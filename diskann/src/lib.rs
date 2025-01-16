#![feature(pointer_is_aligned_to)]
#![feature(test)]

extern crate test;

use foldhash::{HashSet, HashSetExt};
use fastrand::Rng;
use rayon::prelude::*;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard, Mutex};

pub mod vector;
use vector::{dot, fast_dot, fast_dot_noprefetch, to_svector, VectorRef, SVector, VectorList};

// ParlayANN improves parallelism by not using locks like this and instead using smarter batch operations
// but I don't have enough cores that it matters
#[derive(Debug)]
pub struct IndexGraph {
    pub graph: Vec<RwLock<Vec<u32>>>
}

impl IndexGraph {
    pub fn empty(n: usize, capacity: usize) -> IndexGraph {
        let mut graph = Vec::with_capacity(n);
        for _ in 0..n {
            graph.push(RwLock::new(Vec::with_capacity(capacity)));
        }
        IndexGraph {
            graph
        }
    }

    fn out_neighbours(&self, pt: u32) -> RwLockReadGuard<Vec<u32>> {
        self.graph[pt as usize].read().unwrap()
    }

    fn out_neighbours_mut(&self, pt: u32) -> RwLockWriteGuard<Vec<u32>> {
        self.graph[pt as usize].write().unwrap()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct IndexBuildConfig {
    pub r: usize,
    pub l: usize,
    pub maxc: usize,
    pub alpha: i64,
    pub saturate_graph: bool,
    pub query_breakpoint: u32, // above this nodes are queries and not base vectors
    pub max_add_per_stitch_iter: usize,
    pub query_alpha: i64
}


fn centroid(vecs: &VectorList) -> SVector {
    let mut centroid = SVector::zero(vecs.d_emb);

    for (i, vec) in vecs.iter().enumerate() {
        let weight = 1.0 / (i + 1) as f32;
        centroid += (to_svector(vec) - &centroid) * weight;
    }

    centroid
}

pub fn medioid(vecs: &VectorList) -> u32 {
    let centroid = centroid(vecs).half();
    vecs.iter().map(|vec| dot(vec, &*centroid)).enumerate().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0 as u32
}

// neighbours list sorted by score descending
// TODO: this may actually be an awful datastructure
// we could also have a heap of unvisited things, but the algorithm's stopping condition cares about visited things, and this is probably still the easiest way
#[derive(Clone, Debug)]
pub struct NeighbourBuffer {
    pub ids: Vec<u32>,
    scores: Vec<i64>,
    visited: Vec<bool>,
    next_unvisited: Option<u32>,
    size: usize
}

impl NeighbourBuffer {
    pub fn new(size: usize) -> Self {
        NeighbourBuffer {
            ids: Vec::with_capacity(size + 1),
            scores: Vec::with_capacity(size + 1),
            visited: Vec::with_capacity(size + 1), //bitvec::vec::BitVec::with_capacity(size),
            next_unvisited: None,
            size
        }
    }

    pub fn next_unvisited(&mut self) -> Option<u32> {
        //println!("next_unvisited: {:?}", self);
        let mut cur = self.next_unvisited? as usize;
        let old_cur = cur;
        self.visited[cur] = true;
        while cur < self.len() && self.visited[cur] {
            cur += 1;
        }
        if cur == self.len() {
            self.next_unvisited = None;
        } else {
            self.next_unvisited = Some(cur as u32);
        }
        Some(self.ids[old_cur])
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn cap(&self) -> usize {
        self.size
    }

    pub fn insert(&mut self, id: u32, score: i64) {
        if self.len() == self.cap() && self.scores[self.len() - 1] > score {
            return;
        }

        let loc = match self.scores.binary_search_by(|x| score.partial_cmp(&x).unwrap()) {
            Ok(loc) => loc,
            Err(loc) => loc
        };

        if self.ids.get(loc) == Some(&id) {
            return;
        }

        // slightly inefficient but we avoid unsafe code
        self.ids.insert(loc, id);
        self.scores.insert(loc, score);
        self.visited.insert(loc, false);
        self.ids.truncate(self.size);
        self.scores.truncate(self.size);
        self.visited.truncate(self.size);

        match self.next_unvisited {
            Some(ref mut next_unvisited) => {
                *next_unvisited = (loc as u32).min(*next_unvisited);
            },
            None => {
                self.next_unvisited = Some(loc as u32);
            }
        }
    }

    pub fn clear(&mut self) {
        self.ids.clear();
        self.scores.clear();
        self.visited.clear();
        self.next_unvisited = None;
    }
}

pub struct Scratch {
    visited: HashSet<u32>,
    pub neighbour_buffer: NeighbourBuffer,
    neighbour_pre_buffer: Vec<u32>,
    visited_list: Vec<(u32, i64)>,
    robust_prune_scratch_buffer: Vec<(usize, u32)>
}

impl Scratch {
    pub fn new(IndexBuildConfig { l, r, .. }: IndexBuildConfig) -> Self {
        Scratch {
            visited: HashSet::with_capacity(l * 8),
            neighbour_buffer: NeighbourBuffer::new(l),
            neighbour_pre_buffer: Vec::with_capacity(r),
            visited_list: Vec::with_capacity(l * 8),
            robust_prune_scratch_buffer: Vec::with_capacity(r)
        }
    }
}

pub struct GreedySearchCounters {
    pub distances: usize
}

// Algorithm 1 from the DiskANN paper
// We support the dot product metric only, so we want to keep things with the HIGHEST dot product
pub fn greedy_search(scratch: &mut Scratch, start: u32, base_vectors_only: bool, query: VectorRef, vecs: &VectorList, graph: &IndexGraph, config: IndexBuildConfig) -> GreedySearchCounters {
    scratch.visited.clear();
    scratch.neighbour_buffer.clear();
    scratch.visited_list.clear();

    scratch.neighbour_buffer.insert(start, fast_dot_noprefetch(query, &vecs[start as usize]));
    scratch.visited.insert(start);

    let mut counters = GreedySearchCounters { distances: 0 };

    while let Some(pt) = scratch.neighbour_buffer.next_unvisited() {
        scratch.neighbour_pre_buffer.clear();
        for &neighbour in graph.out_neighbours(pt).iter() {
            let neighbour_is_query = neighbour >= config.query_breakpoint; // OOD-DiskANN page 4: if we are searching for a query, only consider results in base vectors
            if scratch.visited.insert(neighbour) && !(base_vectors_only && neighbour_is_query) {
                scratch.neighbour_pre_buffer.push(neighbour);
            }
        }
        for (i, &neighbour) in scratch.neighbour_pre_buffer.iter().enumerate() {
            let next_neighbour = scratch.neighbour_pre_buffer[(i + 1) % scratch.neighbour_pre_buffer.len()]; // TODO
            let distance = fast_dot(query, &vecs[neighbour as usize], &vecs[next_neighbour as usize]);
            counters.distances += 1;
            scratch.neighbour_buffer.insert(neighbour, distance);
            scratch.visited_list.push((neighbour, distance));
        }
    }

    counters
}

type CandidateList = Vec<(u32, i64)>;

fn merge_existing_neighbours(candidates: &mut CandidateList, point: u32, neigh: &[u32], vecs: &VectorList) {
    let p_vec = &vecs[point as usize];
    for (i, &n) in neigh.iter().enumerate() {
        let dot = fast_dot(p_vec, &vecs[n as usize], &vecs[neigh[(i + 1) % neigh.len() as usize] as usize]);
        candidates.push((n, dot));
    }
}

// "Robust prune" algorithm, kind of
// The algorithm in the paper does not actually match the code as implemented in microsoft/DiskANN
// and that's slightly different from the one in ParlayANN for no clear reason
// This is closer to ParlayANN
fn robust_prune(scratch: &mut Scratch, p: u32, neigh: &mut Vec<u32>, vecs: &VectorList, config: IndexBuildConfig) {
    neigh.clear();

    let candidates = &mut scratch.visited_list;

    // distance low to high = score high to low
    candidates.sort_unstable_by_key(|&(_id, score)| -score);
    candidates.truncate(config.maxc);

    let mut candidate_index = 0;
    while neigh.len() < config.r && candidate_index < candidates.len() {
        let p_star = candidates[candidate_index].0;
        let p_star_score = candidates[candidate_index].1;
        candidate_index += 1;
        if p_star == p || p_star_score == i64::MIN {
            continue;
        }

        neigh.push(p_star);

        scratch.robust_prune_scratch_buffer.clear();

        // mark remaining candidates as not-to-be-used if "not much better than" current candidate
        for i in (candidate_index+1)..candidates.len() {
            let p_prime = candidates[i].0;
            if candidates[i].1 != i64::MIN {
                scratch.robust_prune_scratch_buffer.push((i, p_prime));
            }
        }

        for (i, &(ci, p_prime)) in scratch.robust_prune_scratch_buffer.iter().enumerate() {
            let next_vec = &vecs[scratch.robust_prune_scratch_buffer[(i + 1) % scratch.robust_prune_scratch_buffer.len()].0 as usize];
            let p_star_prime_score = fast_dot(&vecs[p_prime as usize], &vecs[p_star as usize], next_vec);
            let p_prime_p_score = candidates[ci].1;
            let con_alpha = if p_prime >= config.query_breakpoint {
                config.query_alpha
            } else {
                config.alpha
            };
            let alpha_times_p_star_prime_score = (con_alpha * p_star_prime_score) >> 16;

            if alpha_times_p_star_prime_score >= p_prime_p_score {
                candidates[ci].1 = i64::MIN;
            }
        }
    }

    // saturate graph on for query points - otherwise they get no neighbours, more or less
    if config.saturate_graph || p >= config.query_breakpoint {
        for &(id, _score) in candidates.iter() {
            if neigh.len() == config.r {
                return;
            }
            if !neigh.contains(&id) {
                neigh.push(id);
            }
        }
    }
}

pub fn build_graph(rng: &mut Rng, graph: &mut IndexGraph, medioid: u32, vecs: &VectorList, config: IndexBuildConfig) {
    assert!(vecs.len() < u32::MAX as usize);
    assert_eq!(vecs.len(), graph.graph.len());

    let mut sigmas: Vec<u32> = (0..(vecs.len() as u32)).collect();
    rng.shuffle(&mut sigmas);

    //let scratch = &mut Scratch::new(config);
    //let mut rng = rng.lock().unwrap();
    sigmas.into_par_iter().for_each_init(|| Scratch::new(config), |scratch, sigma_i| {
    //sigmas.into_iter().for_each(|sigma_i| {
        let is_query = sigma_i >= config.query_breakpoint;
        greedy_search(scratch, medioid, is_query, &vecs[sigma_i as usize], vecs, &graph, config);

        {
            let n = graph.out_neighbours(sigma_i);
            merge_existing_neighbours(&mut scratch.visited_list, sigma_i, &*n, vecs);
        }

        {
            let mut n = graph.out_neighbours_mut(sigma_i);
            robust_prune(scratch, sigma_i, &mut *n, vecs, config);
        }

        let neighbours = graph.out_neighbours(sigma_i).to_owned();
        for neighbour in neighbours {
            let mut neighbour_neighbours = graph.out_neighbours_mut(neighbour);
            if neighbour_neighbours.len() == config.r {
                scratch.visited_list.clear();
                merge_existing_neighbours(&mut scratch.visited_list, neighbour, &neighbour_neighbours, vecs);
                merge_existing_neighbours(&mut scratch.visited_list, neighbour, &vec![sigma_i], vecs);
                robust_prune(scratch, neighbour, &mut neighbour_neighbours, vecs, config);
            } else if !neighbour_neighbours.contains(&sigma_i) && neighbour_neighbours.len() < config.r {
                neighbour_neighbours.push(sigma_i);
            }
        }
    });
}

pub fn robust_stitch(rng: &mut Rng, graph: &mut IndexGraph, vecs: &VectorList, config: IndexBuildConfig) {
    let n_queries = graph.graph.len() as u32 - config.query_breakpoint;
    let mut in_edges = Vec::with_capacity(n_queries as usize);
    for _i in 0..(n_queries as usize) {
        in_edges.push(Vec::with_capacity(config.r as usize));
    }

    let mut queries_order = (config.query_breakpoint..(graph.graph.len() as u32)).collect::<Vec<u32>>();
    rng.shuffle(&mut queries_order);

    for base_i in 0..config.query_breakpoint {
        let mut out_neighbours = graph.out_neighbours_mut(base_i);
        // store out-edges (to queries) from each base data node with corresponding query node and drop out-edges to queries
        out_neighbours.retain(|&out_neighbour_out_edge| {
            let is_query = out_neighbour_out_edge >= config.query_breakpoint;
            if is_query {
                in_edges[(out_neighbour_out_edge - config.query_breakpoint) as usize].push(base_i);
            }
            !is_query
        });
    }

    queries_order.into_par_iter().for_each(|query_i| {
        // For each query, fill spare space at in-neighbours with query's out-neighbours
        // The OOD-DiskANN paper itself seems to fill *all* the spare space at once with (out-neighbours of) the first query which is encountered, which feels like an odd choice.
        // We have a switch for that instead.
        let query_out_neighbours = graph.out_neighbours(query_i);
        for &in_neighbour in in_edges[(query_i - config.query_breakpoint) as usize].iter() {
            let mut candidates = Vec::with_capacity(query_out_neighbours.len());
            for (i, &neigh) in query_out_neighbours.iter().enumerate() {
                let score = fast_dot(&vecs[in_neighbour as usize], &vecs[neigh as usize], &vecs[query_out_neighbours[(i + 1) % query_out_neighbours.len()] as usize]);
                candidates.push((neigh, score));
            }
            candidates.sort_unstable_by_key(|(_neigh, score)| -*score);
            let mut in_neighbour_out_edges = graph.out_neighbours_mut(in_neighbour);
            let mut added = 0;
            for (neigh, _score) in candidates {
                if added >= config.max_add_per_stitch_iter || in_neighbour_out_edges.len() >= config.r {
                    break;
                }
                if in_neighbour_out_edges.contains(&neigh) {
                    continue;
                }
                in_neighbour_out_edges.push(neigh);
                added += 1;
            }
        }
    });
}

pub fn random_fill_graph(rng: &mut Rng, graph: &mut IndexGraph, r: usize) {
    let rng = Mutex::new(rng.fork());
    (0..graph.graph.len() as u32).into_par_iter().for_each_init(|| rng.lock().unwrap().fork(), |rng, i| {
        let mut neighbours = graph.out_neighbours_mut(i);
        while neighbours.len() < r {
            let next = rng.u32(0..(graph.graph.len() as u32));
            if !neighbours.contains(&next) {
                neighbours.push(next);
            }
        }
    });
}

pub struct Timer(&'static str, std::time::Instant);

impl Timer {
    pub fn new(name: &'static str) -> Self {
        Timer(name, std::time::Instant::now())
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        println!("{}: {:.2}s", self.0, self.1.elapsed().as_secs_f32());
    }
}

pub fn report_degrees(graph: &IndexGraph) {
    let mut total_degree = 0;
    let mut degrees = Vec::with_capacity(graph.graph.len());
    for out_neighbours in graph.graph.iter() {
        let deg = out_neighbours.read().unwrap().len();
        total_degree += deg;
        degrees.push(deg);
    }
    degrees.sort_unstable();
    println!("average degree {}", (total_degree as f64) / (graph.graph.len() as f64));
    println!("median degree {}", degrees[degrees.len() / 2]);
    println!("min degree {}", degrees[0]);
    println!("max degree {}", degrees[degrees.len() - 1]);
}
