#![feature(pointer_is_aligned_to)]
#![feature(test)]

extern crate test;

use foldhash::{HashSet, HashMap, HashMapExt, HashSetExt};
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
    pub fn random_r_regular(rng: &mut Rng, n: usize, r: usize, capacity: usize) -> Self {
        let mut graph = Vec::with_capacity(n);
        for _ in 0..n {
            let mut adjacency = Vec::with_capacity(capacity);
            for _ in 0..r {
                adjacency.push(rng.u32(0..(n as u32)));
            }
            graph.push(RwLock::new(adjacency));
        }
        IndexGraph {
            graph
        }
    }

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
    pub alpha: i64
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
    pub fn new(IndexBuildConfig { l, r, maxc, .. }: IndexBuildConfig) -> Self {
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
pub fn greedy_search(scratch: &mut Scratch, start: u32, query: VectorRef, vecs: &VectorList, graph: &IndexGraph, config: IndexBuildConfig) -> GreedySearchCounters {
    scratch.visited.clear();
    scratch.neighbour_buffer.clear();
    scratch.visited_list.clear();

    scratch.neighbour_buffer.insert(start, fast_dot_noprefetch(query, &vecs[start as usize]));
    scratch.visited.insert(start);

    let mut counters = GreedySearchCounters { distances: 0 };

    while let Some(pt) = scratch.neighbour_buffer.next_unvisited() {
        scratch.neighbour_pre_buffer.clear();
        for &neighbour in graph.out_neighbours(pt).iter() {
            if scratch.visited.insert(neighbour) {
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

fn merge_existing_neighbours(candidates: &mut CandidateList, point: u32, neigh: &[u32], vecs: &VectorList, config: IndexBuildConfig) {
    let p_vec = &vecs[point as usize];
    for (i, &n) in neigh.iter().enumerate() {
        let dot = fast_dot(p_vec, &vecs[n as usize], &vecs[neigh[(i + 1) % neigh.len() as usize] as usize]);
        candidates.push((n, dot));
    }
}

// "Robust prune" algorithm, kind of
// The algorithm in the paper does not actually match the code as implemented in microsoft/DiskANN
// and that's slightly different from the one in ParlayANN for no reason
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
        candidate_index += 1;
        if p_star == p || p_star == u32::MAX {
            continue;
        }

        neigh.push(p_star);

        scratch.robust_prune_scratch_buffer.clear();

        // mark remaining candidates as not-to-be-used if "not much better than" current candidate
        for i in (candidate_index+1)..candidates.len() {
            let p_prime = candidates[i].0;
            if p_prime != u32::MAX {
                scratch.robust_prune_scratch_buffer.push((i, p_prime));
            }
        }

        for (i, &(ci, p_prime)) in scratch.robust_prune_scratch_buffer.iter().enumerate() {
            let next_vec = &vecs[scratch.robust_prune_scratch_buffer[(i + 1) % scratch.robust_prune_scratch_buffer.len()].0 as usize];
            let p_star_prime_score = fast_dot(&vecs[p_prime as usize], &vecs[p_star as usize], next_vec);
            let p_prime_p_score = candidates[ci].1;
            let alpha_times_p_star_prime_score = (config.alpha * p_star_prime_score) >> 16;

            if alpha_times_p_star_prime_score >= p_prime_p_score {
                candidates[ci].0 = u32::MAX;
            }
        }
    }
}

pub fn build_graph(rng: &mut Rng, graph: &mut IndexGraph, medioid: u32, vecs: &VectorList, config: IndexBuildConfig) {
    assert!(vecs.len() < u32::MAX as usize);

    let mut sigmas: Vec<u32> = (0..(vecs.len() as u32)).collect();
    rng.shuffle(&mut sigmas);

    let rng = Mutex::new(rng.fork());

    //let scratch = &mut Scratch::new(config);
    //let mut rng = rng.lock().unwrap();
    sigmas.into_par_iter().for_each_init(|| (Scratch::new(config), rng.lock().unwrap().fork()), |(scratch, rng), sigma_i| {
    //sigmas.into_iter().for_each(|sigma_i| {
        greedy_search(scratch, medioid, &vecs[sigma_i as usize], vecs, &graph, config);

        {
            let n = graph.out_neighbours(sigma_i);
            merge_existing_neighbours(&mut scratch.visited_list, sigma_i, &*n, vecs, config);
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
                merge_existing_neighbours(&mut scratch.visited_list, neighbour, &neighbour_neighbours, vecs, config);
                merge_existing_neighbours(&mut scratch.visited_list, neighbour, &vec![sigma_i], vecs, config);
                robust_prune(scratch, neighbour, &mut neighbour_neighbours, vecs, config);
            } else if !neighbour_neighbours.contains(&sigma_i) && neighbour_neighbours.len() < config.r {
                neighbour_neighbours.push(sigma_i);
            }
        }
    });
}

// RoarGraph's AcquireNeighbours algorithm is actually almost identical to Vamana/DiskANN's RobustPrune, but with fixed α = 1.0.
// We replace Vamana's random initialization of the graph with Neighbourhood-Aware Projection from RoarGraph - there's no way to use a large enough
// query set that I would be confident in using *only* RoarGraph's algorithm
pub fn project_bipartite(rng: &mut Rng, graph: &mut IndexGraph, query_knns: &Vec<Vec<u32>>, query_knns_bwd: &Vec<Vec<u32>>, config: IndexBuildConfig, vecs: &VectorList) {
    let mut sigmas: Vec<u32> = (0..(graph.graph.len() as u32)).collect();
    rng.shuffle(&mut sigmas);

    // Iterate through graph vertices in a random order
    let rng = Mutex::new(rng.fork());
    sigmas.into_par_iter().for_each_init(|| (rng.lock().unwrap().fork(), Scratch::new(config)), |(rng, scratch), sigma_i| {
        scratch.visited.clear();
        scratch.visited_list.clear();
        scratch.neighbour_pre_buffer.clear();
        for &query_neighbour in query_knns[sigma_i as usize].iter() {
            for &projected_neighbour in query_knns_bwd[query_neighbour as usize].iter() {
                if scratch.visited.insert(projected_neighbour) {
                    scratch.neighbour_pre_buffer.push(projected_neighbour);
                }
            }
        }
        rng.shuffle(&mut scratch.neighbour_pre_buffer);
        scratch.neighbour_pre_buffer.truncate(config.maxc * 2);
        for (i, &projected_neighbour) in scratch.neighbour_pre_buffer.iter().enumerate() {
            let score = fast_dot(&vecs[sigma_i as usize], &vecs[projected_neighbour as usize], &vecs[scratch.neighbour_pre_buffer[(i + 1) % scratch.neighbour_pre_buffer.len()] as usize]);
            scratch.visited_list.push((projected_neighbour, score));
        }
        let mut neighbours = graph.out_neighbours_mut(sigma_i);
        robust_prune(scratch, sigma_i, &mut *neighbours, vecs, config);
    })
}

pub fn augment_bipartite(rng: &mut Rng, graph: &mut IndexGraph, query_knns: Vec<Vec<u32>>, query_knns_bwd: Vec<Vec<u32>>, config: IndexBuildConfig) {
    let mut sigmas: Vec<u32> = (0..(graph.graph.len() as u32)).collect();
    rng.shuffle(&mut sigmas);

    // Iterate through graph vertices in a random order
    let rng = Mutex::new(rng.fork());
    sigmas.into_par_iter().for_each_init(|| rng.lock().unwrap().fork(), |rng, sigma_i| {
        let mut neighbours = graph.out_neighbours_mut(sigma_i);
        let mut i = 0;
        while neighbours.len() < config.r && i < 100 {
            let query_neighbour = *rng.choice(&query_knns[sigma_i as usize]).unwrap();
            let projected_neighbour = *rng.choice(&query_knns_bwd[query_neighbour as usize]).unwrap();
            if !neighbours.contains(&projected_neighbour) {
                neighbours.push(projected_neighbour);
            }
            i += 1;
        }
    })
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
