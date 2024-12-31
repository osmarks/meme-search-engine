#![feature(test)]
#![feature(pointer_is_aligned_to)]

extern crate test;

use std::{io::Read, time::Instant};
use anyhow::Result;
use half::f16;

use diskann::{build_graph, IndexBuildConfig, medioid, IndexGraph, greedy_search, Scratch, vector::{fast_dot, dot, VectorList, self}, Timer};
use simsimd::SpatialSimilarity;

const D_EMB: usize = 1152;

fn load_file(path: &str, trunc: Option<usize>) -> Result<VectorList> {
    let mut input = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    input.read_to_end(&mut buf)?;
    // TODO: this is not particularly efficient
    let f16s = bytemuck::cast_slice::<_, f16>(&buf)[0..trunc.unwrap_or(buf.len()/2)].iter().copied().collect();
    Ok(VectorList::from_f16s(f16s, D_EMB))
}

const PQ_TEST_SIZE: usize = 1000;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    {
        let file = std::fs::File::open("opq.msgpack")?;
        let codec: vector::ProductQuantizer = rmp_serde::from_read(file)?;
        let input = load_file("embeddings.bin", Some(D_EMB * PQ_TEST_SIZE))?.data.into_iter().map(|a| a.to_f32()).collect::<Vec<_>>();
        let codes = codec.quantize_batch(&input);
        println!("{:?}", codes);
        let raw_query = load_file("query.bin", Some(D_EMB))?.data.into_iter().map(|a| a.to_f32()).collect::<Vec<_>>();
        let query = codec.preprocess_query(&raw_query);
        let mut real_scores = vec![];
        for i in 0..PQ_TEST_SIZE {
            real_scores.push(SpatialSimilarity::dot(&raw_query, &input[i * D_EMB .. (i + 1) * D_EMB]).unwrap() as f32);
        }
        let pq_scores = codec.asymmetric_dot_product(&query, &codes);
        for (x, y) in real_scores.iter().zip(pq_scores.iter()) {
            println!("{} {} {} {}", x, y, x - y, (x - y) / x);
        }
    }

    let mut rng = fastrand::Rng::with_seed(1);

    let n = 100000;
    let vecs = {
        let _timer = Timer::new("loaded vectors");

        &load_file("embeddings.bin", Some(D_EMB * n))?
    };

    let (graph, medioid) = {
        let _timer = Timer::new("index built");

        let mut config = IndexBuildConfig {
            r: 64,
            r_cap: 80,
            l: 128,
            maxc: 750,
            alpha: 65536,
        };

        let mut graph = IndexGraph::random_r_regular(&mut rng, vecs.len(), config.r, config.r_cap);

        let medioid = medioid(&vecs);

        build_graph(&mut rng, &mut graph, medioid, &vecs, config);
        config.alpha = 58000;
        build_graph(&mut rng, &mut graph, medioid, &vecs, config);

        (graph, medioid)
    };

    let mut edge_ctr = 0;

    for adjlist in graph.graph.iter() {
        edge_ctr += adjlist.read().unwrap().len();
    }

    println!("average degree: {}", edge_ctr as f32 / graph.graph.len() as f32);

    let time = Instant::now();
    let mut recall = 0;
    let mut cmps_ctr = 0;
    let mut cmps = vec![];

    let mut config = IndexBuildConfig {
        r: 64,
        r_cap: 64,
        l: 50,
        alpha: 65536,
        maxc: 0,
    };

    let mut scratch = Scratch::new(config);

    for (i, vec) in tqdm::tqdm(vecs.iter().enumerate()) {
        let ctr = greedy_search(&mut scratch, medioid, &vec, &vecs, &graph, config);
        cmps_ctr += ctr.distances;
        cmps.push(ctr.distances);
        if scratch.neighbour_buffer.ids[0] == (i as u32) {
            recall += 1;
        }
    }

    cmps.sort();

    let end = time.elapsed();

    println!("recall@1: {} ({}/{})", recall as f32 / n as f32, recall, n);
    println!("cmps: {} ({}/{})", cmps_ctr as f32 / n as f32, cmps_ctr, n);
    println!("median comparisons: {}", cmps[cmps.len() / 2]);
    //println!("brute force recall@1: {} ({}/{})", brute_force_recall as f32 / brute_force_queries as f32, brute_force_recall, brute_force_queries);
    println!("{} QPS", n as f32 / end.as_secs_f32());

    Ok(())
}