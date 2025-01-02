use anyhow::{Result, Context};
use itertools::Itertools;
use std::io::{BufReader, Write, BufWriter, Seek};
use rmp_serde::decode::Error as DecodeError;
use std::fs;
use std::collections::BinaryHeap;
use diskann::{augment_bipartite, build_graph, project_bipartite, random_fill_graph, vector::{dot, VectorList, scale_dot_result}, IndexBuildConfig, IndexGraph, Timer};
use half::f16;

mod common;

use common::{index_config::{self, QUERY_REVERSE_K}, ShardHeader, ShardInputHeader, ShardedRecord};

const D_EMB: usize = 1152;

fn report_degrees(graph: &IndexGraph) {
    let mut total_degree = 0;
    let mut degrees = Vec::with_capacity(graph.graph.len());
    for out_neighbours in graph.graph.iter() {
        let deg = out_neighbours.read().unwrap().len();
        total_degree += deg;
        degrees.push(deg);
    }
    degrees.sort_unstable();
    println!("average degree {}", (total_degree as f32) / (graph.graph.len() as f32));
    println!("median degree {}", degrees[degrees.len() / 2]);
}

fn main() -> Result<()> {
    let mut rng = fastrand::Rng::new();

    let mut stream = BufReader::new(fs::File::open(std::env::args().nth(1).unwrap()).context("read dump file")?);

    let mut original_ids = vec![];
    let mut vector_data = vec![];

    let header: ShardInputHeader = rmp_serde::from_read(&mut stream)?;
    let centroid_fp16 = header.centroid.iter().map(|x| f16::from_f32(*x)).collect::<Vec<_>>();

    let mut query_knns_bwd = vec![BinaryHeap::new(); header.max_query_id];
    query_knns_bwd.fill_with(|| BinaryHeap::with_capacity(QUERY_REVERSE_K));

    {
        let _timer = Timer::new("read shard vectors");
        loop {
            let res: Result<ShardedRecord, DecodeError> = rmp_serde::from_read(&mut stream);
            match res {
                Ok(x) => {
                    let current_local_id = original_ids.len() as u32;
                    original_ids.push(x.id);
                    vector_data.extend(bytemuck::cast_slice(&x.vector));

                    for (&query_id, &distance) in x.query_knns.iter().zip(x.query_knns_distances.iter()) {
                        let distance = scale_dot_result(distance);
                        // Rust BinaryHeap is a max-heap - we want the lowest-dot-product vectors to be discarded first
                        // So negate the distance
                        let knns = &mut query_knns_bwd[query_id as usize];
                        if knns.len() == QUERY_REVERSE_K {
                            knns.pop();
                        }
                        query_knns_bwd[query_id as usize].push((-distance, current_local_id));
                    }
                },
                Err(DecodeError::InvalidDataRead(x)) | Err(DecodeError::InvalidMarkerRead(x)) if x.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e).context("decode fail")
            }
        }
    }

    let mut query_knns = vec![vec![]; original_ids.len()];
    query_knns.fill_with(|| Vec::with_capacity(8));
    let mut query_knns_bwd_out = vec![vec![]; header.max_query_id];
    query_knns_bwd_out.fill_with(|| Vec::with_capacity(QUERY_REVERSE_K));

    {
        let _timer = Timer::new("initialize bipartite graph");
        // RoarGraph: out-edge from closest base vector to each query vector
        for (query_id, distance_id_pairs) in query_knns_bwd.into_iter().enumerate() {
            let vec = distance_id_pairs.into_sorted_vec();
            let it = vec.into_iter();

            for (i, (_distance, id)) in it.enumerate() {
                if i < index_config::PROJECTION_CUT_POINT {
                    query_knns[id as usize].push(query_id as u32);
                } else {
                    query_knns_bwd_out[query_id].push(id);
                }
            }
        }

    }

    let mut config = common::index_config::BASE_CONFIG;

    let vecs = VectorList {
        data: vector_data,
        d_emb: D_EMB,
        length: original_ids.len()
    };

    let mut graph = IndexGraph::empty(original_ids.len(), config.r_cap);

    {
        let _timer = Timer::new("project bipartite");
        project_bipartite(&mut rng, &mut graph, &query_knns, &query_knns_bwd_out, config, &vecs);
    }

    report_degrees(&graph);

    {
        let _timer = Timer::new("random fill");
        random_fill_graph(&mut rng, &mut graph, config.r);
    }

    report_degrees(&graph);

    let medioid = vecs.iter().position_max_by_key(|&v| {
        dot(v, &centroid_fp16)
    }).unwrap() as u32;

    {
        let _timer = Timer::new("first pass");
        config.alpha = common::index_config::FIRST_PASS_ALPHA;
        build_graph(&mut rng, &mut graph, medioid, &vecs, config);
    }

    report_degrees(&graph);

    {
        let _timer = Timer::new("second pass");
        config.alpha = common::index_config::SECOND_PASS_ALPHA;
        build_graph(&mut rng, &mut graph, medioid, &vecs, config);
    }

    report_degrees(&graph);

    std::mem::drop(vecs);

    let len = original_ids.len();

    {
        let _timer = Timer::new("write shard");
        let mut graph_data = BufWriter::new(fs::File::create(&format!("{}.shard.bin", header.id))?);

        let mut offsets = Vec::with_capacity(original_ids.len());
        let mut offset = 0;
        for out_neighbours in graph.graph.iter() {
            let out_neighbours = out_neighbours.read().unwrap();
            offsets.push(offset);
            let s: &[u8] = bytemuck::cast_slice(&*out_neighbours);
            offset += s.len() as u64;
            graph_data.write_all(s)?;
        }
        offsets.push(offset); // dummy entry for convenience

        let mut header_f = fs::File::create(&format!("{}.shard-header.msgpack", header.id))?;
        header_f.write_all(&rmp_serde::to_vec(&ShardHeader {
            id: header.id,
            max: *original_ids.iter().max().unwrap(),
            centroid: header.centroid,
            medioid,
            offsets,
            mapping: original_ids
        })?)?;
    }

    println!("{} vectors", len);

    Ok(())
}
