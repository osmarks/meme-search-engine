use anyhow::{Result, Context};
use itertools::Itertools;
use std::io::{BufReader, BufWriter, Write};
use rmp_serde::decode::Error as DecodeError;
use std::fs;
use diskann::{augment_bipartite, build_graph, project_bipartite, random_fill_graph, vector::{dot, VectorList}, IndexBuildConfig, IndexGraph, Timer};
use half::f16;

mod common;

use common::{ShardInputHeader, ShardedRecord, ShardHeader};

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
    let mut query_knns = vec![];

    let header: ShardInputHeader = rmp_serde::from_read(&mut stream)?;
    let centroid_fp16 = header.centroid.iter().map(|x| f16::from_f32(*x)).collect::<Vec<_>>();

    {
        let _timer = Timer::new("read shard");
        loop {
            let res: Result<ShardedRecord, DecodeError> = rmp_serde::from_read(&mut stream);
            match res {
                Ok(x) => {
                    original_ids.push(x.id);
                    vector_data.extend(bytemuck::cast_slice(&x.vector));
                    query_knns.push(x.query_knns);
                },
                Err(DecodeError::InvalidDataRead(x)) | Err(DecodeError::InvalidMarkerRead(x)) if x.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e).context("decode fail")
            }
        }
    }

    let mut config = IndexBuildConfig {
        r: 64,
        r_cap: 80,
        l: 500,
        maxc: 950,
        alpha: 65536
    };

    let vecs = VectorList {
        data: vector_data,
        d_emb: D_EMB,
        length: original_ids.len()
    };

    let mut graph = IndexGraph::empty(original_ids.len(), config.r_cap);

    {
        //let _timer = Timer::new("project bipartite");
        //project_bipartite(&mut rng, &mut graph, &query_knns, &query_knns_bwd, config, &vecs);
    }

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
        build_graph(&mut rng, &mut graph, medioid, &vecs, config);
    }

    report_degrees(&graph);

    {
        let _timer = Timer::new("second pass");
        config.alpha = 60000;
        //build_graph(&mut rng, &mut graph, medioid, &vecs, config);
    }

    report_degrees(&graph);

    std::mem::drop(vecs);

    let mut query_knns_bwd = vec![vec![]; header.max_query_id];

    {
        let _timer = Timer::new("compute backward edges");
        for (record_id, knns) in query_knns.iter().enumerate() {
            for &k in knns {
                query_knns_bwd[k as usize].push(record_id as u32);
            }
        }
    }

    {
        let _timer = Timer::new("augment bipartite");
        //augment_bipartite(&mut rng, &mut graph, query_knns, query_knns_bwd, config);
    }

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
