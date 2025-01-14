use anyhow::{bail, Context, Result};
use diskann::vector::scale_dot_result_f64;
use serde::{Serialize, Deserialize};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::os::unix::prelude::FileExt;
use std::path::PathBuf;
use std::fs;
use base64::Engine;
use argh::FromArgs;
use chrono::{TimeZone, Utc, DateTime};
use itertools::Itertools;
use foldhash::{HashSet, HashSetExt};
use half::f16;
use diskann::{NeighbourBuffer, vector::{fast_dot_noprefetch, ProductQuantizer, DistanceLUT, scale_dot_result}};
use simsimd::SpatialSimilarity;
use memmap2::{Mmap, MmapOptions};

mod common;

use common::{PackedIndexEntry, IndexHeader};

#[derive(FromArgs)]
#[argh(description="Query disk index")]
struct CLIArguments {
    #[argh(positional)]
    index_path: String,
    #[argh(option, short='q', description="query vector in base64")]
    query_vector_base64: Option<String>,
    #[argh(option, short='f', description="file of FP16 query vectors")]
    query_vector_file: Option<String>,
    #[argh(switch, short='v', description="verbose")]
    verbose: bool,
    #[argh(option, short='n', description="stop at n queries")]
    n: Option<usize>,
    #[argh(option, short='L', description="search list size")]
    search_list_size: Option<usize>,
    #[argh(switch, description="always use full-precision vectors (slow)")]
    disable_pq: bool
}

fn read_node(id: u32, data_file: &mut fs::File, header: &IndexHeader) -> Result<PackedIndexEntry> {
    let offset = id as usize * header.record_pad_size;
    let mut buf = vec![0; header.record_pad_size as usize];
    data_file.read_exact_at(&mut buf, offset as u64)?;
    let len = u16::from_le_bytes(buf[0..2].try_into().unwrap()) as usize;
    Ok(bitcode::decode(&buf[2..len+2])?)
}

fn read_pq_codes(id: u32, codes: &Mmap, buf: &mut Vec<u8>, pq_code_size: usize) {
    let loc = (id as usize) * pq_code_size;
    buf.extend(&codes[loc..loc+pq_code_size])
}

struct Scratch {
    visited: HashSet<u32>,
    neighbour_buffer: NeighbourBuffer,
    neighbour_pre_buffer: Vec<u32>,
    visited_list: Vec<(u32, i64, String, Vec<u32>)>
}

struct IndexRef<'a> {
    data_file: &'a mut fs::File,
    pq_codes: &'a Mmap,
    header: &'a IndexHeader,
    pq_code_size: usize
}

fn greedy_search(scratch: &mut Scratch, start: u32, query: &[f16], query_preprocessed: &DistanceLUT, index: IndexRef, disable_pq: bool) -> Result<(usize, usize)> {
    scratch.visited.clear();
    scratch.neighbour_buffer.clear();
    scratch.visited_list.clear();

    let mut cmps = 0;
    let mut pq_cmps = 0;

    let node = read_node(start, index.data_file, index.header)?;
    let vector = bytemuck::cast_slice(&node.vector);
    scratch.neighbour_buffer.insert(start, fast_dot_noprefetch(query, &vector));
    scratch.visited.insert(start);

    while let Some(pt) = scratch.neighbour_buffer.next_unvisited() {
        //println!("pt {} {:?}", pt, graph.out_neighbours(pt));
        scratch.neighbour_pre_buffer.clear();
        let node = read_node(pt, index.data_file, index.header)?;
        let vector = bytemuck::cast_slice(&node.vector);
        let distance = fast_dot_noprefetch(query, &vector);
        cmps += 1;
        scratch.visited_list.push((pt, distance, node.url, node.shards));
        for &neighbour in node.vertices.iter() {
            if scratch.visited.insert(neighbour) {
                scratch.neighbour_pre_buffer.push(neighbour);
            }
        }
        let mut pq_codes = Vec::with_capacity(index.pq_code_size * scratch.neighbour_pre_buffer.len());
        for &neighbour in scratch.neighbour_pre_buffer.iter() {
            read_pq_codes(neighbour, index.pq_codes, &mut pq_codes, index.pq_code_size);
        }
        let approx_scores = index.header.quantizer.asymmetric_dot_product(&query_preprocessed, &pq_codes);
        for (i, &neighbour) in scratch.neighbour_pre_buffer.iter().enumerate() {
            if disable_pq {
                //let next_neighbour = scratch.neighbour_pre_buffer[(i + 1) % scratch.neighbour_pre_buffer.len()]; // TODO
                let node = read_node(neighbour, index.data_file, index.header)?;
                let vector = bytemuck::cast_slice(&node.vector);
                let distance = fast_dot_noprefetch(query, &vector);
                scratch.neighbour_buffer.insert(neighbour, distance);
            } else {
                scratch.neighbour_buffer.insert(neighbour, approx_scores[i]);
                pq_cmps += 1;
            }
        }
    }

    Ok((cmps, pq_cmps))
}

fn summary_stats(ranks: &mut [usize]) {
    let sum = ranks.iter().sum::<usize>();
    let mean = sum as f64 / ranks.len() as f64 + 1.0;
    ranks.sort_unstable();
    let median = ranks[ranks.len() / 2] + 1;
    let harmonic_mean = ranks.iter().map(|x| 1.0 / ((x+1) as f64)).sum::<f64>() / ranks.len() as f64;
    println!("median {} mean {:.2} max {} min {} harmonic mean {:.2}", median, mean, ranks[ranks.len() - 1] + 1, ranks[0] + 1, 1.0 / harmonic_mean);
}

const K: usize = 20;

fn main() -> Result<()> {
    let args: CLIArguments = argh::from_env();

    let mut queries = vec![];

    if let Some(query_vector_base64) = args.query_vector_base64 {
        let query_vector: Vec<f16> = common::chunk_fp16_buffer(&base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(query_vector_base64.as_bytes()).context("invalid base64")?);
        queries.push(query_vector);
    }
    if let Some(query_vector_file) = args.query_vector_file {
        let query_vectors = fs::read(query_vector_file)?;
        queries.extend(common::chunk_fp16_buffer(&query_vectors).chunks(1152).map(|x| x.to_vec()).collect::<Vec<_>>());
    }

    if let Some(n) = args.n {
        queries.truncate(n);
    }

    let index_path = PathBuf::from(&args.index_path);
    let header: IndexHeader = rmp_serde::from_read(BufReader::new(fs::File::open(index_path.join("index.msgpack"))?))?;
    let mut data_file = fs::File::open(index_path.join("index.bin"))?;
    let pq_codes_file = fs::File::open(index_path.join("index.pq-codes.bin"))?;
    let pq_codes = unsafe {
        // This is unsafe because other processes could in principle edit the mmap'd file.
        // It would be annoying to do anything about this possibility, so ignore it.
        MmapOptions::new().populate().map(&pq_codes_file)?
    };

    println!("{} items {} dead {} shards", header.count, header.dead_count, header.shards.len());

    let mut top_k_ranks_best_shard = vec![];
    let mut top_rank_best_shard = vec![];
    let mut pq_cmps = vec![];
    let mut cmps = vec![];
    let mut recall_total = 0;

    for query_vector in queries.iter() {
        let query_vector_fp32 = query_vector.iter().map(|x| x.to_f32()).collect::<Vec<f32>>();
        let query_preprocessed = header.quantizer.preprocess_query(&query_vector_fp32);

        // TODO slightly dubious
        let selected_shard = header.shards.iter().position_max_by_key(|x| {
            scale_dot_result_f64(SpatialSimilarity::dot(&x.0, &query_vector_fp32).unwrap())
        }).unwrap();

        if args.verbose {
            println!("selected shard is {}", selected_shard);
        }

        let mut matches = vec![];
        // brute force scan
        for i in 0..header.count {
            let node = read_node(i, &mut data_file, &header)?;
            //println!("{} {}", i, node.url);
            let vector = bytemuck::cast_slice(&node.vector);
            matches.push((i, fast_dot_noprefetch(&query_vector, &vector), node.url, node.shards));
        }

        matches.sort_unstable_by_key(|x| -x.1);
        let mut matches = matches.into_iter().enumerate().map(|(i, (id, distance, url, shards))| (id, i)).collect::<Vec<_>>();
        matches.sort_unstable();

        /*for (id, distance, url, shards) in matches.iter().take(20) {
            println!("brute force: {} {} {} {:?}", id, distance, url, shards);
        }*/

        let mut top_ranks = vec![usize::MAX; K];

        for shard in 0..header.shards.len() {
            let selected_start = header.shards[shard].1;

            let mut scratch = Scratch {
                visited: HashSet::new(),
                neighbour_buffer: NeighbourBuffer::new(args.search_list_size.unwrap_or(1000)),
                neighbour_pre_buffer: Vec::new(),
                visited_list: Vec::new()
            };

            //let query_vector = diskann::vector::quantize(&query_vector, &header.quantizer, &mut rng);
            let cmps_result = greedy_search(&mut scratch, selected_start, &query_vector, &query_preprocessed, IndexRef {
                data_file: &mut data_file,
                header: &header,
                pq_codes: &pq_codes,
                pq_code_size: header.quantizer.n_dims / header.quantizer.n_dims_per_code,
            }, args.disable_pq)?;

            // slightly dubious because this is across shards
            pq_cmps.push(cmps_result.1);
            cmps.push(cmps_result.0);

            if args.verbose {
                println!("index scan {}: {:?} cmps", shard, cmps);
            }

            scratch.visited_list.sort_by_key(|x| -x.1);
            for (i, (id, distance, url, shards)) in scratch.visited_list.iter().take(20).enumerate() {
                let found_id = match matches.binary_search(&(*id, 0)) {
                    Ok(pos) => pos,
                    Err(pos) => pos
                };
                if args.verbose {
                    println!("index scan: {} {} {} {:?}; rank {}", id, distance, url, shards, matches[found_id].1 + 1);
                };
                top_ranks[i] = std::cmp::min(top_ranks[i], matches[found_id].1);
            }
            if args.verbose { println!("") }
        }

        // results list is always correctly sorted
        for &rank in top_ranks.iter() {
            if rank < K {
                recall_total += 1;
            }
        }

        top_rank_best_shard.push(top_ranks[0]);
        top_k_ranks_best_shard.extend(top_ranks);
    }

    println!("ranks of top 20:");
    summary_stats(&mut top_k_ranks_best_shard);
    println!("ranks of top 1:");
    summary_stats(&mut top_rank_best_shard);
    println!("pq comparisons:");
    summary_stats(&mut pq_cmps);
    println!("comparisons:");
    summary_stats(&mut cmps);
    println!("recall@{}: {}", K, recall_total as f64 / (K * queries.len()) as f64);

    Ok(())
}
