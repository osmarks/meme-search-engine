use anyhow::{bail, Context, Result};
use diskann::vector::scale_dot_result_f64;
use serde::{Serialize, Deserialize};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::fs;
use base64::Engine;
use argh::FromArgs;
use chrono::{TimeZone, Utc, DateTime};
use std::collections::VecDeque;
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
    query_vector: String,
    #[argh(positional)]
    index_path: String
}

fn read_node(id: u32, data_file: &mut fs::File, header: &IndexHeader) -> Result<PackedIndexEntry> {
    let offset = id as usize * header.record_pad_size;
    data_file.seek(SeekFrom::Start(offset as u64))?;
    let mut buf = vec![0; header.record_pad_size as usize];
    data_file.read_exact(&mut buf)?;
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

fn greedy_search(scratch: &mut Scratch, start: u32, query: &[f16], query_preprocessed: &DistanceLUT, index: IndexRef) -> Result<(usize, usize)> {
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
            //let next_neighbour = scratch.neighbour_pre_buffer[(i + 1) % scratch.neighbour_pre_buffer.len()]; // TODO
            //let node = read_node(neighbour, index.data_file, index.header)?;
            //let vector = bytemuck::cast_slice(&node.vector);
            //let distance = fast_dot_noprefetch(query, &vector);
            pq_cmps += 1;
            scratch.neighbour_buffer.insert(neighbour, approx_scores[i]);
            //scratch.neighbour_buffer.insert(neighbour, distance);
        }
    }

    Ok((cmps, pq_cmps))
}

fn main() -> Result<()> {
    let args: CLIArguments = argh::from_env();

    let query_vector: Vec<f16> = common::chunk_fp16_buffer(&base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(args.query_vector.as_bytes()).context("invalid base64")?);
    let query_vector_fp32 = query_vector.iter().map(|x| x.to_f32()).collect::<Vec<f32>>();

    let index_path = PathBuf::from(&args.index_path);
    let header: IndexHeader = rmp_serde::from_read(BufReader::new(fs::File::open(index_path.join("index.msgpack"))?))?;
    let mut data_file = fs::File::open(index_path.join("index.bin"))?;
    let pq_codes_file = fs::File::open(index_path.join("index.pq-codes.bin"))?;
    let pq_codes = unsafe {
        // This is unsafe because other processes could in principle edit the mmap'd file.
        // It would be annoying to do anything about this possibility, so ignore it.
        MmapOptions::new().populate().map(&pq_codes_file)?
    };

    let query_preprocessed = header.quantizer.preprocess_query(&query_vector_fp32);

    println!("{} items {} dead {} shards", header.count, header.dead_count, header.shards.len());

    // TODO slightly dubious
    let selected_shard = header.shards.iter().position_max_by_key(|x| {
        scale_dot_result_f64(SpatialSimilarity::dot(&x.0, &query_vector_fp32).unwrap())
    }).unwrap();

    println!("best shard is {}", selected_shard);

    for shard in 0..header.shards.len() {
        let selected_start = header.shards[shard].1;

        let mut scratch = Scratch {
            visited: HashSet::new(),
            neighbour_buffer: NeighbourBuffer::new(5000),
            neighbour_pre_buffer: Vec::new(),
            visited_list: Vec::new()
        };

        //let query_vector = diskann::vector::quantize(&query_vector, &header.quantizer, &mut rng);
        let cmps = greedy_search(&mut scratch, selected_start, &query_vector, &query_preprocessed, IndexRef {
            data_file: &mut data_file,
            header: &header,
            pq_codes: &pq_codes,
            pq_code_size: header.quantizer.n_dims / header.quantizer.n_dims_per_code,
        })?;

        println!("index scan {}: {:?} cmps", shard, cmps);

        scratch.visited_list.sort_by_key(|x| -x.1);
        for (id, distance, url, shards) in scratch.visited_list.iter().take(20) {
            println!("index scan: {} {} {} {:?}", id, distance, url, shards);
        }
        println!("");
    }

    let mut matches = vec![];
    // brute force scan
    for i in 0..header.count {
        let node = read_node(i, &mut data_file, &header)?;
        //println!("{} {}", i, node.url);
        let vector = bytemuck::cast_slice(&node.vector);
        matches.push((i, fast_dot_noprefetch(&query_vector, &vector), node.url, node.shards));
    }

    matches.sort_by_key(|x| -x.1);
    for (id, distance, url, shards) in matches.iter().take(20) {
        println!("brute force: {} {} {} {:?}", id, distance, url, shards);
    }

    Ok(())
}
