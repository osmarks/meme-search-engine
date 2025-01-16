use anyhow::{Result, Context};
use std::io::{BufReader, BufWriter, Write, Read};
use rmp_serde::decode::Error as DecodeError;
use std::fs;
use diskann::{build_graph, random_fill_graph, vector::VectorList, IndexBuildConfig, IndexGraph, Timer, report_degrees, medioid, robust_stitch};
use half::f16;
use argh::FromArgs;

mod common;

use common::{ShardInputHeader, ShardedRecord, ShardHeader};

#[derive(FromArgs)]
#[argh(description="Generate indices from shard files")]
struct CLIArguments {
    #[argh(positional)]
    input_file: String,
    #[argh(positional)]
    out_dir: String,
    #[argh(positional)]
    queries_bin: Option<String>,
    #[argh(option, short='L', default="192", description="search list size (higher is better but slower)")]
    l: usize,
    #[argh(option, short='R', default="64", description="graph degree")]
    r: usize,
    #[argh(option, short='C', default="750", description="max candidate list size")]
    maxc: usize,
    #[argh(option, short='A', default="65536", description="first pass relaxation factor (times 2^16)")]
    alpha: i64,
    #[argh(option, short='Q', default="65536", description="query set special relaxation factor (times 2^16)")]
    query_alpha: i64,
    #[argh(option, short='B', default="65536", description="second pass relaxation factor (times 2^16)")]
    alpha_2: i64,
    #[argh(switch, short='s', description="do second pass")]
    second_pass: bool,
    #[argh(option, short='N', description="number of vectors to allocate for")]
    n: Option<usize>
}

const D_EMB: usize = 1152;
const READ_CHUNK_SIZE: usize = D_EMB * size_of::<f16>() * 1024;

fn main() -> Result<()> {
    let args: CLIArguments = argh::from_env();

    let mut rng = fastrand::Rng::new();

    let mut stream = BufReader::new(fs::File::open(args.input_file)?);

    // There is no convenient way to pass the actual size along, so accursedly do it manually
    let mut original_ids = Vec::with_capacity(args.n.unwrap_or(0));
    let mut vector_data = Vec::with_capacity(args.n.unwrap_or(0) * D_EMB);

    let header: ShardInputHeader = rmp_serde::from_read(&mut stream)?;

    {
        let _timer = Timer::new("read shard");
        loop {
            let res: Result<ShardedRecord, DecodeError> = rmp_serde::from_read(&mut stream);
            match res {
                Ok(x) => {
                    original_ids.push(x.id);
                    vector_data.extend(bytemuck::cast_slice(&x.vector));
                },
                Err(DecodeError::InvalidDataRead(x)) | Err(DecodeError::InvalidMarkerRead(x)) if x.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e).context("decode fail")
            }
        }
    }

    let query_breakpoint = original_ids.len();

    if let Some(ref queries_bin) = args.queries_bin {
        let mut queries_file = BufReader::new(fs::File::open(queries_bin)?);
        let mut buf = vec![0; READ_CHUNK_SIZE];
        loop {
            let n = queries_file.by_ref().take(READ_CHUNK_SIZE as u64).read_to_end(&mut buf)?;
            if n == 0 {
                break;
            }
            vector_data.extend(bytemuck::cast_slice(&buf[..n]));
        }
    }

    let mut config = IndexBuildConfig {
        r: args.r,
        l: args.l,
        maxc: args.maxc,
        alpha: args.alpha,
        query_alpha: args.query_alpha,
        saturate_graph: false,
        query_breakpoint: query_breakpoint as u32,
        max_add_per_stitch_iter: 16
    };

    let vecs = VectorList {
        length: vector_data.len() / D_EMB,
        data: vector_data,
        d_emb: D_EMB
    };

    let mut graph = IndexGraph::empty(vecs.len(), config.r);

    {
        let _timer = Timer::new("random fill");
        random_fill_graph(&mut rng, &mut graph, config.r);
    }

    report_degrees(&graph);

    let medioid = medioid(&vecs);

    {
        let _timer = Timer::new("first pass");
        build_graph(&mut rng, &mut graph, medioid, &vecs, config);
    }

    report_degrees(&graph);

    if args.second_pass {
        {
            let _timer = Timer::new("second pass");
            config.alpha = args.alpha_2;
            build_graph(&mut rng, &mut graph, medioid, &vecs, config);
        }
        report_degrees(&graph);
    }

    if query_breakpoint < graph.graph.len() {
        let _timer = Timer::new("robust stitch");
        robust_stitch(&mut rng, &mut graph, &vecs, config);
        report_degrees(&graph);
    }

    std::mem::drop(vecs);

    let len = original_ids.len();

    {
        let _timer = Timer::new("write shard");
        let mut graph_data = BufWriter::new(fs::File::create(&format!("{}/{}.shard.bin", args.out_dir, header.id))?);

        let mut offsets = Vec::with_capacity(original_ids.len());
        let mut offset = 0;
        for (i, out_neighbours) in graph.graph.iter().enumerate() {
            if i >= query_breakpoint { break; }
            let out_neighbours = out_neighbours.read().unwrap();
            offsets.push(offset);
            let s: &[u8] = bytemuck::cast_slice(&*out_neighbours);
            offset += s.len() as u64;
            graph_data.write_all(s)?;
        }
        offsets.push(offset); // dummy entry for convenience

        let mut header_f = fs::File::create(&format!("{}/{}.shard-header.msgpack", args.out_dir, header.id))?;
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
