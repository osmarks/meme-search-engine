use anyhow::{Context, Result};
use lazy_static::lazy_static;
use monoio::fs;
use std::path::PathBuf;
use base64::Engine;
use argh::FromArgs;
use itertools::Itertools;
use foldhash::{HashSet, HashSetExt};
use half::f16;
use diskann::{NeighbourBuffer, vector::{fast_dot_noprefetch, QueryLUT, scale_dot_result, scale_dot_result_f64}};
use simsimd::SpatialSimilarity;
use memmap2::{Mmap, MmapOptions};
use std::rc::Rc;
use monoio::net::{TcpListener, TcpStream};
use monoio::io::IntoPollIo;
use hyper::{body::{Body, Bytes, Incoming, Frame}, server::conn::http1, Method, Request, Response, StatusCode};
use http_body_util::{BodyExt, Empty, Full};
use prometheus::{register_int_counter, register_int_counter_vec, register_int_gauge, Encoder, IntCounter, IntGauge, IntCounterVec};
use std::pin::Pin;
use std::future::Future;
use serde::Serialize;
use std::str::FromStr;
use std::collections::HashMap;

mod common;

use common::{resize_for_embed_sync, FrontendInit, IndexHeader, InferenceServerConfig, PackedIndexEntry, QueryRequest, QueryResult};

#[derive(FromArgs, Clone)]
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
    disable_pq: bool,
    #[argh(option, short='l', description="listen address")]
    listen_address: Option<String>,
    #[argh(option, short='c', description="clip server")]
    clip_server: Option<String>,
}

lazy_static! {
    static ref QUERIES_COUNTER: IntCounter = register_int_counter!("mse_queries", "queries executed").unwrap();
    static ref TERMS_COUNTER: IntCounterVec = register_int_counter_vec!("mse_terms", "terms used in queries, by type", &["type"]).unwrap();
    static ref NODE_READS: IntCounter = register_int_counter!("mse_node_reads", "graph nodes read").unwrap();
    static ref PQ_COMPARISONS: IntCounter = register_int_counter!("mse_pq_comparisons", "product quantization comparisons").unwrap();
}

async fn read_node<'a>(id: u32, index: Rc<Index>) -> Result<PackedIndexEntry> {
    let offset = id as usize * index.header.record_pad_size;
    let buf = vec![0; index.header.record_pad_size as usize];
    let (res, buf) = index.data_file.read_exact_at(buf, offset as u64).await;
    res?;
    NODE_READS.inc();
    let len = u16::from_le_bytes(buf[0..2].try_into().unwrap()) as usize;
    Ok(bitcode::decode(&buf[2..len+2])?)
}

fn next_several_unvisited(s: &mut NeighbourBuffer, n: usize) -> Option<Vec<u32>> {
    let mut result = Vec::new();
    for _ in 0..n {
        if let Some(neighbour) = s.next_unvisited() {
            result.push(neighbour);
        } else {
            break;
        }
    }
    if result.len() > 0 {
        Some(result)
    } else {
        None
    }
}

fn read_pq_codes(id: u32, index: Rc<Index>, buf: &mut Vec<u8>) {
    let loc = (id as usize) * index.pq_code_size;
    buf.extend(&index.pq_codes[loc..loc+index.pq_code_size])
}

struct Scratch {
    visited_adjacent: HashSet<u32>,
    visited: HashSet<u32>,
    neighbour_buffer: NeighbourBuffer,
    neighbour_pre_buffer: Vec<u32>,
    visited_list: Vec<(u32, i64, String, Vec<u32>, Vec<f32>)>
}

struct Index {
    data_file: fs::File,
    pq_codes: Mmap,
    header: Rc<IndexHeader>,
    pq_code_size: usize,
    descriptors: Mmap,
    n_descriptors: usize
}

struct DescriptorScales(Vec<f32>);

fn descriptor_product(index: Rc<Index>, scales: &DescriptorScales, neighbour: u32) -> i64 {
    let mut result = 0;
    // effectively an extra part of the vector to dot product
    for (j, d) in scales.0.iter().enumerate() {
        result += scale_dot_result(d * index.descriptors[neighbour as usize * index.n_descriptors + j] as f32);
    }
    result
}

async fn greedy_search<'a>(scratch: &mut Scratch, start: u32, query: &[f16], query_preprocessed: &QueryLUT, descriptor_scales: &DescriptorScales, index: Rc<Index>, disable_pq: bool, beamwidth: usize) -> Result<(usize, usize)> {
    scratch.visited_adjacent.clear();
    scratch.neighbour_buffer.clear();
    scratch.visited_list.clear();
    scratch.visited.clear();

    let mut cmps = 0;
    let mut pq_cmps = 0;

    scratch.neighbour_buffer.insert(start, 0);
    scratch.visited_adjacent.insert(start);

    while let Some(pts) = next_several_unvisited(&mut scratch.neighbour_buffer, beamwidth) {
        scratch.neighbour_pre_buffer.clear();

        let mut join_handles = Vec::with_capacity(pts.len());

        for &pt in pts.iter() {
            join_handles.push(monoio::spawn(read_node(pt, index.clone())));
        }

        for handle in join_handles {
            let index = index.clone();
            let node = handle.await?;
            let vector = bytemuck::cast_slice(&node.vector);
            let distance = fast_dot_noprefetch(query, &vector);
            cmps += 1;
            if scratch.visited.insert(node.id) {
                scratch.visited_list.push((node.id, distance, node.url, node.shards, node.scores));
            };
            for &neighbour in node.vertices.iter() {
                if scratch.visited_adjacent.insert(neighbour) {
                    scratch.neighbour_pre_buffer.push(neighbour);
                }
            }
            let mut pq_codes = Vec::with_capacity(index.pq_code_size * scratch.neighbour_pre_buffer.len());
            for &neighbour in scratch.neighbour_pre_buffer.iter() {
                read_pq_codes(neighbour, index.clone(), &mut pq_codes);
            }
            let mut approx_scores = index.header.quantizer.asymmetric_dot_product(&query_preprocessed, &pq_codes);
            for (i, &neighbour) in scratch.neighbour_pre_buffer.iter().enumerate() {
                if disable_pq {
                    let node = read_node(neighbour, index.clone()).await?;
                    let vector = bytemuck::cast_slice(&node.vector);
                    let mut score = fast_dot_noprefetch(query, &vector);
                    score += descriptor_product(index.clone(), &descriptor_scales, neighbour);
                    scratch.neighbour_buffer.insert(neighbour, score);
                } else {
                    approx_scores[i] += descriptor_product(index.clone(), &descriptor_scales, neighbour);
                    scratch.neighbour_buffer.insert(neighbour, approx_scores[i]);
                    pq_cmps += 1;
                    PQ_COMPARISONS.inc();
                }
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

async fn evaluate(args: &CLIArguments, index: Rc<Index>) -> Result<()> {
    let mut top_k_ranks_best_shard = vec![];
    let mut top_rank_best_shard = vec![];
    let mut pq_cmps = vec![];
    let mut cmps = vec![];
    let mut recall_total = 0;

    let mut queries = vec![];

    if let Some(query_vector_base64) = &args.query_vector_base64 {
        let query_vector: Vec<f16> = common::chunk_fp16_buffer(&base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(query_vector_base64.as_bytes()).context("invalid base64")?);
        queries.push(query_vector);
    }
    if let Some(query_vector_file) = &args.query_vector_file {
        let query_vectors = fs::read(query_vector_file).await?;
        queries.extend(common::chunk_fp16_buffer(&query_vectors).chunks(1152).map(|x| x.to_vec()).collect::<Vec<_>>());
    }

    if let Some(n) = args.n {
        queries.truncate(n);
    }

    for query_vector in queries.iter() {
        let query_vector_fp32 = query_vector.iter().map(|x| x.to_f32()).collect::<Vec<f32>>();
        let query_preprocessed = index.header.quantizer.preprocess_query(&query_vector_fp32);

        // TODO slightly dubious
        let selected_shard = index.header.shards.iter().position_max_by_key(|x| {
            scale_dot_result_f64(SpatialSimilarity::dot(&x.0, &query_vector_fp32).unwrap())
        }).unwrap();

        if args.verbose {
            println!("selected shard is {}", selected_shard);
        }

        let mut matches = vec![];
        // brute force scan
        for i in 0..index.header.count {
            let node = read_node(i, index.clone()).await?;
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

        for shard in 0..index.header.shards.len() {
            let selected_start = index.header.shards[shard].1;

            let beamwidth = 3;

            let mut scratch = Scratch {
                visited: HashSet::new(),
                neighbour_buffer: NeighbourBuffer::new(args.search_list_size.unwrap_or(1000)),
                neighbour_pre_buffer: Vec::new(),
                visited_list: Vec::new(),
                visited_adjacent: HashSet::new()
            };

            let descriptor_scales = DescriptorScales(vec![0.0, 0.0, 0.0, 0.0]);

            let cmps_result = greedy_search(&mut scratch, selected_start, &query_vector, &query_preprocessed, &descriptor_scales, index.clone(), args.disable_pq, beamwidth).await?;

            // slightly dubious because this is across shards
            pq_cmps.push(cmps_result.1);
            cmps.push(cmps_result.0);

            if args.verbose {
                println!("index scan {}: {:?} cmps", shard, cmps_result);
            }

            scratch.visited_list.sort_by_key(|x| -x.1);
            for (i, (id, distance, url, shards, scores)) in scratch.visited_list.iter().take(20).enumerate() {
                let found_id = match matches.binary_search(&(*id, 0)) {
                    Ok(pos) => pos,
                    Err(pos) => pos
                };
                if args.verbose {
                    println!("index scan: {} {} {} {:?} {:?}; rank {}", id, distance, url, shards, scores, matches[found_id].1 + 1);
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

pub async fn query_clip_server<I, O>(base_url: &str, path: &str, data: Option<I>) -> Result<O> where I: Serialize, O: serde::de::DeserializeOwned {
    // TODO connection pool or something
    // also this won't work over TLS

    let url = hyper::Uri::from_str(base_url)?;

    let stream = TcpStream::connect(format!("{}:{}", url.host().unwrap(), url.port_u16().unwrap_or(80))).await?;
    let io = monoio_compat::hyper::MonoioIo::new(stream.into_poll_io()?);

    let (mut sender, conn) = hyper::client::conn::http1::handshake(io).await?;
    monoio::spawn(async move {
        if let Err(err) = conn.await {
            tracing::error!("connection failed: {:?}", err);
        }
    });

    let authority = url.authority().unwrap().clone();

    let req = Request::builder()
        .uri(path)
        .header(hyper::header::HOST, authority.as_str())
        .header(hyper::header::CONTENT_TYPE, "application/msgpack");

    let res = match data {
        Some(data) => sender.send_request(req.method(Method::POST).body(Full::new(Bytes::from(rmp_serde::to_vec_named(&data)?)))?).await?,
        None => sender.send_request(req.method(Method::GET).body(Full::new(Bytes::from("")))?).await?
    };

    if res.status() != StatusCode::OK {
        return Err(anyhow::anyhow!("unexpected status code: {}", res.status()));
    }

    let data = res.collect().await?.to_bytes();

    let result: O = rmp_serde::from_slice(&data)?;
    Ok(result)
}

#[derive(Clone)]
struct Service {
    index: Rc<Index>,
    inference_server_config: Rc<InferenceServerConfig>,
    args: Rc<CLIArguments>
}

impl hyper::service::Service<Request<Incoming>> for Service {
    type Response = Response<Full<Bytes>>;
    type Error = anyhow::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>>>>;

    fn call(&self, req: Request<Incoming>) -> Self::Future {
        let index = self.index.clone();
        let args = self.args.clone();
        let inference_server_config = self.inference_server_config.clone();

        Box::pin(async move {
            let mut body = match (req.method(), req.uri().path()) {
                (&Method::GET, "/") => Response::new(Full::new(Bytes::from(serde_json::to_vec(&FrontendInit {
                    n_total: (index.header.count - index.header.dead_count) as u64,
                    d_emb: index.header.quantizer.n_dims,
                    predefined_embedding_names: vec![]
                })?))),
                (&Method::POST, "/") => {
                    let upper = req.body().size_hint().upper().unwrap_or(u64::MAX);
                    if upper > 1<<23 {
                        let mut resp = Response::new(Full::new(Bytes::from("Body too big")));
                        *resp.status_mut() = hyper::StatusCode::PAYLOAD_TOO_LARGE;
                        return Ok(resp);
                    }

                    let whole_body = req.collect().await?.to_bytes();

                    let body: QueryRequest = serde_json::from_slice(&whole_body)?;

                    let query = common::get_total_embedding(
                        &body.terms,
                        &*inference_server_config,
                        |batch, _config| {
                            query_clip_server(args.clip_server.as_ref().unwrap(), "/", Some(batch))
                        },
                        |image, config| async move {
                            let image = image::load_from_memory(&image)?;
                            Ok(serde_bytes::ByteBuf::from(resize_for_embed_sync(&*config, image)?))
                        },
                        &std::collections::HashMap::new(),
                        inference_server_config.clone(),
                        ()
                    ).await?;

                    let selected_shard = index.header.shards.iter().position_max_by_key(|x| {
                        scale_dot_result_f64(SpatialSimilarity::dot(&x.0, &query).unwrap())
                    }).unwrap();
                    let selected_start = index.header.shards[selected_shard].1;

                    let beamwidth = 3;

                    let mut scratch = Scratch {
                        visited: HashSet::new(),
                        neighbour_buffer: NeighbourBuffer::new(args.search_list_size.unwrap_or(1000)),
                        neighbour_pre_buffer: Vec::new(),
                        visited_list: Vec::new(),
                        visited_adjacent: HashSet::new()
                    };

                    let descriptor_scales = DescriptorScales(vec![0.0, 0.0, 0.0, 0.0]);

                    let query_preprocessed = index.header.quantizer.preprocess_query(&query);

                    let query = query.iter().map(|x| half::f16::from_f32(*x)).collect::<Vec<f16>>();

                    let cmps_result = greedy_search(&mut scratch, selected_start, &query, &query_preprocessed, &descriptor_scales, index.clone(), args.disable_pq, beamwidth).await?;

                    scratch.visited_list.sort_by_key(|x| -x.1);

                    let matches = scratch.visited_list.drain(..).map(|(id, score, url, shards, scores)| (score as f32, url, String::new(), 0, None)).collect::<Vec<_>>();

                    let result = QueryResult {
                        formats: vec![],
                        extensions: HashMap::new(),
                        matches
                    };

                    let result = serde_json::to_vec(&result)?;

                    Response::new(Full::new(Bytes::from(result)))
                },
                (&Method::GET, "/metrics") => {
                    let mut buffer = Vec::new();
                    let encoder = prometheus::TextEncoder::new();
                    let metric_families = prometheus::gather();
                    encoder.encode(&metric_families, &mut buffer).unwrap();
                    Response::builder()
                        .header(hyper::header::CONTENT_TYPE, "text/plain; version=0.0.4")
                        .body(Full::new(Bytes::from(buffer))).unwrap()
                },
                (&Method::OPTIONS, "/") => {
                    Response::builder()
                        .status(StatusCode::NO_CONTENT)
                        .body(Full::new(Bytes::from(""))).unwrap()
                },
                _ => Response::builder()
                        .status(StatusCode::NOT_FOUND)
                        .body(Full::new(Bytes::from("Not Found")))
                        .unwrap()
            };

            body.headers_mut().entry(hyper::header::CONTENT_TYPE).or_insert(hyper::header::HeaderValue::from_static("application/json"));
            body.headers_mut().entry(hyper::header::ACCESS_CONTROL_ALLOW_ORIGIN).or_insert(hyper::header::HeaderValue::from_static("*"));
            body.headers_mut().entry(hyper::header::ACCESS_CONTROL_ALLOW_METHODS).or_insert(hyper::header::HeaderValue::from_static("GET, POST, OPTIONS"));
            body.headers_mut().entry(hyper::header::ACCESS_CONTROL_ALLOW_HEADERS).or_insert(hyper::header::HeaderValue::from_static("Content-Type"));

            Result::<_, anyhow::Error>::Ok(body)
        })
    }
}

async fn get_backend_config(clip_server: &Option<String>) -> Result<InferenceServerConfig> {
    loop {
        match query_clip_server(clip_server.as_ref().unwrap(), "/config", Option::<()>::None).await {
            Ok(config) => return Ok(config),
            Err(err) => {
                tracing::warn!("waiting for clip server: {}", err);
                monoio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        };
    }
}

async fn serve(args: &CLIArguments, index: Rc<Index>) -> Result<()> {
    let service = Service {
        index,
        inference_server_config: Rc::new(get_backend_config(&args.clip_server).await?),
        args: Rc::new(args.clone())
    };

    let listener = TcpListener::bind(args.listen_address.as_ref().unwrap())?;
    println!("Listening");
    loop {
        let (stream, _) = listener.accept().await?;
        let stream_poll = monoio_compat::hyper::MonoioIo::new(stream.into_poll_io()?);
        let service = service.clone();
        monoio::spawn(async move {
            // Handle the connection from the client using HTTP1 and pass any
            // HTTP requests received on that connection to the `hello` function
            if let Err(err) = http1::Builder::new()
                .timer(monoio_compat::hyper::MonoioTimer)
                .serve_connection(stream_poll, service)
                .await
            {
                println!("Error serving connection: {:?}", err);
            }
        });
    }
}

#[monoio::main(threads=1, enable_timer=true)]
async fn main() -> Result<()> {
    let args: CLIArguments = argh::from_env();

    let index_path = PathBuf::from(&args.index_path);
    let header: IndexHeader = rmp_serde::from_slice(&fs::read(index_path.join("index.msgpack")).await?)?;
    let header = Rc::new(header);
    // contains graph structure, full-precision vectors, and bulk metadata
    let data_file = fs::File::open(index_path.join("index.bin")).await?;
    // contains product quantization codes
    let pq_codes_file = fs::File::open(index_path.join("index.pq-codes.bin")).await?;
    let pq_codes = unsafe {
        // This is unsafe because other processes could in principle edit the mmap'd file.
        // It would be annoying to do anything about this possibility, so ignore it.
        MmapOptions::new().populate().map(&pq_codes_file)?
    };
    // contains metadata descriptors
    let descriptors_file = fs::File::open(index_path.join("index.descriptor-codes.bin")).await?;
    let descriptors = unsafe {
        MmapOptions::new().populate().map(&descriptors_file)?
    };

    println!("{} items {} dead {} shards", header.count, header.dead_count, header.shards.len());

    let index = Rc::new(Index {
        data_file,
        header: header.clone(),
        pq_codes,
        pq_code_size: header.quantizer.n_dims / header.quantizer.n_dims_per_code,
        descriptors,
        n_descriptors: header.descriptor_cdfs.len(),
    });

    if args.listen_address.is_some() {
        serve(&args, index).await?;
    } else {
        evaluate(&args, index).await?;
    }

    Ok(())
}
