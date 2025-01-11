use anyhow::{bail, Context, Result};
use serde::{Serialize, Deserialize};
use std::io::{BufReader, Read, Seek, SeekFrom, Write, BufWriter};
use std::path::PathBuf;
use rmp_serde::decode::Error as DecodeError;
use std::fs;
use base64::Engine;
use argh::FromArgs;
use chrono::{TimeZone, Utc, DateTime};
use std::collections::VecDeque;
use faiss::Index;
use std::sync::mpsc::{sync_channel, SyncSender};
use itertools::Itertools;
use simsimd::SpatialSimilarity;
use std::hash::Hasher;
use foldhash::{HashSet, HashSetExt};

use diskann::vector::{scale_dot_result_f64, ProductQuantizer};

mod common;

use common::{ProcessedEntry, ShardInputHeader, ShardedRecord, ShardHeader, PackedIndexEntry, IndexHeader};

#[derive(FromArgs)]
#[argh(description="Process scraper dump files")]
struct CLIArguments {
    #[argh(option, short='s', description="randomly select fraction of records")]
    sample: Option<f32>,
    #[argh(switch, short='p', description="print basic information for records")]
    print_records: bool,
    #[argh(switch, short='e',description="print embeddings")]
    print_embeddings: bool,
    #[argh(switch, short='a', description="print aggregates")]
    print_aggregates: bool,
    #[argh(option, short='E', description="x:y[:f] - load embedding named x from file y, discard record if dot product >= filter threshold f")]
    embedding: Vec<String>,
    #[argh(option, short='H', description="path for histograms of dot with embeddings")]
    histograms: Option<String>,
    #[argh(switch, short='D', description="enable deduplication")]
    deduplicate: bool,
    #[argh(positional)]
    paths: Vec<String>,
    #[argh(option, short='o', description="output embeddings to file")]
    output_embeddings: Option<String>,
    #[argh(option, short='C', description="split input into shards using these centroids")]
    centroids: Option<String>,
    #[argh(option, short='S', description="index shard directory")]
    shards_dir: Option<String>,
    #[argh(option, short='Q', description="query vectors file")]
    queries: Option<String>,
    #[argh(option, short='d', description="random seed")]
    seed: Option<u64>,
    #[argh(option, short='i', description="index output directory")]
    index_output: Option<String>,
    #[argh(switch, short='t', description="print titles")]
    titles: bool,
    #[argh(option, description="truncate centroids list")]
    clip_shards: Option<usize>,
    #[argh(switch, description="print original linked URL")]
    original_url: bool,
    #[argh(option, short='q', description="product quantization codec path")]
    pq_codec: Option<String>,
    #[argh(switch, short='j', description="JSON output")]
    json: bool
}

#[derive(Clone, Deserialize, Serialize, Debug)]
struct Histogram {
    min: f32,
    max: f32,
    buckets: Vec<u64>
}

impl Histogram {
    fn new(min: f32, max: f32, count: usize) -> Self {
        let buckets = (0..count).map(|_| 0).collect();
        Self { min, max, buckets }
    }

    fn add(&mut self, x: f32) {
        let mut bucket = if x < self.min {
            0
        } else if x >= self.max {
            self.buckets.len() - 1
        } else {
            ((x - self.min) / (self.max - self.min) * (self.buckets.len() as f32)) as usize
        };
        bucket = bucket.max(0).min(self.buckets.len() - 1);
        self.buckets[bucket] += 1;
    }

    fn buckets(&self) -> Vec<(f32, u64)> {
        let step = (self.max - self.min) / (self.buckets.len() as f32);
        self.buckets.iter().enumerate().map(|(i, x)| (self.min + (i as f32) * step, *x)).collect()
    }
}

fn binarize(x: &[f32]) -> u64 {
    let mut hasher = seahash::SeaHasher::new();
    for i in 0..(x.len() / 8) {
        hasher.write_u8(((x[i * 8] > 0.0) as u8) + (((x[i * 8 + 1] > 0.0) as u8) << 1) + (((x[i * 8 + 2] > 0.0) as u8) << 2) + (((x[i * 8 + 3] > 0.0) as u8) << 3) + (((x[i * 8 + 4] > 0.0) as u8) << 4) + (((x[i * 8 + 5] > 0.0) as u8) << 5) + (((x[i * 8 + 6] > 0.0) as u8) << 6) + (((x[i * 8 + 7] > 0.0) as u8) << 7));
    }
    hasher.finish()
}

fn reader_thread(paths: &Vec<String>, tx: SyncSender<ProcessedEntry>) -> Result<()> {
    for path in paths {
        let stream = zstd::stream::Decoder::new(fs::File::open(path).context("read dump file")?)?;
        let mut stream = BufReader::new(stream);

        loop {
            let res: Result<ProcessedEntry, DecodeError> = rmp_serde::from_read(&mut stream);
            match res {
                Ok(x) => tx.send(x)?,
                Err(DecodeError::InvalidDataRead(x)) | Err(DecodeError::InvalidMarkerRead(x)) if x.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e).context("decode fail")
            }
        }
    }
    Ok(())
}

const SHARD_SPILL: usize = 2;
const RECORD_PAD_SIZE: usize = 4096; // NVMe disk sector size
const D_EMB: u32 = 1152;
const EMPTY_LOOKUP: (u32, u64, u32) = (u32::MAX, 0, 0);
const KNN_K: usize = 30;
const BALANCE_WEIGHT: f64 = 0.2;
const BATCH_SIZE: usize = 128;

fn main() -> Result<()> {
    let args: CLIArguments = argh::from_env();
    let mut rng = fastrand::Rng::with_seed(args.seed.unwrap_or(0));
    let mut latest_timestamp = DateTime::<Utc>::MIN_UTC;
    let mut earliest_timestamp = DateTime::<Utc>::MAX_UTC;
    let mut count = 0;
    let mut deduped_count = 0;

    // load specified embeddings from files
    let mut embeddings = Vec::new();
    for x in args.embedding {
        let (name, snd) = x.split_once(':').unwrap();
        let (path, threshold) = if let Some((path, threshold)) = snd.split_once(':') {
            (path, Some(threshold.parse::<f32>().context("parse threshold")?))
        } else {
            (snd, None)
        };
        let blob = fs::read(path).context("read embedding")?;
        embeddings.push((name.to_string(), common::decode_fp16_buffer(&blob), Histogram::new(-1.0, 1.0, 512), threshold));
    }

    let pq_codec = if let Some(pq_codec) = args.pq_codec {
        let data = fs::read(pq_codec).context("read pq codec")?;
        let pq_codec: ProductQuantizer = rmp_serde::from_read(&data[..]).context("decode pq codec")?;
        Some(pq_codec)
    } else {
        None
    };

    // construct FAISS index over query vectors for kNNs
    let (mut queries_index, max_query_id) = if let Some(queries_file) = args.queries {
        println!("constructing index");
        // not memory-efficient but this is small
        let data = fs::read(queries_file).context("read queries file")?;
        //let mut index = faiss::index_factory(D_EMB, "HNSW32,SQfp16", faiss::MetricType::InnerProduct)?;
        let mut index = faiss::index_factory(D_EMB, "HNSW32,SQfp16", faiss::MetricType::InnerProduct)?;
        //let mut index = faiss::index_factory(D_EMB, "IVF4096,SQfp16", faiss::MetricType::InnerProduct)?;
        let unpacked = common::decode_fp16_buffer(&data);
        index.train(&unpacked)?;
        index.add(&unpacked)?;
        println!("done");
        (Some(index), unpacked.len() / D_EMB as usize)
    } else {
        (None, 0)
    };

    // if sufficient config to split index exists, set up output files
    let mut shards_out = if let (Some(shards_dir), Some(centroids)) = (&args.shards_dir, &args.centroids) {
        let mut shards = Vec::new();
        let centroids_data = fs::read(centroids).context("read centroids file")?;
        let mut centroids_data = common::decode_fp16_buffer(&centroids_data);

        if let Some(clip) = args.clip_shards {
            centroids_data.truncate(clip * D_EMB as usize);
        }

        for i in 0..(centroids_data.len() / (D_EMB as usize)) {
            let centroid = centroids_data[i * (D_EMB as usize)..(i + 1) * (D_EMB as usize)].to_vec();
            let mut file = fs::File::create(PathBuf::from(shards_dir).join(format!("{}.shard.msgpack", i))).context("create shard file")?;
            rmp_serde::encode::write(&mut file, &ShardInputHeader { id: i as u32, centroid: centroid.clone(), max_query_id })?;
            shards.push((centroid, file, 0, i));
        }

        Some(shards)
    } else {
        None
    };

    // we can't fit all generated shards into RAM or they wouldn't be sharded anyway; keep file handles and locations lookup table
    let (mut read_out_vertices, shard_specs) = if let (Some(shards_dir), Some(_index_output)) = (&args.shards_dir, &args.index_output) {
        let mut original_ids_to_shards = Vec::new(); // locations in shard files of graph vertices: [(shard, offset, len)]
        let mut shard_id_mappings = Vec::new();
        let mut files = Vec::new();
        let mut shard_specs = Vec::new();

        // open shard files and build lookup from their header files
        for file in fs::read_dir(shards_dir)? {
            let file = file?;
            let path = file.path();
            let filename = path.file_name().unwrap().to_str().unwrap();
            let (fst, snd) = filename.split_once(".").unwrap();

            let id: u32 = str::parse(fst)?;
            if let Some(clip) = args.clip_shards {
                if id >= (clip as u32) {
                    continue;
                }
            }

            if snd == "shard-header.msgpack" {
                let header: ShardHeader = rmp_serde::from_read(BufReader::new(fs::File::open(path)?))?;
                if original_ids_to_shards.len() < (header.max as usize + 1) {
                    // probably somewhat inefficient, oh well
                    original_ids_to_shards.resize(header.max as usize + 1, [EMPTY_LOOKUP; SHARD_SPILL]);
                }
                for (i, &id) in header.mapping.iter().enumerate() {
                    let len = header.offsets[i + 1] - header.offsets[i]; // always valid, as we have a dummy entry at the end
                    let mut did_write = false;
                    // write location to next empty slot
                    //println!("{} {} {} {:?}", id, header.offsets[i], header.max, original_ids_to_shards[id as usize]);
                    for rec in original_ids_to_shards[id as usize].iter_mut() {
                        if *rec == EMPTY_LOOKUP {
                            *rec = (header.id, header.offsets[i], len as u32);
                            did_write = true;
                            break;
                        }
                    }
                    // each record should be in exactly SHARD_SPILL shards
                    if !did_write {
                        bail!("shard processing inconsistency");
                    }
                }

                shard_specs.push((header.centroid.clone(), header.mapping[header.medioid as usize]));

                shard_id_mappings.push((header.id, header.mapping));
            } else if snd == "shard.bin" {
                let file = fs::File::open(&path).context("open shard file")?;
                files.push((id, file));
            }
        }

        files.sort_by_key(|(id, _)| *id);
        shard_id_mappings.sort_by_key(|(id, _)| *id);


        let read_out_vertices = move |id: u32| -> Result<(Vec<u32>, Vec<u32>)> {
            let mut out_vertices: Vec<u32> = vec![];
            let mut shards: Vec<u32> = vec![];
            // look up each location in shard files
            for &(shard, offset, len) in original_ids_to_shards[id as usize].iter() {
                if (shard, offset, len) == EMPTY_LOOKUP {
                    continue;
                }

                shards.push(shard);
                let shard = shard as usize;
                // this random access is almost certainly rather slow
                // parallelize?
                files[shard].1.seek(SeekFrom::Start(offset))?;
                let mut buf = vec![0; len as usize];
                files[shard].1.read_exact(&mut buf)?;
                let s: &mut [u32] = bytemuck::cast_slice_mut(&mut *buf);
                for within_shard_id in s.iter_mut() {
                    *within_shard_id = shard_id_mappings[shard].1[*within_shard_id as usize];
                }
                out_vertices.extend(s.iter().unique());
            }

            Ok((out_vertices, shards))
        };

        (Some(read_out_vertices), Some(shard_specs))
    } else {
        (None, None)
    };

    let mut index_output_file = if let Some(index_output) = &args.index_output {
        let main_output = BufWriter::new(fs::File::create(PathBuf::from(index_output).join("index.bin")).context("create index file")?);
        let pq_codes =BufWriter::new(fs::File::create(PathBuf::from(index_output).join("index.pq-codes.bin")).context("create index file")?);
        Some((main_output, pq_codes))
    } else {
        None
    };

    let mut output_file = args.output_embeddings.map(|x| fs::File::create(x).context("create output file")).transpose()?;

    let mut i: u64 = 0;

    let mut dedupe_ring: VecDeque<u64> = VecDeque::with_capacity(2<<20);
    let mut dedupe_hashset: HashSet<u64> = HashSet::with_capacity(2<<21);
    let mut dedupe_url_ring: VecDeque<u64> = VecDeque::with_capacity(2<<20);
    let mut dedupe_url_hashset: HashSet<u64> = HashSet::with_capacity(2<<21);

    let (tx, rx) = sync_channel(1024);

    let th = std::thread::spawn(move || reader_thread(&args.paths, tx));

    let mut rng2 = rng.fork();
    let initial_filter = |x: ProcessedEntry| {
        i += 1;

        if args.sample.is_some() && rng2.f32() > args.sample.unwrap() {
            return None;
        }
        let timestamp = Utc.timestamp_opt(x.timestamp as i64, 0).unwrap();

        let embedding = common::decode_fp16_buffer(&x.embedding);

        latest_timestamp = latest_timestamp.max(timestamp);
        earliest_timestamp = earliest_timestamp.min(timestamp);

        for (_name, vec, histogram, threshold) in &mut embeddings {
            let dot = SpatialSimilarity::dot(&embedding, vec).unwrap() as f32;
            histogram.add(dot);
            if let Some(threshold) = threshold {
                if dot >= *threshold {
                    return None;
                }
            }
        }

        // distance thresholding is too costly to do over a long range so just do it badly
        if args.deduplicate {
            let code = binarize(&embedding);
            let mut hasher = seahash::SeaHasher::new();
            hasher.write(&x.metadata.final_url.as_bytes());
            let url_code = hasher.finish();
            if dedupe_ring.len() == dedupe_ring.capacity() {
                dedupe_ring.pop_front().unwrap();
                dedupe_url_ring.pop_front().unwrap();
            }
            dedupe_ring.push_back(code);
            dedupe_url_ring.push_back(url_code);
            if dedupe_hashset.insert(code) == false || dedupe_url_hashset.insert(url_code) == false {
                deduped_count += 1;
                return None;
            }
        }

        if args.print_records {
            println!("{} {} https://reddit.com/r/{}/comments/{} {}", timestamp, x.title, x.subreddit, x.id, x.metadata.final_url);
        }
        if args.original_url {
            println!("{}", x.url);
        }
        if args.titles {
            println!("{}", x.title);
        }
        if args.print_embeddings {
            println!("https://mse.osmarks.net/?e={}", base64::engine::general_purpose::URL_SAFE.encode(&x.embedding));
        }

        Some((x, embedding))
    };

    let mut dead_count = 0;

    let mut bal_count = 1;

    for batch in &rx.iter().filter_map(initial_filter).chunks(BATCH_SIZE) {
        let batch: Vec<_> = batch.collect();
        let batch_len = batch.len();

        for (x, _embedding) in batch.iter() {
            if let Some(ref mut file) = output_file {
                file.write_all(&x.embedding)?;
            }
        }

        if let Some(shards) = &mut shards_out {
            let mut knn_query = vec![];
            for (_, embedding) in batch.iter() {
                knn_query.extend(embedding);
            }

            let index = queries_index.as_mut().context("need queries")?;
            let knn_result = index.search(&knn_query, KNN_K)?;

            for (i, (x, embedding)) in batch.iter().enumerate() {
                // closest matches first
                shards.sort_by_cached_key(|&(ref centroid, _, shard_count, _shard_index)| {
                    let mut dot = SpatialSimilarity::dot(&centroid, &embedding).unwrap();
                    dot -= BALANCE_WEIGHT * (shard_count as f64 / bal_count as f64);
                    -scale_dot_result_f64(dot)
                });

                let entry = ShardedRecord {
                    id: count + i as u32,
                    vector: x.embedding.clone(),
                    query_knns: knn_result.labels[i * KNN_K..(i + 1)*KNN_K].into_iter().map(|x| x.get().unwrap() as u32).collect()
                };
                let data = rmp_serde::to_vec(&entry)?;
                for (_, file, shard_count, _shard_index) in shards[0..SHARD_SPILL].iter_mut() {
                    file.write_all(&data)?;
                    *shard_count += 1;
                }

                bal_count += 1;
                // it is possible that using the count which is updated at the end of the batch leads to confusing numerics issues
                // also, this one starts at 1, so we avoid a division by zero on the first one
            }
        }

        if let (Some(read_out_vertices), Some(index_output_file)) = (&mut read_out_vertices, &mut index_output_file) {
            let quantizer = pq_codec.as_ref().unwrap();

            let mut batch_embeddings = Vec::with_capacity(batch.len() * D_EMB as usize);
            for (_x, embedding) in batch.iter() {
                batch_embeddings.extend_from_slice(&embedding);
            }
            let codes = quantizer.quantize_batch(&batch_embeddings);

            for (i, (x, _embedding)) in batch.into_iter().enumerate() {
                let (vertices, shards) = read_out_vertices(count)?; // TODO: could parallelize this given the batching
                let mut entry = PackedIndexEntry {
                    id: count + i as u32,
                    vertices,
                    vector: x.embedding.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect(),
                    timestamp: x.timestamp,
                    dimensions: x.metadata.dimension,
                    score: 0.5, // TODO
                    url: x.metadata.final_url,
                    shards
                };
                let mut bytes = bitcode::encode(&entry);
                if bytes.len() > (RECORD_PAD_SIZE - 2) {
                    // we do need the records to fit in a fixed size and can't really drop things, so discard URL so it can exist as a graph node only
                    entry.url = String::new();
                    bytes = bitcode::encode(&entry);
                    dead_count += 1;
                }
                let len = bytes.len() as u16;
                bytes.resize(RECORD_PAD_SIZE - 2, 0);
                index_output_file.0.write_all(&u16::to_le_bytes(len))?;
                index_output_file.0.write_all(&bytes)?;
            }
            index_output_file.1.write_all(&codes)?;
        }

        count += batch_len as u32;
    }

    if args.print_aggregates {
        println!("earliest={} latest={} count={} read={} deduped={}", earliest_timestamp, latest_timestamp, count, i, deduped_count);
    }
    if let Some(histogram_path) = args.histograms {
        let mut file = fs::File::create(histogram_path)?;
        for (name, _, histogram, _) in &embeddings {
            let width = 800.0;
            let padding = 40.0;
            let bars_height = 300 as f64;
            let buckets = histogram.buckets();
            let max_count = *buckets.iter().map(|(_max, count)| count).max().unwrap();
            let bar_width = width / buckets.len() as f64;
            let plot = maud::html! {
                h1 { (name) }
                svg style="border: 1px solid gray;" viewBox=(format!("{} 0 {} {}", -padding * 0.25, width + (padding * 0.75), bars_height + 50.0)) xmlns="http://www.w3.org/2000/svg" width=(format!("{}", width + padding)) height=(format!("{}", bars_height + 50.0)) {
                    @for (i, (min, count)) in buckets.into_iter().enumerate() {
                        @let height = bars_height * (count as f64 / max_count as f64);
                        rect width=(format!("{}", bar_width)) x=(format!("{}", bar_width * i as f64)) height=(format!("{}", height)) y=(format!("{}", bars_height - height)) {
                            title {
                                (format!("{} {}", min, count))
                            }
                        }
                    }
                }
            };
            file.write_all(plot.into_string().as_bytes())?;
        }
    }

    if let Some(index_output) = &args.index_output {
        let mut file = fs::File::create(PathBuf::from(index_output).join("index.msgpack"))?;
        let header = IndexHeader {
            shards: shard_specs.unwrap(),
            count: count as u32,
            record_pad_size: RECORD_PAD_SIZE,
            dead_count,
            quantizer: pq_codec.unwrap()
        };
        file.write_all(rmp_serde::to_vec_named(&header)?.as_slice())?;
    }

    if let Some(shards) = &mut shards_out {
        for (_centroid, _file, count, index) in shards.iter_mut() {
            println!("shard {}: {} records", index, count);
        }
    }

    th.join().unwrap()?;

    Ok(())
}
