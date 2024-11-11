use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use std::io::{BufReader, Write};
use rmp_serde::decode::Error as DecodeError;
use std::fs;
use base64::Engine;
use argh::FromArgs;
use chrono::{TimeZone, Utc, DateTime};
use std::collections::{VecDeque, HashSet};
use std::hash::Hasher;

mod common;

// TODO refactor
#[derive(Clone, Deserialize, Serialize, Debug, PartialEq)]
struct OriginalImageMetadata {
    mime_type: String,
    original_file_size: usize,
    dimension: (u32, u32),
    final_url: String
}

#[derive(Clone, Deserialize, Serialize, Debug)]
struct ProcessedEntry {
    url: String,
    id: String,
    title: String,
    subreddit: String,
    author: String,
    timestamp: u64,
    #[serde(with="serde_bytes")]
    embedding: Vec<u8>,
    metadata: OriginalImageMetadata
}

#[derive(FromArgs)]
#[argh(description="Process scraper dump files")]
struct CLIArguments {
    #[argh(option, short='s', description="read subset of records")]
    sample: Option<f32>,
    #[argh(switch, short='p', description="print basic information for records")]
    print_records: bool,
    #[argh(switch, short='e',description="print embeddings")]
    print_embeddings: bool,
    #[argh(switch, short='a', description="print aggregates")]
    print_aggregates: bool,
    #[argh(option, short='E', description="x:y - load embedding named x from file y")]
    embedding: Vec<String>,
    #[argh(option, short='H', description="path for histograms of dot with embeddings")]
    histograms: Option<String>,
    #[argh(switch, short='D', description="enable deduplicator")]
    deduplicate: bool,
    #[argh(option, short='T', description="deduplication Hamming distance threshold")]
    threshold: Option<u64>,
    #[argh(positional)]
    paths: Vec<String>
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
        let bucket = if x < self.min {
            0
        } else if x >= self.max {
            self.buckets.len() - 1
        } else {
            ((x - self.min) / (self.max - self.min) * (self.buckets.len() as f32)) as usize
        };
        self.buckets[bucket] += 1;
    }

    fn buckets(&self) -> Vec<(f32, u64)> {
        let step = (self.max - self.min) / (self.buckets.len() as f32);
        self.buckets.iter().enumerate().map(|(i, x)| (self.min + (i as f32) * step, *x)).collect()
    }
}

fn dot(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y).map(|(a, b)| a * b).sum::<f32>()
}

fn binarize(x: &[f32]) -> Vec<u8> {
    let mut buf = vec![0; x.len() / 8];
    for i in 0..(x.len() / 8) {
        buf[i] = ((x[i * 8] > 0.0) as u8) + (((x[i * 8 + 1] > 0.0) as u8) << 1) + (((x[i * 8 + 2] > 0.0) as u8) << 2) + (((x[i * 8 + 3] > 0.0) as u8) << 3) + (((x[i * 8 + 4] > 0.0) as u8) << 4) + (((x[i * 8 + 5] > 0.0) as u8) << 5) + (((x[i * 8 + 6] > 0.0) as u8) << 6) + (((x[i * 8 + 7] > 0.0) as u8) << 7);
    }
    buf
}

fn main() -> Result<()> {
    let args: CLIArguments = argh::from_env();
    let mut rng = fastrand::Rng::new();
    let mut latest_timestamp = DateTime::<Utc>::MIN_UTC;
    let mut earliest_timestamp = DateTime::<Utc>::MAX_UTC;
    let mut count = 0;
    let mut deduped_count = 0;
    let mut embeddings = Vec::new();
    for x in args.embedding {
        let (name, path) = x.split_once(':').unwrap();
        let blob = std::fs::read(path).context("read embedding")?;
        embeddings.push((name.to_string(), common::decode_fp16_buffer(&blob), Histogram::new(-1.0, 1.0, 512)));
    }

    // TODO ring of vecs probably has bad cache locality
    let mut dedupe_ring: VecDeque<Vec<u8>> = VecDeque::with_capacity(2<<10);
    let threshold = args.threshold.unwrap_or(3);

    for path in args.paths {
        let stream = zstd::stream::Decoder::new(fs::File::open(path).context("read dump file")?)?;
        let mut stream = BufReader::new(stream);

        loop {
            let res: Result<ProcessedEntry, DecodeError> = rmp_serde::from_read(&mut stream);
            if res.is_ok() {
                count += 1;
            }
            match res {
                Ok(x) => {
                    if args.sample.is_some() && rng.f32() > args.sample.unwrap() {
                        continue;
                    }
                    let timestamp = Utc.timestamp_opt(x.timestamp as i64, 0).unwrap();

                    let embedding = common::decode_fp16_buffer(&x.embedding);

                    latest_timestamp = latest_timestamp.max(timestamp);
                    earliest_timestamp = earliest_timestamp.min(timestamp);

                    if args.deduplicate {
                        let code = binarize(&embedding);
                        if dedupe_ring.len() == dedupe_ring.capacity() {
                            dedupe_ring.pop_front().unwrap();
                        }
                        let has_match = dedupe_ring.iter().any(|x| hamming::distance(x, &code) <= threshold);
                        dedupe_ring.push_back(code);
                        if has_match {
                            deduped_count += 1;
                            continue;
                        }
                    }

                    if args.print_records {
                        println!("{} {} https://reddit.com/r/{}/comments/{} {}", timestamp, x.title, x.subreddit, x.id, x.metadata.final_url);
                    }
                    if args.print_embeddings {
                        println!("https://mse.osmarks.net/?e={}", base64::engine::general_purpose::URL_SAFE.encode(&x.embedding));
                    }
                    for (_name, vec, histogram) in &mut embeddings {
                        let dot = dot(&embedding, vec);
                        histogram.add(dot);
                    }
                },
                Err(DecodeError::InvalidDataRead(x)) | Err(DecodeError::InvalidMarkerRead(x)) if x.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e).context("decode fail")
            }
        }
    }

    if args.print_aggregates {
        println!("earliest={} latest={} count={} deduped={}", earliest_timestamp, latest_timestamp, count, deduped_count);
    }
    if let Some(histogram_path) = args.histograms {
        let mut file = std::fs::File::create(histogram_path)?;
        for (name, _, histogram) in &embeddings {
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
    Ok(())
}
