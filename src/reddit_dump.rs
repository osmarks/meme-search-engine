use anyhow::{anyhow, Context, Result};
use common::resize_for_embed;
use itertools::Itertools;
use std::{collections::HashSet, ffi::OsStr, fs::{self, read_dir}, io::{BufRead, BufReader, BufWriter, Cursor}, path::PathBuf, str::FromStr, sync::Arc, time::Duration};
use serde::{Serialize, Deserialize};
use lazy_static::lazy_static;
use regex::{bytes, Regex, RegexSet, RegexSetBuilder};
use tokio::{sync::{mpsc::{self, Receiver}, Semaphore}, task::{JoinHandle, JoinSet}};
use tokio_stream::wrappers::ReceiverStream;
use reqwest::Client;
use futures_util::stream::{StreamExt, TryStreamExt};
use image::{DynamicImage, ImageReader};
use mimalloc::MiMalloc;
use tracing::instrument;
use prometheus::{Encoder, register_int_counter, IntCounter, register_histogram_vec, HistogramVec};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod common;

use crate::common::{get_backend_config, query_clip_server, EmbeddingRequest};

fn function_which_returns_some_na() -> Option<String> { Some(String::from("na")) }

#[derive(Clone, Deserialize, Serialize, Debug)]
#[serde(untagged)]
enum BadTimestampFormat {
    Int(u64),
    String(String)
}

impl BadTimestampFormat {
    fn to_u64(&self) -> Result<u64> {
        match self {
            BadTimestampFormat::Int(x) => Ok(*x),
            BadTimestampFormat::String(x) => u64::from_str(&x).context("invalid string")
        }
    }
}

#[derive(Clone, Deserialize, Serialize, Debug)]
struct Entry {
    url: String,
    over_18: bool,
    title: String,
    author: Option<String>,
    selftext: String,
    subreddit: Option<String>,
    created_utc: BadTimestampFormat,
    #[serde(default="function_which_returns_some_na")]
    post_hint: Option<String>,
    id: String
}

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
    #[serde(with = "serde_bytes")]
    embedding: Vec<u8>,
    metadata: OriginalImageMetadata
}

lazy_static! {
    // we do exclude galleries doing this but there don't seem to be any in the dataset
    static ref URL_IGNORE: RegexSet = RegexSet::new([
        r"//reddit\.com",
        r"\.html?",
        r"\.php",
        r"\?articleid=",
        r"\.aspx?",
        r"\.xml",
        r"/rss/",
        r"//vimeo\.com",
        r"//www\.reddit\.com",
        r"//v\.redd\.it",
        r"\.gifv$",
        r"youtube\.com/user/"
        // TODO fill in more things, maybe try and collect thumbnails or something
    ]).unwrap();
    static ref URL_MUST_CONTAIN: RegexSet = RegexSetBuilder::new([
        "jpg",
        "jpeg",
        "png",
        "webp",
        r"\.gif",
        "=gif",
        "jpeg",
        "bmp",
        "tiff",
        "avif",
        "webp",
        "imgur",
        "image",
        r"//i\.",
        "img",
        r"cdn\.",
        r"media\.",
        "/i/",
        "/media",
        r"youtu\.be",
        r"youtube\.com",
    ]).case_insensitive(true).build().unwrap();
    static ref ACCEPTABLE_FILETYPES: HashSet<&'static str> = ["image/png", "image/webp", "image/avif", "image/jpeg", "image/gif", "image/webp", "image/apng", "image/bmp", "image/tiff"]
        .into_iter().collect();
    static ref OBJECT_HACKY_IGNORE: bytes::RegexSet = bytes::RegexSet::new([
        r#""author":"\[deleted\]""#,
        r#""promoted":true"#, // these seem to be ads which are in the data for some reason, and lack some important fields
        r#""domain":"self.promos""#, // .......
        r"\x00" // for SOME REASON one of the JSON files contains a lot of null bytes before one particular record, so just ignore that record
    ]).unwrap();
    static ref URL_REPLACEMENT_RULES: Vec<(Regex, &'static str)> = [
        (r"imgur\.com/([A-Za-z0-9]+),", r"imgur.com/$1"),
        (r"//imgur\.com/([A-Za-z0-9]+)$", r"//i.imgur.com/$1.jpg"),
        (r"//www\.imgur\.com/([A-Za-z0-9]+)$", r"//i.imgur.com/$1.jpg"),
        (r"//m\.imgur\.com/([A-Za-z0-9]+)$", r"//i.imgur.com/$1.jpg"),
        (r"^http://", r"https://"),
        (r"//youtu\.be/(.*)", r"//youtube.com/watch?v=$1"),
        (r"//[a-z]+\.youtube\.com/(.*)", r"//youtube.com/$1"),
        (r"//www.youtube.com/attribution_link?.*v%3D([A-Za-z0-9_-]+).*", r"//i.ytimg.com/vi/$1/maxresdefault.jpg"), // redirect to youtube thumbnail API
        (r"//youtube.com/embed/([A-Za-z0-9_-]+)", r"//i.ytimg.com/vi/$1/maxresdefault.jpg"),
        (r"//youtube\.com/(?:.*)v=([A-Za-z0-9_-]+)(?:.*)", r"//i.ytimg.com/vi/$1/maxresdefault.jpg"),
        (r"&amp;", "&") // this is such an intensely cursed feature of the dumps
    ].into_iter().map(|(r, e)| (Regex::new(r).unwrap(), e)).collect();

    static ref HTML_EXTRACTION_RULES: Vec<(Regex, Regex)> = [
        (r"//imgur\.com/a/[A-Za-z0-9]+", r#"<meta name="twitter:image" data-react-helmet="true" content="([^"]+)">"#),
        (r"//imgur\.com/gallery/[A-Za-z0-9]+", r#"<meta name="twitter:image" data-react-helmet="true" content="([^"]+)">"#),
    ].into_iter().map(|(r, e)| (Regex::new(r).unwrap(), Regex::new(e).unwrap())).collect();

    static ref IMAGES_FETCHED_COUNTER: IntCounter = register_int_counter!("mse_scrape_images_fetched", "images fetched").unwrap();
    static ref IMAGES_PROCESSED_COUNTER: IntCounter = register_int_counter!("mse_scrape_images_processed", "images processed").unwrap();
    static ref ENTRIES_PROCESSED_COUNTER: IntCounter = register_int_counter!("mse_scrape_entries_processed", "entries processed").unwrap();
    static ref IMAGES_FAILED_COUNTER: IntCounter = register_int_counter!("mse_scrape_images_failed", "images failed").unwrap();
    static ref IMAGE_FILESIZES_HISTOGRAM: HistogramVec = register_histogram_vec!("mse_scrape_image_filesizes", "filesizes of successfully fetched images", &["format"], prometheus::exponential_buckets(100.0, 1.5, 29).unwrap()).unwrap();
    static ref IMAGE_PIXELS_HISTOGRAM: HistogramVec = register_histogram_vec!("mse_scrape_image_pixels", "pixel count of successfully fetched images", &["format"], prometheus::exponential_buckets(100.0, 1.3, 53).unwrap()).unwrap();
    static ref HTML_EXTRACTS_COUNTER: IntCounter = register_int_counter!("mse_scrape_html_extracts", "html extraction operations").unwrap();
}

#[instrument(skip(tx))]
fn process_file(path: PathBuf, tx: mpsc::Sender<Entry>, timestamp_threshold: Option<u64>) -> Result<()> {
    let mut stream = zstd::stream::Decoder::new(fs::File::open(path)?)?;
    stream.window_log_max(31)?;
    let mut stream = BufReader::new(stream);
    let mut buf = Vec::new();
    loop {
        if stream.read_until(0x0A, &mut buf)? == 0 {
            break
        }
        // we would discard these later, but they have a different format so they can't be deserialized straight into Entries
        if OBJECT_HACKY_IGNORE.is_match(&buf) {
            buf.clear();
            continue;
        }
        ENTRIES_PROCESSED_COUNTER.inc();
        let entry = match sonic_rs::serde::from_slice::<Entry>(buf.as_slice()) {
            Ok(x) => x,
            Err(e) => {
                tracing::warn!("parse failed, please validate {:?} {:?}", e, String::from_utf8_lossy(&buf));
                return Ok(())
            }
        };
        if entry.selftext.is_empty() && !entry.over_18 && entry.author.is_some() && entry.subreddit.is_some() {
            if !URL_IGNORE.is_match(&entry.url) && URL_MUST_CONTAIN.is_match(&entry.url) {
                match &entry.post_hint {
                    Some(x) if x == "na" || x == "image" => {
                        // Technically this is slightly wrong because we reorder images slightly, but as long as it is not restarted all the time this is "fine".
                        let after_threshold = match timestamp_threshold {
                            Some(threshold) => {
                                let timestamp = match &entry.created_utc {
                                    BadTimestampFormat::Int(x) => *x,
                                    BadTimestampFormat::String(s) => u64::from_str(s).unwrap()
                                };
                                timestamp > threshold
                            },
                            None => true
                        };

                        if after_threshold { tx.blocking_send(entry)?; }
                    },
                    _ => ()
                }
            }
        }
        buf.clear();
    }
    Ok(())
}

#[derive(Debug)]
struct Config {
    max_content_length: usize,
    input: String,
    output: String,
    backend: String,
    mode: OperatingMode,
    filename_threshold: Option<String>,
    metrics_addr: String,
    contact_info: String
}

#[instrument(skip(client, config))]
#[async_recursion::async_recursion]
async fn fetch_file(client: reqwest::Client, config: Arc<Config>, url: &str) -> Result<(Vec<u8>, String, String)> {
    let mut url = url.to_string();
    for (regex, replacement) in URL_REPLACEMENT_RULES.iter() {
        url = regex.replace(&url, *replacement).to_string();
    }

    let mut html_extract_rule = None;

    for (url_rule, extract_rule) in HTML_EXTRACTION_RULES.iter() {
        if url_rule.is_match(&url) {
            html_extract_rule = Some(extract_rule);
            break;
        }
    }

    let mut response = client.get(&*url).send().await?;
    let content_type = std::str::from_utf8(&response.headers().get(reqwest::header::CONTENT_TYPE).context("no content type")?.as_bytes())?.to_owned();
    if !(ACCEPTABLE_FILETYPES.contains(&content_type[..]) || (html_extract_rule.is_some() && content_type == "text/html")) {
        return Err(anyhow!("invalid Content-Type"));
    }
    match response.content_length() {
        Some(x) if x > (config.max_content_length as u64) => return Err(anyhow!("response too large")),
        _ => ()
    }
    let mut buffer = vec![];
    while let Some(chunk) = response.chunk().await? {
        buffer.extend(chunk);
        if buffer.len() > config.max_content_length {
            return Err(anyhow!("response too large"));
        }
    }
    if let Some(extract_rule) = html_extract_rule {
        if content_type == "text/html" {
            let buffer = String::from_utf8_lossy(&buffer).to_string();
            if let Some(mat) = extract_rule.captures(&buffer) {
                let new_url = mat.get(1).unwrap().as_str();
                HTML_EXTRACTS_COUNTER.inc();
                tracing::debug!("found new URL: {}", new_url);
                return fetch_file(client, config, new_url).await;
            } else {
                return Err(anyhow!("no extraction match"));
            }
        }
    }
    Ok((buffer, content_type, response.url().to_string()))
}

fn write_output(config: Arc<Config>, mut rx: Receiver<ProcessedEntry>, seqnum: usize) -> Result<()> {
    let mut out = fs::File::create(PathBuf::from(&config.output).join(format!("{}.dump-zst", seqnum)))?;
    let stream = zstd::Encoder::new(&mut out, 15)?.auto_finish();
    let mut buf_stream = BufWriter::new(stream);
    while let Some(x) = rx.blocking_recv() {
        rmp_serde::encode::write(&mut buf_stream, &x)?;
    }
    Ok(())
}

#[derive(Debug)]
enum OperatingMode {
    Count,
    Sample(f32),
    FullRun
}

#[instrument]
fn readback_output(path: &str) -> Result<(u64, usize, usize)> {
    use rmp_serde::decode::Error;

    let mut highest_seqnum: Option<usize> = None;
    for entry in read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(OsStr::to_str).map(|x| x == "dump-zst").unwrap_or(false) {
            let seqnum = path.file_stem().context("invalid file structure")?.to_str().context("non-UTF8 path")?.parse::<usize>().context("invalid file name")?;
            highest_seqnum = Some(highest_seqnum.map(|x| x.max(seqnum)).unwrap_or(seqnum));
        }
    }

    let seqnum = highest_seqnum.context("no files found")?;

    let stream = zstd::stream::Decoder::new(fs::File::open(PathBuf::from(path).join(&format!("{}.dump-zst", seqnum)))?)?;
    let mut stream = BufReader::new(stream);
    let mut latest_timestamp = 0;
    let mut count = 0;
    loop {
        let res: Result<ProcessedEntry, Error> = rmp_serde::from_read(&mut stream);
        if res.is_ok() {
            count += 1;
        }
        match res {
            Ok(x) => latest_timestamp = latest_timestamp.max(x.timestamp),
            Err(Error::InvalidDataRead(x)) | Err(Error::InvalidMarkerRead(x)) if x.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e).context("decode fail")
        }
    }
    Ok((latest_timestamp, count, seqnum))
}

async fn serve_metrics(config: Arc<Config>) -> Result<()> {
    let metrics = axum::Router::new().route("/metrics", axum::routing::get(|| async move {
        let mut buffer = Vec::new();
        let encoder = prometheus::TextEncoder::new();
        let metric_families = prometheus::gather();
        encoder.encode(&metric_families, &mut buffer).unwrap();
        buffer
    }));
    let listener = tokio::net::TcpListener::bind(&config.metrics_addr).await?;
    tokio::task::spawn(async move {
        let _ = axum::serve(listener, metrics).await;
    });
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    console_subscriber::init();

    let cpus = num_cpus::get();

    let config = Arc::new(Config {
        max_content_length: 1<<24,
        input: String::from("./reddit_subs_202212/"),
        output: String::from("."),
        backend: String::from("http://localhost:1708"),
        mode: OperatingMode::FullRun,
        filename_threshold: Some(String::from("RS_2017-08.zst")),
        metrics_addr: String::from("0.0.0.0:9914"),
        contact_info: String::from("scraping-ops@osmarks.net")
    });

    serve_metrics(config.clone()).await?;

    let timestamp_threshold = match config.mode {
        OperatingMode::Count => None,
        _ => {
            match readback_output(&config.output) {
                Ok(x) => Some(x),
                Err(e) => {
                    tracing::warn!("could not read output: {}", e);
                    None
                }
            }
        }
    };

    let mut seqnum = 0;
    if let Some((threshold, count, existing_seqnum)) = timestamp_threshold {
        tracing::info!("threshold is {}, {} items, seq {}", threshold, count, existing_seqnum);
        seqnum = existing_seqnum + 1;
    }

    let backend = get_backend_config(&config.backend).await;

    tracing::info!("connected to inference server");

    let (entries_tx, mut entries_rx) = mpsc::channel::<Entry>(32768);
    let (buffers_tx, buffers_rx) = mpsc::channel(128);
    let (resized_tx, resized_rx) = mpsc::channel(backend.batch);
    let (final_write_tx, final_write_rx) = mpsc::channel::<ProcessedEntry>(32768);
    let client = Client::builder()
        .user_agent(format!("{}/{} (contact {})", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"), config.contact_info))
        .timeout(Duration::from_secs(30))
        .build()?;

    let load_task: JoinHandle<Result<()>> = match config.mode {
        OperatingMode::Count => tokio::task::spawn(async move {
            let mut counter = 0;
            while let Some(_) = entries_rx.recv().await {
                counter += 1
            }
            println!("{}", counter);
            Ok(())
        }),
        _ => tokio::task::spawn({
            let client = client.clone();
            let config = config.clone();
            let stream = ReceiverStream::new(entries_rx);
            stream.map(Ok).try_for_each_concurrent(Some(512), move |entry| {
                let client = client.clone();
                let config = config.clone();
                let buffers_tx = buffers_tx.clone();
                async move {
                    if let OperatingMode::Sample(rate) = config.mode {
                        if fastrand::f32() > rate {
                            return Ok(())
                        }
                    }
                    match fetch_file(client, config.clone(), &entry.url).await {
                        Ok(buf) => {
                            IMAGES_FETCHED_COUNTER.inc();
                            tracing::debug!("got {}", &entry.url);
                            buffers_tx.send((entry, buf)).await?;
                        },
                        Err(e) => {
                            IMAGES_FAILED_COUNTER.inc();
                            tracing::debug!("{} failed: {}", &entry.url, e)
                        }
                    }
                    Ok(())
                }
            }
        )})
    };

    let resize_task = match config.mode {
        OperatingMode::Count => None,
        _ => Some(tokio::task::spawn({
            let stream = ReceiverStream::new(buffers_rx);
            let backend = backend.clone();
            stream.map(Ok).try_for_each_concurrent(Some(cpus), move |(entry, (buffer, mime_type, final_url))| {
                let backend = backend.clone();
                let size = buffer.len();
                IMAGE_FILESIZES_HISTOGRAM.with_label_values(&[&mime_type]).observe(size as f64);
                let resized_tx = resized_tx.clone();
                async move {
                    let image_result = tokio::task::spawn_blocking(|| {
                        let csr = Cursor::new(buffer);
                        let image = ImageReader::new(csr).with_guessed_format()?.decode()?;
                        Result::<DynamicImage, anyhow::Error>::Ok(image)
                    }).await?;
                    let image = match image_result {
                        Ok(image) => image,
                        Err(e) => {
                            tracing::debug!("loading {} failed: {}", entry.url, e);
                            return Result::<(), anyhow::Error>::Ok(());
                        }
                    };
                    let dim = (image.width(), image.height());
                    IMAGE_PIXELS_HISTOGRAM.with_label_values(&[&mime_type]).observe(dim.0 as f64 * dim.1 as f64);
                    let metadata = OriginalImageMetadata {
                        mime_type,
                        original_file_size: size,
                        dimension: dim,
                        final_url
                    };
                    let resized = resize_for_embed(backend.clone(), image).await?;
                    resized_tx.send((entry, resized, metadata)).await?;
                    Ok(())
                }
            })
        }))
    };

    let embedding_generation_task: Option<JoinHandle<Result<()>>> = match config.mode {
        OperatingMode::Count => None,
        _ => Some(tokio::spawn({
            let stream = ReceiverStream::new(resized_rx).chunks(backend.batch);
            let client = client.clone();
            let config = config.clone();
            // keep multiple embedding requests in flight
            stream.map(Ok).try_for_each_concurrent(Some(3), move |batch| {
                let (entries, bytes, batch_dimensions): (Vec<Entry>, Vec<Vec<u8>>, Vec<OriginalImageMetadata>) = batch.into_iter().multiunzip();
                let client = client.clone();
                let config = config.clone();
                let final_write_tx = final_write_tx.clone();
                async move {
                    let result: Vec<serde_bytes::ByteBuf> = query_clip_server(
                        &client,
                        &config.backend,
                        "",
                        EmbeddingRequest::Images {
                            images: bytes.into_iter().map(serde_bytes::ByteBuf::from).collect(),
                        },
                    ).await.context("querying CLIP server")?;

                    for (vector, entry,
                        metadata) in itertools::izip!(result.into_iter(), entries, batch_dimensions) {
                        final_write_tx.send(ProcessedEntry {
                            url: entry.url,
                            id: entry.id,
                            title: entry.title,
                            subreddit: entry.subreddit.unwrap(),
                            author: entry.author.unwrap(),
                            embedding: vector.into_vec(),
                            timestamp: entry.created_utc.to_u64()?,
                            metadata
                        }).await?;
                        IMAGES_PROCESSED_COUNTER.inc();
                    }
                    anyhow::Result::Ok(())
                }
            })
        }))
    };

    let config_ = config.clone();
    let output_writer_task = match config.mode {
        OperatingMode::Sample(_) | OperatingMode::FullRun => Some(tokio::task::spawn_blocking(move || write_output(config_, final_write_rx, seqnum))),
        _ => None
    };

    tracing::info!("working...");

    let mut paths = vec![];
    for file in fs::read_dir(&config.input)? {
        let path = file?.path();
        let last_segment = path.file_name().context("invalid file structure")?.to_str().context("non-UTF8 path")?.to_string();
        match &config.filename_threshold {
            Some(threshold) if threshold >= &last_segment => (),
            _ => paths.push(path)
        }
    }

    paths.sort();

    let mut file_readers = JoinSet::new();

    let readers = match config.mode {
        OperatingMode::Count | OperatingMode::Sample(_) => cpus,
        OperatingMode::FullRun => 1
    };

    let semaphore = Arc::new(Semaphore::new(readers));

    for path in paths {
        let semaphore = semaphore.clone();
        let permit = semaphore.acquire_owned().await?;
        let entries_tx = entries_tx.clone();
        let path_ = path.clone();
        tracing::info!("reading {:?}", path);
        file_readers.spawn_blocking(move || {
            match process_file(path_, entries_tx, timestamp_threshold.map(|(x, _, _)| x)) {
                Ok(_) => (),
                Err(e) => tracing::error!("could not parse {:?} {:?}", &path, e)
            }
            std::mem::drop(permit);
        });
    }

    while let Some(x) = file_readers.try_join_next() {
        x?;
    }

    std::mem::drop(entries_tx);
    println!("{:?}", load_task.await?);
    if let Some(task) = resize_task { println!("resize: {:?}", task.await?); }
    if let Some(task) = embedding_generation_task { println!("embedding: {:?}", task.await?) };
    if let Some(task) = output_writer_task { println!("output: {:?}", task.await?) };

    Ok(())
}
