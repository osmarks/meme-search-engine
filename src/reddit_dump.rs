use anyhow::{anyhow, Context, Result};
use common::resize_for_embed;
use std::{collections::HashSet, fs, io::{BufReader, Cursor, BufRead, BufWriter}, time::Duration, sync::Arc, str::FromStr, path::PathBuf};
use serde::{Serialize, Deserialize};
use lazy_static::lazy_static;
use regex::{RegexSet, bytes, Regex};
use tokio::{sync::{mpsc::{self, Receiver}, Semaphore}, task::{JoinHandle, JoinSet}};
use tokio_stream::wrappers::ReceiverStream;
use reqwest::Client;
use futures_util::stream::{StreamExt, TryStreamExt};
use image::{DynamicImage, io::Reader as ImageReader};
use mimalloc::MiMalloc;

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

#[derive(Clone, Deserialize, Serialize, Debug)]
struct ProcessedEntry {
    url: String,
    id: String,
    title: String,
    subreddit: String,
    author: String,
    timestamp: u64,
    blob: Vec<u8>
}

lazy_static! {
    static ref URL_IGNORE: RegexSet = RegexSet::new([
        r"//reddit\.com",
        r"\.html?",
        r"\.php",
        r"\?articleid=",
        r"\.aspx?",
        r"\.xml",
        r"//youtube\.com",
        r"/rss/",
        r"//vimeo\.com",
        r"//www\.youtube\.com",
        r"//youtu\.be",
        r"//www\.reddit\.com",
        // TODO fill in more things, maybe try and collect thumbnails or something
    ]).unwrap();
    static ref ACCEPTABLE_FILETYPES: HashSet<&'static [u8]> = ["image/png", "image/webp", "image/avif", "image/jpeg", "image/gif", "image/webp", "image/apng", "image/bmp", "image/tiff"]
        .into_iter().map(str::as_bytes).collect();
    static ref OBJECT_HACKY_IGNORE: bytes::RegexSet = bytes::RegexSet::new([
        r#""author":"\[deleted\]""#,
        r#""promoted":true"#, // these seem to be ads which are in the data for some reason, and lack some important fields
        r#""domain":"self.promos""#, // .......
        r"\x00" // for SOME REASON one of the JSON files contains a lot of null bytes before one particular record, so just ignore that record
    ]).unwrap();
    static ref URL_REPLACEMENT_RULES: Vec<(Regex, &'static str)> = [
        (r"//imgur.com/([A-Za-z0-9]+)", r"//i.imgur.com/$1.jpg"),
        (r"^http://", r"https://")
    ].into_iter().map(|(r, e)| (Regex::new(r).unwrap(), e)).collect();
}

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
        let entry = match sonic_rs::serde::from_slice::<Entry>(buf.as_slice()) {
            Ok(x) => x,
            Err(e) => {
                log::warn!("parse failed, please validate {:?} {:?}", e, String::from_utf8_lossy(&buf));
                return Ok(())
            }
        };
        if entry.selftext.is_empty() && !entry.over_18 && entry.author.is_some() && entry.subreddit.is_some() {
            if !URL_IGNORE.is_match(&entry.url) {
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

struct Config {
    max_content_length: usize,
    input: String,
    output: String,
    backend: String,
    mode: OperatingMode,
    filename_threshold: Option<String>
}

async fn fetch_file(client: reqwest::Client, config: Arc<Config>, url: &str) -> Result<Vec<u8>> {
    // inelegant but I can't get it to work using Cows
    let mut url = url.to_string();
    for (regex, replacement) in URL_REPLACEMENT_RULES.iter() {
        url = regex.replace(&url, *replacement).to_string();
    }
    let mut response = client.get(&*url).send().await?;
    if !ACCEPTABLE_FILETYPES.contains(response.headers().get(reqwest::header::CONTENT_TYPE).context("no contept type")?.as_bytes()) {
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
    Ok(buffer)
}

fn write_output(config: Arc<Config>, mut rx: Receiver<ProcessedEntry>) -> Result<()> {
    let mut out = fs::File::options().append(true).open(&config.output)?;
    let stream = zstd::Encoder::new(&mut out, 15)?.auto_finish();
    let mut buf_stream = BufWriter::new(stream);
    while let Some(x) = rx.blocking_recv() {
        rmp_serde::encode::write(&mut buf_stream, &x)?;
    }
    Ok(())
}

enum OperatingMode {
    Count,
    Sample(f32),
    FullRun
}

fn readback_output(path: &str) -> Result<(u64, usize)> {
    use rmp_serde::decode::Error;
    let stream = zstd::stream::Decoder::new(fs::File::open(path)?)?;
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
    Ok((latest_timestamp, count))
}

#[tokio::main]
async fn main() -> Result<()> {
    pretty_env_logger::init();
    let cpus = num_cpus::get();

    let config = Arc::new(Config {
        max_content_length: 1<<23,
        input: String::from("./submissions"),
        output: String::from("./sample.zst"),
        backend: String::from("http://localhost:1708"),
        mode: OperatingMode::Sample(0.004),
        filename_threshold: None
    });

    let timestamp_threshold = match config.mode {
        OperatingMode::Count => None,
        _ => {
            match readback_output(&config.output) {
                Ok(x) => Some(x),
                Err(e) => {
                    log::warn!("could not read output: {}", e);
                    None
                }
            }
        }
    };


    if let Some((threshold, count)) = timestamp_threshold {
        log::info!("threshold is {}, {} items", threshold, count);
    }
    
    let backend = get_backend_config(&config.backend).await;

    log::info!("connected to inference server");

    let (entries_tx, mut entries_rx) = mpsc::channel::<Entry>(32768);
    let (buffers_tx, buffers_rx) = mpsc::channel(128);
    let (resized_tx, resized_rx) = mpsc::channel(backend.batch);
    let (final_write_tx, final_write_rx) = mpsc::channel::<ProcessedEntry>(32768);
    let client = Client::builder()
        .user_agent(concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION")))
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
                            log::debug!("got {}", &entry.url);
                            buffers_tx.send((entry, buf)).await?;
                        },
                        Err(e) => {
                            log::warn!("{} failed: {}", &entry.url, e)
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
            stream.map(Ok).try_for_each_concurrent(Some(cpus), move |(entry, buffer)| {
                let backend = backend.clone();
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
                            log::warn!("loading {} failed: {}", entry.url, e);
                            return Result::<(), anyhow::Error>::Ok(());
                        }
                    };
                    let resized = resize_for_embed(backend.clone(), image).await?;
                    resized_tx.send((entry, resized)).await?;
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
                let (entries, bytes): (Vec<Entry>, Vec<Vec<u8>>) = batch.into_iter().unzip();
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
                    
                    for (vector, entry) in result.into_iter().zip(entries) {
                        final_write_tx.send(ProcessedEntry {
                            url: entry.url,
                            id: entry.id,
                            title: entry.title,
                            subreddit: entry.subreddit.unwrap(),
                            author: entry.author.unwrap(),
                            blob: vector.into_vec(),
                            timestamp: entry.created_utc.to_u64()?
                        }).await?;
                    }
                    anyhow::Result::Ok(())
                }
            })
        }))
    };

    let config_ = config.clone();
    let output_writer_task = match config.mode {
        OperatingMode::Sample(_) | OperatingMode::FullRun => Some(tokio::task::spawn_blocking(move || write_output(config_, final_write_rx))),
        _ => None
    };

    log::info!("working...");

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

    match config.mode {
        OperatingMode::Count | OperatingMode::Sample(_) => {
            let semaphore = Arc::new(Semaphore::new(cpus));

            for path in paths {
                let semaphore = semaphore.clone();
                let permit = semaphore.acquire_owned().await?;
                let entries_tx = entries_tx.clone();
                let path_ = path.clone();
                log::info!("reading {:?}", path);
                file_readers.spawn_blocking(move || {
                    match process_file(path_, entries_tx, timestamp_threshold.map(|(x, _)| x)) {
                        Ok(_) => (),
                        Err(e) => log::error!("could not parse {:?} {:?}", &path, e)
                    }
                    std::mem::drop(permit);
                });
            }
        },
        OperatingMode::FullRun => {
            for path in paths {
                let entries_tx = entries_tx.clone();
                let path_ = path.clone();
                log::info!("reading {:?}", path);
                file_readers.spawn_blocking(move || match process_file(path_, entries_tx, timestamp_threshold.map(|(x, _)| x)) {
                    Ok(_) => (),
                    Err(e) => log::error!("could not parse {:?} {:?}", &path, e)
                });
            }
        }
    }

    while let Some(x) = file_readers.try_join_next() {
        x?;
    }

    std::mem::drop(entries_tx);
    println!("{:?}", load_task.await?);
    if let Some(task) = resize_task { println!("{:?}", task.await?); }
    if let Some(task) = embedding_generation_task { println!("{:?}", task.await?) };
    if let Some(task) = output_writer_task { println!("{:?}", task.await?) };

    Ok(())
}