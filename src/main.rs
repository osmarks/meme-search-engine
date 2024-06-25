use std::collections::HashSet;
use std::{collections::HashMap, io::Cursor};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, Context};
use axum::body::Body;
use axum::response::Response;
use axum::{
    extract::{Json, DefaultBodyLimit},
    response::IntoResponse,
    routing::{get, post},
    Router,
    http::StatusCode
};
use common::resize_for_embed_sync;
use compact_str::CompactString;
use image::RgbImage;
use image::{imageops::FilterType, io::Reader as ImageReader, DynamicImage, ImageFormat};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sqlx::SqliteConnection;
use sqlx::{sqlite::SqliteConnectOptions, SqlitePool};
use tokio::sync::{broadcast, mpsc, RwLock};
use tokio::task::JoinHandle;
use walkdir::WalkDir;
use base64::prelude::*;
use faiss::{ConcurrentIndex, Index};
use futures_util::stream::{StreamExt, TryStreamExt};
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;
use faiss::index::scalar_quantizer;
use lazy_static::lazy_static;
use prometheus::{register_int_counter, register_int_counter_vec, register_int_gauge, Encoder, IntCounter, IntGauge, IntCounterVec};
use ndarray::ArrayBase;

mod ocr;
mod common;
mod video_reader;

use crate::ocr::scan_image;
use crate::common::{InferenceServerConfig, resize_for_embed, EmbeddingRequest, get_backend_config, query_clip_server};

lazy_static! {
    static ref RELOADS_COUNTER: IntCounter = register_int_counter!("mse_reloads", "reloads executed").unwrap();
    static ref QUERIES_COUNTER: IntCounter = register_int_counter!("mse_queries", "queries executed").unwrap();
    static ref TERMS_COUNTER: IntCounterVec = register_int_counter_vec!("mse_terms", "terms used in queries, by type", &["type"]).unwrap();
    static ref IMAGES_LOADED_COUNTER: IntCounter = register_int_counter!("mse_loads", "images loaded by ingest process").unwrap();
    static ref IMAGES_LOADED_ERROR_COUNTER: IntCounter = register_int_counter!("mse_load_errors", "image load fails by ingest process").unwrap();
    static ref VIDEOS_LOADED_COUNTER: IntCounter = register_int_counter!("mse_video_loads", "video loaded by ingest process").unwrap();
    static ref VIDEOS_LOADED_ERROR_COUNTER: IntCounter = register_int_counter!("mse_video_load_errors", "video load fails by ingest process").unwrap();
    static ref IMAGES_EMBEDDED_COUNTER: IntCounter = register_int_counter!("mse_embeds", "images embedded by ingest process").unwrap();
    static ref IMAGES_OCRED_COUNTER: IntCounter = register_int_counter!("mse_ocrs", "images OCRed by ingest process").unwrap();
    static ref IMAGES_OCRED_ERROR_COUNTER: IntCounter = register_int_counter!("mse_ocr_errors", "image OCR fails by ingest process").unwrap();
    static ref IMAGES_THUMBNAILED_COUNTER: IntCounter = register_int_counter!("mse_thumbnails", "images thumbnailed by ingest process").unwrap();
    static ref THUMBNAILS_GENERATED_COUNTER: IntCounterVec = register_int_counter_vec!("mse_thumbnail_outputs", "thumbnails produced by ingest process", &["output_format"]).unwrap();
    static ref LAST_INDEX_SIZE: IntGauge = register_int_gauge!("mse_index_size", "images in loaded index").unwrap();
}

fn function_which_returns_50() -> usize { 50 }

#[derive(Debug, Deserialize, Clone)]
struct Config {
    clip_server: String,
    db_path: String,
    port: u16,
    files: String,
    #[serde(default)]
    enable_ocr: bool,
    #[serde(default)]
    thumbs_path: String,
    #[serde(default)]
    enable_thumbs: bool,
    #[serde(default="function_which_returns_50")]
    ocr_concurrency: usize,
    #[serde(default)]
    no_run_server: bool
}

#[derive(Debug)]
struct IIndex {
    vectors: scalar_quantizer::ScalarQuantizerIndexImpl,
    filenames: Vec<Filename>,
    format_codes: Vec<u64>,
    format_names: Vec<String>,
}

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS files (
    filename TEXT NOT NULL PRIMARY KEY,
    embedding_time INTEGER,
    ocr_time INTEGER,
    thumbnail_time INTEGER,
    embedding BLOB,
    ocr TEXT,
    raw_ocr_segments BLOB,
    thumbnails BLOB
);

CREATE TABLE IF NOT EXISTS predefined_embeddings (
    name TEXT NOT NULL PRIMARY KEY,
    embedding BLOB NOT NULL
);

DROP TRIGGER IF EXISTS ocr_fts_upd;
DROP TRIGGER IF EXISTS ocr_fts_ins;
DROP TRIGGER IF EXISTS ocr_fts_del;
DROP TABLE IF EXISTS ocr_fts;
"#;

#[derive(Debug, sqlx::FromRow, Clone)]
struct RawFileRecord {
    filename: Vec<u8>,
    embedding_time: Option<i64>,
    ocr_time: Option<i64>,
    thumbnail_time: Option<i64>,
    embedding: Option<Vec<u8>>,
    // this totally "will" be used later
    ocr: Option<String>,
    raw_ocr_segments: Option<Vec<u8>>,
    thumbnails: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
struct FileRecord {
    filename: CompactString,
    needs_embed: bool,
    needs_ocr: bool,
    needs_thumbnail: bool
}

#[derive(Debug, Clone)]
struct WConfig {
    backend: InferenceServerConfig,
    service: Config,
    predefined_embeddings: HashMap<String, ArrayBase<ndarray::OwnedRepr<f32>, ndarray::prelude::Dim<[usize; 1]>>>
}

#[derive(Debug)]
struct LoadedImage {
    image: Arc<DynamicImage>,
    filename: Filename,
    original_size: Option<usize>,
    fast_thumbnails_only: bool
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash)]
enum Filename {
    Actual(CompactString),
    VideoFrame(CompactString, u64)
}

// this is a somewhat horrible hack, but probably nobody has NUL bytes at the start of filenames?
impl Filename {
    fn decode(buf: Vec<u8>) -> Result<Self> {
        Ok(match buf.strip_prefix(&[0]) {
            Some(remainder) => rmp_serde::from_read(&*remainder)?,
            None => Filename::Actual(CompactString::from_utf8(buf)?)
        })
    }

    fn encode(&self) -> Result<Vec<u8>> {
        match self {
            Self::Actual(s) => Ok(s.to_string().into_bytes()),
            x => {
                let mut out = rmp_serde::to_vec(x).context("should not happen")?;
                out.insert(0, 0);
                Ok(out)
            }
        }
    }

    fn container_filename(&self) -> String {
        match self {
            Self::Actual(s) => s.to_string(),
            Self::VideoFrame(s, _) => s.to_string()
        }
    }
}

#[derive(Debug)]
struct EmbeddingInput {
    image: Vec<u8>,
    filename: Filename,
}

fn timestamp() -> i64 {
    chrono::Utc::now().timestamp_micros()
}

#[derive(Debug, Clone)]
struct ImageFormatConfig {
    target_width: u32,
    target_filesize: usize,
    quality: u8,
    format: ImageFormat,
    extension: String,
    is_fast: bool
}

fn generate_filename_hash(filename: &Filename) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = fnv::FnvHasher::default();
    match filename {
        Filename::Actual(x) => x.hash(&mut hasher),
        _ => filename.hash(&mut hasher)
    };
    BASE64_URL_SAFE_NO_PAD.encode(hasher.finish().to_le_bytes())
}

fn generate_thumbnail_filename(
    filename: &Filename,
    format_name: &str,
    format_config: &ImageFormatConfig,
) -> String {
    format!(
        "{}{}.{}",
        generate_filename_hash(filename),
        format_name,
        format_config.extension
    )
}

async fn initialize_database(config: &Config) -> Result<SqlitePool> {
    let connection_options = SqliteConnectOptions::new()
    .filename(&config.db_path)
    .create_if_missing(true);
    let pool = SqlitePool::connect_with(connection_options).await?;
    sqlx::query(SCHEMA).execute(&pool).await?;
    Ok(pool)
}

fn image_formats(_config: &Config) -> HashMap<String, ImageFormatConfig> {
    let mut formats = HashMap::new();
    formats.insert(
        "jpegl".to_string(),
        ImageFormatConfig {
            target_width: 800,
            target_filesize: 0,
            quality: 70,
            format: ImageFormat::Jpeg,
            extension: "jpg".to_string(),
            is_fast: true
        },
    );
    formats.insert(
        "jpegh".to_string(),
        ImageFormatConfig {
            target_width: 1600,
            target_filesize: 0,
            quality: 80,
            format: ImageFormat::Jpeg,
            extension: "jpg".to_string(),
            is_fast: true
        },
    );
    formats.insert(
        "jpeg256kb".to_string(),
        ImageFormatConfig {
            target_width: 500,
            target_filesize: 256000,
            quality: 0,
            format: ImageFormat::Jpeg,
            extension: "jpg".to_string(),
            is_fast: false
        },
    );
    formats.insert(
        "avifh".to_string(),
        ImageFormatConfig {
            target_width: 1600,
            target_filesize: 0,
            quality: 80,
            format: ImageFormat::Avif,
            extension: "avif".to_string(),
            is_fast: false
        },
    );
    formats.insert(
        "avifl".to_string(),
        ImageFormatConfig {
            target_width: 800,
            target_filesize: 0,
            quality: 70,
            format: ImageFormat::Avif,
            extension: "avif".to_string(),
            is_fast: false
        },
    );
    formats
}

async fn ensure_filename_record_exists(conn: &mut SqliteConnection, filename_enc: &Vec<u8>) -> Result<()> {
    sqlx::query!("INSERT OR IGNORE INTO files (filename) VALUES (?)", filename_enc)
        .execute(conn)
        .await?;
    Ok(())
}

async fn ingest_files(config: Arc<WConfig>) -> Result<()> {
    let pool = initialize_database(&config.service).await?;
    let client = Client::new();
    
    let formats = image_formats(&config.service);
    
    let (to_process_tx, to_process_rx) = mpsc::channel::<FileRecord>(100);
    let (to_embed_tx, to_embed_rx) = mpsc::channel(config.backend.batch as usize);
    let (to_thumbnail_tx, to_thumbnail_rx) = mpsc::channel(30);
    let (to_ocr_tx, to_ocr_rx) = mpsc::channel(30);
    
    let cpus = num_cpus::get();

    let video_lengths = Arc::new(RwLock::new(HashMap::new()));
    let video_thumb_times = Arc::new(RwLock::new(HashMap::new()));
    let video_embed_times = Arc::new(RwLock::new(HashMap::new()));

    // Image loading and preliminary resizing
    let image_loading: JoinHandle<Result<()>> = tokio::spawn({
        let config = config.clone();
        let video_lengths = video_lengths.clone();
        let stream = ReceiverStream::new(to_process_rx).map(Ok);
        stream.try_for_each_concurrent(Some(cpus), move |record| {
            let config = config.clone();
            let to_embed_tx = to_embed_tx.clone();
            let to_thumbnail_tx = to_thumbnail_tx.clone();
            let to_ocr_tx = to_ocr_tx.clone();
            let video_lengths = video_lengths.clone();
            async move {
                let path = Path::new(&config.service.files).join(&*record.filename);
                let image: Result<Arc<DynamicImage>> = tokio::task::block_in_place(|| Ok(Arc::new(ImageReader::open(&path)?.with_guessed_format()?.decode()?)));
                let image = match image {
                    Ok(image) => image,
                    Err(e) => {
                        log::warn!("Could not read {} as image: {}", record.filename, e);
                        let filename = record.filename.clone();
                        IMAGES_LOADED_ERROR_COUNTER.inc();
                        let video_length = tokio::task::spawn_blocking(move || -> Result<Option<u64>> {
                            let mut i = 0;
                            let callback = |frame: RgbImage| {
                                let frame: Arc<DynamicImage> = Arc::new(frame.into());
                                let embed_buf = resize_for_embed_sync(config.backend.clone(), frame.clone())?;
                                to_embed_tx.blocking_send(EmbeddingInput {
                                    image: embed_buf,
                                    filename: Filename::VideoFrame(filename.clone(), i)
                                })?;
                                if config.service.enable_thumbs {
                                    to_thumbnail_tx.blocking_send(LoadedImage {
                                        image: frame.clone(),
                                        filename: Filename::VideoFrame(filename.clone(), i),
                                        original_size: None,
                                        fast_thumbnails_only: true
                                    })?;
                                }
                                i += 1;
                                Ok(())
                            };
                            match video_reader::run(&path, callback) {
                                Ok(()) => {
                                    VIDEOS_LOADED_COUNTER.inc();
                                    return anyhow::Result::Ok(Some(i))
                                },
                                Err(e) => {
                                    log::error!("Could not read {} as video: {}", filename, e);
                                    VIDEOS_LOADED_ERROR_COUNTER.inc();
                                }
                            }
                            return anyhow::Result::Ok(None)
                        }).await??;
                        if let Some(length) = video_length {
                            video_lengths.write().await.insert(record.filename, length);
                        }
                        return Ok(())
                    }
                };
                IMAGES_LOADED_COUNTER.inc();
                if record.needs_embed {
                    let resized = resize_for_embed(config.backend.clone(), image.clone()).await?;
                    
                    to_embed_tx.send(EmbeddingInput { image: resized, filename: Filename::Actual(record.filename.clone()) }).await?
                }
                if record.needs_thumbnail {
                    to_thumbnail_tx
                        .send(LoadedImage {
                            image: image.clone(),
                            filename: Filename::Actual(record.filename.clone()),
                            original_size: Some(std::fs::metadata(&path)?.len() as usize),
                            fast_thumbnails_only: false
                        })
                        .await?;
                }
                if record.needs_ocr {
                    to_ocr_tx
                        .send(LoadedImage {
                            image,
                            filename: Filename::Actual(record.filename.clone()),
                            original_size: None,
                            fast_thumbnails_only: true
                        })
                        .await?;
                }
                Ok(())
            }
        })
    });
    
    // Thumbnail generation
    let thumbnail_generation: Option<JoinHandle<Result<()>>> = if config.service.enable_thumbs {
        let config = config.clone();
        let pool = pool.clone();
        let stream = ReceiverStream::new(to_thumbnail_rx).map(Ok);
        let formats = Arc::new(formats);
        let video_thumb_times = video_thumb_times.clone();
        Some(tokio::spawn({
            stream.try_for_each_concurrent(Some(cpus), move |image| {
                use image::codecs::*;

                let formats = formats.clone();
                let config = config.clone();
                let pool = pool.clone();
                let video_thumb_times = video_thumb_times.clone();
                async move {
                    let filename = image.filename.clone();
                    log::debug!("thumbnailing {:?}", filename);
                    let generated_formats = tokio::task::spawn_blocking(move || {
                        let mut generated_formats = Vec::new();
                        let rgb = DynamicImage::from(image.image.to_rgb8());
                        for (format_name, format_config) in &*formats {
                            if !format_config.is_fast && image.fast_thumbnails_only { continue }
                            let resized = if format_config.target_filesize != 0 {
                                let mut lb = 1;
                                let mut ub = 100;
                                loop {
                                    let quality = (lb + ub) / 2;
                                    let thumbnail = rgb.resize(
                                        format_config.target_width.min(rgb.width()),
                                        u32::MAX,
                                        FilterType::Lanczos3,
                                    );
                                    let mut buf: Vec<u8> = Vec::new();
                                    let mut csr = Cursor::new(&mut buf);
                                    // this is ugly but I don't actually know how to fix it (cannot factor it out due to issues with dyn Trait)
                                    match format_config.format {
                                        ImageFormat::Avif => thumbnail.write_with_encoder(avif::AvifEncoder::new_with_speed_quality(&mut csr, 4, quality)),
                                        ImageFormat::Jpeg => thumbnail.write_with_encoder(jpeg::JpegEncoder::new_with_quality(&mut csr, quality)),
                                        _ => unimplemented!()
                                    }?;
                                    if buf.len() > format_config.target_filesize {
                                        ub = quality;
                                    } else {
                                        lb = quality + 1;
                                    }
                                    if lb >= ub {
                                        break buf;
                                    }
                                }
                            } else {
                                let thumbnail = rgb.resize(
                                    format_config.target_width.min(rgb.width()),
                                    u32::MAX,
                                    FilterType::Lanczos3,
                                );
                                let mut buf: Vec<u8> = Vec::new();
                                let mut csr = Cursor::new(&mut buf);
                                match format_config.format {
                                    ImageFormat::Avif => thumbnail.write_with_encoder(avif::AvifEncoder::new_with_speed_quality(&mut csr, 4, format_config.quality)),
                                    ImageFormat::Jpeg => thumbnail.write_with_encoder(jpeg::JpegEncoder::new_with_quality(&mut csr, format_config.quality)),
                                    ImageFormat::WebP => thumbnail.write_with_encoder(webp::WebPEncoder::new_lossless(&mut csr)),
                                    _ => unimplemented!()
                                }?;
                                buf
                            };
                            if resized.len() < image.original_size.unwrap_or(usize::MAX) {
                                generated_formats.push(format_name.clone());
                                let thumbnail_path = Path::new(&config.service.thumbs_path).join(
                                    generate_thumbnail_filename(
                                        &image.filename,
                                        format_name,
                                        format_config,
                                    ),
                                );
                                THUMBNAILS_GENERATED_COUNTER.get_metric_with_label_values(&[format_name]).unwrap().inc();
                                std::fs::write(thumbnail_path, resized)?;
                            }
                        }
                        Ok::<Vec<String>, anyhow::Error>(generated_formats)
                    }).await??;
                    IMAGES_THUMBNAILED_COUNTER.inc();
                    let formats_data = rmp_serde::to_vec(&generated_formats)?;
                    let ts = timestamp();
                    let filename_enc = filename.encode()?;
                    let mut conn = pool.acquire().await?;
                    ensure_filename_record_exists(&mut conn, &filename_enc).await?;
                    match filename {
                        Filename::VideoFrame(container, _) => { video_thumb_times.write().await.insert(container.clone(), timestamp()); },
                        _ => ()
                    }
                    sqlx::query!(
                        "UPDATE files SET thumbnails = ?, thumbnail_time = ? WHERE filename = ?",
                        formats_data,
                        ts,
                        filename_enc
                    )
                        .execute(&mut *conn)
                        .await?;
                    Ok(())
                }
            })
        }))
    } else {
        None
    };
    
    // OCR
    // TODO: save OCR errors and don't retry
    let ocr: Option<JoinHandle<Result<()>>> = if config.service.enable_ocr {
        let client = client.clone();
        let pool = pool.clone();
        let stream = ReceiverStream::new(to_ocr_rx).map(Ok);
        Some(tokio::spawn({
            stream.try_for_each_concurrent(Some(config.service.ocr_concurrency), move |image| {
                let client = client.clone();
                let pool = pool.clone();
                async move {
                    log::debug!("OCRing {:?}", image.filename);
                    let scan = match scan_image(&client, &image.image).await {
                        Ok(scan) => scan,
                        Err(e) => {
                            IMAGES_OCRED_ERROR_COUNTER.inc();
                            log::error!("OCR failure {:?}: {}", image.filename, e);
                            return Ok(())
                        }
                    };
                    IMAGES_OCRED_COUNTER.inc();
                    let ocr_text = scan
                        .iter()
                        .map(|segment| segment.text.clone())
                        .collect::<Vec<_>>()
                        .join("\n");
                    let ocr_data = rmp_serde::to_vec(&scan)?;
                    let ts = timestamp();
                    let filename_enc = image.filename.encode()?;
                    let mut conn = pool.acquire().await?;
                    ensure_filename_record_exists(&mut conn, &filename_enc).await?;
                    sqlx::query!(
                        "UPDATE files SET ocr = ?, raw_ocr_segments = ?, ocr_time = ? WHERE filename = ?",
                        ocr_text,
                        ocr_data,
                        ts,
                        filename_enc
                    )
                        .execute(&mut *conn)
                        .await?;
                    Ok(())
                }
            })
        }))
    } else {
        None
    };

    let embedding_generation: JoinHandle<Result<()>> = tokio::spawn({
        let stream = ReceiverStream::new(to_embed_rx).chunks(config.backend.batch);
        let client = client.clone();
        let config = config.clone();
        let pool = pool.clone();
        let video_embed_times = video_embed_times.clone();
        // keep multiple embedding requests in flight
        stream.map(Ok).try_for_each_concurrent(Some(3), move |batch| {
            let client = client.clone();
            let config = config.clone();
            let pool = pool.clone();
            let video_embed_times = video_embed_times.clone();
            async move {
                let result: Vec<serde_bytes::ByteBuf> = query_clip_server(
                    &client,
                    &config.service.clip_server,
                    "",
                    EmbeddingRequest::Images {
                        images: batch.iter().map(|input| serde_bytes::ByteBuf::from(input.image.clone())).collect(),
                    },
                ).await.context("querying CLIP server")?;
                
                let mut tx = pool.begin().await?;
                let ts = timestamp();
                for (i, vector) in result.into_iter().enumerate() {
                    let vector = vector.into_vec();
                    log::debug!("embedded {:?}", batch[i].filename);
                    let encoded_filename = batch[i].filename.encode()?;
                    IMAGES_EMBEDDED_COUNTER.inc();
                    ensure_filename_record_exists(&mut *tx, &encoded_filename).await?;
                    match &batch[i].filename {
                        Filename::VideoFrame(container, _) => { video_embed_times.write().await.insert(container.clone(), timestamp()); },
                        _ => ()
                    }
                    sqlx::query!(
                        "UPDATE files SET embedding_time = ?, embedding = ? WHERE filename = ?",
                        ts,
                        vector,
                        encoded_filename
                    )
                        .execute(&mut *tx)
                        .await?;
                }
                tx.commit().await?;
                anyhow::Result::Ok(())
            }
        })
    });
                
    let mut actual_filenames = HashMap::new();
    
    // blocking OS calls
    tokio::task::block_in_place(|| -> anyhow::Result<()> {
        for entry in WalkDir::new(config.service.files.as_str()) {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                let filename = CompactString::from(path.strip_prefix(&config.service.files)?.to_str().unwrap());
                let modtime = entry.metadata()?.modified()?.duration_since(std::time::UNIX_EPOCH)?;
                let modtime = modtime.as_micros() as i64;
                actual_filenames.insert(filename.clone(), (path.to_path_buf(), modtime));
            }
        }
        Ok(())
    })?;

    log::debug!("finished reading filenames");

    for (filename, (_path, modtime)) in actual_filenames.iter() {
        let modtime = *modtime;
        let filename_arr = filename.as_bytes();
        let record = sqlx::query_as!(RawFileRecord, "SELECT * FROM files WHERE filename = ?", filename_arr)
            .fetch_optional(&pool)
            .await?;

        let new_record = match record {
            None => Some(FileRecord {
                filename: filename.clone(),
                needs_embed: true,
                needs_ocr: config.service.enable_ocr,
                needs_thumbnail: config.service.enable_thumbs
            }),
            Some(r) => {
                let needs_embed = modtime > r.embedding_time.unwrap_or(i64::MIN);
                let needs_ocr = modtime > r.ocr_time.unwrap_or(i64::MIN) && config.service.enable_ocr;
                let needs_thumbnail = modtime > r.thumbnail_time.unwrap_or(i64::MIN) && config.service.enable_thumbs;
                if needs_embed || needs_ocr || needs_thumbnail {
                    Some(FileRecord {
                        filename: filename.clone(),
                        needs_embed, needs_ocr, needs_thumbnail
                    })
                } else {
                    None
                }
            }
        };
        if let Some(record) = new_record {
            log::debug!("processing {}", record.filename);
            // we need to exit here to actually capture the error
            if !to_process_tx.send(record).await.is_ok() {
                break
            }
        }
    }

    drop(to_process_tx);
    
    embedding_generation.await?.context("generating embeddings")?;
    
    if let Some(thumbnail_generation) = thumbnail_generation {
        thumbnail_generation.await?.context("generating thumbnails")?;
    }
    
    if let Some(ocr) = ocr {
        ocr.await?.context("OCRing")?;
    }

    image_loading.await?.context("loading images")?;
    
    let stored: Vec<Vec<u8>> = sqlx::query_scalar("SELECT filename FROM files").fetch_all(&pool).await?;
    let mut tx = pool.begin().await?;
    let video_lengths = video_lengths.read().await;
    for filename in stored {
        let parsed_filename = Filename::decode(filename.clone())?;
        match parsed_filename {
            Filename::Actual(s) => {
                let s = &*s;
                let raw = &filename;
                if !actual_filenames.contains_key(s) {
                    sqlx::query!("DELETE FROM files WHERE filename = ?", raw)
                        .execute(&mut *tx)
                        .await?;
                }
            },
            // This might fail in some cases where for whatever reason a video is replaced with a file of the same name which is not a video. Don't do that.
            Filename::VideoFrame(container, frame) => if !actual_filenames.contains_key(&container) {
                if let Some(length) = video_lengths.get(&container) {
                    if frame > *length {
                        sqlx::query!("DELETE FROM files WHERE filename = ?", filename)
                            .execute(&mut *tx)
                            .await?;
                    }
                }
            }
        }
    }

    let video_thumb_times = video_thumb_times.read().await;
    let video_embed_times = video_embed_times.read().await;
    for container_filename in video_lengths.keys() {
        let embed_time = video_embed_times.get(container_filename);
        let thumb_time = video_thumb_times.get(container_filename);
        let container_filename: &[u8] = container_filename.as_bytes();
        sqlx::query!("INSERT OR REPLACE INTO files (filename, embedding_time, thumbnail_time) VALUES (?, ?, ?)", container_filename, embed_time, thumb_time)
            .execute(&mut *tx)
            .await?;
    }

    tx.commit().await?;

    log::info!("Ingest done");
    
    Result::Ok(())
}

const INDEX_ADD_BATCH: usize = 512;

async fn build_index(config: Arc<WConfig>) -> Result<IIndex> {
    let pool = initialize_database(&config.service).await?;

    let mut index = IIndex {
        vectors: scalar_quantizer::ScalarQuantizerIndexImpl::new(config.backend.embedding_size as u32, scalar_quantizer::QuantizerType::QT_fp16, faiss::MetricType::InnerProduct)?,
        filenames: Vec::new(),
        format_codes: Vec::new(),
        format_names: Vec::new(),
    };

    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM files")
        .fetch_one(&pool)
        .await?;

    index.filenames = Vec::with_capacity(count as usize);
    index.format_codes = Vec::with_capacity(count as usize);
    let mut buffer = Vec::with_capacity(INDEX_ADD_BATCH * config.backend.embedding_size as usize);
    index.format_names = Vec::with_capacity(5);
    index.format_names.push(String::from("VIDEO"));
    let video_format_code = 1<<0;

    let mut rows = sqlx::query_as::<_, RawFileRecord>("SELECT * FROM files").fetch(&pool);
    while let Some(record) = rows.try_next().await? {
        if let Some(emb) = record.embedding {
            let parsed = Filename::decode(record.filename)?;

            let mut format_code = match parsed {
                Filename::VideoFrame(_, _) => video_format_code,
                _ => 0
            };

            index.filenames.push(parsed);
            for i in (0..emb.len()).step_by(2) {
                buffer.push(
                    half::f16::from_le_bytes([emb[i], emb[i + 1]])
                        .to_f32(),
                );
            }
            if buffer.len() == buffer.capacity() {
                index.vectors.add(&buffer)?;
                buffer.clear();
            }

            let mut formats: Vec<String> = Vec::new();
            if let Some(t) = record.thumbnails {
                formats = rmp_serde::from_slice(&t)?;
            }
            
            for format_string in &formats {
                let mut found = false;
                for (i, name) in index.format_names.iter().enumerate() {
                    if name == format_string {
                        format_code |= 1 << i;
                        found = true;
                        break;
                    }
                }
                if !found {
                    let new_index = index.format_names.len();
                    format_code |= 1 << new_index;
                    index.format_names.push(format_string.clone());
                }
            }
            index.format_codes.push(format_code);
        }
    }
    if !buffer.is_empty() {
        index.vectors.add(&buffer)?;
    }

    Ok(index)
}

fn decode_fp16_buffer(buf: &[u8]) -> Vec<f32> {
    buf.chunks_exact(2)
        .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect()
}

type EmbeddingVector = Vec<f32>;

#[derive(Debug, Serialize)]
struct QueryResult {
    matches: Vec<(f32, String, String, u64)>,
    formats: Vec<String>,
    extensions: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct QueryTerm {
    embedding: Option<EmbeddingVector>,
    image: Option<String>,
    text: Option<String>,
    predefined_embedding: Option<String>,
    weight: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct QueryRequest {
    terms: Vec<QueryTerm>,
    k: Option<usize>,
    #[serde(default)]
    include_video: bool
}

async fn query_index(index: &IIndex, query: EmbeddingVector, k: usize, video: bool) -> Result<QueryResult> {
    let result = index.vectors.search(&query, k as usize)?;

    let mut seen_videos = HashSet::new();

    let items = result.distances
        .into_iter()
        .zip(result.labels)
        .filter_map(|(distance, id)| {
            let id = id.get()? as usize;
            match (video, &index.filenames[id]) {
                (_, Filename::Actual(_)) => (),
                (false, Filename::VideoFrame(_, _)) => return None,
                (true, Filename::VideoFrame(container, _)) => {
                    if !seen_videos.insert(container) {
                        return None
                    }
                }
            }
            Some((
                distance,
                index.filenames[id].container_filename(),
                generate_filename_hash(&index.filenames[id as usize]).clone(),
                index.format_codes[id]
            ))
        })
        .collect();

    Ok(QueryResult {
        matches: items,
        formats: index.format_names.clone(),
        extensions: HashMap::new(),
    })
}

async fn handle_request(config: Arc<WConfig>, client: Arc<Client>, index: &IIndex, req: Json<QueryRequest>) -> Result<Response<Body>> {
    let mut total_embedding = ndarray::Array::from(vec![0.0; config.backend.embedding_size]);

    let mut image_batch = Vec::new();
    let mut image_weights = Vec::new();
    let mut text_batch = Vec::new();
    let mut text_weights = Vec::new();

    for term in &req.terms {
        if let Some(image) = &term.image {
            TERMS_COUNTER.get_metric_with_label_values(&["image"]).unwrap().inc();
            let bytes = BASE64_STANDARD.decode(image)?;
            let image = Arc::new(tokio::task::block_in_place(|| image::load_from_memory(&bytes))?);
            image_batch.push(serde_bytes::ByteBuf::from(resize_for_embed(config.backend.clone(), image).await?));
            image_weights.push(term.weight.unwrap_or(1.0));
        }
        if let Some(text) = &term.text {
            TERMS_COUNTER.get_metric_with_label_values(&["text"]).unwrap().inc();
            text_batch.push(text.clone());
            text_weights.push(term.weight.unwrap_or(1.0));
        }
        if let Some(embedding) = &term.embedding {
            TERMS_COUNTER.get_metric_with_label_values(&["embedding"]).unwrap().inc();
            let weight = term.weight.unwrap_or(1.0);
            for (i, value) in embedding.iter().enumerate() {
                total_embedding[i] += value * weight;
            }
        }
        if let Some(name) = &term.predefined_embedding {
            let embedding = config.predefined_embeddings.get(name).context("name invalid")?;
            total_embedding = total_embedding + embedding * term.weight.unwrap_or(1.0);
        }
    }
    
    let mut batches = vec![];    

    if !image_batch.is_empty() {
        batches.push(
            (EmbeddingRequest::Images {
                images: image_batch
            }, image_weights)
        );
    }
    if !text_batch.is_empty() {
        batches.push(
            (EmbeddingRequest::Text {
                text: text_batch,
            }, text_weights)
        );
    }

    for (batch, weights) in batches {
        let embs: Vec<Vec<u8>> = query_clip_server(&client, &config.service.clip_server, "/", batch).await?;
        for (emb, weight) in embs.into_iter().zip(weights) {
            total_embedding += &(ndarray::Array::from_vec(decode_fp16_buffer(&emb)) * weight);
        }
    }

    let k = req.k.unwrap_or(1000);
    let qres = query_index(index, total_embedding.to_vec(), k, req.include_video).await?;

    let mut extensions = HashMap::new();
    for (k, v) in image_formats(&config.service) {
        extensions.insert(k, v.extension);
    }

    Ok(Json(QueryResult {
        matches: qres.matches,
        formats: qres.formats,
        extensions,
    }).into_response())
}

#[derive(Serialize, Deserialize)]
struct FrontendInit {
    n_total: u64,
    predefined_embedding_names: Vec<String>
}

#[tokio::main]
async fn main() -> Result<()> {
    pretty_env_logger::init();

    let config_path = std::env::args().nth(1).expect("Missing config file path");
    let config = serde_json::from_slice(&std::fs::read(config_path)?)?;

    let pool = initialize_database(&config).await?;
    sqlx::query(SCHEMA).execute(&pool).await?;

    let backend = get_backend_config(&config.clip_server).await;

    let mut predefined_embeddings = HashMap::new();

    {
        let db = initialize_database(&config).await?;
        let result = sqlx::query!("SELECT * FROM predefined_embeddings")
            .fetch_all(&db).await?;
        for row in result {
            predefined_embeddings.insert(row.name, ndarray::Array::from(decode_fp16_buffer(&row.embedding)));
        }
    }

    let config = Arc::new(WConfig {
        service: config,
        backend,
        predefined_embeddings
    });

    if config.service.no_run_server {
        ingest_files(config.clone()).await?;
        return Ok(())
    }

    let (request_ingest_tx, mut request_ingest_rx) = mpsc::channel(1);

    let index = Arc::new(tokio::sync::RwLock::new(build_index(config.clone()).await?));

    let (ingest_done_tx, _ingest_done_rx) = broadcast::channel(1);
    let done_tx = Arc::new(ingest_done_tx.clone());

    let _ingest_task = tokio::spawn({
        let config = config.clone();
        let index = index.clone();
        async move {
            loop {
                log::info!("Ingest running");
                match ingest_files(config.clone()).await {
                    Ok(_) => {
                        match build_index(config.clone()).await {
                            Ok(new_index) => {
                                LAST_INDEX_SIZE.set(new_index.vectors.ntotal() as i64);
                                *index.write().await = new_index;
                                log::info!("Index loaded");
                            }
                            Err(e) => {
                                log::error!("Index build failed: {:?}", e);
                                ingest_done_tx.send((false, format!("{:?}", e))).unwrap();
                            }
                        }
                    }
                    Err(e) => {
                        log::error!("Ingest failed: {:?}", e);
                        ingest_done_tx.send((false, format!("{:?}", e))).unwrap();
                    }
                }
                ingest_done_tx.send((true, format!("OK"))).unwrap();
                RELOADS_COUNTER.inc();
                request_ingest_rx.recv().await;
            }
        }
    });

    let cors = CorsLayer::permissive();

    let config_ = config.clone();
    let client = Arc::new(Client::new());
    let index_ = index.clone();
    let config__ = config.clone();
    let app = Router::new()
        .route("/", post(|req| async move {
            let config = config.clone();
            let index = index.read().await; // TODO: use ConcurrentIndex here
            let client = client.clone();
            QUERIES_COUNTER.inc();
            handle_request(config, client.clone(), &index, req).await.map_err(|e| format!("{:?}", e))
        }).layer(DefaultBodyLimit::max(2<<24)))
        .route("/", get(|_req: ()| async move {
            Json(FrontendInit {
                n_total: index_.read().await.vectors.ntotal(),
                predefined_embedding_names: config__.predefined_embeddings.keys().cloned().collect()
            })
        }))
        .route("/reload", post(|_req: ()| async move {
            log::info!("Requesting index reload");
            let mut done_rx = done_tx.clone().subscribe();
            let _ = request_ingest_tx.send(()).await; // ignore possible error, which is presumably because the queue is full
            match done_rx.recv().await {
                Ok((true, status)) => {
                    let mut res = status.into_response();
                    *res.status_mut() = StatusCode::OK;
                    res
                },
                Ok((false, status)) => {
                    let mut res = status.into_response();
                    *res.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                    res
                },
                Err(_) => {
                    let mut res = "internal error".into_response();
                    *res.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                    res
                }
            }
        }))
        .route("/metrics", get(|_req: ()| async move {
            let mut buffer = Vec::new();
            let encoder = prometheus::TextEncoder::new();
            let metric_families = prometheus::gather();
            encoder.encode(&metric_families, &mut buffer).unwrap();
            buffer
        }))
        .layer(cors);

    let addr = format!("0.0.0.0:{}", config_.service.port);
    log::info!("Starting server on {}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await?;

    Ok(())
}            