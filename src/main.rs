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
use image::{imageops::FilterType, io::Reader as ImageReader, DynamicImage, ImageFormat};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sqlx::{sqlite::SqliteConnectOptions, SqlitePool};
use tokio::sync::{broadcast, mpsc};
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

use crate::ocr::scan_image;
use crate::common::{InferenceServerConfig, resize_for_embed, EmbeddingRequest, get_backend_config, query_clip_server};

lazy_static! {
    static ref RELOADS_COUNTER: IntCounter = register_int_counter!("mse_reloads", "reloads executed").unwrap();
    static ref QUERIES_COUNTER: IntCounter = register_int_counter!("mse_queries", "queries executed").unwrap();
    static ref TERMS_COUNTER: IntCounterVec = register_int_counter_vec!("mse_terms", "terms used in queries, by type", &["type"]).unwrap();
    static ref IMAGES_LOADED_COUNTER: IntCounter = register_int_counter!("mse_loads", "images loaded by ingest process").unwrap();
    static ref IMAGES_LOADED_ERROR_COUNTER: IntCounter = register_int_counter!("mse_load_errors", "image load fails by ingest process").unwrap();
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
    filenames: Vec<String>,
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

CREATE VIRTUAL TABLE IF NOT EXISTS ocr_fts USING fts5 (
    filename,
    ocr,
    tokenize='unicode61 remove_diacritics 2',
    content='files'
);

CREATE TABLE IF NOT EXISTS predefined_embeddings (
    name TEXT NOT NULL PRIMARY KEY,
    embedding BLOB NOT NULL
);

CREATE TRIGGER IF NOT EXISTS ocr_fts_ins AFTER INSERT ON files BEGIN
    INSERT INTO ocr_fts (rowid, filename, ocr) VALUES (new.rowid, new.filename, COALESCE(new.ocr, ''));
END;

CREATE TRIGGER IF NOT EXISTS ocr_fts_del AFTER DELETE ON files BEGIN
    INSERT INTO ocr_fts (ocr_fts, rowid, filename, ocr) VALUES ('delete', old.rowid, old.filename, COALESCE(old.ocr, ''));
END;

CREATE TRIGGER IF NOT EXISTS ocr_fts_upd AFTER UPDATE ON files BEGIN
    INSERT INTO ocr_fts (ocr_fts, rowid, filename, ocr) VALUES ('delete', old.rowid, old.filename, COALESCE(old.ocr, ''));
    INSERT INTO ocr_fts (rowid, filename, ocr) VALUES (new.rowid, new.filename, COALESCE(new.ocr, ''));
END;
"#;

#[derive(Debug, sqlx::FromRow, Clone, Default)]
struct FileRecord {
    filename: String,
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
struct WConfig {
    backend: InferenceServerConfig,
    service: Config,
    predefined_embeddings: HashMap<String, ArrayBase<ndarray::OwnedRepr<f32>, ndarray::prelude::Dim<[usize; 1]>>>
}

#[derive(Debug)]
struct LoadedImage {
    image: Arc<DynamicImage>,
    filename: String,
    original_size: usize,
}

#[derive(Debug)]
struct EmbeddingInput {
    image: Vec<u8>,
    filename: String,
}

fn timestamp() -> i64 {
    chrono::Utc::now().timestamp_micros()
}

#[derive(Debug, Clone)]
struct ImageFormatConfig {
    target_width: u32,
    target_filesize: u32,
    quality: u8,
    format: ImageFormat,
    extension: String,
}

fn generate_filename_hash(filename: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = fnv::FnvHasher::default();
    filename.hash(&mut hasher);
    BASE64_URL_SAFE_NO_PAD.encode(hasher.finish().to_le_bytes())
}

fn generate_thumbnail_filename(
    filename: &str,
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
        },
    );
    formats
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

    // Image loading and preliminary resizing
    let image_loading: JoinHandle<Result<()>> = tokio::spawn({
        let config = config.clone();
        let stream = ReceiverStream::new(to_process_rx).map(Ok);
        stream.try_for_each_concurrent(Some(cpus), move |record| {
            let config = config.clone();
            let to_embed_tx = to_embed_tx.clone();
            let to_thumbnail_tx = to_thumbnail_tx.clone();
            let to_ocr_tx = to_ocr_tx.clone();
            async move {
                let path = Path::new(&config.service.files).join(&record.filename);
                let image: Result<Arc<DynamicImage>> = tokio::task::block_in_place(|| Ok(Arc::new(ImageReader::open(&path)?.with_guessed_format()?.decode()?)));
                let image = match image {
                    Ok(image) => image,
                    Err(e) => {
                        log::error!("Could not read {}: {}", record.filename, e);
                        IMAGES_LOADED_ERROR_COUNTER.inc();
                        return Ok(())
                    }
                };
                IMAGES_LOADED_COUNTER.inc();
                if record.embedding.is_none() {
                    let resized = resize_for_embed(config.backend.clone(), image.clone()).await?;
                    
                    to_embed_tx.send(EmbeddingInput { image: resized, filename: record.filename.clone() }).await?
                }
                if record.thumbnails.is_none() && config.service.enable_thumbs {
                    to_thumbnail_tx
                    .send(LoadedImage {
                        image: image.clone(),
                        filename: record.filename.clone(),
                        original_size: std::fs::metadata(&path)?.len() as usize,
                    })
                    .await?;
                }
                if record.raw_ocr_segments.is_none() && config.service.enable_ocr {
                    to_ocr_tx
                    .send(LoadedImage {
                        image,
                        filename: record.filename.clone(),
                        original_size: 0,
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
        Some(tokio::spawn({
            stream.try_for_each_concurrent(Some(cpus), move |image| {
                use image::codecs::*;

                let formats = formats.clone();
                let config = config.clone();
                let pool = pool.clone();
                async move {
                    let filename = image.filename.clone();
                    log::debug!("thumbnailing {}", filename);
                    let generated_formats = tokio::task::spawn_blocking(move || {
                        let mut generated_formats = Vec::new();
                        let rgb = DynamicImage::from(image.image.to_rgb8());
                        for (format_name, format_config) in &*formats {
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
                                    if buf.len() > image.original_size {
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
                            if resized.len() < image.original_size {
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
                    sqlx::query!(
                        "UPDATE files SET thumbnails = ?, thumbnail_time = ? WHERE filename = ?",
                        formats_data,
                        ts,
                        filename
                    )
                        .execute(&pool)
                        .await?;
                    Ok(())
                }
            })
        }))
    } else {
        None
    };
    
    // OCR
    let ocr: Option<JoinHandle<Result<()>>> = if config.service.enable_ocr {
        let client = client.clone();
        let pool = pool.clone();
        let stream = ReceiverStream::new(to_ocr_rx).map(Ok);
        Some(tokio::spawn({
            stream.try_for_each_concurrent(Some(config.service.ocr_concurrency), move |image| {
                let client = client.clone();
                let pool = pool.clone();
                async move {
                    log::debug!("OCRing {}", image.filename);
                    let scan = match scan_image(&client, &image.image).await {
                        Ok(scan) => scan,
                        Err(e) => {
                            IMAGES_OCRED_ERROR_COUNTER.inc();
                            log::error!("OCR failure {}: {}", image.filename, e);
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
                    sqlx::query!(
                        "UPDATE files SET ocr = ?, raw_ocr_segments = ?, ocr_time = ? WHERE filename = ?",
                        ocr_text,
                        ocr_data,
                        ts,
                        image.filename
                    )
                        .execute(&pool)
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
        // keep multiple embedding requests in flight
        stream.map(Ok).try_for_each_concurrent(Some(3), move |batch| {
            let client = client.clone();
            let config = config.clone();
            let pool = pool.clone();
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
                    log::debug!("embedded {}", batch[i].filename);
                    IMAGES_EMBEDDED_COUNTER.inc();
                    sqlx::query!(
                        "UPDATE files SET embedding_time = ?, embedding = ? WHERE filename = ?",
                        ts,
                        vector,
                        batch[i].filename
                    )
                        .execute(&mut *tx)
                        .await?;
                }
                tx.commit().await?;
                anyhow::Result::Ok(())
            }
        })
    });
                
    let mut filenames = HashMap::new();
    
    // blocking OS calls
    tokio::task::block_in_place(|| -> anyhow::Result<()> {
        for entry in WalkDir::new(config.service.files.as_str()) {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                let filename = path.strip_prefix(&config.service.files)?.to_str().unwrap().to_string();
                let modtime = entry.metadata()?.modified()?.duration_since(std::time::UNIX_EPOCH)?;
                let modtime = modtime.as_micros() as i64;
                filenames.insert(filename.clone(), (path.to_path_buf(), modtime));
            }
        }
        Ok(())
    })?;

    log::debug!("finished reading filenames");

    for (filename, (_path, modtime)) in filenames.iter() {
        let modtime = *modtime;
        let record = sqlx::query_as!(FileRecord, "SELECT * FROM files WHERE filename = ?", filename)
            .fetch_optional(&pool)
            .await?;

        let new_record = match record {
            None => Some(FileRecord {
                filename: filename.clone(),
                ..Default::default()
            }),
            Some(r) if modtime > r.embedding_time.unwrap_or(i64::MIN) || (modtime > r.ocr_time.unwrap_or(i64::MIN) && config.service.enable_ocr) || (modtime > r.thumbnail_time.unwrap_or(i64::MIN) && config.service.enable_thumbs) => {
                Some(r)
            },
            _ => None
        };
        if let Some(mut record) = new_record {
            log::debug!("processing {}", record.filename);
            sqlx::query!("INSERT OR IGNORE INTO files (filename) VALUES (?)", filename)
                .execute(&pool)
                .await?;
            if modtime > record.embedding_time.unwrap_or(i64::MIN) {
                record.embedding = None;
            }
            if modtime > record.ocr_time.unwrap_or(i64::MIN) {
                record.raw_ocr_segments = None;
            }
            if modtime > record.thumbnail_time.unwrap_or(i64::MIN) {
                record.thumbnails = None;
            }
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
    
    let stored: Vec<String> = sqlx::query_scalar("SELECT filename FROM files").fetch_all(&pool).await?;
    let mut tx = pool.begin().await?;
    for filename in stored {
        if !filenames.contains_key(&filename) {
            sqlx::query!("DELETE FROM files WHERE filename = ?", filename)
                .execute(&mut *tx)
                .await?;
        }
    }
    tx.commit().await?;

    log::info!("ingest done");
    
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

    let mut rows = sqlx::query_as::<_, FileRecord>("SELECT * FROM files").fetch(&pool);
    while let Some(record) = rows.try_next().await? {
        if let Some(emb) = record.embedding {
            index.filenames.push(record.filename);
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

            let mut format_code = 0;
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
}

async fn query_index(index: &IIndex, query: EmbeddingVector, k: usize) -> Result<QueryResult> {
    let result = index.vectors.search(&query, k as usize)?;

    let items = result.distances
        .into_iter()
        .zip(result.labels)
        .filter_map(|(distance, id)| {
            let id = id.get()? as usize;
            Some((
                distance,
                index.filenames[id].clone(),
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
    let qres = query_index(index, total_embedding.to_vec(), k).await?;

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