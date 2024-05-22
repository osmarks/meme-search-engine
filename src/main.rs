use std::{collections::HashMap, io::Cursor};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, Context};
use axum::body::Body;
use axum::response::Response;
use axum::{
    extract::Json,
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
use faiss::Index;
use futures_util::stream::{StreamExt, TryStreamExt};
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;

mod ocr;

use crate::ocr::scan_image;

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
    vectors: faiss::index::IndexImpl,
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

CREATE TRIGGER IF NOT EXISTS ocr_fts_ins AFTER INSERT ON files BEGIN
    INSERT INTO ocr_fts (rowid, filename, ocr) VALUES (new.rowid, new.filename, COALESCE(new.ocr, ''));
END;

CREATE TRIGGER IF NOT EXISTS ocr_fts_del AFTER DELETE ON files BEGIN
    INSERT INTO ocr_fts (ocr_fts, rowid, filename, ocr) VALUES ('delete', old.rowid, old.filename, COALESCE(old.ocr, ''));
END;

CREATE TRIGGER IF NOT EXISTS ocr_fts_upd AFTER UPDATE ON files BEGIN
    INSERT INTO ocr_fts (ocr_fts, rowid, filename, ocr) VALUES ('delete', old.rowid, old.filename, COALESCE(old.ocr, ''));
    INSERT INTO ocr_fts (rowid, filename, text) VALUES (new.rowid, new.filename, COALESCE(new.ocr, ''));
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

#[derive(Debug, Deserialize, Clone)]
struct InferenceServerConfig {
    batch: usize,
    image_size: (u32, u32),
    embedding_size: usize,
}

async fn query_clip_server<I, O>(
    client: &Client,
    config: &Config,
    path: &str,
    data: I,
) -> Result<O> where I: Serialize, O: serde::de::DeserializeOwned,
{
    let response = client
        .post(&format!("{}{}", config.clip_server, path))
        .header("Content-Type", "application/msgpack")
        .body(rmp_serde::to_vec_named(&data)?)
        .send()
        .await?;
    let result: O = rmp_serde::from_slice(&response.bytes().await?)?;
    Ok(result)
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

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum EmbeddingRequest {
    Images { images: Vec<serde_bytes::ByteBuf> },
    Text { text: Vec<String> }
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

async fn resize_for_embed(backend_config: Arc<InferenceServerConfig>, image: Arc<DynamicImage>) -> Result<Vec<u8>> {
    let resized = tokio::task::spawn_blocking(move || {
        let new = image.resize(
            backend_config.image_size.0,
            backend_config.image_size.1,
            FilterType::Lanczos3
        );
        let mut buf = Vec::new();
        let mut csr = Cursor::new(&mut buf);
        new.write_to(&mut csr, ImageFormat::Png)?;
        Ok::<Vec<u8>, anyhow::Error>(buf)
    }).await??;
    Ok(resized)
}

async fn ingest_files(config: Arc<Config>, backend: Arc<InferenceServerConfig>) -> Result<()> {
    let pool = initialize_database(&config).await?;
    let client = Client::new();
    
    let formats = image_formats(&config);
    
    let (to_process_tx, to_process_rx) = mpsc::channel::<FileRecord>(100);
    let (to_embed_tx, to_embed_rx) = mpsc::channel(backend.batch as usize);
    let (to_thumbnail_tx, to_thumbnail_rx) = mpsc::channel(30);
    let (to_ocr_tx, to_ocr_rx) = mpsc::channel(30);
    
    let cpus = num_cpus::get();

    // Image loading and preliminary resizing
    let image_loading: JoinHandle<Result<()>> = tokio::spawn({
        let config = config.clone();
        let backend = backend.clone();
        let stream = ReceiverStream::new(to_process_rx).map(Ok);
        stream.try_for_each_concurrent(Some(cpus), move |record| {
            let config = config.clone();
            let backend = backend.clone();
            let to_embed_tx = to_embed_tx.clone();
            let to_thumbnail_tx = to_thumbnail_tx.clone();
            let to_ocr_tx = to_ocr_tx.clone();
            async move {
                let path = Path::new(&config.files).join(&record.filename);
                let image: Result<Arc<DynamicImage>> = tokio::task::block_in_place(|| Ok(Arc::new(ImageReader::open(&path)?.with_guessed_format()?.decode()?)));
                let image = match image {
                    Ok(image) => image,
                    Err(e) => {
                        log::error!("Could not read {}: {}", record.filename, e);
                        return Ok(())
                    }
                };
                if record.embedding.is_none() {
                    let resized = resize_for_embed(backend.clone(), image.clone()).await?;
                    
                    to_embed_tx.send(EmbeddingInput { image: resized, filename: record.filename.clone() }).await?
                }
                if record.thumbnails.is_none() && config.enable_thumbs {
                    to_thumbnail_tx
                    .send(LoadedImage {
                        image: image.clone(),
                        filename: record.filename.clone(),
                        original_size: std::fs::metadata(&path)?.len() as usize,
                    })
                    .await?;
                }
                if record.raw_ocr_segments.is_none() && config.enable_ocr {
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
    let thumbnail_generation: Option<JoinHandle<Result<()>>> = if config.enable_thumbs {
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
                                let thumbnail_path = Path::new(&config.thumbs_path).join(
                                    generate_thumbnail_filename(
                                        &image.filename,
                                        format_name,
                                        format_config,
                                    ),
                                );
                                std::fs::write(thumbnail_path, resized)?;
                            }
                        }
                        Ok::<Vec<String>, anyhow::Error>(generated_formats)
                    }).await??;
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
    let ocr: Option<JoinHandle<Result<()>>> = if config.enable_ocr {
        let client = client.clone();
        let pool = pool.clone();
        let stream = ReceiverStream::new(to_ocr_rx).map(Ok);
        Some(tokio::spawn({
            stream.try_for_each_concurrent(Some(config.ocr_concurrency), move |image| {
                let client = client.clone();
                let pool = pool.clone();
                async move {
                    log::debug!("OCRing {}", image.filename);
                    let scan = match scan_image(&client, &image.image).await {
                        Ok(scan) => scan,
                        Err(e) => {
                            log::error!("OCR failure {}: {}", image.filename, e);
                            return Ok(())
                        }
                    };
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
        let stream = ReceiverStream::new(to_embed_rx).chunks(backend.batch);
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
                    &config,
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
        for entry in WalkDir::new(config.files.as_str()) {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                let filename = path.strip_prefix(&config.files)?.to_str().unwrap().to_string();
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
            Some(r) if modtime > r.embedding_time.unwrap_or(i64::MIN) || (modtime > r.ocr_time.unwrap_or(i64::MIN) && config.enable_ocr) || (modtime > r.thumbnail_time.unwrap_or(i64::MIN) && config.enable_thumbs) => {
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

async fn build_index(config: Arc<Config>, backend: Arc<InferenceServerConfig>) -> Result<IIndex> {
    let pool = initialize_database(&config).await?;

    let mut index = IIndex {
        vectors: faiss::index_factory(backend.embedding_size as u32, "SQfp16", faiss::MetricType::InnerProduct)?,
        filenames: Vec::new(),
        format_codes: Vec::new(),
        format_names: Vec::new(),
    };

    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM files")
        .fetch_one(&pool)
        .await?;

    index.filenames = Vec::with_capacity(count as usize);
    index.format_codes = Vec::with_capacity(count as usize);
    let mut buffer = Vec::with_capacity(INDEX_ADD_BATCH * backend.embedding_size as usize);
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
    weight: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct QueryRequest {
    terms: Vec<QueryTerm>,
    k: Option<usize>,
}

async fn query_index(index: &mut IIndex, query: EmbeddingVector, k: usize) -> Result<QueryResult> {
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

async fn handle_request(
    config: &Config,
    backend_config: Arc<InferenceServerConfig>,
    client: Arc<Client>,
    index: &mut IIndex,
    req: Json<QueryRequest>,
) -> Result<Response<Body>> {
    let mut total_embedding = ndarray::Array::from(vec![0.0; backend_config.embedding_size]);

    let mut image_batch = Vec::new();
    let mut image_weights = Vec::new();
    let mut text_batch = Vec::new();
    let mut text_weights = Vec::new();

    for term in &req.terms {
        if let Some(image) = &term.image {
            let bytes = BASE64_STANDARD.decode(image)?;
            let image = Arc::new(tokio::task::block_in_place(|| image::load_from_memory(&bytes))?);
            image_batch.push(serde_bytes::ByteBuf::from(resize_for_embed(backend_config.clone(), image).await?));
            image_weights.push(term.weight.unwrap_or(1.0));
        }
        if let Some(text) = &term.text {
            text_batch.push(text.clone());
            text_weights.push(term.weight.unwrap_or(1.0));
        }
        if let Some(embedding) = &term.embedding {
            let weight = term.weight.unwrap_or(1.0);
            for (i, value) in embedding.iter().enumerate() {
                total_embedding[i] += value * weight;
            }
        }
    }
    
    let mut batches = vec![];    

    if !image_batch.is_empty() {
        batches.push(
            EmbeddingRequest::Images {
                images: image_batch
            }
        );
    }
    if !text_batch.is_empty() {
        batches.push(
            EmbeddingRequest::Text {
                text: text_batch,
            }
        );
    }

    for batch in batches {
        let embs: Vec<Vec<u8>> = query_clip_server(&client, config, "/", batch).await?;
        for emb in embs {
            total_embedding += &ndarray::Array::from_vec(decode_fp16_buffer(&emb));
        }
    }

    let k = req.k.unwrap_or(1000);
    let qres = query_index(index, total_embedding.to_vec(), k).await?;

    let mut extensions = HashMap::new();
    for (k, v) in image_formats(config) {
        extensions.insert(k, v.extension);
    }

    Ok(Json(QueryResult {
        matches: qres.matches,
        formats: qres.formats,
        extensions,
    }).into_response())
}

async fn get_backend_config(config: &Config) -> Result<InferenceServerConfig> {
    let res = Client::new().get(&format!("{}/config", config.clip_server)).send().await?;
    Ok(rmp_serde::from_slice(&res.bytes().await?)?)
}

#[tokio::main]
async fn main() -> Result<()> {
    pretty_env_logger::init();

    let config_path = std::env::args().nth(1).expect("Missing config file path");
    let config: Arc<Config> = Arc::new(serde_json::from_slice(&std::fs::read(config_path)?)?);

    let pool = initialize_database(&config).await?;
    sqlx::query(SCHEMA).execute(&pool).await?;

    let backend = Arc::new(loop {
        match get_backend_config(&config).await {
            Ok(backend) => break backend,
            Err(e) => {
                log::error!("Backend failed (fetch): {}", e);
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        }
    });

    if config.no_run_server {
        ingest_files(config.clone(), backend.clone()).await?;
        return Ok(())
    }

    let (request_ingest_tx, mut request_ingest_rx) = mpsc::channel(1);

    let index = Arc::new(tokio::sync::Mutex::new(build_index(config.clone(), backend.clone()).await?));

    let (ingest_done_tx, _ingest_done_rx) = broadcast::channel(1);
    let done_tx = Arc::new(ingest_done_tx.clone());

    let _ingest_task = tokio::spawn({
        let config = config.clone();
        let backend = backend.clone();
        let index = index.clone();
        async move {
            loop {
                log::info!("Ingest running");
                match ingest_files(config.clone(), backend.clone()).await {
                    Ok(_) => {
                        match build_index(config.clone(), backend.clone()).await {
                            Ok(new_index) => {
                                *index.lock().await = new_index;
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
                request_ingest_rx.recv().await;
            }
        }
    });

    let cors = CorsLayer::permissive();

    let config_ = config.clone();
    let client = Arc::new(Client::new());
    let app = Router::new()
        .route("/", post(|req| async move {
            let config = config.clone();
            let backend_config = backend.clone();
            let mut index = index.lock().await; // TODO: use ConcurrentIndex here
            let client = client.clone();
            handle_request(&config, backend_config, client.clone(), &mut index, req).await.map_err(|e| format!("{:?}", e))
        }))
        .route("/", get(|_req: axum::http::Request<axum::body::Body>| async move {
            "OK"
        }))
        .route("/reload", post(|_req: axum::http::Request<axum::body::Body>| async move {
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
        .layer(cors);

    let addr = format!("0.0.0.0:{}", config_.port);
    log::info!("Starting server on {}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await?;

    Ok(())
}            