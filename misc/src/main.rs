use serde::{Deserialize, Serialize};
use tower_http::{services::ServeDir, add_extension::AddExtensionLayer, services::ServeFile};
use axum::{extract::{Json, Extension, Multipart, Path as AxumPath}, http::{StatusCode, Request}, response::{IntoResponse}, body::Body, routing::{get, post, get_service}, Router};
use std::sync::Arc;
use tokio::{sync::RwLock, runtime::Handle, fs::File};
use anyhow::Result;
use rusqlite::Connection;
use std::collections::HashMap;
use futures::{stream, StreamExt};
use std::path::Path;
use image::{io::Reader as ImageReader, imageops};
use tokio::task::block_in_place;
use rayon::prelude::*;
use std::io::Cursor;

mod util;

use util::CONFIG;

#[derive(Serialize, Deserialize)]
struct InferenceServerConfig {
    model: String,
    embedding_size: usize,
    batch: usize,
    image_size: usize
}

struct Index {
    vectors: faiss::FlatIndex // we need the index to implement Send, which an arbitrary boxed one might not
}

async fn build_index() -> Result<Index> {
    let mut conn = block_in_place(|| Connection::open(&CONFIG.db_path))?;
    block_in_place(|| conn.execute("CREATE TABLE IF NOT EXISTS files (
        filename TEXT PRIMARY KEY,
        modtime REAL NOT NULL,
        embedding_vector BLOB NOT NULL
    )", ()))?;

    let SIZE = 1024; // TODO
    let IMAGE_SIZE = 384; // TODO
    let BS = 32;

    let mut files = HashMap::new();
    let mut new_files: HashMap<String, (i64, Option<Vec<u8>>)> = HashMap::new();
    let mut vectors = faiss::index_factory(SIZE, "Flat", faiss::MetricType::InnerProduct)?.into_flat()?;

    block_in_place(|| -> Result<()> {
        let mut stmt = conn.prepare_cached("SELECT filename, modtime FROM files")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let filename: String = row.get(0)?;
            let modtime: i64 = row.get(1)?;
            files.insert(filename, modtime);
        }

        for entry in walkdir::WalkDir::new(&CONFIG.images_path).follow_links(true) {
            let entry = entry?;
            if entry.file_type().is_file() {
                let metadata = entry.metadata()?;
                let modtime = metadata.modified()?.duration_since(std::time::UNIX_EPOCH)?.as_secs() as i64;
                let filename = entry.path().strip_prefix(&CONFIG.images_path)?.to_string_lossy().to_string();
                match files.get(&filename) {
                    Some(old_modtime) if *old_modtime < modtime => new_files.insert(filename, (modtime, None)),
                    None => new_files.insert(filename, (modtime, None)),
                    _ => None
                };
            }
        }

        Ok(())
    })?;

    let (itx, mut irx) = tokio::sync::mpsc::channel(BS * 2);

    let new_files_ = new_files.clone();
    let image_reader_task = tokio::task::spawn_blocking(move || {
        new_files_.par_iter().try_for_each(|(filename, _)| -> Result<()> {
            let mut path = Path::new(&CONFIG.images_path).to_path_buf();
            path.push(filename);
            let image = ImageReader::open(path)?.with_guessed_format()?.decode()?;
            let resized = imageops::resize(&image.into_rgb8(), IMAGE_SIZE, IMAGE_SIZE, imageops::Lanczos3);
            let mut bytes: Vec<u8> = Vec::new();
            resized.write_to(&mut Cursor::new(&mut bytes), image::ImageOutputFormat::Png)?;
            itx.blocking_send((filename.to_string(), bytes))?;
            Ok(())
        })
    });

    let dispatch_batch = |batch| {

    };

    let mut batch = vec![];
    while let Some((filename, image)) = irx.recv().await {
        if batch.len() == BS {
            dispatch_batch(std::mem::replace(&mut batch, vec![]));
        }
        batch.push((filename, image));
    }
    if batch.len() > 0 {
        dispatch_batch(std::mem::replace(&mut batch, vec![]));
    }

    // TODO switch to blocking
    {
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare_cached("INSERT OR REPLACE INTO files VALUES (?, ?, ?)")?;
            for (filename, (modtime, embedding)) in new_files {
                stmt.execute((filename, modtime, embedding.unwrap()))?;
            }
        }
        tx.commit()?;
    }

    Ok(Index {
        vectors: vectors
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var_os("RUST_LOG").is_none() {
        std::env::set_var("RUST_LOG", format!("meme-search-engine={}", CONFIG.log_level))
    }

    let notify = tokio::sync::Notify::new();

    tokio::spawn(async move {
        loop {
            notify.notified().await;
            let index = build_index().await.unwrap();
        }
    });

    tracing_subscriber::fmt::init();

    //let db = Arc::new(RwLock::new(DB::init().await?));

    let app = Router::new()
        .route("/", get(health))
        .route("/", post(run_query));
        //.layer(AddExtensionLayer::new(db));

    let addr = CONFIG.listen_address.parse().unwrap();
    tracing::info!("listening on {}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await?;
    Ok(())
}

async fn health() -> String {
    format!("OK")
}

#[derive(Debug, Serialize, Deserialize)]
struct RawQuery {
    text: Vec<String>,
    images: Vec<String> // base64 (sorry)
}

async fn run_query(query: Json<RawQuery>) -> Json<Vec<(String, f32)>> {
    tracing::info!("{:?}", query);
    Json(vec![])
}