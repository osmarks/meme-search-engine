use image::codecs::bmp::BmpEncoder;
use lazy_static::lazy_static;
use serde::{Serialize, Deserialize};
use std::borrow::Borrow;
use std::cell::RefCell;
use image::{DynamicImage, ExtendedColorType, ImageEncoder};
use anyhow::Result;
use std::io::Cursor;
use reqwest::Client;
use tracing::instrument;
use fast_image_resize::{Resizer, ResizeOptions, ResizeAlg};
use fast_image_resize::images::{Image, ImageRef};
use anyhow::Context;
use std::collections::HashMap;
use ndarray::ArrayBase;
use prometheus::{register_int_counter_vec, IntCounterVec};
use base64::prelude::*;
use std::future::Future;

std::thread_local! {
    static RESIZER: RefCell<Resizer> = RefCell::new(Resizer::new());
}

#[derive(Debug, Deserialize, Clone)]
pub struct InferenceServerConfig {
    pub batch: usize,
    pub image_size: (u32, u32),
    pub embedding_size: usize,
}

pub fn resize_for_embed_sync<T: Borrow<DynamicImage> + Send + 'static>(config: &InferenceServerConfig, image: T) -> Result<Vec<u8>> {
    // the model currently in use wants aspect ratio 1:1 regardless of input
    // I think this was previously being handled in the CLIP server but that is slightly lossy

    let src_rgb = match image.borrow() {
        DynamicImage::ImageRgb8(x) => x,
        x => &x.to_rgb8()
    };
    let (original_width, original_height) = src_rgb.dimensions();
    let src_rgb = ImageRef::new(original_width, original_height, src_rgb.as_raw(), fast_image_resize::PixelType::U8x3)?;

    let mut dst_image = Image::new(config.image_size.0, config.image_size.1, fast_image_resize::PixelType::U8x3);
    // use Hamming for downscaling, Lanczos3 for upscaling (I don't expect much upscaling)
    let opts = ResizeOptions::default().resize_alg(ResizeAlg::Convolution(if original_width > config.image_size.0 && original_height > config.image_size.1 { fast_image_resize::FilterType::Hamming } else { fast_image_resize::FilterType::Lanczos3 }));

    RESIZER.with_borrow_mut(|resizer| {
        resizer.resize(&src_rgb, &mut dst_image, Some(&opts))
    }).context("resize failure")?;

    let mut buf = Vec::new();
    let mut csr = Cursor::new(&mut buf);
    BmpEncoder::new(&mut csr).write_image(dst_image.buffer(), config.image_size.0, config.image_size.1, ExtendedColorType::Rgb8)?;
    Ok::<Vec<u8>, anyhow::Error>(buf)
}

pub async fn resize_for_embed<T: Borrow<DynamicImage> + Send + 'static>(config: InferenceServerConfig, image: T) -> Result<Vec<u8>> {
    let resized = tokio::task::spawn_blocking(move || resize_for_embed_sync(&config, image)).await??;
    Ok(resized)
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum EmbeddingRequest {
    Images { images: Vec<serde_bytes::ByteBuf> },
    Text { text: Vec<String> }
}

async fn fetch_backend_config(base_url: &str) -> Result<InferenceServerConfig> {
    let res = Client::new().get(&format!("{}/config", base_url)).send().await?;
    Ok(rmp_serde::from_slice(&res.bytes().await?)?)
}

pub async fn get_backend_config(clip_server: &str) -> InferenceServerConfig {
    loop {
        match fetch_backend_config(&clip_server).await {
            Ok(backend) => break backend,
            Err(e) => {
                tracing::error!("Backend failed (fetch): {}", e);
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        }
    }
}

#[instrument(skip(client, data))]
pub async fn query_clip_server<I, O>(client: &Client, base_url: &str, path: &str, data: I) -> Result<O> where I: Serialize, O: serde::de::DeserializeOwned,
{
    let response = client
        .post(&format!("{}{}", base_url, path))
        .header("Content-Type", "application/msgpack")
        .body(rmp_serde::to_vec_named(&data)?)
        .send()
        .await?;
    let result: O = rmp_serde::from_slice(&response.bytes().await?)?;
    Ok(result)
}

pub fn decode_fp16_buffer(buf: &[u8]) -> Vec<f32> {
    buf.chunks_exact(2)
        .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect()
}

pub fn chunk_fp16_buffer(buf: &[u8]) -> Vec<half::f16> {
    buf.chunks_exact(2)
        .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]))
        .collect()
}

#[derive(Clone, Deserialize, Serialize, Debug, PartialEq)]
pub struct OriginalImageMetadata {
    pub mime_type: String,
    pub original_file_size: usize,
    pub dimension: (u32, u32),
    pub final_url: String
}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct ProcessedEntry {
    pub url: String,
    pub id: String,
    pub title: String,
    pub subreddit: String,
    pub author: String,
    pub timestamp: u64,
    #[serde(with="serde_bytes")]
    pub embedding: Vec<u8>,
    pub metadata: OriginalImageMetadata
}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct ShardInputHeader {
    pub id: u32,
    pub centroid: Vec<f32>
}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct ShardedRecord {
    pub id: u32,
    #[serde(with="serde_bytes")]
    pub vector: Vec<u8> // FP16
}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct ShardHeader {
    pub id: u32,
    pub max: u32,
    pub centroid: Vec<f32>,
    pub medioid: u32,
    pub offsets: Vec<u64>,
    pub mapping: Vec<u32>
}

#[derive(Clone, Debug, bitcode::Encode, bitcode::Decode)]
pub struct PackedIndexEntry {
    pub vector: Vec<u16>, // FP16 values cast to u16 for storage
    pub vertices: Vec<u32>,
    pub id: u32,
    pub timestamp: u64,
    pub dimensions: (u32, u32),
    pub scores: Vec<f32>,
    pub url: String,
    pub shards: Vec<u32>
}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct IndexHeader {
    pub shards: Vec<(Vec<f32>, u32)>,
    pub count: u32,
    pub dead_count: u32,
    pub record_pad_size: usize,
    pub quantizer: diskann::vector::ProductQuantizer,
    pub descriptor_cdfs: Vec<Vec<f32>>
}

#[derive(Serialize, Deserialize)]
pub struct FrontendInit {
    pub n_total: u64,
    pub predefined_embedding_names: Vec<String>,
    pub d_emb: usize
}

pub type EmbeddingVector = Vec<f32>;

#[derive(Debug, Serialize)]
pub struct QueryResult {
    pub matches: Vec<(f32, String, String, u64, Option<(u32, u32)>)>,
    pub formats: Vec<String>,
    pub extensions: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
pub struct QueryTerm {
    pub embedding: Option<EmbeddingVector>,
    pub image: Option<String>,
    pub text: Option<String>,
    pub predefined_embedding: Option<String>,
    pub weight: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    pub terms: Vec<QueryTerm>,
    pub k: Option<usize>,
    #[serde(default)]
    pub include_video: bool
}

lazy_static::lazy_static! {
    static ref TERMS_COUNTER: IntCounterVec = register_int_counter_vec!("mse_terms", "terms used in queries, by type", &["type"]).unwrap();
}

pub async fn get_total_embedding<A: Future<Output = Result<Vec<Vec<u8>>>>, B: Future<Output = Result<serde_bytes::ByteBuf>>, S: Clone, T: Clone, F: Fn(EmbeddingRequest, S) -> A, G: Fn(Vec<u8>, T) -> B>(terms: &Vec<QueryTerm>, ic: &InferenceServerConfig, query_server: F, resize_image: G, predefined_embeddings: &HashMap<String, ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 1]>>>, image_state: T, query_state: S) -> Result<Vec<f32>> {
    let mut total_embedding = ndarray::Array::from(vec![0.0; ic.embedding_size]);

    let mut image_batch = Vec::new();
    let mut image_weights = Vec::new();
    let mut text_batch = Vec::new();
    let mut text_weights = Vec::new();

    for term in terms {
        if let Some(image) = &term.image {
            TERMS_COUNTER.get_metric_with_label_values(&["image"]).unwrap().inc();
            let bytes = BASE64_STANDARD.decode(image)?;
            image_batch.push(resize_image(bytes, image_state.clone()).await?);
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
            let embedding = predefined_embeddings.get(name).context("name invalid")?;
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
        let embs: Vec<Vec<u8>> = query_server(batch, query_state.clone()).await?;
        for (emb, weight) in embs.into_iter().zip(weights) {
            total_embedding += &(ndarray::Array::from_vec(decode_fp16_buffer(&emb)) * weight);
        }
    }

    Ok(total_embedding.to_vec())
}
