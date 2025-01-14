use image::codecs::bmp::BmpEncoder;
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

std::thread_local! {
    static RESIZER: RefCell<Resizer> = RefCell::new(Resizer::new());
}

#[derive(Debug, Deserialize, Clone)]
pub struct InferenceServerConfig {
    pub batch: usize,
    pub image_size: (u32, u32),
    pub embedding_size: usize,
}

pub fn resize_for_embed_sync<T: Borrow<DynamicImage> + Send + 'static>(config: InferenceServerConfig, image: T) -> Result<Vec<u8>> {
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
    let resized = tokio::task::spawn_blocking(move || resize_for_embed_sync(config, image)).await??;
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
    pub centroid: Vec<f32>,
    pub max_query_id: usize
}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct ShardedRecord {
    pub id: u32,
    #[serde(with="serde_bytes")]
    pub vector: Vec<u8>, // FP16
    pub query_knns: Vec<u32>,
    pub query_knns_distances: Vec<f32>
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
    pub score: f32,
    pub url: String,
    pub shards: Vec<u32>
}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct IndexHeader {
    pub shards: Vec<(Vec<f32>, u32)>,
    pub count: u32,
    pub dead_count: u32,
    pub record_pad_size: usize,
    pub quantizer: diskann::vector::ProductQuantizer
}

pub mod index_config {
    use diskann::IndexBuildConfig;

    pub const BASE_CONFIG: IndexBuildConfig = IndexBuildConfig {
        r: 40,
        l: 200,
        maxc: 900,
        alpha: 65200,
        saturate_graph: false
    };

    pub const PROJECTION_CUT_POINT: usize = 3;

    pub const FIRST_PASS_ALPHA: i64 = 65200;

    //pub const SECOND_PASS_ALPHA: i64 = 62000;

    pub const QUERY_SEARCH_K: usize = 200; // we want each query to have QUERY_REVERSE_K results, but some queries are likely more common than others in the top-k lists, so oversample a bit
    pub const QUERY_REVERSE_K: usize = 100;
}
