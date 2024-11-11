use image::codecs::bmp::BmpEncoder;
use serde::{Serialize, Deserialize};
use std::borrow::Borrow;
use std::cell::RefCell;
use image::{DynamicImage, ExtendedColorType, ImageEncoder};
use anyhow::Result;
use std::io::Cursor;
use reqwest::Client;
use tracing::instrument;
use fast_image_resize::Resizer;
use fast_image_resize::images::Image;
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

    let src_rgb = DynamicImage::from(image.borrow().to_rgb8()); // TODO this might be significantly inefficient for RGB8->RGB8 case

    let mut dst_image = Image::new(config.image_size.0, config.image_size.1, fast_image_resize::PixelType::U8x3);

    RESIZER.with_borrow_mut(|resizer| {
        resizer.resize(&src_rgb, &mut dst_image, None)
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
