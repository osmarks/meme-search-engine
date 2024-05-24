use serde::{Serialize, Deserialize};
use std::borrow::Borrow;
use image::{DynamicImage, imageops::FilterType, ImageFormat};
use anyhow::Result;
use std::io::Cursor;
use reqwest::Client;

#[derive(Debug, Deserialize, Clone)]
pub struct InferenceServerConfig {
    pub batch: usize,
    pub image_size: (u32, u32),
    pub embedding_size: usize,
}

pub async fn resize_for_embed<T: Borrow<DynamicImage> + Send + 'static>(config: InferenceServerConfig, image: T) -> Result<Vec<u8>> {
    let resized = tokio::task::spawn_blocking(move || {
        let new = image.borrow().resize(
            config.image_size.0,
            config.image_size.1,
            FilterType::Lanczos3
        );
        let mut buf = Vec::new();
        let mut csr = Cursor::new(&mut buf);
        new.write_to(&mut csr, ImageFormat::Png)?;
        Ok::<Vec<u8>, anyhow::Error>(buf)
    }).await??;
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
                log::error!("Backend failed (fetch): {}", e);
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        }
    }
}

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
