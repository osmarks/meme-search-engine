use anyhow::{anyhow, Result};
use image::{DynamicImage, GenericImageView, ImageFormat};
use regex::Regex;
use reqwest::{
    header::{HeaderMap, HeaderValue},
    multipart::{Form, Part},
    Client,
};
use serde_json::Value;
use std::{io::Cursor, time::{SystemTime, UNIX_EPOCH}};
use serde::{Deserialize, Serialize};
use tracing::instrument;

const CALLBACK_REGEX: &str = r">AF_initDataCallback\((\{key: 'ds:1'.*?\})\);</script>";
const MAX_DIM: u32 = 1024;

#[derive(Debug, Serialize, Deserialize)]
pub struct SegmentCoords {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Segment {
    pub coords: SegmentCoords,
    pub text: String,
}

pub type ScanResult = Vec<Segment>;

fn rationalize_coords_format1(
    image_w: f64,
    image_h: f64,
    center_x_fraction: f64,
    center_y_fraction: f64,
    width_fraction: f64,
    height_fraction: f64,
) -> SegmentCoords {
    SegmentCoords {
        x: ((center_x_fraction - width_fraction / 2.0) * image_w).round() as i32,
        y: ((center_y_fraction - height_fraction / 2.0) * image_h).round() as i32,
        w: (width_fraction * image_w).round() as i32,
        h: (height_fraction * image_h).round() as i32,
    }
}

#[instrument(skip(client, image))]
async fn scan_image_chunk(
    client: &Client,
    image: &[u8],
    image_width: u32,
    image_height: u32,
) -> Result<ScanResult> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros();

    let part = Part::bytes(image.to_vec())
        .file_name(format!("ocr{}.png", timestamp))
        .mime_str("image/png")?;

    let form = Form::new().part("encoded_image", part);

    let mut headers = HeaderMap::new();
    headers.insert(
        "User-Agent",
        HeaderValue::from_static("Mozilla/5.0 (Linux; Android 13; RMX3771) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.6167.144 Mobile Safari/537.36"),
    );
    headers.insert("Cookie", HeaderValue::from_str(&format!("SOCS=CAESEwgDEgk0ODE3Nzk3MjQaAmVuIAEaBgiA_LyaBg; stcs={}", timestamp))?);

    let response = client
        .post(&format!("https://lens.google.com/v3/upload?stcs={}", timestamp))
        .multipart(form)
        .headers(headers)
        .send()
        .await?;

    let body = response.text().await?;

    let re = Regex::new(CALLBACK_REGEX)?;
    let captures = re
        .captures(&body)
        .ok_or_else(|| anyhow!("invalid API response"))?;
    let match_str = captures.get(1).unwrap().as_str();

    let lens_object: Value = json5::from_str(match_str)?;

    if lens_object.get("errorHasStatus").is_some() {
        return Err(anyhow!("lens failed"));
    }

    let root = lens_object["data"].as_array().unwrap();

    let mut text_segments = Vec::new();
    let mut text_regions = Vec::new();

    let text_segments_raw = root[3][4][0][0]
        .as_array()
        .ok_or_else(|| anyhow!("invalid text segments"))?;
    let text_regions_raw = root[2][3][0]
        .as_array()
        .ok_or_else(|| anyhow!("invalid text regions"))?;

    for region in text_regions_raw {
        let region_data = region.as_array().unwrap();
        if region_data[11].as_str().unwrap().starts_with("text:") {
            let raw_coords = region_data[1].as_array().unwrap();
            let coords = rationalize_coords_format1(
                image_width as f64,
                image_height as f64,
                raw_coords[0].as_f64().unwrap(),
                raw_coords[1].as_f64().unwrap(),
                raw_coords[2].as_f64().unwrap(),
                raw_coords[3].as_f64().unwrap(),
            );
            text_regions.push(coords);
        }
    }

    for segment in text_segments_raw {
        let text_segment = segment.as_str().unwrap().to_string();
        text_segments.push(text_segment);
    }

    Ok(text_segments
        .into_iter()
        .zip(text_regions.into_iter())
        .map(|(text, coords)| Segment { text, coords })
        .collect())
}

#[instrument(skip(client))]
pub async fn scan_image(client: &Client, image: &DynamicImage) -> Result<ScanResult> {
    let mut result = ScanResult::new();
    let (width, height) = image.dimensions();

    let (width, height, image) = if width > MAX_DIM {
        let height = ((height as f64) * (MAX_DIM as f64) / (width as f64)).round() as u32;
        let new_image = tokio::task::block_in_place(|| image.resize_exact(MAX_DIM, height, image::imageops::FilterType::CatmullRom));
        (MAX_DIM, height, std::borrow::Cow::Owned(new_image))
    } else {
        (width, height, std::borrow::Cow::Borrowed(image))
    };

    let mut y = 0;
    while y < height {
        let chunk_height = (height - y).min(MAX_DIM);
        let chunk = tokio::task::block_in_place(|| {
            let chunk = image.view(0, y, width, chunk_height).to_image();
            let mut buf = Vec::new();
            let mut csr = Cursor::new(&mut buf);
            chunk.write_to(&mut csr, ImageFormat::Png)?;
            Ok::<Vec<u8>, anyhow::Error>(buf)
        })?;

        let res = scan_image_chunk(client, &chunk, width, chunk_height).await?;
        for segment in res {
            result.push(Segment {
                text: segment.text,
                coords: SegmentCoords {
                    y: segment.coords.y + y as i32,
                    x: segment.coords.x,
                    w: segment.coords.w,
                    h: segment.coords.h,
                },
            });
        }

        y += chunk_height;
    }

    Ok(result)
}
