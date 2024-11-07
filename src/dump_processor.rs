use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use std::io::BufReader;
use rmp_serde::decode::Error as DecodeError;
use std::fs;
use base64::{engine::general_purpose::URL_SAFE, Engine as _};

// TODO refactor
#[derive(Clone, Deserialize, Serialize, Debug, PartialEq)]
struct OriginalImageMetadata {
    mime_type: String,
    original_file_size: usize,
    dimension: (u32, u32),
    final_url: String
}

#[derive(Clone, Deserialize, Serialize, Debug)]
struct ProcessedEntry {
    url: String,
    id: String,
    title: String,
    subreddit: String,
    author: String,
    timestamp: u64,
    #[serde(with = "serde_bytes")]
    embedding: Vec<u8>,
    metadata: OriginalImageMetadata
}

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("missing path")?;
    let stream = zstd::stream::Decoder::new(fs::File::open(path)?)?;
    let mut stream = BufReader::new(stream);
    let mut latest_timestamp = 0;
    let mut count = 0;
    loop {
        let res: Result<ProcessedEntry, DecodeError> = rmp_serde::from_read(&mut stream);
        if res.is_ok() {
            count += 1;
        }
        match res {
            Ok(x) => {
                if x.timestamp > latest_timestamp {
                    println!("{} {} https://reddit.com/r/{}/comments/{} {} https://mse.osmarks.net/?e={}", x.timestamp, count, x.subreddit, x.id, x.metadata.final_url, URL_SAFE.encode(x.embedding));
                    latest_timestamp = x.timestamp;
                }
            },
            Err(DecodeError::InvalidDataRead(x)) | Err(DecodeError::InvalidMarkerRead(x)) if x.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e).context("decode fail")
        }
    }
    println!("{} {}", latest_timestamp, count);
    Ok(())
}
