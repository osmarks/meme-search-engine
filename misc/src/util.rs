use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub log_level: String,
    pub listen_address: String,
    pub images_path: String,
    pub db_path: String,
    pub backend_url: String
}

fn load_config() -> Result<Config> {
    use config::{Config, File};
    let s = Config::builder()
        .add_source(File::with_name("./config"))
        .build().context("loading config")?;
    Ok(s.try_deserialize().context("parsing config")?)
}

lazy_static::lazy_static! {
    pub static ref CONFIG: Config = load_config().unwrap();
}