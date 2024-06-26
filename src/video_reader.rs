extern crate ffmpeg_the_third as ffmpeg;
use anyhow::{Result, Context};
use image::RgbImage;
use std::env;
use ffmpeg::{codec, filter, format::{self, Pixel}, media::Type, util::frame::video::Video};

const BYTES_PER_PIXEL: usize = 3;

pub fn run<P: AsRef<std::path::Path>, F: FnMut(RgbImage) -> Result<()>>(path: P, mut frame_callback: F, frame_interval: f32) -> Result<()> {
    let mut ictx = format::input(&path).context("parsing video")?;
    let video = ictx.streams().best(Type::Video).context("no video stream")?;
    let video_index = video.index();

    let context_decoder = codec::Context::from_parameters(video.parameters()).context("initializing decoder context")?;
    let mut decoder = context_decoder.decoder().video().context("initializing decoder")?;

    let mut graph = filter::Graph::new();
    let afr = video.avg_frame_rate();
    let afr = ((frame_interval * (afr.0 as f32) / (afr.1 as f32)).round() as i64).max(1);

    // I don't actually know what this does, but in some cases it's 0/0 so "correct" for that.
    let aspect_ratio = decoder.aspect_ratio();
    let mut aspect_ratio = (aspect_ratio.0, aspect_ratio.1);
    if aspect_ratio == (0, 0) {
        aspect_ratio = (1, 1);
    }

    graph.add(&filter::find("buffer").unwrap(), "in", 
        &format!("video_size={}x{}:pix_fmt={}:time_base={}/{}:pixel_aspect={}/{}", decoder.width(), decoder.height(), decoder.format().descriptor().unwrap().name(), video.time_base().0, video.time_base().1, aspect_ratio.0, aspect_ratio.1))?;
    graph.add(&filter::find("buffersink").unwrap(), "out", "")?;
    // I don't know exactly where, but some of my videos apparently have the size vary throughout them.
    // This causes horrible segfaults somewhere.
    // Rescale to initial width to fix this. We could do this with a separate swscaler but this is easier.
    let filterspec = format!("[in] scale={}:{} [scaled]; [scaled] thumbnail=n={}:log=verbose [thumbs]; [thumbs] select='gt(scene,0.05)+eq(n,0)' [out]", decoder.width(), decoder.height(), afr);
    graph.output("in", 0)?.input("out", 0)?.parse(&filterspec).context("filtergraph parse failed")?;
    let mut out = graph.get("out").unwrap();
    out.set_pixel_format(Pixel::RGB24);

    graph.validate().context("filtergraph build failed")?;

    let mut receive_and_process_decoded_frames = |decoder: &mut ffmpeg::decoder::Video, filter_graph: &mut filter::Graph| -> Result<()> {
        let mut decoded = Video::empty();
        let mut filtered = Video::empty();
        loop {
            if !decoder.receive_frame(&mut decoded).is_ok() { break }

            let mut in_ctx = filter_graph.get("in").unwrap();
            // The filters really do not like 
            let mut src = in_ctx.source();
            src.add(&decoded).context("add frame")?;
            
            while filter_graph.get("out").unwrap().sink().frame(&mut filtered).is_ok() {
                let mut image = vec![0u8; filtered.width() as usize * filtered.height() as usize * BYTES_PER_PIXEL];
                let stride = filtered.stride(0);
                let data = filtered.data(0);
                let width = filtered.width() as usize * BYTES_PER_PIXEL;
                let height = filtered.height() as usize;
                for y in 0..height {
                    image[y * width .. (y + 1) * width].copy_from_slice(&data[y * stride .. y * stride + width]);
                }
                frame_callback(image::ImageBuffer::from_vec(filtered.width(), filtered.height(), image).unwrap())?;
            }
        }
        Ok(())
    };

    for (stream, packet) in ictx.packets().filter_map(Result::ok) {
        if stream.index() == video_index {
            decoder.send_packet(&packet).context("decoder")?;
            receive_and_process_decoded_frames(&mut decoder, &mut graph).context("processing frame")?;
        }
    }
    decoder.send_eof()?;
    receive_and_process_decoded_frames(&mut decoder, &mut graph)?;

    Ok(())
}

fn main() -> Result<()> {
    let mut count = 0;
    let callback = |frame: RgbImage| {
        frame.save(format!("/tmp/output-{}.png", count))?;
        count += 1;
        Ok(())
    };
    run(&env::args().nth(1).unwrap(), callback, 1.0)
}