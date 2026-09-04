extern crate ffmpeg_the_third as ffmpeg;
use anyhow::{ensure, Context, Result};
use ffmpeg::{
    codec, filter,
    format::{self, Pixel},
    media::Type,
    util::frame::video::Video,
};
use image::{ImageBuffer, RgbImage};
use std::env;

const BYTES_PER_PIXEL: usize = 3;

fn frame_to_image(frame: &Video) -> Result<RgbImage> {
    ensure!(
        frame.format() == Pixel::RGB24,
        "filter returned {:?}, expected RGB24",
        frame.format()
    );

    let width = frame.width();
    let height = frame.height();
    ensure!(
        width > 0 && height > 0,
        "invalid RGB24 dimensions: {width}x{height}"
    );
    let row_bytes = (width as usize)
        .checked_mul(BYTES_PER_PIXEL)
        .context("RGB frame row is too large")?;
    let image_bytes = row_bytes
        .checked_mul(height as usize)
        .context("RGB frame is too large")?;
    let stride = frame.stride(0);
    ensure!(
        stride >= row_bytes,
        "invalid RGB24 stride: {stride} bytes for a {row_bytes}-byte row"
    );

    let data = frame.data(0);
    let required_bytes = ((height as usize) - 1)
        .checked_mul(stride)
        .and_then(|last_row_start| last_row_start.checked_add(row_bytes))
        .context("RGB frame layout is too large")?;
    ensure!(
        data.len() >= required_bytes,
        "truncated RGB24 frame: need {required_bytes} bytes, got {}",
        data.len()
    );

    let image = if stride == row_bytes {
        data[..image_bytes].to_vec()
    } else {
        let mut image = vec![0; image_bytes];
        image
            .chunks_exact_mut(row_bytes)
            .zip(data.chunks_exact(stride))
            .for_each(|(destination, source)| destination.copy_from_slice(&source[..row_bytes]));
        image
    };

    ImageBuffer::from_raw(width, height, image).context("invalid RGB24 image dimensions")
}

fn drain_filter<F: FnMut(RgbImage) -> Result<()>>(
    filter_graph: &mut filter::Graph,
    frame_callback: &mut F,
) -> Result<()> {
    let mut filtered = Video::empty();
    loop {
        let result = filter_graph
            .get("out")
            .context("missing output filter")?
            .sink()
            .frame(&mut filtered);
        match result {
            Ok(()) => frame_callback(frame_to_image(&filtered)?)?,
            Err(ffmpeg::Error::Other { errno }) if errno == libc::EAGAIN => break,
            Err(ffmpeg::Error::Eof) => break,
            Err(error) => return Err(error).context("receiving filtered frame"),
        }
    }
    Ok(())
}

pub fn run<P: AsRef<std::path::Path>, F: FnMut(RgbImage) -> Result<()>>(
    path: P,
    mut frame_callback: F,
    frame_interval: f32,
) -> Result<()> {
    let mut ictx = format::input(path.as_ref().as_os_str()).context("parsing video")?;
    let video = ictx
        .streams()
        .best(Type::Video)
        .context("no video stream")?;
    let video_index = video.index();

    let context_decoder = codec::Context::from_parameters(video.parameters())
        .context("initializing decoder context")?;
    let mut decoder = context_decoder
        .decoder()
        .video()
        .context("initializing decoder")?;

    let mut graph = filter::Graph::new();
    let afr = video.avg_frame_rate();
    let afr = ((frame_interval * (afr.0 as f32) / (afr.1 as f32)).round() as i64).max(1);

    // I don't actually know what this does, but in some cases it's 0/0 so "correct" for that.
    let aspect_ratio = decoder.aspect_ratio();
    let mut aspect_ratio = (aspect_ratio.0, aspect_ratio.1);
    if aspect_ratio == (0, 0) {
        aspect_ratio = (1, 1);
    }

    graph.add(
        &filter::find("buffer").unwrap(),
        "in",
        &format!(
            "video_size={}x{}:pix_fmt={}:time_base={}/{}:pixel_aspect={}/{}",
            decoder.width(),
            decoder.height(),
            decoder
                .format()
                .descriptor()
                .context("invalid format")?
                .name(),
            video.time_base().0,
            video.time_base().1,
            aspect_ratio.0,
            aspect_ratio.1
        ),
    )?;
    graph.add(&filter::find("buffersink").unwrap(), "out", "")?;
    // I don't know exactly where, but some of my videos apparently have the size vary throughout them.
    // This causes horrible segfaults somewhere.
    // Rescale to initial width to fix this. We could do this with a separate swscaler but this is easier.
    let filterspec = format!("[in] scale={}:{} [scaled]; [scaled] thumbnail=n={}:log=verbose [thumbs]; [thumbs] select='gt(scene,0.05)+eq(n,0)' [selected]; [selected] format=pix_fmts=rgb24 [out]", decoder.width(), decoder.height(), afr);
    graph
        .output("in", 0)?
        .input("out", 0)?
        .parse(&filterspec)
        .context("filtergraph parse failed")?;
    graph.validate().context("filtergraph build failed")?;

    let mut receive_and_process_decoded_frames =
        |decoder: &mut ffmpeg::decoder::Video, filter_graph: &mut filter::Graph| -> Result<()> {
            let mut decoded = Video::empty();
            loop {
                match decoder.receive_frame(&mut decoded) {
                    Ok(()) => {}
                    Err(ffmpeg::Error::Other { errno }) if errno == libc::EAGAIN => break,
                    Err(ffmpeg::Error::Eof) => break,
                    Err(error) => return Err(error).context("receiving decoded frame"),
                }

                let mut in_ctx = filter_graph.get("in").context("missing input filter")?;
                let mut src = in_ctx.source();
                src.add(&decoded).context("add frame")?;
                drain_filter(filter_graph, &mut frame_callback)?;
            }
            Ok(())
        };

    for packet in ictx.packets() {
        let (stream, packet) = packet.context("reading packet")?;
        if stream.index() == video_index {
            decoder.send_packet(&packet).context("decoder")?;
            receive_and_process_decoded_frames(&mut decoder, &mut graph)
                .context("processing frame")?;
        }
    }
    decoder.send_eof()?;
    receive_and_process_decoded_frames(&mut decoder, &mut graph)?;
    graph
        .get("in")
        .context("missing input filter")?
        .source()
        .flush()
        .context("flushing filter graph")?;
    drain_filter(&mut graph, &mut frame_callback)?;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_padded_rgb24_frames() {
        let width = 5;
        let height = 2;
        let mut frame = Video::new(Pixel::RGB24, width, height);
        let stride = frame.stride(0);
        assert!(stride >= width as usize * BYTES_PER_PIXEL);

        let data = frame.data_mut(0);
        for y in 0..height as usize {
            for x in 0..width as usize * BYTES_PER_PIXEL {
                data[y * stride + x] = (y * 100 + x) as u8;
            }
        }

        let image = frame_to_image(&frame).unwrap();
        assert_eq!(image.dimensions(), (width, height));
        assert_eq!(
            image.into_raw(),
            (0..15).chain(100..115).collect::<Vec<_>>()
        );
    }

    #[test]
    fn rejects_planar_frames() {
        let frame = Video::new(Pixel::GBRP, 5, 2);
        let error = frame_to_image(&frame).unwrap_err();
        assert!(error.to_string().contains("expected RGB24"));
    }
}
