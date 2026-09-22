// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4 image preprocessing and sentinel layout.

#[path = "deepseek_vision_preprocess/bicubic.rs"]
pub(crate) mod bicubic;
#[path = "deepseek_vision_preprocess/geometry.rs"]
mod geometry;
#[path = "deepseek_vision_preprocess/prompt.rs"]
mod prompt;

use std::io::Cursor;

use anyhow::{Context, Result, ensure};
use atlas_core::config::DeepSeekVisionConfig;
use base64::Engine;
use image::{ImageFormat, ImageReader, RgbImage};

#[cfg(test)]
use atlas_core::config::{ImageTokenType, build_deepseek_image_block as build_image_block};
use geometry::resize_plan;
pub(crate) use prompt::{IMAGE_PLACEHOLDER, expand_image_placeholders, reject_unprepared_tokens};

const MAX_ENCODED_BYTES: usize = 32 * 1024 * 1024;
const MAX_DECODED_BYTES: usize = 24 * 1024 * 1024;
const MAX_IMAGE_DIMENSION: u32 = 32_768;
const MAX_SOURCE_PIXELS: u64 = 40_000_000;

#[derive(Debug)]
pub(crate) struct PreparedImage {
    /// Row-major [grid_h*grid_w, 3, patch_size, patch_size], BF16 values in f32.
    pub patches: Vec<f32>,
    pub grid_h: usize,
    pub grid_w: usize,
    pub grid_llm_h: usize,
    pub grid_llm_w: usize,
}

/// Decode only bounded PNG/JPEG base64 payloads. URLs and paths are never read.
fn decode_image(data_uri: &str) -> Result<RgbImage> {
    ensure!(
        data_uri.len() <= MAX_ENCODED_BYTES,
        "DeepSeek image base64 exceeds size limit"
    );
    let (payload, declared) = if data_uri.starts_with("data:") {
        let (header, payload) = data_uri.split_once(',').context("Invalid image data URI")?;
        let format = match header {
            "data:image/png;base64" => ImageFormat::Png,
            "data:image/jpeg;base64" => ImageFormat::Jpeg,
            _ => anyhow::bail!("DeepSeek images require base64 PNG or JPEG data URIs"),
        };
        (payload, Some(format))
    } else {
        (data_uri, None)
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload)
        .context("Invalid image base64")?;
    ensure!(
        !bytes.is_empty() && bytes.len() <= MAX_DECODED_BYTES,
        "Invalid image byte length"
    );
    let format = image::guess_format(&bytes).context("Unknown image format")?;
    ensure!(
        matches!(format, ImageFormat::Png | ImageFormat::Jpeg),
        "Unsupported image format"
    );
    ensure!(
        declared.is_none_or(|expected| expected == format),
        "Image MIME and payload format disagree"
    );
    let dimensions = ImageReader::with_format(Cursor::new(&bytes), format)
        .into_dimensions()
        .context("Invalid image dimensions")?;
    validate_source_size(dimensions.0, dimensions.1)?;
    let mut reader = ImageReader::with_format(Cursor::new(&bytes), format);
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_DIMENSION);
    limits.max_image_height = Some(MAX_IMAGE_DIMENSION);
    limits.max_alloc = Some(MAX_SOURCE_PIXELS * 8);
    reader.limits(limits);
    let decoded = reader.decode().context("Image decode failed")?;
    ensure!(
        matches!(
            decoded.color(),
            image::ColorType::L8
                | image::ColorType::La8
                | image::ColorType::Rgb8
                | image::ColorType::Rgba8
        ),
        "DeepSeek images currently require 8-bit channels; high-bit-depth conversion is not qualified"
    );
    let rgb = decoded.to_rgb8();
    ensure!(
        rgb.dimensions() == dimensions,
        "Image dimensions changed during decoding"
    );
    Ok(rgb)
}

fn validate_source_size(width: u32, height: u32) -> Result<()> {
    ensure!(
        width > 0 && height > 0 && width <= MAX_IMAGE_DIMENSION && height <= MAX_IMAGE_DIMENSION,
        "DeepSeek image dimensions outside supported range"
    );
    ensure!(
        u64::from(width) * u64::from(height) <= MAX_SOURCE_PIXELS,
        "DeepSeek image exceeds source pixel limit"
    );
    Ok(())
}

pub(crate) fn preprocess_image(
    data_uri: &str,
    config: &DeepSeekVisionConfig,
) -> Result<PreparedImage> {
    config.validate()?;
    preprocess_rgb(&decode_image(data_uri)?, config)
}

pub(crate) fn preprocess_images(
    uris: &[String],
    config: &DeepSeekVisionConfig,
) -> Result<Vec<(Vec<f32>, usize, usize)>> {
    ensure!(
        uris.len() <= 8,
        "DeepSeek Vision accepts at most 8 images per request"
    );
    uris.iter()
        .map(|uri| {
            let prepared = preprocess_image(uri, config)?;
            debug_assert_eq!(
                prepared.grid_llm_h,
                prepared.grid_h.div_ceil(config.downsample_ratio)
            );
            debug_assert_eq!(
                prepared.grid_llm_w,
                prepared.grid_w.div_ceil(config.downsample_ratio)
            );
            Ok((prepared.patches, prepared.grid_h, prepared.grid_w))
        })
        .collect()
}

pub(crate) fn preprocess_rgb(
    image: &RgbImage,
    config: &DeepSeekVisionConfig,
) -> Result<PreparedImage> {
    validate_source_size(image.width(), image.height())?;
    let plan = resize_plan(image.height(), image.width(), config)?;
    let resized = if plan.stretch {
        bicubic::resize(image, plan.width, plan.height)?
    } else {
        bicubic::pad(image, plan.width, plan.height)?
    };
    let p = config.patch_size;
    let elements = plan
        .grid_h
        .checked_mul(plan.grid_w)
        .and_then(|n| n.checked_mul(3 * p * p))
        .context("DeepSeek patch extent overflow")?;
    let mut patches = Vec::with_capacity(elements);
    for ph in 0..plan.grid_h {
        for pw in 0..plan.grid_w {
            for channel in 0..3 {
                for py in 0..p {
                    for px in 0..p {
                        let pixel =
                            resized.get_pixel((pw * p + px) as u32, (ph * p + py) as u32)[channel];
                        // Keep the two F32 operations, then the official BF16 boundary.
                        let normalized = (f32::from(pixel) / 255.0 - 0.5) / 0.5;
                        patches.push(round_bf16(normalized));
                    }
                }
            }
        }
    }
    Ok(PreparedImage {
        patches,
        grid_h: plan.grid_h,
        grid_w: plan.grid_w,
        grid_llm_h: plan.grid_llm_h,
        grid_llm_w: plan.grid_llm_w,
    })
}

pub(crate) fn round_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(rounded)
}

#[cfg(test)]
#[path = "deepseek_vision_preprocess/tests.rs"]
mod tests;
