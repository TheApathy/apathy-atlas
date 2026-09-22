// SPDX-License-Identifier: AGPL-3.0-only

//! RGB bicubic resampling: half-pixel centers, antialias widening, normalized
//! signed 22-bit weights and byte rounding after each separable pass. These
//! arithmetic boundaries match Pillow 10.4's RGB BICUBIC reference.

use anyhow::{Result, ensure};
use image::{Rgb, RgbImage};

const FRACTION: u32 = 22;

fn cubic(distance: f64) -> f64 {
    let x = distance.abs();
    if x < 1.0 {
        ((1.5 * x - 2.5) * x) * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}

fn weights(input: u32, output: u32) -> Vec<(u32, Vec<i32>)> {
    let scale = f64::from(input) / f64::from(output);
    let widen = scale.max(1.0);
    let support = 2.0 * widen;
    (0..output)
        .map(|position| {
            let center = (f64::from(position) + 0.5) * scale;
            let begin = ((center - support + 0.5) as i64).max(0) as u32;
            let end = ((center + support + 0.5) as u32).min(input);
            let kernel: Vec<f64> = (begin..end)
                .map(|x| cubic((f64::from(x) - center + 0.5) * (1.0 / widen)))
                .collect();
            let sum: f64 = kernel.iter().sum();
            let fixed = kernel
                .into_iter()
                .map(|weight| (weight / sum * f64::from(1 << FRACTION)).round() as i32)
                .collect();
            (begin, fixed)
        })
        .collect()
}

fn byte(sum: i64) -> u8 {
    (sum >> FRACTION).clamp(0, 255) as u8
}

pub(crate) fn resize(source: &RgbImage, width: u32, height: u32) -> Result<RgbImage> {
    super::validate_source_size(source.width(), source.height())?;
    super::validate_source_size(width, height)?;
    super::validate_source_size(width, source.height())?;
    let horizontal = if width == source.width() {
        source.clone()
    } else {
        let columns = weights(source.width(), width);
        RgbImage::from_fn(width, source.height(), |x, y| {
            let (begin, ref taps) = columns[x as usize];
            Rgb(std::array::from_fn(|channel| {
                let sum =
                    taps.iter()
                        .enumerate()
                        .fold(1i64 << (FRACTION - 1), |sum, (offset, &tap)| {
                            sum + i64::from(source.get_pixel(begin + offset as u32, y)[channel])
                                * i64::from(tap)
                        });
                byte(sum)
            }))
        })
    };
    if height == source.height() {
        return Ok(horizontal);
    }
    let rows = weights(source.height(), height);
    Ok(RgbImage::from_fn(width, height, |x, y| {
        let (begin, ref taps) = rows[y as usize];
        Rgb(std::array::from_fn(|channel| {
            let sum =
                taps.iter()
                    .enumerate()
                    .fold(1i64 << (FRACTION - 1), |sum, (offset, &tap)| {
                        sum + i64::from(horizontal.get_pixel(x, begin + offset as u32)[channel])
                            * i64::from(tap)
                    });
            byte(sum)
        }))
    }))
}

pub(crate) fn pad(source: &RgbImage, width: u32, height: u32) -> Result<RgbImage> {
    super::validate_source_size(width, height)?;
    super::validate_source_size(source.width(), source.height())?;
    let ratio = f64::from(source.width()) / f64::from(source.height());
    let destination_ratio = f64::from(width) / f64::from(height);
    let (inside_w, inside_h) = if ratio > destination_ratio {
        (
            width,
            (f64::from(source.height()) / f64::from(source.width()) * f64::from(width))
                .round_ties_even() as u32,
        )
    } else if ratio < destination_ratio {
        (
            (f64::from(source.width()) / f64::from(source.height()) * f64::from(height))
                .round_ties_even() as u32,
            height,
        )
    } else {
        (width, height)
    };
    ensure!(
        inside_w > 0 && inside_h > 0,
        "DeepSeek padding would produce an empty contained image"
    );
    let resized = resize(source, inside_w, inside_h)?;
    let offset_x = (f64::from(width - inside_w) / 2.0).round_ties_even() as u32;
    let offset_y = (f64::from(height - inside_h) / 2.0).round_ties_even() as u32;
    let mut output = RgbImage::from_pixel(width, height, Rgb([127, 127, 127]));
    image::imageops::replace(
        &mut output,
        &resized,
        i64::from(offset_x),
        i64::from(offset_y),
    );
    Ok(output)
}
