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
    validate_source_size(source.width(), source.height())?;
    validate_source_size(width, height)?;
    validate_source_size(width, source.height())?;
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
    validate_source_size(width, height)?;
    validate_source_size(source.width(), source.height())?;
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

// Size limits of the V4 preprocessor this resampler came from
// (deepseek_vision_preprocess.rs on dsv41/integration).
const MAX_IMAGE_DIMENSION: u32 = 32_768;
const MAX_SOURCE_PIXELS: u64 = 40_000_000;

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

pub(crate) fn round_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(rounded)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern() -> image::RgbImage {
        image::RgbImage::from_fn(7, 5, |x, y| {
            let i = y * 7 + x;
            image::Rgb([(i * 29) as u8, (i * 73 + 17) as u8, (i * 131 + 3) as u8])
        })
    }

    #[test]
    fn bicubic_pixels_match_pillow_10_4_golden() {
        // Independent Pillow RGB resize golden; includes antialias downsampling.
        assert_eq!(
            resize(&pattern(), 3, 2).unwrap().into_raw(),
            vec![
                126, 78, 86, 87, 161, 94, 102, 136, 100, 119, 76, 134, 163, 159, 142, 139, 134, 148
            ]
        );
        let up = resize(&pattern(), 9, 8).unwrap().into_raw();
        assert_eq!(
            &up[..27],
            &[
                0, 14, 0, 2, 61, 104, 35, 122, 82, 69, 184, 18, 91, 236, 148, 114, 81, 22, 136, 80, 91,
                160, 155, 118, 179, 202, 3
            ]
        );
        assert_eq!(
            &up[189..],
            &[
                39, 10, 72, 58, 57, 191, 82, 118, 169, 104, 180, 105, 127, 232, 235, 150, 77, 109, 171,
                76, 178, 202, 151, 205, 234, 198, 90
            ]
        );
        assert_eq!(resize(&pattern(), 7, 5).unwrap(), pattern());
    }

    #[test]
    fn centered_gray_padding_matches_pillow_golden() {
        let padded = pad(&pattern(), 6, 6).unwrap();
        assert_eq!(
            padded.into_raw(),
            vec![
                127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
                127, 46, 24, 48, 61, 107, 88, 45, 209, 72, 91, 130, 76, 131, 95, 99, 164, 192, 66, 201,
                23, 103, 186, 106, 94, 94, 208, 104, 65, 129, 108, 45, 94, 105, 97, 191, 121, 119, 21,
                130, 156, 104, 121, 209, 206, 131, 169, 127, 135, 154, 92, 132, 44, 189, 148, 54, 20,
                126, 87, 103, 166, 120, 205, 150, 163, 126, 154, 205, 91, 177, 182, 188, 144, 127, 127,
                127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127
            ]
        );
    }

    #[test]
    fn bf16_boundary_rounds_ties_to_even() {
        assert_eq!(
            round_bf16(f32::from_bits(0x3f80_8000)).to_bits(),
            0x3f80_0000
        );
        assert_eq!(
            round_bf16(f32::from_bits(0x3f81_8000)).to_bits(),
            0x3f82_0000
        );
        assert_eq!(round_bf16(-1.0), -1.0);
        assert_eq!(
            round_bf16((127.0 / 255.0 - 0.5) / 0.5).to_bits(),
            0xbb81_0000
        );
    }
}
