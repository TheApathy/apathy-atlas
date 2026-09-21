// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn n_layout_matches_reference_odd_and_even_rows() {
    use ImageTokenType::{End as E, Image as I, NewLine as N, Pad as P, Start as S};
    let even = build_image_block(2, 2, 0, 100).unwrap();
    assert_eq!(even.types, vec![P, P, P, S, I, I, I, I, N, N, P, P, E]);
    assert_eq!(even.aligner_permutation, vec![0, 2, 1, 3]);
    assert_eq!(
        even.token_ids,
        vec![
            101, 101, 101, 100, 102, 102, 102, 102, 103, 103, 101, 101, 104
        ]
    );
    let odd = build_image_block(3, 2, 3, 100).unwrap();
    assert_eq!(odd.types, vec![S, I, I, I, I, N, N, I, P, I, P, N, P, E]);
    assert_eq!(odd.aligner_permutation, vec![0, 2, 1, 3, 4, 5]);
}

#[test]
fn n_layout_preserves_every_aligner_row_and_absolute_alignment() {
    for h in 1..10 {
        for w in 1..10 {
            for start in 0..4 {
                let plan = build_image_block(h, w, start, 100).unwrap();
                let marker = plan
                    .types
                    .iter()
                    .position(|t| *t == ImageTokenType::Start)
                    .unwrap();
                assert_eq!((start + marker) % 4, 3);
                assert_eq!((start + plan.types.len() - 1) % 4, 0);
                assert_eq!(
                    plan.types
                        .iter()
                        .filter(|t| **t == ImageTokenType::Image)
                        .count(),
                    h * w
                );
                assert_eq!(
                    plan.types
                        .iter()
                        .filter(|t| **t == ImageTokenType::NewLine)
                        .count(),
                    h
                );
                let mut rows = plan.aligner_permutation;
                rows.sort_unstable();
                assert_eq!(rows, (0..h * w).collect::<Vec<_>>());
            }
        }
    }
}

#[test]
fn layout_rejects_empty_overflow_and_unbounded_inputs() {
    assert!(build_image_block(0, 3, 0, 100).is_err());
    assert!(build_image_block(3, 0, 0, 100).is_err());
    assert!(build_image_block(usize::MAX, 3, 0, 100).is_err());
    assert!(build_image_block(3, usize::MAX, 0, 100).is_err());
    assert!(build_image_block(2, 2, usize::MAX, 100).is_err());
    assert!(build_image_block(2, 2, 0, u32::MAX - 3).is_err());
    assert!(build_image_block(10_000, 10_000, 0, 100).is_err());
}

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
        bicubic::resize(&pattern(), 3, 2).unwrap().into_raw(),
        vec![
            126, 78, 86, 87, 161, 94, 102, 136, 100, 119, 76, 134, 163, 159, 142, 139, 134, 148
        ]
    );
    let up = bicubic::resize(&pattern(), 9, 8).unwrap().into_raw();
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
    assert_eq!(bicubic::resize(&pattern(), 7, 5).unwrap(), pattern());
}

#[test]
fn centered_gray_padding_matches_pillow_golden() {
    let padded = bicubic::pad(&pattern(), 6, 6).unwrap();
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

pub(super) fn config() -> DeepSeekVisionConfig {
    DeepSeekVisionConfig {
        hidden_size: 1024,
        intermediate_size: 2816,
        num_hidden_layers: 32,
        num_attention_heads: 16,
        patch_size: 14,
        downsample_ratio: 3,
        max_tokens: 384,
        max_wh_ratio: Some(8.0),
        min_pixels: 147456,
        rope_theta: 10000.0,
    }
}

#[test]
fn resize_dimensions_match_pinned_official_python() {
    // Official image_processor.py @6821d6ad, actual Vision-Exp config.
    for (ih, iw, h, w, gh, gw, stretch) in [
        (1, 1, 392, 392, 10, 10, false),
        (1024, 768, 840, 630, 20, 15, false),
        (768, 1024, 658, 882, 16, 21, false),
        (100, 5000, 140, 1092, 4, 26, true),
        (5000, 100, 4200, 84, 100, 2, false),
        (301, 499, 308, 504, 8, 12, false),
        (40, 320, 140, 1092, 4, 26, true),
        (40, 321, 140, 1092, 4, 26, true),
        (10000, 1, 7896, 42, 188, 1, false),
        (1, 10000, 140, 1092, 4, 26, true),
    ] {
        let plan = resize_plan(ih, iw, &config()).unwrap();
        assert_eq!(
            (
                plan.height,
                plan.width,
                plan.grid_llm_h,
                plan.grid_llm_w,
                plan.stretch
            ),
            (h, w, gh, gw, stretch),
            "input {ih}x{iw}"
        );
        for start in 0..4 {
            assert!(
                build_image_block(gh, gw, start, 129280)
                    .unwrap()
                    .types
                    .len()
                    <= 384
            );
        }
    }
}

#[test]
fn patch_order_and_bf16_match_channel_major_reference() {
    let mut cfg = config();
    cfg.min_pixels = 1;
    let source = image::RgbImage::from_fn(28, 14, |x, y| image::Rgb([x as u8, y as u8, 127]));
    let prepared = preprocess_rgb(&source, &cfg).unwrap();
    assert_eq!(
        (
            prepared.grid_h,
            prepared.grid_w,
            prepared.grid_llm_h,
            prepared.grid_llm_w
        ),
        (1, 2, 1, 1)
    );
    assert_eq!(prepared.patches.len(), 2 * 3 * 14 * 14);
    for patch in 0..2 {
        for channel in 0..3 {
            for y in 0..14 {
                for x in 0..14 {
                    let pixel = [patch * 14 + x, y, 127][channel];
                    let expected = round_bf16((pixel as f32 / 255.0 - 0.5) / 0.5);
                    assert_eq!(
                        prepared.patches[patch * 588 + channel * 196 + y * 14 + x],
                        expected
                    );
                }
            }
        }
    }
    assert_eq!(prepared.patches[2 * 196].to_bits(), 0xbb81_0000);
}

#[test]
fn decoding_rejects_urls_wrong_mime_and_bad_or_large_shapes() {
    for text in [
        "https://example.com/a.png",
        "/etc/passwd",
        "data:image/png,AAAA",
        "data:image/gif;base64,AAAA",
        "data:image/png;base64,!!!!",
        "",
        "AAAA",
    ] {
        assert!(decode_image(text).is_err(), "accepted {text}");
    }
    assert!(resize_plan(0, 1, &config()).is_err());
    assert!(resize_plan(u32::MAX, 1, &config()).is_err());
    assert!(resize_plan(10_000, 10_000, &config()).is_err());
    let mut invalid = config();
    invalid.patch_size = 0;
    assert!(resize_plan(40, 40, &invalid).is_err());
    let mut image_bytes = Cursor::new(Vec::new());
    pattern()
        .write_to(&mut image_bytes, ImageFormat::Png)
        .unwrap();
    let payload = base64::engine::general_purpose::STANDARD.encode(image_bytes.into_inner());
    assert_eq!(
        decode_image(&format!("data:image/png;base64,{payload}")).unwrap(),
        pattern()
    );
    assert!(decode_image(&format!("data:image/jpeg;base64,{payload}")).is_err());
    let mut small = config();
    small.min_pixels = 1;
    let prepared = preprocess_image(&format!("data:image/png;base64,{payload}"), &small).unwrap();
    assert_eq!((prepared.grid_h, prepared.grid_w), (1, 1));
    assert!(preprocess_images(&vec![String::new(); 9], &small).is_err());
    let sixteen_bit = image::DynamicImage::ImageLuma16(image::ImageBuffer::from_pixel(
        1,
        1,
        image::Luma([256u16]),
    ));
    let mut high_depth = Cursor::new(Vec::new());
    sixteen_bit
        .write_to(&mut high_depth, ImageFormat::Png)
        .unwrap();
    let encoded = base64::engine::general_purpose::STANDARD.encode(high_depth.into_inner());
    assert!(decode_image(&encoded).is_err());
}
