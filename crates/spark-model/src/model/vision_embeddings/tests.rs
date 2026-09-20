// SPDX-License-Identifier: AGPL-3.0-only

use super::test_gpu::MemoryGpu;
use super::*;
use layout::{Geometry, ImageLayout};

pub(super) fn geometry() -> Geometry {
    Geometry {
        merge: 2,
        max_patches: 16,
        hidden: 2,
        deepstack: 1,
        pixel_width: 1536,
    }
}

pub(super) fn image(value: f32, gh: usize, gw: usize) -> (Vec<f32>, usize, usize) {
    (vec![value; gh * gw * 1536], gh, gw)
}

pub(super) fn encode(
    gpu: &MemoryGpu,
    scratch: DevicePtr,
    pixels: &[f32],
    gh: usize,
    gw: usize,
) -> Result<(DevicePtr, usize)> {
    let rows = gh * gw / 4;
    let mut bytes = vec![0xcc; rows * 4 * 2];
    bytes[..rows * 4].fill(pixels[0] as u8);
    gpu.copy_h2d(&bytes, scratch)?;
    Ok((scratch, rows * 2))
}

#[test]
fn preserves_each_final_output_excludes_deepstack_and_survives_scratch_reuse() {
    let gpu = MemoryGpu::new();
    let scratch = gpu.alloc(128).unwrap();
    let mut state = VisionEmbeddingState::new(128, 1);
    let images = vec![image(3.0, 2, 2), image(9.0, 2, 4)];
    state
        .prepare(&gpu, &images, geometry(), true, 3, 7, |p, h, w| {
            encode(&gpu, scratch, p, h, w)
        })
        .unwrap();
    gpu.copy_h2d(&[0xee; 128], scratch).unwrap();
    let dst = gpu.alloc(24).unwrap();
    gpu.copy_h2d(&[0xab; 24], dst).unwrap();
    state
        .splice(&gpu, &[1, 99, 2, 99, 99, 3], 99, 0, 6, dst, 2, 2, 7)
        .unwrap();
    assert_eq!(
        gpu.bytes(dst),
        [
            vec![0xab; 4],
            vec![3; 4],
            vec![0xab; 4],
            vec![9; 8],
            vec![0xab; 4]
        ]
        .concat()
    );
    state.release(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 2);
}

#[test]
fn cache_hit_uses_owned_bytes_but_unsampled_pixel_and_order_changes_miss() {
    let gpu = MemoryGpu::new();
    let scratch = gpu.alloc(128).unwrap();
    let mut state = VisionEmbeddingState::new(128, 1);
    let mut images = vec![image(3.0, 2, 2), image(9.0, 2, 4)];
    assert!(
        !state
            .prepare(&gpu, &images, geometry(), true, 3, 7, |p, h, w| encode(
                &gpu, scratch, p, h, w
            ))
            .unwrap()
    );
    assert!(
        state
            .prepare(&gpu, &images, geometry(), true, 3, 7, |_, _, _| panic!(
                "cache hit must not encode"
            ))
            .unwrap()
    );
    images[0].0[1] = 4.0;
    assert!(
        !state
            .prepare(&gpu, &images, geometry(), true, 3, 7, |p, h, w| encode(
                &gpu, scratch, p, h, w
            ))
            .unwrap()
    );
    images.reverse();
    assert!(
        !state
            .prepare(&gpu, &images, geometry(), true, 3, 7, |p, h, w| encode(
                &gpu, scratch, p, h, w
            ))
            .unwrap()
    );
    assert!(
        !state
            .prepare(&gpu, &images, geometry(), true, 2, 7, |p, h, w| encode(
                &gpu, scratch, p, h, w
            ))
            .unwrap()
    );
    assert!(
        !state
            .prepare(&gpu, &images, geometry(), false, 2, 7, |p, h, w| encode(
                &gpu, scratch, p, h, w
            ))
            .unwrap()
    );
}

#[test]
fn chunk_inside_second_image_uses_global_rows_and_correct_positions() {
    let layout =
        ImageLayout::validate(&[image(3.0, 2, 2), image(9.0, 2, 4)], geometry(), 128).unwrap();
    let tokens = [1, 99, 2, 99, 99, 3];
    let full = layout.plan(&tokens, 99, 0, 6).unwrap();
    let tail = layout.plan(&tokens, 99, 4, 2).unwrap();
    assert_eq!(
        tail.copies,
        vec![layout::CopyRun {
            source_row: 2,
            dest_row: 0,
            rows: 1
        }]
    );
    assert_eq!(
        full.positions,
        [
            vec![0, 1, 2, 3, 3, 5],
            vec![0, 1, 2, 3, 3, 5],
            vec![0, 1, 2, 3, 4, 5]
        ]
    );
    for axis in 0..3 {
        assert_eq!(tail.positions[axis], full.positions[axis][4..]);
    }
    assert_eq!(
        layout.plan(&[99, 99, 99], 99, 0, 3).unwrap().copies.len(),
        2
    );
}
