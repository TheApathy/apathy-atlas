// SPDX-License-Identifier: AGPL-3.0-only

use super::layout::{Geometry, ImageLayout};
use super::test_gpu::MemoryGpu;
use super::tests::{encode, geometry, image};
use super::*;

#[test]
fn invalid_geometry_or_partial_encoding_never_publishes_and_empty_clears() {
    let gpu = MemoryGpu::new();
    let scratch = gpu.alloc(128).unwrap();
    let mut state = VisionEmbeddingState::new(128, 1);
    let good = [image(3.0, 2, 2)];
    state
        .prepare(&gpu, &good, geometry(), true, 0, 7, |p, h, w| {
            encode(&gpu, scratch, p, h, w)
        })
        .unwrap();
    let mut bad = image(2.0, 2, 2);
    bad.0.pop();
    let before = gpu.copy_count();
    assert!(
        state
            .prepare(&gpu, &[bad], geometry(), true, 0, 7, |_, _, _| panic!(
                "invalid input encoded"
            ))
            .is_err()
    );
    assert!(!state.is_ready());
    assert_eq!(gpu.copy_count(), before);
    assert!(
        state
            .prepare(
                &gpu,
                &[image(3.0, 2, 2), image(9.0, 2, 4)],
                geometry(),
                true,
                0,
                7,
                |p, h, w| {
                    if p[0] == 9.0 {
                        anyhow::bail!("injected encoder failure");
                    }
                    encode(&gpu, scratch, p, h, w)
                }
            )
            .is_err()
    );
    assert!(!state.is_ready());
    assert!(state.plan(&[99], 99, 0, 1).is_err());
    state
        .prepare(&gpu, &good, geometry(), true, 0, 7, |p, h, w| {
            encode(&gpu, scratch, p, h, w)
        })
        .unwrap();
    state
        .prepare(&gpu, &[], geometry(), true, 0, 7, |_, _, _| unreachable!())
        .unwrap();
    assert!(!state.is_ready());
    assert!(state.plan(&[99], 99, 0, 1).is_err());
    assert_eq!(
        state.plan(&[1, 2], 99, 1, 1).unwrap().positions,
        [vec![1], vec![1], vec![1]]
    );
}

#[test]
fn invalid_pad_counts_shapes_dtypes_and_ranges_fail_before_copy() {
    let gpu = MemoryGpu::new();
    let scratch = gpu.alloc(128).unwrap();
    let dst = gpu.alloc(128).unwrap();
    let mut state = VisionEmbeddingState::new(128, 1);
    state
        .prepare(
            &gpu,
            &[image(3.0, 2, 4)],
            geometry(),
            false,
            0,
            7,
            |p, h, w| encode(&gpu, scratch, p, h, w),
        )
        .unwrap();
    let before = gpu.copy_count();
    for tokens in [&[99][..], &[99, 99, 99], &[99, 1, 99], &[1, 2]] {
        assert!(
            state
                .splice(&gpu, tokens, 99, 0, tokens.len(), dst, 2, 2, 7)
                .is_err()
        );
    }
    assert!(
        state
            .splice(&gpu, &[99, 99], 99, 0, 2, dst, 3, 2, 7)
            .is_err()
    );
    assert!(
        state
            .splice(&gpu, &[99, 99], 99, 0, 2, dst, 2, 4, 7)
            .is_err()
    );
    assert!(
        state
            .splice(&gpu, &[99, 99], 99, usize::MAX, 2, dst, 2, 2, 7)
            .is_err()
    );
    assert_eq!(gpu.copy_count(), before);
}

#[test]
fn bounds_and_nonfinite_inputs_rejected_without_gpu_work() {
    let gpu = MemoryGpu::new();
    let mut state = VisionEmbeddingState::new(128, 2);
    assert!(
        state
            .prepare(
                &gpu,
                &[image(1.0, 2, 2)],
                geometry(),
                true,
                0,
                7,
                |_, _, _| unreachable!()
            )
            .is_err()
    );
    for (mut value, geo, limit) in [
        (image(1.0, 2, 2), geometry(), 0),
        (image(1.0, 3, 2), geometry(), 128),
        (
            image(1.0, 2, 2),
            Geometry {
                merge: 0,
                ..geometry()
            },
            128,
        ),
        (
            image(1.0, 2, 2),
            Geometry {
                hidden: usize::MAX,
                ..geometry()
            },
            128,
        ),
        (
            image(1.0, 2, 2),
            Geometry {
                max_patches: 3,
                ..geometry()
            },
            128,
        ),
        (
            image(1.0, 2, 2),
            Geometry {
                pixel_width: 12,
                ..geometry()
            },
            128,
        ),
    ] {
        assert!(ImageLayout::validate(&[value.clone()], geo, limit).is_err());
        value.0[0] = f32::NAN;
        assert!(ImageLayout::validate(&[value], geometry(), 128).is_err());
    }
    assert!(
        ImageLayout::validate(
            &[image(1.0, 2, 2)],
            Geometry {
                hidden: MAX_AGGREGATE_BYTES,
                ..geometry()
            },
            128
        )
        .is_err()
    );
    assert_eq!(gpu.alloc_count(), 0);
}
