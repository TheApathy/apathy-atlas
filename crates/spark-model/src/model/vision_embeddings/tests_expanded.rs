// SPDX-License-Identifier: AGPL-3.0-only

use super::test_gpu::MemoryGpu;
use super::tests::{encode, geometry, image};
use super::*;

#[test]
fn expanded_rows_preserve_all_hyper_streams_text_padding_and_chunk_offsets() {
    let gpu = MemoryGpu::new();
    let scratch = gpu.alloc(128).unwrap();
    let dst = gpu.alloc(6 * 16).unwrap();
    let mut state = VisionEmbeddingState::new(128, 1);
    state
        .prepare(
            &gpu,
            &[image(3.0, 2, 2), image(9.0, 2, 4)],
            geometry(),
            false,
            0,
            7,
            |p, h, w| encode(&gpu, scratch, p, h, w),
        )
        .unwrap();
    let tokens = [1, 99, 2, 99, 99, 3];
    let expand = |source, target: DevicePtr| -> Result<()> {
        for channel in 0..4 {
            gpu.copy_d2d(source, target.offset(channel * 4), 4)?;
        }
        Ok(())
    };
    state
        .splice_expanded(&tokens, 99, 0, 4, dst, 2, 4, 7, expand)
        .unwrap();
    state
        .splice_expanded(&tokens, 99, 4, 2, dst.offset(64), 2, 4, 7, expand)
        .unwrap();
    assert_eq!(
        gpu.bytes(dst),
        [
            vec![0xdd; 16],
            vec![3; 16],
            vec![0xdd; 16],
            vec![9; 32],
            vec![0xdd; 16]
        ]
        .concat()
    );
    state.release(&gpu).unwrap();
}

#[test]
fn expanded_invalid_stride_alias_and_partial_kernel_error_fail_closed() {
    let gpu = MemoryGpu::new();
    let scratch = gpu.alloc(128).unwrap();
    let dst = gpu.alloc(32).unwrap();
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
    for (hidden, streams) in [(3, 4), (2, 0), (2, usize::MAX)] {
        assert!(
            state
                .splice_expanded(&[99, 99], 99, 0, 2, dst, hidden, streams, 7, |_, _| panic!(
                    "invalid shape reached expansion"
                ))
                .is_err()
        );
    }
    assert!(
        state
            .splice_expanded(
                &[99, 99],
                99,
                0,
                2,
                state.buffer.offset(2),
                2,
                4,
                7,
                |_, _| panic!("alias reached expansion")
            )
            .is_err()
    );
    let mut calls = 0;
    assert!(
        state
            .splice_expanded(&[99, 99], 99, 0, 2, dst, 2, 4, 7, |source, target| {
                calls += 1;
                if calls == 2 {
                    anyhow::bail!("injected expansion failure");
                }
                for channel in 0..4 {
                    gpu.copy_d2d(source, target.offset(channel * 4), 4)?;
                }
                Ok(())
            })
            .is_err()
    );
    assert_eq!(calls, 2);
    assert_eq!(gpu.bytes(dst), [vec![3; 16], vec![0xdd; 16]].concat());
    assert!(!state.is_ready());
    state.release(&gpu).unwrap();
}
