// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::gpu::KernelHandle;

use super::exact_projection::exact_projection_route;
use crate::layers::ops::{ExactLmHeadRoute, ExactLmHeadTier, W4a16ExactLmHeadKernels};

fn kernels(handles: [u64; 4]) -> W4a16ExactLmHeadKernels {
    W4a16ExactLmHeadKernels::new(
        KernelHandle(handles[0]),
        KernelHandle(handles[1]),
        KernelHandle(handles[2]),
        KernelHandle(handles[3]),
    )
}

#[test]
fn exact_projection_uses_the_smallest_present_tier() {
    let available = kernels([1, 2, 3, 4]);
    for (rows, tier) in [
        (2, ExactLmHeadTier::M4),
        (4, ExactLmHeadTier::M4),
        (5, ExactLmHeadTier::M8),
        (8, ExactLmHeadTier::M8),
        (9, ExactLmHeadTier::M17),
        (17, ExactLmHeadTier::M17),
        (18, ExactLmHeadTier::M32),
        (32, ExactLmHeadTier::M32),
    ] {
        assert_eq!(
            exact_projection_route(available, rows).unwrap(),
            ExactLmHeadRoute::Exact(tier)
        );
    }
}

#[test]
fn missing_selected_tier_fails_closed_to_serial_k1() {
    for (available, rows, tier) in [
        ([0, 2, 3, 4], 2, ExactLmHeadTier::M4),
        ([1, 0, 3, 4], 5, ExactLmHeadTier::M8),
        ([1, 2, 0, 4], 9, ExactLmHeadTier::M17),
        ([1, 2, 3, 0], 18, ExactLmHeadTier::M32),
    ] {
        assert_eq!(
            exact_projection_route(kernels(available), rows).unwrap(),
            ExactLmHeadRoute::SerialK1(tier)
        );
    }
    assert_eq!(
        exact_projection_route(kernels([1, 0, 3, 4]), 4).unwrap(),
        ExactLmHeadRoute::Exact(ExactLmHeadTier::M4)
    );
}

#[test]
fn projection_route_is_bounded_to_dynamic_verify_rows() {
    let available = kernels([1, 2, 3, 4]);
    for rows in [0, 1, 33] {
        assert!(exact_projection_route(available, rows).is_err());
    }
}

#[test]
fn source_retains_all_tiers_and_a_row_major_k1_fallback() {
    let layer = include_str!("mod.rs");
    let init = include_str!("init.rs");
    let route = include_str!("exact_projection.rs");

    assert!(layer.contains("w4a16_exact_projection_kernels"));
    for tier in ["M4", "M8", "M17", "M32"] {
        assert!(init.contains(&format!("ExactLmHeadTier::{tier}.symbol()")));
    }
    assert!(route.contains("w4a16_gemv_batch_logits_exact"));
    assert!(route.contains("ExactLmHeadRoute::SerialK1"));
    assert!(route.contains("ops::w4a16_decode_gemv("));
    assert!(route.contains("self.w4a16_gemv_sw_k"));
    assert!(route.contains("self.gemv_sw"));
    assert!(route.contains("row as usize * k as usize * BF16_BYTES"));
    assert!(route.contains("row as usize * n as usize * BF16_BYTES"));
}
