// SPDX-License-Identifier: AGPL-3.0-only

//! Tests split out of `moe/mod.rs` for the ≤500 LoC file-size cap.

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

#[test]
fn test_moe_kernel_loading() {
    let gpu = MockGpuBackend::new();
    assert!(gpu.kernel("gemv", "dense_gemv_bf16").is_ok());
    assert!(gpu.kernel("w4a16_gemv", "w4a16_gemv").is_ok());
    assert!(gpu.kernel("moe_topk", "moe_topk_softmax").is_ok());
    assert!(
        gpu.kernel("moe_expert_gemv_fused", "moe_expert_gemv_gate_up")
            .is_ok()
    );
    assert!(
        gpu.kernel("moe_expert_gemv_fused", "moe_expert_gemv_gate_up_2x")
            .is_ok()
    );
    assert!(
        gpu.kernel("moe_expert_gemv_fused", "moe_expert_gemv_silu_down")
            .is_ok()
    );
    assert!(
        gpu.kernel("moe_expert_gemv_fused", "moe_expert_gemv_silu_down_2x")
            .is_ok()
    );
    assert!(
        gpu.kernel("moe_shared_expert_fused", "moe_expert_gate_up_shared")
            .is_ok()
    );
    assert!(
        gpu.kernel("moe_shared_expert_fused", "moe_expert_silu_down_shared")
            .is_ok()
    );
    assert!(
        gpu.kernel("moe_expert_gemv", "moe_weighted_sum_blend")
            .is_ok()
    );
    // K=2 batch dispatch
    assert!(gpu.kernel("moe_topk", "moe_topk_softmax_batched").is_ok());
}

#[test]
fn adaptive_topk_prune_kernel_is_loadable() {
    let gpu = MockGpuBackend::new();
    assert!(
        gpu.kernel("moe_adaptive_topk", "moe_adaptive_topk_prune")
            .is_ok()
    );
}

/// Host reference for `moe_adaptive_topk_prune` — the specification the CUDA
/// kernel implements, line for line. Returns `(indices, weights)` after prune.
fn prune_ref(
    idx: &[u32],
    w: &[f32],
    skip_index: u32,
    threshold: f32,
    renormalize: bool,
) -> (Vec<u32>, Vec<f32>) {
    let mut idx = idx.to_vec();
    let mut w = w.to_vec();
    let sum_all: f32 = w.iter().sum();
    if threshold <= 0.0 || sum_all <= 1e-20 {
        return (idx, w);
    }
    let argmax = w
        .iter()
        .enumerate()
        .fold(
            (0usize, f32::MIN),
            |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
        )
        .0;
    let mut sum_kept = 0.0f32;
    let mut kept = 0usize;
    for t in 0..w.len() {
        if t == argmax || w[t] / sum_all >= threshold {
            sum_kept += w[t];
            kept += 1;
        } else {
            idx[t] = skip_index;
            w[t] = 0.0;
        }
    }
    if kept < idx.len() && renormalize && sum_kept > 1e-20 {
        let rescale = sum_all / sum_kept;
        for v in w.iter_mut() {
            if *v > 0.0 {
                *v *= rescale;
            }
        }
    }
    (idx, w)
}

#[test]
fn adaptive_topk_renormalizes_survivors_to_the_original_total() {
    // DeepSeek-V4 shape: norm_topk_prob => the six weights sum to
    // routed_scaling_factor (1.5), and the router emits them in SELECTION
    // order (score + correction_bias), which is NOT descending weight — slot 3
    // here is smaller than slot 5. A prune that keyed off slot position would
    // drop the wrong expert; keying off the weight drops slots 3 and 4.
    let idx = [7u32, 40, 12, 99, 3, 51];
    let w = [0.60f32, 0.42, 0.24, 0.015, 0.015, 0.21];
    let total: f32 = w.iter().sum();
    assert!((total - 1.5).abs() < 1e-5);

    let (i2, w2) = prune_ref(&idx, &w, 144, 0.02, true);
    assert_eq!(i2, [7, 40, 12, 144, 144, 51]);
    assert_eq!(w2[3], 0.0);
    assert_eq!(w2[4], 0.0);
    // Renormalized: total preserved, so the routed branch keeps its magnitude.
    assert!((w2.iter().sum::<f32>() - total).abs() < 1e-4);
    // ...and survivors keep their RELATIVE proportions exactly.
    assert!((w2[0] / w2[1] - w[0] / w[1]).abs() < 1e-5);
}

#[test]
fn adaptive_topk_without_norm_topk_prob_only_subtracts() {
    // norm_topk_prob = false: the weights are raw scores, their sum carries
    // meaning, and rescaling would invent magnitude. Drop and leave the rest.
    let idx = [1u32, 2, 3, 4];
    let w = [0.9f32, 0.5, 0.4, 0.02];
    let (i2, w2) = prune_ref(&idx, &w, 144, 0.05, false);
    assert_eq!(i2, [1, 2, 3, 144]);
    assert_eq!(w2, [0.9, 0.5, 0.4, 0.0]);
}

#[test]
fn adaptive_topk_never_drops_every_slot() {
    // A pathologically flat router under an absurd threshold: every slot is
    // below it, but the arg-max is kept unconditionally so the token still
    // reaches a routed expert.
    let idx = [1u32, 2, 3, 4];
    let w = [0.25f32, 0.25, 0.25, 0.25];
    let (i2, w2) = prune_ref(&idx, &w, 144, 0.9, true);
    assert_eq!(i2[0], 1);
    assert_eq!(&i2[1..], &[144, 144, 144]);
    assert!((w2[0] - 1.0).abs() < 1e-5); // renormalized back to the full mass
}

#[test]
fn adaptive_topk_threshold_zero_is_exactly_current_behaviour() {
    let idx = [1u32, 2, 3, 4];
    let w = [0.9f32, 0.5, 0.4, 0.02];
    let (i2, w2) = prune_ref(&idx, &w, 144, 0.0, true);
    assert_eq!(i2, idx);
    assert_eq!(w2, w);
}

#[test]
fn bf16_shared_expert_requires_three_non_null_weights() {
    let gate = DenseWeight {
        weight: DevicePtr(11),
    };
    let up = DenseWeight {
        weight: DevicePtr(22),
    };
    let down = DenseWeight {
        weight: DevicePtr(33),
    };

    let shared = Bf16SharedExpert::new(gate, up, down).expect("valid BF16 shared expert");
    assert_eq!(shared.gate_proj.weight, gate.weight);
    assert_eq!(shared.up_proj.weight, up.weight);
    assert_eq!(shared.down_proj.weight, down.weight);

    let null = DenseWeight {
        weight: DevicePtr::NULL,
    };
    assert!(Bf16SharedExpert::new(null, up, down).is_err());
    assert!(Bf16SharedExpert::new(gate, null, down).is_err());
    assert!(Bf16SharedExpert::new(gate, up, null).is_err());
}
