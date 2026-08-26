// SPDX-License-Identifier: AGPL-3.0-only

pub mod dense_ffn;
pub mod dflash_head;
pub mod ep_dispatch;
pub mod fp8_calibration;
pub mod moe;
pub mod mtp_head;
pub mod mtp_multi;
pub mod nemotron_mamba2;
pub mod nemotron_moe;
pub mod ops;
pub mod qwen3_attention;
pub mod qwen3_ssm;
pub mod qwen4_hyper;
pub mod qwen4_mtp;
pub mod qwen4_ple;
pub mod vision_encoder;

pub use dense_ffn::{DenseFfnLayer, FfnActivation};
pub use dflash_head::{
    BlockDiffusionDraftHead, DDTreePayload, DflashLayer, DflashProposerState, DflashQuantization,
};
pub use moe::MoeLayer;
pub use mtp_head::{MtpHead, MtpQuantization};
pub use nemotron_mamba2::NemotronMamba2Layer;
pub use nemotron_moe::NemotronMoeLayer;
pub use qwen3_attention::Qwen3AttentionLayer;
pub use qwen3_ssm::Qwen3SsmLayer;
pub use qwen4_hyper::Qwen4HyperConnection;
pub use qwen4_mtp::Qwen4MtpHead;
#[cfg(all(feature = "cuda", target_os = "linux"))]
pub use qwen4_ple::Qwen4PleLayer;
pub use qwen4_ple::{PleRowSelection, QWEN4_PLE_HEADS, Qwen4PleHasher};
pub use vision_encoder::{MergerLayer, ViTBlock, VisionEncoder};

use crate::layer::ForwardContext;
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

/// Try to load an optional kernel, logging at debug level if it's not found.
/// Returns `KernelHandle(0)` (null) on failure — callers must check before use.
///
/// Debug (not warn) because misses are expected when a model doesn't use a
/// given feature: e.g. Qwen3-Coder-Next (GDN+attention) never calls MLA
/// kernels, but the layer builder still probes them. Warning on expected
/// misses drowned out genuine problems in startup logs. The place a miss
/// DOES deserve a warning is the dispatch site of a knob the operator
/// explicitly turned on — see [`warn_kernel_fallback`].
///
/// `#[track_caller]` so `kernel_audit` records the LAYER CONSTRUCTOR's
/// `file:line`, not this wrapper's. Every one of the 167 optional lookups in
/// this crate funnels through here; without it `--check-kernels` would name a
/// single line for all of them.
#[track_caller]
pub fn try_kernel(gpu: &dyn GpuBackend, module: &str, func: &str) -> KernelHandle {
    match gpu.kernel(module, func) {
        Ok(h) => h,
        Err(_) => {
            tracing::debug!("Optional kernel '{module}::{func}' not loaded");
            KernelHandle(0)
        }
    }
}

/// Warn ONCE that an explicitly-enabled knob is a no-op because its kernel
/// symbol did not resolve.
///
/// The failure mode this exists for: a knob is gated on
/// `env == "1" && handle.0 != 0`, the operator sets the env var, the PTX cache
/// is stale, and the second conjunct quietly turns the whole thing off. The
/// run then measures the fallback path while the recipe, the logs and the
/// operator all say the knob is on. Our champion recipe sets 42 such
/// variables; upstream's LESSON 11 was that six of ten in their published
/// recipe did nothing.
///
/// `latch` must be a `static` at the call site — these are dispatch-path
/// checks, so an unlatched `warn!` would fire once per layer per token.
/// `consequence` names the path actually taken, because "kernel missing" on
/// its own does not tell an operator whether the number they just recorded is
/// usable.
pub fn warn_kernel_fallback(
    latch: &'static std::sync::Once,
    env_var: &str,
    symbol: &str,
    consequence: &str,
) {
    latch.call_once(|| {
        tracing::warn!(
            "{env_var} is set but the `{symbol}` kernel symbol did not resolve — {consequence}. \
             The knob is a SILENT NO-OP for this run; any measurement taken from it is a \
             measurement of the fallback path. Rebuild the kernel cache, and use \
             `spark serve --check-kernels` to list every unresolved lookup with its dispatch \
             site."
        );
    });
}

/// Returns true when `ATLAS_TC_NVFP4_M16=1` is set in the process env.
///
/// Gates the small-M `w4a16_gemm_t_m16` dispatch (K=γ verify path,
/// M ≤ 32) so the parent `w4a16_gemm_n128` (M_TILE=64) stays the
/// default until the new kernel is benched and validated. Cached
/// via `OnceLock` — `std::env::var` is non-trivial to call on every
/// decode hop.
pub fn tc_nvfp4_m16_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_TC_NVFP4_M16").ok().as_deref() == Some("1"))
}

/// Returns true when `ATLAS_TC_NVFP4_K3=1` is set in the process env.
///
/// Gates the tensor-core M=3 FFN path for K=3 MTP verify on dense
/// Qwen3.6-27B. With the flag on AND the transposed FFN weights loaded
/// (`ATLAS_FFN_M16_TRANSPOSED=1` at startup), `forward_k3` short-circuits
/// to `forward_kgamma(n=3)` which dispatches `w4a16_gemm_t_m16` (m16n8k16
/// BF16 MMA). The kernel pads M=3 → 16 internally and writes only the
/// real 3 rows via its `if (r < M)` bounds checks. Trades 81% MMA-tile
/// waste at the kernel level for tensor-core FLOPS density vs the M=3
/// `w4a16_gemv_dual_batch3` GEMV path. Default off until A/B-validated
/// (production fallback stays the GEMV).
pub fn tc_nvfp4_k3_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_TC_NVFP4_K3").ok().as_deref() == Some("1"))
}

/// Returns true when `ATLAS_SSM_OUT_BATCH3=1` is set in the process env.
///
/// Gates the K=3 triple-GEMV (`w4a16_gemv_batch3`) fast path for the SSM
/// `out_proj` projection (4096→5120 NVFP4). The general `w4a16_gemm`
/// M_TILE=64 path wastes ~96% of MMA work at M=3; the GEMV avoids this
/// but is unverified for byte-equivalence with the FP8 / NVFP4-transposed
/// fall-throughs. Default off so the existing path stays the baseline;
/// `ATLAS_SSM_OUT_BATCH3=1` enables the experiment. Cached via `OnceLock`.
pub fn ssm_out_batch3_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_SSM_OUT_BATCH3").ok().as_deref() == Some("1"))
}

/// Gates routing the SSM `out_proj` [M=17, N=5120, K=6144] through the
/// `w4a16_gemm_t_m32_n64` kernel (N_TILE=64 → 80 CTAs) at the K=γ verify
/// (3 < M ≤ 32) instead of the legacy `w4a16_gemm_t` (N_TILE=128 → only 40
/// CTAs at N=5120, SM-starved like the pre-split-K ffn_down). Same proven
/// T-weight + m32_n64 path as qkv/o/FFN — single K-chain, bit-exact.
/// Default OFF: A/B (2026-06-18) measured NO throughput win — out_proj's
/// K=6144 loop is 3× shorter and its weight 3× smaller than ffn_down's, so
/// it is NOT occupancy-starved; doubling CTAs 40→80 left counting flat
/// (82.5→81.5, within noise). Kept as an opt-in (`ATLAS_SSM_OUT_M32N64=1`,
/// bit-exact: counting md5 unchanged) but off by default to preserve the
/// baseline. Cached via `OnceLock`.
pub fn ssm_out_proj_m32n64() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_SSM_OUT_M32N64").ok().as_deref() == Some("1"))
}

/// Split-K factor for the SSM `out_proj` [M=17, N=5120, K=6144] on the
/// K=γ verify (`ATLAS_SSM_OUT_SPLITK`). Floor-map microbench (2026-07-05,
/// clean per-launch event timing, min-of-150): the production
/// `w4a16_gemm_t` route runs 234.7µs = 28% of the 66µs DRAM floor (40
/// CTAs on 48 SMs, 77 GB/s), and the earlier `ATLAS_SSM_OUT_M32N64`
/// verdict of "not occupancy-starved" was wrong — the m32_n64 single
/// slice measures 167µs (40%) and split-K×4 measures 90.8µs (84%, 228
/// GB/s), a 2.6× kernel-level win worth ~6.7ms/step across 48 layers.
/// Same lossless FP32-partials + `reduce_splitk_f32_to_bf16` pattern as
/// the shipped `ffn_down` split-K. Returns 0 when unset/0/1; else the
/// factor clamped to [2, 8]. Default OFF pending the counting-md5 gate.
pub fn ssm_out_splitk() -> u32 {
    static GATE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_SSM_OUT_SPLITK")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .map(|v| if v < 2 { 0 } else { v.min(8) })
            .unwrap_or(0)
    })
}

/// Split-K factor for the SSM `qkvz` projection [M=17, N=12288, K=5120]
/// on the K=γ verify (`ATLAS_SSM_QKVZ_SPLITK`). Floor-map microbench
/// (2026-07-05): production `w4a16_gemm_t` route = 220.5µs (59% of the
/// 132µs floor); split-K×2 = 167.3µs (85%, 232 GB/s) — ~2.2ms/step
/// across 48 layers. Lossless FP32 partials, mirrors `ffn_down` split-K.
/// Returns 0 when unset/0/1; else clamped to [2, 8]. Default OFF.
pub fn ssm_qkvz_splitk() -> u32 {
    static GATE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_SSM_QKVZ_SPLITK")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .map(|v| if v < 2 { 0 } else { v.min(8) })
            .unwrap_or(0)
    })
}

/// `ATLAS_SSM_PROJ_TC=1` REFREEZE switch: un-shadow the split-K tensor-core
/// transposed-weight paths for the SSM QKVZ and out projections, which the
/// bit-exact `ssm_qkvz_exact` / `ssm_out_exact` branches otherwise catch
/// first for the sequential Qwen3.x layout. The split-K kernels use FP32
/// partials + `reduce_splitk_f32_to_bf16`, so the reduction order differs
/// from the serial FMA oracle and token output can change: the reference
/// completion hash must be re-established after enabling. Default off (the
/// exact branches keep shadowing the TC paths).
pub fn ssm_proj_tc_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_SSM_PROJ_TC").ok().as_deref() == Some("1"))
}

/// Diagnostic-only: force every row of the batched SSM QKVZ projection
/// through the same NVFP4 GEMV used by single-token decode.  This is a
/// losslessness bisection switch, not a serving optimization.
pub fn ssm_qkvz_serial_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_SSM_QKVZ_SERIAL").ok().as_deref() == Some("1"))
}

/// Diagnostic-only counterpart for the SSM output projection.  Re-reading
/// the weight once per verify row is intentionally slow but exactly matches
/// the single-token NVFP4 GEMV dispatch.
pub fn ssm_out_serial_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_SSM_OUT_SERIAL").ok().as_deref() == Some("1"))
}

/// Diagnostic correctness oracle for the recurrent half of a batched SSM
/// verify.  It preserves the FP32 conv/GDN contract used by ordinary decode
/// and processes rows sequentially so recurrent state advances identically.
pub fn ssm_recurrent_serial_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_SSM_RECURRENT_SERIAL").ok().as_deref() == Some("1"))
}

/// Diagnostic-only: use the fused single-token BA projection and gate
/// transform for each verify row, matching ordinary decode exactly.
pub fn ssm_ba_serial_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_SSM_BA_SERIAL").ok().as_deref() == Some("1"))
}

/// Returns true when `ATLAS_SSM_BA_BATCHED=1` is set in the process env.
///
/// Gates the K=3 batched-GEMV path for the SSM BA projection. The
/// unbatched path runs `dense_gemv` once per token (3 launches/layer ×
/// 48 SSM layers/verify), and each launch is ~24μs on GB10 — pure
/// launch-overhead since the kernel itself does N=64 outputs × K=5120
/// reductions (trivial). Batching collapses to 1 launch/layer.
///
/// Default off so the existing path stays the baseline; flip on to
/// enable the experiment. Cached via `OnceLock` — `std::env::var` is
/// non-trivial to call on every decode hop.
pub fn ssm_ba_batched_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_SSM_BA_BATCHED").ok().as_deref() == Some("1"))
}

/// Split-K factor for the K/V projections on the DFlash K=γ verify
/// attention QKV path ([M=n, N=nkv*hd, K=hidden]).
///
/// On AEON-27B the K/V projections are N=nkv*hd=4*256=1024 → the
/// single-slice `w4a16_gemm_t_m32_n64` fields only ceil(1024/64)=16 CTAs on
/// GB10's 48 SMs (severely occupancy-starved), while the Q projection at
/// N=q_proj_dim=12288 already fields 192 CTAs (well-provisioned; split-K is
/// a no-op there, like FFN gate/up). This factor slices the K axis of the
/// K and V GEMMs across gridDim.z into an FP32 workspace, then
/// `reduce_splitk_f32_to_bf16` sums+downcasts. FP32 partials, so nothing
/// rounds to BF16 mid-accumulation — but that is NOT token-exactness, and
/// the earlier wording here claiming it was is wrong: slicing K reassociates
/// the FP32 sum and can move the committed token. See
/// `ops::w4a16_gemm_n64_m32_splitk`. Q stays on the single-slice kernel.
///
/// Note this factor is inert on gated Qwen3.8 anyway: it is read only inside
/// `ms_qkv_batched_plain`, which `exact_attention_qkv_route` short-circuits
/// for every n in 4..=17. Returns 0 (disabled) when unset/0/1; else the
/// parsed factor clamped to [2, 8]. A/B against the single-slice baseline —
/// the win is only real if the extra CTAs raise effective bandwidth on the
/// tiny K/V weights.
pub fn attn_qkv_splitk() -> u32 {
    static GATE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_ATTN_QKV_SPLITK")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .map(|v| if v < 2 { 0 } else { v.min(8) })
            .unwrap_or(0)
    })
}

/// Diagnostic correctness oracle for the attention output projection during
/// multi-token verify. Process every row with the same NVFP4 decode GEMV
/// dispatcher as single-token decode, including the software-kernel choice.
pub fn attn_out_serial_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_ATTN_OUT_SERIAL").ok().as_deref() == Some("1"))
}

/// Diagnostic correctness oracle for paged attention during multi-token
/// verification.  Launch each verification row as an independent one-row
/// paged-decode call so kernel choice, split geometry, and reduction layout
/// match ordinary decode.  Flat chains only; tree indirection intentionally
/// stays on the batched implementation.
pub fn attn_paged_serial_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_ATTN_PAGED_SERIAL").ok().as_deref() == Some("1"))
}

/// V-dim split factor for the DFlash K=17 `gated_delta_rule_wy17` GDN verify.
///
/// The single-slice wy17 launches grid=(num_v_heads=48, batch=1) = 48 CTAs
/// on the 48-SM GB10 — 1 CTA/SM, 4 warps, no second resident block to hide
/// the two k_dim=128 H-state streaming passes. This factor fans each head's
/// v_dim=128 columns across `ATLAS_WY17_SPLIT` CTAs (gridDim.z), so an SM
/// hosts that many blocks and can overlap memory stalls. Each split
/// recomputes the shared kd_flat k-dots (136 block-reductions) — the
/// occupancy/recompute trade. Bit-identical to the single-slice kernel
/// (per-column FP32 math + reduction order unchanged). Returns 0 (disabled)
/// when unset/0/1; else the parsed factor clamped to [2, 4] (v_dim=128 → 64
/// or 32 columns/CTA; beyond 4 the kd_flat recompute dominates).
pub fn wy17_split() -> u32 {
    static GATE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_WY17_SPLIT")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .map(|v| if v < 2 { 0 } else { v.min(4) })
            .unwrap_or(0)
    })
}

/// Returns the WY17 LAZY Hi-write stride J (`ATLAS_WY17_LAZY=J`).
///
/// The wy17 PASS-2 writes 16 per-token intermediate H states (Hi_0..Hi_15),
/// which are 86% of the kernel's DRAM traffic. They exist ONLY for partial-
/// accept rollback, and the commit consumer reads at most ONE of them
/// (inter[num_accepted-1] on partial accept; full accept reads none). J>1
/// makes the kernel persist only CHECKPOINT slots (0, K-2, every J-th); a
/// partial accept whose slot was skipped is reconstructed bit-exactly by
/// `gated_delta_rule_wy17_replay` (root re-seed, same FP32 recurrence).
/// Returns 1 (disabled — write all, bit-identical to the historical kernel)
/// when unset/0/1; else the parsed J clamped to [2, 16]. Outputs and final
/// h_state are byte-identical for every J. md5-gated. Cached via `OnceLock`.
pub fn wy17_lazy() -> u32 {
    static GATE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_WY17_LAZY")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .map(|v| if v < 2 { 1 } else { v.min(16) })
            .unwrap_or(1)
    })
}

/// Returns true when `ATLAS_WY17_LAZY_COMMIT=1` is set in the process env.
///
/// Gates the commit-side half of the WY17 lazy Hi-writes optimisation. When
/// on (and `ATLAS_WY17_LAZY>1`), the wy17 verify kernel persists only
/// checkpoint intermediate slots, and the async-checkpoint commit path
/// reconstructs a skipped non-checkpoint partial-accept slot via the
/// `gated_delta_rule_wy17_replay` kernel instead of a plain intermediate
/// → h_state D2D copy. Requires the per-layer k/v/gate/beta retention buffers
/// (see `SsmLayerState::wy17_kv_retain` / `wy17_gate_retain`). Default OFF.
/// Cached via `OnceLock`.
pub fn wy17_lazy_commit() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_WY17_LAZY_COMMIT").ok().as_deref() == Some("1"))
}

/// Returns true when `ATLAS_DFLASH_ASYNC_PROBE=1` is set in the process env.
///
/// Measurement-only gate for the task-#20 async propose‖verify design probe.
/// When on:
///   * `commit_verify_state_async_dispatch` fences the secondary stream after
///     enqueue and logs the TRUE GPU duration of the SSM commit tail
///     (`ASYNC_PROBE commit_tail`), which the CPU-side `STEP_TIMING`
///     `commit=` figure (enqueue-only) understates;
///   * `forward_block` logs the enqueue-vs-GPU split of the drafter propose
///     (`ASYNC_PROBE propose`), i.e. how much of the propose wall is CPU
///     launch overhead vs actual drafter kernels.
///
/// Adds one stream sync per step per site — measurement only, never enable
/// in production. Default OFF (zero cost). Cached via `OnceLock`.
pub fn dflash_async_probe() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_DFLASH_ASYNC_PROBE").ok().as_deref() == Some("1"))
}

/// Is intermediate slot `s` (0..K-1, i.e. Hi_0..Hi_{K-2}) persisted by the
/// lazy wy17 kernel under stride `j`?
///
/// This MUST match `wy17_is_checkpoint` in `gated_delta_rule_wy17.cu` exactly,
/// because the commit path uses it to decide whether `h_intermediate[s]` holds
/// a real state (checkpoint → plain D2D copy) or a skipped slot that must be
/// reconstructed via `gated_delta_rule_wy17_replay` (non-checkpoint → replay).
///
/// Kernel predicate:  `s == 0 || s == K-2 || ((s+1) % j == 0)`.
/// `j <= 1` disables lazy (every slot written) so all slots are checkpoints.
///
/// `s` is the pool `token_idx` (== the kernel's Hi index) and is expected in
/// `0..k` where `k = K = num verify tokens = γ+1` (so valid Hi slots are
/// `0..=K-2`). A caller passing `s >= K-1` (e.g. the final live-h_state slot)
/// is treated as a checkpoint (no replay) — replay only reconstructs the
/// `Hi_0..Hi_{K-2}` intermediates the kernel could have skipped.
#[inline]
pub fn wy17_is_checkpoint(s: usize, j: u32, k: usize) -> bool {
    if j <= 1 {
        return true;
    }
    // Slots outside the intermediate range (0..=K-2) are never lazily skipped.
    if k < 2 || s >= k - 1 {
        return true;
    }
    let last_inter = k - 2; // K-2
    s == 0 || s == last_inter || ((s as u32) + 1).is_multiple_of(j)
}

#[cfg(test)]
mod wy17_checkpoint_tests {
    use super::wy17_is_checkpoint;

    const K: usize = 17; // DFlash γ+1

    #[test]
    fn lazy_disabled_j_le_1_all_checkpoints() {
        for s in 0..K {
            assert!(wy17_is_checkpoint(s, 0, K), "j=0 s={s}");
            assert!(wy17_is_checkpoint(s, 1, K), "j=1 s={s}");
        }
    }

    #[test]
    fn slot_0_and_k_minus_2_always_checkpoints() {
        for j in 2..=16u32 {
            assert!(wy17_is_checkpoint(0, j, K), "slot 0 j={j}");
            assert!(wy17_is_checkpoint(K - 2, j, K), "slot K-2 j={j}");
        }
    }

    #[test]
    fn final_slot_and_beyond_treated_as_checkpoint() {
        // K-1 (the live-h_state slot) and any OOB slot are never skipped.
        for j in 2..=16u32 {
            assert!(wy17_is_checkpoint(K - 1, j, K), "slot K-1 j={j}");
            assert!(wy17_is_checkpoint(K, j, K), "slot K j={j}");
        }
    }

    #[test]
    fn j8_matches_kernel_predicate() {
        // j=8, K=17: checkpoints are s==0, s==15, or (s+1)%8==0 → s∈{7,15}.
        // Memory note (J=8): 3 written slots {0, 7, 15}.
        let expected_ckpt: Vec<usize> = vec![0, 7, 15];
        for s in 0..(K - 1) {
            let is_ckpt = wy17_is_checkpoint(s, 8, K);
            assert_eq!(
                is_ckpt,
                expected_ckpt.contains(&s),
                "j=8 s={s} got {is_ckpt}"
            );
        }
        // Exactly 3 checkpoint intermediates (matches microbench "3 writes").
        let n_ckpt = (0..(K - 1))
            .filter(|&s| wy17_is_checkpoint(s, 8, K))
            .count();
        assert_eq!(n_ckpt, 3, "J=8 must persist 3 intermediate slots");
    }

    #[test]
    fn j4_matches_kernel_predicate() {
        // j=4, K=17: s==0, s==15, or (s+1)%4==0 → s∈{3,7,11,15}. Plus 0.
        // → {0, 3, 7, 11, 15} = 5 written slots (matches microbench "5 writes").
        let expected_ckpt: Vec<usize> = vec![0, 3, 7, 11, 15];
        for s in 0..(K - 1) {
            assert_eq!(
                wy17_is_checkpoint(s, 4, K),
                expected_ckpt.contains(&s),
                "j=4 s={s}"
            );
        }
        let n_ckpt = (0..(K - 1))
            .filter(|&s| wy17_is_checkpoint(s, 4, K))
            .count();
        assert_eq!(n_ckpt, 5, "J=4 must persist 5 intermediate slots");
    }

    #[test]
    fn j2_writes_alternating_plus_endpoints() {
        // j=2, K=17: s==0, s==15, or (s+1)%2==0 → all odd s, plus 0.
        // odd s in 0..=15 = {1,3,5,7,9,11,13,15} (8) plus s=0 = 9 slots.
        let n_ckpt = (0..(K - 1))
            .filter(|&s| wy17_is_checkpoint(s, 2, K))
            .count();
        assert_eq!(n_ckpt, 9, "J=2 checkpoint count");
        assert!(wy17_is_checkpoint(1, 2, K));
        assert!(!wy17_is_checkpoint(2, 2, K));
    }

    #[test]
    fn small_k_never_panics() {
        // Guard the k<2 branch and tiny verify windows (K=2/3).
        for k in 0..=3usize {
            for s in 0..=4usize {
                for j in 0..=4u32 {
                    let _ = wy17_is_checkpoint(s, j, k);
                }
            }
        }
    }
}

/// Returns true when `ATLAS_SSM_BA_BATCH=1` is set in the process env.
///
/// Gates the GENERAL (any num_tokens) batched BA projection for the SSM
/// in_proj_ba. The baseline runs `dense_gemv` once per token — at DFlash
/// γ=16 that is 17 launches/layer × 48 SSM layers = 816 tiny GEMV launches
/// per K=γ verify, each computing only N=64 outputs × K=5120 reductions.
/// The weight (`in_proj_ba`) is IDENTICAL across tokens, so the 17 GEMVs
/// collapse into ONE `dense_gemm` at M=num_tokens (weight read once, all
/// tokens' rows computed together). This cuts BOTH launch overhead AND the
/// 17× redundant weight streaming — reducing graph *execution* time, not
/// just launch count. Distinct from `ATLAS_SSM_BA_BATCHED` which only
/// covers the K=3 MTP path via `dense_gemv_batch3`. Bit-exact: `dense_gemm`
/// and `dense_gemv` share the same BF16 accumulation math. md5-gated.
/// Default OFF for A/B safety. Cached via `OnceLock`.
pub fn ssm_ba_batch_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_SSM_BA_BATCH").ok().as_deref() == Some("1"))
}

/// Returns true when `ATLAS_FFN_DUAL_TUNED=1` is set in the process env.
///
/// Gates the tuned `w4a16_gemv_dual_batch3_tuned` kernel that fuses the
/// gate and up projections into the SAME CTA (8 outputs / CTA instead of
/// 4×2-CTA dispatch) so the 3-token activation vector is loaded only once
/// per CTA. Targets the FFN K=3 verify path on dense Qwen3.6-27B where
/// `ffn_gate_up_dual_batch3` is the bandwidth-bound bleeder (~475μs/call,
/// 64 layers × 31 verifies/gen = ~940ms / generation). Default off until
/// byte-equivalence is verified. Cached via `OnceLock`.
pub fn ffn_dual_tuned_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_FFN_DUAL_TUNED").ok().as_deref() == Some("1"))
}

/// Returns true when `ATLAS_FFN_KGAMMA_M16=1` is set in the process env.
///
/// Gates the batched K=γ verify FFN path that replaces the per-token loop
/// (`for i in 0..n { ffn.forward(...) }`) with 3 `w4a16_gemm` calls at
/// M=n. At DFlash γ=16 the loop costs 64 layers × 17 weight reloads per
/// step (~145 GB of redundant LPDDR5X traffic); the batched path loads
/// each weight once per layer (~8.6 GB) — an 18× reduction on the
/// dominant cost in the K=γ profile. Default off so the per-token loop
/// stays the baseline until A/B-validated. Cached via `OnceLock`.
pub fn ffn_kgamma_m16_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_FFN_KGAMMA_M16").ok().as_deref() == Some("1"))
}

/// Returns true when `ATLAS_FFN_KGAMMA_M128=1` is set in the process env.
///
/// Routes the batched K=γ verify FFN through `w4a16_gemm_t_m128`
/// (M_TILE=128) instead of `w4a16_gemm_t_m16`. At M=17 the m16 kernel
/// runs ceil(17/16)=2 M-tile rows and each tile row re-reads the FULL
/// weight matrix — 2× B DRAM traffic on a memory-bound GEMM. The m128
/// kernel covers M=17 in ONE tile (single weight read); its compute
/// waste on 111 phantom rows is irrelevant because the kernel is
/// bandwidth-bound (see forward_prefill's fp8_fast_path note). Expected
/// ~40ms verify reduction at K=17 on Qwen3.6-27B Full.
pub fn ffn_kgamma_m128_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_FFN_KGAMMA_M128").ok().as_deref() == Some("1"))
}

/// ATLAS_FFN_KGAMMA_WIDE=1: extend the transposed K=γ FFN family to
/// 32 < n <= 256 via `w4a16_gemm_t_m128` (SASS audit 2026-07-08: at c>=2
/// batched verify M=17c>32 previously fell back SILENTLY to the legacy
/// `w4a16_gemm` — no cp.async, scalar LDG.U8, ~4x sector overfetch,
/// issue-capped at ~47% of DRAM BW — the measured cause of the concurrency
/// ceiling). Default OFF until the c=2/4 token-exactness A/B passes
/// (m128-vs-legacy accumulation order differs, so md5 must be re-proven).
pub fn ffn_kgamma_wide_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_FFN_KGAMMA_WIDE").ok().as_deref() == Some("1"))
}

/// Split count for the DFlash K=γ verify FFN `down_proj` GEMM.
///
/// The down projection ([M=17, N=5120, K=16384]) is occupancy-starved on
/// the single-slice `w4a16_gemm_t_m32_n64` kernel: N=5120 → only 80 CTAs
/// vs gate/up's 256 at N=16384, and it grinds a 512-iteration K-loop. Per
/// full_profile (2026-06-18) it runs at ~91 GB/s vs gate/up ~163 GB/s on
/// the same-size weight. Split-K multiplies the CTA count by this factor
/// (80 → 320 at 4) to restore occupancy. Returns 0 (disabled, single
/// slice) when the env var is unset/0/1; otherwise the parsed split count
/// clamped to [2, 8]. `ATLAS_FFN_DOWN_SPLITK=4` is the records-grade
/// default applied by the serve script.
pub fn ffn_down_splitk() -> u32 {
    static GATE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_FFN_DOWN_SPLITK")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .map(|v| if v < 2 { 0 } else { v.min(8) })
            .unwrap_or(0)
    })
}

/// Returns true when `ATLAS_FFN_FUSED_GATEUP=1` is set in the process env.
///
/// Routes the K=γ verify FFN gate_proj + up_proj + SiLU·mul through the
/// single fused kernel `w4a16_gemm_t_m32_n64_gateup_silu` instead of two
/// separate m32_n64 GEMMs + a standalone `moe_silu_mul`. Loads the shared
/// [M,K] input tile once and writes only the fused silu(gate)*up [M,N]
/// activation (eliminates the two [M,N] activation writes + reads of the
/// standalone silu_mul and one kernel launch). Requires transposed FFN
/// weights + the fused kernel symbol; falls back to the m32/m16 path
/// otherwise. Mutually exclusive with ATLAS_FFN_GATEUP_SPLITK (the
/// fused path supersedes gate/up split-K). Byte-exact vs the unfused
/// path: the fused kernel rounds each gate/up accumulator to BF16 and
/// back to FP32 before the silu·mul, exactly reproducing the baseline's
/// BF16 activation round-trip (m32_n64 writes gate_out/up_out as BF16 →
/// moe_silu_mul reloads FP32). md5-gated. Default OFF for A/B safety.
/// Returns true when `ATLAS_DEQUANT_PIPE=1` is set in the process env.
///
/// When the fused gate+up+SiLU path is active (`ffn_fused_gateup_enabled`),
/// this routes it through the DEQUANT-IN-REGISTERS variant
/// `w4a16_gemm_t_m32_n64_gateup_silu_pipe` instead of the SMEM-staged
/// baseline `w4a16_gemm_t_m32_n64_gateup_silu`. Same shape, same caller
/// signature, byte-exact output — the NVFP4→FP8 dequant runs in registers
/// immediately before each `mma.sync.e4m3` (from resident packed W4 bytes)
/// rather than being materialized into a `smem_B_fp8` staging array behind
/// two `__syncthreads`. This (a) drops the second per-K-step barrier, (b)
/// shrinks SMEM ~27% (10.9 KB vs 15.0 KB → higher occupancy — the opposite
/// of the `pipe3` fork which grew SMEM and lost a block/SM), and (c) uses
/// `cp.async.wait_group<1>` so the next tile's memory load overlaps the
/// current tile's register dequant + MMA instead of a full drain. Requires
/// the `_pipe` kernel symbol; falls back to the staged fused kernel when
/// missing. Default OFF for A/B safety — md5-gated (greedy-counting
/// constitution 91a6ff90d50736f779c09db67a96db2d).
pub fn dequant_pipe_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_DEQUANT_PIPE").ok().as_deref() == Some("1"))
}

pub fn ffn_fused_gateup_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_FFN_FUSED_GATEUP").ok().as_deref() == Some("1"))
}

/// Returns true when `ATLAS_GATEUP_K64=1` is set — routes the fused
/// gate+up+SiLU kernel to the K_STEP=64 register-dequant variant
/// (`w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64`). Halves K-loop
/// iterations (80 vs 160) and sync count while doubling per-step load
/// volume for better memory-latency overlap. Takes priority over
/// ATLAS_DEQUANT_PIPE when both are set. Requires K divisible by 64.
pub fn gateup_k64_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_GATEUP_K64").ok().as_deref() == Some("1"))
}

/// Split-K factor for the FFN **gate/up** projections ([M=17, N=inter=16384,
/// K=hidden=5120]) on the K=γ verify path. Default OFF (0).
///
/// Unlike `ffn_down`, gate/up are NOT occupancy-starved on the single-slice
/// `w4a16_gemm_t_m32_n64` kernel — at N=16384 they field ceil(16384/64)=256
/// CTAs (~2.4/SM on GB10's ~108 SMs) vs down's 80 at N=5120. But the K=5120
/// loop (160 K-steps) may still leave latency-hiding headroom: `ATLAS_FFN_
/// GATEUP_SPLITK=2` slices K across gridDim.z, doubling CTAs to 512 (~4.7/SM)
/// into an FP32 workspace, then `reduce_splitk_f32_to_bf16` sums+downcasts.
/// Lossless (FP32 partials) and token-exact — mirrors the proven ffn_down
/// path exactly. Returns 0 when unset/0/1; else the parsed count clamped to
/// [2, 8]. A/B this against the single-slice baseline before shipping — the
/// win is only real if the extra CTAs actually raise effective bandwidth.
pub fn ffn_gateup_splitk() -> u32 {
    static GATE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_FFN_GATEUP_SPLITK")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .map(|v| if v < 2 { 0 } else { v.min(8) })
            .unwrap_or(0)
    })
}

/// Returns true when `ATLAS_SSM_MULTI_SEQ_BATCHED=1` is set in the process env.
///
/// Gates the batched-projections multi-seq SSM decode path. With this off
/// (default), `Qwen3SsmLayer::decode_multi_seq_inner` delegates to a per-seq
/// loop of `self.decode(...)` — proven correct but each layer's GEMV weights
/// are re-read for every sequence, so c=4 aggregate is stuck near c=1's
/// throughput (~22-29 tok/s on AEON-27B). With this on, the projections
/// (QKVZ, BA, out_proj, FFN gate/up/down) are batched at M=n via the same
/// M=n kernels `decode_batched` uses for multi-token verify; the SSM
/// recurrent ops (conv1d_update, gdn_decode, compute_gdn_gates) stay in a
/// per-seq loop because their state pointer (h_state, conv_state) is
/// per-sequence. Each per-seq op uses per-seq input/output offsets so no
/// scratch aliasing occurs across sequences. The post-norm + FFN + residual
/// stack reuses the same batched K=2 / K=3 / K=γ paths as the existing
/// `decode_batched` and `qwen3_attention::ms_phase_ffn` helpers. Default
/// off until A/B-validated; opt-in via env var for safety. Cached via
/// `OnceLock`.
pub fn ssm_multi_seq_batched_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_SSM_MULTI_SEQ_BATCHED").ok().as_deref() == Some("1"))
}

/// Returns true when `ATLAS_SSM_MULTI_SEQ_KERNEL=1` is set in the process env.
///
/// Gates the multi-seq state-advance kernels (conv1d_update_multi_seq,
/// gdn_decode_multi_seq, compute_gdn_gates_multi_seq) that collapse the
/// per-seq SSM-recurrence loop in `decode_multi_seq_batched` into ONE
/// launch per op, advancing all c sequences in parallel. Each (vh, seq)
/// CTA reads its own per-seq state pointer from a small device array
/// uploaded once per call. Requires `ATLAS_SSM_MULTI_SEQ_BATCHED=1` to
/// be on first; checked at call time so the per-seq loop kernels still
/// run when this gate is off. Cached via `OnceLock`.
pub fn ssm_multi_seq_kernel_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_SSM_MULTI_SEQ_KERNEL").ok().as_deref() == Some("1"))
}

/// Returns true when `ATLAS_SSM_MULTI_SEQ_GRAPH=1` is set in the process env.
///
/// Gates CUDA graph capture for the c-batched (n≥2) decode dispatch path.
/// Default off because the SSM state pointers (h_state, conv_state) are
/// per-sequence and per-pool-slot — a captured graph that bakes those
/// pointers into kernel args becomes stale whenever the active sequence
/// set changes (batch composition drift via scheduler `swap_remove` on
/// sequence completion, new arrivals taking different pool slots). The
/// multi-seq SSM kernels accept the per-seq h_state / conv_state arrays
/// via a device-resident pointer table (`SsmStatePool::multi_seq_ptr_table`).
/// When this gate is on, the dispatcher uploads that table BEFORE
/// `begin_capture` each step, so the captured graph holds only the
/// (fixed) table address and replays consume the freshly-uploaded
/// per-step pointers without re-capture.
///
/// Requires `ATLAS_SSM_MULTI_SEQ_BATCHED=1` + `ATLAS_SSM_MULTI_SEQ_KERNEL=1`
/// (the multi-seq kernel path that consumes the pointer table) AND the
/// dispatcher's non-EP, no-comm path (graph capture is illegal under
/// NCCL all-reduce). Cached via `OnceLock`.
pub fn ssm_multi_seq_graph_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_SSM_MULTI_SEQ_GRAPH").ok().as_deref() == Some("1"))
}

/// Returns true when `ATLAS_MULTISEQ_GRAPHS=1` is set in the process env.
///
/// Gates the **piecewise CUDA-graph** multi-seq decode dispatch
/// (`decode_batch_dispatch_piecewise`). This is the pragmatic sibling of
/// the monolithic `ATLAS_SSM_MULTI_SEQ_GRAPH` path: instead of trying to
/// capture the *entire* per-step forward into one graph (which requires
/// indirecting every per-slot device address a captured kernel reads — the
/// SSM state pointers, KV block tables, slot mappings, attention split-K
/// launch geometry, and embedding offsets), it captures only the segments
/// that are provably address-stable and runs the rest eagerly:
///
///   * **SSM + FFN layer runs** — captured. The only per-slot addresses
///     they read (the recurrent `h_state` / `conv_state` pointers) are
///     already indirected through the layer-stable
///     `ssm_multi_seq_ptr_scratch` device buffer (see
///     `qwen3_ssm::decode_multi_seq_inner`, "Fix B"): the graph bakes only
///     the *fixed* scratch address, and the freshly-uploaded per-step
///     pointer table is consumed on replay. Requires the multi-seq SSM
///     kernel path (`ssm_multi_seq_kernel_enabled`) so that indirection is
///     actually taken; otherwise the per-seq fallback bakes raw
///     `SsmLayerState` pointers and the segment is *not* captured.
///   * **FullAttention layers** — run EAGER, never captured. Their paged
///     decode picks a split-K partition (`num_splits`, hence the kernel
///     grid dims) from a *host* scalar `max_seq_len_host = max(seq_lens)+1`
///     (see `qwen3_attention::multi_seq::mod`). Baking that grid geometry
///     into a graph makes replay stale the moment any sequence grows across
///     a split-K threshold — the documented "one token corrupted per N=4
///     stream". Running attention eagerly sidesteps the whole class.
///   * **final norm + LM head** — captured as a tail segment (all fixed
///     model-buffer addresses).
///
/// Because every captured segment now reads ONLY fixed device addresses,
/// the segment-graph cache is keyed on **`(padded_n, segment_id)` alone**
/// — NOT on the active slot tuple. A graph captured for a given padded
/// batch size replays for *any* slot set of that size. Pad slots point at
/// the dedicated `ssm_pool.dummy_slot()` / `dummy_kv_block` sentinels
/// (vLLM PAD_SLOT_ID pattern), so one max-batch-size capture serves any
/// `n <= padded_n`.
///
/// Metadata (positions / slot mapping / seq_len / block table / SSM ptr
/// table) is uploaded to those fixed addresses *before* each segment's
/// replay, OUTSIDE the captured region, mirroring vLLM's persistent-batch
/// gather-before-replay. Default off; opt-in via env var. Cached via
/// `OnceLock`.
pub fn multiseq_graphs_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_MULTISEQ_GRAPHS").ok().as_deref() == Some("1"))
}

/// Returns true when `ATLAS_MTP_K3_BATCH_CSEQ=1` is set in the process env.
///
/// Gates the c-batched K=3 verify path (Path B): instead of running c
/// independent single-sequence K=3 verify forwards (each at M=3 = c × 3 =
/// 4 × 3 = 12 forwards per layer per step at c=4), runs K=3 SEQUENTIAL
/// single-step c-batched decodes (each at M=c = 4 = 3 forwards per layer
/// per step at c=4, K=3). Trades 12 → 3 forward calls per layer per step,
/// keeping the same M=c×K total work but folding c-dimensional weight
/// reuse onto the SSM+attn+FFN GEMMs (which are LPDDR5X bandwidth-bound
/// per `forward_kgamma` profile).
///
/// **Implementation strategy (per-step c-batched K-loop):**
/// 1. Saves per-seq SSM h_state / conv_state into `ssm_pool.h_intermediate(
///    layer, slot, k)` after each step k of the loop, so the existing
///    `commit_verify_state_async(seq, num_accepted, K)` partial-rollback
///    path (which reads `h_intermediate(layer, slot, num_accepted - 1)`)
///    works unchanged.
/// 2. Each step k runs `decode_batch_dispatch(tokens[..c], seqs[..c])`,
///    which already uses `decode_multi_seq` (batched projections + per-seq
///    SSM state advance) when `ATLAS_SSM_MULTI_SEQ_BATCHED=1` and the
///    multi-seq SSM kernels when `ATLAS_SSM_MULTI_SEQ_KERNEL=1`.
/// 3. After K=3 steps, the scheduler computes per-seq `accept_count[i] ∈
///    [1, K]` (last_token + matched drafts) by comparing each step's
///    argmax to the next-expected draft, and routes each seq through the
///    standard per-seq commit/emit/propose pipeline from `verify_k3_step`.
///
/// **Constraints / fallback:** only activates when ALL c active seqs have
/// `drafts.len() == 2` (K=3 path), no grammar boundary, not finished, and
/// `c >= 2`. Otherwise falls back to the per-seq `step_verify_k3` loop.
/// Requires `ATLAS_SSM_MULTI_SEQ_BATCHED=1` + `ATLAS_SSM_MULTI_SEQ_KERNEL=1`
/// (set both for the c-batched SSM kernel path) so the per-step decode
/// actually batches across c — without those gates, `decode_multi_seq`
/// falls back to a per-seq loop and the batched dispatch yields no win.
///
/// ## Status (2026-05-23 measured) — DISABLED by default, blocked on
/// graphed multi-seq decode
///
/// Scaffolding shipped and operationally correct (output coherent at
/// c=2/4/8) but performance REGRESSES vs the per-seq single-graphed-K=3
/// path:
///   baseline (A): c=1 31.8 / c=2 24.8 / c=4 23.8 / c=8 21.4 tok/s
///   CSK     (B): c=1 14.2 / c=2 15.9 / c=4 17.3 / c=8 19.0 tok/s
///
/// **Root cause:** `decode_batch_dispatch` for n≥2 explicitly disables
/// CUDA graphs (decode_a2.rs:99 — `let use_graphs = false`) because the
/// SSM state pointers (h_state, conv_state) are baked into per-seq
/// kernel args of `gdn_decode` / `conv1d_update` at capture time, and
/// batch composition changes (sequences finishing, swap_remove) replay
/// graphs with stale pointers and corrupt SSM state. The per-step
/// c-batched K-loop runs 3 UNGRAPHED c-batched forwards per verify;
/// each costs ~600-1300 ms at c=2-3 vs ~85 ms for ONE graphed single-
/// seq K=3 verify. The 3 × 1 s × c structural cost dwarfs the ~12 → 3
/// FFN-call savings.
///
/// **Unblocker:** make `decode_batch_dispatch` graphable for n≥2 by
/// either (a) marshalling per-seq SSM state pointers through an
/// indirection table (`ssm_pool.ptr_scratch`) that the kernel reads at
/// launch time, OR (b) per-slot graph cache keyed on slot tuple. Once
/// that lands, CSK should hit its 4× FFN-call savings target.
///
/// Default off until the multi-seq graph unblocker lands. Cached via
/// `OnceLock`.
pub fn mtp_k3_batch_cseq_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_MTP_K3_BATCH_CSEQ").ok().as_deref() == Some("1"))
}

/// Returns true when `ATLAS_MTP_K2_BATCH_CSEQ=1` is set in the process env.
///
/// K=2 sibling of `mtp_k3_batch_cseq_enabled()`. Gates the c-batched K=2
/// verify path: instead of running c independent single-sequence K=2
/// graphed verify forwards (each at M=2 = c × 2 = 2c forwards per layer
/// per step), runs K=2 SEQUENTIAL single-step c-batched decodes (each
/// at M=c = 2 forwards per layer per step). Trades 2c → 2 forward calls
/// per layer per step, keeping the same M=c×K total work but folding
/// c-dimensional weight reuse onto the SSM+attn+FFN GEMMs.
///
/// **Why K=2 specifically:** production Q36-35B-A3B runs with
/// `--num-drafts 1` (K=2 path). The K=3 CSK gate `mtp_k3_batch_cseq_enabled`
/// requires `drafts.len() == 2`, never activated in production. K=2 CSK
/// requires `drafts.len() == 1` (the common production case).
///
/// **Constraints / fallback:** only activates when ALL c active seqs have
/// `drafts.len() == 1` (K=2 path), no grammar boundary, not finished, and
/// `c >= 2`. Otherwise falls back to the per-seq `step_verify_k2` loop.
/// Requires `ATLAS_SSM_MULTI_SEQ_BATCHED=1` + `ATLAS_SSM_MULTI_SEQ_KERNEL=1`
/// (and ideally `ATLAS_SSM_MULTI_SEQ_GRAPH=1`) for the underlying
/// `decode_batch_dispatch` to actually batch SSM work across c instead
/// of falling back to a per-seq loop.
///
/// See the K=3 sibling's status doc for context: the win is bounded by
/// whether `decode_batch_dispatch` (n≥2) hits the graphed multi-seq
/// kernel path. Cached via `OnceLock`.
pub fn mtp_k2_batch_cseq_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_MTP_K2_BATCH_CSEQ").ok().as_deref() == Some("1"))
}

/// Returns true when `ATLAS_PREFILL_FFN_FAST=1` is set in the process env.
///
/// Gates the dense FFN large-M prefill path's routing through the
/// `w4a16_gemm_t_m128` (M_TILE=128, N_TILE=128) kernel using the
/// transposed (`nvfp4_t`) FFN weight layout (already populated when
/// `ATLAS_FFN_M16_TRANSPOSED=1`). At M ≥ 128 the default `w4a16_gemm`
/// (M_TILE=64, N_TILE=64) is bandwidth-bound on weight DRAM traffic —
/// the M_TILE=128 kernel loads each weight tile once for 128 rows of A
/// instead of twice for two 64-row tiles, halving the per-layer weight
/// re-read count. Kernel comment claims ~2x speedup at ISL>128 vs
/// `w4a16_gemm_t`. Profile (task #98, ISL=3575): moe_ffn = 17.5s = 84%
/// of 21s TTFT (273 ms/layer × 64 layers) at ~7% of GB10 BF16 peak;
/// theoretical bandwidth-bound ceiling is ~3.5x higher. Mirror of the
/// attention prefill `w4a16_gemm_m128_dispatch` pattern (see
/// `qwen3_attention/prefill_weights.rs:14`). Requires the transposed
/// weights AND the `w4a16_gemm_t_m128` kernel symbol; falls back to
/// the M_TILE=64 path silently when either is missing. Default off
/// until A/B-validated. Cached via `OnceLock`.
pub fn prefill_ffn_fast_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // Default ON (opt-out via ATLAS_PREFILL_FFN_FAST=0). A/B-validated on
    // AEON-Q36-27B (qwen3.6-27b target): the M_TILE=128 transposed-weight
    // path runs the dense FFN prefill GEMM at ~5.4 ms/layer vs ~41.8 ms/layer
    // on the M_TILE=64 baseline (7.7×), cutting 510-tok prefill 3166→840 ms
    // (161→607 tok/s) with no output-coherence change. Memory cost of the
    // transposed weights is ~150 MB total. Honors an explicit "0" to disable.
    *GATE.get_or_init(|| std::env::var("ATLAS_PREFILL_FFN_FAST").ok().as_deref() != Some("0"))
}

/// Returns true when the prefill FFN baseline path should use the cp.async
/// double-buffered byte-exact shadow kernel (`w4a16_gemm_pipe`). Default OFF
/// (`ATLAS_PREFILL_FFN_PIPE=1` to enable) — the pipe kernel preserves the
/// baseline's dequant + MMA arithmetic exactly, so the route is bit-exact;
/// it only changes the load pipeline. When the pipe kernel symbol is missing
/// (older kernel caches) the dispatch falls back to the baseline silently.
pub fn prefill_ffn_pipe_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_PREFILL_FFN_PIPE").ok().as_deref() == Some("1"))
}

/// Returns true when the prefill projection non-transposed fallback routes
/// through the byte-exact pipelined `w4a16_gemm_pipe` instead of the
/// latency-bound baseline `w4a16_gemm` (`ATLAS_PREFILL_PROJ_PIPE=1`).
///
/// With `ATLAS_PREFILL_PROJ_FAST=0` (the champion bit-exact config), the
/// attention QKV/O and SSM QKVZ/out prefill projections fall to the baseline
/// non-transposed M_TILE=64 `w4a16_gemm`, which is latency-bound at small M
/// (~21 GB/s vs the decode kernels' ~190 GB/s). The pipe kernel is a
/// byte-exact shadow (same dequant arithmetic, same m16n8k16 MMA order, only
/// the cp.async load pipeline differs), so routing these projections through
/// it preserves exactness while removing the per-layer latency floor.
/// Default off; the projection dispatch falls back to the baseline when the
/// handle is missing.
pub fn prefill_proj_pipe_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_PREFILL_PROJ_PIPE").ok().as_deref() == Some("1"))
}

/// Returns true when the attention AND SSM prefill projection transposed-TC
/// fast paths are enabled. Default ON (opt-out via `ATLAS_PREFILL_PROJ_FAST=0`).
///
/// The attention Q/K/V/O and the SSM QKVZ/out PREFILL projections all route
/// through transposed (`nvfp4_t`) tensor-core GEMMs (`w4a16_gemm_n128*`) by
/// default — the SAME class of kernel as the FFN prefill fast path that
/// drifted the prompt hidden state and flipped hard.05/expert.02. Opting out
/// routes them through the non-transposed M_TILE=64 `w4a16_gemm` instead,
/// mirroring the FFN fix. This only affects the PREFILL projections; the
/// transposed weights remain installed for the decode/verify path. Cached via
/// `OnceLock`.
pub fn prefill_proj_fast_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_PREFILL_PROJ_FAST").ok().as_deref() != Some("0"))
}

/// Returns true when `ATLAS_FFN_PREDEQUANT_FP8=1` is set in the process env.
///
/// Gates the dense FFN large-M prefill path's routing through the
/// `fp8_gemm_t_m128` kernel using pre-dequanted FP8 [N, K] weights
/// allocated by `DenseFfnLayer::predequant_for_prefill`. This sidesteps
/// the per-K-step DEQUANT phase + one __syncthreads in
/// `w4a16_gemm_t_m128` (the existing fast path), trading additional
/// GPU memory (~17 GB for the 3 × N×K FP8 buffers across 64 layers of
/// Qwen3.6-27B) for compute-time savings. Mirrors the attention
/// `predequant_for_prefill` pattern
/// (qwen3_attention/prefill_weights.rs:161). Requires the
/// `fp8_gemm_t_m128` kernel symbol AND the loader to call
/// `predequant_for_prefill` after `DenseFfnLayer::new` (gated by the
/// SAME env var on the loader side). Default off until A/B-validated
/// and memory cost is acceptable in the production deploy. Cached via
/// `OnceLock`.
pub fn prefill_ffn_fp8_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_FFN_PREDEQUANT_FP8").ok().as_deref() == Some("1"))
}

/// Returns true when `ATLAS_E2M1_GEMM=1` is set in the process env.
///
/// Gates the dense FFN large-M prefill path's routing through the new
/// W4A4 (NVFP4 activation × NVFP4 weight) tensor-core GEMM kernel
/// (`nvfp4_nvfp4_gemm_t_m64`). On SM120/SM121 (GB10), the kernel emits
/// native `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32`
/// instructions — the same hardware path FlashInfer's CUTLASS NVFP4 GEMM
/// uses under the hood, without the CUTLASS host stack. Theoretical 2×
/// MFU lift over the existing `w4a16_gemm_t_m128` (BF16×NVFP4) path.
///
/// Requires:
///   - the `nvfp4_cutlass::nvfp4_nvfp4_gemm_t_m64` kernel symbol
///   - the `quantize_nvfp4` module (already loaded for weight quantization)
///   - prefill `M >= 128` (kernel's intended window)
///
/// Activations are prequantized BF16 → NVFP4 inline before each GEMM
/// using `nvfp4_global_absmax` + `quantize_bf16_to_nvfp4`. Falls back to
/// the existing dispatch chain (fp8 / v2 / m128 fast paths, then the
/// M_TILE=64 baseline) when the gate is off, kernels are missing, or
/// `M < 128`. Default off until A/B-validated. Cached via `OnceLock`.
pub fn prefill_ffn_e2m1_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_E2M1_GEMM").ok().as_deref() == Some("1"))
}

/// Per-shape dispatch: route ONLY down_proj through E2M1 hardware MMA.
/// gate_proj + up_proj stay on w4a16_gemm_t_m128 (faster for K=5120,N=17408).
/// down_proj (K=17408,N=5120) is 1.31x faster via E2M1 hardware MMA path.
/// Net: ~30% down_proj savings, no gate/up regression.
pub fn prefill_ffn_e2m1_down_only_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_E2M1_GEMM_DOWN_ONLY").ok().as_deref() == Some("1"))
}

/// Returns true when `ATLAS_FFN_M128_V2=1` is set in the process env.
///
/// Gates the dense FFN large-M prefill path's routing through the
/// `w4a16_gemm_t_m128_v2` kernel (8-warp shadow of the v1 m128 kernel).
/// Same SMEM + same 2-stage cp.async pipeline as v1, but parallelizes
/// chunk 0 and chunk 1 MMAs across warps {0-3} and {4-7} instead of
/// serializing both chunks on 4 warps. 2× warps/SM → more MMA pipeline
/// slots in flight; best on compute-bound large-K GEMMs. Originally a
/// MiniMax-only kernel (kernels/gb10/minimax-m2-229b/nvfp4/
/// w4a16_gemm_v2.cu) — copied into qwen3.6-27b to A/B against the
/// existing v1 fast path. Requires the transposed (`nvfp4_t`) FFN
/// weights AND the `w4a16_gemm_t_m128_v2` kernel symbol. Falls back
/// to v1 silently when either is missing. Default off until
/// A/B-validated. Cached via `OnceLock`.
pub fn prefill_ffn_m128_v2_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_FFN_M128_V2").ok().as_deref() == Some("1"))
}

/// Returns true when `ATLAS_FFN_M16_TRANSPOSED=1` is set in the process env.
///
/// Gates the dense FFN K=γ verify path's routing through the transposed
/// (`nvfp4_t`) weight layout + `w4a16_gemm_n128_m16` (M_TILE=16) kernel.
/// When on, `Qwen35DenseWeightLoader` calls `transpose_for_gemm` on each
/// FFN projection (gate/up/down) at model-load time and installs the
/// transposed copies onto `DenseFfnLayer` via `set_transposed_weights`;
/// `forward_kgamma` then dispatches to `w4a16_gemm_n128_m16` instead of
/// the M_TILE=64 `w4a16_gemm` fallback. Memory cost: ~equivalent to the
/// original FFN weights (~150 MB on qwen3.6-27b — 64 layers × 3 FFN
/// projections × ~780 KB each). One-time host-side transpose cost at
/// load (~89 MB H↔D per projection). When this gate is off OR the
/// `w4a16_gemm_t_m16` kernel symbol is missing, the existing M_TILE=64
/// path stays the baseline. Cached via `OnceLock`.
pub fn ffn_m16_transposed_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // Default ON (opt-out via ATLAS_FFN_M16_TRANSPOSED=0). The transposed
    // NVFP4 FFN weights (~150 MB total) are the prerequisite for the
    // default-on `prefill_ffn_fast_enabled` M_TILE=128 prefill path AND the
    // M_TILE=16 verify path. Building them unconditionally removes the
    // foot-gun where the fast prefill gate is on but silently falls back to
    // the M_TILE=64 baseline because the transposed copies were never built.
    *GATE.get_or_init(|| std::env::var("ATLAS_FFN_M16_TRANSPOSED").ok().as_deref() != Some("0"))
}

/// Returns true when `ATLAS_DFLASH_FFN_KGAMMA=1` is set in the process env.
///
/// Gates the DFlash drafter's per-layer FFN (gate/up/down) routing through
/// the small-M `w4a16_gemm_n128_m16` (M_TILE=16) kernel instead of the
/// default `w4a16_gemm` (M_TILE=64). At DFlash γ=16 the drafter forward
/// runs all 17 (= γ+1) noise+bonus tokens through a single batched FFN per
/// layer (so M=17 in practice); the M_TILE=64 kernel discards 47/64 = 73%
/// of accumulator writes, while the M_TILE=16 specialization redesigns
/// warp partitioning so all 4 warps share the same 16 rows across N
/// sub-tiles. Profile showed 32μs gate_up + 17μs down_proj per drafter
/// layer × 5 layers = 49ms of FFN time, ~44% of the 113ms propose body.
/// Theoretical at 273 GB/s: ~5.5ms. The transposed-weight M=16 kernel
/// requires `nvfp4_t` layout; the drafter's `quantize_to_nvfp4` produces
/// the standard HuggingFace `[N, K/2]` layout, so when this gate is on
/// the loader runs an additional `transpose_for_gemm` per FFN projection
/// (one-time at model build, ~1.3 GB H↔D round-trip across 5 layers × 3
/// projections — measured separately). Default off so the M_TILE=64 path
/// stays the baseline until A/B-validated. Cached via `OnceLock`.
pub fn dflash_ffn_kgamma_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_DFLASH_FFN_KGAMMA").ok().as_deref() == Some("1"))
}

/// Returns true when `ATLAS_DFLASH_ATTN_KGAMMA=1` is set in the process env.
///
/// Gates the DFlash drafter's per-layer attention projections (q/k/v/o)
/// routing through the small-M `w4a16_gemm_n128_m16` (M_TILE=16) kernel
/// instead of the default `w4a16_gemm` (M_TILE=64). At DFlash γ=16 the
/// drafter forwards `n_attn = ctx_window + γ+1` rows through Q (and `noise_count
/// = γ+1` through K/V noise), but the M_TILE=64 kernel discards 47/64 = 73%
/// of accumulator writes when M=17 (verify rows). Drafter q_proj observed
/// ~5.5ms × 5 layers = ~27ms; o_proj ~4-6ms × 5 layers = ~20-30ms. Targets
/// the same M_TILE=16 specialization the FFN kgamma unlock uses, but
/// applied to attention projections.
///
/// Like the FFN gate, requires the transposed `nvfp4_t` weight layout
/// (~one-time `transpose_for_gemm` per projection at model build). Default
/// off so the M_TILE=64 path stays baseline until A/B-validated.
/// Cached via `OnceLock`.
pub fn dflash_attn_kgamma_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_DFLASH_ATTN_KGAMMA").ok().as_deref() == Some("1"))
}

/// Returns true when `ATLAS_MEASURE_FFN_SPARSITY=1` is set in the process env.
///
/// Gates the TEAL-style FFN activation-sparsity MEASUREMENT harness — the
/// go/no-go feasibility gate for sparsity-drafted self-speculation. When on,
/// `DenseFfnLayer::forward` runs `ffn_sparsity_measure` at two sites per
/// layer:
///   1. on `input` (gate/up in, K=hidden=5120) before the dual GEMV
///   2. on `gate_out` (down in, K=intermediate=17408) after silu_mul
///
/// accumulating per-site below-threshold histograms at {0.5,1,2,5}%×rowmax.
/// A periodic D2H dump (see `DenseFfnLayer::maybe_dump_sparsity`) prints the
/// averaged fractions so the operator can read the down_proj-input sparsity —
/// the single number that decides whether the whole flagship is worth building
/// (>=40% at <=1% threshold → BUILD; <25% → KILL).
///
/// This is a PURE OBSERVER: the measure kernel only READS `input`/`gate_out`
/// and writes into dedicated per-layer counter buffers — it never mutates the
/// token-producing path. Enabling the gate therefore keeps the greedy-counting
/// md5 at the 91a6ff90 constitution (the token stream is byte-identical whether
/// the gate is on or off; only extra observer kernels are launched).
/// Default off. Cached via `OnceLock`.
pub fn measure_ffn_sparsity_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_MEASURE_FFN_SPARSITY").ok().as_deref() == Some("1"))
}

/// Number of decode-token FFN forwards between periodic sparsity-histogram
/// dumps (`ATLAS_MEASURE_FFN_SPARSITY_EVERY`, default 512). The dump is
/// per-site, averaged over all rows seen since process start, and printed at
/// `tracing::info`. Returns the parsed value clamped to `>= 1`.
pub fn measure_ffn_sparsity_dump_every() -> u64 {
    static GATE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_MEASURE_FFN_SPARSITY_EVERY")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(|v| v.max(1))
            .unwrap_or(512)
    })
}

/// Returns true when `ATLAS_SELF_SPEC_SPARSE=1` is set in the process env.
///
/// Selects the SPARSE self-speculative DRAFT path (`decode_draft_sparse`) in
/// `step_self_spec` instead of the default dense layer-skip `decode_draft`.
/// The sparse draft reuses the existing self-spec layer-skip shape (SSM layers
/// skipped, so rewind stays a trivial truncate — NO new SSM checkpoint/rollback)
/// but swaps the FFN's down_proj (and gate/up) GEMV for the column-sparse path
/// (`ffn_build_keep_chunks` → `w4a16_gemv_sparse_cols`), reading fewer weight
/// bytes. The draft need NOT be bit-exact — the dense verify (`decode_verify`,
/// untouched) is the lossless oracle. When OFF, `step_self_spec` calls the
/// ORIGINAL `decode_draft` byte-for-byte. Default off. Cached via `OnceLock`.
pub fn self_spec_sparse_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_SELF_SPEC_SPARSE").ok().as_deref() == Some("1"))
}

/// Threshold (fraction of per-row max-abs) for the sparse self-spec draft's
/// keep-chunk selection (`ATLAS_SELF_SPEC_SPARSE_THRESH`, as a PERCENT — e.g.
/// `1.0` means 1% of rowmax). A k8 chunk survives iff any of its 8 activations
/// is `>= (percent/100) * rowmax`. Higher percent → more columns skipped →
/// cheaper draft but lower acceptance. Returns the parsed percent as a raw
/// fraction (percent/100), defaulting to 0.01 (1%). Clamped to (0, 1].
pub fn self_spec_sparse_thresh() -> f32 {
    static GATE: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        let pct = std::env::var("ATLAS_SELF_SPEC_SPARSE_THRESH")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(1.0);
        parse_sparse_thresh_pct(pct)
    })
}

/// Pure conversion + clamp for `ATLAS_SELF_SPEC_SPARSE_THRESH` (a PERCENT) into
/// a keep-threshold FRACTION of per-row max-abs. Extracted from
/// `self_spec_sparse_thresh` so it is unit-testable without touching process
/// env (the OnceLock caches on first call). Keeps the result in `(0, 1]`:
/// `>= 1` would skip everything but the rowmax chunk (degenerate); `<= 0`
/// would keep all chunks (== dense). Clamped to a sane draft window.
#[inline]
pub fn parse_sparse_thresh_pct(pct: f32) -> f32 {
    let frac = pct / 100.0;
    if !frac.is_finite() {
        return 0.01; // default 1% for NaN/inf input
    }
    frac.clamp(1e-4, 1.0)
}

/// Returns true when `ATLAS_FLASH_ATTN_KGAMMA=1` is set in the process env.
///
/// Gates the FlashAttention-v2 inspired Q-tile fused paged-decode kernel
/// (`paged_decode_attn_kgamma_nvfp4`) for the K=γ verify attention path.
/// At DFlash γ=16 the legacy kernel launches `num_q_heads × (γ+1)=68 CTAs`
/// each independently scanning the full KV history (17× redundant HBM
/// traffic per layer). The fused kernel collapses the QTILE axis into a
/// single CTA per q_head: 8 warps each own a slice of queries, K and V
/// vectors are loaded ONCE and reused across owned queries — direct
/// 2-3× per-warp HBM reduction with L1/L2 reuse compounding across warps.
///
/// Mirrors AEON-7's vLLM FLASH_ATTN backend strategy of fusing the Q-tile
/// against a shared KV history at the kernel level.
///
/// Active only when:
///   - `ATLAS_FLASH_ATTN_KGAMMA=1`
///   - `num_seqs >= 8` (K=γ verify shape)
///   - NVFP4 KV cache
///   - No tree-aware indirection active (legacy kernel handles tree mode)
///   - `head_dim == 256` (kernel compiled with HDIM=256)
///
/// Default off until proven; falls back to the legacy per-query path.
/// Cached via `OnceLock`.
pub fn flash_attn_kgamma_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_FLASH_ATTN_KGAMMA").ok().as_deref() == Some("1"))
}

/// Quarantined FP8 Kgamma M=15 experiment.
///
/// This route is intentionally unreachable from the former public experiment
/// gate. The replacement preserves the legacy reduction topology statically,
/// but remains unverified on device. A deliberately alarming developer-only
/// name permits captured-context diagnosis without accidental promotion.
pub fn flash_attn_kgamma_fp8_bc4_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_UNSAFE_UNVERIFIED_FP8_KGAMMA_EXACT")
            .ok()
            .as_deref()
            == Some("1")
    })
}

/// BF16-KV sibling of the flat K-gamma fused attention route. Kept on an
/// independent gate so mixed-KV models can qualify BF16 and compressed layers
/// separately and a missing PTX symbol always falls back to the proven path.
pub fn flash_attn_kgamma_bf16_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_UNSAFE_UNVERIFIED_BF16_KGAMMA_EXACT")
            .ok()
            .as_deref()
            == Some("1")
    })
}

/// Returns true when `ATLAS_KGAMMA_VECDEQUANT=1` is set in the process env.
///
/// Gates the VEC variant of the K=γ paged-decode kernel
/// (`paged_decode_attn_kgamma_nvfp4_vec`) which compounds two optimizations
/// over the baseline single-CTA kgamma kernel:
///
///   1. NUM_WARPS bumped 8 → 16 so each warp owns at most 2 queries
///      (γ=16 → QTILE=17 → my_count ≤ 2). Kills the per-pos divergent
///      QPER_WARP slot loop dominating inner-loop instruction throughput.
///   2. 2-position dequant batching in the inner KV scan: 4 NVFP4 dequants
///      (K0, V0, K1, V1) are issued together so the compiler can interleave
///      loads and ALU. Same total HBM bytes, fewer LD-stall bubbles.
///
/// Per kernel-author diagnosis on aeon-27b counting (γ=16): NVFP4 dequant
/// inside the inner K/V scan was the bottleneck. This path widens the
/// instruction-level parallelism without changing the algorithm.
///
/// Active only when:
///   - `ATLAS_KGAMMA_VECDEQUANT=1`
///   - `ATLAS_FLASH_ATTN_KGAMMA=1` (the gate that activates the kgamma path)
///   - `ATLAS_FLASH_ATTN_KGAMMA_SPLITK` is NOT effective (splitk path keeps
///     using its own kernel for now — vec optimization can be ported later)
///   - vec kernel resolved at init
///
/// Default off until proven. Cached via `OnceLock`.
pub fn flash_attn_kgamma_vecdequant_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_KGAMMA_VECDEQUANT").ok().as_deref() == Some("1"))
}

/// Returns true when `ATLAS_FLASH_ATTN_KGAMMA_SPLITK=1` is set in the
/// process env. Gates the split-K variant of the kgamma kernel that
/// partitions the KV history across `num_splits` CTAs per q_head to
/// reclaim SM occupancy on long-context decode (4 CTAs → 48 CTAs on
/// a 48-SM GB10). Only consulted when `flash_attn_kgamma_enabled()`
/// is also true; default off until proven. Cached via `OnceLock`.
pub fn flash_attn_kgamma_splitk_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_FLASH_ATTN_KGAMMA_SPLITK")
            .ok()
            .as_deref()
            == Some("1")
    })
}

/// Returns true when `ATLAS_FA2_KGAMMA=1` is set in the process env.
///
/// Gates the FA2-grafted variant of the K=γ paged-decode kernel
/// (`paged_decode_attn_kgamma_nvfp4_fa2`). This is the partial port of
/// Dao-AILab FlashAttention-v2's inner-loop technique to Atlas's PTX-only
/// model:
///
///   1. K/V tiled into FA2_TILE_N=32 positions, loaded into shared memory
///      via `cp.async.cg.shared.global` (vectorized 16 B per thread).
///   2. Double-buffered staging (FA2_STAGES=2): issue cp.async loads for
///      tile N+1 while computing tile N, with `cp.async.wait_group 1`
///      barriers — same shape as FA2's `compute_attn_1rowblock` loop.
///   3. Dequant + dot product execute against SMEM, freeing the LSU to
///      keep HBM saturated while the math units run.
///
/// Same caller contract as `paged_decode_attn_kgamma_nvfp4`:
///   - num_seqs == 1 (one real sequence), num_qtile = γ+1 ≤ QTILE_MAX
///   - NVFP4 KV cache, HDIM=256
///   - kv_indirection MUST be nullptr (tree-aware path uses legacy)
///   - All `num_qtile` rows of `block_tables` identical (K=γ verify)
///
/// Active only when:
///   - `ATLAS_FA2_KGAMMA=1`
///   - `ATLAS_FLASH_ATTN_KGAMMA=1` (the gate that activates the kgamma path)
///   - fa2 kernel resolved at init (`paged_decode_kgamma_fa2_k`)
///
/// Takes precedence over the VEC variant when both are enabled.
/// Default off until proven. Cached via `OnceLock`.
pub fn flash_attn_kgamma_fa2_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_FA2_KGAMMA").ok().as_deref() == Some("1"))
}

/// FFN component: MoE (expert routing), dense SwiGLU, or None (standalone attention).
#[allow(clippy::large_enum_variant)]
pub enum FfnComponent {
    Moe(MoeLayer),
    Dense(DenseFfnLayer),
    /// No FFN — used by Nemotron-H standalone attention layers.
    None,
}

impl FfnComponent {
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    pub fn forward(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        match self {
            Self::Moe(m) => m.forward(input, ctx, stream),
            // Sparsity-drafted self-spec DRAFT path: when the context requests
            // it (`decode_draft_sparse` sets `self_spec_sparse_draft=Some(t)`),
            // route the dense FFN through the column-sparse draft GEMV. Every
            // other caller leaves the field `None` → the exact dense `forward`
            // runs, byte-for-byte as before. `forward_draft_sparse` itself
            // falls back to `forward` when the sparse kernels are missing.
            Self::Dense(d) => match ctx.self_spec_sparse_draft {
                Some(thresh) => d.forward_draft_sparse(input, ctx, thresh, stream),
                None => d.forward(input, ctx, stream),
            },
            Self::None => Ok(input),
        }
    }

    pub fn forward_k2(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_k2(input, ctx, stream),
            Self::Dense(d) => d.forward_k2(input, ctx, stream),
            Self::None => Ok(()),
        }
    }

    pub fn forward_k3(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_k3(input, ctx, stream),
            Self::Dense(d) => d.forward_k3(input, ctx, stream),
            Self::None => Ok(()),
        }
    }

    /// K=γ verify batched FFN. Returns `true` when the call was serviced
    /// by the batched path (output in `ctx.buffers.moe_output()`), `false`
    /// when the caller must fall back to the per-token `forward()` loop.
    ///
    /// Only implemented for `Dense` today — MoE / `None` always return
    /// `false`. Caller is expected to check `ffn_kgamma_m16_enabled()`
    /// before invoking so the gate stays at one place per call site.
    pub fn forward_kgamma(
        &self,
        input: DevicePtr,
        n: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        match self {
            Self::Dense(d) => {
                d.forward_kgamma(input, ctx, n, stream)?;
                Ok(true)
            }
            Self::Moe(_) | Self::None => Ok(false),
        }
    }

    /// True when the dense exact-W4 dispatcher owns this row count even if
    /// the legacy `ATLAS_FFN_KGAMMA_M16` optimization gate is unset.
    pub fn exact_kgamma_applicable(&self, rows: u32) -> bool {
        match self {
            Self::Dense(d) => d.exact_kgamma_applicable(rows),
            Self::Moe(_) | Self::None => false,
        }
    }

    pub fn forward_prefill(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_prefill(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_prefill(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }

    pub fn forward_batched(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_batched(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_batched(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod sparse_thresh_tests {
    use super::parse_sparse_thresh_pct;

    #[test]
    fn default_one_percent() {
        // 1.0 percent → 0.01 fraction.
        assert!((parse_sparse_thresh_pct(1.0) - 0.01).abs() < 1e-9);
    }

    #[test]
    fn half_percent() {
        assert!((parse_sparse_thresh_pct(0.5) - 0.005).abs() < 1e-9);
    }

    #[test]
    fn two_percent() {
        assert!((parse_sparse_thresh_pct(2.0) - 0.02).abs() < 1e-9);
    }

    #[test]
    fn clamps_below_floor() {
        // A tiny percent floors at 1e-4 (keeps the draft from becoming dense).
        assert!((parse_sparse_thresh_pct(0.0) - 1e-4).abs() < 1e-9);
        assert!((parse_sparse_thresh_pct(-5.0) - 1e-4).abs() < 1e-9);
    }

    #[test]
    fn clamps_above_ceiling() {
        // >= 100% clamps to 1.0 (degenerate but bounded).
        assert!((parse_sparse_thresh_pct(150.0) - 1.0).abs() < 1e-9);
        assert!((parse_sparse_thresh_pct(100.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn non_finite_falls_back_to_default() {
        assert!((parse_sparse_thresh_pct(f32::NAN) - 0.01).abs() < 1e-9);
        assert!((parse_sparse_thresh_pct(f32::INFINITY) - 0.01).abs() < 1e-9);
    }
}
