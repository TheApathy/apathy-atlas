// SPDX-License-Identifier: AGPL-3.0-only

//! Split-K dispatch for the DFlash drafter's own NVFP4 projections.
//!
//! # Why the drafter needs this
//!
//! Under the production flag set (`ATLAS_DFLASH_NOISE_ONLY=1` +
//! `ATLAS_DFLASH_ATTN_KGAMMA=1` + `ATLAS_DFLASH_FFN_KGAMMA=1`) every drafter
//! projection runs at `M = γ+1 = 13` rows through `w4a16_gemm_t_m32_n64`,
//! whose grid is `(ceil(N/64), ceil(M/32), 1)`. At M ≤ 32 that is exactly
//! `ceil(N/64)` CTAs — the N dimension alone decides occupancy, and the whole
//! machine has to be filled from that one dimension. (An earlier draft of this
//! comment asserted "110 SMs"; nothing in the tree had ever queried the
//! device, so treat that as unverified — `GpuBackend::sm_count()` now reports
//! the real number, and the threshold below is anchored on measurement rather
//! than on any SM literal.)
//!
//! Measured achieved bandwidth per drafter kernel (6 layers, γ_eff=12, from
//! `qwen38/benchmark/results/kprof-raw.txt`), against the CTA count its N
//! implies:
//!
//! | kernel      |    N  | CTAs | weight bytes | measured | achieved  |
//! |-------------|-------|------|--------------|----------|-----------|
//! | kv_noise    |  1024 |   16 |    35.4 MB   | 1.16 ms  |  30 GB/s  |
//! | q_proj      |  4096 |   64 |    70.8 MB   | 0.93 ms  |  76 GB/s  |
//! | o_proj      |  5120 |   80 |    70.8 MB   | 0.78 ms  |  91 GB/s  |
//! | down_proj   |  5120 |   80 |   301.0 MB   | 3.10 ms  |  97 GB/s  |
//! | gate_up     | 17408 |  272 |   602.0 MB   | 3.10 ms  | 194 GB/s  |
//!
//! Achieved bandwidth is a monotone function of CTA count and saturates only
//! once the grid exceeds the SM count by ~2.5×. `gate_up` — the one drafter
//! GEMM with a wide N — already runs at 84% of the 232 GB/s achievable floor
//! and has nothing left to give. Every other drafter GEMM is occupancy-bound,
//! not bandwidth-bound, and that is the whole of the propose path's gap to its
//! memory-traffic floor.
//!
//! This is the identical diagnosis (and identical remedy) already applied on
//! the target side: see the `w4a16_gemm_n64_m32_splitk` doc comment in
//! `layers/ops/gemm_dense.rs`, which records the K=17 verify `down_proj`
//! running at "~91 GB/s vs gate/up ~163 GB/s on the same-size weight" and
//! fixes it by multiplying the CTA count through `gridDim.z`. Production
//! already sets `ATLAS_FFN_DOWN_SPLITK`, `ATLAS_FLASH_ATTN_KGAMMA_SPLITK` and
//! `ATLAS_PAGED_DECODE_SPLITK`; the drafter's own layers are the last consumer
//! of the m32_n64 kernel that was never given the same treatment.
//!
//! # Numerical contract
//!
//! Split-K is *reassociated*, not bit-identical: the K-loop is sliced across
//! `gridDim.z`, each slice accumulates in FP32 into a workspace band, and
//! `reduce_splitk_f32_to_bf16` sums the bands. Every partial is FP32 (no
//! precision is dropped) but the summation ORDER changes, so results can
//! differ in the last ULP of the BF16 result.
//!
//! For the drafter this cannot corrupt output: speculation is verified by a
//! strict argmax match against the target, so a drafter logit that moves by an
//! ULP can only change WHICH token is proposed, never which is committed. The
//! honest risk is therefore acceptance, not correctness — hence the default
//! OFF gate below and the A/B requirement.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops;
use crate::weight_map::QuantizedWeight;

use super::BlockDiffusionDraftHead;

/// `ATLAS_DFLASH_DRAFT_SPLITK` — number of K-slices for the drafter's own
/// occupancy-starved projections. `0` (the default) disables split-K entirely
/// and every dispatch below is byte-for-byte the pre-existing kernel choice.
/// Values `1` and below are treated as off; the kernel's workspace is sized
/// for at most 8 slices, matching `ffn_down_splitk`'s clamp.
pub(super) fn draft_splitk() -> u32 {
    static GATE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        let v = std::env::var("ATLAS_DFLASH_DRAFT_SPLITK")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .map(|v| if v < 2 { 0 } else { v.min(8) })
            .unwrap_or(0);
        if v > 0 {
            tracing::info!(
                "DFlash drafter split-K ENABLED (ATLAS_DFLASH_DRAFT_SPLITK={v}): \
                 q/kv/o/down projections slice K across gridDim.z. Reassociated \
                 FP32 partials — drafts may shift, commits cannot (target argmax \
                 verifies)."
            );
        }
        v
    })
}

/// Largest CTA count below which a grid is considered occupancy-starved.
///
/// MEASUREMENT-ANCHORED, NOT SM-DERIVED. This is the one number in this file
/// that decides slice counts, and it comes from the table above, not from an
/// SM literal: `gate_up` at 272 CTAs measures 194 GB/s (84% of achievable) and
/// has nothing left to give, while `o_proj`/`down_proj` at 80 CTAs manage only
/// 91-97 GB/s. 256 is the round number that sits just under the one shape
/// measured saturated. The older "~2.5× GB10's 110 SMs" gloss was a post-hoc
/// rationalization and does not reproduce this value — 2.5×110 = 275 would
/// also split `gate_up`, which the same table says is pointless.
///
/// So changing the assumed SM count does NOT change any slice count here. Use
/// [`log_sm_count_assumption`] to surface what the device actually reports;
/// re-deriving this constant from that number would regress the sizing,
/// because the measured saturation point of this kernel is ~5.7 CTAs/SM on a
/// 48-SM part, not the 2.5 the old comment assumed.
const STARVED_CTA_LIMIT: u32 = 256;

/// Report the device's real SM count against the assumption baked into
/// [`STARVED_CTA_LIMIT`], once per process.
///
/// Purely observational — it changes no dispatch. It exists because nothing in
/// this tree had ever asked the device how many SMs it has, so every occupancy
/// comment was an unverified literal (48 in `layers/mod.rs`, 110 here). Now the
/// number shows up in the log next to the threshold it is supposed to justify.
fn log_sm_count_assumption(gpu: &dyn GpuBackend) {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| match gpu.sm_count() {
        Some(sms) => {
            let ctas_per_sm = f64::from(STARVED_CTA_LIMIT) / f64::from(sms);
            tracing::info!(
                "drafter split-K: device reports {sms} SMs; STARVED_CTA_LIMIT={STARVED_CTA_LIMIT} \
                 = {ctas_per_sm:.1} CTAs/SM. This threshold is measured, not derived from the SM \
                 count — see its doc comment before retuning it."
            );
        }
        None => tracing::info!(
            "drafter split-K: backend cannot report an SM count; \
             STARVED_CTA_LIMIT={STARVED_CTA_LIMIT} stands on its measured anchor."
        ),
    });
}

/// CTAs the single-slice m32_n64 kernel would field for output width `n`.
fn cta_count(n: u32) -> u32 {
    n.div_ceil(64)
}

/// Slices needed to lift `n`'s grid past [`STARVED_CTA_LIMIT`], clamped to the
/// caller's configured budget. Returns 0 when the shape is already saturated.
fn slices_for(n: u32, budget: u32) -> u32 {
    let ctas = cta_count(n);
    if ctas >= STARVED_CTA_LIMIT {
        return 0;
    }
    STARVED_CTA_LIMIT.div_ceil(ctas.max(1)).min(budget)
}

/// The kernel choice for one drafter projection, as a value.
///
/// Extracted from `draft_gemm_t` so the choice is testable WITHOUT a GPU or a
/// constructed drafter head. The regression this guards: a refactor that
/// funnels nine call sites through one helper must, with the gate off, pick
/// exactly the kernel each site picked before. See `plan_matches_legacy_*`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DispatchPlan {
    /// `ops::w4a16_gemm_n64_m32` with `kernels.w4a16_gemm_t_m32_n64`.
    M32,
    /// `ops::w4a16_gemm_n128_m16` with `kernels.w4a16_gemm_t_m16`.
    M16,
    /// `ops::w4a16_gemm_n64_m32_splitk` with `k_splits` K-slices.
    SplitK { splits: u32 },
}

/// Everything `plan_dispatch` needs that would otherwise require a live head.
#[derive(Debug, Clone, Copy)]
pub(super) struct DispatchCaps {
    pub has_m32: bool,
    pub has_splitk: bool,
    pub has_reduce: bool,
    pub has_workspace: bool,
    /// Output width the split-K workspace was sized for.
    pub workspace_n: u32,
    /// `ATLAS_DFLASH_DRAFT_SPLITK`; 0 disables split-K entirely.
    pub budget: u32,
}

/// The pre-refactor dispatch, reproduced exactly: each of the nine NVFP4 call
/// sites used `if <site guard> && m32_handle != 0 { m32 } else { m16 }`.
/// Kept as its own function so the equivalence test compares against a written
/// specification rather than against the implementation under test.
pub(super) fn legacy_plan(has_m32: bool, allow_m32: bool) -> DispatchPlan {
    if allow_m32 && has_m32 {
        DispatchPlan::M32
    } else {
        DispatchPlan::M16
    }
}

/// Choose the kernel for `[m, k] x [k, n]`.
///
/// With `caps.budget == 0` this is required to equal [`legacy_plan`] for every
/// input — that is the "unset is byte-identical" contract, and it is asserted
/// exhaustively over the production shapes in this module's tests.
pub(super) fn plan_dispatch(caps: &DispatchCaps, m: u32, n: u32, allow_m32: bool) -> DispatchPlan {
    let legacy = legacy_plan(caps.has_m32, allow_m32);
    if legacy != DispatchPlan::M32 || m > 32 {
        // Split-K is a K-sliced variant of the m32 kernel and fields one
        // M-tile, so it is only reachable where m32 itself was chosen.
        return legacy;
    }
    if caps.budget < 2
        || !caps.has_splitk
        || !caps.has_reduce
        || !caps.has_workspace
        || n > caps.workspace_n
    {
        return legacy;
    }
    let splits = slices_for(n, caps.budget);
    if splits >= 2 {
        DispatchPlan::SplitK { splits }
    } else {
        legacy
    }
}

impl BlockDiffusionDraftHead {
    /// Capability snapshot for [`plan_dispatch`].
    fn dispatch_caps(&self) -> DispatchCaps {
        DispatchCaps {
            has_m32: self.kernels.w4a16_gemm_t_m32_n64.0 != 0,
            has_splitk: self.kernels.w4a16_gemm_t_m32_n64_splitk.0 != 0,
            has_reduce: self.kernels.reduce_splitk_k.0 != 0,
            has_workspace: self.scratch.splitk_ws != DevicePtr::NULL,
            // The workspace is sized for N = hidden_size, which bounds every
            // shape we split (q_dim 4096, kv_dim 1024, hidden 5120). gate_up's
            // N = intermediate is never split — `slices_for` returns 0 for it.
            workspace_n: self.hidden_size as u32,
            budget: draft_splitk(),
        }
    }

    /// Dispatch one drafter NVFP4 projection `[M, K] x [K, N] -> [M, N]`.
    ///
    /// `allow_m32` is the call site's pre-existing guard on whether the
    /// M_TILE=32 kernel is legal for this shape (the kv-context-new site
    /// requires `new_ctx_count <= 32`; the others pass `true`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draft_gemm_t(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        allow_m32: bool,
        stream: u64,
    ) -> Result<()> {
        // Observational only, `Once`-guarded: puts the device's real SM count
        // in the log next to the threshold it is supposed to justify.
        log_sm_count_assumption(gpu);
        match plan_dispatch(&self.dispatch_caps(), m, n, allow_m32) {
            DispatchPlan::SplitK { splits } => ops::w4a16_gemm_n64_m32_splitk(
                gpu,
                self.kernels.w4a16_gemm_t_m32_n64_splitk,
                self.kernels.reduce_splitk_k,
                input,
                weight,
                output,
                self.scratch.splitk_ws,
                m,
                n,
                k,
                n, // ldb == N for tightly-packed T-weights
                splits,
                stream,
            ),
            DispatchPlan::M32 => ops::w4a16_gemm_n64_m32(
                gpu,
                self.kernels.w4a16_gemm_t_m32_n64,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            ),
            DispatchPlan::M16 => ops::w4a16_gemm_n128_m16(
                gpu,
                self.kernels.w4a16_gemm_t_m16,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DispatchCaps, DispatchPlan, STARVED_CTA_LIMIT, cta_count, legacy_plan, plan_dispatch,
        slices_for,
    };

    const HIDDEN: u32 = 5120;
    const Q_DIM: u32 = 4096;
    const KV_DIM: u32 = 1024;
    const INTER: u32 = 17408;
    /// gamma+1 noise rows under the production GAMMA.
    const M_NOISE: u32 = 13;

    /// The nine NVFP4 GEMM call sites in `forward_block_layer_nvfp4`, as
    /// (label, m, n, k, allow_m32) — transcribed from the call sites.
    fn production_sites() -> Vec<(&'static str, u32, u32, u32, bool)> {
        let mut v = vec![
            ("q_proj", M_NOISE, Q_DIM, HIDDEN, true),
            ("kv_noise_k", M_NOISE, KV_DIM, HIDDEN, true),
            ("kv_noise_v", M_NOISE, KV_DIM, HIDDEN, true),
            ("o_proj", M_NOISE, HIDDEN, Q_DIM, true),
            ("gate", M_NOISE, INTER, HIDDEN, true),
            ("up", M_NOISE, INTER, HIDDEN, true),
            ("down_proj", M_NOISE, HIDDEN, INTER, true),
        ];
        // kv_ctx_new runs at `new_ctx_count`, which varies per step and CAN
        // exceed 32 (first propose after a prefill) — hence its site guard.
        for ctx in [1u32, 8, 13, 32, 33, 64, 512] {
            v.push(("kv_ctx_new_k", ctx, KV_DIM, HIDDEN, ctx <= 32));
            v.push(("kv_ctx_new_v", ctx, KV_DIM, HIDDEN, ctx <= 32));
        }
        v
    }

    fn caps(budget: u32, has_m32: bool) -> DispatchCaps {
        DispatchCaps {
            has_m32,
            has_splitk: true,
            has_reduce: true,
            has_workspace: true,
            workspace_n: HIDDEN,
            budget,
        }
    }

    /// THE regression guard: gate unset must reproduce the pre-refactor
    /// ternary at every call site, with and without the m32 kernel present.
    #[test]
    fn gate_off_is_identical_to_the_pre_refactor_ternary() {
        for has_m32 in [true, false] {
            let c = caps(0, has_m32);
            for (label, m, n, k, allow) in production_sites() {
                let got = plan_dispatch(&c, m, n, allow);
                let want = legacy_plan(has_m32, allow);
                assert_eq!(
                    got, want,
                    "{label} [m={m} n={n} k={k} allow_m32={allow} has_m32={has_m32}]                      diverged from the legacy dispatch with the gate OFF"
                );
            }
        }
    }

    /// Missing kernels, a missing workspace, or an oversized N must all fall
    /// back to the legacy choice even with the gate on — a null handle or an
    /// unallocated workspace must never reach a launch.
    #[test]
    fn gate_on_still_falls_back_when_prerequisites_are_missing() {
        let sites = production_sites();
        let degradations: [(&str, DispatchCaps); 4] = [
            (
                "no splitk kernel",
                DispatchCaps {
                    has_splitk: false,
                    ..caps(8, true)
                },
            ),
            (
                "no reduce kernel",
                DispatchCaps {
                    has_reduce: false,
                    ..caps(8, true)
                },
            ),
            (
                "no workspace",
                DispatchCaps {
                    has_workspace: false,
                    ..caps(8, true)
                },
            ),
            (
                "workspace too small",
                DispatchCaps {
                    workspace_n: 0,
                    ..caps(8, true)
                },
            ),
        ];
        for (why, c) in degradations {
            for (label, m, n, _k, allow) in &sites {
                assert_eq!(
                    plan_dispatch(&c, *m, *n, *allow),
                    legacy_plan(c.has_m32, *allow),
                    "{label} did not fall back to the legacy dispatch when {why}"
                );
            }
        }
    }

    /// With the gate on, only the occupancy-starved shapes are split, only
    /// where the m32 kernel was already legal, and never above one M-tile.
    #[test]
    fn gate_on_splits_exactly_the_starved_shapes() {
        let c = caps(8, true);
        // gate/up (N=intermediate) is already saturated — never split.
        assert_eq!(plan_dispatch(&c, M_NOISE, INTER, true), DispatchPlan::M32);
        // The starved shapes are split.
        assert_eq!(
            plan_dispatch(&c, M_NOISE, KV_DIM, true),
            DispatchPlan::SplitK { splits: 8 }
        );
        assert_eq!(
            plan_dispatch(&c, M_NOISE, Q_DIM, true),
            DispatchPlan::SplitK { splits: 4 }
        );
        assert_eq!(
            plan_dispatch(&c, M_NOISE, HIDDEN, true),
            DispatchPlan::SplitK { splits: 4 }
        );
        // M above one M-tile, and sites whose guard forbids m32, stay legacy.
        assert_eq!(plan_dispatch(&c, 33, KV_DIM, false), DispatchPlan::M16);
        assert_eq!(plan_dispatch(&c, 64, KV_DIM, true), DispatchPlan::M32);
    }

    #[test]
    fn only_the_wide_n_shape_is_considered_saturated() {
        assert_eq!(cta_count(KV_DIM), 16);
        assert_eq!(cta_count(Q_DIM), 64);
        assert_eq!(cta_count(HIDDEN), 80);
        assert_eq!(cta_count(INTER), 272);
        assert!(cta_count(INTER) >= STARVED_CTA_LIMIT);
    }

    #[test]
    fn saturated_shapes_are_never_split() {
        assert_eq!(slices_for(INTER, 8), 0);
    }

    #[test]
    fn starved_shapes_are_lifted_toward_the_limit_within_budget() {
        assert_eq!(slices_for(KV_DIM, 8), 8);
        assert_eq!(slices_for(KV_DIM, 4), 4);
        assert_eq!(slices_for(Q_DIM, 8), 4);
        assert_eq!(slices_for(HIDDEN, 8), 4);
    }

    #[test]
    fn a_budget_below_two_yields_no_usable_split() {
        assert!(slices_for(HIDDEN, 1) < 2);
    }
}
