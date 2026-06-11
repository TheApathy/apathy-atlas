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
/// misses drowned out genuine problems in startup logs.
pub fn try_kernel(gpu: &dyn GpuBackend, module: &str, func: &str) -> KernelHandle {
    match gpu.kernel(module, func) {
        Ok(h) => h,
        Err(_) => {
            tracing::debug!("Optional kernel '{module}::{func}' not loaded");
            KernelHandle(0)
        }
    }
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
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_SSM_MULTI_SEQ_BATCHED").ok().as_deref() == Some("1")
    })
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
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_SSM_MULTI_SEQ_KERNEL").ok().as_deref() == Some("1")
    })
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
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_SSM_MULTI_SEQ_GRAPH").ok().as_deref() == Some("1")
    })
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
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_MTP_K3_BATCH_CSEQ").ok().as_deref() == Some("1")
    })
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
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_MTP_K2_BATCH_CSEQ").ok().as_deref() == Some("1")
    })
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
    *GATE.get_or_init(|| std::env::var("ATLAS_PREFILL_FFN_FAST").ok().as_deref() == Some("1"))
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
    *GATE
        .get_or_init(|| std::env::var("ATLAS_FFN_PREDEQUANT_FP8").ok().as_deref() == Some("1"))
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
    *GATE.get_or_init(|| std::env::var("ATLAS_FFN_M16_TRANSPOSED").ok().as_deref() == Some("1"))
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
    *GATE
        .get_or_init(|| std::env::var("ATLAS_FLASH_ATTN_KGAMMA_SPLITK").ok().as_deref() == Some("1"))
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
            Self::Dense(d) => d.forward(input, ctx, stream),
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
