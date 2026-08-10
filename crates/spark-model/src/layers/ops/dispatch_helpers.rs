// SPDX-License-Identifier: AGPL-3.0-only

//! GEMM-path dispatch helpers + roofline instrumentation. Extracted from the
//! `ops` module root during the ≤500-line split. Re-exported at
//! `crate::layers::ops::*` via `ops.rs`.

#![allow(unused_imports)]

use super::*;

/// Whether block-scaled FP8 prefill (per-128-block weight scales + per-token
/// activation scales via `fp8_gemm_t_blockscaled` / `moe_w8a8_grouped_gemm`)
/// is enabled. This is the DEFAULT for block-scaled FP8 checkpoints as of
/// 2026-06-17: it matches vLLM's per-block precision and avoids the
/// single-scale `fp8_gemm_n128` path, whose collapse of per-block dynamic
/// range pushed long-context tool-arg decode into the FP8 argmax-flip regime
/// (B1 drift gauge ~1400 → ~100 once block-scaled prefill is on).
///
/// Opt out with `ATLAS_FP8_SINGLE_SCALE=1` to restore the old single-scale
/// prefill (diagnostic / fallback only). Call sites still guard on the
/// presence of block-scaled weights + kernel handles, so builds/models
/// without those fall back automatically regardless of this flag.
pub fn fp8_blockscaled_prefill_enabled() -> bool {
    !matches!(
        std::env::var("ATLAS_FP8_SINGLE_SCALE").ok().as_deref(),
        Some("1")
    )
}

/// Whether chunk-zero streams may use the paged batched-prefill path.
///
/// `ATLAS_PREFILL_CODISPATCH` is the end-to-end request-admission flag;
/// keep the older Q12 spelling as a compatibility alias for existing recipes.
pub fn prefill_batched_first_chunk_enabled() -> bool {
    ["ATLAS_Q12_BATCHED_FIRST_CHUNK", "ATLAS_PREFILL_CODISPATCH"]
        .iter()
        .map(|name| std::env::var(name).ok())
        .any(|value| bool_value_enabled(value.as_deref()))
}

/// VARLEN (ragged) batched prefill enabled? (`ATLAS_PREFILL_VARLEN=1`).
///
/// SSOT for both the admission predicate (`check_kernel_batched_eligible`) and
/// the batched-attention layer's chunk-0 guard. Those two must agree: if
/// admission accepts a batch the layer then rejects, the bail happens
/// mid-Phase-A with streams already mutated, and the per-stream fallback
/// re-runs setup on dirty state.
pub fn prefill_varlen_enabled() -> bool {
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| bool_value_enabled(std::env::var("ATLAS_PREFILL_VARLEN").ok().as_deref()))
}

/// cuBLASLt GEMM path enabled? (`ATLAS_CUBLAS_GEMM=1`), cached. The hand-written
/// mma.sync projection GEMMs hit only ~30% of the cuBLAS bf16 ceiling on GB10.
pub fn cublas_gemm_enabled() -> bool {
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| std::env::var("ATLAS_CUBLAS_GEMM").ok().as_deref() == Some("1"))
}

/// Route the BF16 *prefill* dense-FFN and head-gate GEMMs through cuBLASLt
/// instead of `dense_gemm_tc`? (`ATLAS_PREFILL_CUBLAS=1`), cached.
///
/// `dense_gemm_tc` runs the large dense-FFN shapes (Laguna layer-0 gate/up/down,
/// N=12288/3072 K=3072) at ~1.4 TFLOP/s against 90+ on cuBLASLt, and tiles the
/// head-gate's N=nq=72 output into a 2-block grid. Both are the same
/// BF16xBF16->FP32 GEMM q/k/v/o already send to `cublas_bf16_proj_dense`.
///
/// Kept as a separate gate from [`cublas_gemm_enabled`] so a single binary can
/// serve both arms of the A/B: production exports `ATLAS_CUBLAS_GEMM=1`
/// already, so folding this into that flag would make the control leg
/// unreachable without a second build.
///
/// NOT bit-exact — cuBLASLt and the mma.sync kernel differ in accumulation
/// order, so this needs an output-hash check on long prompts, not just tok/s.
pub fn prefill_cublas_dense_enabled() -> bool {
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| {
        cublas_gemm_enabled() && std::env::var("ATLAS_PREFILL_CUBLAS").ok().as_deref() == Some("1")
    })
}

/// Fused elementwise-swarm kernels enabled? (`ATLAS_FUSED_ELEMWISE=1`), cached.
/// Default OFF — gates the fused q/k norm+rope+cache-write epilogue of the
/// multi-seq flat verify AND the serial (M=1) decode path, plus the fused MoE
/// blend+residual tail (multi-seq KN verify and serial decode). All fused
/// kernels are bit-identical to the unfused chains they replace (see
/// kernels/gb10/common/fused_verify_elemwise.cu).
pub fn fused_elemwise_enabled() -> bool {
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| std::env::var("ATLAS_FUSED_ELEMWISE").ok().as_deref() == Some("1"))
}

/// Native-FP8 cuBLASLt GEMM path enabled? (`ATLAS_CUBLAS_FP8=1`), cached.
pub fn cublas_fp8_enabled() -> bool {
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| std::env::var("ATLAS_CUBLAS_FP8").ok().as_deref() == Some("1"))
}

/// CUTLASS GEMM path enabled? (`ATLAS_CUTLASS_GEMM=1`), cached. M0 is scoped to
/// dense BF16 projections using the same FP8→BF16 cached dequant as cuBLASLt.
pub fn cutlass_gemm_enabled() -> bool {
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| std::env::var("ATLAS_CUTLASS_GEMM").ok().as_deref() == Some("1"))
}

/// Serial (M=1) decode FP8-mirror GEMM dispatch enabled?
/// (`ATLAS_SERIAL_MIRROR_GEMM=1`), cached. Default OFF.
///
/// When on, the M=1 decode attention-mirror sites (q/k/v/o) and the serial
/// FP8 lm_head tier launch the proven M≤8 verify GEMMs
/// (`fp8_gemm_t_row_scaled_mtile8` / `_n32`) instead of `dense_gemv_fp8w`.
///
/// Cold M=1 microbench verdict (fp8gemv_m1_serial_microtest, GB10,
/// 2026-07-25): the GEMV is FASTER at every serial Laguna shape — 231-256
/// GB/s cold (at the ~245 GB/s LPDDR5x wall) vs 152-211 GB/s for the
/// mtile8/n32 tiles at M=1. This gate exists purely for serve-side
/// falsification (in-graph, real mirrors, real L2 state); expectation from
/// the microbench is that enabling it REGRESSES serial decode.
///
/// Numerics: mtile8 MMA accumulation differs from the GEMV's reduction
/// order (known knife-edge EAR class) — A/B accept-impact serve-side.
/// Graph-capture safe: pure device args, host-side constant decision.
pub fn serial_mirror_gemm_enabled() -> bool {
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| std::env::var("ATLAS_SERIAL_MIRROR_GEMM").ok().as_deref() == Some("1"))
}

/// Native CUTLASS NVFP4 GEMM path enabled? (`ATLAS_CUTLASS_NVFP4_GEMM=1`).
/// This path quantizes activations to CUTLASS NVFP4 and consumes transposed
/// Atlas NVFP4 weights after repacking scales into CUTLASS SM120 layout.
pub fn cutlass_nvfp4_gemm_enabled() -> bool {
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| std::env::var("ATLAS_CUTLASS_NVFP4_GEMM").ok().as_deref() == Some("1"))
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name).ok().as_deref() == Some("1")
}

fn bool_value_enabled(value: Option<&str>) -> bool {
    matches!(value, Some("1")) || value.is_some_and(|value| value.eq_ignore_ascii_case("true"))
}

/// Native CUTLASS NVFP4 SSM QKVZ path enabled.
pub fn cutlass_nvfp4_qkvz_enabled() -> bool {
    cutlass_nvfp4_gemm_enabled() || env_flag_enabled("ATLAS_CUTLASS_NVFP4_QKVZ")
}

/// Native CUTLASS NVFP4 attention Q/K/V path enabled for the named projection.
pub fn cutlass_nvfp4_attn_qkv_enabled(label: &str) -> bool {
    cutlass_nvfp4_gemm_enabled()
        || match label {
            "q_proj" => env_flag_enabled("ATLAS_CUTLASS_NVFP4_ATTN_Q"),
            "k_proj" | "v_proj" => env_flag_enabled("ATLAS_CUTLASS_NVFP4_ATTN_KV"),
            _ => false,
        }
}

/// Native CUTLASS NVFP4 attention O path enabled.
pub fn cutlass_nvfp4_attn_o_enabled() -> bool {
    cutlass_nvfp4_gemm_enabled() || env_flag_enabled("ATLAS_CUTLASS_NVFP4_ATTN_O")
}

/// Native CUTLASS NVFP4 SSM out-projection path enabled.
pub fn cutlass_nvfp4_ssm_out_enabled() -> bool {
    env_flag_enabled("ATLAS_CUTLASS_NVFP4_SSM_OUT")
}

pub fn log_cutlass_nvfp4_route(name: &str, m: u32, n: u32, k: u32) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u64, u32, u32, u32)>>> = OnceLock::new();
    let mut h: u64 = 1469598103934665603;
    for b in name.bytes() {
        h = (h ^ b as u64).wrapping_mul(1099511628211);
    }
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    if seen.lock().unwrap().insert((h, m, n, k)) {
        tracing::warn!("CUTLASS_NVFP4_ROUTE {name} M={m} N={n} K={k}");
    }
}

/// Roofline instrumentation: log each unique (kernel, M, N, K) GEMM shape once,
/// gated by `ATLAS_GEMM_SHAPE_LOG=1`. Used to cross-reference nsys per-call
/// durations → achieved TFLOPS/bandwidth vs GB10 peak.
#[track_caller]
pub fn log_gemm_shape(name: &str, m: u32, n: u32, k: u32) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    if std::env::var("ATLAS_GEMM_SHAPE_LOG").ok().as_deref() != Some("1") {
        return;
    }
    static SEEN: OnceLock<Mutex<HashSet<(u64, u32, u32, u32)>>> = OnceLock::new();
    let mut h: u64 = 1469598103934665603;
    for b in name.bytes() {
        h = (h ^ b as u64).wrapping_mul(1099511628211);
    }
    let key = (h, m, n, k);
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    if seen.lock().unwrap().insert(key) {
        let flop = 2.0 * m as f64 * n as f64 * k as f64;
        // #[track_caller] on this fn and the thin ops wrappers means `caller`
        // names the layer-code dispatch site — added after the decode audit
        // spent a search cycle hunting which layer owned two anonymous BF16
        // dense_gemv shapes.
        let loc = std::panic::Location::caller();
        tracing::warn!(
            "GEMM_SHAPE {name} M={m} N={n} K={k} FLOP={flop:.3e} at={}:{}",
            loc.file(),
            loc.line()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::bool_value_enabled;

    #[test]
    fn accepts_boolean_environment_spellings() {
        assert!(bool_value_enabled(Some("1")));
        assert!(bool_value_enabled(Some("true")));
        assert!(bool_value_enabled(Some("TRUE")));
        assert!(!bool_value_enabled(Some("0")));
        assert!(!bool_value_enabled(Some("false")));
        assert!(!bool_value_enabled(None));
    }

    #[test]
    fn accepts_the_codispatch_alias_for_chunk_zero() {
        let enabled = [None, Some("1")].into_iter().any(bool_value_enabled);
        assert!(enabled);
    }
}
