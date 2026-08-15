// SPDX-License-Identifier: AGPL-3.0-only

//! Metal-build stub of the cuda-only `cublaslt` module.
//!
//! The real [`crate::cublaslt`] module (cuBLASLt BF16 act·weightᵀ GEMMs) is
//! gated behind `feature = "cuda"` because it links cuBLASLt, which does not
//! exist on macOS. spark-model names these entry points unconditionally, so
//! the metal build (cuda off) needs the symbols to resolve even though the
//! cuBLASLt route never runs there. The bodies are `unreachable!` — reaching
//! one on metal is a bug (the `ATLAS_CUBLAS_GEMM` gate must stay off).

use anyhow::Result;

pub fn bf16_gemm_act_weight_t(
    _act: u64,
    _weight: u64,
    _out: u64,
    _m: u32,
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cublaslt::bf16_gemm_act_weight_t is cuda-only (not built for metal)")
}

pub fn bf16_gemm_act_weight_t_tuned(
    _act: u64,
    _weight: u64,
    _out: u64,
    _m: u32,
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cublaslt::bf16_gemm_act_weight_t_tuned is cuda-only (not built for metal)")
}
