// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1 cross-layer sparse index attention.
//!
//! `core::Dsv41SparseCore` is the forward's `AttnCore` + `PassHook`: compressor, indexer,
//! L20 replay tail and the one-pass sparse attention (kernels in
//! `kernels/gb10/deepseek-v4.1/cb3/dsv41_sparse_{attn,index}.cu`). Gated against
//! runF_faithful / runG_replay by `examples/dsv41_attn_seam.rs` (SPEC.md 8f-8g). A
//! `full_attention` fallback for this architecture is a DIFFERENT MODEL, not a slower one.

pub mod fp8_gemm;
pub mod index;
pub mod schedule;
pub mod core;

pub use schedule::{
    SparseSchedule, context_bucket, indexer_score_rows, indexer_topk_width, pad_topk,
};
