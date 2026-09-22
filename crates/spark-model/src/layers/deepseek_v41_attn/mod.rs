// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1 cross-layer sparse index attention.
//!
//! Status: schedule arithmetic only. The kernel lives in
//! `kernels/gb10/deepseek-v4.1/attn/` and is not yet wired in; until it is,
//! nothing here may be used to claim the model runs. A `full_attention`
//! fallback for this architecture is a DIFFERENT MODEL producing wrong output,
//! not a slower one, and must hard-stop rather than ship silently.

pub mod index;
pub mod schedule;
pub mod seam;

pub use schedule::{
    SparseSchedule, context_bucket, indexer_score_rows, indexer_topk_width, pad_topk,
};
