// SPDX-License-Identifier: AGPL-3.0-only

//! Exactly one genuine ordinary decode, after any staged transaction restored.
//! No sequence ownership is moved and no staged verifier logits are reused.

use super::{ActiveSeq, Model, Settings};
use anyhow::{Context, Result, ensure};

pub(super) fn ordinary(
    model: &dyn Model,
    a: &mut ActiveSeq,
    settings: &Settings<'_>,
) -> Result<()> {
    ensure!(
        !a.finished && a.terminal_error.is_none() && a.remaining > 0,
        "GLM ordinary replay cannot advance a terminal request"
    );
    let start = a.seq.seq_len;
    let anchor = a.last_token;
    let end = start
        .checked_add(1)
        .context("GLM ordinary position overflow")?;
    ensure!(
        a.seq.tokens.len() == start && a.seq.kv_valid_tokens == start,
        "GLM ordinary host prefix extents disagree"
    );
    let t0 = std::time::Instant::now();
    let logits = model
        .decode_batch(&[anchor], &mut [&mut a.seq], settings.stream)
        .context("GLM genuine ordinary decode failed")?;
    ensure!(
        a.seq.seq_len == end
            && a.seq.kv_valid_tokens == end
            && a.seq.tokens.len() == end
            && a.seq.tokens.last() == Some(&anchor),
        "GLM ordinary decode did not publish its consumed anchor"
    );
    crate::scheduler::decode_logits_step::process_decode_logits_slice(
        model,
        std::slice::from_mut(a),
        logits,
        t0,
        settings.logits.think_end_token,
        settings.logits.think_start_token,
        settings.code_fence_token,
        settings.logits.tool_call_start_token,
        settings.logits.tool_call_end_token,
        settings.adaptive_sampling,
    )
}
