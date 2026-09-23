// SPDX-License-Identifier: AGPL-3.0-only

//! Ordinary decode's policy-preserving GPU argmax admission.

use super::fast_greedy::{self, PenaltyGate};
use super::logit_processors::LogitsContext;
use super::sample_step::{PositionKind, penalty_history_scope, penalty_params_for};
use super::{ActiveSeq, Model};
use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

/// `None` requests the existing full host pipeline without having mutated any
/// sequence policy. `Some` carries only fully admitted raw GPU picks. A bad
/// backend result is an error, never permission to publish unchecked tokens.
pub(super) fn try_pick_batch(
    model: &dyn Model,
    active: &[ActiveSeq],
    logits: DevicePtr,
    ctx: &LogitsContext,
    adaptive_sampling: bool,
) -> Result<Option<Vec<u32>>> {
    if active.is_empty() {
        return Ok(Some(Vec::new()));
    }
    // Preserve the existing host-only dtype, grammar, logprob and thinking
    // paths. Adaptive entropy/zone observations are required even at temp=0.
    if model.decode_logits_fp32()
        || adaptive_sampling
        || active.iter().any(|a| requires_pipeline(a, ctx))
    {
        return Ok(None);
    }

    let mut gates = Vec::with_capacity(active.len());
    for a in active {
        // Classify the actual FinalDecode parameters, not raw request fields:
        // the builder owns in-tool DRY/opener-bias gating and caller bias.
        let params = penalty_params_for(
            a,
            PositionKind::FinalDecode,
            a.temperature,
            a.seed.map(|s| s.wrapping_add(a.output_tokens.len() as u64)),
            a.logit_bias.clone(),
        );
        let gate = fast_greedy::classify_penalties(&params);
        if gate == PenaltyGate::Blocked {
            return Ok(None);
        }
        gates.push(gate);
    }

    let vocab = model.vocab_size();
    checked_bf16_extent(logits, active.len(), vocab)?;
    ensure!(
        !model.logits_ptr_is_fp32(logits),
        "ordinary greedy BF16 admission disagrees with logits pointer dtype"
    );
    let tokens = model.argmax_batch(logits, active.len(), 0)?;
    // Validate ALL rows first: a malformed later ID may not follow even an
    // earlier valid row's scalar read, let alone host state publication.
    validate_argmax(&tokens, active.len(), vocab)?;
    for (row, ((a, gate), token)) in active.iter().zip(gates).zip(&tokens).enumerate() {
        if gate == PenaltyGate::ReduceOnly {
            let history = penalty_history_scope(&a.output_tokens, ctx.tool_call_end_token);
            if !fast_greedy::argmax_immune(*token, history, || {
                fast_greedy::logit_is_positive(model, logits, row, vocab, *token)
            }) {
                // Includes a failed scalar D2H: the SSOT proof fails closed,
                // then ordinary decode copies/samples the original full rows.
                return Ok(None);
            }
        }
    }
    Ok(Some(tokens))
}

/// Conservative admission for speculative policy rows. Unlike ordinary
/// decode's reduce-only proof, compact verification cannot fetch a contender
/// logit after staging, so only a fully neutral policy is eligible.
pub(super) fn policy_raw_argmax_exclusions(
    a: &ActiveSeq,
    ctx: &LogitsContext,
    adaptive_sampling: bool,
) -> Option<[u32; 2]> {
    let rejection = if adaptive_sampling {
        Some("adaptive")
    } else if a.temperature != 0.0 {
        Some("temperature")
    } else if a.top_logprobs.is_some() {
        Some("logprobs")
    } else if a.grammar_state.is_some() {
        Some("grammar")
    } else if a.inside_thinking {
        Some("inside-thinking")
    } else if ctx.tool_call_start_token.is_some()
        && (a.suppress_tool_call
            || (a.think_just_ended && a.require_tool_call && !a.tool_call_opened))
    {
        Some("tool-mask")
    } else {
        None
    };
    if let Some(reason) = rejection {
        return report_policy_argmax_rejection(reason);
    }
    let params = penalty_params_for(
        a,
        PositionKind::FinalDecode,
        a.temperature,
        a.seed.map(|s| s.wrapping_add(a.output_tokens.len() as u64)),
        a.logit_bias.clone(),
    );
    match fast_greedy::classify_penalties(&params) {
        PenaltyGate::Neutral => {}
        PenaltyGate::ReduceOnly => return report_policy_argmax_rejection("reduce-only"),
        PenaltyGate::Blocked => return report_policy_argmax_rejection("blocked-penalty"),
    }
    let mut excluded = [u32::MAX; 2];
    if a.think_ended {
        excluded = [
            ctx.think_end_token.unwrap_or(u32::MAX),
            a.think_start_token.unwrap_or(u32::MAX),
        ];
    }
    Some(excluded)
}

fn report_policy_argmax_rejection(reason: &'static str) -> Option<[u32; 2]> {
    static REPORTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::info!(
            schema = "atlas.glm53.policy_argmax_rejection.v1",
            reason,
            "GLM_POLICY_DEVICE_ARGMAX_REJECTED"
        );
    }
    None
}

fn requires_pipeline(a: &ActiveSeq, ctx: &LogitsContext) -> bool {
    a.temperature != 0.0
        || a.top_logprobs.is_some()
        || a.grammar_state.is_some()
        || a.inside_thinking
        || a.think_ended
        || (ctx.tool_call_start_token.is_some()
            && (a.suppress_tool_call
                || (a.think_just_ended && a.require_tool_call && !a.tool_call_opened)))
    // min_tokens and an unpinned require_tool_call affect post-selection EOS
    // handling, which remains in advance_ordinary on BOTH paths.
}

fn checked_bf16_extent(logits: DevicePtr, rows: usize, vocab: usize) -> Result<()> {
    ensure!(!logits.is_null(), "ordinary greedy logits pointer is null");
    ensure!(
        rows > 0 && vocab > 0,
        "ordinary greedy logits geometry is empty"
    );
    let bytes = rows
        .checked_mul(vocab)
        .and_then(|n| n.checked_mul(2))
        .context("ordinary greedy logits extent overflow")?;
    logits
        .0
        .checked_add(u64::try_from(bytes)?)
        .context("ordinary greedy logits address overflow")?;
    Ok(())
}

fn validate_argmax(tokens: &[u32], rows: usize, vocab: usize) -> Result<()> {
    ensure!(
        rows > 0 && vocab > 0,
        "ordinary greedy argmax geometry is empty"
    );
    ensure!(
        tokens.len() == rows,
        "ordinary greedy argmax row count mismatch"
    );
    ensure!(
        tokens.iter().all(|&token| (token as usize) < vocab),
        "ordinary greedy argmax token is outside vocabulary"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_shape_and_every_token_are_validated_before_row_reads() {
        assert!(validate_argmax(&[1, 4], 2, 5).is_ok());
        assert!(validate_argmax(&[1], 2, 5).is_err());
        assert!(validate_argmax(&[1, 2, 3], 2, 5).is_err());
        assert!(validate_argmax(&[1, 5], 2, 5).is_err());
        assert!(validate_argmax(&[u32::MAX, 1], 2, 5).is_err());
    }

    #[test]
    fn bf16_matrix_extent_rejects_empty_null_and_overflow() {
        assert!(checked_bf16_extent(DevicePtr(4096), 2, 5).is_ok());
        assert!(checked_bf16_extent(DevicePtr::NULL, 2, 5).is_err());
        assert!(checked_bf16_extent(DevicePtr(4096), 0, 5).is_err());
        assert!(checked_bf16_extent(DevicePtr(4096), 2, 0).is_err());
        assert!(checked_bf16_extent(DevicePtr(4096), usize::MAX, 5).is_err());
        assert!(checked_bf16_extent(DevicePtr(u64::MAX - 2), 1, 5).is_err());
    }
}
