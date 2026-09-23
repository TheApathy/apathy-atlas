// SPDX-License-Identifier: AGPL-3.0-only

//! GLM's policy-before-commit scheduler. This model owns recurrent state and
//! proposer captures; the generic verifier's later commit hooks do not apply.

#[path = "glm53_verify_policy_adapter.rs"]
mod adapter;
#[path = "glm53_policy_replay.rs"]
mod replay;

use super::logit_processors::LogitsContext;
use super::{ActiveSeq, Model, ResponseSink, adaptive_spec, mark_sequence_error};
use adapter::{NeedOrdinaryReplay, OrdinaryEffectRequest, OrdinaryVerifyPolicy, PolicyOptions};
use anyhow::{Context, Result, ensure};
use spark_model::model::glm53::verify_policy_transaction::{
    PolicyAdvance, VerifyOutcome, VerifyPolicy,
};

struct Settings<'a> {
    logits: &'a LogitsContext,
    num_drafts: usize,
    raw_proposal: bool,
    adaptive_sampling: bool,
    code_fence_token: Option<u32>,
    stream: u64,
}

pub(super) fn step(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    num_drafts: usize,
    verify_ctx: &LogitsContext,
    dflash_verify_raw_argmax: bool,
    adaptive_sampling: bool,
    code_fence_token: Option<u32>,
) {
    if active.is_empty() {
        return;
    }
    let settings = Settings {
        logits: verify_ctx,
        num_drafts,
        raw_proposal: dflash_verify_raw_argmax,
        adaptive_sampling,
        code_fence_token,
        stream: model.default_stream(),
    };
    if active.len() != 1 {
        let error = anyhow::anyhow!("GLM policy driver requires exactly one active sequence");
        for a in active {
            mark_sequence_error(a, "GLM policy admission", &error);
        }
        return;
    }
    let a = &mut active[0];
    if a.finished || a.terminal_error.is_some() {
        return;
    }
    if let Err(error) = step_one(model, a, &settings) {
        // Includes failed host publication/grammar installation after a device
        // commit. The retirement owner will emit one error and release once.
        model.poison_verify_policy(settings.stream);
        mark_sequence_error(a, "GLM ordinary-policy driver", &error);
    }
}

fn step_one(model: &dyn Model, a: &mut ActiveSeq, settings: &Settings<'_>) -> Result<()> {
    ensure!(
        model.requires_verify_policy(),
        "GLM policy driver requires typed model admission"
    );
    ensure!(
        settings.num_drafts <= 7,
        "GLM supports zero ordinary or 1..=7 draft tokens"
    );
    ensure!(
        a.seq.tokens.len() == a.seq.seq_len && a.seq.kv_valid_tokens == a.seq.seq_len,
        "GLM host prefix extents require reconciliation"
    );
    ensure!(
        (a.last_token as usize) < model.vocab_size(),
        "GLM anchor outside vocabulary"
    );
    if a.remaining == 0 {
        a.pending_drafts.clear();
        a.finished = true;
        return Ok(());
    }
    let ceiling = super::max_seq_len_ceiling();
    // The scheduler's explicit zero value disables its served-context guard;
    // the model-owned binding still enforces actual allocation capacity.
    ensure!(
        ceiling == 0 || a.seq.seq_len < ceiling,
        "GLM served sequence limit reached"
    );
    model
        .sync_secondary()
        .context("GLM policy secondary-stream ordering")?;

    let num_drafts = settings.num_drafts;
    let allowed = num_drafts != 0 && adaptive_spec::spec_allowed(a);
    if num_drafts == 0 || !allowed {
        a.pending_drafts.clear();
        return ordinary_and_propose(model, a, settings);
    }
    if a.pending_drafts.is_empty() {
        let serial_seam = super::verify_pipeline_helper::dflash_seam_serial_enabled();
        if !settings.raw_proposal || serial_seam {
            return ordinary_and_propose(model, a, settings);
        }
        a.pending_drafts = propose(model, a, settings)?;
        if a.pending_drafts.is_empty() {
            // An explicitly empty proposal carries no staged target state.
            return ordinary_and_propose(model, a, settings);
        }
    }
    let drafts = a.pending_drafts.clone();
    a.pending_drafts.clear();
    ensure!(
        (1..=7).contains(&drafts.len()),
        "GLM pending draft extent must be 1..=7"
    );
    verify(model, a, &drafts, settings)?;
    propose_next(model, a, settings)
}

fn ordinary_and_propose(
    model: &dyn Model,
    a: &mut ActiveSeq,
    settings: &Settings<'_>,
) -> Result<()> {
    a.pending_drafts.clear();
    replay::ordinary(model, a, settings)?;
    adaptive_spec::tick_serial(a);
    propose_next(model, a, settings)
}

fn propose_next(model: &dyn Model, a: &mut ActiveSeq, settings: &Settings<'_>) -> Result<()> {
    if !a.finished
        && a.terminal_error.is_none()
        && settings.num_drafts != 0
        && adaptive_spec::spec_allowed(a)
    {
        a.pending_drafts = propose(model, a, settings)?;
    }
    Ok(())
}

fn propose(model: &dyn Model, a: &mut ActiveSeq, settings: &Settings<'_>) -> Result<Vec<u32>> {
    // This verifier advances the live grammar for every accepted prefix row.
    // The legacy one-draft clamp applies to verifiers without that policy.
    let requested = settings.num_drafts;
    ensure!(
        (1..=7).contains(&requested),
        "GLM proposal request outside trained block extent"
    );
    let mask = super::mtp_grammar_mask_for(a);
    let drafts = model
        .run_mtp_propose_multi(
            a.last_token,
            a.seq.seq_len,
            requested,
            &mut a.seq,
            settings.stream,
            mask.as_deref(),
        )
        .context("GLM policy proposal failed")?;
    ensure!(
        drafts.len() <= requested && drafts.iter().all(|&v| (v as usize) < model.vocab_size()),
        "GLM proposer returned an invalid extent or token"
    );
    Ok(drafts)
}

fn verify(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    settings: &Settings<'_>,
) -> Result<()> {
    let start = a.seq.seq_len;
    let before_output = a.output_tokens.len();
    let mut tokens = Vec::with_capacity(drafts.len() + 1);
    tokens.push(a.last_token);
    tokens.extend_from_slice(drafts);
    let bound = model.bind_verify_policy_request(&tokens, &a.seq, settings.stream)?;
    let (outcome, prepared) = {
        let policy = OrdinaryVerifyPolicy::new(
            model,
            a,
            settings.logits,
            PolicyOptions {
                vocab_size: model.vocab_size(),
                max_seq_len: super::max_seq_len_ceiling(),
                adaptive_sampling: settings.adaptive_sampling,
                code_fence_token: settings.code_fence_token,
            },
        )?;
        let mut bridge = PolicyBridge(policy);
        let outcome = model.decode_verify_with_policy(bound, &mut bridge)?;
        ensure!(
            bridge.0.sequence().last_token == tokens[0],
            "GLM transaction did not restore the policy anchor"
        );
        let prepared = match &outcome {
            VerifyOutcome::Committed(_) => Some(bridge.0.take_prepared()?),
            VerifyOutcome::RestoredForOrdinaryReplay(_) => None,
        };
        drop(bridge); // Release the exclusive ActiveSeq borrow before publication.
        (outcome, prepared)
    };
    match outcome {
        VerifyOutcome::Committed(receipt) => {
            let accepted = receipt.accepted_drafts();
            let published = receipt.publish(
                &mut a.seq.tokens,
                &mut a.seq.seq_len,
                &mut a.seq.kv_valid_tokens,
            )?;
            let end = a.seq.seq_len;
            let selected_count = published.emitted_tokens().len();
            let events = prepared
                .context("GLM committed outcome lost retained policy")?
                .install(a, published.emitted_tokens(), end)?;
            ensure!(
                a.finished == published.terminal(),
                "GLM terminal policy/commit disagreement"
            );
            adaptive_spec::record_verify(a, accepted);
            for event in events {
                let ResponseSink::Streaming(sender) = &a.sink else {
                    anyhow::bail!("GLM retained streaming event lost its stream owner");
                };
                if !super::mod_helpers::bounded_stream_send(sender, event, "GLM committed policy") {
                    a.finished = true;
                    break;
                }
            }
            crate::metrics::SPEC_DECODE_VERIFY
                .with_label_values(&[
                    "dflash",
                    if accepted == drafts.len() {
                        "accept_all"
                    } else {
                        "accept_partial"
                    },
                ])
                .inc();
            tracing::info!(
                schema = "atlas.glm53.policy_verify.v1",
                start,
                end,
                proposed = drafts.len(),
                accepted,
                selected = selected_count,
                output_tokens_added = a.output_tokens.len().saturating_sub(before_output),
                terminal = a.finished,
                "GLM ordinary-policy verification committed"
            );
        }
        VerifyOutcome::RestoredForOrdinaryReplay(restored) => {
            restored.validate_host_prefix(&a.seq.tokens, a.seq.seq_len, a.seq.kv_valid_tokens)?;
            ensure!(
                a.last_token == restored.anchor() && a.seq.seq_len == restored.start(),
                "GLM restored ordinary replay anchor/position drift"
            );
            replay::ordinary(model, a, settings)?;
            adaptive_spec::tick_serial(a);
        }
    }
    Ok(())
}

/// Only the typed ordinary effect permits replay; copy/sampling/restore errors
/// never grant fallback authority. The model transaction owns both restorations.
struct PolicyBridge<'a>(OrdinaryVerifyPolicy<'a>);

impl VerifyPolicy for PolicyBridge<'_> {
    fn checkpoint(&mut self) -> Result<()> {
        self.0.checkpoint()
    }
    fn compact_argmax_exclusions(&self) -> Option<[u32; 2]> {
        self.0.compact_argmax_exclusions()
    }
    fn pick(&mut self, row: usize, logits: &[u8]) -> Result<u32> {
        self.0.pick(row, logits)
    }
    fn pick_argmax(&mut self, row: usize, token: u32) -> Result<u32> {
        self.0.pick_argmax(row, token)
    }
    fn advance(&mut self, token: u32) -> Result<PolicyAdvance> {
        match self.0.advance(token) {
            Ok(true) => Ok(PolicyAdvance::Continue),
            Ok(false) => Ok(PolicyAdvance::Terminal),
            Err(error) => match error.downcast_ref::<NeedOrdinaryReplay>() {
                Some(request) => match request.effect {
                    OrdinaryEffectRequest::Rollback { .. }
                    | OrdinaryEffectRequest::BoundaryCheckpoint
                    | OrdinaryEffectRequest::GrammarBudgetClose => {
                        Ok(PolicyAdvance::OrdinaryReplay)
                    }
                },
                None => Err(error),
            },
        }
    }
    fn restore(&mut self) -> Result<()> {
        self.0.restore()
    }
}
