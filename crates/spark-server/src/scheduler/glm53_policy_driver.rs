// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 DFlash2 speculative step (`Model::requires_verify_policy`).
//!
//! GLM owns recurrent KDA/DSA state and its drafter captures, so the generic
//! verifiers (`step_mtp`) do not apply; the model runs a policy-before-commit
//! transaction instead (`decode_verify_with_policy`). This driver supplies the
//! policy, and it adds no sampler or token transition of its own: every row
//! the transaction offers is sampled and committed by the ordinary decode
//! path, `process_decode_logits`, reading that row's logits where the model
//! staged them on the device. A speculative step therefore emits exactly the
//! tokens, stream events and request state that one ordinary decode per row
//! would, and it stops at the first row whose ordinary pick differs from the
//! draft. Steps with no usable proposal run `step_decode_only` itself.
//!
//! Because each pick is a real emission, the transaction's checkpoint and
//! restore are no-ops and the driver never asks for an ordinary replay. Any
//! error after the transaction staged target state poisons the model, and
//! the sequence is retired with the error.

use super::{ActiveSeq, Model, process_decode_logits, send_error, step_decode_only};
use anyhow::{Context, Result, bail, ensure};
use spark_model::model::glm53::verify_policy_transaction::{
    PolicyAdvance, VerifyOutcome, VerifyPolicy,
};
use spark_runtime::gpu::DevicePtr;

/// The token ids and sampler settings `step_decode_only` receives.
pub(super) struct DecodeSettings<'a> {
    pub think_end_token: Option<u32>,
    pub think_start_token: Option<u32>,
    pub code_fence_token: Option<u32>,
    pub tool_call_start_token: Option<u32>,
    pub tool_call_end_token: Option<u32>,
    pub reflection_suppress_ids: &'a [u32],
    pub adaptive_sampling: bool,
}

/// One scheduler step for the single active GLM sequence.
pub(super) fn step(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    num_drafts: usize,
    settings: &DecodeSettings<'_>,
) {
    if active.is_empty() {
        return;
    }
    if active.len() != 1 {
        for mut a in active.drain(..) {
            send_error(
                model,
                &mut a,
                "GLM speculation requires exactly one active sequence",
            );
        }
        return;
    }
    if active[0].finished {
        return;
    }
    let stream = model.default_stream();
    if let Err(error) = step_one(model, active, num_drafts, settings, stream) {
        model.poison_verify_policy(stream);
        tracing::error!("GLM speculative step failed: {error:#}");
        for mut a in active.drain(..) {
            send_error(model, &mut a, &format!("GLM speculative step: {error:#}"));
        }
    }
}

fn step_one(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    num_drafts: usize,
    settings: &DecodeSettings<'_>,
    stream: u64,
) -> Result<()> {
    ensure!(
        model.requires_verify_policy(),
        "GLM speculative step requires the model's policy verifier"
    );
    ensure!(
        (1..=7).contains(&num_drafts),
        "GLM speculation takes 1..=7 drafts, got {num_drafts}"
    );
    // Drafts are proposed and verified within one step; nothing carries over.
    active[0].pending_drafts.clear();
    let drafts = propose(model, &mut active[0], num_drafts, stream)?;
    if drafts.is_empty() {
        ordinary(model, active, settings);
        return Ok(());
    }
    verify(model, active, &drafts, settings, stream)
}

fn ordinary(model: &dyn Model, active: &mut Vec<ActiveSeq>, settings: &DecodeSettings<'_>) {
    step_decode_only(
        model,
        active,
        settings.think_end_token,
        settings.think_start_token,
        settings.code_fence_token,
        settings.tool_call_start_token,
        settings.tool_call_end_token,
        settings.reflection_suppress_ids,
        settings.adaptive_sampling,
    );
}

fn propose(
    model: &dyn Model,
    a: &mut ActiveSeq,
    requested: usize,
    stream: u64,
) -> Result<Vec<u32>> {
    if a.remaining < 2 {
        // A one-token budget gains nothing from a draft.
        return Ok(Vec::new());
    }
    ensure!(
        a.seq.tokens.len() == a.seq.seq_len && a.seq.kv_valid_tokens == a.seq.seq_len,
        "GLM host prefix extents require reconciliation"
    );
    let mask = super::spec_step::mtp_grammar_mask_for(a);
    let drafts = model
        .run_mtp_propose_multi(
            a.last_token,
            a.seq.seq_len,
            requested,
            &mut a.seq,
            stream,
            mask.as_deref(),
        )
        .context("GLM DFlash2 proposal failed")?;
    ensure!(
        drafts.len() <= requested && drafts.iter().all(|&v| (v as usize) < model.vocab_size()),
        "GLM proposer returned an invalid extent or token"
    );
    Ok(drafts)
}

fn verify(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    drafts: &[u32],
    settings: &DecodeSettings<'_>,
    stream: u64,
) -> Result<()> {
    let (start, before_output) = (active[0].seq.seq_len, active[0].output_tokens.len());
    let mut inputs = Vec::with_capacity(drafts.len() + 1);
    inputs.push(active[0].last_token);
    inputs.extend_from_slice(drafts);
    let bound = model.bind_verify_policy_request(&inputs, &active[0].seq, stream)?;
    let outcome = {
        let mut policy = OrdinaryDecodePolicy {
            model,
            active: &mut *active,
            settings,
            t0: std::time::Instant::now(),
        };
        model.decode_verify_with_policy(bound, &mut policy)?
    };
    let VerifyOutcome::Committed(receipt) = outcome else {
        bail!("GLM verify restored for an ordinary replay this policy never requests");
    };
    let accepted = receipt.accepted_drafts();
    let a = active
        .first_mut()
        .context("GLM sequence retired during a committed verify")?;
    let published = receipt.publish(
        &mut a.seq.tokens,
        &mut a.seq.seq_len,
        &mut a.seq.kv_valid_tokens,
    )?;
    let emitted = published.emitted_tokens();
    ensure!(
        emitted.last() == Some(&a.last_token),
        "GLM committed verify disagrees with the ordinary emission"
    );
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
    tracing::debug!(
        start,
        end = a.seq.seq_len,
        proposed = drafts.len(),
        accepted,
        output_tokens_added = a.output_tokens.len().saturating_sub(before_output),
        finished = a.finished,
        "GLM DFlash2 verify committed"
    );
    Ok(())
}

/// Samples and commits each verify row through the ordinary decode path.
struct OrdinaryDecodePolicy<'a, 's> {
    model: &'a dyn Model,
    active: &'a mut Vec<ActiveSeq>,
    settings: &'a DecodeSettings<'s>,
    t0: std::time::Instant,
}

impl VerifyPolicy for OrdinaryDecodePolicy<'_, '_> {
    fn checkpoint(&mut self) -> Result<()> {
        Ok(())
    }

    fn pick(&mut self, _row: usize, _logits: &[u8]) -> Result<u32> {
        bail!("GLM ordinary-decode policy needs the staged logits resident on the device")
    }

    fn pick_resident(
        &mut self,
        row: usize,
        _logits: &[u8],
        device_row: Option<DevicePtr>,
    ) -> Result<u32> {
        let device_row = device_row
            .context("GLM ordinary-decode policy needs the staged logits resident on the device")?;
        ensure!(
            self.active.len() == 1 && !self.active[0].finished,
            "GLM verify row {row} offered to a retired sequence"
        );
        let s = self.settings;
        process_decode_logits(
            self.model,
            self.active,
            device_row,
            self.t0,
            s.think_end_token,
            s.think_start_token,
            s.code_fence_token,
            s.tool_call_start_token,
            s.tool_call_end_token,
            s.reflection_suppress_ids,
            s.adaptive_sampling,
        );
        // The ordinary path retires the sequence itself on a sampling error.
        let a = self
            .active
            .first()
            .context("GLM sequence was retired by the ordinary sampler")?;
        Ok(a.last_token)
    }

    fn advance(&mut self, token: u32) -> Result<PolicyAdvance> {
        let a = self
            .active
            .first()
            .context("GLM sequence was retired by the ordinary sampler")?;
        ensure!(
            a.last_token == token,
            "GLM policy advanced an unpicked token"
        );
        Ok(if a.finished {
            PolicyAdvance::Terminal
        } else {
            PolicyAdvance::Continue
        })
    }

    fn restore(&mut self) -> Result<()> {
        Ok(())
    }
}
