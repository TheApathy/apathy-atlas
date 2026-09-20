// SPDX-License-Identifier: AGPL-3.0-only

//! GLM EXL3 ordinary-policy adapter. The caller must obtain a typed, owned
//! GLM model/request binding before borrowing ActiveSeq; model verification
//! must not simultaneously borrow its SequenceState. No GPU state is moved.

use crate::api::TokenLogprobs;
use crate::scheduler::logit_processors::LogitsContext;
use crate::scheduler::ordinary_transition::{TransitionContext, advance_ordinary};
use crate::scheduler::{ActiveSeq, Model, StreamEvent};
use anyhow::{Context, Result, ensure};
use std::time::Instant;

#[path = "glm53_policy_effects.rs"]
mod effects;
#[path = "glm53_policy_snapshot.rs"]
mod snapshot;
use effects::RecordedEffects;
pub(in crate::scheduler) use effects::{NeedOrdinaryReplay, OrdinaryEffectRequest};
use snapshot::PolicySnapshot;

pub(in crate::scheduler) struct PolicyOptions {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub adaptive_sampling: bool,
    pub code_fence_token: Option<u32>,
}

pub(in crate::scheduler) struct OrdinaryVerifyPolicy<'a> {
    model: &'a dyn Model,
    seq: &'a mut ActiveSeq,
    context: &'a LogitsContext,
    options: PolicyOptions,
    start: usize,
    saved: Option<PolicySnapshot>,
    pending: Option<(u32, Option<TokenLogprobs>)>,
    selected: Vec<u32>,
    grammar_tokens: Vec<u32>,
    events: Vec<StreamEvent>,
    prepared: Option<PreparedPolicy>,
    closed: bool,
    failed: bool,
}

impl<'a> OrdinaryVerifyPolicy<'a> {
    pub(in crate::scheduler) fn new(
        model: &'a dyn Model,
        seq: &'a mut ActiveSeq,
        context: &'a LogitsContext,
        options: PolicyOptions,
    ) -> Result<Self> {
        ensure!(
            (1..=154_880).contains(&options.vocab_size),
            "GLM policy vocabulary is outside its bound"
        );
        ensure!(
            model.vocab_size() == options.vocab_size,
            "GLM policy model vocabulary mismatch"
        );
        ensure!(
            seq.seq.tokens.len() == seq.seq.seq_len && seq.seq.kv_valid_tokens == seq.seq.seq_len,
            "GLM policy host/model prefix requires reconciliation before staging"
        );
        ensure!(
            options.max_seq_len == 0
                || (seq.seq.seq_len < options.max_seq_len
                    && seq.output_tokens.len() <= options.max_seq_len),
            "GLM policy request exceeds its served context bound"
        );
        ensure!(
            !seq.finished && seq.terminal_error.is_none() && seq.remaining > 0,
            "GLM policy cannot verify an already finished request"
        );
        let start = seq.seq.seq_len;
        Ok(Self {
            model,
            seq,
            context,
            options,
            start,
            saved: None,
            pending: None,
            selected: Vec::with_capacity(8),
            grammar_tokens: Vec::with_capacity(16),
            events: Vec::with_capacity(8),
            prepared: None,
            closed: false,
            failed: false,
        })
    }

    pub(in crate::scheduler) fn sequence(&self) -> &ActiveSeq {
        self.seq
    }

    pub(in crate::scheduler) fn checkpoint(&mut self) -> Result<()> {
        ensure!(
            !self.closed && self.saved.is_none() && self.selected.is_empty(),
            "GLM policy checkpoint is not repeatable"
        );
        self.saved = Some(PolicySnapshot::capture(self.seq));
        Ok(())
    }

    pub(in crate::scheduler) fn pick(&mut self, index: usize, logits: &[u8]) -> Result<u32> {
        ensure!(
            self.saved.is_some() && !self.closed && !self.failed && !self.seq.finished,
            "GLM policy pick outside a live checkpoint"
        );
        ensure!(
            self.pending.is_none() && index == self.selected.len() && index < 8,
            "GLM policy row is out of order or already selected"
        );
        ensure!(
            logits.len() == self.options.vocab_size * 2,
            "GLM policy BF16 row extent mismatch"
        );
        ensure!(
            !logits
                .chunks_exact(2)
                .any(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7fff > 0x7f80),
            "GLM policy cannot sample NaN target logits"
        );
        let (token, logprobs) = crate::scheduler::decode_logits_seq::process_seq_logits(
            self.model,
            self.seq,
            logits,
            0,
            self.options.vocab_size,
            2,
            false,
            self.context,
            self.options.adaptive_sampling,
        );
        ensure!(
            (token as usize) < self.options.vocab_size,
            "GLM ordinary sampler returned an invalid token"
        );
        self.pending = Some((token, logprobs));
        Ok(token)
    }

    pub(in crate::scheduler) fn advance(&mut self, token: u32) -> Result<bool> {
        ensure!(
            self.saved.is_some() && !self.closed && !self.failed,
            "GLM policy advance outside a live checkpoint"
        );
        ensure!(
            self.pending
                .as_ref()
                .is_some_and(|(picked, _)| *picked == token),
            "GLM policy cannot advance an unselected token"
        );
        let position = self
            .start
            .checked_add(self.selected.len() + 1)
            .context("GLM policy position overflow")?;
        ensure!(
            self.options.max_seq_len == 0 || position <= self.options.max_seq_len,
            "GLM policy row exceeds served context"
        );
        let (_, logprobs) = self
            .pending
            .take()
            .context("GLM policy selected token disappeared")?;
        let context = TransitionContext {
            logits: *self.context,
            code_fence_token: self.options.code_fence_token,
            max_seq_len: self.options.max_seq_len,
            // A speculative view does not publish an emission timestamp.
            now: self.seq.last_token_time,
        };
        let mut effects = RecordedEffects {
            position,
            grammar_tokens: &mut self.grammar_tokens,
            events: &mut self.events,
        };
        if let Err(error) = advance_ordinary(self.seq, token, logprobs, &context, &mut effects) {
            self.failed = true;
            return Err(error);
        }
        self.selected.push(token);
        Ok(!self.seq.finished)
    }

    pub(in crate::scheduler) fn restore(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        let Some(saved) = self.saved.take() else {
            self.closed = true;
            return Ok(());
        };
        let prepared =
            (!self.failed && self.pending.is_none() && !self.selected.is_empty()).then(|| {
                PreparedPolicy {
                    start: self.start,
                    grammar_before: saved.grammar_depth(),
                    final_state: PolicySnapshot::capture(self.seq),
                    selected: self.selected.clone(),
                    grammar_tokens: std::mem::take(&mut self.grammar_tokens),
                    events: std::mem::take(&mut self.events),
                }
            });
        let restored = saved.restore(self.seq);
        self.closed = true;
        restored?;
        self.prepared = prepared;
        Ok(())
    }

    /// Transfer the retained decisions only after successful policy restore.
    /// Root's committed-receipt bridge owns installation and actual emission.
    pub(in crate::scheduler) fn take_prepared(&mut self) -> Result<PreparedPolicy> {
        ensure!(
            self.closed,
            "GLM policy must restore before retaining publication decisions"
        );
        self.prepared
            .take()
            .context("GLM policy has no successful selected prefix to publish")
    }
}

impl Drop for OrdinaryVerifyPolicy<'_> {
    fn drop(&mut self) {
        if !self.closed {
            // Normal transaction code calls restore explicitly and propagates
            // its error to model poison. Drop is only an abandonment safeguard.
            if let Err(error) = self.restore() {
                crate::scheduler::sequence_error::mark_sequence_error(
                    self.seq,
                    "abandoned GLM policy restore",
                    &error,
                );
            }
        }
    }
}

#[must_use = "install only after a consumed committed host-prefix receipt"]
pub(in crate::scheduler) struct PreparedPolicy {
    start: usize,
    grammar_before: Option<usize>,
    final_state: PolicySnapshot,
    selected: Vec<u32>,
    grammar_tokens: Vec<u32>,
    events: Vec<StreamEvent>,
}

impl PreparedPolicy {
    /// The root bridge supplies `selected` and `committed_end` from its
    /// consumed model receipt, never from raw verification picks. All checks
    /// and grammar replay complete before any event leaves this function.
    /// Failure after model commit requires poisoning/reset, not a raw fallback.
    pub(in crate::scheduler) fn install(
        self,
        a: &mut ActiveSeq,
        selected: &[u32],
        committed_end: usize,
    ) -> Result<Vec<StreamEvent>> {
        ensure!(
            selected == self.selected,
            "GLM publication does not match ordinary policy decisions"
        );
        ensure!(
            committed_end > self.start
                && committed_end <= self.start + self.selected.len() + 1
                && a.seq.seq_len == committed_end
                && a.seq.kv_valid_tokens == committed_end
                && a.seq.tokens.len() == committed_end,
            "GLM policy installation requires a published target prefix"
        );
        ensure!(
            a.grammar_state.as_ref().map(|g| g.num_history_steps()) == self.grammar_before,
            "GLM policy grammar changed between restore and publication"
        );
        for token in self.grammar_tokens {
            let grammar = a
                .grammar_state
                .as_mut()
                .context("GLM publication lost its grammar")?;
            ensure!(
                grammar.accept_token(token),
                "GLM publication grammar refused a retained policy token"
            );
        }
        ensure!(
            a.grammar_state.as_ref().map(|g| g.num_history_steps())
                == self.final_state.grammar_depth(),
            "GLM publication grammar depth differs from the selected policy"
        );
        self.final_state.install_fields(a);
        a.last_token_time = Instant::now();
        Ok(self.events)
    }
}
