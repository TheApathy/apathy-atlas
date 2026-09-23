// SPDX-License-Identifier: AGPL-3.0-only

//! Observational accounting for one completed DFlash verification.
//! "Emitted" means appended to the scheduler's output-token ledger, including
//! its existing EOS accounting, not necessarily visible text or an SSE event.
//! Suppressed/cancelled tokens and injected grammar-close tokens are distinct.

#[derive(Clone, Copy, Debug)]
pub(super) enum ReceiptExit {
    TerminalDraft,
    TerminalBonus,
    Continue,
}

impl ReceiptExit {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::TerminalDraft => "terminal_draft",
            Self::TerminalBonus => "terminal_bonus",
            Self::Continue => "continue",
        }
    }
}

pub(super) struct DflashVerifyReceipt {
    proposed: usize,
    accepted: usize,
    initial_output: usize,
    last_output: usize,
    attempted_drafts: usize,
    emitted_drafts: usize,
    bonus_attempted: bool,
    emitted_bonus: usize,
    error: Option<&'static str>,
}

#[derive(Debug)]
pub(super) struct ReceiptSummary {
    pub(super) proposed: usize,
    pub(super) accepted: usize,
    pub(super) rejected: usize,
    pub(super) attempted_drafts: usize,
    pub(super) emitted_drafts: usize,
    pub(super) emitted_bonus: usize,
    pub(super) output_tokens_added: usize,
    pub(super) injected_output_tokens: usize,
    pub(super) error: Option<&'static str>,
}

impl DflashVerifyReceipt {
    pub(super) fn new(proposed: usize, accepted: usize, output_tokens: usize) -> Self {
        Self {
            proposed,
            accepted,
            initial_output: output_tokens,
            last_output: output_tokens,
            attempted_drafts: 0,
            emitted_drafts: 0,
            bonus_attempted: false,
            emitted_bonus: 0,
            error: (proposed == 0 || accepted > proposed).then_some("invalid_acceptance_counts"),
        }
    }

    pub(super) fn observe_draft(&mut self, token: u32, before: usize, output: &[u32]) {
        if self.attempted_drafts >= self.accepted || self.bonus_attempted {
            self.error.get_or_insert("draft_emission_order");
        }
        match self.attempted_drafts.checked_add(1) {
            Some(count) => self.attempted_drafts = count,
            None => {
                self.error.get_or_insert("draft_counter_overflow");
            }
        }
        if self.observe(token, before, output) {
            match self.emitted_drafts.checked_add(1) {
                Some(count) => self.emitted_drafts = count,
                None => {
                    self.error.get_or_insert("emitted_counter_overflow");
                }
            }
        }
    }

    pub(super) fn observe_bonus(&mut self, token: u32, before: usize, output: &[u32]) {
        if self.bonus_attempted || self.attempted_drafts != self.accepted {
            self.error.get_or_insert("bonus_emission_order");
        }
        self.bonus_attempted = true;
        self.emitted_bonus = usize::from(self.observe(token, before, output));
    }

    fn observe(&mut self, token: u32, before: usize, output: &[u32]) -> bool {
        if before != self.last_output {
            self.error.get_or_insert("output_ledger_gap");
        }
        self.last_output = output.len();
        let Some(appended) = output.get(before..) else {
            self.error.get_or_insert("output_ledger_shrank");
            return false;
        };
        match appended.first() {
            None => false, // Cancellation or suppression before output_tokens.push.
            Some(&first) if first == token => true,
            Some(_) => {
                self.error.get_or_insert("unexpected_output_token");
                false
            }
        }
    }

    pub(super) fn finish(&self, exit: ReceiptExit, output_tokens: usize) -> ReceiptSummary {
        let mut error = self.error;
        if output_tokens != self.last_output {
            error.get_or_insert("unobserved_output_change");
        }
        let closed = match exit {
            ReceiptExit::TerminalDraft => self.attempted_drafts > 0 && !self.bonus_attempted,
            ReceiptExit::TerminalBonus | ReceiptExit::Continue => {
                self.attempted_drafts == self.accepted && self.bonus_attempted
            }
        };
        if !closed {
            error.get_or_insert("incomplete_emission_accounting");
        }
        let output_tokens_added = output_tokens
            .checked_sub(self.initial_output)
            .unwrap_or_else(|| {
                error.get_or_insert("output_ledger_shrank");
                0
            });
        let injected_output_tokens = self
            .emitted_drafts
            .checked_add(self.emitted_bonus)
            .and_then(|emitted| output_tokens_added.checked_sub(emitted))
            .unwrap_or_else(|| {
                error.get_or_insert("invalid_output_accounting");
                0
            });
        ReceiptSummary {
            proposed: self.proposed,
            accepted: self.accepted,
            rejected: self.proposed.saturating_sub(self.accepted),
            attempted_drafts: self.attempted_drafts,
            emitted_drafts: self.emitted_drafts,
            emitted_bonus: self.emitted_bonus,
            output_tokens_added,
            injected_output_tokens,
            error,
        }
    }
}
