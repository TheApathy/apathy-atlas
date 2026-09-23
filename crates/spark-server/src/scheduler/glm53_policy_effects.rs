// SPDX-License-Identifier: AGPL-3.0-only

//! Speculative effects: record reversible grammar/emission decisions; request
//! a genuine restored ordinary round before any unsupported persistent effect.

use crate::grammar::GrammarState;
use crate::scheduler::ordinary_transition::OrdinaryEffects;
use crate::scheduler::rollback::RollbackOutcome;
use crate::scheduler::{ActiveSeq, StreamEvent};
use anyhow::{Result, ensure};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::scheduler) enum OrdinaryEffectRequest {
    Rollback { min_keep: usize },
    BoundaryCheckpoint,
    GrammarBudgetClose,
}

#[derive(Debug)]
pub(in crate::scheduler) struct NeedOrdinaryReplay {
    pub effect: OrdinaryEffectRequest,
}

impl std::fmt::Display for NeedOrdinaryReplay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GLM policy needs a restored ordinary round: {:?}",
            self.effect
        )
    }
}
impl std::error::Error for NeedOrdinaryReplay {}

pub(super) struct RecordedEffects<'a> {
    pub position: usize,
    pub grammar_tokens: &'a mut Vec<u32>,
    pub events: &'a mut Vec<StreamEvent>,
}

impl OrdinaryEffects for RecordedEffects<'_> {
    fn position(&self, _: &ActiveSeq) -> usize {
        self.position
    }

    fn accept_grammar(&mut self, grammar: &mut GrammarState, token: u32) -> Result<()> {
        ensure!(
            grammar.accept_token(token),
            "GLM ordinary policy grammar refused selected token {token}"
        );
        self.grammar_tokens.push(token);
        Ok(())
    }

    fn rollback(&mut self, _: &mut ActiveSeq, min_keep: usize) -> Result<RollbackOutcome> {
        Err(NeedOrdinaryReplay {
            effect: OrdinaryEffectRequest::Rollback { min_keep },
        }
        .into())
    }

    fn checkpoint(&mut self, a: &mut ActiveSeq) -> Result<()> {
        if a.ssm_rollback_ring.is_enabled() {
            return Err(NeedOrdinaryReplay {
                effect: OrdinaryEffectRequest::BoundaryCheckpoint,
            }
            .into());
        }
        // This adapter's binding MUST be GLM EXL3: its disabled decode ring
        // and default-no-op Marconi hook make this ordinary effect inert.
        // Other architectures are not admitted by the root model transaction.
        Ok(())
    }

    fn stream(&mut self, _: &ActiveSeq, event: StreamEvent, _: &'static str) -> Result<bool> {
        ensure!(
            matches!(
                event,
                StreamEvent::Token(_) | StreamEvent::TokenWithLogprobs(_, _)
            ),
            "unexpected event in GLM speculative token policy"
        );
        self.events.push(event);
        Ok(true)
    }

    fn grammar_close(&mut self, a: &mut ActiveSeq) -> Result<()> {
        if a.inside_thinking || !crate::scheduler::helpers::grammar_budget_close_enabled() {
            return Ok(());
        }
        let Some(grammar) = a.grammar_state.as_mut() else {
            return Ok(());
        };
        if grammar.is_terminated() || grammar.stop_legal(&a.eos_tokens) {
            return Ok(());
        }
        // The original ordinary close routine remains the sole owner of its
        // bounded completion search and synthetic tokens, even if it finds no
        // close. Never reproduce that search or append target-less tokens here.
        Err(NeedOrdinaryReplay {
            effect: OrdinaryEffectRequest::GrammarBudgetClose,
        }
        .into())
    }
}
