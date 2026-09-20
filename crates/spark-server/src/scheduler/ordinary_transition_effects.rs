// SPDX-License-Identifier: AGPL-3.0-only

//! Explicit effects for the ordinary transition. Live decode retains the
//! original effect ordering; GLM speculation supplies a reversible recorder.

use crate::grammar::GrammarState;
use crate::scheduler::rollback::{RollbackOutcome, rollback_to_boundary, snapshot_boundary_if_ssm};
use crate::scheduler::{ActiveSeq, Model, ResponseSink, StreamEvent};
use anyhow::Result;

pub(in crate::scheduler) trait OrdinaryEffects {
    fn position(&self, a: &ActiveSeq) -> usize;
    fn accept_grammar(&mut self, grammar: &mut GrammarState, token: u32) -> Result<()>;
    fn rollback(&mut self, a: &mut ActiveSeq, min_keep: usize) -> Result<RollbackOutcome>;
    fn checkpoint(&mut self, a: &mut ActiveSeq) -> Result<()>;
    fn stream(&mut self, a: &ActiveSeq, event: StreamEvent, label: &'static str) -> Result<bool>;
    fn grammar_close(&mut self, a: &mut ActiveSeq) -> Result<()>;
}

pub(in crate::scheduler) struct LiveEffects<'a> {
    pub model: &'a dyn Model,
}

impl OrdinaryEffects for LiveEffects<'_> {
    fn position(&self, a: &ActiveSeq) -> usize {
        a.seq.seq_len
    }

    fn accept_grammar(&mut self, grammar: &mut GrammarState, token: u32) -> Result<()> {
        // Preserve ordinary refusal handling. Token advancement itself is
        // single-owner in advance_ordinary, including tool boundaries.
        // Strict GLM staged policy reports refusal through its separate effect.
        grammar.accept_token(token);
        Ok(())
    }

    fn rollback(&mut self, a: &mut ActiveSeq, min_keep: usize) -> Result<RollbackOutcome> {
        Ok(rollback_to_boundary(a, min_keep, self.model))
    }

    fn checkpoint(&mut self, a: &mut ActiveSeq) -> Result<()> {
        snapshot_boundary_if_ssm(a, self.model);
        self.model.decode_marconi_checkpoint(&mut a.seq);
        Ok(())
    }

    fn stream(&mut self, a: &ActiveSeq, event: StreamEvent, label: &'static str) -> Result<bool> {
        let ResponseSink::Streaming(ref tx) = a.sink else {
            return Ok(true);
        };
        Ok(crate::scheduler::mod_helpers::bounded_stream_send(
            tx, event, label,
        ))
    }

    fn grammar_close(&mut self, a: &mut ActiveSeq) -> Result<()> {
        crate::scheduler::emit_step::emit_grammar_close(a);
        Ok(())
    }
}
