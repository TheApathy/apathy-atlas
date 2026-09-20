// SPDX-License-Identifier: AGPL-3.0-only

//! Reversible ordinary-policy state. GPU sequence ownership is never moved,
//! cloned, replaced, or "restored" by adjusting a host-only position.

use crate::adaptive_sampler::AdaptiveSamplingState;
use crate::api::TokenLogprobs;
use crate::scheduler::ActiveSeq;
use anyhow::{Result, ensure};
use std::time::Instant;

fn owned<T: Clone>(value: &T) -> T {
    value.clone()
}

macro_rules! policy_snapshot {
    ($($field:ident: $ty:ty),+ $(,)?) => {
        #[derive(Clone)]
        pub(super) struct PolicySnapshot {
            $($field: $ty,)+
            grammar_depth: Option<usize>,
            prefix: Vec<u32>,
            seq_len: usize,
            kv_valid: usize,
        }

        impl PolicySnapshot {
            pub(super) fn grammar_depth(&self) -> Option<usize> { self.grammar_depth }

            pub(super) fn capture(a: &ActiveSeq) -> Self {
                Self {
                    $($field: owned(&a.$field),)+
                    grammar_depth: a.grammar_state.as_ref().map(|g| g.num_history_steps()),
                    prefix: a.seq.tokens.clone(),
                    seq_len: a.seq.seq_len,
                    kv_valid: a.seq.kv_valid_tokens,
                }
            }

            /// Install policy fields only, after an independently validated
            /// commit receipt. Grammar replay is separate and must succeed
            /// before publishing any stream event.
            pub(super) fn install_fields(&self, a: &mut ActiveSeq) {
                $(a.$field = owned(&self.$field);)+
            }

            pub(super) fn restore(&self, a: &mut ActiveSeq) -> Result<()> {
                let grammar = self.restore_grammar(a);
                self.install_fields(a);
                // Never disguise an unexpected target-prefix mutation as a
                // successful GPU rollback. The caller must poison on failure.
                ensure!(a.seq.tokens == self.prefix && a.seq.seq_len == self.seq_len
                    && a.seq.kv_valid_tokens == self.kv_valid,
                    "GLM policy mutated the target prefix before commit");
                grammar
            }

            fn restore_grammar(&self, a: &mut ActiveSeq) -> Result<()> {
                match (self.grammar_depth, a.grammar_state.as_mut()) {
                    (None, None) => Ok(()),
                    (Some(before), Some(grammar)) => {
                        let now = grammar.num_history_steps();
                        ensure!(now >= before, "GLM speculative grammar rewound committed history");
                        let added = now - before;
                        ensure!(added <= i32::MAX as usize, "GLM grammar rollback exceeds matcher ABI");
                        // rollback(0) invalidates speculative mask caches too.
                        grammar.rollback(added);
                        ensure!(grammar.num_history_steps() == before,
                            "GLM speculative grammar rollback did not restore its exact depth");
                        Ok(())
                    }
                    _ => anyhow::bail!("GLM speculative policy changed grammar ownership"),
                }
            }
        }
    };
}

// Only fields mutated by ordinary sampling/transition are snapshotted. Request
// knobs (bias, penalties, seed, masks, EOS IDs, etc.) remain authoritative on
// the exclusively borrowed ActiveSeq and are never rewritten by the adapter.
policy_snapshot! {
    last_token: u32,
    output_tokens: Vec<u32>,
    remaining: usize,
    finished: bool,
    terminal_error: Option<String>,
    guard_stop: Option<&'static str>,
    param_close_pending: u8,
    inside_thinking: bool,
    thinking_budget: Option<u32>,
    thinking_tokens: u32,
    force_end_thinking: bool,
    think_force_closed: bool,
    sentence_defer_count: u32,
    consecutive_confident: u32,
    in_code_fence: bool,
    think_ended: bool,
    think_just_ended: bool,
    post_think_emitted: u32,
    think_skip_count: u32,
    require_tool_call: bool,
    tool_call_opened: bool,
    inside_tool_body: bool,
    tool_call_completed: bool,
    post_completion_tool_opens: u32,
    tool_body_streak_tokens: u32,
    inside_parameter_body: bool,
    param_body_chars_emitted: u32,
    content_started: bool,
    content_tokens: u32,
    prose_tokens_since_last_tool: u32,
    think_watchdog_fires: u32,
    rollback_count: u32,
    last_token_time: Instant,
    logprobs_data: Vec<TokenLogprobs>,
    adaptive: AdaptiveSamplingState,
}
