// SPDX-License-Identifier: AGPL-3.0-only

//! Phase: start new requests — either single-shot prefill (legacy) or
//! chunked prefill that pushes onto `prefilling`. Handles SSM-pool-full
//! preemption.

use spark_model::traits::Model;

use super::*;
use crate::api::InferenceRequest;
use crate::grammar::GrammarEngine;

fn initial_prefill_chunk_budget(max_prefill_tokens: usize, max_batch_tokens: usize) -> usize {
    max_prefill_tokens.min(max_batch_tokens)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn start_new_requests(
    model: &dyn Model,
    new_reqs: Vec<InferenceRequest>,
    chunked: bool,
    max_prefill_tokens: usize,
    max_batch_tokens: usize,
    eos_tokens: &[u32],
    prefill_stream: u64,
    prefill_event: u64,
    grammar_engine: &mut Option<GrammarEngine>,
    spontaneous_think_budget: u32,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    active: &mut Vec<ActiveSeq>,
    prefilling: &mut Vec<PrefillInProgress>,
) {
    for req in new_reqs {
        if chunked {
            // The arena includes extra decode-slot capacity, but that capacity
            // is not part of the explicit prefill ceiling. In particular, an
            // 8192-token kernel qualification must never become an 8193-token
            // launch merely because max_batch_size is one.
            let budget = initial_prefill_chunk_budget(max_prefill_tokens, max_batch_tokens);
            match start_chunked_prefill(
                think_end_token,
                think_start_token,
                tool_call_start_token,
                tool_call_end_token,
                model,
                req,
                eos_tokens,
                budget,
                prefill_stream,
                prefill_event,
                grammar_engine,
                spontaneous_think_budget,
            ) {
                Ok(StartPrefillResult::Active(a)) => {
                    tracing::info!(
                        "Prefilled (single chunk): seq_len={}, remaining={}",
                        a.seq.seq_len,
                        a.remaining,
                    );
                    active.push(a);
                }
                Ok(StartPrefillResult::InProgress(p)) => {
                    tracing::info!(
                        "Prefill chunk 0/{}: {}/{} tokens",
                        p.prompt_tokens.len(),
                        p.chunk_offset,
                        p.prompt_tokens.len(),
                    );
                    prefilling.push(p);
                }
                Ok(StartPrefillResult::Finished) => {} // EOS on first token
                Err(e) => {
                    handle_prefill_start_error(model, &e, active);
                }
            }
        } else {
            // Legacy non-chunked path.
            match prefill_request(
                think_end_token,
                think_start_token,
                tool_call_start_token,
                tool_call_end_token,
                model,
                req,
                eos_tokens,
                grammar_engine,
                spontaneous_think_budget,
            ) {
                Ok(Some(a)) => {
                    tracing::info!(
                        "Prefilled: seq_len={}, remaining={}",
                        a.seq.seq_len,
                        a.remaining,
                    );
                    active.push(a);
                }
                Ok(None) => {}
                Err(e) => {
                    handle_prefill_start_error(model, &e, active);
                }
            }
        }
    }
}

/// SSM-pool-full preemption: free oldest active sequence and surface a
/// 503-equivalent error to the preempted request. Mirrors vLLM's
/// preemption strategy — never return HTTP 500 for resource exhaustion.
fn handle_prefill_start_error(model: &dyn Model, e: &anyhow::Error, active: &mut Vec<ActiveSeq>) {
    let err_msg = format!("{e:#}");
    if err_msg.contains("pool exhausted") && !active.is_empty() {
        let victim_idx = active.len() - 1;
        let mut victim = active.swap_remove(victim_idx);
        tracing::warn!(
            "SSM pool full: preempting seq (slot={}, tokens={}) for new request",
            victim.seq.slot_idx,
            victim.output_tokens.len(),
        );
        send_error(model, &mut victim, "Preempted: server resource pressure");
    } else {
        tracing::error!("Prefill start error: {err_msg}");
    }
}

#[cfg(test)]
mod tests {
    use super::initial_prefill_chunk_budget;

    #[test]
    fn initial_prefill_never_consumes_decode_slot_capacity() {
        assert_eq!(initial_prefill_chunk_budget(8_192, 8_193), 8_192);
        assert_eq!(initial_prefill_chunk_budget(2_048, 8_193), 2_048);
    }
}
