// SPDX-License-Identifier: AGPL-3.0-only

//! Independent ordinary-entry reference: real post-processing, synthetic host
//! logits only. No CUDA backend, inference output fabrication, or environment mutation.

use super::adapter::{NeedOrdinaryReplay, OrdinaryEffectRequest};
use super::fixture::{MODEL, fingerprint, row, sequence};
use super::{OrdinaryVerifyPolicy, context, options};
use crate::scheduler::{Model, SequenceState};
use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

struct HostLogits(Vec<u8>);

macro_rules! forbidden_gpu_methods {
    ($(fn $name:ident($($arg:ident: $ty:ty),*) -> $ret:ty;)*) => {
        $(fn $name(&self, $($arg: $ty),*) -> $ret { MODEL.$name($($arg),*) })*
    };
}

impl Model for HostLogits {
    fn vocab_size(&self) -> usize {
        5
    }
    fn copy_logits_to_host(&self, _: DevicePtr, dst: &mut [u8]) -> Result<()> {
        anyhow::ensure!(dst.len() == self.0.len(), "host fixture extent mismatch");
        dst.copy_from_slice(&self.0);
        Ok(())
    }
    forbidden_gpu_methods! {
        fn prefill(tokens: &[u32], seq: &mut SequenceState, stream: u64) -> Result<DevicePtr>;
        fn prefill_chunk(tokens: &[u32], seq: &mut SequenceState, start: usize, len: usize, last: bool, stream: u64) -> Result<DevicePtr>;
        fn decode(token: u32, seq: &mut SequenceState, stream: u64) -> Result<DevicePtr>;
        fn decode_batch(tokens: &[u32], seqs: &mut [&mut SequenceState], stream: u64) -> Result<DevicePtr>;
        fn bind_gpu_to_thread() -> Result<()>;
        fn alloc_sequence() -> Result<SequenceState>;
        fn logits_buffer_ptr() -> DevicePtr;
        fn argmax_on_device(ptr: DevicePtr, stream: u64) -> Result<u32>;
        fn argmax_batch(ptr: DevicePtr, rows: usize, stream: u64) -> Result<Vec<u32>>;
        fn hidden_after_norm() -> DevicePtr;
        fn decode_verify(tokens: &[u32], seq: &mut SequenceState, stream: u64) -> Result<Vec<u32>>;
        fn checkpoint_ssm_states(seq: &mut SequenceState) -> Result<()>;
        fn rollback_ssm_states(seq: &mut SequenceState, accepted: usize) -> Result<()>;
        fn generate_speculative(tokens: &[u32], params: &spark_runtime::sampler::SamplingParams, drafts: usize) -> Result<spark_model::engine::GenerateResult>;
        fn has_proposer() -> bool;
        fn has_self_speculative() -> bool;
        fn decode_draft(token: u32, seq: &mut SequenceState, stream: u64) -> Result<DevicePtr>;
        fn cache_sequence(seq: &SequenceState) -> ();
        fn free_sequence(seq: &mut SequenceState) -> Result<()>;
        fn compact_sequence(seq: &mut SequenceState, slot: usize) -> Result<()>;
        fn detach_slot_for_reuse(seq: &mut SequenceState) -> ();
        fn decode_verify_graphed(tokens: &[u32; 2], seq: &mut SequenceState, stream: u64) -> Result<[u32; 2]>;
        fn decode_verify_graphed_k3(tokens: &[u32; 3], seq: &mut SequenceState, stream: u64) -> Result<[u32; 3]>;
        fn decode_verify_graphed_k4(tokens: &[u32; 4], seq: &mut SequenceState, stream: u64) -> Result<[u32; 4]>;
        fn save_hidden_for_mtp(index: usize, stream: u64) -> Result<()>;
        fn run_mtp_propose(token: u32, position: usize, seq: &mut SequenceState, stream: u64) -> Result<Option<u32>>;
        fn run_mtp_propose_multi(token: u32, position: usize, drafts: usize, seq: &mut SequenceState, stream: u64, mask: Option<&[i32]>) -> Result<Vec<u32>>;
        fn trim_proposer_state(seq: &mut SequenceState, accepted: usize, stream: u64) -> Result<()>;
    }
}

#[test]
fn adapter_transition_matches_actual_ordinary_entry_not_emit_token() {
    for case in [
        "content",
        "thinking_close",
        "spontaneous",
        "tool_open",
        "tool_close",
        "eos",
        "budget",
    ] {
        let mut reference = sequence();
        let mut candidate = sequence();
        let mut ctx = context();
        let token = match case {
            "thinking_close" => {
                ctx.think_end_token = Some(1);
                1
            }
            "spontaneous" => {
                ctx.think_start_token = Some(4);
                4
            }
            "tool_open" => {
                ctx.tool_call_start_token = Some(3);
                3
            }
            "tool_close" => {
                ctx.tool_call_end_token = Some(3);
                3
            }
            "eos" => 3,
            _ => 2,
        };
        for a in [&mut reference, &mut candidate] {
            a.top_logprobs = Some(2); // forces the real ordinary host sampler
            a.think_end_token = ctx.think_end_token;
            a.think_start_token = ctx.think_start_token;
            a.tool_call_start_token = ctx.tool_call_start_token;
            a.tool_call_end_token = ctx.tool_call_end_token;
            match case {
                "thinking_close" => {
                    a.inside_thinking = true;
                    a.in_code_fence = true;
                    a.consecutive_confident = 5;
                }
                "spontaneous" => a.think_watchdog_fires = 2,
                "tool_open" => {
                    a.require_tool_call = true;
                    a.tools_present = true;
                }
                "tool_close" => {
                    a.inside_tool_body = true;
                    a.tools_present = true;
                    a.think_ended = true;
                }
                "eos" => a.eos_tokens = vec![3],
                "budget" => a.remaining = 1,
                _ => {}
            }
        }
        let mut scores = [0.0; 5];
        scores[token] = 30.0;
        let bytes = row(&scores);
        let model = HostLogits(bytes.clone());
        // Ordinary entry sees the just-decoded anchor already in its target
        // prefix; the speculative policy is forbidden to publish that prefix.
        reference.seq.tokens.push(0);
        reference.seq.seq_len += 1;
        reference.seq.kv_valid_tokens += 1;
        let mut active = vec![reference];
        crate::scheduler::decode_logits_step::process_decode_logits(
            &model,
            &mut active,
            DevicePtr::NULL,
            std::time::Instant::now(),
            ctx.think_end_token,
            ctx.think_start_token,
            None,
            ctx.tool_call_start_token,
            ctx.tool_call_end_token,
            false,
        );
        let mut expected = fingerprint(&active[0]);
        // Exclude only target-prefix publication and timestamps (not policy).
        expected.pop();
        expected.remove(1);
        let mut policy =
            OrdinaryVerifyPolicy::new(&model, &mut candidate, &ctx, options(false)).unwrap();
        policy.checkpoint().unwrap();
        assert_eq!(policy.pick(0, &bytes).unwrap(), token as u32, "{case}");
        assert_eq!(
            policy.advance(token as u32).unwrap(),
            !active[0].finished,
            "{case}"
        );
        let mut actual = fingerprint(policy.sequence());
        actual.pop();
        actual.remove(1);
        assert_eq!(
            actual, expected,
            "{case}: actual ordinary entry is authoritative"
        );
        let logprobs = |a: &crate::scheduler::ActiveSeq| {
            a.logprobs_data
                .iter()
                .map(|lp| {
                    (
                        lp.token_id,
                        lp.logprob.to_bits(),
                        lp.top
                            .iter()
                            .map(|&(id, value)| (id, value.to_bits()))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(logprobs(policy.sequence()), logprobs(&active[0]));
        policy.restore().unwrap();
        let prepared = policy.take_prepared().unwrap();
        drop(policy);
        // CPU installation fixture only: the model transaction tests separately
        // require a successful commit receipt before this host publication.
        candidate.seq.tokens.push(0);
        candidate.seq.seq_len += 1;
        candidate.seq.kv_valid_tokens += 1;
        let installed = prepared
            .install(&mut candidate, &[token as u32], 3)
            .unwrap();
        assert!(
            installed.is_empty(),
            "blocking fixture has no stream events"
        );
        let mut actual = fingerprint(&candidate);
        actual.pop();
        actual.remove(1);
        assert_eq!(
            actual, expected,
            "{case}: retained policy installs without resampling"
        );
        assert_eq!(logprobs(&candidate), logprobs(&active[0]));
    }
}

#[test]
fn rollback_request_is_typed_replay_not_error_string_or_terminal_substitution() {
    assert!(!crate::scheduler::helpers::disable_watchdogs());
    let mut live = sequence();
    live.tool_request = true;
    live.prose_tokens_since_last_tool = u32::MAX;
    let before = fingerprint(&live);
    let ctx = context();
    {
        let mut policy =
            OrdinaryVerifyPolicy::new(&MODEL, &mut live, &ctx, options(false)).unwrap();
        policy.checkpoint().unwrap();
        assert_eq!(
            policy.pick(0, &row(&[0.0, 20.0, 0.0, 0.0, 0.0])).unwrap(),
            1
        );
        let error = policy
            .advance(1)
            .expect_err("watchdog must request a genuine ordinary step");
        let replay = error
            .downcast_ref::<NeedOrdinaryReplay>()
            .expect("typed replay classification");
        assert!(matches!(
            replay.effect,
            OrdinaryEffectRequest::Rollback { .. }
        ));
        assert!(
            !policy.sequence().finished,
            "do not turn re-steer into terminal generation"
        );
        assert_eq!(policy.sequence().seq.tokens, [0, 0]);
        assert_eq!(policy.sequence().seq.seq_len, 2);
        policy.restore().unwrap();
    }
    assert_eq!(fingerprint(&live), before);
}
