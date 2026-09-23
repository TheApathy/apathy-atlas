// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only fixtures for the unregistered GLM ordinary-policy adapter RED gate.

use crate::scheduler::{ActiveSeq, Model, ResponseSink, SequenceState, SsmDecodeRing};
use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use std::time::Instant;

pub(super) struct NoGpuModel;
pub(super) const MODEL: NoGpuModel = NoGpuModel;

// process_seq_logits does not use its model argument. Every inference method
// deliberately panics: this fixture must never silently simulate model output.
impl Model for NoGpuModel {
    fn prefill(&self, _: &[u32], _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
        panic!("no GPU")
    }
    fn prefill_chunk(
        &self,
        _: &[u32],
        _: &mut SequenceState,
        _: usize,
        _: usize,
        _: bool,
        _: u64,
    ) -> Result<DevicePtr> {
        panic!("no GPU")
    }
    fn decode(&self, _: u32, _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
        panic!("no GPU")
    }
    fn decode_batch(&self, _: &[u32], _: &mut [&mut SequenceState], _: u64) -> Result<DevicePtr> {
        panic!("no GPU")
    }
    fn vocab_size(&self) -> usize {
        5
    }
    fn bind_gpu_to_thread(&self) -> Result<()> {
        panic!("no GPU")
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        panic!("no GPU")
    }
    fn copy_logits_to_host(&self, _: DevicePtr, _: &mut [u8]) -> Result<()> {
        panic!("no GPU")
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        panic!("no GPU")
    }
    fn argmax_on_device(&self, _: DevicePtr, _: u64) -> Result<u32> {
        panic!("no GPU")
    }
    fn argmax_batch(&self, _: DevicePtr, _: usize, _: u64) -> Result<Vec<u32>> {
        panic!("no GPU")
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        panic!("no GPU")
    }
    fn decode_verify(&self, _: &[u32], _: &mut SequenceState, _: u64) -> Result<Vec<u32>> {
        panic!("no GPU")
    }
    fn checkpoint_ssm_states(&self, _: &mut SequenceState) -> Result<()> {
        panic!("no GPU")
    }
    fn rollback_ssm_states(&self, _: &mut SequenceState, _: usize) -> Result<()> {
        panic!("no GPU")
    }
    fn generate_speculative(
        &self,
        _: &[u32],
        _: &spark_runtime::sampler::SamplingParams,
        _: usize,
    ) -> Result<spark_model::engine::GenerateResult> {
        panic!("no GPU")
    }
    fn has_proposer(&self) -> bool {
        false
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn decode_draft(&self, _: u32, _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
        panic!("no GPU")
    }
    fn cache_sequence(&self, _: &SequenceState) {
        panic!("no GPU")
    }
    fn free_sequence(&self, _: &mut SequenceState) -> Result<()> {
        panic!("no GPU")
    }
    fn compact_sequence(&self, _: &mut SequenceState, _: usize) -> Result<()> {
        panic!("no GPU")
    }
    fn detach_slot_for_reuse(&self, _: &mut SequenceState) {
        panic!("no GPU")
    }
    fn decode_verify_graphed(
        &self,
        _: &[u32; 2],
        _: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 2]> {
        panic!("no GPU")
    }
    fn decode_verify_graphed_k3(
        &self,
        _: &[u32; 3],
        _: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 3]> {
        panic!("no GPU")
    }
    fn decode_verify_graphed_k4(
        &self,
        _: &[u32; 4],
        _: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 4]> {
        panic!("no GPU")
    }
    fn save_hidden_for_mtp(&self, _: usize, _: u64) -> Result<()> {
        panic!("no GPU")
    }
    fn run_mtp_propose(
        &self,
        _: u32,
        _: usize,
        _: &mut SequenceState,
        _: u64,
    ) -> Result<Option<u32>> {
        panic!("no GPU")
    }
    fn run_mtp_propose_multi(
        &self,
        _: u32,
        _: usize,
        _: usize,
        _: &mut SequenceState,
        _: u64,
        _: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        panic!("no GPU")
    }
    fn trim_proposer_state(&self, _: &mut SequenceState, _: usize, _: u64) -> Result<()> {
        panic!("no GPU")
    }
}

pub(super) fn row(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

pub(super) fn sequence() -> ActiveSeq {
    let now = Instant::now();
    let mut seq = SequenceState::for_model_owned_state(0);
    seq.tokens = vec![0, 0];
    seq.seq_len = 2;
    seq.kv_valid_tokens = 2;
    seq.prompt_len = 2;
    ActiveSeq {
        seq,
        session_hash: 0,
        last_token: 0,
        output_tokens: vec![],
        remaining: 20,
        min_tokens: 0,
        eos_tokens: vec![],
        finished: false,
        terminal_error: None,
        guard_stop: None,
        param_close_pending: 0,
        sink: ResponseSink::Blocking(None),
        cancel_flag: None,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        repetition_penalty_window: 256,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_sequence_breakers: vec![],
        logit_bias: vec![],
        inside_thinking: false,
        enable_thinking: false,
        thinking_budget: None,
        repetition_detection: None,
        spontaneous_think_budget: 128,
        thinking_tokens: 0,
        force_end_thinking: false,
        think_force_closed: false,
        sentence_defer_count: 0,
        consecutive_confident: 0,
        in_code_fence: false,
        think_end_token: None,
        think_start_token: None,
        think_ended: false,
        think_just_ended: false,
        post_think_emitted: 0,
        spec_adapt: Default::default(),
        think_skip_count: 0,
        tool_call_end_token: None,
        require_tool_call: false,
        tool_request: false,
        tools_present: false,
        tool_call_start_token: None,
        tool_call_opened: false,
        inside_tool_body: false,
        tool_call_completed: false,
        post_completion_tool_opens: 0,
        tool_body_streak_tokens: 0,
        inside_parameter_body: false,
        param_body_chars_emitted: 0,
        suppress_tool_call: false,
        disable_mtp: false,
        content_started: false,
        content_tokens: 0,
        prose_tokens_since_last_tool: 0,
        think_watchdog_fires: 0,
        rollback_count: 0,
        ssm_rollback_ring: SsmDecodeRing::new(0),
        grammar_state: None,
        pending_drafts: vec![],
        last_token_time: now,
        request_start: now,
        decode_start: now,
        seed: Some(42),
        top_logprobs: None,
        logprobs_data: vec![],
        timeout_at: None,
        adaptive: crate::adaptive_sampler::AdaptiveSamplingState::new(0.0),
        cached_prompt_tokens: 0,
    }
}

pub(super) fn grammar() -> crate::grammar::GrammarState {
    let vocab: Vec<String> = ["{", "}", "x", "<eos>", "</think>"]
        .into_iter()
        .map(String::from)
        .collect();
    let mut engine = crate::grammar::GrammarEngine::new(&vocab, &[3]).unwrap();
    let compiled = engine.compile_json_grammar().unwrap();
    crate::grammar::GrammarState::new(&compiled, vocab.len())
        .unwrap()
        .with_stop_tokens(&[3])
}

/// Compare rollback-sensitive public policy state, not timer/GPU ownership.
pub(super) fn fingerprint(a: &ActiveSeq) -> Vec<String> {
    vec![
        format!(
            "{:?}/{}/{}/{:?}",
            a.output_tokens, a.last_token, a.remaining, a.logit_bias
        ),
        format!(
            "{:?}/{}/{}/{}",
            a.seq.tokens, a.seq.seq_len, a.seq.kv_valid_tokens, a.finished
        ),
        format!(
            "{}/{}/{:?}/{}/{}",
            a.inside_thinking,
            a.think_ended,
            a.thinking_budget,
            a.thinking_tokens,
            a.think_skip_count
        ),
        format!(
            "{}/{}/{}/{}/{}",
            a.force_end_thinking,
            a.think_force_closed,
            a.sentence_defer_count,
            a.consecutive_confident,
            a.in_code_fence
        ),
        format!(
            "{}/{}/{}/{}/{}",
            a.think_just_ended,
            a.post_think_emitted,
            a.require_tool_call,
            a.tool_call_opened,
            a.inside_tool_body
        ),
        format!(
            "{}/{}/{}/{}/{}",
            a.tool_call_completed,
            a.post_completion_tool_opens,
            a.tool_body_streak_tokens,
            a.inside_parameter_body,
            a.param_close_pending
        ),
        format!(
            "{}/{}/{}/{}/{}",
            a.param_body_chars_emitted,
            a.content_started,
            a.content_tokens,
            a.prose_tokens_since_last_tool,
            a.think_watchdog_fires
        ),
        format!(
            "{:?}/{:?}/{:?}/{}",
            a.guard_stop,
            a.terminal_error,
            a.adaptive.zone,
            a.adaptive.effective_temperature()
        ),
        format!(
            "{:?}/{}",
            a.grammar_state.as_ref().map(|g| g.num_history_steps()),
            a.logprobs_data.len()
        ),
        format!(
            "{:?}/{:?}/{:?}",
            a.last_token_time, a.request_start, a.decode_start
        ),
    ]
}
