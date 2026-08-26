// SPDX-License-Identifier: AGPL-3.0-only

//! Pure, GPU-free swap metadata round-trip tests.
use super::*;
use spark_model::traits::SequenceState;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;
fn seq_state() -> SequenceState {
    SequenceState {
        tokens: vec![3, 5, 8],
        block_table: vec![13, 21],
        seq_len: 3,
        qwen4_qsa_required: false,
        layer_states: Vec::new(),
        proposer_state: None,
        proposer_state_alt: None,
        slot_idx: 0,
        marconi_skip_to: 0,
        session_hash: 0,
        chunked_prefill_meta: None,
        cached_prefix_tokens: 0,
        prompt_len: 3,
        disk_block_ids: Vec::new(),
        mtp_lastk_host_buf: Vec::new(),
        mtp_lastk_host_filled: 0,
        mtp_lastk_end_abs: 0,
        disk_last_offloaded_per_layer: Vec::new(),
    }
}
fn restored_seq_state() -> SequenceState {
    let mut seq = seq_state();
    seq.tokens.clear();
    seq.block_table.clear();
    seq.seq_len = 0;
    seq
}
pub(super) type Fixture = (
    ActiveSeq,
    Arc<AtomicBool>,
    tokio::sync::mpsc::Receiver<StreamEvent>,
);
pub(super) fn active_seq(committed_probe: bool) -> Fixture {
    let cancel = Arc::new(AtomicBool::new(false));
    let (tx, rx) = tokio::sync::mpsc::channel(4);
    let now = Instant::now();
    let output_tokens = vec![34; 64];
    let mut adaptive = crate::adaptive_sampler::AdaptiveSamplingState::new(0.9);
    adaptive.update_zone(false, false, true);
    for _ in 0..9 {
        adaptive.observe_entropy(&[8.0, -8.0]);
    }
    adaptive.update_lz_ratio(&output_tokens);
    let mut difficulty_probe = crate::scheduler::thinking_efficiency::DifficultyProbe::default();
    for _ in 0..if committed_probe { 48 } else { 47 } {
        difficulty_probe.observe(0.75);
    }
    let thinking_budget = Some(if committed_probe {
        difficulty_probe.commit(512).expect("probe must commit")
    } else {
        512
    });
    let mut ssm_rollback_ring = SsmDecodeRing::new(2);
    assert_eq!(ssm_rollback_ring.record(32), Some(0));

    (
        ActiveSeq {
            seq: seq_state(),
            session_hash: 55,
            last_token: 34,
            output_tokens,
            max_output_tokens: 1000,
            remaining: 811,
            min_tokens: 7,
            eos_tokens: vec![99],
            finished: true,
            sink: ResponseSink::Streaming(tx),
            cancel_flag: Some(Arc::clone(&cancel)),
            temperature: 0.9,
            top_k: 17,
            top_p: 0.91,
            top_n_sigma: 0.2,
            min_p: 0.03,
            repetition_penalty: 0.87,
            repetition_penalty_window: 19,
            presence_penalty: 0.4,
            frequency_penalty: 0.6,
            lz_penalty: 0.42,
            dry_multiplier: 0.9,
            dry_base: 1.6,
            dry_allowed_length: 4,
            dry_sequence_breakers: vec![89],
            logit_bias: vec![(144, -0.5)],
            inside_thinking: false,
            enable_thinking: true,
            thinking_budget,
            spontaneous_think_budget: 511,
            thinking_tokens: 37,
            force_end_thinking: true,
            consecutive_confident: 6,
            in_code_fence: true,
            think_end_token: Some(233),
            think_start_token: Some(234),
            think_ended: true,
            think_just_ended: true,
            think_skip_count: 3,
            post_think_gate_steps: 4,
            tool_call_end_token: Some(377),
            require_tool_call: false,
            tool_call_start_token: Some(378),
            tool_call_opened: true,
            inside_tool_body: true,
            suppress_tool_call: true,
            disable_mtp: true,
            content_started: true,
            content_tokens: 61,
            prose_tokens_since_last_tool: 19,
            think_watchdog_fires: 2,
            rollback_count: 1,
            ssm_rollback_ring,
            grammar_state: None,
            pending_drafts: vec![610, 987],
            draft_origin: DraftOrigin::default(),
            last_verify_accepted: 0,
            self_context: Default::default(),
            pending_tree_payload: Some(spark_model::layers::DDTreePayload {
                tree_token_ids: vec![610, 987],
                parent_indices: vec![-1, 0],
            }),
            last_token_time: now - Duration::from_secs(9),
            request_start: now - Duration::from_secs(30),
            decode_start: now - Duration::from_secs(20),
            seed: Some(1597),
            top_logprobs: Some(3),
            logprobs_data: Vec::new(),
            timeout_at: Some(now + Duration::from_secs(30)),
            adaptive,
            cached_prompt_tokens: 11,
            difficulty_probe,
        },
        cancel,
        rx,
    )
}
pub(super) fn round_trip(active: ActiveSeq, resumed_at: Instant) -> ActiveSeq {
    let tokens = active.seq.tokens.clone();
    let seq_len = active.seq.seq_len;
    let num_blocks = active.seq.block_table.len();
    let swapped = pack_swapped_seq(active, tokens, seq_len, num_blocks, 41);
    assert_eq!(swapped.num_blocks, 2);
    assert_eq!(swapped.swap_id, 41);
    restore_swapped_seq(restored_seq_state(), swapped, 3, resumed_at)
}
#[test]
fn round_trip_preserves_live_policy_and_resets_gpu_transients() {
    let (active, external_cancel, mut rx) = active_seq(false);
    let request_start = active.request_start;
    let decode_start = active.decode_start;
    let timeout_at = active.timeout_at;
    let adaptive_temperature = active.adaptive.effective_temperature();
    let difficulty_mean = active.difficulty_probe.mean_confidence();
    let resumed_at = Instant::now() + Duration::from_secs(2);
    let mut resumed = round_trip(active, resumed_at);
    assert_eq!(resumed.seq.tokens, vec![3, 5, 8]);
    assert_eq!(resumed.seq.seq_len, 3);
    assert_eq!(resumed.output_tokens, vec![34; 64]);
    assert_eq!(resumed.max_output_tokens, 1000);
    assert_eq!(resumed.remaining, 811);
    assert_eq!(resumed.min_tokens, 7);
    assert_eq!(resumed.eos_tokens, vec![99]);
    assert_eq!(resumed.temperature.to_bits(), 0.9_f32.to_bits());
    assert_eq!(resumed.top_k, 17);
    assert_eq!(resumed.top_p.to_bits(), 0.91_f32.to_bits());
    assert_eq!(resumed.thinking_budget, Some(512));
    assert_eq!(resumed.spontaneous_think_budget, 511);
    assert_eq!(resumed.thinking_tokens, 37);
    assert!(resumed.force_end_thinking);
    assert_eq!(resumed.repetition_penalty_window, 19);
    assert_eq!(resumed.lz_penalty.to_bits(), 0.42_f32.to_bits());
    assert!(resumed.content_started);
    assert_eq!(resumed.content_tokens, 61);
    assert_eq!(resumed.prose_tokens_since_last_tool, 19);
    assert_eq!(resumed.think_watchdog_fires, 2);
    assert_eq!(resumed.rollback_count, 1);
    assert!(resumed.inside_tool_body);
    assert_eq!(
        resumed.adaptive.effective_temperature().to_bits(),
        adaptive_temperature.to_bits()
    );
    assert_eq!(resumed.difficulty_probe.mean_confidence(), difficulty_mean);
    assert!(!resumed.difficulty_probe.ready());
    resumed.difficulty_probe.observe(0.75);
    assert!(resumed.difficulty_probe.ready());
    assert!(resumed.difficulty_probe.commit(512).is_some());
    assert_eq!(resumed.difficulty_probe.commit(512), None);
    assert!(!resumed.finished);
    assert!(resumed.grammar_state.is_none());
    assert!(resumed.pending_drafts.is_empty());
    assert!(resumed.pending_tree_payload.is_none());
    assert!(resumed.ssm_rollback_ring.is_enabled());
    assert_eq!(resumed.ssm_rollback_ring.len(), 0);
    assert_eq!(resumed.ssm_rollback_ring.record(65), Some(0));
    assert_eq!(resumed.ssm_rollback_ring.record(66), Some(1));
    assert_eq!(resumed.ssm_rollback_ring.record(67), Some(2));
    assert_eq!(resumed.ssm_rollback_ring.len(), 3);
    assert_eq!(resumed.last_token_time, resumed_at);
    assert_eq!(resumed.request_start, request_start);
    assert_eq!(resumed.decode_start, decode_start);
    assert_eq!(resumed.timeout_at, timeout_at);

    let restored_cancel = resumed
        .cancel_flag
        .as_ref()
        .expect("cancel flag must survive");
    assert!(Arc::ptr_eq(restored_cancel, &external_cancel));
    let ResponseSink::Streaming(tx) = &resumed.sink else {
        panic!("sink kind changed")
    };
    tx.try_send(StreamEvent::Error("sink-sentinel".into()))
        .unwrap();
    assert!(matches!(rx.try_recv(), Ok(StreamEvent::Error(msg)) if msg == "sink-sentinel"));
    external_cancel.store(true, Ordering::Release);
    let output_before_cancel = resumed.output_tokens.clone();
    let remaining_before_cancel = resumed.remaining;
    crate::scheduler::emit_step::emit_token(&mut resumed, 42, None);
    assert!(resumed.finished);
    assert_eq!(resumed.output_tokens, output_before_cancel);
    assert_eq!(resumed.remaining, remaining_before_cancel);
}
#[test]
fn round_trip_keeps_committed_difficulty_probe_closed_with_evidence() {
    let (active, _cancel, _rx) = active_seq(true);
    let mean_before = active
        .difficulty_probe
        .mean_confidence()
        .expect("fixture evidence");
    let committed_budget = active.thinking_budget;

    let mut resumed = round_trip(active, Instant::now());
    assert_eq!(
        resumed
            .difficulty_probe
            .mean_confidence()
            .expect("round-trip must retain committed probe evidence")
            .to_bits(),
        mean_before.to_bits()
    );
    assert!(!resumed.difficulty_probe.ready());
    assert_eq!(resumed.difficulty_probe.commit(512), None);
    assert_eq!(resumed.thinking_budget, committed_budget);
}
