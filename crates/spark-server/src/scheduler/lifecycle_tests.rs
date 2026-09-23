// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-free finish-reason parity tests shared by blocking and streaming sinks.

use super::*;

mod model_support {
    mod implementation {
        use crate::scheduler::proposal_lifecycle::SchedulerProposalFrame;
        use spark_model::layers::DDTreePayload;

        include!("proposal_lifecycle_model_test_support.rs");
    }
    pub(super) fn model() -> impl spark_model::traits::Model {
        implementation::TestModel::new(Ok(Vec::new()), None)
    }

    pub(super) fn seq_state() -> spark_model::traits::SequenceState {
        implementation::seq_state()
    }

    /// A model whose (single) logits row is `row`.
    pub(super) fn logits_model(row: Vec<f32>) -> impl spark_model::traits::Model {
        let m = implementation::TestModel::new(Ok(Vec::new()), None);
        *m.host_logits.lock().unwrap() = row;
        m
    }

    /// A model whose greedy decode step returns `tokens`.
    pub(super) fn greedy_model(tokens: Vec<u32>) -> impl spark_model::traits::Model {
        let m = implementation::TestModel::new(Ok(Vec::new()), None);
        *m.argmax.lock().unwrap() = tokens;
        m
    }
}

const EOS: u32 = 17;
const TOOL_CLOSE: u32 = 23;
const IM_START: u32 = u32::MAX - 1;
const CONTENT: u32 = 31;
static HARD_STOP_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct HardStopReset;

impl Drop for HardStopReset {
    fn drop(&mut self) {
        set_im_start_hard_stop(0);
    }
}

fn active_seq(sink: ResponseSink, last_token: u32) -> ActiveSeq {
    let now = std::time::Instant::now();
    ActiveSeq {
        seq: model_support::seq_state(),
        session_hash: 0,
        last_token,
        output_tokens: vec![last_token],
        max_output_tokens: 1,
        remaining: 0,
        min_tokens: 0,
        eos_tokens: vec![EOS],
        finished: true,
        sink,
        cancel_flag: None,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        repetition_penalty_window: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 3,
        dry_sequence_breakers: Vec::new(),
        logit_bias: Vec::new(),
        inside_thinking: false,
        enable_thinking: false,
        thinking_budget: None,
        spontaneous_think_budget: 512,
        thinking_tokens: 0,
        force_end_thinking: false,
        consecutive_confident: 0,
        in_code_fence: false,
        think_end_token: None,
        think_start_token: None,
        think_ended: true,
        think_just_ended: false,
        think_skip_count: 0,
        post_think_gate_steps: 0,
        tool_call_end_token: Some(TOOL_CLOSE),
        require_tool_call: false,
        tool_call_start_token: None,
        tool_call_opened: false,
        inside_tool_body: false,
        suppress_tool_call: false,
        disable_mtp: false,
        content_started: true,
        content_tokens: 1,
        prose_tokens_since_last_tool: 0,
        think_watchdog_fires: 0,
        rollback_count: 0,
        ssm_rollback_ring: SsmDecodeRing::new(0),
        grammar_state: None,
        pending_drafts: Vec::new(),
        draft_origin: DraftOrigin::default(),
        last_verify_accepted: 0,
        self_context: Default::default(),
        pending_tree_payload: None,
        last_token_time: now,
        request_start: now,
        decode_start: now,
        seed: None,
        top_logprobs: None,
        logprobs_data: Vec::new(),
        timeout_at: None,
        adaptive: crate::adaptive_sampler::AdaptiveSamplingState::new(0.0),
        cached_prompt_tokens: 0,
        difficulty_probe: Default::default(),
    }
}

fn delivered_finish_reasons(last_token: u32) -> (String, String) {
    let model = model_support::model();

    let (blocking_tx, mut blocking_rx) = tokio::sync::oneshot::channel();
    let mut blocking = active_seq(ResponseSink::Blocking(Some(blocking_tx)), last_token);
    finish_sequence_with_cache(&model, &mut blocking, false);
    let blocking_reason = blocking_rx
        .try_recv()
        .expect("blocking response must be delivered")
        .expect("blocking finish must succeed")
        .finish_reason;

    let (streaming_tx, mut streaming_rx) = tokio::sync::mpsc::channel(1);
    let mut streaming = active_seq(ResponseSink::Streaming(streaming_tx), last_token);
    finish_sequence_with_cache(&model, &mut streaming, false);
    let streaming_reason = match streaming_rx
        .try_recv()
        .expect("streaming Done must be delivered")
    {
        StreamEvent::Done { finish_reason, .. } => finish_reason,
        _ => panic!("finish must deliver a Done event"),
    };

    (blocking_reason, streaming_reason)
}

#[test]
fn blocking_and_streaming_sinks_deliver_the_shared_finish_reason() {
    let _guard = HARD_STOP_TEST_LOCK.lock().expect("hard-stop test lock");
    let _reset = HardStopReset;
    set_im_start_hard_stop(IM_START);

    for (last_token, expected) in [(IM_START, "stop"), (TOOL_CLOSE, "tool_calls")] {
        let (blocking_reason, streaming_reason) = delivered_finish_reasons(last_token);
        assert_eq!(blocking_reason, expected);
        assert_eq!(streaming_reason, blocking_reason);
    }
}

#[test]
fn shared_sink_finish_reason_obeys_protocol_stop_boundaries() {
    let _guard = HARD_STOP_TEST_LOCK.lock().expect("hard-stop test lock");
    let _reset = HardStopReset;
    set_im_start_hard_stop(0);

    assert_eq!(finish_reason(&[EOS], &[EOS], Some(TOOL_CLOSE)), "stop");
    assert_eq!(
        finish_reason(&[TOOL_CLOSE], &[EOS], Some(TOOL_CLOSE)),
        "tool_calls"
    );
    assert_eq!(
        finish_reason(&[CONTENT], &[EOS], Some(TOOL_CLOSE)),
        "length",
        "an ordinary token at the request cap must remain a length stop"
    );
    assert_eq!(
        finish_reason(&[], &[EOS], Some(TOOL_CLOSE)),
        "length",
        "an empty history must not alias an unregistered ChatML boundary"
    );
    assert_eq!(
        finish_reason(&[IM_START], &[EOS], Some(TOOL_CLOSE)),
        "length",
        "an unregistered ChatML token must not become a stop"
    );
    assert_eq!(
        finish_reason(&[], &[EOS], None),
        "length",
        "an empty history must not alias an absent tool-close token"
    );
    assert_eq!(
        finish_reason(&[CONTENT], &[EOS], None),
        "length",
        "an absent tool-close token can never classify as tool_calls"
    );

    set_im_start_hard_stop(IM_START);
    assert_eq!(
        finish_reason(&[IM_START], &[EOS], Some(TOOL_CLOSE)),
        "stop",
        "the exact registered ChatML boundary must be an ordinary stop"
    );
    assert_eq!(
        finish_reason(&[IM_START - 1], &[EOS], Some(TOOL_CLOSE)),
        "length",
        "a token different from the registered boundary must remain length"
    );
    assert_eq!(
        finish_reason(&[IM_START], &[EOS], Some(IM_START)),
        "stop",
        "the registered ChatML stop must take precedence over tool close"
    );
}

/// A stream-side stop string sets the request's cancel flag. The serial decode
/// path must end the sequence there (dropping the sample), as speculative
/// `emit_token` already did; before the fix it committed the token and ran on.
#[test]
fn a_cancelled_sequence_stops_on_the_serial_decode_path() {
    let _guard = HARD_STOP_TEST_LOCK.lock().expect("hard-stop test lock");
    let _reset = HardStopReset;
    set_im_start_hard_stop(0);
    let run = |cancelled: bool| {
        let model = model_support::greedy_model(vec![CONTENT]);
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let mut a = active_seq(ResponseSink::Blocking(Some(tx)), CONTENT);
        a.finished = false;
        a.max_output_tokens = 100;
        a.remaining = 100;
        a.think_ended = false;
        a.cancel_flag = Some(std::sync::Arc::new(std::sync::atomic::AtomicBool::new(cancelled)));
        let mut active = vec![a];
        process_decode_logits(
            &model,
            &mut active,
            spark_runtime::gpu::DevicePtr::NULL,
            std::time::Instant::now(),
            None,
            None,
            None,
            None,
            Some(TOOL_CLOSE),
            &[],
            false,
        );
        let a = active.pop().expect("sequence");
        (a.finished, a.output_tokens.len())
    };
    // cancelled: finished, the sampled token is not committed
    assert_eq!(run(true), (true, 1));
    // CONTROL: without the flag the same step commits the token and continues
    assert_eq!(run(false), (false, 2));
}

/// The first generated token is sampled from the prefill logits; with
/// `top_logprobs` requested its logprobs entry must come with it (it used to be
/// missing, so logprobs.content had completion_tokens - 1 entries), and its
/// stream event must carry them.
#[test]
fn the_first_token_gets_its_logprobs_from_the_prefill_logits() {
    let model = model_support::logits_model(vec![0.0, 3.0, 1.0, 2.0]);
    let null = spark_runtime::gpu::DevicePtr::NULL;
    let (tok, lp) = sample_token_with_logprobs(&model, null, 0.0, 0, 1.0, &[], Some(2)).unwrap();
    assert_eq!(tok, 1);
    let lp = lp.expect("first-token logprobs");
    assert_eq!(lp.token_id, 1);
    assert_eq!(lp.top.iter().map(|t| t.0).collect::<Vec<_>>(), vec![1, 3]);
    let z = [0.0f32, 3.0, 1.0, 2.0].iter().map(|v| v.exp()).sum::<f32>().ln();
    assert!((lp.logprob - (3.0 - z)).abs() < 1e-5);
    assert!(matches!(
        first_token_event(tok, &Some(lp)),
        StreamEvent::TokenWithLogprobs(1, _)
    ));
    // Sampled (T>0) path returns an entry for whatever it drew.
    let (t2, lp2) = sample_token_with_logprobs(&model, null, 1.0, 0, 1.0, &[], Some(1)).unwrap();
    assert_eq!(lp2.expect("sampled first-token logprobs").token_id, t2);
    // CONTROL: without top_logprobs nothing is produced (and the plain event is sent),
    // which is what every request got before -- one entry short.
    let (_, none) = sample_token_with_logprobs(&model, null, 0.5, 0, 1.0, &[], None).unwrap();
    assert!(none.is_none());
    assert!(matches!(first_token_event(1, &None), StreamEvent::Token(1)));
}
