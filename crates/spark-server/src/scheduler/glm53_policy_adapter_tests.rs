// SPDX-License-Identifier: AGPL-3.0-only

//! Unregistered RED tests. Root may register this test module in an isolated
//! CPU snapshot; production registration and GPU use are deliberately absent.
//! Uses the real ordinary sampler and xgrammar, not PR834's emit_token oracle.
//! Run without ATLAS_FORCE_TEMP_ZERO or other diagnostic sampler overrides.

#[path = "glm53_verify_policy_adapter.rs"]
mod adapter;
#[path = "glm53_policy_adapter_fixture.rs"]
mod fixture;
#[path = "glm53_ordinary_transition_tests.rs"]
mod ordinary_transition_tests;
#[path = "glm53_policy_tool_boundary_tests.rs"]
mod tool_boundary_tests;
#[path = "glm53_policy_adapter_transaction_tests.rs"]
mod transaction_tests;

use crate::scheduler::logit_processors::LogitsContext;
use crate::scheduler::{ActiveSeq, ResponseSink};
use adapter::{OrdinaryVerifyPolicy, PolicyOptions, policy_device_argmax_enabled_from};
use fixture::{MODEL, fingerprint, grammar, row, sequence};

fn context() -> LogitsContext {
    LogitsContext {
        think_end_token: None,
        think_start_token: None,
        tool_call_start_token: None,
        tool_call_end_token: None,
    }
}

fn options(adaptive_sampling: bool) -> PolicyOptions {
    PolicyOptions {
        vocab_size: 5,
        max_seq_len: 64,
        adaptive_sampling,
        code_fence_token: None,
    }
}

#[test]
fn compact_policy_selector_is_strict_and_default_off() {
    assert!(!policy_device_argmax_enabled_from(None).unwrap());
    assert!(!policy_device_argmax_enabled_from(Some("0")).unwrap());
    assert!(policy_device_argmax_enabled_from(Some("1")).unwrap());
    for invalid in ["", "true", "2", " 1"] {
        assert!(policy_device_argmax_enabled_from(Some(invalid)).is_err());
    }
}

#[test]
fn compact_policy_admission_is_neutral_greedy_only() {
    let ctx = context();
    let mut live = sequence();
    assert_eq!(
        crate::scheduler::ordinary_greedy::policy_raw_argmax_exclusions(&live, &ctx, false),
        Some([u32::MAX; 2])
    );
    live.logit_bias = vec![(2, 1.0)];
    assert_eq!(
        crate::scheduler::ordinary_greedy::policy_raw_argmax_exclusions(&live, &ctx, false),
        None
    );
    live.logit_bias.clear();
    live.grammar_state = Some(grammar());
    assert_eq!(
        crate::scheduler::ordinary_greedy::policy_raw_argmax_exclusions(&live, &ctx, false),
        None
    );
    live.grammar_state = None;
    assert_eq!(
        crate::scheduler::ordinary_greedy::policy_raw_argmax_exclusions(&live, &ctx, true),
        None
    );
}

#[test]
fn compact_policy_carries_exact_post_close_thinking_exclusions() {
    let mut live = sequence();
    live.think_ended = true;
    live.think_start_token = Some(4);
    let mut ctx = context();
    ctx.think_end_token = Some(1);
    assert_eq!(
        crate::scheduler::ordinary_greedy::policy_raw_argmax_exclusions(&live, &ctx, false),
        Some([1, 4])
    );
}

fn ordinary(a: &mut ActiveSeq, bytes: &[u8], ctx: &LogitsContext, adaptive: bool) -> u32 {
    crate::scheduler::decode_logits_seq::process_seq_logits(
        &MODEL, a, bytes, 0, 5, 2, false, ctx, adaptive,
    )
    .0
}

#[test]
fn caller_bias_is_applied_on_first_row_at_greedy_and_seeded_temperatures() {
    for temperature in [0.0, 0.7, 1.0] {
        let mut live = sequence();
        live.temperature = temperature;
        live.logit_bias = vec![
            (0, f32::NEG_INFINITY),
            (2, f32::NEG_INFINITY),
            (3, f32::NEG_INFINITY),
            (4, f32::NEG_INFINITY),
        ];
        let mut reference = sequence();
        reference.temperature = temperature;
        reference.logit_bias = live.logit_bias.clone();
        let before = fingerprint(&live);
        let ctx = context();
        let bytes = row(&[10.0, 1.0, 0.0, 0.0, 0.0]);
        let expected = ordinary(&mut reference, &bytes, &ctx, false);
        assert_eq!(expected, 1);
        {
            let mut policy =
                OrdinaryVerifyPolicy::new(&MODEL, &mut live, &ctx, options(false)).unwrap();
            policy.checkpoint().unwrap();
            assert_eq!(policy.pick(0, &bytes).unwrap(), expected);
            policy.restore().unwrap();
        }
        assert_eq!(fingerprint(&live), before);
    }
}

#[test]
fn unique_raw_max_can_lose_after_actual_repetition_penalty() {
    let mut live = sequence();
    live.output_tokens = vec![1];
    live.repetition_penalty = 2.0;
    let mut reference = sequence();
    reference.output_tokens = live.output_tokens.clone();
    reference.repetition_penalty = 2.0;
    let ctx = context();
    let bytes = row(&[0.0, 10.0, 9.0, 0.0, 0.0]);
    assert_eq!(ordinary(&mut reference, &bytes, &ctx, false), 2);
    let mut policy = OrdinaryVerifyPolicy::new(&MODEL, &mut live, &ctx, options(false)).unwrap();
    policy.checkpoint().unwrap();
    assert_eq!(policy.pick(0, &bytes).unwrap(), 2);
    policy.restore().unwrap();
}

#[test]
fn accepted_history_and_seed_advance_before_next_row_without_touching_target_prefix() {
    for temperature in [0.0, 0.8] {
        let mut live = sequence();
        live.temperature = temperature;
        live.presence_penalty = 1.5;
        let before = fingerprint(&live);
        let mut reference = sequence();
        reference.temperature = temperature;
        reference.presence_penalty = 1.5;
        let ctx = context();
        let rows = [
            row(&[-20.0, 20.0, -20.0, -20.0, -20.0]),
            row(&[0.0, 3.0, 2.0, -20.0, -20.0]),
        ];
        {
            let mut policy =
                OrdinaryVerifyPolicy::new(&MODEL, &mut live, &ctx, options(false)).unwrap();
            policy.checkpoint().unwrap();
            for (i, bytes) in rows.iter().enumerate() {
                let expected = ordinary(&mut reference, bytes, &ctx, false);
                let picked = policy.pick(i, bytes).unwrap();
                assert_eq!(picked, expected);
                assert!(policy.advance(picked).unwrap());
                reference.output_tokens.push(picked);
                if temperature == 0.0 {
                    assert_eq!(picked, [1, 2][i]);
                }
            }
            assert_eq!(policy.sequence().seq.tokens, vec![0, 0]);
            assert_eq!(policy.sequence().seq.seq_len, 2);
            assert_eq!(policy.sequence().seq.kv_valid_tokens, 2);
            policy.restore().unwrap();
        }
        assert_eq!(fingerprint(&live), before);
    }
}

#[test]
fn ordinary_last_index_tie_rule_is_not_replaced_by_raw_verify_argmax() {
    let mut live = sequence();
    let mut reference = sequence();
    let ctx = context();
    let bytes = row(&[0.0, 8.0, 8.0, 0.0, 0.0]);
    let expected = ordinary(&mut reference, &bytes, &ctx, false);
    assert_eq!(expected, 2);
    let mut policy = OrdinaryVerifyPolicy::new(&MODEL, &mut live, &ctx, options(false)).unwrap();
    policy.checkpoint().unwrap();
    assert_eq!(policy.pick(0, &bytes).unwrap(), expected);
    policy.restore().unwrap();
}

#[test]
fn thinking_close_resets_ordinary_fence_counters_and_pins_required_tool() {
    let mut live = sequence();
    live.inside_thinking = true;
    live.think_end_token = Some(1);
    live.require_tool_call = true;
    live.tools_present = true;
    live.tool_call_start_token = Some(3);
    live.in_code_fence = true;
    live.consecutive_confident = 5;
    live.sentence_defer_count = 2;
    let before = fingerprint(&live);
    let ctx = LogitsContext {
        think_end_token: Some(1),
        tool_call_start_token: Some(3),
        ..context()
    };
    {
        let mut policy =
            OrdinaryVerifyPolicy::new(&MODEL, &mut live, &ctx, options(false)).unwrap();
        policy.checkpoint().unwrap();
        assert_eq!(
            policy.pick(0, &row(&[0.0, 20.0, 0.0, 0.0, 0.0])).unwrap(),
            1
        );
        assert!(policy.advance(1).unwrap());
        assert!(!policy.sequence().inside_thinking);
        assert!(!policy.sequence().in_code_fence);
        assert_eq!(policy.sequence().consecutive_confident, 0);
        assert_eq!(policy.sequence().sentence_defer_count, 0);
        assert!(policy.sequence().think_just_ended);
        assert_eq!(
            policy.pick(1, &row(&[0.0, 0.0, 20.0, 0.0, 0.0])).unwrap(),
            3
        );
        assert!(policy.advance(3).unwrap());
        assert!(policy.sequence().tool_call_opened);
        assert!(!policy.sequence().require_tool_call);
        policy.restore().unwrap();
    }
    assert_eq!(fingerprint(&live), before);
}

#[test]
fn ordinary_spontaneous_thinking_budget_decay_and_fence_toggle_are_preserved() {
    let mut live = sequence();
    live.think_start_token = Some(4);
    live.think_watchdog_fires = 2;
    let before = fingerprint(&live);
    let ctx = LogitsContext {
        think_start_token: Some(4),
        ..context()
    };
    let mut config = options(false);
    config.code_fence_token = Some(2);
    {
        let mut policy = OrdinaryVerifyPolicy::new(&MODEL, &mut live, &ctx, config).unwrap();
        policy.checkpoint().unwrap();
        assert_eq!(
            policy.pick(0, &row(&[0.0, 0.0, 0.0, 0.0, 20.0])).unwrap(),
            4
        );
        assert!(policy.advance(4).unwrap());
        assert_eq!(policy.sequence().thinking_budget, Some(32));
        assert!(policy.sequence().output_tokens.is_empty());
        assert_eq!(
            policy.pick(1, &row(&[0.0, 0.0, 20.0, 0.0, 0.0])).unwrap(),
            2
        );
        assert!(policy.advance(2).unwrap());
        assert!(policy.sequence().in_code_fence);
        policy.restore().unwrap();
    }
    assert_eq!(fingerprint(&live), before);
}

#[test]
fn real_json_grammar_advances_per_row_and_restores_without_detaching() {
    let mut live = sequence();
    live.grammar_state = Some(grammar());
    live.eos_tokens = vec![3];
    let before = fingerprint(&live);
    let ctx = context();
    {
        let mut policy =
            OrdinaryVerifyPolicy::new(&MODEL, &mut live, &ctx, options(false)).unwrap();
        policy.checkpoint().unwrap();
        let bytes = row(&[20.0, 1.0, 30.0, 0.0, 0.0]);
        assert_eq!(policy.pick(0, &bytes).unwrap(), 0);
        assert!(policy.advance(0).unwrap());
        let bytes = row(&[0.0, 1.0, 30.0, 0.0, 0.0]);
        assert_eq!(policy.pick(1, &bytes).unwrap(), 1);
        assert!(policy.advance(1).unwrap());
        assert_eq!(
            policy
                .sequence()
                .grammar_state
                .as_ref()
                .unwrap()
                .num_history_steps(),
            2
        );
        assert_eq!(
            policy.pick(2, &row(&[0.0, 0.0, 0.0, 30.0, 0.0])).unwrap(),
            3
        );
        assert!(!policy.advance(3).unwrap());
        assert!(
            policy.pick(3, &bytes).is_err(),
            "terminal row must prevent suffix policy work"
        );
        policy.restore().unwrap();
    }
    assert_eq!(fingerprint(&live), before);
    assert!(!live.grammar_state.as_mut().unwrap().stop_legal(&[3]));
}

#[test]
fn grammar_resumes_after_thinking_close_without_consuming_close_token() {
    let mut live = sequence();
    live.grammar_state = Some(grammar());
    live.inside_thinking = true;
    live.think_end_token = Some(4);
    let ctx = LogitsContext {
        think_end_token: Some(4),
        ..context()
    };
    let mut policy = OrdinaryVerifyPolicy::new(&MODEL, &mut live, &ctx, options(false)).unwrap();
    policy.checkpoint().unwrap();
    assert_eq!(
        policy.pick(0, &row(&[0.0, 0.0, 0.0, 0.0, 30.0])).unwrap(),
        4
    );
    assert!(policy.advance(4).unwrap());
    assert_eq!(
        policy
            .sequence()
            .grammar_state
            .as_ref()
            .unwrap()
            .num_history_steps(),
        0
    );
    assert_eq!(
        policy.pick(1, &row(&[1.0, 0.0, 30.0, 0.0, 0.0])).unwrap(),
        0
    );
    assert!(policy.advance(0).unwrap());
    assert_eq!(
        policy
            .sequence()
            .grammar_state
            .as_ref()
            .unwrap()
            .num_history_steps(),
        1
    );
    policy.restore().unwrap();
}

#[test]
fn adaptive_configuration_is_explicit_and_private_history_survives_restore() {
    for adaptive_enabled in [false, true] {
        let mut live = sequence();
        let mut reference = sequence();
        for a in [&mut live, &mut reference] {
            a.temperature = 0.8;
            a.tool_call_opened = true;
            a.adaptive = crate::adaptive_sampler::AdaptiveSamplingState::new(0.8);
            for _ in 0..7 {
                a.adaptive.observe_entropy(&[20.0, 0.0, 0.0, 0.0, 0.0]);
            }
        }
        let before = fingerprint(&live);
        let ctx = context();
        let bytes = row(&[1.0, 1.1, 1.2, 1.3, 1.4]);
        let expected = ordinary(&mut reference, &bytes, &ctx, adaptive_enabled);
        {
            let mut policy =
                OrdinaryVerifyPolicy::new(&MODEL, &mut live, &ctx, options(adaptive_enabled))
                    .unwrap();
            policy.checkpoint().unwrap();
            assert_eq!(policy.pick(0, &bytes).unwrap(), expected);
            assert_eq!(policy.sequence().adaptive.zone, reference.adaptive.zone);
            assert_eq!(
                policy.sequence().adaptive.effective_temperature(),
                reference.adaptive.effective_temperature()
            );
            policy.restore().unwrap();
        }
        assert_eq!(fingerprint(&live), before);
        // A reset-to-new rollback would lose seven private low-entropy steps.
        live.adaptive.observe_entropy(&[20.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(live.adaptive.effective_temperature(), 0.8 + 0.1);
    }
}

#[test]
fn malformed_or_unselected_advance_fails_closed_and_drop_restores_without_events() {
    for malformed in [true, false] {
        let mut live = sequence();
        live.grammar_state = Some(grammar());
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        live.sink = ResponseSink::Streaming(tx);
        let before = fingerprint(&live);
        let ctx = context();
        {
            let mut policy =
                OrdinaryVerifyPolicy::new(&MODEL, &mut live, &ctx, options(false)).unwrap();
            policy.checkpoint().unwrap();
            if malformed {
                assert!(policy.pick(0, &[0, 0]).is_err());
            } else {
                assert_eq!(
                    policy.pick(0, &row(&[30.0, 0.0, 0.0, 0.0, 0.0])).unwrap(),
                    0
                );
                assert!(
                    policy.advance(2).is_err(),
                    "cannot substitute a grammar-invalid/unselected token"
                );
            }
            assert!(rx.try_recv().is_err());
        }
        assert_eq!(fingerprint(&live), before);
        assert!(live.grammar_state.is_some());
        assert!(rx.try_recv().is_err());
    }
}
