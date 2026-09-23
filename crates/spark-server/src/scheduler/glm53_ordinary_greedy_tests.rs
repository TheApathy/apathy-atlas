// SPDX-License-Identifier: AGPL-3.0-only

//! Actual greedy-entry RED: no logprobs, forced host flag, CUDA, or environment
//! mutation. Reference is the real sampler plus the real ordinary transition.

#[path = "glm53_ordinary_greedy_fixture.rs"]
mod boundary;
#[path = "glm53_policy_adapter_fixture.rs"]
mod fixture;

use crate::adaptive_sampler::GenerationZone;
use crate::scheduler::ActiveSeq;
use crate::scheduler::logit_processors::LogitsContext;
use crate::scheduler::ordinary_transition::{LiveEffects, TransitionContext, advance_ordinary};
use boundary::{BASE, Boundary, VOCAB};
use fixture::{fingerprint, grammar, sequence};
use std::time::Instant;

fn context() -> LogitsContext {
    LogitsContext {
        think_end_token: None,
        think_start_token: None,
        tool_call_start_token: None,
        tool_call_end_token: None,
    }
}

fn run(model: &Boundary, active: &mut Vec<ActiveSeq>, ctx: LogitsContext, adaptive: bool) {
    for a in active.iter() {
        assert_eq!(a.temperature, 0.0);
        // API top_logprobs=0 means no requested logprobs, not Some(0): any
        // Some value would bypass the ordinary greedy admission under test.
        assert!(a.top_logprobs.is_none());
        assert!(a.logprobs_data.is_empty());
    }
    crate::scheduler::decode_logits_step::process_decode_logits(
        model,
        active,
        BASE,
        Instant::now(),
        ctx.think_end_token,
        ctx.think_start_token,
        None,
        ctx.tool_call_start_token,
        ctx.tool_call_end_token,
        adaptive,
    );
    assert!(!active.is_empty(), "fixture may not conceal a decode error");
}

fn reference(
    model: &Boundary,
    a: &mut ActiveSeq,
    index: usize,
    ctx: LogitsContext,
    adaptive: bool,
) -> u32 {
    let (token, logprobs) = crate::scheduler::decode_logits_seq::process_seq_logits(
        model,
        a,
        &model.bytes,
        index,
        VOCAB,
        2,
        false,
        &ctx,
        adaptive,
    );
    assert!(logprobs.is_none());
    advance_ordinary(
        a,
        token,
        logprobs,
        &TransitionContext {
            logits: ctx,
            code_fence_token: None,
            max_seq_len: 4096,
            now: Instant::now(),
        },
        &mut LiveEffects { model },
    )
    .unwrap();
    token
}

fn same_policy(actual: &ActiveSeq, expected: &ActiveSeq) {
    let mut actual = fingerprint(actual);
    let mut expected = fingerprint(expected);
    actual.pop(); // Only independent wall-clock timestamps are excluded.
    expected.pop();
    assert_eq!(
        actual, expected,
        "actual greedy entry must preserve ordinary policy state"
    );
}

fn compare(
    scores: [f32; VOCAB],
    ctx: LogitsContext,
    adaptive: bool,
    setup: impl Fn(&mut ActiveSeq),
) -> (Boundary, ActiveSeq, u32) {
    let model = Boundary::new(&[scores]);
    let mut expected = sequence();
    let mut candidate = sequence();
    for a in [&mut expected, &mut candidate] {
        a.think_end_token = ctx.think_end_token;
        a.think_start_token = ctx.think_start_token;
        a.tool_call_start_token = ctx.tool_call_start_token;
        a.tool_call_end_token = ctx.tool_call_end_token;
        setup(a);
    }
    let token = reference(&model, &mut expected, 0, ctx, adaptive);
    let mut active = vec![candidate];
    run(&model, &mut active, ctx, adaptive);
    same_policy(&active[0], &expected);
    (model, active.pop().unwrap(), token)
}

#[test]
fn neutral_greedy_keeps_gpu_argmax_without_any_logits_copy() {
    let (model, actual, token) = compare([0., 10., 9., 0., 0.], context(), false, |_| {});
    assert_eq!(token, 1);
    assert_eq!(actual.last_token, 1);
    assert_eq!(model.argmax_calls(), 1);
    assert_eq!(model.full_reads(), 0);
    assert!(model.scalar_offsets().is_empty());
}

#[test]
fn caller_bias_changes_unique_raw_winner_at_actual_greedy_entry() {
    let (model, _, token) = compare([0., 10., 9., 0., 0.], context(), false, |a| {
        a.logit_bias = vec![(2, 2.0)]
    });
    assert_eq!(token, 2);
    assert!(model.full_reads() > 0);
}

#[test]
fn repetition_presence_and_frequency_can_change_the_raw_winner() {
    for kind in ["repetition", "presence", "frequency"] {
        let (model, _, token) = compare([0., 10., 9., 0., 0.], context(), false, |a| {
            a.output_tokens = vec![1, 1];
            match kind {
                "repetition" => a.repetition_penalty = 2.0,
                "presence" => a.presence_penalty = 2.0,
                "frequency" => a.frequency_penalty = 1.0,
                _ => unreachable!(),
            }
        });
        assert_eq!(token, 2, "{kind}");
        assert!(model.full_reads() > 0, "{kind}");
    }
}

#[test]
fn competitor_raising_penalties_are_not_admitted_as_reduce_only() {
    for kind in ["repetition", "presence", "frequency"] {
        let (model, _, token) = compare([0., 10., 9., 0., 0.], context(), false, |a| {
            a.output_tokens = vec![2, 2];
            match kind {
                "repetition" => a.repetition_penalty = 0.5,
                "presence" => a.presence_penalty = -2.0,
                "frequency" => a.frequency_penalty = -1.0,
                _ => unreachable!(),
            }
        });
        assert_eq!(token, 2, "{kind}");
        assert!(model.full_reads() > 0, "{kind}");
    }
}

#[test]
fn reduce_only_immune_positive_winner_uses_existing_proof_not_full_copy() {
    let (model, _, token) = compare([0., 8., 10., 0., 0.], context(), false, |a| {
        a.output_tokens = vec![1, 1];
        a.repetition_penalty = 1.25;
        a.presence_penalty = 0.5;
        a.frequency_penalty = 0.5;
    });
    assert_eq!(token, 2);
    assert_eq!(model.argmax_calls(), 1);
    assert_eq!(model.full_reads(), 0);
    assert_eq!(model.scalar_offsets(), vec![2 * 2]);
}

#[test]
fn nonpositive_immune_winner_fails_conservative_gpu_proof() {
    let (model, _, token) = compare([-3., -4., -1., -5., -6.], context(), false, |a| {
        a.output_tokens = vec![1];
        a.repetition_penalty = 1.25;
    });
    assert_eq!(token, 2);
    assert!(model.full_reads() > 0);
}

#[test]
fn failed_single_logit_copy_falls_back_to_real_host_policy() {
    let mut model = Boundary::new(&[[0., 8., 10., 0., 0.]]);
    model.fail_scalar = true;
    let mut expected = sequence();
    let mut candidate = sequence();
    for a in [&mut expected, &mut candidate] {
        a.output_tokens = vec![1];
        a.repetition_penalty = 1.25;
    }
    assert_eq!(reference(&model, &mut expected, 0, context(), false), 2);
    let mut active = vec![candidate];
    run(&model, &mut active, context(), false);
    same_policy(&active[0], &expected);
    assert_eq!(model.scalar_offsets(), vec![2 * 2]);
    assert!(model.full_reads() > 0);
}

#[test]
fn immunity_uses_post_tool_history_and_each_batch_row_stride() {
    let model = Boundary::new(&[[0., 8., 10., 0., 0.], [0., 10., 8., 0., 0.]]);
    let mut ctx = context();
    ctx.tool_call_end_token = Some(4);
    let mut active = vec![sequence(), sequence()];
    let mut expected = vec![sequence(), sequence()];
    for (index, history) in [vec![2, 4, 1], vec![1, 4, 2]].into_iter().enumerate() {
        for a in [&mut active[index], &mut expected[index]] {
            a.output_tokens = history.clone();
            a.repetition_penalty = 1.25;
            a.tool_call_end_token = Some(4);
        }
        assert_eq!(
            reference(&model, &mut expected[index], index, ctx, false),
            [2, 1][index]
        );
    }
    run(&model, &mut active, ctx, false);
    for (actual, expected) in active.iter().zip(&expected) {
        same_policy(actual, expected);
    }
    assert_eq!(model.full_reads(), 0);
    assert_eq!(model.scalar_offsets(), vec![2 * 2, (VOCAB + 1) * 2]);
}

#[test]
fn tool_loop_suppression_runs_even_outside_thinking() {
    let mut ctx = context();
    ctx.tool_call_start_token = Some(3);
    let (model, _, token) = compare([0., 0., 9., 10., 0.], ctx, false, |a| {
        a.suppress_tool_call = true
    });
    assert_eq!(token, 2);
    assert!(model.full_reads() > 0);
}

#[test]
fn existing_thinking_and_postclose_masks_still_use_actual_host_pipeline() {
    let mut ctx = context();
    ctx.tool_call_start_token = Some(3);
    ctx.think_end_token = Some(1);
    ctx.think_start_token = Some(4);
    let (model, _, token) = compare([0., 0., 9., 10., 0.], ctx, false, |a| {
        a.inside_thinking = true
    });
    assert_eq!(token, 2);
    assert!(model.full_reads() > 0);
    let (model, _, token) = compare([0., 8., 9., 0., 10.], ctx, false, |a| a.think_ended = true);
    assert_eq!(token, 2);
    assert!(model.full_reads() > 0);
}

#[test]
fn required_tool_pin_preserves_one_shot_transition() {
    let mut ctx = context();
    ctx.tool_call_start_token = Some(3);
    let (model, actual, token) = compare([0., 0., 10., 0., 0.], ctx, false, |a| {
        a.think_ended = true;
        a.think_just_ended = true;
        a.require_tool_call = true;
        a.tools_present = true;
    });
    assert_eq!(token, 3);
    assert!(actual.tool_call_opened);
    assert!(!actual.require_tool_call);
    assert!(model.full_reads() > 0);
}

#[test]
fn min_tokens_and_unpinned_required_tool_keep_postselection_eos_semantics() {
    for required in [false, true] {
        let (model, actual, token) = compare([0., 0., 0., 10., 0.], context(), false, |a| {
            a.eos_tokens = vec![3];
            a.min_tokens = if required { 0 } else { 4 };
            a.require_tool_call = required;
        });
        assert_eq!(token, 3); // Existing ordinary behavior: select then suppress EOS emission.
        assert!(!actual.finished);
        assert!(actual.output_tokens.is_empty());
        assert_eq!(model.argmax_calls(), 1);
        assert_eq!(model.full_reads(), 0);
    }
}

#[test]
fn enabled_adaptive_observation_is_not_lost_for_temperature_zero() {
    let (model, actual, token) = compare([0., 10., 9., 0., 0.], context(), true, |a| {
        a.tool_call_opened = true
    });
    assert_eq!(token, 1);
    assert_eq!(actual.adaptive.zone, GenerationZone::ToolCall);
    assert!(model.full_reads() > 0);
}

#[test]
fn disabled_adaptive_control_keeps_neutral_gpu_path() {
    let (model, actual, token) = compare([0., 10., 9., 0., 0.], context(), false, |a| {
        a.tool_call_opened = true
    });
    assert_eq!(token, 1);
    assert_eq!(actual.adaptive.zone, GenerationZone::FreeText);
    assert_eq!(model.argmax_calls(), 1);
    assert_eq!(model.full_reads(), 0);
}

#[test]
fn real_grammar_mask_and_matcher_transition_are_not_bypassed() {
    let (model, actual, token) = compare([9., 0., 10., 0., 0.], context(), false, |a| {
        a.grammar_state = Some(grammar())
    });
    assert_eq!(token, 0);
    assert!(actual.grammar_state.as_ref().unwrap().num_history_steps() > 0);
    assert!(model.full_reads() > 0);
}

#[test]
fn pattern_penalties_require_pipeline_but_in_tool_dry_is_classified_after_gating() {
    for lz in [false, true] {
        let (model, _, _) = compare([0., 8., 10., 0., 0.], context(), false, |a| {
            a.output_tokens = vec![1, 2, 1, 2, 1];
            if lz {
                a.lz_penalty = 0.8;
            } else {
                a.dry_multiplier = 0.8;
            }
        });
        assert!(model.full_reads() > 0);
    }
    let (model, _, token) = compare([0., 8., 10., 0., 0.], context(), false, |a| {
        a.inside_tool_body = true;
        a.dry_multiplier = 0.8;
    });
    assert_eq!(token, 2);
    assert_eq!(model.argmax_calls(), 1);
    assert_eq!(model.full_reads(), 0);
}

#[test]
fn in_tool_positive_opener_bias_is_removed_by_actual_parameter_builder() {
    let mut ctx = context();
    ctx.tool_call_start_token = Some(3);
    let (model, _, token) = compare([0., 8., 10., 0., 0.], ctx, false, |a| {
        a.inside_tool_body = true;
        a.logit_bias = vec![(3, 50.0)];
    });
    assert_eq!(token, 2);
    assert_eq!(model.argmax_calls(), 1);
    assert_eq!(model.full_reads(), 0);
}

#[test]
fn one_biased_sequence_cannot_inherit_another_rows_neutral_admission() {
    let model = Boundary::new(&[[0., 10., 9., 0., 0.], [0., 8., 10., 0., 0.]]);
    let mut active = vec![sequence(), sequence()];
    let mut expected = vec![sequence(), sequence()];
    active[1].logit_bias = vec![(1, 3.0)];
    expected[1].logit_bias = vec![(1, 3.0)];
    for (index, a) in expected.iter_mut().enumerate() {
        assert_eq!(reference(&model, a, index, context(), false), 1);
    }
    run(&model, &mut active, context(), false);
    for (actual, expected) in active.iter().zip(&expected) {
        same_policy(actual, expected);
    }
    assert!(model.full_reads() > 0);
}
