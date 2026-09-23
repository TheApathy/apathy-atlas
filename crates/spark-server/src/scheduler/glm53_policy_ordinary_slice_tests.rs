// SPDX-License-Identifier: AGPL-3.0-only

//! Real ordinary-entry extraction gate: only the logits transport is synthetic.

#[path = "glm53_ordinary_greedy_fixture.rs"]
mod boundary;
#[path = "glm53_policy_adapter_fixture.rs"]
mod fixture;

use crate::scheduler::logit_processors::LogitsContext;
use crate::scheduler::{ActiveSeq, DevicePtr};
use boundary::{BASE, Boundary};
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

fn logprobs(a: &ActiveSeq) -> Vec<(u32, u32, Vec<(u32, u32)>)> {
    a.logprobs_data
        .iter()
        .map(|lp| {
            (
                lp.token_id,
                lp.logprob.to_bits(),
                lp.top
                    .iter()
                    .map(|&(id, value)| (id, value.to_bits()))
                    .collect(),
            )
        })
        .collect()
}

fn reference(model: &Boundary, seq: ActiveSeq, ctx: &LogitsContext, adaptive: bool) -> ActiveSeq {
    let mut active = vec![seq];
    super::decode_logits_step::process_decode_logits(
        model,
        &mut active,
        BASE,
        Instant::now(),
        ctx.think_end_token,
        ctx.think_start_token,
        Some(2),
        ctx.tool_call_start_token,
        ctx.tool_call_end_token,
        adaptive,
    );
    assert_eq!(active.len(), 1);
    active.pop().unwrap()
}

#[test]
fn borrowed_singleton_preserves_actual_ordinary_sampling_and_live_transition() {
    for case in [
        "neutral", "bias", "repeat", "grammar", "thinking", "adaptive", "logprobs",
    ] {
        let model = Boundary::new(&[[1.0, 10.0, 9.0, 0.0, -1.0]]);
        let mut expected = sequence();
        let mut actual = sequence();
        let mut ctx = context();
        if case == "thinking" {
            ctx.think_end_token = Some(4);
        }
        for a in [&mut expected, &mut actual] {
            // A genuine decode has consumed the anchor before this entry.
            a.seq.tokens.push(a.last_token);
            a.seq.seq_len += 1;
            a.seq.kv_valid_tokens += 1;
            match case {
                "bias" => a.logit_bias = vec![(2, 4.0)],
                "repeat" => {
                    a.output_tokens = vec![1];
                    a.repetition_penalty = 2.0;
                }
                "grammar" => {
                    a.grammar_state = Some(grammar());
                    a.eos_tokens = vec![3];
                }
                "thinking" => {
                    a.inside_thinking = true;
                    a.think_end_token = Some(4);
                    a.logit_bias = vec![(2, 4.0)];
                }
                "adaptive" => a.inside_tool_body = true,
                "logprobs" => a.top_logprobs = Some(3),
                _ => {}
            }
        }
        let expected = reference(&model, expected, &ctx, case == "adaptive");
        let sequence_address = &actual.seq as *const _ as usize;
        super::decode_logits_step::process_decode_logits_slice(
            &model,
            std::slice::from_mut(&mut actual),
            BASE,
            Instant::now(),
            ctx.think_end_token,
            ctx.think_start_token,
            Some(2),
            ctx.tool_call_start_token,
            ctx.tool_call_end_token,
            case == "adaptive",
        )
        .unwrap();
        assert_eq!(&actual.seq as *const _ as usize, sequence_address);
        let mut expected_state = fingerprint(&expected);
        let mut actual_state = fingerprint(&actual);
        expected_state.pop(); // Independent wall-clock timestamps only.
        actual_state.pop();
        assert_eq!(actual_state, expected_state, "{case}");
        assert_eq!(logprobs(&actual), logprobs(&expected));
        if case == "neutral" {
            assert_eq!(model.argmax_calls(), 2);
            assert_eq!(model.full_reads(), 0);
            assert!(model.scalar_offsets().is_empty());
        }
    }
}

#[test]
fn borrowed_entry_returns_transport_error_without_freeing_or_replacing_sequence() {
    let model = Boundary::new(&[[1.0, 10.0, 9.0, 0.0, -1.0]]);
    let mut actual = sequence();
    let before = fingerprint(&actual);
    let address = &actual.seq as *const _ as usize;
    let result = super::decode_logits_step::process_decode_logits_slice(
        &model,
        std::slice::from_mut(&mut actual),
        DevicePtr(BASE.0 - 1),
        Instant::now(),
        None,
        None,
        None,
        None,
        None,
        false,
    );
    assert!(
        result.is_err(),
        "caller retains exactly one retirement owner"
    );
    assert_eq!(fingerprint(&actual), before);
    assert_eq!(&actual.seq as *const _ as usize, address);
}
