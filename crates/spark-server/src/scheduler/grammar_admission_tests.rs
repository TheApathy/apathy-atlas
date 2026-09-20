// SPDX-License-Identifier: AGPL-3.0-only

//! Actual prefill entry points must reject a lost constraint before any model
//! allocation or inference. The shared fixture panics on every model I/O call.

use super::*;
use crate::tool_parser::ToolCallFormat;
use std::sync::Arc;

#[allow(dead_code)]
#[path = "glm53_policy_adapter_fixture.rs"]
mod fixture;

enum Reply {
    Blocking(tokio::sync::oneshot::Receiver<anyhow::Result<InferenceResponse>>),
    Streaming(tokio::sync::mpsc::Receiver<StreamEvent>),
}

fn request(
    spec: Option<GrammarSpec>,
    required: bool,
    streaming: bool,
) -> (InferenceRequest, Reply) {
    macro_rules! make {
        ($variant:ident, $($sink:ident: $value:expr),+ $(,)?) => {
            InferenceRequest::$variant {
                prompt_tokens: Arc::new(vec![0]), session_hash: 0, adapter_slot: -1,
                src_lang_id: 0, tgt_lang_id: 0, num_beams: 1, length_penalty: 1.0,
                early_stopping: false, image_pixels: vec![], max_tokens: 8, min_tokens: 0,
                temperature: 0.0, top_k: 0, top_p: 1.0, top_n_sigma: 0.0, min_p: 0.0,
                repetition_penalty: 1.0, presence_penalty: 0.0, frequency_penalty: 0.0,
                dry_multiplier: 0.0, dry_base: 1.75, dry_allowed_length: 2, lz_penalty: 0.0,
                logit_bias: vec![], stop_tokens: vec![], enable_thinking: false,
                thinking_budget: None, repetition_detection: None, require_tool_call: required,
                tools_present: required, suppress_tool_call: false, disable_mtp: false,
                grammar_spec: spec, seed: Some(42), top_logprobs: None, prompt_logprobs: None,
                echo: false, timeout_at: None, $($sink: $value),+
            }
        };
    }
    if streaming {
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        (
            make!(Streaming, token_tx: tx, cancel_flag: Arc::new(std::sync::atomic::AtomicBool::new(false))),
            Reply::Streaming(rx),
        )
    } else {
        let (tx, rx) = tokio::sync::oneshot::channel();
        (make!(Blocking, response_tx: tx), Reply::Blocking(rx))
    }
}

fn engine() -> GrammarEngine {
    let vocab: Vec<_> = (0u8..128).map(|b| char::from(b).to_string()).collect();
    GrammarEngine::new(&vocab, &[]).unwrap()
}

fn tools(format: ToolCallFormat, auto: bool) -> GrammarSpec {
    GrammarSpec::ToolCall {
        tools: vec![],
        parser: Arc::from(format.into_parser()),
        use_triggers: auto,
    }
}

fn rejected(spec: Option<GrammarSpec>, required: bool, with_engine: bool) {
    for chunked in [false, true] {
        for streaming in [false, true] {
            let (req, reply) = request(spec.clone(), required, streaming);
            let mut engine = with_engine.then(engine);
            let observed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if chunked {
                    start_chunked_prefill(
                        None,
                        None,
                        None,
                        None,
                        &fixture::MODEL,
                        req,
                        &[],
                        8,
                        0,
                        0,
                        &mut engine,
                        0,
                        false,
                        None,
                        None,
                    )
                    .map(|_| ())
                } else {
                    prefill_request(
                        None,
                        None,
                        None,
                        None,
                        &fixture::MODEL,
                        req,
                        &[],
                        &mut engine,
                        0,
                        None,
                    )
                    .map(|_| ())
                }
            }));
            let result = observed.expect("lost grammar reached model allocation/inference");
            let error = result
                .expect_err("constraint failure must reject the request")
                .to_string();
            assert!(
                error.contains("grammar"),
                "precise constraint error missing: {error}"
            );
            match reply {
                Reply::Blocking(mut rx) => {
                    let response = rx
                        .try_recv()
                        .expect("request error must reach blocking sink");
                    let message = match response {
                        Ok(_) => panic!("unexpected output"),
                        Err(e) => e.to_string(),
                    };
                    assert_eq!(message, error);
                }
                Reply::Streaming(mut rx) => {
                    let message = match rx.try_recv().expect("request error must reach stream sink")
                    {
                        StreamEvent::Error(message) => message,
                        _ => panic!("constraint error published a non-error event"),
                    };
                    assert_eq!(message, error);
                    assert!(rx.try_recv().is_err(), "duplicate terminal error");
                }
            }
        }
    }
}

#[test]
fn grammar_admission_missing_engine_rejects_json_in_both_prefill_paths() {
    rejected(Some(GrammarSpec::JsonObject), false, false);
}

#[test]
fn grammar_admission_bad_schema_rejects_before_model_and_reports_error() {
    rejected(
        Some(GrammarSpec::JsonSchema { schema: "{".into() }),
        false,
        true,
    );
}

#[test]
fn grammar_admission_required_without_spec_rejects_instead_of_suppression_only() {
    rejected(None, true, true);
}

#[test]
fn grammar_admission_opted_out_required_parser_rejects() {
    rejected(Some(tools(ToolCallFormat::Mistral, false)), true, true);
}

#[test]
fn grammar_admission_empty_required_tool_set_rejects() {
    rejected(Some(tools(ToolCallFormat::Hermes, false)), true, true);
}

#[test]
fn grammar_admission_auto_compile_failure_does_not_silently_remove_constraint() {
    rejected(Some(tools(ToolCallFormat::Hermes, true)), false, true);
}

#[test]
fn grammar_admission_json_cannot_stand_in_for_required_tool_grammar() {
    rejected(Some(GrammarSpec::JsonObject), true, true);
}

#[test]
fn grammar_admission_optional_opt_out_and_no_spec_remain_explicitly_unconstrained() {
    assert!(
        compile_grammar_state(&mut None, &None, &[], false)
            .unwrap()
            .is_none()
    );
    for present in [false, true] {
        let mut engine = present.then(engine);
        assert!(
            compile_grammar_state(
                &mut engine,
                &Some(tools(ToolCallFormat::Mistral, true)),
                &[],
                false
            )
            .unwrap()
            .is_none()
        );
    }
}

#[test]
fn grammar_admission_valid_json_keeps_a_real_state_and_mask() {
    let mut state = compile_grammar_state(
        &mut Some(engine()),
        &Some(GrammarSpec::JsonObject),
        &[],
        false,
    )
    .unwrap()
    .unwrap();
    assert!(state.fill_bitmask());
    assert!(state.is_token_allowed(b'{' as u32));
    assert!(!state.is_token_allowed(b'x' as u32));
}
