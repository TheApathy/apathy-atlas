// SPDX-License-Identifier: AGPL-3.0-only

//! RED wiring gates for the new adapter only. No existing scheduler route is
//! changed or claimed qualified by these source checks.

fn adapter_source() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/scheduler/glm53_verify_policy_adapter.rs"
    ))
    .unwrap_or_default()
}

#[test]
fn adapter_calls_real_ordinary_sampler_with_explicit_adaptive_configuration() {
    let source = adapter_source();
    let start = source
        .find("process_seq_logits(")
        .expect("ordinary policy adapter must call the real ordinary sampler");
    let call = &source[start + "process_seq_logits(".len()..];
    let mut depth = 1usize;
    let mut end = None;
    for (index, ch) in call.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(index);
                    break;
                }
            }
            _ => {}
        }
    }
    let call = &call[..end.expect("ordinary sampler call must be complete")];
    assert!(
        call.contains(".adaptive_sampling"),
        "pass explicit request/server configuration, not PR834's hard-coded false"
    );
}

#[test]
fn reversible_policy_adapter_does_not_emit_or_commit_model_state() {
    let source = adapter_source();
    assert!(
        source.contains("OrdinaryVerifyPolicy"),
        "production adapter is missing"
    );
    let code = source
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    for forbidden in [
        "emit_token(",
        "send_stream_event(",
        "commit_accepted_prefix(",
        "blocking_send(",
        "try_send(",
        "decode_verify(",
    ] {
        assert!(
            !code.contains(forbidden),
            "reversible policy must not call {forbidden}"
        );
    }
}
