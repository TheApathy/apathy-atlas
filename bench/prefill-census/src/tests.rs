// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

#[test]
fn deepseek_chat_pins_native_unicode_and_thinking_off_template() {
    assert_eq!(
        parse_prompt_format(Some("deepseek-chat")).unwrap(),
        "deepseek-chat"
    );
    let (prefix, suffix) = prompt_parts("deepseek-chat").unwrap();
    assert_eq!(
        prefix,
        "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}><\u{ff5c}User\u{ff5c}>Reference facts:\n"
    );
    assert_eq!(
        suffix,
        "\n\nWrite a short factual summary of the reference facts above.<\u{ff5c}Assistant\u{ff5c}></think>"
    );
    assert_eq!(prefix.chars().filter(|c| *c == '\u{ff5c}').count(), 4);
    assert_eq!(prefix.chars().filter(|c| *c == '\u{2581}').count(), 2);
    assert_eq!(suffix.chars().filter(|c| *c == '\u{ff5c}').count(), 2);
    assert!(!prefix.contains("<think>"));
    assert!(!prefix.contains('|'));
    assert!(!suffix.contains("<|im_"));
    assert!(suffix.ends_with("</think>"));
    assert!(corpus(prefix).starts_with(&format!("{prefix}Fact 0000:")));
}

#[test]
fn plain_format_is_default_and_preserves_v2_prompt_bytes() {
    assert_eq!(parse_prompt_format(None).unwrap(), "plain");
    let (prefix, suffix) = prompt_parts("plain").unwrap();
    assert_eq!(prefix, "<think></think>\n\nReference facts:\n");
    assert_eq!(
        suffix,
        "\n\nWrite a short factual summary of the reference facts above.\nSummary:"
    );
    let text = corpus(prefix);
    assert!(text.starts_with("<think></think>\n\nReference facts:\nFact 0000: shelf 0 contains 1 blue folders and 1 red folders.\n"));
    assert_eq!(text.matches("Fact ").count(), 4000);
    assert!(!text.contains("<|im_start|>"));
}

#[test]
fn qwen_chatml_has_one_user_and_complete_assistant_non_thinking_prefix() {
    assert_eq!(
        parse_prompt_format(Some("qwen-chatml")).unwrap(),
        "qwen-chatml"
    );
    let (prefix, suffix) = prompt_parts("qwen-chatml").unwrap();
    assert_eq!(prefix, "<|im_start|>user\nReference facts:\n");
    assert_eq!(
        suffix,
        "\n\nWrite a short factual summary of the reference facts above.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    assert!(!prefix.contains("<think>"));
    assert_eq!(suffix.matches("<|im_end|>").count(), 1);
    assert_eq!(suffix.matches("<|im_start|>assistant").count(), 1);
    assert!(suffix.ends_with("<think>\n\n</think>\n\n"));
    for bad in ["", "chatml", "QWEN", "qwen-chatml "] {
        assert!(parse_prompt_format(Some(bad)).is_err());
        assert!(prompt_parts(bad).is_err());
    }
}

#[test]
fn selected_bins_default_all_or_unique_ascending_subset_only() {
    assert_eq!(parse_bins(None).unwrap(), vec![256, 2048, 8192]);
    assert_eq!(parse_bins(Some("256,2048")).unwrap(), vec![256, 2048]);
    assert_eq!(parse_bins(Some("8192")).unwrap(), vec![8192]);
    assert_eq!(parse_bins(Some("256,8192")).unwrap(), vec![256, 8192]);
    for invalid in [
        "",
        "256,256",
        "8192,2048",
        "512",
        "256,",
        "256, 2048",
        "0",
        "-1",
        "256,2048,8192,8192",
    ] {
        assert!(parse_bins(Some(invalid)).is_err(), "accepted {invalid}");
    }
}

#[test]
fn canary_explicitly_disables_both_supported_thinking_switches() {
    let request = canary_request("test-model");
    assert_eq!(request["enable_thinking"], false);
    assert_eq!(request["chat_template_kwargs"]["enable_thinking"], false);
    assert_eq!(request["max_tokens"], 32);
    assert_eq!(request["temperature"], 0.0);
}

fn response() -> Value {
    json!({"choices":[{"index":0,"text":"Useful result.","finish_reason":"length"}],
        "usage":{"prompt_tokens":256,"completion_tokens":32,"total_tokens":288,
        "prompt_tokens_details":{"cached_tokens":0},"time_to_first_token_ms":128.0,
        "response_token/s":42.0}})
}

#[test]
fn measured_usage_is_exact_uncached_positive_and_distinct_from_wall() {
    let got = validate_response(&response(), Some(256), false, 1000.0).unwrap();
    assert_eq!(got["effective_prefill_tokens_per_second"], 2000.0);
    assert_eq!(got["client_wall_ms"], 1000.0);
    assert_eq!(got["completion_tokens"], 32);
}

#[test]
fn missing_cache_is_not_treated_as_zero() {
    let mut v = response();
    v["usage"]
        .as_object_mut()
        .unwrap()
        .remove("prompt_tokens_details");
    assert!(validate_response(&v, Some(256), false, 1000.0).is_err());
    for bad in [json!(null), json!(1), json!(-1), json!(0.0), json!("0")] {
        let mut v = response();
        v["usage"]["prompt_tokens_details"]["cached_tokens"] = bad;
        assert!(validate_response(&v, Some(256), false, 1000.0).is_err());
    }
}

#[test]
fn rejects_bad_ttft_and_prompt_or_completion_counts() {
    for bad in [json!(0), json!(-1), json!("NaN"), json!(null), json!(true)] {
        let mut v = response();
        v["usage"]["time_to_first_token_ms"] = bad;
        assert!(validate_response(&v, Some(256), false, 1000.0).is_err());
    }
    for (field, bad) in [
        ("prompt_tokens", 255),
        ("completion_tokens", 0),
        ("completion_tokens", 33),
        ("total_tokens", 287),
    ] {
        let mut v = response();
        v["usage"][field] = json!(bad);
        assert!(validate_response(&v, Some(256), false, 1000.0).is_err());
    }
}

#[test]
fn malformed_errors_and_empty_output_fail_closed() {
    assert!(parse_json_response(b"{", 200).is_err());
    assert!(parse_json_response(b"{}{}", 200).is_err());
    assert!(parse_json_response(b"{}", 503).is_err());
    assert!(parse_json_response(br#"{"error":{"message":"bad"}}"#, 200).is_err());
    let mut v = response();
    v["choices"][0]["text"] = json!(" \n");
    assert!(validate_response(&v, Some(256), false, 1000.0).is_err());
    v["choices"] = json!([]);
    assert!(validate_response(&v, Some(256), false, 1000.0).is_err());
}

#[test]
fn canary_reports_verbatim_content_without_semantic_pass_claim() {
    let mut v = response();
    v["choices"][0]["message"] = json!({"content":"Wrong answer"});
    let got = validate_response(&v, None, true, 1000.0).unwrap();
    assert_eq!(got["output"], "Wrong answer");
    assert_eq!(got["semantic_status"], "unreviewed");
}

#[test]
fn exact_bins_keep_suffix_and_cannot_underfill() {
    let ids = exact_prompt(&(0..1000).collect::<Vec<u32>>(), &[9001, 9002], 256).unwrap();
    assert_eq!(ids.len(), 256);
    assert_eq!(&ids[254..], &[9001, 9002]);
    assert!(exact_prompt(&[1, 2], &[9], 256).is_err());
    assert!(exact_prompt(&[1, 2], &[9, 10], 2).is_err());
}

#[test]
fn tokenize_schema_requires_integral_tokens_and_matching_count() {
    assert_eq!(
        parse_tokens(&json!({"tokens":[1,2],"count":2})).unwrap(),
        vec![1, 2]
    );
    for v in [
        json!({"tokens":[1],"count":2}),
        json!({"tokens":[-1],"count":1}),
        json!({"tokens":[1.0],"count":1}),
        json!({"tokens":[],"count":0}),
    ] {
        assert!(parse_tokens(&v).is_err());
    }
}

#[test]
fn provenance_secret_screen_does_not_confuse_token_tuning() {
    assert!(secret_name("ATLAS_API_KEY"));
    assert!(secret_name("--auth-token"));
    assert!(!secret_name("ATLAS_MAX_TOKENS"));
    assert!(!secret_name("--max-tokens"));
    assert!(tuning_name("ATLAS_DFLASH2"));
    assert!(!tuning_name("GITHUB_TOKEN"));
}
