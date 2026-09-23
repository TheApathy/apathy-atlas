// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only fixtures for the real blocking and complete-stream publication
//! boundaries. No scheduler/model mock, schema repair, or replacement parser.

use std::sync::{Arc, Mutex, RwLock, atomic::AtomicBool};
use std::time::Duration;

use serde_json::{Value, json};

use crate::AppState;
use crate::tool_parser::{GlmXmlParser, ToolCallParser, ToolChoice, ToolDefinition};

pub(super) struct Case {
    pub tools: Vec<ToolDefinition>,
    pub choice: ToolChoice,
    pub wire: String,
    pub expected: Option<Value>,
}

fn tool(name: &str, schema: Value) -> ToolDefinition {
    serde_json::from_value(json!({
        "type": "function", "function": {"name": name, "parameters": schema}
    }))
    .unwrap()
}

fn schema(kind: &str) -> Value {
    json!({"type": "object", "properties": {"value": {"type": kind}},
        "required": ["value"], "additionalProperties": false})
}

pub(super) fn wire(name: &str, args: &[(&str, &str)]) -> String {
    let mut body = format!("<tool_call>{name}");
    for (key, value) in args {
        body.push_str(&format!(
            "<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"
        ));
    }
    body.push_str("</tool_call>");
    body
}

pub(super) fn case(name: &str) -> Case {
    let mut case = Case {
        tools: vec![tool("inspect", schema("string"))],
        choice: ToolChoice::Mode("required".into()),
        wire: wire("inspect", &[("value", "ok")]),
        expected: None,
    };
    match name {
        "fractional_integer" => {
            case.tools = vec![tool("inspect", schema("integer"))];
            case.wire = wire("inspect", &[("value", "1.5")]);
        }
        "wrong_object" => {
            case.tools = vec![tool("inspect", schema("object"))];
            case.wire = wire("inspect", &[("value", "[]")]);
        }
        "missing_required_integer" => {
            case.tools = vec![tool("inspect", schema("integer"))];
            case.wire = wire("inspect", &[]);
        }
        "missing_required_string" => case.wire = wire("inspect", &[]),
        "unknown_name" => case.wire = wire("unrelated_unknown_tool", &[("value", "ok")]),
        "fuzzy_name" => {
            case.tools = vec![tool("get_weather", schema("string"))];
            case.wire = wire("weather", &[("value", "ok")]);
        }
        "specific_wrong_name" => {
            case.tools.push(tool("inspect_more", schema("string")));
            case.choice = serde_json::from_value(json!({"function": {"name": "inspect"}})).unwrap();
            case.wire = wire("inspect_more", &[("value", "ok")]);
        }
        "closed_extra_key" => {
            case.wire = wire("inspect", &[("value", "ok"), ("extra", "must not publish")]);
        }
        "renamed_key" => case.wire = wire("inspect", &[("VALUE", "ok")]),
        "raw_whitespace" | "raw_empty" => {
            let value = if name == "raw_empty" { "" } else { " \t\n  " };
            // This schema expressly permits empty strings. The legacy
            // description backfill must not invent replacement bytes.
            case.tools = vec![tool(
                "inspect",
                json!({"type": "object",
                "properties": {"description": {"type": "string"}},
                "required": ["description"], "additionalProperties": false}),
            )];
            case.wire = wire("inspect", &[("description", value)]);
            case.expected = Some(json!({"description": value}));
        }
        "raw_true_string" => {
            case.wire = wire("inspect", &[("value", "true")]);
            case.expected = Some(json!({"value": "true"}));
        }
        "valid_typed_arguments" => {
            case.tools = vec![tool(
                "inspect",
                json!({"type": "object", "properties": {
                "count": {"type": "integer"}, "enabled": {"type": "boolean"},
                "ratio": {"type": "number"}, "object": {"type": "object"},
                "items": {"type": "array", "items": {"type": "integer"}},
                "label": {"type": "string"}},
                "required": ["count", "enabled", "ratio", "object", "items", "label"],
                "additionalProperties": false}),
            )];
            case.wire = wire(
                "inspect",
                &[
                    ("count", "7"),
                    ("enabled", "true"),
                    ("ratio", "1.5"),
                    ("object", "{\"nested\":2}"),
                    ("items", "[1,2]"),
                    ("label", "true"),
                ],
            );
            case.expected = Some(json!({"count": 7, "enabled": true, "ratio": 1.5,
                "object": {"nested": 2}, "items": [1, 2], "label": "true"}));
        }
        _ => panic!("unknown publication fixture {name}"),
    }
    case
}

pub(super) fn request(case: &Case) -> crate::ir::ChatRequest {
    let wire: crate::openai::ChatCompletionRequest = serde_json::from_value(json!({
        "model": "glm-publication-cpu", "messages": [{"role": "user", "content": "inspect"}],
        "tools": case.tools, "max_tokens": 64
    }))
    .unwrap();
    let mut req: crate::ir::ChatRequest = wire.into();
    req.tool_choice = Some(case.choice.clone());
    req
}

pub(super) fn response() -> crate::api::InferenceResponse {
    crate::api::InferenceResponse {
        output_tokens: Vec::new(),
        finish_reason: "stop".into(),
        time_to_first_token_ms: 0.0,
        decode_time_ms: 0.0,
        logprobs: Vec::new(),
        reasoning_tokens: 0,
        cached_prompt_tokens: 0,
        prompt_logprobs: Vec::new(),
    }
}

pub(super) fn parser() -> Arc<dyn ToolCallParser> {
    Arc::new(GlmXmlParser)
}

pub(super) fn app_state() -> Arc<AppState> {
    let dir = tempfile::tempdir().unwrap();
    let tokenizer = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
    tokenizer
        .save(dir.path().join("tokenizer.json"), false)
        .unwrap();
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        r#"{"chat_template":"{{ messages | length }}"}"#,
    )
    .unwrap();
    let tokenizer = crate::tokenizer::ChatTokenizer::from_model_dir(
        dir.path(),
        0,
        false,
        "glm_publication_cpu_fixture",
        Some(dir.path()),
    )
    .unwrap();
    let (request_tx, _request_rx) = tokio::sync::mpsc::channel(1);
    Arc::new(AppState {
        tokenizer,
        model_name: "glm-publication-cpu".into(),
        adapter_name: None,
        adapter_names: Vec::new(),
        active_adapter: Arc::new(Mutex::new(None)),
        max_seq_len: 2048,
        max_batch_size: 1,
        yarn_context: false,
        request_tx,
        rotation_tx: None,
        vision_config: None,
        vision_max_pixels: None,
        default_temperature: 0.0,
        default_top_k: 0,
        default_top_p: 1.0,
        default_top_n_sigma: 0.0,
        default_min_p: 0.0,
        tool_call_parser: Some(parser()),
        reasoning_parser: None,
        think_end_token_id: None,
        think_start_token_id: None,
        tool_max_tokens: 64,
        sampling_presets: Default::default(),
        tool_call_start_token_id: None,
        auto_compact_threshold: None,
        model_ready: Arc::new(AtomicBool::new(false)),
        request_timeout: 0,
        effective_context: 2048,
        behavior: Default::default(),
        disable_thinking: true,
        default_thinking: crate::ir::ThinkingDirective::Off,
        response_store: crate::response_store::ResponseStore::with_config(
            1,
            Duration::from_secs(60),
        ),
        rate_limiter: crate::rate_limiter::RateLimiter::with_config(
            crate::rate_limiter::RateLimitConfig {
                rpm: 0,
                tpm: 0,
                burst_rpm: 1,
                burst_tpm: 1,
            },
        ),
        conversation_store: crate::conversation_store::ConversationStore::with_config(
            1,
            Duration::from_secs(60),
        ),
        dump_writer: None,
        auth: None,
        lora_stageable: Default::default(),
        lora_peer_addr: None,
        promotion: None,
        promoted_slots: Arc::new(RwLock::new(Default::default())),
        lora_disk_stageable: Default::default(),
    })
}
