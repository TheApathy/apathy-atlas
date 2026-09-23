// SPDX-License-Identifier: AGPL-3.0-only
//
// Jinja template helpers for ChatTokenizer. Free fns split out of
// chat_impl.rs to keep the parent under 500 LoC.

use anyhow::{Context, Result};
use std::path::Path;

pub(super) const TEMPLATE_OVERRIDE_DIR: &str = "jinja-templates";

/// HF's chat-template `tojson`: `json.dumps(x, ensure_ascii=False, indent=None,
/// separators=None, sort_keys=False)`, the same keywords accepted (and an
/// `indent` positional, as Jinja's own filter takes).
fn hf_tojson(
    value: &minijinja::Value,
    indent: Option<minijinja::Value>,
    kwargs: minijinja::value::Kwargs,
) -> std::result::Result<minijinja::Value, minijinja::Error> {
    use minijinja::{Error, ErrorKind};
    let bad = |m: String| Error::new(ErrorKind::InvalidOperation, m);
    let indent = match kwargs.get::<Option<minijinja::Value>>("indent")?.or(indent) {
        None => None,
        Some(v) if v.is_none() || v.is_undefined() => None,
        // Python: an int n is n spaces (n <= 0 -> newlines only), a str is used as is.
        Some(v) => Some(match v.as_str() {
            Some(s) => s.to_string(),
            None => {
                let n = i64::try_from(v.clone())
                    .map_err(|_| bad(format!("tojson: indent must be an int or str, got {v}")))?;
                " ".repeat(n.max(0) as usize)
            }
        }),
    };
    let separators = match kwargs.get::<Option<Vec<String>>>("separators")? {
        None => None,
        Some(s) if s.len() == 2 => Some((s[0].clone(), s[1].clone())),
        Some(s) => return Err(bad(format!("tojson: separators must be a pair, got {s:?}"))),
    };
    let opts = crate::pyjson::DumpOpts {
        ensure_ascii: kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false),
        indent,
        separators,
        sort_keys: kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false),
    };
    kwargs.assert_all_used()?;
    let json = serde_json::to_value(value)
        .map_err(|e| bad(format!("tojson: value is not JSON-serialisable: {e}")))?;
    Ok(minijinja::Value::from_safe_string(crate::pyjson::dumps_with(&json, &opts)))
}

/// Resolve the template source without letting the stale dense-Qwen override
/// hide a newer checkpoint contract.  Qwen3.5/3.6 retain the override (and
/// therefore their established prompt bytes); a dense `qwen3_5` checkpoint
/// that declares both Qwen3.8-only controls gets its own template.
pub(super) fn select_chat_template(
    model_type: &str,
    override_tmpl: Option<String>,
    checkpoint_tmpl: Option<String>,
    supports_thinking: bool,
) -> (String, bool) {
    let checkpoint_is_qwen38 = model_type == "qwen3_5"
        && checkpoint_tmpl.as_ref().is_some_and(|t| {
            t.contains("resolved_reasoning_effort") && t.contains("preserve_thinking")
        });
    if checkpoint_is_qwen38 {
        tracing::info!(
            "Using checkpoint-owned dense-Qwen template (native reasoning_effort/preserve_thinking contract)"
        );
        return (checkpoint_tmpl.expect("checked Some above"), true);
    }
    if let Some(t) = override_tmpl {
        return (t, false);
    }
    if let Some(t) = checkpoint_tmpl {
        return (t, false);
    }
    tracing::warn!("No chat template found — using default ChatML");
    (default_chatml_template(supports_thinking), false)
}

/// Build a precompiled minijinja Environment from the chat template.
/// Leaks the template string to 'static — acceptable since one ChatTokenizer
/// lives for the entire server lifetime.
pub(super) fn build_jinja_env(chat_template: &str) -> Result<minijinja::Environment<'static>> {
    let template_static: &'static str = Box::leak(chat_template.to_string().into_boxed_str());
    let mut env = minijinja::Environment::new();
    env.set_lstrip_blocks(true);
    env.set_trim_blocks(true);

    env.add_function(
        "raise_exception",
        |msg: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );
    env.add_filter("rtrim", |s: String| -> String {
        s.trim_end_matches('\n').to_string()
    });
    env.add_filter("ltrim", |s: String| -> String {
        s.trim_start_matches('\n').to_string()
    });
    env.add_filter("split_first", |s: String, sep: String| -> String {
        s.split(&sep).next().unwrap_or("").to_string()
    });
    env.add_filter("split_last", |s: String, sep: String| -> String {
        s.rsplit(&sep).next().unwrap_or("").to_string()
    });
    // HF transformers replaces Jinja's `tojson` with `json.dumps(x,
    // ensure_ascii=False, indent=None, separators=None, sort_keys=False)`
    // (chat_template_utils._compile_jinja_template), so tool schemas render
    // in the request's key order with `", "`/`": "` separators. minijinja's
    // built-in is compact and escapes HTML characters; templates were written
    // against HF's.
    env.add_filter("tojson", hf_tojson);

    // F76 (2026-04-29): bridge Python-style `.items()` / `.keys()` /
    // `.values()` methods on Maps to the corresponding minijinja
    // filters. MiniMax M2.7's chat_template.jinja line 112 calls
    // `_args.items()` on the tool_call arguments dict; without this
    // callback minijinja raises `UnknownMethod: map has no method
    // named items` and the second turn of every tool-use
    // conversation 500's. The callback only fires on
    // `UnknownMethod` errors, so it is a no-cost compat layer.
    //
    // Also bridges Python-style `str.split(sep)` to the minijinja
    // `split` filter. Gemma-4's chat template calls `text.split('<channel|>')`
    // and `part.split('<|channel>')` inside its `strip_thinking` macro
    // (gemma4.jinja:143/145). minijinja has no `.split()` *method* on
    // strings — only a filter — so every assistant (model-role) turn
    // raised `UnknownMethod: string has no method named split`. The
    // existing `convert_python_jinja_to_minijinja` text-rewrites only
    // cover the literal `.split('<think>')`/`.split('</think>')`
    // patterns; this callback makes `.split()` work for any separator.
    env.set_unknown_method_callback(
        |state, value, method, args| -> Result<minijinja::Value, minijinja::Error> {
            use minijinja::value::{ValueKind, from_args};
            if value.kind() == ValueKind::String && method == "split" {
                // Python `str.split(sep)` → minijinja `split` filter.
                // The separator is forwarded verbatim; an absent
                // separator falls through to the filter's default
                // (whitespace split), matching Python semantics.
                let (sep,): (Option<minijinja::Value>,) = from_args(args)?;
                let mut filter_args = vec![value.clone()];
                if let Some(sep) = sep {
                    filter_args.push(sep);
                }
                return state.apply_filter("split", &filter_args);
            }
            if value.kind() == ValueKind::Map {
                match method {
                    "items" => {
                        let _: () = from_args(args)?;
                        return state.apply_filter("items", std::slice::from_ref(value));
                    }
                    "keys" => {
                        let _: () = from_args(args)?;
                        return state
                            .apply_filter("dictsort", std::slice::from_ref(value))
                            .and_then(|sorted| {
                                state.apply_filter(
                                    "map",
                                    &[
                                        sorted,
                                        minijinja::Value::from("attribute"),
                                        minijinja::Value::from(0u32),
                                    ],
                                )
                            })
                            .or_else(|_| state.apply_filter("list", std::slice::from_ref(value)));
                    }
                    "values" => {
                        let _: () = from_args(args)?;
                        return state
                            .apply_filter("dictsort", std::slice::from_ref(value))
                            .and_then(|sorted| {
                                state.apply_filter(
                                    "map",
                                    &[
                                        sorted,
                                        minijinja::Value::from("attribute"),
                                        minijinja::Value::from(1u32),
                                    ],
                                )
                            });
                    }
                    "get" => {
                        let (key, default): (minijinja::Value, Option<minijinja::Value>) =
                            from_args(args)?;
                        return Ok(value
                            .get_item(&key)
                            .ok()
                            .filter(|v| !v.is_undefined())
                            .unwrap_or_else(|| default.unwrap_or(minijinja::Value::UNDEFINED)));
                    }
                    _ => {}
                }
            }
            Err(minijinja::Error::from(minijinja::ErrorKind::UnknownMethod))
        },
    );

    env.add_template("chat", template_static)
        .context("Failed to compile Jinja chat template")?;
    Ok(env)
}

/// Try loading an override template from jinja-templates/{model_type}.jinja.
pub(super) fn load_override_template(model_type: &str, repo_root: Option<&Path>) -> Option<String> {
    // Check relative to repo root (Docker: /build, dev: /workspace/atlas)
    let candidates = [
        repo_root.map(|r| {
            r.join(TEMPLATE_OVERRIDE_DIR)
                .join(format!("{model_type}.jinja"))
        }),
        Some(std::path::PathBuf::from(TEMPLATE_OVERRIDE_DIR).join(format!("{model_type}.jinja"))),
    ];
    for candidate in candidates.into_iter().flatten() {
        if candidate.exists() {
            match std::fs::read_to_string(&candidate) {
                Ok(raw) => {
                    let converted = convert_python_jinja_to_minijinja(&raw);
                    tracing::info!(
                        "Using override Jinja template from {} ({} chars)",
                        candidate.display(),
                        converted.len(),
                    );
                    return Some(converted);
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to read override template {}: {e}",
                        candidate.display()
                    );
                }
            }
        }
    }
    None
}

/// Try loading an OpenAI-variant template from jinja-templates/openai/{model_type}.jinja.
pub(super) fn load_openai_template(model_type: &str, repo_root: Option<&Path>) -> Option<String> {
    let candidates = [
        repo_root.map(|r| {
            r.join(TEMPLATE_OVERRIDE_DIR)
                .join("openai")
                .join(format!("{model_type}.jinja"))
        }),
        Some(
            std::path::PathBuf::from(TEMPLATE_OVERRIDE_DIR)
                .join("openai")
                .join(format!("{model_type}.jinja")),
        ),
    ];
    for candidate in candidates.into_iter().flatten() {
        if candidate.exists() {
            match std::fs::read_to_string(&candidate) {
                Ok(raw) => {
                    return Some(convert_python_jinja_to_minijinja(&raw));
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to read OpenAI template {}: {e}",
                        candidate.display()
                    );
                }
            }
        }
    }
    None
}

/// Load template from tokenizer_config.json in the model directory.
pub(super) fn load_config_template(model_dir: &Path) -> Result<Option<String>> {
    let config_path = model_dir.join("tokenizer_config.json");
    if !config_path.exists() {
        return Ok(None);
    }
    let config_json =
        std::fs::read_to_string(&config_path).context("Failed to read tokenizer_config.json")?;
    let config: serde_json::Value =
        serde_json::from_str(&config_json).context("Failed to parse tokenizer_config.json")?;
    match config.get("chat_template").and_then(|v| v.as_str()) {
        Some(t) => {
            let converted = convert_python_jinja_to_minijinja(t);
            tracing::info!(
                "Loaded Jinja chat template from tokenizer_config.json ({} chars)",
                converted.len()
            );
            Ok(Some(converted))
        }
        None => {
            // Some models ship chat_template.jinja as a standalone file
            // (e.g., Nemotron-H Super 120B) instead of embedding it in
            // tokenizer_config.json.
            let jinja_path = model_dir.join("chat_template.jinja");
            if jinja_path.exists() {
                let raw = std::fs::read_to_string(&jinja_path)
                    .context("Failed to read chat_template.jinja")?;
                let converted = convert_python_jinja_to_minijinja(&raw);
                tracing::info!(
                    "Loaded standalone chat_template.jinja ({} chars)",
                    converted.len()
                );
                return Ok(Some(converted));
            }
            Ok(None)
        }
    }
}

/// Default ChatML template for models without tokenizer_config.json.
/// Always includes an empty system block if no system message (required by Nemotron-H).
pub(super) fn default_chatml_template(supports_thinking: bool) -> String {
    let gen_prompt = if supports_thinking {
        "{{ '<|im_start|>assistant\\n<think>\\n' }}"
    } else {
        "{{ '<|im_start|>assistant\\n' }}"
    };
    format!(
        r#"{{% if messages[0].role != 'system' %}}{{{{ '<|im_start|>system\n<|im_end|>\n' }}}}{{% endif %}}{{% for message in messages %}}{{% if message.role == 'system' %}}{{% if loop.first %}}{{{{ '<|im_start|>system\n' + message.content + '<|im_end|>\n' }}}}{{% endif %}}{{% elif message.role == 'user' %}}{{{{ '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}}}{{% elif message.role == 'assistant' %}}{{{{ '<|im_start|>assistant\n' + message.content + '<|im_end|>\n' }}}}{{% endif %}}{{% endfor %}}{{% if add_generation_prompt %}}{gen_prompt}{{% endif %}}"#
    )
}

/// Convert Python Jinja2 syntax to minijinja-compatible syntax.
/// Handles: slice reversal, string methods, etc.
pub(super) fn convert_python_jinja_to_minijinja(template: &str) -> String {
    let mut t = template.to_string();

    // messages[::-1] → messages | reverse
    t = t.replace("messages[::-1]", "messages | reverse");

    // content.startswith('X') → content is startingwith('X')
    // Need regex for this — use simple string replacements for known patterns
    t = t.replace(
        ".startswith('<tool_response>')",
        " is startingwith '<tool_response>'",
    );
    t = t.replace(".startswith('\\n')", " is startingwith '\\n'");
    t = t.replace(
        ".endswith('</tool_response>')",
        " is endingwith '</tool_response>'",
    );
    t = t.replace(".endswith('\\n')", " is endingwith '\\n'");

    // content.split('X') → content | split('X') — minijinja has split filter
    // content.split('</think>')[0] → (content | split('</think>'))[0]
    // This is complex — use a custom filter instead
    // For now, register split as a method via custom function

    // content.rstrip('\n') → content | rtrim  (custom filter)
    // content.lstrip('\n') → content | ltrim  (custom filter)
    // content.strip('\n')  → content | rtrim | ltrim  (chain, custom filters)
    //
    // MiniMax M2's chat_template.jinja chains `.split('</think>')[0]
    // .strip('\n').split('<think>')[-1].strip('\n')`. After each
    // substring replace the chain remains a valid filter pipeline
    // because minijinja filters associate left-to-right.
    t = t.replace(".rstrip('\\n')", " | rtrim");
    t = t.replace(".lstrip('\\n')", " | ltrim");
    t = t.replace(".strip('\\n')", " | rtrim | ltrim");

    // .split('</think>')[0] → handled by custom split_first/split_last filters
    // Replace: content.split('</think>')[0] → content | split_first('</think>')
    // Replace: content.split('</think>')[-1] → content | split_last('</think>')
    t = t.replace(".split('</think>')[0]", " | split_first('</think>')");
    t = t.replace(".split('</think>')[-1]", " | split_last('</think>')");

    // .split('<think>')[-1] → | split_last('<think>')
    t = t.replace(".split('<think>')[-1]", " | split_last('<think>')");
    // .split('<think>')[0] → | split_first('<think>')
    t = t.replace(".split('<think>')[0]", " | split_first('<think>')");

    // `tojson(ensure_ascii=...)`, `tojson(indent=...)` etc. are handled by the
    // registered HF-compatible `tojson` filter (`hf_tojson`), kwargs included.

    // messages[1:] — minijinja 2.x supports slice syntax natively

    t
}

#[cfg(test)]
mod template_selection_tests {
    use super::select_chat_template;

    #[test]
    fn qwen38_checkpoint_contract_outranks_stale_dense_override() {
        let checkpoint = "resolved_reasoning_effort preserve_thinking".to_string();
        let (selected, is_qwen38) = select_chat_template(
            "qwen3_5",
            Some("retired-qwen3_5-override".into()),
            Some(checkpoint.clone()),
            true,
        );
        assert_eq!(selected, checkpoint);
        assert!(is_qwen38);
    }

    #[test]
    fn qwen36_keeps_existing_override_prompt_contract() {
        let (selected, is_qwen38) = select_chat_template(
            "qwen3_5",
            Some("qwen36-established-override".into()),
            Some("preserve_thinking only".into()),
            true,
        );
        assert_eq!(selected, "qwen36-established-override");
        assert!(!is_qwen38);
    }
}

#[cfg(test)]
mod hf_tojson_tests {
    use super::build_jinja_env;

    fn render(tmpl: &str) -> String {
        let x: serde_json::Value = serde_json::from_str(
            r#"{"type": "function", "function": {"name": "f", "description": "<é>", "parameters": {"type": "object", "properties": {"b": {"type": "integer"}, "a": {"type": "number", "default": 1.5}}}}}"#,
        )
        .unwrap();
        let env = build_jinja_env(tmpl).unwrap();
        let t = env.get_template("chat").unwrap();
        t.render(minijinja::context! { x => minijinja::Value::from_serialize(&x) })
            .unwrap()
    }

    /// Expected strings are CPython `json.dumps` with HF's tojson defaults.
    #[test]
    fn tojson_is_hf_json_dumps() {
        assert_eq!(
            render("{{ x | tojson }}"),
            r#"{"type": "function", "function": {"name": "f", "description": "<é>", "parameters": {"type": "object", "properties": {"b": {"type": "integer"}, "a": {"type": "number", "default": 1.5}}}}}"#
        );
        assert_eq!(
            render("{{ x.function.parameters | tojson(ensure_ascii=True, indent=2) }}"),
            "{\n  \"type\": \"object\",\n  \"properties\": {\n    \"b\": {\n      \"type\": \"integer\"\n    },\n    \"a\": {\n      \"type\": \"number\",\n      \"default\": 1.5\n    }\n  }\n}"
        );
        assert_eq!(render("{{ x.function.description | tojson(ensure_ascii=True) }}"), r#""<\u00e9>""#);
        assert_eq!(
            render("{{ x.function.parameters.properties | tojson(sort_keys=True, separators=[',', ':']) }}"),
            r#"{"a":{"type":"number","default":1.5},"b":{"type":"integer"}}"#
        );
        // CONTROL: minijinja's built-in tojson (what ran before) is compact and
        // escapes '<' -- it must not be what this environment renders.
        let builtin = minijinja::Environment::new()
            .render_str("{{ x | tojson }}", minijinja::context! { x => "<é>" })
            .unwrap();
        assert_ne!(builtin, render("{{ x.function.description | tojson }}"));
    }
}
