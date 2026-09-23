// SPDX-License-Identifier: AGPL-3.0-only

use serde::{Deserialize, Serialize};

use super::*;

// ── Legacy /v1/completions types (OpenAI standard) ──

/// Completion request (non-chat, raw prompt).
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct CompletionRequest {
    /// Optional, as on the DeepSeek-V4.1 Python server (which ignores it).
    #[serde(default)]
    pub model: String,
    /// OpenAI's four `prompt` shapes: a string, an array of strings (joined),
    /// an array of token ids, or an array of token-id arrays (concatenated,
    /// as the string array is joined). Token prompts are prefilled exactly,
    /// like `prompt_token_ids`.
    #[serde(default, deserialize_with = "deserialize_prompt")]
    pub prompt: CompletionPrompt,
    /// Optional raw prompt token IDs. When present, bypasses tokenization
    /// AND the think-prefix injection — the model prefills exactly these
    /// tokens. Used by the DFlash drafter-retrain hidden-capture harness so
    /// the captured hidden states align 1:1 with the trainer's own
    /// (chat-templated) input_ids. `prompt` is ignored when this is set.
    #[serde(default)]
    pub prompt_token_ids: Option<Vec<u32>>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    pub temperature: Option<f32>,
    /// Top-k: keep only the k highest-probability tokens before sampling.
    pub top_k: Option<u32>,
    /// Top-p (nucleus): keep smallest set of tokens whose cumulative probability >= p.
    pub top_p: Option<f32>,
    /// Top-n-sigma: filter tokens in logit space before temperature scaling.
    /// 0.0 = disabled.
    pub top_n_sigma: Option<f32>,
    /// Min-p: keep tokens with prob >= min_p * max_prob (post-softmax).
    /// 0.0 = disabled.
    pub min_p: Option<f32>,
    /// Repetition penalty: penalize tokens that have already been generated.
    /// 1.0 = disabled.
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// Per-token logit bias.
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    #[serde(default)]
    pub stream: bool,
    /// Stop sequences (same as chat completions).
    #[serde(default, deserialize_with = "deserialize_stop")]
    pub stop: Vec<String>,
    /// Seed for deterministic sampling (same as chat completions).
    pub seed: Option<u64>,
}

/// A `/v1/completions` prompt: text to tokenize, or token ids to prefill as given.
#[derive(Debug, Clone, PartialEq)]
pub enum CompletionPrompt {
    Text(String),
    Tokens(Vec<u32>),
}

impl Default for CompletionPrompt {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

/// Parse OpenAI's `prompt`: a string, an array of strings (joined), an array
/// of token ids, or an array of token-id arrays (concatenated). An empty array
/// (or an empty batch) and anything else is refused with a message naming it.
fn deserialize_prompt<'de, D>(d: D) -> Result<CompletionPrompt, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    let v = serde_json::Value::deserialize(d)?;
    parse_prompt(&v).map_err(D::Error::custom)
}

fn parse_prompt(v: &serde_json::Value) -> Result<CompletionPrompt, String> {
    use serde_json::Value;
    let token = |t: &Value| -> Result<u32, String> {
        t.as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| format!("prompt token ids must be non-negative integers, got {t}"))
    };
    match v {
        Value::String(s) => Ok(CompletionPrompt::Text(s.clone())),
        Value::Array(items) if items.is_empty() => {
            Err("prompt must not be an empty array".to_string())
        }
        Value::Array(items) => match &items[0] {
            Value::String(_) => items
                .iter()
                .map(|s| {
                    s.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| format!("prompt array mixes strings with {s}"))
                })
                .collect::<Result<Vec<_>, _>>()
                .map(|parts| CompletionPrompt::Text(parts.join(""))),
            Value::Array(_) => {
                let mut ids = Vec::new();
                for batch in items {
                    let Some(batch) = batch.as_array() else {
                        return Err(format!("prompt token batches must be arrays, got {batch}"));
                    };
                    if batch.is_empty() {
                        return Err("prompt token batches must not be empty".to_string());
                    }
                    for t in batch {
                        ids.push(token(t)?);
                    }
                }
                Ok(CompletionPrompt::Tokens(ids))
            }
            _ => items
                .iter()
                .map(token)
                .collect::<Result<Vec<_>, _>>()
                .map(CompletionPrompt::Tokens),
        },
        other => Err(format!(
            "prompt must be a string, an array of strings, an array of token ids or an array of token-id arrays, got {other}"
        )),
    }
}

#[cfg(test)]
mod prompt_shape_tests {
    use super::{CompletionPrompt, CompletionRequest};
    use serde_json::json;

    fn prompt(v: serde_json::Value) -> Result<CompletionPrompt, String> {
        serde_json::from_value::<CompletionRequest>(json!({"prompt": v}))
            .map(|r| r.prompt)
            .map_err(|e| e.to_string())
    }

    #[test]
    fn the_four_openai_prompt_shapes_parse() {
        assert_eq!(prompt(json!("hi")), Ok(CompletionPrompt::Text("hi".into())));
        assert_eq!(prompt(json!(["a", "b"])), Ok(CompletionPrompt::Text("ab".into())));
        assert_eq!(prompt(json!([1, 2, 3])), Ok(CompletionPrompt::Tokens(vec![1, 2, 3])));
        assert_eq!(
            prompt(json!([[1, 2], [3]])),
            Ok(CompletionPrompt::Tokens(vec![1, 2, 3]))
        );
        // absent prompt is the empty text, as before
        let r: CompletionRequest = serde_json::from_value(json!({})).unwrap();
        assert_eq!(r.prompt, CompletionPrompt::Text(String::new()));
    }

    #[test]
    fn invalid_prompt_shapes_are_refused_with_a_reason() {
        for (v, why) in [
            (json!([]), "empty array"),
            (json!([[1], []]), "batches must not be empty"),
            (json!([1, -2]), "non-negative integers"),
            (json!([1, 2.5]), "non-negative integers"),
            (json!([1, 4294967296u64]), "non-negative integers"),
            (json!(["a", 1]), "mixes strings"),
            (json!([[1], 2]), "batches must be arrays"),
            (json!(7), "must be a string"),
            (json!({"x": 1}), "must be a string"),
        ] {
            let err = prompt(v.clone()).expect_err(&v.to_string());
            assert!(err.contains(why), "{v}: {err}");
        }
    }
}

/// Completion response.
#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: String,
}

impl CompletionResponse {
    pub fn new(model: &str, text: String, usage: Usage, finish_reason: &str) -> Self {
        Self {
            id: format!("cmpl-{}", uuid_v4()),
            object: "text_completion".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            choices: vec![CompletionChoice {
                index: 0,
                text,
                finish_reason: finish_reason.to_string(),
            }],
            usage,
        }
    }
}

/// SSE streaming chunk for completions.
#[derive(Debug, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChunkChoice {
    pub index: usize,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

impl CompletionChunk {
    /// Content text chunk.
    pub fn text_chunk(model: &str, id: &str, text: String) -> Self {
        Self {
            id: id.to_string(),
            object: "text_completion".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            choices: vec![CompletionChunkChoice {
                index: 0,
                text,
                finish_reason: None,
            }],
            usage: None,
        }
    }

    /// Final chunk with finish_reason and usage.
    pub fn done_chunk(model: &str, id: &str, finish_reason: &str, usage: Usage) -> Self {
        Self {
            id: id.to_string(),
            object: "text_completion".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            choices: vec![CompletionChunkChoice {
                index: 0,
                text: String::new(),
                finish_reason: Some(finish_reason.to_string()),
            }],
            usage: Some(usage),
        }
    }
}

// ── Tokenize endpoint types ──

/// Request body for POST /tokenize.
#[derive(Debug, Deserialize)]
pub struct TokenizeRequest {
    #[allow(dead_code)]
    pub model: Option<String>,
    /// Raw text to tokenize (mutually exclusive with `messages`).
    pub prompt: Option<String>,
    /// Chat messages to tokenize via the chat template (mutually exclusive with `prompt`).
    pub messages: Option<Vec<IncomingMessage>>,
}

/// Response body for POST /tokenize.
#[derive(Debug, Serialize)]
pub struct TokenizeResponse {
    pub tokens: Vec<u32>,
    pub count: usize,
}
