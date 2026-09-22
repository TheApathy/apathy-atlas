// SPDX-License-Identifier: AGPL-3.0-only

//! Parser for **DeepSeek-V4.1-Flash-Next**, the checkpoint Atlas actually has on disk
//! (`/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K`).
//!
//! This is a DIFFERENT RELEASE from the `deepseek_v4` lane, which implements
//! DeepSeek-V4-Flash-0731. Every major shape differs:
//!
//! | field | V4-Flash-0731 | V4.1-Flash-Next |
//! |---|---|---|
//! | layers / hidden | 43 / 4096 | 40 / 5120 |
//! | routed experts | 256 | 384 |
//! | moe_intermediate | 2048 | 2304 |
//! | q_lora_rank | 1024 | 1280 |
//! | index_n_heads | 64 | 32 |
//! | expert quant | EXL3 2-bit | CB3 3-bit + FP8 block-32 |
//!
//! **Scope of this file is CONFIG PARSING ONLY.** It makes the engine read the real
//! checkpoint and report correct shapes. It does NOT provide: CB3 expert kernels, engram
//! tables (layers 1 and 14, ~384M rows each), the block-candidate compressor, cross-layer
//! KV/index sourcing, or the V4.1 vision tower. Loading weights will still fail; that is
//! expected and is the next stage's work.
//!
//! ## The nesting, which is the whole reason this file exists
//! V4-Flash-0731 ships a FLAT config.json. V4.1 nests the language model under
//! `text_config` and keeps `quantization_config`, `vision_config`, `architectures` and the
//! token ids at the top level. Aliasing `model_type` alone is not enough — serde would
//! read the top level and yield `hidden_size = 0`. So we flatten first, then delegate to
//! the shared V4 body.

use anyhow::{Context, Result};
use serde_json::{Map, Value};

use super::super::ModelConfig;
use super::deepseek_v4::parse_deepseek_family;

/// Top-level keys that are meaningful to the parser and are NOT inside `text_config`.
/// `vision_config` is deliberately absent: see the note in `flatten_text_config`.
const TOP_LEVEL_CARRIED: &[&str] = &[
    "architectures",
    "quantization_config",
    "dtype",
    "torch_dtype",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "image_token_id",
];

pub fn parse_deepseek_v41(json: &str) -> Result<ModelConfig> {
    let raw: Value =
        serde_json::from_str(json).context("Invalid JSON in DeepSeek-V4.1 config.json")?;
    let flat = flatten_text_config(&raw)?;
    let flat_json = serde_json::to_string(&Value::Object(flat))
        .context("Failed to re-serialize flattened DeepSeek-V4.1 config")?;
    parse_deepseek_family(&flat_json, true)
}

/// Merge `text_config` up to the top level.
///
/// `text_config` WINS on any key collision: it is the language model's own statement of
/// its shape, and the top level carries container-level metadata that can disagree (for
/// example `model_type` is `deepseek_v41` at the top and `deepseek_v41_text` inside).
/// Only the keys in `TOP_LEVEL_CARRIED` are lifted, so unrelated container fields cannot
/// shadow a language-model field by accident.
fn flatten_text_config(raw: &Value) -> Result<Map<String, Value>> {
    let top = raw
        .as_object()
        .context("DeepSeek-V4.1 config.json is not an object")?;
    let text = top
        .get("text_config")
        .context("deepseek_v41 config missing text_config")?
        .as_object()
        .context("deepseek_v41 text_config is not an object")?;

    let mut flat = text.clone();
    for key in TOP_LEVEL_CARRIED {
        if let Some(value) = top.get(*key) {
            flat.entry((*key).to_string()).or_insert(value.clone());
        }
    }

    // `text_config.model_type` is "deepseek_v41_text"; the shared body stamps the canonical
    // model_type itself, but leaving the "_text" suffix here would make any raw-JSON reader
    // downstream disagree with `config.model_type`.
    flat.insert("model_type".into(), Value::String("deepseek_v41".into()));

    // VISION IS NOT WIRED. V4.1 nests a vision tower under `vision_config` with HF-style
    // names (hidden_size / intermediate_size / num_hidden_layers), whereas
    // `DeepSeekVisionConfig::parse_flat` looks for flat `vision_*` keys (vision_dim,
    // vision_inter_dim, vision_n_layers) as 0731 ships them. Because we do not lift
    // `vision_config`, that detector finds no `vision_` prefix and correctly returns None.
    // This is a deliberate omission, not an oversight: silently renaming the keys would
    // claim vision support the engine does not have. Text-only is the S0 deliverable.
    Ok(flat)
}
