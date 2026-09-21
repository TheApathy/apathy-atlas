// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_model::factory::vision_speculation::{Qualification, Request};
use std::path::Path;

pub(crate) fn validate_request(
    args: &crate::cli::ServeArgs,
    model_dir: &Path,
    config: &ModelConfig,
    raw: &str,
) -> Result<bool> {
    let policy = Qualification::from_env().map_err(anyhow::Error::msg)?;
    let target_only =
        !args.dflash && !args.speculative && !args.self_speculative && !args.ngram_speculative;
    if !policy.enabled() || target_only {
        return Ok(false);
    }
    let draft_path = args
        .draft_model
        .as_deref()
        .context("Vision DSpark needs an explicit --draft-model equal to --model-from-path")?;
    let embedded = Path::new(draft_path)
        .canonicalize()
        .context("resolve local Vision DSpark path")?
        == model_dir.canonicalize()?;
    policy
        .admit(Request {
            is_vision: config.deepseek_vision.is_some(),
            target_only,
            dflash_only: args.dflash
                && !args.speculative
                && !args.self_speculative
                && !args.ngram_speculative,
            embedded_drafter: embedded,
            high_speed_swap: args.high_speed_swap,
            verify_capacity: if args.max_prefill_tokens == 0 {
                args.max_seq_len
            } else {
                args.max_prefill_tokens
            },
            has_adapters: !args.lora_adapter.is_empty()
                || !args.lora_stageable.is_empty()
                || !args.lora_stageable_disk.is_empty(),
        })
        .map_err(anyhow::Error::msg)?;
    policy.validate_runtime().map_err(anyhow::Error::msg)?;
    ensure!(
        args.dflash_gamma == 6,
        "Vision DSpark qualification requires --dflash-gamma 6 (five proposals)"
    );
    let raw: serde_json::Value = serde_json::from_str(raw)?;
    for (key, expected) in [
        ("dspark_block_size", 5),
        ("dspark_markov_rank", 256),
        ("dspark_noise_token_id", 128799),
        ("sliding_window", 128),
        ("num_nextn_predict_layers", 3),
    ] {
        ensure!(
            raw.get(key).and_then(serde_json::Value::as_u64) == Some(expected),
            "unsupported Vision DSpark metadata: {key}"
        );
    }
    ensure!(
        raw.get("dspark_target_layer_ids") == Some(&serde_json::json!([40, 41, 42])),
        "unsupported Vision DSpark capture layers"
    );
    let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        model_dir.join("model.safetensors.index.json"),
    )?)?;
    ensure!(
        index
            .get("weight_map")
            .and_then(|map| map.get("mtp.0.main_proj.weight"))
            .and_then(serde_json::Value::as_str)
            .is_some(),
        "Vision checkpoint index has no embedded DSpark projection"
    );
    Ok(true)
}
