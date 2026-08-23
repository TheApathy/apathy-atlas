// SPDX-License-Identifier: AGPL-3.0-only

//! Weight-store loading: main checkpoint, prefix auto-detect, DFlash drafter.

use std::path::Path;

use anyhow::{Context, Result};

use atlas_core::config::ModelConfig;

use crate::cli;

pub(crate) fn quant_multiplier(config: &ModelConfig) -> Option<f64> {
    if config.model_type == "minimax_m2" {
        return Some(1.02);
    }
    let qc = config.quantization_config.as_ref()?;
    if qc.quant_method == "fp8" {
        return Some(1.05);
    }
    // NVFP4: weights are mmap'd zero-copy, so the *store* itself costs
    // ~1× on-disk. This ratio covers only that. Everything the model
    // builder retains on top of the store is architecture-dependent and
    // is estimated in absolute bytes by `construction_overhead_bytes`
    // below — do NOT fold it into this constant. A blanket ratio big
    // enough for a hybrid linear-attention model false-OOMs a plain
    // one: the upstream fallback of 1.3× already false-OOM'd 76 GB
    // heretic 122B on a 119 GB GPU.
    let is_nvfp4 = (qc.quant_method == "modelopt" && qc.quant_algo.eq_ignore_ascii_case("NVFP4"))
        || qc.quant_method == "compressed-tensors";
    if is_nvfp4 {
        return Some(1.02);
    }
    None
}

/// NVFP4 scale group size used by `QuantizedWeight`
/// (`spark-model/src/weight_map/quantized.rs`).
const NVFP4_GROUP: usize = 16;

/// Bytes an NVFP4 `QuantizedWeight` of logical shape `[n, k]` occupies:
/// packed nibbles `[n, k/2]` + per-group FP8 scales `[n, k/NVFP4_GROUP]`.
fn nvfp4_bytes(n: usize, k: usize) -> usize {
    n * k / 2 + n * k / NVFP4_GROUP
}

/// Bytes a `QuantizedWeight::transpose_for_gemm(_, n, k)` result occupies.
/// The transposed layout pads N up to 64 for cp.async alignment — see
/// `spark-model/src/weight_map/quantized.rs:59`.
fn nvfp4_transposed_bytes(n: usize, k: usize) -> usize {
    let n_pad = n.div_ceil(64) * 64;
    (k / 2) * n_pad + (k / NVFP4_GROUP) * n_pad
}

/// GPU bytes the model *builder* retains on top of the weight store,
/// broken down by source. Absolute bytes, not a ratio.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ConstructionOverhead {
    /// Per-linear-attention-layer buffers built by the SSM loader arms
    /// (`weight_loader/qwen35_dense.rs` and
    /// `weight_loader/qwen35/load_layers/linear_attn_arms.rs`).
    pub(crate) ssm: usize,
    /// Transposed NVFP4 copies of the dense FFN projections
    /// (`ATLAS_FFN_M16_TRANSPOSED`, default on).
    pub(crate) ffn_transposed: usize,
    /// Transposed NVFP4 q/k/v/o copies on full-attention layers.
    pub(crate) attn_transposed: usize,
}

impl ConstructionOverhead {
    pub(crate) fn total(&self) -> usize {
        self.ssm + self.ffn_transposed + self.attn_transposed
    }
}

/// Estimate the GPU memory the model builder allocates *and keeps* after
/// the weight store finishes loading.
///
/// Why this exists: the load pre-flight is the only guard against
/// over-allocating at startup, and on GB10's unified memory an
/// over-allocation is a host OOM, not a clean CUDA failure. Before this
/// function the pre-flight modelled construction as a flat 1.02× of the
/// on-disk bytes, on the belief that the SSM QKV/Z concat was freed once
/// the layer had been quantized. It is not: `qkvz_dense` is moved into
/// `SsmWeights.in_proj_qkvz` (`qwen35_dense.rs:378`,
/// `linear_attn_arms.rs:241`) and lives for the process lifetime,
/// alongside *both* NVFP4 copies. Neither `DenseWeight` nor
/// `QuantizedWeight` implements `Drop`, and the loader deliberately does
/// not call `gpu.free()` (BUG #29: freeing on GB10 UVM posts in-band TLB
/// invalidations that corrupt neighbouring allocations). So every buffer
/// counted here is retained, by design.
///
/// Scope: only hybrid linear-attention (GDN/SSM) checkpoints. Models with
/// no `linear_attention` layers return zero and keep their previous
/// pre-flight behaviour exactly. MoE expert transposition is excluded on
/// purpose — it has its own free-memory guard at
/// `weight_loader/qwen35/load_layers.rs:102` and skips itself when it
/// would not fit, so charging it here would false-OOM MoE checkpoints.
///
/// Not modelled (all post-store, all comparatively small): the `lm_head`
/// NVFP4 quantization, the MTP head, ViT scratch, the DFlash drafter, and
/// the split-K FP32 workspaces. Together ~2 GB on Qwen3.8-27B.
pub(crate) fn construction_overhead_bytes(config: &ModelConfig) -> ConstructionOverhead {
    let ssm_layers = config.num_ssm_layers();
    if ssm_layers == 0 {
        return ConstructionOverhead::default();
    }

    let h = config.hidden_size;
    let qkvz = config.ssm_qkvz_size();
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;

    // `out_proj` is dequantized to a fresh BF16 buffer by
    // `load_ssm_qwen35` only when it is packed on disk. Checkpoints that
    // list it in the quantizer's ignore list ship it as BF16 and the
    // loader aliases the store pointer instead — no extra allocation.
    let out_proj_packed_on_disk = config.quantization_config.as_ref().is_none_or(|qc| {
        !qc.ignore_modules
            .iter()
            .any(|m| m.contains("linear_attn.out_proj"))
    });
    let out_proj_bf16_dequant = if out_proj_packed_on_disk {
        h * value_dim * 2
    } else {
        0
    };

    let per_ssm_layer =
        // qkvz BF16 concat, retained as `SsmWeights.in_proj_qkvz`.
        qkvz * h * 2
        // qkvz NVFP4 copy + its transposed duplicate.
        + nvfp4_bytes(qkvz, h)
        + nvfp4_transposed_bytes(qkvz, h)
        // BF16 dequant of a packed on-disk out_proj, never freed.
        + out_proj_bf16_dequant
        // out_proj NVFP4 + transposed duplicate + FP8 prefill predequant
        // (`layers/qwen3_ssm/init.rs::predequant_for_prefill`).
        + nvfp4_bytes(h, value_dim)
        + nvfp4_transposed_bytes(h, value_dim)
        + h * value_dim
        // interleave_ba BF16 output.
        + config.ssm_ba_size() * h * 2;

    // Dense-FFN transposed NVFP4 copies. MoE checkpoints take the
    // expert path instead (guarded separately, see above).
    let ffn_transposed =
        if config.num_experts == 0 && spark_model::layers::ffn_m16_transposed_enabled() {
            let inter = config.intermediate_size;
            let per_layer = nvfp4_transposed_bytes(inter, h)   // gate
                + nvfp4_transposed_bytes(inter, h)             // up
                + nvfp4_transposed_bytes(h, inter); // down
            per_layer * config.num_hidden_layers
        } else {
            0
        };

    // Transposed q/k/v/o on full-attention layers. Unconditional on both
    // the dense (`qwen35_dense.rs:293`) and MoE
    // (`load_layers/attention_arms.rs:196`) hybrid paths.
    let attn_transposed = {
        let head_dim = config.head_dim;
        let q_out = config.num_attention_heads * head_dim * if config.attn_gated { 2 } else { 1 };
        let kv_out = config.num_key_value_heads * head_dim;
        let per_layer = nvfp4_transposed_bytes(q_out, h)
            + 2 * nvfp4_transposed_bytes(kv_out, h)
            + nvfp4_transposed_bytes(h, config.num_attention_heads * head_dim);
        per_layer * config.num_attention_layers()
    };

    ConstructionOverhead {
        ssm: per_ssm_layer * ssm_layers,
        ffn_transposed,
        attn_transposed,
    }
}

pub(crate) fn load_weight_store(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    model_dir: &Path,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    ep_rank: usize,
    ep_size: usize,
    oom_reserve_bytes: usize,
) -> Result<spark_runtime::weights::WeightStore> {
    use spark_runtime::weights::WeightLoader;
    let mult = quant_multiplier(config);
    let overhead = construction_overhead_bytes(config);
    let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
    if overhead.total() > 0 {
        // The window between "weight store loaded" and "KV cache sized"
        // used to consume tens of GB with no byte-count log line at all,
        // which is how the 1.02× under-estimate stayed hidden. Print the
        // prediction here so it can be diffed against the free-memory
        // deltas either side of model construction without an audit.
        tracing::info!(
            "Model-construction overhead estimate: {:.2} GB retained after the weight \
             store loads — SSM linear-attn {:.2} GB ({} layers), dense-FFN transposes \
             {:.2} GB, attention transposes {:.2} GB ({} layers). Charged to the load \
             pre-flight on top of the {:.2}x on-disk ratio. Excludes lm_head / MTP / \
             ViT / DFlash drafter.",
            gib(overhead.total()),
            gib(overhead.ssm),
            config.num_ssm_layers(),
            gib(overhead.ffn_transposed),
            gib(overhead.attn_transposed),
            config.num_attention_layers(),
            mult.unwrap_or(1.0),
        );
    }
    let use_fast_load =
        !args.no_fast_load && std::env::var("ATLAS_FAST_LOAD").ok().as_deref() != Some("0");
    let store = if use_fast_load {
        #[cfg(unix)]
        {
            tracing::info!("Using fast weight loader (O_DIRECT + pipelined read/copy)");
            let mut loader = if ep_size > 1 {
                spark_runtime::fast_weights::FastSafetensorsLoader::with_ep(
                    ep_rank,
                    ep_size,
                    config.num_experts,
                )
            } else {
                spark_runtime::fast_weights::FastSafetensorsLoader::new()
            };
            loader.peak_memory_multiplier = mult;
            loader.construction_overhead_bytes = overhead.total();
            loader
                .load(model_dir, gpu, oom_reserve_bytes)
                .context("Failed to load model weights (fast loader)")?
        }
        #[cfg(not(unix))]
        {
            anyhow::bail!("--fast-load requires a Unix host (needs O_DIRECT / posix_fadvise)");
        }
    } else {
        let mut loader = if ep_size > 1 {
            spark_runtime::weights::SafetensorsLoader::with_ep(ep_rank, ep_size, config.num_experts)
        } else {
            spark_runtime::weights::SafetensorsLoader::new()
        };
        loader.peak_memory_multiplier = mult;
        loader.construction_overhead_bytes = overhead.total();
        loader
            .load(model_dir, gpu, oom_reserve_bytes)
            .context("Failed to load model weights")?
    };
    tracing::info!("Loaded {} weight tensors", store.len());
    Ok(store)
}

pub(crate) fn auto_detect_weight_prefix(
    store: &spark_runtime::weights::WeightStore,
    config: &mut ModelConfig,
) {
    if config.weight_prefix.is_empty() && config.nested_config {
        config.weight_prefix = if store.contains("language_model.model.embed_tokens.weight") {
            "language_model.model".to_string()
        } else if store.contains("model.language_model.embed_tokens.weight") {
            "model.language_model".to_string()
        } else {
            let scanned = store
                .names()
                .find(|k| k.contains(".layers.0."))
                .and_then(|k| k.split(".layers.0.").next())
                .map(|s| s.to_string());
            if let Some(ref prefix) = scanned {
                tracing::info!("Auto-detected weight prefix: '{prefix}'");
            }
            scanned.unwrap_or_else(|| "model".to_string())
        };
    }
    if !config.weight_prefix.is_empty() {
        tracing::info!("Weight prefix: {}", config.weight_prefix);
    }
}

pub(crate) fn load_dflash_drafter(
    args: &cli::ServeArgs,
    ptx_set: &atlas_kernels::TargetPtxSet,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    verify_mode: spark_model::weight_loader::DsparkVerifyMode,
) -> Result<
    Option<(
        spark_runtime::weights::WeightStore,
        spark_model::weight_loader::DflashConfig,
    )>,
> {
    use spark_runtime::weights::WeightLoader;
    if !args.dflash {
        return Ok(None);
    }
    let drafter_id = args
        .draft_model
        .clone()
        .or_else(|| ptx_set.dflash.as_ref().map(|d| d.draft_model.to_string()))
        .context(
            "--dflash set but no drafter HF id provided: pass --draft-model <ID> \
             or use a target whose MODEL.toml has a [dflash] section",
        )?;
    tracing::info!("DFlash: resolving drafter '{drafter_id}'");
    let drafter_dir =
        crate::model_resolver::resolve_model_dir(&drafter_id, args.cache_dir.as_deref())
            .context("Failed to resolve DFlash drafter checkpoint")?;
    let drafter_config_json = std::fs::read_to_string(drafter_dir.join("config.json"))
        .with_context(|| {
            format!(
                "Failed to read drafter config.json at {}",
                drafter_dir.display()
            )
        })?;
    let drafter_config =
        spark_model::weight_loader::dflash_loader::parse_dflash_config(&drafter_config_json)?;
    // Validate the requested planner before mapping the multi-GB drafter.
    // Dynamic modes must never degrade silently to static verify-all.
    drafter_config.validate_verify_mode(verify_mode)?;
    let mut loader = spark_runtime::weights::SafetensorsLoader::new();
    loader.peak_memory_multiplier = None;
    let drafter_store = loader
        .load(&drafter_dir, gpu, 0)
        .context("Failed to load DFlash drafter weights")?;
    tracing::info!(
        "DFlash drafter store: {} tensors, {} bytes",
        drafter_store.len(),
        drafter_store.total_bytes()
    );
    Ok(Some((drafter_store, drafter_config)))
}
