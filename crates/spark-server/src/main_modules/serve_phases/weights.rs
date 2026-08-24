// SPDX-License-Identifier: AGPL-3.0-only

//! Weight-store loading: main checkpoint, prefix auto-detect, DFlash drafter.

use std::path::Path;

use anyhow::{Context, Result};

use atlas_core::config::ModelConfig;

use crate::cli;

pub(crate) fn quant_multiplier(config: &ModelConfig) -> Option<f64> {
    // Manual override for checkpoints whose peak/on-disk ratio the format
    // heuristics below misjudge (e.g. bring-up of a new quant format).
    if let Some(m) = std::env::var("ATLAS_PEAK_MEM_MULT")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|&m| m >= 1.0)
    {
        return Some(m);
    }
    if config.model_type == "minimax_m2" || config.model_type == "step3p7" {
        Some(1.02)
    } else if config
        .quantization_config
        .as_ref()
        .is_some_and(|qc| qc.quant_method == "exl3")
    {
        // EXL3 trellis experts load zero-copy as-is (no transpose pass, no
        // NVFP4 tables, no BF16 dequant staging) and the base FP8 tensors
        // follow the normal FP8 path; without this arm the loader falls to
        // the has_fp8 default of 1.5x and the ~99 GiB tp1 checkpoint fails
        // the pre-flight on a GB10 that fits it comfortably.
        Some(1.05)
    } else if config
        .quantization_config
        .as_ref()
        .is_some_and(|qc| qc.quant_method == "fp8")
    {
        Some(1.05)
    } else {
        None
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
            loader.prefetch_shards = args.fast_load_prefetch_shards
                || std::env::var("ATLAS_FAST_LOAD_PREFETCH_SHARDS")
                    .ok()
                    .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
            if loader.prefetch_shards {
                tracing::info!("Fast weight loader shard prefetch/readahead enabled");
            }
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

/// Everything the factory needs about the drafter checkpoint: its on-device
/// store, its parsed DFlash config (`None` = DSpark), and — for a DSpark
/// drafter under `ATLAS_DSPARK_REF_DRAFT=1` — the compact-draft routed-expert
/// subset resolved from the checkpoint's own REAP provenance map.
pub(crate) type DrafterState = (
    spark_runtime::weights::WeightStore,
    Option<spark_model::weight_loader::DflashConfig>,
    Option<spark_model::weight_loader::deepseek_v4::dspark_reap::DraftExpertSubset>,
);

/// Does this tensor belong to the DSpark drafter rather than the target?
///
/// Only meaningful for the shared-checkpoint layout (the reference tp1 build),
/// where the drafter's `mtp.*` stages sit in the same directory as the target
/// weights. `load_dspark_drafter` consumes the mtp stages plus the Markov and
/// confidence heads; everything else in that directory is the target's and is
/// already resident by the time the drafter loads.
fn is_drafter_tensor(name: &str) -> bool {
    name.starts_with("mtp.")
        || name.contains("markov")
        || name.contains("confidence")
        || name.starts_with("dspark")
}

pub(crate) fn load_dflash_drafter(
    args: &cli::ServeArgs,
    ptx_set: &atlas_kernels::TargetPtxSet,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    target_store: &spark_runtime::weights::WeightStore,
) -> Result<Option<DrafterState>> {
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
    // Does the drafter live in the target's own directory? Compare canonical
    // paths so a trailing slash or a symlink cannot defeat it.
    let shares_target_dir = args
        .model
        .as_deref()
        .map(std::path::Path::new)
        .and_then(|p| p.canonicalize().ok())
        .zip(std::path::Path::new(&drafter_id).canonicalize().ok())
        .is_some_and(|(a, b)| a == b);
    let drafter_dir =
        crate::model_resolver::resolve_model_dir(&drafter_id, args.cache_dir.as_deref())
            .context("Failed to resolve DFlash drafter checkpoint")?;
    // DSpark block drafter (docs/dspark_port.md): the official 0731 drafter
    // shards carry no drafter config.json; the `mtp.0.main_proj.weight`
    // tensor in the safetensors index is the marker. `None` config tells the
    // factory to build the DSpark head instead of DFlash.
    let is_dspark = std::fs::read_to_string(drafter_dir.join("model.safetensors.index.json"))
        .map(|j| j.contains("mtp.0.main_proj.weight"))
        .unwrap_or(false);
    let drafter_config = if is_dspark {
        tracing::info!("Drafter store detected as DSpark (mtp.0.main_proj marker)");
        None
    } else {
        let drafter_config_json = std::fs::read_to_string(drafter_dir.join("config.json"))
            .with_context(|| {
                format!(
                    "Failed to read drafter config.json at {}",
                    drafter_dir.display()
                )
            })?;
        Some(spark_model::weight_loader::dflash_loader::parse_dflash_config(&drafter_config_json)?)
    };
    // Compact DSpark draft (ATLAS_DSPARK_REF_DRAFT=1, default off): resolve
    // WHICH routed experts the draft keeps before loading, so the other ~70%
    // of the drafter's expert bytes are never read off disk or allocated.
    // Resolved here because this is where the drafter DIRECTORY is known — the
    // REAP provenance map sits next to the shards.
    let dspark_subset = if is_dspark {
        spark_model::weight_loader::deepseek_v4::dspark::resolve_ref_draft_subset(&drafter_dir)?
    } else {
        None
    };

    // A shared K2/K3 checkpoint already loaded every embedded `mtp.*`,
    // Markov, and confidence tensor into the target store. Build a filtered
    // pointer view instead of allocating the same ~5.5 GiB a second time.
    // WeightStore is metadata-only, so this neither copies nor double-owns
    // device memory.
    if shares_target_dir && is_dspark {
        let compact_skip = dspark_subset
            .as_ref()
            .map(spark_model::weight_loader::deepseek_v4::dspark::compact_draft_skip_fn);
        let drafter_store = target_store.filtered_view(|name| {
            is_drafter_tensor(name) && !compact_skip.as_ref().is_some_and(|skip| skip(name))
        });
        tracing::info!(
            "DFlash drafter store: reusing {} target-store tensors ({} bytes, zero-copy){}",
            drafter_store.len(),
            drafter_store.total_bytes(),
            match dspark_subset {
                Some(ref s) => format!(" (compact draft: {} routed experts)", s.len()),
                None => String::new(),
            }
        );
        return Ok(Some((drafter_store, drafter_config, dspark_subset)));
    }

    let mut loader = spark_runtime::weights::SafetensorsLoader::new();
    // Honour the SAME multiplier the target used, instead of falling to the
    // 1.3x default. When the drafter shares the target's directory — which is
    // exactly the DSpark-in-checkpoint layout of the reference tp1 build — the
    // pre-flight sizes the WHOLE shared dir, so 99.26 GB x 1.3 = 129.04 GB and
    // the serve aborts before loading a byte, even though the drafter only
    // reads the mtp.* shards (~8.6 GiB). Passing the target's multiplier makes
    // `ATLAS_PEAK_MEM_MULT` and the EXL3 1.05 case reach this path.
    //
    // NOTE: this is the conservative half of the fix. The estimate is still
    // computed over the whole directory rather than the mtp.* subset the skip
    // function actually loads, so it remains an OVER-estimate — safe, just not
    // tight. Sizing from the subset is the real fix and needs the loader to
    // know its own skip predicate before the pre-flight runs.
    loader.peak_memory_multiplier = std::env::var("ATLAS_PEAK_MEM_MULT")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|&m| m >= 1.0);
    // The two predicates are ORTHOGONAL and must COMPOSE: the compact-draft one
    // prunes routed experts WITHIN mtp.*, the shared-dir one excludes the
    // target's tensors entirely. Applying only the first (the original
    // `else if`) still admitted every target tensor minus pruned experts —
    // measured 35.58 GB where the drafter needs 2.94.
    if let Some(ref subset) = dspark_subset {
        let compact =
            spark_model::weight_loader::deepseek_v4::dspark::compact_draft_skip_fn(subset);
        loader.extra_skip = if shares_target_dir {
            Some(std::sync::Arc::new(move |name: &str| {
                !is_drafter_tensor(name) || compact(name)
            }))
        } else {
            Some(compact)
        };
    } else if shares_target_dir {
        // SHARED-CHECKPOINT LAYOUT (the reference tp1 build): the DSpark
        // drafter's `mtp.*` shards sit in the SAME directory as the target
        // weights. With no skip predicate the drafter load admits every tensor
        // in the directory — it would pull the target's ~99 GB into a second
        // store, and the OOM pre-flight was correctly reporting that intent
        // (99.26 GB estimate) rather than mis-measuring.
        //
        // Admit only what the drafter actually consumes: the `mtp.*` stages
        // plus the Markov / confidence heads that `load_dspark_drafter` looks
        // up by name. Everything else is the target's, already resident.
        loader.extra_skip = Some(std::sync::Arc::new(|name: &str| !is_drafter_tensor(name)));
    }
    let drafter_store = loader
        .load(&drafter_dir, gpu, 0)
        .context("Failed to load DFlash drafter weights")?;
    tracing::info!(
        "DFlash drafter store: {} tensors, {} bytes{}",
        drafter_store.len(),
        drafter_store.total_bytes(),
        match dspark_subset {
            Some(ref s) => format!(" (compact draft: {} routed experts)", s.len()),
            None => String::new(),
        }
    );
    Ok(Some((drafter_store, drafter_config, dspark_subset)))
}

/// Startup-loaded LoRA adapter: its own WeightStore + parsed PEFT config.
/// One `LoraAdapterState` per repeated `--lora-adapter NAME=PATH`; each becomes
/// one resident pool slot. A single adapter is byte-identical to the v0 path.
pub(crate) struct LoraAdapterState {
    pub name: String,
    pub peft_config: atlas_core::config::PeftAdapterConfig,
    pub store: spark_runtime::weights::WeightStore,
}

/// Resolve + load every `--lora-adapter` into its own on-device `WeightStore`
/// (slot 0..N-1). Empty when no adapter is requested. Rejects >`--max-loras`
/// adapters and duplicate names up front.
pub(crate) fn load_lora_adapters(
    args: &cli::ServeArgs,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
) -> Result<Vec<LoraAdapterState>> {
    if args.lora_adapter.is_empty() {
        return Ok(Vec::new());
    }
    if args.lora_adapter.len() > args.max_loras {
        anyhow::bail!(
            "--lora-adapter given {} times but --max-loras={} (pool has {} slots); \
             raise --max-loras or stage the extras on an $ATLAS_LORA_PEER",
            args.lora_adapter.len(),
            args.max_loras,
            args.max_loras,
        );
    }
    let mut states: Vec<LoraAdapterState> = Vec::with_capacity(args.lora_adapter.len());
    for (name, spec) in &args.lora_adapter {
        if states.iter().any(|s| &s.name == name) {
            anyhow::bail!("--lora-adapter name '{name}' given twice (names must be unique)");
        }
        tracing::info!("LoRA: resolving adapter '{name}' from '{spec}'");
        let adapter_dir =
            crate::model_resolver::resolve_adapter_dir(spec, args.cache_dir.as_deref())
                .context("Failed to resolve LoRA adapter")?;
        let cfg_path = adapter_dir.join("adapter_config.json");
        let raw = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("Failed to read {}", cfg_path.display()))?;
        // Hard-error parser (atlas-core config/parsers/lora.rs) — scaling is read
        // per adapter (alpha/r, alpha/sqrt(r) under use_rslora), NEVER defaulted.
        let peft_config = atlas_core::config::parse_peft_adapter_config(&raw)
            .with_context(|| format!("Failed to parse {}", cfg_path.display()))?;
        if peft_config.r > args.max_lora_rank {
            anyhow::bail!(
                "LoRA adapter '{}' has r={} > --max-lora-rank {} — raise the flag \
                 (slot pool is rank-padded to it) or use a smaller adapter",
                name,
                peft_config.r,
                args.max_lora_rank,
            );
        }
        let store = spark_runtime::weights::adapter::load_adapter_safetensors(&adapter_dir, gpu, 0)
            .context("Failed to load LoRA adapter weights")?;
        tracing::info!(
            "LoRA adapter '{}': {} tensors, {} bytes loaded; r={}, alpha={}, \
             use_rslora={}, scaling={:.6}, target_modules={:?}",
            name,
            store.len(),
            store.total_bytes(),
            peft_config.r,
            peft_config.lora_alpha,
            peft_config.use_rslora,
            peft_config.scaling(),
            peft_config.target_modules,
        );
        states.push(LoraAdapterState {
            name: name.clone(),
            peft_config,
            store,
        });
    }
    Ok(states)
}
