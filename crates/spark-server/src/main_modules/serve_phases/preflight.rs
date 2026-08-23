// SPDX-License-Identifier: AGPL-3.0-only

//! GPU init + pre-load reserve preflight + post-load OOM check.

use anyhow::{Context, Result};

use atlas_core::config::ModelConfig;
use spark_model::model::ssm_pool_geometry::{
    SsmPoolGeometryInput, checked_ssm_pool_geometry, checked_ssm_speculative_geometry,
};

use crate::cli;

pub(crate) struct ReservePreflight {
    pub(crate) inference_reserve: usize,
    pub(crate) buffer_arena_bytes: usize,
    pub(crate) gdn_two_phase_bytes: usize,
    pub(crate) ssm_prefill_chunk: usize,
    pub(crate) max_batch_tokens_pre: usize,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct SsmPreflightCapacity {
    ssm_pool_bytes: usize,
    ssm_snapshot_bytes: usize,
}

fn checked_preflight_mul(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .with_context(|| format!("SSM preflight capacity overflow: {label} ({left} * {right})"))
}

fn checked_preflight_add(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_add(right)
        .with_context(|| format!("SSM preflight capacity overflow: {label} ({left} + {right})"))
}

fn checked_ssm_preflight_capacity(
    ssm_pool_bytes: usize,
    max_batch_size: usize,
    num_ssm_layers: usize,
    h_bytes: usize,
    conv_bytes: usize,
    ssm_cache_slots: usize,
    decode_ring_slots: usize,
) -> Result<SsmPreflightCapacity> {
    if num_ssm_layers == 0 {
        return Ok(SsmPreflightCapacity::default());
    }
    let snapshot_enabled = ssm_cache_slots > 0 || (max_batch_size > 0 && decode_ring_slots > 0);
    let ssm_snapshot_bytes = if snapshot_enabled {
        if h_bytes == 0 {
            anyhow::bail!("SSM preflight capacity: enabled snapshot h bytes must be positive");
        }
        if conv_bytes == 0 {
            anyhow::bail!("SSM preflight capacity: enabled snapshot conv bytes must be positive");
        }
        let bytes_per_layer = checked_preflight_add(h_bytes, conv_bytes, "SSM bytes per layer")?;
        let decode_region =
            checked_preflight_mul(max_batch_size, decode_ring_slots, "decode region slots")?;
        let snapshot_slots =
            checked_preflight_add(ssm_cache_slots, decode_region, "snapshot slots")?;
        let layer_bytes =
            checked_preflight_mul(num_ssm_layers, bytes_per_layer, "snapshot layer bytes")?;
        checked_preflight_mul(snapshot_slots, layer_bytes, "snapshot total bytes")?
    } else {
        0
    };

    Ok(SsmPreflightCapacity {
        ssm_pool_bytes,
        ssm_snapshot_bytes,
    })
}

fn checked_reserve_totals(
    ssm_pool_bytes: usize,
    ssm_snapshot_bytes: usize,
    gdn_two_phase_bytes: usize,
    cuda_headroom: usize,
    buffer_arena_bytes: usize,
) -> Result<(usize, usize)> {
    let ssm_total = checked_preflight_add(
        ssm_pool_bytes,
        ssm_snapshot_bytes,
        "reserve SSM pool + snapshot",
    )?;
    let with_gdn = checked_preflight_add(ssm_total, gdn_two_phase_bytes, "reserve + GDN")?;
    let inference_reserve =
        checked_preflight_add(with_gdn, cuda_headroom, "reserve + CUDA headroom")?;
    let total_reserve = checked_preflight_add(
        inference_reserve,
        buffer_arena_bytes,
        "reserve + buffer arena",
    )?;
    Ok((inference_reserve, total_reserve))
}

fn speculative_cuda_headroom(speculative_drafts: Option<usize>) -> usize {
    match speculative_drafts {
        Some(_) => 4 * 1024 * 1024 * 1024,
        None => 512 * 1024 * 1024,
    }
}

pub(crate) fn preflight_reserve(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    free_mem: usize,
) -> Result<ReservePreflight> {
    let num_ssm_layers = config.num_ssm_layers();
    let speculative_drafts = super::speculative_draft_count(args);
    let has_mtp = speculative_drafts.is_some();
    let num_drafts = speculative_drafts.unwrap_or(0);
    let requested_ddtree_capacity = std::env::var("ATLAS_DDTREE_MAX_NODES")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok());
    let speculative_geometry = checked_ssm_speculative_geometry(
        has_mtp,
        args.dflash,
        num_drafts,
        requested_ddtree_capacity,
    )?;
    let spec_tokens_pre = speculative_geometry.num_intermediates.max(1);
    let cuda_headroom = speculative_cuda_headroom(speculative_drafts);
    let decode_ring_slots = if num_ssm_layers > 0 {
        (atlas_kernels::ROLLBACK_RESTEER_CAP as usize)
            .checked_add(1)
            .context("SSM preflight decode ring slot count overflow")?
    } else {
        0
    };
    let pool_geometry = checked_ssm_pool_geometry(SsmPoolGeometryInput::from_config(
        config,
        args.max_batch_size,
        has_mtp,
        speculative_geometry.num_intermediates,
        spark_model::layers::wy17_lazy_commit(),
    )?)?;
    let ssm_capacity = checked_ssm_preflight_capacity(
        pool_geometry.total_bytes,
        args.max_batch_size,
        num_ssm_layers,
        pool_geometry.h_bytes,
        pool_geometry.conv_bytes,
        args.ssm_cache_slots,
        decode_ring_slots,
    )?;
    let ssm_pool_bytes = ssm_capacity.ssm_pool_bytes;
    let ssm_snapshot_bytes = ssm_capacity.ssm_snapshot_bytes;
    let ssm_prefill_chunk: usize = if num_ssm_layers > 0 {
        args.max_seq_len.min(8192)
    } else {
        0
    };
    let user_set_prefill_pre = args.max_prefill_tokens != 8192;
    let prefill_budget_pre = if user_set_prefill_pre && args.max_prefill_tokens > 0 {
        args.max_prefill_tokens
    } else if ssm_prefill_chunk > 0 {
        ssm_prefill_chunk
    } else if args.max_prefill_tokens > 0 {
        args.max_prefill_tokens
    } else {
        args.max_seq_len
    };
    // Mirror of the auto-clamp in resolve_prefill_budget (kv_cache.rs).
    // See issue #15: when prefix caching + SSM snapshots are both on,
    // single-chunk prefill produces no reachable intermediate snapshots.
    let prefill_budget_pre = if !user_set_prefill_pre
        && args.enable_prefix_caching
        && args.ssm_checkpoint_interval > 0
        && args.ssm_cache_slots > 0
    {
        let target = args.ssm_checkpoint_interval * args.block_size;
        if prefill_budget_pre > target && target > 0 {
            target
        } else {
            prefill_budget_pre
        }
    } else {
        prefill_budget_pre
    };
    let max_batch_tokens_pre = prefill_budget_pre
        .max(spec_tokens_pre)
        .max(args.max_batch_size);
    let buffer_arena_bytes = spark_runtime::buffers::BufferSizes::from_config(
        config,
        max_batch_tokens_pre,
        args.max_seq_len,
        args.block_size,
    )
    .total_bytes();
    let gdn_two_phase_bytes: usize = {
        let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let nv = config.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;
        if conv_dim > 0 && config.num_ssm_layers() > 0 {
            let sl = max_batch_tokens_pre;
            sl * conv_dim * 2 + sl * nv * 2 * 4 + sl * value_dim * 2 + sl * value_dim * 2
        } else {
            0
        }
    };
    let (inference_reserve, total_reserve) = checked_reserve_totals(
        ssm_pool_bytes,
        ssm_snapshot_bytes,
        gdn_two_phase_bytes,
        cuda_headroom,
        buffer_arena_bytes,
    )?;
    if total_reserve > free_mem {
        let need_gb = total_reserve as f64 / (1024.0 * 1024.0 * 1024.0);
        let free_gb = free_mem as f64 / (1024.0 * 1024.0 * 1024.0);
        let fixed = ssm_pool_bytes + ssm_snapshot_bytes + cuda_headroom;
        let budget_for_seq_term = free_mem.saturating_sub(fixed) / 2;
        let per_tok_bytes = {
            let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
            let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
            let nv = config.linear_num_value_heads;
            let conv_dim = key_dim * 2 + value_dim;
            if conv_dim > 0 && config.num_ssm_layers() > 0 {
                (conv_dim * 2) + (nv * 2 * 4) + (value_dim * 2) + (value_dim * 2)
            } else {
                0
            }
        };
        let suggested = budget_for_seq_term
            .checked_div(per_tok_bytes)
            .map(|q| q.max(2048))
            .unwrap_or(0);
        let hint = if suggested > 0 && suggested < args.max_seq_len {
            format!(
                " Try --max-seq-len {} (or lower --max-batch-size / --num-drafts).",
                suggested
            )
        } else if args.max_batch_size > 1 {
            " Reduce --max-batch-size.".to_string()
        } else {
            " Use a smaller model or a GPU with more memory.".to_string()
        };
        anyhow::bail!(
            "Preflight failed: inference buffers alone need {:.2} GB but only {:.2} GB is free on the GPU \
             (before weights load). SSM pool + GDN chunked prefill scales with --max-seq-len={} × --max-batch-size={}.{}",
            need_gb,
            free_gb,
            args.max_seq_len,
            args.max_batch_size,
            hint,
        );
    }
    tracing::info!(
        "Preflight reserve: inference={} MB, buffer_arena={} MB (pre-load free: {:.1} GB)",
        inference_reserve / (1024 * 1024),
        buffer_arena_bytes / (1024 * 1024),
        free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    // Q09: per-component breakdown so future MTP/spec-decode reserve
    // jumps are diagnosable from the log alone. Each line is dropped at
    // debug to avoid noise on hot startup paths; flip to info if you
    // need to trace a specific deployment's reserve.
    let spec_on = speculative_drafts.is_some();
    tracing::debug!(
        "Preflight reserve breakdown: \
         ssm_pool={} MB ({} dummy-inclusive slots × {} layers × {} state copies), \
         ssm_snapshot={} MB ({} slots), \
         gdn_two_phase={} MB ({} tokens), \
         cuda_headroom={} MB ({}), \
         spec_on={}, num_drafts={}",
        ssm_pool_bytes / (1024 * 1024),
        pool_geometry.total_slots,
        config.num_ssm_layers(),
        pool_geometry.state_copies,
        ssm_snapshot_bytes / (1024 * 1024),
        args.ssm_cache_slots,
        gdn_two_phase_bytes / (1024 * 1024),
        max_batch_tokens_pre,
        cuda_headroom / (1024 * 1024),
        if spec_on { "spec/MTP on" } else { "no spec" },
        spec_on,
        speculative_drafts.map_or(-1, |drafts| drafts as i64),
    );
    Ok(ReservePreflight {
        inference_reserve,
        buffer_arena_bytes,
        gdn_two_phase_bytes,
        ssm_prefill_chunk,
        max_batch_tokens_pre,
    })
}

/// Initialize the GPU backend for the active feature.
///
/// Compile-time dispatch:
/// - `cuda` feature → `AtlasCudaBackend` loading PTX modules from `ptx_set`.
/// - `metal` feature → `MetalGpuBackend` loading metallib modules from
///   `atlas_kernels::metallib_modules()`. The `ptx_set` argument is
///   accepted (for ABI symmetry with the cuda variant) but ignored;
///   metal kernels live in a parallel registry.
#[cfg(feature = "cuda")]
pub(crate) fn init_gpu_backend(
    args: &cli::ServeArgs,
    ptx_set: &atlas_kernels::TargetPtxSet,
) -> Result<(Box<dyn spark_runtime::gpu::GpuBackend>, usize)> {
    let gpu: Box<dyn spark_runtime::gpu::GpuBackend> = Box::new(
        spark_runtime::cuda_backend::AtlasCudaBackend::new(args.gpu_ordinal, &ptx_set.modules)
            .context("Failed to initialize CUDA backend")?,
    );
    let total_mem = gpu.total_memory()?;
    let free_mem = gpu.free_memory()?;
    tracing::info!(
        "GPU {}: {:.1} GB total, {:.1} GB free",
        args.gpu_ordinal,
        total_mem as f64 / (1024.0 * 1024.0 * 1024.0),
        free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    Ok((gpu, free_mem))
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(crate) fn init_gpu_backend(
    args: &cli::ServeArgs,
    _ptx_set: &atlas_kernels::TargetPtxSet,
) -> Result<(Box<dyn spark_runtime::gpu::GpuBackend>, usize)> {
    let modules = atlas_kernels::metallib_modules();
    let gpu: Box<dyn spark_runtime::gpu::GpuBackend> = Box::new(
        spark_runtime::metal_backend::MetalGpuBackend::new(args.gpu_ordinal, &modules)
            .context("Failed to initialize Metal backend")?,
    );
    let total_mem = gpu.total_memory()?;
    let free_mem = gpu.free_memory()?;
    tracing::info!(
        "Metal device {}: {:.1} GB total, {:.1} GB free",
        args.gpu_ordinal,
        total_mem as f64 / (1024.0 * 1024.0 * 1024.0),
        free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    Ok((gpu, free_mem))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn post_load_memory_audit(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    weight_bytes: usize,
    free_mem: usize,
    inference_reserve: usize,
    total_reserve: usize,
    gdn_two_phase_bytes: usize,
    max_batch_tokens_pre: usize,
) -> Result<()> {
    let estimated_free = free_mem.saturating_sub(weight_bytes);
    let actual_free = gpu.free_memory().unwrap_or(estimated_free);
    let available_free = if actual_free > 0 {
        actual_free
    } else {
        estimated_free
    };
    if available_free < total_reserve {
        let avail_gb = available_free as f64 / (1024.0 * 1024.0 * 1024.0);
        let need_gb = total_reserve as f64 / (1024.0 * 1024.0 * 1024.0);
        let hint = if args.max_batch_size > 1 {
            format!(
                " Reduce --max-batch-size (currently {}) or --max-seq-len (currently {}).",
                args.max_batch_size, args.max_seq_len
            )
        } else {
            format!(
                " Reduce --max-seq-len (currently {}) or use a smaller model.",
                args.max_seq_len
            )
        };
        anyhow::bail!(
            "Insufficient GPU memory for inference buffers. \
             After loading {:.2} GB of weights, only {:.2} GB remains \
             but {:.2} GB is needed for SSM state pool ({} slots × {} layers) + scratch buffers.{}",
            weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            avail_gb,
            need_gb,
            args.max_batch_size,
            config.num_ssm_layers(),
            hint,
        );
    }
    if gdn_two_phase_bytes > 0 {
        tracing::info!(
            "GDN chunked prefill reserve: {} MB (chunk_size={}, max_seq_len={})",
            gdn_two_phase_bytes / (1024 * 1024),
            max_batch_tokens_pre,
            args.max_seq_len,
        );
    }
    tracing::info!(
        "Weights: {:.2} GB, estimated free: {:.1} GB, actual free: {:.1} GB (reserve: {} MB)",
        weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        estimated_free as f64 / (1024.0 * 1024.0 * 1024.0),
        actual_free as f64 / (1024.0 * 1024.0 * 1024.0),
        inference_reserve / (1024 * 1024),
    );
    Ok(())
}

/// Initialize the REST retrieval draft store against the tokenizer the
/// server actually loaded.
///
/// Call once, after the tokenizer is available: the store carries the
/// fingerprint of the `tokenizer.json` it was built from, and a mismatch
/// is FATAL. Token ids minted by a different tokenizer are not
/// lower-quality drafts, they are noise — every one of them would be
/// rejected at verify while still costing a verify slot.
///
/// A no-op returning `Ok(())` when `ATLAS_REST_STORE` is unset, which is
/// the default.
pub(crate) fn init_rest_store(model_dir: &std::path::Path) -> Result<()> {
    if std::env::var_os("ATLAS_REST_STORE").is_none() {
        return Ok(());
    }
    // The same file `ChatTokenizer::from_model_dir` loads, so the
    // fingerprint is taken over the bytes actually in use.
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer_json = std::fs::read(&tokenizer_path).with_context(|| {
        format!(
            "ATLAS_REST_STORE is set but {} could not be read to validate the store",
            tokenizer_path.display()
        )
    })?;
    crate::rest_store::init(&tokenizer_json).context("REST draft store rejected at startup")?;
    Ok(())
}

#[cfg(test)]
mod dflash_preflight_shape_tests {
    use super::speculative_cuda_headroom;

    #[test]
    fn dflash_block16_reserves_for_fifteen_drafts() {
        assert_eq!(speculative_cuda_headroom(Some(15)), 4 * 1024 * 1024 * 1024);
    }

    #[test]
    fn plain_decode_keeps_the_small_reserve() {
        assert_eq!(speculative_cuda_headroom(None), 512 * 1024 * 1024);
    }
}

#[cfg(test)]
#[path = "preflight_ssm_capacity_tests.rs"]
mod ssm_capacity_tests;
