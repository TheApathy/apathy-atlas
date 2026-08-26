// SPDX-License-Identifier: AGPL-3.0-only

//! `build_model` — entry point that wires up the configured loader,
//! buffers, KV cache, and (optional) DFlash drafter into a `TransformerModel`.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use spark_runtime::prefix_cache::PrefixCache;
use spark_runtime::weights::WeightStore;

use super::DflashBuildArgs;
use super::loader_for_config;
use super::m2_setup::maybe_run_minimax_m2_moe_transpose;
use crate::layers::MtpQuantization;
use crate::model::TransformerModel;
use crate::traits::Model;
use crate::weight_loader::load_dflash_weights;
use crate::weight_map::quantize_to_nvfp4;

pub fn build_model(
    mut config: ModelConfig,
    store: &WeightStore,
    gpu: Box<dyn GpuBackend>,
    max_batch_tokens: usize,
    kv_block_size: usize,
    max_seq_len: usize,
    max_batch_size: usize,
    mtp_quant: MtpQuantization,
    use_speculative: bool,
    prefix_cache: Box<dyn PrefixCache>,
    mtp_vocab_size: u32,
    comm: Option<std::sync::Arc<dyn spark_comm::CommBackend>>,
    self_speculative: bool,
    num_drafts: usize,
    kv_dtype: KvCacheDtype,
    inference_reserve: usize,
    gpu_memory_utilization: f64,
    ssm_cache_slots: usize,
    layer_dtypes: Vec<KvCacheDtype>,
    ssm_checkpoint_interval: usize,
    // Phase 6.1.f: per-sequence HBM cache cap. `Some(N)` enables
    // `--high-speed-swap` HBM-shrink behavior. `None` preserves the
    // pre-Phase-6 unbounded behavior.
    hss_cache_blocks_per_seq: Option<u32>,
    // DFlash speculative-decoding pairing. `None` = no DFlash; existing
    // MTP / no-spec paths unchanged.
    dflash_args: Option<DflashBuildArgs<'_>>,
) -> Result<Box<dyn Model>> {
    // ── Step 1: Select weight loader (only model-specific dispatch) ──
    let loader = loader_for_config(&config)?;

    // Pre-construction: when DFlash is active, populate the target's
    // capture-layer indices from the drafter's `dflash_config.target_layer_ids`
    // so `TransformerModel::new` allocates the 5×hidden_size capture buffer.
    //
    // OFFSET 0 IS CORRECT. Do not "fix" it to 1 — see below, because the
    // comment that used to live here argued for 1 and was wrong.
    //
    // The drafter's raw `target_layer_ids` are used verbatim: Atlas must
    // capture the SAME tensors SpecForge captured when it generated the
    // training data, and it already does.
    //
    //   training  — specforge/modeling/target/dflash_target_model.py:270-276
    //               reads `outputs.hidden_states[idx + 1]`. That `+1` skips
    //               the HF tuple's element 0 (the embedding output), so
    //               `hidden_states[N+1]` IS the OUTPUT OF LAYER N. It is a
    //               tuple-index shift, NOT a semantic layer shift.
    //   serving   — `trait_impl/decode_a.rs:188-205` calls
    //               `try_dflash_capture(i, ..)` immediately AFTER
    //               `layer.decode()` for index i, i.e. it already holds the
    //               OUTPUT OF LAYER i. There is no tuple to index.
    //
    // Both sides therefore reference output-of-layer-N with no adjustment,
    // and alignment requires offset == 0.
    //
    // The superseded comment cited vLLM PR #40898 (@jianc99) applying a "+1
    // correctness fix" and concluded Atlas needs it too. vLLM adds 1 for the
    // same tuple-indexing reason SpecForge does; importing it here would
    // double-count the correction and read one layer too DEEP. The old text
    // even contained its own disproof — "Atlas captures AFTER layer.decode()
    // for the listed index, so we add 1" — where the premise is exactly why
    // the conclusion does not follow. Git history agrees: shipped at -1 (one
    // layer too shallow, genuinely broken), corrected to 0 in f81ae296, and
    // never 1.
    //
    // ATLAS_DFLASH_CAPTURE_LAYER_OFFSET exists only for A/B testing that
    // claim. Every nonzero value misaligns the drafter against its training
    // data and shows up as degraded acceptance, never as an error — hence the
    // warning below.
    if let Some(ref args) = dflash_args
        && let Some(ref sub) = args.drafter_config.dflash_config
    {
        let offset: i64 = std::env::var("ATLAS_DFLASH_CAPTURE_LAYER_OFFSET")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        if offset != 0 {
            tracing::warn!(
                "ATLAS_DFLASH_CAPTURE_LAYER_OFFSET={offset}: capture layers are now \
                 MISALIGNED against the drafter's training data, which captured \
                 output-of-layer-N for the raw target_layer_ids. This degrades draft \
                 acceptance silently — it will not error. Only 0 is correct; use a \
                 nonzero value for A/B testing that claim and nothing else."
            );
        }
        config.dflash_capture_layers = sub
            .target_layer_ids
            .iter()
            .map(|&id| (id as i64 + offset).max(0) as usize)
            .collect();
        tracing::info!(
            "DFlash: target layer capture indices = {:?} (offset={offset} from raw {:?})",
            config.dflash_capture_layers,
            sub.target_layer_ids,
        );
    }

    // ── Step 2: Load weights (model-agnostic from here) ──
    let attn_layer_dtypes: Vec<KvCacheDtype> = if layer_dtypes.is_empty() {
        vec![kv_dtype; config.num_attention_layers()]
    } else {
        layer_dtypes.clone()
    };

    // Populate per-layer KV dims for heterogeneous-attention models (Gemma-4).
    // Homogeneous models return an empty Vec which the KV cache treats as
    // "use global num_kv_heads/head_dim for all layers" (backward compatible).
    config.kv_layer_dims = loader.kv_layer_dims(&config);

    let mut layers = loader.load_layers(store, &config, gpu.as_ref(), &attn_layer_dtypes)?;
    let embed = loader.load_embedding(store, &config)?;
    let final_norm = loader.load_final_norm(store, &config, gpu.as_ref())?;
    let qwen4_final_mixer = loader.load_qwen4_final_mixer(store, &config, gpu.as_ref())?;
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    let qwen4_ple =
        crate::layers::Qwen4PleLayer::load(store, &config, gpu.as_ref(), max_batch_size)?;
    if config.is_qwen4_exp() {
        crate::weight_loader::transform_cache::finish();
    }
    let lm_head = loader.load_lm_head(store, &config, gpu.as_ref())?;
    let mtp_weights = loader.load_mtp_weights_multi(store, &config, gpu.as_ref())?;
    // Probe dense MTP path for non-MoE models (Qwen3.5/3.6 27B family,
    // AEON-7 re-quants). Loader returns None for MoE models so this is a
    // no-op there. The full DenseMtpHead layer is not yet wired — for now
    // we just log presence so the user sees the loader works.
    let mtp_dense_weights = loader.load_mtp_dense_weights(store, &config, gpu.as_ref())?;
    let vision_encoder = loader.load_vision_encoder(store, &config, gpu.as_ref())?;

    // If the checkpoint's `quantization_config.ignore_modules` lists MTP
    // (e.g. Sehyo/Qwen3.5-35B-A3B-NVFP4 ignores `mtp.*`), the MTP weights
    // were stored as BF16 on disk. Runtime-quantizing them to NVFP4
    // anyway — which is what `mtp_quant` would otherwise do — produces
    // garbage drafts (vllm PR #38832). Force BF16 in that case.
    let effective_mtp_quant = if !mtp_weights.is_empty() {
        let quant_fmt = crate::quant_format::detect_quant_format(&config, store);
        if quant_fmt.is_ignored("mtp.fc.weight")
            || quant_fmt.is_ignored("mtp.layers.0.self_attn.q_proj.weight")
        {
            if mtp_quant != MtpQuantization::Bf16 {
                tracing::info!(
                    "MTP head listed in checkpoint ignore_modules — overriding \
                     --mtp-quantization {:?} → Bf16 to preserve precision",
                    mtp_quant,
                );
            }
            MtpQuantization::Bf16
        } else {
            mtp_quant
        }
    } else {
        mtp_quant
    };

    // ── Step 3: Quantize LM head to NVFP4 for fast decode ──
    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();
    let lm_head_nvfp4 = if config.skip_lm_head_quantization() {
        tracing::info!("LM head kept as BF16 (skip NVFP4 quantization per model config)");
        None
    } else {
        let q = quantize_to_nvfp4(
            &lm_head,
            config.vocab_size,
            config.hidden_size,
            gpu.as_ref(),
            absmax_k,
            quantize_k,
            stream,
        )?;
        tracing::info!("LM head quantized to NVFP4 (vocab={})", config.vocab_size);
        Some(q)
    };

    // ── Step 3b: Post-load MoE prefill transpose (MiniMax EP=2 TTFT fix) ──
    //
    // MiniMax M2.7-NVFP4 EP=2 has ~46 GB free at layer-0 load time but
    // ~65 GB free here (the BF16 lm_head just freed ~22 GB during NVFP4
    // quantization). The transpose costs ~59 GB — fits in the post-load
    // window but not the pre-load one. Other loaders (qwen35, qwen3,
    // gemma4) still call `transpose_for_prefill` inline during layer
    // construction; this default-no-op hook doesn't perturb them.
    maybe_run_minimax_m2_moe_transpose(&config, gpu.as_ref(), &mut layers)?;
    // ── Step 4: Create buffer arena ──
    let buffers = BufferArena::new(
        &config,
        max_batch_tokens,
        max_seq_len,
        kv_block_size,
        gpu.as_ref(),
    )?;

    // ── Step 5: Size KV cache from actual free memory ──
    // MLA absorbed: cache compressed latent [kv_lora + rope] instead of expanded [nkv * hd]
    // This gives 12.8x smaller KV cache AND better precision (no expand→cache→read roundtrip)
    let (kv_num_heads, kv_head_dim) = if config.kv_lora_rank > 0 {
        let mla_cache_dim = config.kv_lora_rank + config.qk_rope_head_dim;
        tracing::info!(
            "MLA absorbed KV cache: 1 head × {} dims ({}+{}) per token (vs {} heads × {})",
            mla_cache_dim,
            config.kv_lora_rank,
            config.qk_rope_head_dim,
            config.num_key_value_heads,
            config.head_dim,
        );
        (1, mla_cache_dim)
    } else {
        (config.num_key_value_heads, config.head_dim)
    };
    let kv_config = KvCacheConfig {
        block_size: kv_block_size,
        num_kv_heads: kv_num_heads,
        head_dim: kv_head_dim,
        num_layers: config.num_attention_layers(),
        dtype: kv_dtype,
        layer_dtypes: layer_dtypes.clone(),
        layer_dims: config.kv_layer_dims.clone(),
        cache_blocks_per_seq: hss_cache_blocks_per_seq,
    };

    // Phase 6.2.c — KV-dtype gating for `--high-speed-swap`.
    //
    // All quantization variants are now supported via host-side dequant before
    // disk-write (the orchestrator's tiled-attention kernel reads BF16):
    //   - BF16    : direct stream; predictor anchor (K_lr) computed natively.
    //   - FP8     : E4M3 → BF16 (per-tensor calibration scale). Predictor
    //               degrades to LRU (BF16-only kernel can't read FP8 layout).
    //   - NVFP4   : E2M1 nibble + per-group FP8 scale → BF16. Predictor LRU.
    //   - Turbo4  : Lloyd-Max 16-level + per-group FP8 scale + WHT(K/V) on
    //               disk. Decode flow's WHT(Q)/iWHT(out) bookends handle the
    //               Walsh-Hadamard round-trip transparently. Predictor LRU.
    //   - Turbo3  : 3-bit packed (8 vals per 3 bytes), 8-level codebook,
    //               per-group FP8 scales, WHT bookended. Predictor LRU.
    //   - Turbo8  : FP8 E4M3 + per-group FP8 scales + WHT bookended.
    //               Predictor LRU.
    fn dtype_label(dt: KvCacheDtype) -> &'static str {
        match dt {
            KvCacheDtype::Bf16
            | KvCacheDtype::Bf16KTurbo4V
            | KvCacheDtype::Bf16KTurbo3V
            | KvCacheDtype::Bf16KTurbo2V => "BF16",
            KvCacheDtype::Fp8
            | KvCacheDtype::Fp8KTurbo4V
            | KvCacheDtype::Fp8KTurbo3V
            | KvCacheDtype::Fp8KTurbo2V => "FP8",
            KvCacheDtype::Nvfp4 => "NVFP4",
            KvCacheDtype::Turbo3 | KvCacheDtype::Turbo3KTurbo8V | KvCacheDtype::Turbo2 => "Turbo3",
            KvCacheDtype::Turbo4 | KvCacheDtype::Turbo4KTurbo3V | KvCacheDtype::Turbo4KTurbo8V => {
                "Turbo4"
            }
            KvCacheDtype::Turbo8 => "Turbo8",
        }
    }
    if hss_cache_blocks_per_seq.is_some() {
        let mut counts: std::collections::BTreeMap<&'static str, usize> =
            std::collections::BTreeMap::new();
        if kv_config.layer_dtypes.is_empty() {
            *counts.entry(dtype_label(kv_config.dtype)).or_default() += kv_config.num_layers;
        } else {
            for dt in &kv_config.layer_dtypes {
                *counts.entry(dtype_label(*dt)).or_default() += 1;
            }
        }
        let total: usize = counts.values().sum();
        let summary: Vec<String> = counts
            .iter()
            .map(|(name, n)| format!("{n} {name}"))
            .collect();
        tracing::info!(
            "--high-speed-swap KV: {} attn layers ({}); HBM-shrink applies to all \
             (Phase 6.2.c proper — host dequant for FP8/NVFP4/Turbo3/Turbo4/Turbo8; \
             predictor scoring uses LRU for non-BF16 layers)",
            total,
            summary.join(" + ")
        );
    }
    let actual_free = gpu.free_memory()?;
    let allocatable = actual_free.saturating_sub(inference_reserve);
    let kv_budget = (allocatable as f64 * gpu_memory_utilization) as usize;
    // Phase 6.1.f: when HBM-shrink is active, size the production cache to
    // `max_batch_size × cache_blocks_per_seq` rather than the unbounded
    // budget-driven sum. This is the *whole point* of the HBM-shrink
    // feature — the production cache becomes write staging only; older
    // blocks live on disk under the orchestrator's control.
    let num_kv_blocks = match hss_cache_blocks_per_seq {
        Some(cap) => {
            // Phase 6.3: pool = max_batch × cap + 1 dummy + 1 spare per seq.
            // Reasons:
            //   * +1 dummy: the dummy_kv_block (allocated once at model init,
            //     used for OOB-safe paged-kernel reads) permanently consumes
            //     one slot.
            //   * +1 spare per seq: the slide-then-alloc round-trip in
            //     `ensure_blocks_through_decode` needs the just-freed block
            //     back from the LIFO free list. With exactly cap blocks, the
            //     last grow-to-cap step has zero free blocks, and the next
            //     step's alloc fires before the slide can free one (the loop
            //     orders slide-before-alloc, but alloc-without-slide hits the
            //     final block first).
            //
            // The +1-per-seq covers the gap between bt_len=cap-1 (last
            // pre-slide alloc) and bt_len=cap (slide-then-alloc steady state).
            let n = max_batch_size * (cap as usize + 1) + 1;
            tracing::info!(
                "--high-speed-swap: HBM cache sized to {n} blocks ({} batch × ({cap}+1 spare) + 1 dummy); \
                 older blocks stream from disk via the orchestrator",
                max_batch_size
            );
            n
        }
        None => {
            let budget_blocks = PagedKvCache::compute_num_blocks(&kv_config, kv_budget)?;

            // Cap the pool at what the configured sequences can actually
            // address. Without this, KV is sized purely from `free_memory()`
            // and the server grabs a cache far larger than any request can
            // reach: on Qwen3.8-27B at --max-seq-len 8192 --max-batch-size 4
            // the budget path produced 28079 blocks = 449,264 tokens (22.3 GB)
            // when only 4 x 8192 = 32,768 tokens are addressable — 13.7x waste.
            //
            // On a discrete GPU that is merely wasteful. On GB10 the memory is
            // UNIFIED, so `free_memory()` reports most of host RAM and the
            // oversized cache drives the *host* into a global OOM that takes
            // the machine down (observed 2026-08-14). The HBM-shrink branch
            // above already caps this way; the budget path simply never did.
            //
            // Same shape as the hss cap: +1 spare block per sequence for the
            // slide-then-alloc round trip, +1 dummy block for OOB-safe reads.
            let blocks_per_seq = max_seq_len.div_ceil(kv_block_size);
            let reachable = max_batch_size
                .saturating_mul(blocks_per_seq.saturating_add(1))
                .saturating_add(1);
            let n = budget_blocks.min(reachable);
            if n < budget_blocks {
                tracing::info!(
                    "KV cache capped to addressable size: {} blocks ({} batch × \
                     ({} blocks/seq + 1 spare) + 1 dummy) instead of the \
                     budget-derived {} blocks — saves {:.1} GB",
                    n,
                    max_batch_size,
                    blocks_per_seq,
                    budget_blocks,
                    // bytes/block derived from the budget that produced
                    // `budget_blocks`, so this needs no extra KvCacheConfig API.
                    (budget_blocks - n) as f64 * (kv_budget as f64 / budget_blocks.max(1) as f64)
                        / (1024.0 * 1024.0 * 1024.0),
                );
            }
            let max_kv_tokens = n * kv_block_size;
            tracing::info!(
                "KV cache (post-construction): {:.1} GB free, {:.1} GB allocatable, \
                 {} blocks × {} tok/block = {} max tokens",
                actual_free as f64 / (1024.0 * 1024.0 * 1024.0),
                allocatable as f64 / (1024.0 * 1024.0 * 1024.0),
                n,
                kv_block_size,
                max_kv_tokens,
            );
            n
        }
    };
    let _max_kv_tokens = num_kv_blocks * kv_block_size;
    // Phase 6.1.f / 6.2.c — when --high-speed-swap is on with HBM-shrink, the
    // production KV cache only has to fit the per-seq HBM window, not the full
    // sequence (older blocks live on disk). Compare against `cache_blocks_per_seq`
    // in that mode; the legacy "blocks per max_seq_len" check is invalid for
    // HBM-shrunk pools by design.
    let blocks_per_seq = match hss_cache_blocks_per_seq {
        Some(cap) => cap as usize,
        None => max_seq_len.div_ceil(kv_block_size),
    };
    let max_concurrent = num_kv_blocks / blocks_per_seq.max(1);
    if max_concurrent < max_batch_size {
        // Suggest a max_seq_len that lets the requested batch size fit.
        let suggested_max_seq_len = (num_kv_blocks / max_batch_size.max(1)) * kv_block_size;
        anyhow::bail!(
            "KV cache can hold at most {} concurrent sequence(s) at --max-seq-len={}, \
             but --max-batch-size={} was requested. \
             KV pool has {} block(s) of {} tokens each; each sequence needs {} block(s). \
             Try --max-seq-len {} (keeps max_batch_size={}) or reduce --max-batch-size.",
            max_concurrent,
            max_seq_len,
            max_batch_size,
            num_kv_blocks,
            kv_block_size,
            blocks_per_seq,
            suggested_max_seq_len.max(kv_block_size),
            max_batch_size,
        );
    }
    let kv_cache = PagedKvCache::new(kv_config, num_kv_blocks, gpu.as_ref())?;

    // ── Step 6: Assemble model ──
    // Capture pointers for any post-construction sharing (DFlash drafter
    // shares embed_tokens + lm_head with the target). DenseWeight is Copy
    // so this clones the device pointer cheaply.
    let target_embed_for_dflash = embed.weight;
    let target_lm_head_for_dflash = lm_head.weight;
    // DFlash trains against the target model's exposed intermediate states.
    // Qwen4 exposes the full four-stream hyperconnection row (4H); ordinary
    // targets have residual_width()==hidden_size, preserving their ABI.
    let target_hidden_for_dflash = config.residual_width();
    // Honor --mtp-vocab for the DFlash drafter lm_head, mirroring the MTP
    // head: drafts only need argmax over the high-frequency vocab prefix,
    // and the full 248k-row lm_head GEMM at M=γ+1 dominates the propose
    // tail (~32ms of 67ms propose at ctx≈390, DFLASH_KP 2026-06-11).
    // mtp_vocab_size=0 means uncapped.
    let target_vocab_for_dflash = if mtp_vocab_size > 0 {
        (mtp_vocab_size as usize).min(config.vocab_size)
    } else {
        config.vocab_size
    };

    let mut model = TransformerModel::new(
        config,
        embed,
        final_norm,
        qwen4_final_mixer,
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        qwen4_ple,
        lm_head,
        lm_head_nvfp4,
        layers,
        buffers,
        kv_cache,
        mtp_weights,
        mtp_dense_weights,
        gpu,
        max_seq_len,
        max_batch_size,
        effective_mtp_quant,
        use_speculative,
        prefix_cache,
        mtp_vocab_size,
        comm,
        self_speculative,
        num_drafts,
        vision_encoder,
        ssm_cache_slots,
        ssm_checkpoint_interval,
    )?;

    // ── Step 7: DFlash drafter (optional, post-construction) ──
    //
    // Loaded last because it depends on the target's `embed_tokens` and
    // `lm_head` pointers (the drafter checkpoint omits these — they're
    // shared at runtime, mirroring vLLM PR #40898's `skip_substrs` flow).
    if let Some(args) = dflash_args {
        let weights = load_dflash_weights(
            args.drafter_store,
            &args.drafter_config,
            model.gpu_backend(),
            1, // tp_size for the drafter side: replicated, so always 1
        )?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "DFlash was explicitly requested, but the drafter checkpoint contains no \
                 DFlash weight schema (`fc.weight` or `model.fc.weight`)"
            )
        })?;
        let mut head = crate::layers::BlockDiffusionDraftHead::from_weights(
            weights,
            target_embed_for_dflash,
            target_lm_head_for_dflash,
            target_hidden_for_dflash,
            target_vocab_for_dflash,
            args.gamma,
            model.ddtree_parent_ids_capacity,
            args.dspark_verify_mode,
            args.window_size,
            model.gpu_backend(),
            max_seq_len,
            args.quantization,
        )?;
        // Share the target's NVFP4-T lm_head (ATLAS_LM_HEAD_T) with the
        // drafter's propose lm_head fast path (gated at the call site by
        // ATLAS_DFLASH_LM_HEAD_NVFP4=1). Same device allocation — the
        // drafter reads the --mtp-vocab column prefix via ldb.
        if let Some((t, ldb)) = model.dflash_lm_head_t() {
            head.lm_head_shared_t = Some(t);
            head.lm_head_shared_t_ldb = ldb;
        }
        model.set_dflash_proposer(std::sync::Arc::new(head))?;
        tracing::info!("DFlash drafter installed as the active proposer");
    }

    Ok(Box::new(model))
}
