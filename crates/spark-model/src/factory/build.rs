// SPDX-License-Identifier: AGPL-3.0-only

//! `build_model` — entry point that wires up the configured loader,
//! buffers, KV cache, and (optional) DFlash drafter into a `TransformerModel`.

use std::sync::Arc;

use anyhow::{Context, Result};
use atlas_core::config::{DflashCaptureMode, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use spark_runtime::prefix_cache::PrefixCache;
use spark_runtime::weights::WeightDtype;
use spark_runtime::weights::WeightStore;

use super::DflashBuildArgs;
use super::loader_for_config;
use super::m2_setup::maybe_run_minimax_m2_moe_transpose;
use super::qwen4_stream_t::maybe_setup_qwen4_stream_t;
use crate::layers::MtpQuantization;
use crate::model::TransformerModel;
use crate::traits::Model;
use crate::weight_loader::load_dflash_weights;
use crate::weight_map::quantize_to_nvfp4;

const QWEN38_FLASH_NEXT_HIDDEN: usize = 2560;
const QWEN38_FLASH_NEXT_VOCAB: usize = 248_320;
const QWEN38_FLASH_NEXT_TARGET_LAYERS: usize = 48;
const QWEN38_FLASH_NEXT_DRAFT_LAYERS: usize = 6;
const QWEN38_FLASH_NEXT_DRAFT_BLOCK_SIZE: usize = 16;
const QWEN38_FLASH_NEXT_DRAFT_GAMMA: usize = QWEN38_FLASH_NEXT_DRAFT_BLOCK_SIZE - 1;
const QWEN38_FLASH_NEXT_CAPTURE_LAYERS: [usize; 8] = [1, 7, 13, 20, 26, 33, 39, 46];
const QWEN38_FLASH_NEXT_SELECTOR_VOCAB: usize = 248_077;
const QWEN38_FLASH_NEXT_DFLASH2_LAYER_TYPES: [&str; QWEN38_FLASH_NEXT_DRAFT_LAYERS] = [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen4DflashPairing {
    Native,
    DenseDonorBridge,
}

fn is_qwen38_flash_next_target_vocab(vocab_size: usize) -> bool {
    matches!(
        vocab_size,
        QWEN38_FLASH_NEXT_VOCAB | QWEN38_FLASH_NEXT_SELECTOR_VOCAB
    )
}

fn is_native_qwen38_flash_next_drafter(
    target: &ModelConfig,
    drafter: &crate::weight_loader::DflashConfig,
) -> bool {
    let Some(sub) = drafter.dflash_config.as_ref() else {
        return false;
    };
    let common = target.is_qwen4_exp()
        && target.hidden_size == QWEN38_FLASH_NEXT_HIDDEN
        && target.residual_width() == 4 * QWEN38_FLASH_NEXT_HIDDEN
        // The target store is loaded and preflighted at physical V248320,
        // then server admission narrows ModelConfig to the tokenizer's
        // canonical logical V248077 before factory construction. Both are
        // exact identities for this target; no other cap is native.
        && is_qwen38_flash_next_target_vocab(target.vocab_size)
        && target.num_hidden_layers == QWEN38_FLASH_NEXT_TARGET_LAYERS
        && drafter.hidden_size == QWEN38_FLASH_NEXT_HIDDEN
        && drafter.vocab_size == QWEN38_FLASH_NEXT_VOCAB
        && drafter.num_hidden_layers == QWEN38_FLASH_NEXT_DRAFT_LAYERS
        && drafter.num_target_layers == QWEN38_FLASH_NEXT_TARGET_LAYERS
        && drafter.intermediate_size == 8704
        && drafter.num_attention_heads == 20
        && drafter.num_key_value_heads == 4
        && drafter.head_dim == 128
        && drafter.model_type.as_deref() == Some("qwen3")
        && drafter.root_block_size_explicit
        && drafter.block_size == QWEN38_FLASH_NEXT_DRAFT_BLOCK_SIZE
        && drafter.resolved_block_size() == QWEN38_FLASH_NEXT_DRAFT_BLOCK_SIZE
        && drafter.is_causal == Some(false)
        && !drafter.tie_word_embeddings
        && drafter.draft_vocab_size.is_none()
        && drafter.markov_rank == 0
        && drafter
            .confidence_head_config()
            .is_ok_and(|confidence| confidence.is_none())
        && sub.block_size == Some(QWEN38_FLASH_NEXT_DRAFT_BLOCK_SIZE)
        && sub.mask_token_id == 248_077
        && sub.target_layer_ids == QWEN38_FLASH_NEXT_CAPTURE_LAYERS
        && sub.projector_type.is_none()
        && !sub.fc_layernorm;
    if !common {
        return false;
    }
    match drafter.architectures.as_slice() {
        [architecture] if architecture == "DFlashDraftModel" => {
            sub.conv_kernel_size == 0
                && sub.conv_group_size == 0
                && sub.selector_rank == 0
                && sub.selector_top_k == 0
                && sub.selector_vocab_size.is_none()
        }
        [architecture] if architecture == "DFlash2DraftModel" => {
            sub.conv_kernel_size == 2
                && sub.conv_group_size == 16
                && sub.selector_rank == 256
                && sub.selector_top_k == 16
                && sub.selector_vocab_size == Some(QWEN38_FLASH_NEXT_SELECTOR_VOCAB)
                && drafter.sliding_window == Some(4_096)
                && drafter.layer_types.as_ref().is_some_and(|layer_types| {
                    layer_types
                        .iter()
                        .map(String::as_str)
                        .eq(QWEN38_FLASH_NEXT_DFLASH2_LAYER_TYPES)
                })
        }
        _ => false,
    }
}

fn resolve_native_dflash2_proposal_vocab(
    drafter: &crate::weight_loader::DflashConfig,
    requested_vocab: u32,
) -> Result<usize> {
    anyhow::ensure!(
        drafter.is_dflash2(),
        "native Flash-Next DFlash2 proposal-vocabulary resolution requires DFlash2 semantics"
    );
    let selector_vocab = drafter
        .dflash_config
        .as_ref()
        .and_then(|sub| sub.selector_vocab_size)
        .context("native Flash-Next DFlash2 requires selector_vocab_size")?;
    anyhow::ensure!(
        selector_vocab == QWEN38_FLASH_NEXT_SELECTOR_VOCAB,
        "native Flash-Next DFlash2 selector_vocab_size={selector_vocab}; expected {QWEN38_FLASH_NEXT_SELECTOR_VOCAB}"
    );
    anyhow::ensure!(
        requested_vocab == 0 || requested_vocab as usize == selector_vocab,
        "native Flash-Next DFlash2 proposal vocabulary must be uncapped or exactly {selector_vocab}; got {requested_vocab}"
    );
    Ok(selector_vocab)
}

fn classify_qwen4_dflash_pairing(
    native_geometry: bool,
    has_donor: bool,
) -> Result<Qwen4DflashPairing> {
    match (native_geometry, has_donor) {
        (true, false) => Ok(Qwen4DflashPairing::Native),
        (false, true) => Ok(Qwen4DflashPairing::DenseDonorBridge),
        (true, true) => anyhow::bail!(
            "native Qwen3.8-Flash-Next DFlash must share the target embedding/lm_head; remove --dflash-donor-model"
        ),
        (false, false) => anyhow::bail!(
            "Qwen3.8-Flash-Next requires either the exact native H2560/V248320/T48 DFlash checkpoint or --dflash-donor-model for an explicit dense-DFlash bridge"
        ),
    }
}

fn validate_native_qwen4_dflash_width(
    drafter: &crate::weight_loader::DflashConfig,
    requested_gamma: Option<usize>,
) -> Result<()> {
    let trained_block = drafter.resolved_block_size();
    anyhow::ensure!(
        trained_block == QWEN38_FLASH_NEXT_DRAFT_BLOCK_SIZE,
        "native Qwen3.8-Flash-Next V3 requires the exact B{QWEN38_FLASH_NEXT_DRAFT_BLOCK_SIZE}/gamma{QWEN38_FLASH_NEXT_DRAFT_GAMMA} recipe, but checkpoint block_size={trained_block}"
    );
    let gamma = requested_gamma.unwrap_or(QWEN38_FLASH_NEXT_DRAFT_GAMMA);
    anyhow::ensure!(
        gamma == QWEN38_FLASH_NEXT_DRAFT_GAMMA,
        "native Qwen3.8-Flash-Next V3 uses bidirectional B{QWEN38_FLASH_NEXT_DRAFT_BLOCK_SIZE} attention and requires gamma={QWEN38_FLASH_NEXT_DRAFT_GAMMA}, but gamma={gamma} was requested"
    );
    Ok(())
}

fn remap_capture_depths(
    ids: &[usize],
    source_layers: usize,
    target_layers: usize,
) -> Result<Vec<usize>> {
    anyhow::ensure!(
        source_layers > 1 && target_layers > 1,
        "invalid DFlash capture-depth bridge {source_layers}->{target_layers}"
    );
    let source_last = source_layers - 1;
    let target_last = target_layers - 1;
    ids.iter()
        .map(|&id| {
            anyhow::ensure!(
                id < source_layers,
                "DFlash capture layer {id} is outside donor depth {source_layers}"
            );
            Ok((id * target_last + source_last / 2) / source_last)
        })
        .collect()
}

fn donor_tensor<'a>(
    store: &'a WeightStore,
    candidates: &[&str],
    expected_shape: &[usize],
    role: &str,
) -> Result<&'a spark_runtime::weights::WeightTensor> {
    let matches: Vec<_> = candidates
        .iter()
        .filter_map(|name| store.get(name).ok().map(|tensor| (*name, tensor)))
        .collect();
    anyhow::ensure!(
        matches.len() == 1,
        "DFlash donor must contain exactly one {role}; found {:?}",
        matches.iter().map(|(name, _)| *name).collect::<Vec<_>>()
    );
    let (name, tensor) = matches[0];
    anyhow::ensure!(
        tensor.dtype == WeightDtype::BF16,
        "DFlash donor tensor {name} must be BF16, got {:?}",
        tensor.dtype
    );
    anyhow::ensure!(
        tensor.shape == expected_shape,
        "DFlash donor tensor {name} has shape {:?}; expected {:?}",
        tensor.shape,
        expected_shape
    );
    Ok(tensor)
}

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
    // DeepSeek-V4.1 (forward-ported from dsv41/integration): standalone model over the CED +
    // SWA-replay forward (`model::dsv41`). Dispatched FIRST: none of the decoder-only
    // construction below describes it.
    #[cfg(feature = "cuda")]
    if config.model_type == "deepseek_v41" {
        if max_batch_size > 1 {
            tracing::warn!(
                "DeepSeek-V4.1 serves ONE live sequence (model-wide compressed-KV caches); \
                 --max-batch-size {max_batch_size} is ignored"
            );
        }
        let model_dir = crate::weight_loader::deepseek_v41::resolve_model_dir()?;
        let max_chunk = std::env::var("ATLAS_DSV41_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
        let model = crate::model::dsv41::Dsv41Model::new(&config, store, gpu, &model_dir, max_seq_len, max_chunk)?;
        return Ok(Box::new(model));
    }
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
        let raw_with_offset: Vec<_> = sub
            .target_layer_ids
            .iter()
            .map(|&id| (id as i64 + offset).max(0) as usize)
            .collect();
        if config.is_qwen4_exp() {
            let pairing = classify_qwen4_dflash_pairing(
                is_native_qwen38_flash_next_drafter(&config, &args.drafter_config),
                args.donor_store.is_some(),
            )?;
            config.dflash_capture_width = args.drafter_config.hidden_size;
            config.dflash_capture_offset = 0;
            match pairing {
                Qwen4DflashPairing::Native => {
                    validate_native_qwen4_dflash_width(&args.drafter_config, args.gamma)?;
                    anyhow::ensure!(
                        std::env::var("ATLAS_EXPERIMENTAL_NATIVE_QWEN4_DFLASH")
                            .ok()
                            .as_deref()
                            == Some("1"),
                        "native Qwen4 DFlash support is unqualified and disabled by default; set ATLAS_EXPERIMENTAL_NATIVE_QWEN4_DFLASH=1 only for correctness research"
                    );
                    anyhow::ensure!(
                        offset == 0,
                        "native Qwen3.8-Flash-Next DFlash requires exact unshifted capture layers"
                    );
                    config.dflash_capture_layers = raw_with_offset;
                    config.dflash_capture_mode = DflashCaptureMode::Qwen4HyperProjected;
                    tracing::warn!(
                        "EXPERIMENTAL unqualified native Qwen3.8-Flash-Next DFlash: exact capture layers {:?}; projecting each Qwen4 4H residual through the terminal hyperconnection mixer to H={}. Do not use as a performance candidate until acceptance qualification passes.",
                        config.dflash_capture_layers,
                        config.dflash_capture_width,
                    );
                }
                Qwen4DflashPairing::DenseDonorBridge => {
                    let source_layers = args.drafter_config.num_target_layers;
                    anyhow::ensure!(
                        source_layers > 1,
                        "Qwen4 DFlash bridge requires drafter config num_target_layers"
                    );
                    config.dflash_capture_layers = remap_capture_depths(
                        &raw_with_offset,
                        source_layers,
                        config.num_hidden_layers,
                    )?;
                    config.dflash_capture_mode = DflashCaptureMode::ResidualSlice;
                    anyhow::ensure!(
                        config.dflash_capture_width <= config.residual_width(),
                        "dense DFlash bridge width {} exceeds Flash-Next residual width {}",
                        config.dflash_capture_width,
                        config.residual_width(),
                    );
                    tracing::warn!(
                        "EXPERIMENTAL Qwen4->dense-DFlash bridge: capture depths {:?}->{:?}; exposing BF16 residual slice [{}..{}) of {}. Target verification remains authoritative; qualification is required before production use.",
                        raw_with_offset,
                        config.dflash_capture_layers,
                        config.dflash_capture_offset,
                        config.dflash_capture_offset + config.dflash_capture_width,
                        config.residual_width(),
                    );
                }
            }
        } else {
            anyhow::ensure!(
                args.donor_store.is_none(),
                "--dflash-donor-model is only valid for an explicit Qwen3.8-Flash-Next bridge"
            );
            config.dflash_capture_layers = raw_with_offset;
        }
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

    crate::weight_loader::transform_cache::configure_construction_mode(use_speculative)?;
    let mut layers = loader.load_layers(store, &config, gpu.as_ref(), &attn_layer_dtypes)?;
    let embed = loader.load_embedding(store, &config, gpu.as_ref())?;
    let final_norm = loader.load_final_norm(store, &config, gpu.as_ref())?;
    let qwen4_final_mixer = loader.load_qwen4_final_mixer(store, &config, gpu.as_ref())?;
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    let qwen4_ple =
        crate::layers::Qwen4PleLayer::load(store, &config, gpu.as_ref(), max_batch_size)?;
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

    let qwen4_mtp_layout = if use_speculative && config.is_qwen4_exp() {
        crate::weight_loader::qwen4_mtp::classify_qwen4_mtp_store(store, &config)?
    } else {
        None
    };
    let qwen4_mtp_proposer: Option<Arc<dyn crate::speculative::DraftProposer>> =
        if let Some(mtp_layout) = qwen4_mtp_layout {
            tracing::info!(
                layout = ?mtp_layout,
                "Qwen4 native MTP schema admitted for direct proposer construction"
            );
            let mtp_config = crate::weight_loader::qwen4_mtp_config(&config);
            let mtp_layer = crate::weight_loader::load_qwen4_mtp_layer(
                store,
                &config,
                gpu.as_ref(),
                KvCacheDtype::Bf16,
            )?;
            let mtp_final_mixer = loader
                .load_qwen4_final_mixer(store, &mtp_config, gpu.as_ref())?
                .context("Qwen4 MTP checkpoint is missing its final hyperconnection mixer")?;
            let shared_lm_head =
                lm_head_nvfp4.context("Qwen4 native MTP requires the target NVFP4 LM head")?;
            Some(Arc::new(crate::layers::Qwen4MtpHead::new(
                store,
                &config,
                mtp_layer,
                mtp_final_mixer,
                embed,
                shared_lm_head,
                gpu.as_ref(),
                mtp_vocab_size,
                max_seq_len,
                max_batch_size,
            )?))
        } else {
            None
        };
    if config.is_qwen4_exp() {
        // Qwen4's native MTP layer, final mixer, and packed BF16 expert bank
        // are constructed after the target layers. Publish only once every
        // selected transform has succeeded; an error leaves no cache index.
        crate::weight_loader::transform_cache::finish();
    }

    // ── Step 3b: Post-load MoE prefill transpose (MiniMax EP=2 TTFT fix) ──
    //
    // MiniMax M2.7-NVFP4 EP=2 has ~46 GB free at layer-0 load time but
    // ~65 GB free here (the BF16 lm_head just freed ~22 GB during NVFP4
    // quantization). The transpose costs ~59 GB — fits in the post-load
    // window but not the pre-load one. Other loaders (qwen35, qwen3,
    // gemma4) still call `transpose_for_prefill` inline during layer
    // construction; this default-no-op hook doesn't perturb them.
    maybe_run_minimax_m2_moe_transpose(&config, gpu.as_ref(), &mut layers)?;
    maybe_setup_qwen4_stream_t(&config, gpu.as_ref(), &mut layers)?;
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
    let (target_embed_for_dflash, target_lm_head_for_dflash) = if let Some(args) =
        dflash_args.as_ref()
        && let Some(donor) = args.donor_store
    {
        let hidden = args.drafter_config.hidden_size;
        let vocab = args.drafter_config.vocab_size;
        let embed = donor_tensor(
            donor,
            &[
                "model.language_model.embed_tokens.weight",
                "model.embed_tokens.weight",
                "embed_tokens.weight",
            ],
            &[vocab, hidden],
            "embedding",
        )?;
        let lm_head = donor_tensor(
            donor,
            &["lm_head.weight", "model.lm_head.weight"],
            &[vocab, hidden],
            "lm_head",
        )?;
        (embed.ptr, lm_head.ptr)
    } else {
        (embed.weight, lm_head.weight)
    };
    // DFlash trains against the target model's exposed intermediate states.
    // Qwen4 exposes the full four-stream hyperconnection row (4H); ordinary
    // targets have residual_width()==hidden_size, preserving their ABI.
    let target_hidden_for_dflash = if config.dflash_capture_width > 0 {
        config.dflash_capture_width
    } else {
        config.residual_width()
    };
    // Honor --mtp-vocab for the DFlash drafter lm_head, mirroring the MTP
    // head: drafts only need argmax over the high-frequency vocab prefix,
    // and the full 248k-row lm_head GEMM at M=γ+1 dominates the propose
    // tail (~32ms of 67ms propose at ctx≈390, DFLASH_KP 2026-06-11).
    // mtp_vocab_size=0 means uncapped.
    let native_dflash2_selector_vocab = dflash_args
        .as_ref()
        .filter(|args| {
            is_native_qwen38_flash_next_drafter(&config, &args.drafter_config)
                && args.drafter_config.is_dflash2()
        })
        .map(|args| resolve_native_dflash2_proposal_vocab(&args.drafter_config, mtp_vocab_size))
        .transpose()?;
    let target_vocab_for_dflash = if let Some(selector_vocab) = native_dflash2_selector_vocab {
        selector_vocab
    } else if mtp_vocab_size > 0 {
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
        qwen4_mtp_proposer,
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
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    model.initialize_qwen4_ple_prefill(max_batch_tokens)?;

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
        if args.donor_store.is_none()
            && let Some((t, ldb)) = model.dflash_lm_head_t()
        {
            head.lm_head_shared_t = Some(t);
            head.lm_head_shared_t_ldb = ldb;
        }
        model.set_dflash_proposer(std::sync::Arc::new(head))?;
        tracing::info!("DFlash drafter installed as the active proposer");
    }

    Ok(Box::new(model))
}

#[cfg(test)]
mod tests {
    use super::{
        Qwen4DflashPairing, classify_qwen4_dflash_pairing, is_native_qwen38_flash_next_drafter,
        is_qwen38_flash_next_target_vocab, remap_capture_depths,
        resolve_native_dflash2_proposal_vocab, validate_native_qwen4_dflash_width,
    };
    use atlas_core::config::ModelConfig;

    fn flash_next_target() -> ModelConfig {
        let mut target = ModelConfig::qwen3_next_80b_nvfp4();
        target.model_type = "qwen4_exp".to_string();
        target.hidden_size = 2560;
        target.hc_count = 4;
        target.vocab_size = 248_320;
        target.num_hidden_layers = 48;
        target
    }

    fn native_v3_json(block_size: usize) -> String {
        format!(
            r#"{{
                "architectures":["DFlashDraftModel"],
                "model_type":"qwen3",
                "hidden_size":2560,
                "num_hidden_layers":6,
                "num_target_layers":48,
                "intermediate_size":8704,
                "num_attention_heads":20,
                "num_key_value_heads":4,
                "head_dim":128,
                "vocab_size":248320,
                "block_size":{block_size},
                "tie_word_embeddings":false,
                "is_causal":false,
                "dflash_config":{{
                    "block_size":{block_size},
                    "mask_token_id":248077,
                    "target_layer_ids":[1,7,13,20,26,33,39,46]
                }}
            }}"#
        )
    }

    fn native_dflash2_json() -> &'static str {
        r#"{
            "architectures":["DFlash2DraftModel"],
            "model_type":"qwen3",
            "hidden_size":2560,
            "num_hidden_layers":6,
            "num_target_layers":48,
            "intermediate_size":8704,
            "num_attention_heads":20,
            "num_key_value_heads":4,
            "head_dim":128,
            "vocab_size":248320,
            "block_size":16,
            "tie_word_embeddings":false,
            "is_causal":false,
            "layer_types":[
                "sliding_attention","sliding_attention","sliding_attention",
                "sliding_attention","sliding_attention","full_attention"
            ],
            "sliding_window":4096,
            "dflash_config":{
                "block_size":16,
                "mask_token_id":248077,
                "target_layer_ids":[1,7,13,20,26,33,39,46],
                "conv_kernel_size":2,
                "conv_group_size":16,
                "selector_rank":256,
                "selector_top_k":16,
                "selector_vocab_size":248077
            }
        }"#
    }

    #[test]
    fn remaps_dense_qwen38_capture_depths_to_flash_next() {
        let mapped =
            remap_capture_depths(&[1, 10, 18, 27, 35, 44, 52, 61], 64, 48).expect("valid bridge");
        assert_eq!(mapped, [1, 7, 13, 20, 26, 33, 39, 46]);
    }

    #[test]
    fn rejects_capture_outside_donor_depth() {
        assert!(remap_capture_depths(&[64], 64, 48).is_err());
    }

    #[test]
    fn native_qwen4_dflash_requires_no_donor() {
        assert_eq!(
            classify_qwen4_dflash_pairing(true, false).unwrap(),
            Qwen4DflashPairing::Native
        );
        assert!(classify_qwen4_dflash_pairing(true, true).is_err());
    }

    #[test]
    fn dense_qwen4_dflash_requires_donor() {
        assert_eq!(
            classify_qwen4_dflash_pairing(false, true).unwrap(),
            Qwen4DflashPairing::DenseDonorBridge
        );
        assert!(classify_qwen4_dflash_pairing(false, false).is_err());
    }

    #[test]
    fn native_qwen38_geometry_is_exact() {
        let target = flash_next_target();
        let mut drafter =
            crate::weight_loader::dflash_loader::parse_dflash_config(&native_v3_json(16))
                .expect("valid native drafter fixture");

        assert!(is_native_qwen38_flash_next_drafter(&target, &drafter));
        assert!(validate_native_qwen4_dflash_width(&drafter, None).is_ok());
        assert!(validate_native_qwen4_dflash_width(&drafter, Some(15)).is_ok());
        assert!(validate_native_qwen4_dflash_width(&drafter, Some(14)).is_err());
        drafter.dflash_config.as_mut().unwrap().target_layer_ids[1] = 8;
        assert!(!is_native_qwen38_flash_next_drafter(&target, &drafter));
    }

    #[test]
    fn native_qwen38_accepts_only_physical_or_canonical_logical_target_vocab() {
        assert!(is_qwen38_flash_next_target_vocab(248_320));
        assert!(is_qwen38_flash_next_target_vocab(248_077));
        assert!(!is_qwen38_flash_next_target_vocab(248_076));
        assert!(!is_qwen38_flash_next_target_vocab(248_319));

        let drafter =
            crate::weight_loader::dflash_loader::parse_dflash_config(native_dflash2_json())
                .expect("valid native DFlash2 fixture");
        let mut target = flash_next_target();
        target.vocab_size = 248_077;
        assert!(is_native_qwen38_flash_next_drafter(&target, &drafter));
        target.vocab_size = 248_076;
        assert!(!is_native_qwen38_flash_next_drafter(&target, &drafter));
    }

    #[test]
    fn native_qwen38_rejects_b32_even_with_matching_gamma() {
        let target = flash_next_target();
        assert!(
            crate::weight_loader::dflash_loader::parse_dflash_config(&native_v3_json(32)).is_err()
        );
        let mut drafter =
            crate::weight_loader::dflash_loader::parse_dflash_config(&native_v3_json(16))
                .expect("valid B16 fixture");
        drafter.block_size = 32;
        drafter.dflash_config.as_mut().unwrap().block_size = Some(32);

        assert!(!is_native_qwen38_flash_next_drafter(&target, &drafter));
        assert!(validate_native_qwen4_dflash_width(&drafter, None).is_err());
        assert!(validate_native_qwen4_dflash_width(&drafter, Some(31)).is_err());
        assert!(validate_native_qwen4_dflash_width(&drafter, Some(15)).is_err());
    }

    #[test]
    fn native_qwen38_dflash2_geometry_and_logical_vocab_are_exact() {
        let target = flash_next_target();
        let mut drafter =
            crate::weight_loader::dflash_loader::parse_dflash_config(native_dflash2_json())
                .expect("valid native DFlash2 fixture");

        assert!(is_native_qwen38_flash_next_drafter(&target, &drafter));
        assert!(validate_native_qwen4_dflash_width(&drafter, None).is_ok());
        assert_eq!(
            resolve_native_dflash2_proposal_vocab(&drafter, 0).unwrap(),
            248_077
        );
        assert_eq!(
            resolve_native_dflash2_proposal_vocab(&drafter, 248_077).unwrap(),
            248_077
        );
        assert!(resolve_native_dflash2_proposal_vocab(&drafter, 100_000).is_err());
        assert!(resolve_native_dflash2_proposal_vocab(&drafter, 248_320).is_err());

        drafter.dflash_config.as_mut().unwrap().selector_vocab_size = Some(248_320);
        assert!(!is_native_qwen38_flash_next_drafter(&target, &drafter));
        assert!(resolve_native_dflash2_proposal_vocab(&drafter, 0).is_err());
    }
}
