// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, ensure};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::{ModelWeightLoader, WeightFormat};
use crate::layer::TransformerLayer;
use crate::layers::{Qwen3AttentionLayer, Qwen3SsmLayer};
use crate::tp_shard::{TpShardKind, load_qkvo_tp, shard_dense_bf16, shard_quantized_nvfp4};
use crate::weight_map::{
    AttentionWeights, DenseWeight, MtpWeights, Nvfp4Variant, QuantizeCtx, SsmWeights, dense,
    dense_auto, detect_nvfp4_variant, gpu_concat_rows, interleave_ba, load_kv_scales,
    load_ssm_qwen35, quantize_to_nvfp4, quantized_any,
};

#[path = "qwen35_dense/ffn.rs"]
mod ffn;

#[cfg(test)]
#[path = "qwen35_dense/w3_integration_tests.rs"]
mod w3_integration_tests;

pub struct Qwen35DenseWeightLoader;

impl ModelWeightLoader for Qwen35DenseWeightLoader {
    fn supports_tp(&self) -> bool {
        // FullAttention layers are TP-sharded (NVFP4-from-disk and BF16
        // → NVFP4 paths). LinearAttention (GDN SSM) layers run
        // full-replica per rank — see qwen35.rs for the rationale.
        true
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let layer_types = if config.layer_types.is_empty() {
            (0..config.num_hidden_layers)
                .map(|i| config.layer_type(i))
                .collect::<Vec<_>>()
        } else {
            config.layer_types.clone()
        };

        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);
        let mut attn_idx = 0usize;

        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();
        let h = config.hidden_size;

        let variant = detect_nvfp4_variant(store, config);
        let weight_format = WeightFormat::detect(store, config);
        let flashinfer_ffn_requested = crate::layers::prefill_ffn_flashinfer_enabled()?;
        let flashinfer_attn_proj_requested = crate::layers::prefill_proj_flashinfer_enabled()?;
        let flashinfer_ssm_proj_requested = crate::layers::prefill_ssm_flashinfer_enabled()?;
        ensure!(
            !flashinfer_ffn_requested || variant == Nvfp4Variant::Standard,
            "ATLAS_PREFILL_FFN_FLASHINFER=1 requires a Standard ModelOpt NVFP4 checkpoint"
        );
        ensure!(
            !flashinfer_attn_proj_requested || variant == Nvfp4Variant::Standard,
            "ATLAS_PREFILL_PROJ_FLASHINFER=1 requires a Standard ModelOpt NVFP4 checkpoint"
        );
        ensure!(
            !flashinfer_ssm_proj_requested || variant == Nvfp4Variant::Standard,
            "ATLAS_PREFILL_SSM_FLASHINFER=1 requires a Standard ModelOpt NVFP4 checkpoint"
        );
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        ensure!(
            !flashinfer_attn_proj_requested,
            "ATLAS_PREFILL_PROJ_FLASHINFER=1 requires a Linux CUDA build"
        );
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        ensure!(
            !flashinfer_ssm_proj_requested,
            "ATLAS_PREFILL_SSM_FLASHINFER=1 requires a Linux CUDA build"
        );
        tracing::info!(
            "Weight format: {:?}, NVFP4 variant: {:?}",
            weight_format,
            variant
        );

        let mut w3_session = ffn::prepare_w3_session(config, gpu)?;
        let ffn_context = ffn::DenseFfnLoadContext::new(
            store,
            config,
            gpu,
            variant,
            absmax_k,
            quantize_k,
            stream,
            flashinfer_ffn_requested,
        );

        // Fast engine recovery: serve the per-layer transposed NVFP4 copies
        // from disk instead of rebuilding them. No-op unless
        // ATLAS_WEIGHT_CACHE=1. `finish` runs only on the success path, so a
        // load that bails leaves the blob unpublished.
        super::transform_cache::init(
            store,
            config,
            gpu,
            &format!("qwen35_dense/{weight_format:?}/{variant:?}"),
        );

        for (i, lt) in layer_types.iter().enumerate() {
            let lp = config.layer_prefix(i);
            let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
            let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;

            let ffn = ffn_context.load(i, &lp, &mut w3_session)?;

            match lt {
                LayerType::FullAttention => {
                    let p = format!("{lp}.self_attn");
                    let tp_rank = config.tp_rank;
                    let tp_size = config.tp_world_size.max(1);
                    let (attn, q_nvfp4, k_nvfp4, v_nvfp4) = match variant {
                        // NVFP4 already packed in the checkpoint
                        // (compressed-tensors `.weight_packed` or
                        // modelopt uint8 `.weight` + `weight_scale_2`).
                        // `quantized_auto` reads either schema; we shard
                        // packed bytes directly. Treating modelopt as
                        // BF16-then-quantize previously crashed at the
                        // first full_attention layer because dense_auto
                        // returned a uint8 ptr aliased as BF16, and the
                        // absmax kernel read 2× the allocation.
                        Nvfp4Variant::CompressedTensors | Nvfp4Variant::Standard => {
                            // NVFP4-from-disk path: column-parallel Q/K/V, row-parallel O.
                            let group_size = 16usize;
                            let load_nvfp4 = |name: &str,
                                              full_n: usize,
                                              full_k: usize,
                                              kind: TpShardKind|
                             -> Result<crate::weight_map::QuantizedWeight> {
                                let src = quantized_any(
                                    store,
                                    &format!("{p}.{name}"),
                                    full_n,
                                    full_k,
                                    gpu,
                                    variant,
                                    QuantizeCtx {
                                        absmax_k,
                                        quantize_k,
                                        stream,
                                    },
                                )?;
                                if tp_size == 1 {
                                    return Ok(src);
                                }
                                let sharded = shard_quantized_nvfp4(
                                    &src, full_n, full_k, kind, tp_rank, tp_size, group_size, gpu,
                                )?;
                                gpu.free(src.weight)?;
                                gpu.free(src.weight_scale)?;
                                Ok(sharded)
                            };
                            let [q, k, v, o] = load_qkvo_tp(config, load_nvfp4)?;
                            let dummy = DenseWeight {
                                weight: spark_runtime::gpu::DevicePtr::NULL,
                            };
                            let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
                            let attn = AttentionWeights {
                                q_proj: dummy,
                                k_proj: dummy,
                                v_proj: dummy,
                                o_proj: o,
                                q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
                                k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
                                q_norm_full: None,
                                k_norm_full: None,
                                k_scale,
                                v_scale,
                            };
                            (attn, Some(q), Some(k), Some(v))
                        }
                        Nvfp4Variant::Fp8Dequanted | Nvfp4Variant::Bf16Raw => {
                            // BF16 → NVFP4 path: shard BF16 then quantize per-rank.
                            let load_bf16_then_nvfp4 = |name: &str,
                                                        full_n: usize,
                                                        full_k: usize,
                                                        kind: TpShardKind|
                             -> Result<(
                                DenseWeight,
                                crate::weight_map::QuantizedWeight,
                            )> {
                                let src = dense_auto(store, &format!("{p}.{name}.weight"), gpu)?;
                                let (sharded_ptr, local_n, local_k) = shard_dense_bf16(
                                    src.weight, full_n, full_k, kind, tp_rank, tp_size, gpu,
                                )?;
                                let sharded = DenseWeight {
                                    weight: sharded_ptr,
                                };
                                let q = quantize_to_nvfp4(
                                    &sharded, local_n, local_k, gpu, absmax_k, quantize_k, stream,
                                )?;
                                if sharded_ptr != src.weight {
                                    gpu.free(sharded_ptr)?;
                                }
                                Ok((src, q))
                            };
                            let [
                                (q_dense, q_nvfp4),
                                (k_dense, k_nvfp4),
                                (v_dense, v_nvfp4),
                                (_o_dense, o_nvfp4),
                            ] = load_qkvo_tp(config, load_bf16_then_nvfp4)?;

                            let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);

                            let attn = AttentionWeights {
                                q_proj: q_dense,
                                k_proj: k_dense,
                                v_proj: v_dense,
                                o_proj: o_nvfp4,
                                q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
                                k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
                                q_norm_full: None,
                                k_norm_full: None,
                                k_scale,
                                v_scale,
                            };
                            (attn, Some(q_nvfp4), Some(k_nvfp4), Some(v_nvfp4))
                        }
                    };

                    let mut layer = Qwen3AttentionLayer::new(
                        input_norm,
                        attn,
                        post_attn_norm,
                        ffn,
                        attn_idx,
                        q_nvfp4,
                        k_nvfp4,
                        v_nvfp4,
                        gpu,
                        layer_kv_dtypes[attn_idx],
                        config.fp8_kv_calibration_tokens,
                        config,
                    )?;

                    // Transpose NVFP4 weights for the small-M prefill /
                    // K=γ verify GEMM path (`w4a16_gemm_t_m16`). Without
                    // these, the multi_seq decode path falls back to
                    // per-token GEMV at γ=16 (3 × 17 launches/layer).
                    let num_heads = config.num_attention_heads;
                    let num_kv_heads = config.num_key_value_heads;
                    let head_dim = config.head_dim;
                    let gated = config.attn_gated;
                    let q_proj_n = if gated {
                        num_heads * head_dim * 2
                    } else {
                        num_heads * head_dim
                    };
                    if let Some(ref qw) = q_nvfp4 {
                        let qt = qw.transpose_for_gemm_cached(
                            gpu,
                            q_proj_n,
                            h,
                            &format!("L{i}.self_attn.q_proj.t"),
                        )?;
                        let kt = k_nvfp4.as_ref().unwrap().transpose_for_gemm_cached(
                            gpu,
                            num_kv_heads * head_dim,
                            h,
                            &format!("L{i}.self_attn.k_proj.t"),
                        )?;
                        let vt = v_nvfp4.as_ref().unwrap().transpose_for_gemm_cached(
                            gpu,
                            num_kv_heads * head_dim,
                            h,
                            &format!("L{i}.self_attn.v_proj.t"),
                        )?;
                        let ot = layer.attn.o_proj.transpose_for_gemm_cached(
                            gpu,
                            h,
                            num_heads * head_dim,
                            &format!("L{i}.self_attn.o_proj.t"),
                        )?;
                        layer.set_prefill_weights(Some(qt), Some(kt), Some(vt), Some(ot));
                        // Eagerly allocate the QKV split-K FP32 workspace
                        // (illegal during CUDA graph capture). Sized for the
                        // K/V output dim (kv_dim = num_kv_heads*head_dim);
                        // Q uses the single-slice kernel. No-op when
                        // ATLAS_ATTN_QKV_SPLITK is unset or kernels missing.
                        layer.alloc_qkv_splitk_workspace(gpu, (num_kv_heads * head_dim) as u32)?;
                    }

                    #[cfg(all(feature = "cuda", target_os = "linux"))]
                    if flashinfer_attn_proj_requested {
                        layer.prepare_flashinfer_projection_prefill(gpu, i, variant, config)?;
                    }

                    layers.push(Box::new(layer));
                    attn_idx += 1;
                }
                LayerType::LinearAttention => {
                    let nv = config.linear_num_value_heads;
                    let nk = config.linear_num_key_heads;
                    let qkv_rows = config.ssm_qkv_size();
                    let z_rows = config.ssm_z_size();
                    let value_dim = nv * config.linear_value_head_dim;

                    // load_ssm_qwen35 returns BF16 dense regardless of source
                    // quantization (modelopt uint8 .weight, compressed-tensors
                    // .weight_packed, or raw BF16). Including in_proj_a / b /
                    // out_proj which previous versions of this loader assumed
                    // were always BF16 — that assumption breaks on Sehyo /
                    // Huihui-style NVFP4 checkpoints which pack all five SSM
                    // projections.
                    let ssm35 = load_ssm_qwen35(store, &lp, gpu, variant, config)?;
                    let qkv_dense = ssm35.in_proj_qkv;
                    let z_dense = ssm35.in_proj_z;
                    let out_proj_dense = ssm35.out_proj;
                    let in_proj_a = ssm35.in_proj_a;
                    let in_proj_b = ssm35.in_proj_b;
                    let conv1d = ssm35.conv1d;
                    let a_log = ssm35.a_log;
                    let dt_bias = ssm35.dt_bias;
                    let norm = ssm35.norm;

                    let qkvz_dense =
                        gpu_concat_rows(&qkv_dense, qkv_rows, &z_dense, z_rows, h, gpu)?;

                    let ba_dense = interleave_ba(&in_proj_a, &in_proj_b, nv, nk, h, gpu)?;

                    let qkvz_size = config.ssm_qkvz_size();
                    let qkvz_nvfp4 = quantize_to_nvfp4(
                        &qkvz_dense,
                        qkvz_size,
                        h,
                        gpu,
                        absmax_k,
                        quantize_k,
                        stream,
                    )?;

                    let qkvz_nvfp4_t = qkvz_nvfp4.transpose_for_gemm_cached(
                        gpu,
                        qkvz_size,
                        h,
                        &format!("L{i}.linear_attn.qkvz.t"),
                    )?;

                    let out_proj_nvfp4 = quantize_to_nvfp4(
                        &out_proj_dense,
                        h,
                        value_dim,
                        gpu,
                        absmax_k,
                        quantize_k,
                        stream,
                    )?;

                    let out_proj_nvfp4_t = out_proj_nvfp4.transpose_for_gemm_cached(
                        gpu,
                        h,
                        value_dim,
                        &format!("L{i}.linear_attn.out_proj.t"),
                    )?;

                    let ssm = SsmWeights {
                        in_proj_qkvz: qkvz_dense,
                        in_proj_ba: ba_dense,
                        conv1d,
                        a_log,
                        dt_bias,
                        norm,
                        out_proj: out_proj_nvfp4,
                    };

                    let mut layer = Qwen3SsmLayer::new_sequential(
                        input_norm,
                        ssm,
                        post_attn_norm,
                        ffn,
                        Some(qkvz_nvfp4),
                        Some(qkvz_nvfp4_t),
                        Some(out_proj_nvfp4_t),
                        config,
                        gpu,
                    )?;
                    layer.predequant_for_prefill(gpu, config, stream)?;
                    #[cfg(all(feature = "cuda", target_os = "linux"))]
                    if flashinfer_ssm_proj_requested {
                        layer.prepare_flashinfer_ssm_prefill(gpu, i, config)?;
                    }
                    layers.push(Box::new(layer));
                }
                LayerType::Moe => unreachable!("Qwen3.5 dense has no standalone MoE layers"),
            }

            if (i + 1) % 10 == 0 {
                tracing::info!("Loaded layers 0..{}", i + 1);
                spark_runtime::progress::layer(i + 1, config.num_hidden_layers);
            }
        }

        tracing::info!(
            "Qwen3.5 dense weight loader: {} layers ({} attention, {} SSM, dense FFN)",
            layers.len(),
            attn_idx,
            layers.len() - attn_idx,
        );

        ffn::finish_w3_session(w3_session)?;
        super::transform_cache::finish();

        Ok(layers)
    }

    fn load_embedding(&self, store: &WeightStore, config: &ModelConfig, _gpu: &dyn GpuBackend) -> Result<DenseWeight> {
        let prefix = &config.weight_prefix;
        dense(store, &format!("{prefix}.embed_tokens.weight"))
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        let prefix = &config.weight_prefix;
        dense(store, &format!("{prefix}.norm.weight"))
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        super::qwen35_mixed_precision::load_lm_head(store, config, gpu)
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        // Dense models don't have MoE-shaped MTP — see `load_mtp_dense_weights`.
        Ok(None)
    }

    fn load_mtp_dense_weights(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<crate::weight_map::MtpDenseWeights>> {
        if !store.contains("mtp.fc.weight") {
            return Ok(None);
        }
        tracing::info!("Loading dense MTP weights (Qwen3.5/3.6 27B-class)...");
        let mtp = crate::weight_map::load_mtp_dense(store)?;
        if mtp.is_some() {
            tracing::info!(
                "Dense MTP weights loaded: 1 attention layer + dense MLP (no expert routing)"
            );
        }
        Ok(mtp)
    }

    /// Load the ViT tower for dense Qwen3.5/3.6 checkpoints that ship one.
    ///
    /// The dense weight loader handles the text-only Qwen3.6-27B-FP8 sibling
    /// (no vision) AND vision-capable dense checkpoints like AEON-Q36-27B,
    /// which carry the full 333-tensor `model.visual.*` tower in the same
    /// `qwen3_5` + `num_experts==0` config that routes here (see
    /// `factory::loader_for_config`). The default trait impl returns `None`,
    /// which silently dropped the vision tower for the latter — the image
    /// `<|image_pad|>` token was then embedded as plain text, producing
    /// fluent-but-wrong "vision" output. The ViT load logic is identical to
    /// the MoE-VL sibling (same `model.visual.*` / nested layout, same FP8→
    /// BF16 auto-dequant), so delegate to it. It returns `Ok(None)` when
    /// `config.vision` is `None` (the text-only dense path), keeping that
    /// path byte-for-byte unchanged.
    fn load_vision_encoder(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<crate::layers::VisionEncoder>> {
        super::Qwen35WeightLoader.load_vision_encoder(store, config, gpu)
    }
}
