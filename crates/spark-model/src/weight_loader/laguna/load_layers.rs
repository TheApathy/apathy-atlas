// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, ensure};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layer::TransformerLayer;
use crate::layers::dense_ffn::DenseFfnWeights;
use crate::layers::qwen3_attention::HeadGateActivation;
use crate::layers::{DenseFfnLayer, FfnComponent, MoeLayer, Qwen3AttentionLayer};
use crate::weight_map::{
    AttentionWeights, DenseWeight, ExpertWeight, MoeWeights, QuantizedWeight, dense, dense_auto,
    quantize_to_nvfp4, quantized_v2,
};

pub(super) fn load_layers(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    ensure!(
        layer_kv_dtypes.len() == config.num_hidden_layers,
        "laguna requires one KV dtype per attention layer"
    );
    ensure!(
        config.shared_expert_intermediate_size == config.moe_intermediate_size,
        "laguna fused shared-expert path requires equal shared/routed widths"
    );

    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();
    let yarn_inv_freq = compute_yarn_inv_freq(config, gpu)?;
    // Sliding layers: theta=10000 over the full head_dim, no YaRN ramp.
    let sliding_inv_freq = if sliding_rope_table_enabled() {
        compute_plain_inv_freq(10_000.0, config.head_dim, gpu)?
    } else {
        DevicePtr::NULL
    };
    let unified_moe_layout =
        unified_moe_layout_enabled(std::env::var("ATLAS_UNIFIED_MOE_LAYOUT").ok().as_deref());
    if unified_moe_layout {
        tracing::info!(
            "Laguna: using unified transposed MoE layout; prefill uses fused K64 kernels and decode uses transposed experts"
        );
    }
    let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(config.num_hidden_layers);

    for i in 0..config.num_hidden_layers {
        let lp = format!("model.layers.{i}");
        let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
        let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;
        let ffn = if config.mlp_only_layers.contains(&i) {
            load_dense_ffn(store, gpu, &lp)?
        } else {
            load_moe_ffn(
                store,
                config,
                gpu,
                &lp,
                absmax_k,
                quantize_k,
                stream,
                unified_moe_layout,
                i,
            )?
        };
        let layer = load_attention(
            store,
            config,
            gpu,
            &lp,
            input_norm,
            post_attn_norm,
            ffn,
            layer_kv_dtypes[i],
            yarn_inv_freq,
            sliding_inv_freq,
            i,
            absmax_k,
            quantize_k,
            stream,
        )?;
        layers.push(Box::new(layer));
    }
    Ok(layers)
}

fn null_dense_ffn_weights() -> DenseFfnWeights {
    DenseFfnWeights {
        gate_proj: QuantizedWeight::null(),
        up_proj: QuantizedWeight::null(),
        down_proj: QuantizedWeight::null(),
        gate_proj_t: None,
        up_proj_t: None,
        down_proj_t: None,
    }
}

fn load_dense_ffn(store: &WeightStore, gpu: &dyn GpuBackend, lp: &str) -> Result<FfnComponent> {
    let mut layer = DenseFfnLayer::new(null_dense_ffn_weights(), gpu)?;
    layer.set_bf16_weights(
        dense_auto(store, &format!("{lp}.mlp.gate_proj.weight"), gpu)?,
        dense_auto(store, &format!("{lp}.mlp.up_proj.weight"), gpu)?,
        dense_auto(store, &format!("{lp}.mlp.down_proj.weight"), gpu)?,
    );
    Ok(FfnComponent::Dense(layer))
}

#[allow(clippy::too_many_arguments)]
fn load_moe_ffn(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    lp: &str,
    absmax_k: spark_runtime::gpu::KernelHandle,
    quantize_k: spark_runtime::gpu::KernelHandle,
    stream: u64,
    unified_moe_layout: bool,
    layer_idx: usize,
) -> Result<FfnComponent> {
    let mlp = format!("{lp}.mlp");
    let gate = dense(store, &format!("{mlp}.gate.weight"))?;
    let correction_bias = dense(store, &format!("{mlp}.experts.e_score_correction_bias"))?;

    // ── ATLAS_MOE_W3=1: 3-bit Lloyd-Max routed experts from the w3cache. ──
    // On success the NVFP4 expert tensors this layer would have used are
    // FREED (never held alongside W3). Any failure (missing file, bad
    // header, missing kernels, incompatible layout envs) warns once and
    // stays NVFP4 — never aborts.
    let w3_layer = maybe_load_w3_layer(config, gpu, layer_idx, unified_moe_layout);

    let experts = if let Some(w3) = &w3_layer {
        // Free the store's NVFP4 packed + scale buffers for every routed
        // expert of this layer (per-tensor allocations; the WeightStore
        // entries become dangling but the laguna loader is their only
        // consumer and reads them exactly once, here).
        let mut freed = 0usize;
        for e in 0..config.num_experts {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["weight_packed", "weight_scale"] {
                    let name = format!("{mlp}.experts.{e}.{proj}.{suffix}");
                    if let Ok(t) = store.get(&name) {
                        freed += t.byte_size();
                        let _ = gpu.free(t.ptr);
                    }
                }
            }
        }
        tracing::debug!(
            "Laguna L{layer_idx}: W3 experts active (+{} MiB W3, -{} MiB NVFP4 freed)",
            w3.device_bytes >> 20,
            freed >> 20,
        );
        w3.experts.clone()
    } else {
        (0..config.num_experts)
            .map(|e| {
                if !config.is_local_expert(e) {
                    return Ok(ExpertWeight::null());
                }
                let ep = format!("{mlp}.experts.{e}");
                Ok(ExpertWeight {
                    gate_proj: quantized_v2(store, &format!("{ep}.gate_proj"), gpu)?,
                    up_proj: quantized_v2(store, &format!("{ep}.up_proj"), gpu)?,
                    down_proj: quantized_v2(store, &format!("{ep}.down_proj"), gpu)?,
                })
            })
            .collect::<Result<Vec<_>>>()?
    };

    let shared = format!("{mlp}.shared_expert");
    let si = config.shared_expert_intermediate_size;
    let h = config.hidden_size;
    // Shared-expert precision differs across Laguna variants: S-2.1 ships it
    // BF16 (in the quant ignore list) and keeps BF16 authoritative; XS-2.1 ships
    // it NVFP4-packed (`.weight_packed`, NOT ignored) like the routed experts.
    // Detect and load accordingly — mirrors the routed `expert_proj` above.
    let shared_packed = store.contains(&format!("{shared}.gate_proj.weight_packed"));
    let shared_proj = |proj: &str, n: usize, k: usize| -> Result<QuantizedWeight> {
        if shared_packed {
            quantized_v2(store, proj, gpu)
        } else {
            let bf16 = dense_auto(store, &format!("{proj}.weight"), gpu)?;
            quantize_to_nvfp4(&bf16, n, k, gpu, absmax_k, quantize_k, stream)
        }
    };
    let shared_expert = ExpertWeight {
        gate_proj: shared_proj(&format!("{shared}.gate_proj"), si, h)?,
        up_proj: shared_proj(&format!("{shared}.up_proj"), si, h)?,
        down_proj: shared_proj(&format!("{shared}.down_proj"), h, si)?,
    };
    // BF16 variant (S-2.1): also keep the raw BF16 tensors to set authoritative
    // (the NVFP4 copies above are placeholders overwritten before blending).
    // NVFP4-packed variant (XS-2.1): the NVFP4 shared_expert IS authoritative.
    let bf16_shared = if shared_packed {
        None
    } else {
        Some((
            dense_auto(store, &format!("{shared}.gate_proj.weight"), gpu)?,
            dense_auto(store, &format!("{shared}.up_proj.weight"), gpu)?,
            dense_auto(store, &format!("{shared}.down_proj.weight"), gpu)?,
        ))
    };
    let weights = MoeWeights {
        gate,
        shared_expert,
        shared_expert_gate: DenseWeight {
            weight: DevicePtr::NULL,
        },
        experts,
        router_pre_norm: None,
        correction_bias: Some(correction_bias),
    };
    let mut layer = MoeLayer::new(weights, config.num_experts, None, gpu, config)?;
    if let Some(w3) = &w3_layer {
        // maybe_load_w3_layer verified the kernel set, so this cannot fail;
        // if it somehow does, that is a real bug — surface it.
        layer.enable_w3(w3.lut_dev)?;
        static W3_LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !W3_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::info!(
                "ATLAS_MOE_W3=1: 3-bit Lloyd-Max routed experts active (w3cache dir {:?}, \
                 lut {:?}) — NVFP4 expert memory freed as layers load",
                crate::weight_map::w3cache_dir(),
                w3.lut,
            );
        }
    }
    // S-2.1 excludes the shared expert from NVFP4 compression: keep its BF16
    // weights authoritative for both prefill and decode; the quantized copies
    // above are placeholders for fused routed kernels and their shared
    // contribution is overwritten before blending. XS-2.1 ships the shared
    // expert NVFP4-packed → no BF16 override, so the NVFP4 `weights.shared_expert`
    // (same machinery as the routed NVFP4 experts) is the authoritative path.
    if let Some((sg, su, sd)) = bf16_shared {
        layer.set_bf16_shared_expert(sg, su, sd)?;
    }
    // Keep-packed GGUF experts: the routed experts are raw Q4_K/Q6_K blocks and
    // carry NO NVFP4 scale tables, so the NVFP4-specific transpose and CUTLASS
    // SFB swizzle below (which read the null NVFP4 expert scales) must be
    // skipped — the keep-packed MoE prefill arm consumes the packed blocks via
    // q4k_mmq instead.
    // (Our tree lacks the GGUF keep-packed MoE arm — packed_experts field
    // comes from later upstream commits not cherry-picked; always NVFP4 here.)
    let experts_keep_packed = false;
    if unified_moe_layout && !experts_keep_packed {
        layer.transpose_for_prefill_unified(gpu, config)?;
    }
    // Native NVFP4 CUTLASS grouped MoE (ATLAS_HOLO_MOE_GROUPED_CUTLASS=1).
    // The routed grouped GEMMs are ~47% of Laguna's C=1 prefill GPU time and
    // otherwise run on the w4a16 kernels, which LUT-dequant NVFP4 to FP8 per
    // tile. The SFB swizzle is built from whichever scale tables exist —
    // transposed [K/16,N] under the unified layout, else the checkpoint's own
    // [N,K/16] via the src_n_major packer path.
    if cutlass_grouped_moe_enabled() {
        layer.build_cutlass_grouped_sfb(gpu, config, gpu.default_stream())?;
    }
    Ok(FfnComponent::Moe(layer))
}

fn unified_moe_layout_enabled(value: Option<&str>) -> bool {
    value.is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

/// ATLAS_MOE_W3: try to load this layer's 3-bit Lloyd-Max experts from the
/// w3cache. Every failure mode warns ONCE and returns `None` (stay NVFP4):
/// missing/invalid cache file, missing `_w3` kernels, or a layout env that
/// the W3 v1 kernel set cannot serve (transposed/unified/hybrid/CUTLASS/FP4
/// prefill lanes and EP all consume NVFP4 bytes the W3 path frees).
fn maybe_load_w3_layer(
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_idx: usize,
    unified_moe_layout: bool,
) -> Option<crate::weight_map::W3LoadedLayer> {
    use std::sync::atomic::{AtomicBool, Ordering};
    if !crate::weight_map::w3_enabled() {
        return None;
    }
    static WARNED: AtomicBool = AtomicBool::new(false);
    let warn_once = |msg: &str| {
        if !WARNED.swap(true, Ordering::Relaxed) {
            tracing::warn!("ATLAS_MOE_W3=1 requested but staying NVFP4: {msg}");
        }
    };

    let env_on = |k: &str| {
        matches!(
            std::env::var(k).as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE")
        )
    };
    // ATLAS_MOE_MIXED43: gate-ranked mixed 4.5/3-bit MoE (top-G experts read
    // NVFP4, the rest W3) was evaluated 2026-07 and is NOT implementable on
    // GB10: it needs BOTH formats resident for every expert (gate ranks vary
    // per token, so there is no stable NVFP4 subset), i.e. 59.5 GiB NVFP4 +
    // 46.3 GiB W3 routed experts in a 119 GiB unified LPDDR pool that also
    // holds attention weights, KV, drafter and the OS. Streaming the NVFP4
    // side from mmap is no escape either: cudaHostRegister pins pages in the
    // SAME physical pool (unified memory), and unpinned page-cache reads
    // leave ~40% of the ~475 MiB/token top-2 working set faulting from NVMe.
    // The fittable substitute is per-layer W3 codebooks (w3-requant
    // `--codebook per-layer`), which this build consumes transparently via
    // the per-layer LUT already present in every .w3x header.
    if env_on("ATLAS_MOE_MIXED43") {
        static MIXED_WARNED: AtomicBool = AtomicBool::new(false);
        if !MIXED_WARNED.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                "ATLAS_MOE_MIXED43=1 ignored: mixed NVFP4/W3 dual-residency does not fit \
                 GB10 unified memory (see load_layers.rs); serving W3-only with per-layer \
                 codebooks instead"
            );
        }
    }
    if unified_moe_layout
        || env_on("ATLAS_HYBRID_MOE_LAYOUT")
        || env_on("ATLAS_HOLO_MOE_GROUPED_CUTLASS")
        || env_on("ATLAS_HOLO_MOE_GATEUP_FP4")
        || env_on("ATLAS_HOLO_MOE_DOWN_FP4")
    {
        warn_once(
            "incompatible MoE layout env (unified/hybrid/CUTLASS/FP4 prefill lanes need NVFP4 bytes)",
        );
        return None;
    }
    if config.ep_world_size > 1 {
        warn_once("expert parallelism not supported by the W3 v1 path");
        return None;
    }
    // Full W3 kernel set must be compiled into this target image.
    let fused = |name: &str| gpu.kernel("moe_fused_w3", name).is_ok();
    if !(fused("moe_expert_gate_up_shared_w3")
        && fused("moe_expert_silu_down_shared_w3")
        && fused("moe_expert_gate_up_shared_batchN_w3")
        && fused("moe_expert_silu_down_shared_batchN_w3")
        && gpu
            .kernel("moe_w3a16", "moe_w3a16_grouped_gemm_ptrtable")
            .is_ok())
    {
        warn_once("W3 kernels (moe_fused_w3 / moe_w3a16 modules) missing from this build");
        return None;
    }
    let dir = crate::weight_map::w3cache_dir();
    match crate::weight_map::load_w3_layer(
        &dir,
        layer_idx,
        config.num_experts,
        config.hidden_size,
        config.moe_intermediate_size,
        gpu,
    ) {
        Ok(w3) => Some(w3),
        Err(e) => {
            warn_once(&format!("layer {layer_idx}: {e:#}"));
            None
        }
    }
}

/// FP8 attention-mirror env gate (`ATLAS_TARGET_ATTN_FP8_MIRROR=1`).
/// Default OFF — the decode/verify dispatch stays byte-identical BF16.
fn attn_fp8_mirror_enabled() -> bool {
    attn_fp8_mirror_enabled_value(
        std::env::var("ATLAS_TARGET_ATTN_FP8_MIRROR")
            .ok()
            .as_deref(),
    )
}

fn attn_fp8_mirror_enabled_value(value: Option<&str>) -> bool {
    value.is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

#[allow(clippy::too_many_arguments)]
fn load_attention(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    lp: &str,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
    kv_dtype: KvCacheDtype,
    yarn_inv_freq: DevicePtr,
    sliding_inv_freq: DevicePtr,
    i: usize,
    absmax_k: spark_runtime::gpu::KernelHandle,
    quantize_k: spark_runtime::gpu::KernelHandle,
    stream: u64,
) -> Result<Qwen3AttentionLayer> {
    let p = format!("{lp}.self_attn");
    let heads = config.num_attention_heads_per_layer[i];
    let q_width = heads * config.head_dim;
    validate_matrix(
        store,
        &format!("{p}.q_proj.weight"),
        q_width,
        config.hidden_size,
    )?;
    validate_matrix(
        store,
        &format!("{p}.g_proj.weight"),
        heads,
        config.hidden_size,
    )?;
    validate_matrix(
        store,
        &format!("{p}.o_proj.weight"),
        config.hidden_size,
        q_width,
    )?;

    let q_proj = dense_auto(store, &format!("{p}.q_proj.weight"), gpu)?;
    let k_proj = dense_auto(store, &format!("{p}.k_proj.weight"), gpu)?;
    let v_proj = dense_auto(store, &format!("{p}.v_proj.weight"), gpu)?;
    let o_proj = dense_auto(store, &format!("{p}.o_proj.weight"), gpu)?;
    let (k_scale, v_scale) = load_kv_scales(store, gpu, &p)?;

    // Lever A (ATLAS_LAGUNA_ATTN_NVFP4=1): the NVFP4 checkpoint keeps attention
    // BF16 (excluded from compression), which is ~33% of the decode weight-byte
    // traffic and the entire concurrency gap to GGUF (roofline: CUTLASS MoE is
    // already at the DRAM floor). Quantizing q/k/v/o to NVFP4 at load halves
    // those bytes → projected C4 43.6→~57.6 (beats vLLM 51 + GGUF 56). The
    // decode/prefill NVFP4 attention GEMV kernels already exist (fire on
    // as_nvfp4().is_some()). Opt-in; quality is uncalibrated static min-max so
    // gate behind the coherence check. Default OFF = BF16 attention unchanged.
    let attn_nvfp4 = std::env::var("ATLAS_LAGUNA_ATTN_NVFP4")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let kv_width = config.num_key_value_heads * config.head_dim;
    // Quantize q/k/v to NVFP4 (multi-seq ms_qkv_batch2/3/n read them batched-ONCE).
    // o_proj is kept BF16: MEASURED, o-NVFP4's w4a16_gemv_batch4/batch8 arms are
    // SLOWER than the BF16 dense_gemv_batchm at C=4/8 (full qkvo-NVFP4 tanked C4
    // 45.9→33.1, C8 55.2→49), so o stays on its batched BF16 read-once path.
    // Net win: single-stream C1 19.3→~30 (+55%, beats vLLM/GGUF) + C2/C4 gains;
    // a faster batched-NVFP4 o (and levers B/C: shared-expert/lm_head) is the
    // follow-up for concurrency parity with GGUF. o_proj is [N=hidden, K=q_width].
    let (q_nvfp4, k_nvfp4, v_nvfp4) = if attn_nvfp4 {
        (
            Some(quantize_to_nvfp4(&q_proj, q_width, config.hidden_size, gpu, absmax_k, quantize_k, stream)?),
            Some(quantize_to_nvfp4(&k_proj, kv_width, config.hidden_size, gpu, absmax_k, quantize_k, stream)?),
            Some(quantize_to_nvfp4(&v_proj, kv_width, config.hidden_size, gpu, absmax_k, quantize_k, stream)?),
        )
    } else {
        (None, None, None)
    };

    let attn = AttentionWeights {
        q_proj,
        k_proj,
        v_proj,
        o_proj: QuantizedWeight::null(),
        q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
        k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
        q_norm_full: None,
        k_norm_full: None,
        k_scale,
        v_scale,
    };
    let mut layer = Qwen3AttentionLayer::new_ungated(
        input_norm,
        attn,
        post_attn_norm,
        ffn,
        i,
        q_nvfp4,
        k_nvfp4,
        v_nvfp4,
        gpu,
        kv_dtype,
        config.fp8_kv_calibration_tokens,
        config,
    )?;
    layer.set_dimension_overrides(config.head_dim, heads, config.num_key_value_heads);
    // Lever A off (default): BF16 o_proj keeps the batched read-once
    // dense_gemv_batchm arm at n=2..8 — unchanged default path. Lever A on:
    // o_proj was installed as NVFP4 in `attn` above; o_dense_bf16 must stay
    // o_proj is always BF16 (batched read-once at n=2..8) — o-NVFP4 measured
    // slower at concurrency, so lever A quantizes only q/k/v.
    layer.set_o_dense_bf16(o_proj);
    // ── FP8-E4M3 row-scaled mirrors of the BF16 attention projections ──
    // (ATLAS_TARGET_ATTN_FP8_MIRROR=1, default OFF = byte-identical BF16).
    // Consumed only by the decode/verify GEMV/GEMM sites; prefill keeps
    // BF16 (cuBLASLt). Soft-fails to BF16 on missing kernels or OOM.
    if attn_fp8_mirror_enabled() {
        let kv_width = config.num_key_value_heads * config.head_dim;
        let bytes = layer.build_attn_fp8_mirrors(gpu, q_width, kv_width, config.hidden_size)?;
        if bytes > 0 {
            tracing::debug!(
                "Laguna L{i}: FP8 attention mirrors built (+{:.1} MiB)",
                bytes as f64 / (1024.0 * 1024.0)
            );
            if i == 0 {
                tracing::info!(
                    "ATLAS_TARGET_ATTN_FP8_MIRROR=1: building FP8 row-scaled attention \
                     mirrors for decode/verify (~{:.1} MiB per attention layer)",
                    bytes as f64 / (1024.0 * 1024.0)
                );
            }
        }
    }
    layer.set_head_gate_weight(
        dense_auto(store, &format!("{p}.g_proj.weight"), gpu)?,
        HeadGateActivation::Softplus,
    );
    match config.layer_types[i] {
        LayerType::SlidingAttention => {
            layer.set_sliding_window(Some(config.sliding_window));
            layer.set_rope_overrides(10_000.0, config.head_dim as u32);
            if !sliding_inv_freq.is_null() {
                // attention_factor = 1.0 => cos/sin unscaled, i.e. plain RoPE.
                layer.set_yarn_rope(sliding_inv_freq, 1.0);
            }
        }
        LayerType::FullAttention => {
            layer.set_sliding_window(None);
            layer.set_rope_overrides(config.rope_theta as f32, config.rotary_dim() as u32);
            layer.set_yarn_rope(yarn_inv_freq, config.yarn_attention_factor);
        }
        other => anyhow::bail!("laguna layer {i} is not attention: {other:?}"),
    }
    Ok(layer)
}

fn validate_matrix(store: &WeightStore, key: &str, rows: usize, cols: usize) -> Result<()> {
    let tensor = store.get(key)?;
    ensure!(
        tensor.shape == [rows, cols],
        "{key} shape {:?}, expected [{rows}, {cols}]",
        tensor.shape
    );
    Ok(())
}

fn load_kv_scales(store: &WeightStore, gpu: &dyn GpuBackend, prefix: &str) -> Result<(f32, f32)> {
    Ok((
        load_scalar(store, gpu, &format!("{prefix}.k_scale"))?,
        load_scalar(store, gpu, &format!("{prefix}.v_scale"))?,
    ))
}

fn load_scalar(store: &WeightStore, gpu: &dyn GpuBackend, key: &str) -> Result<f32> {
    let tensor = store.get(key)?;
    ensure!(
        tensor.shape.iter().product::<usize>() == 1,
        "{key} must be scalar"
    );
    match tensor.dtype {
        WeightDtype::BF16 => {
            let mut bytes = [0u8; 2];
            gpu.copy_d2h(tensor.ptr, &mut bytes)?;
            Ok(f32::from_bits((u16::from_le_bytes(bytes) as u32) << 16))
        }
        WeightDtype::FP32 => {
            let mut bytes = [0u8; 4];
            gpu.copy_d2h(tensor.ptr, &mut bytes)?;
            Ok(f32::from_le_bytes(bytes))
        }
        dtype => anyhow::bail!("{key} must be BF16 or F32, got {dtype:?}"),
    }
}

fn compute_yarn_inv_freq(config: &ModelConfig, gpu: &dyn GpuBackend) -> Result<DevicePtr> {
    let dim = config.rotary_dim();
    let dim_f = dim as f32;
    let theta = config.rope_theta as f32;
    let max_pos = config.yarn_original_max_position_embeddings as f32;
    let correction = |rotations: f32| {
        (dim_f * (max_pos / (rotations * 2.0 * std::f32::consts::PI)).ln()) / (2.0 * theta.ln())
    };
    let low = correction(config.yarn_beta_fast).floor().max(0.0);
    let high = correction(config.yarn_beta_slow)
        .ceil()
        .min((dim - 1) as f32);
    let denominator = if (high - low).abs() < 1e-6 {
        0.001
    } else {
        high - low
    };
    let values = (0..dim / 2)
        .map(|j| {
            let base = theta.powf((2 * j) as f32 / dim_f);
            let ramp = ((j as f32 - low) / denominator).clamp(0.0, 1.0);
            (1.0 - ramp) / base + ramp / (config.yarn_factor * base)
        })
        .collect::<Vec<_>>();
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let ptr = gpu
        .alloc(bytes.len())
        .context("allocate laguna YaRN table")?;
    gpu.copy_h2d(&bytes, ptr)?;
    Ok(ptr)
}

/// Precomputed plain RoPE inv_freq table for the sliding-attention layers.
///
/// Those layers use theta=10000 over the full head_dim with no YaRN ramp, and
/// the default rope kernel recomputes `1/theta^(2j/dim)` on the GPU with an
/// FP64 `pow` per pair index per block (kernels/gb10/common/rope.cu). For
/// Laguna's sliding layers rotary_dim == head_dim == 128, so a block covers
/// only 2 positions and pays 64 doubles to produce them — measured at 6.3% of
/// C=1 prefill GPU time. The table-based `rope_yarn_scaled` kernel is already
/// wired for this model (it serves the full-attention YaRN layers); feeding it
/// a plain table with attention_factor = 1.0 is the same math without the
/// per-block transcendentals.
///
/// Computed in f64 and narrowed once, so the stored values are at least as
/// accurate as the kernel's own FP64 `pow` followed by an f32 store.
/// Build the CUTLASS grouped-NVFP4 SFB tables at load
/// (`ATLAS_HOLO_MOE_GROUPED_CUTLASS=1`). Costs ~7.1 GB of device memory for
/// Laguna (256 experts x 47 layers x 3 projections), so it is opt-in.
fn cutlass_grouped_moe_enabled() -> bool {
    matches!(
        std::env::var("ATLAS_HOLO_MOE_GROUPED_CUTLASS").as_deref(),
        Ok("1") | Ok("true")
    )
}

fn compute_plain_inv_freq(theta: f64, dim: usize, gpu: &dyn GpuBackend) -> Result<DevicePtr> {
    let bytes = (0..dim / 2)
        .map(|j| (1.0f64 / theta.powf((2 * j) as f64 / dim as f64)) as f32)
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let ptr = gpu
        .alloc(bytes.len())
        .context("allocate laguna sliding-layer RoPE table")?;
    gpu.copy_h2d(&bytes, ptr)?;
    Ok(ptr)
}

/// Opt out of the precomputed sliding-layer RoPE table with
/// `ATLAS_LAGUNA_ROPE_TABLE=0` (falls back to the on-the-fly rope kernel).
fn sliding_rope_table_enabled() -> bool {
    std::env::var("ATLAS_LAGUNA_ROPE_TABLE").as_deref() != Ok("0")
}

#[cfg(test)]
mod tests {
    use super::{attn_fp8_mirror_enabled_value, unified_moe_layout_enabled};

    #[test]
    fn unified_moe_layout_is_explicitly_opt_in() {
        assert!(unified_moe_layout_enabled(Some("1")));
        assert!(unified_moe_layout_enabled(Some("true")));
        assert!(unified_moe_layout_enabled(Some("TRUE")));
        assert!(!unified_moe_layout_enabled(None));
        assert!(!unified_moe_layout_enabled(Some("0")));
        assert!(!unified_moe_layout_enabled(Some("full")));
    }

    // ATLAS_TARGET_ATTN_FP8_MIRROR is explicitly opt-in: default OFF keeps
    // the decode/verify dispatch byte-identical to the BF16 baseline.
    #[test]
    fn attn_fp8_mirror_is_explicitly_opt_in() {
        assert!(attn_fp8_mirror_enabled_value(Some("1")));
        assert!(attn_fp8_mirror_enabled_value(Some("true")));
        assert!(attn_fp8_mirror_enabled_value(Some("TRUE")));
        assert!(!attn_fp8_mirror_enabled_value(None));
        assert!(!attn_fp8_mirror_enabled_value(Some("0")));
        assert!(!attn_fp8_mirror_enabled_value(Some("mirror")));
    }
}
