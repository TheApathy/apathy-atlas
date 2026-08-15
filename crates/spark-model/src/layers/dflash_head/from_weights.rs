// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash drafter constructor + target-config validation.
//!
//! Split out of `dflash_head.rs` for file-size budget. Contains
//! [`BlockDiffusionDraftHead::from_weights`] (kernel resolution + KV
//! cache setup) and [`BlockDiffusionDraftHead::validate_against_target`].

use anyhow::Result;
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

use super::{
    BlockDiffusionDraftHead, DflashKernels, DflashLayer, DflashLayerNvfp4, DflashLayerQuantWeights,
    DflashQuantization, DflashScratch,
};
use crate::weight_loader::{DflashWeights, MarkovWeights};
use crate::weight_map::{DenseWeight, quantize_to_nvfp4};

impl BlockDiffusionDraftHead {
    pub fn from_weights(
        weights: DflashWeights,
        embed_tokens_shared: DevicePtr,
        lm_head_shared: DevicePtr,
        target_hidden_size: usize,
        target_vocab_size: usize,
        gamma: Option<usize>,
        window_size: Option<usize>,
        gpu: &dyn GpuBackend,
        max_seq_len: usize,
        quantization: DflashQuantization,
    ) -> Result<Self> {
        // Drafter's `fc` is `[draft_hidden, len(target_layer_ids) * target_hidden]`.
        // We rely on the drafter config's `hidden_size` and the parsed
        // `target_layer_ids` to derive the expected target_hidden, then
        // validate it matches what the caller provided.
        let target_layer_ids = weights
            .config
            .dflash_config
            .as_ref()
            .map(|c| c.target_layer_ids.clone())
            .unwrap_or_default();
        let mask_token_id = weights
            .config
            .dflash_config
            .as_ref()
            .map(|c| c.mask_token_id)
            .unwrap_or(0);

        if target_layer_ids.is_empty() {
            anyhow::bail!(
                "DFlash drafter config.json has no `dflash_config.target_layer_ids` — \
                 cannot determine which target hidden states to capture"
            );
        }

        let _ = target_hidden_size;

        let num_layers = weights.config.num_hidden_layers;
        let hidden_size = weights.config.hidden_size;
        let intermediate_size = weights.config.intermediate_size;
        let num_q_heads = weights.config.num_attention_heads;
        let num_kv_heads = weights.config.num_key_value_heads;
        let head_dim = weights.config.head_dim;
        let vocab_size = weights.config.vocab_size;
        let gamma_val = gamma.unwrap_or(weights.config.block_size);

        // Allocate the drafter's paged FP8 KV cache. One multi-layer cache,
        // sized for `max_seq_len + γ + 1` positions (prompt + γ drafts +
        // 1 bonus). Block size 16 matches the rest of Atlas.
        let block_size = 16;
        let kv_config = KvCacheConfig {
            block_size,
            num_kv_heads,
            head_dim,
            num_layers,
            dtype: KvCacheDtype::Fp8,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let num_blocks = (max_seq_len + gamma_val + 1) / block_size + 1;
        let kv_cache = PagedKvCache::new(kv_config, num_blocks, gpu)?;

        // Resolve kernel handles. All BF16 paths since drafter weights are
        // BF16 (DflashQuantization::Bf16); FP8 cache uses the FP8 reshape +
        // FP8-aware paged-attention kernel. Module/function names verified
        // against existing Atlas resolutions in `qwen3_attention/mod.rs` and
        // `mtp_head.rs` plus the `extern "C" __global__` declarations under
        // `kernels/gb10/nvfp4/`.
        let kernels = DflashKernels {
            // DFlash drafter uses HF's vanilla RMSNorm convention
            // (`out = x * w / RMS(x)`), NOT Atlas's default offset-from-1
            // form (`out = x * (1 + w) / RMS(x)`). Atlas's standard
            // `rms_norm` kernel includes the `+1` for Qwen3-Next-style
            // checkpoints; we must use `rms_norm_vanilla` for the drafter
            // to match the drafter's HF-trained weights exactly.
            rms_norm: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            // `rms_norm_residual` lands the post-attn / post-MLP add+norm in
            // a single launch. Atlas exposes this as a separate kernel — see
            // `mtp_head.rs:469` for the established lookup.
            residual_rms_norm: gpu
                .kernel("norm", "rms_norm_residual")
                .or_else(|_| gpu.kernel("residual_add", "bf16_residual_add"))?,
            dense_gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            dense_gemm: gpu.kernel("gemm", "dense_gemm_bf16")?,
            // Qwen3.6-DFlash uses yarn RoPE — confirmed in the drafter
            // `config.json:rope_scaling.rope_type="yarn"`. Atlas's yarn
            // kernel is `rope::rope_forward_yarn`.
            rope_qwen3: gpu.kernel("rope", "rope_forward_yarn")?,
            // FP8 KV cache writeback. Module name is the .cu stem
            // `reshape_and_cache`, function is `reshape_and_cache_flash_fp8`
            // (qwen3_attention/mod.rs:377-378 uses the same path).
            reshape_cache_fp8: gpu.kernel("reshape_and_cache", "reshape_and_cache_flash_fp8")?,
            // The Phase-2 γ-block kernel — same module as the existing
            // FP8 paged-prefill kernel (we just pass `causal_mask_enabled=0`
            // via a different dispatcher).
            prefill_attn_dflash_fp8: gpu
                .kernel("prefill_paged_fp8", "inferspark_prefill_paged_fp8")?,
            silu_mul: gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
            residual_add: gpu.kernel("residual_add", "bf16_residual_add")?,
            scaled_add: gpu.kernel("residual_add", "bf16_scaled_add")?,
            token_recommit: gpu.kernel("residual_add", "dflash_token_recommit")?,
            argmax: gpu.kernel("argmax", "argmax_bf16")?,
            // Top-K kernel lives in the same `argmax` module .cu file
            // (kernels/gb10/nvfp4/argmax_bf16.cu). Resolved unconditionally
            // so the handle table is shape-stable. Only invoked when the
            // DDTree propose path opts into real top-k tree construction
            // (chain-only mode skips it entirely).
            topk: gpu.kernel("argmax", "topk_bf16")?,
            batched_embed: gpu.kernel("embed_from_argmax", "batched_embed")?,
            // Drafter has head_dim=128, but qwen3.6-35b-a3b target's
            // `inferspark_prefill` is compiled with HDIM=256. Using that
            // kernel produces corrupted attn_out for the drafter (kernel
            // reads 256 elements per head when only 128 are valid →
            // garbage in the back half of SMEM tiles → per-head sign-flip
            // pattern across q-heads). The HDIM=128 specialization
            // `inferspark_prefill_h128` lives in the qwen3.6-35b-a3b
            // model-override kernel dir.
            prefill_attn: gpu.kernel("inferspark_prefill_h128", "inferspark_prefill_h128")?,
            // NVFP4 kernels: resolved unconditionally so the kernel-handle
            // table is shape-stable across both quantization paths. Under
            // `DflashQuantization::Bf16` these handles are never invoked
            // (forward_block_layer's match-dispatch on the layer variant
            // routes BF16 layers through `dense_gemm` / `dense_gemv`).
            // Naming matches the target-side resolutions in
            // `qwen3_attention/init.rs` and `dense_ffn.rs`.
            w4a16_gemv: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemm: gpu.kernel("w4a16", "w4a16_gemm")?,
            // M_TILE=16 specialization for the drafter FFN K=γ batched
            // path (ATLAS_DFLASH_FFN_KGAMMA=1). Resolved via try_kernel so
            // older built kernel caches that pre-date this symbol still
            // link; the dispatch in forward_block_layer_nvfp4 checks for
            // `KernelHandle(0)` and falls back to the M_TILE=64 path.
            w4a16_gemm_t_m16: crate::layers::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m16"),
            w4a16_gemm_t_m32_n64: crate::layers::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m32_n64"),
            fp8_gemm_t: crate::layers::try_kernel(gpu, "w4a16", "fp8_gemm_t"),
        };

        // Per-step scratch buffers. BF16 = 2 bytes/element.
        //
        // Sized for `n_attn_slots = ctx_window + γ` rows in the attention
        // path. The first `ctx_window` slots hold projected target ctx
        // (K/V only — Q is zero-padded so its attention output is
        // discarded). The next γ slots hold the noise tokens. lm_head +
        // logits + argmax tail still operates on γ rows (offset past ctx).
        let bf16 = 2usize;
        let g = gamma_val;
        // Phase 2.5n: ctx_window controls how many captured target positions
        // the drafter attends to per step. The drafter was trained over the
        // FULL captured prefix (paper §A.1), but capping at γ=16 cripples it
        // on prompts past a tiny window — Atlas's 6-10% acceptance vs the
        // paper's 70% is dominated by this cap. Default raised to 512;
        // ATLAS_DFLASH_CTX_WINDOW overrides at construction time.
        //
        // Memory cost: scratch buffers scale linearly with `n_attn = γ + cw`.
        // At cw=512: stream_buf = 528 × 2048 × 2 = 2.1 MB; mlp_intermediate
        // = 528 × 6144 × 2 = 6.3 MB; logits = 528 × 248320 × 2 = 257 MB
        // (largest). Total scratch ~280 MB per head — affordable.
        let ctx_window: usize = std::env::var("ATLAS_DFLASH_CTX_WINDOW")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(512);
        tracing::info!(
            "DFlash ctx_window = {} (set ATLAS_DFLASH_CTX_WINDOW to override; \
             drafter trained on full captured prefix — larger is better, \
             scratch grows linearly)",
            ctx_window
        );
        // n_attn = ctx_window + (γ+1). The +1 holds the bonus token at slot 0
        // of the noise block (γ MASKs at slots 1..γ). Drafter forward produces
        // γ+1 outputs; lm_head reads only the MASK rows [1..γ+1] → γ drafts.
        // Matches vLLM PR #40898 dflash.py `max_query_tokens = 1 + γ`.
        let n_attn = (g + 1) + ctx_window;
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let scratch = DflashScratch {
            stream_buf: gpu.alloc(n_attn * hidden_size * bf16)?,
            norm_buf: gpu.alloc(n_attn * hidden_size * bf16)?,
            q_buf: gpu.alloc(n_attn * q_dim * bf16)?,
            k_buf: gpu.alloc(n_attn * kv_dim * bf16)?,
            v_buf: gpu.alloc(n_attn * kv_dim * bf16)?,
            attn_out: gpu.alloc(n_attn * q_dim * bf16)?,
            mlp_intermediate: gpu.alloc(n_attn * intermediate_size * bf16)?,
            mlp_up: gpu.alloc(n_attn * intermediate_size * bf16)?,
            stream_acc: gpu.alloc(n_attn * hidden_size * bf16)?,
            fc_proj: gpu.alloc(ctx_window * hidden_size * bf16)?,
            logits: gpu.alloc(n_attn * vocab_size * bf16)?,
            draft_tokens_dev: gpu.alloc(n_attn * 4)?,
            position_ids: gpu.alloc(n_attn * 4)?,
            // DDTree M4B v2 top-K scratch. Sized for the max compile-time
            // K (16) × γ rows. u32 indices + f32 logits = 4 bytes each.
            // Allocated unconditionally so the scratch shape is independent
            // of the runtime ATLAS_DFLASH_METHOD value.
            topk_tokens_dev: gpu.alloc(g * super::DDTREE_TOP_K_MAX * 4)?,
            topk_logits_dev: gpu.alloc(g * super::DDTREE_TOP_K_MAX * 4)?,
            // EAGLE-3.1 per-layer FC-normalization scratch
            // (ATLAS_DFLASH_FC_LAYERNORM=1). `fc_norm_in` holds one fc-input
            // slot [n_target_layers * target_hidden] BF16; `fc_norm_zero_w` is
            // an all-zeros BF16 weight of length target_hidden so the
            // `rms_norm` kernel's `x * rms * (1 + w)` reduces to unit-variance
            // `x * rms` per slice. Both are allocated unconditionally (a few
            // KB) so the scratch shape is independent of the runtime flag; the
            // zero weight is memset to 0 below.
            fc_norm_in: gpu.alloc(target_layer_ids.len() * target_hidden_size * bf16)?,
            fc_norm_zero_w: gpu.alloc(target_hidden_size * bf16)?,
            // DSpark Markov head scratch. Allocated only when the checkpoint
            // ships the head (`weights.markov.is_some()`), else NULL — the
            // propose path only touches these on the Some branch.
            markov_w1_row: match weights.markov.as_ref() {
                Some(m) => gpu.alloc(m.rank * bf16)?,
                None => DevicePtr::NULL,
            },
            markov_bias: match weights.markov.as_ref() {
                Some(_) => gpu.alloc(vocab_size * bf16)?,
                None => DevicePtr::NULL,
            },
            markov_prev_dev: match weights.markov.as_ref() {
                Some(_) => gpu.alloc(4)?,
                None => DevicePtr::NULL,
            },
        };
        // Zero the FC-layernorm weight buffer so the per-slice rms_norm uses a
        // unit (1 + 0) scale → plain unit-variance normalization (variant a).
        gpu.memset(scratch.fc_norm_zero_w, 0, target_hidden_size * bf16)?;

        // Capture the DSpark Markov head weights before `weights.layers` is
        // consumed below. Weights stay BF16 on device even under NVFP4 drafter
        // quantization — the head operates directly on the shared-vocab logit
        // space and the per-position GEMV (K=rank=256) is tiny, so quantizing
        // it would add complexity for no measurable win.
        let markov_weights: Option<MarkovWeights> = weights.markov;

        // Pre-compute RoPE inv_freq table. Two paths based on the drafter's
        // `rope_scaling` config:
        //   - `Some(yarn)` → YaRN ramp interpolation (35B drafter)
        //   - `None`       → vanilla RoPE inv_freq (27B drafter — verified
        //                    via `z-lab/Qwen3.6-27B-DFlash/config.json` —
        //                    `rope_scaling: None`, `rope_theta: 10_000_000`)
        // Without this branch, the 27B drafter sees YaRN-rotated K/V and
        // attention scores diverge → ~3% accept rate (verified empirically).
        let rope_theta = weights.config.rope_theta;
        let rotary_dim = head_dim; // Qwen3.6-DFlash applies rope to full head_dim
        let dim_f = rotary_dim as f32;
        let n_pairs = rotary_dim / 2;
        let mut inv_freq_table = vec![0.0f32; n_pairs];
        if let Some(scaling) = weights.config.rope_scaling.as_ref()
            && scaling.rope_type.as_deref() == Some("yarn")
        {
            let factor = scaling.factor.unwrap_or(64.0);
            let beta_fast = scaling.beta_fast.unwrap_or(32.0);
            let beta_slow = scaling.beta_slow.unwrap_or(1.0);
            let orig_max_pos = scaling.original_max_position_embeddings.unwrap_or(4096) as f32;
            let find_correction_dim = |num_rot: f32| -> f32 {
                (dim_f * (orig_max_pos / (num_rot * 2.0 * std::f32::consts::PI)).ln())
                    / (2.0 * rope_theta.ln())
            };
            let low = find_correction_dim(beta_fast).floor().max(0.0);
            let high = find_correction_dim(beta_slow)
                .ceil()
                .min((rotary_dim - 1) as f32);
            let ramp_denom = if (high - low).abs() < 1e-6 {
                high - low + 0.001
            } else {
                high - low
            };
            for j in 0..n_pairs {
                let pos_freq = rope_theta.powf((2 * j) as f32 / dim_f);
                let inv_freq_extrap = 1.0 / pos_freq;
                let inv_freq_interp = 1.0 / (factor * pos_freq);
                let ramp = ((j as f32 - low) / ramp_denom).clamp(0.0, 1.0);
                let extrap_factor = 1.0 - ramp;
                inv_freq_table[j] =
                    inv_freq_interp * (1.0 - extrap_factor) + inv_freq_extrap * extrap_factor;
            }
            tracing::info!(
                "DFlash YaRN inv_freq: {n_pairs} pairs, factor={factor}, \
                 beta_fast={beta_fast}, beta_slow={beta_slow}, \
                 max_pos={orig_max_pos}, low_dim={low:.1}, high_dim={high:.1}"
            );
        } else {
            for j in 0..n_pairs {
                let pos_freq = rope_theta.powf((2 * j) as f32 / dim_f);
                inv_freq_table[j] = 1.0 / pos_freq;
            }
            tracing::info!(
                "DFlash vanilla RoPE inv_freq: {n_pairs} pairs, theta={rope_theta} \
                 (drafter has no rope_scaling)"
            );
        }
        let inv_freq_bytes: Vec<u8> = inv_freq_table
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let yarn_inv_freq = gpu.alloc(inv_freq_bytes.len())?;
        gpu.copy_h2d(&inv_freq_bytes, yarn_inv_freq)?;

        // Per-layer SWA window + causal flag (vLLM PR #40898 — Qwen3.6-27B-DFlash
        // has 4 sliding_attention + 1 full_attention layer).
        let (layer_window_sizes, layer_causal): (Vec<u32>, Vec<bool>) = match (
            weights.config.layer_types.as_ref(),
            weights.config.sliding_window,
        ) {
            (Some(types), Some(sw)) if types.len() == num_layers => {
                let mut sliding_count = 0usize;
                let mut windows = Vec::with_capacity(num_layers);
                let mut causals = Vec::with_capacity(num_layers);
                for t in types {
                    let is_sliding = t == "sliding_attention";
                    if is_sliding {
                        sliding_count += 1;
                        windows.push(sw as u32);
                    } else {
                        windows.push(0);
                    }
                    // Per-layer causal mask matching vLLM dflash.py:424-433.
                    // Sliding layers run causal + SWA (the drafter was
                    // trained with this mask); full-attention layers run
                    // non-causal across the [ctx, noise] block (bidirectional).
                    // Atlas's `inferspark_prefill_h128` kernel implements
                    // `(causal=true, sliding_window=W)` as the lower-triangular
                    // band `k >= q-W+1 AND k <= q`, matching flash-attn's
                    // `window_size=(W-1, 0)` semantics exactly — see
                    // kernels/gb10/qwen3.6-27b/nvfp4/inferspark_prefill_h128.cu
                    // lines 262-281 and vllm flash_attn.py:624 (`(sliding_window-1, 0)`).
                    causals.push(is_sliding);
                }
                tracing::info!(
                    "DFlash per-layer SWA: {sliding_count}/{num_layers} layers \
                     use sliding_window={sw} causal=true (causal+SWA); \
                     full layers causal=false (bidirectional)"
                );
                (windows, causals)
            }
            (Some(types), None) if types.iter().any(|t| t == "sliding_attention") => {
                anyhow::bail!(
                    "DFlash drafter has sliding_attention layers but no \
                     `sliding_window` in config — refusing to silently treat \
                     as full-attention (would break drafts)"
                );
            }
            _ => (Vec::new(), Vec::new()),
        };

        // Per-layer dimensions used for both the BF16 → NVFP4 quantization
        // (when requested) and downstream forward-pass kernel dispatch.
        let q_out = num_q_heads * head_dim;
        let kv_out = num_kv_heads * head_dim;

        // Build the per-layer weight variants. Under BF16, this just rewraps
        // the loaded `DflashLayerWeights` into `DflashLayerQuantWeights::Bf16`
        // — no GPU work. Under NVFP4, every dense projection is quantized
        // on-device via the existing `quantize_to_nvfp4` helper (the same
        // path the Gemma-4 / Qwen3.6-target loaders use), then the BF16
        // source buffer is freed to reclaim ~3.3 GB of GPU memory.
        let (layers, fc_nvfp4, fc_after) = match quantization {
            DflashQuantization::Bf16 => {
                let layers: Vec<DflashLayerQuantWeights> = weights
                    .layers
                    .into_iter()
                    .map(|l| {
                        DflashLayerQuantWeights::Bf16(DflashLayer {
                            input_layernorm: l.input_layernorm,
                            post_attention_layernorm: l.post_attention_layernorm,
                            q_proj: l.q_proj,
                            k_proj: l.k_proj,
                            v_proj: l.v_proj,
                            o_proj: l.o_proj,
                            q_norm: l.q_norm,
                            k_norm: l.k_norm,
                            gate_proj: l.gate_proj,
                            up_proj: l.up_proj,
                            down_proj: l.down_proj,
                        })
                    })
                    .collect();
                (layers, None, weights.fc)
            }
            DflashQuantization::Nvfp4 => {
                // Resolve the two quantize kernels once — same module/symbol
                // names the target-model NVFP4 loaders use.
                let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
                let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
                let qstream = gpu.default_stream();
                let target_hidden_dim = target_layer_ids.len() * target_hidden_size;

                // ATLAS_DFLASH_FFN_KGAMMA=1: also build transposed (nvfp4_t)
                // copies of the per-layer FFN projections so the per-layer
                // forward can route gate/up/down through the M_TILE=16
                // specialization (w4a16_gemm_n128_m16). The transposed
                // weights are *additional* allocations alongside the
                // standard nvfp4 layout — the existing M_TILE=64 path
                // remains the fallback when the gate is off OR the M=16
                // kernel handle failed to resolve. One-time host-side
                // transpose cost at model build (per QuantizedWeight ~89MB
                // H2D+D2H round trip; drafter is 5 layers × 3 FFN projs ~=
                // 1.3 GB total). Disabled when ffn_kgamma kernel handle
                // failed to resolve (older cached PTX).
                let want_ffn_kgamma_t =
                    crate::layers::dflash_ffn_kgamma_enabled() && kernels.w4a16_gemm_t_m16.0 != 0;
                if crate::layers::dflash_ffn_kgamma_enabled() && kernels.w4a16_gemm_t_m16.0 == 0 {
                    tracing::warn!(
                        "ATLAS_DFLASH_FFN_KGAMMA=1 set but w4a16_gemm_t_m16 \
                         kernel symbol missing — drafter FFN will use the \
                         M_TILE=64 (w4a16_gemm) fallback. Rebuild the kernel \
                         cache to enable the M_TILE=16 specialization."
                    );
                }

                // ATLAS_DFLASH_ATTN_KGAMMA=1: same as the FFN kgamma toggle
                // above, but applied to the per-layer attention projections
                // (q/k/v/o). Drafter Q-proj observed ~5.5ms × 5 layers and
                // o_proj ~4-6ms × 5 layers under the M_TILE=64 path because
                // verify-row count (M ≈ 17) leaves 73% of accumulator slots
                // unused. Routing through M_TILE=16 reclaims that time. Adds
                // 4 additional `transpose_for_gemm` H↔D round trips per layer
                // at model build (~1.5 GB × 5 layers ≈ 7.5 GB transient host
                // memory, freed after each transpose). Disabled when the
                // kgamma kernel handle failed to resolve.
                let want_attn_kgamma_t =
                    crate::layers::dflash_attn_kgamma_enabled() && kernels.w4a16_gemm_t_m16.0 != 0;
                if crate::layers::dflash_attn_kgamma_enabled() && kernels.w4a16_gemm_t_m16.0 == 0 {
                    tracing::warn!(
                        "ATLAS_DFLASH_ATTN_KGAMMA=1 set but w4a16_gemm_t_m16 \
                         kernel symbol missing — drafter attention will use \
                         the M_TILE=64 (w4a16_gemm) fallback. Rebuild the \
                         kernel cache to enable the M_TILE=16 specialization."
                    );
                }

                // Quantize the top-level `fc` projection first
                // (`[H, target_layer_ids.len() * target_hidden]`). N=H,
                // K=target_hidden_dim.
                let fc_q = quantize_to_nvfp4(
                    &weights.fc,
                    hidden_size,
                    target_hidden_dim,
                    gpu,
                    absmax_k,
                    quantize_k,
                    qstream,
                )?;
                // BUG #29: skip free; leave a placeholder so callers can't
                // accidentally dispatch the BF16 path.
                let _ = weights.fc.weight;
                let fc_after = DenseWeight {
                    weight: DevicePtr::NULL,
                };

                // Quantize each per-layer dense projection.
                let mut layers: Vec<DflashLayerQuantWeights> =
                    Vec::with_capacity(weights.layers.len());
                for l in weights.layers.into_iter() {
                    let q = quantize_to_nvfp4(
                        &l.q_proj,
                        q_out,
                        hidden_size,
                        gpu,
                        absmax_k,
                        quantize_k,
                        qstream,
                    )?;
                    let k = quantize_to_nvfp4(
                        &l.k_proj,
                        kv_out,
                        hidden_size,
                        gpu,
                        absmax_k,
                        quantize_k,
                        qstream,
                    )?;
                    let v = quantize_to_nvfp4(
                        &l.v_proj,
                        kv_out,
                        hidden_size,
                        gpu,
                        absmax_k,
                        quantize_k,
                        qstream,
                    )?;
                    let o = quantize_to_nvfp4(
                        &l.o_proj,
                        hidden_size,
                        q_out,
                        gpu,
                        absmax_k,
                        quantize_k,
                        qstream,
                    )?;
                    let gate = quantize_to_nvfp4(
                        &l.gate_proj,
                        intermediate_size,
                        hidden_size,
                        gpu,
                        absmax_k,
                        quantize_k,
                        qstream,
                    )?;
                    let up = quantize_to_nvfp4(
                        &l.up_proj,
                        intermediate_size,
                        hidden_size,
                        gpu,
                        absmax_k,
                        quantize_k,
                        qstream,
                    )?;
                    let down = quantize_to_nvfp4(
                        &l.down_proj,
                        hidden_size,
                        intermediate_size,
                        gpu,
                        absmax_k,
                        quantize_k,
                        qstream,
                    )?;
                    // DON'T free BF16 source buffers — BUG #29 (see
                    // crates/spark-model/src/weight_loader/nemotron.rs:49):
                    // gpu.free() on GB10 UVM posts in-band TLB invalidations
                    // that corrupt nearby allocations and trigger CUDA 700
                    // (illegal address) on subsequent kernel launches. 3.3 GB
                    // of headroom for the BF16 leftovers is acceptable in
                    // exchange for stability.
                    let _ = (
                        l.q_proj.weight,
                        l.k_proj.weight,
                        l.v_proj.weight,
                        l.o_proj.weight,
                        l.gate_proj.weight,
                        l.up_proj.weight,
                        l.down_proj.weight,
                    );

                    // Build transposed FFN weights when ATLAS_DFLASH_FFN_KGAMMA=1.
                    // For each projection: N = output dim, K = input dim
                    // (matches the (n, k) parameters of transpose_for_gemm:
                    // see qwen35_dense.rs:232/244 for the established pattern).
                    let (gate_t, up_t, down_t) = if want_ffn_kgamma_t {
                        // gate: [intermediate, hidden]; up: [intermediate, hidden]; down: [hidden, intermediate].
                        let gt = gate.transpose_for_gemm(gpu, intermediate_size, hidden_size)?;
                        let ut = up.transpose_for_gemm(gpu, intermediate_size, hidden_size)?;
                        let dt = down.transpose_for_gemm(gpu, hidden_size, intermediate_size)?;
                        (Some(gt), Some(ut), Some(dt))
                    } else {
                        (None, None, None)
                    };

                    // Build transposed attention weights when
                    // ATLAS_DFLASH_ATTN_KGAMMA=1. Same (n=output, k=input)
                    // convention as the FFN block above.
                    // q: [q_out, hidden]; k: [kv_out, hidden]; v: [kv_out, hidden];
                    // o: [hidden, q_out].
                    let (q_t, k_t, v_t, o_t) = if want_attn_kgamma_t {
                        let qt = q.transpose_for_gemm(gpu, q_out, hidden_size)?;
                        let kt = k.transpose_for_gemm(gpu, kv_out, hidden_size)?;
                        let vt = v.transpose_for_gemm(gpu, kv_out, hidden_size)?;
                        let ot = o.transpose_for_gemm(gpu, hidden_size, q_out)?;
                        (Some(qt), Some(kt), Some(vt), Some(ot))
                    } else {
                        (None, None, None, None)
                    };

                    layers.push(DflashLayerQuantWeights::Nvfp4(DflashLayerNvfp4 {
                        input_layernorm: l.input_layernorm,
                        post_attention_layernorm: l.post_attention_layernorm,
                        q_norm: l.q_norm,
                        k_norm: l.k_norm,
                        q_proj: q,
                        k_proj: k,
                        v_proj: v,
                        o_proj: o,
                        gate_proj: gate,
                        up_proj: up,
                        down_proj: down,
                        gate_proj_t: gate_t,
                        up_proj_t: up_t,
                        down_proj_t: down_t,
                        q_proj_t: q_t,
                        k_proj_t: k_t,
                        v_proj_t: v_t,
                        o_proj_t: o_t,
                    }));
                }

                tracing::info!(
                    "DFlash drafter quantized to NVFP4 ({} layers × 7 dense + fc); \
                     freed BF16 source buffers (~3.3 GB reclaimed); \
                     ffn_kgamma_t={} attn_kgamma_t={}",
                    layers.len(),
                    want_ffn_kgamma_t,
                    want_attn_kgamma_t,
                );
                (layers, Some(fc_q), fc_after)
            }
        };

        // DSpark VanillaMarkov head (runtime form). Weights are BF16 on device
        // (loaded by the drafter weight loader); we just carry the pointers +
        // rank. The scratch buffers above are already sized for `rank`/`vocab`.
        let markov = markov_weights.map(|m: MarkovWeights| super::MarkovHead {
            w1: m.w1,
            w2: m.w2,
            rank: m.rank,
        });

        // ── ATLAS_DFLASH_LM_HEAD_FP8=1: pre-scaled E4M3 copy of the propose
        // lm_head slice (see the field doc on `lm_head_shared_fp8`). One-time
        // load cost: absmax scan + scaled BF16 copy + FP8 cast over
        // lm_vocab × K (~0.5 GB result, ~1 GB transient). The compensating
        // 1/s goes into the final `norm` weight — its only consumer is the
        // propose lm_head GEMM (noise_pass Step 4), so all downstream logit
        // consumers (argmax, top-2 cliff margins) see true-scale values.
        let mut lm_head_shared_fp8 = None;
        let mut norm_after = weights.norm;
        if std::env::var("ATLAS_DFLASH_LM_HEAD_FP8").ok().as_deref() == Some("1") {
            let absmax_k = crate::layers::try_kernel(gpu, "quantize_nvfp4", "nvfp4_global_absmax");
            let tofp8_k = crate::layers::try_kernel(gpu, "w4a16", "bf16_to_fp8");
            if kernels.fp8_gemm_t.0 == 0 || absmax_k.0 == 0 || tofp8_k.0 == 0 {
                tracing::warn!(
                    "ATLAS_DFLASH_LM_HEAD_FP8=1 but fp8_gemm_t/absmax/bf16_to_fp8 \
                     kernel symbols missing — propose lm_head stays BF16"
                );
            } else {
                let n = target_vocab_size.min(vocab_size);
                let k = target_hidden_size;
                let total = (n * k) as u32;
                let qstream = gpu.default_stream();
                let gmax = gpu.alloc(4)?;
                gpu.memset(gmax, 0, 4)?;
                crate::layers::ops::nvfp4_global_absmax(
                    gpu, absmax_k, lm_head_shared, gmax, total, qstream,
                )?;
                gpu.synchronize(qstream)?;
                let mut b4 = [0u8; 4];
                gpu.copy_d2h(gmax, &mut b4)?;
                gpu.free(gmax)?;
                let absmax = f32::from_le_bytes(b4);
                // Power-of-2 scale targeting absmax·s ≈ 256 (E4M3 max 448):
                // exact in BF16 and in the folded norm, so the only lossy
                // step is the E4M3 cast itself.
                let s = if absmax > 0.0 {
                    (256.0f32 / absmax).log2().floor().exp2()
                } else {
                    1.0
                };
                let tmp = gpu.alloc(n * k * 2)?;
                gpu.memset(tmp, 0, n * k * 2)?;
                crate::layers::ops::scaled_add(
                    gpu, kernels.scaled_add, tmp, lm_head_shared, s, total, qstream,
                )?;
                let fp8 = gpu.alloc(n * k)?;
                crate::layers::ops::bf16_to_fp8(gpu, tofp8_k, tmp, fp8, total, qstream)?;
                gpu.synchronize(qstream)?;
                gpu.free(tmp)?;
                let norm_scaled = gpu.alloc(hidden_size * 2)?;
                gpu.memset(norm_scaled, 0, hidden_size * 2)?;
                crate::layers::ops::scaled_add(
                    gpu,
                    kernels.scaled_add,
                    norm_scaled,
                    weights.norm.weight,
                    1.0 / s,
                    hidden_size as u32,
                    qstream,
                )?;
                gpu.synchronize(qstream)?;
                norm_after = DenseWeight {
                    weight: norm_scaled,
                };
                lm_head_shared_fp8 = Some(fp8);
                tracing::info!(
                    "DFlash propose lm_head FP8: [{n} x {k}] absmax={absmax:.4} \
                     scale=2^{} (1/s folded into final norm)",
                    s.log2() as i32,
                );
            }
        }

        let head = Self {
            num_layers,
            hidden_size,
            intermediate_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            draft_vocab_size: weights.config.draft_vocab_size.unwrap_or(vocab_size),
            gamma: gamma_val,
            mask_token_id,
            window_size,
            layer_window_sizes,
            layer_causal,
            target_layer_ids,
            target_hidden_size,
            target_vocab_size,

            embed_tokens_shared,
            lm_head_shared,
            lm_head_shared_t: None,
            lm_head_shared_t_ldb: 0,
            lm_head_shared_fp8,
            hidden_norm: weights.hidden_norm,
            norm: norm_after,
            fc: fc_after,
            fc_nvfp4,
            draft_id_to_target_id: None,
            markov,
            layers,
            kv_cache: Mutex::new(kv_cache),
            scratch,
            kernels,
            max_seq_len,
            yarn_inv_freq,
            rope_theta,
            rotary_dim,
            rms_norm_eps: 1e-6,
            ctx_window,
            quant: quantization,
            async_inflight: Mutex::new(None),
            async_propose_stream: std::sync::OnceLock::new(),
            async_order_event: std::sync::atomic::AtomicU64::new(0),
            fused_event_armed: std::sync::atomic::AtomicBool::new(false),
        };

        tracing::info!(
            "BlockDiffusionDraftHead loaded: {} layers, hidden={}, intermediate={}, \
             GQA {}/{}, head_dim={}, γ={}, vocab={}, mask_token_id={}, target_layers={:?}",
            head.num_layers,
            head.hidden_size,
            head.intermediate_size,
            head.num_q_heads,
            head.num_kv_heads,
            head.head_dim,
            head.gamma,
            head.vocab_size,
            head.mask_token_id,
            head.target_layer_ids,
        );

        Ok(head)
    }

    /// Borrow-validate the drafter dimensions against the target's hidden_size
    /// at construction time. Mismatch is a hard error — the `fc` projection
    /// width is baked from `target_hidden_size` and a runtime mismatch would
    /// produce silent garbage (vLLM's loader hits this same check).
    pub fn validate_against_target(&self, target_hidden_size: usize) -> Result<()> {
        if self.target_hidden_size != target_hidden_size {
            anyhow::bail!(
                "DFlash drafter target_hidden_size mismatch: drafter expects {}, target is {}",
                self.target_hidden_size,
                target_hidden_size
            );
        }
        Ok(())
    }
}
