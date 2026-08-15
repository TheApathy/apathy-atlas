// SPDX-License-Identifier: AGPL-3.0-only

//! Per-layer body of `BlockDiffusionDraftHead::forward_block`. Extracted
//! from `forward_block.rs` so the parent file fits the 500-LoC budget.
//! Contains the 12-step kernel chain (input_layernorm → q/k/v projections
//! → ctx K/V override → q_norm/k_norm → RoPE → attention → o_proj →
//! residual → post_attention_layernorm → MLP gate/up → silu_mul →
//! down_proj → residual). Called once per drafter layer from
//! `forward_block`'s Step 3 loop.

use anyhow::Result;
use std::sync::OnceLock;

use super::{BlockDiffusionDraftHead, DflashLayerNvfp4, DflashLayerQuantWeights};
use crate::layer::ForwardContext;

/// Per-layer SWA gate (vLLM PR #40898). When enabled, the drafter applies
/// the per-layer sliding-window from `layer_window_sizes` so that
/// `sliding_attention` layers respect their trained window (typically 2048)
/// and `full_attention` layers see the full prefix. When disabled, every
/// layer falls back to full attention (the pre-PR-40898 behavior, which
/// silently widens the SWA layers — this hurts acceptance on long-context
/// prompts because the drafter was trained with a finite window).
///
/// Default: ON. Set `ATLAS_DFLASH_SWA=0` to disable (A/B benchmark only).
fn dflash_swa_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let raw = std::env::var("ATLAS_DFLASH_SWA").ok();
        match raw.as_deref() {
            Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => {
                tracing::info!(
                    "DFlash per-layer SWA DISABLED (ATLAS_DFLASH_SWA=0); \
                     all drafter layers run full attention (pre-PR-40898 behavior)"
                );
                false
            }
            _ => {
                tracing::info!("DFlash per-layer SWA enabled (vLLM PR #40898 alignment)");
                true
            }
        }
    })
}

/// Noise-rows-only layer math (upstream dflash.py alignment). The reference
/// drafter runs its decoder layers on the γ+1 noise rows ONLY — ctx enters
/// attention purely as K/V projected from the stationary `fc_proj` output
/// (`k_ctx = k_proj(target_hidden)` per layer, dflash.py:71-76). Atlas
/// historically ran input_norm / q_proj / o_proj / residuals / FFN over the
/// full `n_attn = eff_ctx + γ+1` rows, with ctx-row results either zeroed
/// (q, attn_out) or computed-and-never-read (FFN, residual stream). At
/// ctx_window=512 that is ~30× wasted FFN rows per layer and the dominant
/// propose cost (gate_up 33ms + down 18ms per propose at eff_ctx≈100).
///
/// With this gate ON, the per-row ops shrink to the noise slice
/// [eff_ctx .. n_attn). Ops that genuinely cover ctx rows are unchanged:
/// k_norm + RoPE (ctx K is cached pre-rope; positions shift per step) and
/// the attention kernel itself (ctx rows participate as keys/values).
///
/// Default: OFF until validated. Set `ATLAS_DFLASH_NOISE_ONLY=1` to enable.
fn dflash_noise_only_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let on = std::env::var("ATLAS_DFLASH_NOISE_ONLY").ok().as_deref() == Some("1");
        if on {
            tracing::info!("DFlash noise-rows-only layer math ENABLED (ATLAS_DFLASH_NOISE_ONLY=1)");
        }
        on
    })
}

/// DeepLoop residual scaling (arxiv 2607.13491). When enabled and loop_pass ≥ 1,
/// residual updates are scaled by β = (8·N)^{-½} where N = loop_pass.
/// Gate re-exported from the head module for local use.
use super::dflash_deeploop_enabled;

/// Inputs passed to the per-layer kernel chain. Holds local computations
/// from the surrounding `forward_block` body so the helper can be called
/// without re-deriving them in every layer iteration.
#[allow(clippy::too_many_arguments)]
pub(super) struct LayerArgs {
    pub layer_idx: usize,
    pub n_attn: u32,
    pub eff_ctx: usize,
    pub h: u32,
    pub q_dim: u32,
    pub kv_dim: u32,
    pub inter: u32,
    pub bf16: usize,
    pub inv_sqrt_d: f32,
    pub stream: u64,
    pub needed_start: usize,
    pub window: usize,
    /// 0-indexed denoise pass number. Pass 0 = standard residual (scale 1.0).
    /// Passes ≥1 with `ATLAS_DFLASH_DEEPLOOP=1` apply DeepLoop β scaling.
    pub loop_pass: usize,
}

impl BlockDiffusionDraftHead {
    /// Run one drafter transformer layer. Mutates `self.scratch.*` buffers
    /// in place, leaving `stream_buf` updated with the layer's output.
    ///
    /// Dispatch on the per-layer weight quantization variant: BF16 layers
    /// route every projection through `ops::dense_gemm`; NVFP4 layers
    /// route them through `ops::w4a16_gemm` (same kernel the target model
    /// uses, ~7× faster than BF16 dense GEMM on GB10). All non-GEMM steps
    /// (RMSNorm, RoPE, attention, SiLU, residual add) are identical
    /// because the scratch buffers stay BF16.
    pub(super) fn forward_block_layer(
        &self,
        layer: &DflashLayerQuantWeights,
        args: &LayerArgs,
        ctx: &ForwardContext,
        debug_dump: bool,
        dstate: &mut super::DflashProposerState,
        kprofile: bool,
    ) -> Result<()> {
        match layer {
            DflashLayerQuantWeights::Bf16(l) => {
                self.forward_block_layer_bf16(l, args, ctx, debug_dump, dstate, kprofile)
            }
            DflashLayerQuantWeights::Nvfp4(l) => {
                self.forward_block_layer_nvfp4(l, args, ctx, debug_dump, dstate, kprofile)
            }
        }
    }

    /// BF16 per-layer body — verbatim from the pre-NVFP4 implementation.
    /// Reads `layer.{q,k,v,o,gate,up,down}_proj` as [`DenseWeight`] and
    /// dispatches each through `ops::dense_gemm`.
    fn forward_block_layer_bf16(
        &self,
        layer: &super::DflashLayer,
        args: &LayerArgs,
        ctx: &ForwardContext,
        debug_dump: bool,
        dstate: &mut super::DflashProposerState,
        kprofile: bool,
    ) -> Result<()> {
        use crate::layers::ops;

        let LayerArgs {
            layer_idx,
            n_attn,
            eff_ctx,
            h,
            q_dim,
            kv_dim,
            inter,
            bf16,
            inv_sqrt_d,
            stream,
            needed_start,
            window,
            loop_pass,
        } = *args;
        // DeepLoop (arxiv 2607.13491): β = (8·N)^{-½} on passes ≥1.
        let deeploop = loop_pass > 0 && dflash_deeploop_enabled();
        // β=1.0 default: pure iterative refinement. See NVFP4 path below.
        let loop_beta: f32 = if deeploop {
            std::env::var("ATLAS_DFLASH_DEEPLOOP_BETA")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0)
        } else {
            1.0
        };
        // Kernel profiler helper: synchronize+time when kprofile=true, then
        // accumulate into the thread-local KPROF_ACC field. Free when off.
        macro_rules! kp {
            ($field:ident, $body:expr) => {{
                if kprofile {
                    let t = std::time::Instant::now();
                    let r = $body;
                    ctx.gpu.synchronize(stream)?;
                    let dt = t.elapsed().as_micros();
                    super::kprof_add(|a| a.$field += dt);
                    r
                } else {
                    $body
                }
            }};
        }
        let cache_k = dstate.ctx_k_cache[layer_idx];
        let cache_v = dstate.ctx_v_cache[layer_idx];
        let mut cache_k_start = dstate.cache_k_start[layer_idx];
        let mut cache_k_end = dstate.cache_k_end[layer_idx];
        let mut cache_v_start = dstate.cache_v_start[layer_idx];
        let mut cache_v_end = dstate.cache_v_end[layer_idx];
        let gpu = ctx.gpu;

        let dump_bf16 = |label: &str, ptr: spark_runtime::gpu::DevicePtr, n: usize| -> Result<()> {
            if !debug_dump {
                return Ok(());
            }
            let mut buf = vec![0u8; n * 2];
            gpu.synchronize(stream)?;
            gpu.copy_d2h(ptr, &mut buf)?;
            let vals: Vec<f32> = buf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            tracing::info!("DFLASH DUMP {label} [{n}]: {:?}", &vals);
            Ok(())
        };

        // ATLAS_DFLASH_DEBUG_DUMP_ALL_LAYERS=1: per-layer binary dumps to
        // /tmp/atlas_layer{i}_{tag}.bin so the Python diff harness can
        // compare element-wise against vLLM at every layer.
        let dump_all_layers = std::env::var("ATLAS_DFLASH_DEBUG_DUMP_ALL_LAYERS")
            .ok()
            .as_deref()
            == Some("1");
        let dump_layer_bin =
            |tag: &str, ptr: spark_runtime::gpu::DevicePtr, n_elems: usize| -> Result<()> {
                if !dump_all_layers {
                    return Ok(());
                }
                let n_bytes = n_elems * 2;
                let mut buf = vec![0u8; n_bytes];
                gpu.synchronize(stream)?;
                gpu.copy_d2h(ptr, &mut buf)?;
                let path = format!("/tmp/atlas_layer{}_{}.bin", layer_idx, tag);
                if !std::path::Path::new(&path).exists() {
                    if let Err(e) = std::fs::write(&path, &buf) {
                        tracing::warn!("DFLASH DUMP_ALL_LAYERS: write {path} failed: {e}");
                    } else {
                        tracing::info!(
                            "DFLASH DUMP_ALL_LAYERS: wrote {n_bytes}B ({n_elems} BF16) to {path}"
                        );
                    }
                }
                Ok(())
            };

        // 3a. input_layernorm.
        kp!(
            input_norm_us,
            ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                self.scratch.stream_buf,
                &layer.input_layernorm,
                self.scratch.norm_buf,
                n_attn,
                h,
                self.rms_norm_eps,
                stream,
            )
        )?;

        // 3b. Q projection for all n_attn tokens (ctx + noise).
        kp!(
            q_proj_us,
            ops::dense_gemm_routed(
                gpu,
                self.kernels.dense_gemm,
                self.scratch.norm_buf,
                &layer.q_proj,
                self.scratch.q_buf,
                n_attn,
                q_dim,
                h,
                stream,
            )
        )?;
        if eff_ctx > 0 {
            gpu.memset_async(self.scratch.q_buf, 0, eff_ctx * q_dim as usize * bf16, stream)?;
        }

        // 3b'. K/V projections with persistent context cache.
        // Instead of recomputing K/V for ALL eff_ctx context tokens every
        // step, we copy cached K/V for old positions and only compute
        // new ones. This turns O(seq_len) k_proj/v_proj per layer into
        // O(new_tokens) ≈ O(1) amortized.
        //
        // PERF FIX (2026-05-19): demoted this per-layer log from
        // tracing::info! to tracing::debug! — it fired 5×/propose at INFO
        // level (one per drafter layer) and was on the critical path.
        // Re-enable via `RUST_LOG=spark_model::layers::dflash_head=debug`.
        let needed_end = needed_start + eff_ctx;
        tracing::debug!(
            "DFlash layer={} K/V cache: needed=[{}..{}), k_cached=[{}..{}), v_cached=[{}..{})",
            layer_idx,
            needed_start,
            needed_end,
            cache_k_start,
            cache_k_end,
            cache_v_start,
            cache_v_end,
        );
        let old_ctx_end = needed_end.min(cache_k_end).max(needed_start);
        let old_ctx_count = old_ctx_end.saturating_sub(needed_start);
        let new_ctx_count = eff_ctx.saturating_sub(old_ctx_count);

        if eff_ctx > 0 {
            // 1. Copy cached K/V for old context positions.
            if old_ctx_count > 0 {
                kp!(
                    kv_ctx_copy_us,
                    self.cache_copy_range(
                        gpu,
                        cache_k,
                        cache_k_start,
                        cache_k_end,
                        window,
                        needed_start,
                        old_ctx_end,
                        self.scratch.k_buf,
                        kv_dim as usize * bf16,
                        stream,
                    )
                )?;
                kp!(
                    kv_ctx_copy_us,
                    self.cache_copy_range(
                        gpu,
                        cache_v,
                        cache_v_start,
                        cache_v_end,
                        window,
                        needed_start,
                        old_ctx_end,
                        self.scratch.v_buf,
                        kv_dim as usize * bf16,
                        stream,
                    )
                )?;
            }
            // 2. Compute K/V for new context positions from fc_proj.
            if new_ctx_count > 0 {
                let fc_offset = old_ctx_count * self.hidden_size * bf16;
                let kv_offset = old_ctx_count * kv_dim as usize * bf16;
                kp!(
                    kv_ctx_new_us,
                    ops::dense_gemm_routed(
                        gpu,
                        self.kernels.dense_gemm,
                        self.scratch.fc_proj.offset(fc_offset),
                        &layer.k_proj,
                        self.scratch.k_buf.offset(kv_offset),
                        new_ctx_count as u32,
                        kv_dim,
                        h,
                        stream,
                    )
                )?;
                kp!(
                    kv_ctx_new_us,
                    ops::dense_gemm_routed(
                        gpu,
                        self.kernels.dense_gemm,
                        self.scratch.fc_proj.offset(fc_offset),
                        &layer.v_proj,
                        self.scratch.v_buf.offset(kv_offset),
                        new_ctx_count as u32,
                        kv_dim,
                        h,
                        stream,
                    )
                )?;
                // 3. Write new K/V into persistent cache.
                let (new_k_start, new_k_end) = kp!(
                    cache_write_us,
                    self.cache_write_range(
                        gpu,
                        self.scratch.k_buf.offset(kv_offset),
                        needed_start + old_ctx_count,
                        new_ctx_count,
                        cache_k,
                        cache_k_start,
                        cache_k_end,
                        window,
                        kv_dim as usize * bf16,
                        stream,
                    )
                )?;
                cache_k_start = new_k_start;
                cache_k_end = new_k_end;
                let (new_v_start, new_v_end) = kp!(
                    cache_write_us,
                    self.cache_write_range(
                        gpu,
                        self.scratch.v_buf.offset(kv_offset),
                        needed_start + old_ctx_count,
                        new_ctx_count,
                        cache_v,
                        cache_v_start,
                        cache_v_end,
                        window,
                        kv_dim as usize * bf16,
                        stream,
                    )
                )?;
                cache_v_start = new_v_start;
                cache_v_end = new_v_end;
            }
        }

        // 4. K/V projections for noise tokens (γ+1 rows at offset eff_ctx:
        // 1 bonus + γ MASKs; vLLM PR #40898 alignment).
        let noise_count = n_attn - eff_ctx as u32;
        let noise_offset = eff_ctx * self.hidden_size * bf16;
        let noise_kv_offset = eff_ctx * kv_dim as usize * bf16;
        kp!(
            kv_noise_us,
            ops::dense_gemm_routed(
                gpu,
                self.kernels.dense_gemm,
                self.scratch.norm_buf.offset(noise_offset),
                &layer.k_proj,
                self.scratch.k_buf.offset(noise_kv_offset),
                noise_count,
                kv_dim,
                h,
                stream,
            )
        )?;
        kp!(
            kv_noise_us,
            ops::dense_gemm_routed(
                gpu,
                self.kernels.dense_gemm,
                self.scratch.norm_buf.offset(noise_offset),
                &layer.v_proj,
                self.scratch.v_buf.offset(noise_kv_offset),
                noise_count,
                kv_dim,
                h,
                stream,
            )
        )?;

        if layer_idx == 0 {
            dump_bf16("layer0.k_buf[ctx0].pre_k_norm", self.scratch.k_buf, 10)?;
            dump_bf16("layer0.v_buf[ctx0]", self.scratch.v_buf, 10)?;
            let noise_q_offset = eff_ctx * q_dim as usize * bf16;
            let noise_k_offset = eff_ctx * kv_dim as usize * bf16;
            dump_bf16(
                "layer0.q_buf[noise0].pre_q_norm",
                self.scratch.q_buf.offset(noise_q_offset),
                10,
            )?;
            dump_bf16(
                "layer0.k_buf[noise0].pre_k_norm",
                self.scratch.k_buf.offset(noise_k_offset),
                10,
            )?;
        }

        // Store updated cache ranges back into proposer state.
        dstate.cache_k_start[layer_idx] = cache_k_start;
        dstate.cache_k_end[layer_idx] = cache_k_end;
        dstate.cache_v_start[layer_idx] = cache_v_start;
        dstate.cache_v_end[layer_idx] = cache_v_end;

        // 3c. q_norm / k_norm — per-head RMSNorm over head_dim slices.
        kp!(
            qk_norm_us,
            ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                self.scratch.q_buf,
                &layer.q_norm,
                self.scratch.q_buf,
                n_attn * self.num_q_heads as u32,
                self.head_dim as u32,
                self.rms_norm_eps,
                stream,
            )
        )?;
        kp!(
            qk_norm_us,
            ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                self.scratch.k_buf,
                &layer.k_norm,
                self.scratch.k_buf,
                n_attn * self.num_kv_heads as u32,
                self.head_dim as u32,
                self.rms_norm_eps,
                stream,
            )
        )?;
        if layer_idx == 0 {
            dump_bf16("layer0.k_buf[ctx0].post_k_norm", self.scratch.k_buf, 10)?;
            let noise_q_offset = eff_ctx * q_dim as usize * bf16;
            let noise_k_offset = eff_ctx * kv_dim as usize * bf16;
            dump_bf16(
                "layer0.q_buf[noise0].post_q_norm",
                self.scratch.q_buf.offset(noise_q_offset),
                10,
            )?;
            dump_bf16(
                "layer0.k_buf[noise0].post_k_norm",
                self.scratch.k_buf.offset(noise_k_offset),
                10,
            )?;
        }
        dump_layer_bin(
            "q_post_norm",
            self.scratch.q_buf,
            n_attn as usize * q_dim as usize,
        )?;
        dump_layer_bin(
            "k_post_norm",
            self.scratch.k_buf,
            n_attn as usize * kv_dim as usize,
        )?;
        dump_layer_bin(
            "v_buf",
            self.scratch.v_buf,
            n_attn as usize * kv_dim as usize,
        )?;

        // 3d. yarn RoPE — n_attn positions.
        kp!(
            rope_us,
            ops::rope_yarn(
                gpu,
                self.kernels.rope_qwen3,
                self.scratch.q_buf,
                self.scratch.k_buf,
                self.scratch.position_ids,
                n_attn,
                self.num_q_heads as u32,
                self.num_kv_heads as u32,
                self.head_dim as u32,
                self.rotary_dim as u32,
                self.yarn_inv_freq,
                self.rope_theta,
                stream,
            )
        )?;
        if layer_idx == 0 {
            let noise_q_offset = eff_ctx * q_dim as usize * bf16;
            let noise_k_offset = eff_ctx * kv_dim as usize * bf16;
            dump_bf16(
                "layer0.q_buf[noise0].post_rope",
                self.scratch.q_buf.offset(noise_q_offset),
                10,
            )?;
            dump_bf16(
                "layer0.k_buf[noise0].post_rope",
                self.scratch.k_buf.offset(noise_k_offset),
                10,
            )?;
            dump_bf16("layer0.k_buf[ctx0].post_rope", self.scratch.k_buf, 10)?;
        }
        dump_layer_bin(
            "q_post_rope",
            self.scratch.q_buf,
            n_attn as usize * q_dim as usize,
        )?;
        dump_layer_bin(
            "k_post_rope",
            self.scratch.k_buf,
            n_attn as usize * kv_dim as usize,
        )?;

        // 3e. attention — per-layer SWA window + causal (vLLM PR #40898).
        // sliding_attention layers: window=sliding_window (2048), causal=TRUE.
        // full_attention layers:    window=0 (full attn),       causal=FALSE.
        // Drafter is block-diffusion across γ noise tokens, but vLLM's
        // dflash.py:424-433 specifically applies a causal sliding-window
        // mask to sliding layers (the drafter was trained with this).
        // The flag is wired from `from_weights.rs:280` (causals.push(is_sliding)).
        // ATLAS_DFLASH_SWA=0 force-disables the per-layer window for A/B benchmarks.
        let swa_on = dflash_swa_enabled();
        let layer_window = if swa_on {
            self.layer_window_sizes.get(layer_idx).copied().unwrap_or(0)
        } else {
            0
        };
        let layer_causal_b = if swa_on {
            self.layer_causal.get(layer_idx).copied().unwrap_or(false)
        } else {
            false
        };
        kp!(
            prefill_attn_us,
            ops::prefill_attention(
                gpu,
                self.kernels.prefill_attn,
                self.scratch.q_buf,
                self.scratch.k_buf,
                self.scratch.v_buf,
                self.scratch.attn_out,
                n_attn,
                1,
                self.num_q_heads as u32,
                self.num_kv_heads as u32,
                self.head_dim as u32,
                inv_sqrt_d,
                layer_causal_b,
                layer_window,
                stream,
            )
        )?;
        // Zero context rows in attn_out so garbage uniform scores from zeroed Q
        // don't corrupt the residual stream through o_proj.
        if eff_ctx > 0 {
            gpu.memset_async(self.scratch.attn_out, 0, eff_ctx * q_dim as usize * bf16, stream)?;
        }
        if layer_idx == 0 {
            let noise_q_offset = eff_ctx * q_dim as usize * bf16;
            dump_bf16(
                "layer0.attn_out[noise0]",
                self.scratch.attn_out.offset(noise_q_offset),
                10,
            )?;
            dump_bf16(
                "layer0.attn_out[noise0][1000..1010]",
                self.scratch.attn_out.offset(noise_q_offset + 1000 * bf16),
                10,
            )?;
            dump_bf16(
                "layer0.attn_out[noise0][4086..4096]",
                self.scratch.attn_out.offset(noise_q_offset + 4086 * bf16),
                10,
            )?;
            // ATLAS_DFLASH_DEBUG_DUMP_FULL=1: write the FULL 4096-element
            // attn_out[noise0] row to /tmp/atlas_attn_out.bin so PyTorch
            // can run o_proj on the exact same bytes.
            if std::env::var("ATLAS_DFLASH_DEBUG_DUMP_FULL")
                .ok()
                .as_deref()
                == Some("1")
            {
                let n_bytes = q_dim as usize * bf16;
                let mut buf = vec![0u8; n_bytes];
                gpu.synchronize(stream)?;
                gpu.copy_d2h(self.scratch.attn_out.offset(noise_q_offset), &mut buf)?;
                std::fs::write("/tmp/atlas_attn_out.bin", &buf)
                    .map_err(|e| anyhow::anyhow!("write attn_out dump: {e}"))?;
                tracing::info!(
                    "DFLASH DUMP wrote {} bytes attn_out[noise0] to /tmp/atlas_attn_out.bin",
                    n_bytes
                );
            }
        }
        dump_layer_bin(
            "attn_out",
            self.scratch.attn_out,
            n_attn as usize * q_dim as usize,
        )?;

        // 3f. o_proj.
        kp!(
            o_proj_us,
            ops::dense_gemm_routed(
                gpu,
                self.kernels.dense_gemm,
                self.scratch.attn_out,
                &layer.o_proj,
                self.scratch.stream_acc,
                n_attn,
                h,
                q_dim,
                stream,
            )
        )?;
        if layer_idx == 0 {
            let noise_offset = eff_ctx * self.hidden_size * bf16;
            dump_bf16(
                "layer0.stream_acc[noise0].post_o_proj",
                self.scratch.stream_acc.offset(noise_offset),
                10,
            )?;
            dump_bf16(
                "layer0.stream_buf[noise0].pre_residual",
                self.scratch.stream_buf.offset(noise_offset),
                10,
            )?;
            if std::env::var("ATLAS_DFLASH_DEBUG_DUMP_FULL")
                .ok()
                .as_deref()
                == Some("1")
            {
                let n_bytes = self.hidden_size * bf16;
                let mut buf = vec![0u8; n_bytes];
                gpu.synchronize(stream)?;
                gpu.copy_d2h(self.scratch.stream_acc.offset(noise_offset), &mut buf)?;
                std::fs::write("/tmp/atlas_o_proj_out.bin", &buf)
                    .map_err(|e| anyhow::anyhow!("write o_proj_out: {e}"))?;
            }
        }
        dump_layer_bin(
            "stream_acc_post_o_proj",
            self.scratch.stream_acc,
            n_attn as usize * h as usize,
        )?;

        // 3g. residual: stream_buf += [β·]stream_acc. β=1 on pass 0 (standard);
        // β=(8·N)^{-½} on pass N≥1 with ATLAS_DFLASH_DEEPLOOP=1 (DeepLoop).
        kp!(
            resid1_us,
            if deeploop {
                ops::scaled_add(gpu, self.kernels.scaled_add, self.scratch.stream_buf,
                    self.scratch.stream_acc, loop_beta, n_attn * h, stream)
            } else {
                ops::residual_add(gpu, self.kernels.residual_add, self.scratch.stream_buf,
                    self.scratch.stream_acc, n_attn * h, stream)
            }
        )?;
        if layer_idx == 0 {
            let noise_offset = eff_ctx * self.hidden_size * bf16;
            dump_bf16(
                "layer0.stream_buf[noise0].post_attn_residual",
                self.scratch.stream_buf.offset(noise_offset),
                10,
            )?;
        }

        // 3h. post_attention_layernorm.
        kp!(
            post_norm_us,
            ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                self.scratch.stream_buf,
                &layer.post_attention_layernorm,
                self.scratch.norm_buf,
                n_attn,
                h,
                self.rms_norm_eps,
                stream,
            )
        )?;

        // 3i. MLP: gate + up.
        kp!(
            gate_up_us,
            ops::dense_gemm_routed(
                gpu,
                self.kernels.dense_gemm,
                self.scratch.norm_buf,
                &layer.gate_proj,
                self.scratch.mlp_intermediate,
                n_attn,
                inter,
                h,
                stream,
            )
        )?;
        kp!(
            gate_up_us,
            ops::dense_gemm_routed(
                gpu,
                self.kernels.dense_gemm,
                self.scratch.norm_buf,
                &layer.up_proj,
                self.scratch.mlp_up,
                n_attn,
                inter,
                h,
                stream,
            )
        )?;

        // 3j. silu_mul.
        kp!(
            silu_mul_us,
            ops::silu_mul(
                gpu,
                self.kernels.silu_mul,
                self.scratch.mlp_intermediate,
                self.scratch.mlp_up,
                self.scratch.mlp_intermediate,
                n_attn * inter,
                stream,
            )
        )?;

        // 3k. down_proj.
        kp!(
            down_proj_us,
            ops::dense_gemm_routed(
                gpu,
                self.kernels.dense_gemm,
                self.scratch.mlp_intermediate,
                &layer.down_proj,
                self.scratch.stream_acc,
                n_attn,
                h,
                inter,
                stream,
            )
        )?;

        // 3l. residual (DeepLoop β same as 3g).
        kp!(
            resid2_us,
            if deeploop {
                ops::scaled_add(gpu, self.kernels.scaled_add, self.scratch.stream_buf,
                    self.scratch.stream_acc, loop_beta, n_attn * h, stream)
            } else {
                ops::residual_add(gpu, self.kernels.residual_add, self.scratch.stream_buf,
                    self.scratch.stream_acc, n_attn * h, stream)
            }
        )?;
        if layer_idx == 0 {
            let noise_offset = eff_ctx * self.hidden_size * bf16;
            dump_bf16(
                "layer0.stream_buf[noise0].post_layer",
                self.scratch.stream_buf.offset(noise_offset),
                10,
            )?;
        }
        dump_layer_bin(
            "stream_buf_post_mlp",
            self.scratch.stream_buf,
            n_attn as usize * h as usize,
        )?;

        Ok(())
    }

    /// NVFP4 per-layer body — structurally identical to the BF16 path, but
    /// every dense projection (q/k/v/o + gate/up/down) is dispatched through
    /// `ops::w4a16_gemm` against a [`QuantizedWeight`]. Activations remain
    /// BF16 in scratch buffers; the kernel reads BF16 input, dequantizes
    /// the NVFP4 weight on-chip, and writes BF16 output. This is the same
    /// kernel the target model uses for its prefill GEMMs, so we inherit
    /// its ~7× throughput advantage over BF16 dense_gemm on GB10.
    fn forward_block_layer_nvfp4(
        &self,
        layer: &DflashLayerNvfp4,
        args: &LayerArgs,
        ctx: &ForwardContext,
        debug_dump: bool,
        dstate: &mut super::DflashProposerState,
        kprofile: bool,
    ) -> Result<()> {
        use crate::layers::ops;

        let LayerArgs {
            layer_idx,
            n_attn,
            eff_ctx,
            h,
            q_dim,
            kv_dim,
            inter,
            bf16,
            inv_sqrt_d,
            stream,
            needed_start,
            window,
            loop_pass,
        } = *args;
        let deeploop = loop_pass > 0 && dflash_deeploop_enabled();
        // β=1.0 default: pure iterative refinement. DeepLoop's (8N)^{-0.5}
        // paper formula requires a model trained for it — override via
        // ATLAS_DFLASH_DEEPLOOP_BETA for experimentation.
        let loop_beta: f32 = if deeploop {
            std::env::var("ATLAS_DFLASH_DEEPLOOP_BETA")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0)
        } else {
            1.0
        };
        // Kernel profiler helper (NVFP4 path). Identical to BF16 helper.
        macro_rules! kp {
            ($field:ident, $body:expr) => {{
                if kprofile {
                    let t = std::time::Instant::now();
                    let r = $body;
                    ctx.gpu.synchronize(stream)?;
                    let dt = t.elapsed().as_micros();
                    super::kprof_add(|a| a.$field += dt);
                    r
                } else {
                    $body
                }
            }};
        }
        let cache_k = dstate.ctx_k_cache[layer_idx];
        let cache_v = dstate.ctx_v_cache[layer_idx];
        let mut cache_k_start = dstate.cache_k_start[layer_idx];
        let mut cache_k_end = dstate.cache_k_end[layer_idx];
        let mut cache_v_start = dstate.cache_v_start[layer_idx];
        let mut cache_v_end = dstate.cache_v_end[layer_idx];
        let gpu = ctx.gpu;

        // Noise-rows-only row range (see dflash_noise_only_enabled). Per-row
        // ops run on rows [row0, row0 + m_rows); ctx-row regions of the
        // touched scratch buffers become stale, which is safe because ctx
        // K/V comes from the persistent cache + fc_proj (never from the
        // evolving stream/norm buffers) and the final norm/lm_head in
        // forward_block reads the noise slice only.
        let noise_only = dflash_noise_only_enabled();
        let (m_rows, row0) = if noise_only {
            (n_attn - eff_ctx as u32, eff_ctx)
        } else {
            (n_attn, 0usize)
        };
        let row0_h = row0 * h as usize * bf16;
        let row0_q = row0 * q_dim as usize * bf16;
        let row0_inter = row0 * inter as usize * bf16;

        let dump_bf16 = |label: &str, ptr: spark_runtime::gpu::DevicePtr, n: usize| -> Result<()> {
            if !debug_dump {
                return Ok(());
            }
            let mut buf = vec![0u8; n * 2];
            gpu.synchronize(stream)?;
            gpu.copy_d2h(ptr, &mut buf)?;
            let vals: Vec<f32> = buf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            tracing::info!("DFLASH DUMP {label} [{n}]: {:?}", &vals);
            Ok(())
        };

        // ATLAS_DFLASH_DEBUG_DUMP_ALL_LAYERS=1: per-layer binary dumps to
        // /tmp/atlas_layer{i}_{tag}.bin so the Python diff harness can
        // compare element-wise against vLLM at every layer.
        let dump_all_layers = std::env::var("ATLAS_DFLASH_DEBUG_DUMP_ALL_LAYERS")
            .ok()
            .as_deref()
            == Some("1");
        let _dump_layer_bin =
            |_tag: &str, _ptr: spark_runtime::gpu::DevicePtr, _n_elems: usize| -> Result<()> {
                // NVFP4 path: dumps not wired up yet (BF16 path is enough for
                // first-divergence detection vs vLLM, which is BF16 too).
                if !dump_all_layers {
                    return Ok(());
                }
                Ok(())
            };

        // 3a. input_layernorm. Identical to BF16 path. Noise-only: ctx rows
        // of norm_buf go stale — consumed only by the full-M q/FFN GEMMs
        // that shrink with this gate (noise K/V reads the noise slice).
        kp!(
            input_norm_us,
            ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                self.scratch.stream_buf.offset(row0_h),
                &layer.input_layernorm,
                self.scratch.norm_buf.offset(row0_h),
                m_rows,
                h,
                self.rms_norm_eps,
                stream,
            )
        )?;

        // 3b. Q projection for all n_attn tokens. When the drafter was built
        // with ATLAS_DFLASH_ATTN_KGAMMA=1 and the transposed attention
        // weights are present (q_proj_t / k_proj_t / v_proj_t / o_proj_t all
        // Some), route through the M_TILE=16 specialization
        // (w4a16_gemm_n128_m16) instead of the M_TILE=64 default. Mirrors
        // the FFN-kgamma dispatch (see Step 3i below). Each branch is a
        // straight 1:1 substitution — same scratch buffers, same (m, n, k)
        // shape; only the kernel + weight layout swap.
        let attn_kgamma_t = layer.q_proj_t.is_some()
            && layer.k_proj_t.is_some()
            && layer.v_proj_t.is_some()
            && layer.o_proj_t.is_some()
            && self.kernels.w4a16_gemm_t_m16.0 != 0;
        if attn_kgamma_t {
            let q_t = layer.q_proj_t.as_ref().unwrap();
            kp!(
                q_proj_us,
                (if self.kernels.w4a16_gemm_t_m32_n64.0 != 0 {
                    ops::w4a16_gemm_n64_m32
                } else {
                    ops::w4a16_gemm_n128_m16
                })(
                    gpu,
                    if self.kernels.w4a16_gemm_t_m32_n64.0 != 0 {
                        self.kernels.w4a16_gemm_t_m32_n64
                    } else {
                        self.kernels.w4a16_gemm_t_m16
                    },
                    self.scratch.norm_buf.offset(row0_h),
                    q_t,
                    self.scratch.q_buf.offset(row0_q),
                    m_rows,
                    q_dim,
                    h,
                    stream,
                )
            )?;
        } else {
            kp!(
                q_proj_us,
                ops::w4a16_gemm(
                    gpu,
                    self.kernels.w4a16_gemm,
                    self.scratch.norm_buf.offset(row0_h),
                    &layer.q_proj,
                    self.scratch.q_buf.offset(row0_q),
                    m_rows,
                    q_dim,
                    h,
                    stream,
                )
            )?;
        }
        // ctx rows of q_buf must be ZERO either way: the attention kernel
        // still computes all n_attn query rows, and zero Q keeps ctx rows'
        // (discarded) outputs finite. In noise-only mode the GEMM never
        // touches the ctx region, so the memset fully owns it.
        if eff_ctx > 0 {
            gpu.memset_async(self.scratch.q_buf, 0, eff_ctx * q_dim as usize * bf16, stream)?;
        }

        // 3b'. K/V projections with persistent context cache (NVFP4 path).
        let needed_end = needed_start + eff_ctx;
        let old_ctx_end = needed_end.min(cache_k_end).max(needed_start);
        let old_ctx_count = old_ctx_end.saturating_sub(needed_start);
        let new_ctx_count = eff_ctx.saturating_sub(old_ctx_count);

        if eff_ctx > 0 {
            if old_ctx_count > 0 {
                kp!(
                    kv_ctx_copy_us,
                    self.cache_copy_range(
                        gpu,
                        cache_k,
                        cache_k_start,
                        cache_k_end,
                        window,
                        needed_start,
                        old_ctx_end,
                        self.scratch.k_buf,
                        kv_dim as usize * bf16,
                        stream,
                    )
                )?;
                kp!(
                    kv_ctx_copy_us,
                    self.cache_copy_range(
                        gpu,
                        cache_v,
                        cache_v_start,
                        cache_v_end,
                        window,
                        needed_start,
                        old_ctx_end,
                        self.scratch.v_buf,
                        kv_dim as usize * bf16,
                        stream,
                    )
                )?;
            }
            if new_ctx_count > 0 {
                let fc_offset = old_ctx_count * self.hidden_size * bf16;
                let kv_offset = old_ctx_count * kv_dim as usize * bf16;
                if attn_kgamma_t {
                    let k_t = layer.k_proj_t.as_ref().unwrap();
                    let v_t = layer.v_proj_t.as_ref().unwrap();
                    kp!(
                        kv_ctx_new_us,
                        ops::w4a16_gemm_n128_m16(
                            gpu,
                            self.kernels.w4a16_gemm_t_m16,
                            self.scratch.fc_proj.offset(fc_offset),
                            k_t,
                            self.scratch.k_buf.offset(kv_offset),
                            new_ctx_count as u32,
                            kv_dim,
                            h,
                            stream,
                        )
                    )?;
                    kp!(
                        kv_ctx_new_us,
                        ops::w4a16_gemm_n128_m16(
                            gpu,
                            self.kernels.w4a16_gemm_t_m16,
                            self.scratch.fc_proj.offset(fc_offset),
                            v_t,
                            self.scratch.v_buf.offset(kv_offset),
                            new_ctx_count as u32,
                            kv_dim,
                            h,
                            stream,
                        )
                    )?;
                } else {
                    kp!(
                        kv_ctx_new_us,
                        ops::w4a16_gemm(
                            gpu,
                            self.kernels.w4a16_gemm,
                            self.scratch.fc_proj.offset(fc_offset),
                            &layer.k_proj,
                            self.scratch.k_buf.offset(kv_offset),
                            new_ctx_count as u32,
                            kv_dim,
                            h,
                            stream,
                        )
                    )?;
                    kp!(
                        kv_ctx_new_us,
                        ops::w4a16_gemm(
                            gpu,
                            self.kernels.w4a16_gemm,
                            self.scratch.fc_proj.offset(fc_offset),
                            &layer.v_proj,
                            self.scratch.v_buf.offset(kv_offset),
                            new_ctx_count as u32,
                            kv_dim,
                            h,
                            stream,
                        )
                    )?;
                }
                let (new_k_start, new_k_end) = kp!(
                    cache_write_us,
                    self.cache_write_range(
                        gpu,
                        self.scratch.k_buf.offset(kv_offset),
                        needed_start + old_ctx_count,
                        new_ctx_count,
                        cache_k,
                        cache_k_start,
                        cache_k_end,
                        window,
                        kv_dim as usize * bf16,
                        stream,
                    )
                )?;
                cache_k_start = new_k_start;
                cache_k_end = new_k_end;
                let (new_v_start, new_v_end) = kp!(
                    cache_write_us,
                    self.cache_write_range(
                        gpu,
                        self.scratch.v_buf.offset(kv_offset),
                        needed_start + old_ctx_count,
                        new_ctx_count,
                        cache_v,
                        cache_v_start,
                        cache_v_end,
                        window,
                        kv_dim as usize * bf16,
                        stream,
                    )
                )?;
                cache_v_start = new_v_start;
                cache_v_end = new_v_end;
            }
        }

        // 4. K/V for noise tokens (γ+1 rows: 1 bonus + γ MASKs, vLLM PR #40898).
        let noise_count = n_attn - eff_ctx as u32;
        let noise_offset = eff_ctx * self.hidden_size * bf16;
        let noise_kv_offset = eff_ctx * kv_dim as usize * bf16;
        if attn_kgamma_t {
            let k_t = layer.k_proj_t.as_ref().unwrap();
            let v_t = layer.v_proj_t.as_ref().unwrap();
            kp!(
                kv_noise_us,
                (if self.kernels.w4a16_gemm_t_m32_n64.0 != 0 {
                    ops::w4a16_gemm_n64_m32
                } else {
                    ops::w4a16_gemm_n128_m16
                })(
                    gpu,
                    if self.kernels.w4a16_gemm_t_m32_n64.0 != 0 {
                        self.kernels.w4a16_gemm_t_m32_n64
                    } else {
                        self.kernels.w4a16_gemm_t_m16
                    },
                    self.scratch.norm_buf.offset(noise_offset),
                    k_t,
                    self.scratch.k_buf.offset(noise_kv_offset),
                    noise_count,
                    kv_dim,
                    h,
                    stream,
                )
            )?;
            kp!(
                kv_noise_us,
                (if self.kernels.w4a16_gemm_t_m32_n64.0 != 0 {
                    ops::w4a16_gemm_n64_m32
                } else {
                    ops::w4a16_gemm_n128_m16
                })(
                    gpu,
                    if self.kernels.w4a16_gemm_t_m32_n64.0 != 0 {
                        self.kernels.w4a16_gemm_t_m32_n64
                    } else {
                        self.kernels.w4a16_gemm_t_m16
                    },
                    self.scratch.norm_buf.offset(noise_offset),
                    v_t,
                    self.scratch.v_buf.offset(noise_kv_offset),
                    noise_count,
                    kv_dim,
                    h,
                    stream,
                )
            )?;
        } else {
            kp!(
                kv_noise_us,
                ops::w4a16_gemm(
                    gpu,
                    self.kernels.w4a16_gemm,
                    self.scratch.norm_buf.offset(noise_offset),
                    &layer.k_proj,
                    self.scratch.k_buf.offset(noise_kv_offset),
                    noise_count,
                    kv_dim,
                    h,
                    stream,
                )
            )?;
            kp!(
                kv_noise_us,
                ops::w4a16_gemm(
                    gpu,
                    self.kernels.w4a16_gemm,
                    self.scratch.norm_buf.offset(noise_offset),
                    &layer.v_proj,
                    self.scratch.v_buf.offset(noise_kv_offset),
                    noise_count,
                    kv_dim,
                    h,
                    stream,
                )
            )?;
        }

        // Store updated cache ranges back into proposer state (NVFP4 path).
        dstate.cache_k_start[layer_idx] = cache_k_start;
        dstate.cache_k_end[layer_idx] = cache_k_end;
        dstate.cache_v_start[layer_idx] = cache_v_start;
        dstate.cache_v_end[layer_idx] = cache_v_end;

        // 3c. q_norm / k_norm — RMSNorm reads BF16 weight, ignores quant.
        kp!(
            qk_norm_us,
            ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                self.scratch.q_buf,
                &layer.q_norm,
                self.scratch.q_buf,
                n_attn * self.num_q_heads as u32,
                self.head_dim as u32,
                self.rms_norm_eps,
                stream,
            )
        )?;
        kp!(
            qk_norm_us,
            ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                self.scratch.k_buf,
                &layer.k_norm,
                self.scratch.k_buf,
                n_attn * self.num_kv_heads as u32,
                self.head_dim as u32,
                self.rms_norm_eps,
                stream,
            )
        )?;

        // 3d. yarn RoPE.
        kp!(
            rope_us,
            ops::rope_yarn(
                gpu,
                self.kernels.rope_qwen3,
                self.scratch.q_buf,
                self.scratch.k_buf,
                self.scratch.position_ids,
                n_attn,
                self.num_q_heads as u32,
                self.num_kv_heads as u32,
                self.head_dim as u32,
                self.rotary_dim as u32,
                self.yarn_inv_freq,
                self.rope_theta,
                stream,
            )
        )?;

        // 3e. attention — per-layer SWA window + causal (vLLM PR #40898).
        // ATLAS_DFLASH_SWA=0 force-disables the per-layer window for A/B benchmarks.
        let swa_on = dflash_swa_enabled();
        let layer_window = if swa_on {
            self.layer_window_sizes.get(layer_idx).copied().unwrap_or(0)
        } else {
            0
        };
        let layer_causal_b = if swa_on {
            self.layer_causal.get(layer_idx).copied().unwrap_or(false)
        } else {
            false
        };
        kp!(
            prefill_attn_us,
            ops::prefill_attention(
                gpu,
                self.kernels.prefill_attn,
                self.scratch.q_buf,
                self.scratch.k_buf,
                self.scratch.v_buf,
                self.scratch.attn_out,
                n_attn,
                1,
                self.num_q_heads as u32,
                self.num_kv_heads as u32,
                self.head_dim as u32,
                inv_sqrt_d,
                layer_causal_b,
                layer_window,
                stream,
            )
        )?;
        // Zero context rows in attn_out so garbage uniform scores from zeroed Q
        // don't corrupt the residual stream through o_proj.
        if eff_ctx > 0 {
            gpu.memset_async(self.scratch.attn_out, 0, eff_ctx * q_dim as usize * bf16, stream)?;
        }

        // 3f. o_proj via NVFP4 W4A16 GEMM. N=h, K=q_dim.
        //
        // When ATLAS_DFLASH_ATTN_KGAMMA=1 (transposed o weights present),
        // route through the M_TILE=16 specialization (w4a16_gemm_n128_m16),
        // mirroring q/k/v_proj. Empirical paired KP (seed=42, "Count from
        // 1 to 100", matched n_attn≈257-259):
        //   * M_TILE=64 (w4a16_gemm):           o_proj=10.83ms total=105.9ms
        //   * M_TILE=16 (w4a16_gemm_n128_m16):  o_proj= 3.89ms total=100.6ms
        //   * Kernel-time savings 6.9ms in o_proj per propose
        //   * End-to-end propose savings 5.3ms (~5% of propose)
        // The earlier note that M_TILE=16 collapsed accept to <25% does
        // not reproduce on the current build — likely fixed by the ctx
        // zeroing at line 1102 (the same FFN gate/up at M=528 already
        // run on M_TILE=16 with identical pre-zeroing semantics). Output
        // is coherent and accept rates are comparable to or higher than
        // the M_TILE=64 path across 18+ seed runs. Opt out with
        // ATLAS_DFLASH_ATTN_KGAMMA_DISABLE_O=1 for bisection.
        let disable_o = std::env::var("ATLAS_DFLASH_ATTN_KGAMMA_DISABLE_O")
            .ok()
            .as_deref()
            == Some("1");
        if attn_kgamma_t && !disable_o {
            let o_t = layer.o_proj_t.as_ref().unwrap();
            kp!(
                o_proj_us,
                (if self.kernels.w4a16_gemm_t_m32_n64.0 != 0 {
                    ops::w4a16_gemm_n64_m32
                } else {
                    ops::w4a16_gemm_n128_m16
                })(
                    gpu,
                    if self.kernels.w4a16_gemm_t_m32_n64.0 != 0 {
                        self.kernels.w4a16_gemm_t_m32_n64
                    } else {
                        self.kernels.w4a16_gemm_t_m16
                    },
                    self.scratch.attn_out.offset(row0_q),
                    o_t,
                    self.scratch.stream_acc.offset(row0_h),
                    m_rows,
                    h,
                    q_dim,
                    stream,
                )
            )?;
        } else {
            kp!(
                o_proj_us,
                ops::w4a16_gemm(
                    gpu,
                    self.kernels.w4a16_gemm,
                    self.scratch.attn_out.offset(row0_q),
                    &layer.o_proj,
                    self.scratch.stream_acc.offset(row0_h),
                    m_rows,
                    h,
                    q_dim,
                    stream,
                )
            )?;
        }

        // 3g. residual (DeepLoop β on passes ≥1 with ATLAS_DFLASH_DEEPLOOP=1).
        kp!(
            resid1_us,
            if deeploop {
                ops::scaled_add(gpu, self.kernels.scaled_add,
                    self.scratch.stream_buf.offset(row0_h),
                    self.scratch.stream_acc.offset(row0_h),
                    loop_beta, m_rows * h, stream)
            } else {
                ops::residual_add(gpu, self.kernels.residual_add,
                    self.scratch.stream_buf.offset(row0_h),
                    self.scratch.stream_acc.offset(row0_h),
                    m_rows * h, stream)
            }
        )?;

        // 3h. post_attention_layernorm.
        kp!(
            post_norm_us,
            ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                self.scratch.stream_buf.offset(row0_h),
                &layer.post_attention_layernorm,
                self.scratch.norm_buf.offset(row0_h),
                m_rows,
                h,
                self.rms_norm_eps,
                stream,
            )
        )?;

        // 3i. MLP: gate + up via NVFP4. When the drafter was built with
        // ATLAS_DFLASH_FFN_KGAMMA=1 and the transposed FFN weights are
        // present (gate_proj_t / up_proj_t / down_proj_t all Some), route
        // through the M_TILE=16 specialization (w4a16_gemm_n128_m16)
        // instead of the M_TILE=64 default. At γ=16 the drafter forwards
        // n_attn = ctx_window + γ+1 tokens through a single batched FFN
        // per layer; only the γ+1 noise-block rows actually carry useful
        // signal, but the GEMM operates on the full M=n_attn for layout
        // reasons. M_TILE=64 discards 47/64 = 73% of accumulator writes;
        // the M_TILE=16 variant redesigns warp partitioning so all 4
        // warps share the same 16 rows across N sub-tiles (no waste).
        let ffn_kgamma_t = layer.gate_proj_t.is_some()
            && layer.up_proj_t.is_some()
            && layer.down_proj_t.is_some()
            && self.kernels.w4a16_gemm_t_m16.0 != 0;
        if ffn_kgamma_t {
            let gate_t = layer.gate_proj_t.as_ref().unwrap();
            let up_t = layer.up_proj_t.as_ref().unwrap();
            kp!(
                gate_up_us,
                (if self.kernels.w4a16_gemm_t_m32_n64.0 != 0 {
                    ops::w4a16_gemm_n64_m32
                } else {
                    ops::w4a16_gemm_n128_m16
                })(
                    gpu,
                    if self.kernels.w4a16_gemm_t_m32_n64.0 != 0 {
                        self.kernels.w4a16_gemm_t_m32_n64
                    } else {
                        self.kernels.w4a16_gemm_t_m16
                    },
                    self.scratch.norm_buf.offset(row0_h),
                    gate_t,
                    self.scratch.mlp_intermediate.offset(row0_inter),
                    m_rows,
                    inter,
                    h,
                    stream,
                )
            )?;
            kp!(
                gate_up_us,
                (if self.kernels.w4a16_gemm_t_m32_n64.0 != 0 {
                    ops::w4a16_gemm_n64_m32
                } else {
                    ops::w4a16_gemm_n128_m16
                })(
                    gpu,
                    if self.kernels.w4a16_gemm_t_m32_n64.0 != 0 {
                        self.kernels.w4a16_gemm_t_m32_n64
                    } else {
                        self.kernels.w4a16_gemm_t_m16
                    },
                    self.scratch.norm_buf.offset(row0_h),
                    up_t,
                    self.scratch.mlp_up.offset(row0_inter),
                    m_rows,
                    inter,
                    h,
                    stream,
                )
            )?;
        } else {
            kp!(
                gate_up_us,
                ops::w4a16_gemm(
                    gpu,
                    self.kernels.w4a16_gemm,
                    self.scratch.norm_buf.offset(row0_h),
                    &layer.gate_proj,
                    self.scratch.mlp_intermediate.offset(row0_inter),
                    m_rows,
                    inter,
                    h,
                    stream,
                )
            )?;
            kp!(
                gate_up_us,
                ops::w4a16_gemm(
                    gpu,
                    self.kernels.w4a16_gemm,
                    self.scratch.norm_buf.offset(row0_h),
                    &layer.up_proj,
                    self.scratch.mlp_up.offset(row0_inter),
                    m_rows,
                    inter,
                    h,
                    stream,
                )
            )?;
        }

        // 3j. silu_mul.
        kp!(
            silu_mul_us,
            ops::silu_mul(
                gpu,
                self.kernels.silu_mul,
                self.scratch.mlp_intermediate.offset(row0_inter),
                self.scratch.mlp_up.offset(row0_inter),
                self.scratch.mlp_intermediate.offset(row0_inter),
                m_rows * inter,
                stream,
            )
        )?;

        // 3k. down_proj via NVFP4 — route through M_TILE=16 when
        // ATLAS_DFLASH_FFN_KGAMMA=1 (see gate/up branch above for the
        // rationale on M_TILE=16 vs M_TILE=64).
        if ffn_kgamma_t {
            let down_t = layer.down_proj_t.as_ref().unwrap();
            kp!(
                down_proj_us,
                (if self.kernels.w4a16_gemm_t_m32_n64.0 != 0 {
                    ops::w4a16_gemm_n64_m32
                } else {
                    ops::w4a16_gemm_n128_m16
                })(
                    gpu,
                    if self.kernels.w4a16_gemm_t_m32_n64.0 != 0 {
                        self.kernels.w4a16_gemm_t_m32_n64
                    } else {
                        self.kernels.w4a16_gemm_t_m16
                    },
                    self.scratch.mlp_intermediate.offset(row0_inter),
                    down_t,
                    self.scratch.stream_acc.offset(row0_h),
                    m_rows,
                    h,
                    inter,
                    stream,
                )
            )?;
        } else {
            kp!(
                down_proj_us,
                ops::w4a16_gemm(
                    gpu,
                    self.kernels.w4a16_gemm,
                    self.scratch.mlp_intermediate.offset(row0_inter),
                    &layer.down_proj,
                    self.scratch.stream_acc.offset(row0_h),
                    m_rows,
                    h,
                    inter,
                    stream,
                )
            )?;
        }

        // 3l. residual (DeepLoop β same as 3g).
        kp!(
            resid2_us,
            if deeploop {
                ops::scaled_add(gpu, self.kernels.scaled_add,
                    self.scratch.stream_buf.offset(row0_h),
                    self.scratch.stream_acc.offset(row0_h),
                    loop_beta, m_rows * h, stream)
            } else {
                ops::residual_add(gpu, self.kernels.residual_add,
                    self.scratch.stream_buf.offset(row0_h),
                    self.scratch.stream_acc.offset(row0_h),
                    m_rows * h, stream)
            }
        )?;

        // Debug breadcrumb (cheap when debug_dump=false).
        let _ = (layer_idx, dump_bf16, bf16);

        Ok(())
    }
}
