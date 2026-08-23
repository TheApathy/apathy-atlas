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

/// Per-layer debug/bisection gates, cached.
///
/// These three were read with an uncached `std::env::var` INSIDE
/// `forward_block_layer`, i.e. 3 lookups × 6 layers = 18 per propose, each one
/// taking the global environ lock, linear-scanning ~60 vars and allocating a
/// String — all of it between kernel launches on the propose path, which is
/// NOT CUDA-graph captured and therefore pays its host-side gaps for real.
/// Caching matches how every other gate in this file already behaves.
fn deeploop_beta() -> f32 {
    static CACHED: OnceLock<f32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        // β=1.0 default: pure iterative refinement. DeepLoop's (8N)^{-0.5}
        // paper formula requires a model trained for it.
        std::env::var("ATLAS_DFLASH_DEEPLOOP_BETA")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0)
    })
}

fn dump_all_layers_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| env_flag("ATLAS_DFLASH_DEBUG_DUMP_ALL_LAYERS"))
}

fn attn_kgamma_disable_o() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| env_flag("ATLAS_DFLASH_ATTN_KGAMMA_DISABLE_O"))
}

/// Read one `=1` env flag. Deliberately NOT spelled as the inline
/// `std::env::var(..).ok().as_deref() == Some("1")` chain the call sites used:
/// a scripted rewrite of that chain into these helpers previously matched the
/// helpers' own bodies and turned them into `get_or_init(|| self())`, which
/// deadlocks the first propose. Keeping the read behind one named function
/// means the textual pattern exists exactly once in this file.
fn env_flag(key: &str) -> bool {
    std::env::var(key).ok().as_deref() == Some("1")
}

fn dspark_asymmetric_attention_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let on = std::env::var("ATLAS_DSPARK_ASYMMETRIC_ATTN")
            .ok()
            .as_deref()
            == Some("1");
        if on {
            tracing::info!(
                "DSpark asymmetric attention ENABLED: aligned tail query tiles, full context+draft K/V"
            );
        }
        on
    })
}

fn aligned_tail_query_range(eff_ctx: u32, seq_len: u32, block_rows: u32) -> (u32, u32) {
    debug_assert!(block_rows > 0);
    debug_assert!(eff_ctx <= seq_len);
    let start = eff_ctx - eff_ctx % block_rows;
    (start, seq_len - start)
}

/// DFlash 2 conv slice: the noise rows `[eff_ctx, n_attn)`.
///
/// The reference drafter runs its decoder layers on the γ+1 noise rows only
/// (ctx enters attention purely as K/V from `k_proj(target_hidden)`), and its
/// conv applies the causal zero pad at the block start — i.e. at `eff_ctx`.
/// Both BF16 and NVFP4 paths convolve exactly this slice in place, regardless
/// of noise-only / rectangular-attention row optimizations.
///
/// Returns (byte offset into a [n_attn, hidden] BF16 buffer, row count).
fn conv_noise_range(eff_ctx: usize, n_attn: u32, h: u32) -> (usize, u32) {
    debug_assert!(eff_ctx <= n_attn as usize);
    (eff_ctx * h as usize * 2, n_attn - eff_ctx as u32)
}

impl BlockDiffusionDraftHead {
    /// DFlash 2 conv `prepare` (stage 0): projects the normed noise rows
    /// through `kernel_projection`, convolves them into `conv_out` with the
    /// stage-0 dynamic rows, exports the stage-1 dynamic rows to `conv_dyn1`,
    /// then copies the result back into the norm_buf noise slice. The copy is
    /// required because the causal taps read a previous row of the same
    /// buffer the kernel writes — in-place would race across threads.
    fn dflash2_conv_prepare(
        &self,
        conv: &crate::weight_loader::DflashConvWeights,
        noise_byte_offset: usize,
        noise_count: u32,
        h: u32,
        stream: u64,
        ctx: &ForwardContext,
    ) -> Result<()> {
        use crate::layers::ops;
        let gpu = ctx.gpu;
        let dyn_width = (2 * conv.kernel_size * conv.groups) as u32;
        let noise_bytes = noise_count as usize * h as usize * 2;
        // dynamic = kernel_projection(norm_buf[noise]) → conv_dyn
        ops::dense_gemm_routed(
            gpu,
            self.kernels.dense_gemm,
            self.scratch.norm_buf.offset(noise_byte_offset),
            &conv.kernel_projection,
            self.scratch.conv_dyn,
            noise_count,
            dyn_width,
            h,
            stream,
        )?;
        // stage-0 conv: read norm_buf noise slice → conv_out, + stage-1 export
        ops::dflash2_conv_prepare(
            gpu,
            self.kernels.dflash2_conv_prepare,
            self.scratch.norm_buf.offset(noise_byte_offset),
            self.scratch.conv_dyn,
            conv.base_kernel.weight,
            self.scratch.conv_out,
            self.scratch.conv_dyn1,
            noise_count,
            conv.groups as u32,
            stream,
        )?;
        // conv_out → norm_buf noise slice (stream-ordered D2D)
        gpu.copy_d2d_async(
            self.scratch.norm_buf.offset(noise_byte_offset),
            self.scratch.conv_out,
            noise_bytes,
            stream,
        )?;
        Ok(())
    }

    /// DFlash 2 conv `finish` (stage 1): convolves the sublayer output
    /// (`stream_acc` noise slice, from o_proj or down_proj) into `conv_out`
    /// with the exported stage-1 dynamic rows, then copies it back to the
    /// stream_acc noise slice before the residual add.
    fn dflash2_conv_finish(
        &self,
        conv: &crate::weight_loader::DflashConvWeights,
        noise_byte_offset: usize,
        noise_count: u32,
        stream: u64,
        ctx: &ForwardContext,
    ) -> Result<()> {
        use crate::layers::ops;
        let gpu = ctx.gpu;
        let noise_bytes = noise_count as usize * conv.groups as usize * 16 * 2;
        ops::dflash2_conv_finish(
            gpu,
            self.kernels.dflash2_conv_finish,
            self.scratch.stream_acc.offset(noise_byte_offset),
            self.scratch.conv_dyn1,
            conv.base_kernel.weight,
            self.scratch.conv_out,
            noise_count,
            conv.groups as u32,
            stream,
        )?;
        // conv_out → stream_acc noise slice
        gpu.copy_d2d_async(
            self.scratch.stream_acc.offset(noise_byte_offset),
            self.scratch.conv_out,
            noise_bytes,
            stream,
        )?;
        Ok(())
    }
}

/// DeepLoop residual scaling (arxiv 2607.13491). When enabled and loop_pass ≥ 1,
/// residual updates are scaled by β = (8·N)^{-½} where N = loop_pass.
/// Gate re-exported from the head module for local use.
use super::dflash_deeploop_enabled;

#[cfg(test)]
mod asymmetric_query_range_tests {
    use super::aligned_tail_query_range;

    #[test]
    fn keeps_original_attention_tile_boundaries() {
        assert_eq!(aligned_tail_query_range(512, 526, 32), (512, 14));
        assert_eq!(aligned_tail_query_range(513, 527, 32), (512, 15));
        assert_eq!(aligned_tail_query_range(543, 557, 32), (512, 45));
        assert_eq!(aligned_tail_query_range(0, 14, 32), (0, 14));
    }

    #[test]
    fn retained_cta_starts_match_full_grid_for_every_context_alignment() {
        for eff_ctx in 0..4096u32 {
            let seq_len = eff_ctx + 14;
            let (start, len) = aligned_tail_query_range(eff_ctx, seq_len, 32);
            let control: Vec<u32> = (0..seq_len.div_ceil(32))
                .map(|block| block * 32)
                .filter(|&q_start| q_start >= start)
                .collect();
            let ranged: Vec<u32> = (0..len.div_ceil(32))
                .map(|block| start + block * 32)
                .collect();
            assert_eq!(ranged, control, "eff_ctx={eff_ctx}");
        }
    }
}

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
        let loop_beta: f32 = if deeploop { deeploop_beta() } else { 1.0 };
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
        let dump_all_layers = dump_all_layers_enabled();
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

        // 3a'. DFlash 2 attention-conv `prepare`: convolve the normed noise
        // slice in place (norm_buf) and export the stage-1 dynamic rows.
        // Reference: `Qwen3DFlashDecoderLayer.forward` — the conv wraps the
        // normed input that feeds the q/k/v projections below. Skipped when
        // the checkpoint has no conv (plain DFlash / DSpark).
        let (conv_noise_offset, conv_noise_count) = conv_noise_range(eff_ctx, n_attn, h);
        if let Some(conv) = &layer.attention_conv {
            kp!(
                conv_prepare_us,
                self.dflash2_conv_prepare(
                    conv,
                    conv_noise_offset,
                    conv_noise_count,
                    h,
                    stream,
                    ctx,
                )
            )?;
            dump_layer_bin(
                "d2_attn_conv_out",
                self.scratch.norm_buf.offset(conv_noise_offset),
                conv_noise_count as usize * h as usize,
            )?;
        }

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
            gpu.memset_async(
                self.scratch.q_buf,
                0,
                eff_ctx * q_dim as usize * bf16,
                stream,
            )?;
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
            ops::rope_yarn_scaled(
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
                self.rope_attention_factor,
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
            gpu.memset_async(
                self.scratch.attn_out,
                0,
                eff_ctx * q_dim as usize * bf16,
                stream,
            )?;
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

        // 3f'. DFlash 2 attention-conv `finish`: convolve the o_proj output
        // (stream_acc noise slice) with the stage-1 dynamic rows before the
        // residual add. Reference: `attention_conv.finish(attn_out)` then
        // `hidden = residual + hidden`.
        if let Some(conv) = &layer.attention_conv {
            kp!(
                conv_finish_us,
                self.dflash2_conv_finish(conv, conv_noise_offset, conv_noise_count, stream, ctx)
            )?;
            dump_layer_bin(
                "d2_attn_conv_fin",
                self.scratch.stream_acc.offset(conv_noise_offset),
                conv_noise_count as usize * h as usize,
            )?;
        }

        // 3g. residual: stream_buf += [β·]stream_acc. β=1 on pass 0 (standard);
        // β=(8·N)^{-½} on pass N≥1 with ATLAS_DFLASH_DEEPLOOP=1 (DeepLoop).
        kp!(
            resid1_us,
            if deeploop {
                ops::scaled_add(
                    gpu,
                    self.kernels.scaled_add,
                    self.scratch.stream_buf,
                    self.scratch.stream_acc,
                    loop_beta,
                    n_attn * h,
                    stream,
                )
            } else {
                ops::residual_add(
                    gpu,
                    self.kernels.residual_add,
                    self.scratch.stream_buf,
                    self.scratch.stream_acc,
                    n_attn * h,
                    stream,
                )
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

        // 3h'. DFlash 2 MLP-conv `prepare`: convolve the post-attention-norm
        // noise slice in place (norm_buf) before the MLP projections.
        if let Some(conv) = &layer.mlp_conv {
            kp!(
                conv_prepare_us,
                self.dflash2_conv_prepare(
                    conv,
                    conv_noise_offset,
                    conv_noise_count,
                    h,
                    stream,
                    ctx,
                )
            )?;
            dump_layer_bin(
                "d2_mlp_conv_out",
                self.scratch.norm_buf.offset(conv_noise_offset),
                conv_noise_count as usize * h as usize,
            )?;
        }

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

        // 3k'. DFlash 2 MLP-conv `finish`: convolve the down_proj output
        // (stream_acc noise slice) before the final residual add.
        if let Some(conv) = &layer.mlp_conv {
            kp!(
                conv_finish_us,
                self.dflash2_conv_finish(conv, conv_noise_offset, conv_noise_count, stream, ctx)
            )?;
            dump_layer_bin(
                "d2_mlp_conv_fin",
                self.scratch.stream_acc.offset(conv_noise_offset),
                conv_noise_count as usize * h as usize,
            )?;
        }

        // 3l. residual (DeepLoop β same as 3g).
        kp!(
            resid2_us,
            if deeploop {
                ops::scaled_add(
                    gpu,
                    self.kernels.scaled_add,
                    self.scratch.stream_buf,
                    self.scratch.stream_acc,
                    loop_beta,
                    n_attn * h,
                    stream,
                )
            } else {
                ops::residual_add(
                    gpu,
                    self.kernels.residual_add,
                    self.scratch.stream_buf,
                    self.scratch.stream_acc,
                    n_attn * h,
                    stream,
                )
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
        let loop_beta: f32 = if deeploop { deeploop_beta() } else { 1.0 };
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
        // Rectangular attention: compute attention outputs only for the tail
        // rows [aligned(eff_ctx), n_attn) instead of all n_attn query rows.
        // In noise-only mode the ctx-row attention outputs are dead compute
        // (ctx K/V comes from the persistent cache and the final norm/lm_head
        // reads only the noise slice), and `aligned_tail_query_range` provably
        // starts the retained CTAs on the same 32-row MMA tile boundaries as
        // the square grid, so per-row arithmetic is bit-identical.
        //
        // Originally gated to the Dspark family only; the DFlash-family
        // drafter (v5-ckpt-goheavy, `family=dflash`) uses the same noise-tail
        // row layout and the same kernel, so the gate simply never fired for
        // the production drafter — measured 5.5ms of square ctx-row attention
        // per propose. The env flag now enables it for both families; hash
        // gate is the reference content_sha256 (`12e0c0ad…`).
        let asymmetric_attention = dspark_asymmetric_attention_enabled();
        anyhow::ensure!(
            !asymmetric_attention || noise_only,
            "ATLAS_DSPARK_ASYMMETRIC_ATTN=1 requires ATLAS_DFLASH_NOISE_ONLY=1 so skipped \
             context-query outputs cannot enter q/o/FFN math"
        );
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
        let dump_all_layers = dump_all_layers_enabled();
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

        // 3a'. DFlash 2 attention-conv `prepare` (NVFP4 path). The conv runs
        // on the noise slice [eff_ctx, n_attn) of the n_attn-wide buffers;
        // ctx-row regions are stale but never read downstream (ctx K/V comes
        // from the persistent cache + fc_proj).
        let (conv_noise_offset, conv_noise_count) = conv_noise_range(eff_ctx, n_attn, h);
        if let Some(conv) = &layer.attention_conv {
            // One-shot execution proof. The NVFP4 conv sites carry no kprof
            // label (only the BF16 ones do), so a profile cannot distinguish
            // "ran unwrapped" from "never ran". This settles it.
            {
                static D2_SEEN: std::sync::Once = std::sync::Once::new();
                D2_SEEN.call_once(|| {
                    tracing::info!("DFLASH2_EXEC: nvfp4 attention_conv prepare RAN");
                });
            }
            kp!(
                conv_prepare_us,
                self.dflash2_conv_prepare(
                    conv,
                    conv_noise_offset,
                    conv_noise_count,
                    h,
                    stream,
                    ctx,
                )
            )?;
        }

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
                self.draft_gemm_t(
                    gpu,
                    self.scratch.norm_buf.offset(row0_h),
                    q_t,
                    self.scratch.q_buf.offset(row0_q),
                    m_rows,
                    q_dim,
                    h,
                    true,
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
        // ctx rows of q_buf must be ZERO in square-attention mode: the
        // kernel still computes all n_attn query rows, and zero Q keeps ctx
        // rows' (discarded) outputs finite. In noise-only mode the GEMM
        // never touches the ctx region, so the memset fully owns it. With
        // rectangular attention (`asymmetric_attention`) the ctx query rows
        // are never read at all, so the memset is dead work — skip it.
        if eff_ctx > 0 && !asymmetric_attention {
            gpu.memset_async(
                self.scratch.q_buf,
                0,
                eff_ctx * q_dim as usize * bf16,
                stream,
            )?;
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
                    // These two GEMMs are the same shape class as the kv_noise
                    // pair ~100 lines below, which selects `m32_n64` whenever
                    // the symbol is present because it is "strictly better than
                    // m128 at n <= 32" (single B read AND full SM occupancy —
                    // dense_ffn.rs:2158-2161). kv_ctx_new was left hardcoded on
                    // the m16 path and so never got that win, despite being the
                    // most expensive kernel in propose (3168 us/propose in
                    // benchmark/results/kprof-raw.txt, vs kv_noise's 1153).
                    //
                    // The `<= 32` guard is load-bearing: m32_n64 has M_TILE=32
                    // and does not cover wide M. `noise_count` is gamma+1 and
                    // thus always small, but `new_ctx_count` is the number of
                    // newly captured context rows and CAN exceed 32 (e.g. the
                    // first propose after a prefill), so it must be checked
                    // rather than assumed.
                    //
                    // Drafter-side only: this changes which tokens are
                    // PROPOSED, never which are committed (the target argmaxes
                    // for itself), so completion sha256 must not move.
                    kp!(
                        kv_ctx_new_us,
                        self.draft_gemm_t(
                            gpu,
                            self.scratch.fc_proj.offset(fc_offset),
                            k_t,
                            self.scratch.k_buf.offset(kv_offset),
                            new_ctx_count as u32,
                            kv_dim,
                            h,
                            new_ctx_count <= 32,
                            stream,
                        )
                    )?;
                    kp!(
                        kv_ctx_new_us,
                        self.draft_gemm_t(
                            gpu,
                            self.scratch.fc_proj.offset(fc_offset),
                            v_t,
                            self.scratch.v_buf.offset(kv_offset),
                            new_ctx_count as u32,
                            kv_dim,
                            h,
                            new_ctx_count <= 32,
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
                self.draft_gemm_t(
                    gpu,
                    self.scratch.norm_buf.offset(noise_offset),
                    k_t,
                    self.scratch.k_buf.offset(noise_kv_offset),
                    noise_count,
                    kv_dim,
                    h,
                    true,
                    stream,
                )
            )?;
            kp!(
                kv_noise_us,
                self.draft_gemm_t(
                    gpu,
                    self.scratch.norm_buf.offset(noise_offset),
                    v_t,
                    self.scratch.v_buf.offset(noise_kv_offset),
                    noise_count,
                    kv_dim,
                    h,
                    true,
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
        // With rectangular attention the ctx query rows are never read, so
        // q_norm only needs the noise rows; k_norm must stay full-width
        // (ctx K rows feed the attention KV).
        let q_norm_rows = if asymmetric_attention {
            m_rows * self.num_q_heads as u32
        } else {
            n_attn * self.num_q_heads as u32
        };
        let q_norm_ptr = if asymmetric_attention {
            self.scratch.q_buf.offset(row0_q)
        } else {
            self.scratch.q_buf
        };
        kp!(
            qk_norm_us,
            ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                q_norm_ptr,
                &layer.q_norm,
                q_norm_ptr,
                q_norm_rows,
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
            ops::rope_yarn_scaled(
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
                self.rope_attention_factor,
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
            if asymmetric_attention {
                // Preserve the control path's BR=32 MMA tile boundary. A
                // range starting directly at an unaligned `eff_ctx` moves
                // draft rows to different MMA lanes and perturbs DSpark
                // logits. The partial leading context tile is harmless: its
                // output is discarded by noise-only o/FFN math.
                let (query_start, query_len) = aligned_tail_query_range(eff_ctx as u32, n_attn, 32);
                ops::prefill_attention_query_range(
                    gpu,
                    self.kernels.prefill_attn,
                    self.scratch.q_buf,
                    self.scratch.k_buf,
                    self.scratch.v_buf,
                    self.scratch.attn_out,
                    n_attn,
                    query_start,
                    query_len,
                    1,
                    self.num_q_heads as u32,
                    self.num_kv_heads as u32,
                    self.head_dim as u32,
                    inv_sqrt_d,
                    layer_causal_b,
                    layer_window,
                    stream,
                )
            } else {
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
            }
        )?;
        // Zero context rows in attn_out so garbage uniform scores from zeroed Q
        // don't corrupt the residual stream through o_proj.
        if eff_ctx > 0 {
            gpu.memset_async(
                self.scratch.attn_out,
                0,
                eff_ctx * q_dim as usize * bf16,
                stream,
            )?;
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
        let disable_o = attn_kgamma_disable_o();
        if attn_kgamma_t && !disable_o {
            let o_t = layer.o_proj_t.as_ref().unwrap();
            kp!(
                o_proj_us,
                self.draft_gemm_t(
                    gpu,
                    self.scratch.attn_out.offset(row0_q),
                    o_t,
                    self.scratch.stream_acc.offset(row0_h),
                    m_rows,
                    h,
                    q_dim,
                    true,
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

        // 3f'. DFlash 2 attention-conv `finish` (NVFP4 path): convolve the
        // o_proj output (stream_acc noise slice) before the residual add.
        if let Some(conv) = &layer.attention_conv {
            kp!(
                conv_finish_us,
                self.dflash2_conv_finish(conv, conv_noise_offset, conv_noise_count, stream, ctx)
            )?;
        }

        // 3g. residual (DeepLoop β on passes ≥1 with ATLAS_DFLASH_DEEPLOOP=1).
        kp!(
            resid1_us,
            if deeploop {
                ops::scaled_add(
                    gpu,
                    self.kernels.scaled_add,
                    self.scratch.stream_buf.offset(row0_h),
                    self.scratch.stream_acc.offset(row0_h),
                    loop_beta,
                    m_rows * h,
                    stream,
                )
            } else {
                ops::residual_add(
                    gpu,
                    self.kernels.residual_add,
                    self.scratch.stream_buf.offset(row0_h),
                    self.scratch.stream_acc.offset(row0_h),
                    m_rows * h,
                    stream,
                )
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

        // 3h'. DFlash 2 MLP-conv `prepare` (NVFP4 path): convolve the
        // post-attention-norm noise slice in place before the MLP projections.
        if let Some(conv) = &layer.mlp_conv {
            kp!(
                conv_prepare_us,
                self.dflash2_conv_prepare(
                    conv,
                    conv_noise_offset,
                    conv_noise_count,
                    h,
                    stream,
                    ctx,
                )
            )?;
        }

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
        // FUSED gate_proj + up_proj + SiLU·mul: one launch (A loaded once,
        // both B streams, single [M,N] write) replaces the two m32_n64 GEMMs
        // + the standalone silu_mul. Same geometry as w4a16_gemm_t_m32_n64
        // (M ≤ 32). Drafter-side only — verify still commits the target's own
        // argmax, so output hash is unaffected; acceptance may shift with the
        // fused kernel's register-level SiLU rounding, so this is A/B-gated
        // on the m32_n64 handle being present.
        let fused_gateup = ffn_kgamma_t
            && self.kernels.w4a16_gemm_t_m32_n64_gateup_silu.0 != 0
            && self.kernels.w4a16_gemm_t_m32_n64.0 != 0;
        if fused_gateup {
            let gate_t = layer.gate_proj_t.as_ref().unwrap();
            let up_t = layer.up_proj_t.as_ref().unwrap();
            kp!(
                gate_up_us,
                ops::w4a16_gemm_n64_m32_gateup_silu(
                    gpu,
                    self.kernels.w4a16_gemm_t_m32_n64_gateup_silu,
                    self.scratch.norm_buf.offset(row0_h),
                    gate_t,
                    up_t,
                    self.scratch.mlp_intermediate.offset(row0_inter),
                    m_rows,
                    inter,
                    h,
                    stream,
                )
            )?;
        } else if ffn_kgamma_t {
            let gate_t = layer.gate_proj_t.as_ref().unwrap();
            let up_t = layer.up_proj_t.as_ref().unwrap();
            kp!(
                gate_up_us,
                self.draft_gemm_t(
                    gpu,
                    self.scratch.norm_buf.offset(row0_h),
                    gate_t,
                    self.scratch.mlp_intermediate.offset(row0_inter),
                    m_rows,
                    inter,
                    h,
                    true,
                    stream,
                )
            )?;
            kp!(
                gate_up_us,
                self.draft_gemm_t(
                    gpu,
                    self.scratch.norm_buf.offset(row0_h),
                    up_t,
                    self.scratch.mlp_up.offset(row0_inter),
                    m_rows,
                    inter,
                    h,
                    true,
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

        // 3j. silu_mul — skipped on the fused gateup path (the fused kernel
        // writes silu(gate)*up directly).
        if !fused_gateup {
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
        }

        // 3k. down_proj via NVFP4 — route through M_TILE=16 when
        // ATLAS_DFLASH_FFN_KGAMMA=1 (see gate/up branch above for the
        // rationale on M_TILE=16 vs M_TILE=64).
        if ffn_kgamma_t {
            let down_t = layer.down_proj_t.as_ref().unwrap();
            kp!(
                down_proj_us,
                self.draft_gemm_t(
                    gpu,
                    self.scratch.mlp_intermediate.offset(row0_inter),
                    down_t,
                    self.scratch.stream_acc.offset(row0_h),
                    m_rows,
                    h,
                    inter,
                    true,
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

        // 3k'. DFlash 2 MLP-conv `finish` (NVFP4 path): convolve the
        // down_proj output (stream_acc noise slice) before the final residual.
        if let Some(conv) = &layer.mlp_conv {
            kp!(
                conv_finish_us,
                self.dflash2_conv_finish(conv, conv_noise_offset, conv_noise_count, stream, ctx)
            )?;
        }

        // 3l. residual (DeepLoop β same as 3g).
        kp!(
            resid2_us,
            if deeploop {
                ops::scaled_add(
                    gpu,
                    self.kernels.scaled_add,
                    self.scratch.stream_buf.offset(row0_h),
                    self.scratch.stream_acc.offset(row0_h),
                    loop_beta,
                    m_rows * h,
                    stream,
                )
            } else {
                ops::residual_add(
                    gpu,
                    self.kernels.residual_add,
                    self.scratch.stream_buf.offset(row0_h),
                    self.scratch.stream_acc.offset(row0_h),
                    m_rows * h,
                    stream,
                )
            }
        )?;

        // Debug breadcrumb (cheap when debug_dump=false).
        let _ = (layer_idx, dump_bf16, bf16);

        Ok(())
    }
}

/// Regression guard for the process-wide cached gate helpers in this file.
///
/// A scripted refactor once rewrote two of these helpers into
/// `get_or_init(|| <the helper itself>())`. That compiles, passes every
/// existing test (nothing called them), and clippy only reports it as a
/// "redundant closure" — but at runtime the reentrant `OnceLock::get_or_init`
/// parks the calling thread forever. Symptom: the server loads, answers
/// /v1/models, then the first generation hangs with 0% GPU, 0% CPU and no
/// further log lines.
///
/// Every gate below is therefore CALLED, on a worker thread with a deadline,
/// so a reentrant or otherwise blocking initializer surfaces as a test failure
/// instead of a hung process. Add any new cached gate in this file here.
#[cfg(test)]
mod gate_liveness_tests {
    /// Generous relative to the real work (a handful of `getenv` calls) but
    /// short enough that a deadlock fails the suite quickly rather than
    /// hanging CI until the harness times out.
    const DEADLINE: std::time::Duration = std::time::Duration::from_secs(10);

    /// Every gate assertion runs on a worker thread behind [`DEADLINE`].
    ///
    /// Calling a deadlocked gate directly from the test body would wedge the
    /// harness thread with no timeout — the failure mode would be a hung CI
    /// job rather than a red test. So the checks are a closure run elsewhere,
    /// and the test body only waits for the verdict.
    fn within_deadline<F>(what: &str, checks: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            checks();
            let _ = tx.send(());
        });
        if rx.recv_timeout(DEADLINE).is_err() {
            panic!(
                "{what}: a cached gate in forward_block_layer.rs did not return within \
                 {DEADLINE:?} — almost certainly a OnceLock initializer that re-enters its \
                 own get_or_init (self-recursive helper). That deadlocks the first propose: \
                 the server loads, answers /v1/models, then hangs at 0% GPU and 0% CPU."
            );
        }
    }

    /// Touch every cached gate defined in this module. Values are
    /// environment-dependent and deliberately not asserted here; the property
    /// under test is that each call RETURNS. Add any new cached gate here.
    #[test]
    fn every_cached_gate_terminates_and_is_not_self_recursive() {
        within_deadline("gate liveness", || {
            let _ = super::dflash_swa_enabled();
            let _ = super::dflash_noise_only_enabled();
            let _ = super::dspark_asymmetric_attention_enabled();
            let _ = super::dump_all_layers_enabled();
            let _ = super::attn_kgamma_disable_o();
            // Unset env must yield the documented default, which also proves
            // the initializer ran rather than returning a default-constructed
            // value.
            if std::env::var("ATLAS_DFLASH_DEEPLOOP_BETA").is_err() {
                assert_eq!(super::deeploop_beta(), 1.0, "deeploop_beta default changed");
            }
        });
    }

    /// The second half of the same defect: each gate must agree with the raw
    /// environment, so a rewrite that swaps two helpers' keys is caught too.
    #[test]
    fn cached_gates_match_the_raw_environment() {
        within_deadline("gate/env agreement", || {
            let raw = |k: &str| std::env::var(k).ok().as_deref() == Some("1");
            assert_eq!(
                super::dump_all_layers_enabled(),
                raw("ATLAS_DFLASH_DEBUG_DUMP_ALL_LAYERS"),
                "dump_all_layers_enabled reads the wrong key"
            );
            assert_eq!(
                super::attn_kgamma_disable_o(),
                raw("ATLAS_DFLASH_ATTN_KGAMMA_DISABLE_O"),
                "attn_kgamma_disable_o reads the wrong key"
            );
            assert_eq!(
                super::dflash_noise_only_enabled(),
                raw("ATLAS_DFLASH_NOISE_ONLY"),
                "dflash_noise_only_enabled reads the wrong key"
            );
            assert_eq!(
                super::dspark_asymmetric_attention_enabled(),
                raw("ATLAS_DSPARK_ASYMMETRIC_ATTN"),
                "dspark_asymmetric_attention_enabled reads the wrong key"
            );
        });
    }
}
