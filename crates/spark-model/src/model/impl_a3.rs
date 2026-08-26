// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use spark_runtime::kv_cache::PagedKvCache;

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

/// Verify-side LM-head vocab truncation cap (`ATLAS_TARGET_LMHEAD_VOCAB`).
///
/// The TARGET model's verify `lm_head` GEMV computes logits over the FULL
/// 248320-row vocab for every spec-decode verify step (k≈2 rows/step). The
/// argmax that picks the verified token only needs the top scoring token,
/// and BPE places frequent tokens at low IDs — so reading only the first N
/// weight rows makes the GEMV proportionally cheaper (it's memory-bound on
/// the NVFP4 weight read at 273 GB/s) with negligible quality risk for N
/// large enough to cover normal text.
///
/// 0 (default) = full vocab (no truncation). Any value ≥ vocab_size is also
/// treated as full. Only the batched-verify and single-token DECODE argmax
/// paths honor this — PREFILL keeps the full vocab so the first token stays
/// exact.
pub(super) fn target_lmhead_vocab() -> u32 {
    static CACHE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("ATLAS_TARGET_LMHEAD_VOCAB")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(0)
    })
}

/// `ATLAS_LM_HEAD_TC=1` routes the target verify lm_head (M=2..=32) through
/// the transposed-NVFP4 m32_n64 tensor-core kernel (`w4a16_gemm_t_m32_n64`)
/// instead of the byte-exact scalar-FMA family. This is the same kernel the
/// DFlash drafter already uses for its propose head (`draft_lm_head_nvfp4_t`),
/// so the weight layout and MMA rounding are proven coherent; the target
/// committed token may differ from the scalar oracle by MMA-vs-FMA rounding
/// (a re-reference in the same class as `ATLAS_FFN_TC=1`). Default OFF: the
/// exact scalar path remains the commit authority until this is qualified.
pub(super) fn lm_head_tc_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("ATLAS_LM_HEAD_TC").ok().as_deref() == Some("1"))
}

fn log_lm_head_tc_engagement(rows: u32, vocab: u32, hidden: u32) {
    static SEEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !SEEN.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::info!(
            target: "atlas::lm_head",
            route = "tc_m32_n64",
            rows,
            vocab,
            hidden,
            output_dtype = "bf16",
            output_layout = "row_major",
            "LM_HEAD_TC_ENGAGEMENT"
        );
    }
}

fn log_exact_lm_head_engagement(route: ops::ExactLmHeadRoute, rows: u32, vocab: u32, hidden: u32) {
    static SEEN: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
    let tier_bit = match route.tier() {
        ops::ExactLmHeadTier::M4 => 0,
        ops::ExactLmHeadTier::M8 => 1,
        ops::ExactLmHeadTier::M17 => 2,
        ops::ExactLmHeadTier::M32 => 3,
    };
    let route_bit = match route {
        ops::ExactLmHeadRoute::Exact(_) => tier_bit,
        ops::ExactLmHeadRoute::SerialK1(_) => tier_bit + 4,
    };
    let mask = 1u8 << route_bit;
    if SEEN.fetch_or(mask, std::sync::atomic::Ordering::Relaxed) & mask == 0 {
        tracing::info!(
            target: "atlas::lm_head",
            route = route.provenance(),
            tier = route.tier().label(),
            rows,
            vocab,
            hidden,
            output_dtype = "bf16",
            output_layout = "row_major",
            "LM_HEAD_EXACT_ENGAGEMENT"
        );
    }
}

impl TransformerModel {
    /// Collapse Qwen4's four persistent streams to the core hidden width.
    /// Conventional architectures retain their terminal RMSNorm path at the
    /// call site.
    pub(super) fn qwen4_final_hidden(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        let Some(mixer) = &self.qwen4_final_mixer else {
            return Ok(None);
        };
        let (mixed, inject) = mixer.prepare_decode(
            hidden,
            residual,
            &self.buffers,
            self.gpu.as_ref(),
            self.config.rms_norm_eps as f32,
            stream,
        )?;
        debug_assert!(inject.is_none());
        Ok(Some(mixed))
    }

    pub(super) fn embed(&self, token: u32, output: DevicePtr, stream: u64) -> Result<()> {
        let h = self.config.hidden_size;
        let row_bytes = h * 2; // BF16 embedding row
        let src = self.embed_tokens.weight.offset(token as usize * row_bytes);
        if self.config.is_qwen4_exp() {
            let scratch = self.buffers.norm_output();
            self.gpu.copy_d2d_async(src, scratch, row_bytes, stream)?;
            self.expand_qwen4_embedding(scratch, output, stream)?;
            return Ok(());
        }
        if self.bf16_to_f32_kernel.0 != 0 {
            // FP32 residual: embed BF16 to scratch, convert to FP32 output.
            // The scratch buffer is norm_output which is BF16 regardless of
            // residual dtype — use the BF16 scaler explicitly.
            let scratch = self.buffers.norm_output();
            self.gpu.copy_d2d_async(src, scratch, row_bytes, stream)?;
            self.scale_embeddings_bf16(scratch, 1, stream)?;
            crate::layers::ops::bf16_to_f32(
                self.gpu.as_ref(),
                self.bf16_to_f32_kernel,
                scratch,
                output,
                h as u32,
                stream,
            )
        } else {
            self.gpu.copy_d2d_async(src, output, row_bytes, stream)?;
            // Scale embeddings (Gemma-4: sqrt(hidden_size))
            self.scale_embeddings(output, 1, stream)
        }
    }

    /// Expand one BF16 `[hidden_size]` embedding into Qwen4's
    /// `[hc_count, hidden_size]` residual-stream layout.
    pub(super) fn expand_qwen4_embedding(
        &self,
        input: DevicePtr,
        output: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let kernel = self
            .gpu
            .kernel("qwen4_hyper", "qwen4_hc_expand_embedding")?;
        KernelLaunch::new(self.gpu.as_ref(), kernel)
            .grid([1, div_ceil(self.config.residual_width() as u32, 256), 1])
            .block([256, 1, 1])
            .arg_ptr(input)
            .arg_ptr(output)
            .arg_u32(self.config.hidden_size as u32)
            .arg_u32(self.config.hc_count as u32)
            .launch(stream)
    }

    /// Scale in-place embeddings by config.embed_scale. Picks the kernel
    /// matching `data`'s actual dtype:
    ///   - when `use_fp32_residual()` is true, `hidden` is FP32 and we
    ///     dispatch `embed_scale::f32_scale_inplace`
    ///   - otherwise (`hidden` is BF16) we dispatch the usual
    ///     `embed_scale::bf16_scale_inplace`
    ///
    /// For the rare case of scaling a BF16 buffer while FP32 residual is
    /// ALSO active (e.g. the decode embed() scratch which is deliberately
    /// BF16 before a bf16_to_f32 cast), use `scale_embeddings_bf16`.
    pub(super) fn scale_embeddings(
        &self,
        data: DevicePtr,
        num_tokens: usize,
        stream: u64,
    ) -> Result<()> {
        if self.config.use_fp32_residual() {
            self.scale_embeddings_fp32(data, num_tokens, stream)
        } else {
            self.scale_embeddings_bf16(data, num_tokens, stream)
        }
    }

    pub(super) fn scale_embeddings_bf16(
        &self,
        data: DevicePtr,
        num_tokens: usize,
        stream: u64,
    ) -> Result<()> {
        if self.embed_scale_kernel.0 == 0 {
            return Ok(());
        }
        use spark_runtime::kernel_args::KernelLaunch;
        let n = (num_tokens * self.config.hidden_size) as u32;
        KernelLaunch::new(self.gpu.as_ref(), self.embed_scale_kernel)
            .grid([n.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(data)
            .arg_u32(n)
            .arg_f32(self.config.embed_scale)
            .launch(stream)
    }

    pub(super) fn scale_embeddings_fp32(
        &self,
        data: DevicePtr,
        num_tokens: usize,
        stream: u64,
    ) -> Result<()> {
        // Symmetric with scale_embeddings_bf16: models without embedding
        // scaling (non-Gemma, e.g. qwen3.6-27b) have no embed_scale kernel
        // registered (handle == 0). Without this guard the FP8 fp32-residual
        // path hard-fails ("Module 'embed_scale' not loaded").
        if self.embed_scale_kernel.0 == 0 {
            return Ok(());
        }
        use spark_runtime::kernel_args::KernelLaunch;
        let kernel = self.gpu.kernel("embed_scale", "f32_scale_inplace")?;
        let n = (num_tokens * self.config.hidden_size) as u32;
        KernelLaunch::new(self.gpu.as_ref(), kernel)
            .grid([n.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(data)
            .arg_u32(n)
            .arg_f32(self.config.embed_scale)
            .launch(stream)
    }

    /// LM head for K tokens: hidden[K, H] → logits[K, V].
    /// Effective verify-side vocab for the batched-decode/verify `lm_head`
    /// argmax: the full vocab, or the `ATLAS_TARGET_LMHEAD_VOCAB` cap when it
    /// is set and smaller. The scheduler's verify argmax MUST read this (not
    /// `config.vocab_size`) so the per-row logits stride and the argmax range
    /// match what `lm_head_batched` actually wrote. Returns the full vocab in
    /// every non-truncated case (cap==0 or cap≥vocab).
    pub(super) fn verify_lmhead_vocab(&self) -> u32 {
        let v = self.config.vocab_size as u32;
        let cap = target_lmhead_vocab();
        if cap == 0 || cap >= v { v } else { cap }
    }

    pub(super) fn lm_head_batched(
        &self,
        hidden: DevicePtr,
        num_tokens: u32,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = self.config.hidden_size as u32;
        // Verify-side vocab truncation: shrink the logical GEMM/GEMV output
        // dimension (and the matching argmax range in the scheduler) to the
        // first `v` rows. BPE places frequent tokens at low IDs, so reading
        // fewer vocab rows is a clean bandwidth reduction with negligible
        // quality risk for `v` large enough to cover normal text.
        // `verify_lmhead_vocab()` returns the full vocab when truncation is
        // disabled (`ATLAS_TARGET_LMHEAD_VOCAB` unset / 0 / ≥ vocab).
        let v = self.verify_lmhead_vocab();
        let logits = self.buffers.logits();
        if let Some(ref nvfp4) = self.lm_head_nvfp4 {
            anyhow::ensure!(
                (1..=32).contains(&num_tokens),
                "NVFP4 speculative LM-head rows must be in 1..=32, got {num_tokens}"
            );
            if num_tokens == 1 {
                // A one-row tail (for example, a chunked diagnostic verify)
                // stays on the ordinary qualified K1 LM-head path.
                ops::w4a16_gemv(
                    self.gpu.as_ref(),
                    self.w4a16_gemv_kernel,
                    hidden,
                    nvfp4,
                    logits,
                    v,
                    h,
                    stream,
                )?;
            } else if lm_head_tc_enabled()
                && self.lm_head_nvfp4_t.is_some()
                && self.w4a16_gemm_t_m32_n64_kernel.0 != 0
            {
                // ATLAS_LM_HEAD_TC=1: tensor-core m32_n64 route over the
                // transposed NVFP4 weight — the same kernel + layout the
                // DFlash drafter uses for its propose head, so the MMA
                // rounding and packing are proven coherent. ldb is the
                // 64-padded vocab stride (== vocab for 248320). Output stays
                // row-major [M, v]. Re-reference class: ATLAS_FFN_TC=1.
                let ldb = (self.config.vocab_size.div_ceil(64) * 64) as u32;
                log_lm_head_tc_engagement(num_tokens, v, h);
                ops::w4a16_gemm_n64_m32_ldb(
                    self.gpu.as_ref(),
                    self.w4a16_gemm_t_m32_n64_kernel,
                    hidden,
                    self.lm_head_nvfp4_t.as_ref().expect("checked above"),
                    logits,
                    num_tokens,
                    v,
                    h,
                    ldb,
                    stream,
                )?;
            } else {
                let route = self
                    .w4a16_exact_lm_head_kernels
                    .route_for_rows(num_tokens)
                    .expect("NVFP4 M=2..=32 has an exact tier");
                log_exact_lm_head_engagement(route, num_tokens, v, h);
                match route {
                    ops::ExactLmHeadRoute::Exact(_) => {
                        ops::w4a16_gemv_batch_logits_exact(
                            self.gpu.as_ref(),
                            self.w4a16_exact_lm_head_kernels,
                            hidden,
                            nvfp4,
                            logits,
                            num_tokens,
                            v,
                            h,
                            stream,
                        )?;
                    }
                    ops::ExactLmHeadRoute::SerialK1(_) => {
                        // Missing exact symbols fail closed to M independent
                        // ordinary K1 GEMVs. Row-major output uses the logical
                        // (possibly truncated) vocab as its stride.
                        for row in 0..num_tokens {
                            ops::w4a16_gemv(
                                self.gpu.as_ref(),
                                self.w4a16_gemv_kernel,
                                hidden.offset(row as usize * h as usize * 2),
                                nvfp4,
                                logits.offset(row as usize * v as usize * 2),
                                v,
                                h,
                                stream,
                            )?;
                        }
                    }
                }
            }
        } else if num_tokens == 2 {
            // Preserve dense M=2 as two BF16 GEMVs. The FP32 logits path is
            // decode-only and does not apply to batched verification.
            ops::dense_gemv(
                self.gpu.as_ref(),
                self.dense_gemv_kernel,
                hidden,
                &self.lm_head_weight,
                logits,
                v,
                h,
                stream,
            )?;
            ops::dense_gemv(
                self.gpu.as_ref(),
                self.dense_gemv_kernel,
                hidden.offset(h as usize * 2),
                &self.lm_head_weight,
                logits.offset(v as usize * 2),
                v,
                h,
                stream,
            )?;
        } else {
            ops::dense_gemm(
                self.gpu.as_ref(),
                self.dense_gemm_kernel,
                hidden,
                &self.lm_head_weight,
                logits,
                num_tokens,
                v,
                h,
                stream,
            )?;
        }
        // Apply logit softcapping: logits = cap * tanh(logits / cap)
        if self.logit_softcap_kernel.0 != 0 {
            let cap = self.config.final_logit_softcapping;
            let total = num_tokens * v;
            self.apply_logit_softcap(logits, total, cap, stream)?;
        }
        Ok(logits)
    }

    pub(super) fn lm_head(&self, hidden: DevicePtr, stream: u64) -> Result<DevicePtr> {
        let h = self.config.hidden_size as u32;
        let v = self.config.vocab_size as u32;
        // Pick the output buffer: FP32 scratch when use_fp32_logits is on,
        // shared BF16 buffer otherwise. The sampler must use the matching
        // dtype — see `decode_logits_dtype()`.
        let (logits, fp32) = if self.use_fp32_logits {
            (self.logits_fp32_buf, true)
        } else {
            (self.buffers.logits(), false)
        };
        if let Some(ref nvfp4) = self.lm_head_nvfp4 {
            // Pick FP32-output variant when the FP32 logits buffer is the
            // destination. Same packed-NVFP4 weights, same activation, but the
            // accumulator is NOT downcast to BF16 — closes the 0.125-logit
            // BF16-rounding tiebreak flip that triggers Gemma-4-31B's
            // creative-collapse stop-word loop.
            let kernel = if fp32 {
                self.w4a16_gemv_logits_kernel
            } else {
                self.w4a16_gemv_kernel
            };
            ops::w4a16_gemv(
                self.gpu.as_ref(),
                kernel,
                hidden,
                nvfp4,
                logits,
                v,
                h,
                stream,
            )?;
        } else if fp32 {
            // FP32-output dense GEMV: same precision-preservation reason as
            // the NVFP4 variant above. Used when Gemma keeps the LM head
            // as BF16 (skip_lm_head_quantization=true).
            ops::dense_gemv(
                self.gpu.as_ref(),
                self.dense_gemv_fp32out_kernel,
                hidden,
                &self.lm_head_weight,
                logits,
                v,
                h,
                stream,
            )?;
        } else {
            ops::dense_gemv(
                self.gpu.as_ref(),
                self.dense_gemv_kernel,
                hidden,
                &self.lm_head_weight,
                logits,
                v,
                h,
                stream,
            )?;
        }
        // Apply logit softcapping: logits = cap * tanh(logits / cap)
        if self.logit_softcap_kernel.0 != 0 || self.logit_softcap_fp32_kernel.0 != 0 {
            let cap = self.config.final_logit_softcapping;
            self.apply_logit_softcap_dtype(logits, v, cap, fp32, stream)?;
        }
        Ok(logits)
    }

    /// Apply logit softcapping in-place: `logits[i] = cap * tanh(logits[i] / cap)`.
    /// BF16 path. Use `apply_logit_softcap_dtype` to dispatch by buffer dtype.
    pub(super) fn apply_logit_softcap(
        &self,
        logits: DevicePtr,
        num_elements: u32,
        cap: f32,
        stream: u64,
    ) -> Result<()> {
        use spark_runtime::kernel_args::KernelLaunch;
        let inv_cap = 1.0f32 / cap;
        KernelLaunch::new(self.gpu.as_ref(), self.logit_softcap_kernel)
            .grid([num_elements.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(logits)
            .arg_u32(num_elements)
            .arg_f32(inv_cap)
            .arg_f32(cap)
            .launch(stream)
    }

    /// Dtype-aware softcap dispatcher. Picks the BF16 or FP32 kernel based on
    /// whether the buffer holds FP32 logits. No-op when softcap is disabled
    /// (cap == 0). Used by the single-token decode `lm_head` to keep the FP32
    /// path symmetrical when `use_fp32_logits` is on.
    pub(super) fn apply_logit_softcap_dtype(
        &self,
        logits: DevicePtr,
        num_elements: u32,
        cap: f32,
        is_fp32: bool,
        stream: u64,
    ) -> Result<()> {
        use spark_runtime::kernel_args::KernelLaunch;
        let kernel = if is_fp32 {
            self.logit_softcap_fp32_kernel
        } else {
            self.logit_softcap_kernel
        };
        if kernel.0 == 0 {
            return Ok(());
        }
        let inv_cap = 1.0f32 / cap;
        KernelLaunch::new(self.gpu.as_ref(), kernel)
            .grid([num_elements.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(logits)
            .arg_u32(num_elements)
            .arg_f32(inv_cap)
            .arg_f32(cap)
            .launch(stream)
    }

    /// True when single-token decode `lm_head` writes FP32 logits to
    /// `logits_fp32_buf`. Callers that consume those logits (sampler) MUST
    /// read with the matching dtype. Prefill / batched-decode lm_head still
    /// produce BF16, so this only applies to the `lm_head` (single-token)
    /// return value.
    pub fn decode_logits_fp32(&self) -> bool {
        self.use_fp32_logits
    }

    /// Buffer pointer the single-token decode `lm_head` last wrote to. FP32
    /// scratch when `use_fp32_logits`, otherwise the shared BF16 logits
    /// buffer. Callers that previously hard-coded `self.buffers.logits()`
    /// after `self.lm_head(...)` must use this so the sampler reads the
    /// correct buffer dtype (the BF16 buffer is stale/empty in the FP32
    /// path because lm_head writes elsewhere). Pair with
    /// `logits_ptr_is_fp32` / `decode_logits_fp32` for dtype-aware reads.
    pub fn decode_logits_ptr(&self) -> DevicePtr {
        if self.use_fp32_logits {
            self.logits_fp32_buf
        } else {
            self.buffers.logits()
        }
    }
}
