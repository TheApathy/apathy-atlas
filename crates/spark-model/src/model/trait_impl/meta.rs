// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, HostToDeviceCopy, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::super::ssm_pool::SsmStatePool;
use super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(super) fn vocab_size_dispatch(&self) -> usize {
        self.config.vocab_size
    }

    pub(super) fn high_speed_swap_dims_dispatch(&self) -> Option<spark_storage::ModelDims> {
        // Only attention models have a meaningful sense of K/V blocks; SSM-
        // only models would need a different orchestrator. We expose dims
        // unconditionally and let the scheduler decide whether to install,
        // gated by the user's --high-speed-swap CLI choice.
        Some(spark_storage::ModelDims {
            num_layers: self.config.num_hidden_layers as u32,
            max_blocks_per_layer: self.max_blocks_per_seq,
            num_q_heads: self.config.num_attention_heads as u16,
            num_kv_heads: self.config.num_key_value_heads as u16,
            head_dim: self.config.head_dim as u16,
            block_size: self.kv_cache.lock().block_size() as u16,
        })
    }

    pub(super) fn normalize_ssm_states_dispatch(
        &self,
        seq: &SequenceState,
        stream: u64,
    ) -> Result<()> {
        use spark_runtime::kernel_args::KernelLaunch;

        let num_ssm = self.ssm_pool.num_ssm_layers;
        if num_ssm == 0 || self.ssm_state_norm_kernel.0 == 0 {
            return Ok(());
        }
        let slot = seq.slot_idx;

        // Build pointer table: [layer_0_h_state, layer_1_h_state, ...]
        let ptrs: Vec<u64> = (0..num_ssm)
            .map(|i| self.ssm_pool.h_state(i, slot).0)
            .collect();
        let ptr_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(ptrs.as_ptr() as *const u8, ptrs.len() * 8) };
        self.gpu.copy_h2d_group_on_stream(
            &[HostToDeviceCopy::new(ptr_bytes, self.ssm_norm_ptrs_buf)],
            stream,
        )?;

        let (num_heads, k_dim, v_dim) = self.config.ssm_state_norm_dims();

        KernelLaunch::new(self.gpu.as_ref(), self.ssm_state_norm_kernel)
            .grid([num_heads as u32, num_ssm as u32, 1])
            .block([v_dim as u32, 1, 1])
            .arg_ptr(self.ssm_norm_ptrs_buf)
            .arg_u32(num_heads as u32)
            .arg_u32(k_dim as u32)
            .arg_u32(v_dim as u32)
            .launch(stream)?;

        Ok(())
    }

    /// Flip this sequence's SSM h-state slots from FP32 to FP16
    /// (`ATLAS_SSM_H_FP16`, stage 2). No-op unless the flag is set.
    ///
    /// TWO THINGS HERE ARE LOAD-BEARING AND BOTH FAIL SILENTLY IF CHANGED.
    ///
    /// 1. It must run OUTSIDE CUDA-graph capture. A conversion launched from
    ///    inside a layer is captured into the graph and replayed on every
    ///    later step, re-reading already-FP16 state as FP32. That does not
    ///    crash — an FP32 bit pattern read as two halves is a plausible
    ///    number — it produces fluent, degenerate output. Hence a method on
    ///    the model, called at decode entry, never from a layer.
    ///
    /// 2. The stream is the BACKEND's, not the caller's. On a caller stream
    ///    the conversion is unordered against the decode kernels that read the
    ///    state, and two sequences can interleave through the same scratch.
    ///    Upstream measured NaN h-states on a concurrency-dependent subset
    ///    (7/16 at C=16, clean at C<=8).
    ///
    /// The conversion is a narrowing compaction and cannot be done in place —
    /// thread `2i`'s write lands inside thread `i`'s read — so it stages
    /// through a one-layer scratch and copies back.
    /// Is this sequence's SSM h-state currently stored FP16?
    ///
    /// Read from the sequence's own layer state rather than inferred from the
    /// env flag: a slot that has only PREFILLED is still FP32 even with
    /// `ATLAS_SSM_H_FP16=1`, and the snapshot save path must widen only what is
    /// actually narrow.
    pub(crate) fn seq_h_is_f16(seq: &SequenceState) -> bool {
        seq.layer_states.iter().any(|ls| {
            ls.as_any()
                .downcast_ref::<SsmLayerState>()
                .is_some_and(|s| s.h_is_f16)
        })
    }

    /// Widen one layer's FP16 h-state into an FP32 scratch buffer.
    ///
    /// Returns the scratch pointer, valid until the next call. Callers that
    /// need the bytes must consume them before widening another layer — the
    /// disk-swap path does, copying D2H immediately.
    ///
    /// Exists so the swap FILE stays FP32 and therefore dtype-agnostic; see the
    /// call site in `save_sequence_state_dispatch`.
    pub(crate) fn widen_h_to_f32_scratch(
        &self,
        gpu: &dyn GpuBackend,
        src: DevicePtr,
    ) -> Result<DevicePtr> {
        if self.ssm_h_f16_to_f32_kernel.0 == 0 {
            bail!(
                "ATLAS_SSM_H_FP16: ssm_h_dtype::ssm_h_state_f16_to_f32 did not resolve on this \
                 target — refusing to serialize a half-width h-state into an FP32 swap file."
            );
        }
        let h_bytes = self.ssm_pool.h_bytes;
        let dst = match self.ssm_h_f32_scratch.get() {
            Some(p) => *p,
            None => {
                let p = gpu.alloc(h_bytes)?;
                let _ = self.ssm_h_f32_scratch.set(p);
                p
            }
        };
        let stream = gpu.default_stream();
        crate::layers::ops::ssm_h_state_f16_to_f32(
            gpu,
            self.ssm_h_f16_to_f32_kernel,
            src,
            dst,
            (h_bytes / 4) as u64,
            stream,
        )?;
        gpu.synchronize(stream)?;
        Ok(dst)
    }

    pub(crate) fn ssm_h_to_f16_dispatch(&self, seq: &mut SequenceState) -> Result<()> {
        if !crate::layers::qwen3_ssm::ssm_h_fp16_enabled() || self.ssm_pool.num_ssm_layers == 0 {
            return Ok(());
        }
        let pending = seq.layer_states.iter().any(|ls| {
            ls.as_any()
                .downcast_ref::<SsmLayerState>()
                .is_some_and(|s| !s.h_is_f16)
        });
        if !pending {
            return Ok(());
        }
        if self.ssm_h_f32_to_f16_kernel.0 == 0 {
            bail!(
                "ATLAS_SSM_H_FP16: ssm_h_dtype::ssm_h_state_f32_to_f16 did not resolve on this \
                 target — refusing to run FP16 decode kernels over an FP32 pool."
            );
        }
        let stream = self.gpu.default_stream();
        let h_bytes = self.ssm_pool.h_bytes;
        let f16_bytes = h_bytes / 2;
        let scratch = match self.ssm_h_f16_scratch.get() {
            Some(p) => *p,
            None => {
                let p = self.gpu.alloc(f16_bytes)?;
                let _ = self.ssm_h_f16_scratch.set(p);
                p
            }
        };
        for ls in seq.layer_states.iter_mut() {
            let Some(st) = ls.as_any_mut().downcast_mut::<SsmLayerState>() else {
                continue;
            };
            if st.h_is_f16 {
                continue;
            }
            crate::layers::ops::ssm_h_state_f32_to_f16(
                self.gpu.as_ref(),
                self.ssm_h_f32_to_f16_kernel,
                st.h_state,
                scratch,
                (h_bytes / 4) as u64,
                stream,
            )?;
            self.gpu
                .copy_d2d_async(scratch, st.h_state, f16_bytes, stream)?;
            st.h_is_f16 = true;
        }
        Ok(())
    }

    pub(super) fn bind_gpu_to_thread_dispatch(&self) -> Result<()> {
        self.gpu.bind_to_thread()
    }

    pub(super) fn alloc_sequence_dispatch(&self) -> Result<SequenceState> {
        let alloc_t0 = std::time::Instant::now();
        let mut alloc_ms: Vec<(&str, f64)> = Vec::new();
        let alloc_on = std::env::var("ATLAS_PREFILL_PHASE_PROFILE").ok().as_deref() == Some("1");
        macro_rules! alloc_mark {
            ($name:expr) => {
                if alloc_on {
                    alloc_ms.push(($name, alloc_t0.elapsed().as_secs_f64() * 1000.0));
                }
            };
        }
        let slot = self.ssm_pool.claim_slot()?;
        alloc_mark!("claim_slot");
        // Zero SSM state to prevent stale h_state/conv_state from prior
        // sequences corrupting the recurrent computation during prefill.
        // CRITICAL: use Atlas's own stream (not stream 0) because Atlas's stream
        // is CU_STREAM_NON_BLOCKING and does NOT synchronize with stream 0.
        // Using stream 0 would race with the subsequent prefill kernel.
        let stream = self.gpu.default_stream();
        self.ssm_pool.zero_slot(slot, self.gpu.as_ref(), stream)?;
        alloc_mark!("zero_slot");
        // Ensure zero completes before any prefill kernels touch this slot.
        self.gpu.synchronize(stream)?;
        alloc_mark!("zero_slot_sync");
        // Capability, not arm selection: the SSM checkpoint/intermediate
        // buffers must exist if EITHER arm can speculate on this sequence.
        let has_mtp = self.any_proposer() || self.self_speculative;

        // Build layer states: SSM layers point into the pool (fixed addresses),
        // attention layers use their own alloc_state (EmptyLayerState).
        // When MTP is available, pre-allocate checkpoint + K=2 intermediate
        // buffers so CUDA graph capture doesn't trigger lazy allocation.
        let mut ssm_layer_idx = 0usize;
        let mut layer_states: Vec<Box<dyn LayerState>> = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let mut ssm_state = SsmLayerState {
                    h_state: self.ssm_pool.h_state(ssm_layer_idx, slot),
                    conv_state: self.ssm_pool.conv_state(ssm_layer_idx, slot),
                    h_state_checkpoint: None,
                    conv_state_checkpoint: None,
                    h_state_intermediates: Vec::new(),
                    conv_state_intermediates: Vec::new(),
                    wy17_kv_retain: None,
                    wy17_gate_retain: None,
                    h_is_f16: false,
                };

                if has_mtp {
                    // Use pool-based fixed addresses (stable across sequence
                    // lifetimes → CUDA graph can replay without stale pointers).
                    ssm_state.h_state_checkpoint =
                        Some(self.ssm_pool.h_checkpoint(ssm_layer_idx, slot));
                    ssm_state.conv_state_checkpoint =
                        Some(self.ssm_pool.conv_checkpoint(ssm_layer_idx, slot));

                    for t in 0..self.ssm_pool.num_intermediates {
                        ssm_state
                            .h_state_intermediates
                            .push(self.ssm_pool.h_intermediate(ssm_layer_idx, slot, t));
                        ssm_state
                            .conv_state_intermediates
                            .push(self.ssm_pool.conv_intermediate(ssm_layer_idx, slot, t));
                    }

                    // WY17 LAZY-commit retention (None unless the pools were
                    // allocated, i.e. ATLAS_WY17_LAZY_COMMIT=1).
                    ssm_state.wy17_kv_retain = self.ssm_pool.wy17_kv_retain(ssm_layer_idx, slot);
                    ssm_state.wy17_gate_retain =
                        self.ssm_pool.wy17_gate_retain(ssm_layer_idx, slot);
                }

                layer_states.push(Box::new(ssm_state));
                ssm_layer_idx += 1;
            } else {
                layer_states.push(layer.alloc_state(self.gpu.as_ref())?);
            }
        }

        // Zero SSM states for the new sequence.
        // Synchronous reset: memset + stream sync ensures zero is visible
        // before any subsequent kernel reads the state.
        self.ssm_pool.reset_slot(slot, self.gpu.as_ref())?;
        alloc_mark!("reset_slot");
        // Double-check: explicit sync to guarantee zero is complete
        self.gpu.synchronize(self.gpu.default_stream())?;
        alloc_mark!("reset_slot_sync");

        // Allocate proposer state (owns its own KV cache block table) for
        // EVERY installed arm, not just the live one: an arm switch mid-run
        // moves `proposer_state_alt` into `proposer_state`, and allocating
        // the second arm's state lazily at that moment would put a device
        // allocation on the switch path — inside the scheduler's step loop,
        // where it can fail and where the cost would be misattributed to the
        // arm being switched TO.
        //
        // On single-proposer builds `proposer_alt` is None, so this allocates
        // exactly what it always did and `state_alt` stays None.
        let state_primary = match &self.proposer {
            Some(p) => Some(p.alloc_state(self.gpu.as_ref())?),
            None => None,
        };
        alloc_mark!("proposer_state");
        let state_secondary = match &self.proposer_alt {
            Some(p) => Some(p.alloc_state(self.gpu.as_ref())?),
            None => None,
        };
        alloc_mark!("proposer_state_alt");
        // Maintain the invariant that `proposer_state` belongs to the LIVE
        // arm: a sequence admitted while the gate sits on arm 1 must come up
        // with arm 1's state in the active slot.
        let (proposer_state, proposer_state_alt) = if self.proposer_arm() == 1 {
            (state_secondary, state_primary)
        } else {
            (state_primary, state_secondary)
        };

        // No graph invalidation needed — pool addresses are stable across sequences.

        // Phase 6.1.d critical fix: pre-size disk_last_offloaded_per_layer
        // to the model's attention-layer count. The vector stays empty
        // when HSS isn't engaged (cache_blocks_per_seq is None) — the
        // helper short-circuits before reading from it. Sized here once
        // so the layer-0 offload helper doesn't need to grow a Vec on
        // every sequence's first decode step.
        let num_attn_layers = self.config.num_attention_layers();
        alloc_mark!("build_state");
        if alloc_on {
            let mut joined: Vec<String> = Vec::with_capacity(alloc_ms.len());
            let mut prev = 0.0f64;
            for (name, t) in &alloc_ms {
                joined.push(format!("{name}={:.1}", t - prev));
                prev = *t;
            }
            tracing::info!(
                "ALLOC_SEQUENCE | total={:.1} | {}",
                alloc_t0.elapsed().as_secs_f64() * 1000.0,
                joined.join(" ")
            );
        }
        Ok(SequenceState {
            tokens: Vec::new(),
            block_table: Vec::new(),
            seq_len: 0,
            layer_states,
            proposer_state,
            proposer_state_alt,
            slot_idx: slot,
            marconi_skip_to: 0,
            session_hash: 0,
            chunked_prefill_meta: None,
            cached_prefix_tokens: 0,
            prompt_len: 0,
            disk_block_ids: Vec::new(),
            mtp_lastk_host_buf: Vec::new(),
            mtp_lastk_host_filled: 0,
            mtp_lastk_end_abs: 0,
            disk_last_offloaded_per_layer: vec![0; num_attn_layers],
        })
    }

    pub(super) fn copy_logits_to_host_dispatch(
        &self,
        logits_ptr: DevicePtr,
        dst: &mut [u8],
    ) -> Result<()> {
        self.gpu.copy_d2h(logits_ptr, dst)
    }

    pub(super) fn logits_ptr_is_fp32_dispatch(&self, logits_ptr: DevicePtr) -> bool {
        self.use_fp32_logits && logits_ptr.0 == self.logits_fp32_buf.0
    }

    pub(super) fn logits_buffer_ptr_dispatch(&self) -> DevicePtr {
        self.buffers.logits()
    }

    pub(super) fn argmax_on_device_dispatch(
        &self,
        logits_ptr: DevicePtr,
        _stream: u64,
    ) -> Result<u32> {
        // Use backend's default stream (same as decode) to avoid implicit
        // sync overhead from legacy default stream (handle 0).
        let stream = self.gpu.default_stream();
        // Use first 4 bytes of scratch buffer for the u32 output
        let out_ptr = self.buffers.scratch();
        // Dispatch by buffer dtype: when the logits pointer is the model's
        // FP32 scratch (single-token decode lm_head with use_fp32_logits),
        // run argmax_fp32; otherwise the buffer is BF16 (prefill /
        // batched-decode / non-Gemma-4 paths) and argmax_bf16 applies.
        // The kernel arg layout is identical (ptr, ptr, u32), so dispatch
        // is just a kernel-handle swap.
        let is_fp32 = self.use_fp32_logits && logits_ptr.0 == self.logits_fp32_buf.0;
        let kernel = if is_fp32 {
            self.argmax_logits_kernel
        } else {
            self.argmax_kernel
        };
        ops::argmax_bf16(
            self.gpu.as_ref(),
            kernel,
            logits_ptr,
            out_ptr,
            self.config.vocab_size as u32,
            stream,
        )?;
        // D2H: copy 4 bytes (single u32) instead of vocab_size*2 = 304KB
        let mut buf = [0u8; 4];
        self.gpu.copy_d2h(out_ptr, &mut buf)?;
        let gpu_token = u32::from_le_bytes(buf);

        // ── ATLAS_DUMP_HIDDEN: append emitted-token record ──
        // Pairs with the 5 hidden-state records dumped during the layer loop.
        // Record format: u32 magic | u32 token_id | u32 0 | u32 0 (16 bytes).
        // env::var lookup cached via OnceLock helper (was a per-token syscall).
        if let Some(path) = crate::model::env_diag::dump_hidden_path() {
            const TOKEN_DUMP_MAGIC: u32 = 0xA71B5DEE;
            use std::io::Write;
            tracing::trace!("ATLAS_DUMP_HIDDEN: argmax_dispatch → token {}", gpu_token);
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                let _ = f.write_all(&TOKEN_DUMP_MAGIC.to_le_bytes());
                let _ = f.write_all(&gpu_token.to_le_bytes());
                let _ = f.write_all(&0u32.to_le_bytes());
                let _ = f.write_all(&0u32.to_le_bytes());
            }
        }

        Ok(gpu_token)
    }

    pub(super) fn argmax_batch_dispatch(
        &self,
        logits_ptr: DevicePtr,
        n: usize,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        let stream = self.gpu.default_stream();
        let v = self.config.vocab_size;
        let bf16 = 2usize;
        let _fp32 = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        let out_ptr = self.buffers.scratch();
        for i in 0..n {
            let logits_i = logits_ptr.offset(i * v * bf16);
            let out_i = out_ptr.offset(i * 4);
            ops::argmax_bf16(
                self.gpu.as_ref(),
                self.argmax_kernel,
                logits_i,
                out_i,
                v as u32,
                stream,
            )?;
        }
        let mut buf = vec![0u8; n * 4];
        self.gpu.copy_d2h(out_ptr, &mut buf)?;
        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            results.push(u32::from_le_bytes([
                buf[i * 4],
                buf[i * 4 + 1],
                buf[i * 4 + 2],
                buf[i * 4 + 3],
            ]));
        }

        // ── ATLAS_DUMP_HIDDEN: append emitted-token records (batch path) ──
        // Pairs with the per-decode-step hidden-state records.
        // env::var lookup cached via OnceLock helper.
        if let Some(path) = crate::model::env_diag::dump_hidden_path() {
            const TOKEN_DUMP_MAGIC: u32 = 0xA71B5DEE;
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                for &tok in &results {
                    let _ = f.write_all(&TOKEN_DUMP_MAGIC.to_le_bytes());
                    let _ = f.write_all(&tok.to_le_bytes());
                    let _ = f.write_all(&0u32.to_le_bytes());
                    let _ = f.write_all(&0u32.to_le_bytes());
                }
            }
        }

        Ok(results)
    }

    pub(super) fn hidden_after_norm_dispatch(&self) -> DevicePtr {
        // norm_output() holds the post-final-norm hidden state from the last decode
        self.buffers.norm_output()
    }
}
