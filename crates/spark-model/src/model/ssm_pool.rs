// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

/// Pre-allocated contiguous GPU memory pool for SSM layer states.
///
/// Each pool slot has fixed GPU addresses for h_state and conv_state across
/// all SSM layers. This enables CUDA graph capture at batch sizes > 1 because
/// the graph embeds memory addresses that remain stable across replays.
pub(crate) struct SsmStatePool {
    pub(super) h_state_pools: Vec<DevicePtr>,
    pub(super) conv_state_pools: Vec<DevicePtr>,
    /// Per-slot K=3 intermediate checkpoint pools (only allocated when has_mtp).
    /// Layout: `[num_ssm_layers]`, each allocation = max_slots * 3 * h_bytes.
    pub(super) h_intermediate_pools: Vec<DevicePtr>,
    pub(super) conv_intermediate_pools: Vec<DevicePtr>,
    /// Per-slot SSM state checkpoint pools (only allocated when has_mtp).
    pub(super) h_checkpoint_pools: Vec<DevicePtr>,
    pub(super) conv_checkpoint_pools: Vec<DevicePtr>,
    /// WY17 LAZY-commit retention pools (only allocated when has_mtp AND the
    /// lazy-commit gate is on). Each holds one full per-verify k/q/v buffer
    /// (`[K, conv_dim]` BF16 = `kv_retain_bytes`) and gate/beta buffer
    /// (`[K, 2*nv]` FP32 = `gate_retain_bytes`) per slot per layer, so the
    /// commit path can feed the SAME inputs to `gated_delta_rule_wy17_replay`.
    /// Fixed addresses (like the intermediate pools) for CUDA-graph stability.
    pub(super) wy17_kv_retain_pools: Vec<DevicePtr>,
    pub(super) wy17_gate_retain_pools: Vec<DevicePtr>,
    /// Byte size of one retained k/q/v buffer (`K * conv_dim * 2`), 0 if unused.
    pub(super) kv_retain_bytes: usize,
    /// Byte size of one retained gate/beta buffer (`K * 2*nv * 4`), 0 if unused.
    pub(super) gate_retain_bytes: usize,
    pub(super) h_bytes: usize,
    pub(super) conv_bytes: usize,
    /// Number of CLAIMABLE slots (excludes the reserved dummy slot at
    /// index `max_slots`). All claim_slot/release_slot operations work
    /// in `[0, max_slots)`.
    pub(super) max_slots: usize,
    pub(super) num_ssm_layers: usize,
    pub(super) has_mtp: bool,
    pub(super) num_intermediates: usize,
    pub(super) free_slots: Mutex<Vec<usize>>,
}

impl SsmStatePool {
    pub(super) fn new(
        config: &ModelConfig,
        max_slots: usize,
        has_mtp: bool,
        num_intermediates: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let _d_conv = config.linear_conv_kernel_dim;

        let h_bytes = config.ssm_h_state_bytes();
        let conv_bytes = config.ssm_conv_state_bytes();
        let num_ssm_layers = config.num_ssm_layers();

        // Reserve one extra slot at index `max_slots` as a dedicated
        // dummy used by `decode_batch` / `mixed_forward` padding (see
        // `dummy_slot()` below). Without this, pad positions write to
        // pool slot indices `n..padded_n` which can collide with
        // claimed slots if the scheduler invariant ("active sequences
        // occupy contiguous slots [0..n)") is ever broken — silent SSM
        // state corruption. Costs `(h_bytes + conv_bytes) *
        // num_ssm_layers` extra GPU memory (~kilobytes per pool).
        let total_slots = max_slots + 1;

        let mut h_state_pools = Vec::with_capacity(num_ssm_layers);
        let mut conv_state_pools = Vec::with_capacity(num_ssm_layers);
        let mut h_intermediate_pools = Vec::new();
        let mut conv_intermediate_pools = Vec::new();
        let mut h_checkpoint_pools = Vec::new();
        let mut conv_checkpoint_pools = Vec::new();
        let mut wy17_kv_retain_pools = Vec::new();
        let mut wy17_gate_retain_pools = Vec::new();

        // WY17 LAZY-commit retention sizing. Retained buffers replicate the
        // per-verify forward scratch (`conv_out_buf` / `gates_buf`) so the
        // replay kernel can re-derive a skipped intermediate slot. Layout must
        // match the wy17 kernel's `K_TOKENS`-strided reads: k/q/v = `[K,
        // conv_dim]` BF16, gate/beta = `[K, 2*nv]` FP32, where K = the verify
        // window (= num_intermediates). Only allocated when the lazy-commit
        // gate is on to avoid the (small) memory cost on the default path.
        let lazy_commit = crate::layers::wy17_lazy_commit();
        let nk = config.linear_num_key_heads;
        let kd = config.linear_key_head_dim;
        let nv = config.linear_num_value_heads;
        let vd = config.linear_value_head_dim;
        let conv_dim = nk * kd * 2 + nv * vd;
        let k_tokens = num_intermediates; // verify window K = γ+1
        let kv_retain_bytes = if has_mtp && lazy_commit {
            k_tokens * conv_dim * 2 // BF16
        } else {
            0
        };
        let gate_retain_bytes = if has_mtp && lazy_commit {
            k_tokens * (2 * nv) * 4 // FP32
        } else {
            0
        };

        // Predict the pool footprint BEFORE allocating any of it.
        //
        // These pools are the single largest allocation Atlas makes on a
        // hybrid-GDN model — larger than the weights. The size is
        //   num_ssm_layers * total_slots * ni * (h_bytes + conv_bytes)
        // which is linear in BOTH the batch slots and the verify window
        // ni = γ+1. On Qwen3.8-27B (48 GDN layers, γ=16 → ni=33) that measured
        // 5.15 GB *per slot*: 25.2 GB at --max-batch-size 4, and ~85 GB at 16.
        //
        // Until this check existed the size was only logged AFTER the
        // allocations succeeded, so an over-large request was not reported —
        // it just OOMed. On GB10 the memory is unified, so that OOM is a
        // *global* one: it takes down the host, not just this process
        // (observed 2026-08-14, --max-batch-size 16 → hard reboot).
        {
            let per_slot = num_ssm_layers
                * if has_mtp {
                    num_intermediates * (h_bytes + conv_bytes)
                        + h_bytes
                        + conv_bytes
                        + kv_retain_bytes
                        + gate_retain_bytes
                } else {
                    h_bytes + conv_bytes
                };
            let predicted = total_slots.saturating_mul(per_slot);
            let free = gpu.free_memory().unwrap_or(0);
            tracing::info!(
                "SSM pool pre-flight: {} layers × {} slots × (ni={}) = {:.1} GB predicted \
                 ({:.2} GB/slot), {:.1} GB free",
                num_ssm_layers,
                total_slots,
                num_intermediates,
                predicted as f64 / (1024.0 * 1024.0 * 1024.0),
                per_slot as f64 / (1024.0 * 1024.0 * 1024.0),
                free as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            if free > 0 && predicted > free {
                anyhow::bail!(
                    "SSM pools need {:.1} GB but only {:.1} GB is free. This scales \
                     linearly with --max-batch-size ({:.2} GB per slot) and with the \
                     verify window (--dflash-gamma {} → ni={}). Reduce --max-batch-size, \
                     lower --dflash-gamma, or disable MTP/DFlash. \
                     NOTE: on unified-memory parts (GB10) exceeding this global-OOMs the \
                     host rather than failing cleanly, so this is a hard error.",
                    predicted as f64 / (1024.0 * 1024.0 * 1024.0),
                    free as f64 / (1024.0 * 1024.0 * 1024.0),
                    per_slot as f64 / (1024.0 * 1024.0 * 1024.0),
                    num_intermediates.saturating_sub(1),
                    num_intermediates,
                );
            }
        }

        for _ in 0..num_ssm_layers {
            let h_pool = gpu.alloc(total_slots * h_bytes)?;
            gpu.memset(h_pool, 0, total_slots * h_bytes)?;
            h_state_pools.push(h_pool);

            let conv_pool = gpu.alloc(total_slots * conv_bytes)?;
            gpu.memset(conv_pool, 0, total_slots * conv_bytes)?;
            conv_state_pools.push(conv_pool);
        }

        if has_mtp {
            let ni = num_intermediates;
            for _ in 0..num_ssm_layers {
                let h_inter = gpu.alloc(total_slots * ni * h_bytes)?;
                gpu.memset(h_inter, 0, total_slots * ni * h_bytes)?;
                h_intermediate_pools.push(h_inter);

                let conv_inter = gpu.alloc(total_slots * ni * conv_bytes)?;
                gpu.memset(conv_inter, 0, total_slots * ni * conv_bytes)?;
                conv_intermediate_pools.push(conv_inter);

                // 1 checkpoint per slot per layer
                let h_ckpt = gpu.alloc(total_slots * h_bytes)?;
                gpu.memset(h_ckpt, 0, total_slots * h_bytes)?;
                h_checkpoint_pools.push(h_ckpt);

                let conv_ckpt = gpu.alloc(total_slots * conv_bytes)?;
                gpu.memset(conv_ckpt, 0, total_slots * conv_bytes)?;
                conv_checkpoint_pools.push(conv_ckpt);

                if kv_retain_bytes > 0 {
                    let kv_ret = gpu.alloc(total_slots * kv_retain_bytes)?;
                    gpu.memset(kv_ret, 0, total_slots * kv_retain_bytes)?;
                    wy17_kv_retain_pools.push(kv_ret);

                    let gate_ret = gpu.alloc(total_slots * gate_retain_bytes)?;
                    gpu.memset(gate_ret, 0, total_slots * gate_retain_bytes)?;
                    wy17_gate_retain_pools.push(gate_ret);
                }
            }

            if kv_retain_bytes > 0 {
                let retain_mb =
                    num_ssm_layers * total_slots * (kv_retain_bytes + gate_retain_bytes)
                        / (1024 * 1024);
                tracing::info!(
                    "SSM WY17 LAZY-commit retention pools (K={k_tokens}): {retain_mb} MB"
                );
            }

            let mtp_mb = num_ssm_layers
                * total_slots
                * (ni * h_bytes + ni * conv_bytes + h_bytes + conv_bytes)
                / (1024 * 1024);
            tracing::info!("SSM MTP pools ({ni} intermediates + checkpoints): {mtp_mb} MB");
        }

        // free_slots holds claimable indices only; the dummy at index
        // `max_slots` is permanently reserved.
        let free_slots: Vec<usize> = (0..max_slots).rev().collect();

        let total_mb = num_ssm_layers * max_slots * (h_bytes + conv_bytes) / (1024 * 1024);
        tracing::info!(
            "SSM state pool: {max_slots} slots × {num_ssm_layers} layers = {total_mb} MB",
        );

        Ok(Self {
            h_state_pools,
            conv_state_pools,
            h_intermediate_pools,
            conv_intermediate_pools,
            h_checkpoint_pools,
            conv_checkpoint_pools,
            wy17_kv_retain_pools,
            wy17_gate_retain_pools,
            kv_retain_bytes,
            gate_retain_bytes,
            h_bytes,
            conv_bytes,
            max_slots,
            num_ssm_layers,
            has_mtp,
            num_intermediates,
            free_slots: Mutex::new(free_slots),
        })
    }

    pub(super) fn claim_slot(&self) -> Result<usize> {
        self.free_slots.lock().pop().ok_or_else(|| {
            anyhow::anyhow!("SSM state pool exhausted (max {} slots)", self.max_slots)
        })
    }

    pub(super) fn release_slot(&self, idx: usize) {
        self.free_slots.lock().push(idx);
    }

    /// Reserved pool slot used by `decode_batch` / `mixed_forward` padding.
    /// Never claimed by `claim_slot()`, never released. SSM kernels are
    /// free to read/write this slot's pool memory without affecting any
    /// active sequence.
    #[inline]
    pub(super) fn dummy_slot(&self) -> usize {
        self.max_slots
    }

    /// Zero h_state and conv_state for a slot across all SSM layers.
    /// Must be called on slot allocation to prevent stale SSM state
    /// from prior sequences from corrupting new prefill output.
    pub(super) fn zero_slot(&self, idx: usize, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        for i in 0..self.num_ssm_layers {
            gpu.memset_async(self.h_state(i, idx), 0, self.h_bytes, stream)?;
            gpu.memset_async(self.conv_state(i, idx), 0, self.conv_bytes, stream)?;
        }
        Ok(())
    }

    pub(super) fn h_state(&self, ssm_layer_idx: usize, slot: usize) -> DevicePtr {
        self.h_state_pools[ssm_layer_idx].offset(slot * self.h_bytes)
    }

    pub(super) fn conv_state(&self, ssm_layer_idx: usize, slot: usize) -> DevicePtr {
        self.conv_state_pools[ssm_layer_idx].offset(slot * self.conv_bytes)
    }

    /// Get fixed-address intermediate h_state for K=2/3/4 verify.
    /// `token_idx` is 0..3 (which token in the verify pass).
    pub(super) fn h_intermediate(
        &self,
        ssm_layer_idx: usize,
        slot: usize,
        token_idx: usize,
    ) -> DevicePtr {
        let ni = self.num_intermediates;
        self.h_intermediate_pools[ssm_layer_idx].offset((slot * ni + token_idx) * self.h_bytes)
    }

    pub(super) fn conv_intermediate(
        &self,
        ssm_layer_idx: usize,
        slot: usize,
        token_idx: usize,
    ) -> DevicePtr {
        let ni = self.num_intermediates;
        self.conv_intermediate_pools[ssm_layer_idx]
            .offset((slot * ni + token_idx) * self.conv_bytes)
    }

    pub(super) fn h_checkpoint(&self, ssm_layer_idx: usize, slot: usize) -> DevicePtr {
        self.h_checkpoint_pools[ssm_layer_idx].offset(slot * self.h_bytes)
    }

    pub(super) fn conv_checkpoint(&self, ssm_layer_idx: usize, slot: usize) -> DevicePtr {
        self.conv_checkpoint_pools[ssm_layer_idx].offset(slot * self.conv_bytes)
    }

    /// WY17 LAZY-commit k/q/v retention buffer for `(layer, slot)`. `None` when
    /// retention pools weren't allocated (lazy-commit gate off or `!has_mtp`).
    pub(super) fn wy17_kv_retain(&self, ssm_layer_idx: usize, slot: usize) -> Option<DevicePtr> {
        if self.wy17_kv_retain_pools.is_empty() {
            return None;
        }
        Some(self.wy17_kv_retain_pools[ssm_layer_idx].offset(slot * self.kv_retain_bytes))
    }

    /// WY17 LAZY-commit gate/beta retention buffer for `(layer, slot)`.
    pub(super) fn wy17_gate_retain(&self, ssm_layer_idx: usize, slot: usize) -> Option<DevicePtr> {
        if self.wy17_gate_retain_pools.is_empty() {
            return None;
        }
        Some(self.wy17_gate_retain_pools[ssm_layer_idx].offset(slot * self.gate_retain_bytes))
    }

    pub(super) fn reset_slot(&self, slot: usize, gpu: &dyn GpuBackend) -> Result<()> {
        for i in 0..self.num_ssm_layers {
            gpu.memset(self.h_state(i, slot), 0, self.h_bytes)?;
            gpu.memset(self.conv_state(i, slot), 0, self.conv_bytes)?;
            if self.has_mtp {
                for t in 0..self.num_intermediates {
                    gpu.memset(self.h_intermediate(i, slot, t), 0, self.h_bytes)?;
                    gpu.memset(self.conv_intermediate(i, slot, t), 0, self.conv_bytes)?;
                }
                gpu.memset(self.h_checkpoint(i, slot), 0, self.h_bytes)?;
                gpu.memset(self.conv_checkpoint(i, slot), 0, self.conv_bytes)?;
            }
        }
        Ok(())
    }

    pub(super) fn copy_slot(
        &self,
        from: usize,
        to: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        for i in 0..self.num_ssm_layers {
            gpu.copy_d2d_async(
                self.h_state(i, from),
                self.h_state(i, to),
                self.h_bytes,
                stream,
            )?;
            gpu.copy_d2d_async(
                self.conv_state(i, from),
                self.conv_state(i, to),
                self.conv_bytes,
                stream,
            )?;
            if self.has_mtp {
                for t in 0..self.num_intermediates {
                    gpu.copy_d2d_async(
                        self.h_intermediate(i, from, t),
                        self.h_intermediate(i, to, t),
                        self.h_bytes,
                        stream,
                    )?;
                    gpu.copy_d2d_async(
                        self.conv_intermediate(i, from, t),
                        self.conv_intermediate(i, to, t),
                        self.conv_bytes,
                        stream,
                    )?;
                }
                gpu.copy_d2d_async(
                    self.h_checkpoint(i, from),
                    self.h_checkpoint(i, to),
                    self.h_bytes,
                    stream,
                )?;
                gpu.copy_d2d_async(
                    self.conv_checkpoint(i, from),
                    self.conv_checkpoint(i, to),
                    self.conv_bytes,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}
