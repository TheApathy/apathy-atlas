// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
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
    pub(super) fn start_checkpoint_async_dispatch(&self, seq: &mut SequenceState) -> Result<()> {
        use crate::layer::SsmLayerState;

        let stream = self.secondary_stream;
        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                let nv = self.config.linear_num_value_heads;
                let vd = self.config.linear_value_head_dim;
                let nk = self.config.linear_num_key_heads;
                let kd = self.config.linear_key_head_dim;
                let h_bytes = nv * vd * kd * 4;
                let conv_dim = nk * kd * 2 + nv * vd;
                let d_conv = self.config.linear_conv_kernel_dim;
                let conv_bytes = conv_dim * d_conv * 4;

                if ssm.h_state_checkpoint.is_none() {
                    ssm.h_state_checkpoint = Some(self.gpu.alloc(h_bytes)?);
                }
                if ssm.conv_state_checkpoint.is_none() {
                    ssm.conv_state_checkpoint = Some(self.gpu.alloc(conv_bytes)?);
                }

                self.gpu.copy_d2d_async(
                    ssm.h_state,
                    ssm.h_state_checkpoint.unwrap(),
                    h_bytes,
                    stream,
                )?;
                self.gpu.copy_d2d_async(
                    ssm.conv_state,
                    ssm.conv_state_checkpoint.unwrap(),
                    conv_bytes,
                    stream,
                )?;
            }
        }
        // Record event so default stream can wait (GPU-side, no CPU block).
        self.gpu.record_event(self.secondary_event, stream)?;
        Ok(())
    }

    pub(super) fn start_rollback_and_checkpoint_async_dispatch(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
    ) -> Result<()> {
        use crate::layer::SsmLayerState;

        let stream = self.secondary_stream;
        let mut ssm_layer_idx = 0usize;

        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                let nv = self.config.linear_num_value_heads;
                let vd = self.config.linear_value_head_dim;
                let nk = self.config.linear_num_key_heads;
                let kd = self.config.linear_key_head_dim;
                let h_bytes = nv * vd * kd * 4;
                let conv_dim = nk * kd * 2 + nv * vd;
                let d_conv = self.config.linear_conv_kernel_dim;
                let conv_bytes = conv_dim * d_conv * 4;

                // Rollback: restore h_state and conv_state from the appropriate source.
                if num_accepted == 0 {
                    // No tokens accepted: restore from checkpoint (pre-verify state).
                    if let Some(ckpt) = ssm.h_state_checkpoint {
                        self.gpu
                            .copy_d2d_async(ckpt, ssm.h_state, h_bytes, stream)?;
                    }
                    if let Some(ckpt) = ssm.conv_state_checkpoint {
                        self.gpu
                            .copy_d2d_async(ckpt, ssm.conv_state, conv_bytes, stream)?;
                    }
                } else {
                    // Partial acceptance: restore from intermediate[num_accepted - 1].
                    let slot = seq.slot_idx;
                    let inter_idx = num_accepted - 1;
                    let h_inter = self.ssm_pool.h_intermediate(ssm_layer_idx, slot, inter_idx);
                    let conv_inter =
                        self.ssm_pool
                            .conv_intermediate(ssm_layer_idx, slot, inter_idx);
                    self.gpu
                        .copy_d2d_async(h_inter, ssm.h_state, h_bytes, stream)?;
                    self.gpu
                        .copy_d2d_async(conv_inter, ssm.conv_state, conv_bytes, stream)?;
                }

                // Checkpoint the (now rolled-back) state for the next verify.
                if let Some(ckpt) = ssm.h_state_checkpoint {
                    self.gpu
                        .copy_d2d_async(ssm.h_state, ckpt, h_bytes, stream)?;
                }
                if let Some(ckpt) = ssm.conv_state_checkpoint {
                    self.gpu
                        .copy_d2d_async(ssm.conv_state, ckpt, conv_bytes, stream)?;
                }

                ssm_layer_idx += 1;
            }
        }
        // Record event so default stream can wait (GPU-side, no CPU block).
        self.gpu.record_event(self.secondary_event, stream)?;
        Ok(())
    }

    pub(super) fn sync_secondary_dispatch(&self) -> Result<()> {
        // GPU-side event sync: make the default stream wait for the secondary
        // event. Zero CPU cost — the GPU scheduler handles the dependency.
        self.gpu
            .stream_wait_event(self.gpu.default_stream(), self.secondary_event)
    }

    pub(super) fn pre_verify_copy_async_dispatch(&self, seq: &mut SequenceState) -> Result<()> {
        use crate::layer::SsmLayerState;

        let stream = self.gpu.default_stream();
        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                // No-op if checkpoint isn't populated (non-MTP path).
                let Some(h_ckpt) = ssm.h_state_checkpoint else {
                    continue;
                };
                let Some(conv_ckpt) = ssm.conv_state_checkpoint else {
                    continue;
                };

                let nv = self.config.linear_num_value_heads;
                let vd = self.config.linear_value_head_dim;
                let nk = self.config.linear_num_key_heads;
                let kd = self.config.linear_key_head_dim;
                let h_bytes = nv * vd * kd * 4;
                let conv_dim = nk * kd * 2 + nv * vd;
                let d_conv = self.config.linear_conv_kernel_dim;
                let conv_bytes = conv_dim * d_conv * 4;

                // canonical → scratch (live → kernel input/output).
                self.gpu
                    .copy_d2d_async(h_ckpt, ssm.h_state, h_bytes, stream)?;
                self.gpu
                    .copy_d2d_async(conv_ckpt, ssm.conv_state, conv_bytes, stream)?;
            }
        }
        Ok(())
    }

    pub(super) fn commit_verify_state_async_dispatch(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        k: usize,
        last_inter_slot: usize,
    ) -> Result<()> {
        use crate::layer::SsmLayerState;

        // Task #34: consume the flat-chain injection flag FIRST (even on the
        // full-reject early return below) so a stale `true` can never leak
        // into an unrelated later commit (e.g. an MTP K=2/3/4 verify).
        let flat_tree_injected = self
            .dflash_flat_tree_route
            .swap(false, std::sync::atomic::Ordering::Acquire);

        if num_accepted == 0 {
            // Full reject: canonical state untouched — no commit needed.
            // Still record the event so sync_secondary has something to wait
            // on (defensive: ensures pre-verify ordering on next iteration).
            self.gpu
                .record_event(self.secondary_event, self.secondary_stream)?;
            return Ok(());
        }

        // M8A: if the just-finished verify ran the tree-aware GDN kernel, the
        // kernel only wrote h_state_intermediates[t] per token and LEFT h_state
        // untouched. The full-accept fast path below (line 236) assumes wy_k
        // semantics where h_state was updated in-place to post-K state — wrong
        // for tree mode. Force the partial-accept copy path (copy intermediate
        // [last_inter_slot] → h_state) regardless of num_accepted when tree
        // mode was active.
        //
        // Task #34: "tree mode" has TWO producers, and both must be visible:
        //   * the scheduler-set payload stash (`ddtree_parent_ids_dev`), and
        //   * the verify's own graph-safe FLAT-CHAIN injection
        //     (`dflash_flat_tree_route`, set by verify_d.rs when
        //     `k == ddtree_parent_ids_capacity` with graphs on and
        //     ATLAS_DISABLE_TREE_WY unset).
        // Missing the second one made every FULL accept on an injected verify
        // (e.g. K=12 / γ=11 DSpark) commit the STALE pre-verify h_state.
        let was_tree_mode = super::commit_plan::commit_sees_tree_mode(
            self.ddtree_parent_ids_dev.lock().is_some(),
            flat_tree_injected,
        );

        // Defensive bounds check: `last_inter_slot` must be a valid pool slot.
        // The intermediate pool was sized to `num_intermediates = γ+1 = k` so
        // valid slots are `0..k`. Out-of-bounds indicates a caller bug — fail
        // loudly rather than corrupting an unrelated sequence's state pool.
        if last_inter_slot >= self.ssm_pool.num_intermediates {
            bail!(
                "commit_verify_state_async: last_inter_slot={} OOB (num_intermediates={}, k={}, num_accepted={}, tree_mode={})",
                last_inter_slot,
                self.ssm_pool.num_intermediates,
                k,
                num_accepted,
                was_tree_mode,
            );
        }

        // WY17 LAZY commit: reconstruct a skipped intermediate slot via the
        // replay kernel instead of the intermediate → h_state D2D copy. Active
        // only when the gate is on, lazy J>1, and the tree path was NOT used
        // (tree writes all slots; the flat wy17 path is the only lazy producer).
        //
        // Task #34 sibling hazard: the env gates alone are NOT sufficient — a
        // K≠17 verify runs the CHUNKED wy4/wy3/wy2 path, which persists ALL
        // intermediate slots and never populates the k/v/gate/beta retention
        // buffers the replay kernel reads. The per-layer
        // `wy17_lazy_engaged(k)` check below (shared with the dispatch)
        // guarantees replay fires only when the lazy wy17 kernel actually
        // produced this verify's intermediates.
        let lazy_j = crate::layers::wy17_lazy();
        let lazy_commit_env = crate::layers::wy17_lazy_commit();

        // ATLAS_DFLASH_ASYNC_PROBE=1: measure the TRUE GPU duration of this
        // commit tail (enqueue + secondary-stream drain), not just the CPU
        // enqueue time STEP_TIMING reports. Measurement-only.
        let probe = crate::layers::dflash_async_probe();
        let t_probe = if probe {
            Some(std::time::Instant::now())
        } else {
            None
        };

        let stream = self.secondary_stream;
        let mut ssm_layer_idx = 0usize;

        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                // Snapshot the replay kernel handle + retention pointers BEFORE
                // taking the &mut borrow of the layer state (avoids borrowing
                // `self.layers` while `seq.layer_states` is mutably borrowed).
                let replay_kernel = self.layers[i].wy17_replay_kernel();
                // Did THIS layer's dispatch run the lazy wy17 kernel for a
                // k-token verify? Pure function of k + env + kernel handles
                // (graph-replay safe); shared with the dispatch. False for
                // every K≠17 (chunked/fused) verify — replay must not fire
                // there (task #34 sibling hazard).
                let lazy_engaged = self.layers[i].wy17_lazy_engaged(k);

                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                let Some(h_ckpt) = ssm.h_state_checkpoint else {
                    ssm_layer_idx += 1;
                    continue;
                };
                let Some(conv_ckpt) = ssm.conv_state_checkpoint else {
                    ssm_layer_idx += 1;
                    continue;
                };

                let nv = self.config.linear_num_value_heads;
                let vd = self.config.linear_value_head_dim;
                let nk = self.config.linear_num_key_heads;
                let kd = self.config.linear_key_head_dim;
                let h_bytes = nv * vd * kd * 4;
                let conv_dim = nk * kd * 2 + nv * vd;
                let d_conv = self.config.linear_conv_kernel_dim;
                let conv_bytes = conv_dim * d_conv * 4;

                if num_accepted == k && !was_tree_mode {
                    // Full accept (wy_k path): h_state already holds state-
                    // after-K, which is the canonical post-step state. Mirror
                    // into the checkpoint so a future rollback (if any) has a
                    // valid restore point.
                    self.gpu
                        .copy_d2d_async(ssm.h_state, h_ckpt, h_bytes, stream)?;
                    self.gpu
                        .copy_d2d_async(ssm.conv_state, conv_ckpt, conv_bytes, stream)?;
                } else {
                    // Partial accept (or any tree-mode accept): h_state holds
                    // state-after-K (includes rejected drafts) — WRONG for the
                    // next forward. The kernel-slot semantics are:
                    //   slot 0     = post-root (= previously-committed bonus)
                    //   slot k     = post-compact-index-k for k ∈ [1, T-1]
                    // Chain mode: `last_inter_slot = num_accepted - 1`.
                    // Tree mode: `last_inter_slot = accepted_compact_indices
                    //             .last()` — may be > num_accepted-1 when the
                    // accepted path crosses tree forks (non-contiguous compact
                    // indices, e.g. [1, 4, 7]). Reading
                    // `inter[num_accepted - 1]` instead would silently grab
                    // an unrelated branch's state and produce gibberish.
                    let slot = seq.slot_idx;
                    let conv_inter =
                        self.ssm_pool
                            .conv_intermediate(ssm_layer_idx, slot, last_inter_slot);

                    // ── WY17 LAZY commit: replay a skipped H slot ──
                    // Under lazy J>1 the wy17 kernel persisted only checkpoint
                    // H intermediates. If this partial accept targets a
                    // NON-checkpoint slot, `h_intermediate[last_inter_slot]`
                    // holds a STALE H — reconstruct the true H-after-slot via
                    // `gated_delta_rule_wy17_replay`, re-seeding from the
                    // pre-verify ROOT (== h_ckpt) and replaying the SAME FP32
                    // WY recurrence over tokens [0..=last_inter_slot] with the
                    // retained k/v/gate/beta inputs. Bit-exact vs the state the
                    // kernel would have written. Conv intermediates are ALWAYS
                    // persisted (lazy only skips H), so conv still uses D2D.
                    let use_replay = super::commit_plan::wy17_replay_allowed(
                        lazy_commit_env,
                        lazy_j,
                        was_tree_mode,
                        num_accepted,
                        k,
                        last_inter_slot,
                        lazy_engaged,
                    ) && replay_kernel.0 != 0
                        && ssm.wy17_kv_retain.is_some()
                        && ssm.wy17_gate_retain.is_some();

                    if use_replay {
                        // Retained forward inputs (this layer's snapshot).
                        let kv_ret = ssm.wy17_kv_retain.unwrap();
                        let gate_ret = ssm.wy17_gate_retain.unwrap();
                        let bf16 = 2usize;
                        let fp32 = 4usize;
                        let key_dim = nk * kd;
                        let q_ptr = kv_ret;
                        let k_ptr = kv_ret.offset(key_dim * bf16);
                        let v_ptr = kv_ret.offset(key_dim * 2 * bf16);
                        let gate_ptr = gate_ret;
                        let beta_ptr = gate_ret.offset(nv * fp32);
                        // out_h = live h_state (read by next forward).
                        crate::layers::ops::gdn_wy17_replay(
                            self.gpu.as_ref(),
                            replay_kernel,
                            h_ckpt, // h_root == pre-verify ROOT state
                            q_ptr,
                            k_ptr,
                            v_ptr,
                            gate_ptr,
                            beta_ptr,
                            ssm.h_state, // out_h
                            0,           // ckpt_first_token (root replay)
                            last_inter_slot as u32,
                            1, // batch_size
                            nk as u32,
                            nv as u32,
                            kd as u32,
                            vd as u32,
                            conv_dim as u32, // qk_stride
                            conv_dim as u32, // v_stride
                            (nv * 2) as u32, // gb_stride
                            stream,
                        )?;
                        // Mirror reconstructed H into the checkpoint for any
                        // future rollback (parity with the D2D path below).
                        self.gpu
                            .copy_d2d_async(ssm.h_state, h_ckpt, h_bytes, stream)?;
                        // Conv is always persisted → plain D2D.
                        self.gpu
                            .copy_d2d_async(conv_inter, ssm.conv_state, conv_bytes, stream)?;
                        self.gpu
                            .copy_d2d_async(conv_inter, conv_ckpt, conv_bytes, stream)?;
                    } else {
                        let h_inter =
                            self.ssm_pool
                                .h_intermediate(ssm_layer_idx, slot, last_inter_slot);
                        // canonical → live (h_state read by next forward)
                        self.gpu
                            .copy_d2d_async(h_inter, ssm.h_state, h_bytes, stream)?;
                        self.gpu
                            .copy_d2d_async(conv_inter, ssm.conv_state, conv_bytes, stream)?;
                        // canonical → checkpoint (for any future rollback)
                        self.gpu.copy_d2d_async(h_inter, h_ckpt, h_bytes, stream)?;
                        self.gpu
                            .copy_d2d_async(conv_inter, conv_ckpt, conv_bytes, stream)?;
                    }
                }

                ssm_layer_idx += 1;
            }
        }

        self.gpu.record_event(self.secondary_event, stream)?;
        if let Some(t0) = t_probe {
            let enqueue_us = t0.elapsed().as_micros();
            self.gpu.synchronize(stream)?;
            let total_us = t0.elapsed().as_micros();
            tracing::info!(
                "ASYNC_PROBE commit_tail: enqueue={enqueue_us}μs gpu_total={total_us}μs \
                 (num_accepted={num_accepted} k={k})",
            );
        }
        Ok(())
    }
}
