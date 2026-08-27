// SPDX-License-Identifier: AGPL-3.0-only

//! K=γ (DFlash) verify path.
//!
//! ## Safety
//!
//! `unsafe { from_raw_parts(...) }` blocks reinterpret stack arrays
//! / `Vec`s of POD integers (`u32`, `i32`, `i64`, `usize`) as byte
//! slices for H2D upload. See `verify_c.rs` module docs for the full
//! safety contract — same pattern, same invariants here.

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

// ── ATLAS_LAYER_RESOLUTION_PROBE ────────────────────────────────────────────
// Measures at which layer the K=γ verify argmax stabilises.  Run with
// ATLAS_LAYER_RESOLUTION_PROBE=1 (forces eager mode).  Every
// ATLAS_LRP_DUMP_EVERY steps (default 100) logs per-checkpoint match rates.

fn layer_resolution_probe_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_LAYER_RESOLUTION_PROBE")
            .ok()
            .as_deref()
            == Some("1")
    })
}

fn lrp_dump_every() -> u64 {
    static VAL: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *VAL.get_or_init(|| {
        std::env::var("ATLAS_LRP_DUMP_EVERY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(100)
    })
}

/// 0-indexed layer indices at which hidden state is snapshotted.
/// Covers the full 64-layer depth at ~8-layer resolution.
const LRP_PROBE_LAYERS: &[usize] = &[7, 15, 23, 31, 39, 47, 55];

struct LrpAccum {
    step: u64,
    /// matches[probe_idx][token_pos] = steps where probe argmax == final argmax.
    matches: Vec<Vec<u64>>,
    k: usize,
}

impl LrpAccum {
    fn new(k: usize) -> Self {
        Self {
            step: 0,
            matches: vec![vec![0u64; k]; LRP_PROBE_LAYERS.len()],
            k,
        }
    }

    fn accumulate(&mut self, final_am: &[u32], probe_ams: &[Vec<u32>]) {
        let k = self.k.min(final_am.len());
        self.step += 1;
        for (pi, pam) in probe_ams.iter().enumerate() {
            for t in 0..k {
                if pam.get(t).copied() == Some(final_am[t]) {
                    self.matches[pi][t] += 1;
                }
            }
        }
    }

    fn dump(&self) {
        let steps = self.step.max(1);
        let mut msg = format!("LRP step={} K={}:", self.step, self.k);
        for (pi, &layer_idx) in LRP_PROBE_LAYERS.iter().enumerate() {
            let matched: u64 = self.matches[pi].iter().sum();
            let total = steps * self.k as u64;
            let rate = matched as f64 / total as f64;
            msg.push_str(&format!(" L{}={:.3}", layer_idx + 1, rate));
        }
        msg.push_str(" L64=1.000");
        tracing::info!("{msg}");
    }
}

static LRP_ACCUM: std::sync::OnceLock<Mutex<LrpAccum>> = std::sync::OnceLock::new();
// ── end LRP ─────────────────────────────────────────────────────────────────

/// Static verify-layer-skip set parsed once from `ATLAS_VERIFY_SKIP_LAYERS`
/// (comma-separated layer indices). Empty when unset. See the layer loop in
/// `decode_verify_graphed_kgamma_dispatch` for semantics (identity pass-through,
/// graph-safe, pass@1-gated). The "past-the-dense-read" bandwidth lever.
fn verify_skip_layer(idx: usize) -> bool {
    use std::sync::OnceLock;
    static SET: OnceLock<Vec<usize>> = OnceLock::new();
    SET.get_or_init(|| {
        std::env::var("ATLAS_VERIFY_SKIP_LAYERS")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|x| x.trim().parse::<usize>().ok())
                    .collect()
            })
            .unwrap_or_default()
    })
    .contains(&idx)
}

impl TransformerModel {
    pub(super) fn decode_verify_graphed_kgamma_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        let k = tokens.len();
        if k == 0 {
            return Ok(Vec::new());
        }
        // Keep the prior K=2 verifier as an explicit diagnostic oracle. The
        // production Qwen4 path below now owns K=2 and K=3 with batched MoE.
        if self.config.is_qwen4_exp()
            && k == 2
            && std::env::var("ATLAS_QWEN4_VERIFY_LEGACY_K2")
                .ok()
                .as_deref()
                == Some("1")
        {
            let pair = [tokens[0], tokens[1]];
            return Ok(self
                .decode_verify_graphed_dispatch(&pair, seq, _stream)?
                .to_vec());
        }

        // ATLAS_SSM_H_FP16 stage 2. This entry point does NOT exist upstream —
        // the speculative verify is ours — and it is the one that matters most
        // here, because the WY kernels it dispatches are the h-state readers and
        // writers whenever --speculative is on. Without this hook the FP16 twins
        // selected in `wy_chunk_kernel` would run over an unconverted FP32 pool.
        // Outside the graph capture that begins further down.
        self.ssm_h_to_f16_dispatch(seq)?;
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let persistent_width = self.config.residual_width();
        let bf16 = 2usize;
        let fp32 = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        let serial = crate::model::env_diag::DflashSerialControls::current();
        let serial_family = serial.active_family()?;
        let control_requests = crate::model::control_engagement::ControlRequests {
            attn_paged: crate::layers::attn_paged_serial_enabled(),
            attn_out: crate::layers::attn_out_serial_enabled(),
            ffn: serial.ffn,
            layer_norms: serial.layer_norms,
        };
        let controlled_verify = k > 1 && control_requests.any();
        let stage_capture = crate::model::k1_stage_diag::requested_at(seq.seq_len, tokens)?;
        crate::model::k1_stage_diag::validate_serial_control_overlap(
            crate::model::k1_stage_diag::enabled(),
            serial_family,
        )?;
        if stage_capture {
            if self.use_fp32_logits || self.verify_lmhead_vocab() as usize != self.config.vocab_size
            {
                bail!("DFLASH_K1_STAGE_DIAG requires BF16 full-vocabulary logits");
            }
            crate::model::k1_stage_diag::begin_batch(seq.seq_len, tokens)?;
        }
        if let Some(family) = serial_family {
            static PROOF: std::sync::Once = std::sync::Once::new();
            PROOF.call_once(|| {
                tracing::warn!(family, c = 1, k, "DFLASH_K1_BISECT C1 active");
            });
        }
        let kgamma_debug_dump = std::env::var("ATLAS_KGAMMA_DEBUG_DUMP").is_ok()
            && std::env::var("ATLAS_KGAMMA_DEBUG_SEQ_LEN")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .is_none_or(|wanted| wanted == seq.seq_len);

        // F62 (2026-04-27): SpecMamba dual-buffer pre-verify copy.
        self.pre_verify_copy_async(seq)?;

        // ATLAS_FULL_PROFILE=1: bump per-step counter so per-kernel kprof!
        // accumulators fire on K=γ verify. begin_step() early-returns when
        // env var is unset — see full_profile.rs:103-111 for the safety
        // contract (must NOT flip ACTIVE while CUDA graph capture is live).
        crate::full_profile::begin_step();

        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        // ── Phase 1: Pre-graph (varies per step, NOT captured) ──
        //
        // DFS reorder (option C, ATLAS_DDTREE_DFS_REORDER=1):
        // Permute the K kernel-frame tokens into DFS pre-order so each
        // ancestor chain is contiguous in slot order. The paged-decode
        // attention kernel iterates [0..seq_lens[t]) sequentially — DFS
        // reorder ensures that the deepest ancestor chain (the one
        // greedy_sample_ddtree will walk) reads its TRUE ancestors,
        // not sibling/cousin slots. Sibling root children at the end of
        // the DFS order get short seq_lens and only read the bonus +
        // already-visited subtree — wrong but acceptable because the
        // greedy walker only ever commits one chain.
        //
        // `dfs_perm[i] = j` means "DFS slot i contains kernel slot j".
        // `dfs_inv_perm[j] = i` means "kernel slot j is at DFS slot i".
        // Empty when DFS reorder is disabled or no tree payload is active.
        let dfs_enabled = std::env::var("ATLAS_DDTREE_DFS_REORDER").ok().as_deref() == Some("1");
        let (dfs_perm, dfs_inv_perm, dfs_depths): (Vec<usize>, Vec<usize>, Vec<usize>) = {
            let host_parents_lock = self.ddtree_parent_ids_host.lock();
            // Allow payload shorter than k (we pad with linear-chain tail to
            // match k below). Only require non-empty payload.
            if dfs_enabled && !host_parents_lock.is_empty() && host_parents_lock.len() <= k {
                use crate::layers::dflash_head::ddtree::dfs_reorder;
                // Pad up to k if payload shorter (mirror the linear-chain
                // padding logic from the existing depth derivation).
                let mut hp = host_parents_lock.clone();
                drop(host_parents_lock);
                while hp.len() < k {
                    let next_chain_parent = hp.len() as i32 - 1;
                    hp.push(next_chain_parent);
                }
                let (perm, inv, depths) = dfs_reorder(&hp);
                // If perm is identity (flat chain or trivially-flat payload),
                // skip the reorder machinery — it's a no-op and the existing
                // chain path is already optimal.
                let is_identity = perm.iter().enumerate().all(|(i, &j)| i == j);
                if is_identity {
                    (Vec::new(), Vec::new(), Vec::new())
                } else {
                    static DFS_DBG_DONE: std::sync::atomic::AtomicBool =
                        std::sync::atomic::AtomicBool::new(false);
                    if !DFS_DBG_DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        tracing::info!(
                            "ATLAS_DDTREE_DFS_REORDER: k={k} perm={:?} depths={:?}",
                            perm,
                            depths
                        );
                    }
                    (perm, inv, depths)
                }
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            }
        };
        let dfs_active = !dfs_perm.is_empty();

        // ATLAS_DDTREE_DFS_REORDER_METADATA_ONLY=1: skip the token permutation
        // (embedding stays in linear-drafts order) and only permute attention
        // metadata + parent_ids. Use case: when the verify input `tokens`
        // is the LINEAR top-1 drafts (current Atlas plumbing), not the
        // tree's tree_token_ids. Permuting linear drafts into DFS slots
        // places semantically wrong tokens (e.g. drafts[5] at the chain-child
        // slot of drafts[0]) and the model output becomes incoherent.
        //
        // With METADATA_ONLY=1 we keep token-at-slot consistency for
        // attention (preserves the existing chain accept behavior) while
        // letting depth-based RoPE positions still fire. This is a
        // research-only mode used to A/B the metadata permutation alone.
        let dfs_permute_tokens = std::env::var("ATLAS_DDTREE_DFS_REORDER_METADATA_ONLY")
            .ok()
            .as_deref()
            != Some("1");

        // 1a. Embed K tokens — in DFS order if reorder active AND
        // METADATA_ONLY=0 (default).
        crate::kprof!(self.gpu.as_ref(), stream, "embed", {
            for t in 0..k {
                let src_token = if dfs_active && dfs_permute_tokens {
                    tokens[dfs_perm[t]]
                } else {
                    tokens[t]
                };
                self.embed(
                    src_token,
                    hidden.offset(t * persistent_width * fp32),
                    stream,
                )?;
            }
            anyhow::Result::<()>::Ok(())
        })?;
        self.capture_k1_stage("embed", hidden, k, persistent_width * fp32, stream)?;

        // 1b. Allocate KV blocks for all K positions
        let bs = kv_cache.block_size();
        for t in 0..k {
            let pos = seq.seq_len + t;
            let blocks_needed = (pos / bs) + 1;
            ensure_blocks_through_decode(
                seq,
                blocks_needed - 1,
                &mut kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
            )?;
        }

        // 1c. Upload K-entry attention metadata. Layout in scratch (after
        // mtp metadata reservation): positions[K*4] | slots[K*8] | seq_lens[K*4]
        // | block_table[K*max_blocks*4]. Need K*16 + K*max_blocks*4 bytes per
        // call — at K=17 max_blocks=512 that's ~36 KB which fits comfortably
        // in the scratch arena (offset 32768).
        let meta_base = self.buffers.scratch().offset(32768);
        let max_blocks = self.max_blocks_per_seq;

        // Tree-aware metadata (ATLAS_DDTREE_TREE_AWARE_VERIFY=1):
        //
        // The chain-mode meta below treats the γ+1 verify tokens as a FLAT
        // chain (positions = seq.seq_len + t, seq_lens = seq.seq_len + t + 1).
        // That is wrong for the real top-K tree topology produced by M4B v2:
        //
        //   - Sibling root children (e.g. compact 0..4 with parent=-1) get
        //     DIFFERENT RoPE positions (1, 2, 3, 4, 5) when they are all at
        //     tree depth 1 and should share position 1. RoPE mis-alignment
        //     skews query directions per sibling and degrades the per-position
        //     argmax accuracy.
        //   - The per-row seq_lens upper bound also grows with compact index,
        //     so a sibling at compact index 4 reads positions [seq.seq_len ..
        //     seq.seq_len + 4] = bonus + 4 sibling KVs (not its ancestor chain).
        //
        // The full fix (per-row KV indirection so each query attends only to
        // its ancestors) requires modifying the paged-decode attention kernel
        // — see paged_decode_attn_nvfp4.cu / paged_decode_attn_fp8.cu, which
        // today iterate positions [0..seq_lens[t]) and read in-block offset
        // `pos % bs` with no slot-indirection knob. Until that lands,
        // ATLAS_DDTREE_TREE_AWARE_VERIFY=1 implements a *partial* fix that
        // produces coherent output without changing tok/s vs flat chain:
        //
        //   - positions[t] = seq.seq_len + depth[t]   (depth-based RoPE so
        //     sibling root children share RoPE position 1 instead of 1..N).
        //   - seq_lens[t]  unchanged (still compact-index-based). Setting it
        //     to depth-based caused deep chain queries to skip legitimate
        //     prior chain context and drop accept rate even further.
        //
        // Slots are still written at compact-index physical positions so the
        // KV cache stays consistent for the post-verify commit path. Block-
        // tables are unchanged (the same full sequence row replicated per t).
        //
        // RESULT: B (tree-aware) text == A (flat chain) text on essay prompt
        //         but tok/s does NOT clear A — needs the kernel-level KV
        //         indirection to land for a real win on non-flat topologies.
        //
        // depth[t] is derived from the kernel-frame parent_ids stashed by
        // set_ddtree_parent_ids — index 0 is the bonus (depth 0), index i+1
        // is draft i. parent[i] = -1 means "child of pre-tree state" (= bonus,
        // depth 1); parent[i] = k means depth[t] = 1 + depth[k].
        let tree_aware_enabled = std::env::var("ATLAS_DDTREE_TREE_AWARE_VERIFY")
            .ok()
            .as_deref()
            == Some("1");
        let tree_depths: Option<Vec<usize>> = if tree_aware_enabled {
            // host_parents may be shorter than k when the tree payload has
            // fewer nodes than γ_eff (e.g. budget < γ_eff caps the tree at
            // budget). The persistent device buffer is padded by
            // clear_ddtree_parent_ids' linear-chain default — replicate the
            // same padding here so depth lookups don't OOB.
            let mut host_parents = self.ddtree_parent_ids_host.lock().clone();
            // DDTree budgets MAX_NODES (e.g. 32) but SSM verify window is k
            // (e.g. 17). When len > k the == k guard below was falling through
            // to None, keeping tree_depths always None and preventing the
            // causal_conv1d_tree_reroot kernel from ever firing. Truncate first.
            if host_parents.len() > k {
                host_parents.truncate(k);
            }
            if !host_parents.is_empty() && host_parents.len() < k {
                let last_payload_idx = host_parents.len() as i32 - 1;
                while host_parents.len() < k {
                    let next_chain_parent = host_parents.len() as i32 - 1;
                    let _ = last_payload_idx;
                    host_parents.push(next_chain_parent);
                }
            }
            if host_parents.len() == k && !host_parents.is_empty() {
                // Build kernel-frame depths. Bonus (index 0) is depth 0.
                let mut depths = vec![0usize; k];
                // For index i >= 1: parent < 0 means "attaches to pre-tree
                // state" — treat as a child of the bonus (depth 1). Otherwise
                // parent kernel slot p (0 = bonus) means depth = 1 + depth[p].
                for i in 1..k {
                    let p = host_parents[i];
                    if p < 0 {
                        depths[i] = 1;
                    } else {
                        let pi = p as usize;
                        if pi >= i {
                            // Defensive: malformed payload → fall back to
                            // chain depth so we don't panic. Set to compact
                            // index so downstream stays self-consistent.
                            depths[i] = i;
                        } else {
                            depths[i] = depths[pi].saturating_add(1);
                        }
                    }
                }
                // Refuse to apply when the payload is degenerate (flat chain)
                // — chain mode is already optimal.
                let is_flat_chain = (1..k).all(|i| depths[i] == i);
                if is_flat_chain {
                    None
                } else {
                    static TREE_AWARE_DBG_DONE: std::sync::atomic::AtomicBool =
                        std::sync::atomic::AtomicBool::new(false);
                    if !TREE_AWARE_DBG_DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        tracing::info!("ATLAS_DDTREE_TREE_AWARE_VERIFY: k={k} depths={:?}", depths);
                    }
                    Some(depths)
                }
            } else {
                None
            }
        } else {
            None
        };

        // ATLAS_TREE_AWARE_ATTN: full per-row KV indirection. Activated
        // when the env var is set AND tree mode is producing non-flat
        // depths AND DFS reorder is OFF (DFS already provides contiguous
        // ancestor reads, so the two should not be combined).
        //
        // STATUS (2026-05-21): correctness verified end-to-end (coherent
        // outputs, ancestor chains built correctly per log) but the kernel
        // single-position fallback in the tree window is ~3.6x slower than
        // the batched chain-mode path on the qwen3.6-27b config (K=γ=17,
        // depth up to 14, 16 q_heads × 17 rows = 272 CTAs each walking
        // 1-14 single-position iterations). The original chain path uses
        // BC=4 batched loads which dominate even though tree mode reads
        // fewer total KV positions.
        //
        // Default OFF until the kernel single-position path is batched
        // (e.g. by re-introducing BC=4 over indirected ancestors, which
        // requires same-block packing — non-trivial because ancestor slots
        // are scattered across blocks for deep trees).
        let tree_aware_attn_enabled =
            std::env::var("ATLAS_TREE_AWARE_ATTN").ok().as_deref() == Some("1");
        // A selective-boundary cache may mix attention KV dtypes. Do not pay
        // the indirection cost on the supported subset when any attention
        // layer would ignore the map: that verify cannot safely commit a
        // branch anyway. This also keeps the fallback on the established
        // batched chain kernel instead of running a strictly slower partial
        // tree computation whose result must be discarded.
        let all_attention_layers_exact = self
            .layers
            .iter()
            .all(|layer| layer.ddtree_ancestor_attention_exact());
        let tree_kv_active = tree_aware_attn_enabled
            && !dfs_active
            && tree_depths.is_some()
            && all_attention_layers_exact
            && self.tree_kv_indir_stride > 0
            && self.tree_kv_indir_persistent.0 != 0
            && k <= self.tree_kv_indir_stride;

        // ── Ancestor-exact attention flag (2026-07-08 deep-branch root cause) ──
        //
        // Record whether THIS verify gives EVERY tree node ancestor-exact
        // attention (each row attends to exactly the pre-context + its true
        // ancestors + itself):
        //   * flat/chain payload (or none)  → exact: prefix reads ARE the
        //     conditioning (byte-identical baseline path);
        //   * per-row KV indirection active → exact: the kernel remaps the
        //     tree window through each row's true ancestor chain;
        //   * NON-flat payload under prefix metadata (chain-mode or DFS
        //     depth-prefix reads) → NOT exact. DFS pre-order lays only the
        //     LEFTMOST path at slot == depth; any branch node at DFS slot
        //     s > depth reads DFS slots [0..depth] — its spine SIBLING at
        //     slot `depth` instead of its own key at slot `s`
        //     (`ddtree::dfs_prefix_reads_are_ancestor_exact`). Branch rows'
        //     logits are then wrong, and the deep tree-commit walker would
        //     consume them for child-acceptance and the bonus → committed
        //     tokens diverge from the greedy oracle (VALIDATION-36 TEST 1).
        //
        // The scheduler reads this via `dflash_tree_ancestor_attn_exact()`
        // and degrades the FULL tree-commit walker to the flat-safe walker
        // when not exact — turning that silent-corruption class into a safe
        // (spine-only) accept.
        let payload_non_flat = {
            let hp = self.ddtree_parent_ids_host.lock();
            hp.iter()
                .enumerate()
                .skip(1)
                .any(|(i, &p)| p != i as i32 - 1)
        };
        // Conv-exact assertion (ATLAS_DDTREE_TREE_CONV_EXACT=1 + kernel loaded).
        // The causal_conv1d_tree_reroot kernel re-roots each branch token's conv
        // shift-register from its true ancestor's intermediate, making the conv
        // output oracle-correct for ALL tree topologies. Branch commits without
        // this kernel are WRONG (BUG 1 from freeslots-rootcause.md): the conv
        // window propagates compact-predecessor tokens instead of true ancestors,
        // corrupting all 48 GDN layer outputs for branch rows.
        // Gating branch commits on conv_exact prevents silent non-oracle commits.
        // ATLAS_DDTREE_ASSUME_CONV_EXACT=1 is the research escape hatch (repro only).
        let tree_conv_kernel_available = self
            .layers
            .iter()
            .all(|layer| layer.ddtree_conv_state_exact());
        let tree_conv_exact_on = (tree_conv_kernel_available
            && std::env::var("ATLAS_DDTREE_TREE_CONV_EXACT")
                .ok()
                .as_deref()
                == Some("1"))
            || std::env::var("ATLAS_DDTREE_ASSUME_CONV_EXACT")
                .ok()
                .as_deref()
                == Some("1");
        // For a non-flat tree payload, BOTH attention indirection AND conv reroot
        // must be active to guarantee oracle-correct branch logits. A flat payload
        // (or no payload) is always exact by construction.
        // Indirection support is a per-attention-layer property. Selective
        // boundary caches can mix BF16 and FP8 layers; treating a live FP8
        // indirection buffer as a model-wide capability incorrectly certified
        // the BF16 layers even though their decode kernel dropped the tree
        // metadata. Require every layer to attest support and fail closed.
        let ancestor_attn_exact = !payload_non_flat
            || (tree_kv_active && tree_conv_exact_on && all_attention_layers_exact);
        self.dflash_tree_ancestor_attn
            .store(ancestor_attn_exact, std::sync::atomic::Ordering::Release);
        if payload_non_flat && ancestor_attn_exact {
            static EXACT_CERT_DBG: std::sync::atomic::AtomicUsize =
                std::sync::atomic::AtomicUsize::new(0);
            let n = EXACT_CERT_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 8 {
                use spark_runtime::kv_cache::KvCacheDtype;
                let attention_certificates: Vec<_> = self
                    .layers
                    .iter()
                    .filter_map(|layer| layer.ddtree_attention_certificate())
                    .collect();
                let count = |dtype| {
                    attention_certificates
                        .iter()
                        .filter(|(actual, _)| *actual == dtype)
                        .count()
                };
                let exact = |dtype| {
                    attention_certificates
                        .iter()
                        .filter(|(actual, supported)| *actual == dtype && *supported)
                        .count()
                };
                let bf16 = count(KvCacheDtype::Bf16);
                let fp8 = count(KvCacheDtype::Fp8);
                let other = attention_certificates.len().saturating_sub(bf16 + fp8);
                let conv_total = self
                    .layers
                    .iter()
                    .filter(|layer| layer.is_ssm_layer())
                    .count();
                let conv_exact = self
                    .layers
                    .iter()
                    .filter(|layer| layer.is_ssm_layer() && layer.ddtree_conv_state_exact())
                    .count();
                tracing::info!(
                    "DDTREE_EXACT_CERT #{n}: k={k} attention_layers={} \
                     bf16_tree_handles={}/{} fp8_indirection={}/{} \
                     other_attention={} conv_reroot={}/{}",
                    attention_certificates.len(),
                    exact(KvCacheDtype::Bf16),
                    bf16,
                    exact(KvCacheDtype::Fp8),
                    fp8,
                    other,
                    conv_exact,
                    conv_total,
                );
            }
        }
        if payload_non_flat && !ancestor_attn_exact {
            static NONEXACT_DBG: std::sync::atomic::AtomicUsize =
                std::sync::atomic::AtomicUsize::new(0);
            let n = NONEXACT_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 16 {
                tracing::info!(
                    "K=γ verify: non-flat tree payload — branch commits degraded to flat-safe \
                     (k={k}, dfs_active={dfs_active}, tree_aware_attn={tree_aware_attn_enabled}, \
                     tree_depths_some={}, indir_stride={}, indir_ptr_ok={}, all_attn_exact={}, \
                     tree_conv_kernel_ok={tree_conv_kernel_available}, \
                     tree_conv_exact_on={tree_conv_exact_on}) — \
                     Enable ATLAS_DDTREE_TREE_AWARE_VERIFY=1 + ATLAS_TREE_AWARE_ATTN=1 + \
                     ATLAS_DDTREE_TREE_CONV_EXACT=1 for lossless deep-branch commits.",
                    tree_depths.is_some(),
                    self.tree_kv_indir_stride,
                    self.tree_kv_indir_persistent.0 != 0,
                    all_attention_layers_exact,
                );
            }
        }

        let positions: Vec<u32> = if dfs_active {
            // DFS slot i contains kernel slot dfs_perm[i]; its tree depth
            // (kernel frame) is dfs_depths[dfs_perm[i]]. RoPE position is
            // seq.seq_len + depth so siblings at the same depth share
            // a position.
            (0..k)
                .map(|t| (seq.seq_len + dfs_depths[dfs_perm[t]]) as u32)
                .collect()
        } else if let Some(ref depths) = tree_depths {
            // Kernel slot 0 (bonus) is depth 0 → position seq.seq_len (the
            // last_token's slot). For drafts at depth d, position is
            // seq.seq_len + d.
            (0..k).map(|t| (seq.seq_len + depths[t]) as u32).collect()
        } else {
            (0..k).map(|t| (seq.seq_len + t) as u32).collect()
        };
        let pos_bytes =
            unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, k * 4) };
        let mut slots = vec![0i64; k];
        for t in 0..k {
            let pos = seq.seq_len + t;
            let block_idx = pos / bs;
            let block_offset = pos % bs;
            let physical_block = seq.physical_block_for(block_idx).unwrap_or(0);
            slots[t] = (physical_block as i64) * (bs as i64) + (block_offset as i64);
        }
        // 256-byte gap mirrors K=4 layout for ABI compatibility with
        // attention kernels that index meta_base + fixed offsets.
        let slot_bytes = unsafe { std::slice::from_raw_parts(slots.as_ptr() as *const u8, k * 8) };
        // seq_lens stays compact-index-based even in tree-aware mode. Reasoning:
        // setting seq_lens to depth+1 makes a depth-d query read positions
        // [0..seq.seq_len+d] which still are compact-ordered slots — those
        // are sibling/cousin slots, not actual ancestors, for any non-flat
        // tree. Without per-row block_table indirection (next TODO), the
        // chain-based seq_len is no worse than depth-based and avoids
        // dropping legitimate chain context inside ancestor-only subtrees.
        //
        // DFS reorder changes this: in DFS pre-order, when we visit a node at
        // depth d, the immediate ancestor chain (root → … → us) occupies the
        // d slots immediately preceding the current one (their stack frames).
        // So `seq_lens[t] = seq.seq_len + depth[dfs_perm[t]] + 1` gives a
        // depth-d query exactly its ancestor chain (the bonus + d ancestors),
        // *contiguously* in slot order. Sibling subtrees that were already
        // visited (and would otherwise contaminate the read) are skipped
        // because seq_len cuts off the read before reaching them.
        let seq_lens: Vec<i32> = if dfs_active {
            (0..k)
                .map(|t| (seq.seq_len + dfs_depths[dfs_perm[t]] + 1) as i32)
                .collect()
        } else if tree_kv_active {
            // ATLAS_TREE_AWARE_ATTN: kernel only walks the ancestor chain
            // for the tree window, so seq_len = prior_context + depth[t] + 1.
            // The kernel remaps positions ≥ seq.seq_len via the indirection
            // table so they read the right ancestors, not sibling slots.
            let depths = tree_depths.as_ref().unwrap();
            (0..k)
                .map(|t| (seq.seq_len + depths[t] + 1) as i32)
                .collect()
        } else {
            (0..k).map(|t| (seq.seq_len + t + 1) as i32).collect()
        };
        let sl_bytes = unsafe { std::slice::from_raw_parts(seq_lens.as_ptr() as *const u8, k * 4) };
        let mb = max_blocks as usize;
        let needed = k * mb;
        let mut bt_buf = vec![0i32; needed];
        for row in 0..k {
            for (j, &block) in seq.block_table.iter().enumerate().take(mb) {
                bt_buf[row * mb + j] = block as i32;
            }
        }
        let bt_bytes =
            unsafe { std::slice::from_raw_parts(bt_buf.as_ptr() as *const u8, needed * 4) };
        let metadata_copies = [
            HostToDeviceCopy::new(pos_bytes, meta_base),
            HostToDeviceCopy::new(slot_bytes, meta_base.offset(256)),
            HostToDeviceCopy::new(sl_bytes, meta_base.offset(512)),
            HostToDeviceCopy::new(bt_bytes, meta_base.offset(768)),
        ];
        self.gpu
            .copy_h2d_group_on_stream(&metadata_copies, stream)?;

        // ATLAS_TREE_AWARE_ATTN CUDA graph fix: upload `seq.seq_len as i32`
        // into the persistent 1×i32 device buffer that backs the kernel's
        // `kv_indir_base_ptr` / `abs_base_ptr` args. The captured K=γ verify
        // graph reads this via pointer indirection so the fresh value lands
        // on every replay (previously the scalar was baked into the kernel-
        // launch node at capture time and went stale → wrong `local_off`).
        // Upload happens BEFORE `begin_capture` (same pattern as `seq_lens`),
        // so it stays outside the graph.
        // Stamp the per-step base value into the persistent pinned-host
        // shadow then fire the async H2D copy. The kernels read this via
        // `kv_indir_base_ptr` / `abs_base_ptr` so captured CUDA graphs
        // pick up the fresh value on each replay (the prior scalar arg was
        // baked into the kernel-launch node at capture time).
        if tree_kv_active && !self.tree_kv_indir_base_persistent.is_null() {
            let pinned = unsafe { &mut *self.tree_kv_indir_base_host_pinned.get() };
            let pinned = pinned
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("tree KV base pinned host allocation is missing"))?;
            // Write the new base into the persistent pinned-host shadow
            // (in-place, host-side, no GPU involved) THEN fire the async
            // H2D copy from the pinned shadow. Pinned-host sources establish
            // a proper stream-ordered dependency for captured graph kernels;
            // pageable Vec sources do not (small pageable copies can race
            // ahead of subsequent graph launches → stale `kv_indir_base`).
            pinned
                .as_mut_slice()
                .copy_from_slice(&(seq.seq_len as i32).to_ne_bytes());
            let base_bytes = pinned.pinned_slice(std::mem::size_of::<i32>())?;
            unsafe {
                self.gpu.copy_h2d_pinned_async(
                    base_bytes,
                    self.tree_kv_indir_base_persistent,
                    stream,
                )?;
            }
        }

        // ATLAS_TREE_AWARE_ATTN: build + upload per-row KV indirection table.
        // Row t = ancestor chain of compact slot t in top-down (bonus → t)
        // order, indexed by depth in [0..depth[t]+1). Kernel reads
        // indirection[seq_idx][i] for i in [0..seq_lens[t]-kv_indir_base).
        if tree_kv_active {
            let stride = self.tree_kv_indir_stride;
            let depths_ref = tree_depths.as_ref().unwrap();
            let host_parents = self.ddtree_parent_ids_host.lock().clone();
            let mut indir = vec![0i32; stride * stride];
            for t in 0..k {
                // True ancestor chain [root(0), …, t] — unit-tested against
                // the free-slots branch shapes in ddtree.rs
                // (`ancestor_chain_matches_true_ancestors_for_free_slots_branch`).
                let mut chain: Vec<usize> =
                    crate::layers::dflash_head::ddtree::ancestor_chain_topdown(&host_parents, t, k);
                let want = depths_ref[t] + 1;
                while chain.len() < want {
                    chain.push(*chain.last().unwrap_or(&t));
                }
                for (j, &anc) in chain.iter().enumerate().take(stride) {
                    indir[t * stride + j] = anc as i32;
                }
                let last_anc = *chain.last().unwrap_or(&t) as i32;
                for j in chain.len()..stride {
                    indir[t * stride + j] = last_anc;
                }
            }
            let indir_bytes_view = unsafe {
                std::slice::from_raw_parts(
                    indir.as_ptr() as *const u8,
                    indir.len() * std::mem::size_of::<i32>(),
                )
            };
            self.gpu.copy_h2d_group_on_stream(
                &[HostToDeviceCopy::new(
                    indir_bytes_view,
                    self.tree_kv_indir_persistent,
                )],
                stream,
            )?;
            static TREE_KV_DBG_DONE: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !TREE_KV_DBG_DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                let preview: Vec<Vec<i32>> = (0..k)
                    .map(|t| {
                        let d = (depths_ref[t] + 1).min(stride);
                        indir[t * stride..t * stride + d].to_vec()
                    })
                    .collect();
                tracing::info!(
                    "ATLAS_TREE_AWARE_ATTN: k={k} indir_stride={stride} chains={:?}",
                    preview
                );
            }
        }

        let metadata = AttnMetadataDev {
            qwen4_qsa_required: seq.qwen4_qsa_required,
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(256),
            seq_len: meta_base.offset(512),
            block_table: meta_base.offset(768),
            max_blocks_per_seq: max_blocks,
            num_seqs: k as u32,
        };

        // Phase 6.2.c — HSS host I/O is illegal under CUDA graph capture.
        let hss_engaged = kv_cache.config().cache_blocks_per_seq.is_some();
        // ATLAS_DFLASH_DEBUG_NO_GRAPH=1 forces eager (no graph capture) so
        // CUDA_LAUNCH_BLOCKING=1 reports the exact failing kernel — used
        // to localize K=γ illegal-address crashes downstream of SSM.
        let force_eager = std::env::var("ATLAS_DFLASH_DEBUG_NO_GRAPH").ok().as_deref() == Some("1")
            || kgamma_debug_dump;
        // Mirror verify_b.rs (K=2) auto-unsuppress logic. Without this the
        // K=γ verify path stays EAGER forever once FP8/turbo KV calibration
        // sets suppress_graphs=true at startup, and never gets recaptured
        // into a CUDA graph. Eager K=γ verify is ~5x slower than graphed
        // (observed: K=9 eager 600ms/step vs K=4 graphed 80ms/step). Once
        // seq_len passes the calibration window, FP8 scales are frozen and
        // graph capture is safe.
        if self
            .suppress_graphs
            .load(std::sync::atomic::Ordering::Relaxed)
            && seq.seq_len > self.config.fp8_kv_calibration_tokens + 10
            && std::env::var("ATLAS_DUMP_HIDDEN").is_err()
        {
            self.suppress_graphs
                .store(false, std::sync::atomic::Ordering::Relaxed);
            tracing::info!("FP8 calibration frozen — re-enabling CUDA graphs (K=γ verify)");
        }
        // ATLAS_FULL_PROFILE=1 disables graph capture so per-kernel sync is
        // legal (CUDA graph capture forbids host syncs / D2H copies inside).
        let full_profile = crate::full_profile::is_enabled();
        let use_graphs = self.comm.is_none()
            && !self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            && !hss_engaged
            && !force_eager
            && !full_profile
            && !layer_resolution_probe_enabled()
            && serial_family.is_none()
            && !crate::model::k1_stage_diag::enabled()
            && !controlled_verify
            // Qwen4 PLE performs host-backed sparse row fetches and the
            // four-stream layer traversal carries row-specific inject
            // scratch. Keep this path eager until a graph-safe batched PLE
            // fetch exists.
            && !self.config.is_qwen4_exp();

        // M8A: upload DDTree parent_ids (when stashed) so the GDN dispatch
        // can fire the tree-aware kernel. Stash lives on the model so we
        // don't have to thread it through every call site here.
        // verify_dflash_step.rs sets ddtree_parent_ids_dev before calling
        // this verify entry (via model.set_ddtree_parent_ids) so we just
        // read here.
        //
        // CUDA-graph-safe path: when graphs are enabled AND the verify K
        // matches the configured γ+1 (= dflash_kgamma = persistent buffer
        // capacity), substitute the persistent linear-chain buffer when
        // no tree payload was stashed. This forces every K=γ verify
        // through the `gdn_tree_wy_k` branch in the SSM dispatch, so the
        // captured graph always references the same kernel and the same
        // device pointer. Linear-chain parents (= `[-1, 0, 1, ..., γ-1]`)
        // make `gated_delta_rule_tree_wy` bit-equivalent to the old wy17
        // path (verified cos=1.0 across all 17 tokens in a prior session),
        // so output text is identical to the pre-fix eager path.
        let mut ddtree_parent_ids_dev = *self.ddtree_parent_ids_dev.lock();
        // ATLAS_DISABLE_TREE_WY=1 forces flat-chain wy_k path (wy17 etc.)
        // instead of `gated_delta_rule_tree_wy`. The tree-WY kernel is
        // NOT bit-equivalent to wy17 on linear chains (see propose.rs
        // comment around line 343 — "first-pass tree kernel isn't bit-
        // equivalent to wy17 — flat-chain tokens drift numerically and
        // drafter accept collapses"). When γ matches
        // `ddtree_parent_ids_capacity`, the auto-injection at this site
        // would otherwise force the broken kernel. Setting the env var
        // skips the injection and lets the proven wy_k path run.
        let disable_tree_wy = std::env::var("ATLAS_DISABLE_TREE_WY").ok().as_deref() == Some("1");
        let flat_injected = super::commit_plan::flat_tree_wy_injection_applies(
            disable_tree_wy,
            use_graphs,
            ddtree_parent_ids_dev.is_some(),
            self.ddtree_parent_ids_capacity,
            k,
        );
        if flat_injected {
            ddtree_parent_ids_dev = Some(self.ddtree_parent_ids_persistent);
        }
        // Task #34 fix: the injected flat-chain verify runs
        // `gated_delta_rule_tree_wy`, which leaves the live `h_state`
        // UNTOUCHED (it writes per-token states into the intermediate pool
        // only). The commit's full-accept fast path (`async_chkpt.rs`)
        // previously read only the scheduler-set `ddtree_parent_ids_dev`
        // stash to detect this — the injection was invisible, so a FULL
        // accept committed the STALE pre-verify `h_state` (SSM state froze →
        // non-lossless at high acceptance; observed at K=12 / γ=11 in the
        // DSpark A/B). Record the injection on the model so the commit
        // routes full accepts through `h_intermediate[K-1]`.
        //
        // The flag is only honored when the SSM dispatch would actually take
        // the tree branch (`gdn_tree_k` loaded and `ATLAS_FORCE_WY17` unset —
        // mirrors `trait_decode_batched_conv_gdn.rs`); otherwise the flat
        // wy17/chunked kernels run, which DO write `h_state`.
        // `any` assumes GDN kernel homogeneity: every SSM layer is built from
        // the same PTX bundle, so either all have the tree kernel or none do
        // (non-SSM layers return false). A mixed target would need a
        // filtered `all` here — the per-layer dispatch checks its OWN handle,
        // and a layer falling back to wy17/chunked writes h_state live,
        // making the InterSlot commit source wrong for that layer.
        let injection_routes_tree = flat_injected
            && std::env::var("ATLAS_FORCE_WY17").ok().as_deref() != Some("1")
            && self.layers.iter().any(|l| l.gdn_tree_kernel_loaded());
        // Release/Acquire pairing with the commit-side swap: the scheduler
        // calls verify and commit sequentially on one thread today, but the
        // flag guards h_state commit-source selection (silent corruption if
        // desynced), so pay the fence and stay correct under any future
        // cross-thread step orchestration.
        self.dflash_flat_tree_route
            .store(injection_routes_tree, std::sync::atomic::Ordering::Release);

        // DFS reorder: re-stamp parent_ids in DFS frame so the SSM (GDN
        // tree-WY) kernel processes tokens in DFS order, matching the
        // attention KV layout. After this, h_state_inter[i] holds the SSM
        // state AFTER DFS slot i. The commit path must map original-compact
        // indices → DFS slots via dfs_inv_perm to read the right inter slot.
        if dfs_active && self.ddtree_parent_ids_capacity > 0 {
            use crate::layers::dflash_head::ddtree::permute_parent_ids;
            // host_parents is the current kernel-frame parent_ids (pre-DFS).
            // Pad with linear-chain tail to match k, mirroring the depth
            // derivation logic above.
            let mut host_parents = self.ddtree_parent_ids_host.lock().clone();
            while host_parents.len() < k {
                let next_chain_parent = host_parents.len() as i32 - 1;
                host_parents.push(next_chain_parent);
            }
            if host_parents.len() == k {
                let permuted = permute_parent_ids(&host_parents, &dfs_perm, &dfs_inv_perm);
                let bytes: Vec<u8> = permuted.iter().flat_map(|p| p.to_le_bytes()).collect();
                self.gpu.copy_h2d_group_on_stream(
                    &[HostToDeviceCopy::new(
                        &bytes,
                        self.ddtree_parent_ids_persistent,
                    )],
                    stream,
                )?;
                // Make sure dispatch uses persistent buffer (which now holds
                // DFS-frame parents).
                ddtree_parent_ids_dev = Some(self.ddtree_parent_ids_persistent);
            }
        }

        // Stash dfs_inv_perm so commit_verify_state_async_with_slot can map
        // original-compact `last_inter_slot` (from greedy_sample_ddtree) →
        // DFS slot (where the SSM kernel actually wrote the canonical state).
        // Cleared by clear_ddtree_parent_ids() after commit finishes reading.
        //
        // METADATA_ONLY=1: SSM still writes h_state_inter in kernel-original
        // slot order (because parent_ids re-stamping happens above either
        // way), so we still need the inv_perm mapping for commit.
        if dfs_active {
            *self.ddtree_dfs_inv_perm.lock() = dfs_inv_perm.clone();
        } else {
            self.ddtree_dfs_inv_perm.lock().clear();
        }

        // ATLAS_TREE_AWARE_ATTN: wire the per-row KV indirection plumbing
        // into the forward context so paged-decode attention can remap
        // tree-window positions to true ancestors. None when the env var
        // is off or DFS reorder is active.
        //
        // ATLAS_TREE_KV_PACK (additional): also pack ancestor KV into a
        // contiguous per-layer scratch pool and upload `seq_lens =
        // depth[t]+1` so the consumer kernel can run with NULL indirection
        // over the scratch — restores the fast BC=4 batched attention path
        // that the per-position fallback gives up.
        let tree_aware_attn = if tree_kv_active {
            let pack = if self.tree_kv_pack_active
                && !self.tree_kv_pack_scratch_k.is_empty()
                && !self.tree_kv_pack_scratch_v.is_empty()
            {
                // Upload per-step `seq_lens = depth[t]+1` into the dedicated
                // packed-KV seq_lens buffer (separate from the main attn
                // metadata `seq_len` which still holds `seq.seq_len + d + 1`
                // for the unpacked path / fallback).
                let depths_ref = tree_depths.as_ref().unwrap();
                let mut packed_seq_lens = vec![0i32; self.tree_kv_indir_stride];
                for t in 0..k {
                    packed_seq_lens[t] = (depths_ref[t] + 1) as i32;
                }
                let sl_bytes_view = unsafe {
                    std::slice::from_raw_parts(
                        packed_seq_lens.as_ptr() as *const u8,
                        packed_seq_lens.len() * std::mem::size_of::<i32>(),
                    )
                };
                self.gpu.copy_h2d_group_on_stream(
                    &[HostToDeviceCopy::new(
                        sl_bytes_view,
                        self.tree_kv_pack_seq_lens,
                    )],
                    stream,
                )?;
                // NOTE: kv_cache mutex is already held earlier in this fn
                // (`let mut kv_cache = self.kv_cache.lock();`), so we must
                // NOT re-lock it. Read block_size via the already-held
                // borrow (`kv_cache.block_size()`).
                let cache_block_size = kv_cache.block_size() as u32;
                static PACK_DBG_DONE: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !PACK_DBG_DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    tracing::info!(
                        "ATLAS_TREE_KV_PACK: k={k} chain_lens={:?} stride={} cache_bs={}",
                        &packed_seq_lens[..k],
                        self.tree_kv_indir_stride,
                        cache_block_size
                    );
                }
                Some(crate::layer::TreeKvPack {
                    num_attn_layers: self.tree_kv_pack_scratch_k.len() as u32,
                    scratch_k_ptrs: self.tree_kv_pack_scratch_k.as_ptr(),
                    scratch_v_ptrs: self.tree_kv_pack_scratch_v.as_ptr(),
                    identity_block_table: self.tree_kv_pack_block_table,
                    seq_lens: self.tree_kv_pack_seq_lens,
                    block_stride_bytes: self.tree_kv_pack_block_stride_bytes,
                    data_section_bytes: self.tree_kv_pack_data_section_bytes,
                    block_size: self.tree_kv_indir_stride as u32,
                    scatter_fp8_kernel: self.tree_kv_pack_scatter_fp8_kernel,
                    scatter_nvfp4_kernel: self.tree_kv_pack_scatter_nvfp4_kernel,
                    cache_block_size,
                    cache_max_blocks_per_seq: max_blocks,
                    // CUDA graph fix: read abs_base via device buffer so a
                    // captured graph picks up the fresh `seq.seq_len` on
                    // each replay. Buffer is updated above (before
                    // begin_capture).
                    abs_base_ptr: self.tree_kv_indir_base_persistent,
                })
            } else {
                None
            };
            Some(crate::layer::TreeAwareAttn {
                kv_indir: self.tree_kv_indir_persistent,
                // CUDA graph fix: see `abs_base_ptr` above.
                kv_indir_base_ptr: self.tree_kv_indir_base_persistent,
                kv_indir_stride: self.tree_kv_indir_stride as u32,
                pack,
            })
        } else {
            None
        };

        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(metadata),
            profile: full_profile,
            comm: self.comm_ref(),
            graph_capture: use_graphs,
            ddtree_parent_ids_dev,
            tree_aware_attn,
            ssm_multi_seq_ptr_table_override: None,
            self_spec_sparse_draft: None,
            ffn_defer: None,
        };

        // ── Phase 2: CUDA graph capture / replay ──

        let mut graph_cache = if use_graphs {
            Some(self.verify_kgamma_graph.lock())
        } else {
            None
        };

        // Cache key includes (slot, k, graph-shape flags). The pack flag is
        // critical: when ATLAS_TREE_KV_PACK is enabled, the first K=γ verify
        // (often pre-tree, flat-chain) captures a graph WITHOUT pack-pool
        // kernel arguments. Later verifies with non-flat trees can't reuse
        // that graph — its kernel sequence has stale arg pointers. Treating
        // pack as a separate cache key forces a fresh capture for the
        // pack-active path.
        // Bit 1: tree-aware indirection active. A graph captured WITHOUT the
        // indirection kernel arguments (flat step, kv_indirection=NULL) must
        // never be replayed for a tree-indirected step of the same K (and
        // vice versa) — the kernel argument sets differ.
        // Bit 2: a real/synthetic SSM tree parent pointer is active. Flat K4
        // uses exact FP32 sequence recurrence, whereas a genuine DDTree K4
        // uses the tree kernel family; those graph topologies cannot alias.
        let pack_key = super::commit_plan::verify_graph_shape_key(
            ctx.tree_aware_attn.and_then(|t| t.pack).is_some(),
            ctx.tree_aware_attn.is_some(),
            ctx.ddtree_parent_ids_dev.is_some(),
        );
        let cache_key = (seq.slot_idx, k, pack_key);
        let cached_for_slot = graph_cache
            .as_ref()
            .and_then(|c| c.get(&cache_key).copied());
        if let Some(graph) = cached_for_slot
            && graph.0 != 0
        {
            self.gpu.launch_graph(graph, stream)?;
        }
        let mut lrp_snapshots: Vec<Vec<u8>> = Vec::new();
        let need_run = cached_for_slot.is_none();
        if need_run {
            let seq_lens_vec: Vec<usize> = (0..k).map(|t| seq.seq_len + t).collect();
            let block_tables_vec: Vec<Vec<u32>> = vec![seq.block_table.clone(); k];

            let control_guard = if controlled_verify {
                let counts = self.layers.iter().enumerate().fold(
                    crate::model::control_engagement::LayerCounts::default(),
                    |mut counts, (layer_idx, _)| {
                        if !verify_skip_layer(layer_idx) {
                            if self.config.layer_type(layer_idx) == LayerType::FullAttention {
                                counts.attention += 1;
                            } else {
                                counts.ssm += 1;
                            }
                        }
                        counts
                    },
                );
                if hss_engaged && counts.attention > 0 && control_requests.attention_requested() {
                    bail!(
                        "DFLASH_CONTROL_PATH_PROOF requested=true engaged=false \
                         requirement=attention controls require the multi-sequence HBM path"
                    );
                }
                crate::model::control_engagement::begin(control_requests, counts)?
            } else {
                None
            };

            if use_graphs {
                self.gpu.begin_capture(stream)?;
            }

            let mut qwen4_ssm_layer_idx = 0usize;
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let layer_type = self.config.layer_type(layer_idx);

                // ATLAS_VERIFY_SKIP_LAYERS="2,3,7,...": static layer-skip in the
                // single-stream γ verify. A skipped layer is identity (residual
                // passes through unchanged) at EVERY step, so its SSM/KV state is
                // never written and never read — consistent, graph-capture-safe.
                // Lossy vs md5 but pass@1-gated (measured: skip-16 = 25% verify
                // bandwidth cut at ~pass@1-preserving on HumanEval-style coding).
                // Capture layers [1,10,18,27,35,44,52,61] must NOT be in the set.
                if verify_skip_layer(layer_idx) {
                    continue;
                }

                if self.config.is_qwen4_exp() {
                    if layer_idx == 1
                        && let Some(ple) = &self.qwen4_ple
                    {
                        for (row, &token) in tokens.iter().enumerate() {
                            let mut prior = seq.tokens.clone();
                            prior.extend_from_slice(&tokens[..row]);
                            ple.forward_token(
                                token,
                                &prior,
                                hidden.offset(row * persistent_width * fp32),
                                seq.slot_idx,
                                false,
                                self.gpu.as_ref(),
                                stream,
                            )?;
                            ple.save_intermediate(seq.slot_idx, row, self.gpu.as_ref(), stream)?;
                        }
                    }

                    let (h_inter, conv_inter) = if layer_type == LayerType::LinearAttention {
                        (
                            self.ssm_pool
                                .h_intermediate(qwen4_ssm_layer_idx, seq.slot_idx, 0),
                            self.ssm_pool
                                .conv_intermediate(qwen4_ssm_layer_idx, seq.slot_idx, 0),
                        )
                    } else {
                        (DevicePtr::NULL, DevicePtr::NULL)
                    };
                    // The generic Qwen4 batched layer path first diverges
                    // from ordinary decode at layer 0 for K=5. Because its
                    // recurrent intermediates may be committed, even a
                    // logits-tolerant numeric delta becomes cross-step state
                    // corruption. Keep the released gamma-4 geometry on the
                    // lossless row-serial layer path until each batched stage
                    // has an in-process parity proof.
                    let qwen4_k5_hybrid = k == 5
                        && std::env::var("ATLAS_QWEN4_K5_HYBRID").ok().as_deref() == Some("1");
                    if k == 5 && !qwen4_k5_hybrid {
                        let metadata = ctx.attn_metadata.ok_or_else(|| {
                            anyhow::anyhow!("Qwen4 K=5 verify requires attention metadata")
                        })?;
                        for row in 0..k {
                            let token_metadata = AttnMetadataDev {
                                positions: metadata.positions.offset(row * 4),
                                positions_h: metadata.positions_h.offset(row * 4),
                                positions_w: metadata.positions_w.offset(row * 4),
                                slot: metadata.slot.offset(row * 8),
                                seq_len: metadata.seq_len.offset(row * 4),
                                block_table: metadata.block_table.offset(row * mb * 4),
                                num_seqs: 1,
                                ..metadata
                            };
                            let token_ctx = ForwardContext {
                                attn_metadata: Some(token_metadata),
                                ..ctx
                            };
                            layer.decode(
                                hidden.offset(row * persistent_width * fp32),
                                residual.offset(row * persistent_width * fp32),
                                seq.layer_states[layer_idx].as_mut(),
                                &mut kv_cache,
                                seq.seq_len + row,
                                &mut seq.block_table,
                                &mut seq.disk_block_ids,
                                &mut seq.disk_last_offloaded_per_layer,
                                &token_ctx,
                                stream,
                            )?;
                            if layer_type == LayerType::LinearAttention {
                                let ssm = seq.layer_states[layer_idx]
                                    .as_any_mut()
                                    .downcast_mut::<SsmLayerState>()
                                    .ok_or_else(|| {
                                        anyhow::anyhow!(
                                            "Qwen4 K=5 verify expected SSM state at layer {layer_idx}"
                                        )
                                    })?;
                                self.gpu.copy_d2d_async(
                                    ssm.h_state,
                                    h_inter.offset(row * self.ssm_pool.h_bytes),
                                    self.ssm_pool.h_bytes,
                                    stream,
                                )?;
                                self.gpu.copy_d2d_async(
                                    ssm.conv_state,
                                    conv_inter.offset(row * self.ssm_pool.conv_bytes),
                                    self.ssm_pool.conv_bytes,
                                    stream,
                                )?;
                            }
                        }
                    } else {
                        layer.decode_qwen4_batched(
                            hidden,
                            residual,
                            k,
                            seq.layer_states[layer_idx].as_mut(),
                            &mut kv_cache,
                            seq.seq_len,
                            &mut seq.block_table,
                            &mut seq.disk_block_ids,
                            &mut seq.disk_last_offloaded_per_layer,
                            h_inter,
                            conv_inter,
                            self.ssm_pool.h_bytes,
                            self.ssm_pool.conv_bytes,
                            &ctx,
                            stream,
                        )?;
                    }
                    if layer_type == LayerType::LinearAttention {
                        qwen4_ssm_layer_idx += 1;
                    }
                } else if layer_type == LayerType::FullAttention {
                    if hss_engaged {
                        // HSS path: decode_multi_seq's paged-decode kernel
                        // reads K/V from HBM only, missing the long-context
                        // history on disk. Fall back to decode_batched
                        // (sequential single-token decodes via the HSS
                        // orchestrator). See verify_b.rs for full rationale.
                        layer.decode_batched(
                            hidden,
                            residual,
                            k,
                            seq.layer_states[layer_idx].as_mut(),
                            &mut kv_cache,
                            seq.seq_len,
                            &mut seq.block_table,
                            &mut seq.disk_block_ids,
                            &mut seq.disk_last_offloaded_per_layer,
                            &ctx,
                            stream,
                        )?;
                    } else {
                        let mut dummy_states: Vec<Box<dyn LayerState>> = (0..k)
                            .map(|_| layer.alloc_state(self.gpu.as_ref()))
                            .collect::<Result<_>>()?;
                        let mut refs: Vec<&mut (dyn LayerState + 'static)> =
                            dummy_states.iter_mut().map(|s| s.as_mut()).collect();
                        crate::kprof!(self.gpu.as_ref(), stream, "verify_attn", {
                            layer.decode_multi_seq(
                                hidden,
                                residual,
                                k,
                                &mut refs,
                                &mut kv_cache,
                                &seq_lens_vec,
                                &block_tables_vec,
                                &ctx,
                                stream,
                            )
                        })?;
                    }
                } else {
                    crate::kprof!(self.gpu.as_ref(), stream, "verify_ssm", {
                        layer.decode_batched(
                            hidden,
                            residual,
                            k,
                            seq.layer_states[layer_idx].as_mut(),
                            &mut kv_cache,
                            seq.seq_len,
                            &mut seq.block_table,
                            &mut seq.disk_block_ids,
                            &mut seq.disk_last_offloaded_per_layer,
                            &ctx,
                            stream,
                        )
                    })?;
                }
                self.capture_k1_stage(
                    &format!("layer_{layer_idx:02}"),
                    hidden,
                    k,
                    persistent_width * fp32,
                    stream,
                )?;
                // DFlash hidden capture for ctx conditioning. Save ALL k
                // tokens so the scheduler can pick the correct one
                // (num_accepted) after verify. Layout:
                // [token_idx, capture_layer, hidden] in dflash_hidden_save.
                for t in 0..k {
                    self.try_dflash_capture(layer_idx, t, stream)?;
                }
                // ATLAS_LAYER_RESOLUTION_PROBE: snapshot [K, H] BF16 hidden
                // at checkpoint layers. GPU sync required (probe is eager-only).
                if layer_resolution_probe_enabled() && LRP_PROBE_LAYERS.contains(&layer_idx) {
                    let h_bytes = k * h * 2;
                    let mut snap = vec![0u8; h_bytes];
                    self.gpu.synchronize(stream)?;
                    self.gpu.copy_d2h(hidden, &mut snap)?;
                    lrp_snapshots.push(snap);
                }
                // ATLAS_KGAMMA_DEBUG_DUMP=1: dump per-position hidden state
                // for K=γ verify so we can diff against HF reference
                // (modelforge inspect-batched). Writes
                // /tmp/atlas_kgamma_layer{L}_pos{t}.bin one-shot per pair.
                // Dump only when NOT capturing a CUDA graph (sync ops illegal
                // during capture). Set ATLAS_DFLASH_DEBUG_NO_GRAPH=1 too.
                if kgamma_debug_dump && !use_graphs {
                    let h_bytes = h * 2;
                    for t in 0..k {
                        let path = format!("/tmp/atlas_kgamma_layer{}_pos{}.bin", layer_idx, t);
                        if !std::path::Path::new(&path).exists() {
                            let mut buf = vec![0u8; h_bytes];
                            self.gpu.synchronize(stream)?;
                            self.gpu.copy_d2h(hidden.offset(t * h_bytes), &mut buf)?;
                            let _ = std::fs::write(&path, &buf);
                        }
                    }
                }
            }

            if let Some(guard) = control_guard {
                guard.finish()?;
            }

            let normed = self.buffers.norm_output();
            if self.config.is_qwen4_exp() {
                let saved_row0 = self.buffers.attn_output();
                for t in 0..k {
                    let row_offset = t * persistent_width * fp32;
                    self.qwen4_final_hidden(
                        hidden.offset(row_offset),
                        residual.offset(row_offset),
                        stream,
                    )?
                    .ok_or_else(|| anyhow::anyhow!("qwen4_exp verify missing final mixer"))?;
                    if t == 0 {
                        self.gpu
                            .copy_d2d_async(normed, saved_row0, h * bf16, stream)?;
                    } else {
                        self.gpu.copy_d2d_async(
                            normed,
                            normed.offset(t * h * bf16),
                            h * bf16,
                            stream,
                        )?;
                    }
                }
                self.gpu
                    .copy_d2d_async(saved_row0, normed, h * bf16, stream)?;
            } else if serial.final_norm {
                self.dflash_k1_final_norm(hidden, k, stream)?;
            } else {
                crate::kprof!(self.gpu.as_ref(), stream, "final_norm", {
                    ops::rms_norm(
                        self.gpu.as_ref(),
                        self.rms_norm_kernel,
                        hidden,
                        &self.final_norm,
                        normed,
                        k as u32,
                        h as u32,
                        self.config.rms_norm_eps as f32,
                        stream,
                    )?;
                    anyhow::Result::<()>::Ok(())
                })?;
            }
            self.capture_k1_stage("final_norm", normed, k, h * bf16, stream)?;

            // DEBUG: dump post-final-norm hidden states (K positions)
            if kgamma_debug_dump && !use_graphs {
                let h_bytes = h * 2;
                for t in 0..k {
                    let path = format!("/tmp/atlas_kgamma_final_norm_pos{}.bin", t);
                    if !std::path::Path::new(&path).exists() {
                        let mut buf = vec![0u8; h_bytes];
                        self.gpu.synchronize(stream)?;
                        self.gpu.copy_d2h(normed.offset(t * h_bytes), &mut buf)?;
                        let _ = std::fs::write(&path, &buf);
                    }
                }
            }

            let lm_head_proof = if serial.lm_head {
                let (_, proof) = self.dflash_k1_lm_head_argmax(normed, k, stream)?;
                Some(proof)
            } else {
                crate::kprof!(self.gpu.as_ref(), stream, "lm_head", {
                    self.lm_head_batched(normed, k as u32, stream)?;
                    anyhow::Result::<()>::Ok(())
                })?;
                None
            };
            self.capture_k1_stage(
                "logits",
                self.buffers.logits(),
                k,
                self.config.vocab_size * bf16,
                stream,
            )?;

            if stage_capture {
                if let Some(proof) = lm_head_proof {
                    let family = serial_family.ok_or_else(|| {
                        anyhow::anyhow!("DFLASH_K1_LM_HEAD_PATH_PROOF lacks a named serial family")
                    })?;
                    let proof_line = proof.proof_line(family, seq.seq_len, tokens)?;
                    tracing::warn!("{proof_line}");
                }
                let report = crate::model::k1_stage_diag::finish_batch()?;
                if let Some(first) = report.first {
                    tracing::warn!(
                        run_id = report.manifest.run_id,
                        verify_step = report.manifest.verify_step,
                        pre_verify_len = report.manifest.pre_verify_len,
                        tokens = ?report.manifest.tokens,
                        absolute_seq_lens = ?report.manifest.absolute_seq_lens,
                        family = report.manifest.family,
                        stage = first.stage,
                        row = first.row,
                        mismatch_rows = ?first.mismatch_rows,
                        first_byte = first.first_byte,
                        serial_hash = format_args!("{:016x}", first.serial_hash),
                        batch_hash = format_args!("{:016x}", first.batch_hash),
                        stages = report.stages,
                        terminal_stage = report.terminal_stage,
                        logits_compared = report.logits_compared,
                        "DFLASH_K1_STAGE_FIRST_DIVERGENCE in_process_exact"
                    );
                } else {
                    tracing::info!(
                        run_id = report.manifest.run_id,
                        verify_step = report.manifest.verify_step,
                        pre_verify_len = report.manifest.pre_verify_len,
                        tokens = ?report.manifest.tokens,
                        absolute_seq_lens = ?report.manifest.absolute_seq_lens,
                        family = report.manifest.family,
                        stages = report.stages,
                        terminal_stage = report.terminal_stage,
                        logits_compared = report.logits_compared,
                        "DFLASH_K1_STAGE_MATCH in_process_exact"
                    );
                }
            }

            // DEBUG: dump logits for first 100 vocab entries per position
            if kgamma_debug_dump && !use_graphs {
                let vocab = self.config.vocab_size;
                let bf16 = 2usize;
                let dump_n = 100.min(vocab);
                for t in 0..k {
                    let path = format!("/tmp/atlas_kgamma_logits_pos{}.bin", t);
                    if !std::path::Path::new(&path).exists() {
                        let mut buf = vec![0u8; dump_n * bf16];
                        self.gpu.synchronize(stream)?;
                        self.gpu
                            .copy_d2h(self.buffers.logits().offset(t * vocab * bf16), &mut buf)?;
                        let _ = std::fs::write(&path, &buf);
                    }
                }
            }

            // Argmax inside graph (fixed scratch addresses — graph-safe).
            // Use the verify-side (possibly truncated) vocab so the per-row
            // logits stride AND the argmax range match exactly what
            // `lm_head_batched` wrote for the K=γ transposed GEMM — full vocab
            // when `ATLAS_TARGET_LMHEAD_VOCAB` truncation is off.
            if !serial.lm_head {
                let vocab = self.verify_lmhead_vocab() as usize;
                let argmax_out = self.buffers.scratch();
                crate::kprof!(self.gpu.as_ref(), stream, "argmax", {
                    for t in 0..k {
                        let logits_t = self.buffers.logits().offset(t * vocab * bf16);
                        let out_t = argmax_out.offset(t * 4);
                        ops::argmax_bf16(
                            self.gpu.as_ref(),
                            self.argmax_kernel,
                            logits_t,
                            out_t,
                            vocab as u32,
                            stream,
                        )?;
                    }
                    anyhow::Result::<()>::Ok(())
                })?;
            }

            if use_graphs {
                let graph = self.gpu.end_capture(stream)?;
                if graph.0 != 0 {
                    tracing::info!(
                        "Captured CUDA graph for K=γ verify (slot={} K={})",
                        seq.slot_idx,
                        k
                    );
                    if let Some(ref mut cache) = graph_cache {
                        cache.insert(cache_key, graph);
                    }
                    self.gpu.launch_graph(graph, stream)?;
                }
            }
        }

        // ── Phase 3: Post-graph (D2H copy only) ──

        // ATLAS_DUMP_HIDDEN: flush captured layer hiddens to file. See
        // verify_b.rs for the eager-mode safety contract. On the DFlash
        // path try_dflash_capture is already called per layer (line 790
        // above), so by Phase 3 dflash_hidden_save holds k tokens' worth
        // of (n_capture, hidden) records ready to flush.
        self.flush_hidden_dump(k)?;

        let out_ptr = self.buffers.scratch();
        let mut buf = vec![0u8; k * 4];
        self.gpu.copy_d2h(out_ptr, &mut buf)?;
        let mut out = Vec::with_capacity(k);
        for t in 0..k {
            let off = t * 4;
            out.push(u32::from_le_bytes([
                buf[off],
                buf[off + 1],
                buf[off + 2],
                buf[off + 3],
            ]));
        }

        // ATLAS_LAYER_RESOLUTION_PROBE: for each captured snapshot, run
        // norm+lm_head+argmax on GPU and compare to final argmax.  Accumulate
        // match rates into LRP_ACCUM, dump every lrp_dump_every() steps.
        if layer_resolution_probe_enabled() && !lrp_snapshots.is_empty() {
            let normed = self.buffers.norm_output();
            let vocab = self.verify_lmhead_vocab() as usize;
            let argmax_out = self.buffers.scratch();
            let mut probe_ams: Vec<Vec<u32>> = Vec::with_capacity(lrp_snapshots.len());
            for snap in &lrp_snapshots {
                // Upload snapshot → hidden buffer, then norm+lm_head+argmax.
                // `hidden` is not used after the probe block so reuse is safe.
                self.gpu.copy_h2d(snap, hidden)?;
                ops::rms_norm(
                    self.gpu.as_ref(),
                    self.rms_norm_kernel,
                    hidden,
                    &self.final_norm,
                    normed,
                    k as u32,
                    h as u32,
                    self.config.rms_norm_eps as f32,
                    stream,
                )?;
                self.lm_head_batched(normed, k as u32, stream)?;
                for t in 0..k {
                    let logits_t = self.buffers.logits().offset(t * vocab * bf16);
                    let out_t = argmax_out.offset(t * 4);
                    ops::argmax_bf16(
                        self.gpu.as_ref(),
                        self.argmax_kernel,
                        logits_t,
                        out_t,
                        vocab as u32,
                        stream,
                    )?;
                }
                self.gpu.synchronize(stream)?;
                let mut pbuf = vec![0u8; k * 4];
                self.gpu.copy_d2h(argmax_out, &mut pbuf)?;
                let pam: Vec<u32> = (0..k)
                    .map(|t| {
                        let o = t * 4;
                        u32::from_le_bytes([pbuf[o], pbuf[o + 1], pbuf[o + 2], pbuf[o + 3]])
                    })
                    .collect();
                probe_ams.push(pam);
            }
            let accum = LRP_ACCUM.get_or_init(|| Mutex::new(LrpAccum::new(k)));
            let mut g = accum.lock();
            g.accumulate(&out, &probe_ams);
            if g.step.is_multiple_of(lrp_dump_every()) {
                g.dump();
            }
        }

        // DFS reorder: un-permute verified outputs back to original-compact
        // frame so the caller (verify_dflash_step.rs::greedy_sample_ddtree)
        // sees outputs indexed by the ORIGINAL compact (kernel) frame.
        //
        // `out[i]` (DFS frame) is the argmax-after-`tokens[dfs_perm[i]]`.
        // We want `out_orig[j]` = argmax-after-`tokens[j]` = `out[dfs_inv_perm[j]]`.
        //
        // Un-permute is only needed when token permutation was actually
        // applied (METADATA_ONLY=0). With METADATA_ONLY=1, tokens stayed
        // in linear order so out[i] already corresponds to drafts[i-1]'s
        // next-token prediction in the same kernel slot.
        if dfs_active && dfs_permute_tokens {
            let mut out_orig = vec![0u32; k];
            for j in 0..k {
                out_orig[j] = out[dfs_inv_perm[j]];
            }
            out = out_orig;
        }

        // See decode_verify_graphed for rationale on `seq_len += k` fix.
        for &t in tokens {
            seq.tokens.push(t);
        }
        seq.seq_len += k;

        Ok(out)
    }

    /// FIX 2 — KV compaction for sparse (tree-fork) accepts.
    ///
    /// During K=γ verify, compact slot `t` writes its attention K/V to the
    /// linear KV position `pre_verify_len + t` (see the `slots` build above).
    /// When the target's greedy walk crosses a sibling fork, the accepted
    /// path's compact indices are NON-contiguous (e.g. `[1, 2, 3, 7]`), so the
    /// accepted tokens' KV is scattered across positions `pre+1, pre+2, pre+3,
    /// pre+7`. The next decode reads a CONTIGUOUS sequence `pre+1 .. pre+N`, so
    /// the scattered entries must be GATHERED down to the contiguous run that
    /// mirrors how the flat chain lays KV.
    ///
    /// `accepted_compact` is `greedy_sample_ddtree_full`'s path (compact
    /// indices, in walk order). `pre_verify_len` is the sequence length BEFORE
    /// this verify advanced it. For each accepted step `j` (1-based), the
    /// token lives at compact index `accepted_compact[j-1]` (KV pos
    /// `pre + that`); its contiguous home is `pre + j`. A copy fires only when
    /// source != destination (a flat prefix is already in place → no-op).
    ///
    /// LOSSLESS: this only relocates already-committed K/V bytes; it never
    /// changes a value. Bonus row needs no KV (it is fed as verify input next
    /// step). Supported KV dtypes: bf16 / fp8 (contiguous per-position element
    /// layout). Quantized cache dtypes (NVFP4/Turbo*) have split data+scale
    /// sections and are refused so the feature never silently corrupts.
    pub(super) fn compact_verify_kv_dispatch(
        &self,
        seq: &SequenceState,
        accepted_compact: &[usize],
        pre_verify_len: usize,
    ) -> Result<()> {
        // TASK #29 root-cause fix. Two things this must get right:
        //
        //   1. FRAME. The accepted path is in the COMPACT frame, but the KV was
        //      written by the kernel in the KERNEL (DFS-reordered) frame. When
        //      ATLAS_DDTREE_DFS_REORDER=1 (the portfolio / deep-DDTree commit
        //      config), compact index `c`'s KV lives at KV position
        //      `pre + dfs_inv_perm[c]`, NOT `pre + c`. The pre-fix code gathered
        //      from `pre + compact` → wrong bytes → counting md5 corruption on a
        //      forked accept. We map through the SAME `ddtree_dfs_inv_perm` the
        //      SSM commit uses (`commit_verify_state_async_with_slot`).
        //   2. OVERLAP. Once sources are permuted kernel slots, a source may be
        //      another move's destination; a naive in-place copy clobbers data a
        //      later move still needs. `plan_kv_compaction_moves` evacuates such
        //      conflicts through free tail verify slots first, so the whole
        //      relocation is a lossless permutation.
        //
        // Empty `inv_perm` (chain mode / DFS off) ⇒ identity map ⇒ this reduces
        // to the legacy ascending-dst plan (and is a no-op for a contiguous
        // path), so the flat path is byte-for-byte unchanged.
        use crate::layers::dflash_head::ddtree::plan_kv_compaction_moves;
        let inv_perm = self.ddtree_dfs_inv_perm.lock().clone();
        // Verify width (kernel-slot count). Under DFS reorder inv_perm has
        // exactly k entries; otherwise the general (scratch) path is never
        // taken, so any bound >= accepted_compact.len()+1 is safe. Use the
        // accepted-path max compact index as a floor so tail scratch exists.
        let k = if !inv_perm.is_empty() {
            inv_perm.len()
        } else {
            accepted_compact.iter().copied().max().unwrap_or(0) + 1
        };
        let planned = plan_kv_compaction_moves(accepted_compact, &inv_perm, pre_verify_len, k);
        let moves: Vec<(usize, usize)> = planned.iter().map(|m| (m.src_pos, m.dst_pos)).collect();
        if moves.is_empty() {
            return Ok(());
        }

        let stream = self.gpu.default_stream();
        let kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let num_layers = kv_cache.num_layers();

        for layer_idx in 0..num_layers {
            let dtype = kv_cache.dtype_for_layer(layer_idx);
            let (nkv, hd) = kv_cache.config().dims_for_layer(layer_idx);
            let elem_per_pos = nkv * hd;
            // Per-position byte stride for the contiguous element dtypes.
            let pos_bytes = match dtype {
                spark_runtime::kv_cache::KvCacheDtype::Bf16 => elem_per_pos * 2,
                spark_runtime::kv_cache::KvCacheDtype::Fp8 => elem_per_pos,
                other => {
                    bail!(
                        "compact_verify_kv: unsupported KV dtype {other:?} for layer {layer_idx} \
                         (tree-fork KV compaction only supports contiguous bf16/fp8); \
                         disable ATLAS_DFLASH_TREE_COMMIT or use --kv-cache-dtype fp8"
                    );
                }
            };

            for &(src_pos, dst_pos) in &moves {
                let src_block = seq
                    .physical_block_for(src_pos / bs)
                    .ok_or_else(|| anyhow::anyhow!("compact_verify_kv: src block evicted"))?;
                let dst_block = seq
                    .physical_block_for(dst_pos / bs)
                    .ok_or_else(|| anyhow::anyhow!("compact_verify_kv: dst block evicted"))?;
                let src_off = (src_pos % bs) * pos_bytes;
                let dst_off = (dst_pos % bs) * pos_bytes;

                let k_src = kv_cache.k_cache_ptr(layer_idx, src_block).offset(src_off);
                let k_dst = kv_cache.k_cache_ptr(layer_idx, dst_block).offset(dst_off);
                let v_src = kv_cache.v_cache_ptr(layer_idx, src_block).offset(src_off);
                let v_dst = kv_cache.v_cache_ptr(layer_idx, dst_block).offset(dst_off);

                self.gpu.copy_d2d_async(k_src, k_dst, pos_bytes, stream)?;
                self.gpu.copy_d2d_async(v_src, v_dst, pos_bytes, stream)?;
            }
        }
        drop(kv_cache);

        static COMPACT_DBG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = COMPACT_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < 8 {
            tracing::info!(
                "compact_verify_kv #{n}: pre_len={pre_verify_len} accepted_compact={:?} moves={:?}",
                accepted_compact,
                moves,
            );
        }
        Ok(())
    }
}
