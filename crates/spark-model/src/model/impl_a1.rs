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

impl TransformerModel {
    pub fn new(
        config: ModelConfig,
        embed_tokens: DenseWeight,
        final_norm: DenseWeight,
        lm_head_weight: DenseWeight,
        lm_head_nvfp4: Option<QuantizedWeight>,
        layers: Vec<Box<dyn TransformerLayer>>,
        buffers: BufferArena,
        kv_cache: PagedKvCache,
        mtp_weights: Vec<MtpWeights>,
        mtp_dense_weights: Option<crate::weight_map::MtpDenseWeights>,
        gpu: Box<dyn GpuBackend>,
        max_seq_len: usize,
        max_batch_size: usize,
        mtp_quant: crate::layers::MtpQuantization,
        use_speculative: bool,
        prefix_cache: Box<dyn spark_runtime::prefix_cache::PrefixCache>,
        mtp_vocab_size: u32,
        comm: Option<std::sync::Arc<dyn spark_comm::CommBackend>>,
        self_speculative: bool,
        num_drafts: usize,
        vision_encoder: Option<crate::layers::VisionEncoder>,
        ssm_cache_slots: usize,
        ssm_checkpoint_interval: usize,
    ) -> Result<Self> {
        let fp32_residual = config.use_fp32_residual();
        let rms_norm_kernel = if fp32_residual {
            gpu.kernel("norm", "rms_norm_f32")
                .or_else(|_| gpu.kernel("norm", "rms_norm"))?
        } else {
            gpu.kernel("norm", "rms_norm")?
        };
        let bf16_to_f32_kernel = if fp32_residual {
            gpu.kernel("residual_add", "bf16_to_f32")
                .unwrap_or(KernelHandle(0))
        } else {
            KernelHandle(0) // BF16 models don't need conversion
        };
        let dense_gemv_kernel = gpu.kernel("gemv", "dense_gemv_bf16")?;
        // FP32-output dense GEMV — only loaded when LM head needs FP32 logits.
        // For models that don't use FP32 residual, this stays KernelHandle(0)
        // and the BF16 path is taken. The kernel lives in the same `gemv`
        // module as `dense_gemv_bf16` so this lookup is cheap.
        let dense_gemv_fp32out_kernel = if fp32_residual {
            gpu.kernel("gemv", "dense_gemv_bf16_fp32out")
                .unwrap_or(KernelHandle(0))
        } else {
            KernelHandle(0)
        };
        let w4a16_gemv_kernel = gpu.kernel("w4a16_gemv", "w4a16_gemv")?;
        let w4a16_gemv_logits_kernel = gpu.kernel("w4a16_gemv", "w4a16_gemv_logits")?;
        let w4a16_gemm_kernel = gpu.kernel("w4a16", "w4a16_gemm")?;
        let w4a16_gemm_t_m32_n64_kernel =
            crate::layers::try_kernel(gpu.as_ref(), "w4a16", "w4a16_gemm_t_m32_n64");
        let w4a16_gemv_batch2_kernel = gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?;
        let w4a16_gemv_batch3_logits_kernel =
            gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3_logits")?;
        let dense_gemm_kernel = gpu.kernel("gemm", "dense_gemm_bf16")?;
        let argmax_kernel = gpu.kernel("argmax", "argmax_bf16")?;
        let argmax_logits_kernel = gpu.kernel("argmax", "argmax_fp32")?;
        let batched_embed_kernel = if fp32_residual {
            gpu.kernel("embed_from_argmax", "batched_embed_f32")
                .or_else(|_| gpu.kernel("embed_from_argmax", "batched_embed"))?
        } else {
            gpu.kernel("embed_from_argmax", "batched_embed")?
        };
        let fill_slots_kernel = gpu.kernel("metadata_fill", "fill_slots_from_block_table")?;
        let profile = std::env::var("ATLAS_PROFILE").is_ok();
        let profile_first = std::env::var("ATLAS_PROFILE_FIRST").is_ok();

        tracing::info!(
            "TransformerModel: {} layers, vocab={}, hidden={}{}{}",
            layers.len(),
            config.vocab_size,
            config.hidden_size,
            if profile { " [PROFILE MODE]" } else { "" },
            if profile_first {
                " [PROFILE_FIRST]"
            } else {
                ""
            },
        );

        // Build SSM state pool (with MTP intermediate/checkpoint pools only if speculative decoding enabled)
        // num_intermediates = K (per-token SSM h/conv state snapshots).
        // For MTP K=2/3/4 verify: K = num_drafts + 1.
        // For DFlash K=γ verify: K = γ + 1 (drafter's γ drafts + 1 verified bonus slot).
        // Pool size = max of both so DFlash and MTP can coexist on the same model.
        let dflash_kgamma = if !config.dflash_capture_layers.is_empty() {
            // `dflash_kgamma` is the verify token count T = γ+1 (the bonus
            // last-token slot + γ draft slots) — it sizes every verify-side
            // persistent buffer: parent_ids capacity (= kernel_parents.len()
            // = T, see trait_impl/mod.rs), tree_kv_indir stride, and the
            // tree_kv_pack num_seqs. The verify entry uses `k == capacity`
            // (verify_d.rs) where k = tokens.len() = T, so the capacity MUST
            // equal T, not γ.
            //
            // Drafter's γ is plumbed through `num_drafts`: for DFlash the
            // scheduler is configured with `num_drafts = γ - 1` (serve.rs /
            // build.rs), so T = γ+1 = num_drafts + 2. This MUST track the
            // ACTUAL γ, not a literal 17, or γ>16 OOBs these buffers. For the
            // canonical γ=16 run this evaluates to 15+2 = 17 (unchanged).
            num_drafts + 2
        } else {
            0
        };
        // ── DDTree wide-tree verify capacity (ATLAS_DDTREE_MAX_NODES) ──
        // The drafter emits γ draft positions, so the FLAT verify width is
        // `dflash_kgamma` (= γ+1). But a DDTree branch tree can hold MORE
        // nodes than γ (top-k siblings expanded per depth, up to a budget).
        // `ddtree_cap` sizes every verify-side PERSISTENT buffer (SSM
        // intermediates, parent_ids, tree-KV indirection, hidden-save) to
        // admit a wider tree. It is clamped to the tree-WY kernel's
        // compile-time K_MAX=32. Default = dflash_kgamma ⇒ NO behavior change
        // unless the operator opts in; the flat/counting path keeps verifying
        // exactly `dflash_kgamma` tokens. When widened, flat verify (k <
        // ddtree_cap) routes through the proven wy_k path (the persistent
        // parent injection at `k == capacity` no longer matches at the flat
        // width — validated bit-identical to the tree_wy-linear-chain path).
        const DDTREE_KERNEL_KMAX: usize = 32; // must match gated_delta_rule_tree_wy.cu K_MAX
        let ddtree_cap = if dflash_kgamma > 0 {
            std::env::var("ATLAS_DDTREE_MAX_NODES")
                .ok()
                .and_then(|s| s.trim().parse::<usize>().ok())
                .unwrap_or(dflash_kgamma)
                .clamp(dflash_kgamma, DDTREE_KERNEL_KMAX)
        } else {
            0
        };
        if ddtree_cap > dflash_kgamma {
            tracing::info!(
                "ATLAS_DDTREE_MAX_NODES: wide-tree verify capacity = {ddtree_cap} \
                 (flat width dflash_kgamma = {dflash_kgamma}, kernel K_MAX = {DDTREE_KERNEL_KMAX})"
            );
        }
        // DFlash needs the SSM verify pools regardless of MTP weight presence
        // or lm_head quantization — its K=γ verify path checkpoints SSM state
        // for partial-accept rollback. Force `has_mtp` on whenever DFlash is
        // active so the checkpoint pools exist.
        let has_mtp = self_speculative
            || (use_speculative && !mtp_weights.is_empty() && lm_head_nvfp4.is_some())
            || (use_speculative && mtp_dense_weights.is_some() && lm_head_nvfp4.is_some())
            || dflash_kgamma > 0;
        let num_intermediates = if has_mtp {
            // wy17 writes K-1 inter slots (final state in h_state) so dflash_kgamma
            // suffices for that path. M8A tree_wy / general-K verify writes ALL
            // T = γ+1 slots into intermediates because tree topology needs every
            // state addressable by token. With dflash_kgamma = T (= num_drafts+2),
            // `dflash_kgamma + 1` = T+1 keeps one slot of headroom so tree mode
            // never OOBs. For γ=16 this is 17+1 = 18 (unchanged from the prior
            // hardcoded `.max(17 + 1)`). The MTP branch (`num_drafts + 2`) covers
            // the K = num_drafts+1 verify when DFlash is inactive.
            // `ddtree_cap + 1` keeps one slot of headroom so wide-tree mode
            // (which writes all T tree slots into intermediates) never OOBs.
            (num_drafts + 2).max(ddtree_cap + 1)
        } else {
            0
        };
        let ssm_pool = SsmStatePool::new(
            &config,
            max_batch_size,
            has_mtp,
            num_intermediates,
            gpu.as_ref(),
        )?;

        // M8A graph-safe persistent parent_ids buffer.
        //
        // The K=γ verify CUDA graph reads parent_ids from a FIXED device
        // pointer baked into the captured kernel arguments. Allocating a
        // fresh scratch buffer per verify (the pre-fix behavior in
        // set_ddtree_parent_ids) made the captured pointer stale on
        // replay → kernel read garbage. We allocate ONCE here and let
        // both default (linear-chain) and tree-mode payloads share the
        // same address, mutating only the contents.
        //
        // Capacity = dflash_kgamma (17 on qwen3.6-27b). Tree-mode upload
        // never exceeds K=γ tokens. Pre-stamped with the linear chain
        // `[-1, 0, 1, ..., K-2]` so flat-payload verify reuses tree_wy
        // without an upload (zero-copy bit-equivalence path).
        let (parent_ids_persistent, parent_ids_capacity) = if dflash_kgamma > 0 {
            let cap = ddtree_cap;
            let bytes = cap * std::mem::size_of::<i32>();
            let buf = gpu.alloc(bytes)?;
            // Stamp the linear-chain default: parents[0] = -1, parents[i] = i-1.
            let mut chain = Vec::<i32>::with_capacity(cap);
            chain.push(-1);
            for i in 1..cap {
                chain.push((i - 1) as i32);
            }
            let byte_view = unsafe {
                std::slice::from_raw_parts(chain.as_ptr() as *const u8, bytes)
            };
            gpu.copy_h2d_async(byte_view, buf, gpu.default_stream())?;
            gpu.synchronize(gpu.default_stream())?;
            tracing::info!(
                "M8A: allocated graph-safe parent_ids buffer ({cap} slots, ptr={:?})",
                buf
            );
            (buf, cap)
        } else {
            (DevicePtr::NULL, 0usize)
        };

        // ATLAS_TREE_AWARE_ATTN persistent indirection buffer.
        //
        // For K=γ verify, when tree-mode is active and the user opted in via
        // `ATLAS_TREE_AWARE_ATTN=1`, verify_d.rs builds a per-row indirection
        // table that maps each query's compact tree slot back to its ancestor
        // chain. Layout: `[num_rows=dflash_kgamma, stride=dflash_kgamma]` i32,
        // row-major. The kernel reads `indir[seq_idx*stride + (pos - base)]`
        // to remap positions in the tree window.
        //
        // Pre-filled with identity (`indir[t][j] = j`) so a stale-but-graph-
        // captured pointer always behaves like chain-mode if someone forgets
        // to update it before launch.
        let (tree_kv_indir_persistent, tree_kv_indir_stride) = if dflash_kgamma > 0 {
            let stride = ddtree_cap;
            let cells = stride * stride;
            let bytes = cells * std::mem::size_of::<i32>();
            let buf = gpu.alloc(bytes)?;
            let mut identity = Vec::<i32>::with_capacity(cells);
            for _row in 0..stride {
                for j in 0..stride {
                    identity.push(j as i32);
                }
            }
            let byte_view = unsafe {
                std::slice::from_raw_parts(identity.as_ptr() as *const u8, bytes)
            };
            gpu.copy_h2d_async(byte_view, buf, gpu.default_stream())?;
            gpu.synchronize(gpu.default_stream())?;
            tracing::info!(
                "ATLAS_TREE_AWARE_ATTN: allocated indirection buffer ({}x{} i32, ptr={:?})",
                stride, stride, buf
            );
            (buf, stride)
        } else {
            (DevicePtr::NULL, 0usize)
        };

        // ATLAS_TREE_AWARE_ATTN CUDA graph fix: persistent 1×i32 device buffer
        // holding the current tree-window base position. The kernel reads this
        // via pointer arg so a captured graph sees the fresh value on each
        // replay. Pre-stamped to 0 (matches the chain-mode default); the host
        // writes the current `seq.seq_len` here before each K=γ verify step.
        //
        // A pinned-host shadow buffer is also allocated so the per-step H2D
        // upload establishes a proper stream-ordered dependency for the
        // captured graph kernels. A pageable-source (Vec-backed) async copy
        // can race ahead of the kernel launch on graph replay paths.
        let (tree_kv_indir_base_persistent, tree_kv_indir_base_host_pinned) =
            if dflash_kgamma > 0 {
                let buf = gpu.alloc(std::mem::size_of::<i32>())?;
                gpu.memset_async(buf, 0, std::mem::size_of::<i32>(), gpu.default_stream())?;
                gpu.synchronize(gpu.default_stream())?;
                let host_ptr = gpu.alloc_host_pinned(std::mem::size_of::<i32>())?;
                // Pre-zero the pinned shadow so any pre-tree verify (rare) is
                // consistent with the device buffer's zero-init.
                unsafe {
                    std::ptr::write_bytes(host_ptr, 0u8, std::mem::size_of::<i32>());
                }
                tracing::info!(
                    "ATLAS_TREE_AWARE_ATTN: allocated kv_indir_base buffer (1xi32, dev={:?}, host_pinned={:?})",
                    buf, host_ptr
                );
                (buf, host_ptr)
            } else {
                (DevicePtr::NULL, std::ptr::null_mut::<u8>())
            };

        // ATLAS_TREE_KV_PACK: per-attention-layer packed-KV scratch pool.
        //
        // When `ATLAS_TREE_KV_PACK=1` + `ATLAS_TREE_AWARE_ATTN=1`, the K=γ
        // verify path scatters ancestor KV from the paged cache into a
        // contiguous `[num_seqs × stride]` block per layer, then re-runs
        // paged_decode_attn against the scratch with NULL indirection so
        // the fast BC=4 batched path fires (vs the 3.6x-slower
        // single-position fallback inside the indirected kernel).
        //
        // Sizing (qwen3.6-27b, FP8, stride=17, num_seqs=17, nkv=4, hd=128):
        //   per-layer K bytes = num_seqs * stride * nkv * hd * elem_bytes
        //                     = 17 * 17 * 4 * 128 * 1 = 147,968 B (~145 KB)
        //   per-layer (K+V)   = ~290 KB
        //   16 attn layers    = ~4.6 MB total (negligible).
        //
        // NVFP4: per-block = data + scale section, similar order of magnitude.
        let tree_kv_pack_active_env = std::env::var("ATLAS_TREE_KV_PACK").ok().as_deref() == Some("1");
        let (
            tree_kv_pack_scratch_k,
            tree_kv_pack_scratch_v,
            tree_kv_pack_block_table,
            tree_kv_pack_seq_lens,
            tree_kv_pack_block_stride_bytes,
            tree_kv_pack_data_section_bytes,
            tree_kv_pack_scatter_fp8_kernel,
            tree_kv_pack_scatter_nvfp4_kernel,
            tree_kv_pack_active,
        ) = if tree_kv_pack_active_env && dflash_kgamma > 0 {
            let num_attn = config.num_attention_layers();
            // num_seqs sized by k_max (= γ+1). Conservative; matches the K=γ
            // verify batch dimension. The scatter kernel only writes the rows
            // actually used per step.
            let num_seqs = ddtree_cap;
            let stride = ddtree_cap; // = max_chain_len (wide-tree capacity)
            let kv_dtype = kv_cache.dtype();
            // Compute per-block bytes following the same layout the
            // paged_decode_attn kernels expect for the given dtype.
            let nkv = config.num_key_value_heads;
            let hd = config.head_dim;
            let (block_bytes, data_section_bytes) = match kv_dtype {
                spark_runtime::kv_cache::KvCacheDtype::Nvfp4 => {
                    // data + per-group FP8 scales (group size 16)
                    let elems = stride * nkv * hd;
                    let data = elems / 2;
                    let scales = elems / 16;
                    ((data + scales) as u64, data as u64)
                }
                spark_runtime::kv_cache::KvCacheDtype::Fp8 => {
                    let elems = stride * nkv * hd;
                    (elems as u64, 0u64)
                }
                _ => {
                    // Other dtypes (BF16, Turbo*) not yet wired for pack.
                    (0u64, 0u64)
                }
            };
            if block_bytes == 0 {
                tracing::warn!(
                    "ATLAS_TREE_KV_PACK requested but KV dtype {:?} not supported — disabling",
                    kv_dtype
                );
                (
                    Vec::new(),
                    Vec::new(),
                    DevicePtr::NULL,
                    DevicePtr::NULL,
                    0u64,
                    0u64,
                    KernelHandle(0),
                    KernelHandle(0),
                    false,
                )
            } else {
                let pool_bytes_per_layer = block_bytes as usize * num_seqs;
                let mut scratch_k = Vec::with_capacity(num_attn);
                let mut scratch_v = Vec::with_capacity(num_attn);
                for _ in 0..num_attn {
                    scratch_k.push(gpu.alloc(pool_bytes_per_layer)?);
                    scratch_v.push(gpu.alloc(pool_bytes_per_layer)?);
                }
                // Identity block table: bt[seq] = seq (i32). Each row needs
                // `max_blocks_per_seq` entries (the kernel multiplies seq_idx
                // by it), so we allocate `num_seqs * max_blocks_per_seq` slots
                // and fill only column 0; trailing columns stay zero (only
                // read when seq_lens overflows stride, which never happens in
                // pack mode because the packed seq_len ≤ stride).
                //
                // For the packed path, max_blocks_per_seq = 1 (only one
                // synthetic block per seq). The consumer kernel passes this
                // 1 in instead of the real max_blocks_per_seq.
                let bt_entries = num_seqs;
                let bt_bytes = bt_entries * std::mem::size_of::<i32>();
                let bt_buf = gpu.alloc(bt_bytes)?;
                let identity: Vec<i32> = (0..num_seqs as i32).collect();
                let bt_view = unsafe {
                    std::slice::from_raw_parts(identity.as_ptr() as *const u8, bt_bytes)
                };
                gpu.copy_h2d_async(bt_view, bt_buf, gpu.default_stream())?;

                let seq_lens_bytes = num_seqs * std::mem::size_of::<i32>();
                let seq_lens_buf = gpu.alloc(seq_lens_bytes)?;
                gpu.memset_async(seq_lens_buf, 0, seq_lens_bytes, gpu.default_stream())?;
                gpu.synchronize(gpu.default_stream())?;

                let scatter_fp8 = gpu
                    .kernel("tree_kv_scatter", "tree_kv_scatter_fp8")
                    .unwrap_or_else(|e| {
                        tracing::warn!("tree_kv_scatter_fp8 not loadable: {e}");
                        KernelHandle(0)
                    });
                let scatter_nvfp4 = gpu
                    .kernel("tree_kv_scatter", "tree_kv_scatter_nvfp4")
                    .unwrap_or_else(|e| {
                        tracing::warn!("tree_kv_scatter_nvfp4 not loadable: {e}");
                        KernelHandle(0)
                    });
                let active = match kv_dtype {
                    spark_runtime::kv_cache::KvCacheDtype::Fp8 => scatter_fp8.0 != 0,
                    spark_runtime::kv_cache::KvCacheDtype::Nvfp4 => scatter_nvfp4.0 != 0,
                    _ => false,
                };
                tracing::info!(
                    "ATLAS_TREE_KV_PACK: allocated {} attn-layer scratch pools ({} bytes each K/V), \
                     stride={}, num_seqs={}, dtype={:?}, active={}",
                    num_attn, pool_bytes_per_layer, stride, num_seqs, kv_dtype, active
                );
                (
                    scratch_k,
                    scratch_v,
                    bt_buf,
                    seq_lens_buf,
                    block_bytes,
                    data_section_bytes,
                    scatter_fp8,
                    scatter_nvfp4,
                    active,
                )
            }
        } else {
            (
                Vec::new(),
                Vec::new(),
                DevicePtr::NULL,
                DevicePtr::NULL,
                0u64,
                0u64,
                KernelHandle(0),
                KernelHandle(0),
                false,
            )
        };

        // Marconi SSM snapshot pool for prefix caching.
        // PR #74 added decode_ring_slots + decode_max_seqs args for the
        // Phase-C decode-rollback region. We set both to 0 (decode rollback
        // disabled) — Marconi caching still works the same.
        let ssm_snapshots = SsmSnapshotPool::new(
            ssm_cache_slots,
            ssm_pool.h_bytes,
            ssm_pool.conv_bytes,
            ssm_pool.num_ssm_layers,
            0,  // decode_ring_slots
            0,  // decode_max_seqs
            gpu.as_ref(),
        )?;
        if ssm_checkpoint_interval > 0 && ssm_cache_slots > 0 {
            tracing::info!(
                "Marconi intermediate checkpoints: every {} blocks ({} tokens at block_size={})",
                ssm_checkpoint_interval,
                ssm_checkpoint_interval * kv_cache.block_size(),
                kv_cache.block_size(),
            );
        }

        // Fixed metadata stride for CUDA graph compatibility
        let max_blocks_per_seq = (max_seq_len / kv_cache.block_size() + 1) as u32;

        // Permanent dummy KV block for padding sequences. Must be explicitly
        // zeroed: `gpu.alloc()` returns uninitialized memory, and any kernel
        // OOB-read (now routed here via the sentinel block_table_flat default
        // fill in upload_batch_metadata_*) would otherwise dequant random
        // bytes and inject garbage into attention scores.
        let mut kv_cache = kv_cache;
        let dummy_kv_block = kv_cache.alloc_block()?;
        kv_cache.zero_block(dummy_kv_block, gpu.as_ref(), gpu.default_stream())?;
        gpu.synchronize(gpu.default_stream())?;

        // ATLAS_LM_HEAD_T=1: transposed NVFP4 lm_head copy so the K=γ
        // verify lm_head routes through w4a16_gemm_t_m32_n64 (single
        // coalesced B read at M ≤ 32) instead of the strided plain
        // w4a16_gemm (~5× off the bandwidth floor at M=17, ~15 ms/step
        // on qwen3.6-27b's 248k vocab). ~0.63 GB extra device memory.
        let lm_head_nvfp4_t = if std::env::var("ATLAS_LM_HEAD_T").ok().as_deref() == Some("1")
            && w4a16_gemm_t_m32_n64_kernel.0 != 0
        {
            match lm_head_nvfp4.as_ref() {
                Some(w) => match w.transpose_for_gemm(gpu.as_ref(), config.vocab_size, config.hidden_size) {
                    Ok(t) => {
                        tracing::info!("lm_head NVFP4-T built for K=γ m32 path (vocab={})", config.vocab_size);
                        Some(t)
                    }
                    Err(e) => {
                        tracing::warn!("lm_head transpose failed ({e:#}); K=γ lm_head stays on plain w4a16_gemm");
                        None
                    }
                },
                None => None,
            }
        } else {
            None
        };

        // Build MTP proposer (extracted to keep `new` under the file cap).
        let proposer: Option<Arc<dyn DraftProposer>> = super::impl_a1_init::build_mtp_proposer(
            use_speculative,
            mtp_weights,
            mtp_dense_weights,
            embed_tokens,
            lm_head_nvfp4,
            &config,
            gpu.as_ref(),
            mtp_quant,
            mtp_vocab_size,
            max_seq_len,
        );

        if self_speculative {
            let num_ssm = config.num_ssm_layers();
            let num_attn = config.num_attention_layers();
            tracing::info!(
                "Self-speculative decoding: ENABLED (skipping {} SSM layers, keeping {} attention layers)",
                num_ssm,
                num_attn,
            );
        }

        // MTP hidden state save buffer (1 × hidden_size FP32)
        let mtp_hidden_save = gpu.alloc(config.hidden_size * 4)?;

        // Last-K prompt-tail target hidden capture buffer for MTP prefill.
        // Gated by ATLAS_MTP_LASTK_PREFILL=N (N>0 enables, default 0 disabled).
        // Sized at fp32 width for safety regardless of `use_fp32_residual()`.
        let mtp_lastk_capacity: usize = std::env::var("ATLAS_MTP_LASTK_PREFILL")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        let mtp_lastk_buf = if mtp_lastk_capacity > 0 && proposer.is_some() {
            tracing::info!(
                "MTP last-K prefill: ENABLED (K={mtp_lastk_capacity} tokens, \
                 {} KiB hidden capture buffer)",
                mtp_lastk_capacity * config.hidden_size * 4 / 1024,
            );
            Some(gpu.alloc(mtp_lastk_capacity * config.hidden_size * 4)?)
        } else {
            None
        };

        // DFlash 5-layer hidden-state stack. Allocated when a
        // BlockDiffusionDraftHead is the active proposer (`config.dflash_capture_layers`
        // populated by the loader from the drafter's `dflash_config.target_layer_ids`).
        // Size: k_max × N_capture × hidden_size × bf16 (17 × 5 × 2048 × 2 = 348 KB).
        // k_max = γ+1 = 17 covers all verify paths (K=2..4 and DFlash γ=16).
        //
        // When ATLAS_DUMP_HIDDEN is set we still force a capture even if the
        // DFlash proposer isn't active, so training data can be collected
        // off the production K=3 MTP path. Layer-selection precedence:
        //
        //   1. ATLAS_DUMP_LAYERS env var (comma-separated, eg "1,16,31,46,61")
        //      — explicit override, use when retraining a drafter with a
        //      different layer spec than the canonical formula.
        //
        //   2. Derived from `num_hidden_layers` via the canonical 5-layer
        //      DFlash spacing: `[1, 1+s, 1+2s, 1+3s, 1+4s]` where
        //      `s = (N - 4) / 4`. This matches the z-lab DFlash drafter
        //      family's `target_layer_ids` across all observed targets:
        //      - 40-layer 35B-A3B-abl: s=9   → [1, 10, 19, 28, 37]
        //      - 64-layer AEON-Q36-27B: s=15 → [1, 16, 31, 46, 61]
        //
        // The previous patch (2026-05-09) hardcoded the 40-layer indices —
        // a stale value left over from the qwen3.6-35b-a3b-abl session
        // that silently mis-captured hiddens on AEON-27B (training data
        // from those dumps was unusable for the z-lab Qwen3.6-27B drafter
        // which expects [1, 16, 31, 46, 61]).
        let mut dflash_capture_layers: Vec<usize> = config.dflash_capture_layers.clone();
        if dflash_capture_layers.is_empty() && std::env::var("ATLAS_DUMP_HIDDEN").is_ok() {
            let (layers, source) = if let Ok(raw) = std::env::var("ATLAS_DUMP_LAYERS") {
                let parsed: Vec<usize> = raw
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
                if parsed.is_empty() {
                    return Err(anyhow::anyhow!(
                        "ATLAS_DUMP_LAYERS set but parsed to empty list: {:?}",
                        raw
                    ));
                }
                (parsed, "ATLAS_DUMP_LAYERS")
            } else {
                let n = config.num_hidden_layers;
                if n < 5 {
                    return Err(anyhow::anyhow!(
                        "ATLAS_DUMP_HIDDEN: cannot derive 5 capture layers from \
                         num_hidden_layers={} (need ≥ 5). Set ATLAS_DUMP_LAYERS \
                         explicitly.",
                        n
                    ));
                }
                // s = floor((N - 4) / 4). For N ≥ 5 this gives s ≥ 0.
                // Range check below catches N = 5/6/7/8 where indices would
                // collide; require N ≥ 12 for a useful 5-point spacing.
                let s = (n.saturating_sub(4)) / 4;
                if s == 0 || 1 + 4 * s >= n {
                    return Err(anyhow::anyhow!(
                        "ATLAS_DUMP_HIDDEN: derived spacing s={} from N={} is \
                         degenerate. Set ATLAS_DUMP_LAYERS explicitly.",
                        s, n
                    ));
                }
                let derived = vec![1, 1 + s, 1 + 2 * s, 1 + 3 * s, 1 + 4 * s];
                (derived, "derived from num_hidden_layers")
            };
            // Sanity: all indices must be in-range for this model.
            for &li in &layers {
                if li >= config.num_hidden_layers {
                    return Err(anyhow::anyhow!(
                        "ATLAS_DUMP_HIDDEN: capture layer {} out of range \
                         (num_hidden_layers={})",
                        li, config.num_hidden_layers
                    ));
                }
            }
            tracing::info!(
                "ATLAS_DUMP_HIDDEN set: dflash_capture_layers = {:?} \
                 (source: {}, num_hidden_layers={})",
                layers, source, config.num_hidden_layers
            );
            dflash_capture_layers = layers;
        }
        let dflash_hidden_save = if dflash_capture_layers.is_empty() {
            None
        } else {
            let n = dflash_capture_layers.len();
            // Max verify width T = γ+1 (= dflash_kgamma): the K=γ verify
            // captures hidden state for ALL T input rows via
            // `try_dflash_capture(token_idx=0..T)`. Hardcoding 17 (γ=16)
            // OOB-writes this buffer for γ>16 → GPU illegal-memory crash.
            // Track the real γ so γ=20/24/32 size correctly.
            let k_max = dflash_kgamma.max(ddtree_cap).max(17);
            Some(gpu.alloc(k_max * n * config.hidden_size * 2)?)
        };

        // EP command buffer for token broadcast (4 bytes, u32)
        let ep_cmd_buf = gpu.alloc(4)?;

        // Secondary stream + event for pipelining checkpoint D2D with MTP propose.
        let secondary_stream = gpu.create_stream()?;
        let secondary_event = gpu.create_event()?;

        // EP: register moe_output buffer with NCCL and provide bf16_add kernel.
        if let Some(ref comm) = comm
            && comm.world_size() == 2
        {
            let moe_ptr = buffers.moe_output().0;
            let moe_bytes = buffers.sizes().moe_output;
            match comm.register_buffer(moe_ptr, moe_bytes) {
                Ok(_) => tracing::info!("Registered moe_output ({moe_bytes} B) with NCCL"),
                Err(e) => tracing::warn!("ncclCommRegister moe_output failed (non-fatal): {e}"),
            }
            match gpu.kernel("bf16_add", "bf16_add_inplace") {
                Ok(k) => comm.set_add_kernel(k.0),
                Err(e) => {
                    tracing::warn!("bf16_add_inplace kernel not found (send/recv disabled): {e}")
                }
            }
        }

        // Allocate pinned host staging buffer for batched metadata H2D.
        let pinned_bytes = buffers.sizes().scratch.max(64 * 1024);
        let pinned_ptr = gpu.alloc_host_pinned(pinned_bytes)?;
        tracing::info!("Pinned metadata staging: {} KB", pinned_bytes / 1024);
        let max_batch_tokens = buffers.max_batch_tokens();
        let pinned_staging = std::cell::UnsafeCell::new(PinnedMetaStaging {
            ptr: pinned_ptr,
            bytes: pinned_bytes,
            positions: Vec::with_capacity(max_batch_tokens),
            positions_h: Vec::with_capacity(max_batch_tokens),
            positions_w: Vec::with_capacity(max_batch_tokens),
            slots: Vec::with_capacity(max_batch_tokens),
        });

        // SSM state normalization kernel + pointer buffer (for chunked prefill).
        let ssm_norm_k = gpu
            .kernel("ssm_state_norm", "ssm_state_clamp_norm_fused")
            .unwrap_or(KernelHandle(0));

        // Logit softcapping (Gemma-4: cap=30.0). Only load if model uses it.
        let logit_softcap_kernel = if config.final_logit_softcapping > 0.0 {
            gpu.kernel("logit_softcap", "logit_softcap_bf16")
                .unwrap_or_else(|e| {
                    tracing::warn!("logit_softcap kernel not found: {e}");
                    KernelHandle(0)
                })
        } else {
            KernelHandle(0)
        };
        // FP32 softcap variant — only loaded when both softcap and FP32
        // residual are active (i.e. Gemma-4 dense). Other models keep the
        // BF16 softcap (or no softcap at all).
        let logit_softcap_fp32_kernel = if config.final_logit_softcapping > 0.0 && fp32_residual {
            gpu.kernel("logit_softcap", "logit_softcap_fp32")
                .unwrap_or_else(|e| {
                    tracing::warn!("logit_softcap_fp32 kernel not found: {e}");
                    KernelHandle(0)
                })
        } else {
            KernelHandle(0)
        };
        // FP32 logits gate. The LM head produces FP32 (rather than BF16)
        // logits when the residual stream is FP32 AND the LM head is a
        // dense BF16 weight (no NVFP4 quant). NVFP4 LM heads keep their
        // existing path because that quantization is a much larger
        // precision floor than the BF16 store; FP32 wouldn't help there.
        // Today this only affects Gemma-4 dense (model_type=="gemma4",
        // num_experts==0, tied BF16 embed→lm_head).
        // Gemma-4-31B FP32 lm_head experiment. Disabled by default —
        // session 2026-05-01 verified the BF16 lm_head store is NOT the
        // source of Gemma-4's haiku argmax flip: FP32 view of step-1
        // logits keeps top1=` a` (21.85), top2=` waves` (21.706) — same
        // 0.14-margin tiebreak as BF16. The drift is upstream in attention
        // or MLP, not in the lm_head precision boundary. Code paths kept
        // wired so a future bisection (Phase 2 of the plan) can re-enable
        // via `ATLAS_GEMMA4_FP32_LMHEAD=1`. Keep `use_fp32_logits=false`
        // by default so the rest of the model behaves identically to the
        // pre-fix BF16 path on every model family.
        // FP32 lm_head + softcap. Default OFF — empirically the gain on
        // Gemma-4-31B is marginal (Creative occasionally cleaner; fib still
        // fails the same broken-indentation pattern) but the cost is huge:
        // FP32 forces host-side sampling (vocab=262144 × 4 bytes per
        // decode step → ~1 MB D2H per token) which crushes decode TPS
        // from ~35 tok/s to ~6 tok/s on Gemma-4-31B. Not worth it without
        // a GPU-side FP32 argmax kernel. `ATLAS_GEMMA4_FP32_LMHEAD=1`
        // re-enables for bisection / future work.
        //
        // The earlier "FP32 doesn't fix haiku" comment in this file was
        // arrived at via incomplete bisection (the scheduler readback
        // always assumed BF16 — see commit 16b2f3a's commit body). The
        // 2026-05-01 evening run with the dispatch wired confirmed the
        // bisection's *qualitative* conclusion: FP32 lm_head + softcap
        // doesn't materially fix Gemma-4's structural NVFP4 attention
        // drift on greedy code generation. Fix is upstream of lm_head.
        let env_override = std::env::var("ATLAS_GEMMA4_FP32_LMHEAD").ok();
        let fp32_requested = matches!(env_override.as_deref(), Some("1") | Some("true"));
        let use_fp32_logits = fp32_requested
            && fp32_residual
            && ((lm_head_nvfp4.is_none() && dense_gemv_fp32out_kernel.0 != 0)
                || (lm_head_nvfp4.is_some() && w4a16_gemv_logits_kernel.0 != 0));
        // Dedicated FP32 logits scratch — only the single-token decode path
        // uses it. Prefill and batched-decode lm_head still write BF16 to the
        // shared `buffers.logits()`. Sized for one row of `vocab_size` FP32.
        let logits_fp32_buf = if use_fp32_logits {
            let bytes = config.vocab_size * 4;
            let p = gpu.alloc(bytes)?;
            tracing::info!(
                "FP32 LM head + softcap active (model_type={}, vocab={}). \
                 Decode logits scratch: {} bytes.",
                config.model_type,
                config.vocab_size,
                bytes,
            );
            p
        } else {
            DevicePtr::NULL
        };

        // Embedding scale (Gemma-4: sqrt(hidden_size)). Only load if model uses it.
        let embed_scale_kernel = if config.embed_scale > 0.0 {
            gpu.kernel("embed_scale", "bf16_scale_inplace")
                .unwrap_or_else(|e| {
                    tracing::warn!("embed_scale kernel not found: {e}");
                    KernelHandle(0)
                })
        } else {
            KernelHandle(0)
        };
        if config.embed_scale > 0.0 {
            tracing::info!(
                "Embedding scale: {:.4} (sqrt({}))",
                config.embed_scale,
                config.hidden_size
            );
        }
        let ssm_norm_ptrs = if ssm_pool.num_ssm_layers > 0 {
            gpu.alloc(ssm_pool.num_ssm_layers * 8)
                .unwrap_or(DevicePtr::NULL)
        } else {
            DevicePtr::NULL
        };

        // GDN prefill buffers: sized for max_batch_tokens (the prefill chunk size),
        // NOT max_seq_len. For prompts longer than this, prefill_twophase falls back
        // to standard chunked prefill which carries h_state/conv_state between chunks.
        // The GDN recurrence is sequential anyway, so chunking is mathematically identical.
        let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let nv = config.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;
        // GDN buffers only needed when GDN linear attention layers exist
        // (conv_dim > 0). Mamba-2 models (Nemotron) have conv_dim=0 — skip alloc
        // to avoid cuMemAlloc(0) error.
        let gdn_buf_len = max_batch_tokens.min(max_seq_len);
        let (gdn_qkv, gdn_gate_beta, gdn_out, gdn_z) = if conv_dim > 0 {
            let qkv = gpu.alloc(gdn_buf_len * conv_dim * 2)?;
            let gb = gpu.alloc(gdn_buf_len * nv * 2 * 4)?;
            let o = gpu.alloc(gdn_buf_len * value_dim * 2)?;
            let z = gpu.alloc(gdn_buf_len * value_dim * 2)?;
            let total_mb =
                (gdn_buf_len * (conv_dim * 2 + nv * 2 * 4 + value_dim * 2 * 2)) / (1024 * 1024);
            tracing::info!(
                "GDN prefill buffers: {total_mb} MB for {gdn_buf_len} tokens (chunked SSM prefill)"
            );
            (qkv, gb, o, z)
        } else {
            (
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
            )
        };

        // FP8 calibration only runs when the cache is actually FP8 — the
        // observe() call in decode.rs sits inside the FP8 cache branch. For
        // BF16 or NVFP4 caches the MODEL.toml fp8_kv_calibration_tokens
        // value is dead code and must not suppress CUDA graphs.
        let has_fp8_calibration = config.fp8_kv_calibration_tokens > 0
            && kv_cache.dtype() == spark_runtime::kv_cache::KvCacheDtype::Fp8;
        Ok(Self {
            config,
            ddtree_parent_ids_dev: Mutex::new(None),
            ddtree_num_tree_tokens: Mutex::new(0),
            ddtree_parent_ids_persistent: parent_ids_persistent,
            ddtree_parent_ids_capacity: parent_ids_capacity,
            ddtree_parent_ids_host: Mutex::new(Vec::new()),
            ddtree_dfs_inv_perm: Mutex::new(Vec::new()),
            tree_kv_indir_persistent,
            tree_kv_indir_stride,
            tree_kv_indir_base_persistent,
            tree_kv_indir_base_host_pinned,
            tree_kv_pack_scratch_k,
            tree_kv_pack_scratch_v,
            tree_kv_pack_block_table,
            tree_kv_pack_seq_lens,
            tree_kv_pack_block_stride_bytes,
            tree_kv_pack_data_section_bytes,
            tree_kv_pack_scatter_fp8_kernel,
            tree_kv_pack_scatter_nvfp4_kernel,
            tree_kv_pack_active,
            embed_tokens,
            final_norm,
            lm_head_weight,
            lm_head_nvfp4,
            lm_head_nvfp4_t,
            layers,
            buffers,
            kv_cache: Mutex::new(kv_cache),
            gpu,
            rms_norm_kernel,
            bf16_to_f32_kernel,
            dense_gemv_kernel,
            dense_gemv_fp32out_kernel,
            w4a16_gemv_kernel,
            w4a16_gemv_logits_kernel,
            w4a16_gemm_kernel,
            w4a16_gemm_t_m32_n64_kernel,
            w4a16_gemv_batch2_kernel,
            w4a16_gemv_batch3_logits_kernel,
            dense_gemm_kernel,
            argmax_kernel,
            argmax_logits_kernel,
            batched_embed_kernel,
            fill_slots_kernel,
            decode_graph: Mutex::new(std::collections::HashMap::new()),
            batch_decode_graphs: Mutex::new(HashMap::new()),
            // Suppress graphs during FP8 calibration only. MLA used to be
            // suppressed because an internal sync was placed inside the graph
            // capture region — that sync is now conditional on eager mode
            // (see line ~3881), so graphs work for MLA too. The zero_all call
            // at line ~3751 runs in Phase 1 BEFORE begin_capture, so it is
            // naturally outside the captured region.
            suppress_graphs: std::sync::atomic::AtomicBool::new(
                has_fp8_calibration
                    || std::env::var("ATLAS_DIAG_GEMMA4").is_ok_and(|v| v == "1" || v == "true")
                    || std::env::var("ATLAS_DUMP_HIDDEN").is_ok(),
            ),
            ssm_pool,
            ssm_snapshots,
            max_blocks_per_seq,
            dummy_kv_block,
            profile,
            profile_first_pending: std::sync::atomic::AtomicBool::new(profile_first),
            proposer,
            mtp_hidden_save,
            mtp_lastk_buf,
            mtp_lastk_capacity,
            dflash_hidden_save,
            dflash_capture_layers,
            verify2_graph: Mutex::new(std::collections::HashMap::new()),
            verify3_graph: Mutex::new(std::collections::HashMap::new()),
            verify4_graph: Mutex::new(std::collections::HashMap::new()),
            verify_kgamma_graph: Mutex::new(std::collections::HashMap::new()),
            prefix_cache,
            secondary_stream,
            secondary_event,
            comm,
            ep_cmd_buf,
            self_speculative,
            last_mtp_hidden_idx: std::sync::atomic::AtomicUsize::new(0),
            vision_encoder,
            vision_embed_patches: Mutex::new(0),
            vision_image_grids: Mutex::new(Vec::new()),
            vision_cache_fp: std::sync::atomic::AtomicU64::new(0),
            vision_cache_grids: Mutex::new(Vec::new()),
            vision_cache_buf: Mutex::new(spark_runtime::gpu::DevicePtr::NULL),
            vision_cache_bytes: std::sync::atomic::AtomicUsize::new(0),
            pinned_staging,
            ssm_checkpoint_interval,
            ssm_state_norm_kernel: ssm_norm_k,
            ssm_norm_ptrs_buf: ssm_norm_ptrs,
            gdn_buf_qkv: gdn_qkv,
            gdn_buf_gate_beta: gdn_gate_beta,
            gdn_buf_out: gdn_out,
            gdn_buf_z: gdn_z,
            gdn_buf_max_len: gdn_buf_len,
            logit_softcap_kernel,
            logit_softcap_fp32_kernel,
            use_fp32_logits,
            logits_fp32_buf,
            embed_scale_kernel,
        })
    }
}
