// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen4 sparse-attention index side branch.
//!
//! The first implementation is fail-closed: C=1, BF16 main KV, compression
//! ratio four, and the released 4x128/1x128 indexer geometry. Side-cache rows
//! are keyed by main paged-KV physical blocks, preserving block ownership.

use std::sync::OnceLock;

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use crate::layer::AttnMetadataDev;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

const INDEX_HEADS: u32 = 4;
const INDEX_DIM: u32 = 128;
const INDEX_WIDTH: u32 = 640;
const COMPRESS_RATIO: usize = 4;
const OUTPUT_WIDTH: usize = 2051;
const QUERY_HEADS: usize = 24;
const SPARSE_SPLITS: usize = 8;

pub fn compute_yarn_inv_freq(
    config: &atlas_core::config::ModelConfig,
    rotary_dim: usize,
    gpu: &dyn GpuBackend,
) -> Result<DevicePtr> {
    ensure!(config.yarn_factor > 0.0, "YaRN factor must be positive");
    let factor = config.yarn_factor;
    let beta_fast = config.yarn_beta_fast.max(32.0);
    let beta_slow = if config.yarn_beta_slow > 0.0 {
        config.yarn_beta_slow
    } else {
        1.0
    };
    let original = config.yarn_original_max_position_embeddings as f32;
    ensure!(
        original > 0.0 && rotary_dim >= 2,
        "invalid Qwen4 YaRN geometry"
    );
    let dim = rotary_dim as f32;
    let theta = config.rope_theta as f32;
    let correction = |rotations: f32| {
        (dim * (original / (rotations * 2.0 * std::f32::consts::PI)).ln()) / (2.0 * theta.ln())
    };
    let low = correction(beta_fast).floor().max(0.0);
    let high = correction(beta_slow).ceil().min((rotary_dim - 1) as f32);
    let denom = if (high - low).abs() < 1e-6 {
        0.001
    } else {
        high - low
    };
    let mut table = Vec::with_capacity(rotary_dim / 2);
    for j in 0..rotary_dim / 2 {
        let base = theta.powf((2 * j) as f32 / dim);
        let ramp = ((j as f32 - low) / denom).clamp(0.0, 1.0);
        table.push((1.0 / base) * (1.0 - ramp) + (1.0 / (factor * base)) * ramp);
    }
    let bytes: Vec<u8> = table.iter().flat_map(|v| v.to_le_bytes()).collect();
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, ptr)?;
    Ok(ptr)
}

struct QsaCache {
    raw_ring: DevicePtr,
    compressed_keys: DevicePtr,
    projected_qk: DevicePtr,
    pooled_key: DevicePtr,
    first_position: DevicePtr,
    logits: DevicePtr,
    token_indices: DevicePtr,
    sparse_partial_output: DevicePtr,
    sparse_partial_max: DevicePtr,
    sparse_partial_sum: DevicePtr,
    max_compressed_groups: usize,
    num_main_blocks: usize,
    block_size: usize,
}

pub struct Qwen4QsaIndexer {
    index_qk_proj: DenseWeight,
    q_norm: DenseWeight,
    k_norm: DenseWeight,
    dense_gemv: KernelHandle,
    rms_norm: KernelHandle,
    rope: KernelHandle,
    rope_yarn_scaled: KernelHandle,
    yarn_inv_freq: DevicePtr,
    yarn_attention_factor: f32,
    stage_pool: KernelHandle,
    store_compressed: KernelHandle,
    score: KernelHandle,
    select_expand: KernelHandle,
    sparse_attention_bf16_partial: KernelHandle,
    sparse_attention_bf16_reduce: KernelHandle,
    cache: OnceLock<QsaCache>,
}

impl Qwen4QsaIndexer {
    pub fn new(
        index_qk_proj: DenseWeight,
        q_norm: DenseWeight,
        k_norm: DenseWeight,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<Self> {
        let yarn_inv_freq = if config.yarn_factor > 0.0 {
            compute_yarn_inv_freq(config, config.rotary_dim(), gpu)?
        } else {
            DevicePtr::NULL
        };
        Ok(Self {
            index_qk_proj,
            q_norm,
            k_norm,
            dense_gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            rms_norm: gpu.kernel("norm", "rms_norm")?,
            rope: gpu.kernel("rope", "rope_forward")?,
            rope_yarn_scaled: gpu.kernel("rope", "rope_forward_yarn_scaled")?,
            yarn_inv_freq,
            yarn_attention_factor: if config.yarn_factor > 0.0 {
                1.0 + 0.1 * config.yarn_factor.ln()
            } else {
                1.0
            },
            stage_pool: gpu.kernel("qwen4_qsa", "qwen4_qsa_stage_pool")?,
            store_compressed: gpu.kernel("qwen4_qsa", "qwen4_qsa_store_compressed")?,
            score: gpu.kernel("qwen4_qsa", "qwen4_qsa_score")?,
            select_expand: gpu.kernel("qwen4_qsa", "qwen4_qsa_select_expand")?,
            sparse_attention_bf16_partial: gpu
                .kernel("qwen4_qsa", "qwen4_qsa_sparse_attention_bf16_partial")?,
            sparse_attention_bf16_reduce: gpu
                .kernel("qwen4_qsa", "qwen4_qsa_sparse_attention_bf16_reduce")?,
            cache: OnceLock::new(),
        })
    }

    fn cache<'a>(
        &'a self,
        gpu: &dyn GpuBackend,
        kv_cache: &PagedKvCache,
        _meta: AttnMetadataDev,
    ) -> Result<&'a QsaCache> {
        if let Some(cache) = self.cache.get() {
            ensure!(
                cache.num_main_blocks == kv_cache.num_blocks()
                    && cache.block_size == kv_cache.block_size(),
                "QSA main-cache geometry changed after initialization"
            );
            return Ok(cache);
        }
        let block_size = kv_cache.block_size();
        ensure!(
            block_size >= COMPRESS_RATIO && block_size.is_multiple_of(COMPRESS_RATIO),
            "QSA requires KV block size divisible by {COMPRESS_RATIO}, got {block_size}"
        );
        let num_main_blocks = kv_cache.num_blocks();
        let groups_per_page = block_size / COMPRESS_RATIO;
        // Size from physical ownership, not the first request's logical block
        // table. A short first request must not permanently cap this OnceLock
        // and reject a later long-context request.
        let max_compressed_groups = num_main_blocks * groups_per_page;
        ensure!(
            num_main_blocks > 0 && max_compressed_groups > 0,
            "empty QSA cache geometry"
        );
        let raw_bytes = num_main_blocks * COMPRESS_RATIO * INDEX_DIM as usize * 2;
        let compressed_bytes = num_main_blocks * groups_per_page * INDEX_DIM as usize * 2;
        let sparse_parts = QUERY_HEADS * SPARSE_SPLITS;
        let cache = QsaCache {
            raw_ring: gpu.alloc(raw_bytes)?,
            compressed_keys: gpu.alloc(compressed_bytes)?,
            projected_qk: gpu.alloc(INDEX_WIDTH as usize * 2)?,
            pooled_key: gpu.alloc(INDEX_DIM as usize * 2)?,
            first_position: gpu.alloc(std::mem::size_of::<u32>())?,
            logits: gpu.alloc(max_compressed_groups * std::mem::size_of::<f32>())?,
            token_indices: gpu.alloc(OUTPUT_WIDTH * std::mem::size_of::<i32>())?,
            sparse_partial_output: gpu
                .alloc(sparse_parts * INDEX_DIM as usize * 2 * std::mem::size_of::<f32>())?,
            sparse_partial_max: gpu.alloc(sparse_parts * std::mem::size_of::<f32>())?,
            sparse_partial_sum: gpu.alloc(sparse_parts * std::mem::size_of::<f32>())?,
            max_compressed_groups,
            num_main_blocks,
            block_size,
        };
        gpu.memset_async(cache.raw_ring, 0, raw_bytes, gpu.default_stream())?;
        gpu.memset_async(
            cache.compressed_keys,
            0,
            compressed_bytes,
            gpu.default_stream(),
        )?;
        self.cache
            .set(cache)
            .map_err(|_| anyhow::anyhow!("QSA cache initialized concurrently"))?;
        tracing::info!(
            layer_cache_mib = (raw_bytes + compressed_bytes) as f64 / (1024.0 * 1024.0),
            max_compressed_groups,
            "Qwen4 QSA physical side cache initialized"
        );
        Ok(self.cache.get().expect("QSA cache just initialized"))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_and_select(
        &self,
        hidden: DevicePtr,
        position: usize,
        sequence_length: usize,
        kv_cache: &PagedKvCache,
        meta: AttnMetadataDev,
        hidden_size: u32,
        eps: f32,
        rope_theta: f32,
        rotary_dim: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(
            meta.num_seqs == 1,
            "QSA correctness path currently supports C=1"
        );
        ensure!(position < sequence_length, "invalid QSA position/length");
        let cache = self.cache(gpu, kv_cache, meta)?;
        let visible_groups = sequence_length / COMPRESS_RATIO;
        ensure!(
            visible_groups <= cache.max_compressed_groups,
            "QSA context requires {visible_groups} groups but cache holds {}",
            cache.max_compressed_groups
        );

        ops::dense_gemv(
            gpu,
            self.dense_gemv,
            hidden,
            &self.index_qk_proj,
            cache.projected_qk,
            INDEX_WIDTH,
            hidden_size,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.rms_norm,
            cache.projected_qk,
            &self.q_norm,
            cache.projected_qk,
            INDEX_HEADS,
            INDEX_DIM,
            eps,
            stream,
        )?;
        KernelLaunch::new(gpu, self.stage_pool)
            .grid([1, 1, 1])
            .block([INDEX_DIM, 1, 1])
            .arg_ptr(cache.projected_qk)
            .arg_ptr(cache.raw_ring)
            .arg_ptr(cache.pooled_key)
            .arg_ptr(cache.first_position)
            .arg_ptr(meta.slot)
            .arg_ptr(meta.positions)
            .arg_u32(cache.block_size as u32)
            .launch(stream)?;
        self.apply_rope(
            cache.projected_qk,
            meta.positions,
            INDEX_HEADS,
            rotary_dim,
            rope_theta,
            gpu,
            stream,
        )?;
        if (position + 1).is_multiple_of(COMPRESS_RATIO) {
            ops::rms_norm(
                gpu,
                self.rms_norm,
                cache.pooled_key,
                &self.k_norm,
                cache.pooled_key,
                1,
                INDEX_DIM,
                eps,
                stream,
            )?;
            self.apply_rope(
                cache.pooled_key,
                cache.first_position,
                1,
                rotary_dim,
                rope_theta,
                gpu,
                stream,
            )?;
            KernelLaunch::new(gpu, self.store_compressed)
                .grid([1, 1, 1])
                .block([INDEX_DIM, 1, 1])
                .arg_ptr(cache.pooled_key)
                .arg_ptr(cache.compressed_keys)
                .arg_ptr(meta.slot)
                .arg_ptr(meta.positions)
                .arg_u32(cache.block_size as u32)
                .launch(stream)?;
        }
        if visible_groups > 512 {
            KernelLaunch::new(gpu, self.score)
                .grid([visible_groups.div_ceil(8) as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(cache.projected_qk)
                .arg_ptr(cache.compressed_keys)
                .arg_ptr(meta.block_table)
                .arg_ptr(cache.logits)
                .arg_u32(visible_groups as u32)
                .arg_u32(cache.block_size as u32)
                .launch(stream)?;
        }
        KernelLaunch::new(gpu, self.select_expand)
            .grid([1, 1, 1])
            .block([1, 1, 1])
            .arg_ptr(cache.logits)
            .arg_ptr(cache.token_indices)
            .arg_u32(visible_groups as u32)
            .arg_u32(position as u32)
            .arg_u32(sequence_length as u32)
            .launch(stream)?;
        Ok(cache.token_indices)
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_rope(
        &self,
        tensor: DevicePtr,
        positions: DevicePtr,
        heads: u32,
        rotary_dim: u32,
        rope_theta: f32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if self.yarn_inv_freq.is_null() {
            ops::rope(
                gpu,
                self.rope,
                tensor,
                DevicePtr::NULL,
                positions,
                1,
                heads,
                0,
                INDEX_DIM,
                rotary_dim,
                rope_theta,
                stream,
            )
        } else {
            ops::rope_yarn_scaled(
                gpu,
                self.rope_yarn_scaled,
                tensor,
                DevicePtr::NULL,
                positions,
                1,
                heads,
                0,
                INDEX_DIM,
                rotary_dim,
                self.yarn_inv_freq,
                rope_theta,
                self.yarn_attention_factor,
                stream,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sparse_attention_bf16(
        &self,
        query: DevicePtr,
        indices: DevicePtr,
        output: DevicePtr,
        kv_cache: &PagedKvCache,
        meta: AttnMetadataDev,
        attn_layer_idx: usize,
        num_query_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        softmax_scale: f32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            kv_cache.dtype_for_layer(attn_layer_idx) == KvCacheDtype::Bf16,
            "QSA exact path currently requires BF16 main KV cache"
        );
        ensure!(head_dim == 256, "released Qwen4 QSA expects head_dim=256");
        ensure!(
            num_query_heads as usize == QUERY_HEADS,
            "released Qwen4 QSA expects {QUERY_HEADS} query heads"
        );
        let cache = self.cache(gpu, kv_cache, meta)?;
        let k_stride = kv_cache.k_block_stride_bytes_for_layer(attn_layer_idx);
        let v_stride = kv_cache.v_block_stride_bytes_for_layer(attn_layer_idx);
        ensure!(
            k_stride % 2 == 0 && v_stride % 2 == 0,
            "invalid BF16 KV stride"
        );
        KernelLaunch::new(gpu, self.sparse_attention_bf16_partial)
            .grid([num_query_heads, SPARSE_SPLITS as u32, 1])
            .block([head_dim, 1, 1])
            .arg_ptr(query)
            .arg_ptr(kv_cache.k_cache_ptr(attn_layer_idx, 0))
            .arg_ptr(kv_cache.v_cache_ptr(attn_layer_idx, 0))
            .arg_ptr(indices)
            .arg_ptr(meta.block_table)
            .arg_ptr(cache.sparse_partial_output)
            .arg_ptr(cache.sparse_partial_max)
            .arg_ptr(cache.sparse_partial_sum)
            .arg_u64((k_stride / 2) as u64)
            .arg_u64((v_stride / 2) as u64)
            .arg_u32(kv_cache.block_size() as u32)
            .arg_u32(num_query_heads)
            .arg_u32(num_kv_heads)
            .arg_u32(head_dim)
            .arg_f32(softmax_scale)
            .arg_u32(SPARSE_SPLITS as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.sparse_attention_bf16_reduce)
            .grid([num_query_heads, 1, 1])
            .block([head_dim, 1, 1])
            .arg_ptr(cache.sparse_partial_output)
            .arg_ptr(cache.sparse_partial_max)
            .arg_ptr(cache.sparse_partial_sum)
            .arg_ptr(output)
            .arg_u32(num_query_heads)
            .arg_u32(head_dim)
            .arg_u32(SPARSE_SPLITS as u32)
            .launch(stream)
    }
}
