// SPDX-License-Identifier: AGPL-3.0-only

//! Exact four-row QSA history transaction for dense-window prefill.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::PagedKvCache;

use super::{COMPRESS_RATIO, INDEX_DIM, INDEX_HEADS, Qwen4QsaIndexer};
use crate::layer::AttnMetadataDev;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

impl Qwen4QsaIndexer {
    pub(crate) fn validate_exact_group4(&self) -> Result<()> {
        ensure!(
            self.dense_gemv_batchn.0 != 0,
            "exact QSA group4 requires dense_gemv_bf16_batchn"
        );
        ensure!(
            !self.index_qk_proj.weight.is_null() && !self.k_norm.weight.is_null(),
            "exact QSA group4 requires key projection and normalization weights"
        );
        Ok(())
    }

    pub(crate) fn prepare_exact_group4(
        &self,
        gpu: &dyn GpuBackend,
        kv_cache: &PagedKvCache,
    ) -> Result<()> {
        self.validate_exact_group4()?;
        let _ = self.cache(gpu, kv_cache)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn update_prefill_exact_group4(
        &self,
        hidden: DevicePtr,
        group_start: usize,
        kv_cache: &PagedKvCache,
        meta: AttnMetadataDev,
        hidden_size: u32,
        eps: f32,
        rope_theta: f32,
        rotary_dim: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        self.validate_exact_group4()?;
        ensure!(
            group_start.is_multiple_of(COMPRESS_RATIO),
            "exact QSA group4 requires an aligned complete group"
        );
        ensure!(meta.num_seqs == 1, "exact QSA group4 requires C1");
        ensure!(
            !hidden.is_null() && !meta.positions.is_null() && !meta.slot.is_null(),
            "exact QSA group4 requires complete inputs"
        );
        let cache = self.cache(gpu, kv_cache)?;
        ensure!(
            group_start
                .checked_add(COMPRESS_RATIO)
                .is_some_and(|end| end / COMPRESS_RATIO <= cache.max_compressed_groups),
            "exact QSA group4 exceeds side-cache capacity"
        );
        let key_weight = DenseWeight {
            weight: self
                .index_qk_proj
                .weight
                .offset(INDEX_HEADS as usize * INDEX_DIM as usize * hidden_size as usize * 2),
        };
        ops::dense_gemv_batchn(
            gpu,
            self.dense_gemv_batchn,
            hidden,
            &key_weight,
            cache.projected_qk,
            COMPRESS_RATIO as u32,
            INDEX_DIM,
            hidden_size,
            stream,
        )?;
        KernelLaunch::new(gpu, self.stage_prefill_raw)
            .grid([COMPRESS_RATIO as u32, 1, 1])
            .block([INDEX_DIM, 1, 1])
            .arg_ptr(cache.projected_qk)
            .arg_ptr(cache.raw_ring)
            .arg_ptr(meta.slot)
            .arg_ptr(meta.positions)
            .arg_u32(COMPRESS_RATIO as u32)
            .arg_u32(cache.block_size as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.pool_prefill)
            .grid([1, 1, 1])
            .block([INDEX_DIM, 1, 1])
            .arg_ptr(cache.raw_ring)
            .arg_ptr(cache.pooled_key)
            .arg_ptr(cache.first_position)
            .arg_ptr(meta.slot)
            .arg_ptr(meta.positions)
            .arg_u32((COMPRESS_RATIO - 1) as u32)
            .arg_u32(1)
            .arg_u32(cache.block_size as u32)
            .launch(stream)?;
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
        self.apply_rope_n(
            cache.pooled_key,
            cache.first_position,
            1,
            1,
            rotary_dim,
            rope_theta,
            gpu,
            stream,
        )?;
        KernelLaunch::new(gpu, self.store_prefill_compressed)
            .grid([1, 1, 1])
            .block([INDEX_DIM, 1, 1])
            .arg_ptr(cache.pooled_key)
            .arg_ptr(cache.compressed_keys)
            .arg_ptr(meta.slot)
            .arg_u32((COMPRESS_RATIO - 1) as u32)
            .arg_u32(1)
            .arg_u32(cache.block_size as u32)
            .launch(stream)
    }
}
