// SPDX-License-Identifier: AGPL-3.0-only

//! Exact four-row QSA history transaction for dense-window prefill.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
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
    /// Tile-batched twin of [`Self::update_prefill_exact_group4`]: one
    /// projection GEMV over `rows` tokens, one pool/norm/rope/store launch
    /// over `rows / 4` groups, and the ring staged only for the last group
    /// (the only ring state observable after the tile). Every per-row and
    /// per-group kernel is the same kernel with a larger grid, so results are
    /// bit-identical to the per-group sequence.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn update_prefill_exact_tile(
        &self,
        hidden: DevicePtr,
        tile_start: usize,
        rows: usize,
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
            tile_start.is_multiple_of(COMPRESS_RATIO) && rows.is_multiple_of(COMPRESS_RATIO) && rows > 0 && rows <= 32,
            "exact QSA tile requires aligned complete groups (<= 32 rows)"
        );
        ensure!(meta.num_seqs == 1, "exact QSA tile requires C1");
        let cache = self.cache(gpu, kv_cache)?;
        let groups = rows / COMPRESS_RATIO;
        ensure!(
            tile_start
                .checked_add(rows)
                .is_some_and(|end| end / COMPRESS_RATIO <= cache.max_compressed_groups),
            "exact QSA tile exceeds side-cache capacity"
        );
        static SCRATCH: std::sync::Mutex<Option<(DevicePtr, DevicePtr, DevicePtr)>> = std::sync::Mutex::new(None);
        static POOL_K: std::sync::OnceLock<Result<KernelHandle, String>> = std::sync::OnceLock::new();
        let pool_k = POOL_K
            .get_or_init(|| gpu.kernel("qwen4_qsa_tile", "qwen4_qsa_pool_prefill_from_raw").map_err(|e| e.to_string()))
            .clone()
            .map_err(|e| anyhow::anyhow!("QSA tile pool kernel unavailable: {e}"))?;
        let (projected, pooled, first_pos) = {
            let mut g = SCRATCH.lock().unwrap();
            if g.is_none() {
                *g = Some((
                    gpu.alloc(32 * INDEX_DIM as usize * 2)?,
                    gpu.alloc(8 * INDEX_DIM as usize * 2)?,
                    gpu.alloc(8 * std::mem::size_of::<u32>())?,
                ));
            }
            g.unwrap()
        };
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
            projected,
            rows as u32,
            INDEX_DIM,
            hidden_size,
            stream,
        )?;
        KernelLaunch::new(gpu, pool_k)
            .grid([groups as u32, 1, 1])
            .block([INDEX_DIM, 1, 1])
            .arg_ptr(projected)
            .arg_ptr(pooled)
            .arg_ptr(first_pos)
            .arg_ptr(meta.positions)
            .arg_u32(groups as u32)
            .launch(stream)?;
        ops::rms_norm(
            gpu,
            self.rms_norm,
            pooled,
            &self.k_norm,
            pooled,
            groups as u32,
            INDEX_DIM,
            eps,
            stream,
        )?;
        self.apply_rope_n(pooled, first_pos, groups, 1, rotary_dim, rope_theta, gpu, stream)?;
        KernelLaunch::new(gpu, self.store_prefill_compressed)
            .grid([groups as u32, 1, 1])
            .block([INDEX_DIM, 1, 1])
            .arg_ptr(pooled)
            .arg_ptr(cache.compressed_keys)
            .arg_ptr(meta.slot)
            .arg_u32((COMPRESS_RATIO - 1) as u32)
            .arg_u32(groups as u32)
            .arg_u32(cache.block_size as u32)
            .launch(stream)?;
        // Ring: only the last group's rows are observable after the tile.
        let last = rows - COMPRESS_RATIO;
        KernelLaunch::new(gpu, self.stage_prefill_raw)
            .grid([COMPRESS_RATIO as u32, 1, 1])
            .block([INDEX_DIM, 1, 1])
            .arg_ptr(projected.offset(last * INDEX_DIM as usize * 2))
            .arg_ptr(cache.raw_ring)
            .arg_ptr(meta.slot.offset(last * 8))
            .arg_ptr(meta.positions.offset(last * 4))
            .arg_u32(COMPRESS_RATIO as u32)
            .arg_u32(cache.block_size as u32)
            .launch(stream)?;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| tracing::info!("QSA_PREFILL_TILE_ENGAGED selector=ATLAS_QWEN4_PREFILL_QSA_TILE rows={rows}"));
        Ok(())
    }
}
