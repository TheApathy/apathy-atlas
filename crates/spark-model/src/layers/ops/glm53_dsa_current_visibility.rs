// SPDX-License-Identifier: AGPL-3.0-only

//! Transaction-only GLM-5.3 DSA current-token visibility materialization.
//!
//! This dormant primitive writes only physically future latent/pool rows for
//! all eleven DSA layers. It never changes a logical length, index, or tail;
//! readiness requires exact end/generation/nonce/status metadata, with the
//! ready nonce published last. Clearing ready metadata therefore hides stale
//! future bytes on a clean rollback.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::GgmlIqBuffer;

const LAYERS: u32 = 11;
const LATENT_RANK: u32 = 512;
const INDEX_RANK: u32 = 128;
const KPOOL: u32 = 4;
const MAX_POSITIONS: u32 = 1_048_576;
const THREADS: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Glm53DsaCurrentVisibilityPlan {
    pub capacity: u32,
    pub pool_capacity: u32,
    pub logical_position: u32,
    pub end_position: u32,
    pub pool_row: u32,
    pub writes_pool: bool,
    pub generation: u64,
    pub transaction_nonce: u64,
    pub latent_overlay_bytes: usize,
    pub pool_overlay_bytes: usize,
    pub pool_overlay_validity_bytes: usize,
    pub persistent_latent_bytes: usize,
    pub persistent_pool_bytes: usize,
    pub persistent_pool_validity_bytes: usize,
    pub ends_bytes: usize,
    pub generations_bytes: usize,
    pub nonces_bytes: usize,
    pub statuses_bytes: usize,
}

impl Glm53DsaCurrentVisibilityPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        capacity: u32,
        pool_capacity: u32,
        logical_position: u32,
        end_position: u32,
        layers: u32,
        latent_rank: u32,
        index_rank: u32,
        kpool: u32,
        generation: u64,
        transaction_nonce: u64,
    ) -> Result<Self> {
        if !(1..=MAX_POSITIONS).contains(&capacity)
            || logical_position >= capacity
            || layers != LAYERS
            || latent_rank != LATENT_RANK
            || index_rank != INDEX_RANK
            || kpool != KPOOL
            || generation == 0
            || transaction_nonce == 0
        {
            bail!("GLM DSA current visibility has invalid exact geometry or transaction");
        }
        let computed_end = logical_position
            .checked_add(1)
            .context("GLM DSA current visibility end overflow")?;
        let computed_pool_capacity = capacity / KPOOL;
        if computed_end != end_position || pool_capacity != computed_pool_capacity {
            bail!("GLM DSA current visibility end or pool capacity mismatch");
        }
        let writes_pool = logical_position % KPOOL == KPOOL - 1;
        let pool_row = logical_position / KPOOL;
        let layer64 = u64::from(LAYERS);
        let meta = |width| extent(layer64, 1, width);
        Ok(Self {
            capacity,
            pool_capacity,
            logical_position,
            end_position,
            pool_row,
            writes_pool,
            generation,
            transaction_nonce,
            latent_overlay_bytes: extent(layer64, u64::from(LATENT_RANK), 2)?,
            pool_overlay_bytes: if writes_pool {
                extent(layer64, u64::from(INDEX_RANK), 2)?
            } else {
                0
            },
            pool_overlay_validity_bytes: if writes_pool { meta(1)? } else { 0 },
            persistent_latent_bytes: extent(
                layer64,
                u64::from(capacity) * u64::from(LATENT_RANK),
                2,
            )?,
            persistent_pool_bytes: extent(
                layer64,
                u64::from(pool_capacity) * u64::from(INDEX_RANK),
                2,
            )?,
            persistent_pool_validity_bytes: extent(layer64, u64::from(pool_capacity), 1)?,
            ends_bytes: meta(4)?,
            generations_bytes: meta(8)?,
            nonces_bytes: meta(8)?,
            statuses_bytes: meta(4)?,
        })
    }

    pub fn validate(self) -> Result<()> {
        if Self::new(
            self.capacity,
            self.pool_capacity,
            self.logical_position,
            self.end_position,
            LAYERS,
            LATENT_RANK,
            INDEX_RANK,
            KPOOL,
            self.generation,
            self.transaction_nonce,
        )? != self
        {
            bail!("forged GLM DSA current visibility plan");
        }
        Ok(())
    }
}

fn extent(rows: u64, columns: u64, element_bytes: u64) -> Result<usize> {
    usize::try_from(
        rows.checked_mul(columns)
            .and_then(|elements| elements.checked_mul(element_bytes))
            .context("GLM DSA current visibility extent overflow")?,
    )
    .context("GLM DSA current visibility extent exceeds usize")
}

#[derive(Clone, Copy, Debug)]
pub struct Glm53DsaCurrentVisibilityBuffers {
    pub latent_overlay_bf16: GgmlIqBuffer,
    pub pool_overlay_bf16: GgmlIqBuffer,
    pub pool_overlay_validity_u8: GgmlIqBuffer,
    pub staged_ends_u32: GgmlIqBuffer,
    pub staged_generations_u64: GgmlIqBuffer,
    pub staged_nonces_u64: GgmlIqBuffer,
    pub staged_statuses_u32: GgmlIqBuffer,
    pub persistent_latent_bf16: GgmlIqBuffer,
    pub persistent_pool_bf16: GgmlIqBuffer,
    pub persistent_pool_validity_u8: GgmlIqBuffer,
    pub ready_ends_u32: GgmlIqBuffer,
    pub ready_generations_u64: GgmlIqBuffer,
    pub ready_statuses_u32: GgmlIqBuffer,
    pub ready_nonces_u64: GgmlIqBuffer,
}

pub struct Glm53DsaCurrentVisibilityKernel {
    materialize: KernelHandle,
}

impl Glm53DsaCurrentVisibilityKernel {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            materialize: gpu.kernel(
                "glm53_dsa_current_visibility",
                "atlas_glm53_dsa_current_visibility_bf16",
            )?,
        })
    }

    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53DsaCurrentVisibilityPlan,
        buffers: Glm53DsaCurrentVisibilityBuffers,
        stream: u64,
    ) -> Result<()> {
        plan.validate()?;
        validate_buffers(plan, buffers)?;
        clear_ready(gpu, plan, buffers, stream)?;
        KernelLaunch::new(gpu, self.materialize)
            .grid([1, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.latent_overlay_bf16.ptr)
            .arg_ptr(buffers.pool_overlay_bf16.ptr)
            .arg_ptr(buffers.pool_overlay_validity_u8.ptr)
            .arg_ptr(buffers.staged_ends_u32.ptr)
            .arg_ptr(buffers.staged_generations_u64.ptr)
            .arg_ptr(buffers.staged_nonces_u64.ptr)
            .arg_ptr(buffers.staged_statuses_u32.ptr)
            .arg_ptr(buffers.persistent_latent_bf16.ptr)
            .arg_ptr(buffers.persistent_pool_bf16.ptr)
            .arg_ptr(buffers.persistent_pool_validity_u8.ptr)
            .arg_ptr(buffers.ready_ends_u32.ptr)
            .arg_ptr(buffers.ready_generations_u64.ptr)
            .arg_ptr(buffers.ready_statuses_u32.ptr)
            .arg_ptr(buffers.ready_nonces_u64.ptr)
            .arg_u32(plan.capacity)
            .arg_u32(plan.pool_capacity)
            .arg_u32(plan.logical_position)
            .arg_u32(plan.end_position)
            .arg_u64(plan.generation)
            .arg_u64(plan.transaction_nonce)
            .launch(stream)
    }

    /// Same-stream clean rollback. Nonzero nonce is the sole visibility gate;
    /// physical future bytes are intentionally left for overwrite.
    pub fn rollback_ready(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53DsaCurrentVisibilityPlan,
        buffers: Glm53DsaCurrentVisibilityBuffers,
        stream: u64,
    ) -> Result<()> {
        plan.validate()?;
        validate_buffers(plan, buffers)?;
        clear_ready(gpu, plan, buffers, stream)
    }
}

fn clear_ready(
    gpu: &dyn GpuBackend,
    plan: Glm53DsaCurrentVisibilityPlan,
    buffers: Glm53DsaCurrentVisibilityBuffers,
    stream: u64,
) -> Result<()> {
    gpu.memset_async(buffers.ready_nonces_u64.ptr, 0, plan.nonces_bytes, stream)?;
    gpu.memset_async(
        buffers.ready_statuses_u32.ptr,
        0xff,
        plan.statuses_bytes,
        stream,
    )?;
    gpu.memset_async(buffers.ready_ends_u32.ptr, 0, plan.ends_bytes, stream)?;
    gpu.memset_async(
        buffers.ready_generations_u64.ptr,
        0,
        plan.generations_bytes,
        stream,
    )
}

fn validate_buffers(
    plan: Glm53DsaCurrentVisibilityPlan,
    buffers: Glm53DsaCurrentVisibilityBuffers,
) -> Result<()> {
    let named = [
        (
            "latent overlay",
            buffers.latent_overlay_bf16,
            plan.latent_overlay_bytes,
            2,
        ),
        (
            "pool overlay",
            buffers.pool_overlay_bf16,
            plan.pool_overlay_bytes,
            2,
        ),
        (
            "pool overlay validity",
            buffers.pool_overlay_validity_u8,
            plan.pool_overlay_validity_bytes,
            1,
        ),
        ("staged ends", buffers.staged_ends_u32, plan.ends_bytes, 4),
        (
            "staged generations",
            buffers.staged_generations_u64,
            plan.generations_bytes,
            8,
        ),
        (
            "staged nonces",
            buffers.staged_nonces_u64,
            plan.nonces_bytes,
            8,
        ),
        (
            "staged statuses",
            buffers.staged_statuses_u32,
            plan.statuses_bytes,
            4,
        ),
        (
            "persistent latent",
            buffers.persistent_latent_bf16,
            plan.persistent_latent_bytes,
            2,
        ),
        (
            "persistent pool",
            buffers.persistent_pool_bf16,
            plan.persistent_pool_bytes,
            2,
        ),
        (
            "persistent pool validity",
            buffers.persistent_pool_validity_u8,
            plan.persistent_pool_validity_bytes,
            1,
        ),
        ("ready ends", buffers.ready_ends_u32, plan.ends_bytes, 4),
        (
            "ready generations",
            buffers.ready_generations_u64,
            plan.generations_bytes,
            8,
        ),
        (
            "ready statuses",
            buffers.ready_statuses_u32,
            plan.statuses_bytes,
            4,
        ),
        (
            "ready nonces",
            buffers.ready_nonces_u64,
            plan.nonces_bytes,
            8,
        ),
    ];
    let mut ranges = Vec::with_capacity(named.len());
    for (name, buffer, expected, alignment) in named {
        if expected == 0 {
            if buffer.ptr != DevicePtr::NULL || buffer.bytes != 0 {
                bail!("GLM DSA current visibility absent {name} buffer is not empty");
            }
            continue;
        }
        if buffer.ptr == DevicePtr::NULL
            || buffer.bytes != expected
            || buffer.ptr.0 % alignment != 0
        {
            bail!("GLM DSA current visibility {name} buffer is invalid");
        }
        let end = buffer
            .ptr
            .0
            .checked_add(u64::try_from(expected)?)
            .with_context(|| format!("GLM DSA current visibility {name} address overflow"))?;
        ranges.push((buffer.ptr.0, end));
    }
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].0 < ranges[right].1 && ranges[right].0 < ranges[left].1 {
                bail!("GLM DSA current visibility buffers overlap");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "glm53_dsa_current_visibility_tests.rs"]
mod tests;
