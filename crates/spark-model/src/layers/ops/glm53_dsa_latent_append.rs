// SPDX-License-Identifier: AGPL-3.0-only

//! Transaction-overlay staging for the GLM-5.3 DSA BF16 latent cache.
//!
//! This primitive deliberately has no persistent-cache argument. It copies an
//! append into a transaction-private `[B,Q,512]` overlay, then publishes the
//! exclusive end and transaction nonce after a device fence. Callers must
//! discard the overlay on launch/synchronization error or rollback. A later
//! commit must first match both publications for every batch item, and may then
//! copy only an accepted prefix. It must not advance logical length first.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::GgmlIqBuffer;

const LATENT: u32 = 512;
const MAX_QUERIES: u32 = 8;
const MAX_POSITIONS: u32 = 1_048_576;
const THREADS: u32 = 256;
const MAX_GRID_YZ: u64 = 65_535;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53DsaLatentAppendStorage {
    Bf16,
    Fp8E4M3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53DsaLatentAppendPlan {
    pub batch: u32,
    pub queries: u32,
    pub capacity: u32,
    pub start_position: u32,
    pub end_position: u32,
    pub latent: u32,
    pub transaction_nonce: u64,
    pub storage: Glm53DsaLatentAppendStorage,
    pub grid_y: u32,
    pub grid_z: u32,
    pub source_bytes: usize,
    pub overlay_bytes: usize,
    pub published_end_bytes: usize,
    pub published_nonce_bytes: usize,
}

impl Glm53DsaLatentAppendPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batch: u32,
        queries: u32,
        capacity: u32,
        start_position: u32,
        end_position: u32,
        latent: u32,
        transaction_nonce: u64,
        storage: Glm53DsaLatentAppendStorage,
    ) -> Result<Self> {
        if batch == 0 || !(1..=MAX_QUERIES).contains(&queries) {
            bail!("GLM DSA latent append requires batch>0 and Q in 1..=8");
        }
        if !(1..=MAX_POSITIONS).contains(&capacity) || latent != LATENT {
            bail!("GLM DSA latent append has invalid exact geometry");
        }
        if storage != Glm53DsaLatentAppendStorage::Bf16 {
            bail!("GLM DSA latent append rejects FP8 until a scale ABI exists");
        }
        if transaction_nonce == 0 {
            bail!("GLM DSA latent append requires a nonzero transaction nonce");
        }
        let computed_end = start_position
            .checked_add(queries)
            .context("GLM DSA latent append position overflow")?;
        if end_position != computed_end || end_position > capacity {
            bail!("GLM DSA latent append requires exact start/exclusive-end extent");
        }
        let batch_u64 = u64::from(batch);
        let grid_y = batch_u64.min(MAX_GRID_YZ);
        let grid_z = batch_u64
            .checked_add(grid_y - 1)
            .context("GLM DSA latent append grid rounding overflow")?
            / grid_y;
        if grid_z > MAX_GRID_YZ {
            bail!("GLM DSA latent append batch grid exceeds CUDA y/z capacity");
        }
        let rows = batch_u64
            .checked_mul(u64::from(queries))
            .context("GLM DSA latent append row overflow")?;
        let payload_bytes = extent(rows, u64::from(LATENT), 2)?;
        Ok(Self {
            batch,
            queries,
            capacity,
            start_position,
            end_position,
            latent,
            transaction_nonce,
            storage,
            grid_y: u32::try_from(grid_y)?,
            grid_z: u32::try_from(grid_z)?,
            source_bytes: payload_bytes,
            overlay_bytes: payload_bytes,
            published_end_bytes: extent(batch_u64, 1, 4)?,
            published_nonce_bytes: extent(batch_u64, 1, 8)?,
        })
    }

    pub fn validate(self) -> Result<()> {
        if Self::new(
            self.batch,
            self.queries,
            self.capacity,
            self.start_position,
            self.end_position,
            self.latent,
            self.transaction_nonce,
            self.storage,
        )? != self
        {
            bail!("forged GLM DSA latent-append plan");
        }
        Ok(())
    }
}

fn extent(rows: u64, columns: u64, element_bytes: u64) -> Result<usize> {
    usize::try_from(
        rows.checked_mul(columns)
            .and_then(|elements| elements.checked_mul(element_bytes))
            .context("GLM DSA latent-append byte-extent overflow")?,
    )
    .context("GLM DSA latent-append extent does not fit usize")
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53DsaLatentAppendBuffers {
    pub source_bf16: GgmlIqBuffer,
    pub transaction_overlay_bf16: GgmlIqBuffer,
    pub published_ends_u32: GgmlIqBuffer,
    pub published_nonces_u64: GgmlIqBuffer,
}

pub struct Glm53DsaLatentAppendKernel {
    stage: KernelHandle,
}

impl Glm53DsaLatentAppendKernel {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            stage: gpu.kernel(
                "glm53_dsa_latent_append",
                "atlas_glm53_dsa_latent_append_bf16_stage",
            )?,
        })
    }

    pub fn launch_stage(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53DsaLatentAppendPlan,
        buffers: Glm53DsaLatentAppendBuffers,
        stream: u64,
    ) -> Result<()> {
        plan.validate()?;
        validate_buffers(plan, buffers)?;
        // The nonce is the release marker: invalidate it before every attempt.
        // Same-stream order guarantees these clears precede any staged writes.
        gpu.memset_async(
            buffers.published_nonces_u64.ptr,
            0,
            plan.published_nonce_bytes,
            stream,
        )?;
        gpu.memset_async(
            buffers.published_ends_u32.ptr,
            0,
            plan.published_end_bytes,
            stream,
        )?;
        KernelLaunch::new(gpu, self.stage)
            .grid([1, plan.grid_y, plan.grid_z])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.source_bf16.ptr)
            .arg_ptr(buffers.transaction_overlay_bf16.ptr)
            .arg_ptr(buffers.published_ends_u32.ptr)
            .arg_ptr(buffers.published_nonces_u64.ptr)
            .arg_u32(plan.batch)
            .arg_u32(plan.queries)
            .arg_u32(plan.capacity)
            .arg_u32(plan.start_position)
            .arg_u32(plan.end_position)
            .arg_u64(plan.transaction_nonce)
            .launch(stream)
    }
}

fn validate_buffers(
    plan: Glm53DsaLatentAppendPlan,
    buffers: Glm53DsaLatentAppendBuffers,
) -> Result<()> {
    let named = [
        ("source", buffers.source_bf16, plan.source_bytes, 2),
        (
            "transaction overlay",
            buffers.transaction_overlay_bf16,
            plan.overlay_bytes,
            2,
        ),
        (
            "published ends",
            buffers.published_ends_u32,
            plan.published_end_bytes,
            4,
        ),
        (
            "published nonces",
            buffers.published_nonces_u64,
            plan.published_nonce_bytes,
            8,
        ),
    ];
    // Sized from the table it mirrors, never a literal: a hand-written
    // length silently goes out of bounds the moment the table grows.
    let mut ranges = Vec::with_capacity(named.len());
    for (name, buffer, expected, alignment) in named.iter().copied() {
        if buffer.bytes != expected
            || buffer.ptr == DevicePtr::NULL
            || buffer.ptr.0 % alignment != 0
        {
            bail!("GLM DSA latent-append {name} buffer has invalid pointer, alignment, or extent");
        }
        ranges.push((
            buffer.ptr.0,
            buffer
                .ptr
                .0
                .checked_add(u64::try_from(expected)?)
                .with_context(|| format!("GLM DSA latent-append {name} address overflow"))?,
        ));
    }
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].0 < ranges[right].1 && ranges[right].0 < ranges[left].1 {
                bail!("GLM DSA latent-append device buffers overlap");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "glm53_dsa_latent_append_tests.rs"]
mod tests;
