// SPDX-License-Identifier: AGPL-3.0-only

//! Commit a nonce-bound GLM-5.3 DSA BF16 latent transaction overlay.
//!
//! CUDA globally validates every batch item's end, nonce, and old logical
//! length before effects. It then copies only the accepted prefix, fences all
//! writers, publishes new lengths, and retires ends before nonces. Callers must
//! serialize stage/commit/attention on one stream and treat launch or stream
//! failure as fatal for the affected cache allocation.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::GgmlIqBuffer;

const LATENT: u32 = 512;
const MAX_QUERIES: u32 = 8;
const MAX_POSITIONS: u32 = 1_048_576;
const THREADS: u32 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53DsaLatentCommitStorage {
    Bf16,
    Fp8E4M3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53DsaLatentCommitPlan {
    pub batch: u32,
    pub queries: u32,
    pub accepted: u32,
    pub capacity: u32,
    pub start_position: u32,
    pub end_position: u32,
    pub latent: u32,
    pub transaction_nonce: u64,
    pub storage: Glm53DsaLatentCommitStorage,
    pub overlay_bytes: usize,
    pub persistent_cache_bytes: usize,
    pub published_end_bytes: usize,
    pub published_nonce_bytes: usize,
    pub logical_length_bytes: usize,
}

impl Glm53DsaLatentCommitPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batch: u32,
        queries: u32,
        accepted: u32,
        capacity: u32,
        start_position: u32,
        end_position: u32,
        latent: u32,
        transaction_nonce: u64,
        storage: Glm53DsaLatentCommitStorage,
    ) -> Result<Self> {
        if batch == 0 || !(1..=MAX_QUERIES).contains(&queries) || accepted > queries {
            bail!("GLM DSA latent commit requires batch>0, Q in 1..=8, and accepted<=Q");
        }
        if !(1..=MAX_POSITIONS).contains(&capacity) || latent != LATENT {
            bail!("GLM DSA latent commit has invalid exact geometry");
        }
        if storage != Glm53DsaLatentCommitStorage::Bf16 {
            bail!("GLM DSA latent commit rejects FP8 until a scale ABI exists");
        }
        if transaction_nonce == 0 {
            bail!("GLM DSA latent commit requires a nonzero transaction nonce");
        }
        let computed_end = start_position
            .checked_add(queries)
            .context("GLM DSA latent commit position overflow")?;
        if end_position != computed_end || end_position > capacity {
            bail!("GLM DSA latent commit requires exact start/exclusive-end extent");
        }
        let batch_u64 = u64::from(batch);
        let rows = batch_u64
            .checked_mul(u64::from(queries))
            .context("GLM DSA latent commit overlay-row overflow")?;
        let cache_rows = batch_u64
            .checked_mul(u64::from(capacity))
            .context("GLM DSA latent commit cache-row overflow")?;
        Ok(Self {
            batch,
            queries,
            accepted,
            capacity,
            start_position,
            end_position,
            latent,
            transaction_nonce,
            storage,
            overlay_bytes: extent(rows, u64::from(LATENT), 2)?,
            persistent_cache_bytes: extent(cache_rows, u64::from(LATENT), 2)?,
            published_end_bytes: extent(batch_u64, 1, 4)?,
            published_nonce_bytes: extent(batch_u64, 1, 8)?,
            logical_length_bytes: extent(batch_u64, 1, 4)?,
        })
    }

    pub fn validate(self) -> Result<()> {
        if Self::new(
            self.batch,
            self.queries,
            self.accepted,
            self.capacity,
            self.start_position,
            self.end_position,
            self.latent,
            self.transaction_nonce,
            self.storage,
        )? != self
        {
            bail!("forged GLM DSA latent-commit plan");
        }
        Ok(())
    }
}

fn extent(rows: u64, columns: u64, element_bytes: u64) -> Result<usize> {
    usize::try_from(
        rows.checked_mul(columns)
            .and_then(|elements| elements.checked_mul(element_bytes))
            .context("GLM DSA latent-commit byte-extent overflow")?,
    )
    .context("GLM DSA latent-commit extent does not fit usize")
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53DsaLatentCommitBuffers {
    pub transaction_overlay_bf16: GgmlIqBuffer,
    pub persistent_cache_bf16: GgmlIqBuffer,
    pub published_ends_u32: GgmlIqBuffer,
    pub published_nonces_u64: GgmlIqBuffer,
    pub logical_lengths_u32: GgmlIqBuffer,
}

pub struct Glm53DsaLatentCommitKernel {
    commit: KernelHandle,
}

impl Glm53DsaLatentCommitKernel {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            commit: gpu.kernel(
                "glm53_dsa_latent_commit",
                "atlas_glm53_dsa_latent_commit_bf16",
            )?,
        })
    }

    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53DsaLatentCommitPlan,
        buffers: Glm53DsaLatentCommitBuffers,
        stream: u64,
    ) -> Result<()> {
        plan.validate()?;
        validate_buffers(plan, buffers)?;
        KernelLaunch::new(gpu, self.commit)
            .grid([1, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.transaction_overlay_bf16.ptr)
            .arg_ptr(buffers.persistent_cache_bf16.ptr)
            .arg_ptr(buffers.published_ends_u32.ptr)
            .arg_ptr(buffers.published_nonces_u64.ptr)
            .arg_ptr(buffers.logical_lengths_u32.ptr)
            .arg_u32(plan.batch)
            .arg_u32(plan.queries)
            .arg_u32(plan.accepted)
            .arg_u32(plan.capacity)
            .arg_u32(plan.start_position)
            .arg_u32(plan.end_position)
            .arg_u64(plan.transaction_nonce)
            .launch(stream)
    }
}

fn validate_buffers(
    plan: Glm53DsaLatentCommitPlan,
    buffers: Glm53DsaLatentCommitBuffers,
) -> Result<()> {
    let named = [
        (
            "transaction overlay",
            buffers.transaction_overlay_bf16,
            plan.overlay_bytes,
            2,
        ),
        (
            "persistent cache",
            buffers.persistent_cache_bf16,
            plan.persistent_cache_bytes,
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
        (
            "logical lengths",
            buffers.logical_lengths_u32,
            plan.logical_length_bytes,
            4,
        ),
    ];
    let mut ranges = [(0u64, 0u64); 5];
    for (slot, (name, buffer, expected, alignment)) in named.iter().copied().enumerate() {
        if buffer.bytes != expected
            || buffer.ptr == DevicePtr::NULL
            || buffer.ptr.0 % alignment != 0
        {
            bail!("GLM DSA latent-commit {name} buffer has invalid pointer, alignment, or extent");
        }
        ranges[slot] = (
            buffer.ptr.0,
            buffer
                .ptr
                .0
                .checked_add(u64::try_from(expected)?)
                .with_context(|| format!("GLM DSA latent-commit {name} address overflow"))?,
        );
    }
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].0 < ranges[right].1 && ranges[right].0 < ranges[left].1 {
                bail!("GLM DSA latent-commit device buffers overlap");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "glm53_dsa_latent_commit_tests.rs"]
mod tests;
