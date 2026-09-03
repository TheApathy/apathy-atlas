// SPDX-License-Identifier: AGPL-3.0-only

//! Atomic index-tail publication for one GLM-5.3 DSA decode transaction.
//!
//! This primitive treats the eleven DSA layers as one device transaction. It
//! validates every latent, index, and current-visibility receipt before any
//! write. Acceptance publishes only staged index tails; rejection retires all
//! private receipts without touching persistent state. Latent publication and
//! the CPU logical-length commit deliberately remain later operations.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::GgmlIqBuffer;

const DSA_LAYERS: u32 = 11;
const INDEX_DIM: u32 = 128;
const TAIL_ROWS: u32 = 3;
const KPOOL: u32 = 4;
const CAPACITY: u32 = 1_048_576;
const THREADS: u32 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53DsaIndexCommitPlan {
    pub accepted: u32,
    pub capacity: u32,
    pub start_position: u32,
    pub end_position: u32,
    pub owner_generation: u64,
    pub transaction_nonce: u64,
    pub tail_vector_bytes: usize,
    pub tail_validity_bytes: usize,
    pub pool_validity_bytes: usize,
    pub layer_u32_bytes: usize,
    pub layer_u64_bytes: usize,
}

impl Glm53DsaIndexCommitPlan {
    pub fn new(
        accepted: u32,
        capacity: u32,
        start_position: u32,
        end_position: u32,
        owner_generation: u64,
        transaction_nonce: u64,
    ) -> Result<Self> {
        if accepted > 1 || capacity != CAPACITY {
            bail!("GLM DSA index commit requires accepted in 0..=1 and full 1M capacity");
        }
        if start_position >= capacity || end_position != start_position + 1 {
            bail!("GLM DSA index commit requires an exact in-range T1 extent");
        }
        if owner_generation == 0 || transaction_nonce == 0 {
            bail!("GLM DSA index commit requires nonzero generation and nonce");
        }
        Ok(Self {
            accepted,
            capacity,
            start_position,
            end_position,
            owner_generation,
            transaction_nonce,
            tail_vector_bytes: extent(DSA_LAYERS, TAIL_ROWS * INDEX_DIM, 2)?,
            tail_validity_bytes: extent(DSA_LAYERS, TAIL_ROWS, 1)?,
            pool_validity_bytes: extent(DSA_LAYERS, capacity / KPOOL, 1)?,
            layer_u32_bytes: extent(DSA_LAYERS, 1, 4)?,
            layer_u64_bytes: extent(DSA_LAYERS, 1, 8)?,
        })
    }

    pub fn validate(self) -> Result<()> {
        if Self::new(
            self.accepted,
            self.capacity,
            self.start_position,
            self.end_position,
            self.owner_generation,
            self.transaction_nonce,
        )? != self
        {
            bail!("forged GLM DSA index-commit plan");
        }
        Ok(())
    }
}

fn extent(rows: u32, columns: u32, bytes: u32) -> Result<usize> {
    usize::try_from(
        u64::from(rows)
            .checked_mul(u64::from(columns))
            .and_then(|elements| elements.checked_mul(u64::from(bytes)))
            .context("GLM DSA index-commit extent overflow")?,
    )
    .context("GLM DSA index-commit extent does not fit usize")
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53DsaIndexCommitBuffers {
    pub staged_tail_keys_bf16: GgmlIqBuffer,
    pub staged_tail_gates_bf16: GgmlIqBuffer,
    pub staged_tail_validity_u8: GgmlIqBuffer,
    pub persistent_tail_keys_bf16: GgmlIqBuffer,
    pub persistent_tail_gates_bf16: GgmlIqBuffer,
    pub persistent_tail_validity_u8: GgmlIqBuffer,
    pub persistent_pool_validity_u8: GgmlIqBuffer,
    pub logical_lengths_u32: GgmlIqBuffer,
    pub owner_generations_u64: GgmlIqBuffer,
    pub latent_ends_u32: GgmlIqBuffer,
    pub latent_nonces_u64: GgmlIqBuffer,
    pub index_ends_u32: GgmlIqBuffer,
    pub index_nonces_u64: GgmlIqBuffer,
    pub visibility_ends_u32: GgmlIqBuffer,
    pub visibility_nonces_u64: GgmlIqBuffer,
    pub visibility_status_u32: GgmlIqBuffer,
}

pub struct Glm53DsaIndexCommitKernel {
    commit: KernelHandle,
}

impl Glm53DsaIndexCommitKernel {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            commit: gpu.kernel(
                "glm53_dsa_index_commit",
                "atlas_glm53_dsa_index_commit_all11",
            )?,
        })
    }

    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53DsaIndexCommitPlan,
        buffers: Glm53DsaIndexCommitBuffers,
        stream: u64,
    ) -> Result<()> {
        plan.validate()?;
        validate_buffers(plan, buffers)?;
        KernelLaunch::new(gpu, self.commit)
            .grid([1, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.staged_tail_keys_bf16.ptr)
            .arg_ptr(buffers.staged_tail_gates_bf16.ptr)
            .arg_ptr(buffers.staged_tail_validity_u8.ptr)
            .arg_ptr(buffers.persistent_tail_keys_bf16.ptr)
            .arg_ptr(buffers.persistent_tail_gates_bf16.ptr)
            .arg_ptr(buffers.persistent_tail_validity_u8.ptr)
            .arg_ptr(buffers.persistent_pool_validity_u8.ptr)
            .arg_ptr(buffers.logical_lengths_u32.ptr)
            .arg_ptr(buffers.owner_generations_u64.ptr)
            .arg_ptr(buffers.latent_ends_u32.ptr)
            .arg_ptr(buffers.latent_nonces_u64.ptr)
            .arg_ptr(buffers.index_ends_u32.ptr)
            .arg_ptr(buffers.index_nonces_u64.ptr)
            .arg_ptr(buffers.visibility_ends_u32.ptr)
            .arg_ptr(buffers.visibility_nonces_u64.ptr)
            .arg_ptr(buffers.visibility_status_u32.ptr)
            .arg_u32(plan.accepted)
            .arg_u32(plan.capacity)
            .arg_u32(plan.start_position)
            .arg_u32(plan.end_position)
            .arg_u64(plan.owner_generation)
            .arg_u64(plan.transaction_nonce)
            .launch(stream)
    }
}

fn validate_buffers(
    plan: Glm53DsaIndexCommitPlan,
    buffers: Glm53DsaIndexCommitBuffers,
) -> Result<()> {
    let named = [
        (
            "staged tail keys",
            buffers.staged_tail_keys_bf16,
            plan.tail_vector_bytes,
            2,
        ),
        (
            "staged tail gates",
            buffers.staged_tail_gates_bf16,
            plan.tail_vector_bytes,
            2,
        ),
        (
            "staged tail validity",
            buffers.staged_tail_validity_u8,
            plan.tail_validity_bytes,
            1,
        ),
        (
            "persistent tail keys",
            buffers.persistent_tail_keys_bf16,
            plan.tail_vector_bytes,
            2,
        ),
        (
            "persistent tail gates",
            buffers.persistent_tail_gates_bf16,
            plan.tail_vector_bytes,
            2,
        ),
        (
            "persistent tail validity",
            buffers.persistent_tail_validity_u8,
            plan.tail_validity_bytes,
            1,
        ),
        (
            "persistent pool validity",
            buffers.persistent_pool_validity_u8,
            plan.pool_validity_bytes,
            1,
        ),
        (
            "logical lengths",
            buffers.logical_lengths_u32,
            plan.layer_u32_bytes,
            4,
        ),
        (
            "owner generations",
            buffers.owner_generations_u64,
            plan.layer_u64_bytes,
            8,
        ),
        (
            "latent ends",
            buffers.latent_ends_u32,
            plan.layer_u32_bytes,
            4,
        ),
        (
            "latent nonces",
            buffers.latent_nonces_u64,
            plan.layer_u64_bytes,
            8,
        ),
        (
            "index ends",
            buffers.index_ends_u32,
            plan.layer_u32_bytes,
            4,
        ),
        (
            "index nonces",
            buffers.index_nonces_u64,
            plan.layer_u64_bytes,
            8,
        ),
        (
            "visibility ends",
            buffers.visibility_ends_u32,
            plan.layer_u32_bytes,
            4,
        ),
        (
            "visibility nonces",
            buffers.visibility_nonces_u64,
            plan.layer_u64_bytes,
            8,
        ),
        (
            "visibility status",
            buffers.visibility_status_u32,
            plan.layer_u32_bytes,
            4,
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
            bail!("GLM DSA index-commit {name} buffer has invalid pointer, alignment, or extent");
        }
        ranges.push((
            buffer.ptr.0,
            buffer
                .ptr
                .0
                .checked_add(u64::try_from(expected)?)
                .with_context(|| format!("GLM DSA index-commit {name} address overflow"))?,
        ));
    }
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].0 < ranges[right].1 && ranges[right].0 < ranges[left].1 {
                bail!("GLM DSA index-commit device buffers overlap");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "glm53_dsa_index_commit_tests.rs"]
mod tests;
