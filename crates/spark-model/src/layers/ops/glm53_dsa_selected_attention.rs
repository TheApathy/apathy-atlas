// SPDX-License-Identifier: AGPL-3.0-only

//! BF16 selected-latent attention for GLM-5.3 DSA.
//!
//! The CUDA primitive canonicalizes the selector's score-ordered raw indices
//! into unique ascending absolute-token order before attention. This matches
//! the chronological key traversal induced by upstream's dense boolean mask.
//! `absorbed_query_bf16` is the unscaled per-head `q_h @ k_b`; the kernel
//! applies `256^-0.5`, casts FP32 softmax probabilities to BF16, and emits the
//! BF16 weighted rank-512 latent consumed by the per-head `v_b` bank.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::GgmlIqBuffer;

const HEADS: u32 = 64;
const LATENT: u32 = 512;
const SELECTED: u32 = 2_051;
const MAX_QUERIES: u32 = 8;
const MAX_POSITIONS: u32 = 1_048_576;
const THREADS: u32 = 256;
const MAX_GRID_YZ: u64 = 65_535;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53DsaSelectedStorage {
    Bf16,
    Fp8E4M3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53DsaSelectedAttentionPlan {
    pub batch: u32,
    pub queries: u32,
    pub kv_capacity: u32,
    pub heads: u32,
    pub latent: u32,
    pub selected: u32,
    pub storage: Glm53DsaSelectedStorage,
    pub grid_y: u32,
    pub grid_z: u32,
    pub query_bytes: usize,
    pub latent_cache_bytes: usize,
    pub selected_index_bytes: usize,
    pub sequence_length_bytes: usize,
    pub query_position_bytes: usize,
    pub query_validity_bytes: usize,
    pub output_bytes: usize,
}

impl Glm53DsaSelectedAttentionPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batch: u32,
        queries: u32,
        kv_capacity: u32,
        heads: u32,
        latent: u32,
        selected: u32,
        storage: Glm53DsaSelectedStorage,
    ) -> Result<Self> {
        if batch == 0 || !(1..=MAX_QUERIES).contains(&queries) {
            bail!("GLM DSA selected attention requires batch>0 and Q in 1..=8");
        }
        if !(1..=MAX_POSITIONS).contains(&kv_capacity)
            || heads != HEADS
            || latent != LATENT
            || selected != SELECTED
        {
            bail!("GLM DSA selected attention has invalid exact geometry");
        }
        if storage != Glm53DsaSelectedStorage::Bf16 {
            bail!("GLM DSA selected attention rejects FP8 until a scale ABI exists");
        }
        let rows = u64::from(batch)
            .checked_mul(u64::from(queries))
            .context("GLM DSA selected-attention row overflow")?;
        let grid_y = rows.min(MAX_GRID_YZ);
        let grid_z = rows
            .checked_add(grid_y - 1)
            .context("GLM DSA selected-attention grid rounding overflow")?
            / grid_y;
        if grid_z > MAX_GRID_YZ {
            bail!("GLM DSA selected-attention row grid exceeds CUDA y/z capacity");
        }
        let head_latent = u64::from(HEADS)
            .checked_mul(u64::from(LATENT))
            .context("GLM DSA selected-attention head geometry overflow")?;
        let cache_row = u64::from(kv_capacity)
            .checked_mul(u64::from(LATENT))
            .context("GLM DSA selected-attention cache-row overflow")?;
        let query_bytes = extent(rows, head_latent, 2)?;
        Ok(Self {
            batch,
            queries,
            kv_capacity,
            heads,
            latent,
            selected,
            storage,
            grid_y: u32::try_from(grid_y)?,
            grid_z: u32::try_from(grid_z)?,
            query_bytes,
            latent_cache_bytes: extent(u64::from(batch), cache_row, 2)?,
            selected_index_bytes: extent(rows, u64::from(SELECTED), 4)?,
            sequence_length_bytes: extent(u64::from(batch), 1, 4)?,
            query_position_bytes: extent(rows, 1, 4)?,
            query_validity_bytes: usize::try_from(rows)?,
            output_bytes: query_bytes,
        })
    }

    pub fn validate(self) -> Result<()> {
        if Self::new(
            self.batch,
            self.queries,
            self.kv_capacity,
            self.heads,
            self.latent,
            self.selected,
            self.storage,
        )? != self
        {
            bail!("forged GLM DSA selected-attention plan");
        }
        Ok(())
    }
}

fn extent(rows: u64, columns: u64, element_bytes: u64) -> Result<usize> {
    usize::try_from(
        rows.checked_mul(columns)
            .and_then(|elements| elements.checked_mul(element_bytes))
            .context("GLM DSA selected-attention byte-extent overflow")?,
    )
    .context("GLM DSA selected-attention extent does not fit usize")
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53DsaSelectedAttentionBuffers {
    pub absorbed_query_bf16: GgmlIqBuffer,
    pub latent_cache_bf16: GgmlIqBuffer,
    pub selected_indices_i32: GgmlIqBuffer,
    pub sequence_lengths_u32: GgmlIqBuffer,
    pub query_positions_u32: GgmlIqBuffer,
    pub query_validity_u8: GgmlIqBuffer,
    pub output_weighted_latent_bf16: GgmlIqBuffer,
}

pub struct Glm53DsaSelectedAttentionKernel {
    attention: KernelHandle,
}

impl Glm53DsaSelectedAttentionKernel {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            attention: gpu.kernel(
                "glm53_dsa_selected_attention",
                "atlas_glm53_dsa_selected_attention_bf16",
            )?,
        })
    }

    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53DsaSelectedAttentionPlan,
        buffers: Glm53DsaSelectedAttentionBuffers,
        stream: u64,
    ) -> Result<()> {
        plan.validate()?;
        validate_buffers(plan, buffers)?;
        KernelLaunch::new(gpu, self.attention)
            .grid([HEADS, plan.grid_y, plan.grid_z])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.absorbed_query_bf16.ptr)
            .arg_ptr(buffers.latent_cache_bf16.ptr)
            .arg_ptr(buffers.selected_indices_i32.ptr)
            .arg_ptr(buffers.sequence_lengths_u32.ptr)
            .arg_ptr(buffers.query_positions_u32.ptr)
            .arg_ptr(buffers.query_validity_u8.ptr)
            .arg_ptr(buffers.output_weighted_latent_bf16.ptr)
            .arg_u32(plan.batch)
            .arg_u32(plan.queries)
            .arg_u32(plan.kv_capacity)
            .launch(stream)
    }
}

fn validate_buffers(
    plan: Glm53DsaSelectedAttentionPlan,
    buffers: Glm53DsaSelectedAttentionBuffers,
) -> Result<()> {
    let named = [
        (
            "absorbed query",
            buffers.absorbed_query_bf16,
            plan.query_bytes,
            2,
        ),
        (
            "latent cache",
            buffers.latent_cache_bf16,
            plan.latent_cache_bytes,
            2,
        ),
        (
            "selected indices",
            buffers.selected_indices_i32,
            plan.selected_index_bytes,
            4,
        ),
        (
            "sequence lengths",
            buffers.sequence_lengths_u32,
            plan.sequence_length_bytes,
            4,
        ),
        (
            "query positions",
            buffers.query_positions_u32,
            plan.query_position_bytes,
            4,
        ),
        (
            "query validity",
            buffers.query_validity_u8,
            plan.query_validity_bytes,
            1,
        ),
        (
            "output",
            buffers.output_weighted_latent_bf16,
            plan.output_bytes,
            2,
        ),
    ];
    let mut ranges = [(0u64, 0u64); 7];
    for (slot, (name, buffer, expected, alignment)) in named.iter().copied().enumerate() {
        if buffer.bytes != expected
            || buffer.ptr == DevicePtr::NULL
            || buffer.ptr.0 % alignment != 0
        {
            bail!(
                "GLM DSA selected-attention {name} buffer has invalid pointer, alignment, or extent"
            );
        }
        ranges[slot] = (
            buffer.ptr.0,
            buffer
                .ptr
                .0
                .checked_add(u64::try_from(expected)?)
                .with_context(|| format!("GLM DSA selected-attention {name} address overflow"))?,
        );
    }
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].0 < ranges[right].1 && ranges[right].0 < ranges[left].1 {
                bail!("GLM DSA selected-attention device buffers overlap");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "glm53_dsa_selected_attention_tests.rs"]
mod tests;
