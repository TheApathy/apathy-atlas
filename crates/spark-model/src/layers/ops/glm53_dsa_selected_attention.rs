// SPDX-License-Identifier: AGPL-3.0-only

//! BF16 selected-latent attention for GLM-5.3 DSA.
//!
//! The CUDA primitive canonicalizes the selector's score-ordered raw indices
//! into unique ascending absolute-token order before attention. This matches
//! the chronological key traversal induced by upstream's dense boolean mask.
//! `absorbed_query_bf16` is the unscaled per-head `q_h @ k_b`; the kernel
//! applies `256^-0.5`, casts FP32 softmax probabilities to BF16, and emits the
//! BF16 weighted rank-512 latent consumed by the per-head `v_b` bank.

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::{GLM53_EXL3_MAX_WIDE_ROWS, GgmlIqBuffer};

const HEADS: u32 = 64;
const LATENT: u32 = 512;
const SELECTED: u32 = 2_051;
const MAX_QUERIES: u32 = GLM53_EXL3_MAX_WIDE_ROWS as u32;
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
            bail!("GLM DSA selected attention requires batch>0 and Q in 1..={MAX_QUERIES}");
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

#[derive(Debug, Clone, Copy)]
pub struct Glm53DsaHeadTransposeBuffers {
    pub input_bf16: GgmlIqBuffer,
    pub output_bf16: GgmlIqBuffer,
}

pub struct Glm53DsaSelectedAttentionKernel {
    attention: KernelHandle,
    dense_causal: KernelHandle,
    transpose_heads: KernelHandle,
}

impl Glm53DsaSelectedAttentionKernel {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            attention: gpu.kernel(
                "glm53_dsa_selected_attention",
                "atlas_glm53_dsa_selected_attention_bf16",
            )?,
            dense_causal: gpu.kernel(
                "glm53_dsa_selected_attention",
                "atlas_glm53_dsa_dense_causal_bf16",
            )?,
            transpose_heads: gpu.kernel(
                "glm53_dsa_selected_attention",
                "atlas_glm53_dsa_transpose_heads_bf16",
            )?,
        })
    }

    /// Dense causal rank-512 attention for the selector's proven full-coverage
    /// prefix. The caller retains responsibility for admitting only the
    /// layer-major `sequence_length <= SELECTED` region.
    pub fn launch_dense_causal(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53DsaSelectedAttentionPlan,
        buffers: Glm53DsaSelectedAttentionBuffers,
        query_offset: u32,
        sequence_length: u32,
        stream: u64,
    ) -> Result<()> {
        plan.validate()?;
        validate_buffers(plan, buffers)?;
        ensure!(
            plan.batch == 1
                && plan.queries > 1
                && sequence_length <= SELECTED
                && query_offset
                    .checked_add(plan.queries)
                    .is_some_and(|end| end == sequence_length),
            "GLM DSA dense causal attention requires B1, Q>1 and an exact <=2051 prefix"
        );
        // `ATLAS_GLM53_DSA_DENSE_FAST=1`: the GLM-private dense-causal kernel.
        // V and K are the same latent pointer (see the two `.arg_ptr` lines
        // below), so it keeps ONE KV tile instead of two and spends the freed
        // shared memory on a padded row stride that breaks the donor's 8-way
        // bank conflict. Bit-identical; see glm53_dsa_dense_causal_fast.cu.
        let (kernel, shared) = if dense_fast()? {
            (
                gpu.kernel(
                    "glm53_dsa_dense_causal_fast",
                    "atlas_glm53_dsa_dense_causal_fast_bf16",
                )?,
                69_376,
            )
        } else {
            (self.dense_causal, 101_120)
        };
        KernelLaunch::new(gpu, kernel)
            .grid([HEADS, plan.queries.div_ceil(32), 1])
            .block([THREADS, 1, 1])
            .shared_mem(shared)
            .arg_ptr(buffers.absorbed_query_bf16.ptr)
            .arg_ptr(buffers.latent_cache_bf16.ptr)
            .arg_ptr(buffers.latent_cache_bf16.ptr)
            .arg_ptr(buffers.output_weighted_latent_bf16.ptr)
            .arg_ptr(DevicePtr::NULL)
            .arg_u32(plan.queries)
            .arg_u32(sequence_length)
            .arg_u32(query_offset)
            .arg_u32(HEADS)
            .arg_u32(1)
            .arg_u32(LATENT)
            .arg_u32(1)
            .arg_u32(0)
            .arg_u32(1)
            .arg_f32(0.0625)
            .launch(stream)
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

    pub fn transpose_heads(
        &self,
        gpu: &dyn GpuBackend,
        rows: u32,
        width: u32,
        to_head_major: bool,
        buffers: Glm53DsaHeadTransposeBuffers,
        stream: u64,
    ) -> Result<()> {
        if !(1..=u32::try_from(GLM53_EXL3_MAX_WIDE_ROWS)?).contains(&rows)
            || !matches!(width, 256 | 512)
        {
            bail!(
                "GLM DSA head transpose requires rows 1..={GLM53_EXL3_MAX_WIDE_ROWS} and width 256 or 512"
            );
        }
        let bytes = usize::try_from(rows)?
            .checked_mul(HEADS as usize * width as usize * 2)
            .context("GLM DSA head transpose extent overflow")?;
        for (name, buffer) in [
            ("head transpose input", buffers.input_bf16),
            ("head transpose output", buffers.output_bf16),
        ] {
            if buffer.ptr == DevicePtr::NULL || buffer.bytes != bytes {
                bail!("GLM DSA {name} buffer is null or has the wrong extent");
            }
        }
        if buffers.input_bf16.ptr == buffers.output_bf16.ptr {
            bail!("GLM DSA head transpose cannot run in place");
        }
        let elements = u32::try_from(bytes / 2)?;
        KernelLaunch::new(gpu, self.transpose_heads)
            .grid([elements.div_ceil(THREADS), 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.input_bf16.ptr)
            .arg_ptr(buffers.output_bf16.ptr)
            .arg_u32(rows)
            .arg_u32(width)
            .arg_u32(u32::from(to_head_major))
            .launch(stream)
    }
}

/// `ATLAS_GLM53_DSA_DENSE_FAST=1`: single-KV-tile, bank-conflict-free
/// dense-causal DSA attention. Bit-identical to the donor kernel.
fn dense_fast() -> Result<bool> {
    use std::sync::OnceLock;
    static ON: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ON.get_or_init(|| match std::env::var("ATLAS_GLM53_DSA_DENSE_FAST") {
        Ok(v) if v == "1" => Ok(true),
        Ok(v) if v == "0" => Ok(false),
        Ok(other) => Err(format!(
            "ATLAS_GLM53_DSA_DENSE_FAST must be 0 or 1, got {other:?}"
        )),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(e) => Err(format!("ATLAS_GLM53_DSA_DENSE_FAST: {e}")),
    })
    .clone()
    .map_err(anyhow::Error::msg)
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
    // Sized from the table it mirrors, never a literal: a hand-written
    // length silently goes out of bounds the moment the table grows.
    let mut ranges = Vec::with_capacity(named.len());
    for (name, buffer, expected, alignment) in named.iter().copied() {
        if buffer.bytes != expected
            || buffer.ptr == DevicePtr::NULL
            || buffer.ptr.0 % alignment != 0
        {
            bail!(
                "GLM DSA selected-attention {name} buffer has invalid pointer, alignment, or extent"
            );
        }
        ranges.push((
            buffer.ptr.0,
            buffer
                .ptr
                .0
                .checked_add(u64::try_from(expected)?)
                .with_context(|| format!("GLM DSA selected-attention {name} address overflow"))?,
        ));
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
