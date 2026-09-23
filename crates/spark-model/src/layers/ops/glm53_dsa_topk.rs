// SPDX-License-Identifier: AGPL-3.0-only

//! Exact GLM-5.3 DSA pool selection and raw-token expansion.
//! Atlas deliberately resolves upstream's unspecified equal-score `topk` order
//! as `(score descending, pool index ascending)`. All other ordering, causal
//! masking, tail, padding, and `i32[B,Q,2051]` behavior follows GLM5Next.
//! Atlas sequence slots are dense logical streams: raw indices are derived from
//! pool IDs, while only constant-size length/position metadata is passed here.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::{GLM53_EXL3_MAX_WIDE_ROWS, GgmlIqBuffer};

const INDEX_TOPK: u32 = 2_048;
const KPOOL: u32 = 4;
#[cfg(test)]
const SELECTED_POOLS: u32 = INDEX_TOPK / KPOOL;
const OUTPUT_WIDTH: u32 = INDEX_TOPK + KPOOL - 1;
const MAX_QUERIES: u32 = GLM53_EXL3_MAX_WIDE_ROWS as u32;
const MAX_POOLS: u32 = 262_144;
const MAX_POSITIONS: u32 = 1_048_576;
const THREADS: u32 = 256;
const MAX_GRID_YZ: u64 = 65_535;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53DsaTopkPlan {
    pub batch: u32,
    pub queries: u32,
    pub pools: u32,
    pub kv_capacity: u32,
    pub grid_y: u32,
    pub grid_z: u32,
    pub score_bytes: usize,
    pub pool_validity_bytes: usize,
    pub sequence_length_bytes: usize,
    pub query_position_bytes: usize,
    pub query_validity_bytes: usize,
    pub tail_validity_bytes: usize,
    pub output_bytes: usize,
}

impl Glm53DsaTopkPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batch: u32,
        queries: u32,
        pools: u32,
        kv_capacity: u32,
        index_topk: u32,
        index_kpool: u32,
        always_select_tail: bool,
    ) -> Result<Self> {
        if batch == 0 || !(1..=MAX_QUERIES).contains(&queries) {
            bail!("GLM DSA top-k requires batch>0 and Q in 1..={MAX_QUERIES}");
        }
        if !(1..=MAX_POOLS).contains(&pools) || !(1..=MAX_POSITIONS).contains(&kv_capacity) {
            bail!("GLM DSA top-k has invalid pool or sequence geometry");
        }
        if pools > kv_capacity.div_ceil(KPOOL) {
            bail!("GLM DSA top-k pool count exceeds kpool4 capacity");
        }
        if index_topk != INDEX_TOPK || index_kpool != KPOOL || !always_select_tail {
            bail!("GLM DSA top-k requires exact topk2048/kpool4/tail-true policy");
        }
        let rows = u64::from(batch)
            .checked_mul(u64::from(queries))
            .context("GLM DSA top-k row overflow")?;
        let grid_y = rows.min(MAX_GRID_YZ);
        let grid_z = rows
            .checked_add(grid_y - 1)
            .context("GLM DSA top-k grid rounding overflow")?
            / grid_y;
        if grid_z > MAX_GRID_YZ {
            bail!("GLM DSA top-k row grid exceeds CUDA y/z capacity");
        }
        let pool_rows = u64::from(batch)
            .checked_mul(u64::from(pools))
            .context("GLM DSA top-k pool-row overflow")?;
        Ok(Self {
            batch,
            queries,
            pools,
            kv_capacity,
            grid_y: u32::try_from(grid_y)?,
            grid_z: u32::try_from(grid_z)?,
            score_bytes: extent(rows, u64::from(pools), 4)?,
            pool_validity_bytes: usize::try_from(pool_rows)?,
            sequence_length_bytes: extent(u64::from(batch), 1, 4)?,
            query_position_bytes: extent(rows, 1, 4)?,
            query_validity_bytes: usize::try_from(rows)?,
            tail_validity_bytes: extent(u64::from(batch), u64::from(KPOOL - 1), 1)?,
            output_bytes: extent(rows, u64::from(OUTPUT_WIDTH), 4)?,
        })
    }

    pub fn validate(self) -> Result<()> {
        let rebuilt = Self::new(
            self.batch,
            self.queries,
            self.pools,
            self.kv_capacity,
            INDEX_TOPK,
            KPOOL,
            true,
        )?;
        if rebuilt != self {
            bail!("forged GLM DSA top-k plan");
        }
        Ok(())
    }
}

fn extent(rows: u64, columns: u64, element_bytes: u64) -> Result<usize> {
    usize::try_from(
        rows.checked_mul(columns)
            .and_then(|elements| elements.checked_mul(element_bytes))
            .context("GLM DSA top-k byte-extent overflow")?,
    )
    .context("GLM DSA top-k extent does not fit usize")
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53DsaTopkBuffers {
    pub scores_f32: GgmlIqBuffer,
    pub pool_validity_u8: GgmlIqBuffer,
    pub sequence_lengths_u32: GgmlIqBuffer,
    pub query_positions_u32: GgmlIqBuffer,
    pub query_validity_u8: GgmlIqBuffer,
    pub tail_validity_u8: GgmlIqBuffer,
    pub output_indices_i32: GgmlIqBuffer,
}

pub struct Glm53DsaTopkKernel {
    topk: KernelHandle,
}

impl Glm53DsaTopkKernel {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            topk: gpu.kernel("glm53_dsa_topk", "atlas_glm53_dsa_topk_k4")?,
        })
    }

    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53DsaTopkPlan,
        buffers: Glm53DsaTopkBuffers,
        stream: u64,
    ) -> Result<()> {
        plan.validate()?;
        validate_buffers(plan, buffers)?;
        KernelLaunch::new(gpu, self.topk)
            .grid([1, plan.grid_y, plan.grid_z])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.scores_f32.ptr)
            .arg_ptr(buffers.pool_validity_u8.ptr)
            .arg_ptr(buffers.sequence_lengths_u32.ptr)
            .arg_ptr(buffers.query_positions_u32.ptr)
            .arg_ptr(buffers.query_validity_u8.ptr)
            .arg_ptr(buffers.tail_validity_u8.ptr)
            .arg_ptr(buffers.output_indices_i32.ptr)
            .arg_u32(plan.batch)
            .arg_u32(plan.queries)
            .arg_u32(plan.pools)
            .arg_u32(plan.kv_capacity)
            .launch(stream)
    }
}

fn validate_buffers(plan: Glm53DsaTopkPlan, buffers: Glm53DsaTopkBuffers) -> Result<()> {
    let named = [
        ("scores", buffers.scores_f32, plan.score_bytes, 4u64),
        (
            "pool validity",
            buffers.pool_validity_u8,
            plan.pool_validity_bytes,
            1,
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
            "tail validity",
            buffers.tail_validity_u8,
            plan.tail_validity_bytes,
            1,
        ),
        ("output", buffers.output_indices_i32, plan.output_bytes, 4),
    ];
    // Sized from the table it mirrors, never a literal: a hand-written
    // length silently goes out of bounds the moment the table grows.
    let mut ranges = Vec::with_capacity(named.len());
    for (name, buffer, expected, alignment) in named.iter().copied() {
        if buffer.bytes != expected
            || buffer.ptr == DevicePtr::NULL
            || buffer.ptr.0 % alignment != 0
        {
            bail!("GLM DSA top-k {name} buffer has invalid pointer, alignment, or extent");
        }
        ranges.push((
            buffer.ptr.0,
            buffer
                .ptr
                .0
                .checked_add(u64::try_from(expected)?)
                .with_context(|| format!("GLM DSA top-k {name} address overflow"))?,
        ));
    }
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].0 < ranges[right].1 && ranges[right].0 < ranges[left].1 {
                bail!("GLM DSA top-k device buffers overlap");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "glm53_dsa_topk_tests.rs"]
mod tests;
