// SPDX-License-Identifier: AGPL-3.0-only

//! Exact GLM-5.3 KDA q/k/v short convolution with transactional state.
//!
//! The three projections are BF16 `[B,Q,64*128]`; each depthwise weight is
//! F32 GGUF `[4,1,8192]`; state is F32 `[B,3,8192,4]`. Stage reads persistent
//! state, writes private output/state, and publishes a nonce only after every
//! stream finishes. Commit globally validates the receipt and old lengths
//! before replacing persistent state. Callers must serialize both operations
//! and downstream KDA work on one stream and invalidate the allocation after
//! any launch/stream failure. Before token zero, callers initialize both state
//! buffers and every logical length to zero; this primitive never guesses an
//! initial state. Full acceptance copies the staged snapshot; partial acceptance
//! reconstructs state from the still-live raw projections without publishing
//! rejected tokens.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::GgmlIqBuffer;

const STREAMS: u32 = 3;
const HEADS: u32 = 64;
const HEAD_DIM: u32 = 128;
const CHANNELS: u32 = HEADS * HEAD_DIM;
const KERNEL: u32 = 4;
const MAX_QUERIES: u32 = 65_520;
const MAX_POSITIONS: u32 = 1_048_576;
const MAX_BATCH: u32 = 65_535;
const THREADS: u32 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53KdaConvPlan {
    pub batch: u32,
    pub queries: u32,
    pub capacity: u32,
    pub start_position: u32,
    pub end_position: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub kernel: u32,
    pub transaction_nonce: u64,
    pub stream_bytes: usize,
    pub weight_bytes: usize,
    pub state_bytes: usize,
    pub end_bytes: usize,
    pub nonce_bytes: usize,
    pub length_bytes: usize,
}

impl Glm53KdaConvPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batch: u32,
        queries: u32,
        capacity: u32,
        start_position: u32,
        end_position: u32,
        heads: u32,
        head_dim: u32,
        kernel: u32,
        transaction_nonce: u64,
    ) -> Result<Self> {
        if !(1..=MAX_BATCH).contains(&batch) || !(1..=MAX_QUERIES).contains(&queries) {
            bail!("GLM KDA conv requires B in 1..=65535 and Q in 1..=65520");
        }
        if !(1..=MAX_POSITIONS).contains(&capacity)
            || heads != HEADS
            || head_dim != HEAD_DIM
            || kernel != KERNEL
            || transaction_nonce == 0
        {
            bail!("GLM KDA conv has invalid exact geometry or nonce");
        }
        let computed_end = start_position
            .checked_add(queries)
            .context("GLM KDA conv position overflow")?;
        if computed_end != end_position || end_position > capacity {
            bail!("GLM KDA conv requires exact start/exclusive-end extent");
        }
        let batch64 = u64::from(batch);
        let rows = batch64
            .checked_mul(u64::from(queries))
            .context("GLM KDA conv row overflow")?;
        Ok(Self {
            batch,
            queries,
            capacity,
            start_position,
            end_position,
            heads,
            head_dim,
            kernel,
            transaction_nonce,
            stream_bytes: extent(rows, u64::from(CHANNELS), 2)?,
            weight_bytes: extent(u64::from(CHANNELS), u64::from(KERNEL), 4)?,
            state_bytes: extent(batch64, u64::from(STREAMS * CHANNELS * KERNEL), 4)?,
            end_bytes: extent(batch64, 1, 4)?,
            nonce_bytes: extent(batch64, 1, 8)?,
            length_bytes: extent(batch64, 1, 4)?,
        })
    }

    pub fn validate(self) -> Result<()> {
        if Self::new(
            self.batch,
            self.queries,
            self.capacity,
            self.start_position,
            self.end_position,
            self.heads,
            self.head_dim,
            self.kernel,
            self.transaction_nonce,
        )? != self
        {
            bail!("forged GLM KDA conv plan");
        }
        Ok(())
    }
}

fn extent(rows: u64, columns: u64, element_bytes: u64) -> Result<usize> {
    usize::try_from(
        rows.checked_mul(columns)
            .and_then(|elements| elements.checked_mul(element_bytes))
            .context("GLM KDA conv byte-extent overflow")?,
    )
    .context("GLM KDA conv extent does not fit usize")
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53KdaConvBuffers {
    pub q_input_bf16: GgmlIqBuffer,
    pub k_input_bf16: GgmlIqBuffer,
    pub v_input_bf16: GgmlIqBuffer,
    pub q_weight_f32: GgmlIqBuffer,
    pub k_weight_f32: GgmlIqBuffer,
    pub v_weight_f32: GgmlIqBuffer,
    pub persistent_state_f32: GgmlIqBuffer,
    pub staged_state_f32: GgmlIqBuffer,
    pub q_output_bf16: GgmlIqBuffer,
    pub k_output_bf16: GgmlIqBuffer,
    pub v_output_bf16: GgmlIqBuffer,
    pub published_ends_u32: GgmlIqBuffer,
    pub published_nonces_u64: GgmlIqBuffer,
    pub logical_lengths_u32: GgmlIqBuffer,
}

pub struct Glm53KdaConvKernel {
    stage: KernelHandle,
    finalize: KernelHandle,
    commit: KernelHandle,
}

impl Glm53KdaConvKernel {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            stage: gpu.kernel("glm53_kda_conv", "atlas_glm53_kda_conv_f32_stage")?,
            finalize: gpu.kernel("glm53_kda_conv", "atlas_glm53_kda_conv_finalize")?,
            commit: gpu.kernel("glm53_kda_conv", "atlas_glm53_kda_conv_commit")?,
        })
    }

    pub fn launch_stage(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53KdaConvPlan,
        buffers: Glm53KdaConvBuffers,
        stream: u64,
    ) -> Result<()> {
        plan.validate()?;
        validate_buffers(plan, buffers)?;
        gpu.memset_async(
            buffers.published_nonces_u64.ptr,
            0,
            plan.nonce_bytes,
            stream,
        )?;
        gpu.memset_async(buffers.published_ends_u32.ptr, 0, plan.end_bytes, stream)?;
        KernelLaunch::new(gpu, self.stage)
            .grid([CHANNELS.div_ceil(THREADS), plan.batch, STREAMS])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.q_input_bf16.ptr)
            .arg_ptr(buffers.k_input_bf16.ptr)
            .arg_ptr(buffers.v_input_bf16.ptr)
            .arg_ptr(buffers.q_weight_f32.ptr)
            .arg_ptr(buffers.k_weight_f32.ptr)
            .arg_ptr(buffers.v_weight_f32.ptr)
            .arg_ptr(buffers.persistent_state_f32.ptr)
            .arg_ptr(buffers.staged_state_f32.ptr)
            .arg_ptr(buffers.q_output_bf16.ptr)
            .arg_ptr(buffers.k_output_bf16.ptr)
            .arg_ptr(buffers.v_output_bf16.ptr)
            .arg_u32(plan.batch)
            .arg_u32(plan.queries)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.finalize)
            .grid([1, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.published_ends_u32.ptr)
            .arg_ptr(buffers.published_nonces_u64.ptr)
            .arg_ptr(buffers.logical_lengths_u32.ptr)
            .arg_u32(plan.batch)
            .arg_u32(plan.queries)
            .arg_u32(plan.capacity)
            .arg_u32(plan.start_position)
            .arg_u32(plan.end_position)
            .arg_u64(plan.transaction_nonce)
            .launch(stream)
    }

    pub fn launch_commit(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53KdaConvPlan,
        accepted: u32,
        buffers: Glm53KdaConvBuffers,
        stream: u64,
    ) -> Result<()> {
        plan.validate()?;
        if accepted > plan.queries {
            bail!("GLM KDA conv commit requires accepted<=Q");
        }
        validate_buffers(plan, buffers)?;
        KernelLaunch::new(gpu, self.commit)
            .grid([1, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.q_input_bf16.ptr)
            .arg_ptr(buffers.k_input_bf16.ptr)
            .arg_ptr(buffers.v_input_bf16.ptr)
            .arg_ptr(buffers.staged_state_f32.ptr)
            .arg_ptr(buffers.persistent_state_f32.ptr)
            .arg_ptr(buffers.published_ends_u32.ptr)
            .arg_ptr(buffers.published_nonces_u64.ptr)
            .arg_ptr(buffers.logical_lengths_u32.ptr)
            .arg_u32(plan.batch)
            .arg_u32(plan.queries)
            .arg_u32(accepted)
            .arg_u32(plan.capacity)
            .arg_u32(plan.start_position)
            .arg_u32(plan.end_position)
            .arg_u64(plan.transaction_nonce)
            .launch(stream)
    }
}

fn validate_buffers(plan: Glm53KdaConvPlan, buffers: Glm53KdaConvBuffers) -> Result<()> {
    let named = [
        ("q input", buffers.q_input_bf16, plan.stream_bytes, 2),
        ("k input", buffers.k_input_bf16, plan.stream_bytes, 2),
        ("v input", buffers.v_input_bf16, plan.stream_bytes, 2),
        ("q weight", buffers.q_weight_f32, plan.weight_bytes, 4),
        ("k weight", buffers.k_weight_f32, plan.weight_bytes, 4),
        ("v weight", buffers.v_weight_f32, plan.weight_bytes, 4),
        (
            "persistent state",
            buffers.persistent_state_f32,
            plan.state_bytes,
            4,
        ),
        (
            "staged state",
            buffers.staged_state_f32,
            plan.state_bytes,
            4,
        ),
        ("q output", buffers.q_output_bf16, plan.stream_bytes, 2),
        ("k output", buffers.k_output_bf16, plan.stream_bytes, 2),
        ("v output", buffers.v_output_bf16, plan.stream_bytes, 2),
        (
            "published ends",
            buffers.published_ends_u32,
            plan.end_bytes,
            4,
        ),
        (
            "published nonces",
            buffers.published_nonces_u64,
            plan.nonce_bytes,
            8,
        ),
        (
            "logical lengths",
            buffers.logical_lengths_u32,
            plan.length_bytes,
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
            bail!("GLM KDA conv {name} buffer has invalid pointer, alignment, or extent");
        }
        ranges.push((
            buffer.ptr.0,
            buffer
                .ptr
                .0
                .checked_add(u64::try_from(expected)?)
                .with_context(|| format!("GLM KDA conv {name} address overflow"))?,
        ));
    }
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].0 < ranges[right].1 && ranges[right].0 < ranges[left].1 {
                bail!("GLM KDA conv device buffers overlap");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "glm53_kda_conv_tests.rs"]
mod tests;
