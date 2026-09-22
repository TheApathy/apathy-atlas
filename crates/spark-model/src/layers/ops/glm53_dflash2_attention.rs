// SPDX-License-Identifier: AGPL-3.0-only

//! Checked paged rectangular attention for GLM-5.3 DFlash2.
//! Caller slices target input at `source_context_skip_tokens`, rebases after
//! `past_drop_tokens`, then applies the published success/failure cache length.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::DenseWeight;

use super::{GgmlIqBuffer, fill_slots_from_block_table, reshape_and_cache, rms_norm};

const MAX_ABSOLUTE_POSITIONS: u64 = 1_048_576;
const MAX_NOISE_ROWS: u64 = 8;
const MAX_RETAINED_CONTEXT: u32 = 2047;
const NUM_Q_HEADS: u32 = 32;
const NUM_KV_HEADS: u32 = 8;
const HEAD_DIM: u32 = 128;
const RMS_NORM_EPS: f32 = 1.0e-5;
const ROPE_THETA: f32 = 10_000.0;
const SLIDING_WINDOW: u32 = 2048;
const CACHE_BLOCK_SIZE: u32 = 16;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53Dflash2AttentionPlan {
    pub batch: u32,
    pub noise_tokens: u32,
    pub new_context_tokens: u32,
    pub past_retained_tokens: u32,
    pub absolute_context_end: u32,
    pub target_tail_tokens: u32,
    pub source_context_skip_tokens: u32,
    pub kept_past_tokens: u32,
    pub past_drop_tokens: u32,
    pub local_context_tokens: u32,
    pub provisional_cache_len: u32,
    pub logical_cache_blocks: u32,
    pub physical_cache_blocks: u32,
    pub q_bytes: usize,
    pub target_kv_bytes: usize,
    pub noise_kv_bytes: usize,
    pub norm_weight_bytes: usize,
    pub target_slots_bytes: usize,
    pub noise_slots_bytes: usize,
    pub block_tables_bytes: usize,
    pub cache_pool_bytes: usize,
}
impl Glm53Dflash2AttentionPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batch: u32,
        noise_tokens: u32,
        new_context_tokens: u32,
        past_retained_tokens: u32,
        absolute_context_end: u32,
        physical_cache_blocks: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        rms_norm_eps: f32,
        rope_theta: f32,
        sliding_window: u32,
        cache_block_size: u32,
    ) -> Result<Self> {
        if batch == 0 || noise_tokens == 0 || new_context_tokens == 0 {
            bail!("GLM DFlash2 attention requires nonzero batch, noise, and target context");
        }
        let q_rows = u64::from(batch)
            .checked_mul(u64::from(noise_tokens))
            .context("GLM DFlash2 attention Q row overflow")?;
        if noise_tokens > MAX_NOISE_ROWS as u32 || q_rows > MAX_NOISE_ROWS {
            bail!("GLM DFlash2 attention requires T and B*T in 1..=8");
        }
        if past_retained_tokens > MAX_RETAINED_CONTEXT {
            bail!("GLM DFlash2 attention retained context exceeds its visible window");
        }
        if u64::from(new_context_tokens) > u64::from(absolute_context_end)
            || u64::from(absolute_context_end) + u64::from(noise_tokens) > MAX_ABSOLUTE_POSITIONS
        {
            bail!("GLM DFlash2 attention absolute position exceeds 1,048,575");
        }
        if num_q_heads != NUM_Q_HEADS
            || num_kv_heads != NUM_KV_HEADS
            || head_dim != HEAD_DIM
            || cache_block_size != CACHE_BLOCK_SIZE
        {
            bail!("GLM DFlash2 attention geometry is not Q32/KV8/HD128/block16");
        }
        if rms_norm_eps.to_bits() != RMS_NORM_EPS.to_bits()
            || rope_theta.to_bits() != ROPE_THETA.to_bits()
            || sliding_window != SLIDING_WINDOW
        {
            bail!("GLM DFlash2 attention normalization, RoPE, or window config drift");
        }
        let target_tail_tokens = new_context_tokens.min(MAX_RETAINED_CONTEXT);
        let source_context_skip_tokens = new_context_tokens - target_tail_tokens;
        let past_capacity = MAX_RETAINED_CONTEXT - target_tail_tokens;
        let kept_past_tokens = past_retained_tokens.min(past_capacity);
        let past_drop_tokens = past_retained_tokens - kept_past_tokens;
        let local_context_tokens = kept_past_tokens + target_tail_tokens;
        let provisional_cache_len = local_context_tokens
            .checked_add(noise_tokens)
            .context("GLM DFlash2 attention local cache length overflow")?;
        let logical_cache_blocks = provisional_cache_len.div_ceil(CACHE_BLOCK_SIZE);
        let minimum_physical_blocks = u64::from(batch)
            .checked_mul(u64::from(logical_cache_blocks))
            .context("GLM DFlash2 attention cache-block count overflow")?;
        if physical_cache_blocks == 0 || u64::from(physical_cache_blocks) < minimum_physical_blocks
        {
            bail!("GLM DFlash2 attention cache pool cannot hold all local windows");
        }
        let target_rows = u64::from(batch) * u64::from(target_tail_tokens);
        let elements = |rows: u64, heads: u32, label: &str| -> Result<u64> {
            rows.checked_mul(u64::from(heads))
                .and_then(|count| count.checked_mul(u64::from(HEAD_DIM)))
                .with_context(|| format!("GLM DFlash2 attention {label} element overflow"))
        };
        let bytes = |count: u64, width: u64, label: &str| -> Result<usize> {
            usize::try_from(
                count
                    .checked_mul(width)
                    .with_context(|| format!("GLM DFlash2 attention {label} byte overflow"))?,
            )
            .context("GLM DFlash2 attention extent exceeds usize")
        };
        let cache_elements = u64::from(physical_cache_blocks)
            * u64::from(CACHE_BLOCK_SIZE)
            * u64::from(NUM_KV_HEADS)
            * u64::from(HEAD_DIM);
        Ok(Self {
            batch,
            noise_tokens,
            new_context_tokens,
            past_retained_tokens,
            absolute_context_end,
            target_tail_tokens,
            source_context_skip_tokens,
            kept_past_tokens,
            past_drop_tokens,
            local_context_tokens,
            provisional_cache_len,
            logical_cache_blocks,
            physical_cache_blocks,
            q_bytes: bytes(elements(q_rows, NUM_Q_HEADS, "Q")?, 2, "Q")?,
            target_kv_bytes: bytes(
                elements(target_rows, NUM_KV_HEADS, "target KV")?,
                2,
                "target KV",
            )?,
            noise_kv_bytes: bytes(elements(q_rows, NUM_KV_HEADS, "noise KV")?, 2, "noise KV")?,
            norm_weight_bytes: bytes(u64::from(HEAD_DIM), 2, "norm weight")?,
            target_slots_bytes: bytes(target_rows, 8, "target slots")?,
            noise_slots_bytes: bytes(q_rows, 8, "noise slots")?,
            block_tables_bytes: bytes(
                u64::from(batch) * u64::from(logical_cache_blocks),
                4,
                "block tables",
            )?,
            cache_pool_bytes: bytes(cache_elements, 2, "cache pool")?,
        })
    }
    fn validate(self) -> Result<()> {
        let expected = Self::new(
            self.batch,
            self.noise_tokens,
            self.new_context_tokens,
            self.past_retained_tokens,
            self.absolute_context_end,
            self.physical_cache_blocks,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            RMS_NORM_EPS,
            ROPE_THETA,
            SLIDING_WINDOW,
            CACHE_BLOCK_SIZE,
        )?;
        if self != expected {
            bail!("GLM DFlash2 attention plan was modified after admission");
        }
        Ok(())
    }
    pub fn success_cache_len(self) -> u32 {
        self.local_context_tokens
    }
    pub fn failure_cache_len(self) -> u32 {
        self.kept_past_tokens
    }
}
#[derive(Debug, Clone, Copy)]
pub struct Glm53Dflash2AttentionBuffers {
    pub q_noise_bf16: GgmlIqBuffer,
    pub target_tail_k_bf16: GgmlIqBuffer,
    pub target_tail_v_bf16: GgmlIqBuffer,
    pub noise_k_bf16: GgmlIqBuffer,
    pub noise_v_bf16: GgmlIqBuffer,
    pub output_bf16: GgmlIqBuffer,
    pub q_norm_weight_bf16: GgmlIqBuffer,
    pub k_norm_weight_bf16: GgmlIqBuffer,
    pub target_slots_i64: GgmlIqBuffer,
    pub noise_slots_i64: GgmlIqBuffer,
    pub block_tables_u32: GgmlIqBuffer,
    pub k_cache_bf16: GgmlIqBuffer,
    pub v_cache_bf16: GgmlIqBuffer,
}
pub struct Glm53Dflash2AttentionKernels {
    rms_norm_vanilla: KernelHandle,
    rectangular_rope: KernelHandle,
    fill_slots: KernelHandle,
    reshape_cache: KernelHandle,
    paged_h128: KernelHandle,
}
impl Glm53Dflash2AttentionKernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            rms_norm_vanilla: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            rectangular_rope: gpu
                .kernel("glm53_dflash2_attention", "atlas_glm53_dflash2_rope_rect")?,
            fill_slots: gpu.kernel("metadata_fill", "fill_slots_from_block_table")?,
            reshape_cache: gpu.kernel("reshape_and_cache", "reshape_and_cache_flash")?,
            paged_h128: gpu.kernel(
                "glm53_dflash2_attention",
                "atlas_glm53_dflash2_prefill_paged_h128",
            )?,
        })
    }
    pub fn execute(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53Dflash2AttentionPlan,
        buffers: Glm53Dflash2AttentionBuffers,
        stream: u64,
    ) -> Result<()> {
        plan.validate()?;
        validate_buffers(plan, buffers)?;
        let q_norm_rows = norm_rows(plan.batch, plan.noise_tokens, NUM_Q_HEADS)?;
        let target_norm_rows = norm_rows(plan.batch, plan.target_tail_tokens, NUM_KV_HEADS)?;
        let noise_norm_rows = norm_rows(plan.batch, plan.noise_tokens, NUM_KV_HEADS)?;
        let target_rows = norm_rows(plan.batch, plan.target_tail_tokens, 1)?;
        let noise_rows = norm_rows(plan.batch, plan.noise_tokens, 1)?;
        let q_stride = per_batch(plan.q_bytes, plan.batch)?;
        let target_slots_stride = per_batch(plan.target_slots_bytes, plan.batch)?;
        let noise_slots_stride = per_batch(plan.noise_slots_bytes, plan.batch)?;
        let table_stride = per_batch(plan.block_tables_bytes, plan.batch)?;
        preflight_sequence_offsets(
            plan,
            buffers,
            q_stride,
            target_slots_stride,
            noise_slots_stride,
            table_stride,
        )?;
        let q_weight = DenseWeight {
            weight: buffers.q_norm_weight_bf16.ptr,
        };
        let k_weight = DenseWeight {
            weight: buffers.k_norm_weight_bf16.ptr,
        };
        norm_in_place(
            gpu,
            self.rms_norm_vanilla,
            buffers.q_noise_bf16.ptr,
            &q_weight,
            q_norm_rows,
            stream,
        )?;
        norm_in_place(
            gpu,
            self.rms_norm_vanilla,
            buffers.target_tail_k_bf16.ptr,
            &k_weight,
            target_norm_rows,
            stream,
        )?;
        norm_in_place(
            gpu,
            self.rms_norm_vanilla,
            buffers.noise_k_bf16.ptr,
            &k_weight,
            noise_norm_rows,
            stream,
        )?;
        KernelLaunch::new(gpu, self.rectangular_rope)
            .grid([
                NUM_Q_HEADS + 2 * NUM_KV_HEADS,
                div_ceil(plan.target_tail_tokens.max(plan.noise_tokens), 2),
                plan.batch,
            ])
            .block([128, 1, 1])
            .arg_ptr(buffers.q_noise_bf16.ptr)
            .arg_ptr(buffers.target_tail_k_bf16.ptr)
            .arg_ptr(buffers.noise_k_bf16.ptr)
            .arg_u32(plan.noise_tokens)
            .arg_u32(plan.target_tail_tokens)
            .arg_u32(plan.absolute_context_end - plan.target_tail_tokens)
            .arg_u32(plan.absolute_context_end)
            .arg_u32(NUM_Q_HEADS)
            .arg_u32(NUM_KV_HEADS)
            .arg_u32(HEAD_DIM)
            .arg_f32(ROPE_THETA)
            .launch(stream)?;
        for batch in 0..plan.batch {
            let index = usize::try_from(batch)?;
            let table = checked_offset(
                buffers.block_tables_u32.ptr,
                index,
                table_stride,
                "block table",
            )?;
            fill_slots_from_block_table(
                gpu,
                self.fill_slots,
                checked_offset(
                    buffers.target_slots_i64.ptr,
                    index,
                    target_slots_stride,
                    "target slots",
                )?,
                table,
                plan.kept_past_tokens,
                plan.target_tail_tokens,
                CACHE_BLOCK_SIZE,
                stream,
            )?;
            fill_slots_from_block_table(
                gpu,
                self.fill_slots,
                checked_offset(
                    buffers.noise_slots_i64.ptr,
                    index,
                    noise_slots_stride,
                    "noise slots",
                )?,
                table,
                plan.local_context_tokens,
                plan.noise_tokens,
                CACHE_BLOCK_SIZE,
                stream,
            )?;
        }
        reshape_and_cache(
            gpu,
            self.reshape_cache,
            buffers.target_tail_k_bf16.ptr,
            buffers.target_tail_v_bf16.ptr,
            buffers.k_cache_bf16.ptr,
            buffers.v_cache_bf16.ptr,
            buffers.target_slots_i64.ptr,
            target_rows,
            NUM_KV_HEADS,
            HEAD_DIM,
            CACHE_BLOCK_SIZE,
            NUM_KV_HEADS * HEAD_DIM,
            NUM_KV_HEADS * HEAD_DIM,
            0,
            stream,
        )?;
        reshape_and_cache(
            gpu,
            self.reshape_cache,
            buffers.noise_k_bf16.ptr,
            buffers.noise_v_bf16.ptr,
            buffers.k_cache_bf16.ptr,
            buffers.v_cache_bf16.ptr,
            buffers.noise_slots_i64.ptr,
            noise_rows,
            NUM_KV_HEADS,
            HEAD_DIM,
            CACHE_BLOCK_SIZE,
            NUM_KV_HEADS * HEAD_DIM,
            NUM_KV_HEADS * HEAD_DIM,
            0,
            stream,
        )?;
        for batch in 0..plan.batch {
            let index = usize::try_from(batch)?;
            launch_paged_h128(
                self.paged_h128,
                gpu,
                plan,
                checked_offset(buffers.q_noise_bf16.ptr, index, q_stride, "Q")?,
                checked_offset(buffers.output_bf16.ptr, index, q_stride, "output")?,
                buffers,
                checked_offset(
                    buffers.block_tables_u32.ptr,
                    index,
                    table_stride,
                    "block table",
                )?,
                stream,
            )?;
        }
        Ok(())
    }
}
fn norm_rows(batch: u32, tokens: u32, heads: u32) -> Result<u32> {
    u32::try_from(u64::from(batch) * u64::from(tokens) * u64::from(heads))
        .context("GLM DFlash2 attention norm-row count overflow")
}

fn norm_in_place(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    data: DevicePtr,
    weight: &DenseWeight,
    rows: u32,
    stream: u64,
) -> Result<()> {
    rms_norm(
        gpu,
        kernel,
        data,
        weight,
        data,
        rows,
        HEAD_DIM,
        RMS_NORM_EPS,
        stream,
    )
}

fn per_batch(total: usize, batch: u32) -> Result<usize> {
    let batch = usize::try_from(batch)?;
    if total % batch != 0 {
        bail!("GLM DFlash2 attention per-batch extent is not integral");
    }
    Ok(total / batch)
}

fn preflight_sequence_offsets(
    plan: Glm53Dflash2AttentionPlan,
    b: Glm53Dflash2AttentionBuffers,
    q: usize,
    target: usize,
    noise: usize,
    table: usize,
) -> Result<()> {
    for batch in 0..plan.batch {
        let index = usize::try_from(batch)?;
        checked_offset(b.q_noise_bf16.ptr, index, q, "Q")?;
        checked_offset(b.output_bf16.ptr, index, q, "output")?;
        checked_offset(b.target_slots_i64.ptr, index, target, "target slots")?;
        checked_offset(b.noise_slots_i64.ptr, index, noise, "noise slots")?;
        checked_offset(b.block_tables_u32.ptr, index, table, "block table")?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_paged_h128(
    kernel: KernelHandle,
    gpu: &dyn GpuBackend,
    plan: Glm53Dflash2AttentionPlan,
    q: DevicePtr,
    output: DevicePtr,
    buffers: Glm53Dflash2AttentionBuffers,
    block_table: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([NUM_Q_HEADS, div_ceil(plan.noise_tokens, 32), 1])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(buffers.k_cache_bf16.ptr)
        .arg_ptr(buffers.v_cache_bf16.ptr)
        .arg_ptr(output)
        .arg_ptr(block_table)
        .arg_u32(plan.noise_tokens)
        .arg_u32(plan.provisional_cache_len)
        .arg_u32(plan.local_context_tokens)
        .arg_u32(NUM_Q_HEADS)
        .arg_u32(NUM_KV_HEADS)
        .arg_u32(HEAD_DIM)
        .arg_u32(CACHE_BLOCK_SIZE)
        .arg_u32(SLIDING_WINDOW)
        .arg_u32(0)
        .arg_f32(1.0 / (HEAD_DIM as f32).sqrt())
        .launch(stream)
}

fn checked_offset(base: DevicePtr, index: usize, stride: usize, label: &str) -> Result<DevicePtr> {
    let offset = index
        .checked_mul(stride)
        .with_context(|| format!("GLM DFlash2 attention {label} offset overflow"))?;
    Ok(DevicePtr(
        base.0
            .checked_add(u64::try_from(offset)?)
            .with_context(|| format!("GLM DFlash2 attention {label} address overflow"))?,
    ))
}

fn validate_buffers(
    plan: Glm53Dflash2AttentionPlan,
    b: Glm53Dflash2AttentionBuffers,
) -> Result<()> {
    let named = [
        ("Q", b.q_noise_bf16, plan.q_bytes, 16),
        ("target K", b.target_tail_k_bf16, plan.target_kv_bytes, 16),
        ("target V", b.target_tail_v_bf16, plan.target_kv_bytes, 16),
        ("noise K", b.noise_k_bf16, plan.noise_kv_bytes, 16),
        ("noise V", b.noise_v_bf16, plan.noise_kv_bytes, 16),
        ("output", b.output_bf16, plan.q_bytes, 16),
        ("Q norm", b.q_norm_weight_bf16, plan.norm_weight_bytes, 4),
        ("K norm", b.k_norm_weight_bf16, plan.norm_weight_bytes, 4),
        (
            "target slots",
            b.target_slots_i64,
            plan.target_slots_bytes,
            8,
        ),
        ("noise slots", b.noise_slots_i64, plan.noise_slots_bytes, 8),
        (
            "block tables",
            b.block_tables_u32,
            plan.block_tables_bytes,
            4,
        ),
        ("K cache", b.k_cache_bf16, plan.cache_pool_bytes, 16),
        ("V cache", b.v_cache_bf16, plan.cache_pool_bytes, 16),
    ];
    // Sized from the table it mirrors, never a literal: a hand-written
    // length silently goes out of bounds the moment the table grows.
    let mut ranges = Vec::with_capacity(named.len());
    for (name, buffer, expected, alignment) in named.iter().copied() {
        if buffer.ptr == DevicePtr::NULL
            || buffer.bytes != expected
            || buffer.ptr.0 % alignment != 0
        {
            bail!("GLM DFlash2 attention {name} is null, misaligned, or has a wrong extent");
        }
        ranges.push((
            buffer.ptr.0,
            buffer
                .ptr
                .0
                .checked_add(u64::try_from(buffer.bytes)?)
                .with_context(|| format!("GLM DFlash2 attention {name} address overflow"))?,
        ));
    }
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].0 < ranges[right].1 && ranges[right].0 < ranges[left].1 {
                bail!("GLM DFlash2 attention device buffers overlap");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "glm53_dflash2_attention_tests.rs"]
mod tests;
