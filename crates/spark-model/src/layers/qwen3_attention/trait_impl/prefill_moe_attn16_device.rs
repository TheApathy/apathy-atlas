// SPDX-License-Identifier: AGPL-3.0-only
//! Optional device-only launch consolidation for the exact M16 attention core.
use crate::layers::{ops, qwen4_prefill_moe};
use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
const MODULE: &str = "qwen4_attn16_device";
#[derive(Clone, Copy)]
pub(super) struct DeviceKernels {
    mrope: KernelHandle,
    expand_meta16: KernelHandle,
    expand_meta32: KernelHandle,
}
fn span(ptr: DevicePtr, bytes: usize) -> Result<(u64, u64)> {
    ensure!(!ptr.is_null(), "device16 received a null device pointer");
    let end = ptr
        .0
        .checked_add(u64::try_from(bytes)?)
        .ok_or_else(|| anyhow::anyhow!("device16 pointer span overflow"))?;
    Ok((ptr.0, end))
}

fn disjoint(a: (u64, u64), b: (u64, u64)) -> bool {
    a.1 <= b.0 || b.1 <= a.0
}

impl DeviceKernels {
    pub(super) fn load_if_selected(gpu: &dyn GpuBackend) -> Result<Option<Self>> {
        if !qwen4_prefill_moe::attn16::device_selected()? {
            return Ok(None);
        }
        Ok(Some(Self {
            mrope: gpu.kernel(MODULE, "rope_forward_mrope_interleaved_strided")?,
            expand_meta16: gpu.kernel(MODULE, "qwen4_attn16_expand_meta")?,
            expand_meta32: gpu.kernel(MODULE, "qwen4_attn32_expand_meta")?,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn mrope(
        self,
        gpu: &dyn GpuBackend,
        qkv: DevicePtr,
        pos_t: DevicePtr,
        pos_h: DevicePtr,
        pos_w: DevicePtr,
        row_stride_bf16: u32,
        k_offset_bf16: u32,
        nq: u32,
        nkv: u32,
        head_dim: u32,
        rotary_dim: u32,
        theta: f32,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            row_stride_bf16 == 13_312
                && k_offset_bf16 == 12_288
                && nq == 24
                && nkv == 2
                && head_dim == 256,
            "device16 MRoPE requires canonical packed QKV geometry"
        );
        ensure!(
            rotary_dim >= 2 && rotary_dim <= head_dim && rotary_dim.is_multiple_of(2),
            "device16 MRoPE rotary dimension is invalid"
        );
        ensure!(
            theta.is_finite() && theta > 0.0,
            "device16 MRoPE theta is invalid"
        );
        ensure!(
            rows == 16 || rows == 32,
            "device attention rows must be 16 or 32"
        );
        let qkv_bytes = usize::try_from(rows)?
            .checked_mul(usize::try_from(row_stride_bf16)?)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("device16 QKV span overflow"))?;
        let qkv_span = span(qkv, qkv_bytes)?;
        for positions in [pos_t, pos_h, pos_w] {
            ensure!(
                disjoint(qkv_span, span(positions, rows as usize * 4)?),
                "device16 positions overlap writable QKV"
            );
        }
        let pos_per_block = 128 / (rotary_dim / 2);
        ensure!(
            pos_per_block > 0,
            "device16 MRoPE block geometry is invalid"
        );
        KernelLaunch::new(gpu, self.mrope)
            .grid([nq + nkv, div_ceil(rows, pos_per_block), 1])
            .block([128, 1, 1])
            .arg_ptr(qkv)
            .arg_ptr(pos_t)
            .arg_ptr(pos_h)
            .arg_ptr(pos_w)
            .arg_u32(rows)
            .arg_u32(row_stride_bf16)
            .arg_u32(k_offset_bf16)
            .arg_u32(nq)
            .arg_u32(nkv)
            .arg_u32(head_dim)
            .arg_u32(rotary_dim)
            .arg_f32(theta)
            .launch(stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn expand_metadata(
        self,
        gpu: &dyn GpuBackend,
        source: DevicePtr,
        expanded: DevicePtr,
        row_lengths: DevicePtr,
        final_length: DevicePtr,
        block_count: usize,
        tile_start: usize,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        ensure!(block_count > 0, "device16 requires a nonempty block table");
        ensure!(
            rows == 16 || rows == 32,
            "device metadata rows must be 16 or 32"
        );
        let words = (rows as usize)
            .checked_mul(block_count)
            .ok_or_else(|| anyhow::anyhow!("device16 block-table overflow"))?;
        let bytes = words
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("device16 metadata byte overflow"))?;
        let final_value = tile_start
            .checked_add(rows as usize)
            .ok_or_else(|| anyhow::anyhow!("device16 sequence length overflow"))?;
        ensure!(
            final_value <= 2048,
            "device16 tile exceeds admitted sequence"
        );
        let ranges = [
            span(source, block_count * 4)?,
            span(expanded, bytes)?,
            span(row_lengths, rows as usize * 4)?,
            span(final_length, 4)?,
        ];
        for left in 0..ranges.len() {
            for right in left + 1..ranges.len() {
                ensure!(
                    disjoint(ranges[left], ranges[right]),
                    "device16 metadata spans overlap"
                );
            }
        }
        let kernel = if rows == 32 {
            self.expand_meta32
        } else {
            self.expand_meta16
        };
        KernelLaunch::new(gpu, kernel)
            .grid([div_ceil(u32::try_from(words)?, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(source)
            .arg_ptr(expanded)
            .arg_ptr(row_lengths)
            .arg_ptr(final_length)
            .arg_u32(u32::try_from(block_count)?)
            .arg_u32(u32::try_from(tile_start)?)
            .launch(stream)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mrope_or_scalar(
    device: Option<DeviceKernels>,
    gpu: &dyn GpuBackend,
    scalar_kernel: KernelHandle,
    qkv: DevicePtr,
    qkv_row_bytes: usize,
    pos_t: DevicePtr,
    pos_h: DevicePtr,
    pos_w: DevicePtr,
    row_stride: u32,
    k_offset: u32,
    nq: u32,
    nkv: u32,
    head_dim: u32,
    rotary_dim: u32,
    theta: f32,
    rows: u32,
    stream: u64,
) -> Result<()> {
    if let Some(kernels) = device {
        return kernels.mrope(
            gpu, qkv, pos_t, pos_h, pos_w, row_stride, k_offset, nq, nkv, head_dim, rotary_dim,
            theta, rows, stream,
        );
    }
    for row in 0..rows as usize {
        let q = qkv.offset(row * qkv_row_bytes);
        ops::rope_mrope_interleaved(
            gpu,
            scalar_kernel,
            q,
            q.offset(k_offset as usize * 2),
            pos_t.offset(row * 4),
            pos_h.offset(row * 4),
            pos_w.offset(row * 4),
            1,
            nq,
            nkv,
            head_dim,
            rotary_dim,
            theta,
            stream,
        )?;
    }
    Ok(())
}

#[path = "prefill_moe_attn16_metadata.rs"]
mod metadata;
pub(super) use metadata::expand_or_copy_metadata;
