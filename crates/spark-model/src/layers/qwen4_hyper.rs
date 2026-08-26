// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen4-Exp gated residual (four-stream hyperconnection) dispatch.

use anyhow::{Result, bail};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::ops;
use crate::weight_map::{DenseWeight, QuantizedWeight};

pub struct Qwen4HyperConnection {
    pub norm: DenseWeight,
    pub down: QuantizedWeight,
    pub up: QuantizedWeight,
    pub inject: Option<QuantizedWeight>,
    group_norm_k: KernelHandle,
    silu_div_k: KernelHandle,
    mix_k: KernelHandle,
    inject_k: KernelHandle,
    w4a16_gemv_k: KernelHandle,
    hidden_size: usize,
    hc_count: usize,
    rank: usize,
}

impl Qwen4HyperConnection {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        norm: DenseWeight,
        down: QuantizedWeight,
        up: QuantizedWeight,
        inject: Option<QuantizedWeight>,
        hidden_size: usize,
        hc_count: usize,
        rank: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        if hidden_size == 0 || hc_count <= 1 || rank == 0 {
            bail!(
                "invalid Qwen4 hyperconnection geometry h={hidden_size} hc={hc_count} rank={rank}"
            );
        }
        Ok(Self {
            norm,
            down,
            up,
            inject,
            group_norm_k: gpu.kernel("qwen4_hyper", "qwen4_hc_group_norm")?,
            silu_div_k: gpu.kernel("qwen4_hyper", "qwen4_hc_silu_div")?,
            mix_k: gpu.kernel("qwen4_hyper", "qwen4_hc_mix")?,
            inject_k: gpu.kernel("qwen4_hyper", "qwen4_hc_inject")?,
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            hidden_size,
            hc_count,
            rank,
        })
    }

    pub fn residual_width(&self) -> usize {
        self.hidden_size * self.hc_count
    }

    /// Mix one hyperconnection row into a hidden-size core input. The
    /// normalized four-stream row is staged in `residual`; projection
    /// temporaries use arena buffers that are idle between sublayers.
    pub fn prepare_decode(
        &self,
        hyper: DevicePtr,
        residual: DevicePtr,
        buffers: &BufferArena,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<(DevicePtr, Option<DevicePtr>)> {
        let r = self.residual_width();
        KernelLaunch::new(gpu, self.group_norm_k)
            .grid([1, self.hc_count as u32, 1])
            .block([self.hidden_size.min(1024) as u32, 1, 1])
            .arg_ptr(hyper)
            .arg_ptr(self.norm.weight)
            .arg_ptr(residual)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .arg_f32(eps)
            .launch(stream)?;

        // `scratch` contains live attention metadata (positions, slot map,
        // block table, and sequence lengths) for the whole layer traversal.
        // Reusing its first bytes here silently corrupted paged-attention
        // metadata before the following QSA call. The SSM BA arena is idle in
        // both full-attention and hyperconnection mixing and is amply sized
        // for rank BF16 values plus hc_count injection logits.
        let down_out = buffers.ssm_ba();
        ops::w4a16_gemv(
            gpu,
            self.w4a16_gemv_k,
            residual,
            &self.down,
            down_out,
            self.rank as u32,
            r as u32,
            stream,
        )?;
        KernelLaunch::new(gpu, self.silu_div_k)
            .grid([div_ceil(self.rank as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(down_out)
            .arg_u32(self.rank as u32)
            .arg_f32(self.hc_count as f32)
            .launch(stream)?;

        let mix_logits = buffers.qkv_output();
        ops::w4a16_gemv(
            gpu,
            self.w4a16_gemv_k,
            down_out,
            &self.up,
            mix_logits,
            r as u32,
            self.rank as u32,
            stream,
        )?;

        // The fixed 2 KiB offset is aligned and exceeds rank=320 BF16.
        let inject_logits = buffers.ssm_ba().offset(2048);
        if let Some(ref weight) = self.inject {
            ops::w4a16_gemv(
                gpu,
                self.w4a16_gemv_k,
                residual,
                weight,
                inject_logits,
                self.hc_count as u32,
                r as u32,
                stream,
            )?;
        } else {
            gpu.memset_async(inject_logits, 0, self.hc_count * 2, stream)?;
        }

        let mixed = buffers.norm_output();
        KernelLaunch::new(gpu, self.mix_k)
            .grid([1, div_ceil(self.hidden_size as u32, 256), 1])
            .block([256, 1, 1])
            .arg_ptr(residual)
            .arg_ptr(mix_logits)
            .arg_ptr(inject_logits)
            .arg_ptr(mixed)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .launch(stream)?;
        let saved_inject = residual.offset((r - self.hc_count) * 2);
        if self.inject.is_some() {
            gpu.copy_d2d_async(inject_logits, saved_inject, self.hc_count * 2, stream)?;
        }
        Ok((mixed, self.inject.as_ref().map(|_| saved_inject)))
    }

    pub fn inject_decode(
        &self,
        hyper: DevicePtr,
        core: DevicePtr,
        inject: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.inject_k)
            .grid([1, div_ceil(self.residual_width() as u32, 256), 1])
            .block([256, 1, 1])
            .arg_ptr(hyper)
            .arg_ptr(core)
            .arg_ptr(inject)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .launch(stream)
    }
}
