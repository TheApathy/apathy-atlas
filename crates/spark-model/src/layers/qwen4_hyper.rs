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
    save_inject_k: KernelHandle,
    inject_saved_k: KernelHandle,
    stage_mixed_k: KernelHandle,
    pack_mixed_k: KernelHandle,
    w4a16_gemv_k: KernelHandle,
    w4a16_gemm_k: KernelHandle,
    w4a16_gemv_exact_m4_k: KernelHandle,
    w4a16_gemv_exact_m8_k: KernelHandle,
    w4a16_gemv_exact_m17_k: KernelHandle,
    w4a16_gemv_exact_m32_k: KernelHandle,
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
            save_inject_k: gpu.kernel("qwen4_hyper", "qwen4_hc_save_inject")?,
            inject_saved_k: gpu.kernel("qwen4_hyper", "qwen4_hc_inject_saved")?,
            stage_mixed_k: gpu.kernel("qwen4_hyper", "qwen4_hc_stage_mixed")?,
            pack_mixed_k: gpu.kernel("qwen4_hyper", "qwen4_hc_pack_mixed")?,
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemm_k: gpu.kernel("w4a16", "w4a16_gemm")?,
            w4a16_gemv_exact_m4_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch_logits_exact_m4")?,
            w4a16_gemv_exact_m8_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch_logits_exact_m8")?,
            w4a16_gemv_exact_m17_k: gpu
                .kernel("w4a16_gemv", "w4a16_gemv_batch_logits_exact_m17")?,
            w4a16_gemv_exact_m32_k: gpu
                .kernel("w4a16_gemv", "w4a16_gemv_batch_logits_exact_m32")?,
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

    /// Inject a contiguous token tile in one launch. The CUDA primitive is
    /// already token-indexed; decode historically launched only grid.x=1.
    pub fn inject_batched(
        &self,
        hyper: DevicePtr,
        core: DevicePtr,
        inject: DevicePtr,
        num_tokens: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.inject_k)
            .grid([
                num_tokens as u32,
                div_ceil(self.residual_width() as u32, 256),
                1,
            ])
            .block([256, 1, 1])
            .arg_ptr(hyper)
            .arg_ptr(core)
            .arg_ptr(inject)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .launch(stream)
    }

    /// Inject a full prompt tile using scales scattered into each residual
    /// row by [`prepare_prefill`].
    pub fn inject_saved_batched(
        &self,
        hyper: DevicePtr,
        core: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.inject_saved_k)
            .grid([
                num_tokens as u32,
                div_ceil(self.residual_width() as u32, 256),
                1,
            ])
            .block([256, 1, 1])
            .arg_ptr(hyper)
            .arg_ptr(core)
            .arg_ptr(residual)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .launch(stream)
    }

    /// Full-prompt hyperconnection preparation using NVFP4 GEMM for all
    /// three projections. This is the throughput-owned prefill path; the
    /// small exact-GEMV family remains the decode/speculative path.
    pub fn prepare_prefill(
        &self,
        hyper: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        buffers: &BufferArena,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<DevicePtr> {
        anyhow::ensure!(num_tokens > 0, "empty Qwen4 hyperconnection prefill");
        let r = self.residual_width();
        let down_bytes = num_tokens * self.rank * 2;
        let inject_bytes = num_tokens * self.hc_count * 2;
        anyhow::ensure!(
            down_bytes + inject_bytes <= buffers.sizes().gate_logits,
            "Qwen4 prefill hyper scratch exceeds gate-logit arena"
        );
        anyhow::ensure!(
            num_tokens * r * 2 <= buffers.sizes().qkv_output,
            "Qwen4 prefill hyper logits exceed QKV arena"
        );
        anyhow::ensure!(
            num_tokens * self.hidden_size * 2 <= buffers.sizes().norm_output,
            "Qwen4 prefill mixed rows exceed norm arena"
        );
        KernelLaunch::new(gpu, self.group_norm_k)
            .grid([num_tokens as u32, self.hc_count as u32, 1])
            .block([self.hidden_size.min(1024) as u32, 1, 1])
            .arg_ptr(hyper)
            .arg_ptr(self.norm.weight)
            .arg_ptr(residual)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .arg_f32(eps)
            .launch(stream)?;

        // The BA arena is sized for the model's 64-wide recurrent gates and
        // is too small for M*rank at long prompt lengths. Gate logits are idle
        // until the MoE and scale linearly with max prefill tokens.
        let down_out = buffers.gate_logits();
        ops::w4a16_gemm(
            gpu,
            self.w4a16_gemm_k,
            residual,
            &self.down,
            down_out,
            num_tokens as u32,
            self.rank as u32,
            r as u32,
            stream,
        )?;
        KernelLaunch::new(gpu, self.silu_div_k)
            .grid([div_ceil((num_tokens * self.rank) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(down_out)
            .arg_u32((num_tokens * self.rank) as u32)
            .arg_f32(self.hc_count as f32)
            .launch(stream)?;

        let mix_logits = buffers.qkv_output();
        let inject_logits = down_out.offset(down_bytes);
        ops::w4a16_gemm(
            gpu,
            self.w4a16_gemm_k,
            down_out,
            &self.up,
            mix_logits,
            num_tokens as u32,
            r as u32,
            self.rank as u32,
            stream,
        )?;
        if let Some(ref weight) = self.inject {
            ops::w4a16_gemm(
                gpu,
                self.w4a16_gemm_k,
                residual,
                weight,
                inject_logits,
                num_tokens as u32,
                self.hc_count as u32,
                r as u32,
                stream,
            )?;
        } else {
            gpu.memset_async(inject_logits, 0, inject_bytes, stream)?;
        }
        let mixed = buffers.norm_output();
        KernelLaunch::new(gpu, self.mix_k)
            .grid([num_tokens as u32, div_ceil(self.hidden_size as u32, 256), 1])
            .block([256, 1, 1])
            .arg_ptr(residual)
            .arg_ptr(mix_logits)
            .arg_ptr(inject_logits)
            .arg_ptr(mixed)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.save_inject_k)
            .grid([div_ceil((num_tokens * self.hc_count) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(residual)
            .arg_ptr(inject_logits)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .arg_u32(num_tokens as u32)
            .launch(stream)?;
        Ok(mixed)
    }

    /// Arbitrary-length, byte-exact prefill preparation. Projection weights
    /// are streamed once per 32 prompt rows, and mixed rows are staged in the
    /// residual workspace before one contiguous pack for the core kernels.
    pub fn prepare_prefill_exact(
        &self,
        hyper: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        buffers: &BufferArena,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<DevicePtr> {
        anyhow::ensure!(num_tokens > 0, "empty Qwen4 exact prefill");
        let r = self.residual_width();
        let mut start = 0;
        while start < num_tokens {
            let tile = (num_tokens - start).min(32);
            let hyper_tile = hyper.offset(start * r * 2);
            let residual_tile = residual.offset(start * r * 2);
            let mixed = if tile == 1 {
                self.prepare_decode(hyper_tile, residual_tile, buffers, gpu, eps, stream)?
                    .0
            } else {
                self.prepare_batched(hyper_tile, residual_tile, tile, buffers, gpu, eps, stream)?
            };
            KernelLaunch::new(gpu, self.stage_mixed_k)
                .grid([div_ceil((tile * self.hidden_size) as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(mixed)
                .arg_ptr(residual)
                .arg_u32(start as u32)
                .arg_u32(tile as u32)
                .arg_u32(self.hidden_size as u32)
                .arg_u32(self.hc_count as u32)
                .launch(stream)?;
            start += tile;
        }
        let mixed = buffers.norm_output();
        KernelLaunch::new(gpu, self.pack_mixed_k)
            .grid([div_ceil((num_tokens * self.hidden_size) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(residual)
            .arg_ptr(mixed)
            .arg_u32(num_tokens as u32)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .launch(stream)?;
        Ok(mixed)
    }

    /// Mix up to 32 contiguous hyperconnection rows while streaming each
    /// projection weight once. The exact M4 kernel preserves K1 lane
    /// ownership, operation order, reduction tree, and BF16 rounding, unlike
    /// the numerically different K8 batch2/batch3 or tensor-core kernels.
    pub fn prepare_batched(
        &self,
        hyper: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        buffers: &BufferArena,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<DevicePtr> {
        anyhow::ensure!(
            (2..=32).contains(&num_tokens),
            "Qwen4 hyperconnection batching supports K=2..32, got {num_tokens}"
        );
        let r = self.residual_width();
        let down_bytes = num_tokens * self.rank * 2;
        let inject_bytes = num_tokens * self.hc_count * 2;
        anyhow::ensure!(
            down_bytes + inject_bytes <= buffers.sizes().ssm_ba,
            "Qwen4 batched hyper scratch exceeds SSM BA arena"
        );
        anyhow::ensure!(
            num_tokens * r * 2 <= buffers.sizes().qkv_output,
            "Qwen4 batched hyper logits exceed QKV arena"
        );
        anyhow::ensure!(
            num_tokens * self.hidden_size * 2 <= buffers.sizes().norm_output,
            "Qwen4 batched mixed rows exceed norm arena"
        );

        KernelLaunch::new(gpu, self.group_norm_k)
            .grid([num_tokens as u32, self.hc_count as u32, 1])
            .block([self.hidden_size.min(1024) as u32, 1, 1])
            .arg_ptr(hyper)
            .arg_ptr(self.norm.weight)
            .arg_ptr(residual)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .arg_f32(eps)
            .launch(stream)?;

        let down_out = buffers.ssm_ba();
        let exact_kernels = ops::W4a16ExactLmHeadKernels::new(
            self.w4a16_gemv_exact_m4_k,
            self.w4a16_gemv_exact_m8_k,
            self.w4a16_gemv_exact_m17_k,
            self.w4a16_gemv_exact_m32_k,
        );
        ops::w4a16_gemv_batch_logits_exact_with(
            gpu,
            exact_kernels,
            residual,
            &self.down,
            down_out,
            num_tokens as u32,
            self.rank as u32,
            r as u32,
            stream,
            false,
        )?;
        KernelLaunch::new(gpu, self.silu_div_k)
            .grid([div_ceil((num_tokens * self.rank) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(down_out)
            .arg_u32((num_tokens * self.rank) as u32)
            .arg_f32(self.hc_count as f32)
            .launch(stream)?;

        let mix_logits = buffers.qkv_output();
        let inject_logits = down_out.offset(down_bytes);
        let project = |input: DevicePtr,
                       weight: &QuantizedWeight,
                       output: DevicePtr,
                       n: u32,
                       k: u32|
         -> Result<()> {
            ops::w4a16_gemv_batch_logits_exact_with(
                gpu,
                exact_kernels,
                input,
                weight,
                output,
                num_tokens as u32,
                n,
                k,
                stream,
                false,
            )
        };
        project(down_out, &self.up, mix_logits, r as u32, self.rank as u32)?;
        if let Some(ref weight) = self.inject {
            project(
                residual,
                weight,
                inject_logits,
                self.hc_count as u32,
                r as u32,
            )?;
        } else {
            gpu.memset_async(inject_logits, 0, inject_bytes, stream)?;
        }

        let mixed = buffers.norm_output();
        KernelLaunch::new(gpu, self.mix_k)
            .grid([num_tokens as u32, div_ceil(self.hidden_size as u32, 256), 1])
            .block([256, 1, 1])
            .arg_ptr(residual)
            .arg_ptr(mix_logits)
            .arg_ptr(inject_logits)
            .arg_ptr(mixed)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .launch(stream)?;
        if self.inject.is_some() {
            KernelLaunch::new(gpu, self.save_inject_k)
                .grid([div_ceil((num_tokens * self.hc_count) as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(residual)
                .arg_ptr(inject_logits)
                .arg_u32(self.hidden_size as u32)
                .arg_u32(self.hc_count as u32)
                .arg_u32(num_tokens as u32)
                .launch(stream)?;
        }
        Ok(mixed)
    }

    /// Location where [`prepare_decode`] preserves this row's injection
    /// scales. The tail lives in row-specific residual storage, so callers
    /// may prepare several rows before applying their batched core outputs.
    pub fn saved_inject(&self, residual: DevicePtr) -> DevicePtr {
        residual.offset((self.residual_width() - self.hc_count) * 2)
    }
}
