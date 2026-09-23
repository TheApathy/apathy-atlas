// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    DeepSeekVisionEncoder,
    detail_selection::DetailSelection,
    geometry::bf16_bytes,
    observer::{VisionObserver, VisionStageDtype, observe},
    weights::Linear,
};
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

impl DeepSeekVisionEncoder {
    /// Encodes one row-major RGB patch grid. The BF16 result is row-major
    /// after the 3x3 aligner; the caller applies the model's N-layout order.
    /// This method synchronizes before returning the borrowed output buffer.
    pub fn forward(
        &self,
        gpu: &dyn GpuBackend,
        patches: &[f32],
        grid_h: usize,
        grid_w: usize,
    ) -> Result<DevicePtr> {
        self.forward_inner(gpu, patches, grid_h, grid_w, None, None)
    }

    /// Diagnostic-only stage taps. Adds synchronization and callback overhead;
    /// never use this entry point to measure encoder/prefill performance.
    pub fn forward_observed(
        &self,
        gpu: &dyn GpuBackend,
        patches: &[f32],
        grid_h: usize,
        grid_w: usize,
        observer: &mut VisionObserver<'_>,
    ) -> Result<DevicePtr> {
        self.forward_observed_block(gpu, patches, grid_h, grid_w, 0, observer)
    }

    /// Observes every block exit and detailed intermediates for exactly one
    /// loaded block. Invalid selection fails before input upload or callbacks.
    /// This diagnostic entry adds fences; its timings are not performance data.
    pub fn forward_observed_block(
        &self,
        gpu: &dyn GpuBackend,
        patches: &[f32],
        grid_h: usize,
        grid_w: usize,
        block: usize,
        observer: &mut VisionObserver<'_>,
    ) -> Result<DevicePtr> {
        let detail =
            DetailSelection::new(block, self.weights.blocks.len()).map_err(anyhow::Error::msg)?;
        self.forward_inner(gpu, patches, grid_h, grid_w, Some(detail), Some(observer))
    }

    fn forward_inner(
        &self,
        gpu: &dyn GpuBackend,
        patches: &[f32],
        grid_h: usize,
        grid_w: usize,
        detail: Option<DetailSelection>,
        observer: Option<&mut VisionObserver<'_>>,
    ) -> Result<DevicePtr> {
        let _guard = self.forward_lock.lock();
        let g = &self.geometry;
        let (p, rows) = g.grid(grid_h, grid_w, patches.len())?;
        let pixels = bf16_bytes(patches)?;
        // Blocking upload keeps the host pixel buffer alive through its transfer.
        gpu.copy_h2d(&pixels, self.scratch.pixels)?;
        let result = self.run(gpu, p, rows, (grid_h, grid_w), detail, observer);
        // Also fence on a launch error: no in-flight use may escape teardown.
        let fence = gpu.synchronize(gpu.default_stream());
        result?;
        fence?;
        Ok(self.scratch.output)
    }

    fn run(
        &self,
        gpu: &dyn GpuBackend,
        p: usize,
        rows: usize,
        grid: (usize, usize),
        detail: Option<DetailSelection>,
        mut observer: Option<&mut VisionObserver<'_>>,
    ) -> Result<()> {
        let (grid_h, grid_w) = grid;
        let g = &self.geometry;
        let s = &self.scratch;
        let stream = gpu.default_stream();
        // Geometry bounds p <= 3456; admission pins theta10000/head64. Generate
        // CUDA FP32 angles once into existing scratch on the consumer stream.
        KernelLaunch::new(gpu, self.kernels.angles)
            .grid([div_ceil((p * 32) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.angles)
            .arg_u32(grid_h as u32)
            .arg_u32(grid_w as u32)
            .launch(stream)?;
        self.linear(gpu, s.pixels, self.weights.patch, s.hidden, p, g.hidden)?;
        observe(
            gpu,
            &mut observer,
            "patch",
            s.hidden,
            [p, g.hidden],
            VisionStageDtype::Bf16,
        )?;
        for (layer, block) in self.weights.blocks.iter().enumerate() {
            let selected =
                detail.filter(|selection| observer.is_some() && selection.is_selected(layer));
            self.norm(gpu, s.hidden, block.norm1, s.norm, p)?;
            if let Some(selection) = selected {
                observe(
                    gpu,
                    &mut observer,
                    &selection.stage_name("norm1"),
                    s.norm,
                    [p, g.hidden],
                    VisionStageDtype::Bf16,
                )?;
            }
            self.linear(gpu, s.norm, block.qkv, s.qkv, p, 3 * g.hidden)?;
            if let Some(selection) = selected {
                observe(
                    gpu,
                    &mut observer,
                    &selection.stage_name("qkv"),
                    s.qkv,
                    [p, 3 * g.hidden],
                    VisionStageDtype::Bf16,
                )?;
            }
            KernelLaunch::new(gpu, self.kernels.rope)
                .grid([div_ceil((p * g.hidden / 2) as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(s.qkv)
                .arg_ptr(s.query)
                .arg_ptr(s.key)
                .arg_ptr(s.value)
                .arg_ptr(s.angles)
                .arg_u32(p as u32)
                .arg_u32(g.heads as u32)
                .arg_u32(g.head_dim as u32)
                .launch(stream)?;
            if let Some(selection) = selected {
                for (name, pointer, shape) in [
                    ("query", s.query, [g.heads * p, g.head_dim]),
                    ("key", s.key, [g.heads * p, g.head_dim]),
                    ("value", s.value, [g.heads * g.head_dim, p]),
                ] {
                    observe(
                        gpu,
                        &mut observer,
                        &selection.stage_name(name),
                        pointer,
                        shape,
                        VisionStageDtype::Bf16,
                    )?;
                }
            }
            for head in 0..g.heads {
                let offset = head * p * g.head_dim * 2;
                KernelLaunch::new(gpu, self.kernels.scores)
                    .grid([div_ceil(p as u32, 32), div_ceil(p as u32, 32), 1])
                    .block([128, 1, 1])
                    .arg_ptr(s.query.offset(offset))
                    .arg_ptr(s.key.offset(offset))
                    .arg_ptr(s.scores)
                    .arg_u32(p as u32)
                    .arg_u32(p as u32)
                    .arg_u32(g.head_dim as u32)
                    .arg_u32(p as u32)
                    .launch(stream)?;
                if let Some(selection) = selected.filter(|_| head == 0) {
                    observe(
                        gpu,
                        &mut observer,
                        &selection.stage_name("head-00-scores"),
                        s.scores,
                        [p, p],
                        VisionStageDtype::F32,
                    )?;
                }
                KernelLaunch::new(gpu, self.kernels.softmax)
                    .grid([p as u32, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(s.scores)
                    .arg_ptr(s.probs)
                    .arg_u32(p as u32)
                    .arg_f32((g.head_dim as f32).sqrt().recip())
                    .launch(stream)?;
                if let Some(selection) = selected.filter(|_| head == 0) {
                    observe(
                        gpu,
                        &mut observer,
                        &selection.stage_name("head-00-probs"),
                        s.probs,
                        [p, p],
                        VisionStageDtype::F32,
                    )?;
                }
                // Official math SDPA retains FP32 P and FP32 P@V accumulation.
                // V remains [head, dim, patches], widened in the tiled kernel.
                KernelLaunch::new(gpu, self.kernels.attention_value)
                    .grid([div_ceil(p as u32, 16), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(s.probs)
                    .arg_ptr(s.value.offset(offset))
                    .arg_ptr(s.attention.offset(head * g.head_dim * 2))
                    .arg_u32(p as u32)
                    .arg_u32(g.hidden as u32)
                    .launch(stream)?;
            }
            if let Some(selection) = selected {
                observe(
                    gpu,
                    &mut observer,
                    &selection.stage_name("attention"),
                    s.attention,
                    [p, g.hidden],
                    VisionStageDtype::Bf16,
                )?;
            }
            self.linear(gpu, s.attention, block.proj, s.norm, p, g.hidden)?;
            if let Some(selection) = selected {
                observe(
                    gpu,
                    &mut observer,
                    &selection.stage_name("projection"),
                    s.norm,
                    [p, g.hidden],
                    VisionStageDtype::Bf16,
                )?;
            }
            self.add(gpu, s.hidden, s.norm, p * g.hidden)?;
            if let Some(selection) = selected {
                observe(
                    gpu,
                    &mut observer,
                    &selection.stage_name("residual1"),
                    s.hidden,
                    [p, g.hidden],
                    VisionStageDtype::Bf16,
                )?;
            }
            self.norm(gpu, s.hidden, block.norm2, s.norm, p)?;
            if let Some(selection) = selected {
                observe(
                    gpu,
                    &mut observer,
                    &selection.stage_name("norm2"),
                    s.norm,
                    [p, g.hidden],
                    VisionStageDtype::Bf16,
                )?;
            }
            self.linear(gpu, s.norm, block.fc1, s.wide, p, 2 * g.intermediate)?;
            if let Some(selection) = selected {
                observe(
                    gpu,
                    &mut observer,
                    &selection.stage_name("fc1"),
                    s.wide,
                    [p, 2 * g.intermediate],
                    VisionStageDtype::Bf16,
                )?;
            }
            // QKV is dead now and provides >= intermediate-width storage.
            KernelLaunch::new(gpu, self.kernels.swiglu)
                .grid([div_ceil((p * g.intermediate) as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(s.wide)
                .arg_ptr(s.qkv)
                .arg_u32(p as u32)
                .arg_u32(g.intermediate as u32)
                .launch(stream)?;
            if let Some(selection) = selected {
                observe(
                    gpu,
                    &mut observer,
                    &selection.stage_name("swiglu"),
                    s.qkv,
                    [p, g.intermediate],
                    VisionStageDtype::Bf16,
                )?;
            }
            self.linear(gpu, s.qkv, block.fc2, s.norm, p, g.hidden)?;
            if let Some(selection) = selected {
                observe(
                    gpu,
                    &mut observer,
                    &selection.stage_name("fc2"),
                    s.norm,
                    [p, g.hidden],
                    VisionStageDtype::Bf16,
                )?;
            }
            self.add(gpu, s.hidden, s.norm, p * g.hidden)?;
            if observer.is_some() {
                observe(
                    gpu,
                    &mut observer,
                    &format!("block-{layer:02}-exit"),
                    s.hidden,
                    [p, g.hidden],
                    VisionStageDtype::Bf16,
                )?;
            }
        }
        self.norm(gpu, s.hidden, self.weights.norm, s.norm, p)?;
        observe(
            gpu,
            &mut observer,
            "final-norm",
            s.norm,
            [p, g.hidden],
            VisionStageDtype::Bf16,
        )?;
        let merge_dim = g.hidden * g.ratio * g.ratio;
        KernelLaunch::new(gpu, self.kernels.unfold)
            .grid([div_ceil((rows * merge_dim) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.norm)
            .arg_ptr(s.wide)
            .arg_u32(grid_h as u32)
            .arg_u32(grid_w as u32)
            .arg_u32(g.hidden as u32)
            .arg_u32(g.ratio as u32)
            .launch(stream)?;
        observe(
            gpu,
            &mut observer,
            "aligner-unfold",
            s.wide,
            [rows, merge_dim],
            VisionStageDtype::Bf16,
        )?;
        self.linear(
            gpu,
            s.wide,
            self.weights.align1,
            s.norm,
            rows,
            g.text_hidden,
        )?;
        observe(
            gpu,
            &mut observer,
            "aligner-w1",
            s.norm,
            [rows, g.text_hidden],
            VisionStageDtype::Bf16,
        )?;
        KernelLaunch::new(gpu, self.kernels.gelu)
            .grid([div_ceil((rows * g.text_hidden) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.norm)
            .arg_u32((rows * g.text_hidden) as u32)
            .launch(stream)?;
        observe(
            gpu,
            &mut observer,
            "aligner-gelu",
            s.norm,
            [rows, g.text_hidden],
            VisionStageDtype::Bf16,
        )?;
        self.linear(
            gpu,
            s.norm,
            self.weights.align2,
            s.output,
            rows,
            g.text_hidden,
        )?;
        observe(
            gpu,
            &mut observer,
            "aligner-output",
            s.output,
            [rows, g.text_hidden],
            VisionStageDtype::Bf16,
        )
    }

    fn linear(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: Linear,
        output: DevicePtr,
        m: usize,
        ldc: usize,
    ) -> Result<()> {
        // The native fused-bias WMMA epilogue rounds only after adding bias.
        // K tails, including the 588-element RGB patch projection, are masked.
        KernelLaunch::new(gpu, self.kernels.linear)
            .grid([div_ceil(weight.n as u32, 32), div_ceil(m as u32, 32), 1])
            .block([128, 1, 1])
            .arg_ptr(input)
            .arg_ptr(weight.weight)
            .arg_ptr(weight.bias)
            .arg_ptr(output)
            .arg_u32(m as u32)
            .arg_u32(weight.n as u32)
            .arg_u32(weight.k as u32)
            .arg_u32(ldc as u32)
            .launch(gpu.default_stream())
    }

    fn norm(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: DevicePtr,
        output: DevicePtr,
        rows: usize,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.kernels.norm)
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input)
            .arg_ptr(weight)
            .arg_ptr(output)
            .arg_u32(self.geometry.hidden as u32)
            .arg_f32(atlas_core::config::DeepSeekVisionConfig::RMS_NORM_EPS)
            .launch(gpu.default_stream())
    }

    fn add(&self, gpu: &dyn GpuBackend, dst: DevicePtr, src: DevicePtr, n: usize) -> Result<()> {
        KernelLaunch::new(gpu, self.kernels.add)
            .grid([div_ceil(n as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(dst)
            .arg_ptr(src)
            .arg_u32(n as u32)
            .launch(gpu.default_stream())
    }
}
