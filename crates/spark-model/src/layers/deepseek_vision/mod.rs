// SPDX-License-Identifier: AGPL-3.0-only

//! Native DeepSeek-V4-Flash-Vision encoder, separate from Qwen's ViT.
//!
//! Formula reference: deepseek-ai/DeepSeek-V4-Flash-Vision-Exp (MIT),
//! revision 6821d6ad3681a4b137b066b76094fa82ebd0a380, inference/vision.py.
//! GPU numerical parity is a separate qualification gate.

#[cfg(test)]
mod angle_tests;
mod detail_selection;
mod forward;
mod geometry;
mod observer;
#[cfg(test)]
mod pv_tests;
#[cfg(test)]
mod tests;
mod weights;

use anyhow::Result;
use atlas_core::config::DeepSeekVisionConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::weights::WeightStore;

use geometry::Geometry;
pub use observer::{VisionObserver, VisionStageDtype};
use weights::Weights;

/// The returned forward buffer belongs to this encoder. Consume or copy it
/// before another image is encoded. Calls use the backend's default stream.
pub struct DeepSeekVisionEncoder {
    geometry: Geometry,
    weights: Weights,
    kernels: Kernels,
    arena: DevicePtr,
    /// Owns `arena` when built with an owned backend (V4.1: freed on drop, for
    /// TUI model swap); unowned for V4-Flash-Vision, which frees via `release`.
    allocs: crate::weight_loader::deepseek_v41::device_allocs::DeviceAllocs,
    scratch: Scratch,
    forward_lock: parking_lot::Mutex<()>,
}

struct Kernels {
    angles: KernelHandle,
    linear: KernelHandle,
    scores: KernelHandle,
    attention_value: KernelHandle,
    norm: KernelHandle,
    add: KernelHandle,
    swiglu: KernelHandle,
    gelu: KernelHandle,
    rope: KernelHandle,
    softmax: KernelHandle,
    unfold: KernelHandle,
}

struct Scratch {
    pixels: DevicePtr,
    hidden: DevicePtr,
    norm: DevicePtr,
    qkv: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    attention: DevicePtr,
    wide: DevicePtr,
    scores: DevicePtr,
    probs: DevicePtr,
    angles: DevicePtr,
    output: DevicePtr,
}

impl DeepSeekVisionEncoder {
    /// All 267 native BF16 visual weights are checked before allocating scratch.
    pub fn load(
        store: &WeightStore,
        config: &DeepSeekVisionConfig,
        text_hidden_size: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        config.validate()?;
        let geometry = Geometry::new(
            config.hidden_size,
            config.intermediate_size,
            config.num_attention_heads,
            config.patch_size,
            config.downsample_ratio,
            config.max_tokens,
            text_hidden_size,
        )?;
        anyhow::ensure!(
            config.num_hidden_layers == 32,
            "DeepSeek vision requires 32 ViT blocks"
        );
        anyhow::ensure!(
            config.rope_theta.to_bits() == 10_000.0f64.to_bits() && geometry.head_dim == 64,
            "DeepSeek vision device angles require exact theta 10000 and head dimension 64"
        );
        Self::load_with(store, geometry, config.num_hidden_layers, true, gpu, None)
    }

    /// DeepSeek-V4.1's tower: the same 32-block 1024-wide ViT and 3x3 aligner
    /// as V4-Flash-Vision (the same tensor names and shapes), feeding a
    /// 5120-wide text model, with up to 1024 image tokens and no `image_pad`
    /// tensor. The arguments come from the checkpoint's `vision_config`.
    #[allow(clippy::too_many_arguments)]
    pub fn load_v41(
        store: &WeightStore,
        hidden: usize,
        intermediate: usize,
        heads: usize,
        layers: usize,
        patch: usize,
        downsample: usize,
        rope_theta: f64,
        max_image_tokens: usize,
        text_hidden_size: usize,
        gpu: &dyn GpuBackend,
        owner: Option<crate::weight_loader::deepseek_v41::device_allocs::SharedGpu>,
    ) -> Result<Self> {
        anyhow::ensure!(
            text_hidden_size == 5120,
            "DeepSeek-V4.1 vision expects a 5120-wide text model"
        );
        anyhow::ensure!(
            layers == 32 && rope_theta.to_bits() == 10_000.0f64.to_bits(),
            "DeepSeek-V4.1 vision requires 32 blocks and rope_theta 10000"
        );
        let geometry = Geometry::new(
            hidden,
            intermediate,
            heads,
            patch,
            downsample,
            max_image_tokens,
            text_hidden_size,
        )?;
        anyhow::ensure!(
            geometry.head_dim == 64,
            "DeepSeek vision device angles require head dimension 64"
        );
        Self::load_with(store, geometry, layers, false, gpu, owner)
    }

    fn load_with(
        store: &WeightStore,
        geometry: Geometry,
        depth: usize,
        has_pad: bool,
        gpu: &dyn GpuBackend,
        owner: Option<crate::weight_loader::deepseek_v41::device_allocs::SharedGpu>,
    ) -> Result<Self> {
        let weights = Weights::load(store, &geometry, depth, has_pad)?;
        let kernels = Kernels {
            angles: gpu.kernel("deepseek_vision_angles", "deepseek_vision_angles")?,
            linear: gpu.kernel("deepseek_vision_gemm", "deepseek_vision_linear")?,
            scores: gpu.kernel("deepseek_vision_gemm", "deepseek_vision_scores")?,
            attention_value: gpu.kernel("deepseek_vision_pv", "deepseek_vision_attention_value")?,
            norm: gpu.kernel("deepseek_vision", "deepseek_vision_rms_norm")?,
            add: gpu.kernel("deepseek_vision", "deepseek_vision_add")?,
            swiglu: gpu.kernel("deepseek_vision", "deepseek_vision_swiglu")?,
            gelu: gpu.kernel("deepseek_vision", "deepseek_vision_gelu")?,
            rope: gpu.kernel("deepseek_vision", "deepseek_vision_rope")?,
            softmax: gpu.kernel("deepseek_vision", "deepseek_vision_softmax")?,
            unfold: gpu.kernel("deepseek_vision", "deepseek_vision_unfold")?,
        };
        let (offsets, bytes) = geometry.scratch_layout()?;
        use crate::weight_loader::deepseek_v41::device_allocs::DeviceAllocs;
        let mut allocs = owner.map_or_else(DeviceAllocs::unowned, DeviceAllocs::owned);
        let arena = allocs.alloc(gpu, bytes)?;
        let p: Vec<DevicePtr> = offsets
            .into_iter()
            .map(|offset| arena.offset(offset))
            .collect();
        let scratch = Scratch {
            pixels: p[0],
            hidden: p[1],
            norm: p[2],
            qkv: p[3],
            query: p[4],
            key: p[5],
            value: p[6],
            attention: p[7],
            wide: p[8],
            scores: p[9],
            probs: p[10],
            angles: p[11],
            output: p[12],
        };
        Ok(Self {
            geometry,
            weights,
            kernels,
            arena,
            allocs,
            scratch,
            forward_lock: parking_lot::Mutex::new(()),
        })
    }

    pub fn output_rows(&self, grid_h: usize, grid_w: usize) -> Result<usize> {
        self.geometry.output_rows(grid_h, grid_w)
    }

    /// START, PAD, IMAGE (placeholder PAD), NEWLINE, END learned BF16 rows.
    /// For V4.1 (`load_v41`) the two PAD entries are NULL: that checkpoint
    /// has no learned pad row.
    pub fn image_special_embeddings(&self) -> [DevicePtr; 5] {
        self.weights.special
    }

    pub fn scratch_bytes(&self) -> Result<usize> {
        self.geometry.scratch_bytes()
    }

    /// Explicit teardown for owners retaining the backend. The checkpoint's
    /// allocation owner retains all weights; only our scratch arena is freed.
    /// With an owned backend (`load_v41(.., Some(owner))`) dropping the encoder
    /// frees the arena, so this only synchronizes and drops.
    pub fn release(self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.synchronize(gpu.default_stream())?;
        if self.allocs.is_owned() {
            return Ok(()); // freed by `allocs` on drop
        }
        gpu.free(self.arena)
    }
}
