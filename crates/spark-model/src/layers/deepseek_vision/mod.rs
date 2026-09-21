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
        let weights = Weights::load(store, &geometry, config.num_hidden_layers)?;
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
        let arena = gpu.alloc(bytes)?;
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
            scratch,
            forward_lock: parking_lot::Mutex::new(()),
        })
    }

    pub fn output_rows(&self, grid_h: usize, grid_w: usize) -> Result<usize> {
        self.geometry.output_rows(grid_h, grid_w)
    }

    /// START, PAD, IMAGE (placeholder PAD), NEWLINE, END learned BF16 rows.
    pub fn image_special_embeddings(&self) -> [DevicePtr; 5] {
        self.weights.special
    }

    pub fn scratch_bytes(&self) -> Result<usize> {
        self.geometry.scratch_bytes()
    }

    /// Explicit teardown for owners retaining the backend. The checkpoint's
    /// allocation owner retains all weights; only our scratch arena is freed.
    pub fn release(self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.synchronize(gpu.default_stream())?;
        gpu.free(self.arena)
    }
}
