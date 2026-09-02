// SPDX-License-Identifier: AGPL-3.0-only

//! One-time expansion of the mHC mixing functions into the arena.
//!
//! `Glm53HyperKernels::pre` consumes the mixing function as **F32**
//! `[MIX=24][HC=4][HIDDEN=4096]`, but the checkpoint stores `hc_attn_fn.weight`
//! and `hc_ffn_fn.weight` as Q8_0 `[16384, 24]`. GGUF's fastest-varying axis is
//! `ne[0]`, so the on-disk layout is already 24 contiguous runs of 16,384
//! values — the same order the kernel wants. The expansion is therefore a pure
//! dequantization with **no transpose**; if that ever stops holding, the row/
//! column assertion below fails rather than silently feeding a permuted matrix
//! into every layer.
//!
//! This runs once per load, not per token: the arena reserves
//! `GLM53_MHC_EXPANDED_F32_BYTES` = 141,557,760 bytes, which is exactly
//! 45 layers x 2 branches x 1,572,864 bytes. That the reserved size matches the
//! expansion exactly is asserted here, so an arena change cannot quietly leave
//! the last layers writing outside their region.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops::{GgmlIqBuffer, GgmlQ8F32Kernel, GgmlQ8F32Plan};
use crate::weight_loader::Glm53HyperWeights;

use super::arena::GLM53_MHC_EXPANDED_F32_BYTES;

/// mHC mixing width.
const MIX: u32 = 24;
/// Hidden-channel streams times hidden size: the inner extent of one function.
const FUNCTION_INNER: u32 = 4 * 4096;
/// F32 bytes for one layer-branch mixing function.
pub const GLM53_MHC_FUNCTION_F32_BYTES: u64 = (MIX as u64) * (FUNCTION_INNER as u64) * 4;
/// Layers carrying mHC connections. Layer 45 (NextN) has none.
const LAYERS: usize = 45;
/// Attention and FFN sites, in that order.
const BRANCHES: usize = 2;

/// Which mHC site a mixing function belongs to.
///
/// The reference graph calls `build_hc_pre` twice per layer — once before the
/// attention site and once before the FFN site — with different weights each
/// time, so both must be resident and separately addressable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53HyperBranch {
    Attention,
    Ffn,
}

impl Glm53HyperBranch {
    const fn index(self) -> usize {
        match self {
            Self::Attention => 0,
            Self::Ffn => 1,
        }
    }
}

/// The expanded F32 mixing functions, addressed by (layer, branch).
#[derive(Debug, Clone, Copy)]
pub struct Glm53MhcExpanded {
    base: DevicePtr,
}

impl Glm53MhcExpanded {
    /// Bind the arena region without writing to it.
    ///
    /// `region_bytes` must be the region's true extent; passing a larger value
    /// would move the bounds check off the real allocation.
    pub fn bind(base: DevicePtr, region_bytes: u64) -> Result<Self> {
        ensure!(!base.is_null(), "GLM mHC expansion base is NULL");
        ensure!(
            base.0.is_multiple_of(256),
            "GLM mHC expansion base is 256-byte misaligned"
        );
        ensure!(
            region_bytes >= GLM53_MHC_EXPANDED_F32_BYTES,
            "GLM mHC expansion needs {GLM53_MHC_EXPANDED_F32_BYTES} bytes, region is {region_bytes}"
        );
        Ok(Self { base })
    }

    /// The F32 buffer for one layer's branch.
    pub fn slot(&self, layer: usize, branch: Glm53HyperBranch) -> Result<GgmlIqBuffer> {
        ensure!(layer < LAYERS, "GLM layer {layer} has no mHC connections");
        let index = layer * BRANCHES + branch.index();
        let offset = (index as u64)
            .checked_mul(GLM53_MHC_FUNCTION_F32_BYTES)
            .context("GLM mHC slot offset overflow")?;
        let end = offset
            .checked_add(GLM53_MHC_FUNCTION_F32_BYTES)
            .context("GLM mHC slot extent overflow")?;
        ensure!(
            end <= GLM53_MHC_EXPANDED_F32_BYTES,
            "GLM mHC slot for layer {layer} ends beyond its region"
        );
        Ok(GgmlIqBuffer {
            ptr: DevicePtr(
                self.base
                    .0
                    .checked_add(offset)
                    .context("GLM mHC slot address overflow")?,
            ),
            bytes: usize::try_from(GLM53_MHC_FUNCTION_F32_BYTES)?,
        })
    }

    /// Dequantize every layer's two mixing functions into the region.
    ///
    /// Enqueues exactly `2 * layers` launches on `stream` and writes nothing
    /// else. Returns the number of launches so a caller can assert no layer was
    /// skipped — a skipped branch leaves zeroed mixing coefficients, which
    /// produces fluent, wrong output rather than an error.
    pub fn expand(
        &self,
        gpu: &dyn GpuBackend,
        kernel: &GgmlQ8F32Kernel,
        layers: &[&Glm53HyperWeights],
        stream: u64,
    ) -> Result<usize> {
        ensure!(
            layers.len() == LAYERS,
            "GLM mHC expansion needs exactly {LAYERS} target layers, got {}",
            layers.len()
        );
        // rows = MIX, columns = HC * HIDDEN: the on-disk order, not a transpose.
        let plan = GgmlQ8F32Plan::new(MIX, FUNCTION_INNER)?;
        ensure!(
            plan.destination_bytes == usize::try_from(GLM53_MHC_FUNCTION_F32_BYTES)?,
            "GLM mHC expanded function extent drift"
        );

        let mut launches = 0usize;
        for (index, layer) in layers.iter().enumerate() {
            for (branch, weights) in [
                (Glm53HyperBranch::Attention, &layer.attention),
                (Glm53HyperBranch::Ffn, &layer.ffn),
            ] {
                let source = weights.function.buffer();
                ensure!(
                    source.bytes == plan.source_bytes,
                    "GLM layer {index} {branch:?} mHC function is {} bytes, expected {}",
                    source.bytes,
                    plan.source_bytes
                );
                kernel.launch(gpu, plan, source, self.slot(index, branch)?, stream)?;
                launches += 1;
            }
        }
        ensure!(
            launches == LAYERS * BRANCHES,
            "GLM mHC expansion issued {launches} launches"
        );
        Ok(launches)
    }
}

#[cfg(test)]
#[path = "mhc_expansion_tests.rs"]
mod tests;
