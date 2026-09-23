// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit arithmetic for committed K/V only. Noise stays original.
use super::gemv_projection::GemvKernels;
use super::projection_contract::{ProjectionFamily, checked_spans};
use super::*;
use crate::weight_map::DenseWeight;

#[derive(Clone, Copy)]
pub(super) enum CommittedProjection {
    Original,
    StableTc(KernelHandle),
    StableGemv(GemvKernels),
}

impl CommittedProjection {
    pub(super) fn for_serving(gpu: &dyn GpuBackend, choice: ServingProjection) -> Result<Self> {
        match choice {
            ServingProjection::Original => Ok(Self::Original),
            ServingProjection::StableGemv => Ok(Self::StableGemv(GemvKernels::load(gpu)?)),
        }
    }

    pub(super) fn serving_choice(self) -> Result<ServingProjection> {
        match self {
            Self::Original => Ok(ServingProjection::Original),
            Self::StableGemv(_) => Ok(ServingProjection::StableGemv),
            Self::StableTc(_) => {
                anyhow::bail!("TC projection is diagnostic-only, not a serving choice")
            }
        }
    }

    pub(super) fn for_probe(gpu: &dyn GpuBackend, mode: Dflash2ProbeMode) -> Result<Self> {
        match mode {
            Dflash2ProbeMode::FullRecompute | Dflash2ProbeMode::CachedPrefix => Ok(Self::Original),
            Dflash2ProbeMode::StableFullProjection | Dflash2ProbeMode::StableCachedProjection => {
                let kernel = gpu.kernel("gemm_tc", "dense_gemm_tc")?;
                ensure!(kernel.0 != 0, "stable projection kernel handle is null");
                Ok(Self::StableTc(kernel))
            }
            Dflash2ProbeMode::StableGemvFullProjection
            | Dflash2ProbeMode::StableGemvCachedProjection => {
                Ok(Self::StableGemv(GemvKernels::load(gpu)?))
            }
        }
    }
    pub(super) fn family(self) -> ProjectionFamily {
        match self {
            Self::Original => ProjectionFamily::Original,
            Self::StableTc(_) => ProjectionFamily::StableTc,
            Self::StableGemv(_) => ProjectionFamily::StableGemv,
        }
    }
    pub(super) fn project_committed(
        self,
        gpu: &dyn GpuBackend,
        input: GgmlIqBuffer,
        weight: &DenseWeight,
        output: GgmlIqBuffer,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        checked_spans(
            rows,
            MAX_CONTEXT_TOKENS,
            HIDDEN,
            KV_WIDTH,
            (input.ptr.0, input.bytes),
            (weight.weight.0, KV_WIDTH as usize * HIDDEN as usize * 2),
            (output.ptr.0, output.bytes),
        )?;
        match self {
            Self::Original => dense(
                input.ptr,
                weight.weight,
                output.ptr,
                rows,
                KV_WIDTH,
                HIDDEN,
                stream,
            ),
            Self::StableTc(kernel) => ops::dense_gemm_tc(
                gpu, kernel, input.ptr, weight, output.ptr, rows, KV_WIDTH, HIDDEN, stream,
            ),
            Self::StableGemv(kernels) => kernels.project(gpu, input, weight, output, rows, stream),
        }
    }
}
