// SPDX-License-Identifier: AGPL-3.0-only

//! Exact K1-order routing shared by ordinary NVFP4 SSM projections.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::Qwen3SsmLayer;
use crate::layers::ops::{self, ExactLmHeadRoute, W4a16ExactLmHeadKernels};
use crate::weight_map::QuantizedWeight;

const BF16_BYTES: usize = 2;

pub(super) fn exact_projection_route(
    kernels: W4a16ExactLmHeadKernels,
    rows: u32,
) -> Result<ExactLmHeadRoute> {
    ensure!(
        (2..=32).contains(&rows),
        "exact NVFP4 SSM projection rows must be in 2..=32, got {rows}"
    );
    kernels
        .route_for_rows(rows)
        .ok_or_else(|| anyhow::anyhow!("missing exact NVFP4 SSM projection tier for M={rows}"))
}

impl Qwen3SsmLayer {
    /// Project a row-major BF16 `[M,K]` slab through an ordinary row-major
    /// NVFP4 `[N,K]` weight. The exact tier preserves ordinary K1 accumulation
    /// order; an unavailable tier runs independent ordinary K1 launches.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_nvfp4_rows_exact_or_k1(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &QuantizedWeight,
        output: DevicePtr,
        rows: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<ExactLmHeadRoute> {
        ensure!(n > 0, "NVFP4 SSM projection output width must be non-zero");
        ensure!(k > 0, "NVFP4 SSM projection input width must be non-zero");
        ensure!(
            k.is_multiple_of(16),
            "NVFP4 SSM projection input width must be divisible by 16, got {k}"
        );

        let route = exact_projection_route(self.w4a16_exact_projection_kernels, rows)?;
        match route {
            ExactLmHeadRoute::Exact(_) => ops::w4a16_gemv_batch_logits_exact(
                gpu,
                self.w4a16_exact_projection_kernels,
                input,
                weight,
                output,
                rows,
                n,
                k,
                stream,
            )?,
            ExactLmHeadRoute::SerialK1(_) => {
                for row in 0..rows {
                    ops::w4a16_decode_gemv(
                        gpu,
                        self.w4a16_gemv_k,
                        self.w4a16_gemv_sw_k,
                        self.gemv_sw,
                        input.offset(row as usize * k as usize * BF16_BYTES),
                        weight,
                        output.offset(row as usize * n as usize * BF16_BYTES),
                        n,
                        k,
                        stream,
                    )?;
                }
            }
        }
        Ok(route)
    }
}
