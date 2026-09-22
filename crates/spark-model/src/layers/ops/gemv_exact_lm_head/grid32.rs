// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::{
    W4A16_EXACT_LM_HEAD_OUTS_PER_BLOCK, W4A16_EXACT_LM_HEAD_RT2_TILE, W4a16ExactLmHeadKernels,
};
use crate::weight_map::QuantizedWeight;

/// Launch complete M32 row tiles as one two-dimensional exact RT2 grid.
/// `blockIdx.y` changes only the disjoint A/C row slab; every block retains
/// the incumbent per-row K16 ownership and reduction tree.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_batch_logits_exact_rt2_m32_grid(
    gpu: &dyn GpuBackend,
    kernels: W4a16ExactLmHeadKernels,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    rows: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (32..=2048).contains(&rows) && rows.is_multiple_of(32),
        "exact GRID32 rows must be a non-zero multiple of 32 within 2048, got {rows}"
    );
    ensure!(n > 0, "exact GRID32 output width must be non-zero");
    ensure!(
        k > 0 && k.is_multiple_of(16),
        "exact GRID32 hidden width must be non-zero and divisible by 16, got {k}"
    );
    let kernel = kernels.rt2_m32_grid();
    ensure!(kernel.0 != 0, "missing exact RT2 M32 GRID32 kernel");
    KernelLaunch::new(gpu, kernel)
        .grid([
            div_ceil(
                n,
                W4A16_EXACT_LM_HEAD_OUTS_PER_BLOCK * W4A16_EXACT_LM_HEAD_RT2_TILE,
            ),
            rows / 32,
            1,
        ])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(rows)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}
