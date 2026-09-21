// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Exact 256-expert/top-6 Vision router; a null table means learned routing.
#[allow(clippy::too_many_arguments)]
pub fn deepseek_visual_route(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits: DevicePtr,
    table: DevicePtr,
    token_ids: DevicePtr,
    text_bias: DevicePtr,
    visual_bias: DevicePtr,
    indices: DevicePtr,
    weights: DevicePtr,
    vocab: u32,
    scale: f32,
    rows: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        rows > 0 && vocab > 0 && vocab.checked_add(5).is_some(),
        "invalid DeepSeek Vision routing extent"
    );
    ensure!(
        scale.is_finite() && scale > 0.0 && kernel.0 != 0,
        "invalid DeepSeek Vision routing kernel/scale"
    );
    ensure!(
        [logits, token_ids, text_bias, visual_bias, indices, weights]
            .iter()
            .all(|p| *p != DevicePtr::NULL),
        "null DeepSeek Vision routing operand"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([rows, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(logits)
        .arg_ptr(table)
        .arg_ptr(token_ids)
        .arg_ptr(text_bias)
        .arg_ptr(visual_bias)
        .arg_ptr(indices)
        .arg_ptr(weights)
        .arg_u32(vocab)
        .arg_f32(scale)
        .launch(stream)
}
