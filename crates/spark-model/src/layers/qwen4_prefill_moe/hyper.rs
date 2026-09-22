// SPDX-License-Identifier: AGPL-3.0-only

//! Optional exact HC preparation and one saved-scale injection for grouped FFN.

use crate::{
    layer::ForwardContext,
    layers::{FfnComponent, Qwen4HyperConnection},
};
use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

pub(super) fn finish(
    mlp: &Qwen4HyperConnection,
    ffn: &FfnComponent,
    hidden: DevicePtr,
    residual: DevicePtr,
    rows: usize,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
    let input = mlp.prepare_prefill_exact(
        hidden,
        residual,
        rows,
        ctx.buffers,
        ctx.gpu,
        ctx.config.rms_norm_eps as f32,
        stream,
    )?;
    ffn.forward_prefill(input, rows, ctx, stream)?;
    mlp.inject_saved_batched(
        hidden,
        ctx.buffers.moe_output(),
        residual,
        rows,
        ctx.gpu,
        stream,
    )
}
