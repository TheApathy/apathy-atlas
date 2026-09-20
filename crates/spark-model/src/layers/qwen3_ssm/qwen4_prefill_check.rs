// SPDX-License-Identifier: AGPL-3.0-only

//! Same-input shadow replay; no device allocations or performance eligibility.

use super::qwen4_prefill_check_raw::{Element, compare_finite, finite};
use super::qwen4_prefill_exact_plan::Plan;
use super::*;
use spark_runtime::gpu::HostToDeviceCopy;

const ROW_BYTES: usize = 10_240 * 2;
const TAIL_BYTES: usize = 4 * 2;
const CHUNK: usize = 512 * 1024;

pub(super) struct Snapshot {
    rows: usize,
    hidden: Vec<u8>,
    h: Vec<u8>,
    conv: Vec<u8>,
}

fn read(
    ptr: DevicePtr,
    count: usize,
    element: Element,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<Vec<u8>> {
    let mut bytes = vec![0u8; count];
    ctx.gpu.copy_d2h_on_stream(ptr, &mut bytes, stream)?;
    finite(&bytes, element).map_err(anyhow::Error::msg)?;
    Ok(bytes)
}

impl Snapshot {
    /// The caller has already admitted every state/arena/kernel before effects.
    pub(super) fn capture(
        rows: usize,
        hidden: DevicePtr,
        state: &SsmLayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Self> {
        let p = Plan::new(rows, ctx.buffers.max_batch_tokens(), [usize::MAX; 6])
            .map_err(anyhow::Error::msg)?;
        anyhow::ensure!(
            p.residual_bytes <= ctx.buffers.sizes().hidden_states,
            "SSM check hidden extent exceeds arena"
        );
        Ok(Self {
            rows,
            hidden: read(hidden, p.residual_bytes, Element::Bf16, ctx, stream)?,
            h: read(
                state.h_state,
                Plan::H_STATE_BYTES,
                Element::F32,
                ctx,
                stream,
            )?,
            conv: read(
                state.conv_state,
                Plan::CONV_STATE_BYTES,
                Element::F32,
                ctx,
                stream,
            )?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn verify(
        self,
        layer: &Qwen3SsmLayer,
        hidden: DevicePtr,
        residual: DevicePtr,
        rows: usize,
        state: &mut SsmLayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(rows == self.rows, "SSM check row count changed");
        let candidate = Self::capture(rows, hidden, state, ctx, stream)?;
        let mut saved_inject = vec![0u8; rows * TAIL_BYTES];
        for row in 0..rows {
            ctx.gpu.copy_d2h_on_stream(
                residual.offset((row + 1) * ROW_BYTES - TAIL_BYTES),
                &mut saved_inject[row * TAIL_BYTES..(row + 1) * TAIL_BYTES],
                stream,
            )?;
        }
        finite(&saved_inject, Element::Bf16).map_err(anyhow::Error::msg)?;
        ctx.gpu.copy_h2d_group_on_stream(
            &[
                HostToDeviceCopy::new(&self.hidden, hidden),
                HostToDeviceCopy::new(&self.h, state.h_state),
                HostToDeviceCopy::new(&self.conv, state.conv_state),
            ],
            stream,
        )?;
        // Release initial snapshots before scalar replay/readback. Peak owned
        // host snapshots <87 MiB at M2048; comparison uses <=512 KiB scratch.
        drop(self);
        let attn = layer
            .qwen4_attn_hyper
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("SSM check lost attention HC"))?;
        let eps = ctx.config.rms_norm_eps as f32;
        let trace = std::env::var("ATLAS_SSM_TRACE").ok().as_deref() == Some("1");
        for row in 0..rows {
            let hidden_row = hidden.offset(row * ROW_BYTES);
            let residual_row = residual.offset(row * ROW_BYTES);
            let (mixed, inject) =
                attn.prepare_decode(hidden_row, residual_row, ctx.buffers, ctx.gpu, eps, stream)?;
            let inject = inject.ok_or_else(|| anyhow::anyhow!("SSM check lost injection"))?;
            let output = layer.ssm_forward(mixed, state, ctx, stream, trace)?;
            attn.inject_decode(hidden_row, output, inject, ctx.gpu, stream)?;
            let mut reference = [0u8; TAIL_BYTES];
            ctx.gpu.copy_d2h_on_stream(inject, &mut reference, stream)?;
            compare_finite(
                &saved_inject[row * TAIL_BYTES..(row + 1) * TAIL_BYTES],
                &reference,
                Element::Bf16,
            )
            .map_err(|error| anyhow::anyhow!("SSM exact check injection row {row}: {error}"))?;
        }
        candidate.compare_device(
            "hidden",
            hidden,
            &candidate.hidden,
            Element::Bf16,
            ctx,
            stream,
        )?;
        candidate.compare_device("H", state.h_state, &candidate.h, Element::F32, ctx, stream)?;
        candidate.compare_device(
            "conv",
            state.conv_state,
            &candidate.conv,
            Element::F32,
            ctx,
            stream,
        )?;
        // Keep the validated scalar final state; residual's mixed staging is
        // deliberately not compared because scalar HC uses it as norm scratch.
        tracing::info!(
            "QWEN4_SSM_EXACT_CHECK rows={rows} hidden=byte_exact H=byte_exact conv=byte_exact saved_inject=byte_exact timing_eligible=false"
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn compare_device(
        &self,
        name: &str,
        ptr: DevicePtr,
        candidate: &[u8],
        element: Element,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let mut reference = vec![0u8; candidate.len().min(CHUNK)];
        for (index, chunk) in candidate.chunks(CHUNK).enumerate() {
            let offset = index * CHUNK;
            ctx.gpu.copy_d2h_on_stream(
                ptr.offset(offset),
                &mut reference[..chunk.len()],
                stream,
            )?;
            compare_finite(chunk, &reference[..chunk.len()], element).map_err(|error| {
                anyhow::anyhow!("SSM exact check {name} at byte {offset}: {error}")
            })?;
        }
        Ok(())
    }
}
