// SPDX-License-Identifier: AGPL-3.0-only

//! Untimed same-input replay against the actual shipping original-layout GEMM.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::{ExpertPtrTable, MoeLayer, ops, qwen4_compact_compare as compare};
use crate::layer::ForwardContext;

const BF16_CHUNK_BYTES: usize = 512 * 1024;

pub(super) fn check_planned_after_stream(
    ctx: &ForwardContext,
    status: DevicePtr,
    stream: u64,
) -> Result<()> {
    let mut bytes = [0u8; 4];
    ctx.gpu.copy_d2h_on_stream(status, &mut bytes, stream)?;
    let code = i32::from_le_bytes(bytes);
    super::qwen4_compact_contract::check_status(code)
        .map_err(|error| anyhow::anyhow!("{error}: status={code}"))
}

fn capture(
    ctx: &ForwardContext,
    ptr: DevicePtr,
    bytes: usize,
    stage: &str,
    stream: u64,
) -> Result<Vec<u8>> {
    ctx.gpu.synchronize(stream)?;
    let mut saved = Vec::new();
    saved.try_reserve_exact(bytes)?;
    saved.resize(bytes, 0);
    for (chunk, out) in saved.chunks_mut(BF16_CHUNK_BYTES).enumerate() {
        ctx.gpu
            .copy_d2h(ptr.offset(chunk * BF16_CHUNK_BYTES), out)?;
        compare::finite(out).map_err(|error| {
            anyhow::anyhow!(
                "compact CHECK {stage} candidate at byte {}: {error:?}",
                chunk * BF16_CHUNK_BYTES
            )
        })?;
    }
    Ok(saved)
}

fn compare_device(
    ctx: &ForwardContext,
    ptr: DevicePtr,
    saved: &[u8],
    stage: &str,
    stream: u64,
) -> Result<()> {
    ctx.gpu.synchronize(stream)?;
    let mut actual = vec![0u8; BF16_CHUNK_BYTES.min(saved.len())];
    for (chunk, expected) in saved.chunks(BF16_CHUNK_BYTES).enumerate() {
        let actual = &mut actual[..expected.len()];
        ctx.gpu
            .copy_d2h(ptr.offset(chunk * BF16_CHUNK_BYTES), actual)?;
        compare::exact(expected, actual).map_err(|error| {
            anyhow::anyhow!(
                "compact CHECK {stage} at byte {}: {error:?}",
                chunk * BF16_CHUNK_BYTES
            )
        })?;
    }
    Ok(())
}

impl MoeLayer {
    #[allow(clippy::too_many_arguments)]
    fn check_compact_projection(
        &self,
        input: DevicePtr,
        table: &ExpertPtrTable,
        output: DevicePtr,
        offsets: DevicePtr,
        sorted_ids: DevicePtr,
        rows: usize,
        n: u32,
        k: u32,
        stage: &str,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let bytes = compare::extent(rows, n as usize).map_err(anyhow::Error::msg)?;
        let saved = capture(ctx, output, bytes, stage, stream)?;
        // Independent initialization prevents an omitted shipping store from
        // inheriting the candidate value and making the comparison vacuous.
        ctx.gpu.memset_async(output, 0, bytes, stream)?;
        ops::moe_w4a16_grouped_gemm_ptrtable(
            ctx.gpu,
            self.moe_grouped_gemm,
            input,
            table.packed_ptrs,
            table.scale_ptrs,
            table.scale2_vals,
            output,
            offsets,
            sorted_ids,
            512,
            n,
            k,
            (rows * 10).div_ceil(64) as u32,
            stream,
        )?;
        compare_device(ctx, output, &saved, stage, stream)
    }

    pub(super) fn check_compact_gate_up(
        &self,
        input: DevicePtr,
        offsets: DevicePtr,
        sorted_ids: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.check_compact_projection(
            input,
            &self.gate_ptrs,
            ctx.buffers.expert_gate_out(),
            offsets,
            sorted_ids,
            rows,
            640,
            2560,
            "gate",
            ctx,
            stream,
        )?;
        self.check_compact_projection(
            input,
            &self.up_ptrs,
            ctx.buffers.expert_up_out(),
            offsets,
            sorted_ids,
            rows,
            640,
            2560,
            "up",
            ctx,
            stream,
        )
    }

    pub(super) fn check_compact_down(
        &self,
        offsets: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Shipping down reads expanded activation rows directly (NULL IDs).
        self.check_compact_projection(
            ctx.buffers.expert_gate_out(),
            &self.down_ptrs,
            ctx.buffers.expert_down_out(),
            offsets,
            DevicePtr::NULL,
            rows,
            2560,
            640,
            "down",
            ctx,
            stream,
        )?;
        tracing::info!(
            rows,
            "QWEN4_COMPACT_CHECK PASS gate/up/down finite BF16 byte-exact; timing ineligible"
        );
        Ok(())
    }
}
