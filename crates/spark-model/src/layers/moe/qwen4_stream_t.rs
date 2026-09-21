// SPDX-License-Identifier: AGPL-3.0-only

//! Exact per-layer Qwen4 routed-expert streaming transpose.

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, ensure};

use super::*;

static RECEIPT_LOGGED: AtomicBool = AtomicBool::new(false);

impl MoeLayer {
    pub(crate) fn set_moe_stream_transpose_scratch(&mut self, scratch: MoeStreamTransposeScratch) {
        self.gate_ptrs_t = Some(ExpertPtrTable {
            packed_ptrs: scratch.packed_tables[0],
            scale_ptrs: scratch.scale_tables[0],
            scale2_vals: self.gate_ptrs.scale2_vals,
        });
        self.up_ptrs_t = Some(ExpertPtrTable {
            packed_ptrs: scratch.packed_tables[1],
            scale_ptrs: scratch.scale_tables[1],
            scale2_vals: self.up_ptrs.scale2_vals,
        });
        self.down_ptrs_t = Some(ExpertPtrTable {
            packed_ptrs: scratch.packed_tables[2],
            scale_ptrs: scratch.scale_tables[2],
            scale2_vals: self.down_ptrs.scale2_vals,
        });
        self.stream_t_scratch = Some(scratch);
    }

    fn transpose_projection(
        &self,
        source: &ExpertPtrTable,
        packed_dst: DevicePtr,
        scale_dst: DevicePtr,
        destination: &ExpertPtrTable,
        n: u32,
        k: u32,
        experts: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ops::moe_transpose_u8_batched(
            ctx.gpu,
            self.moe_transpose_u8_batched_k,
            source.packed_ptrs,
            destination.packed_ptrs,
            n,
            k / 2,
            experts,
            stream,
        )?;
        ops::moe_transpose_u8_batched(
            ctx.gpu,
            self.moe_transpose_u8_batched_k,
            source.scale_ptrs,
            destination.scale_ptrs,
            n,
            k / 16,
            experts,
            stream,
        )?;
        ensure!(
            !packed_dst.is_null() && !scale_dst.is_null(),
            "Qwen4 streaming transpose scratch is null"
        );
        Ok(())
    }

    pub(super) fn populate_qwen4_stream_t(
        &self,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(rows > 1, "Qwen4 streaming transpose is prefill-only");
        let scratch = self
            .stream_t_scratch
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Qwen4 streaming transpose scratch is absent"))?;
        let gate = self
            .gate_ptrs_t
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Qwen4 stream gate table is absent"))?;
        let up = self
            .up_ptrs_t
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Qwen4 stream up table is absent"))?;
        let down = self
            .down_ptrs_t
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Qwen4 stream down table is absent"))?;
        ensure!(
            ctx.config.num_experts == 512
                && ctx.config.hidden_size == 2560
                && ctx.config.moe_intermediate_size == 640,
            "Qwen4 streaming transpose geometry changed"
        );
        self.transpose_projection(
            &self.gate_ptrs,
            scratch.packed[0],
            scratch.scale[0],
            gate,
            640,
            2560,
            512,
            ctx,
            stream,
        )?;
        self.transpose_projection(
            &self.up_ptrs,
            scratch.packed[1],
            scratch.scale[1],
            up,
            640,
            2560,
            512,
            ctx,
            stream,
        )?;
        self.transpose_projection(
            &self.down_ptrs,
            scratch.packed[2],
            scratch.scale[2],
            down,
            2560,
            640,
            512,
            ctx,
            stream,
        )?;
        if !RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
            tracing::info!(
                "MOE_PREFILL_STREAM_T_ENGAGED selector={} rows={rows} bytes=1415577600",
                super::qwen4_prefill_compact::STREAM_SELECTOR,
            );
        }
        Ok(())
    }
}
