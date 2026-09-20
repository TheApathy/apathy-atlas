// SPDX-License-Identifier: AGPL-3.0-only

//! Private original-layout gate/up/down dispatch; checked status precedes reduction.

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, HostToDeviceCopy};
use spark_runtime::kernel_args::KernelLaunch;

use super::{ExpertPtrTable, MoeLayer, ops, qwen4_compact_contract as abi, qwen4_prefill_compact};
use crate::layer::ForwardContext;

static K32_RECEIPT_LOGGED: AtomicBool = AtomicBool::new(false);

impl MoeLayer {
    fn compact_tables(
        &self,
        transposed: bool,
    ) -> Result<(&ExpertPtrTable, &ExpertPtrTable, &ExpertPtrTable)> {
        if transposed {
            return Ok((
                self.gate_ptrs_t
                    .as_ref()
                    .context("transposed compact gate table absent")?,
                self.up_ptrs_t
                    .as_ref()
                    .context("transposed compact up table absent")?,
                self.down_ptrs_t
                    .as_ref()
                    .context("transposed compact down table absent")?,
            ));
        }
        Ok((&self.gate_ptrs, &self.up_ptrs, &self.down_ptrs))
    }

    pub(super) fn run_qwen4_compact(
        &self,
        input: DevicePtr,
        offsets: DevicePtr,
        sorted_ids: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        qwen4_prefill_compact::validate_context(ctx, rows)?;
        self.validate_qwen4_prefill_moe()?;
        let check = qwen4_prefill_compact::check_selected()?;
        let kernels = self
            .qwen4_compact
            .context("compact kernel bundle is absent")?;
        if kernels.streamed {
            self.populate_qwen4_stream_t(rows, ctx, stream)?;
        }
        let (gate_table, up_table, down_table) = self.compact_tables(kernels.transposed)?;
        ensure!(
            input == ctx.buffers.norm_output()
                && sorted_ids == ctx.buffers.gate_logits()
                && offsets == ctx.buffers.gate_logits().offset(rows * 10 * 8),
            "compact dispatch requires the exact F8 input/sort layout"
        );
        let contract_bytes = abi::contract(rows).map_err(anyhow::Error::msg)?;
        let workspace = ctx.buffers.moe_worklist();
        let contract = workspace.offset(abi::WORKSPACE_BYTES);
        let status = ctx.buffers.moe_worklist_total();
        // The normal sort stores all metadata in gate_logits. Require exact
        // extents here; neither inputs nor metadata may alias writable scratch.
        let input_region = abi::region(input.0, rows * 2560 * 2, 16).map_err(anyhow::Error::msg)?;
        let metadata = abi::region(
            ctx.buffers.gate_logits().0,
            ctx.buffers.sizes().gate_logits,
            16,
        )
        .map_err(anyhow::Error::msg)?;
        for (ptr, bytes) in [(offsets, 513 * 4), (sorted_ids, rows * 10 * 4)] {
            let region = abi::region(ptr.0, bytes, 4).map_err(anyhow::Error::msg)?;
            ensure!(
                region.0 >= metadata.0 && region.1 <= metadata.1,
                "compact routing metadata is outside its admitted arena"
            );
        }
        for (ptr, bytes) in [
            (workspace, abi::ARENA_BYTES),
            (status, 4),
            (ctx.buffers.expert_gate_out(), rows * 10 * 640 * 2),
            (ctx.buffers.expert_up_out(), rows * 10 * 640 * 2),
            (ctx.buffers.expert_down_out(), rows * 10 * 2560 * 2),
        ] {
            ensure!(
                abi::disjoint(
                    input_region,
                    abi::region(ptr.0, bytes, 4).map_err(anyhow::Error::msg)?
                ),
                "compact input aliases writable scratch"
            );
        }
        let pending = (-1i32).to_le_bytes();
        ctx.gpu.copy_h2d_group_on_stream(
            &[
                HostToDeviceCopy::new(&contract_bytes, contract),
                HostToDeviceCopy::new(&pending, status),
            ],
            stream,
        )?;
        KernelLaunch::new(ctx.gpu, kernels.plan)
            .grid([1, 1, 1])
            .block([512, 1, 1])
            .arg_ptr(offsets)
            .arg_ptr(sorted_ids)
            .arg_ptr(contract)
            .arg_ptr(workspace)
            .arg_ptr(status)
            .launch(stream)?;
        let gemm = |a: DevicePtr, table: &ExpertPtrTable, c: DevicePtr, n: u32, k: u32| {
            KernelLaunch::new(ctx.gpu, kernels.gemm)
                .grid([abi::GRID, 1, 1])
                .block([kernels.block, 1, 1])
                .arg_ptr(a)
                .arg_ptr(table.packed_ptrs)
                .arg_ptr(table.scale_ptrs)
                .arg_ptr(table.scale2_vals)
                .arg_ptr(c)
                .arg_ptr(offsets)
                .arg_ptr(sorted_ids)
                .arg_ptr(contract)
                .arg_ptr(workspace)
                .arg_ptr(status)
                .arg_u32(n)
                .arg_u32(k)
                .launch(stream)
        };
        let gate = ctx.buffers.expert_gate_out();
        let up = ctx.buffers.expert_up_out();
        if let Some(gateup) = kernels.gateup {
            ensure!(!check, "fused gate/up rejects the compact CHECK path");
            // One A tile feeds both accumulators; the epilogue applies the
            // shipping silu*up on the BF16-rounded values and writes the
            // activated result into the gate buffer (what silu_mul produced).
            KernelLaunch::new(ctx.gpu, gateup)
                .grid([abi::GRID, 1, 1])
                .block([128, 1, 1])
                .arg_ptr(input)
                .arg_ptr(gate_table.packed_ptrs)
                .arg_ptr(gate_table.scale_ptrs)
                .arg_ptr(gate_table.scale2_vals)
                .arg_ptr(gate)
                .arg_ptr(up_table.packed_ptrs)
                .arg_ptr(up_table.scale_ptrs)
                .arg_ptr(up_table.scale2_vals)
                .arg_ptr(offsets)
                .arg_ptr(sorted_ids)
                .arg_ptr(contract)
                .arg_ptr(workspace)
                .arg_ptr(status)
                .arg_u32(640)
                .arg_u32(2560)
                .launch(stream)?;
        } else {
        gemm(input, gate_table, gate, 640, 2560)?;
        gemm(input, up_table, up, 640, 2560)?;
        if check {
            super::qwen4_compact_check::check_planned_after_stream(ctx, status, stream)?;
            self.check_compact_gate_up(input, offsets, sorted_ids, rows, ctx, stream)?;
        }
        // Exactly the shipping activation and in-place BF16 boundary.
        ops::silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            gate,
            up,
            gate,
            (rows * 10 * 640) as u32,
            stream,
        )?;
        }
        gemm(gate, down_table, ctx.buffers.expert_down_out(), 2560, 640)?;
        // Env-gated oracle capture (ATLAS_QWEN4_ORACLE_DUMP). Records the exact
        // arguments and results of ONE compact-MoE dispatch so the GEMM can be
        // replayed standalone. Off by default; costs nothing when unset.
        self.oracle_capture_compact(
            input, offsets, sorted_ids, rows, gate_table, up_table, down_table, ctx, stream,
        )?;
        // PLANNED=1 is not a completion marker: the ordered D2H drains every
        // GEMM first. Only then can unchanged status admit unpermute/blend/HC.
        // Every planner/GEMM failure (including still PENDING) propagates.
        let mut result = [0u8; 4];
        ctx.gpu.copy_d2h_on_stream(status, &mut result, stream)?;
        let result = i32::from_le_bytes(result);
        abi::check_status(result).map_err(|error| anyhow::anyhow!("{error}: status={result}"))?;
        if kernels.step_k == 32 && !K32_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
            tracing::info!(
                "MOE_PREFILL_COMPACT_K32_ENGAGED selector={} rows={rows} transposed={}",
                if kernels.transposed {
                    if kernels.streamed {
                        qwen4_prefill_compact::STREAM_SELECTOR
                    } else {
                        qwen4_prefill_compact::TRANSPOSED_SELECTOR
                    }
                } else {
                    qwen4_prefill_compact::K32_SELECTOR
                },
                kernels.transposed,
            );
        }
        if check {
            self.check_compact_down(offsets, rows, ctx, stream)?;
        }
        Ok(())
    }
}
