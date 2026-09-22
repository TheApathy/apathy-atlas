// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit GEMV family for committed rows; borrows every existing device owner.
use super::gemv_plan::{GemvChunk, GemvIo, GemvPlan};
use super::*;
use crate::weight_map::DenseWeight;

#[derive(Clone, Copy)]
pub(super) struct GemvKernels {
    single: KernelHandle,
    pair: KernelHandle,
}

impl GemvKernels {
    pub(super) fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        let single = gpu.kernel("gemv", "dense_gemv_bf16")?;
        let pair = gpu.kernel("dense_gemv_bf16_batch2", "dense_gemv_bf16_batch2")?;
        ensure!(
            single.0 != 0 && pair.0 != 0,
            "GEMV projection kernel handle is null"
        );
        Ok(Self { single, pair })
    }

    pub(super) fn plan(
        self,
        input: GgmlIqBuffer,
        weight: &DenseWeight,
        output: GgmlIqBuffer,
        rows: u32,
    ) -> Result<GemvPlan> {
        // The typed weight loader already admits every K/V tensor as exactly
        // [KV_WIDTH,HIDDEN] BF16; no weight conversion or duplicate storage.
        GemvPlan::new(
            rows,
            MAX_CONTEXT_TOKENS,
            HIDDEN,
            KV_WIDTH,
            (input.ptr.0, input.bytes),
            (weight.weight.0, KV_WIDTH as usize * HIDDEN as usize * 2),
            (output.ptr.0, output.bytes),
            [self.single.0, self.pair.0],
        )
    }

    pub(super) fn project(
        self,
        gpu: &dyn GpuBackend,
        input: GgmlIqBuffer,
        weight: &DenseWeight,
        output: GgmlIqBuffer,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        let plan = self.plan(input, weight, output, rows)?;
        // Existing typed launch-builder host allocations are included in wall
        // timing. No device/metadata allocation, upload, or lookup is done here.
        plan.execute(&mut DeviceGemvIo {
            gpu,
            plan: &plan,
            stream,
        })
    }
}

struct DeviceGemvIo<'a> {
    gpu: &'a dyn GpuBackend,
    plan: &'a GemvPlan,
    stream: u64,
}
impl GemvIo for DeviceGemvIo<'_> {
    fn launch(&mut self, chunk: GemvChunk) -> Result<()> {
        let weight = DenseWeight {
            weight: DevicePtr(self.plan.weight()),
        };
        let [single, pair] = self.plan.handles();
        if chunk.rows == 2 {
            ops::dense_gemv_batch2(
                self.gpu,
                KernelHandle(pair),
                DevicePtr(chunk.input),
                &weight,
                DevicePtr(chunk.output),
                self.plan.width(),
                self.plan.hidden(),
                self.plan.width(),
                self.stream,
            )
        } else {
            ops::dense_gemv(
                self.gpu,
                KernelHandle(single),
                DevicePtr(chunk.input),
                &weight,
                DevicePtr(chunk.output),
                self.plan.width(),
                self.plan.hidden(),
                self.stream,
            )
        }
    }
}

impl Glm53Dflash2Runtime {
    /// Pure admission before any observer or submission. Full-context spans
    /// cover every possible retained tail; offsets are whole HIDDEN-sized rows.
    /// Owners/addresses are immutable for this borrow, and each actual tail is
    /// checked again by GemvPlan before its first pair is submitted.
    pub(super) fn preflight_gemv_projection(&self, projection: CommittedProjection) -> Result<()> {
        let CommittedProjection::StableGemv(kernels) = projection else {
            return Ok(());
        };
        self.plan.validate_layout()?;
        ensure!(self.arena != DevicePtr::NULL, "GEMV runtime arena is null");
        self.arena
            .0
            .checked_add(u64::try_from(self.plan.arena_bytes)?)
            .context("GEMV runtime arena end overflow")?;
        let input = self.region(self.plan.projected_target);
        let key = self.region(self.plan.key);
        let value = self.region(self.plan.value);
        for layer in &self.weights.layers {
            kernels.plan(input, &layer.k_proj, key, self.context_tokens)?;
            kernels.plan(input, &layer.v_proj, value, self.context_tokens)?;
        }
        Ok(())
    }
}
