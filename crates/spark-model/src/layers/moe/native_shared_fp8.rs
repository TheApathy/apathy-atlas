// SPDX-License-Identifier: AGPL-3.0-only

//! Native block-FP8 shared projections, independently of routed precision.

use super::vision_l0_dump::{MoeCapture, Stage};
use super::{DevicePtr, ForwardContext, Fp8ExpertWeight, GpuBackend, KernelHandle, MoeLayer, ops};
use crate::weight_map::WeightQuantFormat;
use anyhow::{Result, ensure};

pub(super) struct NativeFp8SharedExpert {
    weights: Fp8ExpertWeight,
    handles: [KernelHandle; 3], // GEMV, GEMM, Vision SwiGLU clamp10
    exact_batch: KernelHandle,
}

impl NativeFp8SharedExpert {
    fn new(
        weights: Fp8ExpertWeight,
        handles: [KernelHandle; 3],
        exact_batch: KernelHandle,
    ) -> Result<Self> {
        ensure!(
            handles.iter().all(|handle| handle.0 != 0) && exact_batch.0 != 0,
            "native shared FP8 requires GEMV, GEMM, exact batch and Vision clamp10 kernels"
        );
        for (weight, n, k) in [
            (weights.gate_proj, 2048, 4096),
            (weights.up_proj, 2048, 4096),
            (weights.down_proj, 4096, 2048),
        ] {
            ensure!(
                !weight.weight.is_null() && !weight.row_scale.is_null(),
                "native shared FP8 requires non-null weight and block-scale pointers"
            );
            ensure!(
                weight.n == n
                    && weight.k == k
                    && weight.scale_format == WeightQuantFormat::Fp8BlockScaled,
                "native shared FP8 projection shape/scale format mismatch"
            );
        }
        Ok(Self {
            weights,
            handles,
            exact_batch,
        })
    }
}

impl MoeLayer {
    pub(crate) fn set_native_fp8_shared_expert(
        &mut self,
        weights: Fp8ExpertWeight,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        ensure!(
            self.native_shared_fp8.is_none()
                && self.bf16_shared_expert.is_none()
                && self.fp8_shared_expert_mirror.is_none()
                && self.shared_gate_fp8.is_none()
                && self.shared_up_fp8.is_none()
                && self.shared_down_fp8.is_none(),
            "native shared FP8 conflicts with existing shared precision state"
        );
        ensure!(
            self.exl3.is_some(),
            "native shared FP8 requires EXL3 routed state"
        );
        // The pinned actual Vision Expert applies swiglu_limit=10 to SHARED
        // as well as routed experts: gate<=10 and up in [-10,10] before SiLU.
        let handles = [
            gpu.kernel("w8a16_gemv", "w8a16_gemv")?,
            gpu.kernel("w8a16_gemm", "w8a16_gemm")?,
            gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
        ];
        let exact_batch = gpu.kernel("w8a16_gemv_batch4", "w8a16_gemv_batchm_exact")?;
        self.native_shared_fp8 = Some(NativeFp8SharedExpert::new(weights, handles, exact_batch)?);
        Ok(())
    }

    /// Wide verify must retain the single-row decoder's FP8 GEMV reduction
    /// order and Vision clamp10. Prefill's GEMM is intentionally not used here.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_native_fp8_shared_verify(
        &self,
        input: DevicePtr,
        rows: u32,
        h: u32,
        inter: u32,
        gate: DevicePtr,
        up: DevicePtr,
        down: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if self.native_shared_fp8.is_none() {
            return Ok(false);
        }
        ensure!(
            gate == ctx.buffers.logits()
                && up == ctx.buffers.ssm_qkvz()
                && down == ctx.buffers.attn_output(),
            "native FP8 shared verify scratch must use the checked arena allocations"
        );
        let sizes = ctx.buffers.sizes();
        let _plan = super::native_shared_verify::NativeSharedVerifyPlan::new(
            rows,
            ctx.buffers.max_batch_tokens(),
            h,
            inter,
            [input, gate, up, down],
            [sizes.logits, sizes.ssm_qkvz, sizes.attn_output],
        )?;
        let state = self.native_shared_fp8.as_ref().unwrap();
        let project = |input, weight: crate::weight_map::Fp8Weight, output, lda, ldc| {
            ops::w8a16_gemv_batchm_exact(
                ctx.gpu,
                state.exact_batch,
                input,
                weight.weight,
                weight.row_scale,
                output,
                rows,
                weight.n,
                weight.k,
                lda,
                ldc,
                stream,
            )
        };
        project(input, state.weights.gate_proj, gate, h, inter)?;
        project(input, state.weights.up_proj, up, h, inter)?;
        let elements = rows
            .checked_mul(inter)
            .ok_or_else(|| anyhow::anyhow!("native shared FP8 activation extent overflow"))?;
        ops::silu_mul(ctx.gpu, state.handles[2], gate, up, gate, elements, stream)?;
        project(gate, state.weights.down_proj, down, inter, h)?;
        Ok(true)
    }

    pub(super) fn run_native_fp8_shared_expert(
        &self,
        input: DevicePtr,
        rows: u32,
        h: u32,
        inter: u32,
        gate: DevicePtr,
        up: DevicePtr,
        down: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        self.run_native_fp8_shared_expert_observed(
            input, rows, h, inter, gate, up, down, ctx, stream, None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_native_fp8_shared_expert_observed(
        &self,
        input: DevicePtr,
        rows: u32,
        h: u32,
        inter: u32,
        gate: DevicePtr,
        up: DevicePtr,
        down: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
        mut capture: Option<&mut MoeCapture>,
    ) -> Result<bool> {
        let Some(state) = &self.native_shared_fp8 else {
            ensure!(
                capture.is_none(),
                "MoE capture requires native shared state"
            );
            return Ok(false);
        };
        ensure!(
            ctx.config.deepseek_vision.is_some()
                && self.exl3.is_some()
                && !ctx.graph_capture
                && rows > 0
                && rows as usize <= ctx.buffers.max_batch_tokens()
                && h == 4096
                && inter == 2048,
            "native shared FP8 requires eager actual-Vision EXL3 geometry"
        );
        ensure!(
            [input, gate, up, down].iter().all(|ptr| !ptr.is_null()),
            "native shared FP8 activation pointer is null"
        );
        let project = |input, weight: crate::weight_map::Fp8Weight, output| {
            if rows == 1 {
                ops::w8a16_gemv(
                    ctx.gpu,
                    state.handles[0],
                    input,
                    weight.weight,
                    weight.row_scale,
                    output,
                    weight.n,
                    weight.k,
                    stream,
                )
            } else {
                ops::w8a16_gemm(
                    ctx.gpu,
                    state.handles[1],
                    input,
                    weight.weight,
                    weight.row_scale,
                    output,
                    rows,
                    weight.n,
                    weight.k,
                    stream,
                )
            }
        };
        if let Some(capture) = capture.as_deref_mut() {
            capture.stage(Stage::SharedInput, ctx.gpu, input, stream)?;
            capture.native_weights(state.weights, state.handles, ctx.gpu, stream)?;
        }
        project(input, state.weights.gate_proj, gate)?;
        if let Some(capture) = capture.as_deref_mut() {
            capture.stage(Stage::SharedGate, ctx.gpu, gate, stream)?;
        }
        project(input, state.weights.up_proj, up)?;
        if let Some(capture) = capture.as_deref_mut() {
            capture.stage(Stage::SharedUp, ctx.gpu, up, stream)?;
        }
        let elements = rows
            .checked_mul(inter)
            .ok_or_else(|| anyhow::anyhow!("native shared FP8 activation extent overflow"))?;
        ops::silu_mul(ctx.gpu, state.handles[2], gate, up, gate, elements, stream)?;
        if let Some(capture) = capture.as_deref_mut() {
            capture.stage(Stage::SharedActivation, ctx.gpu, gate, stream)?;
        }
        project(gate, state.weights.down_proj, down)?;
        if let Some(capture) = capture.as_deref_mut() {
            capture.stage(Stage::SharedDown, ctx.gpu, down, stream)?;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_map::{Fp8Weight, WeightQuantFormat};

    fn weights() -> Fp8ExpertWeight {
        let weight = |n, k| Fp8Weight {
            weight: DevicePtr(0x1000),
            row_scale: DevicePtr(0x2000),
            n,
            k,
            scale_format: WeightQuantFormat::Fp8BlockScaled,
        };
        Fp8ExpertWeight {
            gate_proj: weight(2048, 4096),
            up_proj: weight(2048, 4096),
            down_proj: weight(4096, 2048),
        }
    }

    #[test]
    fn native_shared_fp8_requires_every_projection_pointer_shape_and_block_format() {
        let handles = [KernelHandle(1), KernelHandle(2), KernelHandle(3)];
        NativeFp8SharedExpert::new(weights(), handles, KernelHandle(4)).unwrap();
        for index in 0..3 {
            for fault in 0..4 {
                let mut weights = weights();
                let weight = match index {
                    0 => &mut weights.gate_proj,
                    1 => &mut weights.up_proj,
                    _ => &mut weights.down_proj,
                };
                match fault {
                    0 => weight.weight = DevicePtr::NULL,
                    1 => weight.row_scale = DevicePtr::NULL,
                    2 => weight.n += 1,
                    _ => weight.scale_format = WeightQuantFormat::Fp8PerRow,
                }
                assert!(NativeFp8SharedExpert::new(weights, handles, KernelHandle(4)).is_err());
            }
        }
    }

    #[test]
    fn native_shared_fp8_requires_gemv_gemm_and_vision_clamped_activation() {
        for index in 0..3 {
            let mut handles = [KernelHandle(1), KernelHandle(2), KernelHandle(3)];
            handles[index] = KernelHandle(0);
            assert!(NativeFp8SharedExpert::new(weights(), handles, KernelHandle(4)).is_err());
        }
        assert!(
            NativeFp8SharedExpert::new(weights(), [KernelHandle(1); 3], KernelHandle(0)).is_err()
        );
    }
}
