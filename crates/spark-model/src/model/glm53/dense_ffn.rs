// SPDX-License-Identifier: AGPL-3.0-only

//! Dense FFN for GLM-5.3 layers 0..3.
//!
//! Layers below index 3 have no router and no experts: a plain
//! gate/up/SwiGLU/down stack over a 12,288-wide intermediate, versus the MoE
//! experts' 2,048. It is the same shape `glm53_moe_serial` runs for the shared
//! expert, but with different weights and width, and it deliberately does not
//! reuse that executor — the serial MoE path performs a mandatory device-to-host
//! copy to decode routed expert ids, and a dense layer has no routing to read.
//! Borrowing it would buy a stream sync per dense layer for nothing.
//!
//! The MMQ bank is duplicated here rather than shared with
//! `Glm53SerialMoeKernels`, whose source contract pins the exact count of
//! `matrix: &Glm53GgufMatrix` signatures; adding an accessor there would break
//! that pin. The duplication is one `match`, and both are checked against the
//! same `GgmlType`.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::gguf::GgmlType;

use crate::layers::ops::{
    GgmlIqBuffer, GgmlIqMmqKernels, GgmlIqMmqPlan, Glm53ActivationKernels, Glm53SwigluBuffers,
    Glm53SwigluPlan,
};
use crate::weight_loader::{Glm53DenseFfnWeights, Glm53GgufMatrix};

use super::walk_scratch::Glm53DenseFfnBuffers;

/// Dense intermediate width. `Glm53SwigluPlan` admits only 2,048 or 12,288.
const DENSE_INTERMEDIATE: u32 = 12_288;
const HIDDEN: u32 = 4_096;
/// gate, up, down.
pub const GLM53_DENSE_FFN_MATMULS: u32 = 3;
/// Three matmuls plus one SwiGLU — but each `GgmlIqMmqKernels::launch` enqueues
/// **two** kernels, an activation quantize followed by the matmul itself, so the
/// stack costs `3 * 2 + 1 = 7`. The serial MoE budget of 64 decomposes the same
/// way: 8 experts x (3 matmuls + 1 SwiGLU) = 56, the shared branch's 7, and one
/// ordered reduce.
pub const GLM53_DENSE_FFN_KERNEL_LAUNCHES: u32 = 7;

/// Typed MMQ handles for every quant a dense GLM layer can carry.
pub struct Glm53DenseFfnKernels {
    q2_k: GgmlIqMmqKernels,
    q3_k: GgmlIqMmqKernels,
    q4_k: GgmlIqMmqKernels,
    q5_k: GgmlIqMmqKernels,
    q6_k: GgmlIqMmqKernels,
    q8_0: GgmlIqMmqKernels,
    iq2_xxs: GgmlIqMmqKernels,
    iq2_xs: GgmlIqMmqKernels,
    iq2_s: GgmlIqMmqKernels,
    iq3_xxs: GgmlIqMmqKernels,
    iq3_s: GgmlIqMmqKernels,
    iq4_xs: GgmlIqMmqKernels,
    activations: Glm53ActivationKernels,
}

impl Glm53DenseFfnKernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            q2_k: GgmlIqMmqKernels::load(gpu, GgmlType::Q2_K)?,
            q3_k: GgmlIqMmqKernels::load(gpu, GgmlType::Q3_K)?,
            q4_k: GgmlIqMmqKernels::load(gpu, GgmlType::Q4_K)?,
            q5_k: GgmlIqMmqKernels::load(gpu, GgmlType::Q5_K)?,
            q6_k: GgmlIqMmqKernels::load(gpu, GgmlType::Q6_K)?,
            q8_0: GgmlIqMmqKernels::load(gpu, GgmlType::Q8_0)?,
            iq2_xxs: GgmlIqMmqKernels::load(gpu, GgmlType::IQ2_XXS)?,
            iq2_xs: GgmlIqMmqKernels::load(gpu, GgmlType::IQ2_XS)?,
            iq2_s: GgmlIqMmqKernels::load(gpu, GgmlType::IQ2_S)?,
            iq3_xxs: GgmlIqMmqKernels::load(gpu, GgmlType::IQ3_XXS)?,
            iq3_s: GgmlIqMmqKernels::load(gpu, GgmlType::IQ3_S)?,
            iq4_xs: GgmlIqMmqKernels::load(gpu, GgmlType::IQ4_XS)?,
            activations: Glm53ActivationKernels::load(gpu)?,
        })
    }

    fn mmq(&self, kind: GgmlType) -> Result<&GgmlIqMmqKernels> {
        Ok(match kind {
            GgmlType::Q2_K => &self.q2_k,
            GgmlType::Q3_K => &self.q3_k,
            GgmlType::Q4_K => &self.q4_k,
            GgmlType::Q5_K => &self.q5_k,
            GgmlType::Q6_K => &self.q6_k,
            GgmlType::Q8_0 => &self.q8_0,
            GgmlType::IQ2_XXS => &self.iq2_xxs,
            GgmlType::IQ2_XS => &self.iq2_xs,
            GgmlType::IQ2_S => &self.iq2_s,
            GgmlType::IQ3_XXS => &self.iq3_xxs,
            GgmlType::IQ3_S => &self.iq3_s,
            GgmlType::IQ4_XS => &self.iq4_xs,
            _ => bail!("GLM dense FFN matrix has no typed IQ-MMQ kernel"),
        })
    }

    /// One projection. `q8` is scratch; the op is handed exactly the activation
    /// extent its plan derived, never the whole buffer.
    ///
    fn linear(
        &self,
        gpu: &dyn GpuBackend,
        matrix: &Glm53GgufMatrix,
        input: GgmlIqBuffer,
        q8: GgmlIqBuffer,
        output: GgmlIqBuffer,
        stream: u64,
    ) -> Result<GgmlIqMmqPlan> {
        let plan = matrix.plan(1)?;
        ensure!(
            q8.bytes >= plan.activation_bytes,
            "GLM dense FFN Q8 scratch is {} bytes, needs {}",
            q8.bytes,
            plan.activation_bytes
        );
        self.mmq(matrix.kind())?.launch(
            gpu,
            plan,
            input.ptr,
            matrix.buffer(),
            GgmlIqBuffer {
                ptr: q8.ptr,
                bytes: plan.activation_bytes,
            },
            output,
            stream,
        )?;
        Ok(plan)
    }

    /// One projection against an arbitrary catalog matrix.
    ///
    /// Exposed for the LM head, which is a plain `output.weight` matmul with no
    /// dense-FFN geometry to check. Everything else should go through
    /// [`Self::execute`], which pins its shapes.
    pub fn project(
        &self,
        gpu: &dyn GpuBackend,
        matrix: &Glm53GgufMatrix,
        input: GgmlIqBuffer,
        q8: GgmlIqBuffer,
        output: GgmlIqBuffer,
        stream: u64,
    ) -> Result<()> {
        self.linear(gpu, matrix, input, q8, output, stream)?;
        Ok(())
    }

    /// Run one dense FFN layer: `down(swiglu(gate(x), up(x)))`.
    ///
    /// `input` is the normalized block input and `output` the block output.
    /// Enqueues exactly [`GLM53_DENSE_FFN_KERNEL_LAUNCHES`] launches and
    /// performs no host copy or stream synchronization.
    pub fn execute(
        &self,
        gpu: &dyn GpuBackend,
        weights: &Glm53DenseFfnWeights,
        input: GgmlIqBuffer,
        buffers: Glm53DenseFfnBuffers,
        output: GgmlIqBuffer,
        stream: u64,
    ) -> Result<u32> {
        // Shape is pinned rather than inferred: a checkpoint whose dense layer
        // is not 4096 -> 12288 -> 4096 must fail here, not silently run a
        // SwiGLU over the wrong width.
        for (name, matrix, inner, columns) in [
            ("gate", &weights.gate, HIDDEN, DENSE_INTERMEDIATE),
            ("up", &weights.up, HIDDEN, DENSE_INTERMEDIATE),
            ("down", &weights.down, DENSE_INTERMEDIATE, HIDDEN),
        ] {
            ensure!(
                matrix.inner() == inner && matrix.columns() == columns,
                "GLM dense FFN {name} is [{}, {}], expected [{inner}, {columns}]",
                matrix.inner(),
                matrix.columns()
            );
        }

        self.linear(
            gpu,
            &weights.gate,
            input,
            buffers.q8_activation,
            buffers.gate_bf16,
            stream,
        )?;
        self.linear(
            gpu,
            &weights.up,
            input,
            buffers.q8_activation,
            buffers.up_bf16,
            stream,
        )?;
        self.activations.swiglu(
            gpu,
            Glm53SwigluPlan::new(1, DENSE_INTERMEDIATE)?,
            Glm53SwigluBuffers {
                gate_bf16: buffers.gate_bf16,
                up_bf16: buffers.up_bf16,
                output_bf16: buffers.swiglu_bf16,
            },
            stream,
        )?;
        self.linear(
            gpu,
            &weights.down,
            buffers.swiglu_bf16,
            buffers.q8_activation,
            output,
            stream,
        )?;
        Ok(GLM53_DENSE_FFN_KERNEL_LAUNCHES)
    }
}

#[cfg(test)]
#[path = "dense_ffn_tests.rs"]
mod tests;
