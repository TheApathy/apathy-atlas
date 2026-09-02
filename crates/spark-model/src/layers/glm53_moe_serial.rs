// SPDX-License-Identifier: AGPL-3.0-only

//! Deliberately slow, rows=1 GLM-5.3 MoE bootstrap executor.
//!
//! This is a correctness seam, not an optimized decode path.  One successful
//! call performs one ordered 32-byte D2H copy (and its one mandatory stream
//! sync) plus 64 kernel launches after the router's work. The repository's
//! opt-in per-kernel debug-sync mode adds exactly 64 diagnostic syncs. This
//! path earns no performance credit.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::gguf::GgmlType;

use crate::layers::ops::{
    GgmlIqBuffer, GgmlIqMmqKernels, GgmlIqMmqPlan, Glm53ActivationKernels,
    Glm53ExpertReduceBuffers, Glm53ExpertReducePlan, Glm53SwigluBuffers, Glm53SwigluPlan,
};
use crate::weight_loader::{Glm53GgufMatrix, Glm53GgufMatrixBank, Glm53MoeWeights};

const HIDDEN: u32 = 4096;
const EXPERTS: u32 = 288;
const TOP_K: u32 = 8;
const EXPERT_INTERMEDIATE: u32 = 2048;
const SHARED_INTERMEDIATE: u32 = 2048;

const HIDDEN_BYTES: usize = HIDDEN as usize * 2;
const ROUTE_BYTES: usize = TOP_K as usize * 4;
const EXPERT_INTERMEDIATE_BYTES: usize = EXPERT_INTERMEDIATE as usize * 2;
const SHARED_INTERMEDIATE_BYTES: usize = SHARED_INTERMEDIATE as usize * 2;
const ROUTED_BYTES: usize = TOP_K as usize * HIDDEN_BYTES;
const MAX_Q8_ACTIVATION_BYTES: usize = 4_608;

pub const GLM53_SERIAL_MOE_D2H_COPIES: u32 = 1;
pub const GLM53_SERIAL_MOE_MANDATORY_STREAM_SYNCS: u32 = 1;
pub const GLM53_SERIAL_MOE_KERNEL_LAUNCHES: u32 = 64;

#[derive(Debug, Clone, Copy)]
pub struct Glm53SerialMoeBuffers {
    pub input_bf16: GgmlIqBuffer,
    pub route_ids_u32: GgmlIqBuffer,
    pub route_weights_f32: GgmlIqBuffer,
    pub q8_activation: GgmlIqBuffer,
    pub expert_gate_bf16: GgmlIqBuffer,
    pub expert_up_bf16: GgmlIqBuffer,
    pub expert_swiglu_bf16: GgmlIqBuffer,
    pub routed_bf16: GgmlIqBuffer,
    pub shared_gate_bf16: GgmlIqBuffer,
    pub shared_up_bf16: GgmlIqBuffer,
    pub shared_swiglu_bf16: GgmlIqBuffer,
    pub shared_bf16: GgmlIqBuffer,
    pub output_bf16: GgmlIqBuffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53SerialMoeReceipt {
    pub route_ids: [u32; TOP_K as usize],
    pub d2h_copies: u32,
    pub mandatory_stream_syncs: u32,
    pub kernel_launches: u32,
}

pub struct Glm53SerialMoeKernels {
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

impl Glm53SerialMoeKernels {
    /// Resolve all existing typed IQ-MMQ/activation handles before serving.
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

    /// Execute exactly one token on exactly one stream, without device allocation.
    pub fn execute(
        &self,
        gpu: &dyn GpuBackend,
        weights: &Glm53MoeWeights,
        buffers: Glm53SerialMoeBuffers,
        stream: u64,
    ) -> Result<Glm53SerialMoeReceipt> {
        if gpu.stream_is_capturing(stream) {
            bail!("serial GLM MoE cannot synchronize inside graph capture");
        }
        let plans = self.preflight(weights, buffers)?;

        let mut raw_ids = [0u8; ROUTE_BYTES];
        // The pipeline drain: cuMemcpyDtoHAsync_v2 + cuStreamSynchronize, 42x
        // per token, to move 32 bytes. Timed because it already blocks fully,
        // so a host clock is exact here and cannot make it worse.
        crate::model::glm53::walk_timing::blocked_moe(|| {
            gpu.copy_d2h_on_stream(buffers.route_ids_u32.ptr, &mut raw_ids, stream)
        })?;
        let route_ids = decode_route_ids(raw_ids)?;

        for (slot, expert_id) in route_ids.into_iter().enumerate() {
            let gate = weights.gate_experts.expert(expert_id)?;
            let up = weights.up_experts.expert(expert_id)?;
            let down = weights.down_experts.expert(expert_id)?;
            self.linear(
                gpu,
                &gate,
                plans.expert_gate,
                buffers.input_bf16.ptr,
                buffers.q8_activation,
                buffers.expert_gate_bf16,
                stream,
            )?;
            self.linear(
                gpu,
                &up,
                plans.expert_up,
                buffers.input_bf16.ptr,
                buffers.q8_activation,
                buffers.expert_up_bf16,
                stream,
            )?;
            self.activations.swiglu(
                gpu,
                plans.expert_swiglu,
                Glm53SwigluBuffers {
                    gate_bf16: buffers.expert_gate_bf16,
                    up_bf16: buffers.expert_up_bf16,
                    output_bf16: buffers.expert_swiglu_bf16,
                },
                stream,
            )?;
            let routed_slot = slot_buffer(buffers.routed_bf16, slot)?;
            self.linear(
                gpu,
                &down,
                plans.expert_down,
                buffers.expert_swiglu_bf16.ptr,
                buffers.q8_activation,
                routed_slot,
                stream,
            )?;
        }

        self.linear(
            gpu,
            &weights.shared_gate,
            plans.shared_gate,
            buffers.input_bf16.ptr,
            buffers.q8_activation,
            buffers.shared_gate_bf16,
            stream,
        )?;
        self.linear(
            gpu,
            &weights.shared_up,
            plans.shared_up,
            buffers.input_bf16.ptr,
            buffers.q8_activation,
            buffers.shared_up_bf16,
            stream,
        )?;
        self.activations.swiglu(
            gpu,
            plans.shared_swiglu,
            Glm53SwigluBuffers {
                gate_bf16: buffers.shared_gate_bf16,
                up_bf16: buffers.shared_up_bf16,
                output_bf16: buffers.shared_swiglu_bf16,
            },
            stream,
        )?;
        self.linear(
            gpu,
            &weights.shared_down,
            plans.shared_down,
            buffers.shared_swiglu_bf16.ptr,
            buffers.q8_activation,
            buffers.shared_bf16,
            stream,
        )?;
        self.activations.reduce(
            gpu,
            plans.reduce,
            Glm53ExpertReduceBuffers {
                routed_bf16: buffers.routed_bf16,
                indices_u32: buffers.route_ids_u32,
                weights_f32: buffers.route_weights_f32,
                shared_bf16: buffers.shared_bf16,
                output_bf16: buffers.output_bf16,
            },
            stream,
        )?;
        Ok(Glm53SerialMoeReceipt {
            route_ids,
            d2h_copies: GLM53_SERIAL_MOE_D2H_COPIES,
            mandatory_stream_syncs: GLM53_SERIAL_MOE_MANDATORY_STREAM_SYNCS,
            kernel_launches: GLM53_SERIAL_MOE_KERNEL_LAUNCHES,
        })
    }

    fn linear(
        &self,
        gpu: &dyn GpuBackend,
        matrix: &Glm53GgufMatrix,
        plan: GgmlIqMmqPlan,
        input: DevicePtr,
        q8: GgmlIqBuffer,
        output: GgmlIqBuffer,
        stream: u64,
    ) -> Result<()> {
        self.mmq(matrix.kind())?.launch(
            gpu,
            plan,
            input,
            matrix.buffer(),
            GgmlIqBuffer {
                ptr: q8.ptr,
                bytes: plan.activation_bytes,
            },
            output,
            stream,
        )
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
            _ => bail!("GLM MoE matrix has no typed IQ-MMQ kernel"),
        })
    }

    fn preflight(
        &self,
        weights: &Glm53MoeWeights,
        buffers: Glm53SerialMoeBuffers,
    ) -> Result<SerialPlans> {
        if weights.router.elements() != HIDDEN as usize * EXPERTS as usize
            || weights.expert_bias.elements() != EXPERTS as usize
        {
            bail!("GLM MoE router or bias geometry mismatch");
        }
        let expert_gate = self.bank_plan(&weights.gate_experts, HIDDEN, EXPERT_INTERMEDIATE)?;
        let expert_up = self.bank_plan(&weights.up_experts, HIDDEN, EXPERT_INTERMEDIATE)?;
        let expert_down = self.bank_plan(&weights.down_experts, EXPERT_INTERMEDIATE, HIDDEN)?;
        let shared_gate = self.matrix_plan(&weights.shared_gate, HIDDEN, SHARED_INTERMEDIATE)?;
        let shared_up = self.matrix_plan(&weights.shared_up, HIDDEN, SHARED_INTERMEDIATE)?;
        let shared_down = self.matrix_plan(&weights.shared_down, SHARED_INTERMEDIATE, HIDDEN)?;
        validate_ranges(weights, buffers)?;
        Ok(SerialPlans {
            expert_gate,
            expert_up,
            expert_down,
            shared_gate,
            shared_up,
            shared_down,
            expert_swiglu: Glm53SwigluPlan::new(1, EXPERT_INTERMEDIATE)?,
            shared_swiglu: Glm53SwigluPlan::new(1, SHARED_INTERMEDIATE)?,
            reduce: Glm53ExpertReducePlan::new(1, HIDDEN, EXPERTS, TOP_K)?,
        })
    }

    fn bank_plan(
        &self,
        bank: &Glm53GgufMatrixBank,
        inner: u32,
        columns: u32,
    ) -> Result<GgmlIqMmqPlan> {
        if bank.len() != EXPERTS {
            bail!("GLM MoE matrix bank must contain exactly 288 experts");
        }
        let first = self.matrix_plan(&bank.expert(0)?, inner, columns)?;
        for expert in 1..EXPERTS {
            let next = self.matrix_plan(&bank.expert(expert)?, inner, columns)?;
            if next != first {
                bail!("GLM MoE matrix bank changes kind or geometry");
            }
        }
        Ok(first)
    }

    fn matrix_plan(
        &self,
        matrix: &Glm53GgufMatrix,
        inner: u32,
        columns: u32,
    ) -> Result<GgmlIqMmqPlan> {
        if matrix.inner() != inner || matrix.columns() != columns {
            bail!("GLM MoE matrix geometry mismatch");
        }
        self.mmq(matrix.kind())?;
        matrix.plan(1)
    }
}

#[derive(Debug, Clone, Copy)]
struct SerialPlans {
    expert_gate: GgmlIqMmqPlan,
    expert_up: GgmlIqMmqPlan,
    expert_down: GgmlIqMmqPlan,
    shared_gate: GgmlIqMmqPlan,
    shared_up: GgmlIqMmqPlan,
    shared_down: GgmlIqMmqPlan,
    expert_swiglu: Glm53SwigluPlan,
    shared_swiglu: Glm53SwigluPlan,
    reduce: Glm53ExpertReducePlan,
}

fn decode_route_ids(raw: [u8; ROUTE_BYTES]) -> Result<[u32; TOP_K as usize]> {
    let mut ids = [0u32; TOP_K as usize];
    for slot in 0..TOP_K as usize {
        let at = slot * 4;
        ids[slot] = u32::from_le_bytes(raw[at..at + 4].try_into().expect("four bytes"));
        if ids[slot] >= EXPERTS || ids[..slot].contains(&ids[slot]) {
            bail!("GLM MoE route IDs are duplicate or out of range");
        }
    }
    Ok(ids)
}

fn slot_buffer(routed: GgmlIqBuffer, slot: usize) -> Result<GgmlIqBuffer> {
    let offset = slot
        .checked_mul(HIDDEN_BYTES)
        .context("routed slot offset overflow")?;
    let address = routed
        .ptr
        .0
        .checked_add(u64::try_from(offset)?)
        .context("routed slot address overflow")?;
    Ok(GgmlIqBuffer {
        ptr: DevicePtr(address),
        bytes: HIDDEN_BYTES,
    })
}

fn validate_ranges(weights: &Glm53MoeWeights, b: Glm53SerialMoeBuffers) -> Result<()> {
    let work = [
        ("input", b.input_bf16, HIDDEN_BYTES),
        ("route IDs", b.route_ids_u32, ROUTE_BYTES),
        ("route weights", b.route_weights_f32, ROUTE_BYTES),
        ("Q8 activation", b.q8_activation, MAX_Q8_ACTIVATION_BYTES),
        ("expert gate", b.expert_gate_bf16, EXPERT_INTERMEDIATE_BYTES),
        ("expert up", b.expert_up_bf16, EXPERT_INTERMEDIATE_BYTES),
        (
            "expert SwiGLU",
            b.expert_swiglu_bf16,
            EXPERT_INTERMEDIATE_BYTES,
        ),
        ("routed", b.routed_bf16, ROUTED_BYTES),
        ("shared gate", b.shared_gate_bf16, SHARED_INTERMEDIATE_BYTES),
        ("shared up", b.shared_up_bf16, SHARED_INTERMEDIATE_BYTES),
        (
            "shared SwiGLU",
            b.shared_swiglu_bf16,
            SHARED_INTERMEDIATE_BYTES,
        ),
        ("shared output", b.shared_bf16, HIDDEN_BYTES),
        ("output", b.output_bf16, HIDDEN_BYTES),
    ];
    for &(name, buffer, bytes) in &work {
        if buffer.bytes != bytes || buffer.ptr == DevicePtr::NULL {
            bail!("GLM serial MoE {name} buffer extent or pointer mismatch");
        }
        checked_end(buffer)?;
    }
    let weight_ranges = [
        GgmlIqBuffer {
            ptr: weights.router.ptr(),
            bytes: weights.router.bytes(),
        },
        GgmlIqBuffer {
            ptr: weights.expert_bias.ptr(),
            bytes: weights.expert_bias.bytes(),
        },
        bank_buffer(&weights.gate_experts)?,
        bank_buffer(&weights.up_experts)?,
        bank_buffer(&weights.down_experts)?,
        weights.shared_gate.buffer(),
        weights.shared_up.buffer(),
        weights.shared_down.buffer(),
    ];
    for &buffer in &weight_ranges {
        checked_end(buffer)?;
    }
    for left in 0..work.len() {
        for right in left + 1..work.len() {
            reject_overlap(work[left].1, work[right].1)?;
        }
        for &weight in &weight_ranges {
            reject_overlap(work[left].1, weight)?;
        }
    }
    for left in 0..weight_ranges.len() {
        for right in left + 1..weight_ranges.len() {
            reject_overlap(weight_ranges[left], weight_ranges[right])?;
        }
    }
    Ok(())
}

fn bank_buffer(bank: &Glm53GgufMatrixBank) -> Result<GgmlIqBuffer> {
    let first = bank.expert(0)?.buffer();
    let bytes = first
        .bytes
        .checked_mul(EXPERTS as usize)
        .context("GLM MoE matrix-bank extent overflow")?;
    Ok(GgmlIqBuffer {
        ptr: first.ptr,
        bytes,
    })
}

fn checked_end(buffer: GgmlIqBuffer) -> Result<u64> {
    buffer
        .ptr
        .0
        .checked_add(u64::try_from(buffer.bytes)?)
        .context("GLM serial MoE device address overflow")
}

fn reject_overlap(left: GgmlIqBuffer, right: GgmlIqBuffer) -> Result<()> {
    if left.ptr.0 < checked_end(right)? && right.ptr.0 < checked_end(left)? {
        bail!("GLM serial MoE device ranges overlap");
    }
    Ok(())
}

#[cfg(test)]
#[path = "glm53_moe_serial_tests.rs"]
mod tests;
