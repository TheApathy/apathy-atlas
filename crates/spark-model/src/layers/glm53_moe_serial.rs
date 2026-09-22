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
    GLM53_EXL3_MAX_WIDE_ROWS, GgmlIqBuffer, GgmlIqMmqKernels, GgmlIqMmqPlan,
    Glm53ActivationKernels, Glm53Exl3Buffer, Glm53Exl3CastKernels, Glm53Exl3CastPlan,
    Glm53Exl3MoeBuffers, Glm53Exl3MoeKernels, Glm53Exl3MoePointerTables, Glm53Exl3MoeScratch,
    Glm53Exl3Projection, Glm53Exl3ProjectionBuffers, Glm53Exl3ProjectionScratch,
    Glm53Exl3RoutePolicy, Glm53ExpertReduceBuffers, Glm53ExpertReducePlan, Glm53SwigluBuffers,
    Glm53SwigluPlan,
};
use crate::weight_loader::{
    Glm53Exl3ExpertWeights, Glm53Exl3MoeWeights, Glm53GgufMatrix, Glm53GgufMatrixBank,
    Glm53MoeWeights,
};

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
pub const GLM53_EXL3_SERIAL_MOE_KERNEL_LAUNCHES: u32 = 91;

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

/// The four buffers the grouped path needs that the serial seam does not:
/// TOP_K-wide gate, up and swiglu, plus one quantized activation block per
/// (row, slot) pair for the down projection.
#[derive(Debug, Clone, Copy)]
pub struct Glm53GroupedMoeScratch {
    pub gate_bf16: GgmlIqBuffer,
    pub up_bf16: GgmlIqBuffer,
    pub swiglu_bf16: GgmlIqBuffer,
    pub down_q8: GgmlIqBuffer,
}

/// What the grouped path cost. `d2h_copies` and `mandatory_stream_syncs` are
/// ZERO by construction and that is the whole point: the serial seam's are 1
/// and 1 per MoE layer, 42 per token, 3.91 ms each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53GroupedMoeReceipt {
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
    exl3_moe: Option<Glm53Exl3MoeKernels>,
    exl3_cast: Option<Glm53Exl3CastKernels>,
    /// The grouped kernels, loaded ALONGSIDE the serial ones rather than
    /// instead of them. One struct, two methods, so the A/B is on one binary
    /// and a regression localises in a single run.
    grouped: Vec<(GgmlType, super::glm53_moe_grouped::Glm53GroupedMoeKernels)>,
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
            exl3_moe: None,
            exl3_cast: None,
            // Only the eleven types the grouped kernel is DEFINED for. Q3_K has
            // no grouped definition, so a Q3_K bank falls to the serial path by
            // failing to resolve here rather than by resolving a symbol that was
            // never emitted.
            grouped: [
                GgmlType::Q2_K,
                GgmlType::Q4_K,
                GgmlType::Q5_K,
                GgmlType::Q6_K,
                GgmlType::Q8_0,
                GgmlType::IQ2_XXS,
                GgmlType::IQ2_XS,
                GgmlType::IQ2_S,
                GgmlType::IQ3_XXS,
                GgmlType::IQ3_S,
                GgmlType::IQ4_XS,
            ]
            .into_iter()
            .map(|kind| {
                super::glm53_moe_grouped::Glm53GroupedMoeKernels::load(gpu, kind)
                    .map(|kernels| (kind, kernels))
            })
            .collect::<Result<Vec<_>>>()?,
        })
    }

    pub fn load_exl3(gpu: &dyn GpuBackend) -> Result<Self> {
        let policy = Glm53Exl3RoutePolicy::parse(
            std::env::var_os("ATLAS_GLM53_EXL3_ROUTE_PRIVATE").as_deref(),
            std::env::var_os("ATLAS_GLM53_EXL3_ROUTE_PRIVATE_PREFILL").as_deref(),
        )?;
        policy.validate_moe_mode(std::env::var_os("ATLAS_GLM53_EXL3_MOE").as_deref())?;
        Self::load_exl3_with_route_policy(gpu, policy)
    }

    pub fn load_exl3_with_route_policy(
        gpu: &dyn GpuBackend,
        policy: Glm53Exl3RoutePolicy,
    ) -> Result<Self> {
        let mut kernels = Self::load(gpu)?;
        kernels.exl3_moe = Some(Glm53Exl3MoeKernels::load_with_route_policy(gpu, policy)?);
        kernels.exl3_cast = Some(Glm53Exl3CastKernels::load(gpu)?);
        Ok(kernels)
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

        self.shared_and_reduce(gpu, weights, buffers, plans, stream)?;
        Ok(Glm53SerialMoeReceipt {
            route_ids,
            d2h_copies: GLM53_SERIAL_MOE_D2H_COPIES,
            mandatory_stream_syncs: GLM53_SERIAL_MOE_MANDATORY_STREAM_SYNCS,
            kernel_launches: GLM53_SERIAL_MOE_KERNEL_LAUNCHES,
        })
    }

    /// Correctness-first EXL3 MoE path. Route ids are still read once to the
    /// host per layer; the later grouped EXL3 lane removes that decode drain.
    pub fn execute_exl3(
        &self,
        gpu: &dyn GpuBackend,
        weights: &Glm53Exl3MoeWeights,
        buffers: Glm53SerialMoeBuffers,
        scratch: Glm53Exl3ProjectionScratch,
        stream: u64,
    ) -> Result<Glm53SerialMoeReceipt> {
        if gpu.stream_is_capturing(stream) {
            bail!("serial GLM EXL3 MoE cannot synchronize inside graph capture");
        }
        self.preflight_exl3(weights, buffers)?;
        let mut raw_ids = [0u8; ROUTE_BYTES];
        crate::model::glm53::walk_timing::blocked_moe(|| {
            gpu.copy_d2h_on_stream(buffers.route_ids_u32.ptr, &mut raw_ids, stream)
        })?;
        let route_ids = decode_route_ids(raw_ids)?;

        for (slot, expert_id) in route_ids.into_iter().enumerate() {
            let expert = weights
                .experts
                .get(expert_id as usize)
                .context("GLM EXL3 routed expert missing")?;
            self.exl3_expert(
                gpu,
                expert,
                buffers.input_bf16,
                buffers.expert_gate_bf16,
                buffers.expert_up_bf16,
                buffers.expert_swiglu_bf16,
                slot_buffer(buffers.routed_bf16, slot)?,
                scratch,
                stream,
            )?;
        }
        self.exl3_expert(
            gpu,
            &weights.shared,
            buffers.input_bf16,
            buffers.shared_gate_bf16,
            buffers.shared_up_bf16,
            buffers.shared_swiglu_bf16,
            buffers.shared_bf16,
            scratch,
            stream,
        )?;
        self.activations.reduce(
            gpu,
            Glm53ExpertReducePlan::new(1, HIDDEN, EXPERTS, TOP_K)?,
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
            kernel_launches: GLM53_EXL3_SERIAL_MOE_KERNEL_LAUNCHES,
        })
    }

    /// Device-routed EXL3 path used by the target and the K8 verifier. No route
    /// bytes cross to the host and no stream synchronization occurs here.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_exl3_fused(
        &self,
        gpu: &dyn GpuBackend,
        weights: &Glm53Exl3MoeWeights,
        buffers: Glm53SerialMoeBuffers,
        projection_scratch: Glm53Exl3ProjectionScratch,
        moe_scratch: Glm53Exl3MoeScratch,
        pointers: Glm53Exl3MoePointerTables,
        stream: u64,
    ) -> Result<Glm53GroupedMoeReceipt> {
        self.execute_exl3_fused_rows(
            gpu,
            1,
            weights,
            buffers,
            projection_scratch,
            moe_scratch,
            pointers,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_exl3_fused_rows(
        &self,
        gpu: &dyn GpuBackend,
        rows: u32,
        weights: &Glm53Exl3MoeWeights,
        buffers: Glm53SerialMoeBuffers,
        projection_scratch: Glm53Exl3ProjectionScratch,
        moe_scratch: Glm53Exl3MoeScratch,
        pointers: Glm53Exl3MoePointerTables,
        stream: u64,
    ) -> Result<Glm53GroupedMoeReceipt> {
        self.preflight_exl3_rows(weights, buffers, rows)?;
        let fused = self
            .exl3_moe
            .as_ref()
            .context("GLM EXL3 fused MoE kernels were not loaded")?;
        let plan = fused.plan(rows)?;
        let casts = self
            .exl3_cast
            .as_ref()
            .context("GLM EXL3 cast kernels were not loaded")?;
        let input_f16 = prefix(projection_scratch.input_f16, plan.input_f16_bytes)?;
        let exact = Glm53Exl3MoeBuffers {
            input_f16,
            output_f32: prefix(moe_scratch.output_f32, plan.output_f32_bytes)?,
            route_private_f32: prefix(moe_scratch.route_private_f32, plan.route_private_f32_bytes)?,
            route_ids_u32: exl3_buffer(buffers.route_ids_u32),
            route_weights_f32: exl3_buffer(buffers.route_weights_f32),
            expert_count_i64: prefix(moe_scratch.expert_count_i64, plan.expert_count_i64_bytes)?,
            token_sorted_i64: prefix(moe_scratch.token_sorted_i64, plan.token_sorted_i64_bytes)?,
            weight_sorted_f16: prefix(moe_scratch.weight_sorted_f16, plan.weight_sorted_f16_bytes)?,
            temp_state_g_f16: prefix(moe_scratch.temp_state_g_f16, plan.temp_state_f16_bytes)?,
            temp_state_u_f16: prefix(moe_scratch.temp_state_u_f16, plan.temp_state_f16_bytes)?,
            temp_intermediate_g_f16: prefix(
                moe_scratch.temp_intermediate_g_f16,
                plan.temp_intermediate_f16_bytes,
            )?,
            temp_intermediate_u_f16: prefix(
                moe_scratch.temp_intermediate_u_f16,
                plan.temp_intermediate_f16_bytes,
            )?,
            route_status_u32: moe_scratch.route_status_u32,
            locks_i32: moe_scratch.locks_i32,
            pair_expert_u32: prefix(moe_scratch.pair_expert_u32, plan.pair_expert_u32_bytes)?,
            chunk_expert_u32: prefix(
                moe_scratch.chunk_expert_u32,
                plan.chunk_descriptor_u32_bytes,
            )?,
            chunk_start_u32: prefix(moe_scratch.chunk_start_u32, plan.chunk_descriptor_u32_bytes)?,
            chunk_rows_u32: prefix(moe_scratch.chunk_rows_u32, plan.chunk_descriptor_u32_bytes)?,
            chunk_count_u32: prefix(moe_scratch.chunk_count_u32, plan.chunk_count_u32_bytes)?,
            pointers,
        };
        // All selected extents and pointer tables must be admitted before
        // even the first input cast submits work to the caller's stream.
        fused.validate(plan, exact)?;
        casts.bf16_to_f16(
            gpu,
            Glm53Exl3CastPlan::new(rows, HIDDEN)?,
            exl3_buffer(buffers.input_bf16),
            input_f16,
            stream,
        )?;
        let fused_kernel_launches = fused.kernel_launches(rows);
        // Kernel oracle: the staged MoE for one layer. The routing tables are
        // captured BEFORE the launch (pack_routes rewrites them in place) and
        // the staged/output tensors after it. NOTE: layers 3..=44 run inside a
        // CUDA graph, where a D2H copy is illegal -- capture needs
        // ATLAS_GLM53_FFN_GRAPHS=0.
        crate::model::glm53::oracle_dump::dump_site(
            gpu,
            stream,
            "moe_in",
            &[
                crate::model::glm53::oracle_dump::t(
                    "input_bf16", buffers.input_bf16.ptr,
                    rows as usize * HIDDEN as usize * 2, "bf16",
                    &[rows as usize, HIDDEN as usize]),
                crate::model::glm53::oracle_dump::t(
                    "input_f16", input_f16.ptr, plan.input_f16_bytes, "f16",
                    &[rows as usize, HIDDEN as usize]),
                crate::model::glm53::oracle_dump::t(
                    "route_ids", buffers.route_ids_u32.ptr,
                    plan.route_ids_u32_bytes, "u32", &[rows as usize, 8]),
                crate::model::glm53::oracle_dump::t(
                    "route_weights", buffers.route_weights_f32.ptr,
                    plan.route_weights_f32_bytes, "f32", &[rows as usize, 8]),
            ],
            &[
                ("rows", rows.to_string()),
                ("hidden", HIDDEN.to_string()),
                ("intermediate", "2048".into()),
                ("experts", "288".into()),
                ("top_k", "8".into()),
                ("bits", "2".into()),
                ("codebook", "2".into()),
            ],
        )?;
        fused.launch(gpu, plan, exact, stream)?;
        crate::model::glm53::oracle_dump::dump_site(
            gpu,
            stream,
            "moe_out",
            &[
                crate::model::glm53::oracle_dump::t(
                    // 289, not 288: the pack kernel writes an exclusive prefix
                    // sum with a trailing total (plan.expert_count_i64_bytes = 2312).
                    "expert_count", exact.expert_count_i64.ptr,
                    plan.expert_count_i64_bytes, "i64", &[289]),
                crate::model::glm53::oracle_dump::t(
                    "token_sorted", exact.token_sorted_i64.ptr,
                    plan.token_sorted_i64_bytes, "i64", &[rows as usize * 8]),
                crate::model::glm53::oracle_dump::t(
                    "weight_sorted", exact.weight_sorted_f16.ptr,
                    plan.weight_sorted_f16_bytes, "f16", &[rows as usize * 8]),
                crate::model::glm53::oracle_dump::t(
                    "output_f32", exact.output_f32.ptr,
                    plan.output_f32_bytes, "f32",
                    &[rows as usize, HIDDEN as usize]),
            ],
            &[("rows", rows.to_string())],
        )?;
        self.exl3_expert_rows(
            gpu,
            rows,
            &weights.shared,
            buffers.input_bf16,
            buffers.shared_gate_bf16,
            buffers.shared_up_bf16,
            buffers.shared_swiglu_bf16,
            buffers.shared_bf16,
            projection_scratch,
            stream,
        )?;
        fused.combine_shared(
            gpu,
            plan,
            exact.output_f32,
            exact.route_private_f32,
            exl3_buffer(buffers.shared_bf16),
            exl3_buffer(buffers.output_bf16),
            exact.route_status_u32,
            stream,
        )?;
        Ok(Glm53GroupedMoeReceipt {
            d2h_copies: 0,
            mandatory_stream_syncs: 0,
            // Twelve launches surround the fused routed MoE: one input cast,
            // three shared projections times three launches, SwiGLU, combine.
            kernel_launches: 12 + fused_kernel_launches,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn exl3_expert(
        &self,
        gpu: &dyn GpuBackend,
        weights: &Glm53Exl3ExpertWeights,
        input: GgmlIqBuffer,
        gate: GgmlIqBuffer,
        up: GgmlIqBuffer,
        swiglu: GgmlIqBuffer,
        output: GgmlIqBuffer,
        scratch: Glm53Exl3ProjectionScratch,
        stream: u64,
    ) -> Result<()> {
        self.exl3_expert_rows(
            gpu, 1, weights, input, gate, up, swiglu, output, scratch, stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn exl3_expert_rows(
        &self,
        gpu: &dyn GpuBackend,
        rows: u32,
        weights: &Glm53Exl3ExpertWeights,
        input: GgmlIqBuffer,
        gate: GgmlIqBuffer,
        up: GgmlIqBuffer,
        swiglu: GgmlIqBuffer,
        output: GgmlIqBuffer,
        scratch: Glm53Exl3ProjectionScratch,
        stream: u64,
    ) -> Result<()> {
        self.exl3_linear_rows(gpu, rows, &weights.gate, input, gate, scratch, stream)?;
        self.exl3_linear_rows(gpu, rows, &weights.up, input, up, scratch, stream)?;
        self.activations.swiglu(
            gpu,
            Glm53SwigluPlan::new(rows, EXPERT_INTERMEDIATE)?,
            Glm53SwigluBuffers {
                gate_bf16: gate,
                up_bf16: up,
                output_bf16: swiglu,
            },
            stream,
        )?;
        self.exl3_linear_rows(gpu, rows, &weights.down, swiglu, output, scratch, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn exl3_linear_rows(
        &self,
        gpu: &dyn GpuBackend,
        rows: u32,
        linear: &crate::weight_loader::Glm53Exl3Linear,
        input: GgmlIqBuffer,
        output: GgmlIqBuffer,
        scratch: Glm53Exl3ProjectionScratch,
        stream: u64,
    ) -> Result<()> {
        let projection = Glm53Exl3Projection::Compressed(linear);
        let plan = projection.plan(rows)?;
        projection.launch(
            gpu,
            plan,
            Glm53Exl3ProjectionBuffers {
                input_bf16: Glm53Exl3Buffer {
                    ptr: input.ptr,
                    bytes: input.bytes,
                },
                output_bf16: Glm53Exl3Buffer {
                    ptr: output.ptr,
                    bytes: output.bytes,
                },
                scratch,
            },
            stream,
        )
    }

    /// The shared expert and the weighted reduce.
    ///
    /// Extracted so the serial and grouped paths CALL the same code rather than
    /// each carrying a copy. Neither step ever depended on a route id, so this
    /// is the part of the MoE that is identical between them by construction
    /// instead of by inspection -- which is what makes a bit-exactness failure
    /// point at the expert matmuls rather than anywhere in the layer.
    fn shared_and_reduce(
        &self,
        gpu: &dyn GpuBackend,
        weights: &Glm53MoeWeights,
        buffers: Glm53SerialMoeBuffers,
        plans: SerialPlans,
        stream: u64,
    ) -> Result<u32> {
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
        Ok(0)
    }

    /// The GROUPED path: the same MoE, with no host round trip.
    ///
    /// The serial seam above reads the eight route ids to the HOST
    /// (cuMemcpyDtoHAsync_v2 + cuStreamSynchronize, a full pipeline drain) to
    /// compute `base + id*expert_bytes` on the CPU, 42 times per token at
    /// 3.91 ms each -- 91.5% of decode wall time, to move 32 bytes. Here the
    /// route ids stay on the device and the kernel does the multiply-add
    /// itself, so nothing has to synchronize to learn them.
    ///
    /// All THREE expert matmuls are grouped. Down is the one that matters: if
    /// it stayed serial the host would still need the ids to address the bank
    /// and the drain would survive intact, so two of three is not two thirds of
    /// the win, it is none of it. Gate and up share one activation per row; down
    /// takes one per (row, slot) pair, which is why the plan carries an
    /// activation mode.
    ///
    /// The shared expert and the reduce are UNCHANGED from the serial path --
    /// they never depended on a route id.
    pub fn execute_grouped(
        &self,
        gpu: &dyn GpuBackend,
        weights: &Glm53MoeWeights,
        buffers: Glm53SerialMoeBuffers,
        grouped: Glm53GroupedMoeScratch,
        stream: u64,
    ) -> Result<Glm53GroupedMoeReceipt> {
        use super::glm53_moe_grouped::{
            Glm53GroupedActivations, Glm53GroupedBank, Glm53GroupedMoePlan,
        };

        let plans = self.preflight(weights, buffers)?;
        let gate_bank = Glm53GroupedBank::from_bank(&weights.gate_experts)?;
        let up_bank = Glm53GroupedBank::from_bank(&weights.up_experts)?;
        let down_bank = Glm53GroupedBank::from_bank(&weights.down_experts)?;

        let mut launches = 0u32;
        // Gate and up: one activation per row, shared by all top_k experts.
        for (bank, output) in [(gate_bank, grouped.gate_bf16), (up_bank, grouped.up_bf16)] {
            let plan = Glm53GroupedMoePlan::new(bank, 1, TOP_K)?;
            self.grouped_kernels(bank.kind)?.launch(
                gpu,
                plan,
                buffers.input_bf16.ptr,
                buffers.route_ids_u32,
                GgmlIqBuffer {
                    ptr: buffers.q8_activation.ptr,
                    bytes: plan.activation_bytes,
                },
                GgmlIqBuffer {
                    ptr: output.ptr,
                    bytes: plan.output_bytes,
                },
                stream,
            )?;
            launches += 2;
        }

        // One SwiGLU launch for all TOP_K slots; the plan already takes rows.
        self.activations.swiglu(
            gpu,
            Glm53SwigluPlan::new(TOP_K, EXPERT_INTERMEDIATE)?,
            Glm53SwigluBuffers {
                gate_bf16: grouped.gate_bf16,
                up_bf16: grouped.up_bf16,
                output_bf16: grouped.swiglu_bf16,
            },
            stream,
        )?;
        launches += 1;

        // Down: each slot consumes its OWN swiglu row, so the quantize runs at
        // rows = pairs and the tiles stride by pair. Two plans over one buffer,
        // which `with_activations` has already checked describe the same bytes.
        let down = Glm53GroupedMoePlan::with_activations(
            down_bank,
            1,
            TOP_K,
            Glm53GroupedActivations::PerPair,
        )?;
        let down_kernels = self.grouped_kernels(down_bank.kind)?;
        // ONE rows=1 QUANTIZE PER PAIR, not one multi-row quantize sliced up.
        // The MMQ q8_1 layout for rows > 1 is tiled, so row p does not live at
        // p * per_row_bytes; slicing it reads scrambled data, which is what made
        // `routed` differ in 99.92% of its elements from index 0. Each block is
        // quantized independently into its own region, which is byte-for-byte
        // what the serial path produces.
        let quantize_plan = down.quantize_plan()?;
        let row_bytes = u64::from(EXPERT_INTERMEDIATE) * 2;
        for block in 0..down.activation_blocks {
            down_kernels.quantize(
                gpu,
                quantize_plan,
                DevicePtr(
                    grouped
                        .swiglu_bf16
                        .ptr
                        .0
                        .checked_add(u64::from(block) * row_bytes)
                        .context("GLM grouped MoE swiglu row overflow")?,
                ),
                GgmlIqBuffer {
                    ptr: DevicePtr(
                        grouped
                            .down_q8
                            .ptr
                            .0
                            .checked_add(down.activation_block_offset(block)?)
                            .context("GLM grouped MoE activation block overflow")?,
                    ),
                    bytes: quantize_plan.activation_bytes,
                },
                stream,
            )?;
        }
        launches += down.activation_blocks;
        down_kernels.launch_tiles(
            gpu,
            down,
            buffers.route_ids_u32,
            GgmlIqBuffer {
                ptr: grouped.down_q8.ptr,
                bytes: down.activation_bytes,
            },
            GgmlIqBuffer {
                ptr: buffers.routed_bf16.ptr,
                bytes: down.output_bytes,
            },
            stream,
        )?;
        launches += 2;

        // Shared expert and reduce: byte-identical to the serial path.
        launches += self.shared_and_reduce(gpu, weights, buffers, plans, stream)?;

        Ok(Glm53GroupedMoeReceipt {
            d2h_copies: 0,
            mandatory_stream_syncs: 0,
            kernel_launches: launches,
        })
    }

    fn grouped_kernels(
        &self,
        kind: GgmlType,
    ) -> Result<&super::glm53_moe_grouped::Glm53GroupedMoeKernels> {
        self.grouped
            .iter()
            .find(|(loaded, _)| *loaded == kind)
            .map(|(_, kernels)| kernels)
            .with_context(|| {
                format!(
                    "GLM grouped MoE has no kernel for {kind:?}; run this bank on \
                     ATLAS_GLM53_MOE=serial-reference"
                )
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

    fn preflight_exl3(
        &self,
        weights: &Glm53Exl3MoeWeights,
        buffers: Glm53SerialMoeBuffers,
    ) -> Result<()> {
        self.preflight_exl3_rows(weights, buffers, 1)
    }

    fn preflight_exl3_rows(
        &self,
        weights: &Glm53Exl3MoeWeights,
        buffers: Glm53SerialMoeBuffers,
        rows: u32,
    ) -> Result<()> {
        if weights.experts.len() != EXPERTS as usize
            || weights.router.bytes() != HIDDEN as usize * EXPERTS as usize * 4
            || weights.expert_bias.bytes() != EXPERTS as usize * 4
        {
            bail!("GLM EXL3 MoE router, bias, or expert census mismatch");
        }
        for expert in weights
            .experts
            .iter()
            .chain(std::iter::once(&weights.shared))
        {
            for (linear, inner, columns) in [
                (&expert.gate, HIDDEN, EXPERT_INTERMEDIATE),
                (&expert.up, HIDDEN, EXPERT_INTERMEDIATE),
                (&expert.down, EXPERT_INTERMEDIATE, HIDDEN),
            ] {
                if linear.size_k() != inner || linear.size_n() != columns {
                    bail!("GLM EXL3 MoE expert projection geometry mismatch");
                }
            }
        }
        if rows == 1 {
            validate_work_ranges(buffers)
        } else {
            validate_exl3_fused_work_ranges(buffers, rows)
        }
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

fn exl3_buffer(buffer: GgmlIqBuffer) -> Glm53Exl3Buffer {
    Glm53Exl3Buffer {
        ptr: buffer.ptr,
        bytes: buffer.bytes,
    }
}

fn prefix(buffer: Glm53Exl3Buffer, bytes: usize) -> Result<Glm53Exl3Buffer> {
    if buffer.ptr == DevicePtr::NULL || buffer.bytes < bytes {
        bail!("GLM EXL3 fused MoE scratch is null or too small");
    }
    Ok(Glm53Exl3Buffer {
        ptr: buffer.ptr,
        bytes,
    })
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
    let work = work_ranges(b);
    validate_work_ranges(b)?;
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

fn work_ranges(b: Glm53SerialMoeBuffers) -> [(&'static str, GgmlIqBuffer, usize); 13] {
    [
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
    ]
}

fn validate_work_ranges(b: Glm53SerialMoeBuffers) -> Result<()> {
    let work = work_ranges(b);
    for &(name, buffer, bytes) in &work {
        if buffer.bytes != bytes || buffer.ptr == DevicePtr::NULL {
            bail!("GLM serial MoE {name} buffer extent or pointer mismatch");
        }
        checked_end(buffer)?;
    }
    for left in 0..work.len() {
        for right in left + 1..work.len() {
            reject_overlap(work[left].1, work[right].1)?;
        }
    }
    Ok(())
}

fn validate_exl3_fused_work_ranges(b: Glm53SerialMoeBuffers, rows: u32) -> Result<()> {
    if !(2..=GLM53_EXL3_MAX_WIDE_ROWS as u32).contains(&rows) {
        bail!("GLM fused wide MoE rows must be 2..={GLM53_EXL3_MAX_WIDE_ROWS}");
    }
    let rows = usize::try_from(rows)?;
    let work = [
        ("input", b.input_bf16, rows * HIDDEN_BYTES),
        ("route IDs", b.route_ids_u32, rows * ROUTE_BYTES),
        ("route weights", b.route_weights_f32, rows * ROUTE_BYTES),
        (
            "shared gate",
            b.shared_gate_bf16,
            rows * SHARED_INTERMEDIATE_BYTES,
        ),
        (
            "shared up",
            b.shared_up_bf16,
            rows * SHARED_INTERMEDIATE_BYTES,
        ),
        (
            "shared SwiGLU",
            b.shared_swiglu_bf16,
            rows * SHARED_INTERMEDIATE_BYTES,
        ),
        ("shared output", b.shared_bf16, rows * HIDDEN_BYTES),
        ("output", b.output_bf16, rows * HIDDEN_BYTES),
    ];
    for &(name, buffer, bytes) in &work {
        if buffer.bytes != bytes || buffer.ptr == DevicePtr::NULL {
            bail!("GLM fused wide MoE {name} buffer extent or pointer mismatch");
        }
        checked_end(buffer)?;
    }
    for left in 0..work.len() {
        for right in left + 1..work.len() {
            reject_overlap(work[left].1, work[right].1)?;
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
