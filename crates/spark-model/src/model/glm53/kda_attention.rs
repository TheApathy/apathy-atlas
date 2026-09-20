// SPDX-License-Identifier: AGPL-3.0-only

//! KDA attention for the 34 gated-delta-net layers.
//!
//! Transcribed from the reference graph's `build_kda_layer`
//! (`llama.cpp` `src/models/glm5next.cpp`), which is the port oracle:
//!
//! ```text
//! q,k,v  = causal_conv1d(W_{q,k,v} x, ssm_{q,k,v}_conv)
//! g      = gate_lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))
//! beta   = sigmoid(W_beta x)
//! q,k    = l2_norm(q), l2_norm(k)              (inside the recurrence kernel)
//! out    = recurrent_delta_net(q, k, v, g, beta, state)
//! o_gate = g_b(g_a(x))
//! out    = rms_norm(out, ssm_norm) * sigmoid(o_gate)      (SIGMOID, not SiLU)
//! cur    = W_o out
//! ```
//!
//! # Staging only
//!
//! This runs the **staging** half of a KDA layer. The recurrence mutates its
//! state buffer in place, so it is pointed at [`Glm53KdaScratchState`] — never
//! at persistent state — and the scratch is primed from persistent first.
//! Persistent state advances only later, through `kda_recurrent_commit` on
//! `accepted == 1`. The short convolution has its own stage/commit protocol and
//! is likewise only staged here.
//!
//! # Cost
//!
//! Nine matmuls (each a quantize plus a matmul), the two-kernel conv stage, and
//! four elementwise ops. The 4 MiB prime copy per layer dominates: 136 MiB of
//! device-to-device traffic per token across all 34 layers. That is the price of
//! never letting a rejected step touch persistent state; see
//! [`Glm53KdaScratchState::prime`].

use crate::model::glm53::oracle_dump::t;
use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::gguf::GgmlType;

use crate::layers::ops::{
    GLM53_EXL3_MAX_WIDE_ROWS, GgmlIqBuffer, GgmlIqMmqKernels, Glm53Exl3Buffer, Glm53Exl3Projection,
    Glm53Exl3ProjectionBuffers, Glm53Exl3ProjectionScratch, Glm53KdaBetaBuffers, Glm53KdaBetaPlan,
    Glm53KdaConvBuffers, Glm53KdaConvKernel, Glm53KdaConvPlan, Glm53KdaDecodeBuffers,
    Glm53KdaDecodeKernel, Glm53KdaDecodePlan, Glm53KdaForgetBuffers, Glm53KdaForgetPlan,
    Glm53KdaKernels, Glm53KdaNormBuffers, Glm53KdaNormPlan, Glm53KdaPrefillBuffers,
    Glm53KdaPrefillKernel, Glm53KdaPrefillPlan, Glm53KdaQkvBuffers,
    glm53_exact_wide_prefill_active, glm53_layer_major_prefill_active,
};
use crate::weight_loader::{
    Glm53Exl3KdaWeights, Glm53Exl3NativeDtype, Glm53GgufMatrix, Glm53KdaWeights,
};

use super::kda_state_binding::Glm53KdaScratchState;
use super::walk_scratch::Glm53KdaScratchBuffers;

const HEADS: u32 = 64;
const HEAD_DIM: u32 = 128;
const QKV_DIM: u32 = HEADS * HEAD_DIM;
const HIDDEN: u32 = 4_096;
const LOW_RANK: u32 = 128;
const CONV_KERNEL: u32 = 4;

/// Nine matmuls at two kernels each, the two-kernel conv stage, and the forget
/// gate, beta, recurrence and gated norm.
pub const GLM53_KDA_KERNEL_LAUNCHES: u32 = 9 * 2 + 2 + 4;

/// Combined EXL3 QKV and output each use three launches, the five native BF16
/// projections use cuBLASLt once each, followed by conv and four elementwise
/// stages.
pub const GLM53_EXL3_KDA_KERNEL_LAUNCHES: u32 = 2 * 3 + 5 + 2 + 4;

/// The transactional convolution state for one layer.
///
/// Both halves are supplied: the op reads persistent, writes staged, and
/// publishes a receipt. Unlike the recurrence, the conv kernel never mutates
/// persistent state, so there is no in-place hazard to guard against here.
#[derive(Debug, Clone, Copy)]
pub struct Glm53KdaConvSlots {
    pub persistent_state_f32: GgmlIqBuffer,
    pub staged_state_f32: GgmlIqBuffer,
    pub published_ends_u32: GgmlIqBuffer,
    pub published_nonces_u64: GgmlIqBuffer,
    pub logical_lengths_u32: GgmlIqBuffer,
}

/// Typed handles for one KDA layer.
pub struct Glm53KdaAttentionKernels {
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
    kda: Glm53KdaKernels,
    conv: Glm53KdaConvKernel,
    decode: Glm53KdaDecodeKernel,
    prefill: Glm53KdaPrefillKernel,
}

impl Glm53KdaAttentionKernels {
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
            kda: Glm53KdaKernels::load(gpu)?,
            conv: Glm53KdaConvKernel::load(gpu)?,
            decode: Glm53KdaDecodeKernel::load(gpu)?,
            prefill: Glm53KdaPrefillKernel::load(gpu)?,
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
            _ => bail!("GLM KDA matrix has no typed IQ-MMQ kernel"),
        })
    }

    fn linear(
        &self,
        gpu: &dyn GpuBackend,
        matrix: &Glm53GgufMatrix,
        input: GgmlIqBuffer,
        q8: GgmlIqBuffer,
        output: GgmlIqBuffer,
        stream: u64,
    ) -> Result<()> {
        let plan = matrix.plan(1)?;
        ensure!(
            q8.bytes >= plan.activation_bytes,
            "GLM KDA Q8 scratch is {} bytes, needs {}",
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
        )
    }

    fn f32_buffer(ptr: DevicePtr, bytes: usize) -> GgmlIqBuffer {
        GgmlIqBuffer { ptr, bytes }
    }

    fn linear_exl3(
        &self,
        gpu: &dyn GpuBackend,
        projection: Glm53Exl3Projection<'_>,
        input: GgmlIqBuffer,
        output: GgmlIqBuffer,
        scratch: Glm53Exl3ProjectionScratch,
        stream: u64,
    ) -> Result<()> {
        self.linear_exl3_rows(gpu, 1, projection, input, output, scratch, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_exl3_rows(
        &self,
        gpu: &dyn GpuBackend,
        rows: u32,
        projection: Glm53Exl3Projection<'_>,
        input: GgmlIqBuffer,
        output: GgmlIqBuffer,
        scratch: Glm53Exl3ProjectionScratch,
        stream: u64,
    ) -> Result<()> {
        let plan = projection.plan(rows)?;
        projection.launch(
            gpu,
            plan,
            Glm53Exl3ProjectionBuffers {
                input_bf16: exl3_buffer(input),
                output_bf16: exl3_buffer(output),
                scratch,
            },
            stream,
        )
    }

    /// Stage one KDA layer.
    ///
    /// `position` is the token's absolute position and `capacity` the sequence
    /// capacity; together they define the convolution's transactional extent.
    /// `nonce` must be the walk's transaction nonce and non-zero.
    #[allow(clippy::too_many_arguments)]
    pub fn stage(
        &self,
        gpu: &dyn GpuBackend,
        weights: &Glm53KdaWeights,
        input: GgmlIqBuffer,
        buffers: Glm53KdaScratchBuffers,
        state: &Glm53KdaScratchState,
        conv: Glm53KdaConvSlots,
        position: u32,
        capacity: u32,
        nonce: u64,
        output: GgmlIqBuffer,
        stream: u64,
    ) -> Result<u32> {
        self.validate_shapes(weights)?;

        // The recurrence mutates its state in place, so scratch must start as a
        // copy of the last committed state.
        state.prime(gpu, stream)?;

        // q, k, v projections, then the transactional short convolution.
        for (matrix, destination) in [
            (&weights.q, buffers.q_proj_bf16),
            (&weights.k, buffers.k_proj_bf16),
            (&weights.v, buffers.v_proj_bf16),
        ] {
            self.linear(
                gpu,
                matrix,
                input,
                buffers.q8_activation,
                destination,
                stream,
            )?;
        }

        let conv_plan = Glm53KdaConvPlan::new(
            1,
            1,
            capacity,
            position,
            position
                .checked_add(1)
                .context("GLM KDA conv position overflow")?,
            HEADS,
            HEAD_DIM,
            CONV_KERNEL,
            nonce,
        )?;
        self.conv.launch_stage(
            gpu,
            conv_plan,
            Glm53KdaConvBuffers {
                q_input_bf16: buffers.q_proj_bf16,
                k_input_bf16: buffers.k_proj_bf16,
                v_input_bf16: buffers.v_proj_bf16,
                q_weight_f32: Self::f32_buffer(weights.conv_q.ptr(), weights.conv_q.bytes()),
                k_weight_f32: Self::f32_buffer(weights.conv_k.ptr(), weights.conv_k.bytes()),
                v_weight_f32: Self::f32_buffer(weights.conv_v.ptr(), weights.conv_v.bytes()),
                persistent_state_f32: conv.persistent_state_f32,
                staged_state_f32: conv.staged_state_f32,
                q_output_bf16: buffers.q_conv_bf16,
                k_output_bf16: buffers.k_conv_bf16,
                v_output_bf16: buffers.v_conv_bf16,
                published_ends_u32: conv.published_ends_u32,
                published_nonces_u64: conv.published_nonces_u64,
                logical_lengths_u32: conv.logical_lengths_u32,
            },
            stream,
        )?;

        // Forget gate: the low-rank f_a/f_b stack, then the fused decay.
        self.linear(
            gpu,
            &weights.f_a,
            input,
            buffers.q8_activation,
            buffers.f_a_bf16,
            stream,
        )?;
        self.linear(
            gpu,
            &weights.f_b,
            buffers.f_a_bf16,
            buffers.q8_activation,
            buffers.f_b_bf16,
            stream,
        )?;
        self.kda.forget_gate(
            gpu,
            Glm53KdaForgetPlan::new(1, HEADS, HEAD_DIM)?,
            Glm53KdaForgetBuffers {
                projected_bf16: buffers.f_b_bf16,
                dt_bias_f32: Self::f32_buffer(weights.dt_bias.ptr(), weights.dt_bias.bytes()),
                a_log_f32: Self::f32_buffer(weights.a.ptr(), weights.a.bytes()),
                output_f32: buffers.log_decay_f32,
            },
            stream,
        )?;

        // Beta: one projection, then a per-head sigmoid.
        self.linear(
            gpu,
            &weights.beta,
            input,
            buffers.q8_activation,
            buffers.beta_proj_bf16,
            stream,
        )?;
        self.kda.beta(
            gpu,
            Glm53KdaBetaPlan::new(1, HEADS)?,
            Glm53KdaBetaBuffers {
                projected_bf16: buffers.beta_proj_bf16,
                output_bf16: buffers.beta_bf16,
            },
            stream,
        )?;

        // The recurrence. `state.buffer()` is the only buffer this may mutate.
        self.decode.launch(
            gpu,
            Glm53KdaDecodePlan::new(1, HEADS, HEAD_DIM, HEAD_DIM)?,
            Glm53KdaDecodeBuffers {
                state_f32: state.buffer(),
                query_bf16: buffers.q_conv_bf16,
                key_bf16: buffers.k_conv_bf16,
                value_bf16: buffers.v_conv_bf16,
                log_decay_f32: buffers.log_decay_f32,
                beta_bf16: buffers.beta_bf16,
                output_bf16: buffers.recurrent_out_bf16,
            },
            stream,
        )?;
        {
            // Bring-up diagnostic (ATLAS_GLM53_DUMP_DIR): KDA intermediates, so a
            // wrong attention output can be attributed to conv, gate, beta or the
            // recurrence itself rather than guessed. Mirrors the DSA dump.
            // Compare against a llama-eval-callback trace: kda_{q,k,v}_conv-N,
            // kda_gate-N (log_decay), kda_beta-N, new_state-N, attn_output-N.
            use std::sync::atomic::{AtomicU32, Ordering};
            static KDA_CALLS: AtomicU32 = AtomicU32::new(0);
            let call = KDA_CALLS.fetch_add(1, Ordering::Relaxed);
            // 34 KDA layers per token; DSA sits at every layer%4==3, so the
            // i-th KDA layer is i + i/3.
            let i = call % 34;
            let layer = i + i / 3;
            super::walk_dump::Glm53WalkDump::from_env().write(
                gpu,
                stream,
                position,
                &format!("kda-layer{layer}"),
                &[
                    (
                        "q_conv",
                        buffers.q_conv_bf16,
                        super::walk_dump::Glm53DumpDtype::Bf16,
                    ),
                    (
                        "k_conv",
                        buffers.k_conv_bf16,
                        super::walk_dump::Glm53DumpDtype::Bf16,
                    ),
                    (
                        "v_conv",
                        buffers.v_conv_bf16,
                        super::walk_dump::Glm53DumpDtype::Bf16,
                    ),
                    (
                        "log_decay_f32",
                        buffers.log_decay_f32,
                        super::walk_dump::Glm53DumpDtype::F32,
                    ),
                    (
                        "beta",
                        buffers.beta_bf16,
                        super::walk_dump::Glm53DumpDtype::Bf16,
                    ),
                    (
                        "state_f32",
                        state.buffer(),
                        super::walk_dump::Glm53DumpDtype::F32,
                    ),
                    (
                        "recurrent_out",
                        buffers.recurrent_out_bf16,
                        super::walk_dump::Glm53DumpDtype::Bf16,
                    ),
                ],
            )?;
        }

        // Output gate, gated RMS norm, output projection.
        self.linear(
            gpu,
            &weights.g_a,
            input,
            buffers.q8_activation,
            buffers.g_a_bf16,
            stream,
        )?;
        self.linear(
            gpu,
            &weights.g_b,
            buffers.g_a_bf16,
            buffers.q8_activation,
            buffers.g_b_bf16,
            stream,
        )?;
        self.kda.gated_norm(
            gpu,
            Glm53KdaNormPlan::new(1, HEADS, HEAD_DIM)?,
            Glm53KdaNormBuffers {
                input_bf16: buffers.recurrent_out_bf16,
                weight_f32: Self::f32_buffer(weights.norm.ptr(), weights.norm.bytes()),
                gate_bf16: buffers.g_b_bf16,
                output_bf16: buffers.gated_bf16,
            },
            stream,
        )?;
        // W_o writes the layer's block output, which is the walk's `hidden_a`
        // (not `hidden_b` -- that is the FFN site's). The mHC fold owns it.
        self.linear(
            gpu,
            &weights.output,
            buffers.gated_bf16,
            buffers.q8_activation,
            output,
            stream,
        )?;
        {
            // Bring-up diagnostic: post-recurrence chain. The recurrence itself
            // matches llama.cpp exactly (all 7 intermediates within 1%), so a
            // wrong block output has to come from the output gate, the gated
            // RMS norm, or W_o. Compare: norm-0, kda_normed_gated-0, kda_out-0.
            use std::sync::atomic::{AtomicU32, Ordering};
            static KDA_OUT_CALLS: AtomicU32 = AtomicU32::new(0);
            let call = KDA_OUT_CALLS.fetch_add(1, Ordering::Relaxed);
            let i = call % 34;
            let layer = i + i / 3;
            super::walk_dump::Glm53WalkDump::from_env().write(
                gpu,
                stream,
                position,
                &format!("kdaout-layer{layer}"),
                &[
                    (
                        "g_a",
                        buffers.g_a_bf16,
                        super::walk_dump::Glm53DumpDtype::Bf16,
                    ),
                    (
                        "g_b",
                        buffers.g_b_bf16,
                        super::walk_dump::Glm53DumpDtype::Bf16,
                    ),
                    (
                        "gated",
                        buffers.gated_bf16,
                        super::walk_dump::Glm53DumpDtype::Bf16,
                    ),
                    ("out", output, super::walk_dump::Glm53DumpDtype::Bf16),
                ],
            )?;
        }

        Ok(GLM53_KDA_KERNEL_LAUNCHES)
    }

    /// Stage one KDA layer from the pinned EXL3 target checkpoint.
    #[allow(clippy::too_many_arguments)]
    pub fn stage_exl3(
        &self,
        gpu: &dyn GpuBackend,
        weights: &Glm53Exl3KdaWeights,
        input: GgmlIqBuffer,
        buffers: Glm53KdaScratchBuffers,
        projection_scratch: Glm53Exl3ProjectionScratch,
        state: &Glm53KdaScratchState,
        conv: Glm53KdaConvSlots,
        position: u32,
        capacity: u32,
        nonce: u64,
        output: GgmlIqBuffer,
        stream: u64,
    ) -> Result<u32> {
        self.stage_exl3_rows(
            gpu,
            1,
            weights,
            input,
            buffers,
            projection_scratch,
            state,
            conv,
            position,
            capacity,
            nonce,
            output,
            stream,
        )
    }

    /// Stage a causal EXL3 verifier chunk in one layer-major pass. The
    /// recurrence mutates only the staged state and publishes the chunk end;
    /// callers decide later whether that state is accepted.
    #[allow(clippy::too_many_arguments)]
    pub fn stage_exl3_rows(
        &self,
        gpu: &dyn GpuBackend,
        rows: u32,
        weights: &Glm53Exl3KdaWeights,
        input: GgmlIqBuffer,
        buffers: Glm53KdaScratchBuffers,
        projection_scratch: Glm53Exl3ProjectionScratch,
        state: &Glm53KdaScratchState,
        conv: Glm53KdaConvSlots,
        position: u32,
        capacity: u32,
        nonce: u64,
        output: GgmlIqBuffer,
        stream: u64,
    ) -> Result<u32> {
        let max_rows = if glm53_layer_major_prefill_active() {
            u32::try_from(GLM53_EXL3_MAX_WIDE_ROWS)?
        } else {
            8
        };
        ensure!(
            (1..=max_rows).contains(&rows),
            "GLM EXL3 KDA rows must be 1..={max_rows}"
        );
        self.validate_exl3_shapes(weights)?;
        state.prime(gpu, stream)?;

        let qkv = buffers.combined_qkv_bf16;
        self.linear_exl3_rows(
            gpu,
            rows,
            Glm53Exl3Projection::Compressed(&weights.qkv),
            input,
            qkv,
            projection_scratch,
            stream,
        )?;
        if rows > 1 {
            self.kda.split_qkv(
                gpu,
                rows,
                Glm53KdaQkvBuffers {
                    combined_bf16: qkv,
                    query_bf16: buffers.q_proj_bf16,
                    key_bf16: buffers.k_proj_bf16,
                    value_bf16: buffers.v_proj_bf16,
                },
                stream,
            )?;
        }

        let conv_weights = split_conv(&weights.conv)?;
        let exact_wide = rows > 1 && glm53_exact_wide_prefill_active();
        let batch_exact_carry = exact_wide
            && std::env::var("ATLAS_GLM53_EXACT_WIDE_KDA_BATCH_COPY").as_deref() == Ok("1");
        if exact_wide && !batch_exact_carry {
            let slice = |buffer: GgmlIqBuffer, row: usize| -> Result<GgmlIqBuffer> {
                ensure!(
                    buffer.bytes == rows as usize * 16_384,
                    "GLM exact-wide-prefill KDA conv buffer extent drift"
                );
                Ok(GgmlIqBuffer {
                    ptr: buffer.ptr.offset(row * 16_384),
                    bytes: 16_384,
                })
            };
            for row in 0..rows as usize {
                let row_position = position
                    .checked_add(u32::try_from(row)?)
                    .context("GLM exact-wide-prefill KDA conv position overflow")?;
                self.conv.launch_stage(
                    gpu,
                    Glm53KdaConvPlan::new(
                        1,
                        1,
                        capacity,
                        row_position,
                        row_position + 1,
                        HEADS,
                        HEAD_DIM,
                        CONV_KERNEL,
                        nonce,
                    )?,
                    Glm53KdaConvBuffers {
                        q_input_bf16: slice(buffers.q_proj_bf16, row)?,
                        k_input_bf16: slice(buffers.k_proj_bf16, row)?,
                        v_input_bf16: slice(buffers.v_proj_bf16, row)?,
                        q_weight_f32: conv_weights[0],
                        k_weight_f32: conv_weights[1],
                        v_weight_f32: conv_weights[2],
                        persistent_state_f32: conv.persistent_state_f32,
                        staged_state_f32: conv.staged_state_f32,
                        q_output_bf16: slice(buffers.q_conv_bf16, row)?,
                        k_output_bf16: slice(buffers.k_conv_bf16, row)?,
                        v_output_bf16: slice(buffers.v_conv_bf16, row)?,
                        published_ends_u32: conv.published_ends_u32,
                        published_nonces_u64: conv.published_nonces_u64,
                        logical_lengths_u32: conv.logical_lengths_u32,
                    },
                    stream,
                )?;
                gpu.copy_d2d_async(
                    conv.staged_state_f32.ptr,
                    conv.persistent_state_f32.ptr,
                    conv.staged_state_f32.bytes,
                    stream,
                )?;
            }
        } else {
            self.conv.launch_stage(
                gpu,
                Glm53KdaConvPlan::new(
                    1,
                    rows,
                    capacity,
                    position,
                    position
                        .checked_add(rows)
                        .context("GLM KDA conv position overflow")?,
                    HEADS,
                    HEAD_DIM,
                    CONV_KERNEL,
                    nonce,
                )?,
                Glm53KdaConvBuffers {
                    q_input_bf16: buffers.q_proj_bf16,
                    k_input_bf16: buffers.k_proj_bf16,
                    v_input_bf16: buffers.v_proj_bf16,
                    q_weight_f32: conv_weights[0],
                    k_weight_f32: conv_weights[1],
                    v_weight_f32: conv_weights[2],
                    persistent_state_f32: conv.persistent_state_f32,
                    staged_state_f32: conv.staged_state_f32,
                    q_output_bf16: buffers.q_conv_bf16,
                    k_output_bf16: buffers.k_conv_bf16,
                    v_output_bf16: buffers.v_conv_bf16,
                    published_ends_u32: conv.published_ends_u32,
                    published_nonces_u64: conv.published_nonces_u64,
                    logical_lengths_u32: conv.logical_lengths_u32,
                },
                stream,
            )?;
            if batch_exact_carry {
                gpu.copy_d2d_async(
                    conv.staged_state_f32.ptr,
                    conv.persistent_state_f32.ptr,
                    conv.staged_state_f32.bytes,
                    stream,
                )?;
            }
        }

        self.linear_exl3_rows(
            gpu,
            rows,
            Glm53Exl3Projection::NativeBf16(&weights.f_a),
            input,
            buffers.f_a_bf16,
            projection_scratch,
            stream,
        )?;
        self.linear_exl3_rows(
            gpu,
            rows,
            Glm53Exl3Projection::NativeBf16(&weights.f_b),
            buffers.f_a_bf16,
            buffers.f_b_bf16,
            projection_scratch,
            stream,
        )?;
        self.kda.forget_gate(
            gpu,
            Glm53KdaForgetPlan::new(rows, HEADS, HEAD_DIM)?,
            Glm53KdaForgetBuffers {
                projected_bf16: buffers.f_b_bf16,
                dt_bias_f32: native_buffer(&weights.dt_bias),
                a_log_f32: native_buffer(&weights.a_log),
                output_f32: buffers.log_decay_f32,
            },
            stream,
        )?;

        self.linear_exl3_rows(
            gpu,
            rows,
            Glm53Exl3Projection::NativeBf16(&weights.beta),
            input,
            buffers.beta_proj_bf16,
            projection_scratch,
            stream,
        )?;
        self.kda.beta(
            gpu,
            Glm53KdaBetaPlan::new(rows, HEADS)?,
            Glm53KdaBetaBuffers {
                projected_bf16: buffers.beta_proj_bf16,
                output_bf16: buffers.beta_bf16,
            },
            stream,
        )?;

        if rows == 1 {
            self.decode.launch(
                gpu,
                Glm53KdaDecodePlan::new(1, HEADS, HEAD_DIM, HEAD_DIM)?,
                Glm53KdaDecodeBuffers {
                    state_f32: state.buffer(),
                    query_bf16: buffers.q_conv_bf16,
                    key_bf16: buffers.k_conv_bf16,
                    value_bf16: buffers.v_conv_bf16,
                    log_decay_f32: buffers.log_decay_f32,
                    beta_bf16: buffers.beta_bf16,
                    output_bf16: buffers.recurrent_out_bf16,
                },
                stream,
            )?;
        } else {
            let prefill_plan = Glm53KdaPrefillPlan::new(1, rows, HEADS, HEAD_DIM, HEAD_DIM)?;
            let prefill_buffers = Glm53KdaPrefillBuffers {
                state_f32: state.buffer(),
                query_bf16: buffers.q_conv_bf16,
                key_bf16: buffers.k_conv_bf16,
                value_bf16: buffers.v_conv_bf16,
                log_decay_f32: buffers.log_decay_f32,
                beta_bf16: buffers.beta_bf16,
                output_bf16: buffers.recurrent_out_bf16,
            };
            // Kernel oracle: the recurrence overwrites `state_f32` in place, so
            // the pre-state must be captured BEFORE the launch and the
            // post-state after it.
            crate::model::glm53::oracle_dump::dump_site(
                gpu,
                stream,
                "kda_recurrence_in",
                &[
                    t("state_pre", prefill_buffers.state_f32.ptr, prefill_plan.state_bytes, "f32", &[64, 128, 128]),
                    t("query", prefill_buffers.query_bf16.ptr, prefill_plan.vector_bytes, "bf16", &[rows as usize, 64, 128]),
                    t("key", prefill_buffers.key_bf16.ptr, prefill_plan.vector_bytes, "bf16", &[rows as usize, 64, 128]),
                    t("value", prefill_buffers.value_bf16.ptr, prefill_plan.vector_bytes, "bf16", &[rows as usize, 64, 128]),
                    t("log_decay", prefill_buffers.log_decay_f32.ptr, prefill_plan.decay_bytes, "f32", &[rows as usize, 64, 128]),
                    t("beta", prefill_buffers.beta_bf16.ptr, prefill_plan.beta_bytes, "bf16", &[rows as usize, 64]),
                ],
                &[
                    ("batch", "1".into()),
                    ("tokens", rows.to_string()),
                    ("heads", "64".into()),
                    ("key_dim", "128".into()),
                    ("value_dim", "128".into()),
                    ("l2_epsilon", "1.0e-6".into()),
                ],
            )?;
            if rows > 8 && glm53_layer_major_prefill_active() {
                self.prefill.launch_register_resident(
                    gpu,
                    prefill_plan,
                    prefill_buffers,
                    stream,
                )?;
            } else {
                self.prefill
                    .launch(gpu, prefill_plan, prefill_buffers, stream)?;
            }
            crate::model::glm53::oracle_dump::dump_site(
                gpu,
                stream,
                "kda_recurrence_out",
                &[
                    t("output", prefill_buffers.output_bf16.ptr, prefill_plan.vector_bytes, "bf16", &[rows as usize, 64, 128]),
                    t("state_post", prefill_buffers.state_f32.ptr, prefill_plan.state_bytes, "f32", &[64, 128, 128]),
                ],
                &[("tokens", rows.to_string())],
            )?;
        }

        self.linear_exl3_rows(
            gpu,
            rows,
            Glm53Exl3Projection::NativeBf16(&weights.g_a),
            input,
            buffers.g_a_bf16,
            projection_scratch,
            stream,
        )?;
        self.linear_exl3_rows(
            gpu,
            rows,
            Glm53Exl3Projection::NativeBf16(&weights.g_b),
            buffers.g_a_bf16,
            buffers.g_b_bf16,
            projection_scratch,
            stream,
        )?;
        self.kda.gated_norm(
            gpu,
            Glm53KdaNormPlan::new(rows, HEADS, HEAD_DIM)?,
            Glm53KdaNormBuffers {
                input_bf16: buffers.recurrent_out_bf16,
                weight_f32: native_buffer(&weights.norm),
                gate_bf16: buffers.g_b_bf16,
                output_bf16: buffers.gated_bf16,
            },
            stream,
        )?;
        self.linear_exl3_rows(
            gpu,
            rows,
            Glm53Exl3Projection::Compressed(&weights.output),
            buffers.gated_bf16,
            output,
            projection_scratch,
            stream,
        )?;
        Ok(GLM53_EXL3_KDA_KERNEL_LAUNCHES)
    }

    fn validate_exl3_shapes(&self, weights: &Glm53Exl3KdaWeights) -> Result<()> {
        for (name, matrix, inner, columns) in [
            ("qkv", &weights.qkv, HIDDEN, 3 * QKV_DIM),
            ("output", &weights.output, QKV_DIM, HIDDEN),
        ] {
            ensure!(
                matrix.size_k() == inner && matrix.size_n() == columns,
                "GLM EXL3 KDA {name} is [{}, {}], expected [{inner}, {columns}]",
                matrix.size_k(),
                matrix.size_n()
            );
        }
        for (name, tensor, shape, dtype) in [
            (
                "a_log",
                &weights.a_log,
                &[HEADS as u64][..],
                Glm53Exl3NativeDtype::F32,
            ),
            (
                "beta",
                &weights.beta,
                &[HEADS as u64, HIDDEN as u64][..],
                Glm53Exl3NativeDtype::Bf16,
            ),
            (
                "conv",
                &weights.conv,
                &[(3 * QKV_DIM) as u64, 1, CONV_KERNEL as u64][..],
                Glm53Exl3NativeDtype::F32,
            ),
            (
                "dt_bias",
                &weights.dt_bias,
                &[QKV_DIM as u64][..],
                Glm53Exl3NativeDtype::F32,
            ),
            (
                "f_a",
                &weights.f_a,
                &[LOW_RANK as u64, HIDDEN as u64][..],
                Glm53Exl3NativeDtype::Bf16,
            ),
            (
                "f_b",
                &weights.f_b,
                &[QKV_DIM as u64, LOW_RANK as u64][..],
                Glm53Exl3NativeDtype::Bf16,
            ),
            (
                "g_a",
                &weights.g_a,
                &[LOW_RANK as u64, HIDDEN as u64][..],
                Glm53Exl3NativeDtype::Bf16,
            ),
            (
                "g_b",
                &weights.g_b,
                &[QKV_DIM as u64, LOW_RANK as u64][..],
                Glm53Exl3NativeDtype::Bf16,
            ),
            (
                "norm",
                &weights.norm,
                &[HEAD_DIM as u64][..],
                Glm53Exl3NativeDtype::F32,
            ),
        ] {
            ensure!(
                tensor.dtype() == dtype && tensor.shape() == shape,
                "GLM EXL3 KDA {name} dtype/shape drift"
            );
        }
        Ok(())
    }

    /// Every projection's shape is pinned rather than inferred: a checkpoint
    /// that disagrees must fail here, not run a recurrence over the wrong
    /// geometry and emit fluent, wrong output.
    fn validate_shapes(&self, weights: &Glm53KdaWeights) -> Result<()> {
        for (name, matrix, inner, columns) in [
            ("attn_q", &weights.q, HIDDEN, QKV_DIM),
            ("attn_k", &weights.k, HIDDEN, QKV_DIM),
            ("attn_v", &weights.v, HIDDEN, QKV_DIM),
            ("attn_output", &weights.output, QKV_DIM, HIDDEN),
            ("ssm_f_a", &weights.f_a, HIDDEN, LOW_RANK),
            ("ssm_f_b", &weights.f_b, LOW_RANK, QKV_DIM),
            ("ssm_g_a", &weights.g_a, HIDDEN, LOW_RANK),
            ("ssm_g_b", &weights.g_b, LOW_RANK, QKV_DIM),
            ("ssm_beta", &weights.beta, HIDDEN, HEADS),
        ] {
            ensure!(
                matrix.inner() == inner && matrix.columns() == columns,
                "GLM KDA {name} is [{}, {}], expected [{inner}, {columns}]",
                matrix.inner(),
                matrix.columns()
            );
        }
        for (name, elements, expected) in [
            ("ssm_a", weights.a.elements(), HEADS as usize),
            ("ssm_dt.bias", weights.dt_bias.elements(), QKV_DIM as usize),
            ("ssm_norm", weights.norm.elements(), HEAD_DIM as usize),
        ] {
            ensure!(
                elements == expected,
                "GLM KDA {name} has {elements} F32 elements, expected {expected}"
            );
        }
        Ok(())
    }
}

fn exl3_buffer(buffer: GgmlIqBuffer) -> Glm53Exl3Buffer {
    Glm53Exl3Buffer {
        ptr: buffer.ptr,
        bytes: buffer.bytes,
    }
}

fn native_buffer(tensor: &crate::weight_loader::Glm53Exl3NativeTensor) -> GgmlIqBuffer {
    GgmlIqBuffer {
        ptr: tensor.ptr(),
        bytes: tensor.bytes(),
    }
}

fn split_conv(tensor: &crate::weight_loader::Glm53Exl3NativeTensor) -> Result<[GgmlIqBuffer; 3]> {
    ensure!(
        tensor.dtype() == Glm53Exl3NativeDtype::F32
            && tensor.shape() == [(3 * QKV_DIM) as u64, 1, CONV_KERNEL as u64],
        "GLM EXL3 KDA combined conv dtype/shape drift"
    );
    let bytes = QKV_DIM as usize * CONV_KERNEL as usize * size_of::<f32>();
    ensure!(
        tensor.bytes() == 3 * bytes,
        "GLM EXL3 KDA conv extent drift"
    );
    Ok(std::array::from_fn(|index| GgmlIqBuffer {
        ptr: DevicePtr(tensor.ptr().0 + (index * bytes) as u64),
        bytes,
    }))
}

#[cfg(test)]
#[path = "kda_attention_tests.rs"]
mod tests;
