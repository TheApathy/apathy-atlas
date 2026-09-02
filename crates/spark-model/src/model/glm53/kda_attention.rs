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

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::gguf::GgmlType;

use crate::layers::ops::{
    GgmlIqBuffer, GgmlIqMmqKernels, Glm53KdaBetaBuffers, Glm53KdaBetaPlan, Glm53KdaConvBuffers,
    Glm53KdaConvKernel, Glm53KdaConvPlan, Glm53KdaDecodeBuffers, Glm53KdaDecodeKernel,
    Glm53KdaDecodePlan, Glm53KdaForgetBuffers, Glm53KdaForgetPlan, Glm53KdaKernels,
    Glm53KdaNormBuffers, Glm53KdaNormPlan,
};
use crate::weight_loader::{Glm53GgufMatrix, Glm53KdaWeights};

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
                    ("q_conv", buffers.q_conv_bf16, super::walk_dump::Glm53DumpDtype::Bf16),
                    ("k_conv", buffers.k_conv_bf16, super::walk_dump::Glm53DumpDtype::Bf16),
                    ("v_conv", buffers.v_conv_bf16, super::walk_dump::Glm53DumpDtype::Bf16),
                    ("log_decay_f32", buffers.log_decay_f32, super::walk_dump::Glm53DumpDtype::F32),
                    ("beta", buffers.beta_bf16, super::walk_dump::Glm53DumpDtype::Bf16),
                    ("state_f32", state.buffer(), super::walk_dump::Glm53DumpDtype::F32),
                    ("recurrent_out", buffers.recurrent_out_bf16, super::walk_dump::Glm53DumpDtype::Bf16),
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
                    ("g_a", buffers.g_a_bf16, super::walk_dump::Glm53DumpDtype::Bf16),
                    ("g_b", buffers.g_b_bf16, super::walk_dump::Glm53DumpDtype::Bf16),
                    ("gated", buffers.gated_bf16, super::walk_dump::Glm53DumpDtype::Bf16),
                    ("out", output, super::walk_dump::Glm53DumpDtype::Bf16),
                ],
            )?;
        }

        Ok(GLM53_KDA_KERNEL_LAUNCHES)
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

#[cfg(test)]
#[path = "kda_attention_tests.rs"]
mod tests;
