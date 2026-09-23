// SPDX-License-Identifier: AGPL-3.0-only

//! DSA attention for the 11 sparse-latent layers (`layer % 4 == 3`).
//!
//! Transcribed from the reference graph's `build_mla_layer` and
//! `build_dsa_top_k` (`llama.cpp` `src/models/glm5next.cpp`):
//!
//! ```text
//! qr      = rms_norm(W_qa x, q_a_norm)                       [1536]
//! q       = W_qb qr                              -> 64 heads x 256
//! Qcur    = per-head W_kb q                      -> 64 heads x 512   (absorbed)
//! kv_cmpr = rms_norm(W_kv_a_mqa x, kv_a_norm)                [512]
//!
//! index_k = layer_norm(W_ik x, k_norm, k_norm_bias)          [128]
//! index_g = W_gate x                                         [128]
//! pooled  = softmax(index_g + ape) . index_k over 4 members  (k-pool)
//! index_q = W_iqb qr                             -> 32 heads x 128
//! wts     = (W_proj x) * (d*nh)^-1/2                         [32]
//! scores  = sum_h wts[h] * relu(index_q_h . pool_key)
//! sel     = top-512 pools -> 2051 token indices, tail always selected
//!
//! out     = softmax(Qcur . latent[sel]) . latent[sel]        (absorbed space)
//! out     = per-head W_vb out                    -> 64 heads x 256
//! cur     = W_o out
//! ```
//!
//! # Absorption is the cost centre
//!
//! MLA absorption folds `W_kb` into the query and `W_vb` back out of the
//! output, per head. That is 64 + 64 rank-`[256,512]` matmuls per layer, each a
//! quantize plus a matmul, so 256 launches before anything else runs. Across 11
//! DSA layers that is the bulk of a token's launch count. It is correct and it
//! is slow; batching the banks into one grouped GEMM is the obvious later win
//! and is deliberately not attempted here.
//!
//! # Staging only
//!
//! The latent append is staged into the transaction overlay, never written
//! straight to the persistent cache; the index tail and pool commits happen
//! later under `Glm53DsaIndexCommitKernel` on `accepted == 1`. Nothing in this
//! module mutates persistent state.

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::weights::gguf::GgmlType;

use crate::layers::ops::{
    GLM53_EXL3_MAX_WIDE_ROWS, GgmlIqBuffer, GgmlIqMmqKernels, Glm53DsaHeadTransposeBuffers,
    Glm53DsaLatentAppendBuffers, Glm53DsaLatentAppendKernel, Glm53DsaLatentAppendPlan,
    Glm53DsaLatentAppendStorage, Glm53DsaNormProjectionBuffers, Glm53DsaNormProjectionKernels,
    Glm53DsaNormProjectionKind, Glm53DsaNormProjectionPlan, Glm53DsaPoolBuffers,
    Glm53DsaPoolKernel, Glm53DsaPoolPlan, Glm53DsaScoreBuffers, Glm53DsaScoreKernel,
    Glm53DsaScorePlan, Glm53DsaSelectedAttentionBuffers, Glm53DsaSelectedAttentionKernel,
    Glm53DsaSelectedAttentionPlan, Glm53DsaSelectedStorage, Glm53DsaTopkBuffers,
    Glm53DsaTopkKernel, Glm53DsaTopkPlan, Glm53Exl3Buffer, Glm53Exl3Projection,
    Glm53Exl3ProjectionBuffers, Glm53Exl3ProjectionScratch, glm53_exact_verify_active,
    glm53_exact_wide_prefill_active, glm53_layer_major_prefill_active,
};
use crate::weight_loader::{
    Glm53DsaWeights, Glm53Exl3DsaWeights, Glm53Exl3NativeDtype, Glm53GgufMatrix,
};

use super::dsa_dense_indexer::{
    Glm53DsaDenseIndexerPlan, dense_full_coverage, parse_dense_indexer_skip,
};
use super::walk_scratch::Glm53DsaScratchBuffers;
#[path = "dsa_verify_precompute.rs"]
mod dsa_verify_precompute;

const HIDDEN: u32 = 4_096;
const HEADS: u32 = 64;
const HEAD_DIM: u32 = 256;
const LATENT: u32 = 512;
const Q_RANK: u32 = 1_536;
const INDEX_DIM: u32 = 128;
const INDEX_HEADS: u32 = 32;
const KPOOL: u32 = 4;

/// Positive build provenance for the completing-pool publish order. Grep the
/// artifact you are about to execute for this exact string; never infer the
/// build from the ABSENCE of a marker, which LLVM is free to delete.
#[used]
static ATLAS_GLM53_DSA_POOL_PUBLISH_MARKER: &str = "ATLAS_GLM53_DSA_POOL_PUBLISH=pre-score";
const INDEX_TOPK: u32 = 2_048;
pub(super) const SELECTED: u32 = 2_051;
/// The absolute position space every DSA op is compiled against. Distinct from
/// a sequence's runtime capacity, which may be far smaller.
pub(super) const GLM53_DSA_ABSOLUTE_POSITIONS: u32 = 1_048_576;

/// Per-head absorption matmuls: `W_kb` in, `W_vb` out.
pub const GLM53_DSA_ABSORPTION_MATMULS: u32 = 2 * HEADS;

/// The persistent and staged cache slices one DSA layer touches.
///
/// The persistent halves are **read-only** here: the latent append writes the
/// transaction overlay, and publication happens later under the index commit.
#[derive(Debug, Clone, Copy)]
pub struct Glm53DsaCacheSlots {
    pub latent_cache_bf16: GgmlIqBuffer,
    pub latent_overlay_bf16: GgmlIqBuffer,
    pub pool_keys_bf16: GgmlIqBuffer,
    pub pool_validity_u8: GgmlIqBuffer,
    pub prior_tail_keys_bf16: GgmlIqBuffer,
    pub prior_tail_gates_bf16: GgmlIqBuffer,
    pub prior_tail_validity_u8: GgmlIqBuffer,
    pub out_tail_validity_u8: GgmlIqBuffer,
    pub sequence_lengths_u32: GgmlIqBuffer,
    pub query_positions_u32: GgmlIqBuffer,
    pub query_validity_u8: GgmlIqBuffer,
    pub published_ends_u32: GgmlIqBuffer,
    pub published_nonces_u64: GgmlIqBuffer,
}

enum DsaProjection<'a> {
    Gguf(&'a Glm53GgufMatrix),
    GgufOwned(Glm53GgufMatrix),
    Exl3(Glm53Exl3Projection<'a>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DsaStatelessInputs {
    Compute,
    PrecomputedWide,
}

#[derive(Clone, Copy)]
enum DsaWeightsRef<'a> {
    Gguf(&'a Glm53DsaWeights),
    Exl3(&'a Glm53Exl3DsaWeights),
}

impl<'a> DsaWeightsRef<'a> {
    fn projection(
        self,
        gguf: impl FnOnce(&'a Glm53DsaWeights) -> &'a Glm53GgufMatrix,
        exl3: impl FnOnce(&'a Glm53Exl3DsaWeights) -> Glm53Exl3Projection<'a>,
    ) -> DsaProjection<'a> {
        match self {
            Self::Gguf(weights) => DsaProjection::Gguf(gguf(weights)),
            Self::Exl3(weights) => DsaProjection::Exl3(exl3(weights)),
        }
    }

    fn q_a(self) -> DsaProjection<'a> {
        self.projection(|w| &w.q_a, |w| Glm53Exl3Projection::Compressed(&w.q_a))
    }
    fn q_b(self) -> DsaProjection<'a> {
        self.projection(|w| &w.q_b, |w| Glm53Exl3Projection::Compressed(&w.q_b))
    }
    fn kv_a(self) -> DsaProjection<'a> {
        self.projection(
            |w| &w.kv_a_mqa,
            |w| Glm53Exl3Projection::Compressed(&w.kv_a),
        )
    }
    fn output(self) -> DsaProjection<'a> {
        self.projection(
            |w| &w.output,
            |w| Glm53Exl3Projection::Compressed(&w.output),
        )
    }
    fn indexer_q_b(self) -> DsaProjection<'a> {
        self.projection(
            |w| &w.indexer_q_b,
            |w| Glm53Exl3Projection::Compressed(&w.indexer_q_b),
        )
    }
    fn indexer_k(self) -> DsaProjection<'a> {
        self.projection(
            |w| &w.indexer_k,
            |w| Glm53Exl3Projection::NativeBf16(&w.indexer_k),
        )
    }
    fn compressor_gate(self) -> DsaProjection<'a> {
        self.projection(
            |w| &w.compressor_gate,
            |w| Glm53Exl3Projection::NativeBf16(&w.compressor_gate),
        )
    }
    fn k_b(self, head: u32) -> Result<DsaProjection<'a>> {
        Ok(match self {
            Self::Gguf(weights) => DsaProjection::GgufOwned(weights.k_b.expert(head)?),
            Self::Exl3(weights) => DsaProjection::Exl3(Glm53Exl3Projection::NativeBf16ActWeight(
                weights
                    .k_b
                    .get(head as usize)
                    .context("GLM EXL3 DSA k_b head missing")?,
            )),
        })
    }
    fn v_b(self, head: u32) -> Result<DsaProjection<'a>> {
        Ok(match self {
            Self::Gguf(weights) => DsaProjection::GgufOwned(weights.v_b.expert(head)?),
            Self::Exl3(weights) => DsaProjection::Exl3(Glm53Exl3Projection::NativeBf16(
                weights
                    .v_b
                    .get(head as usize)
                    .context("GLM EXL3 DSA v_b head missing")?,
            )),
        })
    }

    fn exl3_absorption_bank(self, key_bank: bool) -> Result<Option<DevicePtr>> {
        let Self::Exl3(weights) = self else {
            return Ok(None);
        };
        let bank = if key_bank { &weights.k_b } else { &weights.v_b };
        ensure!(
            bank.len() == HEADS as usize,
            "GLM EXL3 DSA absorption bank needs {HEADS} heads"
        );
        let base = bank[0].ptr();
        let stride_bytes = u64::from(4 * HEAD_DIM * LATENT);
        for (head, matrix) in bank.iter().enumerate() {
            ensure!(
                matrix.dtype() == Glm53Exl3NativeDtype::Bf16
                    && matrix.shape() == [u64::from(HEAD_DIM), u64::from(LATENT)]
                    && matrix.ptr().0 == base.0 + head as u64 * stride_bytes,
                "GLM EXL3 DSA absorption bank is not packed at head {head}"
            );
        }
        Ok(Some(base))
    }

    fn q_a_norm(self) -> GgmlIqBuffer {
        match self {
            Self::Gguf(w) => native(w.q_a_norm.ptr(), w.q_a_norm.bytes()),
            Self::Exl3(w) => native(w.q_a_norm.ptr(), w.q_a_norm.bytes()),
        }
    }
    fn kv_a_norm(self) -> GgmlIqBuffer {
        match self {
            Self::Gguf(w) => native(w.kv_a_norm.ptr(), w.kv_a_norm.bytes()),
            Self::Exl3(w) => native(w.kv_a_norm.ptr(), w.kv_a_norm.bytes()),
        }
    }
    fn indexer_k_norm(self) -> GgmlIqBuffer {
        match self {
            Self::Gguf(w) => native(w.indexer_k_norm.ptr(), w.indexer_k_norm.bytes()),
            Self::Exl3(w) => native(w.indexer_k_norm.ptr(), w.indexer_k_norm.bytes()),
        }
    }
    fn indexer_k_norm_bias(self) -> GgmlIqBuffer {
        match self {
            Self::Gguf(w) => native(w.indexer_k_norm_bias.ptr(), w.indexer_k_norm_bias.bytes()),
            Self::Exl3(w) => native(w.indexer_k_norm_bias.ptr(), w.indexer_k_norm_bias.bytes()),
        }
    }
    fn indexer_proj(self) -> GgmlIqBuffer {
        match self {
            Self::Gguf(w) => native(w.indexer_proj.ptr(), w.indexer_proj.bytes()),
            Self::Exl3(w) => native(w.indexer_proj.ptr(), w.indexer_proj.bytes()),
        }
    }
    fn compressor_ape(self) -> GgmlIqBuffer {
        match self {
            Self::Gguf(w) => native(w.compressor_ape.ptr(), w.compressor_ape.bytes()),
            Self::Exl3(w) => native(w.compressor_ape.ptr(), w.compressor_ape.bytes()),
        }
    }
}

fn native(ptr: DevicePtr, bytes: usize) -> GgmlIqBuffer {
    GgmlIqBuffer { ptr, bytes }
}

/// Typed handles for one DSA layer.
pub struct Glm53DsaAttentionKernels {
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
    norms: Glm53DsaNormProjectionKernels,
    pool: Glm53DsaPoolKernel,
    score: Glm53DsaScoreKernel,
    topk: Glm53DsaTopkKernel,
    selected: Glm53DsaSelectedAttentionKernel,
    latent_append: Glm53DsaLatentAppendKernel,
    prepare_metadata: KernelHandle,
    absorb_key_bank: Option<KernelHandle>,
    absorb_value_bank: Option<KernelHandle>,
}

impl Glm53DsaAttentionKernels {
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
            norms: Glm53DsaNormProjectionKernels::load(gpu)?,
            pool: Glm53DsaPoolKernel::load(gpu)?,
            score: Glm53DsaScoreKernel::load(gpu)?,
            topk: Glm53DsaTopkKernel::load(gpu)?,
            selected: Glm53DsaSelectedAttentionKernel::load(gpu)?,
            latent_append: Glm53DsaLatentAppendKernel::load(gpu)?,
            prepare_metadata: gpu.kernel("glm53_dsa_topk", "atlas_glm53_dsa_prepare_metadata")?,
            absorb_key_bank: gpu
                .kernel("glm53_exl3_dsa_absorb", "atlas_glm53_dsa_absorb_key_bank")
                .ok(),
            absorb_value_bank: gpu
                .kernel("glm53_exl3_dsa_absorb", "atlas_glm53_dsa_absorb_value_bank")
                .ok(),
        })
    }

    fn absorb_exl3_bank(
        &self,
        gpu: &dyn GpuBackend,
        key_bank: bool,
        rows: u32,
        input: GgmlIqBuffer,
        weight: DevicePtr,
        output: GgmlIqBuffer,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            (1..=u32::try_from(GLM53_EXL3_MAX_WIDE_ROWS)?).contains(&rows),
            "GLM EXL3 DSA bank needs 1..={GLM53_EXL3_MAX_WIDE_ROWS} rows"
        );
        let (kernel, input_dim, output_dim) = if key_bank {
            (self.absorb_key_bank, HEAD_DIM, LATENT)
        } else {
            (self.absorb_value_bank, LATENT, HEAD_DIM)
        };
        let kernel = kernel.context("GLM EXL3 DSA bank kernel was not built")?;
        ensure!(
            input.bytes >= rows as usize * HEADS as usize * input_dim as usize * 2
                && output.bytes >= rows as usize * HEADS as usize * output_dim as usize * 2,
            "GLM EXL3 DSA bank buffer extent drift"
        );
        KernelLaunch::new(gpu, kernel)
            .grid([HEADS, output_dim.div_ceil(32), rows.div_ceil(16)])
            .block([128, 1, 1])
            .arg_ptr(input.ptr)
            .arg_ptr(weight)
            .arg_ptr(output.ptr)
            .arg_u32(rows)
            .launch(stream)
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
            _ => bail!("GLM DSA matrix has no typed IQ-MMQ kernel"),
        })
    }

    fn linear(
        &self,
        gpu: &dyn GpuBackend,
        matrix: &Glm53GgufMatrix,
        input: DevicePtr,
        q8: GgmlIqBuffer,
        output: GgmlIqBuffer,
        stream: u64,
    ) -> Result<()> {
        let plan = matrix.plan(1)?;
        ensure!(
            q8.bytes >= plan.activation_bytes,
            "GLM DSA Q8 scratch is {} bytes, needs {}",
            q8.bytes,
            plan.activation_bytes
        );
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

    fn linear_dsa(
        &self,
        gpu: &dyn GpuBackend,
        projection: DsaProjection<'_>,
        input: GgmlIqBuffer,
        q8: GgmlIqBuffer,
        output: GgmlIqBuffer,
        scratch: Option<Glm53Exl3ProjectionScratch>,
        stream: u64,
    ) -> Result<()> {
        self.linear_dsa_rows(gpu, 1, projection, input, q8, output, scratch, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_dsa_rows(
        &self,
        gpu: &dyn GpuBackend,
        rows: u32,
        projection: DsaProjection<'_>,
        input: GgmlIqBuffer,
        q8: GgmlIqBuffer,
        output: GgmlIqBuffer,
        scratch: Option<Glm53Exl3ProjectionScratch>,
        stream: u64,
    ) -> Result<()> {
        match projection {
            DsaProjection::Gguf(matrix) if rows == 1 => {
                self.linear(gpu, matrix, input.ptr, q8, output, stream)
            }
            DsaProjection::GgufOwned(matrix) => {
                ensure!(rows == 1, "GLM GGUF DSA remains a T1 reference");
                self.linear(gpu, &matrix, input.ptr, q8, output, stream)
            }
            DsaProjection::Gguf(_) => bail!("GLM GGUF DSA remains a T1 reference"),
            DsaProjection::Exl3(projection) => {
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
                        scratch: scratch.context("GLM EXL3 DSA projection scratch missing")?,
                    },
                    stream,
                )
            }
        }
    }

    fn f32_buffer(ptr: DevicePtr, bytes: usize) -> GgmlIqBuffer {
        GgmlIqBuffer { ptr, bytes }
    }

    /// One head's slice of a `[heads][width]` BF16 buffer.
    ///
    /// The slice **length** is one head's width; only the offset scales with the
    /// head index. Conflating the two yields a zero-length slice for head 0 and
    /// an ever-growing one after it, which the op layer rejects as an extent
    /// mismatch rather than silently reading past the head.
    fn head_slice(buffer: GgmlIqBuffer, head: u32, width: u32) -> Result<GgmlIqBuffer> {
        let bytes = u64::from(width)
            .checked_mul(2)
            .context("GLM DSA head width overflow")?;
        let offset = u64::from(head)
            .checked_mul(bytes)
            .context("GLM DSA head offset overflow")?;
        let end = offset
            .checked_add(bytes)
            .context("GLM DSA head extent overflow")?;
        ensure!(
            end <= buffer.bytes as u64,
            "GLM DSA head {head} slice ends at {end}, past the {}-byte buffer",
            buffer.bytes
        );
        Ok(GgmlIqBuffer {
            ptr: DevicePtr(
                buffer
                    .ptr
                    .0
                    .checked_add(offset)
                    .context("GLM DSA head address overflow")?,
            ),
            bytes: usize::try_from(bytes)?,
        })
    }

    fn head_slice_rows(
        buffer: GgmlIqBuffer,
        head: u32,
        rows: u32,
        width: u32,
    ) -> Result<GgmlIqBuffer> {
        let bytes = u64::from(rows)
            .checked_mul(u64::from(width) * 2)
            .context("GLM DSA wide head width overflow")?;
        let offset = u64::from(head)
            .checked_mul(bytes)
            .context("GLM DSA wide head offset overflow")?;
        ensure!(
            offset + bytes <= buffer.bytes as u64,
            "GLM DSA wide head slice exceeds its buffer"
        );
        Ok(GgmlIqBuffer {
            ptr: DevicePtr(buffer.ptr.0 + offset),
            bytes: usize::try_from(bytes)?,
        })
    }

    fn exact_row_slice(
        buffer: GgmlIqBuffer,
        rows: u32,
        row: usize,
        row_bytes: usize,
        name: &str,
    ) -> Result<GgmlIqBuffer> {
        ensure!(
            row < rows as usize,
            "GLM EXL3 DSA {name} row {row} is outside {rows} rows"
        );
        ensure!(
            buffer.bytes == rows as usize * row_bytes,
            "GLM EXL3 DSA {name} extent is {}, expected {}",
            buffer.bytes,
            rows as usize * row_bytes
        );
        Ok(GgmlIqBuffer {
            ptr: buffer.ptr.offset(row * row_bytes),
            bytes: row_bytes,
        })
    }

    /// Compute only row-independent DSA inputs for a wide prompt or an
    /// explicitly admitted exact verifier. Exact scopes retain row-exact projections;
    /// layer-major prefill uses the regular large-M projection kernels.
    /// Everything that reads or advances causal DSA state stays in
    /// `stage_inner`, one token at a time.
    #[allow(clippy::too_many_arguments)]
    fn precompute_wide_exl3_rows(
        &self,
        gpu: &dyn GpuBackend,
        rows: u32,
        weights: &Glm53Exl3DsaWeights,
        input: GgmlIqBuffer,
        buffers: Glm53DsaScratchBuffers,
        projection_scratch: Glm53Exl3ProjectionScratch,
        dense_indexer: Glm53DsaDenseIndexerPlan,
        stream: u64,
    ) -> Result<u32> {
        let exact_wide = glm53_exact_wide_prefill_active() || glm53_exact_verify_active();
        let layer_major = glm53_layer_major_prefill_active();
        let max_rows = if layer_major {
            u32::try_from(GLM53_EXL3_MAX_WIDE_ROWS)?
        } else {
            8
        };
        ensure!(
            (2..=max_rows).contains(&rows),
            "GLM EXL3 DSA precompute rows must be 2..={max_rows}"
        );
        ensure!(
            exact_wide || layer_major,
            "GLM EXL3 DSA precompute requires an exact or layer-major scope"
        );
        if exact_wide {
            ensure!(
                std::env::var("ATLAS_GLM53_EXACT_WIDE_ROWEXACT").as_deref() == Ok("1"),
                "GLM exact-wide DSA precompute requires ATLAS_GLM53_EXACT_WIDE_ROWEXACT=1"
            );
        }
        let weights_ref = DsaWeightsRef::Exl3(weights);
        self.validate_weight_ref(weights_ref)?;
        let q8 = buffers.q8_activation;

        self.linear_dsa_rows(
            gpu,
            rows,
            weights_ref.q_a(),
            input,
            q8,
            buffers.qr_bf16,
            Some(projection_scratch),
            stream,
        )?;
        self.norms.launch(
            gpu,
            Glm53DsaNormProjectionPlan::new(
                Glm53DsaNormProjectionKind::AbsoluteRms1536,
                rows,
                Q_RANK,
                Q_RANK,
                1e-5,
            )?,
            Glm53DsaNormProjectionBuffers {
                input_bf16: buffers.qr_bf16,
                weight_f32: weights_ref.q_a_norm(),
                bias_f32: GgmlIqBuffer {
                    ptr: DevicePtr::NULL,
                    bytes: 0,
                },
                output_bf16: buffers.qr_norm_bf16,
            },
            stream,
        )?;
        self.linear_dsa_rows(
            gpu,
            rows,
            weights_ref.q_b(),
            buffers.qr_norm_bf16,
            q8,
            buffers.q_b_bf16,
            Some(projection_scratch),
            stream,
        )?;
        self.selected.transpose_heads(
            gpu,
            rows,
            HEAD_DIM,
            true,
            Glm53DsaHeadTransposeBuffers {
                input_bf16: buffers.q_b_bf16,
                output_bf16: buffers.head_major_small_bf16,
            },
            stream,
        )?;
        self.absorb_exl3_bank(
            gpu,
            true,
            rows,
            buffers.head_major_small_bf16,
            weights_ref
                .exl3_absorption_bank(true)?
                .context("GLM EXL3 DSA key absorption bank is missing")?,
            buffers.head_major_large_bf16,
            stream,
        )?;
        self.selected.transpose_heads(
            gpu,
            rows,
            LATENT,
            false,
            Glm53DsaHeadTransposeBuffers {
                input_bf16: buffers.head_major_large_bf16,
                output_bf16: buffers.absorbed_q_bf16,
            },
            stream,
        )?;

        if layer_major {
            self.linear_dsa_rows(
                gpu,
                rows,
                weights_ref.indexer_k(),
                input,
                q8,
                buffers.index_k_bf16,
                Some(projection_scratch),
                stream,
            )?;
            self.linear_dsa_rows(
                gpu,
                rows,
                weights_ref.compressor_gate(),
                input,
                q8,
                buffers.index_g_bf16,
                Some(projection_scratch),
                stream,
            )?;
        } else {
            // Preserve exact M=1 cuBLASLt arithmetic for verifier controls,
            // but issue every row before entering the causal loop.
            for row in 0..rows as usize {
                let row_input =
                    Self::exact_row_slice(input, rows, row, HIDDEN as usize * 2, "hidden input")?;
                self.linear_dsa_rows(
                    gpu,
                    1,
                    weights_ref.indexer_k(),
                    row_input,
                    q8,
                    Self::exact_row_slice(
                        buffers.index_k_bf16,
                        rows,
                        row,
                        INDEX_DIM as usize * 2,
                        "indexer_k output",
                    )?,
                    Some(projection_scratch),
                    stream,
                )?;
                self.linear_dsa_rows(
                    gpu,
                    1,
                    weights_ref.compressor_gate(),
                    row_input,
                    q8,
                    Self::exact_row_slice(
                        buffers.index_g_bf16,
                        rows,
                        row,
                        INDEX_DIM as usize * 2,
                        "compressor_gate output",
                    )?,
                    Some(projection_scratch),
                    stream,
                )?;
            }
        }
        self.norms.launch(
            gpu,
            Glm53DsaNormProjectionPlan::new(
                Glm53DsaNormProjectionKind::BiasedLayerNorm128,
                rows,
                INDEX_DIM,
                INDEX_DIM,
                1e-6,
            )?,
            Glm53DsaNormProjectionBuffers {
                input_bf16: buffers.index_k_bf16,
                weight_f32: weights_ref.indexer_k_norm(),
                bias_f32: weights_ref.indexer_k_norm_bias(),
                output_bf16: buffers.index_k_norm_bf16,
            },
            stream,
        )?;

        self.linear_dsa_rows(
            gpu,
            rows,
            weights_ref.kv_a(),
            input,
            q8,
            buffers.kv_cmpr_bf16,
            Some(projection_scratch),
            stream,
        )?;
        self.norms.launch(
            gpu,
            Glm53DsaNormProjectionPlan::new(
                Glm53DsaNormProjectionKind::AbsoluteRms512,
                rows,
                LATENT,
                LATENT,
                1e-5,
            )?,
            Glm53DsaNormProjectionBuffers {
                input_bf16: buffers.kv_cmpr_bf16,
                weight_f32: weights_ref.kv_a_norm(),
                bias_f32: GgmlIqBuffer {
                    ptr: DevicePtr::NULL,
                    bytes: 0,
                },
                output_bf16: buffers.kv_cmpr_norm_bf16,
            },
            stream,
        )?;

        // Dense causal attention does not consume these two temporary outputs.
        // Keep all index keys/gates/norms above for later sparse continuation.
        if !dense_indexer.skip_indexer_query() {
            self.linear_dsa_rows(
                gpu,
                rows,
                weights_ref.indexer_q_b(),
                buffers.qr_norm_bf16,
                q8,
                buffers.index_q_bf16,
                Some(projection_scratch),
                stream,
            )?;
            self.norms.launch(
                gpu,
                Glm53DsaNormProjectionPlan::new(
                    Glm53DsaNormProjectionKind::F32IndexProjection,
                    rows,
                    HIDDEN,
                    INDEX_HEADS,
                    0.0,
                )?,
                Glm53DsaNormProjectionBuffers {
                    input_bf16: input,
                    weight_f32: weights_ref.indexer_proj(),
                    bias_f32: GgmlIqBuffer {
                        ptr: DevicePtr::NULL,
                        bytes: 0,
                    },
                    output_bf16: buffers.head_weights_bf16,
                },
                stream,
            )?;
        }

        // Three required compressed projections use three launches each;
        // q/kv/index norms are three row-independent launches; the query
        // absorption is two exact transposes around one packed-bank launch;
        // the two native projections are large-M only in layer-major mode.
        // Index query/head-weight work contributes four launches unless omitted.
        Ok(3 * 3
            + 3
            + 3
            + dense_indexer.indexer_launches()
            + if layer_major { 2 } else { 2 * rows })
    }

    /// Stage one DSA layer.
    #[allow(clippy::too_many_arguments)]
    pub fn stage(
        &self,
        gpu: &dyn GpuBackend,
        weights: &Glm53DsaWeights,
        input: GgmlIqBuffer,
        buffers: Glm53DsaScratchBuffers,
        cache: Glm53DsaCacheSlots,
        geometry: Glm53DsaLayerGeometry,
        output: GgmlIqBuffer,
        stream: u64,
    ) -> Result<u32> {
        self.stage_inner(
            gpu,
            1,
            DsaWeightsRef::Gguf(weights),
            input,
            buffers,
            None,
            cache,
            geometry,
            output,
            stream,
            DsaStatelessInputs::Compute,
        )
    }

    /// Stage one DSA layer from the pinned EXL3 target checkpoint.
    #[allow(clippy::too_many_arguments)]
    pub fn stage_exl3(
        &self,
        gpu: &dyn GpuBackend,
        weights: &Glm53Exl3DsaWeights,
        input: GgmlIqBuffer,
        buffers: Glm53DsaScratchBuffers,
        projection_scratch: Glm53Exl3ProjectionScratch,
        cache: Glm53DsaCacheSlots,
        geometry: Glm53DsaLayerGeometry,
        output: GgmlIqBuffer,
        stream: u64,
    ) -> Result<u32> {
        self.stage_inner(
            gpu,
            1,
            DsaWeightsRef::Exl3(weights),
            input,
            buffers,
            Some(projection_scratch),
            cache,
            geometry,
            output,
            stream,
            DsaStatelessInputs::Compute,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn stage_exl3_rows(
        &self,
        gpu: &dyn GpuBackend,
        rows: u32,
        weights: &Glm53Exl3DsaWeights,
        input: GgmlIqBuffer,
        buffers: Glm53DsaScratchBuffers,
        projection_scratch: Glm53Exl3ProjectionScratch,
        cache: Glm53DsaCacheSlots,
        geometry: Glm53DsaLayerGeometry,
        output: GgmlIqBuffer,
        stream: u64,
    ) -> Result<u32> {
        let exact_verify = glm53_exact_verify_active();
        let exact_wide = glm53_exact_wide_prefill_active() || exact_verify;
        let layer_major = glm53_layer_major_prefill_active();
        let verify_precompute_flag = match std::env::var("ATLAS_GLM53_DSA_VERIFY_PRECOMPUTE") {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => {
                return Err(error).context("GLM DSA verifier precompute flag must be UTF-8");
            }
        };
        let verify_precompute = dsa_verify_precompute::select(
            verify_precompute_flag.as_deref(),
            rows,
            exact_verify,
            glm53_exact_wide_prefill_active() || layer_major,
            std::env::var("ATLAS_GLM53_EXACT_WIDE_ROWEXACT").as_deref() == Ok("1"),
        )?;
        let max_rows = if layer_major {
            u32::try_from(GLM53_EXL3_MAX_WIDE_ROWS)?
        } else {
            8
        };
        ensure!(
            (1..=max_rows).contains(&rows),
            "GLM EXL3 DSA rows must be 1..={max_rows}"
        );
        geometry.validate()?;
        let skip_requested = match std::env::var("ATLAS_GLM53_DSA_DENSE_INDEXER_SKIP") {
            Ok(value) => parse_dense_indexer_skip(Some(value.as_str()))?,
            Err(std::env::VarError::NotPresent) => parse_dense_indexer_skip(None)?,
            Err(error) => return Err(error).context("GLM DSA dense indexer flag must be UTF-8"),
        };
        let dense_indexer = Glm53DsaDenseIndexerPlan::new(
            rows,
            geometry.position,
            geometry.capacity,
            layer_major,
            skip_requested,
        )?;
        if rows > 1 && layer_major {
            geometry.validate()?;
            ensure!(
                geometry
                    .position
                    .checked_add(rows)
                    .context("GLM EXL3 layer-major DSA position overflow")?
                    <= geometry.capacity,
                "GLM EXL3 layer-major DSA chunk exceeds capacity"
            );
            let mut launches = self.precompute_wide_exl3_rows(
                gpu,
                rows,
                weights,
                input,
                buffers,
                projection_scratch,
                dense_indexer,
                stream,
            )?;
            launches = launches
                .checked_add(self.stage_inner(
                    gpu,
                    rows,
                    DsaWeightsRef::Exl3(weights),
                    input,
                    buffers,
                    Some(projection_scratch),
                    cache,
                    geometry,
                    output,
                    stream,
                    DsaStatelessInputs::PrecomputedWide,
                )?)
                .context("GLM EXL3 layer-major DSA launch count overflow")?;
            return Ok(launches);
        }
        if rows > 1 && exact_wide {
            // Only stateless inputs move ahead of the existing causal M1 loop.
            // The verifier opt-in is independent of the unchanged prompt flag.
            let precompute = if exact_verify {
                verify_precompute
            } else {
                layer_major
                    || std::env::var("ATLAS_GLM53_EXACT_WIDE_DSA_PRECOMPUTE").as_deref() == Ok("1")
            };
            geometry.validate()?;
            ensure!(
                geometry
                    .position
                    .checked_add(rows)
                    .context("GLM EXL3 exact-wide DSA position overflow")?
                    <= geometry.capacity,
                "GLM EXL3 exact-wide DSA chunk exceeds capacity"
            );
            let mut launches = if precompute {
                self.precompute_wide_exl3_rows(
                    gpu,
                    rows,
                    weights,
                    input,
                    buffers,
                    projection_scratch,
                    dense_indexer,
                    stream,
                )?
            } else {
                0
            };
            for row in 0..rows as usize {
                let slice = |buffer: GgmlIqBuffer, row_bytes: usize| -> Result<GgmlIqBuffer> {
                    ensure!(
                        buffer.bytes == rows as usize * row_bytes,
                        "GLM EXL3 serial-wide DSA buffer extent drift"
                    );
                    Ok(GgmlIqBuffer {
                        ptr: buffer.ptr.offset(row * row_bytes),
                        bytes: row_bytes,
                    })
                };
                let row_buffers = Glm53DsaScratchBuffers {
                    qr_bf16: slice(buffers.qr_bf16, 3_072)?,
                    qr_norm_bf16: slice(buffers.qr_norm_bf16, 3_072)?,
                    q_b_bf16: slice(buffers.q_b_bf16, 32_768)?,
                    absorbed_q_bf16: slice(buffers.absorbed_q_bf16, 65_536)?,
                    kv_cmpr_bf16: slice(buffers.kv_cmpr_bf16, 1_024)?,
                    kv_cmpr_norm_bf16: slice(buffers.kv_cmpr_norm_bf16, 1_024)?,
                    index_k_bf16: slice(buffers.index_k_bf16, 256)?,
                    index_k_norm_bf16: slice(buffers.index_k_norm_bf16, 256)?,
                    index_g_bf16: slice(buffers.index_g_bf16, 256)?,
                    index_q_bf16: slice(buffers.index_q_bf16, 8_192)?,
                    head_weights_bf16: slice(buffers.head_weights_bf16, 64)?,
                    scores_f32: slice(buffers.scores_f32, 1_048_576)?,
                    selected_indices_i32: slice(buffers.selected_indices_i32, 8_204)?,
                    weighted_latent_bf16: slice(buffers.weighted_latent_bf16, 65_536)?,
                    unabsorbed_bf16: slice(buffers.unabsorbed_bf16, 32_768)?,
                    pool_keys_bf16: buffers.pool_keys_bf16,
                    pool_validity_u8: buffers.pool_validity_u8,
                    tail_keys_bf16: buffers.tail_keys_bf16,
                    tail_gates_bf16: buffers.tail_gates_bf16,
                    q8_activation: buffers.q8_activation,
                    head_major_large_bf16: buffers.head_major_large_bf16,
                    head_major_small_bf16: buffers.head_major_small_bf16,
                    sequence_length_u32: buffers.sequence_length_u32,
                    query_positions_u32: buffers.query_positions_u32,
                    query_validity_u8: buffers.query_validity_u8,
                    tail_validity_u8: buffers.tail_validity_u8,
                };
                let row_launches = if precompute {
                    self.stage_inner(
                        gpu,
                        1,
                        DsaWeightsRef::Exl3(weights),
                        slice(input, HIDDEN as usize * 2)?,
                        row_buffers,
                        Some(projection_scratch),
                        cache,
                        Glm53DsaLayerGeometry {
                            position: geometry
                                .position
                                .checked_add(u32::try_from(row)?)
                                .context("GLM EXL3 serial-wide DSA position overflow")?,
                            ..geometry
                        },
                        slice(output, HIDDEN as usize * 2)?,
                        stream,
                        DsaStatelessInputs::PrecomputedWide,
                    )?
                } else {
                    self.stage_exl3_rows(
                        gpu,
                        1,
                        weights,
                        slice(input, HIDDEN as usize * 2)?,
                        row_buffers,
                        projection_scratch,
                        cache,
                        Glm53DsaLayerGeometry {
                            position: geometry
                                .position
                                .checked_add(u32::try_from(row)?)
                                .context("GLM EXL3 serial-wide DSA position overflow")?,
                            ..geometry
                        },
                        slice(output, HIDDEN as usize * 2)?,
                        stream,
                    )?
                };
                launches = launches
                    .checked_add(row_launches)
                    .context("GLM EXL3 serial-wide DSA launch count overflow")?;
            }
            return Ok(launches);
        }
        self.stage_inner(
            gpu,
            rows,
            DsaWeightsRef::Exl3(weights),
            input,
            buffers,
            Some(projection_scratch),
            cache,
            geometry,
            output,
            stream,
            DsaStatelessInputs::Compute,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn stage_inner(
        &self,
        gpu: &dyn GpuBackend,
        rows: u32,
        weights: DsaWeightsRef<'_>,
        input: GgmlIqBuffer,
        buffers: Glm53DsaScratchBuffers,
        projection_scratch: Option<Glm53Exl3ProjectionScratch>,
        cache: Glm53DsaCacheSlots,
        geometry: Glm53DsaLayerGeometry,
        output: GgmlIqBuffer,
        stream: u64,
        stateless_inputs: DsaStatelessInputs,
    ) -> Result<u32> {
        self.validate_weight_ref(weights)?;
        geometry.validate()?;
        if stateless_inputs == DsaStatelessInputs::PrecomputedWide {
            ensure!(
                matches!(weights, DsaWeightsRef::Exl3(_))
                    && (rows == 1
                        || (glm53_layer_major_prefill_active()
                            && rows <= u32::try_from(GLM53_EXL3_MAX_WIDE_ROWS)?)),
                "GLM DSA precomputed inputs require an EXL3 tokenwise or layer-major stage"
            );
        }
        let q8 = buffers.q8_activation;

        // Per-token query metadata. topk and selected-attention read these to
        // bound valid indices and apply the causal mask; the arena starts zeroed
        // and nothing else writes them, so without this every DSA layer runs
        // with sequence_length=0 / query_validity=0 — attention over nothing,
        // silently. sequence_length includes the current token because its
        // latent row is published before attending (below).
        let sequence_length = geometry
            .position
            .checked_add(rows)
            .context("GLM DSA sequence length overflow")?;
        ensure!(
            sequence_length <= geometry.capacity,
            "GLM DSA chunk exceeds capacity"
        );
        let dense_full_coverage = dense_full_coverage(
            rows,
            geometry.position,
            geometry.capacity,
            glm53_layer_major_prefill_active(),
        )?;
        let (sequence_lengths, query_positions, query_validity, tail_validity) = if rows == 1 {
            (
                cache.sequence_lengths_u32,
                cache.query_positions_u32,
                cache.query_validity_u8,
                // Scalar metadata leaves this untouched. Pool writes the
                // current validity before top-k; prior remains its input.
                cache.out_tail_validity_u8,
            )
        } else {
            (
                buffers.sequence_length_u32,
                buffers.query_positions_u32,
                buffers.query_validity_u8,
                buffers.tail_validity_u8,
            )
        };
        KernelLaunch::new(gpu, self.prepare_metadata)
            .grid([1, 1, 1])
            .block([if rows > 8 { 128 } else { 8 }, 1, 1])
            .arg_ptr(sequence_lengths.ptr)
            .arg_ptr(query_positions.ptr)
            .arg_ptr(query_validity.ptr)
            .arg_ptr(tail_validity.ptr)
            .arg_u32(geometry.position)
            .arg_u32(rows)
            .launch(stream)?;
        let mut launches = 0u32;

        // Query: low-rank down-projection, RMS norm, up-projection. An exact
        // wide prompt may have populated these three row-local buffers before
        // entering the token-ordered causal stage.
        if stateless_inputs == DsaStatelessInputs::Compute {
            self.linear_dsa_rows(
                gpu,
                rows,
                weights.q_a(),
                input,
                q8,
                buffers.qr_bf16,
                projection_scratch,
                stream,
            )?;
            self.norms.launch(
                gpu,
                Glm53DsaNormProjectionPlan::new(
                    Glm53DsaNormProjectionKind::AbsoluteRms1536,
                    rows,
                    Q_RANK,
                    Q_RANK,
                    1e-5,
                )?,
                Glm53DsaNormProjectionBuffers {
                    input_bf16: buffers.qr_bf16,
                    weight_f32: weights.q_a_norm(),
                    bias_f32: GgmlIqBuffer {
                        ptr: DevicePtr::NULL,
                        bytes: 0,
                    },
                    output_bf16: buffers.qr_norm_bf16,
                },
                stream,
            )?;
            self.linear_dsa_rows(
                gpu,
                rows,
                weights.q_b(),
                buffers.qr_norm_bf16,
                q8,
                buffers.q_b_bf16,
                projection_scratch,
                stream,
            )?;
            launches += 2 * 2 + 1;
        }

        // Absorb W_kb into the query, one head at a time. Exact-wide
        // precompute may already have run the packed bank across all rows.
        if stateless_inputs == DsaStatelessInputs::Compute {
            // Wide GEMMs require contiguous rows for each head, so transpose
            // around the head bank.
            if rows > 1 {
                self.selected.transpose_heads(
                    gpu,
                    rows,
                    HEAD_DIM,
                    true,
                    Glm53DsaHeadTransposeBuffers {
                        input_bf16: buffers.q_b_bf16,
                        output_bf16: buffers.head_major_small_bf16,
                    },
                    stream,
                )?;
            }
            let key_source = if rows == 1 {
                buffers.q_b_bf16
            } else {
                buffers.head_major_small_bf16
            };
            let key_destination = if rows == 1 {
                buffers.absorbed_q_bf16
            } else {
                buffers.head_major_large_bf16
            };
            if let Some(weight) = weights.exl3_absorption_bank(true)? {
                self.absorb_exl3_bank(
                    gpu,
                    true,
                    rows,
                    key_source,
                    weight,
                    key_destination,
                    stream,
                )?;
                launches += 1;
            } else {
                for head in 0..HEADS {
                    let source = Self::head_slice_rows(key_source, head, rows, HEAD_DIM)?;
                    let destination = Self::head_slice_rows(key_destination, head, rows, LATENT)?;
                    self.linear_dsa_rows(
                        gpu,
                        rows,
                        weights.k_b(head)?,
                        source,
                        q8,
                        destination,
                        projection_scratch,
                        stream,
                    )?;
                    launches += 2;
                }
            }
            if rows > 1 {
                self.selected.transpose_heads(
                    gpu,
                    rows,
                    LATENT,
                    false,
                    Glm53DsaHeadTransposeBuffers {
                        input_bf16: buffers.head_major_large_bf16,
                        output_bf16: buffers.absorbed_q_bf16,
                    },
                    stream,
                )?;
            }
        }

        // Compressed KV for this token. This is likewise row-local and may be
        // supplied by the exact-wide precompute.
        if stateless_inputs == DsaStatelessInputs::Compute {
            self.linear_dsa_rows(
                gpu,
                rows,
                weights.kv_a(),
                input,
                q8,
                buffers.kv_cmpr_bf16,
                projection_scratch,
                stream,
            )?;
            self.norms.launch(
                gpu,
                Glm53DsaNormProjectionPlan::new(
                    Glm53DsaNormProjectionKind::AbsoluteRms512,
                    rows,
                    LATENT,
                    LATENT,
                    1e-5,
                )?,
                Glm53DsaNormProjectionBuffers {
                    input_bf16: buffers.kv_cmpr_bf16,
                    weight_f32: weights.kv_a_norm(),
                    bias_f32: GgmlIqBuffer {
                        ptr: DevicePtr::NULL,
                        bytes: 0,
                    },
                    output_bf16: buffers.kv_cmpr_norm_bf16,
                },
                stream,
            )?;
            launches += 3;
        }

        // Indexer key/gate are row-local and may already be populated by the
        // exact-wide precompute. Their native projections remain M=1 there.
        if stateless_inputs == DsaStatelessInputs::Compute {
            self.linear_dsa_rows(
                gpu,
                rows,
                weights.indexer_k(),
                input,
                q8,
                buffers.index_k_bf16,
                projection_scratch,
                stream,
            )?;
            self.norms.launch(
                gpu,
                Glm53DsaNormProjectionPlan::new(
                    Glm53DsaNormProjectionKind::BiasedLayerNorm128,
                    rows,
                    INDEX_DIM,
                    INDEX_DIM,
                    1e-6,
                )?,
                Glm53DsaNormProjectionBuffers {
                    input_bf16: buffers.index_k_bf16,
                    weight_f32: weights.indexer_k_norm(),
                    bias_f32: weights.indexer_k_norm_bias(),
                    output_bf16: buffers.index_k_norm_bf16,
                },
                stream,
            )?;
            self.linear_dsa_rows(
                gpu,
                rows,
                weights.compressor_gate(),
                input,
                q8,
                buffers.index_g_bf16,
                projection_scratch,
                stream,
            )?;
            launches += 2 * 2 + 1;
        }

        let pool_plan =
            // `max_positions` is the ARCHITECTURAL address-space bound, not this
            // sequence's capacity: the pool op validates it equals exactly
            // GLM53_DSA_ABSOLUTE_POSITIONS and uses it only to bounds-check the
            // absolute position. Passing the runtime capacity here is a category
            // error that refuses every DSA layer. The capacity-dependent ops
            // below (topk, selected attention, latent append) do take
            // `geometry.capacity`.
            Glm53DsaPoolPlan::new(
                1,
                rows,
                geometry.position,
                INDEX_DIM,
                KPOOL,
                GLM53_DSA_ABSOLUTE_POSITIONS,
            )?;
        let empty = GgmlIqBuffer {
            ptr: DevicePtr::NULL,
            bytes: 0,
        };
        self.pool.launch(
            gpu,
            pool_plan,
            Glm53DsaPoolBuffers {
                input_keys_bf16: buffers.index_k_norm_bf16,
                input_gates_bf16: buffers.index_g_bf16,
                input_validity_u8: query_validity,
                prior_tail_keys_bf16: cache.prior_tail_keys_bf16,
                prior_tail_gates_bf16: cache.prior_tail_gates_bf16,
                prior_tail_validity_u8: cache.prior_tail_validity_u8,
                ape_f32: weights.compressor_ape(),
                // A step that completes no pool writes nothing here, and the op
                // requires a NULL/0 buffer in that case rather than a stale one.
                // The op validates EXACT extents, but these are fixed-size
                // scratch slots (256 B each) while the plan's pool extents grow
                // with complete_pools. Hand it exactly what it asks for, sliced
                // out of the scratch, or the first step that completes a pool
                // (position 4) fails the extent guard. Found by the multi-token
                // generation test; the single-token parity test never reaches
                // complete_pools > 0.
                output_pool_keys_bf16: if pool_plan.complete_pools == 0 {
                    empty
                } else {
                    ensure!(
                        buffers.pool_keys_bf16.bytes >= pool_plan.pool_vector_bytes,
                        "GLM DSA pool-keys scratch is {} bytes, needs {}",
                        buffers.pool_keys_bf16.bytes,
                        pool_plan.pool_vector_bytes
                    );
                    GgmlIqBuffer {
                        ptr: buffers.pool_keys_bf16.ptr,
                        bytes: pool_plan.pool_vector_bytes,
                    }
                },
                output_pool_validity_u8: if pool_plan.complete_pools == 0 {
                    empty
                } else {
                    ensure!(
                        buffers.pool_validity_u8.bytes >= pool_plan.pool_validity_bytes,
                        "GLM DSA pool-validity scratch is {} bytes, needs {}",
                        buffers.pool_validity_u8.bytes,
                        pool_plan.pool_validity_bytes
                    );
                    GgmlIqBuffer {
                        ptr: buffers.pool_validity_u8.ptr,
                        bytes: pool_plan.pool_validity_bytes,
                    }
                },
                output_tail_keys_bf16: buffers.tail_keys_bf16,
                output_tail_gates_bf16: buffers.tail_gates_bf16,
                output_tail_validity_u8: cache.out_tail_validity_u8,
            },
            stream,
        )?;
        launches += 1;

        // PUBLISH THE COMPLETING POOL BEFORE SCORE/TOPK.
        //
        // The pool op writes a completed pool to per-step SCRATCH
        // (`buffers.pool_keys_bf16`); score and top-k read the CACHE
        // (`cache.pool_keys_bf16`). Publishing this at the END of the step made
        // the pool completing during THIS step invisible to this step's
        // selector, and the raw tail could not cover it either: the tail holds
        // at most KPOOL-1 tokens and at q = KPOOL-1 (mod KPOOL) the completing
        // group is full. Both routes missed, so at q = 7, 11, 15, ... the model
        // attended to a sequence with a KPOOL-1-token HOLE in it -- measured
        // directly in dump_dsafix layer 3: p7 selected [0,1,2,3] and nothing
        // else, with 4, 5 and 6 absent from selected_indices entirely.
        //
        // Publishing here makes `usable_pools()` = (q+1)/KPOOL sound. The two
        // must move together: the (q+1)/KPOOL count on its own points the
        // scorer at an unwritten cache row, which is the stale-row failure the
        // causality guard below exists to catch and which measured p3 at 3.16%
        // -> 44.98%.
        //
        // The TAIL advance stays at the end of the step, unmoved: it is a
        // cross-position carry and its ordering comment there is correct.
        if pool_plan.complete_pools > 0 {
            // The pool covering positions 4k..4k+3 completes at 4k+3, so
            // its row is position / 4.
            let pool_row = u64::from(geometry.position / KPOOL);
            let key_bytes = u64::from(INDEX_DIM) * 2;
            let key_offset = pool_row
                .checked_mul(key_bytes)
                .context("GLM DSA pool key row overflow")?;
            ensure!(
                key_offset + u64::from(pool_plan.complete_pools) * key_bytes
                    <= cache.pool_keys_bf16.bytes as u64,
                "GLM DSA pool rows from {pool_row} are outside the persistent pool region"
            );
            gpu.copy_d2d_async(
                buffers.pool_keys_bf16.ptr,
                DevicePtr(cache.pool_keys_bf16.ptr.0 + key_offset),
                pool_plan.pool_vector_bytes,
                stream,
            )?;
            ensure!(
                pool_row + u64::from(pool_plan.complete_pools)
                    <= cache.pool_validity_u8.bytes as u64,
                "GLM DSA pool validity row {pool_row} is outside its region"
            );
            gpu.copy_d2d_async(
                buffers.pool_validity_u8.ptr,
                DevicePtr(cache.pool_validity_u8.ptr.0 + pool_row),
                pool_plan.pool_validity_bytes,
                stream,
            )?;
        }

        // Indexer query and the exact row-independent F32 head projection may
        // already be populated by exact-wide precompute.
        if stateless_inputs == DsaStatelessInputs::Compute {
            self.linear_dsa_rows(
                gpu,
                rows,
                weights.indexer_q_b(),
                buffers.qr_norm_bf16,
                q8,
                buffers.index_q_bf16,
                projection_scratch,
                stream,
            )?;
            self.norms.launch(
                gpu,
                Glm53DsaNormProjectionPlan::new(
                    Glm53DsaNormProjectionKind::F32IndexProjection,
                    rows,
                    HIDDEN,
                    INDEX_HEADS,
                    0.0,
                )?,
                Glm53DsaNormProjectionBuffers {
                    input_bf16: input,
                    weight_f32: weights.indexer_proj(),
                    bias_f32: GgmlIqBuffer {
                        ptr: DevicePtr::NULL,
                        bytes: 0,
                    },
                    output_bf16: buffers.head_weights_bf16,
                },
                stream,
            )?;
            launches += 3;
        }

        // Score every pool, then take the top-512 and expand to token indices.
        //
        // A pool only completes every `kpool` positions, so the first KPOOL-1
        // tokens of ANY sequence (q = 0, 1, 2) have zero complete pools — and
        // both the score and top-k ops require `pools >= 1`. There is nothing
        // to select from yet: every visible token is still in the raw tail.
        // Position 3 is NOT in this set: it completes pool 0, which is
        // published above and scored here. The selector's
        // output for that case is exactly "the causally visible prefix", which
        // is written directly rather than asking the ops for a top-k over an
        // empty pool set.
        let pools = sequence_length / KPOOL;
        // The causality rule, ENFORCED rather than left in a test's reference
        // model: pool k spans 4k..4k+3 and is usable only once its last token
        // has arrived. The highest usable pool is `pools - 1`, whose final
        // token sits at KPOOL*pools - 1, so that must not exceed the query
        // position. At q ≡ KPOOL-1 this is an equality: the pool completing on
        // this step is usable precisely because its last token IS the query,
        // and its row was published above before this point.
        // Scoring a pool whose tail has not arrived reads stale rows
        // and produces fluent, wrong output rather than an error -- which is
        // exactly how the previous off-by-one survived, silently and for the
        // life of the model.
        ensure!(
            pools == 0 || KPOOL * pools - 1 < sequence_length,
            "GLM DSA pool causality violated: {pools} usable pools implies a last \
             token at position {}, but the query is at {}",
            KPOOL * pools - 1,
            sequence_length - 1
        );
        // THE COVERAGE INVARIANT, enforced rather than assumed: every position
        // 0..=q must be accounted for exactly once, either by a published pool
        // or by a slot in the raw tail. Pooled coverage ends at KPOOL*pools - 1,
        // so the tail must hold the remainder.
        //
        // With pools = (q+1)/KPOOL the occupancy is (q+1) - KPOOL*pools, which
        // is exactly (q+1) mod KPOOL and therefore never exceeds KPOOL-1 =
        // GLM53_DSA_TAIL_CAPACITY. It reaches zero at q ≡ KPOOL-1 (mod KPOOL),
        // where the completing group is fully pooled instead.
        //
        // Under the OLD pools = q/KPOOL this same expression demanded KPOOL
        // slots at q ≡ KPOOL-1, which is what made a capacity of KPOOL look
        // necessary. Raising the capacity would have been a divergence from
        // the reference; the pool count was the wrong half.
        {
            let pooled_coverage = KPOOL * pools;
            let tail_occupancy = sequence_length - pooled_coverage;
            ensure!(
                pooled_coverage + tail_occupancy == sequence_length,
                "GLM DSA coverage invariant violated: {pooled_coverage} pooled + \
                 {tail_occupancy} tail != {} positions",
                sequence_length
            );
            ensure!(
                tail_occupancy <= spark_runtime::kv_cache::GLM53_DSA_TAIL_CAPACITY,
                "GLM DSA raw tail must hold {tail_occupancy} tokens at position {} \
                 ({pooled_coverage} covered by {pools} published pools) but capacity \
                 is {}. The tail is short exactly when position = KPOOL-1 (mod KPOOL), \
                 where the completing group is full and not yet published.",
                geometry.position,
                spark_runtime::kv_cache::GLM53_DSA_TAIL_CAPACITY
            );
        }
        if dense_full_coverage {
            // Top-512 pools cover 2048 tokens and the raw tail covers three,
            // so every causally visible token is selected in this region.
            // The dense causal attention below consumes the latent cache
            // directly; score/top-k would only reconstruct the same prefix.
        } else if pools == 0 {
            let mut indices = vec![-1i32; usize::try_from(rows * SELECTED)?];
            for row in 0..rows {
                let visible = usize::try_from(geometry.position + row + 1)?;
                ensure!(
                    visible <= usize::try_from(SELECTED)?,
                    "GLM DSA cannot have {visible} visible tokens with no complete pools"
                );
                let base = usize::try_from(row * SELECTED)?;
                for (slot, index) in indices[base..base + visible].iter_mut().enumerate() {
                    *index = i32::try_from(slot)?;
                }
            }
            let bytes: Vec<u8> = indices.iter().flat_map(|v| v.to_le_bytes()).collect();
            ensure!(
                bytes.len() <= buffers.selected_indices_i32.bytes,
                "GLM DSA selector buffer holds {} bytes, need {}",
                buffers.selected_indices_i32.bytes,
                bytes.len()
            );
            gpu.copy_h2d(&bytes, buffers.selected_indices_i32.ptr)?;
        } else {
            let score_plan = Glm53DsaScorePlan::new(1, rows, pools, INDEX_HEADS, INDEX_DIM)?;
            ensure!(
                buffers.scores_f32.bytes >= score_plan.output_bytes,
                "GLM DSA score buffer holds {} bytes, {pools} pools need {}",
                buffers.scores_f32.bytes,
                score_plan.output_bytes
            );
            ensure!(
                cache.pool_keys_bf16.bytes >= score_plan.pool_key_bytes
                    && cache.pool_validity_u8.bytes >= score_plan.pool_validity_bytes,
                "GLM DSA pool cache holds {}/{} bytes, {pools} pools need {}/{}",
                cache.pool_keys_bf16.bytes,
                cache.pool_validity_u8.bytes,
                score_plan.pool_key_bytes,
                score_plan.pool_validity_bytes
            );
            let scores = GgmlIqBuffer {
                ptr: buffers.scores_f32.ptr,
                bytes: score_plan.output_bytes,
            };
            self.score.launch(
                gpu,
                score_plan,
                Glm53DsaScoreBuffers {
                    queries_bf16: buffers.index_q_bf16,
                    head_weights_bf16: buffers.head_weights_bf16,
                    // Same exact-extent rule as `scores` above: the cache
                    // buffers are sized for full context capacity, but the op
                    // validates against this step's pool count.
                    pool_keys_bf16: GgmlIqBuffer {
                        ptr: cache.pool_keys_bf16.ptr,
                        bytes: score_plan.pool_key_bytes,
                    },
                    pool_validity_u8: GgmlIqBuffer {
                        ptr: cache.pool_validity_u8.ptr,
                        bytes: score_plan.pool_validity_bytes,
                    },
                    output_scores_f32: scores,
                },
                stream,
            )?;
            // Every buffer below is extent-validated by the op against THIS
            // step's pool count, while the cache buffers are sized for full
            // context capacity. Slice each to exactly what the plan asks for.
            let topk_plan =
                Glm53DsaTopkPlan::new(1, rows, pools, geometry.capacity, INDEX_TOPK, KPOOL, true)?;
            let slice = |b: GgmlIqBuffer, want: usize, name: &str| -> Result<GgmlIqBuffer> {
                ensure!(
                    b.bytes >= want,
                    "GLM DSA top-k {name} cache holds {} bytes, {pools} pools need {want}",
                    b.bytes
                );
                Ok(GgmlIqBuffer {
                    ptr: b.ptr,
                    bytes: want,
                })
            };
            self.topk.launch(
                gpu,
                topk_plan,
                Glm53DsaTopkBuffers {
                    scores_f32: scores,
                    pool_validity_u8: slice(
                        cache.pool_validity_u8,
                        topk_plan.pool_validity_bytes,
                        "pool validity",
                    )?,
                    sequence_lengths_u32: slice(
                        sequence_lengths,
                        topk_plan.sequence_length_bytes,
                        "sequence lengths",
                    )?,
                    query_positions_u32: slice(
                        query_positions,
                        topk_plan.query_position_bytes,
                        "query positions",
                    )?,
                    query_validity_u8: slice(
                        query_validity,
                        topk_plan.query_validity_bytes,
                        "query validity",
                    )?,
                    tail_validity_u8: slice(
                        tail_validity,
                        topk_plan.tail_validity_bytes,
                        "tail validity",
                    )?,
                    output_indices_i32: buffers.selected_indices_i32,
                },
                stream,
            )?;
            launches += 2;
        }

        // Make the current token visible to itself before attending.
        //
        // Causal self-attention requires row `position` of the latent cache to
        // hold THIS token's compressed KV. The reference graph gets that from
        // its `cpy_k` into the cache before `build_attn`. The receipted way to
        // do it here is `glm53_dsa_current_visibility`, which materializes
        // physically-future rows for all 11 layers at once under a generation/
        // nonce gate. That is a walk-level restructure; until it lands, the
        // bring-up path publishes the single current row directly. This is the
        // second documented shortcut past the receipt machinery (the first is
        // `commit_accepted`) and is unreachable with admission closed.
        {
            let row = u64::from(geometry.position)
                .checked_mul(u64::from(LATENT) * 2)
                .context("GLM DSA latent row offset overflow")?;
            let destination = DevicePtr(
                cache
                    .latent_cache_bf16
                    .ptr
                    .0
                    .checked_add(row)
                    .context("GLM DSA latent row address overflow")?,
            );
            let chunk_bytes = u64::from(rows) * u64::from(LATENT) * 2;
            ensure!(
                row + chunk_bytes <= cache.latent_cache_bf16.bytes as u64,
                "GLM DSA position {} is outside the {}-byte latent cache",
                geometry.position,
                cache.latent_cache_bf16.bytes
            );
            // MUST be stream-ordered. `copy_d2d` issues on the DEFAULT stream
            // and synchronizes only that stream, so it does NOT wait for the
            // kernel on `stream` that just wrote `kv_cmpr_norm_bf16` -- it
            // copies the buffer's PREVIOUS contents, i.e. the previous DSA
            // layer's latent. Measured: every layer's cache row 0 held the
            // prior layer's kv_cmpr_norm, and layer 3 (the first) held zero,
            // so its attention returned nothing at all.
            gpu.copy_d2d_async(
                buffers.kv_cmpr_norm_bf16.ptr,
                destination,
                usize::try_from(chunk_bytes)?,
                stream,
            )?;
        }

        // Attention over the selected latent rows, in absorbed space.
        let selected_plan = Glm53DsaSelectedAttentionPlan::new(
            1,
            rows,
            geometry.capacity,
            HEADS,
            LATENT,
            SELECTED,
            Glm53DsaSelectedStorage::Bf16,
        )?;
        let selected_buffers = Glm53DsaSelectedAttentionBuffers {
            absorbed_query_bf16: buffers.absorbed_q_bf16,
            latent_cache_bf16: cache.latent_cache_bf16,
            selected_indices_i32: buffers.selected_indices_i32,
            sequence_lengths_u32: sequence_lengths,
            query_positions_u32: query_positions,
            query_validity_u8: query_validity,
            output_weighted_latent_bf16: buffers.weighted_latent_bf16,
        };
        if dense_full_coverage {
            self.selected.launch_dense_causal(
                gpu,
                selected_plan,
                selected_buffers,
                geometry.position,
                sequence_length,
                stream,
            )?;
        } else {
            self.selected
                .launch(gpu, selected_plan, selected_buffers, stream)?;
        }
        launches += 1;
        {
            // Bring-up diagnostic (ATLAS_GLM53_DUMP_DIR): DSA intermediates, so a
            // zero attention output can be attributed to a gate rather than guessed.
            use std::sync::atomic::{AtomicU32, Ordering};
            static DSA_CALLS: AtomicU32 = AtomicU32::new(0);
            let call = DSA_CALLS.fetch_add(1, Ordering::Relaxed);
            let layer = 3 + 4 * (call % 11);
            super::walk_dump::Glm53WalkDump::from_env().write(
                gpu,
                stream,
                geometry.position,
                &format!("dsa-layer{layer}"),
                &[
                    // Row 0 of THIS layer's latent cache: the row the publish
                    // above writes and the row selected_indices=[0,..] reads.
                    (
                        "latent_row0",
                        GgmlIqBuffer {
                            ptr: cache.latent_cache_bf16.ptr,
                            bytes: usize::try_from(u64::from(LATENT) * 2)?,
                        },
                        super::walk_dump::Glm53DumpDtype::Bf16,
                    ),
                    (
                        "absorbed_q",
                        buffers.absorbed_q_bf16,
                        super::walk_dump::Glm53DumpDtype::Bf16,
                    ),
                    (
                        "kv_cmpr_norm",
                        buffers.kv_cmpr_norm_bf16,
                        super::walk_dump::Glm53DumpDtype::Bf16,
                    ),
                    (
                        "selected_indices",
                        buffers.selected_indices_i32,
                        super::walk_dump::Glm53DumpDtype::U32,
                    ),
                    (
                        "scores",
                        buffers.scores_f32,
                        super::walk_dump::Glm53DumpDtype::F32,
                    ),
                    (
                        "weighted_latent",
                        buffers.weighted_latent_bf16,
                        super::walk_dump::Glm53DumpDtype::Bf16,
                    ),
                    (
                        "sequence_lengths",
                        cache.sequence_lengths_u32,
                        super::walk_dump::Glm53DumpDtype::U32,
                    ),
                    (
                        "query_positions",
                        cache.query_positions_u32,
                        super::walk_dump::Glm53DumpDtype::U32,
                    ),
                    (
                        "query_validity",
                        cache.query_validity_u8,
                        super::walk_dump::Glm53DumpDtype::U32,
                    ),
                ],
            )?;
        }

        // Un-absorb W_vb, one head at a time, then project out.
        if rows > 1 {
            self.selected.transpose_heads(
                gpu,
                rows,
                LATENT,
                true,
                Glm53DsaHeadTransposeBuffers {
                    input_bf16: buffers.weighted_latent_bf16,
                    output_bf16: buffers.head_major_large_bf16,
                },
                stream,
            )?;
        }
        let value_source = if rows == 1 {
            buffers.weighted_latent_bf16
        } else {
            buffers.head_major_large_bf16
        };
        let value_destination = if rows == 1 {
            buffers.unabsorbed_bf16
        } else {
            buffers.head_major_small_bf16
        };
        if let Some(weight) = weights.exl3_absorption_bank(false)? {
            self.absorb_exl3_bank(
                gpu,
                false,
                rows,
                value_source,
                weight,
                value_destination,
                stream,
            )?;
            launches += 1;
        } else {
            for head in 0..HEADS {
                let source = Self::head_slice_rows(value_source, head, rows, LATENT)?;
                let destination = Self::head_slice_rows(value_destination, head, rows, HEAD_DIM)?;
                self.linear_dsa_rows(
                    gpu,
                    rows,
                    weights.v_b(head)?,
                    source,
                    q8,
                    destination,
                    projection_scratch,
                    stream,
                )?;
                launches += 2;
            }
        }
        if rows > 1 {
            self.selected.transpose_heads(
                gpu,
                rows,
                HEAD_DIM,
                false,
                Glm53DsaHeadTransposeBuffers {
                    input_bf16: buffers.head_major_small_bf16,
                    output_bf16: buffers.unabsorbed_bf16,
                },
                stream,
            )?;
        }
        self.linear_dsa_rows(
            gpu,
            rows,
            weights.output(),
            buffers.unabsorbed_bf16,
            q8,
            output,
            projection_scratch,
            stream,
        )?;
        launches += 2;

        // Stage this token's latent row into the transaction overlay. The
        // persistent cache is untouched until the index commit accepts.
        if rows == 1 {
            self.latent_append.launch_stage(
                gpu,
                Glm53DsaLatentAppendPlan::new(
                    1,
                    1,
                    geometry.capacity,
                    geometry.position,
                    geometry
                        .position
                        .checked_add(1)
                        .context("GLM DSA position overflow")?,
                    LATENT,
                    geometry.nonce,
                    Glm53DsaLatentAppendStorage::Bf16,
                )?,
                Glm53DsaLatentAppendBuffers {
                    source_bf16: buffers.kv_cmpr_norm_bf16,
                    transaction_overlay_bf16: cache.latent_overlay_bf16,
                    published_ends_u32: cache.published_ends_u32,
                    published_nonces_u64: cache.published_nonces_u64,
                },
                stream,
            )?;
            launches += 1;
        }

        // Publish this layer's staged index tail (and a completed pool, if this
        // token finished one) into persistent state so the NEXT token's score/
        // topk see it. The receipted path is `Glm53DsaIndexCommitKernel`, an
        // all-11-layer transaction that validates every receipt first; this
        // bring-up path never rejects, so an immediate per-layer publish is
        // effect-equivalent. Third documented shortcut past the receipts.
        //
        // Keep cross-position publication AFTER score/topk/attention. Pool
        // consumed the prior carry and produced the current selector mask;
        // the advanced keys/gates/validity become the NEXT token's carry here.
        {
            // All stream-ordered: this is a cross-position carry ("the advanced
            // tail is what the next token starts from") and the walk stream is
            // CU_STREAM_NON_BLOCKING, so plain copy_d2d would read the tail the
            // attention kernel has not finished writing.
            let tail_bytes = usize::try_from(
                u64::from(spark_runtime::kv_cache::GLM53_DSA_TAIL_CAPACITY)
                    * u64::from(INDEX_DIM)
                    * 2,
            )?;
            gpu.copy_d2d_async(
                buffers.tail_keys_bf16.ptr,
                cache.prior_tail_keys_bf16.ptr,
                tail_bytes,
                stream,
            )?;
            gpu.copy_d2d_async(
                buffers.tail_gates_bf16.ptr,
                cache.prior_tail_gates_bf16.ptr,
                tail_bytes,
                stream,
            )?;
            gpu.copy_d2d_async(
                cache.out_tail_validity_u8.ptr,
                cache.prior_tail_validity_u8.ptr,
                3,
                stream,
            )?;
        }

        Ok(launches)
    }

    fn validate_shapes(&self, weights: &Glm53DsaWeights) -> Result<()> {
        for (name, matrix, inner, columns) in [
            ("attn_q_a", &weights.q_a, HIDDEN, Q_RANK),
            ("attn_q_b", &weights.q_b, Q_RANK, HEADS * HEAD_DIM),
            ("attn_kv_a_mqa", &weights.kv_a_mqa, HIDDEN, LATENT),
            ("attn_output", &weights.output, HEADS * HEAD_DIM, HIDDEN),
            ("indexer.attn_k", &weights.indexer_k, HIDDEN, INDEX_DIM),
            ("indexer.attn_q_b", &weights.indexer_q_b, Q_RANK, HIDDEN),
            (
                "indexer_compressor_gate",
                &weights.compressor_gate,
                HIDDEN,
                INDEX_DIM,
            ),
        ] {
            ensure!(
                matrix.inner() == inner && matrix.columns() == columns,
                "GLM DSA {name} is [{}, {}], expected [{inner}, {columns}]",
                matrix.inner(),
                matrix.columns()
            );
        }
        ensure!(
            weights.k_b.len() == HEADS && weights.v_b.len() == HEADS,
            "GLM DSA absorption banks must hold {HEADS} heads, got k_b {} v_b {}",
            weights.k_b.len(),
            weights.v_b.len()
        );
        Ok(())
    }

    fn validate_weight_ref(&self, weights: DsaWeightsRef<'_>) -> Result<()> {
        let DsaWeightsRef::Exl3(weights) = weights else {
            return self.validate_shapes(match weights {
                DsaWeightsRef::Gguf(weights) => weights,
                DsaWeightsRef::Exl3(_) => unreachable!(),
            });
        };
        for (name, projection, inner, columns) in [
            (
                "attn_q_a",
                Glm53Exl3Projection::Compressed(&weights.q_a),
                HIDDEN,
                Q_RANK,
            ),
            (
                "attn_q_b",
                Glm53Exl3Projection::Compressed(&weights.q_b),
                Q_RANK,
                HEADS * HEAD_DIM,
            ),
            (
                "attn_kv_a",
                Glm53Exl3Projection::Compressed(&weights.kv_a),
                HIDDEN,
                LATENT,
            ),
            (
                "attn_output",
                Glm53Exl3Projection::Compressed(&weights.output),
                HEADS * HEAD_DIM,
                HIDDEN,
            ),
            (
                "indexer_q_b",
                Glm53Exl3Projection::Compressed(&weights.indexer_q_b),
                Q_RANK,
                HIDDEN,
            ),
            (
                "indexer_k",
                Glm53Exl3Projection::NativeBf16(&weights.indexer_k),
                HIDDEN,
                INDEX_DIM,
            ),
            (
                "compressor_gate",
                Glm53Exl3Projection::NativeBf16(&weights.compressor_gate),
                HIDDEN,
                INDEX_DIM,
            ),
        ] {
            let plan = projection.plan(1)?;
            ensure!(
                plan.input == inner && plan.output == columns,
                "GLM EXL3 DSA {name} is [{}, {}], expected [{inner}, {columns}]",
                plan.input,
                plan.output
            );
        }
        ensure!(
            weights.k_b.len() == HEADS as usize && weights.v_b.len() == HEADS as usize,
            "GLM EXL3 DSA absorption banks must hold {HEADS} heads"
        );
        for matrix in weights.k_b.iter().chain(&weights.v_b) {
            ensure!(
                matrix.dtype() == Glm53Exl3NativeDtype::Bf16
                    && matrix.shape() == [HEAD_DIM as u64, LATENT as u64],
                "GLM EXL3 DSA absorption matrix dtype/shape drift"
            );
        }
        for (name, tensor, shape) in [
            ("q_a_norm", &weights.q_a_norm, &[Q_RANK as u64][..]),
            ("kv_a_norm", &weights.kv_a_norm, &[LATENT as u64][..]),
            (
                "indexer_k_norm",
                &weights.indexer_k_norm,
                &[INDEX_DIM as u64][..],
            ),
            (
                "indexer_k_norm_bias",
                &weights.indexer_k_norm_bias,
                &[INDEX_DIM as u64][..],
            ),
            (
                "indexer_proj",
                &weights.indexer_proj,
                &[INDEX_HEADS as u64, HIDDEN as u64][..],
            ),
        ] {
            ensure!(
                tensor.dtype() == Glm53Exl3NativeDtype::F32 && tensor.shape() == shape,
                "GLM EXL3 DSA {name} dtype/shape drift"
            );
        }
        Ok(())
    }
}

/// The per-token geometry a DSA layer needs.
#[derive(Debug, Clone, Copy)]
pub struct Glm53DsaLayerGeometry {
    /// Absolute position of the token being decoded.
    pub position: u32,
    /// Sequence capacity, which fixes the pool and selector extents.
    pub capacity: u32,
    /// The walk's transaction nonce. Never zero.
    pub nonce: u64,
}

impl Glm53DsaLayerGeometry {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.nonce != 0,
            "GLM DSA transaction nonce must be non-zero"
        );
        ensure!(
            self.position < self.capacity,
            "GLM DSA position {} is outside capacity {}",
            self.position,
            self.capacity
        );
        Ok(())
    }

    /// K-pools in the cache that are CAUSALLY USABLE at this query position.
    ///
    /// Named for the cache, not for the step: this is the CUMULATIVE count of
    /// pools the current token may score against. It is a different quantity
    /// from `Glm53DsaPoolPlan::complete_pools_to_write`, which is INCREMENTAL --
    /// how many pools this step completes. The two were previously both called
    /// some form of "complete pools", were nearly equal at chunk_tokens=1, and
    /// diverge by the chunk size the moment the walk batches. Conflating them
    /// is how the off-by-one below survived.
    ///
    /// Pool k spans positions 4k..4k+3 and is complete once position 4k+3 has
    /// arrived, so at query position q the count of complete pools is
    /// (q + 1) / KPOOL -- INCLUDING the pool that completes on this very step.
    ///
    /// That count is only sound because the walk now publishes the completing
    /// pool row into the cache BEFORE score/topk (see the publish site in
    /// `run_layer`). The two are a single change and neither is correct alone:
    ///
    ///  - `(q + 1) / KPOOL` with the publish still at the end of the step
    ///    points the scorer at a cache row that has not been written. Measured:
    ///    p3 went from 3.16% to 44.98%.
    ///  - `q / KPOOL` with the publish moved forward silently DROPS the
    ///    completing pool at q ≡ KPOOL-1 (mod KPOOL). The raw tail cannot
    ///    cover it: the tail holds positions [KPOOL*pools, q], which is at most
    ///    KPOOL-1 entries only under this formula. Measured in dump_dsafix
    ///    layer 3: p7 selected [0,1,2,3] with 4, 5, 6 absent entirely.
    ///
    /// The coverage arithmetic that falls out is exact and is enforced at the
    /// call site: tail occupancy = (q + 1) - KPOOL*pools ∈ [0, KPOOL-1], which
    /// is precisely GLM53_DSA_TAIL_CAPACITY. The reference model in
    /// `glm53_dsa_topk_tests.rs` covers q = 7 as pools {0,1} plus a 3-slot
    /// tail plus the current position via query metadata, and agrees.
    ///
    /// Distinct from `Glm53DsaPoolPlan::complete_pools_to_write`, which is
    /// INCREMENTAL -- how many pools THIS step completes. The two were once
    /// both called some form of "complete pools", are nearly equal at
    /// chunk_tokens=1, and diverge by the chunk size the moment the walk
    /// batches.
    pub fn usable_pools(self) -> u32 {
        (self.position + 1) / KPOOL
    }
}

#[cfg(test)]
#[path = "dsa_attention_tests.rs"]
mod tests;
