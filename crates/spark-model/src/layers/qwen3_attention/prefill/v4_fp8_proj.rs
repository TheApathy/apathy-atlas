// SPDX-License-Identifier: AGPL-3.0-only

//! Shared DeepSeek-V4 prefill projection dispatch.
//!
//! The checkpoint-native FP8 path permits the loader to release superseded
//! BF16 mirrors while keeping cache-skip and paged prefill on one implementation.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::super::{MlaWeights, Qwen3AttentionLayer};
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, Fp8Weight, WeightQuantFormat};

const BF16_BYTES: usize = 2;
const FP8_BLOCK: u32 = 128;

/// ATLAS_V4_PROJ_FP8MMA=1 (cached once): route the released-BF16 projections
/// through the FP8-native m16n8k32 MMA instead of the LUT-dequant W8A16
/// path. Default OFF — activation quantization is a numerics change (same
/// class as the FP8 mirrors), gated on serve A/B + tool-eval.
fn v4_proj_fp8mma_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_V4_PROJ_FP8MMA").as_deref() == Ok("1"))
}

/// ATLAS_V4_PREFILL_CUBLASLT (cached once, DEFAULT ON; =0 opts out): route the
/// V4 prefill projections through cuBLASLt BF16 (`bf16_gemm_act_weight_t`)
/// whenever a BF16 weight is resident. Measured (cublaslt_v4_bench, M=2410,
/// cosine 1.000000 vs dense_gemm_bf16_pipelined — same-math tier, only the
/// accumulation-order class every BF16 GEMM swap shares): wq_b [2410,32768,
/// 1024] 8.28→1.92 ms (84 TF), wo_b [2410,4096,8192] →1.92, wo_a-group
/// [2410,1024,4096] →0.188 (108 TF), wq_a →0.188, kv_proj →0.117 — ≈480 ms
/// less per pass, prefill ~973 → ~1200 tok/s.
///
/// RELEASE_BF16 interaction (do NOT change the release logic): under
/// ATLAS_V4_ATTN_RELEASE_BF16=1 the wq_b/wo_a/wo_b BF16 mirrors are FREED and
/// `dense.weight` is null, so this arm never fires there and the FP8/w8a8
/// dispatch is untouched. The serve-side A/B is therefore
///   RELEASE_BF16=0 + ATLAS_V4_PREFILL_CUBLASLT=1   (cuBLASLt, +~8 GiB resident)
/// vs today's
///   RELEASE_BF16=1 + ATLAS_V4_PROJ_FP8MMA=1        (FP8 MMA, mirrors freed).
/// Dispatch order per site: BF16 present → cuBLASLt (this gate) → pipelined
/// kernel; BF16 released → existing FP8/w8a8 arms.
pub(super) fn v4_prefill_cublaslt_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_V4_PREFILL_CUBLASLT").as_deref() != Ok("0"))
}

/// Sticky failure latch: if cuBLASLt errors once (handle creation, no algo for
/// a shape, launch failure), warn once and stop attempting it for the rest of
/// the process — every caller falls back to its custom-kernel arm, so a broken
/// or absent cuBLASLt runtime degrades to exactly the pre-cuBLASLt behavior.
fn v4_cublaslt_poisoned() -> &'static std::sync::atomic::AtomicBool {
    static POISONED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    &POISONED
}

/// Try one packed V4 prefill projection through cuBLASLt BF16. Returns `true`
/// when cuBLASLt handled it; `false` (gate off / poisoned / error) means the
/// caller must run its own kernel arm — on error the output buffer is
/// untouched (cuBLASLt validates before writing) and the latch trips.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_v4_cublas_prefill(
    act: DevicePtr,
    weight_bf16: DevicePtr,
    out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
    label: &str,
) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    if !v4_prefill_cublaslt_enabled()
        || weight_bf16.is_null()
        || v4_cublaslt_poisoned().load(Relaxed)
    {
        return false;
    }
    match ops::cublas_bf16_proj_dense(act, weight_bf16, out, m, n, k, stream) {
        Ok(()) => true,
        Err(e) => {
            v4_cublaslt_poisoned().store(true, Relaxed);
            tracing::warn!(
                "{label}: cuBLASLt prefill GEMM failed ({e}); falling back to the \
                 custom kernels for the rest of the process"
            );
            false
        }
    }
}

/// Strided sibling of [`try_v4_cublas_prefill`] for the grouped wo_a in-place
/// path: A is a column slice at row stride `lda` elements, C a column slice at
/// row stride `ldc` elements (the weight group is packed `[N,K]`).
#[allow(clippy::too_many_arguments)]
pub(super) fn try_v4_cublas_prefill_strided(
    act: DevicePtr,
    lda: u32,
    weight_bf16: DevicePtr,
    out: DevicePtr,
    ldc: u32,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
    label: &str,
) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    if !v4_prefill_cublaslt_enabled()
        || weight_bf16.is_null()
        || v4_cublaslt_poisoned().load(Relaxed)
    {
        return false;
    }
    match ops::cublas_bf16_proj_dense_strided(act, lda, weight_bf16, out, ldc, m, n, k, stream) {
        Ok(()) => true,
        Err(e) => {
            v4_cublaslt_poisoned().store(true, Relaxed);
            tracing::warn!(
                "{label}: cuBLASLt strided prefill GEMM failed ({e}); falling back \
                 to the custom kernels for the rest of the process"
            );
            false
        }
    }
}

fn validate_fp8(weight: Fp8Weight, n: u32, k: u32, label: &str) -> Result<Fp8Weight> {
    ensure!(
        weight.scale_format == WeightQuantFormat::Fp8BlockScaled,
        "{label}: expected block-scaled FP8, got {:?}",
        weight.scale_format
    );
    ensure!(
        weight.n == n && weight.k == k,
        "{label}: FP8 shape [{}, {}] != expected [{n}, {k}]",
        weight.n,
        weight.k
    );
    ensure!(
        n.is_multiple_of(FP8_BLOCK) && k.is_multiple_of(FP8_BLOCK),
        "{label}: W8A16 requires dimensions divisible by {FP8_BLOCK}, got [{n}, {k}]"
    );
    Ok(weight)
}

impl Qwen3AttentionLayer {
    /// Try the FP8-native MMA path for one packed prefill projection: quantize
    /// the activations per-row into the fp8_act arena and run
    /// `w8a8_gemm_pipelined`. Returns `Ok(false)` — caller falls back to its
    /// existing kernel, bit-exactly as before — when the ATLAS_V4_PROJ_FP8MMA
    /// gate is off, the kernels are absent, the arena cannot hold [M, K], or
    /// the FP8 weight does not match the [n, k] block-scaled contract.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_w8a8_project_prefill(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        weight: Fp8Weight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<bool> {
        if !v4_proj_fp8mma_enabled()
            || self.w8a8_gemm_pipelined_k.0 == 0
            || self.quantize_a_fp8_rows_k.0 == 0
            || ctx.buffers.fp8_act_bytes() < (m as usize) * (k as usize)
            || weight.scale_format != WeightQuantFormat::Fp8BlockScaled
            || weight.n != n
            || weight.k != k
            || !n.is_multiple_of(FP8_BLOCK)
            || !k.is_multiple_of(FP8_BLOCK)
        {
            return Ok(false);
        }
        let a_fp8 = ctx.buffers.fp8_act();
        let a_scale = ctx.buffers.fp8_act_scale();
        ops::quantize_a_fp8_rows(
            ctx.gpu,
            self.quantize_a_fp8_rows_k,
            input,
            a_fp8,
            a_scale,
            m,
            k,
            stream,
        )?;
        ops::w8a8_gemm_pipelined(
            ctx.gpu,
            self.w8a8_gemm_pipelined_k,
            a_fp8,
            a_scale,
            weight.weight,
            weight.row_scale,
            output,
            m,
            n,
            k,
            stream,
        )?;
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn v4_project_prefill(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        dense: &DenseWeight,
        fp8: Option<Fp8Weight>,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
        label: &str,
    ) -> Result<()> {
        if !dense.weight.is_null() {
            // cuBLASLt-first (ATLAS_V4_PREFILL_CUBLASLT, default ON): 1.7-4.3x
            // the pipelined kernel at the V4 prefill shapes, cosine 1.000000
            // (same-math tier). See v4_prefill_cublaslt_enabled() for the
            // measurements and the RELEASE_BF16 A/B; falls through to the
            // pipelined kernel on gate-off or any cuBLASLt failure.
            if try_v4_cublas_prefill(input, dense.weight, output, m, n, k, stream, label) {
                return Ok(());
            }
            // The BF16 arm (wq_b / wo_a groups / wo_b when the mirrors are NOT
            // released) ran the SIMT scalar GEMM: microtest M=2410 N=512 K=4096
            // measures 6.96 ms vs 0.23 ms for dense_gemm_bf16_pipelined, both
            // cosine 1.000000. Same math, 128x128 cp.async tiling.
            if self.dense_gemm_pipelined_k.0 != 0 {
                return ops::dense_gemm_bf16_pipelined(
                    ctx.gpu,
                    self.dense_gemm_pipelined_k,
                    input,
                    dense,
                    output,
                    m,
                    n,
                    k,
                    stream,
                );
            }
            return ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                input,
                dense,
                output,
                m,
                n,
                k,
                stream,
            );
        }

        let weight = validate_fp8(
            fp8.ok_or_else(|| anyhow::anyhow!("{label}: BF16 released but FP8 is absent"))?,
            n,
            k,
            label,
        )?;
        // FP8-native MMA arm: quantize the activations once (row-scaled E4M3
        // into the fp8_act arena) and feed both operands to mma.m16n8k32.e4m3
        // — the W8A16 path below LUT-dequants B per tile instead. Oracle
        // (w8a8_gemm_microtest): 1.04-1.22x at these shapes, cos 0.9997.
        // Falls through when the arena can't hold [M, K] (wo_b's o_latent at
        // full batch is the largest consumer).
        if self.try_w8a8_project_prefill(ctx, input, weight, output, m, n, k, stream)? {
            return Ok(());
        }
        if self.w8a16_gemm_pipelined_k.0 != 0 {
            ops::w8a16_gemm_pipelined(
                ctx.gpu,
                self.w8a16_gemm_pipelined_k,
                input,
                weight.weight,
                weight.row_scale,
                output,
                m,
                n,
                k,
                stream,
            )
        } else {
            ensure!(
                self.w8a16_gemm_k.0 != 0,
                "{label}: BF16 released but no W8A16 prefill kernel is loaded"
            );
            ops::w8a16_gemm(
                ctx.gpu,
                self.w8a16_gemm_k,
                input,
                weight.weight,
                weight.row_scale,
                output,
                m,
                n,
                k,
                stream,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn v4_grouped_wo_a_prefill(
        &self,
        ctx: &ForwardContext,
        mla: &MlaWeights,
        attn_out: DevicePtr,
        o_latent: DevicePtr,
        n: u32,
        nq: u32,
        head_dim: u32,
        o_groups: u32,
        o_lora: u32,
        stream: u64,
    ) -> Result<()> {
        ensure!(o_groups > 0, "V4 wo_a: o_groups must be positive");
        let input_width = nq
            .checked_mul(head_dim)
            .ok_or_else(|| anyhow::anyhow!("V4 wo_a: attention width overflow"))?;
        ensure!(
            input_width.is_multiple_of(o_groups),
            "V4 wo_a: attention width {input_width} is not divisible by {o_groups} groups"
        );
        let group_in = input_width / o_groups;
        let latent_dim = o_groups
            .checked_mul(o_lora)
            .ok_or_else(|| anyhow::anyhow!("V4 wo_a: latent width overflow"))?;

        // BOTH weight formats take the same gather → one GEMM over all `n`
        // rows → scatter shape, differing only in the weight handed to
        // `v4_project_prefill`.
        //
        // The BF16 arm used to loop `for token in 0..n { for group in
        // 0..o_groups { dense_gemv(..) } }` — n × o_groups GEMV launches per
        // layer where the FP8 arm already issued o_groups GEMMs total. On a
        // 911-token prefill that is 911 × 8 × 43 = 313k launches, and nsys
        // measured `dense_gemv_bf16` at 272,414 instances / 9.64 s — 49% of
        // ALL prefill GPU time, against 694 ms for the entire MoE. It is why
        // prefill throughput was FLAT in prompt length (~38 tok/s at both 113
        // and 3281 tokens): the projections got zero batching amortization.
        //
        // `attn_out` is group-strided (`input_width` per row), so each group's
        // columns are gathered into contiguous scratch first — a real GEMM
        // needs contiguous rows. Scratch is the dead Q buffer, live here
        // because wo_a runs after attention has consumed Q.
        let bf16 = !mla.wo_a.weight.is_null();
        let fp8 = if bf16 {
            None
        } else {
            Some(validate_fp8(
                mla.wo_a_fp8
                    .ok_or_else(|| anyhow::anyhow!("V4 wo_a: BF16 released but FP8 is absent"))?,
                latent_dim,
                group_in,
                "V4 wo_a",
            )?)
        };
        let scratch_in = ctx.buffers.qkv_output();
        let scratch_out = scratch_in.offset((n * group_in) as usize * BF16_BYTES);
        ensure!(
            group_in + o_lora <= input_width,
            "V4 wo_a: grouped scratch exceeds the dead Q buffer"
        );
        let weight_group_bytes = (o_lora * group_in) as usize;
        let scale_group_bytes =
            ((o_lora / FP8_BLOCK) * (group_in / FP8_BLOCK)) as usize * size_of::<f32>();

        // ── In-place arm: the pipelined GEMMs now take A/C row strides, so a
        // group reads its column slice of `attn_out` and writes its slice of
        // `o_latent` directly. That deletes 8 gather + 8 scatter
        // copy_d2d_2d_async per layer (~316 MB/layer of pure BF16 copy traffic
        // at N=2410) and 16 launches. Same GEMM, same math, same tiling —
        // lda/ldc only change WHERE the operands are read/written.
        // ATLAS_V4_WOA_INPLACE=0 restores the gather/scatter path.
        let inplace = {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| {
                std::env::var("ATLAS_V4_WOA_INPLACE").as_deref() != Ok("0")
            })
        } && ((fp8.is_some() && self.w8a16_gemm_pipelined_ld_k.0 != 0)
            || (bf16 && self.dense_gemm_pipelined_ld_k.0 != 0));
        if inplace {
            // FP8-native MMA arm for the grouped in-place path: quantize the
            // FULL attn_out [n, input_width] ONCE (one quantize_a_fp8_rows over
            // the widest input — the fp8_act arena is sized m*max(h, nq*hd) so
            // [n, nq*hd] fits by construction), then run each group's GEMM
            // against its FP8 column slice at lda = input_width.
            //
            // Scale-slice correctness: quantize_a_fp8_rows computes ONE scale
            // per ROW over the full input_width row (absmax/448) and quantizes
            // every element of the row with it. A group GEMM consumes only the
            // k-slice [g*group_in, (g+1)*group_in) of that row, but since each
            // slice element was quantized as a/scale[m] and the kernel's
            // epilogue multiplies a_row_scale[m] uniformly across the whole
            // output row (K-independent, per-row), the slice dequantizes with
            // exactly the scale it was quantized at — the math is consistent.
            // The full-row absmax is an upper bound of the slice absmax, so a
            // slice whose peak lives in another group quantizes slightly
            // coarser than a per-slice scale would (bounded by the same E4M3
            // relative-error class as the packed arm; gated by the same
            // microtest cosine + tool-eval). ATLAS_V4_PROJ_FP8MMA=0 restores
            // the w8a16 LUT-dequant groups below bit-exactly.
            let w8a8_inplace = fp8.is_some()
                && v4_proj_fp8mma_enabled()
                && self.w8a8_gemm_pipelined_ld_k.0 != 0
                && self.quantize_a_fp8_rows_k.0 != 0
                && ctx.buffers.fp8_act_bytes() >= (n as usize) * (input_width as usize);
            if w8a8_inplace {
                let a_fp8 = ctx.buffers.fp8_act();
                let a_scale = ctx.buffers.fp8_act_scale();
                ops::quantize_a_fp8_rows(
                    ctx.gpu,
                    self.quantize_a_fp8_rows_k,
                    attn_out,
                    a_fp8,
                    a_scale,
                    n,
                    input_width,
                    stream,
                )?;
                let w = fp8.expect("w8a8_inplace requires the FP8 wo_a");
                for group in 0..o_groups {
                    ops::w8a8_gemm_pipelined_ld(
                        ctx.gpu,
                        self.w8a8_gemm_pipelined_ld_k,
                        // FP8 A: 1 byte/elem, so the column-slice byte offset
                        // is the element offset.
                        a_fp8.offset((group * group_in) as usize),
                        a_scale,
                        w.weight.offset(group as usize * weight_group_bytes),
                        w.row_scale.offset(group as usize * scale_group_bytes),
                        o_latent.offset((group * o_lora) as usize * BF16_BYTES),
                        n,
                        o_lora,
                        group_in,
                        input_width,
                        latent_dim,
                        stream,
                    )?;
                }
                return Ok(());
            }
            for group in 0..o_groups {
                let a_in = attn_out.offset((group * group_in) as usize * BF16_BYTES);
                let c_out = o_latent.offset((group * o_lora) as usize * BF16_BYTES);
                if let Some(w) = fp8 {
                    ops::w8a16_gemm_pipelined_ld(
                        ctx.gpu,
                        self.w8a16_gemm_pipelined_ld_k,
                        a_in,
                        w.weight.offset(group as usize * weight_group_bytes),
                        w.row_scale.offset(group as usize * scale_group_bytes),
                        c_out,
                        n,
                        o_lora,
                        group_in,
                        input_width,
                        latent_dim,
                        stream,
                    )?;
                } else {
                    let w_group = mla
                        .wo_a
                        .weight
                        .offset(group as usize * weight_group_bytes * BF16_BYTES);
                    // cuBLASLt-first (ATLAS_V4_PREFILL_CUBLASLT, default ON):
                    // strided A slice ([n, group_in] at row stride input_width)
                    // and strided C slice ([n, o_lora] at row stride
                    // latent_dim) go through the ld-carrying cublasLt layouts —
                    // measured 0.188 ms/group at [2410, 1024, 4096] vs 0.8 for
                    // the pipelined_ld kernel, cosine 1.000000. 8 calls/layer.
                    if try_v4_cublas_prefill_strided(
                        a_in,
                        input_width,
                        w_group,
                        c_out,
                        latent_dim,
                        n,
                        o_lora,
                        group_in,
                        stream,
                        "V4 wo_a group",
                    ) {
                        continue;
                    }
                    ops::dense_gemm_bf16_pipelined_ld(
                        ctx.gpu,
                        self.dense_gemm_pipelined_ld_k,
                        a_in,
                        w_group,
                        c_out,
                        n,
                        o_lora,
                        group_in,
                        input_width,
                        latent_dim,
                        stream,
                    )?;
                }
            }
            return Ok(());
        }

        for group in 0..o_groups {
            ctx.gpu.copy_d2d_2d_async(
                attn_out.offset((group * group_in) as usize * BF16_BYTES),
                input_width as usize * BF16_BYTES,
                scratch_in,
                group_in as usize * BF16_BYTES,
                group_in as usize * BF16_BYTES,
                n as usize,
                stream,
            )?;
            let dense_group = DenseWeight {
                weight: if bf16 {
                    mla.wo_a.weight.offset(group as usize * weight_group_bytes * BF16_BYTES)
                } else {
                    DevicePtr::NULL
                },
            };
            let fp8_group = fp8.map(|w| Fp8Weight {
                weight: w.weight.offset(group as usize * weight_group_bytes),
                row_scale: w.row_scale.offset(group as usize * scale_group_bytes),
                n: o_lora,
                k: group_in,
                scale_format: w.scale_format,
            });
            self.v4_project_prefill(
                ctx,
                scratch_in,
                &dense_group,
                fp8_group,
                scratch_out,
                n,
                o_lora,
                group_in,
                stream,
                "V4 wo_a group",
            )?;
            ctx.gpu.copy_d2d_2d_async(
                scratch_out,
                o_lora as usize * BF16_BYTES,
                o_latent.offset((group * o_lora) as usize * BF16_BYTES),
                latent_dim as usize * BF16_BYTES,
                o_lora as usize * BF16_BYTES,
                n as usize,
                stream,
            )?;
        }
        Ok(())
    }
}
