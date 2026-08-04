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

        if !mla.wo_a.weight.is_null() {
            for token in 0..n {
                for group in 0..o_groups {
                    let input = attn_out
                        .offset(((token * input_width + group * group_in) as usize) * BF16_BYTES);
                    let weight = DenseWeight {
                        weight: mla
                            .wo_a
                            .weight
                            .offset(((group * o_lora * group_in) as usize) * BF16_BYTES),
                    };
                    let output = o_latent
                        .offset(((token * latent_dim + group * o_lora) as usize) * BF16_BYTES);
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        input,
                        &weight,
                        output,
                        o_lora,
                        group_in,
                        stream,
                    )?;
                }
            }
            return Ok(());
        }

        let weight = validate_fp8(
            mla.wo_a_fp8
                .ok_or_else(|| anyhow::anyhow!("V4 wo_a: BF16 released but FP8 is absent"))?,
            latent_dim,
            group_in,
            "V4 wo_a",
        )?;
        let scratch_in = ctx.buffers.qkv_output();
        let scratch_out = scratch_in.offset((n * group_in) as usize * BF16_BYTES);
        ensure!(
            group_in + o_lora <= input_width,
            "V4 wo_a: grouped scratch exceeds the dead Q buffer"
        );
        let weight_group_bytes = (o_lora * group_in) as usize;
        let scale_group_bytes =
            ((o_lora / FP8_BLOCK) * (group_in / FP8_BLOCK)) as usize * size_of::<f32>();

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
            let group_weight = Fp8Weight {
                weight: weight.weight.offset(group as usize * weight_group_bytes),
                row_scale: weight.row_scale.offset(group as usize * scale_group_bytes),
                n: o_lora,
                k: group_in,
                scale_format: weight.scale_format,
            };
            self.v4_project_prefill(
                ctx,
                scratch_in,
                &DenseWeight {
                    weight: DevicePtr::NULL,
                },
                Some(group_weight),
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
