// SPDX-License-Identifier: AGPL-3.0-only

//! Ownership gate for DeepSeek-V4 attention transcode intermediates.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::weight_map::{DenseWeight, Fp8Weight, WeightQuantFormat};

fn require_shape(weight: Option<Fp8Weight>, n: usize, k: usize, label: &str) -> Result<()> {
    let weight = weight.ok_or_else(|| anyhow::anyhow!("{label}: native FP8 weight is absent"))?;
    ensure!(
        weight.scale_format == WeightQuantFormat::Fp8BlockScaled,
        "{label}: expected block-scaled FP8, got {:?}",
        weight.scale_format
    );
    ensure!(
        weight.n as usize == n && weight.k as usize == k,
        "{label}: native FP8 shape [{}, {}] != expected [{n}, {k}]",
        weight.n,
        weight.k
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn release_bf16_projection_mirrors(
    layer_prefix: &str,
    config: &ModelConfig,
    wq_b: &mut DenseWeight,
    wo_a: &mut DenseWeight,
    o_dense: &mut DenseWeight,
    wq_b_fp8: Option<Fp8Weight>,
    wo_a_fp8: Option<Fp8Weight>,
    wo_b_fp8: Option<Fp8Weight>,
    nvfp4_ready: bool,
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Result<usize> {
    ensure!(
        config.o_lora_rank > 0 && config.o_groups > 0,
        "{layer_prefix}: BF16 attention release is only valid for grouped V4 MLA"
    );
    let q_width = config
        .num_attention_heads
        .checked_mul(config.qk_nope_head_dim + config.qk_rope_head_dim)
        .ok_or_else(|| anyhow::anyhow!("{layer_prefix}: V4 query width overflow"))?;
    ensure!(
        q_width.is_multiple_of(config.o_groups),
        "{layer_prefix}: query width {q_width} is not divisible by o_groups={} ",
        config.o_groups
    );
    let group_in = q_width / config.o_groups;
    let latent = config
        .o_groups
        .checked_mul(config.o_lora_rank)
        .ok_or_else(|| anyhow::anyhow!("{layer_prefix}: V4 output latent width overflow"))?;

    ensure!(
        nvfp4_ready,
        "{layer_prefix}: ATLAS_V4_ATTN_RELEASE_BF16=1 requires successful NVFP4 \
         transcodes for wq_b, wo_a, and wo_b"
    );
    require_shape(wq_b_fp8, q_width, config.q_lora_rank, "V4 wq_b")?;
    require_shape(wo_a_fp8, latent, group_in, "V4 wo_a")?;
    require_shape(wo_b_fp8, config.hidden_size, latent, "V4 wo_b")?;
    ensure!(
        !wq_b.weight.is_null() && !wo_a.weight.is_null() && !o_dense.weight.is_null(),
        "{layer_prefix}: a BF16 projection mirror is already null"
    );
    ensure!(
        wq_b.weight != wo_a.weight
            && wq_b.weight != o_dense.weight
            && wo_a.weight != o_dense.weight,
        "{layer_prefix}: BF16 projection mirrors unexpectedly alias"
    );
    gpu.kernel("w8a16_gemm", "w8a16_gemm").map_err(|error| {
        anyhow::anyhow!("{layer_prefix}: ATLAS_V4_ATTN_RELEASE_BF16=1 requires w8a16_gemm: {error}")
    })?;

    let released_bytes = q_width
        .checked_mul(config.q_lora_rank)
        .and_then(|wq| {
            latent
                .checked_mul(group_in)
                .and_then(|woa| wq.checked_add(woa))
        })
        .and_then(|partial| {
            config
                .hidden_size
                .checked_mul(latent)
                .and_then(|wob| partial.checked_add(wob))
        })
        .and_then(|elements| elements.checked_mul(size_of::<u16>()))
        .ok_or_else(|| anyhow::anyhow!("{layer_prefix}: BF16 release byte count overflow"))?;

    gpu.synchronize(stream)?;
    gpu.free(wq_b.weight)?;
    wq_b.weight = DevicePtr::NULL;
    gpu.free(wo_a.weight)?;
    wo_a.weight = DevicePtr::NULL;
    // `MlaWeights::wo` and `wo_b` are both built from this one allocation.
    gpu.free(o_dense.weight)?;
    o_dense.weight = DevicePtr::NULL;
    tracing::info!(
        "{layer_prefix}: released {released_bytes} bytes of superseded BF16 MLA \
         projections (wq_b, wo_a, wo_b); checkpoint FP8 and NVFP4 remain resident"
    );
    Ok(released_bytes)
}

#[cfg(test)]
mod tests {
    #[test]
    fn production_projection_bytes_match_the_residency_budget() {
        let q_width = 32_768usize;
        let q_lora = 1_024usize;
        let groups = 8usize;
        let o_lora = 1_024usize;
        let group_in = 4_096usize;
        let hidden = 4_096usize;
        let elements = q_width * q_lora + groups * o_lora * group_in + hidden * groups * o_lora;
        assert_eq!(elements * size_of::<u16>(), 201_326_592);
    }
}
