// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in native shared-expert precision; keep valid legacy storage intact.

use crate::{
    layers::MoeLayer,
    weight_map::{Fp8ExpertWeight, load_fp8_block_scaled_as_fp8weight},
};
use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::{
    gpu::GpuBackend,
    weights::{WeightDtype, WeightStore},
};

fn requested(
    flag: Option<&str>,
    vision: bool,
    exl3: bool,
    ep: usize,
    tp: usize,
    mirrors: bool,
) -> Result<bool> {
    match flag {
        None | Some("0") => return Ok(false),
        Some("1") => (),
        _ => anyhow::bail!("ATLAS_V4_SHARED_NATIVE_FP8 must be 0 or 1"),
    }
    ensure!(
        vision && exl3 && ep <= 1 && tp <= 1,
        "ATLAS_V4_SHARED_NATIVE_FP8=1 requires actual Vision/EXL3/single GPU target-only"
    );
    ensure!(
        !mirrors,
        "native shared FP8 conflicts with shared FP8 mirror selectors"
    );
    Ok(true)
}

fn validate_projection(
    dtype: WeightDtype,
    shape: &[usize],
    scale_dtype: WeightDtype,
    scale_shape: &[usize],
    n: usize,
    k: usize,
) -> Result<()> {
    ensure!(
        dtype == WeightDtype::FP8E4M3 && shape == [n, k],
        "native shared FP8 requires exact F8_E4M3 projection shape [{n},{k}]"
    );
    ensure!(
        scale_dtype == WeightDtype::FP8E8M0 && scale_shape == [n / 128, k / 128],
        "native shared FP8 requires exact F8_E8M0 block-128 scales"
    );
    Ok(())
}

fn validate_scales(bytes: &[u8]) -> Result<()> {
    ensure!(
        !bytes.is_empty() && !bytes.contains(&255),
        "native shared FP8 E8M0 scales must be present and finite"
    );
    Ok(())
}

pub(super) fn install(
    moe: &mut MoeLayer,
    store: &WeightStore,
    prefix: &str,
    config: &ModelConfig,
    exl3: bool,
    gpu: &dyn GpuBackend,
) -> Result<()> {
    let flag = std::env::var("ATLAS_V4_SHARED_NATIVE_FP8").ok();
    let mirrors = ["ATLAS_EXL3_SHARED_PREFILL_FP8", "ATLAS_TARGET_SHARED_FP8"]
        .iter()
        .any(|key| std::env::var(key).as_deref() == Ok("1"));
    if !requested(
        flag.as_deref(),
        config.deepseek_vision.is_some(),
        exl3,
        config.ep_world_size,
        config.tp_world_size,
        mirrors,
    )? {
        return Ok(());
    }
    ensure!(
        config.hidden_size == 4096 && config.shared_expert_intermediate_size == 2048,
        "native shared FP8 requires H4096/I2048"
    );
    let specs = [("w1", 2048, 4096), ("w3", 2048, 4096), ("w2", 4096, 2048)];
    // Validate all metadata and tiny scale payloads before any allocations.
    for (suffix, n, k) in specs {
        let key = format!("{prefix}.ffn.shared_experts.{suffix}");
        let w = store.get(&format!("{key}.weight"))?;
        let scale = store.get(&format!("{key}.scale"))?;
        validate_projection(w.dtype, &w.shape, scale.dtype, &scale.shape, n, k)
            .with_context(|| key.clone())?;
        ensure!(
            !w.ptr.is_null() && !scale.ptr.is_null(),
            "{key}: null native tensor"
        );
        ensure!(
            !store.contains(&format!("{key}.weight_scale_inv"))
                && !store.contains(&format!("{key}.weight_scale")),
            "{key}: ambiguous native block-scale source"
        );
        let mut bytes = vec![0; (n / 128) * (k / 128)];
        gpu.copy_d2h(scale.ptr, &mut bytes)?;
        validate_scales(&bytes).with_context(|| key.clone())?;
    }
    let mut built = Vec::with_capacity(3);
    let result = (|| {
        for (suffix, _, _) in specs {
            built.push(load_fp8_block_scaled_as_fp8weight(
                store,
                &format!("{prefix}.ffn.shared_experts.{suffix}"),
                gpu,
            )?);
        }
        moe.set_native_fp8_shared_expert(
            Fp8ExpertWeight {
                gate_proj: built[0],
                up_proj: built[1],
                down_proj: built[2],
            },
            gpu,
        )
    })();
    if result.is_err() {
        for weight in built {
            let _ = gpu.free(weight.row_scale);
        }
    }
    result?;
    tracing::info!(
        "{prefix}: ATLAS_V4_SHARED_NATIVE_FP8=1 ARMED: native block-FP8 shared expert, SwiGLU clamp10, +6144B scales, valid NVFP4 storage retained"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_shared_fp8_accepts_only_exact_checkpoint_projection_metadata() {
        for (n, k) in [(2048, 4096), (4096, 2048)] {
            validate_projection(
                WeightDtype::FP8E4M3,
                &[n, k],
                WeightDtype::FP8E8M0,
                &[n / 128, k / 128],
                n,
                k,
            )
            .unwrap();
            for dtype in [WeightDtype::BF16, WeightDtype::UInt8, WeightDtype::FP32] {
                assert!(
                    validate_projection(
                        dtype,
                        &[n, k],
                        WeightDtype::FP8E8M0,
                        &[n / 128, k / 128],
                        n,
                        k
                    )
                    .is_err()
                );
            }
            assert!(
                validate_projection(
                    WeightDtype::FP8E4M3,
                    &[k, n],
                    WeightDtype::FP8E8M0,
                    &[n / 128, k / 128],
                    n,
                    k
                )
                .is_err()
            );
            assert!(
                validate_projection(
                    WeightDtype::FP8E4M3,
                    &[n, k],
                    WeightDtype::BF16,
                    &[n / 128, k / 128],
                    n,
                    k
                )
                .is_err()
            );
            assert!(
                validate_projection(
                    WeightDtype::FP8E4M3,
                    &[n, k],
                    WeightDtype::FP8E8M0,
                    &[n / 64, k / 128],
                    n,
                    k
                )
                .is_err()
            );
        }
    }

    #[test]
    fn native_shared_fp8_rejects_nonfinite_e8m0_scales() {
        validate_scales(&[0, 1, 127, 254]).unwrap();
        assert!(validate_scales(&[]).is_err());
        assert!(validate_scales(&[127, 255]).is_err());
    }

    #[test]
    fn native_shared_fp8_is_default_off_and_requires_actual_vision_exl3_c1() {
        for flag in [None, Some("0")] {
            assert!(!requested(flag, false, false, 2, 2, true).unwrap());
        }
        assert!(requested(Some("1"), true, true, 1, 1, false).unwrap());
        assert!(requested(Some("typo"), true, true, 1, 1, false).is_err());
        assert!(requested(Some("1"), false, true, 1, 1, false).is_err());
        assert!(requested(Some("1"), true, false, 1, 1, false).is_err());
        assert!(requested(Some("1"), true, true, 2, 1, false).is_err());
        assert!(requested(Some("1"), true, true, 1, 2, false).is_err());
        assert!(requested(Some("1"), true, true, 1, 1, true).is_err());
    }
}
