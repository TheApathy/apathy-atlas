// SPDX-License-Identifier: AGPL-3.0-only

use super::contract::{bf16_stats, sha256_bytes, write_new};
use anyhow::{Result, ensure};
use serde_json::json;
use spark_model::layers::deepseek_vision::{DeepSeekVisionEncoder, VisionStageDtype};
use spark_runtime::gpu::GpuBackend;
use std::path::Path;

pub fn capture(
    encoder: &DeepSeekVisionEncoder,
    gpu: &dyn GpuBackend,
    pixels: &[f32],
    grid: (usize, usize),
    out: &Path,
    expected: &[u8],
    block: usize,
) -> Result<()> {
    let directory = out.join(format!("grid-{}x{}-stages", grid.0, grid.1));
    std::fs::create_dir(&directory)?;
    let mut entries = Vec::new();
    let mut total = 0usize;
    let mut callback = |name: &str, pointer, shape: [usize; 2], dtype| {
        ensure!(
            !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-'),
            "invalid stage name"
        );
        ensure!(entries.len() < 64, "too many encoder stages");
        let (label, bytes_per_value) = match dtype {
            VisionStageDtype::Bf16 => ("bf16", 2),
            VisionStageDtype::F32 => ("f32", 4),
        };
        let bytes = shape[0]
            .checked_mul(shape[1])
            .and_then(|n| n.checked_mul(bytes_per_value))
            .ok_or_else(|| anyhow::anyhow!("stage size overflow"))?;
        ensure!(
            bytes <= 3456 * 3456 * 4,
            "stage exceeds maximum attention matrix"
        );
        total += bytes;
        ensure!(
            total <= 512 * 1024 * 1024,
            "stage dump exceeds per-image budget"
        );
        let mut raw = vec![0; bytes];
        gpu.copy_d2h(pointer, &mut raw)?;
        if dtype == VisionStageDtype::Bf16 {
            bf16_stats(&raw)?;
        } else {
            ensure!(
                raw.chunks_exact(4)
                    .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
                "nonfinite FP32 stage"
            );
        }
        let file = format!("{name}.{label}");
        write_new(&directory.join(&file), &raw)?;
        entries.push(json!({"name":name,"file":file,"shape":shape,"dtype":label,
            "bytes":bytes,"sha256":sha256_bytes(&raw)?}));
        Ok(())
    };
    let result =
        encoder.forward_observed_block(gpu, pixels, grid.0, grid.1, block, &mut callback)?;
    let mut final_raw = vec![0; expected.len()];
    gpu.copy_d2h(result, &mut final_raw)?;
    ensure!(
        final_raw == expected,
        "stage observation changed encoder output"
    );
    write_new(
        &directory.join("manifest.json"),
        &serde_json::to_vec_pretty(&json!({
            "scope":"diagnostic only; not a timing run", "grid":[grid.0,grid.1],
            "selected_detail_block":block,
            "total_bytes":total,"output_byte_equal":true,"stages":entries
        }))?,
    )
}
