// SPDX-License-Identifier: AGPL-3.0-only
//! P14 native capture admission. No authoritative block-8 teacher exists.
use anyhow::{Context, Result, ensure};
use deepseek_vision_p1::{contract::check_stage, driver::guarded_bytes, pins};
use serde_json::{Value, json};
use std::collections::BTreeSet;

// Reuse only unchanged geometry/ABI types. Historical reference accessors are
// not used by the block-8 binary; its admission and reporting are separate.
pub use crate::gemm_geometry::{BoundFc1, DeviceSpan, Fc1Plan, GemmCall, ReductionMode};
pub const INPUT_BYTES: usize = crate::gemm_geometry::INPUT_BYTES;
pub const WEIGHT_BYTES: usize = crate::gemm_geometry::WEIGHT_BYTES;
pub const OUTPUT_BYTES: usize = crate::gemm_geometry::OUTPUT_BYTES;
pub const HEADER_CAP: u64 = crate::gemm_geometry::HEADER_CAP;
pub const OWNED_DEVICE_CAP: usize = 96 * 1024 * 1024;
pub const LT_WORKSPACE: usize = 64 * 1024 * 1024;
pub const WEIGHT: &str = "vision.blocks.8.mlp.w1.weight";
pub const INPUT_SHA: &str = "bae82ae227edfcdf0ec81150cc2f7fc01371fd2da19bea87948c73e773480606";
pub const NATIVE_SHA: &str = "955e1d6608e55e175445f6e215a5c2ddb0058be26f10374fef5a1c8dddd11ef3";
pub const MANIFEST_SHA: &str = "06a0d7b7dc4f91c36b8657a846b1331031efb4a1c3533da407ab8e26da0a1870";
pub const STAGES_SHA: &str = "054ba4f31f606643c5ffa9fdd3e25a49e3237ccc6597eff3d6d36a07fa300e48";
pub const PRODUCER_SHA: &str = "ba5d7177ecd6d9955fd6c292b8a4829420d0c78804f375c3dd1a09ff568ce9d9";
pub const PIXELS_SHA: &str = "319cc2fac27587ad2e9e2ea03e2c9a6b13b481bdd183b9046735e40b70944e2f";
pub const FINAL_SHA: &str = "27b00f915f5f6c34573dfccaff012d03c596227c77c55d5559f565d401602f9d";
pub const PTX_PATH: &str = "/var/tmp/atlas-deepseek-p14-native.W7zeC1AP/release/build/atlas-kernels-57072f598bc1cf82/out/t0__deepseek_vision_gemm.ptx";
pub const PTX_SHA: &str = "9c06771d350de7ad5a2cf6a01b4ce7244f6bd0a1a00cb327a39024d4d0774669";

pub struct CaptureReceipts {
    pub input_sha256: String,
    pub native_sha256: String,
}

/// Semantic checks supplement mandatory full-manifest SHA admission. Only the
/// selected operator payloads are loaded, not the other 154 captured buffers.
pub fn validate_capture(top: &Value, stages: &Value) -> Result<CaptureReceipts> {
    let checkpoint = &top["checkpoint"];
    ensure!(
        checkpoint["model_revision"] == pins::MODEL_REV
            && checkpoint["config_sha256"] == pins::CONFIG_SHA
            && checkpoint["index_sha256"] == pins::INDEX_SHA
            && checkpoint["selected_tensors"] == 267
            && checkpoint["selected_bytes"] == 932786176,
        "P14 checkpoint identity"
    );
    ensure!(
        top["binary_sha256"] == PRODUCER_SHA
            && top["diagnostic_stage_capture"] == true
            && top["selected_detail_block"].as_u64() == Some(8)
            && top["scratch_bytes"] == 206275584,
        "P14 producer/detail identity"
    );
    let cases = top["cases"].as_array().context("P14 cases missing")?;
    ensure!(cases.len() == 3, "complete three-grid P14 corpus required");
    for (h, w, rows) in [(3, 3, 1), (4, 5, 4), (54, 54, 324)] {
        let name = format!("grid-{h}x{w}");
        let matched: Vec<_> = cases.iter().filter(|c| c["name"] == name).collect();
        ensure!(matched.len() == 1, "missing/duplicate P14 case {name}");
        let c = matched[0];
        ensure!(
            c["grid_h"] == h
                && c["grid_w"] == w
                && c["patches"] == h * w
                && c["aligned_rows"] == rows
                && c["hidden_size"] == 4096
                && c["repeat_byte_equal"] == true,
            "P14 case geometry/repeat {name}"
        );
        if h == 4 {
            ensure!(
                c["input_sha256"] == PIXELS_SHA && c["output_sha256"] == FINAL_SHA,
                "P14 image input/final identity"
            );
        }
    }
    ensure!(
        stages["grid"] == json!([4, 5])
            && stages["output_byte_equal"] == true
            && stages["selected_detail_block"].as_u64() == Some(8),
        "selected block-8 stage identity"
    );
    let entries = stages["stages"]
        .as_array()
        .context("stage entries missing")?;
    ensure!((2..=64).contains(&entries.len()), "stage count bound");
    let mut names = BTreeSet::new();
    for entry in entries {
        ensure!(
            names.insert(entry["name"].as_str().context("stage name missing")?),
            "duplicate stage name"
        );
    }
    for (name, shape, sha) in [
        ("block-08-norm2", [20, 1024], INPUT_SHA),
        ("block-08-fc1", [20, 5632], NATIVE_SHA),
    ] {
        let entry = entries
            .iter()
            .find(|e| e["name"] == name)
            .context("selected stage missing")?;
        check_stage(entry, name, shape)?;
        ensure!(entry["sha256"] == sha, "block-8 selected payload identity");
    }
    Ok(CaptureReceipts {
        input_sha256: INPUT_SHA.into(),
        native_sha256: NATIVE_SHA.into(),
    })
}

pub fn fc1_weight_span(header: &Value, header_len: u64, file_len: u64) -> Result<(u64, usize)> {
    ensure!(
        header_len > 0 && header_len <= HEADER_CAP,
        "header size bound"
    );
    let t = &header[WEIGHT];
    ensure!(t["dtype"] == "BF16", "block-8 fc1 must be native BF16");
    let shape = t["shape"].as_array().context("block-8 weight shape")?;
    ensure!(
        shape.len() == 2 && shape[0].as_u64() == Some(5632) && shape[1].as_u64() == Some(1024),
        "block-8 weight geometry"
    );
    let offsets = t["data_offsets"]
        .as_array()
        .context("block-8 weight offsets")?;
    ensure!(offsets.len() == 2, "weight offset count");
    let lo = offsets[0].as_u64().context("integer weight start")?;
    let hi = offsets[1].as_u64().context("integer weight end")?;
    ensure!(
        hi.checked_sub(lo) == Some(WEIGHT_BYTES as u64),
        "weight byte span"
    );
    let start = header_len
        .checked_add(8)
        .and_then(|n| n.checked_add(lo))
        .context("weight start overflow")?;
    let end = start
        .checked_add(WEIGHT_BYTES as u64)
        .context("weight end overflow")?;
    ensure!(end <= file_len, "weight outside shard");
    Ok((start, WEIGHT_BYTES))
}

pub fn owned_device_bytes() -> Result<usize> {
    let mut total = 0;
    for bytes in [
        INPUT_BYTES,
        WEIGHT_BYTES,
        OUTPUT_BYTES,
        OUTPUT_BYTES,
        OUTPUT_BYTES,
        OUTPUT_BYTES,
        OUTPUT_BYTES,
        Fc1Plan::new().workspace_bytes,
        LT_WORKSPACE,
    ] {
        total = guarded_bytes(bytes, total)?.1;
    }
    ensure!(
        total <= OWNED_DEVICE_CAP,
        "block-8 owned device cap exceeded"
    );
    Ok(total)
}
