// SPDX-License-Identifier: AGPL-3.0-only
//! Fixed same-input block-0 fc1 contract; no device or filesystem effects.
use anyhow::{Context, Result, ensure};
use deepseek_vision_p1::{
    admission::reference_hash,
    contract::{self, Arg, LaunchSpec, ReferenceMode},
};
use serde_json::{Value, json};
use std::collections::BTreeSet;

pub const INPUT_SHA: &str = "a74cf71f1f2f958da55bb2ad64961dc79a0e34c5e0ecfad94fbe974ea74144cf";
pub const NATIVE_SHA: &str = "e39b9e80a339988ba215e8f22c6af82e5cf53f69b9da482056e03c8cb1307e41";
pub const DEFAULT_SHA: &str = "5c4d0b299c710760fe95be6a1caa9cba3826ff4739081b2d679f682d413261ad";
pub const WEIGHT: &str = "vision.blocks.0.mlp.w1.weight";
pub const INPUT_BYTES: usize = 20 * 1024 * 2;
pub const WEIGHT_BYTES: usize = 5632 * 1024 * 2;
pub const OUTPUT_BYTES: usize = 20 * 5632 * 2;
pub const HEADER_CAP: u64 = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceSpan {
    pub ptr: u64,
    pub bytes: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReductionMode {
    Default,
    Full,
}
impl ReductionMode {
    pub fn math_mode(self) -> i32 {
        match self {
            Self::Default => 0,
            Self::Full => 16,
        }
    }
    pub fn reference_sha256(self) -> &'static str {
        match self {
            Self::Default => DEFAULT_SHA,
            Self::Full => NATIVE_SHA,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Full => "full",
        }
    }
}
pub struct Fc1Plan {
    pub rows: usize,
    pub outputs: usize,
    pub inner: usize,
    pub workspace_bytes: usize,
}
pub struct BoundFc1 {
    x: DeviceSpan,
    w: DeviceSpan,
    y: DeviceSpan,
    pub workspace: DeviceSpan,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GemmCall {
    pub transa: i32,
    pub transb: i32,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub lda: i32,
    pub ldb: i32,
    pub ldc: i32,
    pub a: u64,
    pub b: u64,
    pub c: u64,
    pub a_type: i32,
    pub b_type: i32,
    pub c_type: i32,
    pub alpha: f32,
    pub beta: f32,
    pub compute_type: i32,
    pub algorithm: i32,
}
impl Fc1Plan {
    pub fn new() -> Self {
        Self {
            rows: 20,
            outputs: 5632,
            inner: 1024,
            workspace_bytes: 8_519_680,
        }
    }
    pub fn bind(
        &self,
        x: DeviceSpan,
        w: DeviceSpan,
        y: DeviceSpan,
        workspace: DeviceSpan,
    ) -> Result<BoundFc1> {
        ensure!(
            (self.rows, self.outputs, self.inner, self.workspace_bytes)
                == (20, 5632, 1024, 8_519_680),
            "modified fixed plan"
        );
        let spans = [x, w, y, workspace];
        let mut ends = [0; 4];
        for (i, bytes) in [
            INPUT_BYTES,
            WEIGHT_BYTES,
            OUTPUT_BYTES,
            self.workspace_bytes,
        ]
        .into_iter()
        .enumerate()
        {
            let s = spans[i];
            ensure!(
                s.ptr != 0 && s.ptr % 256 == 0 && s.bytes == bytes,
                "device extent/alignment {i}"
            );
            ends[i] = s
                .ptr
                .checked_add(u64::try_from(s.bytes)?)
                .context("device pointer overflow")?;
        }
        for i in 0..4 {
            for j in i + 1..4 {
                ensure!(
                    ends[i] <= spans[j].ptr || ends[j] <= spans[i].ptr,
                    "device operands overlap"
                );
            }
        }
        Ok(BoundFc1 { x, w, y, workspace })
    }
}
impl BoundFc1 {
    pub fn call(&self) -> GemmCall {
        GemmCall {
            transa: 1,
            transb: 0,
            m: 5632,
            n: 20,
            k: 1024,
            lda: 1024,
            ldb: 1024,
            ldc: 5632,
            a: self.w.ptr,
            b: self.x.ptr,
            c: self.y.ptr,
            a_type: 14,
            b_type: 14,
            c_type: 14,
            alpha: 1.0,
            beta: 0.0,
            compute_type: 68,
            algorithm: 99,
        }
    }
    pub fn native_launch(&self) -> LaunchSpec {
        LaunchSpec {
            grid: [176, 1, 1],
            block: [128, 1, 1],
            args: vec![
                Arg::Ptr(self.x.ptr),
                Arg::Ptr(self.w.ptr),
                Arg::Ptr(0),
                Arg::Ptr(self.y.ptr),
                Arg::U32(20),
                Arg::U32(5632),
                Arg::U32(1024),
                Arg::U32(5632),
            ],
        }
    }
}
pub struct ReferenceReceipts {
    pub input_sha256: String,
    pub native_sha256: String,
    pub default_sha256: String,
    pub full_sha256: String,
}
pub fn validate_references(
    stages: &Value,
    default: &Value,
    full: &Value,
) -> Result<ReferenceReceipts> {
    ensure!(
        stages["grid"] == json!([4, 5]) && stages["output_byte_equal"] == true,
        "stage grid/repeat"
    );
    let entries = stages["stages"].as_array().context("stages array")?;
    ensure!(
        !entries.is_empty() && entries.len() <= 64,
        "stage count bound"
    );
    let mut names = BTreeSet::new();
    for entry in entries {
        ensure!(
            names.insert(entry["name"].as_str().context("stage name")?),
            "duplicate stage"
        );
    }
    for (name, shape, sha) in [
        ("block-00-norm2", [20, 1024], INPUT_SHA),
        ("block-00-fc1", [20, 5632], NATIVE_SHA),
    ] {
        let entry = entries
            .iter()
            .find(|e| e["name"].as_str() == Some(name))
            .context("missing stage")?;
        contract::check_stage(entry, name, shape)?;
        ensure!(entry["sha256"] == sha, "retained stage pin changed");
    }
    for (v, mode, expected) in [
        (default, ReferenceMode::Default, DEFAULT_SHA),
        (full, ReferenceMode::Full, NATIVE_SHA),
    ] {
        contract::validate_reference(v, mode)?;
        let cases = v["cases"].as_array().context("reference cases")?;
        ensure!(
            !cases.is_empty() && cases.len() <= 3,
            "reference case bound"
        );
        for case in cases {
            ensure!(
                case["operators"].as_array().context("operators")?.len() <= 64,
                "operator bound"
            );
        }
        ensure!(
            reference_hash(v, "grid-4x5", "block-00-fc1", "block-00-norm2")? == expected,
            "fc1 reference pin changed"
        );
    }
    Ok(ReferenceReceipts {
        input_sha256: INPUT_SHA.into(),
        native_sha256: NATIVE_SHA.into(),
        default_sha256: DEFAULT_SHA.into(),
        full_sha256: NATIVE_SHA.into(),
    })
}
pub fn fc1_weight_span(v: &Value, header_len: u64, file_len: u64) -> Result<(u64, usize)> {
    ensure!(
        header_len > 0 && header_len <= HEADER_CAP,
        "header size bound"
    );
    let t = &v[WEIGHT];
    ensure!(t["dtype"].as_str() == Some("BF16"), "fc1 weight dtype");
    let shape = t["shape"].as_array().context("weight shape")?;
    ensure!(
        shape.len() == 2 && shape[0].as_u64() == Some(5632) && shape[1].as_u64() == Some(1024),
        "fc1 weight geometry"
    );
    let offsets = t["data_offsets"].as_array().context("weight offsets")?;
    ensure!(offsets.len() == 2, "weight offset count");
    let lo = offsets[0].as_u64().context("integer start")?;
    let hi = offsets[1].as_u64().context("integer end")?;
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
