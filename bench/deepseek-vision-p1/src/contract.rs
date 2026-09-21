// SPDX-License-Identifier: AGPL-3.0-only
use anyhow::{Result, ensure};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    pub h: usize,
    pub w: usize,
}
impl Grid {
    pub fn new(h: usize, w: usize) -> Result<Self> {
        ensure!(
            matches!((h, w), (3, 3) | (4, 5) | (54, 54)),
            "not a retained corpus grid"
        );
        Ok(Self { h, w })
    }
    pub fn patches(self) -> usize {
        self.h * self.w
    }
    pub fn aligned_rows(self) -> usize {
        self.h.div_ceil(3) * self.w.div_ceil(3)
    }
    pub fn name(self) -> String {
        format!("grid-{}x{}", self.h, self.w)
    }
}
pub fn sha_syntax(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(&v))
}
pub fn check_bf16(raw: &[u8]) -> Result<()> {
    ensure!(
        !raw.is_empty() && raw.len().is_multiple_of(2),
        "invalid BF16 length"
    );
    ensure!(
        raw.chunks_exact(2)
            .all(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7f80 != 0x7f80),
        "nonfinite BF16 input/output"
    );
    Ok(())
}
pub fn check_f32(raw: &[u8]) -> Result<()> {
    ensure!(
        !raw.is_empty() && raw.len().is_multiple_of(4),
        "invalid F32 length"
    );
    ensure!(
        raw.chunks_exact(4)
            .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite()),
        "nonfinite F32 input/output"
    );
    Ok(())
}
pub fn check_stage(v: &Value, name: &str, shape: [usize; 2]) -> Result<()> {
    ensure!(
        v["name"].as_str() == Some(name) && v["file"].as_str() == Some(&format!("{name}.bf16")),
        "wrong stage name/path"
    );
    ensure!(v["dtype"].as_str() == Some("bf16"), "wrong stage dtype");
    let dims = v["shape"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("missing stage shape"))?;
    ensure!(
        dims.len() == 2
            && dims
                .iter()
                .zip(shape)
                .all(|(a, b)| a.as_u64() == Some(b as u64)),
        "wrong stage shape or noninteger dimension"
    );
    let bytes = shape[0]
        .checked_mul(shape[1])
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| anyhow::anyhow!("stage size overflow"))?;
    ensure!(
        bytes <= 64 * 1024 * 1024 && v["bytes"].as_u64() == Some(bytes as u64),
        "wrong stage bytes"
    );
    ensure!(
        v["sha256"].as_str().is_some_and(sha_syntax),
        "wrong stage hash syntax"
    );
    Ok(())
}
#[derive(Clone, Copy)]
pub enum ReferenceMode {
    Default,
    Full,
}
pub fn validate_reference(v: &Value, mode: ReferenceMode) -> Result<()> {
    ensure!(
        v["diagnostic_only"].as_bool() == Some(true),
        "not an operator diagnostic"
    );
    ensure!(
        v["scope"].as_str()
            == Some(
                "independent operators on shared native inputs, not chained-reference qualification"
            ),
        "wrong reference scope"
    );
    ensure!(
        v["official_sha256"].as_str() == Some(crate::pins::OFFICIAL_SHA),
        "official source changed"
    );
    ensure!(
        v["payload_sha256"].as_str() == Some(crate::pins::PAYLOAD_SHA),
        "reference weights changed"
    );
    ensure!(
        v["torch_version"].as_str() == Some("2.10.0+cu130"),
        "reference Torch changed"
    );
    match mode {
        ReferenceMode::Default => ensure!(
            v.get("bf16_reduction").is_none()
                && v["bf16_reduced_precision_reduction"].as_bool() == Some(true),
            "wrong default precision"
        ),
        ReferenceMode::Full => ensure!(
            v["bf16_reduction"].as_str() == Some("full")
                && v["bf16_reduced_precision_reduction"].as_bool() == Some(false),
            "wrong full precision"
        ),
    }
    Ok(())
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Arg {
    Ptr(u64),
    U32(u32),
}
pub struct LaunchSpec {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub args: Vec<Arg>,
}
pub fn rope_launch(g: Grid, ptrs: [u64; 5]) -> Result<LaunchSpec> {
    Grid::new(g.h, g.w)?;
    ensure!(ptrs.iter().all(|&p| p != 0), "null RoPE operand");
    let mut args: Vec<_> = ptrs.into_iter().map(Arg::Ptr).collect();
    args.extend([Arg::U32(g.patches() as u32), Arg::U32(16), Arg::U32(64)]);
    Ok(LaunchSpec {
        grid: [(g.patches() * 512).div_ceil(256) as u32, 1, 1],
        block: [256, 1, 1],
        args,
    })
}
pub fn angles_launch(g: Grid, out: u64) -> Result<LaunchSpec> {
    Grid::new(g.h, g.w)?;
    ensure!(out != 0, "null angle output");
    Ok(LaunchSpec {
        grid: [(g.patches() * 32).div_ceil(256) as u32, 1, 1],
        block: [256, 1, 1],
        args: vec![Arg::Ptr(out), Arg::U32(g.h as u32), Arg::U32(g.w as u32)],
    })
}
pub fn fc2_launch(a: u64, w: u64, out: u64) -> Result<LaunchSpec> {
    ensure!(a != 0 && w != 0 && out != 0, "null fc2 operand");
    Ok(LaunchSpec {
        grid: [32, 1, 1],
        block: [128, 1, 1],
        args: vec![
            Arg::Ptr(a),
            Arg::Ptr(w),
            Arg::Ptr(0),
            Arg::Ptr(out),
            Arg::U32(20),
            Arg::U32(1024),
            Arg::U32(2816),
            Arg::U32(1024),
        ],
    })
}
