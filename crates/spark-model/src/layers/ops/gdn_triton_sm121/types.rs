// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, bail, ensure};

pub const POINTER_ALIGNMENT: u64 = 16;
pub const WORKSPACE_ALIGNMENT: u64 = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Region {
    pub address: u64,
    pub bytes: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct Buffers {
    pub atlas_qkv_bf16: Region,
    pub atlas_gate_beta_f32: Region,
    pub atlas_state_hkv_f32: Region,
    pub output_bf16: Region,
    pub workspace: Region,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Segment {
    pub name: &'static str,
    pub offset: u64,
    pub bytes: u64,
}

#[derive(Clone, Debug)]
pub struct WorkspaceLayout {
    pub m: u32,
    pub nt: u32,
    pub total_bytes: u64,
    pub segments: Vec<Segment>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Stream(u64);

impl Stream {
    pub fn new(raw: u64) -> Result<Self> {
        ensure!(raw != 0, "frozen Triton requires a nondefault CUDA stream");
        Ok(Self(raw))
    }

    pub fn raw(self) -> u64 {
        self.0
    }
}

pub fn checked_product(values: &[u64], label: &str) -> Result<u64> {
    values.iter().try_fold(1u64, |value, factor| {
        value
            .checked_mul(*factor)
            .ok_or_else(|| anyhow::anyhow!("{label}: u64 multiplication overflow"))
    })
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    ensure!(alignment.is_power_of_two(), "invalid alignment");
    value
        .checked_add(value.wrapping_neg() & (alignment - 1))
        .ok_or_else(|| anyhow::anyhow!("workspace alignment overflow"))
}

pub fn workspace_layout(m: u32) -> Result<WorkspaceLayout> {
    ensure!(matches!(m, 2_079 | 8_192), "only M2079/M8192 are qualified");
    let m64 = u64::from(m);
    let nt = m.checked_add(63).context("NT overflow")? / 64;
    let nt64 = u64::from(nt);
    let specs: [(&str, u64); 16] = [
        ("q_bf16", checked_product(&[m64, 16, 128, 2], "q")?),
        ("k_bf16", checked_product(&[m64, 16, 128, 2], "k")?),
        ("v_bf16", checked_product(&[m64, 48, 128, 2], "v")?),
        ("log_gate_f32", checked_product(&[m64, 48, 4], "log_gate")?),
        ("beta_f32", checked_product(&[m64, 48, 4], "beta")?),
        (
            "state_hvk_f32",
            checked_product(&[48, 128, 128, 4], "state")?,
        ),
        ("g_cumsum_f32", checked_product(&[m64, 48, 4], "cumsum")?),
        ("A_bf16", checked_product(&[m64, 48, 64, 2], "A")?),
        ("w_bf16", checked_product(&[m64, 48, 128, 2], "w")?),
        ("u_bf16", checked_product(&[m64, 48, 128, 2], "u")?),
        ("h_bf16", checked_product(&[nt64, 48, 128, 128, 2], "h")?),
        ("v_new_bf16", checked_product(&[m64, 48, 128, 2], "v_new")?),
        ("cu_seqlens_i32", 8),
        ("state_index_i32", 4),
        (
            "chunk_indices_i32",
            checked_product(&[nt64, 2, 4], "indices")?,
        ),
        ("chunk_offsets_i64", 16),
    ];
    let mut cursor = 0u64;
    let mut segments = Vec::with_capacity(specs.len());
    for (name, bytes) in specs {
        cursor = align_up(cursor, WORKSPACE_ALIGNMENT)?;
        segments.push(Segment {
            name,
            offset: cursor,
            bytes,
        });
        cursor = cursor
            .checked_add(bytes)
            .ok_or_else(|| anyhow::anyhow!("{name}: end overflow"))?;
    }
    let total_bytes = align_up(cursor, WORKSPACE_ALIGNMENT)?;
    let expected = if m == 2_079 { 188_241_152 } else { 729_286_400 };
    ensure!(total_bytes == expected, "sealed workspace total drift");
    Ok(WorkspaceLayout {
        m,
        nt,
        total_bytes,
        segments,
    })
}

impl WorkspaceLayout {
    pub fn pointer(&self, workspace: Region, name: &str) -> Result<u64> {
        ensure!(
            workspace.bytes == self.total_bytes,
            "workspace extent drift"
        );
        let segment = self
            .segments
            .iter()
            .find(|segment| segment.name == name)
            .ok_or_else(|| anyhow::anyhow!("missing workspace segment {name}"))?;
        let end = segment
            .offset
            .checked_add(segment.bytes)
            .ok_or_else(|| anyhow::anyhow!("{name}: segment overflow"))?;
        ensure!(end <= workspace.bytes, "{name}: segment exceeds workspace");
        workspace
            .address
            .checked_add(segment.offset)
            .ok_or_else(|| anyhow::anyhow!("{name}: pointer overflow"))
    }
}

fn expected_external(m: u32, layout: &WorkspaceLayout) -> Result<[u64; 5]> {
    let m = u64::from(m);
    Ok([
        checked_product(&[m, 10_240, 2], "atlas_qkv")?,
        checked_product(&[m, 96, 4], "atlas_gate_beta")?,
        checked_product(&[48, 128, 128, 4], "atlas_state")?,
        checked_product(&[m, 48, 128, 2], "output")?,
        layout.total_bytes,
    ])
}

pub fn preflight_buffers(m: u32, buffers: Buffers, stream: Stream) -> Result<WorkspaceLayout> {
    ensure!(stream.raw() != 0, "null/default stream");
    let layout = workspace_layout(m)?;
    let records = [
        ("atlas_qkv_bf16", buffers.atlas_qkv_bf16, POINTER_ALIGNMENT),
        (
            "atlas_gate_beta_f32",
            buffers.atlas_gate_beta_f32,
            POINTER_ALIGNMENT,
        ),
        (
            "atlas_state_hkv_f32",
            buffers.atlas_state_hkv_f32,
            POINTER_ALIGNMENT,
        ),
        ("output_bf16", buffers.output_bf16, POINTER_ALIGNMENT),
        ("workspace", buffers.workspace, WORKSPACE_ALIGNMENT),
    ];
    let expected = expected_external(m, &layout)?;
    let mut intervals = Vec::with_capacity(records.len());
    for ((name, region, alignment), required) in records.into_iter().zip(expected) {
        ensure!(region.address != 0, "{name}: null device pointer");
        ensure!(region.address % alignment == 0, "{name}: unaligned pointer");
        ensure!(region.bytes == required, "{name}: exact extent required");
        let end = region
            .address
            .checked_add(region.bytes)
            .ok_or_else(|| anyhow::anyhow!("{name}: address overflow"))?;
        intervals.push((region.address, end, name));
    }
    intervals.sort_unstable_by_key(|entry| entry.0);
    if intervals.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        bail!("frozen Triton buffers alias");
    }
    Ok(layout)
}

use anyhow::Context;
