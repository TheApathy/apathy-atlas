// SPDX-License-Identifier: AGPL-3.0-only

//! Position-selected, observational wide/replay logit receipt. No file I/O.
//! Raw base64 binds the complete row for an offline SHA256 receipt; FNV below
//! is only a quick non-cryptographic checksum, not a replacement for SHA256.

use anyhow::{Context, Result, bail, ensure};
use base64::Engine;
use serde::Serialize;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const ENV: &str = "ATLAS_GLM53_VERIFY_TRACE_POSITION";
const VOCAB: usize = 154_880;
const ROW_BYTES: usize = VOCAB * 2;

pub(super) fn parse_position(value: Option<&str>) -> Result<Option<u32>> {
    let Some(value) = value else { return Ok(None) };
    ensure!(
        !value.is_empty() && value.len() <= 4 && value.bytes().all(|byte| byte.is_ascii_digit()),
        "{ENV} must be one canonical decimal position in 0..=2047"
    );
    let position: u32 = value.parse()?;
    ensure!(
        position <= 2047 && position.to_string() == value,
        "{ENV} must be one canonical decimal position in 0..=2047"
    );
    Ok(Some(position))
}

pub(super) fn selection_from_env() -> Result<Option<u32>> {
    match std::env::var(ENV) {
        Ok(value) => parse_position(Some(&value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => bail!("{ENV} must be valid UTF-8"),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum TracePhase {
    Wide,
    Replay,
}

#[derive(Clone, Copy)]
pub(super) struct TracePoint {
    pub phase: TracePhase,
    pub start: u32,
    /// Original verifier width, also retained on its first serial replay row.
    pub rows: usize,
    pub anchor: u32,
    pub selected_oracle: Option<u32>,
    pub device_selector: bool,
    pub stream: u64,
}

#[derive(Serialize)]
pub(super) struct TraceRecord {
    pub schema: &'static str,
    pub phase: TracePhase,
    pub start: u32,
    pub rows: usize,
    pub anchor: u32,
    pub selected_oracle: Option<u32>,
    pub device_selector: bool,
    pub stream: u64,
    pub row_index: usize,
    pub dtype: &'static str,
    pub byte_order: &'static str,
    pub vocab_size: usize,
    pub payload_bytes: usize,
    pub newline_bits: u16,
    pub double_newline_bits: u16,
    pub newline_score: Option<f32>,
    pub double_newline_score: Option<f32>,
    pub cpu_first_argmax: Option<u32>,
    pub cpu_last_argmax: Option<u32>,
    pub maximum_bits: Option<u16>,
    pub maximum_score: Option<f32>,
    pub nonfinite_values: usize,
    pub raw_bf16_fnv1a64: u64,
    pub raw_bf16_base64: String,
}

/// Narrow I/O boundary: a single ordered row read and one log receipt.
pub(super) trait TraceIo {
    fn copy_row(&mut self, source: u64, destination: &mut [u8], stream: u64) -> Result<()>;
    fn publish(&mut self, record: TraceRecord) -> Result<()>;
}

fn value(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

pub(super) fn observe(
    io: &mut impl TraceIo,
    selected_position: Option<u32>,
    point: TracePoint,
    source: u64,
) -> Result<()> {
    // This branch precedes allocation, pointer validation and every I/O call.
    if selected_position != Some(point.start) {
        return Ok(());
    }
    ensure!(
        point.start <= 2047 && (2..=8).contains(&point.rows) && point.anchor < VOCAB as u32,
        "GLM verifier trace geometry is invalid"
    );
    ensure!(
        source != 0 && source.checked_add(ROW_BYTES as u64).is_some(),
        "GLM verifier trace row pointer is invalid"
    );
    ensure!(
        point
            .selected_oracle
            .is_none_or(|token| token < VOCAB as u32),
        "GLM verifier trace oracle is outside vocabulary"
    );
    let mut raw = vec![0u8; ROW_BYTES];
    io.copy_row(source, &mut raw, point.stream)?;
    let bits_at = |token: usize| u16::from_le_bytes([raw[token * 2], raw[token * 2 + 1]]);
    let mut first = None;
    let mut last = None;
    let mut maximum = f32::NEG_INFINITY;
    let mut maximum_bits = None;
    let mut nonfinite_values = 0;
    for (index, pair) in raw.chunks_exact(2).enumerate() {
        let bits = u16::from_le_bytes([pair[0], pair[1]]);
        let score = value(bits);
        if !score.is_finite() {
            nonfinite_values += 1;
            continue;
        }
        if score > maximum {
            maximum = score;
            maximum_bits = Some(bits);
            first = Some(index as u32);
            last = first;
        } else if score == maximum {
            last = Some(index as u32);
        }
    }
    let newline_bits = bits_at(198);
    let double_newline_bits = bits_at(271);
    io.publish(TraceRecord {
        schema: "atlas.glm53.verify-row0.v1",
        phase: point.phase,
        start: point.start,
        rows: point.rows,
        anchor: point.anchor,
        selected_oracle: point.selected_oracle,
        device_selector: point.device_selector,
        stream: point.stream,
        row_index: 0,
        dtype: "bf16",
        byte_order: "little",
        vocab_size: VOCAB,
        payload_bytes: ROW_BYTES,
        newline_bits,
        double_newline_bits,
        newline_score: value(newline_bits).is_finite().then(|| value(newline_bits)),
        double_newline_score: value(double_newline_bits)
            .is_finite()
            .then(|| value(double_newline_bits)),
        cpu_first_argmax: first,
        cpu_last_argmax: last,
        maximum_bits,
        maximum_score: first.map(|_| maximum),
        nonfinite_values,
        raw_bf16_fnv1a64: atlas_tier::hash::fnv1a_64(&raw),
        raw_bf16_base64: base64::engine::general_purpose::STANDARD.encode(&raw),
    })
}

struct DeviceTrace<'a>(&'a dyn GpuBackend);

impl TraceIo for DeviceTrace<'_> {
    fn copy_row(&mut self, source: u64, destination: &mut [u8], stream: u64) -> Result<()> {
        self.0
            .copy_d2h_on_stream(DevicePtr(source), destination, stream)
    }

    fn publish(&mut self, record: TraceRecord) -> Result<()> {
        let receipt =
            serde_json::to_string(&record).context("serialize GLM verifier logit trace")?;
        tracing::info!(target: "atlas::glm53_verify_trace", "GLM_VERIFY_ROW0 {receipt}");
        Ok(())
    }
}

pub(super) fn capture_row(
    gpu: &dyn GpuBackend,
    selection: Option<u32>,
    point: TracePoint,
    source: DevicePtr,
) -> Result<()> {
    observe(&mut DeviceTrace(gpu), selection, point, source.0)
}
