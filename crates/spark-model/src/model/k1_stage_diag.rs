// SPDX-License-Identifier: AGPL-3.0-only

//! Exact in-process stage comparison for the C=1 DFlash verifier.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use anyhow::{Result, bail};

#[path = "k1_stage_diag_types.rs"]
mod types;
pub use types::{CaptureManifest, FirstDivergence, StageReport, fnv1a64};
#[path = "k1_stage_diag_selector.rs"]
mod selector;
pub use selector::{parse_selector, parse_tokens_selector, requested_at, selector_matches};

pub const CONTROLLED_SERIAL_FAMILY: &str = "ffn_layer_norms";
pub const CONTROLLED_LM_HEAD_FAMILY: &str = "ffn_layer_norms_lm_head";
pub const BASELINE_FAMILY: &str = "baseline";

fn manifest_family(serial_family: Option<&str>) -> &str {
    serial_family.unwrap_or(BASELINE_FAMILY)
}

pub fn validate_serial_control_overlap(
    stage_diag_enabled: bool,
    serial_family: Option<&str>,
) -> Result<()> {
    match (stage_diag_enabled, serial_family) {
        (false, _)
        | (_, None)
        | (true, Some(CONTROLLED_SERIAL_FAMILY | CONTROLLED_LM_HEAD_FAMILY)) => Ok(()),
        (true, Some(family)) => bail!(
            "DFLASH_K1_STAGE_DIAG serial-family overlap requires exactly \
             {CONTROLLED_SERIAL_FAMILY} or {CONTROLLED_LM_HEAD_FAMILY}; got {family}"
        ),
    }
}

#[derive(Debug, Default)]
struct StageRows {
    name: String,
    serial: Vec<Vec<u8>>,
}

#[derive(Debug)]
struct Capture {
    manifest: CaptureManifest,
    row: usize,
    cursor: usize,
    stages: Vec<StageRows>,
    first: Option<FirstDivergence>,
}

#[derive(Debug, Default)]
enum Phase {
    #[default]
    Idle,
    Serial(Capture),
    Ready(Capture),
    Batch(Capture),
}

fn state() -> &'static Mutex<Phase> {
    static STATE: OnceLock<Mutex<Phase>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(Phase::Idle))
}

pub(super) fn completed_flag() -> &'static AtomicBool {
    static DONE: AtomicBool = AtomicBool::new(false);
    &DONE
}

pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("ATLAS_DFLASH_K1_STAGE_DIAG").ok().as_deref() == Some("1"))
}

pub fn serial_active() -> bool {
    matches!(
        *state().lock().expect("K1 stage diagnostic mutex"),
        Phase::Serial(_)
    )
}

pub fn batch_active() -> bool {
    matches!(
        *state().lock().expect("K1 stage diagnostic mutex"),
        Phase::Batch(_)
    )
}

/// Whether the active exact comparator owns this stage. The combined LM-head
/// control intentionally ends at final norm: its K1 head reuses one logits
/// row, so treating that buffer as a preserved `[K, V]` slab would fabricate
/// raw-logit equality evidence.
pub fn capture_stage(stage: &str) -> bool {
    if stage != "logits" {
        return true;
    }
    let phase = state().lock().expect("K1 stage diagnostic mutex");
    let family = match &*phase {
        Phase::Serial(capture) | Phase::Ready(capture) | Phase::Batch(capture) => {
            capture.manifest.family.as_str()
        }
        Phase::Idle => return true,
    };
    family != CONTROLLED_LM_HEAD_FAMILY
}

pub fn begin_serial(pre_verify_len: usize, tokens: &[u32]) -> Result<()> {
    let serial_family = crate::model::env_diag::DflashSerialControls::current().active_family()?;
    begin_serial_with_family(pre_verify_len, tokens, serial_family)
}

fn begin_serial_with_family(
    pre_verify_len: usize,
    tokens: &[u32],
    serial_family: Option<&str>,
) -> Result<()> {
    validate_serial_control_overlap(true, serial_family)?;
    if tokens.is_empty() || tokens.len() > 32 {
        bail!("DFLASH_K1_STAGE_DIAG rows must be in 1..=32");
    }
    let mut phase = state().lock().expect("K1 stage diagnostic mutex");
    if !matches!(*phase, Phase::Idle) {
        bail!("DFLASH_K1_STAGE_DIAG overlapping capture");
    }
    static STEP: AtomicU64 = AtomicU64::new(0);
    let verify_step = STEP.fetch_add(1, Ordering::Relaxed);
    *phase = Phase::Serial(Capture {
        manifest: CaptureManifest {
            run_id: format!("pid{}-verify{verify_step}", std::process::id()),
            verify_step,
            pre_verify_len,
            tokens: tokens.to_vec(),
            absolute_seq_lens: (0..tokens.len()).map(|row| pre_verify_len + row).collect(),
            family: manifest_family(serial_family).to_owned(),
        },
        row: 0,
        cursor: 0,
        stages: Vec::new(),
        first: None,
    });
    Ok(())
}

pub fn record_serial(stage: &str, bytes: Vec<u8>) -> Result<()> {
    let mut phase = state().lock().expect("K1 stage diagnostic mutex");
    let Phase::Serial(capture) = &mut *phase else {
        bail!("DFLASH_K1_STAGE_DIAG serial stage outside replay");
    };
    if bytes.is_empty() {
        bail!("DFLASH_K1_STAGE_DIAG empty serial stage {stage}");
    }
    if capture.row == 0 {
        capture.stages.push(StageRows {
            name: stage.to_owned(),
            serial: vec![bytes],
        });
    } else {
        let expected = capture
            .stages
            .get_mut(capture.cursor)
            .ok_or_else(|| anyhow::anyhow!("unexpected serial stage {stage}"))?;
        if expected.name != stage || expected.serial.len() != capture.row {
            bail!("DFLASH_K1_STAGE_DIAG serial stage order drift at {stage}");
        }
        expected.serial.push(bytes);
    }
    capture.cursor += 1;
    Ok(())
}

pub fn finish_serial_row() -> Result<()> {
    let mut phase = state().lock().expect("K1 stage diagnostic mutex");
    let Phase::Serial(capture) = &mut *phase else {
        bail!("DFLASH_K1_STAGE_DIAG row finish outside replay");
    };
    if capture.stages.is_empty() || capture.cursor != capture.stages.len() {
        bail!("DFLASH_K1_STAGE_DIAG incomplete serial row {}", capture.row);
    }
    capture.row += 1;
    capture.cursor = 0;
    if capture.row > capture.manifest.tokens.len() {
        bail!("DFLASH_K1_STAGE_DIAG too many serial rows");
    }
    Ok(())
}

pub fn finish_serial() -> Result<()> {
    let mut phase = state().lock().expect("K1 stage diagnostic mutex");
    let Phase::Serial(capture) = std::mem::take(&mut *phase) else {
        bail!("DFLASH_K1_STAGE_DIAG serial finish outside replay");
    };
    if capture.row != capture.manifest.tokens.len() || capture.cursor != 0 {
        bail!("DFLASH_K1_STAGE_DIAG incomplete serial replay");
    }
    *phase = Phase::Ready(capture);
    Ok(())
}

pub fn begin_batch(pre_verify_len: usize, tokens: &[u32]) -> Result<()> {
    let serial_family = crate::model::env_diag::DflashSerialControls::current().active_family()?;
    begin_batch_with_family(pre_verify_len, tokens, serial_family)
}

fn begin_batch_with_family(
    pre_verify_len: usize,
    tokens: &[u32],
    serial_family: Option<&str>,
) -> Result<()> {
    validate_serial_control_overlap(true, serial_family)?;
    let mut phase = state().lock().expect("K1 stage diagnostic mutex");
    let Phase::Ready(capture) = std::mem::take(&mut *phase) else {
        bail!("DFLASH_K1_STAGE_DIAG lacks completed serial replay");
    };
    if capture.manifest.pre_verify_len != pre_verify_len || capture.manifest.tokens != tokens {
        bail!("DFLASH_K1_STAGE_DIAG serial/batch frame identity mismatch");
    }
    if capture.manifest.family != manifest_family(serial_family) {
        bail!("DFLASH_K1_STAGE_DIAG serial/batch control-family identity mismatch");
    }
    *phase = Phase::Batch(capture);
    Ok(())
}

pub fn record_batch(stage: &str, bytes: &[u8], row_bytes: usize) -> Result<()> {
    let mut phase = state().lock().expect("K1 stage diagnostic mutex");
    let Phase::Batch(capture) = &mut *phase else {
        bail!("DFLASH_K1_STAGE_DIAG batch stage outside verify");
    };
    let rows = capture.manifest.tokens.len();
    if row_bytes == 0 || bytes.len() != rows * row_bytes {
        bail!("DFLASH_K1_STAGE_DIAG invalid batch extent for {stage}");
    }
    let expected = capture
        .stages
        .get(capture.cursor)
        .ok_or_else(|| anyhow::anyhow!("unexpected batch stage {stage}"))?;
    if expected.name != stage || expected.serial.len() != rows {
        bail!("DFLASH_K1_STAGE_DIAG batch stage order drift at {stage}");
    }
    if expected.serial.iter().any(|row| row.len() != row_bytes) {
        bail!("DFLASH_K1_STAGE_DIAG serial/batch byte extent mismatch at {stage}");
    }
    let mismatch_rows: Vec<usize> = (0..rows)
        .filter(|&row| expected.serial[row] != bytes[row * row_bytes..(row + 1) * row_bytes])
        .collect();
    if capture.first.is_none()
        && let Some(&row) = mismatch_rows.first()
    {
        let batch = &bytes[row * row_bytes..(row + 1) * row_bytes];
        let serial = &expected.serial[row];
        capture.first = Some(FirstDivergence {
            stage: stage.to_owned(),
            row,
            first_byte: serial
                .iter()
                .zip(batch)
                .position(|(left, right)| left != right)
                .unwrap_or(serial.len().min(batch.len())),
            serial_hash: fnv1a64(serial),
            batch_hash: fnv1a64(batch),
            mismatch_rows,
        });
    }
    capture.cursor += 1;
    Ok(())
}

pub fn finish_batch() -> Result<StageReport> {
    let mut phase = state().lock().expect("K1 stage diagnostic mutex");
    let Phase::Batch(capture) = std::mem::take(&mut *phase) else {
        bail!("DFLASH_K1_STAGE_DIAG batch finish outside verify");
    };
    if capture.cursor != capture.stages.len() {
        bail!("DFLASH_K1_STAGE_DIAG incomplete batched stage sequence");
    }
    let terminal_stage = capture
        .stages
        .last()
        .map(|stage| stage.name.clone())
        .ok_or_else(|| anyhow::anyhow!("DFLASH_K1_STAGE_DIAG empty completed capture"))?;
    let logits_compared = terminal_stage == "logits";
    completed_flag().store(true, Ordering::Relaxed);
    Ok(StageReport {
        manifest: capture.manifest,
        stages: capture.stages.len(),
        terminal_stage,
        logits_compared,
        first: capture.first,
    })
}

pub fn abort() {
    *state().lock().expect("K1 stage diagnostic mutex") = Phase::Idle;
}

#[cfg(test)]
#[path = "k1_stage_diag_tests.rs"]
mod tests;
