// SPDX-License-Identifier: AGPL-3.0-only
//! Actual byte-copying backend: projection only checks ordering, not GPU math.
use super::prefill_capture_ingest::PrefillCaptureIo;
use super::prefill_capture_plan::{CopyStep, ProjectionStep};
use super::{BANK, PROJECTED, STAGING, complete_receipt};
use anyhow::{Result, bail, ensure};

pub(super) struct MemoryIo {
    bank: Vec<u8>,
    staging: Vec<u8>,
    projected: Vec<u8>,
    calls: usize,
    fail_at: Option<usize>,
    fail_drain: bool,
    syncs: usize,
    projections: Vec<u32>,
    consumed: u32,
}

fn tag(slot: usize, row: usize, column: usize) -> u16 {
    (slot * 10000 + row * 200 + column % 199) as u16
}

impl MemoryIo {
    pub(super) fn new() -> Self {
        let mut bank = vec![0; 5 * 32 * 8192];
        for slot in 0..5 {
            for row in 0..32 {
                for column in 0..4096 {
                    let offset = ((slot * 32 + row) * 4096 + column) * 2;
                    bank[offset..offset + 2].copy_from_slice(&tag(slot, row, column).to_le_bytes());
                }
            }
        }
        Self {
            bank,
            staging: vec![0xcc; 8 * 5 * 8192],
            projected: vec![0xa5; 2047 * 8192],
            calls: 0,
            fail_at: None,
            fail_drain: false,
            syncs: 0,
            projections: Vec::new(),
            consumed: 0,
        }
    }

    pub(super) fn calls(&self) -> usize {
        self.calls
    }

    fn step(&mut self, stream: u64) -> Result<()> {
        ensure!(stream == 7, "wrong owned stream");
        self.calls += 1;
        if self.fail_at == Some(self.calls) {
            bail!("injected IO failure");
        }
        Ok(())
    }
}

impl PrefillCaptureIo for MemoryIo {
    fn copy(&mut self, step: CopyStep, stream: u64) -> Result<()> {
        self.step(stream)?;
        let source = usize::try_from(step.source.checked_sub(BANK).unwrap()).unwrap();
        let destination = usize::try_from(step.destination.checked_sub(STAGING).unwrap()).unwrap();
        self.staging[destination..destination + step.bytes]
            .copy_from_slice(&self.bank[source..source + step.bytes]);
        Ok(())
    }

    fn project_norm(&mut self, step: ProjectionStep, stream: u64) -> Result<()> {
        self.step(stream)?;
        ensure!(step.input.address == STAGING && step.rows <= 8 && step.rows > 0);
        ensure!(step.input.bytes == step.rows as usize * 5 * 8192);
        ensure!(step.output.bytes == step.rows as usize * 8192);
        ensure!(step.output.address == PROJECTED + ((11 + self.consumed) * 8192) as u64);
        for row in 0..step.rows as usize {
            for slot in 0..5 {
                for column in 0..4096 {
                    let offset = ((row * 5 + slot) * 4096 + column) * 2;
                    ensure!(
                        self.staging[offset..offset + 2]
                            == tag(slot, self.consumed as usize + row, column).to_le_bytes()
                    );
                }
            }
            let output = (11 + self.consumed as usize + row) * 8192;
            self.projected[output..output + 8192].fill((self.consumed as usize + row) as u8);
        }
        self.consumed += step.rows;
        self.projections.push(step.rows);
        Ok(())
    }

    fn synchronize(&mut self, stream: u64) -> Result<()> {
        self.syncs += 1;
        self.step(stream)?;
        ensure!(!self.fail_drain, "injected drain failure");
        Ok(())
    }
}

#[test]
fn real_copies_gather_all_taps_then_publish_only_after_final_fence() {
    let mut receipt = complete_receipt();
    let mut io = MemoryIo::new();
    let bank_before = io.bank.clone();
    assert_eq!(receipt.published_position(), None);
    assert_eq!(receipt.ingest(&mut io, 28, 11, 7).unwrap(), 28);
    assert_eq!(receipt.published_position(), Some(28));
    assert_eq!(io.projections, [8, 8, 1]);
    assert_eq!(io.syncs, 1);
    assert_eq!(io.calls, 85 + 3 + 1);
    assert_eq!(io.bank, bank_before);
    assert!(io.projected[..11 * 8192].iter().all(|v| *v == 0xa5));
    assert!(io.projected[28 * 8192..].iter().all(|v| *v == 0xa5));
    for row in 0..17 {
        assert!(
            io.projected[(11 + row) * 8192..(12 + row) * 8192]
                .iter()
                .all(|v| *v == row as u8)
        );
    }
    assert!(receipt.ingest(&mut io, 28, 11, 7).is_err());
    assert_eq!(io.calls, 89);
}

#[test]
fn every_copy_projection_and_final_fence_failure_prevents_publication_and_retry() {
    for fail_at in 1..=89 {
        let mut receipt = complete_receipt();
        let mut io = MemoryIo::new();
        io.fail_at = Some(fail_at);
        assert!(
            receipt.ingest(&mut io, 28, 11, 7).is_err(),
            "failure {fail_at}"
        );
        assert_eq!(receipt.published_position(), None, "failure {fail_at}");
        assert!(receipt.failed(), "failure {fail_at}");
        assert!(
            io.syncs >= 1,
            "must drain already-enqueued IO after failure {fail_at}"
        );
        let calls = io.calls;
        assert!(receipt.ingest(&mut io, 28, 11, 7).is_err());
        assert_eq!(io.calls, calls);
    }
}

#[test]
fn failed_error_drain_remains_terminal_and_never_claims_rollback() {
    let mut receipt = complete_receipt();
    let mut io = MemoryIo::new();
    io.fail_at = Some(42); // First slice output already written; second slice copy fails.
    io.fail_drain = true;
    assert!(receipt.ingest(&mut io, 28, 11, 7).is_err());
    assert_eq!(receipt.published_position(), None);
    assert!(receipt.failed());
    assert_eq!(io.projections, [8]);
    assert!(io.projected[11 * 8192..12 * 8192].iter().all(|v| *v == 0));
    assert_eq!(io.syncs, 1);
}
