// SPDX-License-Identifier: AGPL-3.0-only

//! Stream-ordered save/restore and irreversible-commit ownership for DSA verify.

use super::dsa_verify_plan::{DSA_LAYERS, DsaVerifyPlan, MAX_BACKUP_BYTES};
use anyhow::{Context, Result, ensure};

#[derive(Clone, Copy, Debug)]
pub(super) struct DeviceRegion {
    pub address: u64,
    pub bytes: usize,
}

impl DeviceRegion {
    fn end(self) -> Result<u64> {
        ensure!(
            self.address != 0 && self.bytes != 0,
            "DSA snapshot has a null/empty region"
        );
        self.address
            .checked_add(u64::try_from(self.bytes)?)
            .context("DSA snapshot region address overflow")
    }
}

pub(super) trait CopyIo {
    fn copy(&mut self, source: u64, destination: u64, bytes: usize, stream: u64) -> Result<()>;
    fn synchronize(&mut self, stream: u64) -> Result<()>;
}

#[derive(Clone, Copy)]
struct CopySpan {
    source: u64,
    backup: u64,
    bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Ready,
    Saved,
    Restored,
    Committing,
    Complete,
    Aborted,
    Poisoned,
}

pub(super) struct VerifySnapshot {
    spans: Vec<CopySpan>,
    stream: u64,
    phase: Phase,
}

impl VerifySnapshot {
    /// Validate every source and destination before allowing the first copy.
    pub fn bind(
        plan: &DsaVerifyPlan,
        layers: &[[DeviceRegion; 6]],
        backup: DeviceRegion,
        stream: u64,
    ) -> Result<Self> {
        ensure!(
            layers.len() == DSA_LAYERS,
            "DSA snapshot requires all eleven layers"
        );
        ensure!(
            backup.bytes == MAX_BACKUP_BYTES && plan.backup_bytes() <= backup.bytes,
            "DSA snapshot backup must have the exact bounded extent"
        );
        let backup_end = backup.end()?;
        let mut occupied = Vec::with_capacity(DSA_LAYERS * 6);
        let mut spans = Vec::with_capacity(DSA_LAYERS * 6);
        for (layer_index, layer) in layers.iter().enumerate() {
            for (slot, (source, region)) in layer.iter().zip(plan.source_regions()).enumerate() {
                let end = source.end()?;
                ensure!(
                    end <= backup.address || backup_end <= source.address,
                    "DSA snapshot backup aliases persistent state"
                );
                for &(other_start, other_end) in &occupied {
                    ensure!(
                        end <= other_start || other_end <= source.address,
                        "DSA snapshot persistent regions overlap"
                    );
                }
                occupied.push((source.address, end));
                ensure!(
                    region
                        .offset
                        .checked_add(region.bytes)
                        .is_some_and(|end| end <= source.bytes),
                    "DSA snapshot source span exceeds its region"
                );
                let offset = plan.backup_offset(layer_index, slot)?;
                ensure!(
                    offset
                        .checked_add(region.bytes)
                        .is_some_and(|end| end <= plan.backup_bytes()),
                    "DSA snapshot destination span exceeds its plan"
                );
                if region.bytes != 0 {
                    spans.push(CopySpan {
                        source: source
                            .address
                            .checked_add(u64::try_from(region.offset)?)
                            .context("DSA snapshot source overflow")?,
                        backup: backup
                            .address
                            .checked_add(u64::try_from(offset)?)
                            .context("DSA snapshot backup overflow")?,
                        bytes: region.bytes,
                    });
                }
            }
        }
        ensure!(
            spans.iter().map(|span| span.bytes).sum::<usize>() == plan.payload_bytes(),
            "DSA snapshot payload coverage drift"
        );
        Ok(Self {
            spans,
            stream,
            phase: Phase::Ready,
        })
    }

    pub fn save(&mut self, io: &mut impl CopyIo) -> Result<()> {
        ensure!(
            self.phase == Phase::Ready,
            "DSA snapshot save is out of order"
        );
        for span in &self.spans {
            if let Err(error) = io.copy(span.source, span.backup, span.bytes, self.stream) {
                // Target is untouched, but previously enqueued reads must drain
                // before the one reusable backup can be reused or freed.
                self.phase = Phase::Aborted;
                return match io.synchronize(self.stream) {
                    Ok(()) => Err(error.context("DSA snapshot save failed; target untouched")),
                    Err(drain) => {
                        self.phase = Phase::Poisoned;
                        Err(error.context(format!("DSA snapshot save drain failed: {drain:#}")))
                    }
                };
            }
        }
        self.phase = Phase::Saved;
        Ok(())
    }

    pub fn restore(&mut self, io: &mut impl CopyIo) -> Result<()> {
        ensure!(
            self.phase == Phase::Saved,
            "DSA snapshot restore is out of order"
        );
        // Sticky until every restore copy AND its stream completion succeeds.
        self.phase = Phase::Poisoned;
        let mut copy_error = None;
        for span in &self.spans {
            if let Err(error) = io.copy(span.backup, span.source, span.bytes, self.stream) {
                copy_error = Some(error);
                break;
            }
        }
        let drained = io.synchronize(self.stream);
        match (copy_error, drained) {
            (None, Ok(())) => {
                self.phase = Phase::Restored;
                Ok(())
            }
            (Some(error), Ok(())) => {
                Err(error.context("DSA snapshot restore failed; sequence poisoned"))
            }
            (None, Err(error)) => {
                Err(error.context("DSA snapshot restore completion failed; sequence poisoned"))
            }
            (Some(error), Err(drain)) => Err(error.context(format!(
                "DSA snapshot restore also failed to drain: {drain:#}"
            ))),
        }
    }

    pub fn begin_commit(&mut self) -> Result<()> {
        ensure!(
            matches!(self.phase, Phase::Saved | Phase::Restored),
            "DSA snapshot commit is out of order"
        );
        self.phase = Phase::Committing;
        Ok(())
    }

    pub fn finish(&mut self, io: &mut impl CopyIo) -> Result<()> {
        ensure!(
            self.phase == Phase::Committing,
            "DSA snapshot completion is out of order"
        );
        self.phase = Phase::Poisoned;
        io.synchronize(self.stream)
            .context("GLM verify commit completion failed; sequence poisoned")?;
        self.phase = Phase::Complete;
        Ok(())
    }

    pub fn fail_commit(&mut self, io: &mut impl CopyIo) -> Result<()> {
        ensure!(
            self.phase == Phase::Committing,
            "DSA snapshot failed commit is out of order"
        );
        self.phase = Phase::Poisoned;
        io.synchronize(self.stream)
            .context("GLM failed commit also failed to drain")
    }

    pub fn needs_reset(&self) -> bool {
        self.phase == Phase::Poisoned
    }
}
