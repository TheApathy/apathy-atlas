// SPDX-License-Identifier: AGPL-3.0-only

//! One lazily allocated capture bank. Never free or reuse in-flight storage.
//! The model holds its mutex across begin, target execution, commit and ingestion.
//! Explicit release is required; failure retains ownership for reset/shutdown.

use anyhow::{Context, Result, ensure};

use super::prefill_capture_plan::{DeviceSpan, PrefillCapturePlan};

pub(crate) trait CaptureBankIo {
    fn allocate(&mut self, bytes: usize) -> Result<u64>;
    fn synchronize(&mut self, stream: u64) -> Result<()>;
    fn free(&mut self, address: u64) -> Result<()>;
}

#[must_use = "capture bank owns device memory and requires explicit release"]
pub(crate) struct CaptureBankOwner {
    allocation: Option<DeviceSpan>,
    stream: Option<u64>,
    poisoned: bool,
}

impl CaptureBankOwner {
    pub(crate) fn new() -> Self {
        Self {
            allocation: None,
            stream: None,
            poisoned: false,
        }
    }

    pub(crate) fn has_owner(&self) -> bool {
        self.allocation.is_some()
    }
    pub(crate) fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Arms the owner before the caller can enqueue any use of the returned span.
    /// A larger request frees the idle old bank before allocating its replacement.
    pub(crate) fn begin(
        &mut self,
        io: &mut impl CaptureBankIo,
        plan: &PrefillCapturePlan,
        stream: u64,
    ) -> Result<DeviceSpan> {
        ensure!(
            !self.is_poisoned() && self.stream.is_none(),
            "capture bank is active or poisoned; release/reset required"
        );
        let required = plan.bank_bytes();
        if let Some(old) = self.allocation {
            if old.bytes < required {
                self.poisoned = true;
                io.free(old.address)
                    .context("capture bank growth free failed; owner retained")?;
                self.allocation = None;
                self.poisoned = false;
            }
        }
        if self.allocation.is_none() {
            let address = io.allocate(required)?;
            let bank = DeviceSpan {
                address,
                bytes: required,
            };
            // Even a malformed allocator return represents owned storage.
            self.allocation = Some(bank);
            self.poisoned = true;
            bank.validate(required)?;
            self.poisoned = false;
        }
        self.stream = Some(stream);
        self.allocation
            .context("capture allocation missing after successful admission")
    }

    /// Only call after successful target commit and successful drafter ingestion.
    pub(crate) fn complete(&mut self, io: &mut impl CaptureBankIo) -> Result<()> {
        ensure!(
            !self.is_poisoned(),
            "capture bank cannot complete after failure"
        );
        let stream = self
            .stream
            .context("capture bank completion requires active work")?;
        self.poisoned = true;
        io.synchronize(stream)
            .context("capture completion fence failed; owner retained")?;
        self.stream = None;
        self.poisoned = false;
        Ok(())
    }

    /// Draining is not rollback. A failed sequence remains poisoned until release.
    pub(crate) fn abort(&mut self, io: &mut impl CaptureBankIo) -> Result<()> {
        ensure!(self.has_owner(), "capture abort requires an owned bank");
        self.poisoned = true;
        if let Some(stream) = self.stream {
            io.synchronize(stream)
                .context("capture abort drain failed; owner retained")?;
            self.stream = None;
        }
        Ok(())
    }

    pub(crate) fn release(&mut self, io: &mut impl CaptureBankIo) -> Result<()> {
        let Some(bank) = self.allocation else {
            return Ok(());
        };
        self.poisoned = true;
        if let Some(stream) = self.stream {
            io.synchronize(stream)
                .context("capture release drain failed; owner retained")?;
            self.stream = None;
        }
        io.free(bank.address)
            .context("capture bank free failed; owner retained")?;
        self.allocation = None;
        self.poisoned = false;
        Ok(())
    }
}
