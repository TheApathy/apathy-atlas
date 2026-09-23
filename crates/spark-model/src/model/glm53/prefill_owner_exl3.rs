// SPDX-License-Identifier: AGPL-3.0-only

//! Single prepared allocation; asynchronous failure never relinquishes its owner.
//! GPU effects are supplied by a narrow adapter. Descriptors are not completion
//! receipts, and a quarantined allocation cannot be borrowed for another request.

use super::prefill_input_exl3::PreparedRows;
use anyhow::{Context, Result, ensure};

pub(super) trait OwnerIo {
    fn drain(&mut self, stream: u64) -> Result<()>;
    fn free(&mut self, address: u64) -> Result<()>;
}

#[derive(Clone, Copy)]
enum State {
    Empty,
    Ready(PreparedRows),
    Active {
        rows: PreparedRows,
        stream: Option<u64>,
    },
    Quarantined {
        rows: PreparedRows,
        stream: Option<u64>,
    },
}

pub(super) struct PreparedOwner {
    state: State,
}

impl PreparedOwner {
    pub fn new() -> Self {
        Self {
            state: State::Empty,
        }
    }
    pub fn has_owner(&self) -> bool {
        !matches!(self.state, State::Empty)
    }
    pub fn is_quarantined(&self) -> bool {
        matches!(self.state, State::Quarantined { .. })
    }

    pub fn publish(&mut self, rows: PreparedRows) -> Result<()> {
        ensure!(!self.has_owner(), "GLM prepared owner is already occupied");
        rows.validate()?;
        self.state = State::Ready(rows);
        Ok(())
    }

    /// Borrow metadata only; the allocation remains owned by this state machine.
    pub fn begin(&mut self) -> Result<Option<PreparedRows>> {
        match self.state {
            State::Empty => Ok(None),
            State::Ready(rows) => {
                self.state = State::Active { rows, stream: None };
                Ok(Some(rows))
            }
            _ => anyhow::bail!("GLM prepared owner is active or quarantined"),
        }
    }

    /// Arm before the first operation that may enqueue a read or write.
    pub fn arm(&mut self, stream: u64) -> Result<()> {
        match self.state {
            State::Active {
                rows,
                stream: previous,
            } => {
                ensure!(
                    previous.is_none_or(|old| old == stream),
                    "GLM prepared allocation cannot switch active streams"
                );
                self.state = State::Active {
                    rows,
                    stream: Some(stream),
                };
                Ok(())
            }
            _ => anyhow::bail!("GLM prepared owner cannot arm outside an active operation"),
        }
    }

    /// Encoder-only success: expose Ready only after a successful completion fence.
    pub fn complete_preparation(&mut self, io: &mut impl OwnerIo) -> Result<()> {
        let State::Active {
            rows,
            stream: Some(stream),
        } = self.state
        else {
            anyhow::bail!("GLM preparation completion requires an armed owner");
        };
        self.state = State::Quarantined {
            rows,
            stream: Some(stream),
        };
        io.drain(stream)
            .context("GLM encoder output drain failed; owner quarantined")?;
        self.state = State::Ready(rows);
        Ok(())
    }

    pub fn finish(&mut self, io: &mut impl OwnerIo) -> Result<()> {
        match self.state {
            State::Empty => Ok(()),
            State::Active { rows, stream } => {
                self.state = State::Quarantined { rows, stream };
                self.retry_quarantine(io)
            }
            _ => anyhow::bail!("GLM prepared owner is not an active request"),
        }
    }

    pub fn retry_quarantine(&mut self, io: &mut impl OwnerIo) -> Result<()> {
        if !self.is_quarantined() {
            return Ok(());
        }
        let State::Quarantined { rows, stream } = self.state else {
            unreachable!()
        };
        if let Some(stream) = stream {
            io.drain(stream)
                .context("GLM prepared stream must drain; owner retained")?;
            self.state = State::Quarantined { rows, stream: None };
        }
        io.free(rows.region.address)
            .context("GLM prepared free failed; owner retained")?;
        self.state = State::Empty;
        Ok(())
    }

    /// Before preparation/shutdown. A live request is never stolen or freed.
    pub fn clear(&mut self, io: &mut impl OwnerIo) -> Result<()> {
        match self.state {
            State::Active { .. } => anyhow::bail!("GLM prepared owner is still active"),
            State::Ready(rows) => {
                self.state = State::Quarantined { rows, stream: None };
                self.retry_quarantine(io)
            }
            _ => self.retry_quarantine(io),
        }
    }
}

#[cfg(test)]
#[path = "prefill_owner_exl3_tests.rs"]
mod tests;
