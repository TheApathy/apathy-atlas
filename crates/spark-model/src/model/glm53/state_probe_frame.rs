// SPDX-License-Identifier: AGPL-3.0-only

//! Value-only admission for an exclusively borrowed committed model snapshot.
use anyhow::{Result, ensure};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct StateProbeFrame {
    pub generation: u64,
    pub nonce: u64,
    pub position: u32,
    pub context: u32,
    pub capacity: u32,
    pub context_capacity: u32,
    pub stream: u64,
    pub model_stream: u64,
    pub live: bool,
    pub poisoned: bool,
    pub capturing: bool,
    pub prefill: bool,
}

#[derive(Debug)]
pub struct StateProbeStamp(StateProbeFrame);

impl StateProbeStamp {
    pub(super) fn new(frame: StateProbeFrame) -> Result<Self> {
        ensure!(
            frame.generation != 0
                && frame.nonce != 0
                && frame.capacity > 0
                && frame.context_capacity > 0
                && frame.position > 0
                && frame.position <= frame.capacity
                && frame.context == frame.position
                && frame.context <= frame.context_capacity
                && frame.stream == frame.model_stream
                && frame.live
                && !frame.poisoned
                && !frame.capturing
                && !frame.prefill,
            "state probe requires a healthy committed live frame on the model stream"
        );
        Ok(Self(frame))
    }

    pub(super) fn check(&self, frame: StateProbeFrame) -> Result<()> {
        let current = Self::new(frame)?;
        ensure!(
            self.0 == current.0,
            "state probe frame changed since admission"
        );
        Ok(())
    }

    pub fn generation(&self) -> u64 {
        self.0.generation
    }
    pub fn nonce(&self) -> u64 {
        self.0.nonce
    }
    pub fn position(&self) -> u32 {
        self.0.position
    }
    pub fn context(&self) -> u32 {
        self.0.context
    }
    pub fn stream(&self) -> u64 {
        self.0.stream
    }
}
