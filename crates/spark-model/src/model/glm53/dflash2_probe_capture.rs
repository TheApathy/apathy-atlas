// SPDX-License-Identifier: AGPL-3.0-only
//! Actual owned host capture used by the explicit GPU parity example.
use super::dflash2_probe_contract::{ProbeLayout, ProbeStage};
use super::owned_verify_readback::{OwnedReadback, ReadbackIo};
use anyhow::{Context, Result, ensure};

pub trait ProbeIo {
    fn copy(&mut self, source: u64, dst: &mut [u8], stream: u64) -> Result<()>;
    fn drain(&mut self, stream: u64) -> Result<()>;
}
struct CopyIo<'a> {
    source: u64,
    io: &'a mut dyn ProbeIo,
}
impl ReadbackIo for CopyIo<'_> {
    fn copy(&mut self, dst: &mut [u8], stream: u64) -> Result<()> {
        self.io.copy(self.source, dst, stream)
    }
    fn drain(&mut self, stream: u64) -> Result<()> {
        self.io.drain(stream)
    }
}
pub struct ProbeFrame {
    pub stage: ProbeStage,
    pub bytes: Vec<u8>,
}
pub struct ProbeCapture {
    layout: ProbeLayout,
    context: u32,
    stream: u64,
    readback: OwnedReadback,
    frames: Vec<ProbeFrame>,
    failed: bool,
}
impl ProbeCapture {
    pub fn new(layout: ProbeLayout, context: u32, stream: u64) -> Result<Self> {
        ensure!(context > 0, "empty probe context");
        ensure!(
            layout
                .projected_context()
                .is_none_or(|rows| rows == context),
            "projected input context differs from observer context"
        );
        let readback = OwnedReadback::new(layout.max_stage_bytes())?;
        Ok(Self {
            layout,
            context,
            stream,
            readback,
            frames: Vec::new(),
            failed: false,
        })
    }
    pub fn context(&self) -> u32 {
        self.context
    }
    pub fn wants_projected_target(&self) -> bool {
        self.layout.projected_context().is_some()
    }
    pub fn admit(&self, layout: &ProbeLayout, context: u32, stream: u64) -> Result<()> {
        ensure!(
            !self.failed && !self.pending() && self.frames.is_empty(),
            "probe observer is not fresh"
        );
        ensure!(
            context == self.context
                && stream == self.stream
                && layout.vocab() == self.layout.vocab()
                && layout.projected_context() == self.layout.projected_context()
                && layout.stages() == self.layout.stages(),
            "probe observer identity changed"
        );
        for &stage in layout.stages() {
            ensure!(
                layout.bytes(stage)? == self.layout.bytes(stage)?,
                "probe observer geometry changed"
            );
        }
        Ok(())
    }
    pub fn pending(&self) -> bool {
        self.readback.pending()
    }
    pub fn observe(
        &mut self,
        stage: ProbeStage,
        source: u64,
        bytes: usize,
        stream: u64,
        io: &mut dyn ProbeIo,
    ) -> Result<()> {
        ensure!(
            !self.failed && !self.pending(),
            "failed/pending diagnostic cannot be reused"
        );
        ensure!(
            self.layout.stages().get(self.frames.len()) == Some(&stage),
            "diagnostic stage missing, repeated, or out of order"
        );
        ensure!(
            bytes == self.layout.bytes(stage)? && stream == self.stream,
            "diagnostic extent/stream changed"
        );
        let alignment = if stage == ProbeStage::DraftIds { 4 } else { 2 };
        ensure!(
            source > 0
                && source % alignment == 0
                && source.checked_add(u64::try_from(bytes)?).is_some(),
            "invalid diagnostic source span"
        );
        self.failed = true; // Also terminal after a caught I/O panic.
        let raw = self
            .readback
            .read(bytes, stream, &mut CopyIo { source, io })?;
        let mut retained = Vec::new();
        retained
            .try_reserve_exact(bytes)
            .context("diagnostic frame allocation")?;
        retained.extend_from_slice(raw);
        self.frames.push(ProbeFrame {
            stage,
            bytes: retained,
        });
        let raw = &self.frames.last().unwrap().bytes;
        // Even invalid raw payloads are retained for the failure artifact.
        if stage == ProbeStage::DraftIds {
            ensure!(
                raw.chunks_exact(4)
                    .all(|v| u32::from_le_bytes(v.try_into().unwrap()) < self.layout.vocab()),
                "diagnostic draft ID outside vocabulary"
            );
        } else {
            ensure!(
                raw.chunks_exact(2)
                    .all(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7f80 != 0x7f80),
                "diagnostic contains nonfinite BF16"
            );
        }
        self.failed = false;
        Ok(())
    }
    pub fn frames(&self) -> Result<&[ProbeFrame]> {
        ensure!(
            !self.failed && !self.pending() && self.frames.len() == self.layout.stages().len(),
            "diagnostic receipt incomplete or failed"
        );
        Ok(&self.frames)
    }
    pub fn retained_frames(&self) -> &[ProbeFrame] {
        &self.frames
    }
    pub fn drain(&mut self, io: &mut dyn ProbeIo) -> Result<()> {
        self.readback.drain(&mut CopyIo { source: 0, io })
    }
}
