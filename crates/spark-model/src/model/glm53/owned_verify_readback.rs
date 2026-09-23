// SPDX-License-Identifier: AGPL-3.0-only
//! Model-owned host storage for a fallible, potentially asynchronous readback.

use anyhow::{Context, Result, bail, ensure};

/// The adapter binds the source device span and backend. A pending transfer
/// must always be drained through that same backend/context; the owner retains
/// the exact submitted stream, including the valid default-stream value zero.
pub trait ReadbackIo {
    fn copy(&mut self, dst: &mut [u8], stream: u64) -> Result<()>;
    fn drain(&mut self, stream: u64) -> Result<()>;
}

/// Owns only host readback storage, not its device source or backend. Those
/// remain model-owned until completion. No slice is published on a copy or
/// completion error, and pending storage cannot be resized, reused, or freed.
pub struct OwnedReadback {
    max_bytes: usize,
    buffer: Vec<u8>,
    pending_stream: Option<u64>,
}

impl OwnedReadback {
    /// Validate the explicit BF16 byte bound without allocating its maximum.
    pub fn new(max_bytes: usize) -> Result<Self> {
        ensure!(
            max_bytes > 0 && max_bytes % 2 == 0 && max_bytes <= isize::MAX as usize,
            "owned readback capacity must be nonzero, BF16-aligned, and <= isize::MAX"
        );
        Ok(Self {
            max_bytes,
            buffer: Vec::new(),
            pending_stream: None,
        })
    }

    pub fn pending(&self) -> bool {
        self.pending_stream.is_some()
    }

    /// Complete one exact byte span. Set pending before entering foreign I/O:
    /// a caught panic in either copy or drain must still block later reuse.
    pub fn read(&mut self, bytes: usize, stream: u64, io: &mut dyn ReadbackIo) -> Result<&[u8]> {
        ensure!(
            !self.pending(),
            "owned readback is pending; explicit drain required"
        );
        ensure!(
            bytes > 0 && bytes % 2 == 0 && bytes <= self.max_bytes,
            "owned readback size must be nonzero, BF16-aligned, and within capacity"
        );
        if bytes > self.buffer.len() {
            self.buffer
                .try_reserve_exact(bytes - self.buffer.len())
                .context("owned readback host allocation")?;
        }
        self.buffer.resize(bytes, 0);
        self.pending_stream = Some(stream);
        let copied = io.copy(&mut self.buffer, stream);
        // Even a failed copy may have enqueued work before reporting an error.
        let completed = self.drain(io);
        match (copied, completed) {
            (Ok(()), Ok(())) => Ok(&self.buffer),
            (Err(copy), Ok(())) => Err(copy),
            (Ok(()), Err(drain)) => Err(drain),
            (Err(copy), Err(drain)) => {
                bail!("{copy:#}; owned readback completion also failed: {drain:#}")
            }
        }
    }

    /// Recover only by proving completion on the recorded stream. An error or
    /// panic leaves the stream and allocation intact; idle drain performs no I/O.
    pub fn drain(&mut self, io: &mut dyn ReadbackIo) -> Result<()> {
        if let Some(stream) = self.pending_stream {
            io.drain(stream)?;
            self.pending_stream = None;
        }
        Ok(())
    }
}

impl Drop for OwnedReadback {
    fn drop(&mut self) {
        if self.pending() {
            // Completion is unknown: leak this bounded host allocation rather
            // than release memory that an asynchronous D2H may still reference.
            // No driver I/O, optimistic fence, or source-device free in Drop.
            std::mem::forget(std::mem::take(&mut self.buffer));
        }
    }
}
