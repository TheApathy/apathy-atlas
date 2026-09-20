// SPDX-License-Identifier: AGPL-3.0-only
//! Completed committed-prefix authority for the drafter's existing KV buffers.
//!
//! This owner performs no GPU allocation and knows no architecture defaults.
//! The runtime supplies admitted layer count/capacity and retains every device
//! allocation, source and backend until pending work is explicitly drained.

use anyhow::{Context, Result, ensure};

/// Environment access belongs to the runtime; malformed values never opt out.
pub fn parse_kv_prefix_flag(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => anyhow::bail!("ATLAS_GLM53_DFLASH2_KV_PREFIX must be absent, 0, or 1"),
    }
}

/// Immutable committed source interval. Provisional noise is deliberately not
/// represented here and cannot advance the completed cursor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvTail {
    source_row: u32,
    new_rows: u32,
    committed_end: u32,
}

impl KvTail {
    pub fn source_row(self) -> u32 {
        self.source_row
    }
    pub fn retained_rows(self) -> u32 {
        self.source_row
    }
    pub fn new_rows(self) -> u32 {
        self.new_rows
    }
    pub fn committed_end(self) -> u32 {
        self.committed_end
    }

    fn new(completed: u32, context: u32) -> Result<Self> {
        ensure!(
            context > 0 && completed <= context,
            "KV prefix moved backwards or is empty"
        );
        // Existing attention requires a nonempty new tail. A repeated proposal
        // reprojects the last committed source row, not normalized cache data.
        let source_row = if completed == context {
            context.checked_sub(1).context("KV repeat-tail underflow")?
        } else {
            completed
        };
        let new_rows = context
            .checked_sub(source_row)
            .context("KV tail extent underflow")?;
        ensure!(
            new_rows > 0 && source_row.checked_add(new_rows) == Some(context),
            "KV tail extent overflow"
        );
        Ok(Self {
            source_row,
            new_rows,
            committed_end: context,
        })
    }
}

/// Bound to the same runtime, device resources and backend for the complete
/// update. An enqueue success proves submission only, never completion. A
/// failed or panicking callback may already have submitted asynchronous work.
pub trait KvPrefixIo {
    /// Enqueue the actual layer's projection, normalization, RoPE and cache
    /// writes using this source interval; do not retain borrowed host arguments.
    fn enqueue_layer(&mut self, layer: usize, tail: KvTail, stream: u64) -> Result<()>;
    fn synchronize(&mut self, stream: u64) -> Result<()>;
}

struct Pending {
    stream: u64,
    tails: Vec<KvTail>,
    submitted: usize,
}

/// All completed cursors publish together after the final same-stream fence.
/// Dropping this metadata is not a drain and grants no permission to release
/// runtime-owned GPU buffers; the runtime must enforce that outer boundary.
#[must_use = "pending KV work must be drained before its runtime resources are released"]
pub struct KvPrefix {
    max_context: u32,
    completed: Vec<u32>,
    pending: Option<Pending>,
    poisoned: bool,
}

impl KvPrefix {
    pub fn new(layer_count: usize, max_context: u32) -> Result<Self> {
        ensure!(
            layer_count > 0 && max_context > 0,
            "KV prefix dimensions must be nonzero"
        );
        let bytes = layer_count
            .checked_mul(std::mem::size_of::<u32>())
            .context("KV prefix layer allocation overflow")?;
        ensure!(
            bytes <= isize::MAX as usize,
            "KV prefix layer allocation exceeds isize"
        );
        let mut completed = Vec::new();
        completed
            .try_reserve_exact(layer_count)
            .context("KV prefix cursor allocation")?;
        completed.resize(layer_count, 0);
        Ok(Self {
            max_context,
            completed,
            pending: None,
            poisoned: false,
        })
    }

    /// Diagnostic historical cursor only: pending/poisoned state must not be
    /// bypassed by using this value as independent cache-reuse authority.
    pub fn completed_rows(&self, layer: usize) -> Result<u32> {
        self.completed
            .get(layer)
            .copied()
            .context("KV prefix layer is outside the admitted count")
    }

    pub fn pending(&self) -> bool {
        self.pending.is_some()
    }
    pub fn has_completed_rows(&self) -> bool {
        self.completed.iter().any(|rows| *rows != 0)
    }
    pub fn pending_stream(&self) -> Option<u64> {
        self.pending.as_ref().map(|p| p.stream)
    }
    pub fn poisoned(&self) -> bool {
        self.poisoned
    }

    /// The runtime must derive both context and target_position from its actual
    /// committed owners. Matching caller-supplied integers alone prove no GPU
    /// state identity. Capture mode cannot publish this eager completion state.
    pub fn begin(
        &mut self,
        context: u32,
        target_position: u32,
        stream: u64,
        capturing: bool,
    ) -> Result<()> {
        ensure!(
            !self.pending() && !self.poisoned,
            "KV prefix pending or poisoned; reset required"
        );
        ensure!(
            !capturing,
            "committed KV-prefix reuse requires eager execution"
        );
        ensure!(
            context > 0 && context <= self.max_context && context == target_position,
            "KV prefix target/context identity or capacity mismatch"
        );
        let mut tails = Vec::new();
        tails
            .try_reserve_exact(self.completed.len())
            .context("KV prefix tail-plan allocation")?;
        for &completed in &self.completed {
            tails.push(KvTail::new(completed, context)?);
        }
        // All preflight and allocation completed before any in-flight state.
        self.pending = Some(Pending {
            stream,
            tails,
            submitted: 0,
        });
        Ok(())
    }

    pub fn enqueue_layer(&mut self, layer: usize, io: &mut dyn KvPrefixIo) -> Result<()> {
        ensure!(
            !self.poisoned,
            "KV prefix enqueue failed or panicked; reset required"
        );
        let pending = self
            .pending
            .as_mut()
            .context("KV prefix enqueue without an update")?;
        ensure!(
            layer == pending.submitted && layer < pending.tails.len(),
            "KV prefix layer submission must be complete and ordered exactly once"
        );
        let tail = pending.tails[layer];
        // Set the failure latch BEFORE foreign I/O, not merely on its Err arm:
        // caught panics must not permit a retry against uncertain cache writes.
        self.poisoned = true;
        io.enqueue_layer(layer, tail, pending.stream)?;
        pending.submitted += 1;
        self.poisoned = false;
        Ok(())
    }

    pub fn finish(&mut self, io: &mut dyn KvPrefixIo) -> Result<()> {
        ensure!(!self.poisoned, "KV prefix update failed; reset required");
        let pending = self
            .pending
            .as_ref()
            .context("KV prefix finish without an update")?;
        ensure!(
            pending.submitted == self.completed.len()
                && pending.tails.len() == self.completed.len(),
            "KV prefix finish requires every layer's successful enqueue receipt"
        );
        self.poisoned = true;
        io.synchronize(pending.stream)?;
        // No allocation, callback or fallible operation between completed fence
        // and atomic publication under this owner's exclusive mutable borrow.
        for (completed, tail) in self.completed.iter_mut().zip(&pending.tails) {
            *completed = tail.committed_end;
        }
        self.pending = None;
        self.poisoned = false;
        Ok(())
    }

    /// Drain an abandoned proposal without publishing any candidate cursor.
    /// Even a successful drain leaves it poisoned until reset invalidates all
    /// layers: callbacks may have rewritten the last previously valid row.
    pub fn abort(&mut self, io: &mut dyn KvPrefixIo) -> Result<()> {
        ensure!(self.pending(), "KV prefix abort without a pending update");
        self.drain_pending(io)
    }

    /// Retry completion on the recorded original stream, then invalidate all
    /// cursors. An error/panic leaves both the pending owner and old diagnostic
    /// cursor values intact. No cache zeroing is implied by this metadata reset.
    pub fn reset(&mut self, io: &mut dyn KvPrefixIo) -> Result<()> {
        self.drain_pending(io)?;
        self.completed.fill(0);
        self.poisoned = false;
        Ok(())
    }

    fn drain_pending(&mut self, io: &mut dyn KvPrefixIo) -> Result<()> {
        if let Some(stream) = self.pending_stream() {
            self.poisoned = true;
            io.synchronize(stream)?;
            self.pending = None;
        }
        Ok(())
    }
}
