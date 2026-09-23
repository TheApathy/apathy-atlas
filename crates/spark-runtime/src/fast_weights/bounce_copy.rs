// SPDX-License-Identifier: AGPL-3.0-only

//! Bounded-timeout pinned bounce-buffer H2D copy for weight loading.
//!
//! `GpuBackend::copy_h2d` on ordinary pageable memory is a **synchronous**
//! driver call (`cuStreamSynchronize` + `cuMemcpyHtoD_v2`) with no timeout.
//! Observed on the integrated (all-models) binary: Flash-Next loads stalled
//! on ~13% of attempts (3/23 vs 0/12 on the pre-integration binary), always
//! inside that blocking pair, copying an ordinary few-MB tensor with 25-35
//! GB of GPU memory free — a driver/hardware race, not a resource shortage,
//! and one `cuStreamSynchronize` cannot be interrupted once entered.
//!
//! [`PinnedBounceCopier`] avoids the pageable path entirely: tensor bytes
//! are staged into a small pool of reusable page-locked buffers and moved
//! with the async copy API, then waited on by *polling* `poll_event` against
//! a wall-clock deadline rather than calling a blocking driver primitive. A
//! stall becomes a bounded, retried failure instead of a hang.

use crate::gpu::{DevicePtr, GpuBackend, PinnedHostBuffer};
use anyhow::{Result, bail};
use std::time::{Duration, Instant};

/// Each pending chunk waits at most this long for its H2D copy to land
/// before the caller gives up on this attempt and retries.
const EVENT_TIMEOUT: Duration = Duration::from_secs(30);
const POLL_INTERVAL: Duration = Duration::from_millis(2);
/// Reusable pinned staging buffers. More than one lets consecutive chunks
/// overlap (stage chunk N+1 while chunk N's async copy is still in flight);
/// bounded so a run of huge tensors cannot balloon pinned-memory use.
const SLOTS: usize = 3;
const CHUNK_BYTES: usize = 64 * 1024 * 1024;
/// How many times a single tensor's copy is retried end-to-end after a
/// bounded timeout before the load fails outright. Each retry re-runs the
/// whole tensor from the still-valid `src` slice on a fresh event, so a
/// one-off driver stall does not cost the load; a persistent one does not
/// hang it either.
const MAX_RETRIES: u32 = 3;

/// Reusable pool of pinned staging buffers plus their in-flight completion
/// events. Constructed once per shard load and reused across every tensor
/// in that shard so the (comparatively expensive) `cuMemAllocHost_v2` calls
/// happen `SLOTS` times total, not once per tensor.
pub(super) struct PinnedBounceCopier {
    slots: Vec<(PinnedHostBuffer, Option<u64>)>,
    stream: u64,
}

impl PinnedBounceCopier {
    pub(super) fn new(gpu: &dyn GpuBackend) -> Result<Self> {
        let mut slots = Vec::with_capacity(SLOTS);
        for _ in 0..SLOTS {
            slots.push((gpu.alloc_host_pinned(CHUNK_BYTES)?, None));
        }
        Ok(Self {
            slots,
            stream: gpu.default_stream(),
        })
    }

    /// Copy `src` to `dst` on the GPU, chunked through the pinned pool, with
    /// a bounded number of whole-tensor retries on a stalled chunk.
    pub(super) fn copy(&mut self, gpu: &dyn GpuBackend, src: &[u8], dst: DevicePtr) -> Result<()> {
        let mut attempt = 0;
        loop {
            match self.copy_once(gpu, src, dst) {
                Ok(()) => return Ok(()),
                Err(e) if attempt < MAX_RETRIES => {
                    attempt += 1;
                    // NOTE: retries resubmit on the SAME stream. If the whole
                    // stream/context is wedged (not just one copy), every
                    // retry will also time out — this trades an unbounded
                    // hang for a bounded ~MAX_RETRIES*EVENT_TIMEOUT failure,
                    // which is the goal, but it is not a true independent
                    // retry when the underlying stall is stream-wide.
                    tracing::warn!(
                        "pinned H2D copy stalled ({e:#}); retry {attempt}/{MAX_RETRIES} \
                         for a {}-byte tensor",
                        src.len()
                    );
                }
                Err(e) => {
                    return Err(e.context(format!(
                        "pinned H2D copy failed after {MAX_RETRIES} retries \
                         ({} bytes to device offset {:#x})",
                        src.len(),
                        dst.0
                    )));
                }
            }
        }
    }

    fn copy_once(&mut self, gpu: &dyn GpuBackend, src: &[u8], dst: DevicePtr) -> Result<()> {
        let mut offset = 0usize;
        let mut slot_idx = 0usize;
        while offset < src.len() {
            let len = (src.len() - offset).min(CHUNK_BYTES);
            let chunk = &src[offset..offset + len];

            // This slot's previous copy (if any, from an earlier tensor or an
            // earlier chunk of this one) must be complete before we
            // overwrite its pinned bytes — the DMA reads them asynchronously.
            wait_event_bounded(gpu, self.slots[slot_idx].1.take())?;

            let pinned_len = {
                let (buf, _) = &mut self.slots[slot_idx];
                buf.as_mut_slice()[..len].copy_from_slice(chunk);
                len
            };
            let pinned = self.slots[slot_idx].0.pinned_slice(pinned_len)?;
            let chunk_dst = DevicePtr(dst.0 + offset as u64);
            unsafe {
                gpu.copy_h2d_pinned_async(pinned, chunk_dst, self.stream)?;
            }
            let event = gpu.create_event()?;
            gpu.record_event(event, self.stream)?;
            self.slots[slot_idx].1 = Some(event);

            offset += len;
            slot_idx = (slot_idx + 1) % self.slots.len();
        }
        Ok(())
    }

    /// Wait out every slot's in-flight copy. Call once at the end of a shard
    /// (or on error unwind) so no event/pinned-buffer write is still
    /// pending when this copier is dropped.
    pub(super) fn drain(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        for i in 0..self.slots.len() {
            let event = self.slots[i].1.take();
            wait_event_bounded(gpu, event)?;
        }
        Ok(())
    }
}

/// Poll `event` (if any) to completion, bounded by [`EVENT_TIMEOUT`].
///
/// This is the whole point of the module: never call a blocking driver
/// primitive on this path. A timeout returns an error instead of hanging;
/// the event handle is deliberately leaked in that case rather than
/// destroyed, because destroying a CUDA event while its recorded work may
/// still be executing under a driver-side stall is undefined behaviour —
/// leaking one handle on a rare timeout is a better trade than a
/// use-after-free on a driver that is already misbehaving.
fn wait_event_bounded(gpu: &dyn GpuBackend, event: Option<u64>) -> Result<()> {
    let Some(event) = event else { return Ok(()) };
    let deadline = Instant::now() + EVENT_TIMEOUT;
    loop {
        if gpu.poll_event(event)? {
            gpu.destroy_event(event)?;
            return Ok(());
        }
        if Instant::now() >= deadline {
            bail!(
                "H2D copy event did not complete within {EVENT_TIMEOUT:?} \
                 (event handle {event:#x}, leaked — driver stall, not a spin)"
            );
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}
