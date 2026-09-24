// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_BOUNDED_HOST_COPY=1`: host<->device copies through a pinned staging
//! buffer, with every wait bounded by a wall-clock deadline.
//!
//! The default `copy_h2d` / `copy_d2h` pair is `cuStreamSynchronize` plus the
//! synchronous pageable `cuMemcpyHtoD_v2` / `cuMemcpyDtoH_v2`, neither of which
//! can be interrupted. Spec-mode loads of Qwen3.8 (after "DFlash per-layer
//! SWA", in the drafter NVFP4 quantization) and of Flash-Next (after
//! `dense_keep_f32: promoting ...`) stalled forever on ~7% of attempts, each
//! time right before one of these small copies. Here the stream is drained by
//! polling an event, the copy runs async on that stream from/to page-locked
//! staging, and the completion is polled too, so a stall becomes an error that
//! names the phase instead of a hang.

use std::ffi::c_void;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};

use super::{
    cuEventCreate, cuEventDestroy_v2, cuEventQuery, cuEventRecord, cuMemAllocHost_v2,
    cuMemcpyDtoHAsync_v2, cuMemcpyHtoDAsync_v2,
};

const STAGING_BYTES: usize = 16 * 1024 * 1024;
const TIMEOUT: Duration = Duration::from_secs(60);

pub(super) fn enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_BOUNDED_HOST_COPY").ok().as_deref() == Some("1"))
}

thread_local! {
    /// Per-thread page-locked staging, allocated on first use and kept for the
    /// thread's life (no free: a load thread is long-lived and this is 16 MiB).
    static STAGING: std::cell::Cell<*mut u8> = const { std::cell::Cell::new(std::ptr::null_mut()) };
}

fn staging() -> Result<*mut u8> {
    STAGING.with(|cell| {
        if cell.get().is_null() {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let status = unsafe { cuMemAllocHost_v2(&mut ptr, STAGING_BYTES) };
            if status != 0 || ptr.is_null() {
                bail!("cuMemAllocHost_v2({STAGING_BYTES}) for bounded copy failed: status {status}");
            }
            cell.set(ptr.cast());
        }
        Ok(cell.get())
    })
}

/// Wait for all work queued on `stream` so far, polling against a deadline.
pub(super) fn drain(stream: u64, what: &str) -> Result<()> {
    let mut event: u64 = 0;
    let status = unsafe { cuEventCreate(&mut event, 0x02) };
    if status != 0 {
        bail!("{what}: cuEventCreate failed: status {status}");
    }
    let result = (|| {
        let status = unsafe { cuEventRecord(event, stream) };
        if status != 0 {
            bail!("{what}: cuEventRecord failed: status {status}");
        }
        let t0 = Instant::now();
        let mut spins = 0u32;
        loop {
            match unsafe { cuEventQuery(event) } {
                0 => return Ok(()),
                600 => {}
                other => bail!("{what}: cuEventQuery failed: status {other}"),
            }
            if t0.elapsed() > TIMEOUT {
                bail!(
                    "{what}: stream {stream:#x} did not complete within {}s (bounded host copy)",
                    TIMEOUT.as_secs()
                );
            }
            spins += 1;
            if spins < 2000 {
                std::hint::spin_loop();
            } else {
                std::thread::sleep(Duration::from_micros(50));
            }
        }
    })();
    unsafe { cuEventDestroy_v2(event) };
    result
}

pub(super) fn d2h(src: u64, dst: &mut [u8], stream: u64) -> Result<()> {
    drain(stream, "bounded D2H: producer drain")?;
    let stage = staging()?;
    let mut off = 0usize;
    while off < dst.len() {
        let len = (dst.len() - off).min(STAGING_BYTES);
        let status = unsafe { cuMemcpyDtoHAsync_v2(stage.cast(), src + off as u64, len, stream) };
        if status != 0 {
            bail!("bounded D2H: cuMemcpyDtoHAsync_v2 failed: status {status}");
        }
        drain(stream, "bounded D2H: copy")?;
        unsafe { std::ptr::copy_nonoverlapping(stage, dst.as_mut_ptr().add(off), len) };
        off += len;
    }
    Ok(())
}

pub(super) fn h2d(src: &[u8], dst: u64, stream: u64) -> Result<()> {
    drain(stream, "bounded H2D: prior-work drain")?;
    let stage = staging()?;
    let mut off = 0usize;
    while off < src.len() {
        let len = (src.len() - off).min(STAGING_BYTES);
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr().add(off), stage, len) };
        let status = unsafe { cuMemcpyHtoDAsync_v2(dst + off as u64, stage.cast(), len, stream) };
        if status != 0 {
            bail!("bounded H2D: cuMemcpyHtoDAsync_v2 failed: status {status}");
        }
        // The staging buffer is reused by the next chunk, so the copy must land.
        drain(stream, "bounded H2D: copy")?;
        off += len;
    }
    Ok(())
}
