// SPDX-License-Identifier: AGPL-3.0-only

//! Real CUDA GPU backend using AtlasRegistry.
//!
//! SBIO IORouter: all CUDA operations flow through `GpuBackend`.
//! Uses `AtlasRegistry` for kernel loading/launching and raw CUDA
//! driver API for memory management.

use std::ffi::c_void;

use anyhow::{Result, bail};
use atlas_core::registry::AtlasRegistry;

mod gpu_impl;

// ── Raw CUDA driver API for memory operations ──

unsafe extern "C" {
    pub(super) fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
    pub(super) fn cuMemFree_v2(dptr: u64) -> i32;
    pub(super) fn cuMemcpyHtoD_v2(dst: u64, src: *const c_void, bytes: usize) -> i32;
    pub(super) fn cuMemcpyDtoH_v2(dst: *mut c_void, src: u64, bytes: usize) -> i32;
    pub(super) fn cuMemcpyHtoDAsync_v2(
        dst: u64,
        src: *const c_void,
        bytes: usize,
        stream: u64,
    ) -> i32;
    pub(super) fn cuMemcpyDtoDAsync_v2(dst: u64, src: u64, bytes: usize, stream: u64) -> i32;
    pub(super) fn cuStreamSynchronize(stream: u64) -> i32;
    pub(super) fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> i32;
    pub(super) fn cuMemsetD8Async(dst: u64, value: u8, n: usize, stream: u64) -> i32;
    // CUDA graph capture/replay
    pub(super) fn cuStreamBeginCapture(hStream: u64, mode: u32) -> i32;
    pub(super) fn cuStreamEndCapture(hStream: u64, phGraph: *mut u64) -> i32;
    pub(super) fn cuGraphInstantiateWithFlags(
        phGraphExec: *mut u64,
        hGraph: u64,
        flags: u64,
    ) -> i32;
    pub(super) fn cuGraphLaunch(hGraphExec: u64, hStream: u64) -> i32;
    pub(super) fn cuGraphExecDestroy(hGraphExec: u64) -> i32;
    pub(super) fn cuGraphDestroy(hGraph: u64) -> i32;
    fn cuCtxGetCurrent(pctx: *mut u64) -> i32;
    pub(super) fn cuCtxSetCurrent(ctx: u64) -> i32;
    // Device properties. `CUdevice` is an opaque int handle, not a pointer.
    pub(super) fn cuCtxGetDevice(device: *mut i32) -> i32;
    pub(super) fn cuDeviceGetAttribute(pi: *mut i32, attrib: u32, dev: i32) -> i32;
    pub(super) fn cuStreamCreate(phStream: *mut u64, flags: u32) -> i32;
    // Page-locked host memory for efficient async transfers
    pub(super) fn cuMemAllocHost_v2(pp: *mut *mut c_void, bytesize: usize) -> i32;
    pub(super) fn cuMemFreeHost(p: *mut c_void) -> i32;
    // Managed (unified) memory — allows over-subscription with Linux swap paging
    pub(super) fn cuMemAllocManaged(dptr: *mut u64, bytesize: usize, flags: u32) -> i32;
    // CUDA events for inter-stream synchronization
    pub(super) fn cuEventCreate(phEvent: *mut u64, flags: u32) -> i32;
    pub(super) fn cuEventRecord(hEvent: u64, hStream: u64) -> i32;
    pub(super) fn cuStreamWaitEvent(hStream: u64, hEvent: u64, flags: u32) -> i32;
    pub(super) fn cuEventDestroy_v2(hEvent: u64) -> i32;
}

/// Production GPU backend wrapping AtlasRegistry + raw CUDA driver API.
///
/// Initialized once via [`AtlasCudaBackend::new`], which loads all PTX
/// modules from `atlas-kernels` into the global AtlasRegistry singleton.
pub struct AtlasCudaBackend {
    /// Default CUDA stream handle (from AtlasRegistry).
    default_stream: u64,
    /// CUDA context handle for cross-thread binding.
    cuda_ctx: u64,
}

impl AtlasCudaBackend {
    /// Initialize the CUDA backend on the given GPU ordinal.
    ///
    /// Loads the provided PTX modules into AtlasRegistry.
    /// Use `atlas_kernels::ptx_for_model()` or `ptx_modules()` to
    /// obtain the correct module set for the target model.
    /// Subsequent calls reuse the cached singleton.
    pub fn new(ordinal: usize, ptx_modules: &[(&'static str, &str)]) -> Result<Self> {
        let registry = AtlasRegistry::get_or_init(ordinal, ptx_modules)
            .map_err(|e| anyhow::anyhow!("AtlasRegistry init failed: {e}"))?;
        let default_stream = registry.raw_stream();

        // Capture current CUDA context for cross-thread binding.
        let mut cuda_ctx: u64 = 0;
        let status = unsafe { cuCtxGetCurrent(&mut cuda_ctx) };
        if status != 0 || cuda_ctx == 0 {
            bail!("cuCtxGetCurrent failed: status {status}, ctx {cuda_ctx:#x}");
        }

        tracing::info!(
            "AtlasCudaBackend initialized on GPU {ordinal} with {} PTX modules",
            ptx_modules.len()
        );

        Ok(Self {
            default_stream,
            cuda_ctx,
        })
    }

    /// `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`.
    const ATTR_MULTIPROCESSOR_COUNT: u32 = 16;

    /// Device SM count, queried once and cached for the process.
    ///
    /// The query needs only the current context's device handle — no
    /// allocation, no launch — but it still goes through the driver, so it is
    /// resolved on first use and never repeated. A failure is logged once and
    /// cached as `None` so a broken driver cannot turn into a per-launch log
    /// storm.
    pub(super) fn cached_sm_count() -> Option<u32> {
        static CACHE: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
        *CACHE.get_or_init(|| {
            let mut device: i32 = 0;
            let status = unsafe { cuCtxGetDevice(&mut device) };
            if status != 0 {
                tracing::warn!(
                    "cuCtxGetDevice failed (status {status}); occupancy sizing falls back to \
                     its compiled-in SM literal"
                );
                return None;
            }
            let mut count: i32 = 0;
            let status = unsafe {
                cuDeviceGetAttribute(&mut count, Self::ATTR_MULTIPROCESSOR_COUNT, device)
            };
            if status != 0 || count <= 0 {
                tracing::warn!(
                    "cuDeviceGetAttribute(MULTIPROCESSOR_COUNT) failed (status {status}, \
                     count {count}); occupancy sizing falls back to its compiled-in SM literal"
                );
                return None;
            }
            tracing::info!("GPU multiprocessor count: {count}");
            Some(count as u32)
        })
    }
}

// ── OOM Watchdog ────────────────────────────────────────────────────
//
// Background task that polls GPU free memory every `interval` and calls
// `std::process::exit(1)` if it drops below `threshold_bytes`.
// On GB10 unified memory, GPU OOM = system OOM = kernel freeze, so
// killing the process early prevents unrecoverable system hangs.

/// Query GPU free memory without requiring a GpuBackend reference.
/// Safe to call from any thread that shares the CUDA context.
///
/// On unified memory systems (GB10), `cuMemGetInfo` reports Linux's "free" memory
/// which excludes reclaimable buff/cache. This under-reports available memory by
/// 30-50%. We take the max of CUDA's report and `/proc/meminfo` MemAvailable
/// to get the true available memory.
pub fn cuda_free_memory_bytes() -> Option<usize> {
    let mut free: usize = 0;
    let mut total: usize = 0;
    let status = unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
    if status != 0 {
        return None;
    }

    // On unified memory, also check MemAvailable from /proc/meminfo.
    // This includes reclaimable buff/cache that CUDA doesn't account for.
    if let Some(mem_available) = system_available_memory_bytes() {
        free = free.max(mem_available);
    }
    Some(free)
}

/// Read MemAvailable from /proc/meminfo (Linux only).
/// Returns None on non-Linux or if parsing fails.
fn system_available_memory_bytes() -> Option<usize> {
    let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in contents.lines() {
        if line.starts_with("MemAvailable:") {
            let kb: usize = line.split_whitespace().nth(1)?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

/// Start a background OOM watchdog that polls GPU memory every `interval`.
/// If free memory drops below `threshold_mb` MB, the process exits immediately.
///
/// Returns a `tokio::task::JoinHandle` — drop it to stop the watchdog (on shutdown).
pub fn spawn_oom_watchdog(
    threshold_mb: usize,
    interval: std::time::Duration,
) -> tokio::task::JoinHandle<()> {
    let threshold_bytes = threshold_mb * 1024 * 1024;
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(interval);
        // Track consecutive low-memory readings to avoid false positives
        // during transient allocation spikes.
        let mut consecutive_low = 0u32;
        loop {
            tick.tick().await;
            if let Some(free) = cuda_free_memory_bytes() {
                if free < threshold_bytes {
                    consecutive_low += 1;
                    let free_mb = free / (1024 * 1024);
                    tracing::error!(
                        "OOM watchdog: GPU free memory critically low: {} MB (threshold: {} MB) [{}/3]",
                        free_mb,
                        threshold_mb,
                        consecutive_low,
                    );
                    if consecutive_low >= 3 {
                        tracing::error!(
                            "OOM watchdog: 3 consecutive readings below threshold. \
                             Terminating to prevent system freeze."
                        );
                        // Flush logs before exit
                        std::process::exit(1);
                    }
                } else {
                    consecutive_low = 0;
                }
            }
        }
    })
}

#[cfg(test)]
mod tests;
