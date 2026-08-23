// SPDX-License-Identifier: AGPL-3.0-only

//! `impl GpuBackend for AtlasCudaBackend` — production CUDA backend trait body.
//!
//! ## Safety contract for the `unsafe { cu*(...) }` calls below
//!
//! Every unsafe block in this file wraps a single CUDA Driver API call.
//! The invariants the driver requires are uniform:
//!
//! - **Context bound**: a CUDA primary context for the device is current
//!   on the calling thread. `AtlasCudaBackend::new` binds it once via
//!   `cuCtxSetCurrent`, and we never run on a thread that hasn't been
//!   bound.
//! - **Pointer provenance**: every `DevicePtr` came from a prior
//!   successful `cuMemAlloc_v2` / `cuMemAllocHost_v2` /
//!   `cuMemAllocManaged` and has not yet been freed. `DevicePtr(0)` is
//!   treated as "not allocated" by callers.
//! - **Sizes in bytes**: every `bytes: usize` argument is the exact
//!   byte count of the allocation (callers compute it from typed
//!   sizes); the driver does no bounds-checking.
//! - **Stream / event lifetimes**: handles are owned by `Self` and
//!   freed in `Drop` after `cuStreamSynchronize`, so they outlive every
//!   in-flight launch that captured them.
//! - **`extern "C"` ABI**: matches the cudarc-generated bindings used
//!   in `super::*` imports; see `cudarc` for the full ABI surface.
//!
//! Per-site `// SAFETY:` comments are omitted because the contract is
//! identical for every call. Anything that *deviates* from this
//! contract gets a per-site `// SAFETY:` comment explaining the
//! exception.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::OnceLock;

use anyhow::{Result, bail};
use atlas_core::registry::{AtlasRegistry, RawCudaFunc, cuda_error_text};
use cudarc::driver::LaunchConfig;

use super::{
    AtlasCudaBackend, cuCtxSetCurrent, cuEventCreate, cuEventDestroy_v2, cuEventRecord,
    cuGraphDestroy, cuGraphExecDestroy, cuGraphInstantiateWithFlags, cuGraphLaunch, cuMemAlloc_v2,
    cuMemAllocHost_v2, cuMemAllocManaged, cuMemFree_v2, cuMemFreeHost, cuMemGetInfo_v2,
    cuMemcpyDtoDAsync_v2, cuMemcpyDtoH_v2, cuMemcpyHtoD_v2, cuMemcpyHtoDAsync_v2, cuMemsetD8Async,
    cuStreamBeginCapture, cuStreamCreate, cuStreamEndCapture, cuStreamSynchronize,
    cuStreamWaitEvent,
};
use crate::gpu::{
    DevicePtr, GpuBackend, GraphHandle, HostToDeviceCopy, KernelHandle, PinnedHostBuffer,
    PinnedHostSlice, PinnedHostStorage,
};

struct CudaPinnedHostStorage {
    ptr: NonNull<u8>,
    bytes: usize,
}

unsafe impl Send for CudaPinnedHostStorage {}
unsafe impl Sync for CudaPinnedHostStorage {}

impl PinnedHostStorage for CudaPinnedHostStorage {
    fn ptr(&self) -> NonNull<u8> {
        self.ptr
    }

    fn len(&self) -> usize {
        self.bytes
    }
}

impl Drop for CudaPinnedHostStorage {
    fn drop(&mut self) {
        let status = unsafe { cuMemFreeHost(self.ptr.as_ptr().cast()) };
        if status != 0 {
            tracing::warn!(
                "cuMemFreeHost failed while dropping pinned host buffer: status {status}"
            );
        }
    }
}

impl GpuBackend for AtlasCudaBackend {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        let mut dptr: u64 = 0;
        let status = unsafe { cuMemAlloc_v2(&mut dptr, bytes) };
        if status != 0 {
            let mut free: usize = 0;
            let mut total: usize = 0;
            unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
            bail!(
                "cuMemAlloc_v2 failed: status {status}, requested {bytes} bytes \
                 (device reports {:.1} MB free / {:.1} GB total)",
                free as f64 / (1024.0 * 1024.0),
                total as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        }
        Ok(DevicePtr(dptr))
    }

    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        let mut dptr: u64 = 0;
        const CU_MEM_ATTACH_GLOBAL: u32 = 0x1;
        let status = unsafe { cuMemAllocManaged(&mut dptr, bytes, CU_MEM_ATTACH_GLOBAL) };
        if status != 0 {
            bail!(
                "cuMemAllocManaged failed: status {status}, requested {bytes} bytes. \
                 Check system swap space: swapon --show"
            );
        }
        Ok(DevicePtr(dptr))
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        let status = unsafe { cuMemFree_v2(ptr.0) };
        if status != 0 {
            bail!("cuMemFree_v2 failed: status {status}, ptr {ptr}");
        }
        Ok(())
    }

    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        // This API accepts any Rust slice, including ordinary pageable memory.
        // CUDA's async host-copy APIs require page-locked storage; use the
        // synchronous driver call for this intentionally blocking interface.
        // Atlas uses a non-blocking CUDA stream, so drain it first to preserve
        // the old method's ordering against earlier work on that stream.
        let sync = unsafe { cuStreamSynchronize(self.default_stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize before H2D failed: {}",
                cuda_error_text(sync)
            );
        }
        let status = unsafe { cuMemcpyHtoD_v2(dst.0, src.as_ptr() as *const c_void, src.len()) };
        if status != 0 {
            bail!("cuMemcpyHtoD_v2 failed: status {status}");
        }
        Ok(())
    }

    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        // `dst` is not required to be page-locked. The synchronous API is the
        // CUDA-supported path for arbitrary pageable host buffers and returns
        // only after the bytes are safe for the caller to read.
        // Synchronize Atlas's non-blocking stream before the streamless driver
        // copy so prior kernels cannot race the host read.
        let sync = unsafe { cuStreamSynchronize(self.default_stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize before D2H failed: {}",
                cuda_error_text(sync)
            );
        }
        let status = unsafe { cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut c_void, src.0, dst.len()) };
        if status != 0 {
            bail!("cuMemcpyDtoH_v2 failed: status {status}");
        }
        Ok(())
    }

    fn copy_d2h_on_stream(&self, src: DevicePtr, dst: &mut [u8], stream: u64) -> Result<()> {
        // This is a blocking API over an arbitrary host slice. Drain the
        // producer stream for ordering, then use the pageable-safe synchronous
        // copy. The coalesced pair API below uses the same safe ordering.
        let sync = unsafe { cuStreamSynchronize(stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize before D2H on_stream failed: {}",
                cuda_error_text(sync)
            );
        }
        let status = unsafe { cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut c_void, src.0, dst.len()) };
        if status != 0 {
            bail!("cuMemcpyDtoH_v2 (on_stream) failed: status {status}");
        }
        Ok(())
    }

    fn copy_d2h_pair_on_stream(
        &self,
        first_src: DevicePtr,
        first_dst: &mut [u8],
        second_src: DevicePtr,
        second_dst: &mut [u8],
        stream: u64,
    ) -> Result<()> {
        // Both destinations may be ordinary pageable Vec storage. Drain the
        // producer stream once, then use the synchronous driver entry point
        // for each copy. This preserves the hot paths' two-copy/one-sync
        // shape without passing pageable memory to cuMemcpyDtoHAsync_v2.
        let sync = unsafe { cuStreamSynchronize(stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize before D2H pair failed: {}",
                cuda_error_text(sync)
            );
        }
        let first_status = unsafe {
            cuMemcpyDtoH_v2(
                first_dst.as_mut_ptr() as *mut c_void,
                first_src.0,
                first_dst.len(),
            )
        };
        if first_status != 0 {
            bail!("first cuMemcpyDtoH_v2 in pair failed: status {first_status}");
        }
        let second_status = unsafe {
            cuMemcpyDtoH_v2(
                second_dst.as_mut_ptr() as *mut c_void,
                second_src.0,
                second_dst.len(),
            )
        };
        if second_status != 0 {
            bail!("second cuMemcpyDtoH_v2 in pair failed: status {second_status}");
        }
        Ok(())
    }

    fn copy_h2d_group_on_stream(&self, copies: &[HostToDeviceCopy<'_>], stream: u64) -> Result<()> {
        if copies.is_empty() {
            return Ok(());
        }
        let sync = unsafe { cuStreamSynchronize(stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize before H2D group failed: {}",
                cuda_error_text(sync)
            );
        }
        for (index, copy) in copies.iter().copied().enumerate() {
            let src = copy.src();
            let status =
                unsafe { cuMemcpyHtoD_v2(copy.dst().0, src.as_ptr() as *const c_void, src.len()) };
            if status != 0 {
                bail!("cuMemcpyHtoD_v2 group member {index} failed: status {status}");
            }
        }
        Ok(())
    }

    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        let status = unsafe { cuMemcpyDtoDAsync_v2(dst.0, src.0, bytes, self.default_stream) };
        if status != 0 {
            bail!("cuMemcpyDtoDAsync_v2 failed: status {status}");
        }
        // Synchronize to ensure copy completes before kernels on other streams read it.
        let sync = unsafe { cuStreamSynchronize(self.default_stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize after D2D failed: {}",
                cuda_error_text(sync)
            );
        }
        Ok(())
    }

    fn launch(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        let raw_func = RawCudaFunc(func.0 as *mut c_void);
        let cfg = LaunchConfig {
            grid_dim: (grid[0], grid[1], grid[2]),
            block_dim: (block[0], block[1], block[2]),
            shared_mem_bytes: shared_mem,
        };
        let registry = AtlasRegistry::get();
        unsafe {
            registry
                .launch_on_stream(raw_func, cfg, stream, params)
                .map_err(|e| anyhow::anyhow!("Kernel launch failed: {e}"))
        }
    }

    fn synchronize(&self, stream: u64) -> Result<()> {
        let status = unsafe { cuStreamSynchronize(stream) };
        if status != 0 {
            bail!("cuStreamSynchronize failed: {}", cuda_error_text(status));
        }
        Ok(())
    }

    fn default_stream(&self) -> u64 {
        self.default_stream
    }

    fn sm_count(&self) -> Option<u32> {
        super::AtlasCudaBackend::cached_sm_count()
    }

    #[track_caller]
    fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
        // Ephemeral OnceLock — no cross-call caching, but kernel() is only
        // called at model init time. Layers store the returned KernelHandle.
        let cache: OnceLock<RawCudaFunc> = OnceLock::new();
        let registry = AtlasRegistry::get();
        let found = registry.raw_function_cached(&cache, module, func_name);
        // Record BEFORE the `?`: a failed lookup is the only kind worth
        // auditing, and `try_kernel` swallows the error.
        crate::kernel_audit::record(
            module,
            func_name,
            found.is_ok(),
            std::panic::Location::caller(),
        );
        let raw = found.map_err(|e| anyhow::anyhow!("Kernel lookup {module}::{func_name}: {e}"))?;
        Ok(KernelHandle(raw.0 as u64))
    }

    unsafe fn copy_h2d_pinned_async(
        &self,
        src: PinnedHostSlice<'_>,
        dst: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let src = src.as_bytes();
        let status = unsafe {
            cuMemcpyHtoDAsync_v2(dst.0, src.as_ptr() as *const c_void, src.len(), stream)
        };
        if status != 0 {
            bail!("cuMemcpyHtoDAsync_v2 failed: status {status}");
        }
        Ok(())
    }

    fn copy_d2d_async(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        let status = unsafe { cuMemcpyDtoDAsync_v2(dst.0, src.0, bytes, stream) };
        if status != 0 {
            bail!("cuMemcpyDtoDAsync_v2 failed: status {status}");
        }
        Ok(())
    }

    fn begin_capture(&self, stream: u64) -> Result<()> {
        // CU_STREAM_CAPTURE_MODE_RELAXED = 2
        // Relaxed mode allows NCCL's internal streams to operate during
        // graph capture (required for EP all-reduce in CUDA graphs).
        let status = unsafe { cuStreamBeginCapture(stream, 2) };
        if status != 0 {
            bail!("cuStreamBeginCapture failed: status {status}");
        }
        Ok(())
    }

    fn end_capture(&self, stream: u64) -> Result<GraphHandle> {
        let mut graph: u64 = 0;
        let status = unsafe { cuStreamEndCapture(stream, &mut graph) };
        if status != 0 {
            bail!("cuStreamEndCapture failed: status {status}");
        }
        // Instantiate the graph into an executable
        let mut graph_exec: u64 = 0;
        let status = unsafe { cuGraphInstantiateWithFlags(&mut graph_exec, graph, 0) };
        if status != 0 {
            unsafe { cuGraphDestroy(graph) };
            bail!("cuGraphInstantiateWithFlags failed: status {status}");
        }
        // The graph template is no longer needed after instantiation
        unsafe { cuGraphDestroy(graph) };
        Ok(GraphHandle(graph_exec))
    }

    fn launch_graph(&self, graph: GraphHandle, stream: u64) -> Result<()> {
        let status = unsafe { cuGraphLaunch(graph.0, stream) };
        if status != 0 {
            bail!("cuGraphLaunch failed: status {status}");
        }
        Ok(())
    }

    fn destroy_graph(&self, graph: GraphHandle) -> Result<()> {
        if graph.0 != 0 {
            let status = unsafe { cuGraphExecDestroy(graph.0) };
            if status != 0 {
                bail!("cuGraphExecDestroy failed: status {status}");
            }
        }
        Ok(())
    }

    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        let status = unsafe { cuMemsetD8Async(ptr.0, value, bytes, self.default_stream) };
        if status != 0 {
            bail!("cuMemsetD8Async failed: status {status}");
        }
        let sync = unsafe { cuStreamSynchronize(self.default_stream) };
        if sync != 0 {
            bail!("cuStreamSynchronize after memset failed: status {sync}");
        }
        Ok(())
    }

    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
        let status = unsafe { cuMemsetD8Async(ptr.0, value, bytes, stream) };
        if status != 0 {
            bail!("cuMemsetD8Async failed: status {status}");
        }
        Ok(())
    }

    fn total_memory(&self) -> Result<usize> {
        let mut free: usize = 0;
        let mut total: usize = 0;
        let status = unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
        if status != 0 {
            bail!("cuMemGetInfo_v2 failed: status {status}");
        }
        Ok(total)
    }

    fn free_memory(&self) -> Result<usize> {
        let mut free: usize = 0;
        let mut total: usize = 0;
        let status = unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
        if status != 0 {
            bail!("cuMemGetInfo_v2 failed: status {status}");
        }
        // On unified memory (GB10), cuMemGetInfo reports Linux "free" memory
        // which excludes reclaimable buff/cache. Use MemAvailable instead.
        if let Some(mem_available) = super::system_available_memory_bytes() {
            free = free.max(mem_available);
        }
        Ok(free)
    }

    fn create_stream(&self) -> Result<u64> {
        let mut stream: u64 = 0;
        // CU_STREAM_NON_BLOCKING = 1 (does not synchronize with stream 0)
        let status = unsafe { cuStreamCreate(&mut stream, 1) };
        if status != 0 {
            bail!("cuStreamCreate failed: status {status}");
        }
        Ok(stream)
    }

    fn bind_to_thread(&self) -> Result<()> {
        let status = unsafe { cuCtxSetCurrent(self.cuda_ctx) };
        if status != 0 {
            bail!("cuCtxSetCurrent failed: status {status}");
        }
        Ok(())
    }

    fn create_event(&self) -> Result<u64> {
        let mut event: u64 = 0;
        // CU_EVENT_DISABLE_TIMING = 0x02 (skip timing overhead)
        let status = unsafe { cuEventCreate(&mut event, 0x02) };
        if status != 0 {
            bail!("cuEventCreate failed: status {status}");
        }
        Ok(event)
    }

    fn record_event(&self, event: u64, stream: u64) -> Result<()> {
        let status = unsafe { cuEventRecord(event, stream) };
        if status != 0 {
            bail!("cuEventRecord failed: status {status}");
        }
        Ok(())
    }

    fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
        let status = unsafe { cuStreamWaitEvent(stream, event, 0) };
        if status != 0 {
            bail!("cuStreamWaitEvent failed: status {status}");
        }
        Ok(())
    }

    fn destroy_event(&self, event: u64) -> Result<()> {
        if event != 0 {
            let status = unsafe { cuEventDestroy_v2(event) };
            if status != 0 {
                bail!("cuEventDestroy_v2 failed: status {status}");
            }
        }
        Ok(())
    }

    fn alloc_host_pinned(&self, bytes: usize) -> Result<PinnedHostBuffer> {
        if bytes == 0 {
            bail!("pinned host allocation must be non-empty");
        }
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let status = unsafe { cuMemAllocHost_v2(&mut ptr, bytes) };
        if status != 0 {
            bail!("cuMemAllocHost_v2 failed: status {status}, requested {bytes} bytes");
        }
        let ptr = NonNull::new(ptr.cast::<u8>())
            .ok_or_else(|| anyhow::anyhow!("cuMemAllocHost_v2 returned null"))?;
        Ok(PinnedHostBuffer::from_storage(Box::new(
            CudaPinnedHostStorage { ptr, bytes },
        )))
    }
}
