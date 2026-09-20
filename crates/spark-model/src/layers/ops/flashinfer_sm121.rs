// SPDX-License-Identifier: AGPL-3.0-only

//! Default-unrouted dynamic boundary for the native FlashInfer SM121 C ABI.
//!
//! Production call contract:
//! 1. Explicitly open an absolute, trusted `libatlas_fi_fp4_sm121.so` path.
//! 2. Construct a validated [`FlashInferSm121Shape`] and call [`FlashInferSm121::prepare`]
//!    before CUDA graph capture. Preparation binds the backend's CUDA context,
//!    queries tactic-specific workspace bytes, enforces a caller-supplied cap,
//!    and allocates the workspace through that same backend/context.
//! 3. Keep [`PreparedFlashInferSm121`] alive and use exactly its frozen stream,
//!    shape, tactic, library handle, and workspace. Every launch rebinds the
//!    backend context before entering the CUDA-runtime C ABI.
//! 4. For graph capture, create [`FlashInferSm121GraphPin`] before capture. It
//!    freezes all device addresses and rejects the legacy default stream. Keep
//!    the pin alive until the graph is destroyed; replay and destruction go
//!    through the pin so the prepared workspace/library cannot be dropped early.
//! 5. Explicit close or `Drop` synchronizes the frozen stream before freeing the
//!    workspace. No preparation, tactic change, allocation, or unload is legal
//!    while a captured graph can still reference this operation.
//! 6. Callers that only have `&dyn GpuBackend` may instead use
//!    [`FlashInferSm121::prepare_borrowed_zero_workspace`]. That eager-only seam
//!    binds the borrowed backend, requires the ABI query to return exactly zero,
//!    and never allocates or frees. It is deliberately not graph-capable; graph
//!    integration must use the `Arc`-owned preparation/pin lifetime above.
//!
//! Nothing in production constructs this type yet. Merely compiling this module
//! does not route a model projection to FlashInfer.

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle};

const RTLD_NOW: c_int = 2;
const TACTIC_COUNT: u8 = 6;
const MAX_LIBRARY_BYTES: u64 = 64 * 1024 * 1024;
const MFD_CLOEXEC: c_uint = 0x0001;
const MFD_ALLOW_SEALING: c_uint = 0x0002;
const F_ADD_SEALS: c_int = 1_033;
const F_GET_SEALS: c_int = 1_034;
const F_SEAL_SEAL: c_int = 0x0001;
const F_SEAL_SHRINK: c_int = 0x0002;
const F_SEAL_GROW: c_int = 0x0004;
const F_SEAL_WRITE: c_int = 0x0008;
const REQUIRED_MEMFD_SEALS: c_int = F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE;

type WorkspaceFn = unsafe extern "C" fn(c_int, c_int, c_int, c_int, c_int, *mut usize) -> c_int;
type GemmFn = unsafe extern "C" fn(
    c_int,
    *mut c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const f32,
    c_int,
    c_int,
    c_int,
    c_int,
    *mut c_void,
    usize,
    *mut c_void,
) -> c_int;
type LastErrorFn = unsafe extern "C" fn() -> *const c_char;

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
    fn memfd_create(name: *const c_char, flags: c_uint) -> c_int;
    fn fcntl(fd: c_int, operation: c_int, ...) -> c_int;
}

struct RawApi {
    handle: *mut c_void,
    close_handle: bool,
    workspace: WorkspaceFn,
    gemm: GemmFn,
    last_error: LastErrorFn,
    path: PathBuf,
    // Keep the verified, write/grow/shrink-sealed anonymous copy alive for
    // exactly as long as any resolved symbol can be called. `dlopen` is
    // performed through this descriptor, never through the mutable source
    // pathname or inode.
    _sealed_library: Option<File>,
    sha256: [u8; 32],
    device: u64,
    inode: u64,
}

// The loaded library is immutable after construction. Its C wrapper creates a
// fresh CUTLASS runner per call and its diagnostic storage is thread-local.
unsafe impl Send for RawApi {}
unsafe impl Sync for RawApi {}

impl Drop for RawApi {
    fn drop(&mut self) {
        if self.close_handle && !self.handle.is_null() {
            let status = unsafe { dlclose(self.handle) };
            if status != 0 {
                tracing::warn!(
                    path = %self.path.display(),
                    status,
                    "dlclose failed for FlashInfer SM121 C ABI"
                );
            }
        }
    }
}

/// Explicit handle to the default-unrouted native FlashInfer SM121 library.
#[derive(Clone)]
pub struct FlashInferSm121 {
    api: Arc<RawApi>,
}

impl FlashInferSm121 {
    /// Load the exact three-symbol ABI from an absolute regular-file path and
    /// require its bytes to match a caller-frozen SHA-256 identity.
    ///
    /// The source file is opened and hashed, then its verified bytes are copied
    /// into an anonymous memfd with WRITE/GROW/SHRINK/SEAL all applied before
    /// `dlopen(/proc/self/fd/N)`. The sealed descriptor is retained. Path
    /// replacement and post-admission mutation of the source inode therefore
    /// cannot change any mapped or not-yet-faulted library page.
    pub fn open_with_sha256(path: &Path, expected_sha256: [u8; 32]) -> Result<Self> {
        ensure!(
            usize::BITS == 64,
            "FlashInfer SM121 ABI requires a 64-bit host"
        );
        ensure!(
            path.is_absolute(),
            "FlashInfer SM121 library path must be absolute"
        );
        let canonical = path
            .canonicalize()
            .with_context(|| format!("canonicalize FlashInfer library {}", path.display()))?;
        let mut library_file = File::open(&canonical)
            .with_context(|| format!("open FlashInfer library {}", canonical.display()))?;
        let metadata_before = library_file.metadata()?;
        ensure!(
            metadata_before.is_file(),
            "FlashInfer SM121 library must be a regular file: {}",
            canonical.display()
        );
        ensure!(
            metadata_before.len() > 0 && metadata_before.len() <= MAX_LIBRARY_BYTES,
            "FlashInfer SM121 library size {} is outside 1..={MAX_LIBRARY_BYTES} bytes",
            metadata_before.len()
        );
        let mut library_bytes = Vec::new();
        (&mut library_file)
            .take(MAX_LIBRARY_BYTES + 1)
            .read_to_end(&mut library_bytes)?;
        ensure!(
            library_bytes.len() as u64 <= MAX_LIBRARY_BYTES,
            "FlashInfer SM121 library exceeds {MAX_LIBRARY_BYTES} bytes while hashing"
        );
        let sha256 = sha256(&library_bytes);
        ensure!(
            sha256 == expected_sha256,
            "FlashInfer SM121 library SHA-256 mismatch: expected {}, got {}",
            hex_sha256(expected_sha256),
            hex_sha256(sha256)
        );

        let metadata_after_read = library_file.metadata()?;
        ensure!(
            same_file_identity(&metadata_before, &metadata_after_read),
            "FlashInfer SM121 source library changed while it was being read: {}",
            canonical.display()
        );

        let mut sealed_library = sealed_memfd(&library_bytes)?;
        let sealed_sha256 = sha256_file(&mut sealed_library)?;
        ensure!(
            sealed_sha256 == sha256,
            "sealed FlashInfer SM121 copy does not match admitted source SHA-256"
        );
        let sealed_metadata = sealed_library.metadata()?;
        let descriptor_path = format!("/proc/self/fd/{}", sealed_library.as_raw_fd());
        let encoded = CString::new(descriptor_path.as_bytes())
            .context("FlashInfer SM121 descriptor path contains a NUL byte")?;
        clear_dlerror();
        let handle = unsafe { dlopen(encoded.as_ptr(), RTLD_NOW) };
        ensure!(
            !handle.is_null(),
            "dlopen admitted descriptor for {} failed: {}",
            canonical.display(),
            current_dlerror()
        );

        let loaded = (|| -> Result<(WorkspaceFn, GemmFn, LastErrorFn)> {
            let workspace_ptr = resolve_symbol(handle, b"atlas_fi_nvfp4_sm121_workspace_size\0")?;
            let gemm_ptr = resolve_symbol(handle, b"atlas_fi_nvfp4_sm121_bf16\0")?;
            let last_error_ptr = resolve_symbol(handle, b"atlas_fi_nvfp4_sm121_last_error\0")?;
            let workspace =
                unsafe { std::mem::transmute::<*mut c_void, WorkspaceFn>(workspace_ptr) };
            let gemm = unsafe { std::mem::transmute::<*mut c_void, GemmFn>(gemm_ptr) };
            let last_error =
                unsafe { std::mem::transmute::<*mut c_void, LastErrorFn>(last_error_ptr) };
            Ok((workspace, gemm, last_error))
        })();

        let (workspace, gemm, last_error) = match loaded {
            Ok(symbols) => symbols,
            Err(error) => {
                let _ = unsafe { dlclose(handle) };
                return Err(error).with_context(|| {
                    format!("load FlashInfer SM121 ABI from {}", canonical.display())
                });
            }
        };

        Ok(Self {
            api: Arc::new(RawApi {
                handle,
                close_handle: true,
                workspace,
                gemm,
                last_error,
                path: canonical,
                _sealed_library: Some(sealed_library),
                sha256,
                device: sealed_metadata.dev(),
                inode: sealed_metadata.ino(),
            }),
        })
    }

    /// Bind the backend context, query workspace, and allocate it before capture.
    pub fn prepare(
        &self,
        gpu: Arc<dyn GpuBackend>,
        shape: FlashInferSm121Shape,
        stream: u64,
        max_workspace_bytes: usize,
    ) -> Result<PreparedFlashInferSm121> {
        ensure!(
            max_workspace_bytes > 0,
            "FlashInfer workspace cap must be non-zero"
        );
        gpu.bind_to_thread()
            .context("bind Atlas CUDA context before FlashInfer workspace query")?;
        let mut workspace_bytes = 0usize;
        let status = unsafe {
            (self.api.workspace)(
                c_int::from(shape.tactic),
                shape.m,
                shape.n,
                shape.k,
                shape.batch_count,
                &mut workspace_bytes,
            )
        };
        self.check_status("workspace_size", status)?;
        ensure!(
            workspace_bytes <= max_workspace_bytes,
            "FlashInfer workspace request {workspace_bytes} exceeds cap {max_workspace_bytes}"
        );
        let workspace = if workspace_bytes == 0 {
            DevicePtr::NULL
        } else {
            gpu.alloc(workspace_bytes).with_context(|| {
                format!("allocate FlashInfer workspace ({workspace_bytes} bytes)")
            })?
        };
        Ok(PreparedFlashInferSm121 {
            library: self.clone(),
            gpu,
            shape,
            stream,
            workspace,
            workspace_bytes,
        })
    }

    /// Prepare the eager-only borrowed-backend seam for a proven zero-workspace tactic.
    ///
    /// This is the minimally invasive route for model code whose forward context
    /// exposes only `&dyn GpuBackend`. A non-zero result is rejected rather than
    /// allocated, and this object intentionally provides no graph capture API.
    pub fn prepare_borrowed_zero_workspace<'gpu>(
        &self,
        gpu: &'gpu dyn GpuBackend,
        shape: FlashInferSm121Shape,
        stream: u64,
    ) -> Result<BorrowedZeroWorkspaceFlashInferSm121<'gpu>> {
        gpu.bind_to_thread()
            .context("bind Atlas CUDA context before FlashInfer zero-workspace query")?;
        let mut workspace_bytes = usize::MAX;
        let status = unsafe {
            (self.api.workspace)(
                c_int::from(shape.tactic),
                shape.m,
                shape.n,
                shape.k,
                shape.batch_count,
                &mut workspace_bytes,
            )
        };
        self.check_status("workspace_size", status)?;
        ensure!(
            workspace_bytes == 0,
            "borrowed FlashInfer route requires zero workspace, ABI requested {workspace_bytes} bytes"
        );
        Ok(BorrowedZeroWorkspaceFlashInferSm121 {
            library: self.clone(),
            gpu,
            shape,
            stream,
        })
    }

    fn check_status(&self, operation: &str, status: c_int) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        let pointer = unsafe { (self.api.last_error)() };
        let detail = if pointer.is_null() {
            "<null last_error>".to_owned()
        } else {
            unsafe { CStr::from_ptr(pointer) }
                .to_string_lossy()
                .into_owned()
        };
        bail!("FlashInfer {operation} failed with status {status}: {detail}")
    }

    /// Canonical path retained by the loaded library owner.
    pub fn path(&self) -> &Path {
        &self.api.path
    }

    /// Frozen content identity of the retained library descriptor.
    pub fn sha256_hex(&self) -> String {
        hex_sha256(self.api.sha256)
    }

    /// Stable filesystem identity captured before descriptor-based loading.
    pub fn file_identity(&self) -> (u64, u64) {
        (self.api.device, self.api.inode)
    }

    #[cfg(test)]
    fn from_test_api(workspace: WorkspaceFn, gemm: GemmFn, last_error: LastErrorFn) -> Self {
        Self {
            api: Arc::new(RawApi {
                handle: std::ptr::null_mut(),
                close_handle: false,
                workspace,
                gemm,
                last_error,
                path: PathBuf::from("<test-api>"),
                _sealed_library: None,
                sha256: [0; 32],
                device: 0,
                inode: 0,
            }),
        }
    }
}

/// Eager-only zero-workspace launch owner borrowing Atlas's CUDA backend.
///
/// It retains the dynamic library, tactic and stream, but cannot outlive the
/// backend reference and exposes no graph-capture operation. A mutable launch
/// borrow keeps one object from being launched concurrently.
pub struct BorrowedZeroWorkspaceFlashInferSm121<'gpu> {
    library: FlashInferSm121,
    gpu: &'gpu dyn GpuBackend,
    shape: FlashInferSm121Shape,
    stream: u64,
}

impl BorrowedZeroWorkspaceFlashInferSm121<'_> {
    pub fn stream(&self) -> u64 {
        self.stream
    }

    pub fn shape(&self) -> FlashInferSm121Shape {
        self.shape
    }

    /// Launch with a null workspace on the exact stream used for preparation.
    pub fn launch_eager(&mut self, buffers: FlashInferSm121Buffers, stream: u64) -> Result<()> {
        ensure!(
            stream == self.stream,
            "FlashInfer stream mismatch: prepared {:#x}, launch {stream:#x}",
            self.stream
        );
        buffers.validate()?;
        self.gpu
            .bind_to_thread()
            .context("bind Atlas CUDA context before FlashInfer borrowed launch")?;
        let status = unsafe {
            (self.library.api.gemm)(
                c_int::from(self.shape.tactic),
                device_mut(buffers.output_bf16),
                device_const(buffers.activation_fp4),
                device_const(buffers.weight_fp4),
                device_const(buffers.activation_scales),
                device_const(buffers.weight_scales),
                device_const(buffers.global_scale_f32).cast(),
                self.shape.m,
                self.shape.n,
                self.shape.k,
                self.shape.batch_count,
                std::ptr::null_mut(),
                0,
                self.stream as usize as *mut c_void,
            )
        };
        self.library.check_status("bf16", status)
    }
}

/// Validated, frozen GEMM geometry and CUTLASS tactic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashInferSm121Shape {
    tactic: u8,
    m: c_int,
    n: c_int,
    k: c_int,
    batch_count: c_int,
}

impl FlashInferSm121Shape {
    pub fn new(tactic: u8, m: usize, n: usize, k: usize, batch_count: usize) -> Result<Self> {
        ensure!(tactic < TACTIC_COUNT, "FlashInfer tactic must be in 0..6");
        ensure!(m > 0 && n > 0 && k > 0, "FlashInfer M/N/K must be non-zero");
        ensure!(batch_count > 0, "FlashInfer batch_count must be non-zero");
        ensure!(
            k.is_multiple_of(32),
            "FlashInfer K must be a multiple of 32"
        );
        Ok(Self {
            tactic,
            m: c_int::try_from(m).context("FlashInfer M exceeds C ABI i32")?,
            n: c_int::try_from(n).context("FlashInfer N exceeds C ABI i32")?,
            k: c_int::try_from(k).context("FlashInfer K exceeds C ABI i32")?,
            batch_count: c_int::try_from(batch_count)
                .context("FlashInfer batch_count exceeds C ABI i32")?,
        })
    }

    pub fn tactic(self) -> u8 {
        self.tactic
    }
}

/// Fixed device addresses consumed by one prepared GEMM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashInferSm121Buffers {
    pub output_bf16: DevicePtr,
    pub activation_fp4: DevicePtr,
    pub weight_fp4: DevicePtr,
    pub activation_scales: DevicePtr,
    pub weight_scales: DevicePtr,
    pub global_scale_f32: DevicePtr,
}

impl FlashInferSm121Buffers {
    fn validate(self) -> Result<()> {
        let aligned_16 = [
            ("output_bf16", self.output_bf16),
            ("activation_fp4", self.activation_fp4),
            ("weight_fp4", self.weight_fp4),
            ("activation_scales", self.activation_scales),
            ("weight_scales", self.weight_scales),
        ];
        for (name, pointer) in aligned_16 {
            ensure!(!pointer.is_null(), "FlashInfer {name} pointer is null");
            ensure!(
                pointer.0 % 16 == 0,
                "FlashInfer {name} pointer is not 16-byte aligned"
            );
        }
        ensure!(
            !self.global_scale_f32.is_null(),
            "FlashInfer global_scale_f32 pointer is null"
        );
        ensure!(
            self.global_scale_f32.0.is_multiple_of(4),
            "FlashInfer global_scale_f32 pointer is not 4-byte aligned"
        );
        Ok(())
    }
}

/// Owns the tactic-specific workspace and freezes its context/stream lifetime.
pub struct PreparedFlashInferSm121 {
    library: FlashInferSm121,
    gpu: Arc<dyn GpuBackend>,
    shape: FlashInferSm121Shape,
    stream: u64,
    workspace: DevicePtr,
    workspace_bytes: usize,
}

impl PreparedFlashInferSm121 {
    pub fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    pub fn stream(&self) -> u64 {
        self.stream
    }

    pub fn shape(&self) -> FlashInferSm121Shape {
        self.shape
    }

    /// Eager launch. The mutable borrow prevents concurrent workspace reuse.
    pub fn launch_eager(&mut self, buffers: FlashInferSm121Buffers, stream: u64) -> Result<()> {
        ensure!(
            stream == self.stream,
            "FlashInfer stream mismatch: prepared {:#x}, launch {stream:#x}",
            self.stream
        );
        self.launch_fixed(buffers)
    }

    /// Freeze device addresses for capture. Must be created before capture starts.
    pub fn pin_for_graph(
        &mut self,
        buffers: FlashInferSm121Buffers,
    ) -> Result<FlashInferSm121GraphPin<'_>> {
        ensure!(
            self.stream != 0,
            "FlashInfer graph capture rejects the legacy default stream"
        );
        buffers.validate()?;
        Ok(FlashInferSm121GraphPin {
            prepared: self,
            buffers,
        })
    }

    /// Synchronize and release explicitly. `Drop` performs the same operation.
    pub fn close(mut self) -> Result<()> {
        self.release_workspace()
    }

    fn launch_fixed(&mut self, buffers: FlashInferSm121Buffers) -> Result<()> {
        buffers.validate()?;
        self.gpu
            .bind_to_thread()
            .context("bind Atlas CUDA context before FlashInfer launch")?;
        let status = unsafe {
            (self.library.api.gemm)(
                c_int::from(self.shape.tactic),
                device_mut(buffers.output_bf16),
                device_const(buffers.activation_fp4),
                device_const(buffers.weight_fp4),
                device_const(buffers.activation_scales),
                device_const(buffers.weight_scales),
                device_const(buffers.global_scale_f32).cast(),
                self.shape.m,
                self.shape.n,
                self.shape.k,
                self.shape.batch_count,
                device_mut(self.workspace),
                self.workspace_bytes,
                self.stream as usize as *mut c_void,
            )
        };
        self.library.check_status("bf16", status)
    }

    fn release_workspace(&mut self) -> Result<()> {
        if self.workspace.is_null() {
            return Ok(());
        }
        self.gpu
            .bind_to_thread()
            .context("bind Atlas CUDA context before FlashInfer workspace release")?;
        self.gpu
            .synchronize(self.stream)
            .context("synchronize FlashInfer stream before workspace release")?;
        let pointer = std::mem::replace(&mut self.workspace, DevicePtr::NULL);
        if let Err(error) = self.gpu.free(pointer) {
            self.workspace = pointer;
            return Err(error).context("free FlashInfer workspace");
        }
        Ok(())
    }
}

impl Drop for PreparedFlashInferSm121 {
    fn drop(&mut self) {
        if let Err(error) = self.release_workspace() {
            tracing::warn!(%error, "failed to release FlashInfer SM121 workspace");
        }
    }
}

/// Borrowed graph-lifetime token retaining fixed addresses, workspace and SO.
pub struct FlashInferSm121GraphPin<'a> {
    prepared: &'a mut PreparedFlashInferSm121,
    buffers: FlashInferSm121Buffers,
}

impl FlashInferSm121GraphPin<'_> {
    /// Record the fixed-address GEMM while the caller's stream is capturing.
    pub fn capture_launch(&mut self) -> Result<()> {
        self.prepared.launch_fixed(self.buffers)
    }

    /// Replay on the same non-default stream used during preparation/capture.
    pub fn replay(&self, graph: GraphHandle) -> Result<()> {
        self.prepared
            .gpu
            .bind_to_thread()
            .context("bind Atlas CUDA context before FlashInfer graph replay")?;
        self.prepared.gpu.launch_graph(graph, self.prepared.stream)
    }

    /// Synchronize, destroy the graph, and consume the pin before workspace drop.
    pub fn destroy(self, graph: GraphHandle) -> Result<()> {
        self.prepared
            .gpu
            .bind_to_thread()
            .context("bind Atlas CUDA context before FlashInfer graph destroy")?;
        self.prepared
            .gpu
            .synchronize(self.prepared.stream)
            .context("synchronize FlashInfer graph replay stream before destroy")?;
        self.prepared.gpu.destroy_graph(graph)
    }
}

fn device_mut(pointer: DevicePtr) -> *mut c_void {
    pointer.0 as usize as *mut c_void
}

fn device_const(pointer: DevicePtr) -> *const c_void {
    pointer.0 as usize as *const c_void
}

fn same_file_identity(before: &std::fs::Metadata, after: &std::fs::Metadata) -> bool {
    before.dev() == after.dev()
        && before.ino() == after.ino()
        && before.len() == after.len()
        && before.mtime() == after.mtime()
        && before.mtime_nsec() == after.mtime_nsec()
        && before.ctime() == after.ctime()
        && before.ctime_nsec() == after.ctime_nsec()
}

fn sealed_memfd(bytes: &[u8]) -> Result<File> {
    let name = CString::new("atlas-fi-fp4-sm121")?;
    let descriptor = unsafe { memfd_create(name.as_ptr(), MFD_CLOEXEC | MFD_ALLOW_SEALING) };
    ensure!(
        descriptor >= 0,
        "memfd_create for FlashInfer SM121 failed: {}",
        std::io::Error::last_os_error()
    );
    // SAFETY: memfd_create returned a fresh owned descriptor, transferred
    // exactly once into File for RAII close on all following error paths.
    let mut file = unsafe { File::from_raw_fd(descriptor) };
    file.write_all(bytes)
        .context("write verified FlashInfer SM121 bytes into memfd")?;
    file.flush()
        .context("flush verified FlashInfer SM121 memfd")?;
    file.seek(SeekFrom::Start(0))?;
    let status = unsafe { fcntl(file.as_raw_fd(), F_ADD_SEALS, REQUIRED_MEMFD_SEALS) };
    ensure!(
        status == 0,
        "seal FlashInfer SM121 memfd failed: {}",
        std::io::Error::last_os_error()
    );
    let seals = unsafe { fcntl(file.as_raw_fd(), F_GET_SEALS) };
    ensure!(
        seals >= 0 && seals & REQUIRED_MEMFD_SEALS == REQUIRED_MEMFD_SEALS,
        "FlashInfer SM121 memfd is missing required seals: expected {REQUIRED_MEMFD_SEALS:#x}, got {seals:#x}"
    );
    Ok(file)
}

fn sha256_file(file: &mut File) -> Result<[u8; 32]> {
    file.seek(SeekFrom::Start(0))?;
    let mut bytes = Vec::new();
    file.take(MAX_LIBRARY_BYTES + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_LIBRARY_BYTES,
        "FlashInfer SM121 library exceeds {MAX_LIBRARY_BYTES} bytes while hashing"
    );
    Ok(sha256(&bytes))
}

fn hex_sha256(digest: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

// Small dependency-free SHA-256 used only to seal the native library before
// `dlopen`. Keeping it here avoids expanding the server's dependency surface
// for one startup-time 1.9 MiB digest.
fn sha256(input: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut state = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (input.len() as u64).wrapping_mul(8);
    let padded_len = (input.len() + 9).div_ceil(64) * 64;
    let mut padded = Vec::with_capacity(padded_len);
    padded.extend_from_slice(input);
    padded.push(0x80);
    padded.resize(padded_len - 8, 0);
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (index, word) in chunk.chunks_exact(4).enumerate() {
            w[index] = u32::from_be_bytes(word.try_into().expect("four-byte SHA-256 word"));
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7)
                ^ w[index - 15].rotate_right(18)
                ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17)
                ^ w[index - 2].rotate_right(19)
                ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(choice)
                .wrapping_add(K[index])
                .wrapping_add(w[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h].into_iter()) {
            *slot = slot.wrapping_add(value);
        }
    }

    let mut digest = [0u8; 32];
    for (word, bytes) in state.iter().zip(digest.chunks_exact_mut(4)) {
        bytes.copy_from_slice(&word.to_be_bytes());
    }
    digest
}

fn clear_dlerror() {
    let _ = unsafe { dlerror() };
}

fn current_dlerror() -> String {
    let pointer = unsafe { dlerror() };
    if pointer.is_null() {
        "<no dlerror>".to_owned()
    } else {
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    }
}

fn resolve_symbol(handle: *mut c_void, symbol: &'static [u8]) -> Result<*mut c_void> {
    clear_dlerror();
    let pointer = unsafe { dlsym(handle, symbol.as_ptr().cast()) };
    let error = current_dlerror();
    ensure!(
        !pointer.is_null() && error == "<no dlerror>",
        "dlsym({}) failed: {error}",
        String::from_utf8_lossy(&symbol[..symbol.len() - 1])
    );
    Ok(pointer)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicI32, AtomicU64, AtomicUsize, Ordering};
    use std::sync::{LazyLock, Mutex};

    use spark_runtime::gpu::mock::MockGpuBackend;

    use super::*;

    static TEST_LOCK: Mutex<()> = Mutex::new(());
    static WORKSPACE_STATUS: AtomicI32 = AtomicI32::new(0);
    static GEMM_STATUS: AtomicI32 = AtomicI32::new(0);
    static WORKSPACE_BYTES: AtomicUsize = AtomicUsize::new(0);
    static GEMM_CALLS: AtomicUsize = AtomicUsize::new(0);
    static LAST_STREAM: AtomicU64 = AtomicU64::new(0);
    static LAST_WORKSPACE: AtomicU64 = AtomicU64::new(0);
    static LAST_OUTPUT: AtomicU64 = AtomicU64::new(0);
    static ERROR_TEXT: &[u8] = b"forced C ABI failure\0";
    static VALID_BUFFERS: LazyLock<FlashInferSm121Buffers> =
        LazyLock::new(|| FlashInferSm121Buffers {
            output_bf16: DevicePtr(0x2000),
            activation_fp4: DevicePtr(0x3000),
            weight_fp4: DevicePtr(0x4000),
            activation_scales: DevicePtr(0x5000),
            weight_scales: DevicePtr(0x6000),
            global_scale_f32: DevicePtr(0x7000),
        });

    unsafe extern "C" fn fake_workspace(
        _tactic: c_int,
        _m: c_int,
        _n: c_int,
        _k: c_int,
        _batch: c_int,
        output: *mut usize,
    ) -> c_int {
        if !output.is_null() {
            unsafe { *output = WORKSPACE_BYTES.load(Ordering::SeqCst) };
        }
        WORKSPACE_STATUS.load(Ordering::SeqCst)
    }

    unsafe extern "C" fn fake_gemm(
        _tactic: c_int,
        output: *mut c_void,
        _activation: *const c_void,
        _weight: *const c_void,
        _activation_scales: *const c_void,
        _weight_scales: *const c_void,
        _global_scale: *const f32,
        _m: c_int,
        _n: c_int,
        _k: c_int,
        _batch: c_int,
        workspace: *mut c_void,
        _workspace_bytes: usize,
        stream: *mut c_void,
    ) -> c_int {
        GEMM_CALLS.fetch_add(1, Ordering::SeqCst);
        LAST_STREAM.store(stream as usize as u64, Ordering::SeqCst);
        LAST_WORKSPACE.store(workspace as usize as u64, Ordering::SeqCst);
        LAST_OUTPUT.store(output as usize as u64, Ordering::SeqCst);
        GEMM_STATUS.load(Ordering::SeqCst)
    }

    unsafe extern "C" fn fake_last_error() -> *const c_char {
        ERROR_TEXT.as_ptr().cast()
    }

    fn library() -> FlashInferSm121 {
        FlashInferSm121::from_test_api(fake_workspace, fake_gemm, fake_last_error)
    }

    fn shape() -> FlashInferSm121Shape {
        FlashInferSm121Shape::new(3, 2_079, 17_408, 5_120, 1).unwrap()
    }

    fn reset(workspace: usize) {
        WORKSPACE_STATUS.store(0, Ordering::SeqCst);
        GEMM_STATUS.store(0, Ordering::SeqCst);
        WORKSPACE_BYTES.store(workspace, Ordering::SeqCst);
        GEMM_CALLS.store(0, Ordering::SeqCst);
        LAST_STREAM.store(0, Ordering::SeqCst);
        LAST_WORKSPACE.store(0, Ordering::SeqCst);
        LAST_OUTPUT.store(0, Ordering::SeqCst);
    }

    #[test]
    fn rejects_hostile_shapes_and_tactics() {
        assert!(FlashInferSm121Shape::new(6, 1, 1, 32, 1).is_err());
        assert!(FlashInferSm121Shape::new(0, 0, 1, 32, 1).is_err());
        assert!(FlashInferSm121Shape::new(0, 1, 1, 31, 1).is_err());
        assert!(FlashInferSm121Shape::new(0, 1, 1, 32, 0).is_err());
        assert!(FlashInferSm121Shape::new(0, i32::MAX as usize + 1, 1, 32, 1).is_err());
    }

    #[test]
    fn explicit_open_rejects_relative_and_missing_paths() {
        assert!(FlashInferSm121::open_with_sha256(Path::new("relative.so"), [0; 32]).is_err());
        assert!(
            FlashInferSm121::open_with_sha256(
                Path::new("/definitely/missing/atlas-fi.so"),
                [0; 32],
            )
            .is_err()
        );
    }

    #[test]
    fn sha256_matches_standard_vectors() {
        assert_eq!(
            hex_sha256(sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex_sha256(sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sealed_memfd_is_byte_exact_and_rejects_writes() {
        let mut file = sealed_memfd(b"immutable FlashInfer bytes").unwrap();
        assert_eq!(
            sha256_file(&mut file).unwrap(),
            sha256(b"immutable FlashInfer bytes")
        );
        file.seek(SeekFrom::Start(0)).unwrap();
        assert!(file.write_all(b"mutation").is_err());
        assert_eq!(
            sha256_file(&mut file).unwrap(),
            sha256(b"immutable FlashInfer bytes")
        );
    }

    #[test]
    fn workspace_error_and_cap_fail_before_allocation() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset(4_096);
        let gpu = Arc::new(MockGpuBackend::new());
        WORKSPACE_STATUS.store(-2, Ordering::SeqCst);
        let error = library()
            .prepare(gpu.clone(), shape(), 0x88, 8_192)
            .err()
            .unwrap();
        assert!(error.to_string().contains("forced C ABI failure"));
        assert_eq!(gpu.alloc_count(), 0);

        WORKSPACE_STATUS.store(0, Ordering::SeqCst);
        let error = library()
            .prepare(gpu.clone(), shape(), 0x88, 1_024)
            .err()
            .unwrap();
        assert!(error.to_string().contains("exceeds cap"));
        assert_eq!(gpu.alloc_count(), 0);
    }

    #[test]
    fn zero_workspace_avoids_allocation() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset(0);
        let gpu = Arc::new(MockGpuBackend::new());
        let prepared = library().prepare(gpu.clone(), shape(), 0x88, 1).unwrap();
        assert_eq!(prepared.workspace_bytes(), 0);
        assert_eq!(gpu.alloc_count(), 0);
        prepared.close().unwrap();
    }

    #[test]
    fn borrowed_route_fail_closes_nonzero_workspace_and_never_allocates() {
        let _guard = TEST_LOCK.lock().unwrap();
        let gpu = MockGpuBackend::new();

        reset(1);
        let error = library()
            .prepare_borrowed_zero_workspace(&gpu, shape(), 0x88)
            .err()
            .unwrap();
        assert!(error.to_string().contains("requires zero workspace"));
        assert_eq!(gpu.alloc_count(), 0);

        reset(0);
        {
            let mut prepared = library()
                .prepare_borrowed_zero_workspace(&gpu, shape(), 0x88)
                .unwrap();
            assert_eq!(prepared.stream(), 0x88);
            assert_eq!(prepared.shape(), shape());
            assert!(prepared.launch_eager(*VALID_BUFFERS, 0x99).is_err());
            assert_eq!(GEMM_CALLS.load(Ordering::SeqCst), 0);
            prepared.launch_eager(*VALID_BUFFERS, 0x88).unwrap();
            assert_eq!(GEMM_CALLS.load(Ordering::SeqCst), 1);
            assert_eq!(LAST_WORKSPACE.load(Ordering::SeqCst), 0);
            assert_eq!(LAST_STREAM.load(Ordering::SeqCst), 0x88);
        }
        assert_eq!(gpu.alloc_count(), 0);
    }

    #[test]
    fn launch_freezes_stream_workspace_and_propagates_errors() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset(4_096);
        let gpu = Arc::new(MockGpuBackend::new());
        let mut prepared = library()
            .prepare(gpu.clone(), shape(), 0x88, 8_192)
            .unwrap();
        assert_eq!(gpu.alloc_count(), 1);
        assert!(prepared.launch_eager(*VALID_BUFFERS, 0x99).is_err());
        assert_eq!(GEMM_CALLS.load(Ordering::SeqCst), 0);

        let mut misaligned = *VALID_BUFFERS;
        misaligned.weight_fp4 = DevicePtr(0x4001);
        assert!(prepared.launch_eager(misaligned, 0x88).is_err());
        assert_eq!(GEMM_CALLS.load(Ordering::SeqCst), 0);

        prepared.launch_eager(*VALID_BUFFERS, 0x88).unwrap();
        assert_eq!(GEMM_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(LAST_STREAM.load(Ordering::SeqCst), 0x88);
        assert_ne!(LAST_WORKSPACE.load(Ordering::SeqCst), 0);
        assert_eq!(LAST_OUTPUT.load(Ordering::SeqCst), 0x2000);

        GEMM_STATUS.store(-2, Ordering::SeqCst);
        let error = prepared.launch_eager(*VALID_BUFFERS, 0x88).unwrap_err();
        assert!(error.to_string().contains("forced C ABI failure"));
        prepared.close().unwrap();
        assert_eq!(gpu.alloc_count(), 0);
    }

    #[test]
    fn graph_pin_rejects_default_stream_and_retains_fixed_addresses() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset(128);
        let gpu = Arc::new(MockGpuBackend::new());
        let mut default_stream = library().prepare(gpu.clone(), shape(), 0, 1_024).unwrap();
        assert!(default_stream.pin_for_graph(*VALID_BUFFERS).is_err());
        default_stream.close().unwrap();

        let mut prepared = library()
            .prepare(gpu.clone(), shape(), 0x77, 1_024)
            .unwrap();
        {
            let mut pin = prepared.pin_for_graph(*VALID_BUFFERS).unwrap();
            pin.capture_launch().unwrap();
            pin.capture_launch().unwrap();
            assert_eq!(GEMM_CALLS.load(Ordering::SeqCst), 2);
            assert_eq!(LAST_STREAM.load(Ordering::SeqCst), 0x77);
            assert_eq!(LAST_OUTPUT.load(Ordering::SeqCst), 0x2000);
            pin.replay(GraphHandle(0x1234)).unwrap();
            pin.destroy(GraphHandle(0x1234)).unwrap();
        }
        prepared.close().unwrap();
        assert_eq!(gpu.alloc_count(), 0);
    }
}
