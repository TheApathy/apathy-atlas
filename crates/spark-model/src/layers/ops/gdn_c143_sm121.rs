// SPDX-License-Identifier: AGPL-3.0-only

//! Sealed, default-unrouted Rust boundary for the exact GDN c143 SM121 v3 ABI.
//!
//! This module owns no device memory and exposes no graph API. A caller first
//! opens the one frozen library identity, then prepares one of the two exact
//! Qwen3.8 C1 shapes against a non-default stream and a caller-owned workspace
//! capacity. Launches borrow that same prepared object mutably, rebind the
//! Atlas backend context, require the same stream and workspace capacity, and
//! enqueue only the native v3 operation. The caller must retain every device
//! buffer until that stream has completed.
//!
//! Production may declare this module, but the native library is not opened
//! and no launch is possible unless the separate strict SSM selector is set.

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const RTLD_NOW: c_int = 2;
const MAX_LIBRARY_BYTES: u64 = 8 * 1024 * 1024;
const MFD_CLOEXEC: c_uint = 0x0001;
const MFD_ALLOW_SEALING: c_uint = 0x0002;
const F_ADD_SEALS: c_int = 1_033;
const F_GET_SEALS: c_int = 1_034;
const F_SEAL_SEAL: c_int = 0x0001;
const F_SEAL_SHRINK: c_int = 0x0002;
const F_SEAL_GROW: c_int = 0x0004;
const F_SEAL_WRITE: c_int = 0x0008;
const REQUIRED_MEMFD_SEALS: c_int = F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE;

pub const GDN_C143_SM121_SHA256: [u8; 32] = [
    0x27, 0x2c, 0xe1, 0xb1, 0x15, 0xdc, 0x16, 0xea, 0xef, 0x22, 0xbc, 0x2c, 0xbe, 0x5d, 0x1e, 0x82,
    0xec, 0x48, 0x89, 0x78, 0x24, 0x75, 0xf4, 0xc6, 0x85, 0x67, 0xea, 0xb8, 0x43, 0x5c, 0xd3, 0xf0,
];
pub const GDN_C143_SM121_SHA256_HEX: &str =
    "272ce1b115dc16eaef22bc2cbe5d1e82ec4889782475f4c68567eab8435cd3f0";
const GDN_C143_ABI_IDENTITY: &str = concat!(
    "atlas-gdn-c143-abi-v3:qwen38-c1-b1-workspace-v3:",
    "55d0b8db19312dcb25b96f619b5ccdcdf2e72d209df660d80f1cb59889e81940"
);

const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 48;
const HEAD_DIM: usize = 128;
const Q_WIDTH: usize = KEY_HEADS * HEAD_DIM;
const K_WIDTH: usize = KEY_HEADS * HEAD_DIM;
const V_WIDTH: usize = VALUE_HEADS * HEAD_DIM;
const QKV_ROW_STRIDE: usize = Q_WIDTH + K_WIDTH + V_WIDTH;
const GATE_BETA_ROW_STRIDE: usize = VALUE_HEADS * 2;
const STATE_BYTES: usize = VALUE_HEADS * HEAD_DIM * HEAD_DIM * 4;

type IdentityFn = unsafe extern "C" fn() -> *const c_char;
type WorkspaceFn = unsafe extern "C" fn(c_uint) -> usize;
type LaunchFn = unsafe extern "C" fn(
    *mut f32,
    *const c_void,
    usize,
    usize,
    usize,
    c_uint,
    c_uint,
    c_uint,
    *const f32,
    c_uint,
    *mut c_void,
    *mut c_void,
    usize,
    c_uint,
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
    identity: IdentityFn,
    workspace: WorkspaceFn,
    launch: LaunchFn,
    last_error: LastErrorFn,
    path: PathBuf,
    _sealed_library: Option<File>,
    sha256: [u8; 32],
    source_device: u64,
    source_inode: u64,
}

// The exact sealed library is immutable, its workspace query is pure, and its
// diagnostic buffer is thread-local. Launch serialization for one caller arena
// is expressed by the mutable prepared-object borrow.
unsafe impl Send for RawApi {}
unsafe impl Sync for RawApi {}

impl Drop for RawApi {
    fn drop(&mut self) {
        if self.close_handle && !self.handle.is_null() {
            let _ = unsafe { dlclose(self.handle) };
        }
    }
}

/// Frozen handle to the one statically qualified GDN c143 v3 shared object.
#[derive(Clone)]
pub struct GdnC143Sm121 {
    api: Arc<RawApi>,
}

impl GdnC143Sm121 {
    /// Admit only the exact qualified shared-object bytes from an absolute
    /// regular-file path, copy them into a fully sealed memfd, rehash that copy,
    /// and resolve the ABI through `/proc/self/fd/N`.
    pub fn open_exact(path: &Path) -> Result<Self> {
        Self::open_with_expected_sha256(path, GDN_C143_SM121_SHA256)
    }

    fn open_with_expected_sha256(path: &Path, expected_sha256: [u8; 32]) -> Result<Self> {
        ensure!(usize::BITS == 64, "GDN c143 ABI requires a 64-bit host");
        ensure!(path.is_absolute(), "GDN c143 library path must be absolute");
        let canonical = path
            .canonicalize()
            .with_context(|| format!("canonicalize GDN c143 library {}", path.display()))?;
        let mut source = File::open(&canonical)
            .with_context(|| format!("open GDN c143 library {}", canonical.display()))?;
        let before = source.metadata()?;
        ensure!(
            before.is_file(),
            "GDN c143 library must be a regular file: {}",
            canonical.display()
        );
        ensure!(
            before.len() > 0 && before.len() <= MAX_LIBRARY_BYTES,
            "GDN c143 library size {} is outside 1..={MAX_LIBRARY_BYTES} bytes",
            before.len()
        );

        let mut bytes = Vec::new();
        (&mut source)
            .take(MAX_LIBRARY_BYTES + 1)
            .read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() as u64 == before.len() && bytes.len() as u64 <= MAX_LIBRARY_BYTES,
            "GDN c143 source library length changed while reading"
        );
        let admitted_sha256 = sha256(&bytes);
        ensure!(
            admitted_sha256 == expected_sha256,
            "GDN c143 library SHA-256 mismatch: expected {}, got {}",
            hex_sha256(expected_sha256),
            hex_sha256(admitted_sha256)
        );
        let after = source.metadata()?;
        ensure!(
            same_file_identity(&before, &after),
            "GDN c143 source library changed while it was being read: {}",
            canonical.display()
        );

        let mut sealed_library = sealed_memfd(&bytes)?;
        ensure!(
            sha256_file(&mut sealed_library)? == admitted_sha256,
            "sealed GDN c143 copy does not match admitted source SHA-256"
        );
        let descriptor_path = format!("/proc/self/fd/{}", sealed_library.as_raw_fd());
        let encoded = CString::new(descriptor_path.as_bytes())
            .context("GDN c143 descriptor path contains a NUL byte")?;
        clear_dlerror();
        let handle = unsafe { dlopen(encoded.as_ptr(), RTLD_NOW) };
        ensure!(
            !handle.is_null(),
            "dlopen admitted descriptor for {} failed: {}",
            canonical.display(),
            current_dlerror()
        );

        let loaded = (|| -> Result<(IdentityFn, WorkspaceFn, LaunchFn, LastErrorFn)> {
            let identity = unsafe {
                std::mem::transmute::<*mut c_void, IdentityFn>(resolve_symbol(
                    handle,
                    b"atlas_gdn_c143_abi_identity\0",
                )?)
            };
            let workspace = unsafe {
                std::mem::transmute::<*mut c_void, WorkspaceFn>(resolve_symbol(
                    handle,
                    b"atlas_gdn_c143_workspace_size_v3\0",
                )?)
            };
            let launch = unsafe {
                std::mem::transmute::<*mut c_void, LaunchFn>(resolve_symbol(
                    handle,
                    b"atlas_gdn_c143_launch_v3\0",
                )?)
            };
            let last_error = unsafe {
                std::mem::transmute::<*mut c_void, LastErrorFn>(resolve_symbol(
                    handle,
                    b"atlas_gdn_c143_last_error\0",
                )?)
            };
            Ok((identity, workspace, launch, last_error))
        })();
        let (identity, workspace, launch, last_error) = match loaded {
            Ok(symbols) => symbols,
            Err(error) => {
                let _ = unsafe { dlclose(handle) };
                return Err(error)
                    .with_context(|| format!("load GDN c143 ABI from {}", canonical.display()));
            }
        };

        let library = Self {
            api: Arc::new(RawApi {
                handle,
                close_handle: true,
                identity,
                workspace,
                launch,
                last_error,
                path: canonical,
                _sealed_library: Some(sealed_library),
                sha256: admitted_sha256,
                source_device: before.dev(),
                source_inode: before.ino(),
            }),
        };
        library.validate_loaded_api()?;
        Ok(library)
    }

    fn validate_loaded_api(&self) -> Result<()> {
        let identity = unsafe { (self.api.identity)() };
        ensure!(!identity.is_null(), "GDN c143 ABI identity returned null");
        let identity = unsafe { CStr::from_ptr(identity) }
            .to_str()
            .context("GDN c143 ABI identity is not UTF-8")?;
        ensure!(
            identity == GDN_C143_ABI_IDENTITY,
            "GDN c143 ABI identity mismatch: expected {GDN_C143_ABI_IDENTITY}, got {identity}"
        );
        ensure!(
            !unsafe { (self.api.last_error)() }.is_null(),
            "GDN c143 last_error returned null during admission"
        );
        for shape in [GdnC143Shape::m2079(), GdnC143Shape::m8192()] {
            self.query_exact_workspace(shape)?;
        }
        Ok(())
    }

    fn query_exact_workspace(&self, shape: GdnC143Shape) -> Result<usize> {
        let actual = unsafe { (self.api.workspace)(shape.seq_len) };
        ensure!(
            actual == shape.required_workspace_bytes,
            "GDN c143 workspace query drift for M={}: expected {}, got {actual}",
            shape.seq_len,
            shape.required_workspace_bytes
        );
        Ok(actual)
    }

    /// Freeze an exact shape, non-default stream and caller-owned arena
    /// capacity. No allocation, synchronization, or native launch occurs.
    pub fn prepare_borrowed<'gpu>(
        &self,
        gpu: &'gpu dyn GpuBackend,
        shape: GdnC143Shape,
        stream: u64,
        workspace_capacity: usize,
    ) -> Result<PreparedGdnC143Sm121<'gpu>> {
        ensure!(stream != 0, "GDN c143 requires a non-default stream");
        let required_workspace_bytes = self.query_exact_workspace(shape)?;
        ensure!(
            workspace_capacity >= required_workspace_bytes,
            "GDN c143 workspace capacity {workspace_capacity} is below required {required_workspace_bytes}"
        );
        gpu.bind_to_thread()
            .context("bind Atlas CUDA context before GDN c143 preparation")?;
        Ok(PreparedGdnC143Sm121 {
            library: self.clone(),
            gpu,
            shape,
            stream,
            workspace_capacity,
            required_workspace_bytes,
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
        bail!("GDN c143 {operation} failed with status {status}: {detail}")
    }

    pub fn path(&self) -> &Path {
        &self.api.path
    }

    pub fn sha256_hex(&self) -> String {
        hex_sha256(self.api.sha256)
    }

    pub fn source_file_identity(&self) -> (u64, u64) {
        (self.api.source_device, self.api.source_inode)
    }

    #[cfg(test)]
    fn from_test_api(
        identity: IdentityFn,
        workspace: WorkspaceFn,
        launch: LaunchFn,
        last_error: LastErrorFn,
    ) -> Result<Self> {
        let library = Self {
            api: Arc::new(RawApi {
                handle: std::ptr::null_mut(),
                close_handle: false,
                identity,
                workspace,
                launch,
                last_error,
                path: PathBuf::from("<test-api>"),
                _sealed_library: None,
                sha256: GDN_C143_SM121_SHA256,
                source_device: 0,
                source_inode: 0,
            }),
        };
        library.validate_loaded_api()?;
        Ok(library)
    }
}

/// One of the two exact Qwen3.8 C1 production qualification shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnC143Shape {
    seq_len: c_uint,
    required_workspace_bytes: usize,
}

impl GdnC143Shape {
    pub fn qwen38(seq_len: usize) -> Result<Self> {
        match seq_len {
            2_079 => Ok(Self::m2079()),
            8_192 => Ok(Self::m8192()),
            _ => bail!("GDN c143 production route admits only M=2079 or M=8192"),
        }
    }

    const fn m2079() -> Self {
        Self {
            seq_len: 2_079,
            required_workspace_bytes: 130_565_952,
        }
    }

    const fn m8192() -> Self {
        Self {
            seq_len: 8_192,
            required_workspace_bytes: 506_462_208,
        }
    }

    pub fn seq_len(self) -> usize {
        self.seq_len as usize
    }

    pub fn required_workspace_bytes(self) -> usize {
        self.required_workspace_bytes
    }
}

/// Exact Atlas Qwen3.8 C1 device buffers for one B=1 v3 launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnC143Buffers {
    pub state_fp32: DevicePtr,
    pub qkv_bf16: DevicePtr,
    pub gate_beta_fp32: DevicePtr,
    pub output_bf16: DevicePtr,
    pub workspace: DevicePtr,
    pub workspace_bytes: usize,
}

/// Eager-only borrowed boundary. It owns no workspace and has no graph API.
pub struct PreparedGdnC143Sm121<'gpu> {
    library: GdnC143Sm121,
    gpu: &'gpu dyn GpuBackend,
    shape: GdnC143Shape,
    stream: u64,
    workspace_capacity: usize,
    required_workspace_bytes: usize,
}

impl PreparedGdnC143Sm121<'_> {
    pub fn shape(&self) -> GdnC143Shape {
        self.shape
    }

    pub fn stream(&self) -> u64 {
        self.stream
    }

    pub fn required_workspace_bytes(&self) -> usize {
        self.required_workspace_bytes
    }

    /// Validate the complete launch buffer contract without calling native
    /// code. Production uses this before any projection or recurrent-state
    /// mutation so an explicitly requested route fails closed atomically.
    pub fn validate_buffers(&self, buffers: GdnC143Buffers, stream: u64) -> Result<()> {
        ensure!(
            stream == self.stream && stream != 0,
            "GDN c143 stream mismatch: prepared {:#x}, validation {stream:#x}",
            self.stream
        );
        buffers.validate(self.shape, self.workspace_capacity)
    }

    /// Validate all exact extents and enqueue v3 on the frozen stream. The
    /// caller retains the workspace and all inputs/outputs through completion.
    pub fn launch_eager(&mut self, buffers: GdnC143Buffers, stream: u64) -> Result<()> {
        self.validate_buffers(buffers, stream)?;
        self.gpu
            .bind_to_thread()
            .context("bind Atlas CUDA context before GDN c143 launch")?;
        let status = unsafe {
            (self.library.api.launch)(
                device_mut(buffers.state_fp32).cast(),
                device_const(buffers.qkv_bf16),
                0,
                Q_WIDTH,
                Q_WIDTH + K_WIDTH,
                QKV_ROW_STRIDE as c_uint,
                QKV_ROW_STRIDE as c_uint,
                QKV_ROW_STRIDE as c_uint,
                device_const(buffers.gate_beta_fp32).cast(),
                GATE_BETA_ROW_STRIDE as c_uint,
                device_mut(buffers.output_bf16),
                device_mut(buffers.workspace),
                buffers.workspace_bytes,
                self.shape.seq_len,
                stream as usize as *mut c_void,
            )
        };
        self.library.check_status("launch_v3", status)
    }
}

impl GdnC143Buffers {
    fn validate(self, shape: GdnC143Shape, workspace_capacity: usize) -> Result<()> {
        ensure!(
            self.workspace_bytes == workspace_capacity,
            "GDN c143 launch workspace capacity changed: prepared {workspace_capacity}, launch {}",
            self.workspace_bytes
        );
        ensure!(
            self.workspace_bytes >= shape.required_workspace_bytes,
            "GDN c143 workspace is smaller than the exact shape requirement"
        );
        let m = shape.seq_len as usize;
        let spans = [
            device_span("state_fp32", self.state_fp32, STATE_BYTES)?,
            device_span(
                "qkv_bf16",
                self.qkv_bf16,
                checked_bytes(m, QKV_ROW_STRIDE, 2, "QKV")?,
            )?,
            device_span(
                "gate_beta_fp32",
                self.gate_beta_fp32,
                checked_bytes(m, GATE_BETA_ROW_STRIDE, 4, "gate/beta")?,
            )?,
            device_span(
                "output_bf16",
                self.output_bf16,
                checked_bytes(m, V_WIDTH, 2, "output")?,
            )?,
            device_span("workspace", self.workspace, self.workspace_bytes)?,
        ];
        for left in 0..spans.len() {
            for right in left + 1..spans.len() {
                ensure!(
                    spans[left].1 <= spans[right].0 || spans[right].1 <= spans[left].0,
                    "GDN c143 device ranges overlap"
                );
            }
        }
        Ok(())
    }
}

fn checked_bytes(rows: usize, columns: usize, element_bytes: usize, label: &str) -> Result<usize> {
    rows.checked_mul(columns)
        .and_then(|value| value.checked_mul(element_bytes))
        .with_context(|| format!("GDN c143 {label} byte extent overflow"))
}

fn device_span(label: &str, pointer: DevicePtr, bytes: usize) -> Result<(u64, u64)> {
    ensure!(!pointer.is_null(), "GDN c143 {label} pointer is null");
    ensure!(
        pointer.0.is_multiple_of(16),
        "GDN c143 {label} pointer is not 16-byte aligned"
    );
    let bytes = u64::try_from(bytes).context("GDN c143 device extent exceeds u64")?;
    let end = pointer
        .0
        .checked_add(bytes)
        .with_context(|| format!("GDN c143 {label} address overflow"))?;
    Ok((pointer.0, end))
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
    let name = CString::new("atlas-gdn-c143-sm121")?;
    let descriptor = unsafe { memfd_create(name.as_ptr(), MFD_CLOEXEC | MFD_ALLOW_SEALING) };
    ensure!(
        descriptor >= 0,
        "memfd_create for GDN c143 failed: {}",
        std::io::Error::last_os_error()
    );
    // SAFETY: memfd_create returned a fresh descriptor transferred exactly once.
    let mut file = unsafe { File::from_raw_fd(descriptor) };
    file.write_all(bytes)
        .context("write verified GDN c143 bytes into memfd")?;
    file.flush().context("flush verified GDN c143 memfd")?;
    file.seek(SeekFrom::Start(0))?;
    let status = unsafe { fcntl(file.as_raw_fd(), F_ADD_SEALS, REQUIRED_MEMFD_SEALS) };
    ensure!(
        status == 0,
        "seal GDN c143 memfd failed: {}",
        std::io::Error::last_os_error()
    );
    let seals = unsafe { fcntl(file.as_raw_fd(), F_GET_SEALS) };
    ensure!(
        seals >= 0 && seals & REQUIRED_MEMFD_SEALS == REQUIRED_MEMFD_SEALS,
        "GDN c143 memfd is missing required seals: expected {REQUIRED_MEMFD_SEALS:#x}, got {seals:#x}"
    );
    Ok(file)
}

fn sha256_file(file: &mut File) -> Result<[u8; 32]> {
    file.seek(SeekFrom::Start(0))?;
    let mut bytes = Vec::new();
    file.take(MAX_LIBRARY_BYTES + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_LIBRARY_BYTES,
        "sealed GDN c143 library exceeds {MAX_LIBRARY_BYTES} bytes"
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

// Dependency-free SHA-256 keeps this startup-only sealed boundary independent
// of additional server/runtime packages.
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
        let mut words = [0u32; 64];
        for (index, word) in chunk.chunks_exact(4).enumerate() {
            words[index] = u32::from_be_bytes(word.try_into().expect("four-byte SHA-256 word"));
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
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
                .wrapping_add(words[index]);
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
        for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
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
#[path = "gdn_c143_sm121_tests.rs"]
mod tests;
