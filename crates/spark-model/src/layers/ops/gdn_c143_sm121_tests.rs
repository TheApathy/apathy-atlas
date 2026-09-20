// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::Mutex;
use std::sync::atomic::{AtomicI32, AtomicIsize, AtomicU32, AtomicU64, AtomicUsize, Ordering};

use spark_runtime::gpu::mock::MockGpuBackend;

use super::*;

static TEST_LOCK: Mutex<()> = Mutex::new(());
static WORKSPACE_DRIFT: AtomicIsize = AtomicIsize::new(0);
static LAUNCH_STATUS: AtomicI32 = AtomicI32::new(0);
static LAUNCH_CALLS: AtomicUsize = AtomicUsize::new(0);
static LAST_STREAM: AtomicU64 = AtomicU64::new(0);
static LAST_M: AtomicU32 = AtomicU32::new(0);
static LAST_WORKSPACE_BYTES: AtomicUsize = AtomicUsize::new(0);
static LAST_Q_OFFSET: AtomicUsize = AtomicUsize::new(usize::MAX);
static LAST_K_OFFSET: AtomicUsize = AtomicUsize::new(usize::MAX);
static LAST_V_OFFSET: AtomicUsize = AtomicUsize::new(usize::MAX);
static LAST_Q_STRIDE: AtomicU32 = AtomicU32::new(0);
static LAST_GATE_STRIDE: AtomicU32 = AtomicU32::new(0);

const ABI_IDENTITY: &[u8] = b"atlas-gdn-c143-abi-v3:qwen38-c1-b1-workspace-v3:55d0b8db19312dcb25b96f619b5ccdcdf2e72d209df660d80f1cb59889e81940\0";
const BAD_ABI_IDENTITY: &[u8] = b"atlas-gdn-c143-abi-v2:wrong\0";
const ERROR_TEXT: &[u8] = b"forced v3 failure\0";

unsafe extern "C" fn fake_identity() -> *const c_char {
    ABI_IDENTITY.as_ptr().cast()
}

unsafe extern "C" fn bad_identity() -> *const c_char {
    BAD_ABI_IDENTITY.as_ptr().cast()
}

unsafe extern "C" fn null_identity() -> *const c_char {
    std::ptr::null()
}

unsafe extern "C" fn fake_workspace(m: c_uint) -> usize {
    let expected: usize = match m {
        2_079 => 130_565_952,
        8_192 => 506_462_208,
        _ => 0,
    };
    expected.saturating_add_signed(WORKSPACE_DRIFT.load(Ordering::SeqCst))
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn fake_launch(
    _state: *mut f32,
    _qkv: *const c_void,
    q_offset: usize,
    k_offset: usize,
    v_offset: usize,
    q_stride: c_uint,
    _k_stride: c_uint,
    _v_stride: c_uint,
    _gate_beta: *const f32,
    gate_stride: c_uint,
    _output: *mut c_void,
    _workspace: *mut c_void,
    workspace_bytes: usize,
    m: c_uint,
    stream: *mut c_void,
) -> c_int {
    LAUNCH_CALLS.fetch_add(1, Ordering::SeqCst);
    LAST_STREAM.store(stream as usize as u64, Ordering::SeqCst);
    LAST_M.store(m, Ordering::SeqCst);
    LAST_WORKSPACE_BYTES.store(workspace_bytes, Ordering::SeqCst);
    LAST_Q_OFFSET.store(q_offset, Ordering::SeqCst);
    LAST_K_OFFSET.store(k_offset, Ordering::SeqCst);
    LAST_V_OFFSET.store(v_offset, Ordering::SeqCst);
    LAST_Q_STRIDE.store(q_stride, Ordering::SeqCst);
    LAST_GATE_STRIDE.store(gate_stride, Ordering::SeqCst);
    LAUNCH_STATUS.load(Ordering::SeqCst)
}

unsafe extern "C" fn fake_last_error() -> *const c_char {
    ERROR_TEXT.as_ptr().cast()
}

unsafe extern "C" fn null_last_error() -> *const c_char {
    std::ptr::null()
}

fn reset() {
    WORKSPACE_DRIFT.store(0, Ordering::SeqCst);
    LAUNCH_STATUS.store(0, Ordering::SeqCst);
    LAUNCH_CALLS.store(0, Ordering::SeqCst);
    LAST_STREAM.store(0, Ordering::SeqCst);
    LAST_M.store(0, Ordering::SeqCst);
    LAST_WORKSPACE_BYTES.store(0, Ordering::SeqCst);
    LAST_Q_OFFSET.store(usize::MAX, Ordering::SeqCst);
    LAST_K_OFFSET.store(usize::MAX, Ordering::SeqCst);
    LAST_V_OFFSET.store(usize::MAX, Ordering::SeqCst);
    LAST_Q_STRIDE.store(0, Ordering::SeqCst);
    LAST_GATE_STRIDE.store(0, Ordering::SeqCst);
}

fn library() -> GdnC143Sm121 {
    GdnC143Sm121::from_test_api(fake_identity, fake_workspace, fake_launch, fake_last_error)
        .unwrap()
}

fn valid_buffers(workspace_bytes: usize) -> GdnC143Buffers {
    GdnC143Buffers {
        state_fp32: DevicePtr(0x0010_0000),
        qkv_bf16: DevicePtr(0x0100_0000),
        gate_beta_fp32: DevicePtr(0x0400_0000),
        output_bf16: DevicePtr(0x0500_0000),
        workspace: DevicePtr(0x0800_0000),
        workspace_bytes,
    }
}

#[test]
fn exact_hash_and_standard_sha_vectors_are_frozen() {
    assert_eq!(hex_sha256(GDN_C143_SM121_SHA256), GDN_C143_SM121_SHA256_HEX);
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
fn hostile_paths_and_hashes_fail_before_dlopen() {
    assert!(GdnC143Sm121::open_exact(Path::new("relative.so")).is_err());
    assert!(GdnC143Sm121::open_exact(Path::new("/definitely/missing/gdn.so")).is_err());

    let path = std::env::temp_dir().join(format!("atlas-gdn-c143-test-{}", std::process::id()));
    let _ = std::fs::remove_file(&path);
    std::fs::write(&path, b"not an ELF").unwrap();
    let error = GdnC143Sm121::open_exact(&path).err().unwrap();
    assert!(error.to_string().contains("SHA-256 mismatch"));
    std::fs::remove_file(&path).unwrap();

    let directory = std::env::temp_dir();
    assert!(GdnC143Sm121::open_exact(&directory).is_err());
}

#[test]
fn sealed_copy_is_exact_and_write_grow_shrink_seal_locked() {
    let original = b"immutable GDN c143 bytes";
    let mut file = sealed_memfd(original).unwrap();
    assert_eq!(sha256_file(&mut file).unwrap(), sha256(original));
    let seals = unsafe { fcntl(file.as_raw_fd(), F_GET_SEALS) };
    assert_eq!(seals & REQUIRED_MEMFD_SEALS, REQUIRED_MEMFD_SEALS);
    file.seek(SeekFrom::Start(0)).unwrap();
    assert!(file.write_all(b"mutation").is_err());
    assert!(file.set_len(1).is_err());
    assert!(file.set_len(1_000).is_err());
    assert_eq!(sha256_file(&mut file).unwrap(), sha256(original));
}

#[test]
fn admission_rejects_identity_last_error_and_workspace_drift() {
    let _guard = TEST_LOCK.lock().unwrap();
    reset();
    assert!(
        GdnC143Sm121::from_test_api(bad_identity, fake_workspace, fake_launch, fake_last_error)
            .is_err()
    );
    assert!(
        GdnC143Sm121::from_test_api(null_identity, fake_workspace, fake_launch, fake_last_error)
            .is_err()
    );
    assert!(
        GdnC143Sm121::from_test_api(fake_identity, fake_workspace, fake_launch, null_last_error)
            .is_err()
    );
    WORKSPACE_DRIFT.store(1, Ordering::SeqCst);
    let error =
        GdnC143Sm121::from_test_api(fake_identity, fake_workspace, fake_launch, fake_last_error)
            .err()
            .unwrap();
    assert!(error.to_string().contains("workspace query drift"));
}

#[test]
fn exact_shapes_stream_and_capacity_fail_closed_without_allocation() {
    let _guard = TEST_LOCK.lock().unwrap();
    reset();
    assert!(GdnC143Shape::qwen38(0).is_err());
    assert!(GdnC143Shape::qwen38(2_078).is_err());
    assert!(GdnC143Shape::qwen38(2_080).is_err());
    assert!(GdnC143Shape::qwen38(8_191).is_err());
    assert!(GdnC143Shape::qwen38(8_193).is_err());
    assert_eq!(GdnC143Shape::qwen38(2_079).unwrap(), GdnC143Shape::m2079());
    assert_eq!(GdnC143Shape::qwen38(8_192).unwrap(), GdnC143Shape::m8192());

    let gpu = MockGpuBackend::new();
    let shape = GdnC143Shape::m2079();
    assert!(
        library()
            .prepare_borrowed(&gpu, shape, 0, 136_249_344)
            .is_err()
    );
    assert!(
        library()
            .prepare_borrowed(&gpu, shape, 0x88, shape.required_workspace_bytes - 1)
            .is_err()
    );
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn launch_freezes_exact_c1_layout_stream_workspace_and_status() {
    let _guard = TEST_LOCK.lock().unwrap();
    reset();
    let gpu = MockGpuBackend::new();
    let shape = GdnC143Shape::m2079();
    let capacity = 136_249_344;
    let mut prepared = library()
        .prepare_borrowed(&gpu, shape, 0x88, capacity)
        .unwrap();
    assert_eq!(prepared.shape(), shape);
    assert_eq!(prepared.stream(), 0x88);
    assert_eq!(prepared.required_workspace_bytes(), 130_565_952);
    assert!(
        prepared
            .launch_eager(valid_buffers(capacity), 0x99)
            .is_err()
    );
    assert_eq!(LAUNCH_CALLS.load(Ordering::SeqCst), 0);

    let mut short = valid_buffers(capacity);
    short.workspace_bytes -= 1;
    assert!(prepared.validate_buffers(short, 0x88).is_err());
    assert!(prepared.launch_eager(short, 0x88).is_err());
    let mut null = valid_buffers(capacity);
    null.output_bf16 = DevicePtr::NULL;
    assert!(prepared.launch_eager(null, 0x88).is_err());
    let mut misaligned = valid_buffers(capacity);
    misaligned.gate_beta_fp32 = DevicePtr(0x0400_0004);
    assert!(prepared.launch_eager(misaligned, 0x88).is_err());
    let mut overlapping = valid_buffers(capacity);
    overlapping.output_bf16 = overlapping.qkv_bf16;
    assert!(prepared.launch_eager(overlapping, 0x88).is_err());
    assert_eq!(LAUNCH_CALLS.load(Ordering::SeqCst), 0);

    prepared
        .validate_buffers(valid_buffers(capacity), 0x88)
        .unwrap();
    assert_eq!(LAUNCH_CALLS.load(Ordering::SeqCst), 0);

    prepared
        .launch_eager(valid_buffers(capacity), 0x88)
        .unwrap();
    assert_eq!(LAUNCH_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(LAST_STREAM.load(Ordering::SeqCst), 0x88);
    assert_eq!(LAST_M.load(Ordering::SeqCst), 2_079);
    assert_eq!(LAST_WORKSPACE_BYTES.load(Ordering::SeqCst), capacity);
    assert_eq!(LAST_Q_OFFSET.load(Ordering::SeqCst), 0);
    assert_eq!(LAST_K_OFFSET.load(Ordering::SeqCst), 2_048);
    assert_eq!(LAST_V_OFFSET.load(Ordering::SeqCst), 4_096);
    assert_eq!(LAST_Q_STRIDE.load(Ordering::SeqCst), 10_240);
    assert_eq!(LAST_GATE_STRIDE.load(Ordering::SeqCst), 96);

    LAUNCH_STATUS.store(-7, Ordering::SeqCst);
    let error = prepared
        .launch_eager(valid_buffers(capacity), 0x88)
        .unwrap_err();
    assert!(error.to_string().contains("status -7"));
    assert!(error.to_string().contains("forced v3 failure"));
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn source_contract_has_no_device_allocation_sync_or_graph_surface() {
    let source = include_str!("gdn_c143_sm121.rs");
    for forbidden in [
        ".alloc(",
        ".free(",
        ".synchronize(",
        "cudaMalloc",
        "cudaFree",
        "cudaDeviceSynchronize",
        "cudaStreamSynchronize",
        "GraphHandle",
        "capture_",
    ] {
        assert!(
            !source.contains(forbidden),
            "forbidden boundary token: {forbidden}"
        );
    }
    assert!(source.contains("atlas_gdn_c143_workspace_size_v3\\0"));
    assert!(source.contains("atlas_gdn_c143_launch_v3\\0"));
    assert!(source.contains("atlas_gdn_c143_last_error\\0"));
}
