// SPDX-License-Identifier: AGPL-3.0-only

use super::payload::{FileIdentity, OpenShard};
use super::*;
use crate::gpu::mock::MockGpuBackend;
use crate::gpu::{DevicePtr, GpuBackend, KernelHandle};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

#[path = "gguf_device_test_sha256.rs"]
mod test_source_sha256;

static NEXT_FILE: AtomicU64 = AtomicU64::new(0);
const DEVICE_SOURCE: &str = include_str!("gguf/device.rs");
const DEVICE_SOURCE_SHA256: &str =
    "330732f88aef8a74bb9e0fc2c215980b68ca6fbc12aced9415cfe65683c57bda";
const SHA256_SOURCE: &str = include_str!("gguf/sha256.rs");
const SHA256_SOURCE_SHA256: &str =
    "bca28dccb9f28c30455190a9b9582ba0fb3b86430f9a3b76f193fb3db57c058a";

fn source_sha256(source: &str) -> String {
    let (digest, bytes) = super::sha256::HashingReader::new(source.as_bytes())
        .finish()
        .unwrap();
    assert_eq!(bytes, u64::try_from(source.len()).unwrap());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn device_source_contract(source: &str) -> bool {
    source_sha256(source) == DEVICE_SOURCE_SHA256
}

fn sha256_source_contract(source: &str) -> bool {
    test_source_sha256::matches(source, SHA256_SOURCE_SHA256)
}

#[test]
fn device_source_hash_reader_covers_sha256_padding_boundary() {
    assert_eq!(
        source_sha256(std::str::from_utf8(&[b'a'; 55]).unwrap()),
        "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"
    );
    assert_eq!(
        source_sha256(std::str::from_utf8(&[b'a'; 56]).unwrap()),
        "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"
    );
    assert!(sha256_source_contract(SHA256_SOURCE));

    let conditional = SHA256_SOURCE.replacen(
        "        if self.buffered > 56 {",
        "        #[cfg(test)]\n        let needs_extra_block = self.buffered > 56;\n        #[cfg(not(test))]\n        let needs_extra_block = self.buffered >= 56;\n        if needs_extra_block {",
        1,
    );
    assert_ne!(conditional, SHA256_SOURCE);
    assert!(!sha256_source_contract(&conditional));
}

struct FailingFreeGpu {
    inner: MockGpuBackend,
    fail_once: Mutex<BTreeSet<u64>>,
    fail_next_frees: Mutex<usize>,
    fail_copy_call: Mutex<Option<usize>>,
    copy_calls: Mutex<usize>,
    free_attempts: Mutex<Vec<u64>>,
}

impl FailingFreeGpu {
    fn new() -> Self {
        Self {
            inner: MockGpuBackend::new(),
            fail_once: Mutex::new(BTreeSet::new()),
            fail_next_frees: Mutex::new(0),
            fail_copy_call: Mutex::new(None),
            copy_calls: Mutex::new(0),
            free_attempts: Mutex::new(Vec::new()),
        }
    }

    fn fail_next_free(&self, ptr: DevicePtr) {
        self.fail_once.lock().unwrap().insert(ptr.0);
    }

    fn fail_next_frees(&self, count: usize) {
        *self.fail_next_frees.lock().unwrap() = count;
    }

    fn fail_copy_call(&self, call: usize) {
        *self.fail_copy_call.lock().unwrap() = Some(call);
    }

    fn free_attempts(&self) -> Vec<u64> {
        self.free_attempts.lock().unwrap().clone()
    }
}

impl GpuBackend for FailingFreeGpu {
    fn alloc(&self, bytes: usize) -> anyhow::Result<DevicePtr> {
        self.inner.alloc(bytes)
    }

    fn alloc_managed(&self, bytes: usize) -> anyhow::Result<DevicePtr> {
        self.inner.alloc_managed(bytes)
    }

    fn free(&self, ptr: DevicePtr) -> anyhow::Result<()> {
        self.free_attempts.lock().unwrap().push(ptr.0);
        let fail_counted = {
            let mut remaining = self.fail_next_frees.lock().unwrap();
            if *remaining == 0 {
                false
            } else {
                *remaining -= 1;
                true
            }
        };
        if fail_counted || self.fail_once.lock().unwrap().remove(&ptr.0) {
            anyhow::bail!("injected free failure for {ptr}");
        }
        self.inner.free(ptr)
    }

    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> anyhow::Result<()> {
        let call = {
            let mut calls = self.copy_calls.lock().unwrap();
            *calls += 1;
            *calls
        };
        if *self.fail_copy_call.lock().unwrap() == Some(call) {
            anyhow::bail!("injected copy failure on call {call}");
        }
        self.inner.copy_h2d(src, dst)
    }

    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> anyhow::Result<()> {
        self.inner.copy_d2h(src, dst)
    }

    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> anyhow::Result<()> {
        self.inner.copy_d2d(src, dst, bytes)
    }

    fn launch(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        params: &mut [*mut std::ffi::c_void],
    ) -> anyhow::Result<()> {
        self.inner
            .launch(func, grid, block, shared_mem, stream, params)
    }

    fn synchronize(&self, stream: u64) -> anyhow::Result<()> {
        self.inner.synchronize(stream)
    }

    fn default_stream(&self) -> u64 {
        self.inner.default_stream()
    }

    fn kernel(&self, module: &str, func_name: &str) -> anyhow::Result<KernelHandle> {
        self.inner.kernel(module, func_name)
    }

    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> anyhow::Result<()> {
        self.inner.memset(ptr, value, bytes)
    }

    fn memset_async(
        &self,
        ptr: DevicePtr,
        value: u8,
        bytes: usize,
        stream: u64,
    ) -> anyhow::Result<()> {
        self.inner.memset_async(ptr, value, bytes, stream)
    }

    fn total_memory(&self) -> anyhow::Result<usize> {
        self.inner.total_memory()
    }

    fn free_memory(&self) -> anyhow::Result<usize> {
        self.inner.free_memory()
    }
}

fn fixture(second_len: u64) -> Glm53Iq3Files {
    let path = std::env::temp_dir().join(format!(
        "atlas-glm53-device-{}-{}",
        std::process::id(),
        NEXT_FILE.fetch_add(1, Ordering::Relaxed)
    ));
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    std::fs::remove_file(path).unwrap();
    file.write_all(&[0xaa; 16]).unwrap();
    file.write_all(b"device!!second!!").unwrap();
    file.flush().unwrap();
    let identity = FileIdentity::capture(&file).unwrap();
    let tensor = |name: &str, offset, byte_len| LocatedTensor {
        shard_no: 0,
        info: GgufTensorInfo {
            name: name.into(),
            dimensions: vec![byte_len],
            ggml_type: GgmlType::I8,
            offset,
            byte_len,
        },
    };
    Glm53Iq3Files::new_with_profile(
        Glm53QuantProfile::UdIq3Xxs,
        vec![OpenShard {
            split_no: 0,
            data_offset: 16,
            file,
            identity,
        }],
        GgufDirectory {
            architecture: "glm5next".into(),
            split_count: 1,
            tensors: BTreeMap::from([
                ("a".into(), tensor("a", 0, 8)),
                ("b".into(), tensor("b", 8, second_len)),
            ]),
        },
        Glm53Iq3Summary {
            shards: 1,
            tensors: 2,
            tensor_bytes: 8 + second_len,
        },
    )
    .unwrap()
}

#[test]
fn copies_exact_tensor_and_frees_on_identity_failure() {
    let gpu = MockGpuBackend::new();
    let mut files = fixture(8);
    let tensor = load_glm53_iq3_tensor(&mut files, "a", &gpu).unwrap();
    assert_eq!(gpu.read_alloc(tensor.ptr).unwrap(), b"device!!");
    gpu.free(tensor.ptr).unwrap();

    files.shards[0].file.set_len(17).unwrap();
    assert!(load_glm53_iq3_tensor(&mut files, "a", &gpu).is_err());
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn whole_store_is_exact_and_explicitly_freed() {
    let gpu = MockGpuBackend::new();
    let mut files = fixture(8);
    let store = load_glm53_iq3_store(&mut files, &gpu, 1024).unwrap();
    assert_eq!(store.len(), 2);
    assert_eq!(store.names().collect::<Vec<_>>(), ["a", "b"]);
    assert_eq!(store.total_bytes(), 16);
    assert_eq!(
        gpu.read_alloc(store.get("a").unwrap().ptr).unwrap(),
        b"device!!"
    );
    assert_eq!(
        gpu.read_alloc(store.get("b").unwrap().ptr).unwrap(),
        b"second!!"
    );
    store.free(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn failed_free_retains_only_live_allocations_and_retry_releases_them() {
    let gpu = FailingFreeGpu::new();
    let mut files = fixture(8);
    let store = load_glm53_iq3_store(&mut files, &gpu, 0).unwrap();
    let a = store.get("a").unwrap().ptr;
    let b = store.get("b").unwrap().ptr;
    gpu.fail_next_free(a);

    let failure = store.free(&gpu).unwrap_err();
    assert_eq!(gpu.free_attempts(), vec![a.0, b.0]);
    assert_eq!(gpu.inner.alloc_count(), 1);
    assert_eq!(failure.failed_tensor_count(), 1);
    assert_eq!(failure.store().names().collect::<Vec<_>>(), ["a"]);
    assert_eq!(failure.store().total_bytes(), 8);
    assert!(failure.to_string().contains("1 GGUF device tensors"));
    assert!(failure.to_string().contains("8 bytes retained"));
    assert!(
        failure
            .failure()
            .to_string()
            .contains("injected free failure")
    );
    assert!(gpu.inner.read_alloc(a).is_some());
    assert!(gpu.inner.read_alloc(b).is_none());

    gpu.fail_next_free(a);
    let failure = failure.retry(&gpu).unwrap_err();
    assert_eq!(gpu.free_attempts(), vec![a.0, b.0, a.0]);
    assert_eq!(failure.failed_tensor_count(), 1);
    assert_eq!(failure.store().names().collect::<Vec<_>>(), ["a"]);
    assert_eq!(failure.store().total_bytes(), 8);
    assert_eq!(gpu.inner.alloc_count(), 1);

    failure.retry(&gpu).unwrap();
    assert_eq!(gpu.free_attempts(), vec![a.0, b.0, a.0, a.0]);
    assert_eq!(gpu.inner.alloc_count(), 0);
}

#[test]
fn free_api_returns_a_non_generic_retry_owner() {
    let _: fn(
        GgufDeviceStore,
        &dyn GpuBackend,
    ) -> std::result::Result<(), GgufDeviceStoreFreeError> = GgufDeviceStore::free;
    assert!(device_source_contract(DEVICE_SOURCE));
    assert!(!DEVICE_SOURCE.contains("impl std::error::Error for GgufDeviceStoreFreeError"));
    assert!(DEVICE_SOURCE.contains("failed device frees retain allocations"));
    assert!(DEVICE_SOURCE.contains("Some(first_error) => Err(GgufDeviceStoreFreeError"));
}

#[test]
fn copy_failure_with_failed_free_retains_and_retries_the_new_tensor() {
    let gpu = FailingFreeGpu::new();
    gpu.fail_copy_call(1);
    gpu.fail_next_frees(1);
    let mut files = fixture(8);

    let failure = load_glm53_iq3_tensor(&mut files, "a", &gpu).unwrap_err();
    assert!(
        failure
            .failure()
            .to_string()
            .contains("injected copy failure")
    );
    assert_eq!(failure.cleanup_failure_count(), 1);
    assert_eq!(failure.retained_tensor_count(), 1);
    assert_eq!(failure.retained_bytes(), 8);
    let retained = failure.cleanup_failure(0).unwrap().store();
    assert_eq!(retained.names().collect::<Vec<_>>(), ["a"]);
    assert_eq!(gpu.inner.alloc_count(), 1);

    let primary = failure.retry_cleanup(&gpu).unwrap();
    assert!(primary.to_string().contains("injected copy failure"));
    assert_eq!(gpu.inner.alloc_count(), 0);
}

#[test]
fn partial_store_failure_retries_every_independent_cleanup_owner() {
    let gpu = FailingFreeGpu::new();
    gpu.fail_next_frees(2);
    let mut invalid_second_range = fixture(9);

    let failure = load_glm53_iq3_store(&mut invalid_second_range, &gpu, 0)
        .err()
        .expect("invalid second range must retain its typed load failure");
    assert_eq!(failure.cleanup_failure_count(), 2);
    assert_eq!(failure.retained_tensor_count(), 2);
    assert_eq!(failure.retained_bytes(), 17);
    assert_eq!(gpu.inner.alloc_count(), 2);
    assert_eq!(gpu.free_attempts().len(), 2);

    let second = failure
        .cleanup_failure(0)
        .unwrap()
        .store()
        .get("b")
        .unwrap()
        .ptr;
    gpu.fail_next_free(second);
    let failure = failure.retry_cleanup(&gpu).unwrap_err();
    assert_eq!(failure.cleanup_failure_count(), 1);
    assert_eq!(failure.retained_tensor_count(), 1);
    assert_eq!(failure.retained_bytes(), 9);
    assert_eq!(
        failure
            .cleanup_failure(0)
            .unwrap()
            .store()
            .names()
            .collect::<Vec<_>>(),
        ["b"]
    );
    assert_eq!(gpu.free_attempts().len(), 4);
    assert_eq!(gpu.inner.alloc_count(), 1);

    let primary = failure.retry_cleanup(&gpu).unwrap();
    assert!(
        primary
            .to_string()
            .contains("GGUF tensor exceeds its admitted shard")
    );
    assert_eq!(gpu.free_attempts().len(), 5);
    assert_eq!(gpu.inner.alloc_count(), 0);
}

#[test]
fn load_api_returns_a_non_generic_multi_owner_failure() {
    let _: fn(
        &mut Glm53GgufFiles,
        &str,
        &dyn GpuBackend,
    ) -> std::result::Result<GgufDeviceTensor, GgufDeviceLoadError> = load_glm53_tensor;
    let _: fn(
        &mut Glm53GgufFiles,
        &dyn GpuBackend,
        usize,
    ) -> std::result::Result<GgufDeviceStore, GgufDeviceLoadError> = load_glm53_store;
    assert!(device_source_contract(DEVICE_SOURCE));
    assert!(!DEVICE_SOURCE.contains("impl std::error::Error for GgufDeviceLoadError"));
    assert!(DEVICE_SOURCE.contains("cleanup_failures: Vec<GgufDeviceStoreFreeError>"));
    assert!(DEVICE_SOURCE.contains("for cleanup in std::mem::take(&mut self.cleanup_failures)"));
    assert!(!DEVICE_SOURCE.contains("std::mem::take(tensors)"));

    let no_must_use = DEVICE_SOURCE.replacen(
        "#[must_use = \"failed device loads may retain allocations and must be inspected\"]\n",
        "",
        1,
    );
    assert!(!device_source_contract(&no_must_use));
    for alias in ["GgufDeviceLoadError", "GgufDeviceStoreFreeError"] {
        let mutant = format!(
            "{DEVICE_SOURCE}\nuse std::error::Error as Erased;\nimpl Erased for {alias} {{}}\n"
        );
        assert!(!device_source_contract(&mutant));
    }
}

#[test]
fn preflight_and_mid_load_failures_leave_no_allocations() {
    let gpu = MockGpuBackend::new();
    let mut valid = fixture(8);
    assert!(load_glm53_iq3_store(&mut valid, &gpu, gpu.free_memory().unwrap()).is_err());
    assert!(load_glm53_iq3_store(&mut valid, &gpu, usize::MAX).is_err());
    assert_eq!(gpu.alloc_count(), 0);

    let mut invalid_second_range = fixture(9);
    assert!(load_glm53_iq3_store(&mut invalid_second_range, &gpu, 0).is_err());
    assert_eq!(gpu.alloc_count(), 0);
}
