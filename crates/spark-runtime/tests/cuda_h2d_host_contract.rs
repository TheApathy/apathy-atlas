// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only behavior and source contracts for the H2D host boundary.

use std::sync::Mutex;

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, HostToDeviceCopy, KernelHandle};

const GPU_TRAIT: &str = include_str!("../src/gpu.rs");
const GPU_IMPL: &str = include_str!("../src/cuda_backend/gpu_impl.rs");
const MODEL_TYPES: &str = include_str!("../../spark-model/src/model/types.rs");
const MODEL_INIT: &str = include_str!("../../spark-model/src/model/impl_a1.rs");
const MODEL_DROP: &str = include_str!("../../spark-model/src/model/drop.rs");
const PREFILL_BATCH_KERNEL: &str =
    include_str!("../../spark-model/src/model/trait_impl/prefill_b/batch_kernel.rs");
const PREFILL_UPLOAD_META: &str =
    include_str!("../../spark-model/src/model/trait_impl/prefill_b/upload_meta.rs");
const QWEN_SSM: &str = include_str!("../../spark-model/src/layers/qwen3_ssm/mod.rs");
const QWEN_SSM_INIT: &str = include_str!("../../spark-model/src/layers/qwen3_ssm/init.rs");

#[derive(Clone, Copy)]
enum Failure {
    None,
    Sync,
    Copy(usize),
}

struct RecordingGpu {
    events: Mutex<Vec<String>>,
    failure: Failure,
}

impl RecordingGpu {
    fn new(failure: Failure) -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            failure,
        }
    }

    fn events(&self) -> Vec<String> {
        self.events.lock().unwrap().clone()
    }
}

impl GpuBackend for RecordingGpu {
    fn alloc(&self, _bytes: usize) -> Result<DevicePtr> {
        bail!("unused")
    }

    fn alloc_managed(&self, _bytes: usize) -> Result<DevicePtr> {
        bail!("unused")
    }

    fn free(&self, _ptr: DevicePtr) -> Result<()> {
        bail!("unused")
    }

    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        let ordinal = self
            .events
            .lock()
            .unwrap()
            .iter()
            .filter(|event| event.starts_with("copy:"))
            .count()
            + 1;
        self.events
            .lock()
            .unwrap()
            .push(format!("copy:{}:{}", dst.0, src.len()));
        if matches!(self.failure, Failure::Copy(failed) if failed == ordinal) {
            bail!("copy {ordinal} failed")
        }
        Ok(())
    }

    fn copy_d2h(&self, _src: DevicePtr, _dst: &mut [u8]) -> Result<()> {
        bail!("unused")
    }

    fn copy_d2d(&self, _src: DevicePtr, _dst: DevicePtr, _bytes: usize) -> Result<()> {
        bail!("unused")
    }

    fn launch(
        &self,
        _func: KernelHandle,
        _grid: [u32; 3],
        _block: [u32; 3],
        _shared_mem: u32,
        _stream: u64,
        _params: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        bail!("unused")
    }

    fn synchronize(&self, stream: u64) -> Result<()> {
        self.events.lock().unwrap().push(format!("sync:{stream}"));
        if matches!(self.failure, Failure::Sync) {
            bail!("sync failed")
        }
        Ok(())
    }

    fn default_stream(&self) -> u64 {
        0
    }

    fn kernel(&self, _module: &str, _func_name: &str) -> Result<KernelHandle> {
        bail!("unused")
    }

    fn memset(&self, _ptr: DevicePtr, _value: u8, _bytes: usize) -> Result<()> {
        bail!("unused")
    }

    fn memset_async(&self, _ptr: DevicePtr, _value: u8, _bytes: usize, _stream: u64) -> Result<()> {
        bail!("unused")
    }

    fn total_memory(&self) -> Result<usize> {
        bail!("unused")
    }

    fn free_memory(&self) -> Result<usize> {
        bail!("unused")
    }
}

#[test]
fn pageable_group_is_one_sync_then_ordered_copies() {
    let gpu = RecordingGpu::new(Failure::None);
    let first = [1_u8; 3];
    let second = [2_u8; 5];
    let copies = [
        HostToDeviceCopy::new(&first, DevicePtr(17)),
        HostToDeviceCopy::new(&second, DevicePtr(29)),
    ];
    gpu.copy_h2d_group_on_stream(&copies, 41).unwrap();
    assert_eq!(gpu.events(), ["sync:41", "copy:17:3", "copy:29:5"]);
}

#[test]
fn pageable_group_fails_closed_in_operation_order() {
    let first = [1_u8; 3];
    let second = [2_u8; 5];
    let copies = [
        HostToDeviceCopy::new(&first, DevicePtr(17)),
        HostToDeviceCopy::new(&second, DevicePtr(29)),
    ];
    for (failure, expected) in [
        (Failure::Sync, vec!["sync:41"]),
        (Failure::Copy(1), vec!["sync:41", "copy:17:3"]),
        (Failure::Copy(2), vec!["sync:41", "copy:17:3", "copy:29:5"]),
    ] {
        let gpu = RecordingGpu::new(failure);
        assert!(gpu.copy_h2d_group_on_stream(&copies, 41).is_err());
        assert_eq!(gpu.events(), expected);
    }
}

#[test]
fn empty_group_does_not_synchronize() {
    let gpu = RecordingGpu::new(Failure::None);
    gpu.copy_h2d_group_on_stream(&[], 41).unwrap();
    assert!(gpu.events().is_empty());
}

#[test]
fn pinned_owner_is_bounded_and_only_feeds_typed_async() {
    let gpu = RecordingGpu::new(Failure::None);
    assert!(gpu.alloc_host_pinned(0).is_err());
    let mut owner = gpu.alloc_host_pinned(8).unwrap();
    owner
        .as_mut_slice()
        .copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    assert!(owner.pinned_slice(9).is_err());
    assert!(owner.pinned_slice_range(7, 2).is_err());
    assert!(owner.pinned_slice_range(usize::MAX, 2).is_err());
    assert_eq!(owner.pinned_slice_range(2, 4).unwrap().len(), 4);
    let pinned = owner.pinned_slice(8).unwrap();
    unsafe {
        gpu.copy_h2d_pinned_async(pinned, DevicePtr(53), 41)
            .unwrap();
    }
    assert_eq!(gpu.events(), ["copy:53:8"]);
}

#[test]
fn cuda_override_uses_one_sync_and_blocking_members() {
    let start = GPU_IMPL
        .find("    fn copy_h2d_group_on_stream(")
        .expect("group implementation");
    let end = GPU_IMPL[start..]
        .find("\n    fn copy_d2d(")
        .expect("next implementation");
    let body = &GPU_IMPL[start..start + end];
    assert_eq!(body.matches("cuStreamSynchronize(").count(), 1);
    assert_eq!(body.matches("cuMemcpyHtoD_v2(").count(), 1);
    assert!(!body.contains("cuMemcpyHtoDAsync_v2("));
    assert!(body.find("cuStreamSynchronize(") < body.find("cuMemcpyHtoD_v2("));

    assert!(GPU_TRAIT.contains("pub struct PinnedHostBuffer"));
    assert!(GPU_TRAIT.contains("pub struct PinnedHostSlice<'a>"));
    assert!(!GPU_TRAIT.contains("pub fn PinnedHostSlice"));
    assert!(!GPU_TRAIT.contains("fn free_host_pinned("));
    assert!(!GPU_TRAIT.contains("fn copy_h2d_async("));
}

#[test]
fn model_pinned_staging_and_graph_sources_are_raii_owned() {
    assert!(MODEL_TYPES.contains("buffer: PinnedHostBuffer"));
    assert!(MODEL_TYPES.contains("UnsafeCell<Option<PinnedHostBuffer>>"));
    assert!(!MODEL_TYPES.contains("tree_kv_indir_base_host_pinned: *mut u8"));
    assert!(!MODEL_TYPES.contains("ptr: *mut u8"));
    assert!(
        MODEL_TYPES
            .find("pub(super) pinned_staging:")
            .expect("pinned staging field")
            < MODEL_TYPES.find("pub(super) gpu:").expect("GPU field")
    );
    assert!(!MODEL_INIT.contains("free_host_pinned"));
    assert!(GPU_IMPL.contains("impl Drop for CudaPinnedHostStorage"));
    assert_eq!(GPU_IMPL.matches("cuMemFreeHost(").count(), 1);

    assert!(QWEN_SSM.contains("UnsafeCell<PinnedHostBuffer>"));
    assert!(!QWEN_SSM.contains("UnsafeCell<[u64; 64]>"));
    assert!(QWEN_SSM_INIT.contains("gpu.alloc_host_pinned(64 * std::mem::size_of::<u64>())?"));

    assert!(PREFILL_UPLOAD_META.contains("pinned_slice_range(pinned_offset, pinned_bytes)"));
    assert!(PREFILL_BATCH_KERNEL.contains("pinned_cursor"));
    assert!(PREFILL_BATCH_KERNEL.contains("PinnedH2dCompletionGuard"));
    assert!(PREFILL_BATCH_KERNEL.contains("std::process::abort()"));

    let drop_body = MODEL_DROP
        .split_once("impl Drop for TransformerModel")
        .expect("model drop implementation")
        .1;
    let sync = drop_body
        .find("synchronize(")
        .expect("drop synchronization");
    let piecewise = drop_body
        .find("piecewise_decode_graphs")
        .expect("piecewise graph teardown");
    assert!(sync < piecewise);
    assert!(drop_body.contains("destroy_graph"));
    assert!(drop_body.matches("std::process::abort()").count() >= 2);
}
