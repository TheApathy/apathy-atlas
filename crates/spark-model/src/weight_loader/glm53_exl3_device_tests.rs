// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::BTreeMap;

use spark_runtime::gpu::{GpuBackend, mock::MockGpuBackend};

use super::*;
use crate::weight_loader::glm53_exl3::{
    Glm53Exl3Admission, Glm53Exl3ShardInfo, Glm53Exl3TensorInfo,
};

#[test]
fn two_shards_are_two_packed_allocations_with_typed_subviews() {
    let root = unique_temp_dir();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("a.st"), [9, 9, 1, 2, 3, 4]).unwrap();
    std::fs::write(root.join("b.st"), [8, 5, 6, 7, 8]).unwrap();
    let files = fixture(root.clone());
    let gpu = MockGpuBackend::new();
    let store = load_glm53_exl3_store(&files, &gpu, 16).unwrap();
    assert_eq!(store.slab_count(), 2);
    assert_eq!(store.allocated_bytes(), 8);
    assert_eq!(store.payload_bytes(), 8);
    assert_eq!(gpu.alloc_count(), 2);

    let trellis = store.get("x.trellis").unwrap();
    assert_eq!(trellis.ptr.0 % 256, 0);
    assert_eq!(trellis.dtype, Glm53Exl3Dtype::I16);
    assert_eq!(trellis.shape, [2]);
    let mut raw = [0u8; 4];
    gpu.copy_d2h(trellis.ptr, &mut raw).unwrap();
    assert_eq!(raw, [1, 2, 3, 4]);

    let suh = store.get("x.suh").unwrap();
    assert_eq!(suh.ptr.0 % 256, 0);
    assert_eq!(suh.dtype, Glm53Exl3Dtype::F16);
    gpu.copy_d2h(suh.ptr, &mut raw).unwrap();
    assert_eq!(raw, [5, 6, 7, 8]);
    store.free(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn adjacent_source_tensors_get_individually_aligned_device_offsets() {
    let root = unique_temp_dir();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("a.st"), [9, 1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    let files = Glm53Exl3Files {
        root: root.clone(),
        admission: Glm53Exl3Admission {
            tensor_count: 2,
            data_bytes: 8,
            ledger_entries: 1,
            quantized_entries: 1,
            dtype_counts: BTreeMap::from([(Glm53Exl3Dtype::F16, 2)]),
        },
        shards: vec![Glm53Exl3ShardInfo {
            file_name: "a.st".into(),
            data_start: 1,
            data_bytes: 8,
        }],
        tensors: BTreeMap::from([
            (
                "x.first".into(),
                Glm53Exl3TensorInfo {
                    dtype: Glm53Exl3Dtype::F16,
                    shape: vec![2],
                    shard: 0,
                    data_offset: 0,
                    byte_len: 4,
                },
            ),
            (
                "x.second".into(),
                Glm53Exl3TensorInfo {
                    dtype: Glm53Exl3Dtype::F16,
                    shape: vec![2],
                    shard: 0,
                    data_offset: 4,
                    byte_len: 4,
                },
            ),
        ]),
    };
    let gpu = MockGpuBackend::new();
    let store = load_glm53_exl3_store(&files, &gpu, 0).unwrap();
    assert_eq!(store.payload_bytes(), 8);
    assert_eq!(store.allocated_bytes(), 260);
    let first = store.get("x.first").unwrap();
    let second = store.get("x.second").unwrap();
    assert_eq!(first.ptr.0 % 256, 0);
    assert_eq!(second.ptr.0 % 256, 0);
    assert_eq!(second.ptr.0 - first.ptr.0, 256);
    let mut raw = [0u8; 4];
    gpu.copy_d2h(first.ptr, &mut raw).unwrap();
    assert_eq!(raw, [1, 2, 3, 4]);
    gpu.copy_d2h(second.ptr, &mut raw).unwrap();
    assert_eq!(raw, [5, 6, 7, 8]);
    store.free(&gpu).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn bounded_windows_preserve_payload_and_zero_alignment_gaps() {
    let root = unique_temp_dir();
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("packed.st");
    std::fs::write(&path, [9, 9, 1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    let layout = PackedShardLayout {
        allocation_bytes: 260,
        payload_bytes: 8,
        routes: vec![
            PackedTensorRoute {
                name: "x.first".into(),
                device_offset: 0,
                byte_len: 4,
            },
            PackedTensorRoute {
                name: "x.second".into(),
                device_offset: 256,
                byte_len: 4,
            },
        ],
    };
    let gpu = MockGpuBackend::new();
    let base = gpu.alloc(layout.allocation_bytes).unwrap();
    gpu.copy_h2d(&vec![0xa5; layout.allocation_bytes], base)
        .unwrap();
    copy_packed_shard_window(&path, 2, &layout, base, &gpu, 7).unwrap();
    let mut packed = vec![0u8; layout.allocation_bytes];
    gpu.copy_d2h(base, &mut packed).unwrap();
    assert_eq!(&packed[..4], &[1, 2, 3, 4]);
    assert!(packed[4..256].iter().all(|&byte| byte == 0));
    assert_eq!(&packed[256..], &[5, 6, 7, 8]);
    gpu.free(base).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn zero_sized_copy_window_fails_closed() {
    let root = unique_temp_dir();
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("empty.st");
    std::fs::write(&path, []).unwrap();
    let layout = PackedShardLayout {
        allocation_bytes: 0,
        payload_bytes: 0,
        routes: Vec::new(),
    };
    let gpu = MockGpuBackend::new();
    let error = copy_packed_shard_window(&path, 0, &layout, DevicePtr::NULL, &gpu, 0).unwrap_err();
    assert!(error.to_string().contains("window must be nonzero"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn preflight_fails_before_allocating() {
    let files = fixture(std::env::temp_dir());
    let gpu = MockGpuBackend::new();
    let error = load_glm53_exl3_store(&files, &gpu, usize::MAX).unwrap_err();
    assert!(error.failure().to_string().contains("overflows"));
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn copy_and_view_failures_roll_back_every_slab() {
    let root = unique_temp_dir();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("a.st"), [9, 9, 1, 2, 3, 4]).unwrap();
    let gpu = MockGpuBackend::new();
    let copy_error = load_glm53_exl3_store(&fixture(root.clone()), &gpu, 0).unwrap_err();
    assert!(copy_error.failure().to_string().contains("b.st"));
    assert_eq!(gpu.alloc_count(), 0);

    std::fs::write(root.join("b.st"), [8, 5, 6, 7, 8]).unwrap();
    let mut malformed = fixture(root.clone());
    malformed.tensors.get_mut("x.suh").unwrap().byte_len = 6;
    let view_error = load_glm53_exl3_store(&malformed, &gpu, 0).unwrap_err();
    assert!(view_error.failure().to_string().contains("escapes"));
    assert_eq!(gpu.alloc_count(), 0);
    std::fs::remove_dir_all(root).unwrap();
}

fn unique_temp_dir() -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "atlas-exl3-store-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ))
}

fn fixture(root: std::path::PathBuf) -> Glm53Exl3Files {
    Glm53Exl3Files {
        root,
        admission: Glm53Exl3Admission {
            tensor_count: 2,
            data_bytes: 8,
            ledger_entries: 1,
            quantized_entries: 1,
            dtype_counts: BTreeMap::from([(Glm53Exl3Dtype::F16, 1), (Glm53Exl3Dtype::I16, 1)]),
        },
        shards: vec![
            Glm53Exl3ShardInfo {
                file_name: "a.st".into(),
                data_start: 2,
                data_bytes: 4,
            },
            Glm53Exl3ShardInfo {
                file_name: "b.st".into(),
                data_start: 1,
                data_bytes: 4,
            },
        ],
        tensors: BTreeMap::from([
            (
                "x.suh".into(),
                Glm53Exl3TensorInfo {
                    dtype: Glm53Exl3Dtype::F16,
                    shape: vec![2],
                    shard: 1,
                    data_offset: 0,
                    byte_len: 4,
                },
            ),
            (
                "x.trellis".into(),
                Glm53Exl3TensorInfo {
                    dtype: Glm53Exl3Dtype::I16,
                    shape: vec![2],
                    shard: 0,
                    data_offset: 0,
                    byte_len: 4,
                },
            ),
        ]),
    }
}
