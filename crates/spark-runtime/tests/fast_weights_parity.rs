// SPDX-License-Identifier: AGPL-3.0-only

//! Parity test: FastSafetensorsLoader must produce byte-identical weights
//! to the mmap-based SafetensorsLoader for the same file.
//!
//! Builds a tiny synthetic safetensors file in a tempdir, loads it with both
//! loaders against a MockGpuBackend, and asserts every tensor's bytes match.

#![cfg(unix)]

use spark_runtime::fast_weights::FastSafetensorsLoader;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::{SafetensorsLoader, WeightDtype, WeightLoader};
use std::io::Write;

/// Build a minimal `model.safetensors` with two BF16 tensors and one U8 tensor.
/// Layout written by hand so the test doesn't depend on the safetensors crate
/// for encoding (decoding is still needed, used by the baseline loader).
fn write_test_safetensors(dir: &std::path::Path) -> std::path::PathBuf {
    // Tensor A: BF16, shape [4, 8] = 64 bytes.
    // Tensor B: BF16, shape [2, 2] = 8 bytes.
    // Tensor C: U8,   shape [16]   = 16 bytes.
    let a_bytes: Vec<u8> = (0..64).map(|i| i as u8).collect();
    let b_bytes: Vec<u8> = (0..8).map(|i| (128 + i) as u8).collect();
    let c_bytes: Vec<u8> = (0..16).map(|i| (200 + i) as u8).collect();

    let header = serde_json::json!({
        "a": { "dtype": "BF16", "shape": [4, 8], "data_offsets": [0, 64] },
        "b": { "dtype": "BF16", "shape": [2, 2], "data_offsets": [64, 72] },
        "c": { "dtype": "U8",   "shape": [16],   "data_offsets": [72, 88] },
    });
    let header_bytes = serde_json::to_vec(&header).unwrap();

    let path = dir.join("model.safetensors");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    f.write_all(&header_bytes).unwrap();
    f.write_all(&a_bytes).unwrap();
    f.write_all(&b_bytes).unwrap();
    f.write_all(&c_bytes).unwrap();
    f.sync_all().unwrap();
    path
}

#[test]
fn fast_and_mmap_loaders_agree() {
    let tmp = tempdir_like();
    write_test_safetensors(&tmp);

    let gpu_base = MockGpuBackend::new();
    let base = SafetensorsLoader::new()
        .load(&tmp, &gpu_base, 0)
        .expect("baseline load");
    assert_eq!(base.len(), 3);

    let gpu_fast = MockGpuBackend::new();
    let mut fast = FastSafetensorsLoader::new();
    // Force the buffered-read path: tmpfs rejects O_DIRECT on most kernels,
    // but we disable it explicitly so the test is deterministic.
    fast.try_direct_io = false;
    let new = fast.load(&tmp, &gpu_fast, 0).expect("fast load");
    assert_eq!(new.len(), 3);

    for name in ["a", "b", "c"] {
        let wb = base.get(name).unwrap();
        let wn = new.get(name).unwrap();
        assert_eq!(wb.shape, wn.shape, "shape mismatch for {name}");
        assert_eq!(wb.dtype, wn.dtype, "dtype mismatch for {name}");
        let bb = gpu_base.read_alloc(wb.ptr).unwrap();
        let bn = gpu_fast.read_alloc(wn.ptr).unwrap();
        assert_eq!(bb, bn, "byte mismatch for {name}");
    }

    std::fs::remove_dir_all(&tmp).ok();
}

/// `extra_skip` must drop a tensor from BOTH the pre-flight size sum and the upload.
/// Pre-flight: the mock reports 120 GiB free; the reserve leaves 100 bytes of budget. All three
/// tensors (88 bytes x 1.3) do not fit; without the 64-byte "a" (24 x 1.3) they do. CONTROL: the
/// same loader without the skip must count "a" and fail the pre-flight.
#[test]
fn extra_skip_excludes_tensors_from_preflight_and_upload() {
    let tmp = tempdir_like();
    write_test_safetensors(&tmp);
    let reserve = 120 * 1024 * 1024 * 1024 - 100;
    let loader = |skip: bool| {
        let mut l = FastSafetensorsLoader::new();
        l.try_direct_io = false;
        l.peak_memory_multiplier = Some(1.3);
        if skip {
            l.extra_skip = Some(std::sync::Arc::new(|name: &str| name == "a"));
        }
        l
    };

    let gpu = MockGpuBackend::new();
    let store = loader(true).load(&tmp, &gpu, reserve).expect("skip fits the pre-flight");
    assert_eq!(store.len(), 2);
    assert!(store.get("a").is_err(), "skipped tensor was uploaded");
    assert_eq!(gpu.alloc_count(), 2, "an allocation was made for the skipped tensor");

    let err = loader(false).load(&tmp, &MockGpuBackend::new(), reserve).err().expect(
        "CONTROL: without the skip the 88-byte load must exceed the 100-byte budget at 1.3x",
    );
    assert!(format!("{err:#}").contains("OOM pre-flight"), "unexpected error: {err:#}");

    let all = loader(false).load(&tmp, &MockGpuBackend::new(), 0).expect("no reserve");
    assert_eq!(all.len(), 3);
    std::fs::remove_dir_all(&tmp).ok();
}

/// DeepSeek-V4.1's checkpoint dtypes beyond the basic set: F8_E8M0 (UE8M0 block scales) and I8
/// (packed DSpark experts). Both loaders must accept them and agree byte for byte.
#[test]
fn fast_and_mmap_loaders_accept_e8m0_and_i8() {
    let tmp = tempdir_like();
    let s_bytes: Vec<u8> = (0..8).map(|i| 120 + i as u8).collect();
    let p_bytes: Vec<u8> = (0..16).map(|i| (i * 17) as u8).collect();
    let header = serde_json::json!({
        "w.scale": { "dtype": "F8_E8M0", "shape": [2, 4], "data_offsets": [0, 8] },
        "mtp.e": { "dtype": "I8", "shape": [16], "data_offsets": [8, 24] },
    });
    let header_bytes = serde_json::to_vec(&header).unwrap();
    let mut f = std::fs::File::create(tmp.join("model.safetensors")).unwrap();
    f.write_all(&(header_bytes.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&header_bytes).unwrap();
    f.write_all(&s_bytes).unwrap();
    f.write_all(&p_bytes).unwrap();
    f.sync_all().unwrap();

    let (gb, gf) = (MockGpuBackend::new(), MockGpuBackend::new());
    let base = SafetensorsLoader::new().load(&tmp, &gb, 0).expect("baseline load");
    let mut fast = FastSafetensorsLoader::new();
    fast.try_direct_io = false;
    let new = fast.load(&tmp, &gf, 0).expect("fast load");
    for (name, dtype) in [("w.scale", WeightDtype::FP8E8M0), ("mtp.e", WeightDtype::UInt8)] {
        let (wb, wn) = (base.get(name).unwrap(), new.get(name).unwrap());
        assert_eq!(wb.dtype, dtype, "{name}");
        assert_eq!(wn.dtype, dtype, "{name}");
        assert_eq!(gb.read_alloc(wb.ptr).unwrap(), gf.read_alloc(wn.ptr).unwrap(), "{name}");
    }
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn fast_loader_with_direct_io_if_supported() {
    // Best-effort O_DIRECT test — silently succeeds (by falling back to
    // buffered) if the filesystem rejects O_DIRECT.
    let tmp = tempdir_like();
    write_test_safetensors(&tmp);

    let gpu_base = MockGpuBackend::new();
    let base = SafetensorsLoader::new().load(&tmp, &gpu_base, 0).unwrap();

    let gpu_fast = MockGpuBackend::new();
    let fast = FastSafetensorsLoader::new(); // try_direct_io = true by default
    let new = fast
        .load(&tmp, &gpu_fast, 0)
        .expect("fast load with O_DIRECT attempted");
    assert_eq!(new.len(), 3);

    for name in ["a", "b", "c"] {
        let bb = gpu_base.read_alloc(base.get(name).unwrap().ptr).unwrap();
        let bn = gpu_fast.read_alloc(new.get(name).unwrap().ptr).unwrap();
        assert_eq!(bb, bn, "byte mismatch for {name} (O_DIRECT path)");
    }

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn prefix_filter_skips_unrelated_missing_index_shards() {
    let tmp = tempdir_like();
    let header = serde_json::json!({
        "mtp.a": { "dtype": "BF16", "shape": [2], "data_offsets": [0, 4] },
    });
    let header_bytes = serde_json::to_vec(&header).unwrap();
    let mut shard = std::fs::File::create(tmp.join("mtp.safetensors")).unwrap();
    shard
        .write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    shard.write_all(&header_bytes).unwrap();
    shard.write_all(&[1, 2, 3, 4]).unwrap();
    shard.sync_all().unwrap();
    let index = serde_json::json!({
        "weight_map": {
            "mtp.a": "mtp.safetensors",
            "model.base": "missing-base.safetensors"
        }
    });
    std::fs::write(
        tmp.join("model.safetensors.index.json"),
        serde_json::to_vec(&index).unwrap(),
    )
    .unwrap();

    let gpu = MockGpuBackend::new();
    let mut loader = FastSafetensorsLoader::new().with_name_prefixes(["mtp.".to_string()]);
    loader.try_direct_io = false;
    let store = loader
        .load(&tmp, &gpu, 0)
        .expect("prefix-filtered sidecar load");
    assert_eq!(store.len(), 1);
    assert!(store.contains("mtp.a"));
    assert!(!store.contains("model.base"));
    assert_eq!(
        gpu.read_alloc(store.get("mtp.a").unwrap().ptr).unwrap(),
        [1, 2, 3, 4]
    );

    std::fs::remove_dir_all(&tmp).ok();
}

/// Creates a unique temp directory without pulling in the tempfile crate.
fn tempdir_like() -> std::path::PathBuf {
    let pid = std::process::id();
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!("atlas-fwp-{pid}-{ns}"));
    std::fs::create_dir_all(&p).unwrap();
    p
}
