// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;
use std::{
    os::unix::ffi::OsStrExt,
    sync::atomic::{AtomicUsize, Ordering},
};
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "atlas-vision-l0-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn child(&self) -> PathBuf {
        self.0.join("capture")
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn capture(path: &Path) -> Result<Capture> {
    Capture::start(path.as_os_str(), (0..12).collect(), [1e-20, 1e-6, 20.0])
}

#[test]
fn vision_l0_admission_rejects_every_unsupported_mode() {
    let valid = Admission {
        vision: true,
        first: true,
        c1: true,
        eager: true,
        rows: 12,
        geometry: true,
    };
    valid.validate().unwrap();
    for bad in [
        Admission {
            vision: false,
            ..valid
        },
        Admission {
            first: false,
            ..valid
        },
        Admission { c1: false, ..valid },
        Admission {
            eager: false,
            ..valid
        },
        Admission { rows: 0, ..valid },
        Admission { rows: 13, ..valid },
        Admission {
            geometry: false,
            ..valid
        },
    ] {
        assert!(bad.validate().is_err());
    }
}

#[test]
fn vision_l0_rejects_unsafe_paths_and_repeat_destinations() {
    let temp = Temp::new();
    for path in [
        OsStr::new(""),
        OsStr::new("relative"),
        OsStr::new("/"),
        OsStr::new("/tmp/../bad"),
        OsStr::from_bytes(b"/tmp/\xff"),
    ] {
        assert!(create_root(path).is_err());
    }
    let link = temp.0.join("link");
    std::os::unix::fs::symlink(&temp.0, &link).unwrap();
    assert!(capture(&link.join("child")).is_err());
    std::fs::write(temp.0.join("file"), b"sentinel").unwrap();
    assert!(capture(&temp.0.join("file")).is_err());
    let _first = capture(&temp.child()).unwrap();
    assert!(capture(&temp.child()).is_err());
    assert_eq!(std::fs::read(temp.0.join("file")).unwrap(), b"sentinel");
}

#[test]
fn vision_l0_rejects_bad_ids_before_creating_directory() {
    let temp = Temp::new();
    for ids in [vec![], vec![0; 11], vec![VOCAB; 12], vec![u32::MAX; 12]] {
        assert!(Capture::start(temp.child().as_os_str(), ids, [1e-20, 1e-6, 20.0]).is_err());
        assert!(!temp.child().exists());
    }
}

#[test]
fn vision_l0_payloads_are_exact_and_manifest_commits_last() {
    let temp = Temp::new();
    let gpu = MockGpuBackend::new();
    let mut capture = capture(&temp.child()).unwrap();
    for stage in STAGES {
        let (name, _, _, bytes) = stage.spec();
        let input: Vec<_> = (0..bytes).map(|i| (i % 251) as u8).collect();
        let ptr = gpu.alloc(bytes).unwrap();
        gpu.copy_h2d(&input, ptr).unwrap();
        let allocations = gpu.alloc_count();
        capture.stage(stage, &gpu, ptr, 7).unwrap();
        assert!(capture.stage(stage, &gpu, ptr, 7).is_err());
        assert_eq!(gpu.alloc_count(), allocations);
        assert_eq!(
            std::fs::read(temp.child().join(format!("{name}.bin"))).unwrap(),
            input
        );
        assert!(!temp.child().join("manifest.json").exists());
        gpu.free(ptr).unwrap();
    }
    capture.finish().unwrap();
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(temp.child().join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["status"], "COMPLETE");
    assert_eq!(manifest["tensors"].as_object().unwrap().len(), 14);
    assert_eq!(manifest["token_ids"], json!((0..12).collect::<Vec<_>>()));
    assert!(manifest["payload_bytes"].as_u64().unwrap() < MAX_BYTES as u64);
    assert_eq!(std::fs::read_dir(temp.child()).unwrap().count(), 16);
}

#[test]
fn vision_l0_partial_order_and_io_failures_never_commit() {
    let temp = Temp::new();
    let gpu = MockGpuBackend::new();
    let mut capture = capture(&temp.child()).unwrap();
    let ptr = gpu.alloc(12 * 4096 * 2).unwrap();
    assert!(capture.stage(Stage::NormAttn, &gpu, ptr, 0).is_err());
    assert!(
        capture
            .stage(Stage::Embed, &gpu, DevicePtr::NULL, 0)
            .is_err()
    );
    assert!(capture.stage(Stage::Embed, &gpu, DevicePtr(1), 0).is_err());
    capture.total = MAX_BYTES;
    assert!(capture.stage(Stage::Embed, &gpu, ptr, 0).is_err());
    capture.total = 48;
    std::fs::write(temp.child().join("embed.bin"), b"sentinel").unwrap();
    assert!(capture.stage(Stage::Embed, &gpu, ptr, 0).is_err());
    assert_eq!(
        std::fs::read(temp.child().join("embed.bin")).unwrap(),
        b"sentinel"
    );
    assert!(capture.finish().is_err());
    assert!(!temp.child().join("manifest.json").exists());
}

#[test]
fn vision_l0_stage_metadata_has_exact_types_shapes_and_total() {
    let expected = [
        ("embed", "BF16", vec![12, 4096], 98_304),
        ("hc_expanded", "F32", vec![12, 4, 4096], 786_432),
        ("hc_pre_attn", "BF16", vec![12, 4096], 98_304),
        ("post_attn", "F32", vec![12, 4], 192),
        ("comb_attn", "F32", vec![12, 4, 4], 768),
        ("norm_attn", "BF16", vec![12, 4096], 98_304),
        ("attention_out", "BF16", vec![12, 4096], 98_304),
        ("hc_post_attn", "F32", vec![12, 4, 4096], 786_432),
        ("hc_pre_ffn", "BF16", vec![12, 4096], 98_304),
        ("post_ffn", "F32", vec![12, 4], 192),
        ("comb_ffn", "F32", vec![12, 4, 4], 768),
        ("norm_ffn", "BF16", vec![12, 4096], 98_304),
        ("moe_out", "BF16", vec![12, 4096], 98_304),
        ("hc_post_ffn", "F32", vec![12, 4, 4096], 786_432),
    ];
    let mut total = 48;
    for (stage, (name, dtype, shape, bytes)) in STAGES.into_iter().zip(expected) {
        assert_eq!(stage.spec(), (name, dtype, shape.as_slice(), bytes));
        total += bytes;
    }
    assert_eq!(total, 3_049_392);
}

#[test]
fn vision_l0_directory_descriptor_survives_parent_rename() {
    let temp = Temp::new();
    let capture = capture(&temp.child()).unwrap();
    let moved = temp.0.join("moved");
    std::fs::rename(temp.child(), &moved).unwrap();
    std::os::unix::fs::symlink(&temp.0, temp.child()).unwrap();
    capture.write_new("probe.bin", b"bound").unwrap();
    assert_eq!(std::fs::read(moved.join("probe.bin")).unwrap(), b"bound");
    assert!(!temp.0.join("probe.bin").exists());
}
