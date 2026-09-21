// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;
use std::{
    os::unix::ffi::OsStrExt,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "atlas-vision-moe-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn child(&self) -> PathBuf {
        self.0.join("capture")
    }
    fn capture(&self) -> MoeCapture {
        let ids: Vec<_> = IDS.iter().flat_map(|id| id.to_le_bytes()).collect();
        MoeCapture::start(self.child().as_os_str(), &ids).unwrap()
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn vision_moe_dump_admission_and_exact_dispatch_fail_closed() {
    let good = Admission {
        vision: true,
        first: true,
        c1: true,
        eager: true,
        geometry: true,
    };
    good.validate().unwrap();
    for bad in [
        Admission {
            vision: false,
            ..good
        },
        Admission {
            first: false,
            ..good
        },
        Admission { c1: false, ..good },
        Admission {
            eager: false,
            ..good
        },
        Admission {
            geometry: false,
            ..good
        },
    ] {
        assert!(bad.validate().is_err());
    }
    validate_mode(true, true, true, true).unwrap();
    for mask in 0..15 {
        assert!(validate_mode(mask & 1 != 0, mask & 2 != 0, mask & 4 != 0, mask & 8 != 0).is_err());
    }
}

#[test]
fn vision_moe_dump_rejects_ids_paths_and_existing_destinations() {
    let temp = Temp::new();
    for ids in [vec![], vec![0; 48], vec![255; 48], vec![0; 44]] {
        assert!(MoeCapture::start(temp.child().as_os_str(), &ids).is_err());
        assert!(!temp.child().exists());
    }
    for path in [
        OsStr::new(""),
        OsStr::new("relative"),
        OsStr::new("/"),
        OsStr::new("/tmp/../bad"),
        OsStr::from_bytes(b"/tmp/\xff"),
    ] {
        assert!(create_root(path).is_err());
    }
    let _capture = temp.capture();
    assert!(create_root(temp.child().as_os_str()).is_err());
    let link = temp.0.join("link");
    std::os::unix::fs::symlink(&temp.0, &link).unwrap();
    assert!(create_root(link.join("new").as_os_str()).is_err());
}

#[test]
fn vision_moe_dump_exact_budget_streams_chunks_and_commits_last() {
    let temp = Temp::new();
    let mut capture = temp.capture();
    assert_eq!(
        48 + STAGES.iter().map(|s| s.spec().3).sum::<usize>(),
        PAYLOAD_BYTES
    );
    let mut largest_chunk = 0;
    for stage in STAGES {
        assert!(!temp.child().join("manifest.json").exists());
        let mut expected_offset = 0;
        capture
            .write_stage(stage, DevicePtr(0x1000), |offset, data| {
                assert_eq!(offset, expected_offset);
                largest_chunk = largest_chunk.max(data.len());
                assert!(data.len() <= CHUNK);
                data.fill(0xff); // Diagnostic payloads must preserve even NaNs verbatim.
                expected_offset += data.len();
                Ok(())
            })
            .unwrap();
        assert_eq!(expected_offset, stage.spec().3);
        assert_eq!(
            std::fs::metadata(temp.child().join(format!("{}.bin", stage.spec().0)))
                .unwrap()
                .len(),
            stage.spec().3 as u64
        );
    }
    assert_eq!(largest_chunk, CHUNK);
    capture.dispatch = Some(json!({"fixture":true}));
    capture.finish().unwrap();
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(temp.child().join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["status"], "COMPLETE");
    assert_eq!(manifest["schema"], "atlas-vision-moe-l0-dump-v1");
    assert_eq!(manifest["payload_bytes"], PAYLOAD_BYTES);
    assert_eq!(manifest["tensors"].as_object().unwrap().len(), 15);
    assert_eq!(manifest["token_ids"], json!(IDS));
}

#[test]
fn vision_moe_dump_partial_order_repeat_and_io_failure_never_complete() {
    let temp = Temp::new();
    let mut capture = temp.capture();
    assert!(
        capture
            .write_stage(Stage::SharedInput, DevicePtr(1), |_, _| Ok(()))
            .is_err()
    );
    assert!(!temp.child().join("shared_input.bin").exists());
    capture
        .write_stage(Stage::FfnInput, DevicePtr(1), |_, _| Ok(()))
        .unwrap();
    assert!(
        capture
            .write_stage(Stage::FfnInput, DevicePtr(1), |_, _| Ok(()))
            .is_err()
    );
    assert!(
        capture
            .write_stage(Stage::SharedInput, DevicePtr(1), |_, _| anyhow::bail!(
                "injected D2H error"
            ))
            .is_err()
    );
    assert_eq!(capture.next, 1);
    assert!(
        capture
            .write_stage(Stage::SharedInput, DevicePtr(1), |_, _| Ok(()))
            .is_err()
    );
    assert!(capture.finish().is_err());
    assert!(!temp.child().join("manifest.json").exists());
}

#[test]
fn vision_moe_dump_rejects_null_pointer_and_missing_dispatch() {
    let temp = Temp::new();
    let mut capture = temp.capture();
    assert!(
        capture
            .stage(Stage::FfnInput, &MockGpuBackend::new(), DevicePtr::NULL, 0)
            .is_err()
    );
    assert!(!temp.child().join("ffn_input.bin").exists());
    for stage in STAGES {
        capture
            .write_stage(stage, DevicePtr(1), |_, _| Ok(()))
            .unwrap();
    }
    assert!(capture.finish().is_err());
    assert!(!temp.child().join("manifest.json").exists());
}
