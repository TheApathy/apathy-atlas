// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Map, Value, json};

use super::attest_exact_bf16_safetensors;

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
type FixtureTensor<'a> = (&'a str, &'a str, &'a [usize], &'a [u16]);

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        let serial = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "atlas-bf16-receipt-{}-{serial}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

fn manifest() -> BTreeMap<String, Vec<usize>> {
    BTreeMap::from([("a".into(), vec![2]), ("b".into(), vec![1])])
}

fn write_fixture(dir: &Path, tensors: &[FixtureTensor<'_>]) {
    let mut header = Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, values) in tensors {
        let start = data.len();
        for value in *values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            (*name).into(),
            json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, data.len()]
            }),
        );
    }
    let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = Vec::new();
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(&header);
    file.extend_from_slice(&data);
    std::fs::write(dir.join("model.safetensors"), file).unwrap();
}

#[test]
fn attests_exact_finite_bf16_and_binds_content() {
    let dir = TempDir::new();
    write_fixture(
        dir.path(),
        &[
            ("a", "BF16", &[2], &[0x0000, 0x3f80]),
            ("b", "BF16", &[1], &[0x7f7f]),
        ],
    );
    let first = attest_exact_bf16_safetensors(dir.path(), &manifest()).unwrap();
    assert_eq!(first.tensor_count, 2);
    assert_eq!(first.element_count, 3);
    assert_eq!(first.file_bytes, 126);
    // Re-pinned 2026-09-23: serde_json `preserve_order` (workspace-wide since the DeepSeek
    // port) serializes the fixture header's keys in insertion order (dtype, shape,
    // data_offsets) instead of sorted. Both hashes were reproduced from the fixture bytes:
    // sorted keys give the old 75a3591e..., insertion order gives this one; 126 bytes either way.
    assert_eq!(
        first.file_sha256,
        "525a993cec4f6a8643743dff3c15be1c73d63b4453721a9813fb08aa7f383b50"
    );

    write_fixture(
        dir.path(),
        &[
            ("a", "BF16", &[2], &[0x8000, 0x3f80]),
            ("b", "BF16", &[1], &[0x7f7f]),
        ],
    );
    let second = attest_exact_bf16_safetensors(dir.path(), &manifest()).unwrap();
    assert_ne!(first.file_sha256, second.file_sha256);
}

#[test]
fn rejects_nan_and_both_infinities() {
    for hostile in [0x7fc1, 0x7f80, 0xff80] {
        let dir = TempDir::new();
        write_fixture(
            dir.path(),
            &[
                ("a", "BF16", &[2], &[0x3f80, hostile]),
                ("b", "BF16", &[1], &[0x0000]),
            ],
        );
        let error = attest_exact_bf16_safetensors(dir.path(), &manifest()).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("`a` element 1"), "{message}");
        assert!(message.contains("non-finite"), "{message}");
    }
}

#[test]
fn rejects_wrong_dtype_shape_and_tensor_set() {
    let cases: &[(&[FixtureTensor<'_>], &str)] = &[
        (
            &[("a", "F16", &[2], &[0, 0]), ("b", "BF16", &[1], &[0])],
            "expected BF16",
        ),
        (
            &[("a", "BF16", &[1], &[0]), ("b", "BF16", &[1], &[0])],
            "shape",
        ),
        (
            &[("a", "BF16", &[2], &[0, 0]), ("extra", "BF16", &[1], &[0])],
            "manifest mismatch",
        ),
    ];
    for (tensors, expected_error) in cases {
        let dir = TempDir::new();
        write_fixture(dir.path(), tensors);
        let error = attest_exact_bf16_safetensors(dir.path(), &manifest()).unwrap_err();
        assert!(error.to_string().contains(expected_error), "{error:#}");
    }
}

#[test]
fn rejects_noncanonical_extra_shard_and_index() {
    let dir = TempDir::new();
    write_fixture(
        dir.path(),
        &[("a", "BF16", &[2], &[0, 0]), ("b", "BF16", &[1], &[0])],
    );
    std::fs::write(dir.path().join("extra_weights.safetensors"), []).unwrap();
    let error = attest_exact_bf16_safetensors(dir.path(), &manifest()).unwrap_err();
    assert!(error.to_string().contains("requires only"), "{error:#}");

    std::fs::remove_file(dir.path().join("extra_weights.safetensors")).unwrap();
    std::fs::write(dir.path().join("model.safetensors.index.json"), b"{}").unwrap();
    let error = attest_exact_bf16_safetensors(dir.path(), &manifest()).unwrap_err();
    assert!(error.to_string().contains("forbids sharded"), "{error:#}");
}
