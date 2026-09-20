// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::anyhow;
use serde_json::{Map, Value, json};

use super::admission::W3SidecarRequest;
use super::digest::sha256;
use super::session::W3SidecarSession;

const PREFIX_0: &str = "model.language_model.layers.0";
const PREFIX_1: &str = "model.language_model.layers.1";

#[test]
fn sha256_matches_standard_vectors() {
    assert_eq!(
        hex(sha256(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        hex(sha256(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        hex(sha256(&vec![0x5a; 129])),
        "651526df875ac6cec56a649780e20fc4b9c71df77afb62199bf15864cb1f1241"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn exact_session_requires_upload_install_and_full_receipt() {
    let artifact = Fixture::valid(&[PREFIX_0, PREFIX_1]).finish();
    let temp = TempArtifact::new(&artifact);
    let prefixes = vec![PREFIX_0.to_owned(), PREFIX_1.to_owned()];
    let admitted_request = request(&temp.path, &artifact, "0,1", prefixes.len());
    let mut session = W3SidecarSession::prepare(admitted_request, &prefixes, 16, 16).unwrap();

    assert!(
        session
            .upload_layer_with(2, |_, _| -> anyhow::Result<()> {
                panic!("unrequested layer must not invoke the uploader")
            })
            .unwrap()
            .is_none()
    );
    let first = session
        .upload_layer_with(0, |bytes, plan| {
            assert!(!bytes[plan.gate.packed.clone()].is_empty());
            Ok("layer-0")
        })
        .unwrap();
    assert_eq!(first, Some("layer-0"));
    session.mark_installed(0).unwrap();
    session
        .upload_layer_with(1, |_, plan| {
            assert_eq!(plan.down.scale2, 1.0);
            Ok(())
        })
        .unwrap();
    session.mark_installed(1).unwrap();
    let receipt = session.finish().unwrap();
    assert_eq!(receipt.path, temp.path);
    assert_eq!(receipt.sha256, sha256(&artifact));
    assert_eq!(receipt.size, artifact.len() as u64);
    assert_ne!(receipt.inode, 0);
    assert_eq!(receipt.requested_layers, [0, 1]);
    assert_eq!(receipt.validated_layers, [0, 1]);
    assert_eq!(receipt.uploaded_layers, [0, 1]);
    assert_eq!(receipt.installed_layers, [0, 1]);
    assert_eq!(
        [
            receipt.requested_count,
            receipt.validated_count,
            receipt.uploaded_count,
            receipt.installed_count,
        ],
        [2, 2, 2, 2]
    );
}

#[cfg(target_os = "linux")]
#[test]
fn lifecycle_misuse_and_upload_failure_poison_the_session() {
    let artifact = Fixture::valid(&[PREFIX_0]).finish();
    let prefixes = vec![PREFIX_0.to_owned()];

    let temp = TempArtifact::new(&artifact);
    let admitted_request = request(&temp.path, &artifact, "0", 1);
    let mut partial = W3SidecarSession::prepare(admitted_request, &prefixes, 16, 16).unwrap();
    partial.upload_layer_with(0, |_, _| Ok(())).unwrap();
    assert!(partial.finish().is_err());

    let temp = TempArtifact::new(&artifact);
    let admitted_request = request(&temp.path, &artifact, "0", 1);
    let mut failed = W3SidecarSession::prepare(admitted_request, &prefixes, 16, 16).unwrap();
    assert!(
        failed
            .upload_layer_with::<()>(0, |_, _| Err(anyhow!("injected upload failure")))
            .is_err()
    );
    assert!(failed.mark_installed(0).is_err());
    assert!(failed.finish().is_err());

    let temp = TempArtifact::new(&artifact);
    let admitted_request = request(&temp.path, &artifact, "0", 1);
    let mut wrong_mark = W3SidecarSession::prepare(admitted_request, &prefixes, 16, 16).unwrap();
    wrong_mark.upload_layer_with(0, |_, _| Ok(())).unwrap();
    assert!(wrong_mark.mark_installed(1).is_err());
    assert!(wrong_mark.mark_installed(0).is_err());
}

#[cfg(target_os = "linux")]
#[test]
fn awaiting_and_duplicate_uploads_poison_independent_sessions() {
    let artifact = Fixture::valid(&[PREFIX_0]).finish();
    let prefixes = vec![PREFIX_0.to_owned()];

    let temp = TempArtifact::new(&artifact);
    let admitted_request = request(&temp.path, &artifact, "0", 1);
    let mut awaiting = W3SidecarSession::prepare(admitted_request, &prefixes, 16, 16).unwrap();
    awaiting.upload_layer_with(0, |_, _| Ok(())).unwrap();
    let error = awaiting.upload_layer_with(0, |_, _| Ok(())).unwrap_err();
    assert!(format!("{error:#}").contains("must be installed before uploading layer 0"));
    assert!(awaiting.mark_installed(0).is_err());

    let temp = TempArtifact::new(&artifact);
    let admitted_request = request(&temp.path, &artifact, "0", 1);
    let mut duplicate = W3SidecarSession::prepare(admitted_request, &prefixes, 16, 16).unwrap();
    duplicate.upload_layer_with(0, |_, _| Ok(())).unwrap();
    duplicate.mark_installed(0).unwrap();
    let error = duplicate.upload_layer_with(0, |_, _| Ok(())).unwrap_err();
    assert!(format!("{error:#}").contains("uploaded more than once"));
    assert!(duplicate.finish().is_err());
}

#[cfg(target_os = "linux")]
#[test]
fn all_selected_tensors_validate_before_a_session_exists() {
    let mut fixture = Fixture::valid(&[PREFIX_0, PREFIX_1]);
    fixture
        .header
        .remove(&format!("{PREFIX_1}.mlp.down_proj.w3_weight_scale"));
    let artifact = fixture.finish();
    let temp = TempArtifact::new(&artifact);
    let prefixes = vec![PREFIX_0.to_owned(), PREFIX_1.to_owned()];
    let admitted_request = request(&temp.path, &artifact, "0,1", 2);
    assert!(W3SidecarSession::prepare(admitted_request, &prefixes, 16, 16).is_err());

    let mut fixture = Fixture::valid(&[PREFIX_0]);
    fixture.header.insert(
        format!("{PREFIX_0}.mlp.unexpected"),
        json!({"dtype":"U8","shape":[1],"data_offsets":[0,1]}),
    );
    let artifact = fixture.finish();
    let temp = TempArtifact::new(&artifact);
    let admitted_request = request(&temp.path, &artifact, "0", 1);
    assert!(W3SidecarSession::prepare(admitted_request, &[PREFIX_0.to_owned()], 16, 16).is_err());
}

#[cfg(target_os = "linux")]
#[test]
fn malformed_metadata_scale_and_overlapping_spans_fail_closed() {
    let mut wrong_shape = Fixture::valid(&[PREFIX_0]);
    wrong_shape.header[&format!("{PREFIX_0}.mlp.gate_proj.w3_weight")]["shape"] = json!([16, 7]);
    assert_prepare_fails(wrong_shape.finish());

    let mut bad_scale = Fixture::valid(&[PREFIX_0]);
    bad_scale.replace_scale2(PREFIX_0, "up_proj", f32::NAN);
    assert_prepare_fails(bad_scale.finish());
    let mut zero_scale = Fixture::valid(&[PREFIX_0]);
    zero_scale.replace_scale2(PREFIX_0, "down_proj", 0.0);
    assert_prepare_fails(zero_scale.finish());

    let mut overlap = Fixture::valid(&[PREFIX_0]);
    let gate_offsets =
        overlap.header[&format!("{PREFIX_0}.mlp.gate_proj.w3_weight")]["data_offsets"].clone();
    overlap.header[&format!("{PREFIX_0}.mlp.up_proj.w3_weight")]["data_offsets"] = gate_offsets;
    assert_prepare_fails(overlap.finish());
}

#[cfg(target_os = "linux")]
#[test]
fn dtype_descending_oob_and_wrong_span_mutations_fail_specifically() {
    let name = format!("{PREFIX_0}.mlp.gate_proj.w3_weight");

    let mut wrong_dtype = Fixture::valid(&[PREFIX_0]);
    wrong_dtype.header[&name]["dtype"] = json!("F32");
    assert_prepare_error_contains(wrong_dtype.finish(), "wrong dtype");

    let mut descending = Fixture::valid(&[PREFIX_0]);
    descending.header[&name]["data_offsets"] = json!([1, 0]);
    assert_prepare_error_contains(descending.finish(), "descending data offsets");

    let mut out_of_bounds = Fixture::valid(&[PREFIX_0]);
    let beyond_data = out_of_bounds.data.len() + 1;
    out_of_bounds.header[&name]["data_offsets"] = json!([0, beyond_data]);
    assert_prepare_error_contains(out_of_bounds.finish(), "tensor payload is out of bounds");

    let mut wrong_span = Fixture::valid(&[PREFIX_0]);
    wrong_span.header[&name]["data_offsets"] = json!([0, 95]);
    assert_prepare_error_contains(wrong_span.finish(), "tensor span has the wrong length");
}

#[cfg(target_os = "linux")]
#[test]
fn duplicate_json_keys_bad_identity_and_link_aliases_fail_closed() {
    let duplicate = Fixture::valid(&[PREFIX_0]).finish_with_duplicate_dtype();
    assert_prepare_error_contains(duplicate, "duplicate JSON object key");

    let artifact = Fixture::valid(&[PREFIX_0]).finish();
    let temp = TempArtifact::new(&artifact);
    let wrong_digest = "00".repeat(32);
    let admitted_request = W3SidecarRequest::from_values(
        1,
        Some("0"),
        temp.path.to_str(),
        Some(&wrong_digest),
        Some(&artifact.len().to_string()),
    )
    .unwrap()
    .unwrap();
    assert!(W3SidecarSession::prepare(admitted_request, &[PREFIX_0.to_owned()], 16, 16).is_err());

    let wrong_size = W3SidecarRequest::from_values(
        1,
        Some("0"),
        temp.path.to_str(),
        Some(&hex(sha256(&artifact))),
        Some(&(artifact.len() as u64 + 1).to_string()),
    )
    .unwrap()
    .unwrap();
    assert!(W3SidecarSession::prepare(wrong_size, &[PREFIX_0.to_owned()], 16, 16).is_err());

    use std::os::unix::fs::symlink;
    let hardlink = temp.sibling("hardlink");
    std::fs::hard_link(&temp.path, &hardlink).unwrap();
    let hardlink_request = request(&temp.path, &artifact, "0", 1);
    assert!(W3SidecarSession::prepare(hardlink_request, &[PREFIX_0.to_owned()], 16, 16).is_err());
    std::fs::remove_file(&hardlink).unwrap();

    let symlink_path = temp.sibling("symlink");
    symlink(&temp.path, &symlink_path).unwrap();
    let symlink_request = request(&symlink_path, &artifact, "0", 1);
    assert!(W3SidecarSession::prepare(symlink_request, &[PREFIX_0.to_owned()], 16, 16).is_err());
    std::fs::remove_file(symlink_path).unwrap();
}

#[cfg(target_os = "linux")]
fn assert_prepare_fails(artifact: Vec<u8>) {
    let temp = TempArtifact::new(&artifact);
    let admitted_request = request(&temp.path, &artifact, "0", 1);
    assert!(W3SidecarSession::prepare(admitted_request, &[PREFIX_0.to_owned()], 16, 16).is_err());
}

#[cfg(target_os = "linux")]
fn assert_prepare_error_contains(artifact: Vec<u8>, expected: &str) {
    let temp = TempArtifact::new(&artifact);
    let admitted_request = request(&temp.path, &artifact, "0", 1);
    let error = W3SidecarSession::prepare(admitted_request, &[PREFIX_0.to_owned()], 16, 16)
        .err()
        .expect("mutated W3 fixture must fail");
    assert!(
        format!("{error:#}").contains(expected),
        "expected {expected:?}, got {error:#}"
    );
}

pub(super) fn request(
    path: &Path,
    artifact: &[u8],
    layers: &str,
    total_layers: usize,
) -> W3SidecarRequest {
    W3SidecarRequest::from_values(
        total_layers,
        Some(layers),
        path.to_str(),
        Some(&hex(sha256(artifact))),
        Some(&artifact.len().to_string()),
    )
    .unwrap()
    .unwrap()
}

fn hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(super) struct Fixture {
    header: Map<String, Value>,
    data: Vec<u8>,
    scale_offsets: BTreeMap<(String, String), (usize, usize)>,
}

pub(super) fn patterned_bytes(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|index| seed.wrapping_add((index as u8).wrapping_mul(17)))
        .collect()
}

impl Fixture {
    pub(super) fn valid(prefixes: &[&str]) -> Self {
        let mut fixture = Self {
            header: Map::new(),
            data: Vec::new(),
            scale_offsets: BTreeMap::new(),
        };
        for prefix in prefixes {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                fixture.add_tensor(
                    format!("{prefix}.mlp.{projection}.w3_weight"),
                    "U8",
                    &[16, 6],
                    vec![0x5a; 96],
                );
                fixture.add_tensor(
                    format!("{prefix}.mlp.{projection}.w3_weight_scale"),
                    "U8",
                    &[16, 1],
                    vec![0x38; 16],
                );
                let offset = fixture.add_tensor(
                    format!("{prefix}.mlp.{projection}.w3_weight_scale_2"),
                    "F32",
                    &[1],
                    1.0_f32.to_le_bytes().to_vec(),
                );
                fixture
                    .scale_offsets
                    .insert(((*prefix).to_owned(), projection.to_owned()), offset);
            }
        }
        fixture
    }

    pub(super) fn asymmetric() -> Self {
        let mut fixture = Self {
            header: Map::new(),
            data: Vec::new(),
            scale_offsets: BTreeMap::new(),
        };
        for (projection, n, k, packed_seed, scale_seed, scale2) in [
            ("gate_proj", 32, 16, 0x10, 0x40, 1.25_f32),
            ("up_proj", 32, 16, 0x20, 0x50, 2.5_f32),
            ("down_proj", 16, 32, 0x30, 0x60, 3.75_f32),
        ] {
            fixture.add_tensor(
                format!("{PREFIX_0}.mlp.{projection}.w3_weight"),
                "U8",
                &[n, 3 * k / 8],
                patterned_bytes(n * 3 * k / 8, packed_seed),
            );
            fixture.add_tensor(
                format!("{PREFIX_0}.mlp.{projection}.w3_weight_scale"),
                "U8",
                &[n, k / 16],
                patterned_bytes(n * k / 16, scale_seed),
            );
            let offset = fixture.add_tensor(
                format!("{PREFIX_0}.mlp.{projection}.w3_weight_scale_2"),
                "F32",
                &[1],
                scale2.to_le_bytes().to_vec(),
            );
            fixture
                .scale_offsets
                .insert((PREFIX_0.to_owned(), projection.to_owned()), offset);
        }
        fixture
    }

    fn add_tensor(
        &mut self,
        name: String,
        dtype: &str,
        shape: &[usize],
        bytes: Vec<u8>,
    ) -> (usize, usize) {
        let start = self.data.len();
        self.data.extend_from_slice(&bytes);
        let end = self.data.len();
        self.header.insert(
            name,
            json!({"dtype":dtype,"shape":shape,"data_offsets":[start,end]}),
        );
        (start, end)
    }

    fn replace_scale2(&mut self, prefix: &str, projection: &str, value: f32) {
        let (start, end) = self.scale_offsets[&(prefix.to_owned(), projection.to_owned())];
        self.data[start..end].copy_from_slice(&value.to_le_bytes());
    }

    fn finish_with_duplicate_dtype(self) -> Vec<u8> {
        let header = serde_json::to_string(&Value::Object(self.header)).unwrap();
        let duplicate = header.replacen("\"dtype\":\"U8\"", "\"dtype\":\"U8\",\"dtype\":\"U8\"", 1);
        assert_ne!(header, duplicate);
        let mut artifact = Vec::with_capacity(8 + duplicate.len() + self.data.len());
        artifact.extend_from_slice(&(duplicate.len() as u64).to_le_bytes());
        artifact.extend_from_slice(duplicate.as_bytes());
        artifact.extend_from_slice(&self.data);
        artifact
    }

    pub(super) fn finish(self) -> Vec<u8> {
        let header = serde_json::to_vec(&Value::Object(self.header)).unwrap();
        let mut artifact = Vec::with_capacity(8 + header.len() + self.data.len());
        artifact.extend_from_slice(&(header.len() as u64).to_le_bytes());
        artifact.extend_from_slice(&header);
        artifact.extend_from_slice(&self.data);
        artifact
    }
}

#[cfg(target_os = "linux")]
pub(super) struct TempArtifact {
    pub(super) path: PathBuf,
}

#[cfg(target_os = "linux")]
impl TempArtifact {
    pub(super) fn new(bytes: &[u8]) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let sequence = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "atlas-w3-session-{}-{sequence}.safetensors",
            std::process::id()
        ));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(bytes).unwrap();
        file.sync_all().unwrap();
        drop(file);
        Self { path }
    }

    fn sibling(&self, suffix: &str) -> PathBuf {
        self.path.with_extension(format!("safetensors.{suffix}"))
    }
}

#[cfg(target_os = "linux")]
impl Drop for TempArtifact {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}
