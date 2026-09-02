// SPDX-License-Identifier: AGPL-3.0-only

use super::payload::{FileIdentity, OpenShard};
use super::*;
use anyhow::bail;
use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

fn files() -> Glm53Iq3Files {
    let path = std::env::temp_dir().join(format!(
        "atlas-glm53-payload-{}-{}",
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
    file.write_all(b"stream-me").unwrap();
    file.flush().unwrap();
    let identity = FileIdentity::capture(&file).unwrap();
    let info = GgufTensorInfo {
        name: "tensor".into(),
        dimensions: vec![8],
        ggml_type: GgmlType::I8,
        offset: 0,
        byte_len: 8,
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
            tensors: BTreeMap::from([("tensor".into(), LocatedTensor { shard_no: 0, info })]),
        },
        Glm53Iq3Summary {
            shards: 1,
            tensors: 1,
            tensor_bytes: 8,
        },
    )
    .unwrap()
}

#[test]
fn streams_bounded_chunks_from_retained_handle() {
    let mut files = files();
    let mut output = Vec::new();
    let mut offsets = Vec::new();
    files
        .stream_tensor("tensor", &mut [0; 3], |offset, bytes| {
            offsets.push(offset);
            output.extend_from_slice(bytes);
            Ok(())
        })
        .unwrap();
    assert_eq!(offsets, [0, 3, 6]);
    assert_eq!(output, b"stream-m");
}

#[test]
fn rejects_bad_buffers_missing_names_and_identity_drift() {
    let mut files = files();
    assert!(
        files
            .stream_tensor("tensor", &mut [], |_, _| Ok(()))
            .is_err()
    );
    let mut oversized = vec![0; 4 * 1024 * 1024 + 1];
    assert!(
        files
            .stream_tensor("tensor", &mut oversized, |_, _| Ok(()))
            .is_err()
    );
    assert!(
        files
            .stream_tensor("missing", &mut [0], |_, _| Ok(()))
            .is_err()
    );
    files.shards[0].file.set_len(17).unwrap();
    assert!(
        files
            .stream_tensor("tensor", &mut [0], |_, _| Ok(()))
            .is_err()
    );
}

#[test]
fn postcheck_runs_when_the_sink_fails() {
    let mut files = files();
    let mut alias = files.shards[0].file.try_clone().unwrap();
    let err = files
        .stream_tensor("tensor", &mut [0; 3], |_, _| {
            alias.write_all(b"x")?;
            bail!("sink failed")
        })
        .unwrap_err();
    assert!(format!("{err:#}").contains("changed while streaming"));
}
