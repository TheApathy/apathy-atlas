// SPDX-License-Identifier: AGPL-3.0-only

use std::io::Cursor;

use super::{GgmlType, GgufValue, read_gguf_header};

fn u32le(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn u64le(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn string(out: &mut Vec<u8>, value: &str) {
    u64le(out, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}

fn build_file(
    magic: &[u8; 4],
    version: u32,
    alignment: u32,
    tensors: &[(&str, u64, u64)],
    payload_len: usize,
) -> Vec<u8> {
    let tensors: Vec<_> = tensors
        .iter()
        .map(|(name, elements, offset)| (*name, vec![*elements], 8, *offset))
        .collect();
    build_file_dims(magic, version, alignment, &tensors, payload_len)
}

fn build_file_dims(
    magic: &[u8; 4],
    version: u32,
    alignment: u32,
    tensors: &[(&str, Vec<u64>, u32, u64)],
    payload_len: usize,
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(magic);
    u32le(&mut out, version);
    u64le(&mut out, tensors.len() as u64);
    u64le(&mut out, 3);

    string(&mut out, "general.architecture");
    u32le(&mut out, 8); // GGUF_TYPE_STRING
    string(&mut out, "glm5next");
    string(&mut out, "general.alignment");
    u32le(&mut out, 4); // GGUF_TYPE_UINT32
    u32le(&mut out, alignment);
    string(&mut out, "tokenizer.ggml.tokens");
    u32le(&mut out, 9); // GGUF_TYPE_ARRAY
    u32le(&mut out, 8); // strings
    u64le(&mut out, 2);
    string(&mut out, "x");
    string(&mut out, "yz");

    for (name, dimensions, ggml_type, offset) in tensors {
        string(&mut out, name);
        u32le(&mut out, dimensions.len() as u32);
        for dimension in dimensions {
            u64le(&mut out, *dimension);
        }
        u32le(&mut out, *ggml_type);
        u64le(&mut out, *offset);
    }
    let aligned = out.len().div_ceil(alignment as usize) * alignment as usize;
    out.resize(aligned + payload_len, 0);
    out
}

fn build_single_array_metadata(element_type: u32, len: u64, contents: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    u32le(&mut out, 3);
    u64le(&mut out, 0);
    u64le(&mut out, 1);
    string(&mut out, "array");
    u32le(&mut out, 9);
    u32le(&mut out, element_type);
    u64le(&mut out, len);
    out.extend_from_slice(contents);
    out
}

#[test]
fn reads_valid_v3_directory_without_payload() {
    let bytes = build_file(b"GGUF", 3, 32, &[("token_embd.weight", 64, 0)], 68);
    let header = read_gguf_header(&mut Cursor::new(bytes)).unwrap();
    assert_eq!(header.version, 3);
    assert_eq!(header.alignment, 32);
    assert_eq!(
        header.metadata.get("general.architecture"),
        Some(&GgufValue::String("glm5next".to_string()))
    );
    assert_eq!(
        header.metadata.get("tokenizer.ggml.tokens"),
        Some(&GgufValue::Array {
            element_type: 8,
            len: 2
        })
    );
    assert_eq!(header.tensors[0].ggml_type, GgmlType::Q8_0);
    assert_eq!(header.tensors[0].dimensions, [64]);
    assert_eq!(header.tensors[0].byte_len, 68);
}

#[test]
fn rejects_magic_version_and_alignment() {
    for (magic, version, alignment, needle) in [
        (b"NOPE", 3, 32, "magic"),
        (b"GGUF", 2, 32, "version"),
        (b"GGUF", 3, 12, "alignment"),
    ] {
        let bytes = build_file(magic, version, alignment, &[("x", 32, 0)], 34);
        let err = read_gguf_header(&mut Cursor::new(bytes)).unwrap_err();
        assert!(format!("{err:#}").contains(needle), "{err:#}");
    }
}

#[test]
fn accepts_unpadded_metadata_only_shard_and_bounds_names() {
    let mut metadata_only = build_file(b"GGUF", 3, 32, &[], 0);
    while metadata_only.last() == Some(&0) {
        metadata_only.pop();
    }
    let header = read_gguf_header(&mut Cursor::new(metadata_only)).unwrap();
    assert!(header.tensors.is_empty());
    assert!(header.data_offset > header.file_len);

    let long_name = "x".repeat(65);
    let bytes = build_file(b"GGUF", 3, 32, &[(long_name.as_str(), 32, 0)], 34);
    let err = read_gguf_header(&mut Cursor::new(bytes)).unwrap_err();
    assert!(format!("{err:#}").contains("64-byte"), "{err:#}");
}

#[test]
fn rejects_duplicate_names_bad_blocks_and_offsets() {
    let cases = [
        (
            build_file(b"GGUF", 3, 32, &[("x", 32, 0), ("x", 32, 64)], 98),
            "duplicate",
        ),
        (build_file(b"GGUF", 3, 32, &[("x", 31, 0)], 34), "block"),
        (build_file(b"GGUF", 3, 32, &[("x", 32, 1)], 35), "aligned"),
        (build_file(b"GGUF", 3, 32, &[("x", 32, 64)], 34), "bounds"),
    ];
    for (bytes, needle) in cases {
        let err = read_gguf_header(&mut Cursor::new(bytes)).unwrap_err();
        assert!(format!("{err:#}").contains(needle), "{err:#}");
    }
}

#[test]
fn rejects_multidimensional_bad_row_block() {
    let bytes = build_file_dims(b"GGUF", 3, 32, &[("x", vec![1, 32], 8, 0)], 34);
    let err = read_gguf_header(&mut Cursor::new(bytes)).unwrap_err();
    assert!(format!("{err:#}").contains("row width"), "{err:#}");
}

#[test]
fn rejects_counts_and_strings_before_large_allocation() {
    let mut count = Vec::new();
    count.extend_from_slice(b"GGUF");
    u32le(&mut count, 3);
    u64le(&mut count, 100_000);
    u64le(&mut count, 0);
    let err = read_gguf_header(&mut Cursor::new(count)).unwrap_err();
    assert!(format!("{err:#}").contains("bounded directory capacity"));

    let mut string_value = Vec::new();
    string_value.extend_from_slice(b"GGUF");
    u32le(&mut string_value, 3);
    u64le(&mut string_value, 0);
    u64le(&mut string_value, 1);
    string(&mut string_value, "value");
    u32le(&mut string_value, 8);
    u64le(&mut string_value, 128);
    let err = read_gguf_header(&mut Cursor::new(string_value)).unwrap_err();
    assert!(format!("{err:#}").contains("truncated"), "{err:#}");
}

#[test]
fn rejects_unknown_empty_and_invalid_boolean_arrays() {
    let unknown = build_single_array_metadata(99, 0, &[]);
    let err = read_gguf_header(&mut Cursor::new(unknown)).unwrap_err();
    assert!(format!("{err:#}").contains("element type"), "{err:#}");

    let invalid_bool = build_single_array_metadata(7, 1, &[2]);
    let err = read_gguf_header(&mut Cursor::new(invalid_bool)).unwrap_err();
    assert!(format!("{err:#}").contains("boolean"), "{err:#}");
}
