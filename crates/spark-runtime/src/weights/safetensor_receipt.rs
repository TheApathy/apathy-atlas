// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed CPU receipt for small BF16 safetensors checkpoints.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use anyhow::{Context, Result, ensure};
use ring::digest::{Context as DigestContext, SHA256};
use safetensors::tensor::Metadata;

const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
const SCAN_BYTES: usize = 1024 * 1024;

/// Content-bound proof that every value in an exact BF16 checkpoint is finite.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bf16FinitenessReceipt {
    pub file_name: String,
    pub file_bytes: u64,
    pub file_sha256: String,
    pub tensor_count: usize,
    pub element_count: usize,
}

/// Attest one canonical `model.safetensors` before its bytes reach a GPU.
///
/// This intentionally supports only a single-file checkpoint. Callers use it
/// for small, exact-schema sidecars/drafters; large target checkpoints stay on
/// the ordinary sharded loader path.
pub fn attest_exact_bf16_safetensors(
    model_dir: &Path,
    expected: &BTreeMap<String, Vec<usize>>,
) -> Result<Bf16FinitenessReceipt> {
    ensure!(
        !expected.is_empty(),
        "BF16 receipt manifest must not be empty"
    );
    require_canonical_single_file(model_dir)?;
    let path = model_dir.join("model.safetensors");
    let file = File::open(&path)
        .with_context(|| format!("opening BF16 receipt candidate {}", path.display()))?;
    let initial_len = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    let mut digest = DigestContext::new(&SHA256);

    let mut size_bytes = [0u8; 8];
    reader.read_exact(&mut size_bytes)?;
    digest.update(&size_bytes);
    let header_len = u64::from_le_bytes(size_bytes);
    ensure!(
        header_len <= MAX_HEADER_BYTES,
        "safetensors header is {header_len} bytes; limit is {MAX_HEADER_BYTES}"
    );
    let header_len_usize = usize::try_from(header_len).context("header length overflows usize")?;
    let mut header = vec![0u8; header_len_usize];
    reader.read_exact(&mut header)?;
    digest.update(&header);
    let metadata: Metadata =
        serde_json::from_slice(&header).context("invalid safetensors header")?;

    let infos = metadata.tensors();
    let observed: BTreeSet<_> = infos.keys().cloned().collect();
    let wanted: BTreeSet<_> = expected.keys().cloned().collect();
    let missing: Vec<_> = wanted.difference(&observed).cloned().collect();
    let unexpected: Vec<_> = observed.difference(&wanted).cloned().collect();
    ensure!(
        missing.is_empty() && unexpected.is_empty(),
        "BF16 receipt tensor manifest mismatch: missing={missing:?}, unexpected={unexpected:?}"
    );

    let mut element_count = 0usize;
    for (name, shape) in expected {
        let info = infos
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing receipt tensor `{name}`"))?;
        ensure!(
            info.dtype == safetensors::Dtype::BF16,
            "receipt tensor `{name}` has dtype {:?}; expected BF16",
            info.dtype
        );
        ensure!(
            info.shape == *shape,
            "receipt tensor `{name}` has shape {:?}; expected {shape:?}",
            info.shape
        );
        let elements = shape.iter().try_fold(1usize, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or_else(|| anyhow::anyhow!("receipt tensor `{name}` size overflow"))
        })?;
        element_count = element_count
            .checked_add(elements)
            .context("receipt element count overflow")?;
    }
    let data_start = 8u64
        .checked_add(header_len)
        .context("receipt data offset overflow")?;
    let expected_len = data_start
        .checked_add(u64::try_from(metadata.data_len())?)
        .context("receipt file length overflow")?;
    ensure!(
        initial_len == expected_len,
        "safetensors file length is {initial_len}; metadata requires {expected_len}"
    );

    let mut buffer = vec![0u8; SCAN_BYTES];
    for name in metadata.offset_keys() {
        let info = infos
            .get(&name)
            .ok_or_else(|| anyhow::anyhow!("receipt metadata lost tensor `{name}`"))?;
        let mut remaining = info.data_offsets.1 - info.data_offsets.0;
        let mut tensor_element = 0usize;
        while remaining > 0 {
            let take = remaining.min(buffer.len());
            let bytes = &mut buffer[..take];
            reader.read_exact(bytes)?;
            digest.update(bytes);
            for pair in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([pair[0], pair[1]]);
                ensure!(
                    bits & 0x7f80 != 0x7f80,
                    "receipt tensor `{name}` element {tensor_element} is non-finite BF16 (0x{bits:04x})"
                );
                tensor_element += 1;
            }
            ensure!(
                bytes.len().is_multiple_of(2),
                "receipt tensor `{name}` has an odd BF16 byte count"
            );
            remaining -= take;
        }
    }
    ensure!(
        reader.read(&mut buffer[..1])? == 0,
        "safetensors has trailing bytes"
    );
    ensure!(
        reader.get_ref().metadata()?.len() == initial_len,
        "safetensors length changed during receipt scan"
    );

    Ok(Bf16FinitenessReceipt {
        file_name: "model.safetensors".into(),
        file_bytes: initial_len,
        file_sha256: hex_digest(digest.finish().as_ref()),
        tensor_count: expected.len(),
        element_count,
    })
}

fn require_canonical_single_file(model_dir: &Path) -> Result<()> {
    let mut safetensors = Vec::new();
    for entry in std::fs::read_dir(model_dir)
        .with_context(|| format!("reading BF16 receipt directory {}", model_dir.display()))?
    {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".safetensors") {
            safetensors.push(name);
        }
    }
    safetensors.sort();
    ensure!(
        safetensors == ["model.safetensors"],
        "native BF16 receipt requires only model.safetensors; found {safetensors:?}"
    );
    for index in [
        "model.safetensors.index.json",
        "consolidated.safetensors.index.json",
    ] {
        ensure!(
            !model_dir.join(index).exists(),
            "native BF16 receipt forbids sharded index {index}"
        );
    }
    Ok(())
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0xf) as usize] as char);
    }
    output
}

#[cfg(test)]
#[path = "safetensor_receipt_tests.rs"]
mod tests;
