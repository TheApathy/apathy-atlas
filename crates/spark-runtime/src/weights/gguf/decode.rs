// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, bail};
use std::collections::HashSet;
use std::io::{Read, Seek, SeekFrom};
use std::mem::size_of;

use super::values::{DirectoryBudget, bytes, read_metadata, string, u32v, u64v};
use super::{GgmlType, GgufHeader, GgufTensorInfo, GgufValue};

const MAX_DIRECTORY_BYTES: u64 = 64 * 1024 * 1024;
const MAX_RETAINED_DIRECTORY_BYTES: usize = 64 * 1024 * 1024;
const MAX_TENSOR_COUNT: u64 = 100_000;
const MAX_TENSOR_NAME_LEN: u64 = 64;
const MIN_TENSOR_ENTRY_BYTES: u64 = 33;

fn remaining(reader: &mut impl Seek, limit: u64) -> Result<u64> {
    limit
        .checked_sub(reader.stream_position()?)
        .context("GGUF tensor directory exceeds bounded file region")
}

fn read_tensors(
    reader: &mut (impl Read + Seek),
    count: u64,
    alignment: u64,
    directory_limit: u64,
    budget: &mut DirectoryBudget,
) -> Result<Vec<GgufTensorInfo>> {
    if count > MAX_TENSOR_COUNT
        || count > remaining(reader, directory_limit)? / MIN_TENSOR_ENTRY_BYTES
    {
        bail!("GGUF tensor count exceeds bounded directory capacity");
    }
    let count = usize::try_from(count)?;
    budget.charge(
        count
            .checked_mul(size_of::<GgufTensorInfo>() + 64)
            .context("GGUF tensor directory budget overflow")?,
    )?;
    let mut names = HashSet::with_capacity(count);
    let mut tensors = Vec::with_capacity(count);
    for _ in 0..count {
        let name = string(reader, "tensor name", MAX_TENSOR_NAME_LEN, directory_limit)?;
        if name.is_empty() {
            bail!("GGUF tensor name must not be empty");
        }
        budget.charge(name.len())?;
        if !names.insert(name.clone()) {
            bail!("duplicate GGUF tensor name");
        }
        let dimensions_count = u32v(reader)?;
        if !(1..=4).contains(&dimensions_count) {
            bail!("GGUF tensor has invalid dimension count {dimensions_count}");
        }
        budget.charge(dimensions_count as usize * size_of::<u64>())?;
        let mut dimensions = Vec::with_capacity(dimensions_count as usize);
        let mut elements = 1u64;
        for _ in 0..dimensions_count {
            let dimension = u64v(reader)?;
            if dimension == 0 {
                bail!("GGUF tensor has a zero dimension");
            }
            elements = elements
                .checked_mul(dimension)
                .context("GGUF tensor element count overflow")?;
            dimensions.push(dimension);
        }
        let ggml_type = GgmlType::from_raw(u32v(reader)?)?;
        let block = ggml_type.block_size();
        if !dimensions[0].is_multiple_of(block) {
            bail!("GGUF tensor row width violates GGML block size {block}");
        }
        let offset = u64v(reader)?;
        if !offset.is_multiple_of(alignment) {
            bail!("GGUF tensor offset is not aligned to {alignment}");
        }
        tensors.push(GgufTensorInfo {
            name,
            dimensions,
            ggml_type,
            offset,
            byte_len: ggml_type.byte_len(elements)?,
        });
    }
    Ok(tensors)
}

pub(super) fn read_gguf_header_with_len(
    reader: &mut (impl Read + Seek),
    file_len: u64,
) -> Result<GgufHeader> {
    let directory_limit = file_len.min(MAX_DIRECTORY_BYTES);
    let mut budget = DirectoryBudget::new(MAX_RETAINED_DIRECTORY_BYTES);
    if &bytes::<4>(reader)? != b"GGUF" {
        bail!("invalid GGUF magic");
    }
    let version = u32v(reader)?;
    if version != 3 {
        bail!("unsupported GGUF version {version}; expected 3");
    }
    let tensor_count = u64v(reader)?;
    let metadata_count = u64v(reader)?;
    let metadata = read_metadata(reader, metadata_count, directory_limit, &mut budget)?;
    let alignment = match metadata.get("general.alignment") {
        Some(GgufValue::Unsigned(value)) => *value,
        Some(_) => bail!("GGUF general.alignment must be unsigned"),
        None => 32,
    };
    if !(8..=1_048_576).contains(&alignment) || !alignment.is_multiple_of(8) {
        bail!("invalid GGUF alignment {alignment}");
    }
    let tensors = read_tensors(
        reader,
        tensor_count,
        alignment,
        directory_limit,
        &mut budget,
    )?;
    let directory_end = reader.stream_position()?;
    let data_offset = directory_end
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .context("GGUF data offset overflow")?;
    let mut ranges = Vec::with_capacity(tensors.len());
    for tensor in &tensors {
        let start = data_offset
            .checked_add(tensor.offset)
            .context("GGUF tensor offset overflow")?;
        let end = start
            .checked_add(tensor.byte_len)
            .context("GGUF tensor range overflow")?;
        if end > file_len {
            bail!("GGUF tensor exceeds file bounds");
        }
        ranges.push((start, end));
    }
    ranges.sort_unstable_by_key(|range| range.0);
    for pair in ranges.windows(2) {
        if pair[0].1 > pair[1].0 {
            bail!("GGUF tensor payloads overlap");
        }
    }
    Ok(GgufHeader {
        version,
        alignment,
        data_offset,
        file_len,
        metadata,
        tensors,
    })
}

pub fn read_gguf_header(reader: &mut (impl Read + Seek)) -> Result<GgufHeader> {
    let file_len = reader.seek(SeekFrom::End(0))?;
    reader.seek(SeekFrom::Start(0))?;
    read_gguf_header_with_len(reader, file_len)
}
