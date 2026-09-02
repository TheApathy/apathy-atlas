// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, bail};
use std::collections::{BTreeMap, HashSet};

use super::{GgufHeader, GgufTensorInfo, GgufValue};

#[derive(Debug, Clone, Copy)]
pub struct GgufShardRef<'a> {
    pub header: &'a GgufHeader,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocatedTensor {
    pub shard_no: usize,
    pub info: GgufTensorInfo,
}

#[derive(Clone, PartialEq, Eq)]
pub struct GgufDirectory {
    pub architecture: String,
    pub split_count: usize,
    pub tensors: BTreeMap<String, LocatedTensor>,
}

impl std::fmt::Debug for GgufDirectory {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GgufDirectory")
            .field("architecture_bytes", &self.architecture.len())
            .field("split_count", &self.split_count)
            .field("tensors", &self.tensors.len())
            .finish()
    }
}

/// Read a split-header count as a `u64`.
///
/// GGUF has no single canonical width for these fields and writers disagree:
/// llama.cpp's `gguf_split` emits `split.count`/`split.no` as UINT16 but
/// `split.tensors.count` as **INT32**, so every real Unsloth GLM-5.3 shard
/// stores a signed value there. Accepting only `Unsigned` rejected genuine
/// checkpoints — including the pinned Q2_K_XL recipe — so a non-negative
/// `Signed` is admitted here. Negative values remain a hard error: they cannot
/// be a count, and silently saturating one to zero would turn a corrupt header
/// into an empty-but-valid-looking directory.
fn count_field(header: &GgufHeader, key: &str) -> Result<u64> {
    match header.metadata.get(key) {
        Some(GgufValue::Unsigned(value)) => Ok(*value),
        Some(GgufValue::Signed(value)) => {
            u64::try_from(*value).map_err(|_| anyhow::anyhow!("GGUF {key} is negative: {value}"))
        }
        Some(_) => bail!("GGUF {key} must be an integer count"),
        None => bail!("GGUF shard missing {key}"),
    }
}

pub fn assemble_split_shards(shards: &[GgufShardRef<'_>]) -> Result<GgufDirectory> {
    if shards.is_empty() {
        bail!("GGUF shard set must not be empty");
    }
    let split_count = usize::try_from(count_field(shards[0].header, "split.count")?)?;
    let total_tensors = usize::try_from(count_field(shards[0].header, "split.tensors.count")?)?;
    if split_count != shards.len() {
        bail!(
            "GGUF split.count {split_count} does not match {} shards",
            shards.len()
        );
    }

    let mut shard_numbers = HashSet::new();
    let mut tensors = BTreeMap::new();
    let mut architecture = None;
    for shard in shards {
        let header = shard.header;
        if usize::try_from(count_field(header, "split.count")?)? != split_count
            || usize::try_from(count_field(header, "split.tensors.count")?)? != total_tensors
        {
            bail!("GGUF split metadata disagrees across shards");
        }
        let shard_no = usize::try_from(count_field(header, "split.no")?)?;
        if shard_no >= split_count || !shard_numbers.insert(shard_no) {
            bail!("GGUF split.no is missing, duplicate, or out of range");
        }
        if let Some(value) = header.metadata.get("general.architecture") {
            let GgufValue::String(value) = value else {
                bail!("GGUF general.architecture must be a string");
            };
            if architecture.as_ref().is_some_and(|known| known != value) {
                bail!("GGUF architecture disagrees across shards");
            }
            architecture = Some(value.clone());
        }
        for info in &header.tensors {
            let name = info.name.clone();
            if tensors
                .insert(
                    name.clone(),
                    LocatedTensor {
                        shard_no,
                        info: info.clone(),
                    },
                )
                .is_some()
            {
                bail!("duplicate GGUF tensor name across shards");
            }
        }
    }
    if shard_numbers.len() != split_count {
        bail!("GGUF shard numbers are incomplete");
    }
    if tensors.len() != total_tensors {
        bail!(
            "GGUF tensor total {} does not match split metadata {total_tensors}",
            tensors.len()
        );
    }
    Ok(GgufDirectory {
        architecture: architecture.context("GGUF shard set has no general.architecture")?,
        split_count,
        tensors,
    })
}
