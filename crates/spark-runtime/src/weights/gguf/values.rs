// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};

use super::GgufValue;

const MAX_ARRAY_LEN: u64 = 2_000_000;
const MAX_METADATA_COUNT: u64 = 100_000;
const MAX_METADATA_KEY_LEN: u64 = 256;
const MAX_METADATA_STRING_LEN: u64 = 1024 * 1024;
const MIN_METADATA_ENTRY_BYTES: u64 = 13;

pub(super) struct DirectoryBudget {
    remaining: usize,
}

impl DirectoryBudget {
    pub(super) fn new(bytes: usize) -> Self {
        Self { remaining: bytes }
    }

    pub(super) fn charge(&mut self, bytes: usize) -> Result<()> {
        self.remaining = self
            .remaining
            .checked_sub(bytes)
            .context("GGUF retained directory exceeds memory budget")?;
        Ok(())
    }
}

pub(super) fn bytes<const N: usize>(reader: &mut impl Read) -> Result<[u8; N]> {
    let mut out = [0; N];
    reader
        .read_exact(&mut out)
        .context("truncated GGUF header")?;
    Ok(out)
}

fn u8v(reader: &mut impl Read) -> Result<u8> {
    Ok(bytes::<1>(reader)?[0])
}
fn u16v(reader: &mut impl Read) -> Result<u16> {
    Ok(u16::from_le_bytes(bytes(reader)?))
}
pub(super) fn u32v(reader: &mut impl Read) -> Result<u32> {
    Ok(u32::from_le_bytes(bytes(reader)?))
}
pub(super) fn u64v(reader: &mut impl Read) -> Result<u64> {
    Ok(u64::from_le_bytes(bytes(reader)?))
}

fn remaining(reader: &mut impl Seek, file_len: u64) -> Result<u64> {
    file_len
        .checked_sub(reader.stream_position()?)
        .context("GGUF directory exceeds file bounds")
}

pub(super) fn string(
    reader: &mut (impl Read + Seek),
    what: &str,
    max_len: u64,
    file_len: u64,
) -> Result<String> {
    let len = u64v(reader)?;
    if len > max_len {
        bail!("GGUF {what} string length {len} exceeds {max_len}-byte limit");
    }
    if len > remaining(reader, file_len)? {
        bail!("truncated GGUF {what} string");
    }
    let mut value = vec![0; usize::try_from(len)?];
    reader
        .read_exact(&mut value)
        .context("truncated GGUF string")?;
    String::from_utf8(value).with_context(|| format!("GGUF {what} is not UTF-8"))
}

fn skip(reader: &mut (impl Read + Seek), len: u64, file_len: u64) -> Result<()> {
    if len > remaining(reader, file_len)? {
        bail!("GGUF metadata exceeds directory bounds");
    }
    let end = reader
        .stream_position()?
        .checked_add(len)
        .context("GGUF skip offset overflow")?;
    reader.seek(SeekFrom::Start(end))?;
    Ok(())
}

fn validate_array_type(ty: u32) -> Result<()> {
    match ty {
        0..=8 | 10..=12 => Ok(()),
        9 => bail!("nested GGUF metadata arrays are unsupported"),
        _ => bail!("unsupported GGUF metadata array element type {ty}"),
    }
}

fn skip_value(reader: &mut (impl Read + Seek), ty: u32, file_len: u64) -> Result<()> {
    match ty {
        0 | 1 => skip(reader, 1, file_len),
        2 | 3 => skip(reader, 2, file_len),
        4..=6 => skip(reader, 4, file_len),
        7 => match u8v(reader)? {
            0 | 1 => Ok(()),
            other => bail!("invalid GGUF boolean {other}"),
        },
        8 => {
            let len = u64v(reader)?;
            if len > MAX_METADATA_STRING_LEN {
                bail!("GGUF array string length {len} exceeds limit");
            }
            skip(reader, len, file_len)
        }
        10..=12 => skip(reader, 8, file_len),
        9 => bail!("nested GGUF metadata arrays are unsupported"),
        _ => bail!("unsupported GGUF metadata type {ty}"),
    }
}

fn value(
    reader: &mut (impl Read + Seek),
    file_len: u64,
    budget: &mut DirectoryBudget,
) -> Result<GgufValue> {
    let ty = u32v(reader)?;
    Ok(match ty {
        0 => GgufValue::Unsigned(u8v(reader)? as u64),
        1 => GgufValue::Signed(i8::from_le_bytes(bytes(reader)?) as i64),
        2 => GgufValue::Unsigned(u16v(reader)? as u64),
        3 => GgufValue::Signed(i16::from_le_bytes(bytes(reader)?) as i64),
        4 => GgufValue::Unsigned(u32v(reader)? as u64),
        5 => GgufValue::Signed(i32::from_le_bytes(bytes(reader)?) as i64),
        6 => GgufValue::Float(f32::from_le_bytes(bytes(reader)?) as f64),
        7 => match u8v(reader)? {
            0 => GgufValue::Bool(false),
            1 => GgufValue::Bool(true),
            other => bail!("invalid GGUF boolean {other}"),
        },
        8 => {
            let value = string(reader, "metadata value", MAX_METADATA_STRING_LEN, file_len)?;
            budget.charge(value.len())?;
            GgufValue::String(value)
        }
        9 => {
            let element_type = u32v(reader)?;
            validate_array_type(element_type)?;
            let len = u64v(reader)?;
            if len > MAX_ARRAY_LEN {
                bail!("GGUF metadata array length {len} exceeds limit");
            }
            let width = match element_type {
                0 | 1 => Some(1u64),
                2 | 3 => Some(2),
                4..=6 => Some(4),
                10..=12 => Some(8),
                7 | 8 => None,
                _ => unreachable!("array type was validated"),
            };
            if let Some(width) = width {
                skip(
                    reader,
                    len.checked_mul(width).context("GGUF array byte overflow")?,
                    file_len,
                )?;
            } else {
                for _ in 0..len {
                    skip_value(reader, element_type, file_len)?;
                }
            }
            GgufValue::Array { element_type, len }
        }
        10 => GgufValue::Unsigned(u64v(reader)?),
        11 => GgufValue::Signed(i64::from_le_bytes(bytes(reader)?)),
        12 => GgufValue::Float(f64::from_le_bytes(bytes(reader)?)),
        _ => bail!("unsupported GGUF metadata type {ty}"),
    })
}

pub(super) fn read_metadata(
    reader: &mut (impl Read + Seek),
    count: u64,
    file_len: u64,
    budget: &mut DirectoryBudget,
) -> Result<BTreeMap<String, GgufValue>> {
    if count > MAX_METADATA_COUNT || count > remaining(reader, file_len)? / MIN_METADATA_ENTRY_BYTES
    {
        bail!("GGUF metadata count exceeds bounded directory capacity");
    }
    let mut metadata = BTreeMap::new();
    for _ in 0..count {
        let key = string(reader, "metadata key", MAX_METADATA_KEY_LEN, file_len)?;
        budget.charge(
            64usize
                .checked_add(key.len())
                .context("GGUF metadata budget overflow")?,
        )?;
        let entry = value(reader, file_len, budget)?;
        if metadata.insert(key, entry).is_some() {
            bail!("duplicate GGUF metadata key");
        }
    }
    Ok(metadata)
}
