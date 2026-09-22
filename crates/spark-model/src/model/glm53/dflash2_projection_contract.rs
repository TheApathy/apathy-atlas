// SPDX-License-Identifier: AGPL-3.0-only
//! Arithmetic identity only; cursor, poison, stream and completion stay in KvPrefix.
use super::kv_prefix::{KvPrefix, KvPrefixIo};
use anyhow::{Context, Result, ensure};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionFamily {
    Original,
    StableTc,
    StableGemv,
}

#[derive(Default)]
pub struct ProjectionBinding {
    family: Option<ProjectionFamily>,
}

impl ProjectionBinding {
    pub fn new() -> Self {
        Self { family: None }
    }
    pub fn family(&self) -> Option<ProjectionFamily> {
        self.family
    }
    pub fn admit(&self, family: ProjectionFamily, prefix: &KvPrefix) -> Result<()> {
        ensure!(
            !prefix.pending() && !prefix.poisoned(),
            "projection family cannot reuse pending or poisoned KV state"
        );
        if let Some(previous) = self.family() {
            ensure!(
                previous == family,
                "changing committed projection family requires reset"
            );
        } else {
            ensure!(
                !prefix.has_completed_rows(),
                "completed KV prefix has no admitted projection family"
            );
        }
        Ok(())
    }
    pub fn select(&mut self, family: ProjectionFamily, prefix: &KvPrefix) -> Result<()> {
        self.admit(family, prefix)?;
        self.family = Some(family);
        Ok(())
    }
    pub fn reset(&mut self, prefix: &mut KvPrefix, io: &mut dyn KvPrefixIo) -> Result<()> {
        // Failure or unwind leaves the family and real pending owner intact.
        prefix.reset(io)?;
        self.family = None;
        Ok(())
    }
}

/// Validate only derived BF16 operands. The caller supplies admitted model
/// geometry; this does not select an architecture or perform allocation/I/O.
pub fn checked_spans(
    rows: u32,
    max_rows: u32,
    hidden: u32,
    width: u32,
    input: (u64, usize),
    weight: (u64, usize),
    output: (u64, usize),
) -> Result<()> {
    ensure!(
        rows > 0 && rows <= max_rows && hidden > 0 && width > 0,
        "invalid committed projection geometry"
    );
    let bytes = |a: u32, b: u32| -> Result<usize> {
        usize::try_from(a)?
            .checked_mul(usize::try_from(b)?)
            .and_then(|v| v.checked_mul(2))
            .context("committed BF16 extent overflow")
    };
    let a = span(input, bytes(rows, hidden)?)?;
    let b = span(weight, bytes(width, hidden)?)?;
    let c = span(output, bytes(rows, width)?)?;
    let overlaps = |a: (u64, u64), b: (u64, u64)| a.0 < b.1 && b.0 < a.1;
    ensure!(
        !overlaps(a, b) && !overlaps(a, c) && !overlaps(b, c),
        "committed projection operands alias"
    );
    Ok(())
}

fn span(buffer: (u64, usize), bytes: usize) -> Result<(u64, u64)> {
    let (pointer, capacity) = buffer;
    ensure!(
        pointer > 0 && pointer % 2 == 0 && capacity % 2 == 0 && bytes <= capacity,
        "invalid committed projection BF16 address/capacity"
    );
    let end = pointer
        .checked_add(u64::try_from(bytes)?)
        .context("committed projection pointer overflow")?;
    Ok((pointer, end))
}
