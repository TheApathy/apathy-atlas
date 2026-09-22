// SPDX-License-Identifier: AGPL-3.0-only

//! Checked layout for canonical BF16 Flash-Next input and FP32 recurrence.

use std::ffi::OsStr;

pub(super) fn parse_selector(value: Option<&OsStr>) -> Result<bool, &'static str> {
    match value {
        None => Ok(false),
        Some(value) if value == "0" => Ok(false),
        Some(value) if value == "1" => Ok(true),
        Some(_) => Err("selector must be absent or exactly 0 or 1"),
    }
}

pub(super) fn validate_request(rows: usize, start: usize) -> Result<(), &'static str> {
    if rows == 0 || !start.checked_add(rows).is_some_and(|end| end <= 2048) {
        return Err("exact SSM prefill requires a nonempty prompt within 2048 tokens");
    }
    Ok(())
}

pub(super) const fn grid32_partition(rows: usize) -> (usize, usize) {
    let full_rows = rows / 32 * 32;
    (full_rows, rows - full_rows)
}

pub(super) struct Plan {
    pub rows: usize,
    /// norm input, QKVZ, FP32 conv/GDN, gates, normalized GDN, output.
    pub bytes: [usize; 6],
    pub residual_bytes: usize,
}

impl Plan {
    pub const H: usize = 2560;
    pub const KEY_DIM: usize = 2048;
    pub const VALUE_DIM: usize = 6144;
    pub const CONV_DIM: usize = Self::KEY_DIM * 2 + Self::VALUE_DIM;
    pub const QKVZ: usize = Self::CONV_DIM + Self::VALUE_DIM;
    pub const GATES: usize = 96;
    pub const H_STATE_BYTES: usize = 48 * 128 * 128 * 4;
    pub const CONV_STATE_BYTES: usize = Self::CONV_DIM * 4 * 4;

    pub fn new(rows: usize, capacity: usize, limits: [usize; 6]) -> Result<Self, &'static str> {
        validate_request(rows, 0)?;
        if rows < 2 || rows > capacity {
            return Err("exact SSM staging requires 2..2048 rows within arena capacity");
        }
        let strides = [
            Self::H * 2,
            Self::QKVZ * 2,
            Self::QKVZ * 4,
            Self::GATES * 4,
            Self::VALUE_DIM * 2,
            Self::H * 2,
        ];
        let mut bytes = [0; 6];
        for index in 0..bytes.len() {
            bytes[index] = rows
                .checked_mul(strides[index])
                .ok_or("SSM extent overflow")?;
            if bytes[index] > limits[index] {
                return Err("exact SSM staging exceeds its arena");
            }
        }
        let residual_bytes = rows
            .checked_mul(10_240 * 2)
            .ok_or("SSM residual overflow")?;
        Ok(Self {
            rows,
            bytes,
            residual_bytes,
        })
    }

    pub fn tiles(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.tiles_from(0)
    }

    pub fn tiles_from(&self, first: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        (first..self.rows)
            .step_by(32)
            .map(|start| (start, (self.rows - start).min(32)))
    }
}

/// Validate typed, simultaneously live device spans without touching the GPU.
pub(super) fn validate_regions(regions: &[(u64, usize, usize)]) -> Result<(), &'static str> {
    for (index, &(ptr, bytes, alignment)) in regions.iter().enumerate() {
        if ptr == 0 || bytes == 0 || !matches!(alignment, 2 | 4) || ptr % alignment as u64 != 0 {
            return Err("invalid exact SSM typed device span");
        }
        let end = ptr
            .checked_add(bytes.try_into().map_err(|_| "SSM extent overflow")?)
            .ok_or("SSM device address overflow")?;
        for &(other, count, _) in &regions[..index] {
            let other_end = other
                .checked_add(count.try_into().map_err(|_| "SSM extent overflow")?)
                .ok_or("SSM device address overflow")?;
            if end > other && other_end > ptr {
                return Err("exact SSM live device spans overlap");
            }
        }
    }
    Ok(())
}
