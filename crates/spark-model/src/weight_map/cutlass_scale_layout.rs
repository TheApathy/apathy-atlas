// SPDX-License-Identifier: AGPL-3.0-only

//! FlashInfer/CUTLASS NVFP4 128x4 block-scale layout.
//!
//! ModelOpt stores one E4M3 scale byte per group of 16 logical values in a
//! row-major `[N, K / 16]` matrix.  FlashInfer's CUTLASS backend pads that
//! matrix, then applies the following reshape and permutation before GEMM:
//!
//! ```text
//! [1, N/128, 4, 32, (K/16)/4, 4]
//!     -> permute(0, 1, 4, 3, 2, 5)
//! [1, N/128, (K/16)/4, 32, 4, 4]
//! ```
//!
//! For logical coordinates `(n, g)` where `G = K/16`, the exact physical
//! byte offset is:
//!
//! ```text
//! (((((n/128) * (G/4) + g/4) * 32 + n%32) * 4 + (n%128)/32) * 4 + g%4)
//! ```
//!
//! This module deliberately accepts only already-aligned matrices.  Callers
//! must make padding an explicit, separately qualified transform rather than
//! silently changing the logical GEMM shape.

use anyhow::{Result, ensure};

pub const NVFP4_GROUP_SIZE: usize = 16;
pub const CUTLASS_SCALE_ROW_TILE: usize = 128;
pub const CUTLASS_SCALE_GROUP_TILE: usize = 4;
const CUTLASS_SCALE_ROW_LANES: usize = 32;
const CUTLASS_SCALE_ROW_QUADS: usize = CUTLASS_SCALE_ROW_TILE / CUTLASS_SCALE_ROW_LANES;

#[derive(Debug, Clone, Copy)]
struct ScaleShape {
    rows: usize,
    groups: usize,
    len: usize,
}

fn admit_shape(shape: &[usize], group_size: usize, actual_len: usize) -> Result<ScaleShape> {
    ensure!(
        shape.len() == 2,
        "NVFP4 scale matrix must have rank 2, got rank {}",
        shape.len()
    );
    ensure!(
        group_size == NVFP4_GROUP_SIZE,
        "CUTLASS NVFP4 requires group size {NVFP4_GROUP_SIZE}, got {group_size}"
    );

    let rows = shape[0];
    let groups = shape[1];
    ensure!(
        rows > 0 && groups > 0,
        "NVFP4 scale dimensions must be positive, got [{rows}, {groups}]"
    );
    ensure!(
        rows.is_multiple_of(CUTLASS_SCALE_ROW_TILE),
        "NVFP4 scale rows must be divisible by {CUTLASS_SCALE_ROW_TILE}, got {rows}"
    );
    ensure!(
        groups.is_multiple_of(CUTLASS_SCALE_GROUP_TILE),
        "NVFP4 scale groups must be divisible by {CUTLASS_SCALE_GROUP_TILE}, got {groups}"
    );

    // Reconstructing K is part of admission: a shape that cannot represent a
    // logical `[N,K]` tensor must fail before indexing or allocation.
    let _logical_k = groups.checked_mul(group_size).ok_or_else(|| {
        anyhow::anyhow!("NVFP4 logical K overflow: groups={groups} group_size={group_size}")
    })?;
    let len = rows.checked_mul(groups).ok_or_else(|| {
        anyhow::anyhow!("NVFP4 scale byte length overflow: rows={rows} groups={groups}")
    })?;
    ensure!(
        actual_len == len,
        "NVFP4 scale byte length mismatch: shape [{rows}, {groups}] requires {len}, got {actual_len}"
    );

    Ok(ScaleShape { rows, groups, len })
}

/// Convert logical ModelOpt `[N, K/16]` E4M3 bytes to the physical
/// FlashInfer/CUTLASS 128x4 layout.
pub fn interleave_nvfp4_scales_128x4(
    logical: &[u8],
    shape: &[usize],
    group_size: usize,
) -> Result<Vec<u8>> {
    let shape = admit_shape(shape, group_size, logical.len())?;
    let mut physical = vec![0_u8; shape.len];
    let group_blocks = shape.groups / CUTLASS_SCALE_GROUP_TILE;

    for row in 0..shape.rows {
        let row_block = row / CUTLASS_SCALE_ROW_TILE;
        let row_quad = (row % CUTLASS_SCALE_ROW_TILE) / CUTLASS_SCALE_ROW_LANES;
        let row_lane = row % CUTLASS_SCALE_ROW_LANES;
        for group in 0..shape.groups {
            let group_block = group / CUTLASS_SCALE_GROUP_TILE;
            let group_lane = group % CUTLASS_SCALE_GROUP_TILE;
            let dst = (((row_block * group_blocks + group_block) * CUTLASS_SCALE_ROW_LANES
                + row_lane)
                * CUTLASS_SCALE_ROW_QUADS
                + row_quad)
                * CUTLASS_SCALE_GROUP_TILE
                + group_lane;
            physical[dst] = logical[row * shape.groups + group];
        }
    }
    Ok(physical)
}

/// Invert [`interleave_nvfp4_scales_128x4`] into canonical ModelOpt
/// row-major `[N, K/16]` order.  This is intended for byte-exact admission
/// and parity receipts, not a runtime GEMM path.
pub fn deinterleave_nvfp4_scales_128x4(
    physical: &[u8],
    shape: &[usize],
    group_size: usize,
) -> Result<Vec<u8>> {
    let shape = admit_shape(shape, group_size, physical.len())?;
    let mut logical = vec![0_u8; shape.len];
    let group_blocks = shape.groups / CUTLASS_SCALE_GROUP_TILE;

    for row in 0..shape.rows {
        let row_block = row / CUTLASS_SCALE_ROW_TILE;
        let row_quad = (row % CUTLASS_SCALE_ROW_TILE) / CUTLASS_SCALE_ROW_LANES;
        let row_lane = row % CUTLASS_SCALE_ROW_LANES;
        for group in 0..shape.groups {
            let group_block = group / CUTLASS_SCALE_GROUP_TILE;
            let group_lane = group % CUTLASS_SCALE_GROUP_TILE;
            let src = (((row_block * group_blocks + group_block) * CUTLASS_SCALE_ROW_LANES
                + row_lane)
                * CUTLASS_SCALE_ROW_QUADS
                + row_quad)
                * CUTLASS_SCALE_GROUP_TILE
                + group_lane;
            logical[row * shape.groups + group] = physical[src];
        }
    }
    Ok(logical)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(len: usize) -> Vec<u8> {
        (0..len)
            .map(|index| {
                let x = index as u64;
                (x.wrapping_mul(131).wrapping_add(x >> 7).wrapping_add(17) & 0xff) as u8
            })
            .collect()
    }

    fn independent_reference(logical: &[u8], rows: usize, groups: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(logical.len());
        for row_block in 0..rows / 128 {
            for group_block in 0..groups / 4 {
                for row_lane in 0..32 {
                    for row_quad in 0..4 {
                        for group_lane in 0..4 {
                            let row = row_block * 128 + row_quad * 32 + row_lane;
                            let group = group_block * 4 + group_lane;
                            out.push(logical[row * groups + group]);
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    fn mapping_matches_independent_reshape_permute_reference() {
        let rows = 256;
        let groups = 12;
        let logical = fixture(rows * groups);
        let actual =
            interleave_nvfp4_scales_128x4(&logical, &[rows, groups], NVFP4_GROUP_SIZE).unwrap();
        assert_eq!(actual, independent_reference(&logical, rows, groups));
    }

    #[test]
    fn mapping_covers_tile_and_matrix_boundaries_without_aliasing() {
        let rows = 256;
        let groups = 12;
        let mut logical = vec![0_u8; rows * groups];
        let cases = [
            ((0, 0), 0, 0x11),
            ((31, 3), 499, 0x22),
            ((32, 0), 4, 0x33),
            ((127, 11), 1_535, 0x44),
            ((128, 0), 1_536, 0x55),
            ((255, 11), 3_071, 0x66),
        ];
        for &((row, group), _, value) in &cases {
            logical[row * groups + group] = value;
        }

        let physical =
            interleave_nvfp4_scales_128x4(&logical, &[rows, groups], NVFP4_GROUP_SIZE).unwrap();
        for &(_, physical_index, value) in &cases {
            assert_eq!(physical[physical_index], value);
        }
        assert_eq!(
            deinterleave_nvfp4_scales_128x4(&physical, &[rows, groups], NVFP4_GROUP_SIZE,).unwrap(),
            logical
        );
    }

    #[test]
    fn qwen38_ffn_and_projection_shapes_round_trip_every_byte_deterministically() {
        // (N, K): dense FFN gate/up, down, merged gate+up, attention QG,
        // attention K/V, merged QGKV, attention O, SSM QKVZ, and SSM O.
        let shapes = [
            (17_408, 5_120),
            (5_120, 17_408),
            (34_816, 5_120),
            (12_288, 5_120),
            (1_024, 5_120),
            (14_336, 5_120),
            (5_120, 6_144),
            (5_120, 4_096),
        ];

        for (rows, logical_k) in shapes {
            let groups = logical_k / NVFP4_GROUP_SIZE;
            let logical = fixture(rows * groups);
            let first =
                interleave_nvfp4_scales_128x4(&logical, &[rows, groups], NVFP4_GROUP_SIZE).unwrap();
            let second =
                interleave_nvfp4_scales_128x4(&logical, &[rows, groups], NVFP4_GROUP_SIZE).unwrap();
            assert_eq!(
                first, second,
                "non-deterministic physical bytes for N={rows} K={logical_k}"
            );
            assert_eq!(
                deinterleave_nvfp4_scales_128x4(&first, &[rows, groups], NVFP4_GROUP_SIZE,)
                    .unwrap(),
                logical,
                "round-trip mismatch for N={rows} K={logical_k}"
            );
        }
    }

    #[test]
    fn rejects_invalid_rank_group_alignment_and_length() {
        let valid = vec![0_u8; 128 * 4];
        for shape in [&[][..], &[512][..], &[128, 4, 1][..]] {
            assert!(interleave_nvfp4_scales_128x4(&valid, shape, 16).is_err());
            assert!(deinterleave_nvfp4_scales_128x4(&valid, shape, 16).is_err());
        }
        for group_size in [0, 1, 8, 32] {
            assert!(interleave_nvfp4_scales_128x4(&valid, &[128, 4], group_size).is_err());
        }
        for shape in [[0, 4], [128, 0], [127, 4], [129, 4], [128, 3], [128, 5]] {
            assert!(interleave_nvfp4_scales_128x4(&valid, &shape, 16).is_err());
        }
        assert!(interleave_nvfp4_scales_128x4(&valid[..valid.len() - 1], &[128, 4], 16).is_err());
        let mut too_long = valid.clone();
        too_long.push(0);
        assert!(deinterleave_nvfp4_scales_128x4(&too_long, &[128, 4], 16).is_err());
    }

    #[test]
    fn rejects_dimension_arithmetic_overflow_before_indexing_or_allocation() {
        let huge_aligned_groups = usize::MAX - (usize::MAX % 4);
        assert!(interleave_nvfp4_scales_128x4(&[], &[128, huge_aligned_groups], 16).is_err());
        let huge_aligned_rows = usize::MAX - (usize::MAX % 128);
        assert!(deinterleave_nvfp4_scales_128x4(&[], &[huge_aligned_rows, 4], 16).is_err());
    }
}
