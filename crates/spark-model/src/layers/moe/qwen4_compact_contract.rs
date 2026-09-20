// SPDX-License-Identifier: AGPL-3.0-only

//! Private compact planner ABI. Byte encoding never reads Rust padding.

pub(super) const WORKSPACE_BYTES: usize = 14_384;
pub(super) const CONTRACT_BYTES: usize = 64;
pub(super) const ARENA_BYTES: usize = WORKSPACE_BYTES + CONTRACT_BYTES;
pub(super) const GRID: u32 = 4096;
pub(super) const MAX_ITEMS: u32 = 1792;

pub(super) fn parse(value: Option<&str>) -> Result<bool, &'static str> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => Err("must be exactly 0 or 1"),
    }
}

pub(super) fn contract(rows: usize) -> Result<[u8; CONTRACT_BYTES], &'static str> {
    if !(2..=2048).contains(&rows) {
        return Err("compact prefill requires 2..2048 rows");
    }
    let words = [
        1,
        rows as u32,
        2560,
        640,
        512,
        10,
        rows as u32 * 10,
        64,
        64,
        GRID,
        MAX_ITEMS,
        WORKSPACE_BYTES as u32,
        0,
        0,
    ];
    let mut bytes = [0; CONTRACT_BYTES];
    bytes[..8].copy_from_slice(&0x4f52494749363430u64.to_le_bytes());
    for (slot, word) in bytes[8..].chunks_exact_mut(4).zip(words) {
        slot.copy_from_slice(&word.to_le_bytes());
    }
    Ok(bytes)
}

pub(super) fn check_status(status: i32) -> Result<(), &'static str> {
    if status == 1 {
        Ok(())
    } else {
        Err("compact planner/GEMM did not complete successfully")
    }
}

pub(super) fn bundle_matches(selected: bool, handles: Option<(u64, u64)>) -> bool {
    match (selected, handles) {
        (false, None) => true,
        (true, Some((plan, gemm))) => plan != 0 && gemm != 0,
        _ => false,
    }
}

pub(super) fn region(ptr: u64, bytes: usize, alignment: u64) -> Result<(u64, u64), &'static str> {
    if ptr == 0 || bytes == 0 || alignment == 0 || !ptr.is_multiple_of(alignment) {
        return Err("compact buffer is null, empty, or misaligned");
    }
    ptr.checked_add(bytes as u64)
        .map(|end| (ptr, end))
        .ok_or("compact buffer extent overflows")
}

pub(super) fn disjoint(a: (u64, u64), b: (u64, u64)) -> bool {
    a.1 <= b.0 || b.1 <= a.0
}
