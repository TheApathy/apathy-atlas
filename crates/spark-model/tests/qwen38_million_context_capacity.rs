// SPDX-License-Identifier: AGPL-3.0-only

const MAX_SEQ_LEN: usize = 1_000_000;
const BLOCK_SIZE: usize = 16;
const MAX_BATCH_SIZE: usize = 1;

fn reachable_pool_blocks(max_seq_len: usize, block_size: usize, batch: usize) -> usize {
    let blocks_per_seq = max_seq_len.div_ceil(block_size);
    batch
        .saturating_mul(blocks_per_seq.saturating_add(1))
        .saturating_add(1)
}

#[test]
fn exact_million_c1_uses_62502_blocks() {
    let active_blocks = MAX_SEQ_LEN.div_ceil(BLOCK_SIZE);
    assert_eq!(active_blocks, 62_500);
    assert_eq!(
        reachable_pool_blocks(MAX_SEQ_LEN, BLOCK_SIZE, MAX_BATCH_SIZE),
        62_502
    );

    let source = include_str!("../src/factory/build.rs");
    assert!(source.contains("max_seq_len.div_ceil(kv_block_size)"));
    assert!(source.contains(".saturating_mul(blocks_per_seq.saturating_add(1))"));
    assert!(source.contains("let n = budget_blocks.min(reachable);"));
}

#[test]
fn synthetic_1048576_boundary_fits_index_widths() {
    let allocated_blocks = reachable_pool_blocks(1_048_576, BLOCK_SIZE, MAX_BATCH_SIZE);
    assert_eq!(allocated_blocks, 65_538);

    let highest_physical_block = allocated_blocks - 1;
    let highest_slot = allocated_blocks * BLOCK_SIZE - 1;
    assert!(u32::try_from(highest_physical_block).is_ok());
    assert!(u32::try_from(highest_slot).is_ok());

    let nvfp4_bytes_per_side_block = 9_216_u64;
    let final_side_byte = (allocated_blocks as u64)
        .checked_mul(nvfp4_bytes_per_side_block)
        .and_then(|extent| extent.checked_sub(1))
        .unwrap();
    assert_eq!(final_side_byte, 603_998_207);
}
