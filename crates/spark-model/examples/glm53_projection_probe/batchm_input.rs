// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit derived operator fixture, not a captured long model context.
use super::{contract, gemv_contract::HIDDEN};
use anyhow::{Result, ensure};
pub const INPUT_DERIVATION: &str = "repeated-pinned-two-row-input";
pub fn derive_input(source: &[u8], rows: u32) -> Result<Vec<u8>> {
    let row_bytes = HIDDEN as usize * 2;
    ensure!(
        (1..=2048).contains(&rows),
        "derived input requires 1..=2048 rows"
    );
    contract::validate_bf16(source, 2 * row_bytes)?;
    let mut output = Vec::new();
    output.try_reserve_exact(rows as usize * row_bytes)?;
    for row in 0..rows as usize {
        let offset = (row % 2) * row_bytes;
        output.extend_from_slice(&source[offset..offset + row_bytes]);
    }
    Ok(output)
}
