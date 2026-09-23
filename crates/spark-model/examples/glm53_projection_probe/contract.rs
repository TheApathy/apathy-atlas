// SPDX-License-Identifier: AGPL-3.0-only
use anyhow::{Result, ensure};
use serde_json::{Value, json};
pub fn validate_bf16(bytes: &[u8], expected: usize) -> Result<()> {
    ensure!(
        expected > 0 && expected % 2 == 0 && bytes.len() == expected,
        "BF16 operator extent mismatch"
    );
    for pair in bytes.chunks_exact(2) {
        let value = u16::from_le_bytes([pair[0], pair[1]]);
        ensure!(value & 0x7f80 != 0x7f80, "nonfinite BF16 operator data");
    }
    Ok(())
}
pub fn compare(a: &[u8], b: &[u8]) -> Result<Value> {
    validate_bf16(a, a.len())?;
    validate_bf16(b, a.len())?;
    let different_bytes = a.iter().zip(b).filter(|(a, b)| a != b).count();
    let first_byte = a.iter().zip(b).position(|(a, b)| a != b);
    let (mut error, mut norm, mut max_abs) = (0.0f64, 0.0f64, 0.0f64);
    for (a, b) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
        let a = f32::from_bits(u32::from(u16::from_le_bytes([a[0], a[1]])) << 16) as f64;
        let b = f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16) as f64;
        error += (a - b) * (a - b);
        norm += a * a;
        max_abs = max_abs.max((a - b).abs());
    }
    Ok(
        json!({"exact":different_bytes==0,"different_bytes":different_bytes,
        "first_byte":first_byte,"max_abs":max_abs,
        "relative_l2":if norm>0.0 {Some((error/norm).sqrt())} else {None}}),
    )
}
