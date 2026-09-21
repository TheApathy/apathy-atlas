// SPDX-License-Identifier: AGPL-3.0-only

//! CPU plan for DSpark's uncompressed, interleaved-pair RoPE table.

use atlas_core::config::ModelConfig;

pub(super) fn build_rope_table(
    config: &ModelConfig,
    max_seq_len: usize,
    rope_dim: usize,
) -> Result<Vec<f32>, &'static str> {
    let theta = config
        .deepseek_main_rope_theta
        .filter(|theta| theta.is_finite() && *theta > 0.0)
        .ok_or("DSpark requires an explicit finite-positive base rope_theta")?;
    if max_seq_len == 0 || rope_dim == 0 || !rope_dim.is_multiple_of(2) {
        return Err("DSpark RoPE requires nonzero rows and an even nonzero dimension");
    }
    let elements = max_seq_len
        .checked_mul(rope_dim)
        .filter(|elements| *elements <= isize::MAX as usize / size_of::<f32>())
        .ok_or("DSpark RoPE table byte extent overflow")?;
    let mut table = Vec::new();
    table
        .try_reserve_exact(elements)
        .map_err(|_| "cannot allocate DSpark RoPE table")?;
    table.resize(elements, 0.0);
    let half = rope_dim / 2;
    for pos in 0..max_seq_len {
        for j in 0..half {
            let frequency = 1.0f64 / theta.powf(2.0 * j as f64 / rope_dim as f64);
            let angle = pos as f64 * frequency;
            if !angle.is_finite() {
                return Err("DSpark base rope_theta produces nonfinite angles");
            }
            table[(pos * half + j) * 2] = angle.cos() as f32;
            table[(pos * half + j) * 2 + 1] = angle.sin() as f32;
        }
    }
    Ok(table)
}
