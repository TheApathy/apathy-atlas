// SPDX-License-Identifier: AGPL-3.0-only

//! Exact BF16 boundary checks, not an emulation of tensor-core reduction.
use crate::contract::Mode;

pub fn narrow_bf16(value: f32) -> Result<u16, String> {
    if !value.is_finite() {
        return Err("nonfinite F32 input".into());
    }
    let bits = value.to_bits();
    let rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16;
    if rounded & 0x7f80 == 0x7f80 {
        return Err("BF16 rounding overflow".into());
    }
    Ok(rounded)
}

fn widen_bf16(bits: u16) -> Result<f32, String> {
    if bits & 0x7f80 == 0x7f80 {
        return Err("nonfinite BF16 operand".into());
    }
    Ok(f32::from_bits(u32::from(bits) << 16))
}

/// FusedBias shares only the scalar epilogue, not its accumulation order.
pub fn bias_epilogue(sum: f32, bias: u16, mode: Mode) -> Result<u16, String> {
    if !sum.is_finite() {
        return Err("nonfinite F32 accumulation".into());
    }
    let bias = widen_bf16(bias)?;
    let before_bias = match mode {
        Mode::Scalar | Mode::FusedBias => sum,
        Mode::UpstreamSeparate => widen_bf16(narrow_bf16(sum)?)?,
    };
    narrow_bf16(before_bias + bias)
}

/// Sequential F32 product/add with one BF16 narrowing after bias.
pub fn scalar_dot(a: &[u16], b: &[u16], bias: u16) -> Result<u16, String> {
    if a.is_empty() || a.len() != b.len() {
        return Err("dot operand lengths differ or are empty".into());
    }
    let mut sum = 0.0f32;
    for (&a, &b) in a.iter().zip(b) {
        // Intentionally separate product and addition, never mul_add.
        let product = widen_bf16(a)? * widen_bf16(b)?;
        sum += product;
        if !sum.is_finite() {
            return Err("nonfinite F32 accumulation".into());
        }
    }
    bias_epilogue(sum, bias, Mode::Scalar)
}

pub fn validate_output(output: &[u16], expected_len: usize) -> Result<(), String> {
    if expected_len == 0 || output.len() != expected_len {
        return Err("output extent differs or is empty".into());
    }
    if let Some(index) = output.iter().position(|bits| bits & 0x7f80 == 0x7f80) {
        return Err(format!("nonfinite BF16 output at element {index}"));
    }
    Ok(())
}
