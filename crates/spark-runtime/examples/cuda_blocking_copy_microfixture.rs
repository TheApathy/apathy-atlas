// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in GPU microfixture for the blocking pageable-host copy boundary.
//!
//! This is deliberately not a server or model-loader test. It round-trips the
//! exact 96 BF16 bytes from the layer-53 `linear_attn.dt_bias` tensor that was
//! the second observed startup stall, then repeatedly performs only blocking
//! D2H into an ordinary pageable `Vec<u8>`.

use anyhow::{Context, Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;

// model-00005-of-00006.safetensors, absolute file offset 180456.
// SHA-256 of the 96 little-endian bytes:
// e9eed511c3ae7fc96964eabb421164ad67f0fd45d3df56ef27d6dd820558de98
const BF16_WORDS: [u16; 48] = [
    0xc044, 0x3fec, 0xc01a, 0xc095, 0xc0a3, 0xc078, 0x40a8, 0xc0a8, 0x4093, 0xc0a0, 0xc09d, 0xc0a7,
    0xc01d, 0xc080, 0xbfe9, 0xc0a5, 0x3ed3, 0xc09d, 0xc084, 0xc050, 0xc060, 0xbfb7, 0xc064, 0xbf78,
    0xc0a7, 0xc09d, 0xc09e, 0x3e7b, 0xc0a2, 0x3f7a, 0xbe77, 0x400f, 0x3e02, 0x4025, 0xc04c, 0x4046,
    0x3ee4, 0xc097, 0x3f74, 0x3f21, 0xc0b6, 0x3f7d, 0x3f91, 0x3ee1, 0x3f2b, 0x3fb9, 0x3f1a, 0x3f74,
];

fn exact_tensor_bytes() -> [u8; 96] {
    let mut expected = [0_u8; 96];
    for (index, word) in BF16_WORDS.iter().enumerate() {
        let bytes = word.to_le_bytes();
        expected[index * 2] = bytes[0];
        expected[index * 2 + 1] = bytes[1];
    }
    expected
}

fn main() -> Result<()> {
    let iterations = std::env::var("ATLAS_D2H_FIXTURE_ITERS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()
        .context("ATLAS_D2H_FIXTURE_ITERS must be a non-negative integer")?
        .unwrap_or(4_096);
    let expected = exact_tensor_bytes();
    let gpu = AtlasCudaBackend::new(0, &[])?;
    let device = gpu.alloc(expected.len())?;

    let result = (|| -> Result<()> {
        gpu.copy_h2d(&expected, device)?;
        for iteration in 0..iterations {
            let mut observed = vec![0_u8; expected.len()];
            gpu.copy_d2h(device, &mut observed)?;
            if observed.as_slice() != expected {
                bail!("D2H mismatch at iteration {iteration}");
            }
        }
        Ok(())
    })();
    let free_result = gpu.free(device);
    result?;
    free_result?;

    println!("PASS: {iterations} repeated 96-byte blocking pageable D2H copies");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_fixture_is_exactly_48_bf16_words_and_96_bytes() {
        assert_eq!(BF16_WORDS.len(), 48);
        assert_eq!(exact_tensor_bytes().len(), 96);
        assert_eq!(
            &exact_tensor_bytes()[..8],
            &[0x44, 0xc0, 0xec, 0x3f, 0x1a, 0xc0, 0x95, 0xc0]
        );
    }
}
