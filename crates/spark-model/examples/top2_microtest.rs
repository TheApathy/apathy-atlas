// SPDX-License-Identifier: AGPL-3.0-only

//! Correctness gate for `top2_bf16_rows` (block-fork tree cliff detection,
//! doc 16 stage 1): per-row top-2 (idx, val) over BF16 logits vs host ref.

use anyhow::{bail, Result};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::KernelLaunch;

fn f32_to_bf16(f: f32) -> u16 {
    let bits = f.to_bits();
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

fn main() -> Result<()> {
    let rows = 16usize;
    let vocab = 100_352usize;
    let mut rng = Rng(0x51);
    let logits_bits: Vec<u16> = (0..rows * vocab)
        .map(|_| f32_to_bf16(rng.unit() * 20.0 - 10.0))
        .collect();

    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let bytes: Vec<u8> = logits_bits.iter().flat_map(|x| x.to_le_bytes()).collect();
    let d_logits = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, d_logits)?;
    let d_out = gpu.alloc(rows * 16)?;

    let h = gpu.kernel("argmax", "top2_bf16_rows")?;
    KernelLaunch::new(gpu, h)
        .grid([rows as u32, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(d_logits)
        .arg_ptr(d_out)
        .arg_u32(vocab as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;

    let mut raw = vec![0u8; rows * 16];
    gpu.copy_d2h(d_out, &mut raw)?;
    let words: Vec<u32> = raw
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut fails = 0;
    for r in 0..rows {
        // host reference top-2 on the SAME bf16-rounded values
        let row = &logits_bits[r * vocab..(r + 1) * vocab];
        let (mut i1, mut v1, mut i2, mut v2) = (0usize, f32::MIN, 0usize, f32::MIN);
        for (i, &b) in row.iter().enumerate() {
            let v = bf16_to_f32(b);
            if v > v1 {
                v2 = v1;
                i2 = i1;
                v1 = v;
                i1 = i;
            } else if v > v2 {
                v2 = v;
                i2 = i;
            }
        }
        let gi1 = words[r * 4] as usize;
        let gv1 = f32::from_bits(words[r * 4 + 1]);
        let gi2 = words[r * 4 + 2] as usize;
        let gv2 = f32::from_bits(words[r * 4 + 3]);
        // Values must match exactly; indices may differ on exact ties (both
        // valid) — accept if the value at the returned index matches.
        let ok = (gv1 - v1).abs() < 1e-6
            && (gv2 - v2).abs() < 1e-6
            && (bf16_to_f32(row[gi1]) - v1).abs() < 1e-6
            && (bf16_to_f32(row[gi2]) - v2).abs() < 1e-6
            && gi1 != gi2;
        if !ok {
            fails += 1;
            eprintln!(
                "row {r}: gpu ({gi1},{gv1:.4})/({gi2},{gv2:.4}) vs ref ({i1},{v1:.4})/({i2},{v2:.4})"
            );
        }
    }
    if fails > 0 {
        bail!("top2_bf16_rows FAILED on {fails}/{rows} rows");
    }
    println!("top2_bf16_rows: PASS ({rows} rows x {vocab} vocab)");
    Ok(())
}
