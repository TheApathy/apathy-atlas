// SPDX-License-Identifier: AGPL-3.0-only
//! Gate for the small-M fp8 GEMV (`dsv41_fp8_gemv_m8`): every row of an M = 2..8 call must be
//! BIT-IDENTICAL to the same row computed alone by the M = 1 kernel, on real layer-0 weights.
//! That is what makes a speculative verify row equal to the plain decode of the same token.
//!
//! Also reports rel_l2 of the GEMV against dequant + cuBLASLt (a numerics sanity check: only the
//! fp32 accumulation order differs, so it must be ~1e-6, and a wrong scale layout would be O(1)).
//!
//! Control (must FAIL): row r of the M = 8 call against the M = 1 result of row r + 1.
//!
//! ```text
//! cargo run -p spark-model --release --example dsv41_gemv_rows_gate --features cuda,gpu-examples
//! ```

use anyhow::{Result, bail, ensure};
use std::path::Path;
use std::sync::Arc;

use spark_model::weight_loader::deepseek_v41::ops::{Dsv41Kernels, Fp8Linear, Ops};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{SafetensorsLoader, WeightLoader};

const MODEL_DIR: &str = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K";

fn bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn main() -> Result<()> {
    // SAFETY-adjacent: set before any kernel table is built; single-threaded here.
    unsafe {
        std::env::set_var("ATLAS_DSV41_DENSE_GEMV", "1");
        std::env::set_var("ATLAS_DSV41_DENSE_GEMV_SMALL_M", "1");
    }
    let gpu = Arc::new(AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?);
    let mut loader = SafetensorsLoader::new();
    loader.extra_skip = Some(Arc::new(|name: &str| !name.starts_with("layers.0.")));
    let store = loader.load(Path::new(MODEL_DIR), gpu.as_ref(), 0)?;
    let kernels = Dsv41Kernels::load(gpu.as_ref())?;
    let stream = gpu.default_stream();
    let ops = Ops { gpu: gpu.as_ref(), k: &kernels, stream };

    let shapes: [(&str, usize, usize); 6] = [
        ("layers.0.attn.wq_a", 1280, 5120),
        ("layers.0.attn.wq_b", 32768, 1280),
        ("layers.0.attn.wkv", 512, 5120),
        ("layers.0.attn.wo_b", 5120, 8192),
        ("layers.0.ffn.shared_experts.w1", 2304, 5120),
        ("layers.0.ffn.shared_experts.w2", 5120, 2304),
    ];
    let mut failures = 0usize;
    let mut control_fired = true;
    for (name, n, k) in shapes {
        let w = Fp8Linear::load(&store, name, n, k)?;
        // Deterministic activations with a realistic spread (LCG, bf16-rounded).
        let mut seed = 0x9e3779b97f4a7c15u64 ^ (n as u64 * 31 + k as u64);
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0
        };
        let rows = 16usize; // MM_TILE rows for the cuBLAS reference
        let x: Vec<u8> = (0..rows * k).flat_map(|_| bf16_bits(next()).to_le_bytes()).collect();
        let dx = gpu.alloc(x.len())?;
        gpu.copy_h2d(&x, dx)?;
        let out_all = gpu.alloc(rows * n * 2)?;
        let out_one = gpu.alloc(rows * n * 2)?;
        let out_ref = gpu.alloc(rows * n * 2)?;
        let scratch = gpu.alloc(n * k * 2)?;
        let read = |p| -> Result<Vec<u16>> {
            gpu.synchronize(stream)?;
            let mut b = vec![0u8; rows * n * 2];
            gpu.copy_d2h(p, &mut b)?;
            Ok(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
        };
        // Each row alone (M = 1).
        for r in 0..8 {
            ops.fp8_gemv_m1(dx.offset(r * k * 2), &w, out_one.offset(r * n * 2), n, 0)?;
        }
        let one = read(out_one)?;
        // dequant + cuBLASLt reference.
        ops.dequant(&w, scratch)?;
        ops.linear_bf16_tiled(dx, k, scratch, out_ref, n, rows, n, k)?;
        let refv = read(out_ref)?;
        for m in [2usize, 3, 8] {
            ops.fp8_gemv_rows(dx, k, &w, out_all, n, m, n, 0)?;
            let all = read(out_all)?;
            let diff = (0..m * n).filter(|&i| all[i] != one[i]).count();
            let ctl = (0..(m - 1) * n).filter(|&i| all[i] != one[i + n]).count();
            if ctl == 0 {
                control_fired = false;
            }
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for i in 0..m * n {
                let a = f32::from_bits((all[i] as u32) << 16) as f64;
                let b = f32::from_bits((refv[i] as u32) << 16) as f64;
                num += (a - b) * (a - b);
                den += b * b;
            }
            let rel = (num / den).sqrt();
            println!(
                "{name:36} M={m}: {diff} of {} outputs differ from the M=1 rows (control: {ctl} differ vs the wrong row); rel_l2 vs dequant+cuBLASLt {rel:.3e}",
                m * n
            );
            if diff != 0 || rel > 1e-3 {
                failures += 1;
            }
        }
        for p in [dx, out_all, out_one, out_ref, scratch] {
            gpu.free(p)?;
        }
    }
    ensure!(control_fired, "CONTROL DID NOT FIRE: a wrong-row comparison matched; this gate proves nothing");
    if failures > 0 {
        bail!("FAIL: {failures} (weight, M) cases differ from the per-row M=1 result or exceed rel_l2 1e-3");
    }
    println!("PASS: every row of M=2/3/8 is bit-identical to M=1; control fired; rel_l2 vs cuBLASLt within 1e-3");
    Ok(())
}
