// SPDX-License-Identifier: AGPL-3.0-only
//! Autotune probe for the PINNED bf16 GEMMs of DeepSeek-V4.1 prefill
//! (`cublaslt::gemm_act_weight_t_typed_pinned`, one no-split-K algorithm per shape chosen at
//! M = ref_m). For every shape: every heuristic candidate's median time at each M, and whether
//! its output is BYTE-IDENTICAL to the algorithm production pins today, at every M, plus its
//! own M-invariance (rows of a smaller M equal the prefix rows of the largest).
//!
//! Negative control: the production algorithm with ONE weight element changed must differ.
//! Random bf16 operands (a changed summation order shows up on random data).
//!
//!   cargo run --release -p spark-model --features cuda,gpu-examples --example dsv41_gemm_tune -- \
//!       [--ref-m 3968] [--m 3968,2816,128] [--cands 16] [--iters 15] [--shapes name:n:k:lda:ldc,...]

use anyhow::{Context, Result, bail, ensure};
use std::sync::Arc;
use std::time::Instant;

use spark_runtime::cublaslt::{GemmDtype, describe_algo, gemm_act_weight_t_typed_pinned, pinned_candidates, set_pinned_algo};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// (name, n, k, lda, ldc) as V41 prefill issues them (attn_block.rs, fwd.rs SharedExpert).
const SHAPES: [(&str, u32, u32, u32, u32); 7] = [
    ("wq_a", 1280, 5120, 5120, 1280),
    ("wq_b", 32768, 1280, 1280, 32768),
    ("wkv", 512, 5120, 5120, 512),
    ("wo_a_group", 1024, 4096, 32768, 8192),
    ("wo_b", 5120, 8192, 8192, 5120),
    ("shared_w13", 4608, 5120, 5120, 4608),
    ("shared_w2", 5120, 2304, 2304, 5120),
];

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Deterministic bf16 values in about [-1, 1].
fn random_bf16(len: usize, seed: u64) -> Vec<u8> {
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut out = Vec::with_capacity(len * 2);
    for _ in 0..len {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let f = ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
        out.extend_from_slice(&((f.to_bits() >> 16) as u16).to_le_bytes());
    }
    out
}

struct Shape {
    name: String,
    n: u32,
    k: u32,
    lda: u32,
    ldc: u32,
}

fn parse_shapes(v: &str) -> Result<Vec<Shape>> {
    v.split(',')
        .map(|s| {
            let p: Vec<&str> = s.split(':').collect();
            ensure!(p.len() == 5, "shape name:n:k:lda:ldc, got {s}");
            Ok(Shape { name: p[0].into(), n: p[1].parse()?, k: p[2].parse()?, lda: p[3].parse()?, ldc: p[4].parse()? })
        })
        .collect()
}

fn main() -> Result<()> {
    let (mut ref_m, mut n_cands, mut iters) = (3968u32, 16usize, 15usize);
    let mut ms = vec![3968u32, 2816, 128];
    let mut shapes: Vec<Shape> =
        SHAPES.iter().map(|&(name, n, k, lda, ldc)| Shape { name: name.into(), n, k, lda, ldc }).collect();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--ref-m" => ref_m = args.next().context("--ref-m")?.parse()?,
            "--cands" => n_cands = args.next().context("--cands")?.parse()?,
            "--iters" => iters = args.next().context("--iters")?.parse()?,
            "--m" => ms = args.next().context("--m")?.split(',').map(str::parse).collect::<Result<_, _>>()?,
            "--shapes" => shapes = parse_shapes(&args.next().context("--shapes")?)?,
            other => bail!("unknown argument {other}"),
        }
    }
    let max_m = *ms.iter().max().context("no M")?;
    ensure!(ms[0] == max_m, "--m must list the largest M first (the prefix-row check compares against it)");
    let gpu = Arc::new(AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?);
    let stream = gpu.default_stream();
    let bf = GemmDtype::Bf16;
    let mut all_controls_fired = true;

    for sh in &shapes {
        let (n, k, lda, ldc) = (sh.n, sh.k, sh.lda, sh.ldc);
        let w_host = random_bf16((n * k) as usize, 1 + n as u64 * 31 + k as u64);
        let x_host = random_bf16((max_m * lda) as usize, 7 + k as u64);
        let (w, x) = (gpu.alloc(w_host.len())?, gpu.alloc(x_host.len())?);
        gpu.copy_h2d(&w_host, w)?;
        gpu.copy_h2d(&x_host, x)?;
        let out_bytes = (max_m * ldc) as usize * 2;
        let out = gpu.alloc(out_bytes)?;
        let gemm = |wt: DevicePtr, m: u32| gemm_act_weight_t_typed_pinned(x.0, lda, wt.0, out.0, ldc, m, n, k, bf, bf, true, ref_m, stream);
        // Output rows [0, m) x cols [0, n) (ldc may exceed n).
        let read = |m: u32| -> Result<Vec<u8>> {
            gpu.synchronize(stream)?;
            let mut b = vec![0u8; out_bytes];
            gpu.copy_d2h(out, &mut b)?;
            Ok((0..m as usize).flat_map(|r| b[r * ldc as usize * 2..(r * ldc as usize + n as usize) * 2].to_vec()).collect())
        };
        let time = |wt: DevicePtr, m: u32| -> Result<f64> {
            let reps = if m <= 256 { 20 } else { 1 };
            let mut v = Vec::new();
            for i in 0..iters + 2 {
                gpu.synchronize(stream)?;
                let t0 = Instant::now();
                for _ in 0..reps {
                    gemm(wt, m)?;
                }
                gpu.synchronize(stream)?;
                if i >= 2 {
                    v.push(t0.elapsed().as_secs_f64() * 1e3 / reps as f64);
                }
            }
            Ok(median(v))
        };

        // Production: the pinned path on a fresh cache entry picks heuristic #1 by itself.
        let mut prod: Vec<Vec<u8>> = Vec::new();
        for &m in &ms {
            gemm(w, m)?;
            prod.push(read(m)?);
        }
        // NEGATIVE CONTROL: one weight element changed, same algorithm.
        let mut w_bad = w_host.clone();
        w_bad[2 * (k as usize / 2)] ^= 0x40;
        let wc = gpu.alloc(w_bad.len())?;
        gpu.copy_h2d(&w_bad, wc)?;
        gemm(wc, max_m)?;
        let fired = read(max_m)? != prod[0];
        all_controls_fired &= fired;
        println!(
            "== {} n={n} k={k} lda={lda} ldc={ldc}: control (one weight element changed) {}",
            sh.name,
            if fired { "DIFFERS (fired)" } else { "SAME -> this gate proves nothing" }
        );
        let flop = |m: u32| 2.0 * m as f64 * n as f64 * k as f64;
        let prod_ms: Vec<f64> = ms.iter().map(|&m| time(w, m)).collect::<Result<_>>()?;

        let cands = pinned_candidates(n, k, lda, ldc, bf, bf, ref_m, n_cands)?;
        for (ci, algo) in cands.iter().enumerate() {
            set_pinned_algo(n, k, lda, ldc, bf, bf, true, ref_m, *algo)?;
            let mut line = format!("  cand {ci:>2} [{}]", describe_algo(algo));
            let mut identical = true;
            let mut outs: Vec<Vec<u8>> = Vec::new();
            for (mi, &m) in ms.iter().enumerate() {
                if gemm(w, m).is_err() {
                    line.push_str(" LAUNCH-FAILED");
                    identical = false;
                    break;
                }
                let o = read(m)?;
                identical &= o == prod[mi];
                outs.push(o);
                let t = time(w, m)?;
                line.push_str(&format!(
                    " | M={m} {t:.3} ms ({:.1} TF/s, {:+.1}% vs prod)",
                    flop(m) / t / 1e9,
                    100.0 * (t / prod_ms[mi] - 1.0)
                ));
            }
            let m_invariant = outs.len() == ms.len() && outs.iter().all(|o| outs[0][..o.len()] == o[..]);
            line.push_str(&format!(" | byte-identical to prod: {identical} | M-invariant: {m_invariant}"));
            println!("{line}");
        }
        // Leave production's pick in place for any later shape sharing the key.
        if let Some(a) = cands.first() {
            set_pinned_algo(n, k, lda, ldc, bf, bf, true, ref_m, *a)?;
        }
        println!(
            "  prod: {}",
            ms.iter().zip(&prod_ms).map(|(m, t)| format!("M={m} {t:.3} ms ({:.1} TF/s)", flop(*m) / t / 1e9)).collect::<Vec<_>>().join(" | ")
        );
        for p in [w, x, out, wc] {
            gpu.free(p)?;
        }
    }
    ensure!(all_controls_fired, "a negative control did not fire");
    println!("DONE dsv41_gemm_tune");
    Ok(())
}
