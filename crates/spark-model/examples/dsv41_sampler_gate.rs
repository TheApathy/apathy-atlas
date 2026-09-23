// SPDX-License-Identifier: AGPL-3.0-only
//! Gate for the sampled-draft kernel `dspark_decode::dsv41_sample_softmax`: over 2e5 uniforms,
//! the drawn ids must follow softmax(lg / T) (chi-square, p > 0.01), and the q it writes must
//! equal the host softmax (max abs error <= 1e-6). CONTROL (must FAIL): the same draws tested
//! against softmax(lg / 2T). Also: a tiny temperature reproduces the argmax.
//!
//! ```text
//! cargo run -p spark-model --release --example dsv41_sampler_gate --features cuda,gpu-examples
//! ```

use anyhow::{Result, bail, ensure};

use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::KernelLaunch;

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let k = gpu.kernel("dspark_decode", "dsv41_sample_softmax")?;
    let stream = gpu.default_stream();
    // A realistic-size vocab with a peaked head and a long tail; 40 "bins" carry the mass test.
    let v = 129_280usize;
    let lg: Vec<f32> = (0..v)
        .map(|i| if i < 40 { 6.0 - 0.15 * i as f32 } else { -4.0 - ((i * 7919) % 97) as f32 * 0.03 })
        .collect();
    let t = 1.0f32;
    let host_q = |temp: f32| -> Vec<f64> {
        let m = lg.iter().map(|&x| (x / temp) as f64).fold(f64::MIN, f64::max);
        let e: Vec<f64> = lg.iter().map(|&x| ((x / temp) as f64 - m).exp()).collect();
        let s: f64 = e.iter().sum();
        e.iter().map(|x| x / s).collect()
    };
    let d_lg = gpu.alloc(v * 4)?;
    let lb: Vec<u8> = lg.iter().flat_map(|x| x.to_le_bytes()).collect();
    gpu.copy_h2d(&lb, d_lg)?;
    let d_q = gpu.alloc(v * 4)?;
    let n = 200_000usize;
    let d_ids = gpu.alloc(n * 4)?;
    let mut state = 0x243f6a8885a308d3u64;
    for i in 0..n {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        let u = ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32;
        KernelLaunch::new(&gpu, k)
            .grid([1, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(d_lg)
            .arg_u32(v as u32)
            .arg_f32(t)
            .arg_f32(u)
            .arg_ptr(d_q)
            .arg_ptr(d_ids)
            .arg_i32(i as i32)
            .launch(stream)?;
    }
    gpu.synchronize(stream)?;
    let mut ib = vec![0u8; n * 4];
    gpu.copy_d2h(d_ids, &mut ib)?;
    let ids: Vec<u32> = ib.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let mut qb = vec![0u8; v * 4];
    gpu.copy_d2h(d_q, &mut qb)?;
    let q: Vec<f32> = qb.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let hq = host_q(t);
    let qerr = q.iter().zip(&hq).map(|(a, b)| (*a as f64 - b).abs()).fold(0.0, f64::max);
    println!("q vs host softmax: max abs err {qerr:.3e}");
    // chi-square over the 40 head ids + one pooled tail bin.
    let chi = |p: &[f64]| -> (f64, usize) {
        let mut obs = vec![0u64; 41];
        for &id in &ids {
            obs[(id as usize).min(40)] += 1;
        }
        let mut exp: Vec<f64> = (0..40).map(|i| p[i] * n as f64).collect();
        exp.push(p[40..].iter().sum::<f64>() * n as f64);
        let mut s = 0.0;
        let mut df = 0;
        for (o, e) in obs.iter().zip(&exp) {
            if *e >= 5.0 {
                s += (*o as f64 - e).powi(2) / e;
                df += 1;
            }
        }
        (s, df - 1)
    };
    let (stat, df) = chi(&hq);
    let (cstat, _) = chi(&host_q(2.0 * t));
    // chi-square 0.99 quantile ~ df + 2.33*sqrt(2 df) + 2.33^2 (Wilson-Hilferty is overkill here).
    let crit = df as f64 + 2.33 * (2.0 * df as f64).sqrt() + 5.4;
    println!("draws vs softmax(lg/T): chi2 {stat:.1} (df {df}, crit ~{crit:.1}); CONTROL vs softmax(lg/2T): chi2 {cstat:.1}");
    // tiny temperature -> argmax
    KernelLaunch::new(&gpu, k)
        .grid([1, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(d_lg)
        .arg_u32(v as u32)
        .arg_f32(1e-4)
        .arg_f32(0.73)
        .arg_ptr(d_q)
        .arg_ptr(d_ids)
        .arg_i32(0)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut one = [0u8; 4];
    gpu.copy_d2h(d_ids, &mut one)?;
    let am = u32::from_le_bytes(one);
    println!("T=1e-4 draw: {am} (argmax 0)");
    ensure!(cstat > 10.0 * crit, "CONTROL DID NOT FIRE (chi2 {cstat:.1}): the test cannot see a wrong distribution");
    if qerr > 1e-6 || stat > crit || am != 0 {
        bail!("FAIL: q err {qerr:.2e}, chi2 {stat:.1} vs {crit:.1}, tiny-T draw {am}");
    }
    println!("PASS: sampled drafts follow softmax(lg/T); q matches the host; control fired; T->0 = argmax");
    Ok(())
}
