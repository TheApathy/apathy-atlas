// SPDX-License-Identifier: AGPL-3.0-only

//! Correctness + timing oracle for `prefill_attn_compressed_tc` (the
//! m16n8k16 tensor-core rewrite) vs the shipping scalar
//! `prefill_attn_compressed`, at DeepSeek-V4-Flash shapes
//! (head_dim=512, 64 q-heads, MQA kv=1, ratio=4, sliding_window=128).
//!
//! The TC kernel processes 16 keys per online-softmax rescale where the
//! scalar folds one key at a time — same terms, different reduction order —
//! so the gate is COSINE (>= 0.999 per the interleaved-rewrite contract),
//! not byte equality. Edge coverage: S not a multiple of 16 (invalid tail
//! rows), partial key tiles, V==K aliasing on both arms, sinks on.
//!
//! Usage: cargo run --release -p spark-model --example prefill_attn_tc_microtest \
//!            --features cuda,gpu-examples -- [seed]
//! Exit 0 = PASS, 1 = FAIL.

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

const HD: usize = 512;
const NQ: usize = 64;
const NKV: usize = 1;
const RATIO: usize = 4;
const WINDOW: usize = 128;
const ITERS: u32 = 50;

unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

fn gen_bf16(rng: &mut Rng, n: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)).to_le_bytes())
        .collect()
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn bf16_slice(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn launch(
    gpu: &dyn GpuBackend,
    h: spark_runtime::gpu::KernelHandle,
    q: DevicePtr,
    k: DevicePtr,
    v: DevicePtr,
    kc: DevicePtr,
    vc: DevicePtr,
    sinks: DevicePtr,
    o: DevicePtr,
    s: usize,
    n_comp: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, h)
        .grid([NQ as u32, s.div_ceil(16) as u32, 1])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k)
        .arg_ptr(v)
        .arg_ptr(kc)
        .arg_ptr(vc)
        .arg_ptr(sinks)
        .arg_ptr(o)
        .arg_u32(s as u32)
        .arg_u32(NQ as u32)
        .arg_u32(NKV as u32)
        .arg_u32(HD as u32)
        .arg_u32(n_comp as u32)
        .arg_u32(RATIO as u32)
        .arg_u32(WINDOW as u32)
        .arg_f32(1.0 / (HD as f32).sqrt())
        .launch(stream)
}

fn main() -> Result<()> {
    let seed = std::env::args()
        .nth(1)
        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0x7C21);
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let stream = gpu.default_stream();
    let scalar_h = gpu.kernel("prefill_attn_compressed", "prefill_attn_compressed")?;
    let tc_h = gpu.kernel("prefill_attn_compressed", "prefill_attn_compressed_tc")?;
    let mut rng = Rng(seed);

    // ── Correctness at S=253 (non-multiple of 16: invalid tail rows,
    //    partial key tiles on both arms), V==K aliased. ──
    for &(s, aliased) in &[(253usize, true), (160usize, false)] {
        let n_comp = s / RATIO;
        let q = gen_bf16(&mut rng, s * NQ * HD);
        let kraw = gen_bf16(&mut rng, s * NKV * HD);
        let vraw = if aliased {
            kraw.clone()
        } else {
            gen_bf16(&mut rng, s * NKV * HD)
        };
        let kcomp = gen_bf16(&mut rng, n_comp.max(1) * HD);
        let vcomp = if aliased {
            kcomp.clone()
        } else {
            gen_bf16(&mut rng, n_comp.max(1) * HD)
        };
        let sinks: Vec<u8> = (0..NQ)
            .flat_map(|_| rng.uniform(-2.0, 2.0).to_le_bytes())
            .collect();

        let qd = upload(&gpu, &q)?;
        let kd = upload(&gpu, &kraw)?;
        let vd = if aliased { kd } else { upload(&gpu, &vraw)? };
        let kcd = upload(&gpu, &kcomp)?;
        let vcd = if aliased { kcd } else { upload(&gpu, &vcomp)? };
        let sd = upload(&gpu, &sinks)?;
        let o_scalar = gpu.alloc(s * NQ * HD * 2)?;
        let o_tc = gpu.alloc(s * NQ * HD * 2)?;

        launch(&gpu, scalar_h, qd, kd, vd, kcd, vcd, sd, o_scalar, s, n_comp, stream)?;
        launch(&gpu, tc_h, qd, kd, vd, kcd, vcd, sd, o_tc, s, n_comp, stream)?;
        gpu.synchronize(stream)?;

        let mut rs = vec![0u8; s * NQ * HD * 2];
        let mut rt = vec![0u8; s * NQ * HD * 2];
        gpu.copy_d2h(o_scalar, &mut rs)?;
        gpu.copy_d2h(o_tc, &mut rt)?;
        let a = bf16_slice(&rs);
        let b = bf16_slice(&rt);
        let nz = a.iter().filter(|x| **x != 0.0).count();
        if nz == 0 {
            bail!("dead scalar output (S={s})");
        }
        if b.iter().any(|x| !x.is_finite()) {
            let bad = b.iter().position(|x| !x.is_finite()).unwrap();
            bail!("TC produced non-finite output at elem {bad} (S={s}, aliased={aliased})");
        }
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        let mut worst_row_cos = 1.0f64;
        for row in 0..s * NQ {
            let (mut d, mut x2, mut y2) = (0f64, 0f64, 0f64);
            for i in row * HD..(row + 1) * HD {
                d += a[i] * b[i];
                x2 += a[i] * a[i];
                y2 += b[i] * b[i];
            }
            dot += d;
            na += x2;
            nb += y2;
            if x2 > 1e-12 && y2 > 1e-12 {
                worst_row_cos = worst_row_cos.min(d / (x2.sqrt() * y2.sqrt()));
            }
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        println!(
            "  S={s} aliased={aliased}: overall cos={cos:.7} worst-row cos={worst_row_cos:.7} ({nz} nonzero)"
        );
        if cos < 0.999 || worst_row_cos < 0.995 {
            bail!("GATE FAIL: cosine below bar (S={s}, aliased={aliased})");
        }
    }
    println!("  CORRECTNESS PASS (cos gate 0.999 overall / 0.995 worst-row)");

    // ── Timing at S=896 (production-like), V==K aliased. ──
    let s = 896usize;
    let n_comp = s / RATIO;
    let q = gen_bf16(&mut rng, s * NQ * HD);
    let kraw = gen_bf16(&mut rng, s * NKV * HD);
    let kcomp = gen_bf16(&mut rng, n_comp * HD);
    let sinks: Vec<u8> = (0..NQ)
        .flat_map(|_| rng.uniform(-2.0, 2.0).to_le_bytes())
        .collect();
    let qd = upload(&gpu, &q)?;
    let kd = upload(&gpu, &kraw)?;
    let kcd = upload(&gpu, &kcomp)?;
    let sd = upload(&gpu, &sinks)?;
    let od = gpu.alloc(s * NQ * HD * 2)?;

    let time_kernel = |h: spark_runtime::gpu::KernelHandle| -> Result<f64> {
        for _ in 0..5 {
            launch(&gpu, h, qd, kd, kd, kcd, kcd, sd, od, s, n_comp, stream)?;
        }
        gpu.synchronize(stream)?;
        let (mut e0, mut e1) = (0u64, 0u64);
        unsafe {
            cuEventCreate(&mut e0, 0);
            cuEventCreate(&mut e1, 0);
            cuEventRecord(e0, stream);
        }
        for _ in 0..ITERS {
            launch(&gpu, h, qd, kd, kd, kcd, kcd, sd, od, s, n_comp, stream)?;
        }
        unsafe { cuEventRecord(e1, stream) };
        gpu.synchronize(stream)?;
        let mut ms = 0f32;
        unsafe {
            cuEventSynchronize(e1);
            cuEventElapsedTime(&mut ms, e0, e1);
            cuEventDestroy_v2(e0);
            cuEventDestroy_v2(e1);
        }
        Ok(ms as f64 / ITERS as f64)
    };
    let t_scalar = time_kernel(scalar_h)?;
    let t_tc = time_kernel(tc_h)?;
    println!("  S={s}: scalar {t_scalar:.3} ms/call | TC {t_tc:.3} ms/call  [{:.1}x]", t_scalar / t_tc);
    println!(
        "  projected prefill core-attention/pass (43 layers): {:.1} -> {:.1} ms",
        t_scalar * 43.0,
        t_tc * 43.0
    );
    println!("PASS");
    Ok(())
}
