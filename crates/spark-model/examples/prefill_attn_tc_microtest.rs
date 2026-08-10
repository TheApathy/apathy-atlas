// SPDX-License-Identifier: AGPL-3.0-only

//! Correctness + timing oracle for the m16n8k16 tensor-core prefill attention
//! kernels — `prefill_attn_compressed_tc` (round 1) and
//! `prefill_attn_compressed_tc2` (round 2: natural-K staging, single aliased
//! K/V tile, ldmatrix B operands, P kept in registers) — against the shipping
//! scalar `prefill_attn_compressed`, at DeepSeek-V4-Flash shapes
//! (head_dim=512, 64 q-heads, MQA kv=1).
//!
//! The TC kernels process 16 keys per online-softmax rescale where the scalar
//! folds one key at a time — same terms, different reduction order — so the
//! gate vs scalar is COSINE (>= 0.999 overall / 0.995 worst-row, per the
//! interleaved-rewrite contract), not byte equality.
//!
//! tc2 is a pure DATA-MOVEMENT rewrite of tc: same MMA operand values, same
//! sSp summation order, same softmax terms, same bf16 rounding of P. It is
//! therefore held to a much tighter bar against tc (cos >= 0.9999999) and the
//! exact-match fraction is reported — anything below 100% exact is FMA
//! re-association, anything below the cosine bar is a fragment-layout bug.
//!
//! Correctness coverage, run at BOTH production configurations:
//!   - CSA  (sliding_window=128, ratio=4)   — compressor layers
//!   - HCA  (sliding_window=0,   ratio=128) — full-causal layers
//! crossed with S=253 (not a multiple of 16: invalid tail rows + partial key
//! tiles on both arms) V==K aliased, and S=160 NOT aliased (which drives
//! tc2's in-place V re-stage path). Sinks on throughout.
//!
//! Usage: cargo run --release -p spark-model --example prefill_attn_tc_microtest \
//!            --features cuda,gpu-examples -- [seed] [S] [window] [ratio]
//! Timing invocations that reproduce the round-2 report:
//!     ... -- 7C21 2176 128 4     (CSA)
//!     ... -- 7C21 2176 0   128   (HCA, full causal)
//! Exit 0 = PASS, 1 = FAIL.

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

const HD: usize = 512;
const NQ: usize = 64;
const NKV: usize = 1;
const ITERS: u32 = 50;
// Overridable at argv[2..]: S window ratio (defaults 896 128 4).
fn cfg() -> (usize, usize, usize) {
    let a: Vec<usize> = std::env::args()
        .skip(2)
        .filter_map(|s| s.parse().ok())
        .collect();
    (
        a.first().copied().unwrap_or(896),
        a.get(1).copied().unwrap_or(128),
        a.get(2).copied().unwrap_or(4),
    )
}

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
    window: usize,
    ratio: usize,
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
        .arg_u32(ratio as u32)
        .arg_u32(window as u32)
        .arg_f32(1.0 / (HD as f32).sqrt())
        .launch(stream)
}

/// Overall cosine, worst per-row cosine, and the fraction of elements that
/// match bit-for-bit. Rows are the [HD] output vectors.
fn compare(a: &[f64], b: &[f64], ra: &[u8], rb: &[u8]) -> (f64, f64, f64) {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    let mut worst_row_cos = 1.0f64;
    for row in 0..a.len() / HD {
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
    let exact = ra
        .chunks_exact(2)
        .zip(rb.chunks_exact(2))
        .filter(|(x, y)| x == y)
        .count() as f64
        / (ra.len() / 2) as f64;
    (dot / (na.sqrt() * nb.sqrt()), worst_row_cos, exact)
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
    let tc2_h = gpu.kernel("prefill_attn_compressed", "prefill_attn_compressed_tc2")?;
    let mut rng = Rng(seed);

    // ── Correctness: {S=253 aliased, S=160 not aliased} x {CSA, HCA}. ──
    // S=253 is not a multiple of 16 (invalid tail rows + partial key tiles on
    // both arms); the non-aliased case drives tc2's in-place V re-stage.
    for &(s, aliased) in &[(253usize, true), (160usize, false)] {
        for &(window, ratio, arm) in &[(128usize, 4usize, "CSA"), (0usize, 128usize, "HCA")] {
            let n_comp = s / ratio;
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
            let o_tc2 = gpu.alloc(s * NQ * HD * 2)?;

            #[rustfmt::skip]
            launch(&gpu, scalar_h, qd, kd, vd, kcd, vcd, sd, o_scalar, s, n_comp, window, ratio, stream)?;
            #[rustfmt::skip]
            launch(&gpu, tc_h, qd, kd, vd, kcd, vcd, sd, o_tc, s, n_comp, window, ratio, stream)?;
            #[rustfmt::skip]
            launch(&gpu, tc2_h, qd, kd, vd, kcd, vcd, sd, o_tc2, s, n_comp, window, ratio, stream)?;
            gpu.synchronize(stream)?;

            let mut rs = vec![0u8; s * NQ * HD * 2];
            let mut rt = vec![0u8; s * NQ * HD * 2];
            let mut r2 = vec![0u8; s * NQ * HD * 2];
            gpu.copy_d2h(o_scalar, &mut rs)?;
            gpu.copy_d2h(o_tc, &mut rt)?;
            gpu.copy_d2h(o_tc2, &mut r2)?;
            let a = bf16_slice(&rs);
            let b = bf16_slice(&rt);
            let c = bf16_slice(&r2);
            let nz = a.iter().filter(|x| **x != 0.0).count();
            if nz == 0 {
                bail!("dead scalar output (S={s} {arm})");
            }
            for (name, v) in [("tc", &b), ("tc2", &c)] {
                if let Some(bad) = v.iter().position(|x| !x.is_finite()) {
                    bail!(
                        "{name} produced non-finite output at elem {bad} \
                         (S={s}, aliased={aliased}, {arm})"
                    );
                }
            }

            let (cos_tc, wr_tc, _) = compare(&a, &b, &rs, &rt);
            let (cos_2, wr_2, _) = compare(&a, &c, &rs, &r2);
            // tc2 is a data-movement-only rewrite of tc: same MMA operands,
            // same reduction order, same bf16 rounding of P. Only FMA
            // re-association may separate them, so the bar is far tighter.
            let (cos_21, wr_21, exact_21) = compare(&b, &c, &rt, &r2);
            println!(
                "  S={s} aliased={aliased} {arm} (w={window} r={ratio}): \
                 tc cos={cos_tc:.7}/{wr_tc:.7}  tc2 cos={cos_2:.7}/{wr_2:.7}  \
                 tc2-vs-tc cos={cos_21:.9}/{wr_21:.9} exact={:.4}%  ({nz} nonzero)",
                exact_21 * 100.0
            );
            if cos_tc < 0.999 || wr_tc < 0.995 {
                bail!("GATE FAIL: tc cosine below bar (S={s}, aliased={aliased}, {arm})");
            }
            if cos_2 < 0.999 || wr_2 < 0.995 {
                bail!("GATE FAIL: tc2 cosine below bar (S={s}, aliased={aliased}, {arm})");
            }
            if cos_21 < 0.999_999_9 || wr_21 < 0.999_99 {
                bail!(
                    "GATE FAIL: tc2 diverges from tc beyond FMA re-association \
                     (S={s}, aliased={aliased}, {arm}) — suspect an ldmatrix \
                     fragment mapping, not numerics"
                );
            }
        }
    }
    println!(
        "  CORRECTNESS PASS (vs scalar: cos 0.999 overall / 0.995 worst-row; \
         vs tc: cos 0.9999999 / 0.99999 worst-row)"
    );

    // ── Timing (argv-configurable), V==K aliased. ──
    let (s, window, ratio) = cfg();
    let n_comp = s / ratio;
    let q = gen_bf16(&mut rng, s * NQ * HD);
    let kraw = gen_bf16(&mut rng, s * NKV * HD);
    let kcomp = gen_bf16(&mut rng, n_comp.max(1) * HD);
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
            launch(&gpu, h, qd, kd, kd, kcd, kcd, sd, od, s, n_comp, window, ratio, stream)?;
        }
        gpu.synchronize(stream)?;
        let (mut e0, mut e1) = (0u64, 0u64);
        unsafe {
            cuEventCreate(&mut e0, 0);
            cuEventCreate(&mut e1, 0);
            cuEventRecord(e0, stream);
        }
        for _ in 0..ITERS {
            launch(&gpu, h, qd, kd, kd, kcd, kcd, sd, od, s, n_comp, window, ratio, stream)?;
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
    let t_tc2 = time_kernel(tc2_h)?;
    println!(
        "  S={s} window={window} ratio={ratio}: scalar {t_scalar:.3} | TC {t_tc:.3} \
         | TC2 {t_tc2:.3} ms/call   [tc {:.2}x scalar, tc2 {:.2}x scalar, \
         tc2 {:.2}x tc]",
        t_scalar / t_tc,
        t_scalar / t_tc2,
        t_tc / t_tc2
    );
    println!(
        "  projected prefill core-attention/pass (43 layers): \
         scalar {:.1} -> tc {:.1} -> tc2 {:.1} ms",
        t_scalar * 43.0,
        t_tc * 43.0,
        t_tc2 * 43.0
    );
    println!("PASS");
    Ok(())
}
