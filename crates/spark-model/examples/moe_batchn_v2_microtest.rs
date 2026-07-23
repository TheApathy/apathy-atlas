// SPDX-License-Identifier: AGPL-3.0-only

//! Correctness + perf oracle for the batchN **v2** MoE expert kernels
//! (`moe_expert_gate_up_shared_batchN_v2` / `moe_expert_silu_down_shared_batchN_v2`)
//! against the shipping batchN GEMVs, at exact Laguna-S-2.1 verify shapes
//! (hidden K=3072, expert inter N=1024, top_k=10, num_tokens=8, NVFP4
//! non-transposed + E4M3 group-16 scales + per-tensor scale2).
//!
//! v2 dedups expert weight reads across tokens (leader block computes all
//! slots routed to its expert) and loads weights as uint4, so it is NOT
//! bit-identical to v1 (k-stride partition differs) — the gate is cosine vs a
//! CPU reference that mirrors the exact dequant (LUT nibble * e4m3(group
//! scale) * scale2) with FP32 accumulation, plus the BF16 narrowing between
//! gate_up and silu_down that the kernels perform.
//!
//! Usage:
//!   cargo run --release -p spark-model --example moe_batchn_v2_microtest \
//!       -- [num_tokens] [top_k] [pool] [seed-hex]
//! Defaults: 8 10 64 0x9E3   (pool = distinct experts tokens draw from; the
//! pointer tables are 256-wide like production, unused ids are NULL).

use anyhow::{bail, Result};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

const HIDDEN: usize = 3072; // K for gate/up, N for down
const INTER: usize = 1024; // N for gate/up, K for down
const GROUP: usize = 16;
const NUM_EXPERT_SLOTS: usize = 256; // pointer-table width (production)
const COSINE_GATE: f64 = 0.9995;

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

const E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
fn f32_to_bf16(f: f32) -> u16 {
    let bits = f.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}
fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((byte >> 3) & 0x0F) as i32;
    let mant = (byte & 0x07) as i32;
    if exp == 0 {
        sign * (mant as f32 / 8.0) * 2f32.powi(-6)
    } else if exp == 0x0F && mant == 0x07 {
        f32::NAN
    } else {
        sign * (1.0 + mant as f32 / 8.0) * 2f32.powi(exp - 7)
    }
}

/// One NVFP4 weight matrix [n, k]: packed nibbles [n, k/2], e4m3 group-16
/// scales [n, k/16], per-tensor scale2.
struct W {
    packed: Vec<u8>,
    scales: Vec<u8>,
    s2: f32,
    n: usize,
    k: usize,
}
impl W {
    fn random(rng: &mut Rng, n: usize, k: usize) -> Self {
        let packed: Vec<u8> = (0..n * k / 2).map(|_| (rng.next_u64() & 0xFF) as u8).collect();
        // e4m3 exponent 3..=9 keeps decode finite and moderate (no NaN byte).
        let scales: Vec<u8> = (0..n * k / GROUP)
            .map(|_| {
                let s = (rng.next_u64() & 1) as u8;
                let e = 3 + (rng.next_u64() % 7) as u8;
                let m = (rng.next_u64() % 8) as u8;
                (s << 7) | (e << 3) | m
            })
            .collect();
        W { packed, scales, s2: rng.uniform(0.02, 0.06), n, k }
    }
    /// Dequantized weight element [row n, col k] — mirrors the kernel exactly.
    fn at(&self, n: usize, k: usize) -> f32 {
        let byte = self.packed[n * self.k / 2 + k / 2];
        let nib = if k % 2 == 0 { byte & 0xF } else { byte >> 4 };
        let sc = e4m3_to_f32(self.scales[n * (self.k / GROUP) + k / GROUP]) * self.s2;
        E2M1_LUT[nib as usize] * sc
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}
fn u16s(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u32s(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u64s(v: &[u64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[allow(clippy::too_many_arguments)]
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let num_tokens: usize = args.get(1).map_or(8, |s| s.parse().unwrap());
    let top_k: usize = args.get(2).map_or(10, |s| s.parse().unwrap());
    let pool: usize = args.get(3).map_or(64, |s| s.parse().unwrap());
    let seed: u64 = args.get(4).map_or(0x9E3, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x9E3)
    });
    assert!(num_tokens <= 8, "v2 kernels cap M at 8");
    let total_routed = num_tokens * top_k;
    let rows_y = (total_routed + num_tokens) as u32; // routed + shared block rows

    println!(
        "=== batchN v2 microtest: tokens={num_tokens} top_k={top_k} pool={pool} \
         K={HIDDEN} N={INTER} seed=0x{seed:X} ==="
    );

    let mut rng = Rng(seed);
    // A [num_tokens, HIDDEN]
    let a_bf16: Vec<u16> = (0..num_tokens * HIDDEN)
        .map(|_| f32_to_bf16(rng.uniform(-1.0, 1.0)))
        .collect();
    // expert pool: gate/up [INTER, HIDDEN], down [HIDDEN, INTER]
    let gates: Vec<W> = (0..pool).map(|_| W::random(&mut rng, INTER, HIDDEN)).collect();
    let ups: Vec<W> = (0..pool).map(|_| W::random(&mut rng, INTER, HIDDEN)).collect();
    let downs: Vec<W> = (0..pool).map(|_| W::random(&mut rng, HIDDEN, INTER)).collect();
    let sh_gate = W::random(&mut rng, INTER, HIDDEN);
    let sh_up = W::random(&mut rng, INTER, HIDDEN);
    let sh_down = W::random(&mut rng, HIDDEN, INTER);

    // per-token top_k distinct experts from the pool (cross-token dups expected)
    let mut indices = vec![0u32; total_routed];
    for t in 0..num_tokens {
        let mut chosen: Vec<u32> = Vec::with_capacity(top_k);
        while chosen.len() < top_k {
            let e = (rng.next_u64() % pool as u64) as u32;
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        indices[t * top_k..(t + 1) * top_k].copy_from_slice(&chosen);
    }
    let unique: std::collections::HashSet<u32> = indices.iter().copied().collect();
    println!(
        "routing: {} slots -> {} unique experts (dedup factor {:.2}x; shared {}x -> 1x)",
        total_routed,
        unique.len(),
        total_routed as f64 / unique.len() as f64,
        num_tokens
    );

    // ── GPU setup ──
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let a_ptr = upload(gpu, &u16s(&a_bf16))?;
    let up_w = |w: &W, gpu: &dyn GpuBackend| -> Result<(DevicePtr, DevicePtr)> {
        Ok((upload(gpu, &w.packed)?, upload(gpu, &w.scales)?))
    };
    let mut tables: Vec<(Vec<u64>, Vec<u64>, Vec<f32>)> = Vec::new(); // gate/up/down
    for ws in [&gates, &ups, &downs] {
        let mut wp = vec![0u64; NUM_EXPERT_SLOTS];
        let mut sp = vec![0u64; NUM_EXPERT_SLOTS];
        let mut s2 = vec![0f32; NUM_EXPERT_SLOTS];
        for (e, w) in ws.iter().enumerate() {
            let (p, s) = up_w(w, gpu)?;
            wp[e] = p.0;
            sp[e] = s.0;
            s2[e] = w.s2;
        }
        tables.push((wp, sp, s2));
    }
    let gate_tbl = (
        upload(gpu, &u64s(&tables[0].0))?,
        upload(gpu, &u64s(&tables[0].1))?,
        upload(gpu, &f32s(&tables[0].2))?,
    );
    let up_tbl = (
        upload(gpu, &u64s(&tables[1].0))?,
        upload(gpu, &u64s(&tables[1].1))?,
        upload(gpu, &f32s(&tables[1].2))?,
    );
    let down_tbl = (
        upload(gpu, &u64s(&tables[2].0))?,
        upload(gpu, &u64s(&tables[2].1))?,
        upload(gpu, &f32s(&tables[2].2))?,
    );
    let (sh_gate_p, sh_gate_s) = up_w(&sh_gate, gpu)?;
    let (sh_up_p, sh_up_s) = up_w(&sh_up, gpu)?;
    let (sh_down_p, sh_down_s) = up_w(&sh_down, gpu)?;
    let idx_ptr = upload(gpu, &u32s(&indices))?;

    let gate_out = gpu.alloc(total_routed * INTER * 2)?;
    let up_out = gpu.alloc(total_routed * INTER * 2)?;
    let down_out = gpu.alloc(total_routed * HIDDEN * 2)?;
    let sh_gate_out = gpu.alloc(num_tokens * INTER * 2)?;
    let sh_up_out = gpu.alloc(num_tokens * INTER * 2)?;
    let sh_down_out = gpu.alloc(num_tokens * HIDDEN * 2)?;
    // v4 decoupled-silu buffers
    let act_routed = gpu.alloc(total_routed * INTER * 2)?;
    let sh_act = gpu.alloc(num_tokens * INTER * 2)?;

    let launch_gate_up = |name: &str, block: u32, stream: u64| -> Result<()> {
        let h = gpu.kernel("moe_fused_batch2", name)?;
        // v3 stages A per K-tile: num_tokens * 1024 * 2B dynamic smem
        let smem = if name.ends_with("_v3") { (num_tokens * 1024 * 2) as u32 } else { 0 };
        KernelLaunch::new(gpu, h)
            .grid([(INTER as u32).div_ceil(8), rows_y, 2])
            .block([block, 1, 1])
            .shared_mem(smem)
            .arg_ptr(a_ptr)
            .arg_ptr(gate_tbl.0)
            .arg_ptr(gate_tbl.1)
            .arg_ptr(gate_tbl.2)
            .arg_ptr(gate_out)
            .arg_ptr(up_tbl.0)
            .arg_ptr(up_tbl.1)
            .arg_ptr(up_tbl.2)
            .arg_ptr(up_out)
            .arg_ptr(idx_ptr)
            .arg_ptr(sh_gate_p)
            .arg_ptr(sh_gate_s)
            .arg_f32(sh_gate.s2)
            .arg_ptr(sh_gate_out)
            .arg_ptr(sh_up_p)
            .arg_ptr(sh_up_s)
            .arg_f32(sh_up.s2)
            .arg_ptr(sh_up_out)
            .arg_u32(INTER as u32)
            .arg_u32(HIDDEN as u32)
            .arg_u32(top_k as u32)
            .arg_u32(num_tokens as u32)
            .launch(stream)
    };
    let launch_silu_down = |name: &str, block: u32, smem: u32, stream: u64| -> Result<()> {
        let h = gpu.kernel("moe_fused_batch2", name)?;
        // v2 covers 128 rows/block (4 warps x 16 pairs); v1 covers 8
        let rows_per_block = if name.ends_with("_v2") { 128 } else { 8 };
        KernelLaunch::new(gpu, h)
            .grid([(HIDDEN as u32).div_ceil(rows_per_block), rows_y, 1])
            .block([block, 1, 1])
            .shared_mem(smem)
            .arg_ptr(gate_out)
            .arg_ptr(up_out)
            .arg_ptr(down_tbl.0)
            .arg_ptr(down_tbl.1)
            .arg_ptr(down_tbl.2)
            .arg_ptr(down_out)
            .arg_ptr(idx_ptr)
            .arg_ptr(sh_gate_out)
            .arg_ptr(sh_up_out)
            .arg_ptr(sh_down_p)
            .arg_ptr(sh_down_s)
            .arg_f32(sh_down.s2)
            .arg_ptr(sh_down_out)
            .arg_u32(HIDDEN as u32)
            .arg_u32(INTER as u32)
            .arg_u32(top_k as u32)
            .arg_u32(num_tokens as u32)
            .launch(stream)
    };

    // v4 down: silu precompute (gate_out/up_out → act_routed, sh → sh_act),
    // then dedup down reads act directly. rows_y = total_routed + num_tokens
    // leader-block grid; 8 rows/block over HIDDEN (V2 pair layout).
    let launch_down_v4 = |stream: u64| -> Result<()> {
        let pre = gpu.kernel("moe_fused_batch2", "moe_silu_precompute_batchN")?;
        let elems = (total_routed + num_tokens) * INTER;
        KernelLaunch::new(gpu, pre)
            .grid([(elems as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(gate_out)
            .arg_ptr(up_out)
            .arg_ptr(act_routed)
            .arg_ptr(sh_gate_out)
            .arg_ptr(sh_up_out)
            .arg_ptr(sh_act)
            .arg_u32(INTER as u32)
            .arg_u32(total_routed as u32)
            .arg_u32(num_tokens as u32)
            .launch(stream)?;
        let dn = gpu.kernel("moe_fused_batch2", "moe_expert_down_dedup_batchN")?;
        KernelLaunch::new(gpu, dn)
            .grid([(HIDDEN as u32).div_ceil(8), rows_y, 1])
            .block([128, 1, 1])
            .arg_ptr(act_routed)
            .arg_ptr(sh_act)
            .arg_ptr(down_tbl.0)
            .arg_ptr(down_tbl.1)
            .arg_ptr(down_tbl.2)
            .arg_ptr(down_out)
            .arg_ptr(idx_ptr)
            .arg_ptr(sh_down_p)
            .arg_ptr(sh_down_s)
            .arg_f32(sh_down.s2)
            .arg_ptr(sh_down_out)
            .arg_u32(HIDDEN as u32)
            .arg_u32(INTER as u32)
            .arg_u32(top_k as u32)
            .arg_u32(num_tokens as u32)
            .launch(stream)
    };

    let v1_smem = (INTER * 4) as u32;
    let v2_smem = (num_tokens * INTER * 4) as u32;
    // production block choice for hidden>=3072 is 256 (v1); v2 fixes 128
    let run_pair = |mode: u8, stream: u64| -> Result<()> {
        match mode {
            1 => {
                launch_gate_up("moe_expert_gate_up_shared_batchN_v2", 128, stream)?;
                launch_silu_down("moe_expert_silu_down_shared_batchN_v2", 128, v2_smem, stream)
            }
            2 => {
                launch_gate_up("moe_expert_gate_up_shared_batchN_v3", 128, stream)?;
                launch_silu_down("moe_expert_silu_down_shared_batchN", 128, v1_smem, stream)
            }
            3 => {
                // v4: v2 gate_up (shipped best) + decoupled dedup down
                launch_gate_up("moe_expert_gate_up_shared_batchN_v2", 128, stream)?;
                launch_down_v4(stream)
            }
            _ => {
                launch_gate_up("moe_expert_gate_up_shared_batchN", 256, stream)?;
                launch_silu_down("moe_expert_silu_down_shared_batchN", 256, v1_smem, stream)
            }
        }
    };

    let read_bf16 = |ptr: DevicePtr, count: usize| -> Result<Vec<f32>> {
        let mut buf = vec![0u8; count * 2];
        gpu.copy_d2h(ptr, &mut buf)?;
        Ok(buf
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect())
    };

    // ── CPU reference (FP32 accumulate, BF16 narrowing between stages) ──
    let a_f: Vec<f32> = a_bf16.iter().map(|&b| bf16_to_f32(b)).collect();
    let gemv = |w: &W, a_row: &[f32]| -> Vec<f32> {
        (0..w.n)
            .map(|n| (0..w.k).map(|k| a_row[k] * w.at(n, k)).sum())
            .collect()
    };
    let narrow = |v: Vec<f32>| -> Vec<f32> { v.into_iter().map(|x| bf16_to_f32(f32_to_bf16(x))).collect() };
    let mut ref_gate = vec![0f32; total_routed * INTER];
    let mut ref_up = vec![0f32; total_routed * INTER];
    let mut ref_down = vec![0f32; total_routed * HIDDEN];
    let mut ref_sh_gate = vec![0f32; num_tokens * INTER];
    let mut ref_sh_up = vec![0f32; num_tokens * INTER];
    let mut ref_sh_down = vec![0f32; num_tokens * HIDDEN];
    for slot in 0..total_routed {
        let t = slot / top_k;
        let e = indices[slot] as usize;
        let a_row = &a_f[t * HIDDEN..(t + 1) * HIDDEN];
        let g = narrow(gemv(&gates[e], a_row));
        let u = narrow(gemv(&ups[e], a_row));
        let act: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(&gf, &uf)| (gf / (1.0 + (-gf).exp())) * uf)
            .collect();
        let d = narrow(gemv(&downs[e], &act));
        ref_gate[slot * INTER..(slot + 1) * INTER].copy_from_slice(&g);
        ref_up[slot * INTER..(slot + 1) * INTER].copy_from_slice(&u);
        ref_down[slot * HIDDEN..(slot + 1) * HIDDEN].copy_from_slice(&d);
    }
    for t in 0..num_tokens {
        let a_row = &a_f[t * HIDDEN..(t + 1) * HIDDEN];
        let g = narrow(gemv(&sh_gate, a_row));
        let u = narrow(gemv(&sh_up, a_row));
        let act: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(&gf, &uf)| (gf / (1.0 + (-gf).exp())) * uf)
            .collect();
        let d = narrow(gemv(&sh_down, &act));
        ref_sh_gate[t * INTER..(t + 1) * INTER].copy_from_slice(&g);
        ref_sh_up[t * INTER..(t + 1) * INTER].copy_from_slice(&u);
        ref_sh_down[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&d);
    }

    // ── correctness: v1 then v2, each vs the CPU reference ──
    let mut fail = false;
    let mut v1_down_all: Vec<f32> = Vec::new();
    for (label, mode) in [("v1", 0u8), ("v2", 1u8), ("v3", 2u8), ("v4", 3u8)] {
        run_pair(mode, stream)?;
        gpu.synchronize(stream)?;
        let g = read_bf16(gate_out, total_routed * INTER)?;
        let d = read_bf16(down_out, total_routed * HIDDEN)?;
        let sg = read_bf16(sh_gate_out, num_tokens * INTER)?;
        let sd = read_bf16(sh_down_out, num_tokens * HIDDEN)?;
        let cg = cosine(&g, &ref_gate);
        let cd = cosine(&d, &ref_down);
        let csg = cosine(&sg, &ref_sh_gate);
        let csd = cosine(&sd, &ref_sh_down);
        println!(
            "{label}: cos(gate)={cg:.6} cos(down)={cd:.6} cos(sh_gate)={csg:.6} cos(sh_down)={csd:.6}"
        );
        if [cg, cd, csg, csd].iter().any(|c| !(*c >= COSINE_GATE)) {
            fail = true;
        }
        if mode == 0 {
            v1_down_all = d;
        } else {
            let dvd = cosine(&d, &v1_down_all);
            println!("{label}-vs-v1 down cosine: {dvd:.6}");
        }
    }

    // ── perf: event-timed, each kernel separately, several configs ──
    let time_it = |f: &dyn Fn(u64) -> Result<()>, stream: u64| -> Result<f64> {
        let iters = 200;
        for _ in 0..10 {
            f(stream)?;
        }
        gpu.synchronize(stream)?;
        let (mut e0, mut e1): (u64, u64) = (0, 0);
        if unsafe { cuEventCreate(&mut e0, 0) } != 0 || unsafe { cuEventCreate(&mut e1, 0) } != 0 {
            bail!("cuEventCreate failed");
        }
        if unsafe { cuEventRecord(e0, stream) } != 0 {
            bail!("cuEventRecord failed");
        }
        for _ in 0..iters {
            f(stream)?;
        }
        if unsafe { cuEventRecord(e1, stream) } != 0 {
            bail!("cuEventRecord failed");
        }
        if unsafe { cuEventSynchronize(e1) } != 0 {
            bail!("cuEventSynchronize failed");
        }
        let mut ms = 0f32;
        if unsafe { cuEventElapsedTime(&mut ms, e0, e1) } != 0 {
            bail!("cuEventElapsedTime failed");
        }
        unsafe {
            cuEventDestroy_v2(e0);
            cuEventDestroy_v2(e1);
        }
        Ok(ms as f64 / iters as f64)
    };
    // weight bytes actually needed once per layer (packed+scales, dedup):
    let per_expert =
        (INTER * HIDDEN / 2 + INTER * HIDDEN / GROUP) * 2 + HIDDEN * INTER / 2 + HIDDEN * INTER / GROUP;
    let ideal_bytes = (unique.len() + 1) * per_expert;
    println!("dedup-ideal weight read {:.1} MB/layer", ideal_bytes as f64 / 1e6);
    let configs: Vec<(&str, Box<dyn Fn(u64) -> Result<()>>, Box<dyn Fn(u64) -> Result<()>>)> = vec![
        (
            "v1@256",
            Box::new(|s| launch_gate_up("moe_expert_gate_up_shared_batchN", 256, s)),
            Box::new(move |s| launch_silu_down("moe_expert_silu_down_shared_batchN", 256, v1_smem, s)),
        ),
        (
            "v1@128",
            Box::new(|s| launch_gate_up("moe_expert_gate_up_shared_batchN", 128, s)),
            Box::new(move |s| launch_silu_down("moe_expert_silu_down_shared_batchN", 128, v1_smem, s)),
        ),
        (
            "v2@128",
            Box::new(|s| launch_gate_up("moe_expert_gate_up_shared_batchN_v2", 128, s)),
            Box::new(move |s| launch_silu_down("moe_expert_silu_down_shared_batchN_v2", 128, v2_smem, s)),
        ),
        (
            "v3@128",
            Box::new(|s| launch_gate_up("moe_expert_gate_up_shared_batchN_v3", 128, s)),
            Box::new(move |s| launch_silu_down("moe_expert_silu_down_shared_batchN", 128, v1_smem, s)),
        ),
        (
            "v4@128 (dedup-down)",
            Box::new(|s| launch_gate_up("moe_expert_gate_up_shared_batchN_v2", 128, s)),
            Box::new(&launch_down_v4),
        ),
    ];
    for (label, gu, sd) in &configs {
        let t_gu = time_it(gu.as_ref(), stream)?;
        let t_sd = time_it(sd.as_ref(), stream)?;
        println!(
            "{label}: gate_up {t_gu:.4} ms + silu_down {t_sd:.4} ms = {:.4} ms/layer  ({:.0} GB/s eff)",
            t_gu + t_sd,
            ideal_bytes as f64 / ((t_gu + t_sd) / 1e3) / 1e9
        );
    }

    if fail {
        eprintln!("RESULT: FAIL (cosine below {COSINE_GATE})");
        std::process::exit(1);
    }
    println!("RESULT: PASS");
    Ok(())
}
