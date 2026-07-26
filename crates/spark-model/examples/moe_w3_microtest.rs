// SPDX-License-Identifier: AGPL-3.0-only

//! Kernel-vs-CPU-reference microtest for the W3 (3-bit Lloyd-Max) MoE
//! kernels at exact Laguna-S-2.1 shapes (hidden K=3072, expert inter N=1024,
//! top_k=10), following the `moe_batchn_v2_microtest` harness pattern.
//!
//! Covers every `_w3` kernel the ATLAS_MOE_W3 path dispatches:
//!   [1] single-token pair  (moe_expert_gate_up_shared_w3 + silu_down_w3)
//!   [2] batchN v1 pair     (…_batchN_w3 pair, block 256 — production choice
//!                           for hidden >= 3072)
//!   [3] batchN v2 + v4     (…_batchN_v2_w3 + silu_precompute (NVFP4 module,
//!                           weight-free) + …_down_dedup_batchN_w3)
//!   [4] grouped GEMM       (moe_w3a16_grouped_gemm_ptrtable, prefill)
//!
//! Routed weights are random 3-bit Turbo3-packed indices dequanted through a
//! synthetic 8-entry codebook × e4m3 group scale × scale2; the shared-expert
//! slots stay NVFP4 (exercising the dual-format branch). Reference mirrors
//! the kernels' dequant + FP32 accumulate + BF16 stage narrowing.
//!
//! Usage:
//!   cargo run --release -p spark-model --features gpu-examples \
//!       --example moe_w3_microtest -- [num_tokens] [top_k] [pool] [seed-hex]

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

const HIDDEN: usize = 3072; // K for gate/up, N for down
const INTER: usize = 1024; // N for gate/up, K for down
const GROUP: usize = 16;
const NUM_EXPERT_SLOTS: usize = 256;

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
// Synthetic Lloyd-Max codebook (E2M1 units, sign in bit 2).
const W3_LUT: [f32; 8] = [0.57, 1.77, 3.43, 6.0, -0.57, -1.77, -3.43, -6.0];

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

fn random_scales(rng: &mut Rng, count: usize) -> Vec<u8> {
    (0..count)
        .map(|_| {
            let s = (rng.next_u64() & 1) as u8;
            let e = 3 + (rng.next_u64() % 7) as u8;
            let m = (rng.next_u64() % 8) as u8;
            (s << 7) | (e << 3) | m
        })
        .collect()
}

/// W3 weight matrix [n, k]: Turbo3-packed 3-bit indices [n, k*3/8],
/// e4m3 group-16 scales [n, k/16], per-tensor scale2.
struct W3W {
    packed3: Vec<u8>,
    scales: Vec<u8>,
    s2: f32,
    #[allow(dead_code)]
    n: usize,
    k: usize,
}
impl W3W {
    fn random(rng: &mut Rng, n: usize, k: usize) -> Self {
        let mut packed3 = vec![0u8; n * k * 3 / 8];
        for trio in packed3.chunks_exact_mut(3) {
            let mut bits: u32 = 0;
            for j in 0..8 {
                bits |= ((rng.next_u64() % 8) as u32) << (3 * j);
            }
            trio[0] = bits as u8;
            trio[1] = (bits >> 8) as u8;
            trio[2] = (bits >> 16) as u8;
        }
        let scales = random_scales(rng, n * k / GROUP);
        W3W {
            packed3,
            scales,
            s2: rng.uniform(0.02, 0.06),
            n,
            k,
        }
    }
    fn at(&self, n: usize, k: usize) -> f32 {
        let row3 = self.k * 3 / 8;
        let p = &self.packed3[n * row3 + (k / 8) * 3..];
        let bits = p[0] as u32 | ((p[1] as u32) << 8) | ((p[2] as u32) << 16);
        let idx = (bits >> (3 * (k % 8))) & 7;
        let sc = e4m3_to_f32(self.scales[n * (self.k / GROUP) + k / GROUP]) * self.s2;
        W3_LUT[idx as usize] * sc
    }
}

/// NVFP4 weight (shared-expert slots).
struct NvW {
    packed: Vec<u8>,
    scales: Vec<u8>,
    s2: f32,
    #[allow(dead_code)]
    n: usize,
    k: usize,
}
impl NvW {
    fn random(rng: &mut Rng, n: usize, k: usize) -> Self {
        let packed: Vec<u8> = (0..n * k / 2)
            .map(|_| (rng.next_u64() & 0xFF) as u8)
            .collect();
        let scales = random_scales(rng, n * k / GROUP);
        NvW {
            packed,
            scales,
            s2: rng.uniform(0.02, 0.06),
            n,
            k,
        }
    }
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
fn i32s(v: &[i32]) -> Vec<u8> {
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
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let num_tokens: usize = args.get(1).map_or(8, |s| s.parse().unwrap());
    let top_k: usize = args.get(2).map_or(10, |s| s.parse().unwrap());
    let pool: usize = args.get(3).map_or(64, |s| s.parse().unwrap());
    let seed: u64 = args.get(4).map_or(0x3B17, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x3B17)
    });
    assert!(num_tokens <= 8, "v2 kernels cap M at 8");
    let total_routed = num_tokens * top_k;
    let rows_y = (total_routed + num_tokens) as u32;

    println!(
        "=== W3 microtest: tokens={num_tokens} top_k={top_k} pool={pool} \
         K={HIDDEN} N={INTER} seed=0x{seed:X} ==="
    );

    let mut rng = Rng(seed);
    let a_bf16: Vec<u16> = (0..num_tokens * HIDDEN)
        .map(|_| f32_to_bf16(rng.uniform(-1.0, 1.0)))
        .collect();
    let gates: Vec<W3W> = (0..pool)
        .map(|_| W3W::random(&mut rng, INTER, HIDDEN))
        .collect();
    let ups: Vec<W3W> = (0..pool)
        .map(|_| W3W::random(&mut rng, INTER, HIDDEN))
        .collect();
    let downs: Vec<W3W> = (0..pool)
        .map(|_| W3W::random(&mut rng, HIDDEN, INTER))
        .collect();
    let sh_gate = NvW::random(&mut rng, INTER, HIDDEN);
    let sh_up = NvW::random(&mut rng, INTER, HIDDEN);
    let sh_down = NvW::random(&mut rng, HIDDEN, INTER);

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

    // ── GPU setup ──
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let a_ptr = upload(gpu, &u16s(&a_bf16))?;
    let lut_ptr = upload(gpu, &f32s(&W3_LUT))?;
    let mut tables: Vec<(Vec<u64>, Vec<u64>, Vec<f32>)> = Vec::new();
    for ws in [&gates, &ups, &downs] {
        let mut wp = vec![0u64; NUM_EXPERT_SLOTS];
        let mut sp = vec![0u64; NUM_EXPERT_SLOTS];
        let mut s2 = vec![0f32; NUM_EXPERT_SLOTS];
        for (e, w) in ws.iter().enumerate() {
            wp[e] = upload(gpu, &w.packed3)?.0;
            sp[e] = upload(gpu, &w.scales)?.0;
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
    let sh_gate_p = upload(gpu, &sh_gate.packed)?;
    let sh_gate_s = upload(gpu, &sh_gate.scales)?;
    let sh_up_p = upload(gpu, &sh_up.packed)?;
    let sh_up_s = upload(gpu, &sh_up.scales)?;
    let sh_down_p = upload(gpu, &sh_down.packed)?;
    let sh_down_s = upload(gpu, &sh_down.scales)?;
    let idx_ptr = upload(gpu, &u32s(&indices))?;

    let gate_out = gpu.alloc(total_routed * INTER * 2)?;
    let up_out = gpu.alloc(total_routed * INTER * 2)?;
    let down_out = gpu.alloc(total_routed * HIDDEN * 2)?;
    let sh_gate_out = gpu.alloc(num_tokens * INTER * 2)?;
    let sh_up_out = gpu.alloc(num_tokens * INTER * 2)?;
    let sh_down_out = gpu.alloc(num_tokens * HIDDEN * 2)?;
    let act_routed = gpu.alloc(total_routed * INTER * 2)?;
    let sh_act = gpu.alloc(num_tokens * INTER * 2)?;

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
    let narrow =
        |v: Vec<f32>| -> Vec<f32> { v.into_iter().map(|x| bf16_to_f32(f32_to_bf16(x))).collect() };
    let mut ref_gate = vec![0f32; total_routed * INTER];
    let mut ref_up = vec![0f32; total_routed * INTER];
    let mut ref_down = vec![0f32; total_routed * HIDDEN];
    let mut ref_sh_gate = vec![0f32; num_tokens * INTER];
    let mut ref_sh_up = vec![0f32; num_tokens * INTER];
    let mut ref_sh_down = vec![0f32; num_tokens * HIDDEN];
    for slot in 0..total_routed {
        let t = slot / top_k;
        let a_row = &a_f[t * HIDDEN..(t + 1) * HIDDEN];
        let g = &gates[indices[slot] as usize];
        let u = &ups[indices[slot] as usize];
        let gr: Vec<f32> = (0..INTER)
            .map(|n| (0..HIDDEN).map(|k| a_row[k] * g.at(n, k)).sum())
            .collect();
        let ur: Vec<f32> = (0..INTER)
            .map(|n| (0..HIDDEN).map(|k| a_row[k] * u.at(n, k)).sum())
            .collect();
        ref_gate[slot * INTER..(slot + 1) * INTER].copy_from_slice(&narrow(gr));
        ref_up[slot * INTER..(slot + 1) * INTER].copy_from_slice(&narrow(ur));
    }
    for t in 0..num_tokens {
        let a_row = &a_f[t * HIDDEN..(t + 1) * HIDDEN];
        let gr: Vec<f32> = (0..INTER)
            .map(|n| (0..HIDDEN).map(|k| a_row[k] * sh_gate.at(n, k)).sum())
            .collect();
        let ur: Vec<f32> = (0..INTER)
            .map(|n| (0..HIDDEN).map(|k| a_row[k] * sh_up.at(n, k)).sum())
            .collect();
        ref_sh_gate[t * INTER..(t + 1) * INTER].copy_from_slice(&narrow(gr));
        ref_sh_up[t * INTER..(t + 1) * INTER].copy_from_slice(&narrow(ur));
    }
    // silu(gate)*up from the NARROWED stage outputs (what the kernels read).
    let silu = |g: f32| g / (1.0 + (-g).exp());
    for slot in 0..total_routed {
        let d = &downs[indices[slot] as usize];
        let act: Vec<f32> = (0..INTER)
            .map(|i| silu(ref_gate[slot * INTER + i]) * ref_up[slot * INTER + i])
            .collect();
        let dr: Vec<f32> = (0..HIDDEN)
            .map(|n| (0..INTER).map(|k| act[k] * d.at(n, k)).sum())
            .collect();
        ref_down[slot * HIDDEN..(slot + 1) * HIDDEN].copy_from_slice(&narrow(dr));
    }
    for t in 0..num_tokens {
        let act: Vec<f32> = (0..INTER)
            .map(|i| silu(ref_sh_gate[t * INTER + i]) * ref_sh_up[t * INTER + i])
            .collect();
        let dr: Vec<f32> = (0..HIDDEN)
            .map(|n| (0..INTER).map(|k| act[k] * sh_down.at(n, k)).sum())
            .collect();
        ref_sh_down[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&narrow(dr));
    }

    // ── Kernel launch helpers ──
    let launch_gate_up = |name: &str, grid_y: u32, block: u32, ntok: Option<u32>| -> Result<()> {
        let h = gpu.kernel("moe_fused_w3", name)?;
        let mut l = KernelLaunch::new(gpu, h)
            .grid([(INTER as u32).div_ceil(8), grid_y, 2])
            .block([block, 1, 1])
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
            .arg_u32(top_k as u32);
        if let Some(n) = ntok {
            l = l.arg_u32(n);
        }
        l.arg_ptr(lut_ptr).launch(stream)
    };
    let launch_silu_down = |name: &str, grid_y: u32, block: u32, ntok: Option<u32>| -> Result<()> {
        let h = gpu.kernel("moe_fused_w3", name)?;
        let mut l = KernelLaunch::new(gpu, h)
            .grid([(HIDDEN as u32).div_ceil(8), grid_y, 1])
            .block([block, 1, 1])
            .shared_mem((INTER * 4) as u32)
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
            .arg_u32(top_k as u32);
        if let Some(n) = ntok {
            l = l.arg_u32(n);
        }
        l.arg_ptr(lut_ptr).launch(stream)
    };

    let check = |label: &str,
                 got_routed: &[f32],
                 got_shared: &[f32],
                 exp_routed: &[f32],
                 exp_shared: &[f32],
                 gate: f64|
     -> Result<bool> {
        let cr = cosine(got_routed, exp_routed);
        let cs = cosine(got_shared, exp_shared);
        let ok = cr >= gate && cs >= gate;
        println!(
            "  {label}: routed cos={cr:.7} maxdiff={:.4e} | shared cos={cs:.7} maxdiff={:.4e}  {}",
            max_abs_diff(got_routed, exp_routed),
            max_abs_diff(got_shared, exp_shared),
            if ok { "PASS" } else { "FAIL" },
        );
        Ok(ok)
    };
    let mut all_ok = true;

    // [1] single-token pair — token 0 only (grid_y = top_k+1, no num_tokens arg).
    {
        launch_gate_up("moe_expert_gate_up_shared_w3", top_k as u32 + 1, 128, None)?;
        launch_silu_down(
            "moe_expert_silu_down_shared_w3",
            top_k as u32 + 1,
            128,
            None,
        )?;
        gpu.synchronize(stream)?;
        let got_down = read_bf16(down_out, top_k * HIDDEN)?;
        let got_sh = read_bf16(sh_down_out, HIDDEN)?;
        // Single-token kernel clamps routed swiglu at ±10 — reproduce.
        let mut exp = vec![0f32; top_k * HIDDEN];
        for slot in 0..top_k {
            let d = &downs[indices[slot] as usize];
            let act: Vec<f32> = (0..INTER)
                .map(|i| {
                    let g = ref_gate[slot * INTER + i].min(10.0);
                    let u = ref_up[slot * INTER + i].clamp(-10.0, 10.0);
                    silu(g) * u
                })
                .collect();
            let dr: Vec<f32> = (0..HIDDEN)
                .map(|n| (0..INTER).map(|k| act[k] * d.at(n, k)).sum())
                .collect();
            exp[slot * HIDDEN..(slot + 1) * HIDDEN].copy_from_slice(&narrow(dr));
        }
        all_ok &= check(
            "single-token pair ",
            &got_down,
            &got_sh,
            &exp,
            &ref_sh_down[..HIDDEN],
            0.99995,
        )?;
    }

    // [2] batchN v1 pair (block 256, production choice at hidden>=3072).
    {
        launch_gate_up(
            "moe_expert_gate_up_shared_batchN_w3",
            rows_y,
            256,
            Some(num_tokens as u32),
        )?;
        launch_silu_down(
            "moe_expert_silu_down_shared_batchN_w3",
            rows_y,
            256,
            Some(num_tokens as u32),
        )?;
        gpu.synchronize(stream)?;
        let got_gate = read_bf16(gate_out, total_routed * INTER)?;
        let got_down = read_bf16(down_out, total_routed * HIDDEN)?;
        let got_sh = read_bf16(sh_down_out, num_tokens * HIDDEN)?;
        all_ok &= check(
            "batchN v1 gate    ",
            &got_gate,
            &got_gate[..1],
            &ref_gate,
            &got_gate[..1],
            0.99995,
        )?;
        all_ok &= check(
            "batchN v1 down    ",
            &got_down,
            &got_sh,
            &ref_down,
            &ref_sh_down,
            0.99995,
        )?;
    }

    // [3] batchN v2 gate_up + v4 dedup down (silu precompute is the shipped
    // weight-free NVFP4-module kernel).
    {
        launch_gate_up(
            "moe_expert_gate_up_shared_batchN_v2_w3",
            rows_y,
            128,
            Some(num_tokens as u32),
        )?;
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
        let dn = gpu.kernel("moe_fused_w3", "moe_expert_down_dedup_batchN_w3")?;
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
            .arg_ptr(lut_ptr)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let got_gate = read_bf16(gate_out, total_routed * INTER)?;
        let got_down = read_bf16(down_out, total_routed * HIDDEN)?;
        let got_sh = read_bf16(sh_down_out, num_tokens * HIDDEN)?;
        // v2/v4 change the k-partition + narrow the act to BF16 — cosine gate
        // 0.9995 like the NVFP4 v2 microtest.
        all_ok &= check(
            "batchN v2 gate    ",
            &got_gate,
            &got_gate[..1],
            &ref_gate,
            &got_gate[..1],
            0.9995,
        )?;
        all_ok &= check(
            "v4 dedup down     ",
            &got_down,
            &got_sh,
            &ref_down,
            &ref_sh_down,
            0.9995,
        )?;
    }

    // [4] grouped GEMM ptrtable (prefill fallback): run the gate projection
    // for the first `num_tokens` tokens routed to expert indices[0].
    {
        let e0 = indices[0] as usize;
        let mut offsets = vec![0i32; NUM_EXPERT_SLOTS + 1];
        // all num_tokens rows assigned to expert e0
        for o in offsets.iter_mut().skip(e0 + 1) {
            *o = num_tokens as i32;
        }
        let off_ptr = upload(gpu, &i32s(&offsets))?;
        let c_out = gpu.alloc(num_tokens * INTER * 2)?;
        let h = gpu.kernel("moe_w3a16", "moe_w3a16_grouped_gemm_ptrtable")?;
        KernelLaunch::new(gpu, h)
            .grid([
                (INTER as u32).div_ceil(64),
                num_tokens.div_ceil(64) as u32,
                NUM_EXPERT_SLOTS as u32,
            ])
            .block([128, 1, 1])
            .arg_ptr(a_ptr)
            .arg_ptr(gate_tbl.0)
            .arg_ptr(gate_tbl.1)
            .arg_ptr(gate_tbl.2)
            .arg_ptr(c_out)
            .arg_ptr(off_ptr)
            .arg_ptr(DevicePtr(0)) // sorted_token_ids NULL → direct rows
            .arg_u32(NUM_EXPERT_SLOTS as u32)
            .arg_u32(INTER as u32)
            .arg_u32(HIDDEN as u32)
            .arg_ptr(lut_ptr)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let got = read_bf16(c_out, num_tokens * INTER)?;
        let g = &gates[e0];
        let mut exp = vec![0f32; num_tokens * INTER];
        for t in 0..num_tokens {
            let a_row = &a_f[t * HIDDEN..(t + 1) * HIDDEN];
            for n in 0..INTER {
                // kernel dequants to BF16 in smem before the BF16 MMA
                exp[t * INTER + n] = (0..HIDDEN)
                    .map(|k| a_row[k] * bf16_to_f32(f32_to_bf16(g.at(n, k))))
                    .sum();
            }
        }
        let exp = narrow(exp);
        let c = cosine(&got, &exp);
        let ok = c >= 0.999;
        println!(
            "  grouped gemm      : cos={c:.7} maxdiff={:.4e}  {}",
            max_abs_diff(&got, &exp),
            if ok { "PASS" } else { "FAIL" },
        );
        all_ok &= ok;
    }

    if !all_ok {
        bail!("W3 microtest FAILED");
    }
    println!("=== W3 microtest: ALL PASS ===");
    Ok(())
}
