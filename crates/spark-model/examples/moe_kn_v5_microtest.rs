// SPDX-License-Identifier: AGPL-3.0-only

//! Bit-exactness + perf oracle for the batchN **v5** cp.async-staged MoE
//! expert kernels (`moe_expert_gate_up_shared_batchN_v5` /
//! `moe_expert_down_dedup_batchN_v5`) against the shipping v2 gate_up + v4
//! dedup-down chain, at exact Laguna-S-2.1 verify shapes (hidden K=3072,
//! expert inter N=1024, top_k=10, NVFP4 non-transposed + E4M3 group-16
//! scales + per-tensor scale2, 256-wide pointer tables).
//!
//! v5 is required to be BYTE-IDENTICAL to v2/v4 (same lane partition and
//! FMA order; only the load path changes: cp.async bulk staging + parallel
//! leader election), so the gate here is exact u16 equality of every output
//! buffer, not cosine. Runs both the populated-shared and the NULL-shared
//! (production Laguna) variants.
//!
//! Usage:
//!   cargo run --release -p spark-model --example moe_kn_v5_microtest \
//!       -- [num_tokens] [top_k] [pool] [seed-hex]
//! Defaults: 7 10 64 0x9E3   (pool = distinct experts tokens draw from;
//! smaller pool = smaller union, production verify unions are ~30-60).

use anyhow::{Result, bail};
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

fn f32_to_bf16(f: f32) -> u16 {
    let bits = f.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

/// One NVFP4 weight matrix [n, k]: packed nibbles [n, k/2], e4m3 group-16
/// scales [n, k/16], per-tensor scale2.
struct W {
    packed: Vec<u8>,
    scales: Vec<u8>,
    s2: f32,
}
impl W {
    fn random(rng: &mut Rng, n: usize, k: usize) -> Self {
        let packed: Vec<u8> = (0..n * k / 2)
            .map(|_| (rng.next_u64() & 0xFF) as u8)
            .collect();
        // e4m3 exponent 3..=9 keeps decode finite and moderate (no NaN byte).
        let scales: Vec<u8> = (0..n * k / GROUP)
            .map(|_| {
                let s = (rng.next_u64() & 1) as u8;
                let e = 3 + (rng.next_u64() % 7) as u8;
                let m = (rng.next_u64() % 8) as u8;
                (s << 7) | (e << 3) | m
            })
            .collect();
        W {
            packed,
            scales,
            s2: rng.uniform(0.02, 0.06),
        }
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

#[allow(clippy::too_many_arguments)]
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let num_tokens: usize = args.get(1).map_or(7, |s| s.parse().unwrap());
    let top_k: usize = args.get(2).map_or(10, |s| s.parse().unwrap());
    let pool: usize = args.get(3).map_or(64, |s| s.parse().unwrap());
    let seed: u64 = args.get(4).map_or(0x9E3, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x9E3)
    });
    assert!(num_tokens <= 8, "v2/v5 kernels cap M at 8");
    let total_routed = num_tokens * top_k;
    let rows_y = (total_routed + num_tokens) as u32;

    println!(
        "=== batchN v5 microtest: tokens={num_tokens} top_k={top_k} pool={pool} \
         K={HIDDEN} N={INTER} seed=0x{seed:X} ==="
    );

    let mut rng = Rng(seed);
    let a_bf16: Vec<u16> = (0..num_tokens * HIDDEN)
        .map(|_| f32_to_bf16(rng.uniform(-1.0, 1.0)))
        .collect();
    let gates: Vec<W> = (0..pool)
        .map(|_| W::random(&mut rng, INTER, HIDDEN))
        .collect();
    let ups: Vec<W> = (0..pool)
        .map(|_| W::random(&mut rng, INTER, HIDDEN))
        .collect();
    let downs: Vec<W> = (0..pool)
        .map(|_| W::random(&mut rng, HIDDEN, INTER))
        .collect();
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
        "routing: {} slots -> {} unique experts (dedup factor {:.2}x)",
        total_routed,
        unique.len(),
        total_routed as f64 / unique.len() as f64,
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
    let act_routed = gpu.alloc(total_routed * INTER * 2)?;
    let sh_act = gpu.alloc(num_tokens * INTER * 2)?;

    // null_shared=true reproduces production Laguna (dual-format shared
    // expert handled outside the kernel → NULL pointers, zero-filled rows).
    let launch_gate_up = |name: &str, null_shared: bool, stream: u64| -> Result<()> {
        let h = gpu.kernel("moe_fused_batch2", name)?;
        let (sgp, sgs, sup, sus) = if null_shared {
            (DevicePtr(0), DevicePtr(0), DevicePtr(0), DevicePtr(0))
        } else {
            (sh_gate_p, sh_gate_s, sh_up_p, sh_up_s)
        };
        KernelLaunch::new(gpu, h)
            .grid([(INTER as u32).div_ceil(8), rows_y, 2])
            .block([128, 1, 1])
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
            .arg_ptr(sgp)
            .arg_ptr(sgs)
            .arg_f32(sh_gate.s2)
            .arg_ptr(sh_gate_out)
            .arg_ptr(sup)
            .arg_ptr(sus)
            .arg_f32(sh_up.s2)
            .arg_ptr(sh_up_out)
            .arg_u32(INTER as u32)
            .arg_u32(HIDDEN as u32)
            .arg_u32(top_k as u32)
            .arg_u32(num_tokens as u32)
            .launch(stream)
    };
    let launch_precompute = |stream: u64| -> Result<()> {
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
            .launch(stream)
    };
    // v4 covers 8 rows/block reading act_routed/sh_act (from precompute);
    // v5 covers 32 rows (4 pipelined cp.async tiles) and reads act straight
    // from the gate buffers (the v5 gate_up's fused silu epilogue).
    let launch_down = |name: &str, null_shared: bool, stream: u64| -> Result<()> {
        let h = gpu.kernel("moe_fused_batch2", name)?;
        let rows_per_block = if name.ends_with("_v5") { 16 } else { 8 };
        let (act_p, sh_act_p) = (act_routed, sh_act);
        let (sdp, sds) = if null_shared {
            (DevicePtr(0), DevicePtr(0))
        } else {
            (sh_down_p, sh_down_s)
        };
        KernelLaunch::new(gpu, h)
            .grid([(HIDDEN as u32).div_ceil(rows_per_block), rows_y, 1])
            .block([128, 1, 1])
            .arg_ptr(act_p)
            .arg_ptr(sh_act_p)
            .arg_ptr(down_tbl.0)
            .arg_ptr(down_tbl.1)
            .arg_ptr(down_tbl.2)
            .arg_ptr(down_out)
            .arg_ptr(idx_ptr)
            .arg_ptr(sdp)
            .arg_ptr(sds)
            .arg_f32(sh_down.s2)
            .arg_ptr(sh_down_out)
            .arg_u32(HIDDEN as u32)
            .arg_u32(INTER as u32)
            .arg_u32(top_k as u32)
            .arg_u32(num_tokens as u32)
            .launch(stream)
    };

    let read_u16 = |ptr: DevicePtr, count: usize| -> Result<Vec<u16>> {
        let mut buf = vec![0u8; count * 2];
        gpu.copy_d2h(ptr, &mut buf)?;
        Ok(buf
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect())
    };
    let fill_garbage = |ptr: DevicePtr, count: usize| -> Result<()> {
        let junk: Vec<u16> = (0..count)
            .map(|i| (0x7F80 ^ (i as u16)).rotate_left(3))
            .collect();
        gpu.copy_h2d(&u16s(&junk), ptr)
    };

    // ── bit-exactness: v2/v4 chain vs v5 chain, byte-for-byte ──
    let mut fail = false;
    for null_shared in [false, true] {
        let label = if null_shared {
            "null-shared (Laguna)"
        } else {
            "populated-shared"
        };
        let mut captured: Vec<Vec<Vec<u16>>> = Vec::new();
        for chain in ["v2/v4", "v5"] {
            // poison every output buffer so unwritten rows can't fake a match
            fill_garbage(gate_out, total_routed * INTER)?;
            fill_garbage(up_out, total_routed * INTER)?;
            fill_garbage(down_out, total_routed * HIDDEN)?;
            fill_garbage(sh_gate_out, num_tokens * INTER)?;
            fill_garbage(sh_up_out, num_tokens * INTER)?;
            fill_garbage(sh_down_out, num_tokens * HIDDEN)?;
            fill_garbage(act_routed, total_routed * INTER)?;
            fill_garbage(sh_act, num_tokens * INTER)?;
            gpu.synchronize(stream)?;
            if chain == "v5" {
                launch_gate_up("moe_expert_gate_up_shared_batchN_v5", null_shared, stream)?;
                launch_precompute(stream)?;
                launch_down("moe_expert_down_dedup_batchN_v5", null_shared, stream)?;
            } else {
                launch_gate_up("moe_expert_gate_up_shared_batchN_v2", null_shared, stream)?;
                launch_precompute(stream)?;
                launch_down("moe_expert_down_dedup_batchN", null_shared, stream)?;
            }
            gpu.synchronize(stream)?;
            captured.push(vec![
                read_u16(gate_out, total_routed * INTER)?,
                read_u16(up_out, total_routed * INTER)?,
                read_u16(act_routed, total_routed * INTER)?,
                read_u16(down_out, total_routed * HIDDEN)?,
                read_u16(sh_gate_out, num_tokens * INTER)?,
                read_u16(sh_up_out, num_tokens * INTER)?,
                read_u16(sh_act, num_tokens * INTER)?,
                read_u16(sh_down_out, num_tokens * HIDDEN)?,
            ]);
        }
        let names = [
            "gate_out",
            "up_out",
            "act",
            "down_out",
            "sh_gate_out",
            "sh_up_out",
            "sh_act",
            "sh_down_out",
        ];
        for (i, name) in names.iter().enumerate() {
            let a = &captured[0][i];
            let b = &captured[1][i];
            let mismatches = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
            if mismatches != 0 {
                let first = a.iter().zip(b.iter()).position(|(x, y)| x != y).unwrap();
                println!(
                    "  [{label}] {name}: {mismatches}/{} MISMATCHED u16s (first at {first}: \
                     {:#06x} vs {:#06x})",
                    a.len(),
                    a[first],
                    b[first]
                );
                fail = true;
            } else {
                println!("  [{label}] {name}: BIT-EXACT ({} u16s)", a.len());
            }
        }
    }

    // ── perf: event-timed, per phase, v2/v4 vs v5 ──
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

    // dedup-minimum weight bytes per layer (packed+scales; +1 = shared —
    // counted only in the populated-shared leg the timings use... the perf
    // legs below run null_shared=false so the shared expert IS read).
    let gu_bytes = (unique.len() + 1) * 2 * (INTER * HIDDEN / 2 + INTER * HIDDEN / GROUP);
    let dn_bytes = (unique.len() + 1) * (HIDDEN * INTER / 2 + HIDDEN * INTER / GROUP);
    println!(
        "dedup-minimum weight read: gate_up {:.1} MB + down {:.1} MB = {:.1} MB/layer",
        gu_bytes as f64 / 1e6,
        dn_bytes as f64 / 1e6,
        (gu_bytes + dn_bytes) as f64 / 1e6
    );

    type L<'a> = Box<dyn Fn(u64) -> Result<()> + 'a>;
    let legs: Vec<(&str, usize, L)> = vec![
        (
            "gate_up v2",
            gu_bytes,
            Box::new(|s| launch_gate_up("moe_expert_gate_up_shared_batchN_v2", false, s)),
        ),
        (
            "gate_up v5",
            gu_bytes,
            Box::new(|s| launch_gate_up("moe_expert_gate_up_shared_batchN_v5", false, s)),
        ),
        (
            "down v4 (pre+dedup)",
            dn_bytes,
            Box::new(|s| {
                launch_precompute(s)?;
                launch_down("moe_expert_down_dedup_batchN", false, s)
            }),
        ),
        (
            "down v5 (pre+dedup)",
            dn_bytes,
            Box::new(|s| {
                launch_precompute(s)?;
                launch_down("moe_expert_down_dedup_batchN_v5", false, s)
            }),
        ),
    ];
    let mut ms = std::collections::HashMap::new();
    for (label, bytes, f) in &legs {
        let t = time_it(f.as_ref(), stream)?;
        ms.insert(*label, t);
        println!(
            "{label}: {t:.4} ms/layer  ({:.0} GB/s eff)",
            *bytes as f64 / (t / 1e3) / 1e9
        );
    }
    let old = ms["gate_up v2"] + ms["down v4 (pre+dedup)"];
    let new = ms["gate_up v5"] + ms["down v5 (pre+dedup)"];
    let total = (gu_bytes + dn_bytes) as f64;
    println!(
        "chain v2/v4: {old:.4} ms/layer ({:.0} GB/s) -> chain v5: {new:.4} ms/layer \
         ({:.0} GB/s)  [{:+.1}%]",
        total / (old / 1e3) / 1e9,
        total / (new / 1e3) / 1e9,
        (old / new - 1.0) * 100.0
    );

    if fail {
        eprintln!("RESULT: FAIL (v5 not bit-exact vs v2/v4)");
        std::process::exit(1);
    }
    println!("RESULT: PASS");
    Ok(())
}
