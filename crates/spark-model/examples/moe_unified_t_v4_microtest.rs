// SPDX-License-Identifier: AGPL-3.0-only

//! Bit-exactness + bandwidth oracle for the **DeepSeek-V4-Flash unified-`_t`
//! decode MoE** kernels, at production single-token shapes (hidden 4096,
//! moe_intermediate 2048, top-6 routed + 1 shared, routed MXFP4/E8M0 per-32,
//! shared NVFP4/E4M3 per-16).
//!
//! Why this exists: `exp_unified_t` was measured at ~30 ms of a ~65 ms decode
//! token — 4.03 GB of expert weights at ~134 GB/s against a 254 GB/s achievable
//! roofline. Loading the model to test a kernel idea costs ~5 minutes; this
//! costs seconds and reports GB/s per entry point.
//!
//! Hypothesis under test: the shipping kernels give each thread ONE output `n`,
//! so a warp requests 32 contiguous bytes per K iteration — one sector, well
//! under the DRAM burst. The `_v4` entry points give each thread four adjacent
//! `n` (uchar4), so a warp requests 128 contiguous bytes. Per-output
//! accumulation order is unchanged, so `_v4` must be BIT-IDENTICAL — that is
//! the gate here; the timing legs are only meaningful if it holds.
//!
//! Usage:
//!   cargo run --release -p spark-model --example moe_unified_t_v4_microtest \
//!       -- [block] [pool] [seed-hex] [top_k]
//! Defaults: 32 24 0xD54 6

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

const MODULE: &str = "moe_shared_expert_fused_t";
const HIDDEN: usize = 4096; // K for gate/up, N for down
const INTER: usize = 2048; // N for gate/up, K for down
const GS_ROUTED: usize = 32; // MXFP4 E8M0 per-32
const GS_SHARED: usize = 16; // NVFP4 E4M3 per-16
const NUM_EXPERT_SLOTS: usize = 256; // pointer-table width (production)
const NUM_IDX_SETS: usize = 8;
const LAYERS: f64 = 43.0; // DeepSeek-V4-Flash

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(n);
        while v.len() < n {
            v.extend_from_slice(&self.next_u64().to_le_bytes());
        }
        v.truncate(n);
        v
    }
    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
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

/// One transposed weight `[K/2, N]` packed + `[K/GS, N]` scales.
struct Wt {
    packed: Vec<u8>,
    scales: Vec<u8>,
    s2: f32,
}
impl Wt {
    /// E8M0 routed: exponent byte in [113,141] (2^-14..2^14), never 0/255
    /// (those decode to zero and would hide a scale-indexing bug).
    fn routed(rng: &mut Rng, k: usize, n: usize) -> Self {
        let packed = rng.bytes(k / 2 * n);
        let scales = (0..k / GS_ROUTED * n)
            .map(|_| 113 + (rng.next_u64() % 29) as u8)
            .collect();
        Wt {
            packed,
            scales,
            s2: 1.0,
        }
    }
    /// NVFP4 shared: E4M3 scale bytes, 0x7F/0xFF (NaN) remapped to 1.0.
    fn shared(rng: &mut Rng, k: usize, n: usize) -> Self {
        let packed = rng.bytes(k / 2 * n);
        let scales = (0..k / GS_SHARED * n)
            .map(|_| {
                let b = (rng.next_u64() & 0xFF) as u8;
                if b == 0x7F || b == 0xFF { 0x38 } else { b }
            })
            .collect();
        Wt {
            packed,
            scales,
            s2: 0.037,
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

/// Device-side pointer tables for one projection across the expert pool.
struct Table {
    packed: DevicePtr,
    scales: DevicePtr,
    s2: DevicePtr,
}

fn build_table(gpu: &dyn GpuBackend, ws: &[Wt]) -> Result<Table> {
    let mut wp = vec![0u64; NUM_EXPERT_SLOTS];
    let mut sp = vec![0u64; NUM_EXPERT_SLOTS];
    let mut s2 = vec![0f32; NUM_EXPERT_SLOTS];
    for (e, w) in ws.iter().enumerate() {
        wp[e] = upload(gpu, &w.packed)?.0;
        sp[e] = upload(gpu, &w.scales)?.0;
        s2[e] = w.s2;
    }
    Ok(Table {
        packed: upload(gpu, &u64s(&wp))?,
        scales: upload(gpu, &u64s(&sp))?,
        s2: upload(gpu, &f32s(&s2))?,
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let block: u32 = args.get(1).map_or(32, |s| s.parse().unwrap());
    let pool: usize = args.get(2).map_or(24, |s| s.parse().unwrap());
    let seed: u64 = args.get(3).map_or(0xD54, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0xD54)
    });
    // top_k is production-6; raising it is the knob that adds warps at fixed
    // access width, which is how we tell a latency-bound kernel from one
    // capped by per-request sector efficiency.
    let top_k: usize = args.get(4).map_or(6, |s| s.parse().unwrap());
    assert!(pool <= NUM_EXPERT_SLOTS && top_k <= pool);
    // VEC=4 needs every block fully covered — the kernel drops a partial group.
    assert!(
        INTER % (block as usize * 4) == 0 && HIDDEN % (block as usize * 4) == 0,
        "block {block} × VEC 4 must divide both N={INTER} (gate_up) and N={HIDDEN} (down)"
    );

    println!(
        "=== V4 unified_t decode MoE: block={block} pool={pool} top_k={top_k} \
         gate_up(N={INTER},K={HIDDEN}) down(N={HIDDEN},K={INTER}) seed=0x{seed:X} ==="
    );

    // The kernels decode E2M1 nibbles arithmetically instead of via a shared
    // LUT. Mirror the CUDA expression here and check all 16 codes against the
    // table it replaced — signed zero included, so compare raw bits.
    const E2M1: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    for nib in 0u32..16 {
        let (m, e, s) = (nib & 1, (nib >> 1) & 3, (nib >> 3) & 1);
        let mag = if e == 0 {
            if m == 1 { 0x3F00_0000 } else { 0 }
        } else {
            ((126 + e) << 23) | (m << 22)
        };
        let got = f32::from_bits(mag | (s << 31));
        assert_eq!(
            got.to_bits(),
            E2M1[nib as usize].to_bits(),
            "e2m1_decode({nib:#x}) = {got}, table says {}",
            E2M1[nib as usize]
        );
    }
    println!("  e2m1_decode: all 16 codes bit-match the E2M1 table");

    let mut rng = Rng(seed);
    let a_bf16: Vec<u16> = (0..HIDDEN)
        .map(|_| f32_to_bf16(rng.unit() * 2.0 - 1.0))
        .collect();
    println!("generating {pool} experts × 3 projections …");
    let gates: Vec<Wt> = (0..pool)
        .map(|_| Wt::routed(&mut rng, HIDDEN, INTER))
        .collect();
    let ups: Vec<Wt> = (0..pool)
        .map(|_| Wt::routed(&mut rng, HIDDEN, INTER))
        .collect();
    let downs: Vec<Wt> = (0..pool)
        .map(|_| Wt::routed(&mut rng, INTER, HIDDEN))
        .collect();
    let sh_gate = Wt::shared(&mut rng, HIDDEN, INTER);
    let sh_up = Wt::shared(&mut rng, HIDDEN, INTER);
    let sh_down = Wt::shared(&mut rng, INTER, HIDDEN);

    let mut idx_sets_host: Vec<Vec<u32>> = Vec::with_capacity(NUM_IDX_SETS);
    for _ in 0..NUM_IDX_SETS {
        let mut ids: Vec<u32> = Vec::with_capacity(top_k);
        while ids.len() < top_k {
            let e = (rng.next_u64() % pool as u64) as u32;
            if !ids.contains(&e) {
                ids.push(e);
            }
        }
        idx_sets_host.push(ids);
    }
    println!("routing (set 0): {:?}", idx_sets_host[0]);

    // ── GPU setup ──
    let backend =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let a_ptr = upload(gpu, &u16s(&a_bf16))?;
    let gate_tbl = build_table(gpu, &gates)?;
    let up_tbl = build_table(gpu, &ups)?;
    let down_tbl = build_table(gpu, &downs)?;
    let sh = |w: &Wt| -> Result<(DevicePtr, DevicePtr)> {
        Ok((upload(gpu, &w.packed)?, upload(gpu, &w.scales)?))
    };
    let (sh_gate_p, sh_gate_s) = sh(&sh_gate)?;
    let (sh_up_p, sh_up_s) = sh(&sh_up)?;
    let (sh_down_p, sh_down_s) = sh(&sh_down)?;
    let idx_ptrs: Vec<DevicePtr> = idx_sets_host
        .iter()
        .map(|s| upload(gpu, &u32s(s)))
        .collect::<Result<_>>()?;

    // gate/up outputs carry top_k routed slots; the shared slot has its own
    // scratch (production layout).
    let gate_out = gpu.alloc(top_k * INTER * 2)?;
    let up_out = gpu.alloc(top_k * INTER * 2)?;
    let down_out = gpu.alloc(top_k * HIDDEN * 2)?;
    let sh_gate_out = gpu.alloc(INTER * 2)?;
    let sh_up_out = gpu.alloc(INTER * 2)?;
    let sh_down_out = gpu.alloc(HIDDEN * 2)?;

    // Split-K scratch, sized for the largest split under test. gate_up needs
    // [2, split, slots, INTER] f32, down needs [split, slots, HIDDEN] f32 —
    // ~0.5 MB together at split=4, against 94 MB/layer of weight traffic.
    let max_split = 4usize;
    let slots = top_k + 1;
    let partial_gu = gpu.alloc(2 * max_split * slots * INTER * 4)?;
    let partial_dn = gpu.alloc(max_split * slots * HIDDEN * 4)?;

    let launch_gate_up =
        |name: &str, vec: u32, split: u32, idx: DevicePtr, st: u64| -> Result<()> {
            let h: KernelHandle = gpu.kernel(MODULE, name)?;
            let mut l = KernelLaunch::new(gpu, h)
                .grid([(INTER as u32) / (block * vec), top_k as u32 + 1, 2 * split])
                .block([block, 1, 1])
                .arg_ptr(a_ptr)
                .arg_ptr(gate_tbl.packed)
                .arg_ptr(gate_tbl.scales)
                .arg_ptr(gate_tbl.s2)
                .arg_ptr(gate_out)
                .arg_ptr(up_tbl.packed)
                .arg_ptr(up_tbl.scales)
                .arg_ptr(up_tbl.s2)
                .arg_ptr(up_out)
                .arg_ptr(idx)
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
            if split > 1 {
                l = l.arg_ptr(partial_gu);
            }
            l.launch(st)?;
            if split == 1 {
                return Ok(());
            }
            KernelLaunch::new(gpu, gpu.kernel(MODULE, "moe_gate_up_partial_finalize")?)
                .grid([(INTER as u32).div_ceil(block), top_k as u32 + 1, 2])
                .block([block, 1, 1])
                .arg_ptr(partial_gu)
                .arg_ptr(gate_out)
                .arg_ptr(sh_gate_out)
                .arg_ptr(up_out)
                .arg_ptr(sh_up_out)
                .arg_u32(INTER as u32)
                .arg_u32(top_k as u32)
                .arg_u32(split)
                .launch(st)
        };
    let launch_down = |name: &str, vec: u32, split: u32, idx: DevicePtr, st: u64| -> Result<()> {
        let h: KernelHandle = gpu.kernel(MODULE, name)?;
        let mut l = KernelLaunch::new(gpu, h)
            .grid([(HIDDEN as u32) / (block * vec), top_k as u32 + 1, split])
            .block([block, 1, 1])
            // s_act covers only this block's k slice: K*4/split bytes.
            .shared_mem(INTER as u32 * 4 / split)
            .arg_ptr(gate_out)
            .arg_ptr(up_out)
            .arg_ptr(down_tbl.packed)
            .arg_ptr(down_tbl.scales)
            .arg_ptr(down_tbl.s2)
            .arg_ptr(down_out)
            .arg_ptr(idx)
            .arg_ptr(sh_gate_out)
            .arg_ptr(sh_up_out)
            .arg_ptr(sh_down_p)
            .arg_ptr(sh_down_s)
            .arg_f32(sh_down.s2)
            .arg_ptr(sh_down_out)
            .arg_u32(HIDDEN as u32)
            .arg_u32(INTER as u32)
            .arg_u32(top_k as u32);
        if split > 1 {
            l = l.arg_ptr(partial_dn);
        }
        l.launch(st)?;
        if split == 1 {
            return Ok(());
        }
        KernelLaunch::new(gpu, gpu.kernel(MODULE, "moe_down_partial_finalize")?)
            .grid([(HIDDEN as u32).div_ceil(block), top_k as u32 + 1, 1])
            .block([block, 1, 1])
            .arg_ptr(partial_dn)
            .arg_ptr(down_out)
            .arg_ptr(sh_down_out)
            .arg_u32(HIDDEN as u32)
            .arg_u32(top_k as u32)
            .arg_u32(split)
            .launch(st)
    };

    let read_u16 = |ptr: DevicePtr, count: usize| -> Result<Vec<u16>> {
        let mut buf = vec![0u8; count * 2];
        gpu.copy_d2h(ptr, &mut buf)?;
        Ok(buf
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect())
    };
    // Poison every output so an unwritten element can't fake a match.
    let poison = |ptr: DevicePtr, count: usize| -> Result<()> {
        let junk: Vec<u16> = (0..count)
            .map(|i| (0x7F80u16 ^ (i as u16)).rotate_left(3))
            .collect();
        gpu.copy_h2d(&u16s(&junk), ptr)
    };

    // ── correctness: scalar chain vs each wide/split chain ──
    // (vec, split). split>1 sums SPLIT partials instead of one straight sweep,
    // so it is deterministic but not bit-equal to the scalar order; those legs
    // are held to a bf16-ULP bound instead.
    const CFGS: [(u32, u32); 7] = [(1, 1), (2, 1), (4, 1), (2, 2), (2, 4), (4, 2), (4, 4)];
    let suffix_of = |v: u32, s: u32| match (v, s) {
        (1, 1) => String::new(),
        (v, 1) => format!("_v{v}"),
        (v, s) => format!("_v{v}s{s}"),
    };
    let label_of = |v: u32, s: u32| format!("v{v}s{s}");
    let mut captured: Vec<Vec<Vec<u16>>> = Vec::new();
    for (vec, split) in CFGS {
        let suffix = suffix_of(vec, split);
        for (p, c) in [
            (gate_out, top_k * INTER),
            (up_out, top_k * INTER),
            (down_out, top_k * HIDDEN),
            (sh_gate_out, INTER),
            (sh_up_out, INTER),
            (sh_down_out, HIDDEN),
        ] {
            poison(p, c)?;
        }
        gpu.synchronize(stream)?;
        launch_gate_up(
            &format!("moe_expert_gate_up_shared_t_e8m0{suffix}"),
            vec,
            split,
            idx_ptrs[0],
            stream,
        )?;
        launch_down(
            &format!("moe_expert_silu_down_shared_t_e8m0{suffix}"),
            vec,
            split,
            idx_ptrs[0],
            stream,
        )?;
        gpu.synchronize(stream)?;
        captured.push(vec![
            read_u16(gate_out, top_k * INTER)?,
            read_u16(up_out, top_k * INTER)?,
            read_u16(down_out, top_k * HIDDEN)?,
            read_u16(sh_gate_out, INTER)?,
            read_u16(sh_up_out, INTER)?,
            read_u16(sh_down_out, HIDDEN)?,
        ]);
    }
    let names = [
        "gate_out",
        "up_out",
        "down_out",
        "sh_gate_out",
        "sh_up_out",
        "sh_down_out",
    ];
    // Widening the load leaves the per-output summation order untouched, so
    // VEC>1 at split=1 must be BIT-EXACT — that is a hard bug gate.
    //
    // Split-K reassociates the sum, so it cannot be. Judging it by raw ULP is
    // the wrong test: a K=4096 dot product whose result lands near zero by
    // cancellation is many ULP away from itself under any reordering, while
    // being numerically irrelevant. Gate on max|Δ| relative to the tensor's own
    // RMS instead. bf16 carries 8 mantissa bits, so one rounding is 2^-8;
    // allowing 2^-6 of RMS leaves room for accumulated reassociation while
    // still catching the failures that matter — a scale- or index-off-by-one
    // moves an output by a factor, not by a fraction of RMS.
    const SPLIT_REL_MAX: f32 = 1.0 / 64.0;
    let bf16_to_f32 = |x: u16| f32::from_bits((x as u32) << 16);
    let mut fail = false;
    for (ci, (vec, split)) in CFGS.iter().enumerate().skip(1) {
        let tag = label_of(*vec, *split);
        for (i, name) in names.iter().enumerate() {
            let (a, b) = (&captured[0][i], &captured[ci][i]);
            let bad = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
            if *split == 1 {
                if bad == 0 {
                    println!("  {tag} {name}: BIT-EXACT ({} u16s)", a.len());
                } else {
                    let first = a.iter().zip(b.iter()).position(|(x, y)| x != y).unwrap();
                    println!(
                        "  {tag} {name}: {bad}/{} MISMATCHED (first at {first}: {:#06x} vs {:#06x})",
                        a.len(),
                        a[first],
                        b[first]
                    );
                    fail = true;
                }
                continue;
            }
            let rms = (a.iter().map(|x| bf16_to_f32(*x).powi(2)).sum::<f32>() / a.len() as f32)
                .sqrt()
                .max(f32::MIN_POSITIVE);
            let (worst, at) = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (bf16_to_f32(*x) - bf16_to_f32(*y)).abs())
                .enumerate()
                .fold(
                    (0.0f32, 0usize),
                    |(m, mi), (i, d)| {
                        if d > m { (d, i) } else { (m, mi) }
                    },
                );
            let rel = worst / rms;
            let ok = rel <= SPLIT_REL_MAX;
            println!(
                "  {tag} {name}: {bad}/{} differ, max |Δ|/rms = {rel:.2e} {} \
                 (at {at}: {:.5} vs {:.5}, rms {rms:.4})",
                a.len(),
                if ok { "OK" } else { "EXCEEDS 2^-6" },
                bf16_to_f32(a[at]),
                bf16_to_f32(b[at]),
            );
            if !ok {
                fail = true;
            }
        }
    }

    // ── perf ──
    let time_it = |f: &dyn Fn(u64) -> Result<()>| -> Result<f64> {
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

    // Weight bytes actually streamed per layer: top_k routed (E8M0 per-32) +
    // 1 shared (NVFP4 per-16), for gate, up and down.
    let routed_gu = top_k * 2 * (INTER * HIDDEN / 2 + INTER * HIDDEN / GS_ROUTED);
    let shared_gu = 2 * (INTER * HIDDEN / 2 + INTER * HIDDEN / GS_SHARED);
    let routed_dn = top_k * (HIDDEN * INTER / 2 + HIDDEN * INTER / GS_ROUTED);
    let shared_dn = HIDDEN * INTER / 2 + HIDDEN * INTER / GS_SHARED;
    let gu_bytes = routed_gu + shared_gu;
    let dn_bytes = routed_dn + shared_dn;
    println!(
        "weight read: gate_up {:.1} MB + down {:.1} MB = {:.1} MB/layer, \
         {:.2} GB over {LAYERS:.0} layers",
        gu_bytes as f64 / 1e6,
        dn_bytes as f64 / 1e6,
        (gu_bytes + dn_bytes) as f64 / 1e6,
        (gu_bytes + dn_bytes) as f64 * LAYERS / 1e9
    );

    // Rotate the routing every launch so consecutive iterations touch different
    // expert weights — a single hot set would sit in L2 and overstate GB/s.
    let cursor = std::cell::Cell::new(0usize);
    let next_idx = || {
        let c = cursor.get();
        cursor.set(c + 1);
        idx_ptrs[c % idx_ptrs.len()]
    };
    type L<'a> = Box<dyn Fn(u64) -> Result<()> + 'a>;
    let mut legs: Vec<(String, usize, L)> = Vec::new();
    for (v, sp) in CFGS {
        let sfx = suffix_of(v, sp);
        let (g, d) = (sfx.clone(), sfx);
        let tag = label_of(v, sp);
        legs.push((
            format!("gate_up {tag}"),
            gu_bytes,
            Box::new(move |s| {
                let k = format!("moe_expert_gate_up_shared_t_e8m0{g}");
                launch_gate_up(&k, v, sp, next_idx(), s)
            }),
        ));
        legs.push((
            format!("down    {tag}"),
            dn_bytes,
            Box::new(move |s| {
                let k = format!("moe_expert_silu_down_shared_t_e8m0{d}");
                launch_down(&k, v, sp, next_idx(), s)
            }),
        ));
    }
    let mut ms = std::collections::HashMap::new();
    for (label, bytes, f) in &legs {
        let t = time_it(f.as_ref())?;
        ms.insert(label.clone(), t);
        println!(
            "{label}: {t:.4} ms/layer  ({:.0} GB/s eff)",
            *bytes as f64 / (t / 1e3) / 1e9
        );
    }
    let total = (gu_bytes + dn_bytes) as f64;
    let chain = |v: u32, sp: u32| {
        let tag = label_of(v, sp);
        ms[&format!("gate_up {tag}")] + ms[&format!("down    {tag}")]
    };
    let base = chain(1, 1);
    // Best chain is not necessarily one config for both halves: gate_up and
    // down have different N, K and shared-memory footprints, and the dispatch
    // picks them independently.
    for (v, sp) in CFGS {
        let t = chain(v, sp);
        println!(
            "chain {}: {t:.4} ms/layer ({:.0} GB/s) [{:+.1}% vs scalar]   \
             ({LAYERS:.0} layers: {:.1} ms/token MoE)",
            label_of(v, sp),
            total / (t / 1e3) / 1e9,
            (base / t - 1.0) * 100.0,
            t * LAYERS
        );
    }
    let pick = |half: &str| -> (String, f64) {
        CFGS.iter()
            .map(|(v, sp)| {
                let tag = label_of(*v, *sp);
                (tag.clone(), ms[&format!("{half}{tag}")])
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
    };
    let (gu_tag, gu_t) = pick("gate_up ");
    let (dn_tag, dn_t) = pick("down    ");
    let best = (gu_t + dn_t, format!("gate_up {gu_tag} + down {dn_tag}"));
    println!(
        "BEST MIX: {} = {:.4} ms/layer ({:.0} GB/s) [{:+.1}% vs scalar]   \
         ({LAYERS:.0} layers: {:.1} -> {:.1} ms/token MoE)",
        best.1,
        best.0,
        total / (best.0 / 1e3) / 1e9,
        (base / best.0 - 1.0) * 100.0,
        base * LAYERS,
        best.0 * LAYERS
    );

    if fail {
        eprintln!("RESULT: FAIL (a wide variant is not bit-exact vs scalar)");
        std::process::exit(1);
    }
    println!("RESULT: PASS");
    Ok(())
}
