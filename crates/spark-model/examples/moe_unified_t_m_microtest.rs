// SPDX-License-Identifier: AGPL-3.0-only

//! Equivalence + speedup oracle for the **multi-row (`_m`) dedup'd split-K
//! unified-`_t` decode MoE**, at DeepSeek-V4-Flash shapes (hidden 4096,
//! moe_intermediate 2048, top-6 routed + 1 shared, routed MXFP4/E8M0 per-32,
//! shared NVFP4/E4M3 per-16).
//!
//! Why this exists: the MTP K=2 verify runs the MoE for two candidate rows.
//! The two rows' routed expert sets overlap (measured 1.28x on learned-gate
//! layers, 2.01x on hash-routed ones) and the shared expert is duplicated
//! outright, so a kernel that reads each weight byte once and FMAs it into
//! both rows should cut the verify's weight traffic by ~0.72x. The previous
//! attempt at this (`batch2_t`) *lost* — it batched on top of the pre-split-K
//! kernel shape and gave back more to narrow loads than it won on reuse. This
//! measures the rewrite that batches on top of the fast `v2s4` body instead.
//!
//! Two gates, in order:
//!   1. `_m1v2s4` at num_tokens=1 must be **BIT-IDENTICAL** to the shipping
//!      `_v2s4`. The dedup/MROW rewrite must not perturb the k order at all.
//!      If this fails, nothing below it means anything.
//!   2. `_m{MROW}v2s4` at num_tokens=`tokens` must match `tokens` independent
//!      `_v2s4` passes to a bf16-ULP bound. It is NOT bit-equal: a leader block
//!      sums its k window for every row it gathered, which is the same order
//!      per row, but the *shared* row is served by one block-set instead of
//!      `tokens`, and the partial rows differ. In practice this comes out
//!      bit-equal too; the bound is the contract.
//!
//! Then the timing leg: `tokens` `_v2s4` chains (what the verify costs today —
//! the per-token fallback, or `ATLAS_K2_MOE_PER_TOKEN=1` at K=2) vs one
//! `_m{MROW}v2s4` chain.
//!
//! Usage:
//!   cargo run --release -p spark-model --example moe_unified_t_m_microtest \
//!       -- [block] [pool] [seed-hex] [top_k] [overlap] [tokens]
//! Defaults: 32 24 0xD54 6 -1 2
//!
//! `tokens` is the verify width: 2 for the MTP K=2 verify, 6 for the DSpark
//! block (5 proposed rows + the committed one). It selects the narrowest
//! compiled `_m` entry point that covers it, exactly as the Rust dispatch does.
//!
//! `overlap` forces how many of rows 1.. keep of row 0's top_k picks (0..top_k),
//! or -1 for random routing. Sweeping it is how you read the speedup curve:
//! overlap=0 is the worst case (disjoint routing, only the shared expert is
//! amortized), overlap=top_k the best (hash-routed layers, where every row
//! picks the identical top-k).

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
const SPLIT: u32 = 4; // must match `forward_phase::T_SPLIT`
const VEC: u32 = 2; // must match `ops::T_SPLIT_VEC`
const LAYERS: f64 = 43.0; // DeepSeek-V4-Flash
/// Widest compiled `_m` entry point — must track
/// `spark_runtime::buffers::MOE_DECODE_MAX_ROWS` and the `_m6v2s4` entries.
const MROW_MAX: u32 = 6;
const ITERS: usize = 200;

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

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
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
        Wt {
            packed: rng.bytes(k / 2 * n),
            scales: (0..k / GS_ROUTED * n)
                .map(|_| 113 + (rng.next_u64() % 29) as u8)
                .collect(),
            s2: 1.0,
        }
    }
    /// NVFP4 shared: E4M3 scale bytes, 0x7F/0xFF (NaN) remapped to 1.0.
    fn shared(rng: &mut Rng, k: usize, n: usize) -> Self {
        Wt {
            packed: rng.bytes(k / 2 * n),
            scales: (0..k / GS_SHARED * n)
                .map(|_| {
                    let b = (rng.next_u64() & 0xFF) as u8;
                    if b == 0x7F || b == 0xFF { 0x38 } else { b }
                })
                .collect(),
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

/// Buffers one chain writes into. The `_m` chain and the per-row reference get
/// separate sets so a comparison can never read a value the other wrote.
struct Outs {
    gate: DevicePtr,
    up: DevicePtr,
    down: DevicePtr,
    sh_gate: DevicePtr,
    sh_up: DevicePtr,
    sh_down: DevicePtr,
}

impl Outs {
    fn alloc(gpu: &dyn GpuBackend, rows: usize, tokens: usize) -> Result<Self> {
        Ok(Outs {
            gate: gpu.alloc(rows * INTER * 2)?,
            up: gpu.alloc(rows * INTER * 2)?,
            down: gpu.alloc(rows * HIDDEN * 2)?,
            sh_gate: gpu.alloc(tokens * INTER * 2)?,
            sh_up: gpu.alloc(tokens * INTER * 2)?,
            sh_down: gpu.alloc(tokens * HIDDEN * 2)?,
        })
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let block: u32 = args.get(1).map_or(32, |s| s.parse().unwrap());
    let pool: usize = args.get(2).map_or(24, |s| s.parse().unwrap());
    let seed: u64 = args.get(3).map_or(0xD54, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0xD54)
    });
    let top_k: usize = args.get(4).map_or(6, |s| s.parse().unwrap());
    let overlap: i64 = args.get(5).map_or(-1, |s| s.parse().unwrap());
    let tokens: usize = args.get(6).map_or(2, |s| s.parse().unwrap());
    assert!(pool <= NUM_EXPERT_SLOTS && top_k <= pool);
    assert!(overlap <= top_k as i64, "overlap cannot exceed top_k");
    assert!(
        (1..=MROW_MAX as usize).contains(&tokens),
        "tokens must be in 1..={MROW_MAX} (the widest compiled _m entry point)"
    );
    // The pool must hold the forced-overlap construction below: row 0's whole
    // top_k, plus (top_k - overlap) globally fresh picks for each later row.
    let needed = top_k + (tokens - 1) * (top_k - overlap.max(0) as usize);
    assert!(
        overlap < 0 || pool >= needed,
        "pool {pool} too small for tokens={tokens} top_k={top_k} overlap={overlap}: need {needed}"
    );
    assert!(
        INTER % (block as usize * VEC as usize) == 0
            && HIDDEN % (block as usize * VEC as usize) == 0,
        "block {block} x VEC {VEC} must divide both N={INTER} and N={HIDDEN}"
    );

    // Narrowest compiled `_m` entry point that covers `tokens`, exactly as
    // `MoeLayer::splitk_m_t_handles` picks it.
    let mrow: u32 = if tokens <= 2 { 2 } else { MROW_MAX };

    println!(
        "=== V4 multi-row (_m) decode MoE: block={block} pool={pool} top_k={top_k} \
         split={SPLIT} vec={VEC} overlap={overlap} tokens={tokens} (m{mrow}) \
         seed=0x{seed:X} ==="
    );

    let mut rng = Rng(seed);
    // One candidate row per verify slot.
    let a_bf16: Vec<u16> = (0..tokens * HIDDEN)
        .map(|_| f32_to_bf16(rng.unit() * 2.0 - 1.0))
        .collect();
    println!("generating {pool} experts x 3 projections ...");
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

    // Row 0 routes freely; every later row keeps `overlap` of row 0's picks and
    // draws the rest globally fresh, so the union is exactly
    // `top_k + (tokens-1)*(top_k-overlap)` (or is drawn independently when
    // overlap < 0). Every row's own picks stay distinct — that is the top-k
    // invariant the leader election relies on to guarantee an expert appears at
    // most once per row, hence at most `tokens` times overall.
    let mut routing: Vec<Vec<u32>> = Vec::with_capacity(tokens);
    let mut handed_out: Vec<u32> = Vec::new();
    for r in 0..tokens {
        let mut row: Vec<u32> = Vec::with_capacity(top_k);
        if r > 0 && overlap >= 0 {
            row.extend_from_slice(&routing[0][..overlap as usize]);
        }
        while row.len() < top_k {
            let e = (rng.next_u64() % pool as u64) as u32;
            if row.contains(&e) || (overlap >= 0 && r > 0 && handed_out.contains(&e)) {
                continue;
            }
            row.push(e);
            handed_out.push(e);
        }
        routing.push(row);
    }
    let mut union: Vec<u32> = Vec::new();
    for e in routing.iter().flatten() {
        if !union.contains(e) {
            union.push(*e);
        }
    }
    let distinct = union.len();
    for (r, row) in routing.iter().enumerate() {
        println!("routing row{r}={row:?}");
    }
    println!(
        "  distinct routed experts across {tokens} rows: {distinct} of {} slots",
        tokens * top_k
    );

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

    // Flat [tokens*top_k] indices for the `_m` chain; per-row [top_k] slices for
    // the reference chain — the same device memory, so the two chains cannot see
    // different routing.
    let flat: Vec<u32> = routing.iter().flatten().copied().collect();
    let idx_flat = upload(gpu, &u32s(&flat))?;
    let idx_row: Vec<DevicePtr> = (0..tokens).map(|t| idx_flat.offset(t * top_k * 4)).collect();

    let rows = tokens * top_k;
    let m_out = Outs::alloc(gpu, rows, tokens)?;
    let ref_out = Outs::alloc(gpu, rows, tokens)?;

    // Partials. Single-row: [2, SPLIT, top_k+1, INTER] / [SPLIT, top_k+1, HIDDEN].
    // Multi-row: `rows + tokens` accumulator rows instead of `top_k + 1`.
    let s = SPLIT as usize;
    let m_rows = rows + tokens;
    let partial_gu = gpu.alloc(2 * s * m_rows * INTER * 4)?;
    let partial_dn = gpu.alloc(s * m_rows * HIDDEN * 4)?;

    // ── single-row reference chain (the shipping `_v2s4` path), one row ──
    let ref_chain_stages = |row: usize, o: &Outs, stages: u32, st: u64| -> Result<()> {
        let a = a_ptr.offset(row * HIDDEN * 2);
        let idx = idx_row[row];
        let g_off = o.gate.offset(row * top_k * INTER * 2);
        let u_off = o.up.offset(row * top_k * INTER * 2);
        let d_off = o.down.offset(row * top_k * HIDDEN * 2);
        let shg = o.sh_gate.offset(row * INTER * 2);
        let shu = o.sh_up.offset(row * INTER * 2);
        let shd = o.sh_down.offset(row * HIDDEN * 2);
        if stages & 1 != 0 {
        KernelLaunch::new(
            gpu,
            gpu.kernel(MODULE, "moe_expert_gate_up_shared_t_e8m0_v2s4")?,
        )
        .grid([INTER as u32 / (block * VEC), top_k as u32 + 1, 2 * SPLIT])
        .block([block, 1, 1])
        .arg_ptr(a)
        .arg_ptr(gate_tbl.packed)
        .arg_ptr(gate_tbl.scales)
        .arg_ptr(gate_tbl.s2)
        .arg_ptr(g_off)
        .arg_ptr(up_tbl.packed)
        .arg_ptr(up_tbl.scales)
        .arg_ptr(up_tbl.s2)
        .arg_ptr(u_off)
        .arg_ptr(idx)
        .arg_ptr(sh_gate_p)
        .arg_ptr(sh_gate_s)
        .arg_f32(sh_gate.s2)
        .arg_ptr(shg)
        .arg_ptr(sh_up_p)
        .arg_ptr(sh_up_s)
        .arg_f32(sh_up.s2)
        .arg_ptr(shu)
        .arg_u32(INTER as u32)
        .arg_u32(HIDDEN as u32)
        .arg_u32(top_k as u32)
        .arg_ptr(partial_gu)
        .launch(st)?;
        }
        if stages & 2 != 0 {
        KernelLaunch::new(gpu, gpu.kernel(MODULE, "moe_gate_up_partial_finalize")?)
            .grid([(INTER as u32).div_ceil(block), top_k as u32 + 1, 2])
            .block([block, 1, 1])
            .arg_ptr(partial_gu)
            .arg_ptr(g_off)
            .arg_ptr(shg)
            .arg_ptr(u_off)
            .arg_ptr(shu)
            .arg_u32(INTER as u32)
            .arg_u32(top_k as u32)
            .arg_u32(SPLIT)
            .launch(st)?;
        }
        if stages & 4 != 0 {
        KernelLaunch::new(
            gpu,
            gpu.kernel(MODULE, "moe_expert_silu_down_shared_t_e8m0_v2s4")?,
        )
        .grid([HIDDEN as u32 / (block * VEC), top_k as u32 + 1, SPLIT])
        .block([block, 1, 1])
        .shared_mem(INTER as u32 * 4 / SPLIT)
        .arg_ptr(g_off)
        .arg_ptr(u_off)
        .arg_ptr(down_tbl.packed)
        .arg_ptr(down_tbl.scales)
        .arg_ptr(down_tbl.s2)
        .arg_ptr(d_off)
        .arg_ptr(idx)
        .arg_ptr(shg)
        .arg_ptr(shu)
        .arg_ptr(sh_down_p)
        .arg_ptr(sh_down_s)
        .arg_f32(sh_down.s2)
        .arg_ptr(shd)
        .arg_u32(HIDDEN as u32)
        .arg_u32(INTER as u32)
        .arg_u32(top_k as u32)
        .arg_ptr(partial_dn)
        .launch(st)?;
        }
        if stages & 8 != 0 {
        KernelLaunch::new(gpu, gpu.kernel(MODULE, "moe_down_partial_finalize")?)
            .grid([(HIDDEN as u32).div_ceil(block), top_k as u32 + 1, 1])
            .block([block, 1, 1])
            .arg_ptr(partial_dn)
            .arg_ptr(d_off)
            .arg_ptr(shd)
            .arg_u32(HIDDEN as u32)
            .arg_u32(top_k as u32)
            .arg_u32(SPLIT)
            .launch(st)?;
        }
        Ok(())
    };
    let ref_chain =
        |row: usize, o: &Outs, st: u64| -> Result<()> { ref_chain_stages(row, o, 0xF, st) };

    // ── multi-row chain: one launch pair covers `tokens` rows ──
    //
    // `stages` is a bitmask over {gate_up, gu-finalize, down, dn-finalize} so
    // the timing leg can attribute the win (or the loss) to a single kernel;
    // 0xF is the whole chain and the only setting that leaves valid outputs.
    let m_chain_stages = |mrow: u32, tokens: u32, o: &Outs, stages: u32, st: u64| -> Result<()> {
        let gu: KernelHandle = gpu.kernel(
            MODULE,
            &format!("moe_expert_gate_up_shared_t_e8m0_m{mrow}v2s4"),
        )?;
        let dn: KernelHandle = gpu.kernel(
            MODULE,
            &format!("moe_expert_silu_down_shared_t_e8m0_m{mrow}v2s4"),
        )?;
        let total_routed = tokens * top_k as u32;
        let fin_rows = total_routed + tokens;
        if stages & 1 != 0 {
        KernelLaunch::new(gpu, gu)
            .grid([INTER as u32 / (block * VEC), total_routed + 1, 2 * SPLIT])
            .block([block, 1, 1])
            .arg_ptr(a_ptr)
            .arg_ptr(gate_tbl.packed)
            .arg_ptr(gate_tbl.scales)
            .arg_ptr(gate_tbl.s2)
            .arg_ptr(o.gate)
            .arg_ptr(up_tbl.packed)
            .arg_ptr(up_tbl.scales)
            .arg_ptr(up_tbl.s2)
            .arg_ptr(o.up)
            .arg_ptr(idx_flat)
            .arg_ptr(sh_gate_p)
            .arg_ptr(sh_gate_s)
            .arg_f32(sh_gate.s2)
            .arg_ptr(o.sh_gate)
            .arg_ptr(sh_up_p)
            .arg_ptr(sh_up_s)
            .arg_f32(sh_up.s2)
            .arg_ptr(o.sh_up)
            .arg_u32(INTER as u32)
            .arg_u32(HIDDEN as u32)
            .arg_u32(top_k as u32)
            .arg_u32(tokens)
            .arg_ptr(partial_gu)
            .launch(st)?;
        }
        if stages & 2 != 0 {
        KernelLaunch::new(gpu, gpu.kernel(MODULE, "moe_gate_up_partial_finalize_m")?)
            .grid([(INTER as u32).div_ceil(block), fin_rows, 2])
            .block([block, 1, 1])
            .arg_ptr(partial_gu)
            .arg_ptr(o.gate)
            .arg_ptr(o.sh_gate)
            .arg_ptr(o.up)
            .arg_ptr(o.sh_up)
            .arg_u32(INTER as u32)
            .arg_u32(total_routed)
            .arg_u32(tokens)
            .arg_u32(SPLIT)
            .launch(st)?;
        }
        if stages & 4 != 0 {
        KernelLaunch::new(gpu, dn)
            .grid([HIDDEN as u32 / (block * VEC), total_routed + 1, SPLIT])
            .block([block, 1, 1])
            // One s_act k-slice per row the leader may gather: MROW, not
            // `tokens` — the kernel strides slices by the compile-time MROW.
            .shared_mem(mrow * INTER as u32 * 4 / SPLIT)
            .arg_ptr(o.gate)
            .arg_ptr(o.up)
            .arg_ptr(down_tbl.packed)
            .arg_ptr(down_tbl.scales)
            .arg_ptr(down_tbl.s2)
            .arg_ptr(o.down)
            .arg_ptr(idx_flat)
            .arg_ptr(o.sh_gate)
            .arg_ptr(o.sh_up)
            .arg_ptr(sh_down_p)
            .arg_ptr(sh_down_s)
            .arg_f32(sh_down.s2)
            .arg_ptr(o.sh_down)
            .arg_u32(HIDDEN as u32)
            .arg_u32(INTER as u32)
            .arg_u32(top_k as u32)
            .arg_u32(tokens)
            .arg_ptr(partial_dn)
            .launch(st)?;
        }
        if stages & 8 != 0 {
        KernelLaunch::new(gpu, gpu.kernel(MODULE, "moe_down_partial_finalize_m")?)
            .grid([(HIDDEN as u32).div_ceil(block), fin_rows, 1])
            .block([block, 1, 1])
            .arg_ptr(partial_dn)
            .arg_ptr(o.down)
            .arg_ptr(o.sh_down)
            .arg_u32(HIDDEN as u32)
            .arg_u32(total_routed)
            .arg_u32(tokens)
            .arg_u32(SPLIT)
            .launch(st)?;
        }
        Ok(())
    };
    let m_chain = |mrow: u32, tokens: u32, o: &Outs, st: u64| -> Result<()> {
        m_chain_stages(mrow, tokens, o, 0xF, st)
    };

    let read = |ptr: DevicePtr, count: usize| -> Result<Vec<u16>> {
        let mut buf = vec![0u8; count * 2];
        gpu.copy_d2h(ptr, &mut buf)?;
        Ok(buf
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect())
    };
    // Poison every output so an unwritten element cannot fake a match.
    let poison = |o: &Outs| -> Result<()> {
        let junk = |n: usize| -> Vec<u16> {
            (0..n)
                .map(|i| (0x7F80u16 ^ (i as u16)).rotate_left(3))
                .collect()
        };
        gpu.copy_h2d(&u16s(&junk(rows * INTER)), o.gate)?;
        gpu.copy_h2d(&u16s(&junk(rows * INTER)), o.up)?;
        gpu.copy_h2d(&u16s(&junk(rows * HIDDEN)), o.down)?;
        gpu.copy_h2d(&u16s(&junk(tokens * INTER)), o.sh_gate)?;
        gpu.copy_h2d(&u16s(&junk(tokens * INTER)), o.sh_up)?;
        gpu.copy_h2d(&u16s(&junk(tokens * HIDDEN)), o.sh_down)?;
        Ok(())
    };

    // ── GATE 1: MROW=1, one token, must be bit-identical to `_v2s4` ──
    poison(&ref_out)?;
    ref_chain(0, &ref_out, stream)?;
    poison(&m_out)?;
    m_chain(1, 1, &m_out, stream)?;
    gpu.synchronize(stream)?;
    {
        let mut bad = 0usize;
        for (label, a, b, n) in [
            ("gate", ref_out.gate, m_out.gate, top_k * INTER),
            ("up", ref_out.up, m_out.up, top_k * INTER),
            ("down", ref_out.down, m_out.down, top_k * HIDDEN),
            ("sh_gate", ref_out.sh_gate, m_out.sh_gate, INTER),
            ("sh_up", ref_out.sh_up, m_out.sh_up, INTER),
            ("sh_down", ref_out.sh_down, m_out.sh_down, HIDDEN),
        ] {
            let (x, y) = (read(a, n)?, read(b, n)?);
            let diff = x.iter().zip(&y).filter(|(p, q)| p != q).count();
            if diff > 0 {
                let i = x.iter().zip(&y).position(|(p, q)| p != q).unwrap();
                println!(
                    "  GATE1 {label}: {diff}/{n} differ, first at [{i}] \
                     ref={:.6} m1={:.6}",
                    bf16_to_f32(x[i]),
                    bf16_to_f32(y[i])
                );
                bad += diff;
            }
        }
        if bad > 0 {
            bail!("GATE 1 FAILED: _m1v2s4 is not bit-identical to _v2s4 ({bad} elems)");
        }
        println!("  GATE 1 PASS: _m1v2s4 == _v2s4, bit-identical (all 6 outputs)");
    }

    // ── GATE 2: `_m{mrow}`, `tokens` rows, vs `tokens` independent passes ──
    poison(&ref_out)?;
    for r in 0..tokens {
        ref_chain(r, &ref_out, stream)?;
    }
    poison(&m_out)?;
    m_chain(mrow, tokens as u32, &m_out, stream)?;
    gpu.synchronize(stream)?;
    {
        // bf16 carries 8 mantissa bits; 1/64 is 3 ULP of headroom over the
        // reassociation the shared-row block-set merge can introduce.
        const REL_MAX: f32 = 1.0 / 64.0;
        let mut worst = 0.0f32;
        let mut worst_at = String::new();
        let mut exact = true;
        for (label, a, b, n) in [
            ("gate", ref_out.gate, m_out.gate, rows * INTER),
            ("up", ref_out.up, m_out.up, rows * INTER),
            ("down", ref_out.down, m_out.down, rows * HIDDEN),
            ("sh_gate", ref_out.sh_gate, m_out.sh_gate, tokens * INTER),
            ("sh_up", ref_out.sh_up, m_out.sh_up, tokens * INTER),
            ("sh_down", ref_out.sh_down, m_out.sh_down, tokens * HIDDEN),
        ] {
            let (x, y) = (read(a, n)?, read(b, n)?);
            for (i, (p, q)) in x.iter().zip(&y).enumerate() {
                if p != q {
                    exact = false;
                }
                let (fp, fq) = (bf16_to_f32(*p), bf16_to_f32(*q));
                let denom = fp.abs().max(fq.abs()).max(1e-6);
                let rel = (fp - fq).abs() / denom;
                if rel > worst {
                    worst = rel;
                    worst_at = format!("{label}[{i}] ref={fp:.6} m{mrow}={fq:.6}");
                }
            }
        }
        if worst > REL_MAX {
            bail!("GATE 2 FAILED: worst rel {worst:.5} > {REL_MAX:.5} at {worst_at}");
        }
        println!(
            "  GATE 2 PASS: _m{mrow}v2s4 == {tokens}x _v2s4, worst rel {worst:.2e} \
             (bound {REL_MAX:.2e}){}",
            if exact { ", bit-identical" } else { "" }
        );
    }

    // ── timing ──
    let time = |f: &dyn Fn(u64) -> Result<()>| -> Result<f64> {
        for _ in 0..20 {
            f(stream)?;
        }
        gpu.synchronize(stream)?;
        let (mut e0, mut e1) = (0u64, 0u64);
        unsafe {
            cuEventCreate(&mut e0, 0);
            cuEventCreate(&mut e1, 0);
            cuEventRecord(e0, stream);
        }
        for _ in 0..ITERS {
            f(stream)?;
        }
        let mut ms = 0f32;
        unsafe {
            cuEventRecord(e1, stream);
            cuEventSynchronize(e1);
            cuEventElapsedTime(&mut ms, e0, e1);
            cuEventDestroy_v2(e0);
            cuEventDestroy_v2(e1);
        }
        Ok(ms as f64 / ITERS as f64)
    };

    // Weight bytes a single row streams: (gate+up+down) routed + shared, packed
    // at 0.5 B/elem plus the scale table.
    let routed_b = |k: usize, n: usize| (k / 2 * n) + (k / GS_ROUTED * n);
    let shared_b = |k: usize, n: usize| (k / 2 * n) + (k / GS_SHARED * n);
    let per_row_bytes = (top_k * (2 * routed_b(HIDDEN, INTER) + routed_b(INTER, HIDDEN))
        + 2 * shared_b(HIDDEN, INTER)
        + shared_b(INTER, HIDDEN)) as f64;

    let t_ref = time(&|st| {
        for r in 0..tokens {
            ref_chain(r, &ref_out, st)?;
        }
        Ok(())
    })?;
    let t_m = time(&|st| m_chain(mrow, tokens as u32, &m_out, st))?;

    // The reference reads every row's weights in full; the `_m` chain reads the
    // union, so its effective GB/s is quoted against the SAME nominal byte
    // count — the ratio, not the absolute, is the result.
    let nominal = tokens as f64 * per_row_bytes;
    let gbs = |t: f64| nominal / (t * 1e-3) / 1e9;
    println!();
    println!(
        "  {tokens}x _v2s4 (today, per-row verify): {t_ref:.4} ms  ({:.0} GB/s nominal)",
        gbs(t_ref)
    );
    println!(
        "  1x _m{mrow}v2s4 (dedup'd multi-row):     {t_m:.4} ms  ({:.0} GB/s nominal)  \
         [{:+.1}%]",
        gbs(t_m),
        (t_ref / t_m - 1.0) * 100.0
    );
    // Ceiling: only the union of the rows' experts plus ONE shared expert has to
    // be read; the duplicate picks collapse.
    let union_bytes = (distinct * (2 * routed_b(HIDDEN, INTER) + routed_b(INTER, HIDDEN))
        + 2 * shared_b(HIDDEN, INTER)
        + shared_b(INTER, HIDDEN)) as f64;
    println!(
        "  byte-traffic ceiling at this routing: {:.3}x (union {:.2} MB vs {:.2} MB)",
        nominal / union_bytes,
        union_bytes / 1e6,
        nominal / 1e6
    );
    println!(
        "  verify MoE across {LAYERS:.0} layers: {:.2} ms -> {:.2} ms",
        t_ref * LAYERS,
        t_m * LAYERS
    );

    // Per-stage attribution. The two GEMVs carry all the weight traffic; the
    // finalizes only sum SPLIT float partials. If the dedup win lands on one
    // GEMV and not the other, that GEMV is the one to tune.
    println!();
    println!("  stage                {:>10} {:>10} {:>9}", "per-row", "dedup", "gain");
    for (label, bit) in [
        ("gate_up GEMV", 1u32),
        ("gate_up finalize", 2),
        ("silu_down GEMV", 4),
        ("down finalize", 8),
    ] {
        let r = time(&|st| {
            for row in 0..tokens {
                ref_chain_stages(row, &ref_out, bit, st)?;
            }
            Ok(())
        })?;
        let m = time(&|st| m_chain_stages(mrow, tokens as u32, &m_out, bit, st))?;
        println!(
            "  {label:<20} {r:>8.4} ms {m:>8.4} ms {:>+8.1}%",
            (r / m - 1.0) * 100.0
        );
    }
    Ok(())
}
