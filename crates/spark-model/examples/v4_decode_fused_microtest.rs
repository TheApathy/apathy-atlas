// SPDX-License-Identifier: AGPL-3.0-only
//! V4 M=1 decode glue fusion oracle (ATLAS_V4_DECODE_FUSED).
//!
//! Gates the two fused kernels byte-identical against the incumbent chains at
//! the production decode shape (nq=64, hd=512, nope=448, rope=64, cache 576):
//!   1. `v4_decode_rope_fused`      vs extract(Q) + extract(K) +
//!      `rope_forward_yarn_interleaved` + writeback(Q) + writeback(K)
//!   2. `v4_decode_cache_fused_fp8` vs `mla_cache_assemble_batched` +
//!      `reshape_and_cache_flash_fp8`
//!   3. conjugate mode              vs extract + `..._interleaved_inv` + writeback
//! for BOTH layer rope configs: CSA/HCA (θ=160000 yarn table + mscale) and
//! sliding (θ=10000, mscale=1). Tier 1: any byte diff is a failure (exit 1).
//! Timing compares the 7-launch vs 2-launch chain wall clock (launch-bound at
//! M=1; graphed per-node savings are smaller — see the fusion commit message).
//!
//! Build with ATLAS_TARGET_MODEL=deepseek-v4-flash (needs the V4 mla_absorbed
//! module). GPU must be otherwise idle.
//!
//!   ATLAS_TARGET_MODEL=deepseek-v4-flash cargo run --release -p spark-model \
//!     --example v4_decode_fused_microtest --features cuda,gpu-examples -- [pos] [seed]

use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const NQ: u32 = 64;
const HD: u32 = 512;
const NOPE: u32 = 448;
const ROPE: u32 = 64;
const KV_LORA: u32 = 512;
const CACHE_DIM: u32 = 576; // KV_LORA + ROPE
const BS: u32 = 16; // paged cache block size
const NUM_BLOCKS: u32 = 8;
const K_SCALE: f32 = 0.023_41;
const V_SCALE: f32 = 0.031_7;

struct Sm64(u64);
impl Sm64 {
    // splitmix64 — deterministic across runs/machines (playbook §3).
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn f(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * ((self.next() >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn ub(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn uf(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn dl(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

#[allow(clippy::too_many_arguments)]
fn rope_extract(
    g: &dyn GpuBackend,
    k: KernelHandle,
    full: DevicePtr,
    tmp: DevicePtr,
    nh: u32,
    stride: u32,
    stream: u64,
) -> Result<()> {
    let total = nh * ROPE;
    KernelLaunch::new(g, k)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(full)
        .arg_ptr(tmp)
        .arg_u32(1)
        .arg_u32(nh)
        .arg_u32(HD)
        .arg_u32(NOPE)
        .arg_u32(ROPE)
        .arg_u32(stride)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
fn rope_writeback(
    g: &dyn GpuBackend,
    k: KernelHandle,
    tmp: DevicePtr,
    full: DevicePtr,
    nh: u32,
    stride: u32,
    stream: u64,
) -> Result<()> {
    let total = nh * ROPE;
    KernelLaunch::new(g, k)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(tmp)
        .arg_ptr(full)
        .arg_u32(1)
        .arg_u32(nh)
        .arg_u32(HD)
        .arg_u32(NOPE)
        .arg_u32(ROPE)
        .arg_u32(stride)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
fn rope_yarn_interleaved(
    g: &dyn GpuBackend,
    k: KernelHandle,
    q: DevicePtr,
    kk: DevicePtr,
    positions: DevicePtr,
    nq: u32,
    nkv: u32,
    inv_freq: DevicePtr,
    mscale: f32,
    stream: u64,
) -> Result<()> {
    // Mirrors ops::rope_yarn: head_dim = rotary_dim = ROPE on the contiguous tmp.
    let pos_per_block = (128 / (ROPE / 2)).max(1);
    KernelLaunch::new(g, k)
        .grid([nq + nkv, div_ceil(1, pos_per_block), 1])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(kk)
        .arg_ptr(positions)
        .arg_u32(1)
        .arg_u32(nq)
        .arg_u32(nkv)
        .arg_u32(ROPE)
        .arg_u32(ROPE)
        .arg_ptr(inv_freq)
        .arg_f32(mscale)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
fn fused_rope(
    g: &dyn GpuBackend,
    k: KernelHandle,
    q: DevicePtr,
    kk: DevicePtr,
    positions: DevicePtr,
    nq: u32,
    nkv: u32,
    inv_freq: DevicePtr,
    mscale: f32,
    conj: bool,
    stream: u64,
) -> Result<()> {
    let total = (nq + nkv) * (ROPE / 2);
    KernelLaunch::new(g, k)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(kk)
        .arg_ptr(positions)
        .arg_u32(nq)
        .arg_u32(nkv)
        .arg_u32(HD)
        .arg_u32(NOPE)
        .arg_u32(ROPE)
        .arg_ptr(inv_freq)
        .arg_f32(mscale)
        .arg_u32(if conj { 1 } else { 0 })
        .launch(stream)
}

struct Cfg {
    name: &'static str,
    inv_freq: Vec<f32>,
    mscale: f32,
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let pos: u32 = a.get(1).map_or(4242, |s| s.parse().unwrap());
    let seed: u64 = a.get(2).map_or(0xA71A5, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0xA71A5)
    });
    println!(
        "=== v4_decode_fused microtest: nq={NQ} hd={HD} nope={NOPE} rope={ROPE} cache={CACHE_DIM} pos={pos} seed={seed:#x} ==="
    );

    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let stream = 0u64;

    let m = |m: &str, f: &str| -> Result<KernelHandle> { g.kernel(m, f) };
    let k_extract = m("mla_absorbed", "mla_q_rope_extract_batched")?;
    let k_writeback = m("mla_absorbed", "mla_q_rope_writeback_batched")?;
    let k_assemble = m("mla_absorbed", "mla_cache_assemble_batched")?;
    let k_fused_rope = m("mla_absorbed", "v4_decode_rope_fused")?;
    let k_fused_cache = m("mla_absorbed", "v4_decode_cache_fused_fp8")?;
    let k_yarn = m("rope", "rope_forward_yarn_interleaved")?;
    let k_yarn_inv = m("rope", "rope_forward_yarn_interleaved_inv")?;
    let k_reshape_fp8 = m("reshape_and_cache", "reshape_and_cache_flash_fp8")?;

    // Two production layer configs: CSA/HCA layers rope with the θ=160000 YaRN
    // table + attention-temperature mscale; sliding layers use plain θ=10000,
    // mscale=1 (attention_forward_v4.rs step 3).
    let yarn_like = |theta: f64| -> Vec<f32> {
        (0..(ROPE / 2) as usize)
            .map(|i| (1.0 / theta.powf(2.0 * i as f64 / ROPE as f64)) as f32)
            .collect()
    };
    let cfgs = [
        Cfg {
            name: "csa/hca (θ=160000 yarn, mscale)",
            inv_freq: yarn_like(160000.0),
            mscale: 1.187_205,
        },
        Cfg {
            name: "sliding (θ=10000, mscale=1)",
            inv_freq: yarn_like(10000.0),
            mscale: 1.0,
        },
    ];

    let mut rng = Sm64(seed);
    let q_host: Vec<bf16> = (0..(NQ * HD) as usize)
        .map(|_| bf16::from_f64(rng.f(-2.0, 2.0)))
        .collect();
    let k_host: Vec<bf16> = (0..HD as usize)
        .map(|_| bf16::from_f64(rng.f(-2.0, 2.0)))
        .collect();
    let attn_host: Vec<bf16> = (0..(NQ * HD) as usize)
        .map(|_| bf16::from_f64(rng.f(-2.0, 2.0)))
        .collect();

    let pos_buf = {
        let p = g.alloc(4)?;
        g.copy_h2d(&pos.to_le_bytes(), p)?;
        p
    };
    let slot_val: i64 = 37; // block 2, offset 5 at BS=16
    let slot_buf = {
        let p = g.alloc(8)?;
        g.copy_h2d(&slot_val.to_le_bytes(), p)?;
        p
    };

    let pool_bytes = (NUM_BLOCKS * BS * CACHE_DIM) as usize;
    let cache_stride = (BS * CACHE_DIM) as u64;

    let mut failures = 0usize;

    for cfg in &cfgs {
        println!("--- config: {} ---", cfg.name);
        let inv_freq = uf(g, &cfg.inv_freq)?;

        // ── A: incumbent 7-launch chain ──────────────────────────────────
        let q_a = ub(g, &q_host)?;
        let k_a = ub(g, &k_host)?;
        let v_a = ub(g, &k_host)?; // pre-rope latent copy (K==V)
        let q_tmp = g.alloc((NQ * ROPE) as usize * 2)?;
        let k_tmp = g.alloc(ROPE as usize * 2)?;
        let k_asm = g.alloc(CACHE_DIM as usize * 2)?;
        let v_asm = g.alloc(CACHE_DIM as usize * 2)?;
        let k_pool_a = g.alloc(pool_bytes)?;
        let v_pool_a = g.alloc(pool_bytes)?;
        g.memset_async(k_pool_a, 0, pool_bytes, stream)?;
        g.memset_async(v_pool_a, 0, pool_bytes, stream)?;

        rope_extract(g, k_extract, q_a, q_tmp, NQ, NQ * HD, stream)?;
        rope_extract(g, k_extract, k_a, k_tmp, 1, HD, stream)?;
        rope_yarn_interleaved(
            g, k_yarn, q_tmp, k_tmp, pos_buf, NQ, 1, inv_freq, cfg.mscale, stream,
        )?;
        rope_writeback(g, k_writeback, q_tmp, q_a, NQ, NQ * HD, stream)?;
        rope_writeback(g, k_writeback, k_tmp, k_a, 1, HD, stream)?;
        KernelLaunch::new(g, k_assemble)
            .grid([1, 1, 1])
            .block([CACHE_DIM.max(256), 1, 1])
            .arg_ptr(v_a)
            .arg_ptr(k_tmp)
            .arg_ptr(k_asm)
            .arg_ptr(v_asm)
            .arg_u32(KV_LORA)
            .arg_u32(ROPE)
            .arg_u32(CACHE_DIM)
            .launch(stream)?;
        KernelLaunch::new(g, k_reshape_fp8)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(k_asm)
            .arg_ptr(v_asm)
            .arg_ptr(k_pool_a)
            .arg_ptr(v_pool_a)
            .arg_ptr(slot_buf)
            .arg_u32(1)
            .arg_u32(CACHE_DIM)
            .arg_u32(BS)
            .arg_f32(K_SCALE)
            .arg_f32(V_SCALE)
            .arg_u32(CACHE_DIM)
            .arg_u32(CACHE_DIM)
            .arg_u64(cache_stride)
            .launch(stream)?;
        g.synchronize(stream)?;

        // ── B: fused 2-launch chain ──────────────────────────────────────
        let q_b = ub(g, &q_host)?;
        let k_b = ub(g, &k_host)?;
        let v_b = ub(g, &k_host)?;
        let k_pool_b = g.alloc(pool_bytes)?;
        let v_pool_b = g.alloc(pool_bytes)?;
        g.memset_async(k_pool_b, 0, pool_bytes, stream)?;
        g.memset_async(v_pool_b, 0, pool_bytes, stream)?;

        fused_rope(
            g, k_fused_rope, q_b, k_b, pos_buf, NQ, 1, inv_freq, cfg.mscale, false, stream,
        )?;
        KernelLaunch::new(g, k_fused_cache)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(v_b)
            .arg_ptr(k_b.offset(NOPE as usize * 2))
            .arg_ptr(k_pool_b)
            .arg_ptr(v_pool_b)
            .arg_ptr(slot_buf)
            .arg_u32(KV_LORA)
            .arg_u32(ROPE)
            .arg_u32(BS)
            .arg_f32(K_SCALE)
            .arg_f32(V_SCALE)
            .arg_u64(cache_stride)
            .launch(stream)?;
        g.synchronize(stream)?;

        // ── byte-identity gates (tier 1) ─────────────────────────────────
        let gate = |label: &str, x: &[u8], y: &[u8]| -> usize {
            let diff = x.iter().zip(y).filter(|(a, b)| a != b).count();
            println!(
                "  {label}: {} ({diff} byte diffs / {})",
                if diff == 0 { "BYTE-IDENTICAL" } else { "MISMATCH" },
                x.len()
            );
            usize::from(diff != 0)
        };
        failures += gate(
            "q after rope   ",
            &dl(g, q_a, (NQ * HD) as usize * 2)?,
            &dl(g, q_b, (NQ * HD) as usize * 2)?,
        );
        failures += gate(
            "k after rope   ",
            &dl(g, k_a, HD as usize * 2)?,
            &dl(g, k_b, HD as usize * 2)?,
        );
        failures += gate(
            "k_pool (fp8)   ",
            &dl(g, k_pool_a, pool_bytes)?,
            &dl(g, k_pool_b, pool_bytes)?,
        );
        failures += gate(
            "v_pool (fp8)   ",
            &dl(g, v_pool_a, pool_bytes)?,
            &dl(g, v_pool_b, pool_bytes)?,
        );

        // ── C: conjugate (attention-output de-rotation) mode ─────────────
        let o_a = ub(g, &attn_host)?;
        let o_b = ub(g, &attn_host)?;
        let o_tmp = g.alloc((NQ * ROPE) as usize * 2)?;
        rope_extract(g, k_extract, o_a, o_tmp, NQ, NQ * HD, stream)?;
        rope_yarn_interleaved(
            g, k_yarn_inv, o_tmp, o_tmp, pos_buf, NQ, 0, inv_freq, cfg.mscale, stream,
        )?;
        rope_writeback(g, k_writeback, o_tmp, o_a, NQ, NQ * HD, stream)?;
        fused_rope(
            g, k_fused_rope, o_b, o_b, pos_buf, NQ, 0, inv_freq, cfg.mscale, true, stream,
        )?;
        g.synchronize(stream)?;
        failures += gate(
            "attn de-rotate ",
            &dl(g, o_a, (NQ * HD) as usize * 2)?,
            &dl(g, o_b, (NQ * HD) as usize * 2)?,
        );

        // ── timing: 7-launch vs 2-launch chain (launch-bound at M=1) ─────
        const ITERS: usize = 2000;
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            rope_extract(g, k_extract, q_a, q_tmp, NQ, NQ * HD, stream)?;
            rope_extract(g, k_extract, k_a, k_tmp, 1, HD, stream)?;
            rope_yarn_interleaved(
                g, k_yarn, q_tmp, k_tmp, pos_buf, NQ, 1, inv_freq, cfg.mscale, stream,
            )?;
            rope_writeback(g, k_writeback, q_tmp, q_a, NQ, NQ * HD, stream)?;
            rope_writeback(g, k_writeback, k_tmp, k_a, 1, HD, stream)?;
            KernelLaunch::new(g, k_assemble)
                .grid([1, 1, 1])
                .block([CACHE_DIM.max(256), 1, 1])
                .arg_ptr(v_a)
                .arg_ptr(k_tmp)
                .arg_ptr(k_asm)
                .arg_ptr(v_asm)
                .arg_u32(KV_LORA)
                .arg_u32(ROPE)
                .arg_u32(CACHE_DIM)
                .launch(stream)?;
            KernelLaunch::new(g, k_reshape_fp8)
                .grid([1, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(k_asm)
                .arg_ptr(v_asm)
                .arg_ptr(k_pool_a)
                .arg_ptr(v_pool_a)
                .arg_ptr(slot_buf)
                .arg_u32(1)
                .arg_u32(CACHE_DIM)
                .arg_u32(BS)
                .arg_f32(K_SCALE)
                .arg_f32(V_SCALE)
                .arg_u32(CACHE_DIM)
                .arg_u32(CACHE_DIM)
                .arg_u64(cache_stride)
                .launch(stream)?;
        }
        g.synchronize(stream)?;
        let unfused_us = t0.elapsed().as_micros() as f64 / ITERS as f64;
        let t1 = std::time::Instant::now();
        for _ in 0..ITERS {
            fused_rope(
                g, k_fused_rope, q_b, k_b, pos_buf, NQ, 1, inv_freq, cfg.mscale, false, stream,
            )?;
            KernelLaunch::new(g, k_fused_cache)
                .grid([1, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(v_b)
                .arg_ptr(k_b.offset(NOPE as usize * 2))
                .arg_ptr(k_pool_b)
                .arg_ptr(v_pool_b)
                .arg_ptr(slot_buf)
                .arg_u32(KV_LORA)
                .arg_u32(ROPE)
                .arg_u32(BS)
                .arg_f32(K_SCALE)
                .arg_f32(V_SCALE)
                .arg_u64(cache_stride)
                .launch(stream)?;
        }
        g.synchronize(stream)?;
        let fused_us = t1.elapsed().as_micros() as f64 / ITERS as f64;
        println!(
            "  timing: unfused 7-launch {unfused_us:.1} µs/step, fused 2-launch {fused_us:.1} µs/step ({:.1} µs saved/layer; ×43 layers ≈ {:.2} ms/token eager)",
            unfused_us - fused_us,
            (unfused_us - fused_us) * 43.0 / 1000.0
        );
    }

    // ── D: hc_pre vs hc_pre_fused (single-launch multi-block, T=1) ──────
    {
        println!("--- config: hc_pre_fused (H=4096, hc=4, mix=24) ---");
        const H: u32 = 4096;
        const HC: u32 = 4;
        const MIX: u32 = (2 + HC) * HC; // 24
        const SINKHORN: u32 = 4;
        const EPS: f32 = 1e-6;
        const HC_EPS: f32 = 1e-5;
        let k_hc_pre = m("hyper_connection", "hc_pre")?;
        let k_hc_fused = m("hyper_connection", "hc_pre_fused")?;

        let streams: Vec<f32> = (0..(HC * H) as usize)
            .map(|_| rng.f(-1.0, 1.0) as f32)
            .collect();
        let hc_fn: Vec<f32> = (0..(MIX * HC * H) as usize)
            .map(|_| rng.f(-0.05, 0.05) as f32)
            .collect();
        let hc_scale: Vec<f32> = vec![0.7, 0.9, 1.1];
        let hc_base: Vec<f32> = (0..MIX as usize).map(|_| rng.f(-0.5, 0.5) as f32).collect();
        let d_streams = uf(g, &streams)?;
        let d_fn = uf(g, &hc_fn)?;
        let d_scale = uf(g, &hc_scale)?;
        let d_base = uf(g, &hc_base)?;
        let y_a = g.alloc(H as usize * 2)?;
        let y_b = g.alloc(H as usize * 2)?;
        let post_a = g.alloc(HC as usize * 4)?;
        let post_b = g.alloc(HC as usize * 4)?;
        let comb_a = g.alloc((HC * HC) as usize * 4)?;
        let comb_b = g.alloc((HC * HC) as usize * 4)?;
        // [MIX+1] f32 partials + u32 arrival counter, zeroed once.
        let scratch = g.alloc((MIX as usize + 2) * 4)?;
        g.copy_h2d(&vec![0u8; (MIX as usize + 2) * 4], scratch)?;

        let launch_plain = |y: DevicePtr, post: DevicePtr, comb: DevicePtr| -> Result<()> {
            KernelLaunch::new(g, k_hc_pre)
                .grid([1, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(d_streams)
                .arg_ptr(d_fn)
                .arg_ptr(d_scale)
                .arg_ptr(d_base)
                .arg_ptr(y)
                .arg_ptr(post)
                .arg_ptr(comb)
                .arg_u32(H)
                .arg_u32(HC)
                .arg_u32(SINKHORN)
                .arg_f32(EPS)
                .arg_f32(HC_EPS)
                .launch(stream)
        };
        let launch_fused = |y: DevicePtr, post: DevicePtr, comb: DevicePtr| -> Result<()> {
            KernelLaunch::new(g, k_hc_fused)
                .grid([MIX + 1, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(d_streams)
                .arg_ptr(d_fn)
                .arg_ptr(d_scale)
                .arg_ptr(d_base)
                .arg_ptr(y)
                .arg_ptr(post)
                .arg_ptr(comb)
                .arg_ptr(scratch)
                .arg_u32(H)
                .arg_u32(HC)
                .arg_u32(SINKHORN)
                .arg_f32(EPS)
                .arg_f32(HC_EPS)
                .launch(stream)
        };
        launch_plain(y_a, post_a, comb_a)?;
        launch_fused(y_b, post_b, comb_b)?;
        g.synchronize(stream)?;
        let gate = |label: &str, x: &[u8], y: &[u8]| -> usize {
            let diff = x.iter().zip(y).filter(|(a, b)| a != b).count();
            println!(
                "  {label}: {} ({diff} byte diffs / {})",
                if diff == 0 { "BYTE-IDENTICAL" } else { "MISMATCH" },
                x.len()
            );
            usize::from(diff != 0)
        };
        failures += gate(
            "hc y_out       ",
            &dl(g, y_a, H as usize * 2)?,
            &dl(g, y_b, H as usize * 2)?,
        );
        failures += gate(
            "hc post        ",
            &dl(g, post_a, HC as usize * 4)?,
            &dl(g, post_b, HC as usize * 4)?,
        );
        failures += gate(
            "hc comb        ",
            &dl(g, comb_a, (HC * HC) as usize * 4)?,
            &dl(g, comb_b, (HC * HC) as usize * 4)?,
        );
        // Second fused launch proves the arrival counter re-arms itself.
        launch_fused(y_b, post_b, comb_b)?;
        g.synchronize(stream)?;
        failures += gate(
            "hc re-arm      ",
            &dl(g, y_a, H as usize * 2)?,
            &dl(g, y_b, H as usize * 2)?,
        );

        const ITERS: usize = 2000;
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            launch_plain(y_a, post_a, comb_a)?;
        }
        g.synchronize(stream)?;
        let plain_us = t0.elapsed().as_micros() as f64 / ITERS as f64;
        let t1 = std::time::Instant::now();
        for _ in 0..ITERS {
            launch_fused(y_b, post_b, comb_b)?;
        }
        g.synchronize(stream)?;
        let fused_us = t1.elapsed().as_micros() as f64 / ITERS as f64;
        println!(
            "  timing: hc_pre 1-block {plain_us:.1} µs, hc_pre_fused 25-block {fused_us:.1} µs per site (×2 sites ×43 layers)"
        );
    }

    if failures == 0 {
        println!("PASS: all gates byte-identical (tier 1 proven at this shape)");
        Ok(())
    } else {
        println!("FAIL: {failures} gate(s) mismatched");
        std::process::exit(1);
    }
}
