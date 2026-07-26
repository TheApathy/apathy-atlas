// SPDX-License-Identifier: AGPL-3.0-only
//! ATLAS_FUSED_ELEMWISE microbench + bit-exactness oracle.
//!
//! Chain A — flat-verify q/k epilogue (per layer, n=K verify rows):
//!   UNFUSED (the real multi_seq op sequence, launched identically):
//!     3n scatter D2D → n×(q rms_norm_vanilla grid[nq]) → n×(k rms_norm grid[nkv])
//!     → n×rope_forward_yarn_scaled → n×reshape_and_cache_flash → n gather D2D
//!   FUSED: 1× fused_qkv_norm_rope_cache_write_bf16.
//!   PASS requires BYTE-IDENTICAL Q buffer + K cache + V cache.
//!
//! Chain B — MoE blend tail:
//!   UNFUSED: moe_weighted_sum_blend_batch2 (grid Y=n) + bf16_residual_add
//!   FUSED:   1× moe_weighted_sum_blend_residual_batchn
//!   PASS requires BYTE-IDENTICAL output + hidden.
//!
//! Chain A-serial / B-serial — the SERIAL (M=1) decode layouts:
//!   A-serial UNFUSED (the real decode/attention_forward.rs op sequence on
//!   the contiguous qkv_output buffer [q | k | v], NO scatter/gather):
//!     q rms_norm grid[nq] → k rms_norm grid[nkv] → rope_forward_yarn_scaled
//!     (q+k, 1 launch) → reshape_and_cache_flash (1 token)   = 4 launches
//!   A-serial FUSED: 1× fused_qkv_norm_rope_cache_write_bf16 at n=1.
//!   B-serial UNFUSED: moe_weighted_sum_blend (the single-token
//!     moe_expert_gemv kernel) + bf16_residual_add               = 2 launches
//!   B-serial FUSED: 1× moe_weighted_sum_blend_residual_batchn at n=1.
//!   PASS requires BYTE-IDENTICAL q/output/hidden + caches.
//!
//!   cargo run -p spark-model --release --example fused_elemwise_microtest \
//!     --features cuda,gpu-examples
use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const N: usize = 8; // K=γ+1 verify rows (Laguna γ=7)
const NKV: usize = 8;
const HD: usize = 128;
const BS: usize = 16; // kv block size
const NUM_BLOCKS: usize = 64;
const H: usize = 3072;
const TOPK: usize = 10;
const ITERS: usize = 200;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn r(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.f()
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
fn uu(g: &dyn GpuBackend, d: &[u32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn ui64(g: &dyn GpuBackend, d: &[i64]) -> Result<DevicePtr> {
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
fn zeroed(g: &dyn GpuBackend, bytes: usize) -> Result<DevicePtr> {
    let p = g.alloc(bytes)?;
    g.copy_h2d(&vec![0u8; bytes], p)?;
    Ok(p)
}

#[allow(clippy::too_many_arguments)]
struct ChainA {
    nq: usize,
    rot: usize,
    af: f32,
    // device buffers
    q_src: DevicePtr, // pristine inputs (never mutated)
    k_src: DevicePtr,
    v_src: DevicePtr,
    q_work: DevicePtr, // per-run working copy (contiguous scratch layout)
    k_work: DevicePtr,
    v_work: DevicePtr,
    qkv_buf: DevicePtr, // strided per-seq buf (unfused scatter target)
    q_contig: DevicePtr,
    q_norm_w: DevicePtr,
    k_norm_w: DevicePtr,
    positions: DevicePtr,
    inv_freq: DevicePtr,
    slots: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    // kernels
    k_norm: spark_runtime::gpu::KernelHandle,
    k_rope: spark_runtime::gpu::KernelHandle,
    k_cachew: spark_runtime::gpu::KernelHandle,
    k_fused: spark_runtime::gpu::KernelHandle,
    norm_offset_one: u32,
}

impl ChainA {
    fn per_seq_qkv(&self) -> usize {
        (self.nq * HD + 2 * NKV * HD) * 2
    }
    fn q_bytes(&self) -> usize {
        self.nq * HD * 2
    }
    fn kv_bytes(&self) -> usize {
        NKV * HD * 2
    }

    fn reset(&self, g: &dyn GpuBackend, stream: u64) -> Result<()> {
        g.copy_d2d_async(self.q_src, self.q_work, N * self.q_bytes(), stream)?;
        g.copy_d2d_async(self.k_src, self.k_work, N * self.kv_bytes(), stream)?;
        g.copy_d2d_async(self.v_src, self.v_work, N * self.kv_bytes(), stream)?;
        Ok(())
    }

    /// The EXACT unfused multi_seq op sequence (scatter → norms → rope →
    /// cache write → gather), with the same launch geometry the Rust path
    /// uses per row.
    fn run_unfused(&self, g: &dyn GpuBackend, stream: u64) -> Result<()> {
        let psq = self.per_seq_qkv();
        let (qb, kvb) = (self.q_bytes(), self.kv_bytes());
        // scatter (ms_qkv_batchn*: 3n D2D)
        for i in 0..N {
            let q_out = self.qkv_buf.offset(i * psq);
            let k_out = q_out.offset(qb);
            let v_out = k_out.offset(kvb);
            g.copy_d2d_async(self.q_work.offset(i * qb), q_out, qb, stream)?;
            g.copy_d2d_async(self.k_work.offset(i * kvb), k_out, kvb, stream)?;
            g.copy_d2d_async(self.v_work.offset(i * kvb), v_out, kvb, stream)?;
        }
        // per-row q/k per-head norms (ms_qkv_norms: 2n launches)
        for i in 0..N {
            let q_out = self.qkv_buf.offset(i * psq);
            let k_out = q_out.offset(qb);
            for (ptr, rows, w) in [
                (q_out, self.nq as u32, self.q_norm_w),
                (k_out, NKV as u32, self.k_norm_w),
            ] {
                KernelLaunch::new(g, self.k_norm)
                    .grid([rows, 1, 1])
                    .block([HD as u32, 1, 1])
                    .arg_ptr(ptr)
                    .arg_ptr(w)
                    .arg_ptr(ptr)
                    .arg_u32(HD as u32)
                    .arg_f32(1e-6)
                    .launch(stream)?;
            }
        }
        // per-row rope_forward_yarn_scaled (n launches)
        let half_rot = (self.rot / 2).max(1) as u32;
        let pos_per_block = (128 / half_rot).max(1);
        let seq_blocks = div_ceil(1, pos_per_block);
        for i in 0..N {
            let q_out = self.qkv_buf.offset(i * psq);
            let k_out = q_out.offset(qb);
            KernelLaunch::new(g, self.k_rope)
                .grid([(self.nq + NKV) as u32, seq_blocks, 1])
                .block([128, 1, 1])
                .arg_ptr(q_out)
                .arg_ptr(k_out)
                .arg_ptr(self.positions.offset(i * 4))
                .arg_u32(1)
                .arg_u32(self.nq as u32)
                .arg_u32(NKV as u32)
                .arg_u32(HD as u32)
                .arg_u32(self.rot as u32)
                .arg_ptr(self.inv_freq)
                .arg_f32(self.af)
                .launch(stream)?;
        }
        // per-row reshape_and_cache_flash (n launches)
        for i in 0..N {
            let q_out = self.qkv_buf.offset(i * psq);
            let k_out = q_out.offset(qb);
            let v_out = k_out.offset(kvb);
            KernelLaunch::new(g, self.k_cachew)
                .grid([1, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(k_out)
                .arg_ptr(v_out)
                .arg_ptr(self.k_cache)
                .arg_ptr(self.v_cache)
                .arg_ptr(self.slots.offset(i * 8))
                .arg_u32(NKV as u32)
                .arg_u32(HD as u32)
                .arg_u32(BS as u32)
                .arg_u32((NKV * HD) as u32)
                .arg_u32((NKV * HD) as u32)
                .launch(stream)?;
        }
        // gather Q (ms_phase_paged_decode: n D2D)
        for i in 0..N {
            g.copy_d2d_async(
                self.qkv_buf.offset(i * psq),
                self.q_contig.offset(i * qb),
                qb,
                stream,
            )?;
        }
        Ok(())
    }

    fn run_fused(&self, g: &dyn GpuBackend, stream: u64) -> Result<()> {
        KernelLaunch::new(g, self.k_fused)
            .grid([(self.nq + 2 * NKV) as u32, N as u32, 1])
            .block([HD as u32, 1, 1])
            .arg_ptr(self.q_work)
            .arg_ptr(self.k_work)
            .arg_ptr(self.v_work)
            .arg_ptr(self.q_norm_w)
            .arg_ptr(self.k_norm_w)
            .arg_ptr(self.positions)
            .arg_ptr(self.inv_freq)
            .arg_ptr(self.k_cache)
            .arg_ptr(self.v_cache)
            .arg_ptr(self.slots)
            .arg_u32(self.nq as u32)
            .arg_u32(NKV as u32)
            .arg_u32(HD as u32)
            .arg_u32(self.rot as u32)
            .arg_u32(BS as u32)
            .arg_f32(1e-6)
            .arg_f32(self.af)
            .arg_u32(self.norm_offset_one)
            .launch(stream)
    }
}

fn time_us(g: &dyn GpuBackend, stream: u64, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    // warmup
    for _ in 0..20 {
        f()?;
    }
    g.synchronize(stream)?;
    let t = std::time::Instant::now();
    for _ in 0..ITERS {
        f()?;
    }
    g.synchronize(stream)?;
    Ok(t.elapsed().as_micros() as f64 / ITERS as f64)
}

#[allow(clippy::too_many_arguments)]
fn chain_a(
    g: &dyn GpuBackend,
    stream: u64,
    r: &mut Lcg,
    nq: usize,
    rot: usize,
    af: f32,
    norm_module: &str,
    norm_kernel: &str,
    norm_offset_one: u32,
    label: &str,
) -> Result<(f64, f64)> {
    let q: Vec<bf16> = (0..N * nq * HD)
        .map(|_| bf16::from_f64(r.r(-2.0, 2.0)))
        .collect();
    let k: Vec<bf16> = (0..N * NKV * HD)
        .map(|_| bf16::from_f64(r.r(-2.0, 2.0)))
        .collect();
    let v: Vec<bf16> = (0..N * NKV * HD)
        .map(|_| bf16::from_f64(r.r(-2.0, 2.0)))
        .collect();
    let qw: Vec<bf16> = (0..HD).map(|_| bf16::from_f64(r.r(0.5, 1.5))).collect();
    let kw: Vec<bf16> = (0..HD).map(|_| bf16::from_f64(r.r(0.5, 1.5))).collect();
    let pos: Vec<u32> = (0..N as u32).map(|i| 1000 + 7 * i).collect();
    // plain inv_freq table (contents identical for both paths — any table works)
    let inv_freq: Vec<f32> = (0..rot / 2)
        .map(|j| (1.0f64 / 500000f64.powf((2 * j) as f64 / rot as f64)) as f32)
        .collect();
    let slots: Vec<i64> = (0..N as i64).map(|i| i * 17 + 3).collect();

    let cache_bytes = NUM_BLOCKS * BS * NKV * HD * 2;
    let a = ChainA {
        nq,
        rot,
        af,
        q_src: ub(g, &q)?,
        k_src: ub(g, &k)?,
        v_src: ub(g, &v)?,
        q_work: g.alloc(N * nq * HD * 2)?,
        k_work: g.alloc(N * NKV * HD * 2)?,
        v_work: g.alloc(N * NKV * HD * 2)?,
        qkv_buf: g.alloc(N * (nq * HD + 2 * NKV * HD) * 2)?,
        q_contig: g.alloc(N * nq * HD * 2)?,
        q_norm_w: ub(g, &qw)?,
        k_norm_w: ub(g, &kw)?,
        positions: uu(g, &pos)?,
        inv_freq: uf(g, &inv_freq)?,
        slots: ui64(g, &slots)?,
        k_cache: zeroed(g, cache_bytes)?,
        v_cache: zeroed(g, cache_bytes)?,
        k_norm: g.kernel(norm_module, norm_kernel)?,
        k_rope: g.kernel("rope", "rope_forward_yarn_scaled")?,
        k_cachew: g.kernel("reshape_and_cache", "reshape_and_cache_flash")?,
        k_fused: g.kernel(
            "fused_verify_elemwise",
            "fused_qkv_norm_rope_cache_write_bf16",
        )?,
        norm_offset_one,
    };

    // ── bit-exactness: unfused reference ──
    a.reset(g, stream)?;
    a.run_unfused(g, stream)?;
    g.synchronize(stream)?;
    let ref_q = dl(g, a.q_contig, N * nq * HD * 2)?;
    let ref_kc = dl(g, a.k_cache, cache_bytes)?;
    let ref_vc = dl(g, a.v_cache, cache_bytes)?;

    // ── fused (fresh inputs, fresh caches) ──
    g.copy_h2d(&vec![0u8; cache_bytes], a.k_cache)?;
    g.copy_h2d(&vec![0u8; cache_bytes], a.v_cache)?;
    a.reset(g, stream)?;
    a.run_fused(g, stream)?;
    g.synchronize(stream)?;
    let fus_q = dl(g, a.q_work, N * nq * HD * 2)?;
    let fus_kc = dl(g, a.k_cache, cache_bytes)?;
    let fus_vc = dl(g, a.v_cache, cache_bytes)?;

    let q_ok = ref_q == fus_q;
    let k_ok = ref_kc == fus_kc;
    let v_ok = ref_vc == fus_vc;
    let mismatch = |a: &[u8], b: &[u8]| a.iter().zip(b).filter(|(x, y)| x != y).count() / 2;
    println!(
        "[{label}] BIT-EXACT: q={} k_cache={} v_cache={}{}",
        q_ok,
        k_ok,
        v_ok,
        if q_ok && k_ok && v_ok {
            " ✅".to_string()
        } else {
            format!(
                " ❌ (mismatched elems q={} k={} v={})",
                mismatch(&ref_q, &fus_q),
                mismatch(&ref_kc, &fus_kc),
                mismatch(&ref_vc, &fus_vc)
            )
        }
    );

    // ── timing ──
    let t_unfused = time_us(g, stream, || a.run_unfused(g, stream))?;
    let t_fused = time_us(g, stream, || a.run_fused(g, stream))?;
    println!(
        "[{label}] unfused {:.1} µs/layer ({} launches+copies) → fused {:.1} µs/layer (1 launch) | Δ {:.1} µs/layer, ×48 layers = {:.2} ms/step",
        t_unfused,
        8 * N,
        t_fused,
        t_unfused - t_fused,
        (t_unfused - t_fused) * 48.0 / 1000.0
    );
    if !(q_ok && k_ok && v_ok) {
        anyhow::bail!("[{label}] bit-exactness FAILED");
    }
    Ok((t_unfused, t_fused))
}

/// SERIAL (M=1) q/k epilogue: the exact decode/attention_forward.rs unfused
/// op sequence (q norm → k norm → rope → cache write, 4 launches, in place
/// on the contiguous qkv_output layout) vs ONE fused launch at n=1.
#[allow(clippy::too_many_arguments)]
fn chain_a_serial(
    g: &dyn GpuBackend,
    stream: u64,
    r: &mut Lcg,
    nq: usize,
    rot: usize,
    af: f32,
    norm_module: &str,
    norm_kernel: &str,
    norm_offset_one: u32,
    label: &str,
) -> Result<(f64, f64)> {
    let qb = nq * HD * 2;
    let kvb = NKV * HD * 2;
    let qkv_bytes = qb + 2 * kvb;
    let src: Vec<bf16> = (0..qkv_bytes / 2)
        .map(|_| bf16::from_f64(r.r(-2.0, 2.0)))
        .collect();
    let qw: Vec<bf16> = (0..HD).map(|_| bf16::from_f64(r.r(0.5, 1.5))).collect();
    let kw: Vec<bf16> = (0..HD).map(|_| bf16::from_f64(r.r(0.5, 1.5))).collect();
    let inv_freq: Vec<f32> = (0..rot / 2)
        .map(|j| (1.0f64 / 500000f64.powf((2 * j) as f64 / rot as f64)) as f32)
        .collect();

    let src_d = ub(g, &src)?;
    let qkv = g.alloc(qkv_bytes)?; // working copy: [q | k | v] like qkv_output
    let q_norm_w = ub(g, &qw)?;
    let k_norm_w = ub(g, &kw)?;
    let positions = uu(g, &[4321u32])?;
    let inv_freq_d = uf(g, &inv_freq)?;
    let slots = ui64(g, &[42i64])?;
    let cache_bytes = NUM_BLOCKS * BS * NKV * HD * 2;
    let k_cache = zeroed(g, cache_bytes)?;
    let v_cache = zeroed(g, cache_bytes)?;

    let k_norm = g.kernel(norm_module, norm_kernel)?;
    let k_rope = g.kernel("rope", "rope_forward_yarn_scaled")?;
    let k_cachew = g.kernel("reshape_and_cache", "reshape_and_cache_flash")?;
    let k_fused = g.kernel(
        "fused_verify_elemwise",
        "fused_qkv_norm_rope_cache_write_bf16",
    )?;

    let (q_out, k_out, v_out) = (qkv, qkv.offset(qb), qkv.offset(qb + kvb));
    let run_unfused = |g: &dyn GpuBackend| -> Result<()> {
        // q + k per-head norms (ops::rms_norm: grid rows, block hd)
        for (ptr, rows, w) in [(q_out, nq as u32, q_norm_w), (k_out, NKV as u32, k_norm_w)] {
            KernelLaunch::new(g, k_norm)
                .grid([rows, 1, 1])
                .block([HD as u32, 1, 1])
                .arg_ptr(ptr)
                .arg_ptr(w)
                .arg_ptr(ptr)
                .arg_u32(HD as u32)
                .arg_f32(1e-6)
                .launch(stream)?;
        }
        // rope_forward_yarn_scaled — ONE launch covers q+k in the serial path
        let half_rot = (rot / 2).max(1) as u32;
        let pos_per_block = (128 / half_rot).max(1);
        KernelLaunch::new(g, k_rope)
            .grid([(nq + NKV) as u32, div_ceil(1, pos_per_block), 1])
            .block([128, 1, 1])
            .arg_ptr(q_out)
            .arg_ptr(k_out)
            .arg_ptr(positions)
            .arg_u32(1)
            .arg_u32(nq as u32)
            .arg_u32(NKV as u32)
            .arg_u32(HD as u32)
            .arg_u32(rot as u32)
            .arg_ptr(inv_freq_d)
            .arg_f32(af)
            .launch(stream)?;
        // reshape_and_cache_flash (1 token)
        KernelLaunch::new(g, k_cachew)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(k_out)
            .arg_ptr(v_out)
            .arg_ptr(k_cache)
            .arg_ptr(v_cache)
            .arg_ptr(slots)
            .arg_u32(NKV as u32)
            .arg_u32(HD as u32)
            .arg_u32(BS as u32)
            .arg_u32((NKV * HD) as u32)
            .arg_u32((NKV * HD) as u32)
            .launch(stream)
    };
    let run_fused = |g: &dyn GpuBackend| -> Result<()> {
        KernelLaunch::new(g, k_fused)
            .grid([(nq + 2 * NKV) as u32, 1, 1])
            .block([HD as u32, 1, 1])
            .arg_ptr(q_out)
            .arg_ptr(k_out)
            .arg_ptr(v_out)
            .arg_ptr(q_norm_w)
            .arg_ptr(k_norm_w)
            .arg_ptr(positions)
            .arg_ptr(inv_freq_d)
            .arg_ptr(k_cache)
            .arg_ptr(v_cache)
            .arg_ptr(slots)
            .arg_u32(nq as u32)
            .arg_u32(NKV as u32)
            .arg_u32(HD as u32)
            .arg_u32(rot as u32)
            .arg_u32(BS as u32)
            .arg_f32(1e-6)
            .arg_f32(af)
            .arg_u32(norm_offset_one)
            .launch(stream)
    };

    // ── bit-exactness: unfused reference ──
    g.copy_d2d_async(src_d, qkv, qkv_bytes, stream)?;
    run_unfused(g)?;
    g.synchronize(stream)?;
    let ref_q = dl(g, q_out, qb)?;
    let ref_kc = dl(g, k_cache, cache_bytes)?;
    let ref_vc = dl(g, v_cache, cache_bytes)?;

    // ── fused (fresh inputs, fresh caches) ──
    g.copy_h2d(&vec![0u8; cache_bytes], k_cache)?;
    g.copy_h2d(&vec![0u8; cache_bytes], v_cache)?;
    g.copy_d2d_async(src_d, qkv, qkv_bytes, stream)?;
    run_fused(g)?;
    g.synchronize(stream)?;
    let fus_q = dl(g, q_out, qb)?;
    let fus_kc = dl(g, k_cache, cache_bytes)?;
    let fus_vc = dl(g, v_cache, cache_bytes)?;

    let (q_ok, k_ok, v_ok) = (ref_q == fus_q, ref_kc == fus_kc, ref_vc == fus_vc);
    println!(
        "[{label}] BIT-EXACT: q={} k_cache={} v_cache={}{}",
        q_ok,
        k_ok,
        v_ok,
        if q_ok && k_ok && v_ok { " ✅" } else { " ❌" }
    );

    // ── timing (in-place chain is idempotent-enough for timing: we time the
    // launch train itself; the real decode also runs it exactly once on
    // fresh GEMV output, so per-iteration reset would only add D2D noise) ──
    let t_unfused = time_us(g, stream, || run_unfused(g))?;
    let t_fused = time_us(g, stream, || run_fused(g))?;
    println!(
        "[{label}] unfused {:.1} µs/layer (4 launches) → fused {:.1} µs/layer (1 launch) | Δ {:.1} µs/layer",
        t_unfused,
        t_fused,
        t_unfused - t_fused,
    );
    if !(q_ok && k_ok && v_ok) {
        anyhow::bail!("[{label}] bit-exactness FAILED");
    }
    Ok((t_unfused, t_fused))
}

/// SERIAL (M=1) MoE blend tail: the single-token `moe_weighted_sum_blend`
/// (moe_expert_gemv — the kernel MoeLayer::forward launches at decode) +
/// `bf16_residual_add` vs ONE `moe_weighted_sum_blend_residual_batchn` at n=1.
fn chain_b_serial(g: &dyn GpuBackend, stream: u64, r: &mut Lcg) -> Result<(f64, f64)> {
    let expert: Vec<bf16> = (0..TOPK * H)
        .map(|_| bf16::from_f64(r.r(-1.0, 1.0)))
        .collect();
    let wts: Vec<f32> = (0..TOPK).map(|_| r.r(0.0, 0.4) as f32).collect();
    let shared: Vec<bf16> = (0..H).map(|_| bf16::from_f64(r.r(-1.0, 1.0))).collect();
    let input: Vec<bf16> = (0..H).map(|_| bf16::from_f64(r.r(-1.0, 1.0))).collect();
    let hidden0: Vec<bf16> = (0..H).map(|_| bf16::from_f64(r.r(-4.0, 4.0))).collect();

    let expert_d = ub(g, &expert)?;
    let wts_d = uf(g, &wts)?;
    let shared_d = ub(g, &shared)?;
    let input_d = ub(g, &input)?;
    let hidden_src = ub(g, &hidden0)?;
    let hidden_a = g.alloc(H * 2)?;
    let hidden_b = g.alloc(H * 2)?;
    let out_a = g.alloc(H * 2)?;
    let out_b = g.alloc(H * 2)?;

    let k_blend = g.kernel("moe_expert_gemv", "moe_weighted_sum_blend")?;
    let k_resid = g.kernel("residual_add", "bf16_residual_add")?;
    let k_fused = g.kernel(
        "fused_verify_elemwise",
        "moe_weighted_sum_blend_residual_batchn",
    )?;

    let run_unfused = |g: &dyn GpuBackend| -> Result<()> {
        KernelLaunch::new(g, k_blend)
            .grid([div_ceil(H as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(out_a)
            .arg_ptr(expert_d)
            .arg_ptr(wts_d)
            .arg_ptr(shared_d)
            .arg_ptr(input_d)
            .arg_ptr(DevicePtr(0)) // Laguna: NULL shared-expert gate
            .arg_u32(H as u32)
            .arg_u32(TOPK as u32)
            .arg_u32(H as u32)
            .launch(stream)?;
        KernelLaunch::new(g, k_resid)
            .grid([div_ceil(H as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(hidden_a)
            .arg_ptr(out_a)
            .arg_u32(H as u32)
            .launch(stream)
    };
    let run_fused = |g: &dyn GpuBackend| -> Result<()> {
        KernelLaunch::new(g, k_fused)
            .grid([div_ceil(H as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(out_b)
            .arg_ptr(expert_d)
            .arg_ptr(wts_d)
            .arg_ptr(shared_d)
            .arg_ptr(input_d)
            .arg_ptr(DevicePtr(0))
            .arg_ptr(hidden_b)
            .arg_u32(H as u32)
            .arg_u32(TOPK as u32)
            .arg_u32(H as u32)
            .launch(stream)
    };

    // bit-exactness
    g.copy_d2d_async(hidden_src, hidden_a, H * 2, stream)?;
    g.copy_d2d_async(hidden_src, hidden_b, H * 2, stream)?;
    run_unfused(g)?;
    run_fused(g)?;
    g.synchronize(stream)?;
    let (ra, rb) = (dl(g, out_a, H * 2)?, dl(g, out_b, H * 2)?);
    let (ha, hb) = (dl(g, hidden_a, H * 2)?, dl(g, hidden_b, H * 2)?);
    let out_ok = ra == rb;
    let hid_ok = ha == hb;
    println!(
        "[serial blend+residual] BIT-EXACT: output={out_ok} hidden={hid_ok}{}",
        if out_ok && hid_ok { " ✅" } else { " ❌" }
    );

    let t_unfused = time_us(g, stream, || run_unfused(g))?;
    let t_fused = time_us(g, stream, || run_fused(g))?;
    println!(
        "[serial blend+residual] unfused {:.1} µs/layer (2 launches) → fused {:.1} µs/layer (1 launch) | Δ {:.1} µs/layer",
        t_unfused,
        t_fused,
        t_unfused - t_fused,
    );
    if !(out_ok && hid_ok) {
        anyhow::bail!("[serial blend+residual] bit-exactness FAILED");
    }
    Ok((t_unfused, t_fused))
}

fn chain_b(g: &dyn GpuBackend, stream: u64, r: &mut Lcg) -> Result<(f64, f64)> {
    let expert: Vec<bf16> = (0..N * TOPK * H)
        .map(|_| bf16::from_f64(r.r(-1.0, 1.0)))
        .collect();
    let wts: Vec<f32> = (0..N * TOPK).map(|_| r.r(0.0, 0.4) as f32).collect();
    let shared: Vec<bf16> = (0..N * H).map(|_| bf16::from_f64(r.r(-1.0, 1.0))).collect();
    let input: Vec<bf16> = (0..N * H).map(|_| bf16::from_f64(r.r(-1.0, 1.0))).collect();
    let hidden0: Vec<bf16> = (0..N * H).map(|_| bf16::from_f64(r.r(-4.0, 4.0))).collect();

    let expert_d = ub(g, &expert)?;
    let wts_d = uf(g, &wts)?;
    let shared_d = ub(g, &shared)?;
    let input_d = ub(g, &input)?;
    let hidden_src = ub(g, &hidden0)?;
    let hidden_a = g.alloc(N * H * 2)?;
    let hidden_b = g.alloc(N * H * 2)?;
    let out_a = g.alloc(N * H * 2)?;
    let out_b = g.alloc(N * H * 2)?;

    let k_blend = g.kernel("moe_fused_batch2", "moe_weighted_sum_blend_batch2")?;
    let k_resid = g.kernel("residual_add", "bf16_residual_add")?;
    let k_fused = g.kernel(
        "fused_verify_elemwise",
        "moe_weighted_sum_blend_residual_batchn",
    )?;

    let run_unfused = |g: &dyn GpuBackend| -> Result<()> {
        KernelLaunch::new(g, k_blend)
            .grid([div_ceil(H as u32, 256), N as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(out_a)
            .arg_ptr(expert_d)
            .arg_ptr(wts_d)
            .arg_ptr(shared_d)
            .arg_ptr(input_d)
            .arg_ptr(DevicePtr(0)) // Laguna: NULL shared-expert gate
            .arg_u32(H as u32)
            .arg_u32(TOPK as u32)
            .arg_u32(H as u32)
            .launch(stream)?;
        KernelLaunch::new(g, k_resid)
            .grid([div_ceil((N * H) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(hidden_a)
            .arg_ptr(out_a)
            .arg_u32((N * H) as u32)
            .launch(stream)
    };
    let run_fused = |g: &dyn GpuBackend| -> Result<()> {
        KernelLaunch::new(g, k_fused)
            .grid([div_ceil(H as u32, 256), N as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(out_b)
            .arg_ptr(expert_d)
            .arg_ptr(wts_d)
            .arg_ptr(shared_d)
            .arg_ptr(input_d)
            .arg_ptr(DevicePtr(0))
            .arg_ptr(hidden_b)
            .arg_u32(H as u32)
            .arg_u32(TOPK as u32)
            .arg_u32(H as u32)
            .launch(stream)
    };

    // bit-exactness
    g.copy_d2d_async(hidden_src, hidden_a, N * H * 2, stream)?;
    g.copy_d2d_async(hidden_src, hidden_b, N * H * 2, stream)?;
    run_unfused(g)?;
    run_fused(g)?;
    g.synchronize(stream)?;
    let (ra, rb) = (dl(g, out_a, N * H * 2)?, dl(g, out_b, N * H * 2)?);
    let (ha, hb) = (dl(g, hidden_a, N * H * 2)?, dl(g, hidden_b, N * H * 2)?);
    let out_ok = ra == rb;
    let hid_ok = ha == hb;
    println!(
        "[blend+residual] BIT-EXACT: output={out_ok} hidden={hid_ok}{}",
        if out_ok && hid_ok { " ✅" } else { " ❌" }
    );

    let t_unfused = time_us(g, stream, || run_unfused(g))?;
    let t_fused = time_us(g, stream, || run_fused(g))?;
    println!(
        "[blend+residual] unfused {:.1} µs/layer (2 launches) → fused {:.1} µs/layer (1 launch) | Δ {:.1} µs/layer, ×48 layers = {:.2} ms/step",
        t_unfused,
        t_fused,
        t_unfused - t_fused,
        (t_unfused - t_fused) * 48.0 / 1000.0
    );
    if !(out_ok && hid_ok) {
        anyhow::bail!("[blend+residual] bit-exactness FAILED");
    }
    Ok((t_unfused, t_fused))
}

fn main() -> Result<()> {
    let g0 = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;
    let stream = g.default_stream();
    let mut r = Lcg(0x1A6_04A);

    println!(
        "=== fused_elemwise_microtest: n={N} nkv={NKV} hd={HD} bs={BS} h={H} topk={TOPK} iters={ITERS} ==="
    );

    // Laguna full-attention layers: nq=48, rot=64, yarn af=1.3465736, vanilla norm.
    let (u1, f1) = chain_a(
        g,
        stream,
        &mut r,
        48,
        64,
        1.346_573_6,
        "rms_norm_vanilla",
        "rms_norm_vanilla",
        0,
        "qk-epilogue nq=48 rot=64 vanilla",
    )?;
    // Laguna sliding layers: nq=72, rot=128, af=1.0, vanilla norm.
    let (u2, f2) = chain_a(
        g,
        stream,
        &mut r,
        72,
        128,
        1.0,
        "rms_norm_vanilla",
        "rms_norm_vanilla",
        0,
        "qk-epilogue nq=72 rot=128 vanilla",
    )?;
    // Offset-from-1 norm variant (non-Laguna models) — exactness only.
    let (_, _) = chain_a(
        g,
        stream,
        &mut r,
        48,
        64,
        1.0,
        "norm",
        "rms_norm",
        1,
        "qk-epilogue nq=48 rot=64 offset1",
    )?;
    let (u3, f3) = chain_b(g, stream, &mut r)?;

    // Laguna: 12 full (nq=48) + 36 sliding (nq=72) layers.
    let step_unfused = 12.0 * u1 + 36.0 * u2 + 48.0 * u3;
    let step_fused = 12.0 * f1 + 36.0 * f2 + 48.0 * f3;
    println!(
        "=== per-step (48 layers, Laguna 12×nq48 + 36×nq72): unfused {:.2} ms → fused {:.2} ms | saving {:.2} ms/step ===",
        step_unfused / 1000.0,
        step_fused / 1000.0,
        (step_unfused - step_fused) / 1000.0
    );

    // ── SERIAL (M=1) decode layouts ──
    println!("--- serial (M=1) decode ---");
    let (su1, sf1) = chain_a_serial(
        g,
        stream,
        &mut r,
        48,
        64,
        1.346_573_6,
        "rms_norm_vanilla",
        "rms_norm_vanilla",
        0,
        "serial qk-epilogue nq=48 rot=64 vanilla",
    )?;
    let (su2, sf2) = chain_a_serial(
        g,
        stream,
        &mut r,
        72,
        128,
        1.0,
        "rms_norm_vanilla",
        "rms_norm_vanilla",
        0,
        "serial qk-epilogue nq=72 rot=128 vanilla",
    )?;
    // Offset-from-1 norm variant (non-Laguna models) — exactness only.
    let (_, _) = chain_a_serial(
        g,
        stream,
        &mut r,
        48,
        64,
        1.0,
        "norm",
        "rms_norm",
        1,
        "serial qk-epilogue nq=48 rot=64 offset1",
    )?;
    let (su3, sf3) = chain_b_serial(g, stream, &mut r)?;

    let tok_unfused = 12.0 * su1 + 36.0 * su2 + 48.0 * su3;
    let tok_fused = 12.0 * sf1 + 36.0 * sf2 + 48.0 * sf3;
    println!(
        "=== serial per-token (48 layers, Laguna 12×nq48 + 36×nq72): unfused {:.2} ms → fused {:.2} ms | saving {:.2} ms/token ===",
        tok_unfused / 1000.0,
        tok_fused / 1000.0,
        (tok_unfused - tok_fused) / 1000.0
    );
    Ok(())
}
