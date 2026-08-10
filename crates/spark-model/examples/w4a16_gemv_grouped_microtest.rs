// SPDX-License-Identifier: AGPL-3.0-only

//! Bit-exactness + timing oracle for `w4a16_gemv_grouped` (one-launch
//! block-diagonal wo_a) vs the shipping 8-launch per-group `w4a16_gemv`.
//!
//! Shape is the V4 MLA wo_a: o_groups=8 × [o_lora=1024, group_in=4096],
//! weight rows contiguous ([8192, 4096] NVFP4). The dispatch in
//! `attention_forward_v4.rs` (and the drafter's `d_o_proj` in
//! `dspark_head.rs`) launches per group with pointer offsets; this oracle
//! reproduces those exact offsets for the reference leg.
//!
//! Gates, in order:
//!   1. BIT-IDENTICAL: grouped output must equal the 8-launch output byte for
//!      byte. The kernel's per-row math is unchanged, so anything less is a
//!      wiring bug, not noise. STOP if this fails.
//!   2. Timing: CUDA events over ITERS iterations, weights rotated over a
//!      ≥256 MB footprint so no leg benefits from L2 residency (matches the
//!      serve-side reality: 43 layers of cold wo_a every step).
//!      Report per-launch-config effective GB/s and the speedup.
//!   Also times the wo_b single-launch shape [4096, 8192] as the big-launch
//!   reference point on the same silicon.
//!
//! Gate 6 (ATLAS_VERIFY_GEMV_V2 kernels): the compile-time-M V2 entries
//! (`w4a16_gemv_grouped_batchm_v2_m{4,5,6,8}` and
//! `w8a16_gemv_batchm_exact_v2_m{4,5,6,8}`) must be BYTE-IDENTICAL to their
//! runtime-M incumbents at every width, plus cold-weight (rotated) GB/s at
//! M=6 for both families — the DSpark γ=5 verify width.
//!
//! Usage:
//!   cargo run --release -p spark-model --example w4a16_gemv_grouped_microtest \
//!       --features cuda,gpu-examples -- [seed]
//! Exit 0 = PASS (bit-identical + grouped not slower), 1 = FAIL.

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

const GROUP_SIZE: usize = 16;
const O_GROUPS: usize = 8;
const O_LORA: usize = 1024;
const GROUP_IN: usize = 4096;
const ITERS: u32 = 200;
const ROTATION_BYTES: usize = 256 << 20;

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
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

fn u16s_to_le(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// E4M3 group-scale byte from a small representable set (exact round-trip).
fn e4m3_scale_byte(sel: u32) -> u8 {
    let e = 5 + (sel % 5);
    ((e as u8) << 3) & 0x7F
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn gen_weight_bytes(rng: &mut Rng, n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let half_k = k / 2;
    let num_groups = k / GROUP_SIZE;
    let mut packed = vec![0u8; n * half_k];
    let mut scale = vec![0u8; n * num_groups];
    for b in packed.iter_mut() {
        *b = (rng.next_u64() & 0xFF) as u8;
    }
    for s in scale.iter_mut() {
        *s = e4m3_scale_byte(rng.next_u64() as u32);
    }
    (packed, scale)
}

struct Events(u64, u64);
impl Events {
    fn new() -> Self {
        let (mut s, mut e) = (0u64, 0u64);
        unsafe {
            cuEventCreate(&mut s, 0);
            cuEventCreate(&mut e, 0);
        }
        Events(s, e)
    }
    fn elapsed_ms(&self) -> f32 {
        let mut ms = 0f32;
        unsafe {
            cuEventSynchronize(self.1);
            cuEventElapsedTime(&mut ms, self.0, self.1);
        }
        ms
    }
}
impl Drop for Events {
    fn drop(&mut self) {
        unsafe {
            cuEventDestroy_v2(self.0);
            cuEventDestroy_v2(self.1);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_per_group(
    gpu: &dyn GpuBackend,
    h: spark_runtime::gpu::KernelHandle,
    a: DevicePtr,
    packed: DevicePtr,
    scale: DevicePtr,
    scale2: f32,
    c: DevicePtr,
    stream: u64,
) -> Result<()> {
    let half_k = GROUP_IN / 2;
    let num_groups = GROUP_IN / GROUP_SIZE;
    for g in 0..O_GROUPS {
        KernelLaunch::new(gpu, h)
            .grid([(O_LORA.div_ceil(4)) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a.offset(g * GROUP_IN * 2))
            .arg_ptr(packed.offset(g * O_LORA * half_k))
            .arg_ptr(scale.offset(g * O_LORA * num_groups))
            .arg_f32(scale2)
            .arg_ptr(c.offset(g * O_LORA * 2))
            .arg_u32(O_LORA as u32)
            .arg_u32(GROUP_IN as u32)
            .launch(stream)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_grouped(
    gpu: &dyn GpuBackend,
    h: spark_runtime::gpu::KernelHandle,
    a: DevicePtr,
    packed: DevicePtr,
    scale: DevicePtr,
    scale2: f32,
    c: DevicePtr,
    stream: u64,
) -> Result<()> {
    let n_total = O_GROUPS * O_LORA;
    KernelLaunch::new(gpu, h)
        .grid([(n_total.div_ceil(4)) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(packed)
        .arg_ptr(scale)
        .arg_f32(scale2)
        .arg_ptr(c)
        .arg_u32(n_total as u32)
        .arg_u32(GROUP_IN as u32)
        .arg_u32(O_LORA as u32)
        .launch(stream)?;
    Ok(())
}

fn main() -> Result<()> {
    let seed = std::env::args()
        .nth(1)
        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0xB12F);
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let stream = gpu.default_stream();
    let base_h = gpu.kernel("w4a16_gemv", "w4a16_gemv")?;
    let grouped_h = gpu.kernel("w4a16_gemv", "w4a16_gemv_grouped")?;

    let n_total = O_GROUPS * O_LORA; // 8192
    let scale2 = 0.5f32;
    let mut rng = Rng(seed);

    // ── Gate 1: bit-exactness on one weight instance ──
    let a_bf16: Vec<u16> = (0..O_GROUPS * GROUP_IN)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let a_ptr = upload(&gpu, &u16s_to_le(&a_bf16))?;
    let (packed_h, scale_h) = gen_weight_bytes(&mut rng, n_total, GROUP_IN);
    let packed = upload(&gpu, &packed_h)?;
    let scale = upload(&gpu, &scale_h)?;
    let c_ref = gpu.alloc(n_total * 2)?;
    let c_grp = gpu.alloc(n_total * 2)?;

    launch_per_group(&gpu, base_h, a_ptr, packed, scale, scale2, c_ref, stream)?;
    launch_grouped(&gpu, grouped_h, a_ptr, packed, scale, scale2, c_grp, stream)?;
    gpu.synchronize(stream)?;

    let mut rb = vec![0u8; n_total * 2];
    let mut rg = vec![0u8; n_total * 2];
    gpu.copy_d2h(c_ref, &mut rb)?;
    gpu.copy_d2h(c_grp, &mut rg)?;
    let nz = rb.chunks_exact(2).filter(|c| *c != [0, 0]).count();
    if nz == 0 {
        bail!("dead reference output");
    }
    if rb != rg {
        let first = rb
            .iter()
            .zip(rg.iter())
            .position(|(x, y)| x != y)
            .unwrap_or(0);
        bail!(
            "GATE 1 FAIL: grouped != 8-launch reference (first diff at byte {first}, \
             {nz} nonzero ref outputs)"
        );
    }
    println!("  GATE 1 PASS: w4a16_gemv_grouped == 8x w4a16_gemv, bit-identical ({nz} nonzero)");

    // ── Gate 2: timing with weight rotation (defeat L2) ──
    let inst_bytes = packed_h.len() + scale_h.len();
    let n_inst = ROTATION_BYTES.div_ceil(inst_bytes).max(2);
    let mut packs = Vec::with_capacity(n_inst);
    for _ in 0..n_inst {
        let (p, s) = gen_weight_bytes(&mut rng, n_total, GROUP_IN);
        packs.push((upload(&gpu, &p)?, upload(&gpu, &s)?));
    }
    let weight_bytes = (n_total * (GROUP_IN / 2) + n_total * (GROUP_IN / GROUP_SIZE)) as f64;

    let time_leg = |grouped: bool| -> Result<f64> {
        // warmup
        for i in 0..8usize {
            let (p, s) = packs[i % n_inst];
            if grouped {
                launch_grouped(&gpu, grouped_h, a_ptr, p, s, scale2, c_grp, stream)?;
            } else {
                launch_per_group(&gpu, base_h, a_ptr, p, s, scale2, c_ref, stream)?;
            }
        }
        gpu.synchronize(stream)?;
        let ev = Events::new();
        unsafe { cuEventRecord(ev.0, stream) };
        for i in 0..ITERS as usize {
            let (p, s) = packs[i % n_inst];
            if grouped {
                launch_grouped(&gpu, grouped_h, a_ptr, p, s, scale2, c_grp, stream)?;
            } else {
                launch_per_group(&gpu, base_h, a_ptr, p, s, scale2, c_ref, stream)?;
            }
        }
        unsafe { cuEventRecord(ev.1, stream) };
        gpu.synchronize(stream)?;
        Ok(ev.elapsed_ms() as f64 / ITERS as f64)
    };

    let t_ref = time_leg(false)?;
    let t_grp = time_leg(true)?;
    let gbs = |t_ms: f64| weight_bytes / (t_ms / 1e3) / 1e9;
    println!(
        "  8x w4a16_gemv per-group (shipping): {t_ref:.4} ms  ({:.1} GB/s)",
        gbs(t_ref)
    );
    println!(
        "  1x w4a16_gemv_grouped:              {t_grp:.4} ms  ({:.1} GB/s)  [{:+.1}%]",
        gbs(t_grp),
        (t_ref / t_grp - 1.0) * 100.0
    );

    // ── Reference point: the wo_b single-launch shape [4096, 8192] ──
    let (wb_n, wb_k) = (4096usize, 8192usize);
    let a2: Vec<u16> = (0..wb_k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let a2_ptr = upload(&gpu, &u16s_to_le(&a2))?;
    let (p2, s2) = gen_weight_bytes(&mut rng, wb_n, wb_k);
    let inst2 = p2.len() + s2.len();
    let n_inst2 = ROTATION_BYTES.div_ceil(inst2).max(2);
    let mut packs2 = Vec::with_capacity(n_inst2);
    for _ in 0..n_inst2 {
        let (p, s) = gen_weight_bytes(&mut rng, wb_n, wb_k);
        packs2.push((upload(&gpu, &p)?, upload(&gpu, &s)?));
    }
    let c2 = gpu.alloc(wb_n * 2)?;
    for i in 0..8usize {
        let (p, s) = packs2[i % n_inst2];
        KernelLaunch::new(&gpu, base_h)
            .grid([(wb_n.div_ceil(4)) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a2_ptr)
            .arg_ptr(p)
            .arg_ptr(s)
            .arg_f32(scale2)
            .arg_ptr(c2)
            .arg_u32(wb_n as u32)
            .arg_u32(wb_k as u32)
            .launch(stream)?;
    }
    gpu.synchronize(stream)?;
    let ev = Events::new();
    unsafe { cuEventRecord(ev.0, stream) };
    for i in 0..ITERS as usize {
        let (p, s) = packs2[i % n_inst2];
        KernelLaunch::new(&gpu, base_h)
            .grid([(wb_n.div_ceil(4)) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a2_ptr)
            .arg_ptr(p)
            .arg_ptr(s)
            .arg_f32(scale2)
            .arg_ptr(c2)
            .arg_u32(wb_n as u32)
            .arg_u32(wb_k as u32)
            .launch(stream)?;
    }
    unsafe { cuEventRecord(ev.1, stream) };
    gpu.synchronize(stream)?;
    let t_wob = ev.elapsed_ms() as f64 / ITERS as f64;
    let wb_bytes = (wb_n * (wb_k / 2) + wb_n * (wb_k / GROUP_SIZE)) as f64;
    println!(
        "  wo_b shape [4096x8192] single launch: {t_wob:.4} ms  ({:.1} GB/s)",
        wb_bytes / (t_wob / 1e3) / 1e9
    );

    if t_grp > t_ref {
        bail!("grouped SLOWER than per-group — investigate before wiring");
    }

    // ── Gate 3: batched grouped (M=6, the DSpark verify width) must be
    //    BIT-IDENTICAL per row to single-row per-group launches (the
    //    ATLAS_OPROJ_EXACT semantics, at batch speed). ──
    let batch_h = gpu.kernel("w4a16_gemv", "w4a16_gemv_grouped_batchm")?;
    const M: usize = 6;
    let lda = O_GROUPS * GROUP_IN; // dense stride, q_dim-like
    let ldc = n_total;
    let a_batch: Vec<u16> = (0..M * lda)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let ab_ptr = upload(&gpu, &u16s_to_le(&a_batch))?;
    let cb_ref = gpu.alloc(M * ldc * 2)?;
    let cb_bat = gpu.alloc(M * ldc * 2)?;

    // Reference: M x (8 per-group single-row launches), row i input at
    // ab_ptr + i*lda, output at cb_ref + i*ldc.
    for i in 0..M {
        launch_per_group(
            &gpu,
            base_h,
            ab_ptr.offset(i * lda * 2),
            packed,
            scale,
            scale2,
            cb_ref.offset(i * ldc * 2),
            stream,
        )?;
    }
    // One batched grouped launch.
    KernelLaunch::new(&gpu, batch_h)
        .grid([(n_total.div_ceil(4)) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(ab_ptr)
        .arg_ptr(packed)
        .arg_ptr(scale)
        .arg_f32(scale2)
        .arg_ptr(cb_bat)
        .arg_u32(M as u32)
        .arg_u32(n_total as u32)
        .arg_u32(GROUP_IN as u32)
        .arg_u32(lda as u32)
        .arg_u32(ldc as u32)
        .arg_u32(O_LORA as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut rr = vec![0u8; M * ldc * 2];
    let mut rbat = vec![0u8; M * ldc * 2];
    gpu.copy_d2h(cb_ref, &mut rr)?;
    gpu.copy_d2h(cb_bat, &mut rbat)?;
    if rr != rbat {
        let first = rr.iter().zip(rbat.iter()).position(|(x, y)| x != y).unwrap_or(0);
        bail!("GATE 3 FAIL: grouped_batchm != per-row single-row (first diff byte {first})");
    }
    println!("  GATE 3 PASS: w4a16_gemv_grouped_batchm (M={M}) == per-row single-row, bit-identical");

    // Timing: batched grouped vs M x 8-launch per-group (shipping OPROJ_EXACT
    // cost) and vs the amortization ideal.
    let time_bat = |batched: bool| -> Result<f64> {
        for it in 0..8usize {
            let (p, s) = packs[it % n_inst];
            if batched {
                KernelLaunch::new(&gpu, batch_h)
                    .grid([(n_total.div_ceil(4)) as u32, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(ab_ptr)
                    .arg_ptr(p)
                    .arg_ptr(s)
                    .arg_f32(scale2)
                    .arg_ptr(cb_bat)
                    .arg_u32(M as u32)
                    .arg_u32(n_total as u32)
                    .arg_u32(GROUP_IN as u32)
                    .arg_u32(lda as u32)
                    .arg_u32(ldc as u32)
                    .arg_u32(O_LORA as u32)
                    .launch(stream)?;
            } else {
                for i in 0..M {
                    launch_per_group(
                        &gpu,
                        base_h,
                        ab_ptr.offset(i * lda * 2),
                        p,
                        s,
                        scale2,
                        cb_ref.offset(i * ldc * 2),
                        stream,
                    )?;
                }
            }
        }
        gpu.synchronize(stream)?;
        let ev = Events::new();
        unsafe { cuEventRecord(ev.0, stream) };
        for it in 0..ITERS as usize {
            let (p, s) = packs[it % n_inst];
            if batched {
                KernelLaunch::new(&gpu, batch_h)
                    .grid([(n_total.div_ceil(4)) as u32, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(ab_ptr)
                    .arg_ptr(p)
                    .arg_ptr(s)
                    .arg_f32(scale2)
                    .arg_ptr(cb_bat)
                    .arg_u32(M as u32)
                    .arg_u32(n_total as u32)
                    .arg_u32(GROUP_IN as u32)
                    .arg_u32(lda as u32)
                    .arg_u32(ldc as u32)
                    .arg_u32(O_LORA as u32)
                    .launch(stream)?;
            } else {
                for i in 0..M {
                    launch_per_group(
                        &gpu,
                        base_h,
                        ab_ptr.offset(i * lda * 2),
                        p,
                        s,
                        scale2,
                        cb_ref.offset(i * ldc * 2),
                        stream,
                    )?;
                }
            }
        }
        unsafe { cuEventRecord(ev.1, stream) };
        gpu.synchronize(stream)?;
        Ok(ev.elapsed_ms() as f64 / ITERS as f64)
    };
    let t_exact = time_bat(false)?;
    let t_bat = time_bat(true)?;
    println!(
        "  M={M} per-row (OPROJ_EXACT cost):     {t_exact:.4} ms  ({:.1} GB/s eff)",
        gbs(t_exact)
    );
    println!(
        "  1x grouped_batchm (bit-exact, M={M}): {t_bat:.4} ms  ({:.1} GB/s eff)  [{:.2}x vs per-row]",
        gbs(t_bat),
        t_exact / t_bat
    );

    // ── Gate 4: the SHIPPING verify wo_a path (8x batch8_ld per-group) vs
    //    grouped_batchm — head-to-head timing (never measured before the
    //    2026-08-09 serve A/B pointed at verify speed), and a bit-identity
    //    check of the _ld impl's "bit-identical to M x w4a16_gemv" claim. ──
    let ld_h = gpu.kernel("w4a16_gemv", "w4a16_gemv_batch8_ld")?;
    let cb_ld = gpu.alloc(M * ldc * 2)?;
    let half_k = GROUP_IN / 2;
    let ngroups = GROUP_IN / GROUP_SIZE;
    let launch_ld = |p: DevicePtr, s: DevicePtr, c_out: DevicePtr| -> Result<()> {
        for g in 0..O_GROUPS {
            KernelLaunch::new(&gpu, ld_h)
                .grid([(O_LORA.div_ceil(4)) as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(ab_ptr.offset(g * GROUP_IN * 2))
                .arg_ptr(p.offset(g * O_LORA * half_k))
                .arg_ptr(s.offset(g * O_LORA * ngroups))
                .arg_f32(scale2)
                .arg_ptr(c_out.offset(g * O_LORA * 2))
                .arg_u32(M as u32)
                .arg_u32(O_LORA as u32)
                .arg_u32(GROUP_IN as u32)
                .arg_u32(lda as u32)
                .arg_u32(ldc as u32)
                .launch(stream)?;
        }
        Ok(())
    };
    launch_ld(packed, scale, cb_ld)?;
    gpu.synchronize(stream)?;
    let mut rld = vec![0u8; M * ldc * 2];
    gpu.copy_d2h(cb_ld, &mut rld)?;
    let ld_bit_id = rld == rr;
    let (mism, total) = rld
        .chunks_exact(2)
        .zip(rr.chunks_exact(2))
        .fold((0usize, 0usize), |(m, t), (a, b)| {
            (m + usize::from(a != b), t + 1)
        });
    println!(
        "  batch8_ld vs per-row single-row: bit-identical = {ld_bit_id} \
         ({mism}/{total} bf16 outputs differ) — the impl comment claims true"
    );

    for it in 0..8usize {
        let (p, s) = packs[it % n_inst];
        launch_ld(p, s, cb_ld)?;
    }
    gpu.synchronize(stream)?;
    let ev = Events::new();
    unsafe { cuEventRecord(ev.0, stream) };
    for it in 0..ITERS as usize {
        let (p, s) = packs[it % n_inst];
        launch_ld(p, s, cb_ld)?;
    }
    unsafe { cuEventRecord(ev.1, stream) };
    gpu.synchronize(stream)?;
    let t_ld = ev.elapsed_ms() as f64 / ITERS as f64;
    println!(
        "  8x batch8_ld per-group (shipping):  {t_ld:.4} ms  ({:.1} GB/s eff)  \
         [grouped_batchm is {:.2}x vs _ld]",
        gbs(t_ld),
        t_ld / t_bat
    );

    // ── Gate 5: w8a16_gemv_batchm_exact (M=6) must be BIT-IDENTICAL to M x
    //    single-row w8a16_gemv (the Phase-A FP8 projections under
    //    ATLAS_VERIFY_EXACT_GEMV=1). Shape: wq_a-like [1024, 4096]. ──
    let w8_h = gpu.kernel("w8a16_gemv", "w8a16_gemv")?;
    let w8x_h = gpu.kernel("w8a16_gemv_batch4", "w8a16_gemv_batchm_exact")?;
    let (w8n, w8k) = (1024usize, 4096usize);
    let mut w8_bytes = vec![0u8; w8n * w8k];
    for b in w8_bytes.iter_mut() {
        // Avoid NaN encodings (0x7F/0xFF map to 0 in the LUT but keep it tame)
        *b = (rng.next_u64() & 0x7F) as u8;
    }
    let mut w8_scales = vec![0u8; w8n.div_ceil(128) * w8k.div_ceil(128) * 4];
    for c in w8_scales.chunks_exact_mut(4) {
        c.copy_from_slice(&rng.uniform(0.001, 0.02).to_le_bytes());
    }
    let w8_w = upload(&gpu, &w8_bytes)?;
    let w8_s = upload(&gpu, &w8_scales)?;
    let a8: Vec<u16> = (0..6 * w8k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let a8_ptr = upload(&gpu, &u16s_to_le(&a8))?;
    let c8_ref = gpu.alloc(6 * w8n * 2)?;
    let c8_x = gpu.alloc(6 * w8n * 2)?;
    for i in 0..6usize {
        KernelLaunch::new(&gpu, w8_h)
            .grid([(w8n.div_ceil(4)) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a8_ptr.offset(i * w8k * 2))
            .arg_ptr(w8_w)
            .arg_ptr(w8_s)
            .arg_ptr(c8_ref.offset(i * w8n * 2))
            .arg_u32(w8n as u32)
            .arg_u32(w8k as u32)
            .launch(stream)?;
    }
    KernelLaunch::new(&gpu, w8x_h)
        .grid([(w8n.div_ceil(4)) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a8_ptr)
        .arg_ptr(w8_w)
        .arg_ptr(w8_s)
        .arg_ptr(c8_x)
        .arg_u32(6)
        .arg_u32(w8n as u32)
        .arg_u32(w8k as u32)
        .arg_u32(w8k as u32)
        .arg_u32(w8n as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut r8r = vec![0u8; 6 * w8n * 2];
    let mut r8x = vec![0u8; 6 * w8n * 2];
    gpu.copy_d2h(c8_ref, &mut r8r)?;
    gpu.copy_d2h(c8_x, &mut r8x)?;
    let nz8 = r8r.chunks_exact(2).filter(|c| *c != [0, 0]).count();
    if nz8 == 0 {
        bail!("GATE 5: dead w8 reference output");
    }
    if r8r != r8x {
        let first = r8r.iter().zip(r8x.iter()).position(|(x, y)| x != y).unwrap_or(0);
        bail!("GATE 5 FAIL: w8a16_gemv_batchm_exact != per-row w8a16_gemv (first diff byte {first})");
    }
    println!("  GATE 5 PASS: w8a16_gemv_batchm_exact (M=6) == per-row single-row, bit-identical ({nz8} nonzero)");

    // ── Gate 6: ATLAS_VERIFY_GEMV_V2 kernels — byte-identity vs the
    //    runtime-M incumbents at n ∈ {4,5,6,8}, then cold-weight GB/s at
    //    M=6 (the DSpark γ=5 verify width) for both families. The V2
    //    kernels claim tier-1 bit-identity BY CONSTRUCTION (same per-row
    //    FP sequence, only load widths / addressing / guards changed);
    //    this gate is the byte proof. ──
    let widths: [usize; 4] = [4, 5, 6, 8];
    // 8-row activation batches (the earlier gates only built 6).
    let a_w4: Vec<u16> = (0..8 * lda)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let a_w4_ptr = upload(&gpu, &u16s_to_le(&a_w4))?;
    let a_w8: Vec<u16> = (0..8 * w8k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let a_w8_ptr = upload(&gpu, &u16s_to_le(&a_w8))?;
    let c_inc4 = gpu.alloc(8 * ldc * 2)?;
    let c_v2_4 = gpu.alloc(8 * ldc * 2)?;
    let c_inc8 = gpu.alloc(8 * w8n * 2)?;
    let c_v2_8 = gpu.alloc(8 * w8n * 2)?;

    let launch_w4_inc = |m: usize, c_out: DevicePtr| -> Result<()> {
        KernelLaunch::new(&gpu, batch_h)
            .grid([(n_total.div_ceil(4)) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a_w4_ptr)
            .arg_ptr(packed)
            .arg_ptr(scale)
            .arg_f32(scale2)
            .arg_ptr(c_out)
            .arg_u32(m as u32)
            .arg_u32(n_total as u32)
            .arg_u32(GROUP_IN as u32)
            .arg_u32(lda as u32)
            .arg_u32(ldc as u32)
            .arg_u32(O_LORA as u32)
            .launch(stream)
    };
    let launch_w4_v2 = |h: spark_runtime::gpu::KernelHandle, c_out: DevicePtr| -> Result<()> {
        KernelLaunch::new(&gpu, h)
            .grid([(n_total.div_ceil(4)) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a_w4_ptr)
            .arg_ptr(packed)
            .arg_ptr(scale)
            .arg_f32(scale2)
            .arg_ptr(c_out)
            .arg_u32(n_total as u32)
            .arg_u32(GROUP_IN as u32)
            .arg_u32(lda as u32)
            .arg_u32(ldc as u32)
            .arg_u32(O_LORA as u32)
            .launch(stream)
    };
    let launch_w8_inc = |m: usize, c_out: DevicePtr| -> Result<()> {
        KernelLaunch::new(&gpu, w8x_h)
            .grid([(w8n.div_ceil(4)) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a_w8_ptr)
            .arg_ptr(w8_w)
            .arg_ptr(w8_s)
            .arg_ptr(c_out)
            .arg_u32(m as u32)
            .arg_u32(w8n as u32)
            .arg_u32(w8k as u32)
            .arg_u32(w8k as u32)
            .arg_u32(w8n as u32)
            .launch(stream)
    };
    let launch_w8_v2 = |h: spark_runtime::gpu::KernelHandle, c_out: DevicePtr| -> Result<()> {
        KernelLaunch::new(&gpu, h)
            .grid([(w8n.div_ceil(4)) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a_w8_ptr)
            .arg_ptr(w8_w)
            .arg_ptr(w8_s)
            .arg_ptr(c_out)
            .arg_u32(w8n as u32)
            .arg_u32(w8k as u32)
            .arg_u32(w8k as u32)
            .arg_u32(w8n as u32)
            .launch(stream)
    };

    let mut v2_w4_m6 = spark_runtime::gpu::KernelHandle(0);
    let mut v2_w8_m6 = spark_runtime::gpu::KernelHandle(0);
    for &m in &widths {
        let h4 = gpu.kernel("w4a16_gemv", &format!("w4a16_gemv_grouped_batchm_v2_m{m}"))?;
        let h8 = gpu.kernel(
            "w8a16_gemv_batch4",
            &format!("w8a16_gemv_batchm_exact_v2_m{m}"),
        )?;
        if m == 6 {
            v2_w4_m6 = h4;
            v2_w8_m6 = h8;
        }
        // Poison the outputs so untouched rows cannot fake a match.
        let poison4 = vec![0xA5u8; 8 * ldc * 2];
        gpu.copy_h2d(&poison4, c_inc4)?;
        gpu.copy_h2d(&poison4, c_v2_4)?;
        let poison8 = vec![0xA5u8; 8 * w8n * 2];
        gpu.copy_h2d(&poison8, c_inc8)?;
        gpu.copy_h2d(&poison8, c_v2_8)?;

        launch_w4_inc(m, c_inc4)?;
        launch_w4_v2(h4, c_v2_4)?;
        launch_w8_inc(m, c_inc8)?;
        launch_w8_v2(h8, c_v2_8)?;
        gpu.synchronize(stream)?;

        let cmp = |a: DevicePtr, b: DevicePtr, bytes: usize, what: &str| -> Result<()> {
            let mut ra = vec![0u8; bytes];
            let mut rb = vec![0u8; bytes];
            gpu.copy_d2h(a, &mut ra)?;
            gpu.copy_d2h(b, &mut rb)?;
            if ra != rb {
                let first = ra.iter().zip(rb.iter()).position(|(x, y)| x != y).unwrap_or(0);
                bail!("GATE 6 FAIL: {what} v2_m{m} != incumbent (first diff byte {first})");
            }
            Ok(())
        };
        // Compare only the m live rows (rows m..8 keep poison on both sides,
        // which must ALSO match: v2 writes exactly m rows, incumbent m rows).
        cmp(c_inc4, c_v2_4, 8 * ldc * 2, "w4a16_gemv_grouped_batchm")?;
        cmp(c_inc8, c_v2_8, 8 * w8n * 2, "w8a16_gemv_batchm_exact")?;
        println!("  GATE 6 PASS: v2_m{m} == incumbent, byte-identical (w4 grouped + w8 exact)");
    }

    // Cold-weight timing at M=6: rotate weights over >=256 MB so both legs
    // stream from DRAM (decode reality: 43 cold layers/step).
    let time_w4 = |v2: bool| -> Result<f64> {
        for it in 0..8usize {
            let (p, s) = packs[it % n_inst];
            let mut l = KernelLaunch::new(&gpu, if v2 { v2_w4_m6 } else { batch_h })
                .grid([(n_total.div_ceil(4)) as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(a_w4_ptr)
                .arg_ptr(p)
                .arg_ptr(s)
                .arg_f32(scale2)
                .arg_ptr(c_v2_4);
            if !v2 {
                l = l.arg_u32(6);
            }
            l.arg_u32(n_total as u32)
                .arg_u32(GROUP_IN as u32)
                .arg_u32(lda as u32)
                .arg_u32(ldc as u32)
                .arg_u32(O_LORA as u32)
                .launch(stream)?;
        }
        gpu.synchronize(stream)?;
        let ev = Events::new();
        unsafe { cuEventRecord(ev.0, stream) };
        for it in 0..ITERS as usize {
            let (p, s) = packs[it % n_inst];
            let mut l = KernelLaunch::new(&gpu, if v2 { v2_w4_m6 } else { batch_h })
                .grid([(n_total.div_ceil(4)) as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(a_w4_ptr)
                .arg_ptr(p)
                .arg_ptr(s)
                .arg_f32(scale2)
                .arg_ptr(c_v2_4);
            if !v2 {
                l = l.arg_u32(6);
            }
            l.arg_u32(n_total as u32)
                .arg_u32(GROUP_IN as u32)
                .arg_u32(lda as u32)
                .arg_u32(ldc as u32)
                .arg_u32(O_LORA as u32)
                .launch(stream)?;
        }
        unsafe { cuEventRecord(ev.1, stream) };
        gpu.synchronize(stream)?;
        Ok(ev.elapsed_ms() as f64 / ITERS as f64)
    };
    let t_inc_w4 = time_w4(false)?;
    let t_v2_w4 = time_w4(true)?;
    println!(
        "  M=6 w4 grouped_batchm incumbent: {t_inc_w4:.4} ms ({:.1} GB/s)  v2: {t_v2_w4:.4} ms ({:.1} GB/s)  [{:.2}x]",
        gbs(t_inc_w4),
        gbs(t_v2_w4),
        t_inc_w4 / t_v2_w4
    );

    // w8 rotation set (wq_a-shaped instances).
    let w8_inst = w8_bytes.len() + w8_scales.len();
    let n_inst8 = ROTATION_BYTES.div_ceil(w8_inst).max(2);
    let mut packs8 = Vec::with_capacity(n_inst8);
    for _ in 0..n_inst8 {
        let mut wb = vec![0u8; w8n * w8k];
        for b in wb.iter_mut() {
            *b = (rng.next_u64() & 0x7F) as u8;
        }
        let mut ws = vec![0u8; w8n.div_ceil(128) * w8k.div_ceil(128) * 4];
        for c in ws.chunks_exact_mut(4) {
            c.copy_from_slice(&rng.uniform(0.001, 0.02).to_le_bytes());
        }
        packs8.push((upload(&gpu, &wb)?, upload(&gpu, &ws)?));
    }
    let time_w8 = |v2: bool| -> Result<f64> {
        for it in 0..8usize {
            let (p, s) = packs8[it % n_inst8];
            let mut l = KernelLaunch::new(&gpu, if v2 { v2_w8_m6 } else { w8x_h })
                .grid([(w8n.div_ceil(4)) as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(a_w8_ptr)
                .arg_ptr(p)
                .arg_ptr(s)
                .arg_ptr(c_v2_8);
            if !v2 {
                l = l.arg_u32(6);
            }
            l.arg_u32(w8n as u32)
                .arg_u32(w8k as u32)
                .arg_u32(w8k as u32)
                .arg_u32(w8n as u32)
                .launch(stream)?;
        }
        gpu.synchronize(stream)?;
        let ev = Events::new();
        unsafe { cuEventRecord(ev.0, stream) };
        for it in 0..ITERS as usize {
            let (p, s) = packs8[it % n_inst8];
            let mut l = KernelLaunch::new(&gpu, if v2 { v2_w8_m6 } else { w8x_h })
                .grid([(w8n.div_ceil(4)) as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(a_w8_ptr)
                .arg_ptr(p)
                .arg_ptr(s)
                .arg_ptr(c_v2_8);
            if !v2 {
                l = l.arg_u32(6);
            }
            l.arg_u32(w8n as u32)
                .arg_u32(w8k as u32)
                .arg_u32(w8k as u32)
                .arg_u32(w8n as u32)
                .launch(stream)?;
        }
        unsafe { cuEventRecord(ev.1, stream) };
        gpu.synchronize(stream)?;
        Ok(ev.elapsed_ms() as f64 / ITERS as f64)
    };
    let t_inc_w8 = time_w8(false)?;
    let t_v2_w8 = time_w8(true)?;
    let w8_weight_bytes = (w8n * w8k + w8n.div_ceil(128) * w8k.div_ceil(128) * 4) as f64;
    let gbs8 = |t_ms: f64| w8_weight_bytes / (t_ms / 1e3) / 1e9;
    println!(
        "  M=6 w8 batchm_exact incumbent:   {t_inc_w8:.4} ms ({:.1} GB/s)  v2: {t_v2_w8:.4} ms ({:.1} GB/s)  [{:.2}x]",
        gbs8(t_inc_w8),
        gbs8(t_v2_w8),
        t_inc_w8 / t_v2_w8
    );
    if t_v2_w4 > t_inc_w4 * 1.02 || t_v2_w8 > t_inc_w8 * 1.02 {
        bail!("GATE 6 FAIL: a V2 kernel is slower than its incumbent at M=6");
    }

    println!("PASS");
    Ok(())
}
