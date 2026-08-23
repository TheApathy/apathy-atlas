// SPDX-License-Identifier: AGPL-3.0-only

//! Device microtest: DFlash2 `GroupedDynamicCausalConv` (prepare/finish) and
//! `CandidateSelector` greedy walk must match CPU references of the exact
//! reference math (z-lab/dflash dflash/model.py).
//!
//! Conv: `out[l,g,s] = sum_o (base[stage][o][g*GS+s] + dyn[l,stage,o,g]) * x[l-o,g,s]`
//! with x[-1] = 0 (causal pad at the block start).
//! Selector: greedy walk seeded at `anchor`, scores[k] = unary[k] +
//! sum_r (pred[prev][r]*hidden[pos][r]) * succ[cand_k][r], argmax each pos.
//!
//! Run: `cargo run --release --example dflash2_conv_selector_microtest`
//! (requires the qwen3.8-27b kernel cache with dflash2_conv/dflash2_selector).

use anyhow::{Context, Result, ensure};
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const GROUP_SIZE: usize = 16;
const KERNEL_SIZE: usize = 2;
const TOP_K: usize = 16;

// ── data helpers ────────────────────────────────────────────────────────────

fn as_le_bytes(vals: &[u16]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32_to_bf16(v: f32) -> u16 {
    // round-to-nearest-even bf16 via the hardware-convert trick on CPU:
    // add 0x7FFF rounding bias into the upper 16 bits.
    let bits = v.to_bits();
    let mut rounded = bits + 0x7FFF + ((bits >> 16) & 1);
    // NaN/inf passthrough: if exponent is all ones, keep raw high bits.
    if (bits >> 23) & 0xFF == 0xFF {
        rounded = bits;
    }
    (rounded >> 16) as u16
}

fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

fn next_f32(rng: &mut u64) -> f32 {
    // LCG step; mask to 24 bits for a uniform [0,1) sample
    *rng = rng
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*rng >> 8) & 0xFF_FFFF) as f32 / (1u32 << 24) as f32
}

fn random_bf16(rng: &mut u64, n: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let f = (next_f32(rng) - 0.5) * 4.0;
        out.push(f32_to_bf16(f));
    }
    out
}

fn upload_u16(gpu: &dyn GpuBackend, vals: &[u16]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(vals.len() * 2)?;
    gpu.copy_h2d(&as_le_bytes(vals), ptr)?;
    Ok(ptr)
}

fn read_u16(gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize, _stream: u64) -> Result<Vec<u16>> {
    let mut buf = vec![0u8; n * 2];
    gpu.copy_d2h(ptr, &mut buf)?;
    Ok(buf
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn upload_u32(gpu: &dyn GpuBackend, vals: &[u32]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(vals.len() * 4)?;
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    gpu.copy_h2d(&bytes, ptr)?;
    Ok(ptr)
}

fn read_u32(gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize, _stream: u64) -> Result<Vec<u32>> {
    let mut buf = vec![0u8; n * 4];
    gpu.copy_d2h(ptr, &mut buf)?;
    Ok(buf
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

// ── CPU references (exact reference math) ───────────────────────────────────

fn conv_ref(
    hidden: &[u16],  // [rows, groups*GS] bf16
    dynamic: &[u16], // [rows, dyn_stages*KERNEL, groups] bf16 (stage-1 exported slice when dyn_stages==1)
    base: &[u16],    // [2, KERNEL, groups*GS] bf16
    rows: usize,
    groups: usize,
    base_stage: usize, // which base[stage] slice to use (0 for prepare, 1 for finish)
    dyn_stages: usize, // number of stage slices in `dynamic` (2 for prepare, 1 for finish)
) -> Vec<u16> {
    let h = groups * GROUP_SIZE;
    let mut out = vec![0u16; rows * h];
    for l in 0..rows {
        for g in 0..groups {
            for s in 0..GROUP_SIZE {
                let mut acc = 0.0f64;
                for o in 0..KERNEL_SIZE {
                    let x = if l >= o {
                        bf16_to_f32(hidden[(l - o) * h + g * GROUP_SIZE + s]) as f64
                    } else {
                        0.0
                    };
                    let base_v =
                        bf16_to_f32(base[(base_stage * KERNEL_SIZE + o) * h + g * GROUP_SIZE + s]);
                    // The exported stage-1 slice is laid out as [KERNEL, groups] per
                    // row (the stage dimension is consumed), so the dynamic offset is
                    // just `o*groups + g` when dyn_stages == 1.
                    let dyn_off = if dyn_stages == 1 {
                        o * groups + g
                    } else {
                        (base_stage * KERNEL_SIZE + o) * groups + g
                    };
                    let dyn_v =
                        bf16_to_f32(dynamic[l * (dyn_stages * KERNEL_SIZE * groups) + dyn_off]);
                    acc += (base_v as f64 + dyn_v as f64) * x;
                }
                out[l * h + g * GROUP_SIZE + s] = f32_to_bf16(acc as f32);
            }
        }
    }
    out
}

fn selector_ref(
    unary: &[f32],         // [gamma, TOP_K] f32
    candidates: &[u32],    // [gamma, TOP_K]
    hidden_proj: &[u16],   // [gamma, rank] bf16
    pred_codebook: &[u16], // [V, rank] bf16
    succ_codebook: &[u16], // [V, rank] bf16
    gamma: usize,
    rank: usize,
    anchor: u32,
) -> (Vec<u32>, Vec<f64>) {
    let mut path = Vec::with_capacity(gamma);
    let mut margins = Vec::with_capacity(gamma);
    let mut pred = anchor as usize;
    for pos in 0..gamma {
        let mut scores = [0.0f64; TOP_K];
        for k in 0..TOP_K {
            let cand = candidates[pos * TOP_K + k] as usize;
            let mut acc = 0.0f64;
            for r in 0..rank {
                let p = bf16_to_f32(pred_codebook[pred * rank + r]) as f64;
                let h = bf16_to_f32(hidden_proj[pos * rank + r]) as f64;
                let s = bf16_to_f32(succ_codebook[cand * rank + r]) as f64;
                acc += (p * h) * s;
            }
            scores[k] = unary[pos * TOP_K + k] as f64 + acc;
        }
        let mut best = 0usize;
        for k in 1..TOP_K {
            if scores[k] > scores[best] {
                best = k;
            }
        }
        path.push(candidates[pos * TOP_K + best]);
        // top-2 margin (diagnostics for near-tie argmax flips)
        let second = (0..TOP_K)
            .filter(|&k| k != best)
            .map(|k| scores[k])
            .fold(f64::NEG_INFINITY, f64::max);
        margins.push(scores[best] - second);
        pred = candidates[pos * TOP_K + best] as usize;
    }
    (path, margins)
}

// ── tests ───────────────────────────────────────────────────────────────────

fn test_conv(
    gpu: &dyn GpuBackend,
    stream: u64,
    prepare: KernelHandle,
    finish: KernelHandle,
) -> Result<()> {
    let rows = 8usize;
    let groups = 32usize; // H = 512 — fast, exercises the same kernel path
    let h = groups * GROUP_SIZE;
    let dyn_width = 2 * KERNEL_SIZE * groups;
    let mut seed = 0x0ddb_0001u64;

    let hidden = random_bf16(&mut seed, rows * h);
    let dynamic = random_bf16(&mut seed, rows * dyn_width);
    let base = random_bf16(&mut seed, 2 * KERNEL_SIZE * h);

    let hidden_d = upload_u16(gpu, &hidden)?;
    let dynamic_d = upload_u16(gpu, &dynamic)?;
    let base_d = upload_u16(gpu, &base)?;
    let out_d = gpu.alloc(rows * h * 2)?;
    let dyn1_d = gpu.alloc(rows * KERNEL_SIZE * groups * 2)?;

    // Sentinels: fill both outputs with a recognizable pattern before launch so
    // an unwritten store is distinguishable from a zero write.
    let sentinel = vec![0xABABu16; rows * h];
    let sentinel_d = upload_u16(gpu, &sentinel)?;
    gpu.copy_d2d(sentinel_d, out_d, sentinel.len() * 2)?;
    let sentinel1 = vec![0xCDCDu16; rows * KERNEL_SIZE * groups];
    let sentinel1_d = upload_u16(gpu, &sentinel1)?;
    gpu.copy_d2d(sentinel1_d, dyn1_d, sentinel1.len() * 2)?;

    // prepare (stage 0): conv + stage-1 export
    ops::dflash2_conv_prepare(
        gpu,
        prepare,
        hidden_d,
        dynamic_d,
        base_d,
        out_d,
        dyn1_d,
        rows as u32,
        groups as u32,
        stream,
    )?;
    gpu.synchronize(stream)?;
    // stage-1 dynamic export FIRST — proves the kernel ran at all
    let dyn1 = read_u16(gpu, dyn1_d, rows * KERNEL_SIZE * groups, stream)?;
    let dyn1_ok = (0..rows * KERNEL_SIZE * groups).all(|i| {
        let l = i / (KERNEL_SIZE * groups);
        let o = (i / groups) % KERNEL_SIZE;
        let g = i % groups;
        bf16_to_f32(dyn1[i]) == bf16_to_f32(dynamic[l * dyn_width + (KERNEL_SIZE + o) * groups + g])
    });
    eprintln!(
        "DIAG: dyn1_ok={dyn1_ok} dyn1[0..3]={:?} dyn[stage1 0..3]={:?}",
        dyn1[0..3]
            .iter()
            .map(|&v| bf16_to_f32(v))
            .collect::<Vec<_>>(),
        dynamic[KERNEL_SIZE * groups..KERNEL_SIZE * groups + 3]
            .iter()
            .map(|&v| bf16_to_f32(v))
            .collect::<Vec<_>>()
    );
    let got = read_u16(gpu, out_d, rows * h, stream)?;
    let expect = conv_ref(&hidden, &dynamic, &base, rows, groups, 0, 2);
    if got[0] != expect[0] {
        eprintln!(
            "DIAG: hidden[0..2]={:?} base[0..2]={:?} dyn[0..2]={:?}",
            hidden[0..2]
                .iter()
                .map(|&v| bf16_to_f32(v))
                .collect::<Vec<_>>(),
            base[0..2]
                .iter()
                .map(|&v| bf16_to_f32(v))
                .collect::<Vec<_>>(),
            dynamic[0..2]
                .iter()
                .map(|&v| bf16_to_f32(v))
                .collect::<Vec<_>>()
        );
        eprintln!(
            "DIAG: got[0..4]={:?} expect[0..4]={:?}",
            got[0..4]
                .iter()
                .map(|&v| bf16_to_f32(v))
                .collect::<Vec<_>>(),
            expect[0..4]
                .iter()
                .map(|&v| bf16_to_f32(v))
                .collect::<Vec<_>>()
        );
    }
    for (i, (a, b)) in got.iter().zip(&expect).enumerate() {
        let rel = (bf16_to_f32(*a) - bf16_to_f32(*b)).abs() / bf16_to_f32(*b).abs().max(1e-6);
        ensure!(
            rel < 0.01,
            "conv prepare mismatch at {i}: got {} expect {}",
            bf16_to_f32(*a),
            bf16_to_f32(*b)
        );
    }

    // stage-1 dynamic export must equal the stage-1 slice
    let dyn1 = read_u16(gpu, dyn1_d, rows * KERNEL_SIZE * groups, stream)?;
    for l in 0..rows {
        for o in 0..KERNEL_SIZE {
            for g in 0..groups {
                let got_v = bf16_to_f32(dyn1[l * KERNEL_SIZE * groups + o * groups + g]);
                let exp_v = bf16_to_f32(dynamic[l * dyn_width + (KERNEL_SIZE + o) * groups + g]);
                ensure!(
                    (got_v - exp_v).abs() < 1e-6,
                    "conv dyn1 export mismatch at ({l},{o},{g}): {got_v} vs {exp_v}"
                );
            }
        }
    }

    // finish (stage 1) on the exported dyn1
    ops::dflash2_conv_finish(
        gpu,
        finish,
        hidden_d,
        dyn1_d,
        base_d,
        out_d,
        rows as u32,
        groups as u32,
        stream,
    )?;
    gpu.synchronize(stream)?;
    let got = read_u16(gpu, out_d, rows * h, stream)?;
    let expect = conv_ref(&hidden, &dyn1, &base, rows, groups, 1, 1);
    for (i, (a, b)) in got.iter().zip(&expect).enumerate() {
        let rel = (bf16_to_f32(*a) - bf16_to_f32(*b)).abs() / bf16_to_f32(*b).abs().max(1e-6);
        ensure!(
            rel < 0.01,
            "conv finish mismatch at {i}: got {} expect {}",
            bf16_to_f32(*a),
            bf16_to_f32(*b)
        );
    }
    tracing::info!("dflash2 conv prepare+finish: OK (rows={rows} groups={groups})");
    println!("dflash2 conv prepare+finish: OK (rows={rows} groups={groups})");
    Ok(())
}

fn test_selector(gpu: &dyn GpuBackend, stream: u64, walk: KernelHandle) -> Result<()> {
    let gamma = 8usize;
    let rank = 64usize;
    let vocab = 1000u32;
    let mut seed = 0x0ddb_0002u64;

    let unary: Vec<f32> = (0..gamma * TOP_K)
        .map(|_| (next_f32(&mut seed) - 0.5) * 6.0)
        .collect();
    let candidates: Vec<u32> = (0..gamma * TOP_K)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed as u32) % vocab
        })
        .collect();
    let hidden_proj = random_bf16(&mut seed, gamma * rank);
    let pred_codebook = random_bf16(&mut seed, vocab as usize * rank);
    let succ_codebook = random_bf16(&mut seed, vocab as usize * rank);
    let anchor: u32 = 42;

    let unary_d = {
        let ptr = gpu.alloc(unary.len() * 4)?;
        let bytes: Vec<u8> = unary.iter().flat_map(|v| v.to_le_bytes()).collect();
        gpu.copy_h2d(&bytes, ptr)?;
        ptr
    };
    let candidates_d = upload_u32(gpu, &candidates)?;
    let hidden_proj_d = upload_u16(gpu, &hidden_proj)?;
    let pred_d = upload_u16(gpu, &pred_codebook)?;
    let succ_d = upload_u16(gpu, &succ_codebook)?;
    let path_d = gpu.alloc(gamma * 4)?;

    ops::dflash2_selector_walk(
        gpu,
        walk,
        unary_d,
        candidates_d,
        hidden_proj_d,
        pred_d,
        succ_d,
        path_d,
        anchor,
        gamma as i32,
        rank as i32,
        stream,
    )?;
    gpu.synchronize(stream)?;
    let got = read_u32(gpu, path_d, gamma, stream)?;
    let (expect, _margins) = selector_ref(
        &unary,
        &candidates,
        &hidden_proj,
        &pred_codebook,
        &succ_codebook,
        gamma,
        rank,
        anchor,
    );
    ensure!(
        got == expect,
        "selector walk mismatch: got {got:?} expect {expect:?}"
    );
    println!("dflash2 selector walk: OK (gamma={gamma} rank={rank} vocab={vocab})");
    Ok(())
}

/// Sanity probe: a known-good kernel (argmax_bf16) must write correct output
/// in this example process — proves the launch path itself works.
fn probe_launch_path(gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
    let argmax = gpu
        .kernel("argmax", "argmax_bf16")
        .context("resolve argmax_bf16")?;
    let logits: Vec<u16> = vec![
        f32_to_bf16(0.1),
        f32_to_bf16(2.5),
        f32_to_bf16(-1.0),
        f32_to_bf16(2.5),
    ];
    let logits_d = upload_u16(gpu, &logits)?;
    let out_d = gpu.alloc(4)?;
    spark_model::layers::ops::argmax_bf16(gpu, argmax, logits_d, out_d, 4, stream)?;
    gpu.synchronize(stream)?;
    let out = read_u32(gpu, out_d, 1, stream)?;
    ensure!(
        out[0] == 1,
        "launch-path probe failed: argmax returned {:?} (expected 1)",
        out
    );
    println!("launch path probe: OK (argmax returned {})", out[0]);
    Ok(())
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())
        .context("initialize CUDA backend with compiled qwen3.8 kernels")?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    probe_launch_path(gpu, stream)?;
    let prepare = gpu
        .kernel("dflash2_conv", "dflash2_conv_prepare")
        .context("resolve dflash2_conv_prepare")?;
    let finish = gpu
        .kernel("dflash2_conv", "dflash2_conv_finish")
        .context("resolve dflash2_conv_finish")?;
    let walk = gpu
        .kernel("dflash2_selector", "dflash2_selector_walk")
        .context("resolve dflash2_selector_walk")?;
    println!(
        "handles: prepare={} finish={} walk={} argmax_probe_done",
        prepare.0, finish.0, walk.0
    );

    test_conv(gpu, stream, prepare, finish)?;
    test_selector(gpu, stream, walk)?;
    println!("ALL DFLASH2 MICROTESTS PASSED");
    Ok(())
}
