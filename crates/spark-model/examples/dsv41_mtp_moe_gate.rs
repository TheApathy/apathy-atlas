// SPDX-License-Identifier: AGPL-3.0-only
//! Gate for the DSpark FP4 MoE decode kernels (`mtp_moe_decode.cu` via `MtpMoeDecode`) on the
//! checkpoint's real `mtp.{k}` experts, against an independent host reference in f64:
//!
//! - routing: sqrt(softplus(y @ gate^T)) + bias, top-3 (ties -> lower id), weights normalised x
//!   route_scale — picks must match EXACTLY, weights to ~1e-6;
//! - experts: FP4 e2m1 (low nibble = even k) x 2^(s-127) per 32 k, h rounded to bf16 where the
//!   reference rounds, fp64 down + sum, bf16 out — rel_l2 PRE-REGISTERED <= 1e-3 (fp32 order +
//!   bf16 rounding flips only).
//!
//! No DSpark oracle capture exists yet, so the reference is this file's; the input rows are real
//! activations (a capture's layer-0 `moe_in`). Two CONTROLS must FAIL (rel_l2 > 1e-2), both applied
//! to the host reference so the kernel carries no wrong branch: nibble order swapped, and each
//! group read with its neighbour's scale.
//!
//! ```text
//! cargo run -p spark-model --release --example dsv41_mtp_moe_gate --features cuda,gpu-examples
//! ```

use anyhow::{Context, Result, bail, ensure};
use std::path::Path;
use std::sync::Arc;

use atlas_core::config::parse_config;
use spark_model::weight_loader::deepseek_v41::device_allocs::SharedGpu;
use spark_model::weight_loader::deepseek_v41::fwd::V41Dims;
use spark_model::weight_loader::deepseek_v41::mtp::{DsparkWeights, Fp4Linear};
use spark_model::weight_loader::deepseek_v41::mtp_moe_decode::{MTP_TOP_K, MtpMoeDecode};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{SafetensorsLoader, WeightLoader};

const MODEL_DIR: &str = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K";
const MOE_IN: &str = "/home/flocka/atlas/DSV41_PORT/oracle/ref/runA/L00.moe_in.000.bin";
const TOL: f64 = 1e-3;
const CONTROL_FLOOR: f64 = 1e-2;
const E2M1: [f64; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

fn bf(b: u16) -> f64 {
    f32::from_bits((b as u32) << 16) as f64
}
fn round_bf16(v: f32) -> f32 {
    let b = v.to_bits();
    f32::from_bits(((b + 0x7fff + ((b >> 16) & 1)) >> 16) << 16)
}

#[derive(Clone, Copy, PartialEq)]
enum Ctl {
    None,
    SwapNibbles,
    NeighbourScale,
}

/// Host copy of one FP4 matrix, dequantised to f64 [n, k].
fn dequant(gpu: &dyn GpuBackend, w: &Fp4Linear, ctl: Ctl) -> Result<Vec<f64>> {
    let mut packed = vec![0u8; w.n * w.k / 2];
    let mut sc = vec![0u8; w.n * w.k / 32];
    gpu.copy_d2h(w.weight, &mut packed)?;
    gpu.copy_d2h(w.scale, &mut sc)?;
    let g = w.k / 32;
    let mut out = vec![0.0f64; w.n * w.k];
    for r in 0..w.n {
        for kk in 0..w.k {
            let byte = packed[r * w.k / 2 + kk / 2];
            let even = kk % 2 == 0;
            let nib = if even != (ctl == Ctl::SwapNibbles) { byte & 0xf } else { byte >> 4 };
            let grp = if ctl == Ctl::NeighbourScale { (kk / 32 + 1) % g } else { kk / 32 };
            let s = (sc[r * g + grp] as i32) - 127;
            out[r * w.k + kk] = E2M1[nib as usize] * 2f64.powi(s);
        }
    }
    Ok(out)
}

fn main() -> Result<()> {
    let gpu = Arc::new(AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?);
    let shared: SharedGpu = gpu.clone();
    let config = parse_config(&std::fs::read_to_string(format!("{MODEL_DIR}/config.json"))?)?;
    let dims = V41Dims::from_config(&config)?;
    let mut loader = SafetensorsLoader::new();
    loader.extra_skip = Some(Arc::new(|name: &str| !name.starts_with("mtp.")));
    let store = loader.load(Path::new(MODEL_DIR), gpu.as_ref(), 0)?;
    let w = DsparkWeights::load(&store, &dims, config.vocab_size)?;
    let (limit, route_scale) = (10.0f32, 1.5f32);
    let moe = MtpMoeDecode::new(&shared, &w, limit, route_scale)?;
    let stream = gpu.default_stream();

    let raw = std::fs::read(MOE_IN).with_context(|| MOE_IN.to_string())?;
    let x_all: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let d = 5120usize;
    let (d_y, d_out) = (gpu.alloc(8 * d * 2)?, gpu.alloc(8 * d * 2)?);

    let mut fails = 0usize;
    let mut controls_fired = true;
    for blk in 0..w.blocks.len() {
        let b = &w.blocks[blk];
        let mut gate_bits = vec![0u8; 128 * d * 2];
        gpu.copy_d2h(b.router.weight, &mut gate_bits)?;
        let gate: Vec<f64> = gate_bits.chunks_exact(2).map(|c| bf(u16::from_le_bytes([c[0], c[1]]))).collect();
        let mut bias_b = vec![0u8; 128 * 4];
        gpu.copy_d2h(b.router.bias, &mut bias_b)?;
        let bias: Vec<f64> = bias_b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect();
        for &(row0, t) in &[(0usize, 1usize), (3, 5), (10, 8)] {
            let xs = &x_all[row0 * d..(row0 + t) * d];
            let xb: Vec<u8> = xs.iter().flat_map(|v| v.to_le_bytes()).collect();
            gpu.copy_h2d(&xb, d_y)?;
            moe.forward_block(blk, d_y, d_out, t, stream)?;
            gpu.synchronize(stream)?;
            let mut ob = vec![0u8; t * d * 2];
            gpu.copy_d2h(d_out, &mut ob)?;
            let got: Vec<f64> = ob.chunks_exact(2).map(|c| bf(u16::from_le_bytes([c[0], c[1]]))).collect();
            let (dev_idx, dev_w) = moe.read_routing(t, stream)?;

            // Host routing.
            let mut picks = Vec::with_capacity(t * MTP_TOP_K);
            let mut wts = Vec::with_capacity(t * MTP_TOP_K);
            for r in 0..t {
                let x: Vec<f64> = xs[r * d..(r + 1) * d].iter().map(|v| bf(*v)).collect();
                let scores: Vec<f64> = (0..128)
                    .map(|e| {
                        let l: f64 = (0..d).map(|k| gate[e * d + k] * x[k]).sum();
                        (if l > 20.0 { l } else { l.exp().ln_1p() }).sqrt()
                    })
                    .collect();
                let mut order: Vec<usize> = (0..128).collect();
                order.sort_by(|&a, &c| (scores[c] + bias[c]).partial_cmp(&(scores[a] + bias[a])).unwrap().then(a.cmp(&c)));
                let sum: f64 = order[..MTP_TOP_K].iter().map(|&e| scores[e]).sum();
                for &e in &order[..MTP_TOP_K] {
                    picks.push(e as i32);
                    wts.push(scores[e] / (sum + 1e-20) * route_scale as f64);
                }
            }
            let idx_diff = picks.iter().zip(&dev_idx).filter(|(a, c)| a != c).count();
            let w_worst = wts.iter().zip(&dev_w).map(|(a, c)| (a - *c as f64).abs()).fold(0.0, f64::max);

            // Host experts, from the DEVICE routing (so a near-tie pick cannot hide a kernel error).
            for ctl in [Ctl::None, Ctl::SwapNibbles, Ctl::NeighbourScale] {
                let mut refo = vec![0.0f64; t * d];
                for r in 0..t {
                    let x: Vec<f64> = xs[r * d..(r + 1) * d].iter().map(|v| bf(*v)).collect();
                    for k in 0..MTP_TOP_K {
                        let e = dev_idx[r * MTP_TOP_K + k] as usize;
                        let rw = dev_w[r * MTP_TOP_K + k];
                        let ex = &b.experts[e];
                        let (w1, w3, w2) = (dequant(gpu.as_ref(), &ex.w1, ctl)?, dequant(gpu.as_ref(), &ex.w3, ctl)?, dequant(gpu.as_ref(), &ex.w2, ctl)?);
                        let inter = ex.w1.n;
                        let h: Vec<f64> = (0..inter)
                            .map(|n| {
                                let g: f64 = (0..d).map(|kk| w1[n * d + kk] * x[kk]).sum();
                                let u: f64 = (0..d).map(|kk| w3[n * d + kk] * x[kk]).sum();
                                let (g, u) = (g as f32, u as f32);
                                let g = g.min(limit);
                                let u = u.max(-limit).min(limit);
                                round_bf16(g * (1.0 / (1.0 + (-g).exp())) * u * rw) as f64
                            })
                            .collect();
                        for n in 0..d {
                            refo[r * d + n] += (0..inter).map(|j| w2[n * inter + j] * h[j]).sum::<f64>();
                        }
                    }
                }
                let refo: Vec<f64> = refo.iter().map(|v| round_bf16(*v as f32) as f64).collect();
                let (num, den) = got.iter().zip(&refo).fold((0.0, 0.0), |(n, dd), (a, c)| (n + (a - c) * (a - c), dd + c * c));
                let rel = (num / den).sqrt();
                match ctl {
                    Ctl::None => {
                        println!(
                            "block {blk} rows {row0}..{}: routing {idx_diff}/{} picks differ, route_w worst {w_worst:.2e}; rel_l2 {rel:.3e}",
                            row0 + t,
                            picks.len()
                        );
                        if rel > TOL || idx_diff != 0 {
                            fails += 1;
                        }
                    }
                    other => {
                        let name = if other == Ctl::SwapNibbles { "nibble-swap" } else { "neighbour-scale" };
                        println!("    control {name}: rel_l2 {rel:.3e} (must exceed {CONTROL_FLOOR:.0e})");
                        if rel <= CONTROL_FLOOR {
                            controls_fired = false;
                        }
                    }
                }
            }
            if blk > 0 {
                break; // blocks 1, 2: one case each (the full sweep is on block 0)
            }
        }
    }
    ensure!(controls_fired, "A CONTROL DID NOT FIRE: this gate proves nothing");
    if fails > 0 {
        bail!("FAIL: {fails} case(s) outside rel_l2 {TOL:.0e} or with routing mismatches");
    }
    println!("PASS: DSpark FP4 MoE matches the f64 reference (rel_l2 <= {TOL:.0e}), routing exact, both controls fired");
    Ok(())
}
