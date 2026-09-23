// SPDX-License-Identifier: AGPL-3.0-only
//! Replay-indexer gate: the core's DECODER REPLAY at the index layers (24, 28) against a
//! capture that holds only replay layers (runI_plus20: 1044 tokens = 512+512+20 encoder chunks,
//! replay over the last 128 rows, which span chunks 2 and 3).
//!
//! Teacher-forced: L20's compressed cache and the replay tail's candidate mask are SEEDED from
//! the capture (the L24 `ckv`/`ik`/`cand_in` taps ARE L20's cache and tail); `attn_x`, `qr`,
//! `q`, `ring` per layer come from the capture. The core computes the indexer (q/wts
//! projections, score inside the candidate pool, top-k) and the sparse attention.
//!
//! ```text
//! dsv41_attn_replay_gate [runI_plus20]
//! ```

use anyhow::{Context, Result, ensure};
use std::path::Path;
use std::sync::Arc;

use atlas_core::config::parse_config;
use spark_model::layers::deepseek_v41_attn::core::{Dsv41SparseCore, HEAD_DIM, REPLAY_ROWS};
use spark_model::layers::deepseek_v41_attn::index::INDEX_TOPK;
use spark_model::weight_loader::deepseek_v41::attn_block::{AttnCore, CoreArgs, window_positions};
use spark_model::weight_loader::deepseek_v41::forward::{PassHook, PassKind};
use spark_model::weight_loader::deepseek_v41::ops::{Dsv41Kernels, Ops, RopeSpec, bytemuck_i32};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{SafetensorsLoader, WeightLoader};

const MODEL_DIR: &str = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K";
const REF_ROOT: &str = "/home/flocka/atlas/DSV41_PORT/oracle/ref";
const MAX_SEQ: usize = 8192;
const QROW: usize = 64 * HEAD_DIM * 2;

fn i64s(b: &[u8]) -> Vec<i64> {
    b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
}
fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}
fn bf(u: u16) -> f64 {
    f32::from_bits((u as u32) << 16) as f64
}

fn main() -> Result<()> {
    let run = std::env::args().nth(1).unwrap_or_else(|| "runI_plus20".into());
    let dir = format!("{REF_ROOT}/{run}");
    let tap = |l: usize, n: &str| -> Result<Vec<u8>> {
        let p = format!("{dir}/L{l:02}.{n}.000.bin");
        std::fs::read(&p).with_context(|| p)
    };
    let config = parse_config(&std::fs::read_to_string(format!("{MODEL_DIR}/config.json"))?)?;
    let gpu = Arc::new(AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?);
    let g = gpu.as_ref();
    let kernels = Dsv41Kernels::load(g)?;
    let ops = Ops { gpu: g, k: &kernels, stream: g.default_stream() };
    let up = |b: &[u8]| -> Result<DevicePtr> {
        let p = g.alloc(b.len().max(8))?;
        g.copy_h2d(b, p)?;
        Ok(p)
    };
    let down = |p: DevicePtr, n: usize| -> Result<Vec<u8>> {
        g.synchronize(g.default_stream())?;
        let mut v = vec![0u8; n];
        g.copy_d2h(p, &mut v)?;
        Ok(v)
    };

    let mut loader = SafetensorsLoader::new();
    loader.extra_skip = Some(Arc::new(|name: &str| {
        let keep = [2, 8, 14, 20].iter().any(|l| name.starts_with(&format!("layers.{l}.attn.")))
            || [24, 28].iter().any(|l| name.starts_with(&format!("layers.{l}.attn.indexer.")));
        !keep
    }));
    let store = loader.load(Path::new(MODEL_DIR), g, 0)?;
    let spec = RopeSpec { dim: 64, original_seq_len: 65536, base: 160000.0, factor: 16.0, beta_fast: 32.0, beta_slow: 1.0 };
    let core = Dsv41SparseCore::load_prefix(&(gpu.clone() as spark_model::weight_loader::deepseek_v41::device_allocs::SharedGpu), &store, &config, MAX_SEQ, 512, spec.upload(g, MAX_SEQ + 8)?, 29)?;

    // Seed L20's cache + the replay tail from L24's taps (they ARE L20's cache and tail).
    let ckv = tap(24, "ckv")?;
    let rows = ckv.len() / (HEAD_DIM * 2);
    let cand_in = tap(24, "cand_in")?;
    let ld = cand_in.len() / REPLAY_ROWS;
    let start = rows - REPLAY_ROWS; // ratio 1: rows == prompt length
    println!("run {run}: prompt {rows} tokens, replay rows {start}..{rows}, candidate width {ld}");
    core.gate_seed_replay(g, up(&ckv)?, up(&tap(24, "ik")?)?, rows, up(&cand_in)?, ld, REPLAY_ROWS)?;
    ensure!(tap(28, "cand_in")? == cand_in, "L24 and L28 must see the same replay candidate pool");

    let wpos = up(bytemuck_i32(&window_positions(start, REPLAY_ROWS)))?;
    let out = g.alloc(REPLAY_ROWS * QROW)?;
    let mut fail = 0usize;
    // At 1044 tokens every visible block survives L20's top-2048, so the pool equals
    // visibility and dropping it CANNOT change the selection (0 rows, recorded in SPEC 8e).
    // The control that can fail: index keys shifted by one row (a j0 off-by-one).
    for arm in ["real", "CTRL ik shifted 1 row"] {
        if arm != "real" {
            let ik = tap(24, "ik")?;
            let mut shifted = ik[128 * 2..].to_vec();
            shifted.extend(vec![0u8; 128 * 2]);
            core.gate_seed_replay(g, up(&ckv)?, up(&shifted)?, rows, up(&cand_in)?, ld, REPLAY_ROWS)?;
        }
        core.begin_pass(PassKind::Replay, start, REPLAY_ROWS)?;
        for l in [24usize, 28] {
            let win_lo = i64s(&tap(l, "win_lo")?)[0] as usize;
            ensure!(win_lo == start, "L{l} win_lo {win_lo} != {start}");
            let a = CoreArgs {
                layer: l,
                x: up(&tap(l, "attn_x")?)?,
                qr: up(&tap(l, "qr")?)?,
                q: up(&tap(l, "q")?)?,
                ring: up(&tap(l, "ring")?)?,
                wpos,
                win_lo,
                sink: up(&tap(l, "attn_sink")?)?,
                t: REPLAY_ROWS,
                start,
                out,
            };
            core.run(&ops, &a)?;
            let got = i64s(&down(core.current_topk().unwrap(), REPLAY_ROWS * INDEX_TOPK * 8)?);
            let want = i64s(&tap(l, "topk")?);
            let mut bad = 0;
            let mut worst = 1.0f64;
            for r in 0..REPLAY_ROWS {
                let (gr, wr) = (&got[r * INDEX_TOPK..(r + 1) * INDEX_TOPK], &want[r * INDEX_TOPK..(r + 1) * INDEX_TOPK]);
                if gr != wr {
                    bad += 1;
                    let ws: std::collections::HashSet<_> = wr.iter().filter(|&&v| v >= 0).collect();
                    worst = worst.min(gr.iter().filter(|&&v| v >= 0 && ws.contains(&v)).count() as f64 / ws.len().max(1) as f64);
                }
            }
            let o = u16s(&down(out, REPLAY_ROWS * QROW)?);
            let w = u16s(&tap(l, "attn_o_pre_inverse_rope")?);
            let exact = o.iter().zip(&w).filter(|(a, b)| a == b).count() as f64 / w.len() as f64;
            let (mut n, mut d) = (0.0, 0.0);
            for (x, y) in o.iter().zip(&w) {
                n += (bf(*x) - bf(*y)).powi(2);
                d += bf(*y).powi(2);
            }
            println!("{arm:22} L{l}: topk {bad}/{REPLAY_ROWS} rows differ (worst overlap {worst:.4}); attention bit-exact {exact:.4} rel {:.2e}", (n / d).sqrt());
            if arm == "real" {
                fail += usize::from(bad > 3 || exact < 0.98);
            } else {
                fail += usize::from(bad == 0);
            }
        }
    }
    ensure!(fail == 0, "REPLAY GATE FAIL ({fail})");
    println!("REPLAY GATE PASS");
    Ok(())
}
