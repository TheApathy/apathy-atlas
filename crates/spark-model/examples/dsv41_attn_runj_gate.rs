// SPDX-License-Identifier: AGPL-3.0-only
//! The full-chain gate on runJ_decode (1044-token prompt = 512+512+20 encoder chunks, replay ON,
//! then 8 tapped T=1 greedy decode steps; full attention taps at L2/20/24/28/32/36).
//!
//! Teacher-forced at CoreArgs per layer and step, but the CROSS-LAYER state is the core's own:
//!   A. encoder chunks at L2 and L20 (incl. the 20-token chunk) -> own ckv/ik, candidates, tail
//!   B. decoder replay at L24/28/32/36 on the core's OWN tail (spans chunks 2+3, mixed widths)
//!   C. 8 Decode passes at L2/20/24/28/32/36 (split-KV attention, candidates from L20 of the step)
//! Compared per chunk/step: top-k rows and attention (bf16-exact share). Controls must fail.
//!
//! ```text
//! dsv41_attn_runj_gate [runJ_decode]
//! ```

use anyhow::{Context, Result, ensure};
use std::path::Path;
use std::sync::Arc;

use atlas_core::config::parse_config;
use spark_model::layers::deepseek_v41_attn::core::{Dsv41SparseCore, HEAD_DIM, REPLAY_ROWS};
use spark_model::layers::deepseek_v41_attn::index::INDEX_TOPK;
use spark_model::weight_loader::deepseek_v41::attn_block::{AttnCore, CoreArgs};
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

struct G<'a> {
    dir: String,
    gpu: &'a AtlasCudaBackend,
}

impl G<'_> {
    fn tap(&self, l: usize, n: &str, occ: usize) -> Result<Vec<u8>> {
        let p = format!("{}/L{l:02}.{n}.{occ:03}.bin", self.dir);
        std::fs::read(&p).with_context(|| p)
    }
    fn up(&self, b: &[u8]) -> Result<DevicePtr> {
        let p = self.gpu.alloc(b.len().max(8))?;
        self.gpu.copy_h2d(b, p)?;
        Ok(p)
    }
    fn down(&self, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
        self.gpu.synchronize(self.gpu.default_stream())?;
        let mut v = vec![0u8; n];
        self.gpu.copy_d2h(p, &mut v)?;
        Ok(v)
    }
    /// CoreArgs for (layer, occurrence) straight from the capture. `wpos_shift` is the control.
    fn args(&self, l: usize, occ: usize, out: DevicePtr, wpos_shift: i64) -> Result<(CoreArgs, Vec<DevicePtr>)> {
        let q = self.tap(l, "q", occ)?;
        let t = q.len() / QROW;
        let wpos: Vec<i32> = i64s(&self.tap(l, "wpos", occ)?)
            .iter()
            .map(|&p| if p < 0 { -1 } else { (p - wpos_shift).max(-1) as i32 })
            .collect();
        let start = i64s(&self.tap(l, "wpos", occ)?)[127] as usize; // newest slot of row 0 = its position
        let win_lo = i64s(&self.tap(l, "win_lo", occ)?)[0] as usize;
        let bufs = vec![
            self.up(&self.tap(l, "attn_x", occ)?)?,
            self.up(&self.tap(l, "qr", occ)?)?,
            self.up(&q)?,
            self.up(&self.tap(l, "ring", occ)?)?,
            self.up(bytemuck_i32(&wpos))?,
            self.up(&self.tap(l, "attn_sink", occ)?)?,
        ];
        let a = CoreArgs { layer: l, x: bufs[0], qr: bufs[1], q: bufs[2], ring: bufs[3], wpos: bufs[4], win_lo, sink: bufs[5], t, start, out };
        Ok((a, bufs))
    }
}

/// (rows whose top-k differs, attention bf16-exact share, rel_l2) for one core call vs capture.
fn score(g: &G, core: &Dsv41SparseCore, l: usize, occ: usize, t: usize, out: DevicePtr) -> Result<(usize, f64, f64)> {
    let tk = i64s(&g.down(core.current_topk().context("no topk")?, t * INDEX_TOPK * 8)?);
    let want = i64s(&g.tap(l, "topk", occ)?);
    let bad = (0..t).filter(|r| tk[r * INDEX_TOPK..(r + 1) * INDEX_TOPK] != want[r * INDEX_TOPK..(r + 1) * INDEX_TOPK]).count();
    let o = u16s(&g.down(out, t * QROW)?);
    let w = u16s(&g.tap(l, "attn_o_pre_inverse_rope", occ)?);
    let ex = o.iter().zip(&w).filter(|(a, b)| a == b).count() as f64 / w.len() as f64;
    let (mut n, mut d) = (0.0, 0.0);
    for (x, y) in o.iter().zip(&w) {
        n += (bf(*x) - bf(*y)).powi(2);
        d += bf(*y).powi(2);
    }
    Ok((bad, ex, (n / d).sqrt()))
}

fn free(g: &G, bufs: Vec<DevicePtr>) -> Result<()> {
    for b in bufs {
        g.gpu.free(b)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let run = std::env::args().nth(1).unwrap_or_else(|| "runJ_decode".into());
    let config = parse_config(&std::fs::read_to_string(format!("{MODEL_DIR}/config.json"))?)?;
    let gpu = Arc::new(AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?);
    let g = G { dir: format!("{REF_ROOT}/{run}"), gpu: gpu.as_ref() };
    let kernels = Dsv41Kernels::load(gpu.as_ref())?;
    let ops = Ops { gpu: gpu.as_ref(), k: &kernels, stream: gpu.default_stream() };
    let mut loader = SafetensorsLoader::new();
    loader.extra_skip = Some(Arc::new(|name: &str| {
        let keep = [2, 8, 14, 20].iter().any(|l| name.starts_with(&format!("layers.{l}.attn.")))
            || [24, 28, 32, 36].iter().any(|l| name.starts_with(&format!("layers.{l}.attn.indexer.")));
        !keep
    }));
    let store = loader.load(Path::new(MODEL_DIR), gpu.as_ref(), 0)?;
    let spec = RopeSpec { dim: 64, original_seq_len: 65536, base: 160000.0, factor: 16.0, beta_fast: 32.0, beta_slow: 1.0 };
    let core = Dsv41SparseCore::load_prefix(&(gpu.clone() as spark_model::weight_loader::deepseek_v41::device_allocs::SharedGpu), &store, &config, MAX_SEQ, 512, spec.upload(gpu.as_ref(), MAX_SEQ + 8)?, 37)?;
    let out = gpu.alloc(512 * QROW)?;
    let mut fail = 0usize;

    // ---- A. encoder: 3 chunks (512, 512, 20) at L2 and L20
    let chunks = [(0usize, 512usize), (512, 512), (1024, 20)];
    for (c, &(s, t)) in chunks.iter().enumerate() {
        core.begin_pass(PassKind::EncoderChunk, s, t)?;
        for l in [2usize, 20] {
            let (a, bufs) = g.args(l, c, out, 0)?;
            ensure!(a.start == s && a.t == t, "L{l} chunk {c}: capture says ({}, {}), expected ({s}, {t})", a.start, a.t);
            core.run(&ops, &a)?;
            let (bad, ex, rl) = score(&g, &core, l, c, t, out)?;
            println!("ENCODER L{l:02} chunk {c} (S={s}, T={t}): topk {bad}/{t} rows differ; attention bit-exact {ex:.4} rel {rl:.2e}");
            fail += usize::from(bad * 50 > t.max(50) || ex < 0.98);
            free(&g, bufs)?;
        }
    }
    // the tail the replay will install: compare with what the capture's L24 replay received
    {
        let (tk, cand, ld, rows) = core.replay_tail();
        let want = g.tap(24, "cand_in", 0)?;
        let wld = want.len() / REPLAY_ROWS;
        let got = g.down(cand, rows * ld)?;
        let mut diff = 0usize;
        for r in 0..REPLAY_ROWS {
            for c in 0..wld {
                let gv = if c < ld { got[r * ld + c] != 0 } else { false };
                diff += usize::from(gv != (want[r * wld + c] != 0));
            }
        }
        let tk_rows = i64s(&g.down(tk, REPLAY_ROWS * INDEX_TOPK * 8)?);
        let l20_c2 = i64s(&g.tap(20, "topk", 1)?);
        let l20_c3 = i64s(&g.tap(20, "topk", 2)?);
        let want_tk: Vec<i64> = l20_c2[(512 - 108) * INDEX_TOPK..].iter().chain(l20_c3.iter()).copied().collect();
        let tk_bad = (0..REPLAY_ROWS).filter(|r| tk_rows[r * INDEX_TOPK..(r + 1) * INDEX_TOPK] != want_tk[r * INDEX_TOPK..(r + 1) * INDEX_TOPK]).count();
        println!("REPLAY TAIL (108 rows of chunk 2 + 20 of chunk 3, widths 1024/1536): {rows} rows; candidates differ {diff}/{} vs L24.cand_in; topk {tk_bad}/128 rows vs L20's chunk rows", REPLAY_ROWS * wld);
        fail += usize::from(rows != REPLAY_ROWS || diff != 0 || tk_bad > 1);
    }

    // ---- B. replay at L24/28/32/36 on the core's OWN tail
    let replay_start = 1044 - REPLAY_ROWS;
    core.begin_pass(PassKind::Replay, replay_start, REPLAY_ROWS)?;
    for l in [24usize, 28, 32, 36] {
        let (a, bufs) = g.args(l, 0, out, 0)?;
        ensure!(a.start == replay_start && a.win_lo == replay_start, "L{l} replay start/win_lo {} {}", a.start, a.win_lo);
        core.run(&ops, &a)?;
        let (bad, ex, rl) = score(&g, &core, l, 0, REPLAY_ROWS, out)?;
        println!("REPLAY L{l} (own tail): topk {bad}/128 rows differ; attention bit-exact {ex:.4} rel {rl:.2e}");
        fail += usize::from(bad > 3 || ex < 0.98);
        free(&g, bufs)?;
    }

    // ---- C. 8 decode steps, all six layers, each step one Decode pass
    let steps = 8;
    let mut worst = (0usize, 1.0f64);
    for arm in [0i64, 1] {
        core.rollback(gpu.as_ref(), 1044, gpu.default_stream())?; // each arm resumes at the prompt end
        let mut ctrl_exact = Vec::new();
        for st in 0..steps {
            let p = 1044 + st;
            core.begin_pass(PassKind::Decode, p, 1)?;
            for l in [2usize, 20, 24, 28, 32, 36] {
                let occ = if l <= 20 { 3 + st } else { 1 + st };
                let (a, bufs) = g.args(l, occ, out, arm)?;
                ensure!(a.start == p && a.t == 1, "L{l} step {st}: capture row at {} (T={})", a.start, a.t);
                core.run(&ops, &a)?;
                let (bad, ex, rl) = score(&g, &core, l, occ, 1, out)?;
                if arm == 0 {
                    println!("DECODE step {st} (pos {p}) L{l:02}: topk {bad}/1 differ; attention bit-exact {ex:.4} rel {rl:.2e} (win_lo {})", a.win_lo);
                    worst = (worst.0 + bad, worst.1.min(ex));
                    fail += usize::from(ex < 0.98);
                } else {
                    ctrl_exact.push(ex);
                }
                free(&g, bufs)?;
            }
        }
        if arm == 1 {
            let mx = ctrl_exact.iter().cloned().fold(0.0f64, f64::max);
            println!("CTRL DECODE window positions shifted by one: best attention bit-exact over {} layer-steps {mx:.4} (must collapse)", ctrl_exact.len());
            fail += usize::from(mx > 0.5);
        }
    }
    println!("DECODE total: {} top-k rows differ over {} layer-steps (band: at most 1 per step, at ties); worst attention {:.4}", worst.0, steps * 6, worst.1);
    fail += usize::from(worst.0 > steps);

    ensure!(fail == 0, "RUNJ GATE FAIL ({fail})");
    println!("RUNJ GATE PASS");
    Ok(())
}
