// SPDX-License-Identifier: AGPL-3.0-only
//! Gate for the SHAPE-STATIC decode core (`ATLAS_DSV41_CORE_STATIC`, the attention lane's part of
//! the whole-step CUDA-graph contract): Decode/Verify passes whose launches read the pass start
//! from device memory, so one captured graph replays at any position.
//!
//! State: runJ_decode's 1044-token prompt (encoder chunks at L2/L20, replay at L24..36), then
//! decode at L2/20/24/28/32/36 from the capture's per-step inputs. Every arm starts from the same
//! state (rollback to 1044) and records, per layer and pass, the top-k and the attention output;
//! after each arm, the L2/L20 compressed caches up to the final length.
//!
//!   D  dynamic (today's path)                       -- the reference
//!   S  static, the core writing its own device start -- must equal D BYTE FOR BYTE
//!   G  static, ONE graph captured at step 0 and REPLAYED for steps 1..7 with the start set
//!      from the host (memset_u32) and `replay_step` bookkeeping -- must equal D byte for byte
//!   V  Verify T=3/T=2/T=5 + Decode with rollbacks between (both parities, odd and even T),
//!      dynamic vs static -- must be byte-identical
//!   CONTROL: G with the device start frozen at step 0 -- must differ from D
//!   F  the ratio-2 compressor's small-M fp32 GEMV vs the 64x64 GEMM it replaces at M <= 16:
//!      unit A/B at M = 1/2/6/8/16 (bytes + time; control: one flipped input must differ), and
//!      D0 = D with the GEMV off, which must equal D byte for byte
//!
//! ```text
//! dsv41_core_static_gate [runJ_decode]
//! ```

use anyhow::{Context, Result, ensure};
use std::path::Path;
use std::sync::Arc;

use atlas_core::config::parse_config;
use spark_model::layers::deepseek_v41_attn::core::{Dsv41SparseCore, HEAD_DIM, REPLAY_ROWS};
use spark_model::layers::deepseek_v41_attn::index::{GEMM_F32_V1, GEMV_F32_OFF, INDEX_HEAD_DIM, INDEX_TOPK, IndexKernels, IndexOps};
use std::sync::atomic::Ordering;
use spark_model::weight_loader::deepseek_v41::attn_block::{AttnCore, CoreArgs};
use spark_model::weight_loader::deepseek_v41::device_allocs::SharedGpu;
use spark_model::weight_loader::deepseek_v41::forward::{PassHook, PassKind};
use spark_model::weight_loader::deepseek_v41::ops::{Dsv41Kernels, Ops, RopeSpec, bytemuck_i32};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{SafetensorsLoader, WeightLoader};

const MODEL_DIR: &str = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K";
const REF_ROOT: &str = "/home/flocka/atlas/DSV41_PORT/oracle/ref";
/// `ATLAS_GATE_MAX_SEQ` (e.g. 1048576) runs the same gate on the long-context layout: score row
/// blocks, the full-width static decode score, 1M-row caches. The results must not change.
fn max_seq() -> usize {
    std::env::var("ATLAS_GATE_MAX_SEQ").ok().and_then(|v| v.parse().ok()).unwrap_or(8192)
}
const QROW: usize = 64 * HEAD_DIM * 2;
const PROMPT: usize = 1044;
const STEPS: usize = 8;
const DEC_LAYERS: [usize; 6] = [2, 20, 24, 28, 32, 36];

fn i64s(b: &[u8]) -> Vec<i64> {
    b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
}

struct G<'a> {
    dir: String,
    gpu: &'a AtlasCudaBackend,
    stream: u64,
}

/// One pass's host-side inputs for one layer (rows concatenated for T > 1).
struct Inputs {
    x: Vec<u8>,
    qr: Vec<u8>,
    q: Vec<u8>,
    ring: Vec<u8>,
    wpos: Vec<u8>,
    sink: Vec<u8>,
    win_lo: usize,
    start: usize,
    t: usize,
}

impl G<'_> {
    fn tap(&self, l: usize, n: &str, occ: usize) -> Result<Vec<u8>> {
        let p = format!("{}/L{l:02}.{n}.{occ:03}.bin", self.dir);
        std::fs::read(&p).with_context(|| p)
    }
    fn down(&self, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
        self.gpu.synchronize(self.stream)?;
        let mut v = vec![0u8; n];
        self.gpu.copy_d2h(p, &mut v)?;
        Ok(v)
    }
    /// Occurrence `occ` of layer `l` as it was captured (encoder chunk, replay or decode step).
    fn inputs(&self, l: usize, occ: usize) -> Result<Inputs> {
        let q = self.tap(l, "q", occ)?;
        let wpos64 = i64s(&self.tap(l, "wpos", occ)?);
        let wpos: Vec<i32> = wpos64.iter().map(|&p| if p < 0 { -1 } else { p as i32 }).collect();
        Ok(Inputs {
            x: self.tap(l, "attn_x", occ)?,
            qr: self.tap(l, "qr", occ)?,
            t: q.len() / QROW,
            q,
            ring: self.tap(l, "ring", occ)?,
            wpos: bytemuck_i32(&wpos).to_vec(),
            sink: self.tap(l, "attn_sink", occ)?,
            win_lo: i64s(&self.tap(l, "win_lo", occ)?)[0] as usize,
            start: wpos64[127] as usize, // newest slot of row 0 = its position
        })
    }
    /// Decode steps `s0..s0+t` of layer `l` concatenated into one T = t pass (the ring of the last
    /// step). Not a pass the model ran -- the D-vs-S comparison only needs both arms fed alike.
    fn steps(&self, l: usize, s0: usize, t: usize) -> Result<Inputs> {
        let occ = |s: usize| if l <= 20 { 3 + s } else { 1 + s };
        let mut a = self.inputs(l, occ(s0))?;
        for s in s0 + 1..s0 + t {
            let b = self.inputs(l, occ(s))?;
            a.x.extend(b.x);
            a.qr.extend(b.qr);
            a.q.extend(b.q);
            a.wpos.extend(b.wpos);
            a.ring = b.ring;
        }
        a.t = t;
        Ok(a)
    }
}

/// Fixed device buffers for one layer's pass inputs: the same pointers every step (a graph
/// bakes them in). Sized for `t_max` rows.
struct Slots {
    x: DevicePtr,
    qr: DevicePtr,
    q: DevicePtr,
    ring: DevicePtr,
    wpos: DevicePtr,
    sink: DevicePtr,
    out: DevicePtr,
}

impl Slots {
    fn new(gpu: &AtlasCudaBackend, proto: &Inputs, t_max: usize) -> Result<Self> {
        let per = |n: usize| n / proto.t * t_max;
        Ok(Self {
            x: gpu.alloc(per(proto.x.len()))?,
            qr: gpu.alloc(per(proto.qr.len()))?,
            q: gpu.alloc(per(proto.q.len()))?,
            ring: gpu.alloc(proto.ring.len())?,
            wpos: gpu.alloc(per(proto.wpos.len()))?,
            sink: gpu.alloc(proto.sink.len().max(8))?,
            out: gpu.alloc(t_max * QROW)?,
        })
    }
    fn fill(&self, gpu: &AtlasCudaBackend, stream: u64, i: &Inputs) -> Result<()> {
        gpu.synchronize(stream)?; // a replay still reading these must finish first
        gpu.copy_h2d(&i.x, self.x)?;
        gpu.copy_h2d(&i.qr, self.qr)?;
        gpu.copy_h2d(&i.q, self.q)?;
        gpu.copy_h2d(&i.ring, self.ring)?;
        gpu.copy_h2d(&i.wpos, self.wpos)?;
        gpu.copy_h2d(&i.sink, self.sink)?;
        Ok(())
    }
    fn args(&self, l: usize, i: &Inputs) -> CoreArgs {
        CoreArgs { layer: l, x: self.x, qr: self.qr, q: self.q, ring: self.ring, wpos: self.wpos, win_lo: i.win_lo, sink: self.sink, t: i.t, start: i.start, out: self.out }
    }
}

/// Per (pass, layer): top-k rows and attention output bytes.
type Record = Vec<(String, Vec<u8>, Vec<u8>)>;

struct Rig<'a> {
    g: &'a G<'a>,
    core: &'a Dsv41SparseCore,
    ops: &'a Ops<'a>,
    slots: Vec<(usize, Slots)>,
}

impl Rig<'_> {
    fn slot(&self, l: usize) -> &Slots {
        &self.slots.iter().find(|(x, _)| *x == l).expect("slot").1
    }
    /// Run one pass over DEC_LAYERS directly (no graph) and record it.
    fn pass(&self, kind: PassKind, s0: usize, t: usize, rec: &mut Record) -> Result<()> {
        let start = PROMPT + s0;
        self.core.begin_pass(kind, start, t)?;
        for l in DEC_LAYERS {
            let i = self.g.steps(l, s0, t)?;
            ensure!(i.start == start, "L{l} step {s0}: capture at {}", i.start);
            let s = self.slot(l);
            s.fill(self.g.gpu, self.g.stream, &i)?;
            self.core.run(self.ops, &s.args(l, &i))?;
            let tk = self.g.down(self.core.current_topk().context("no topk")?, t * INDEX_TOPK * 8)?;
            rec.push((format!("{kind:?} s{s0} T{t} L{l:02}"), tk, self.g.down(s.out, t * QROW)?));
        }
        Ok(())
    }
    /// L2 (ratio 2) and L20 (ratio 1) compressed caches over the first `len` positions.
    fn caches(&self, len: usize) -> Result<Vec<u8>> {
        let mut v = Vec::new();
        for (l, r) in [(2usize, 2usize), (20, 1)] {
            let (ckv, ik) = self.core.caches(l).context("no caches")?;
            v.extend(self.g.down(ckv, len / r * HEAD_DIM * 2)?);
            v.extend(self.g.down(ik, len / r * INDEX_HEAD_DIM * 2)?);
        }
        Ok(v)
    }
}

/// (passes whose top-k differ, passes whose attention differs, first differing pass).
fn diff(a: &Record, b: &Record) -> (usize, usize, String) {
    let mut first = String::from("-");
    let (mut dt, mut dout) = (0, 0);
    for (x, y) in a.iter().zip(b) {
        let (ft, fo) = (x.1 != y.1, x.2 != y.2);
        dt += usize::from(ft);
        dout += usize::from(fo);
        if (ft || fo) && first == "-" {
            first = x.0.clone();
        }
    }
    (dt, dout, first)
}

fn main() -> Result<()> {
    let run = std::env::args().nth(1).unwrap_or_else(|| "runJ_decode".into());
    let config = parse_config(&std::fs::read_to_string(format!("{MODEL_DIR}/config.json"))?)?;
    let gpu = Arc::new(AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?);
    let stream = gpu.create_stream()?; // capture needs a non-legacy stream
    let g = G { dir: format!("{REF_ROOT}/{run}"), gpu: gpu.as_ref(), stream };
    let kernels = Dsv41Kernels::load(gpu.as_ref())?;
    let ops = Ops { gpu: gpu.as_ref(), k: &kernels, stream };
    let mut loader = SafetensorsLoader::new();
    loader.extra_skip = Some(Arc::new(|name: &str| {
        let keep = [2, 8, 14, 20].iter().any(|l| name.starts_with(&format!("layers.{l}.attn.")))
            || [24, 28, 32, 36].iter().any(|l| name.starts_with(&format!("layers.{l}.attn.indexer.")));
        !keep
    }));
    let store = loader.load(Path::new(MODEL_DIR), gpu.as_ref(), 0)?;
    let spec = RopeSpec { dim: 64, original_seq_len: 65536, base: 160000.0, factor: 16.0, beta_fast: 32.0, beta_slow: 1.0 };
    let core = Dsv41SparseCore::load_prefix(&(gpu.clone() as SharedGpu), &store, &config, max_seq(), 512, spec.upload(gpu.as_ref(), max_seq() + 8)?, 37)?;
    core.set_static_decode(false);

    // ---- prompt state: encoder chunks at L2/L20, replay at L24..36 (all dynamic)
    let big = |l: usize| -> Result<Slots> { Slots::new(gpu.as_ref(), &g.inputs(l, 0)?, 512) };
    let enc = [(2usize, big(2)?), (20, big(20)?)];
    for (c, &(s, t)) in [(0usize, 512usize), (512, 512), (PROMPT - 20, 20)].iter().enumerate() {
        core.begin_pass(PassKind::EncoderChunk, s, t)?;
        for (l, slot) in &enc {
            let i = g.inputs(*l, c)?;
            ensure!(i.start == s && i.t == t, "L{l} chunk {c}: ({}, {})", i.start, i.t);
            slot.fill(gpu.as_ref(), stream, &i)?;
            core.run(&ops, &slot.args(*l, &i))?;
        }
    }
    core.begin_pass(PassKind::Replay, PROMPT - REPLAY_ROWS, REPLAY_ROWS)?;
    for l in [24usize, 28, 32, 36] {
        let i = g.inputs(l, 0)?;
        let slot = Slots::new(gpu.as_ref(), &i, REPLAY_ROWS)?;
        slot.fill(gpu.as_ref(), stream, &i)?;
        core.run(&ops, &slot.args(l, &i))?;
    }
    gpu.synchronize(stream)?;
    println!("prompt state built: {PROMPT} positions");

    let slots = DEC_LAYERS.iter().map(|&l| Ok((l, Slots::new(gpu.as_ref(), &g.steps(l, 0, 1)?, 8)?))).collect::<Result<Vec<_>>>()?;
    let rig = Rig { g: &g, core: &core, ops: &ops, slots };
    let end = PROMPT + STEPS;
    let mut fail = 0usize;

    // ---- D and S: 8 Decode steps, direct
    let arm = |on: bool| -> Result<(Record, Vec<u8>)> {
        core.set_static_decode(on);
        core.set_device_start(None);
        core.rollback(gpu.as_ref(), PROMPT, stream)?;
        let mut rec = Record::new();
        for s in 0..STEPS {
            rig.pass(PassKind::Decode, s, 1, &mut rec)?;
        }
        Ok((rec, rig.caches(end)?))
    };
    let (d, d_c) = arm(false)?;
    GEMV_F32_OFF.store(true, Ordering::Relaxed);
    let (d0, d0_c) = arm(false)?;
    GEMV_F32_OFF.store(false, Ordering::Relaxed);
    let (dt, dout, first) = diff(&d, &d0);
    println!("D0 dynamic with the fp32 GEMV OFF vs ON: top-k differs on {dt}/{n}, attention on {dout}/{n} (first: {first}); caches identical: {}", d0_c == d_c, n = d.len());
    fail += usize::from(dt + dout > 0 || d0_c != d_c);
    let (s, s_c) = arm(true)?;
    let (dt, dout, first) = diff(&d, &s);
    println!("S  static, own device start: top-k differs on {dt}/{n} layer-passes, attention on {dout}/{n} (first: {first}); caches identical: {}", s_c == d_c, n = d.len());
    fail += usize::from(dt + dout > 0 || s_c != d_c);

    // ---- G: one graph captured at step 0, replayed for steps 1..7 (and CONTROL: start frozen)
    let dstart = gpu.alloc(256)?;
    let graph_arm = |frozen: bool| -> Result<(Record, Vec<u8>)> {
        core.set_static_decode(true);
        core.set_device_start(Some(dstart));
        core.rollback(gpu.as_ref(), PROMPT, stream)?;
        let mut rec = Record::new();
        let mut topk = Vec::new();
        for (l, sl) in &rig.slots {
            sl.fill(gpu.as_ref(), stream, &g.steps(*l, 0, 1)?)?;
        }
        gpu.memset_u32_async(dstart, PROMPT as u32, 1, stream)?;
        gpu.synchronize(stream)?;
        core.begin_pass(PassKind::Decode, PROMPT, 1)?;
        gpu.begin_capture(stream)?;
        for (l, sl) in &rig.slots {
            let i = g.steps(*l, 0, 1)?;
            core.run(&ops, &sl.args(*l, &i))?;
            topk.push(core.current_topk().context("no topk")?);
        }
        let graph = gpu.end_capture(stream)?;
        for st in 0..STEPS {
            let p = PROMPT + st;
            if st > 0 {
                for (l, sl) in &rig.slots {
                    sl.fill(gpu.as_ref(), stream, &g.steps(*l, st, 1)?)?;
                }
                if !frozen {
                    gpu.memset_u32_async(dstart, p as u32, 1, stream)?;
                }
                core.replay_step(PassKind::Decode, p, 1)?;
            }
            gpu.launch_graph(graph, stream)?;
            for (k, (l, sl)) in rig.slots.iter().enumerate() {
                rec.push((format!("Decode s{st} T1 L{l:02}"), g.down(topk[k], INDEX_TOPK * 8)?, g.down(sl.out, QROW)?));
            }
        }
        gpu.synchronize(stream)?;
        gpu.destroy_graph(graph)?;
        Ok((rec, rig.caches(end)?))
    };
    let (gr, g_c) = graph_arm(false)?;
    let (dt, dout, first) = diff(&d, &gr);
    println!("G  one graph, replayed 7x at moving starts: top-k differs on {dt}/{n}, attention on {dout}/{n} (first: {first}); caches identical: {}", g_c == d_c, n = d.len());
    fail += usize::from(dt + dout > 0 || g_c != d_c);
    let (ctl, ctl_c) = graph_arm(true)?;
    let (dt, dout, first) = diff(&d, &ctl);
    println!("CONTROL graph with the start FROZEN at {PROMPT}: top-k differs on {dt}/{n}, attention on {dout}/{n} (first: {first}); caches identical: {} (must differ)", ctl_c == d_c, n = d.len());
    fail += usize::from(dt + dout == 0 && ctl_c == d_c);

    // ---- V: Verify/rollback sequence, dynamic vs static (odd and even T, both parities)
    let verify_arm = |on: bool| -> Result<(Record, Vec<u8>)> {
        core.set_static_decode(on);
        core.set_device_start(None);
        core.rollback(gpu.as_ref(), PROMPT, stream)?;
        let mut rec = Record::new();
        rig.pass(PassKind::Verify, 0, 3, &mut rec)?; // start even, T odd: leaves a pending row
        PassHook::rollback(&core, &ops, PROMPT + 1)?; // odd: pending restored from the saved row
        rig.pass(PassKind::Verify, 1, 2, &mut rec)?; // start odd, T even
        PassHook::rollback(&core, &ops, PROMPT + 2)?;
        rig.pass(PassKind::Decode, 2, 1, &mut rec)?; // start even, T 1
        rig.pass(PassKind::Verify, 3, 5, &mut rec)?; // start odd, T odd
        Ok((rec, rig.caches(end)?))
    };
    let (vd, vd_c) = verify_arm(false)?;
    let (vs, vs_c) = verify_arm(true)?;
    let (dt, dout, first) = diff(&vd, &vs);
    println!("V  verify T=3/2/5 + rollbacks, static vs dynamic: top-k differs on {dt}/{n}, attention on {dout}/{n} (first: {first}); caches identical: {}", vs_c == vd_c, n = vd.len());
    fail += usize::from(dt + dout > 0 || vs_c != vd_c);

    // ---- F: unit A/B, GEMV vs GEMM (the compressor's shape: N = 512, K = 5120, fp32)
    {
        let (n, k) = (512usize, 5120usize);
        let ik = IndexKernels::load(gpu.as_ref())?;
        let iops = IndexOps { gpu: gpu.as_ref(), k: &ik, stream };
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut rnd = |len: usize| -> Vec<u8> {
            (0..len)
                .flat_map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5).to_le_bytes()
                })
                .collect()
        };
        let (a_h, b_h) = (rnd(16 * k), rnd(n * k));
        let (a, b, c) = (gpu.alloc(16 * k * 4)?, gpu.alloc(n * k * 4)?, gpu.alloc(16 * n * 4)?);
        gpu.copy_h2d(&a_h, a)?;
        gpu.copy_h2d(&b_h, b)?;
        let run = |m: usize, off: bool| -> Result<(Vec<u8>, f64)> {
            GEMV_F32_OFF.store(off, Ordering::Relaxed);
            iops.gemm_f32(a, b, c, m, n, k)?;
            let out = g.down(c, m * n * 4)?;
            let t0 = std::time::Instant::now();
            for _ in 0..50 {
                iops.gemm_f32(a, b, c, m, n, k)?;
            }
            gpu.synchronize(stream)?;
            Ok((out, t0.elapsed().as_secs_f64() * 1e6 / 50.0))
        };
        for m in [1usize, 2, 6, 8, 16] {
            let (gv, tv) = run(m, false)?;
            let (gm, tm) = run(m, true)?;
            println!("F  M={m}: GEMV byte-identical to GEMM: {} | GEMM {tm:.0} us, GEMV {tv:.0} us ({:.1}x)", gv == gm, tm / tv);
            fail += usize::from(gv != gm);
        }
        // control: one flipped input bit in row 0 must change the GEMV's output
        let mut a2 = a_h.clone();
        a2[4 * 100] ^= 0x10;
        gpu.copy_h2d(&a2, a)?;
        let (gc, _) = run(1, false)?;
        gpu.copy_h2d(&a_h, a)?;
        let (g1, _) = run(1, false)?;
        println!("F  CONTROL one flipped input bit: output differs: {} (must)", gc != g1);
        fail += usize::from(gc == g1);
        GEMV_F32_OFF.store(false, Ordering::Relaxed);

        // PREFILL M: the 128x64 register-blocked GEMM (v2) vs the 64x64 original (v1).
        let mmax = 2048usize;
        let (pa_h, pa, pc) = (rnd(mmax * k), gpu.alloc(mmax * k * 4)?, gpu.alloc(mmax * n * 4)?);
        gpu.copy_h2d(&pa_h, pa)?;
        let prun = |m: usize, v1: bool| -> Result<(Vec<u8>, f64)> {
            GEMM_F32_V1.store(v1, Ordering::Relaxed);
            iops.gemm_f32(pa, b, pc, m, n, k)?;
            let out = g.down(pc, m * n * 4)?;
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                iops.gemm_f32(pa, b, pc, m, n, k)?;
            }
            gpu.synchronize(stream)?;
            Ok((out, t0.elapsed().as_secs_f64() * 1e3 / 10.0))
        };
        for m in [17usize, 128, 512, 2048] {
            let (o2, t2) = prun(m, false)?;
            let (o1, t1) = prun(m, true)?;
            let tf = 2.0 * (m * n * k) as f64 / (t2 * 1e9);
            println!("F  prefill M={m}: GEMM v2 byte-identical to v1: {} | v1 {t1:.3} ms, v2 {t2:.3} ms ({:.2}x, {tf:.1} TF/s)", o2 == o1, t1 / t2);
            fail += usize::from(o2 != o1);
        }
        GEMM_F32_V1.store(false, Ordering::Relaxed);
    }

    ensure!(fail == 0, "CORE STATIC GATE FAIL ({fail})");
    println!("CORE STATIC GATE PASS");
    Ok(())
}
