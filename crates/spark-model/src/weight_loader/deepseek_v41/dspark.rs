// SPDX-License-Identifier: AGPL-3.0-only

//! DSpark speculative decoding for DeepSeek-V4.1 (greedy): seed, draft, verify, accept, rollback.
//!
//! A port of `engine/model.py::{dspark_seed, dspark_draft}` and the lean greedy loop of
//! `engine/v41_engine.py::_decode_loop` (lines 869-1035):
//!
//! ```text
//! step(tok at pos):  draft  ids [tok, NOISE x4] through mtp.{0,1,2} (window = the drafter ring's
//!                           main positions pos-128..pos-1, plus all 5 draft kvs), head,
//!                           5 sequential Markov-bias argmaxes -> drafts d1..d5
//!                    verify [tok, d1..d5] as ONE PassKind::Verify pass (decode-size kernels,
//!                           per-row bit-identical to plain decode) -> 6 logit rows
//!                    accept a = leading argmax matches, bonus = argmax row a  (one readback)
//!                    rollback to pos + a + 1; seed the drafter ring from the verify pass's
//!                           L37-39 hiddens of rows 0..=a
//! ```
//!
//! Correctness does not depend on the drafter: every emitted token is the target model's argmax
//! at its position (verification), so spec greedy output must equal non-spec greedy output token
//! for token -- that is the end-to-end gate. The drafter's numerics only move acceptance.

use anyhow::{Context, Result, ensure};

use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::attn_block::{self, HEAD_DIM, N_HEADS, Q_LORA, RING, WINDOW};
use super::device_allocs::{DeviceAllocs, SharedGpu};
use super::forward::{PassHook, V41Forward, V41Seq};
use super::fwd::{BlockControl, SharedExpert, Tap, V41AttentionBlock, V41BlockWeights, V41RoutedMoe, block};
use super::mtp::{DSPARK_BLOCK, DSPARK_NOISE_TOKEN, DSPARK_TARGET_LAYERS};
use super::mtp_moe_decode::{MTP_LAYER0, MtpMoeDecode};
use super::ops::{MM_TILE, Ops, bytemuck_i32, bytemuck_u32, set_decode_pass, tiled_rows};
use super::attn_block::AttnCore;
use crate::layers::deepseek_v41_attn::core::ATTN_MODULE;

pub const DSPARK_MODULE: &str = "dspark_decode";
/// Draft rows (= drafts per step) and verify rows.
pub const B: usize = DSPARK_BLOCK;
pub const T_VERIFY: usize = B + 1;
/// Largest seed pass (the prefill replay tail).
const MAX_SEED_ROWS: usize = WINDOW;

struct Kernels {
    attn: KernelHandle,
    markov: KernelHandle,
    argmax_f32: KernelHandle,
    sample_softmax: KernelHandle,
    argmax_rows: KernelHandle,
    accept: KernelHandle,
}

/// Result of one spec step.
#[derive(Clone, Debug)]
pub struct StepOut {
    /// Accepted drafts (0..=B).
    pub accepted: usize,
    /// Tokens emitted this step: the accepted drafts, then the bonus token.
    pub emitted: Vec<u32>,
}

pub struct Dspark {
    k: Kernels,
    moe: MtpMoeDecode,
    /// The three drafter blocks in the main block's weight shape (layer ids 40..42).
    bw: Vec<V41BlockWeights>,
    /// Drafter window rings, one per block: `[RING, 512]` bf16, slot = position % RING.
    rings: Vec<DevicePtr>,
    /// Draft kvs `[MM_TILE, 512]` bf16 (the attention's "compressed" keys, NC = B).
    kv: DevicePtr,
    /// `[B, B]` i64: every draft row sees all B draft kvs (non-causal, as the reference).
    cidx: DevicePtr,
    /// `[B, 128]` i32 window positions (the same main window for every draft row).
    wpos: DevicePtr,
    /// Seed positions `[MAX_SEED_ROWS]` i32, and the seed stages.
    seed_pos: DevicePtr,
    main_x: DevicePtr,
    seed_kv: DevicePtr,
    /// bf16 `[MM_TILE, V]` draft logits; fp32 `[V]` Markov-biased row; u32 ids[0..=B]
    /// (ids[0] = the input token, ids[1..] = drafts); u32 verify argmaxes; u32 accept result.
    draft_logits: DevicePtr,
    lg: DevicePtr,
    ids: DevicePtr,
    am: DevicePtr,
    res: DevicePtr,
    vocab: usize,
    /// fp32 `[B, V]`: the distributions the sampled drafts were drawn from (`draft_probs`).
    q: DevicePtr,
    /// The NEXT draft's sampling (temperature, B uniforms in [0,1)), consumed by `draft`;
    /// None = greedy (argmax) drafts.
    sampling: std::sync::Mutex<Option<(f32, [f32; B])>>,
    /// Per-step wall ms (draft, verify+accept, commit) when `DSV41_DSPARK_PHASES=1`.
    pub phase_ms: std::sync::Mutex<Vec<[f64; 3]>>,
    _allocs: DeviceAllocs,
}

impl Dspark {
    pub fn new(shared: &SharedGpu, fwd: &V41Forward, limit: f32, route_scale: f32) -> Result<Self> {
        let gpu: &dyn GpuBackend = shared.as_ref();
        let w = fwd.dspark.as_ref().context("DSpark: the drafter weights are not loaded (ATLAS_DSV41_DSPARK=1)")?;
        ensure!(fwd.dspark_seed.is_some(), "DSpark: the L37-39 seed buffer is not enabled");
        let k = |m: &str, name: &str| gpu.kernel(m, name).with_context(|| format!("{m}::{name} is not in the compiled PTX"));
        let k = Kernels {
            attn: k(ATTN_MODULE, "dsv41_sparse_attn_w32")?,
            markov: k(DSPARK_MODULE, "dsv41_markov_bias")?,
            argmax_f32: k(DSPARK_MODULE, "dsv41_argmax_f32")?,
            sample_softmax: k(DSPARK_MODULE, "dsv41_sample_softmax")?,
            argmax_rows: k(DSPARK_MODULE, "dsv41_argmax_rows_bf16")?,
            accept: k(DSPARK_MODULE, "dsv41_dspark_accept")?,
        };
        let moe = MtpMoeDecode::new(shared, w, limit, route_scale)?;
        let bw = w
            .blocks
            .iter()
            .map(|b| V41BlockWeights {
                layer: MTP_LAYER0 + b.k,
                attn_norm: b.attn_norm,
                ffn_norm: b.ffn_norm,
                hc_attn: b.hc_attn,
                hc_ffn: b.hc_ffn,
                shared: SharedExpert { w1: b.shared.w1, w2: b.shared.w2, w3: b.shared.w3 },
                engram: None,
            })
            .collect();
        let mut allocs = DeviceAllocs::owned(shared.clone());
        let mut a = |bytes: usize| -> Result<DevicePtr> {
            let p = allocs.alloc(gpu, bytes.max(256))?;
            gpu.memset(p, 0, bytes.max(256))?;
            Ok(p)
        };
        let rings = (0..w.blocks.len()).map(|_| a(RING * HEAD_DIM * 2)).collect::<Result<Vec<_>>>()?;
        let d = fwd.dims.hidden;
        let v = fwd.vocab;
        let s = Self {
            k,
            moe,
            bw,
            rings,
            kv: a(MM_TILE * HEAD_DIM * 2)?,
            cidx: a(B * B * 8)?,
            wpos: a(B * WINDOW * 4)?,
            seed_pos: a(MAX_SEED_ROWS * 4)?,
            main_x: a(tiled_rows(MAX_SEED_ROWS) * d * 2)?,
            seed_kv: a(tiled_rows(MAX_SEED_ROWS) * HEAD_DIM * 2)?,
            draft_logits: a(MM_TILE * v * 2)?,
            lg: a(v * 4)?,
            ids: a(16 * 4)?,
            am: a(16 * 4)?,
            res: a(32 * 4)?,
            vocab: v,
            q: a(B * v * 4)?,
            sampling: std::sync::Mutex::new(None),
            phase_ms: std::sync::Mutex::new(Vec::new()),
            _allocs: DeviceAllocs::unowned(),
        };
        let cidx: Vec<u8> = (0..B).flat_map(|_| (0..B as i64).flat_map(|j| j.to_le_bytes())).collect();
        gpu.copy_h2d(&cidx, s.cidx)?;
        Ok(Self { _allocs: allocs, ..s })
    }

    /// Write the drafter rings for main positions `start..start + rows` from the seed rows
    /// `fwd.dspark_seed[0..rows]` (`model.py::dspark_seed`): main_x = rmsnorm(main_proj(seed),
    /// main_norm); per block kv = rope_w(rmsnorm(wkv(main_x), kv_norm)) -> ring[pos % RING].
    pub fn seed(&self, ops: &Ops, fwd: &V41Forward, rows: usize, start: usize) -> Result<()> {
        ensure!(rows >= 1 && rows <= MAX_SEED_ROWS, "DSpark seed: {rows} rows");
        let w = fwd.dspark.as_ref().context("DSpark weights")?;
        let seed = fwd.dspark_seed.context("DSpark seed buffer")?;
        let (d, eps, gpu) = (fwd.dims.hidden, fwd.dims.norm_eps, ops.gpu);
        let wscratch = fwd.scratch.wscratch;
        let pos: Vec<i32> = (start..start + rows).map(|p| p as i32).collect();
        gpu.copy_h2d_async(bytemuck_i32(&pos), self.seed_pos, ops.stream)?;
        ops.linear_fp8_tiled(seed, &w.head.main_proj, wscratch, self.main_x, rows)?;
        ops.rmsnorm(self.main_x, w.head.main_norm, self.main_x, rows, d, eps)?;
        let row = HEAD_DIM * 2;
        for (b, ring) in w.blocks.iter().zip(&self.rings) {
            ops.linear_fp8_tiled(self.main_x, &b.attn.wkv, wscratch, self.seed_kv, rows)?;
            ops.rmsnorm(self.seed_kv, b.attn.kv_norm, self.seed_kv, rows, HEAD_DIM, eps)?;
            ops.rope_tail(self.seed_kv, self.seed_pos, &fwd.freqs_w, rows, 1, HEAD_DIM, false)?;
            let first = start % RING;
            let n1 = rows.min(RING - first);
            gpu.copy_d2d_async(self.seed_kv, ring.offset(first * row), n1 * row, ops.stream)?;
            if n1 < rows {
                gpu.copy_d2d_async(self.seed_kv.offset(n1 * row), *ring, (rows - n1) * row, ops.stream)?;
            }
        }
        Ok(())
    }

    /// SAMPLED drafts for the next [`Self::draft`] only: d_i ~ q_i = softmax(markov_logits_i / T)
    /// (Python `dspark_draft` at temperature > 0: no top_p on q), using `uniforms[i]` (the
    /// caller's request RNG) for draft i. The q rows are kept at [`Self::draft_probs`].
    /// `temperature <= 0` or None = greedy drafts.
    pub fn set_draft_sampling(&self, sampling: Option<(f32, [f32; B])>) {
        *self.sampling.lock().expect("dspark sampling poisoned") = sampling.filter(|(t, _)| *t > 0.0);
    }

    /// fp32 `[B, vocab]`: q_i of the last SAMPLED draft (undefined after a greedy draft).
    pub fn draft_probs(&self) -> DevicePtr {
        self.q
    }

    /// Draft B tokens after `tok` at position `pos` (the drafter ring holds main positions < pos).
    /// Leaves ids[0] = tok, ids[1..=B] = drafts on the device; returns the drafts (one readback).
    pub fn draft(&self, ops: &Ops, fwd: &V41Forward, tok: u32, pos: usize, tap: &Tap) -> Result<Vec<u32>> {
        ensure!(pos >= 1, "DSpark draft needs at least one main position");
        let w = fwd.dspark.as_ref().context("DSpark weights")?;
        let (gpu, d, v) = (ops.gpu, fwd.dims.hidden, self.vocab);
        let s = &fwd.scratch;
        let mut ids = [DSPARK_NOISE_TOKEN; B];
        ids[0] = tok;
        gpu.copy_h2d_async(bytemuck_u32(&ids), fwd.ids_dev, ops.stream)?;
        gpu.copy_h2d_async(bytemuck_u32(&[tok]), self.ids, ops.stream)?;
        ops.embed(fwd.embed, fwd.ids_dev, s.x, B, d)?;
        ops.hc_expand(s.x, s.h, s.pre_mix, B, d)?;
        // The main window the drafts attend to: positions pos-128 .. pos-1 (-1 below 0).
        let main_last = pos as i64 - 1;
        let row: Vec<i32> = (0..WINDOW as i64).map(|j| main_last - (WINDOW as i64 - 1) + j).map(|p| if p >= 0 { p as i32 } else { -1 }).collect();
        let wpos: Vec<i32> = (0..B).flat_map(|_| row.iter().copied()).collect();
        gpu.copy_h2d_async(bytemuck_i32(&wpos), self.wpos, ops.stream)?;
        let qpos: Vec<i32> = (pos..pos + B).map(|p| p as i32).collect();
        gpu.copy_h2d_async(bytemuck_i32(&qpos), fwd.attn_scratch.pos, ops.stream)?;
        // Drafter blocks: decode-size kernels (small-M GEMV where available).
        set_decode_pass(true);
        for (k, bw) in self.bw.iter().enumerate() {
            let attn = DraftAttention { ds: self, fwd, k, blk: &w.blocks[k] };
            block(ops, bw, &fwd.dims, s, B, pos, &attn, &self.moe as &dyn V41RoutedMoe, tap, BlockControl::None)?;
        }
        // Head: hc_pre -> mtp.2.norm -> LM head over the B rows.
        ops.hc_pre(s.h, s.pre_mix, s.x, B, d)?;
        ops.rmsnorm(s.x, w.head.norm, s.x, B, d, fwd.dims.norm_eps)?;
        ops.linear_bf16_tiled(s.x, d, fwd.head, self.draft_logits, v, B, v, d)?;
        // Markov chain: draft i+1 = argmax (or a sample at T > 0) of logits[i] + markov(ids[i]).
        let sampling = self.sampling.lock().expect("dspark sampling poisoned").take();
        for i in 0..B {
            KernelLaunch::new(gpu, self.k.markov)
                .grid([(v as u32).div_ceil(8), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.draft_logits.offset(i * v * 2))
                .arg_ptr(w.head.markov_embed)
                .arg_ptr(w.head.markov_head)
                .arg_ptr(self.ids)
                .arg_i32(i as i32)
                .arg_ptr(self.lg)
                .arg_u32(v as u32)
                .launch(ops.stream)?;
            if let Some((temp, u)) = sampling {
                KernelLaunch::new(gpu, self.k.sample_softmax)
                    .grid([1, 1, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(self.lg)
                    .arg_u32(v as u32)
                    .arg_f32(temp)
                    .arg_f32(u[i])
                    .arg_ptr(self.q.offset(i * v * 4))
                    .arg_ptr(self.ids)
                    .arg_i32(i as i32 + 1)
                    .launch(ops.stream)?;
            } else {
                KernelLaunch::new(gpu, self.k.argmax_f32)
                    .grid([1, 1, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(self.lg)
                    .arg_u32(v as u32)
                    .arg_ptr(self.ids)
                    .arg_i32(i as i32 + 1)
                    .launch(ops.stream)?;
            }
        }
        gpu.synchronize(ops.stream)?;
        let mut out = [0u8; (B + 1) * 4];
        gpu.copy_d2h(self.ids, &mut out)?;
        Ok(out.chunks_exact(4).skip(1).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    /// Greedy accept over the verify logits `[k + 1, V]` bf16 against the first `k` drafts in
    /// ids[1..]. Returns (a, argmax per verify row). One readback.
    pub fn accept(&self, ops: &Ops, logits: DevicePtr, k: usize) -> Result<(usize, Vec<u32>)> {
        ensure!((1..=B).contains(&k), "DSpark accept over {k} drafts");
        let gpu = ops.gpu;
        KernelLaunch::new(gpu, self.k.argmax_rows)
            .grid([k as u32 + 1, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(logits)
            .arg_u32(self.vocab as u32)
            .arg_ptr(self.am)
            .launch(ops.stream)?;
        KernelLaunch::new(gpu, self.k.accept)
            .grid([1, 1, 1])
            .block([1, 1, 1])
            .arg_ptr(self.am)
            .arg_ptr(self.ids.offset(4))
            .arg_i32(k as i32)
            .arg_ptr(self.res)
            .launch(ops.stream)?;
        gpu.synchronize(ops.stream)?;
        let mut r = [0u8; (2 * B + 2) * 4];
        gpu.copy_d2h(self.res, &mut r)?;
        let v: Vec<u32> = r.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Ok((v[0] as usize, v[1..=k + 1].to_vec()))
    }

    /// The first half of a step: draft B tokens after `tok` and run the verify pass over
    /// [tok, d1..dB] at position `seq.len` (T_VERIFY rows of logits in `logits`). Returns the
    /// drafts, the greedy accept count and the argmax of every verify row. `seq` holds the
    /// T_VERIFY verify positions until [`Dspark::commit`] rolls it back to the accepted prefix.
    #[allow(clippy::too_many_arguments)]
    pub fn propose_verify(
        &self,
        ops: &Ops,
        fwd: &V41Forward,
        seq: &mut V41Seq,
        tok: u32,
        hook: &dyn PassHook,
        core: &dyn AttnCore,
        main_moe: &dyn V41RoutedMoe,
        tap: &Tap,
        logits: DevicePtr,
    ) -> Result<(Vec<u32>, usize, Vec<u32>)> {
        let pos = seq.len;
        let drafts = self.draft(ops, fwd, tok, pos, tap)?;
        let mut block_ids = Vec::with_capacity(T_VERIFY);
        block_ids.push(tok);
        block_ids.extend_from_slice(&drafts);
        fwd.verify(ops, seq, &block_ids, hook, core, main_moe, tap, logits)?;
        let (a, am) = self.accept(ops, logits, B)?;
        Ok((drafts, a, am))
    }

    /// The second half: keep `tok` and the first `accepted` drafts of the verify pass that
    /// started at `pos` (roll the rest back) and seed the drafter ring from their rows.
    pub fn commit(&self, ops: &Ops, fwd: &V41Forward, seq: &mut V41Seq, pos: usize, accepted: usize, hook: &dyn PassHook) -> Result<()> {
        ensure!(accepted <= B, "DSpark commit: {accepted} > {B} drafts");
        fwd.rollback(ops, seq, pos + accepted + 1, hook)?;
        self.seed(ops, fwd, accepted + 1, pos)
    }

    /// One greedy spec step for input token `tok` at position `seq.len`, verifying the first `k`
    /// of the B drafts (k + 1 rows; the output is greedy-exact for any k, see dspark_adapt).
    /// `logits` must hold `tiled_rows(T_VERIFY)` rows. `force_accept_all` is a NEGATIVE CONTROL ONLY (accepts every
    /// draft without verifying): its output must diverge from non-spec greedy.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        ops: &Ops,
        fwd: &V41Forward,
        seq: &mut V41Seq,
        tok: u32,
        hook: &dyn PassHook,
        core: &dyn AttnCore,
        main_moe: &dyn V41RoutedMoe,
        tap: &Tap,
        logits: DevicePtr,
        k: usize,
        force_accept_all: bool,
    ) -> Result<StepOut> {
        ensure!((1..=B).contains(&k), "DSpark step verifying {k} drafts");
        let pos = seq.len;
        let phases = std::env::var("DSV41_DSPARK_PHASES").as_deref() == Ok("1");
        let t0 = std::time::Instant::now();
        // (propose_verify's two readbacks already synchronize after the draft and after accept)
        let drafts = self.draft(ops, fwd, tok, pos, tap)?;
        let t1 = std::time::Instant::now();
        let mut block_ids = Vec::with_capacity(k + 1);
        block_ids.push(tok);
        block_ids.extend_from_slice(&drafts[..k]);
        fwd.verify(ops, seq, &block_ids, hook, core, main_moe, tap, logits)?;
        let (mut a, am) = self.accept(ops, logits, k)?;
        let t2 = std::time::Instant::now();
        if force_accept_all {
            a = k;
        }
        self.commit(ops, fwd, seq, pos, a, hook)?;
        if phases {
            ops.gpu.synchronize(ops.stream)?;
            let t3 = std::time::Instant::now();
            let ms = |x: std::time::Instant, y: std::time::Instant| (y - x).as_secs_f64() * 1e3;
            self.phase_ms.lock().expect("phase poisoned").push([ms(t0, t1), ms(t1, t2), ms(t2, t3)]);
        }
        let mut emitted: Vec<u32> = drafts[..a].to_vec();
        emitted.push(am[a]);
        Ok(StepOut { accepted: a, emitted })
    }
}

/// The drafter's attention for block `k` (`model.py::attention` with `mtp_extra`): q/kv at the
/// draft positions (freqs_w); keys = the ring's main window + all B draft kvs; no ring write.
struct DraftAttention<'a> {
    ds: &'a Dspark,
    fwd: &'a V41Forward,
    k: usize,
    blk: &'a super::mtp::MtpBlock,
}

impl V41AttentionBlock for DraftAttention<'_> {
    fn forward(&self, ops: &Ops, _layer: usize, x: DevicePtr, out: DevicePtr, t: usize, _start: usize) -> Result<()> {
        ensure!(t == B, "DSpark draft attention runs over exactly {B} rows");
        let (w, s, fwd) = (&self.blk.attn, &self.fwd.attn_scratch, self.fwd);
        let (eps, wscratch, rope) = (fwd.dims.norm_eps, fwd.scratch.wscratch, &fwd.freqs_w);
        // q (s.pos already holds the draft positions)
        ops.linear_fp8_tiled(x, &w.wq_a, wscratch, s.qr, t)?;
        ops.rmsnorm(s.qr, w.q_norm, s.qr, t, Q_LORA, eps)?;
        ops.linear_fp8_tiled(s.qr, &w.wq_b, wscratch, s.q, t)?;
        ops.rope_tail(s.q, s.pos, rope, t, N_HEADS, HEAD_DIM, false)?;
        // the drafts' own kvs: NOT written to the ring
        ops.linear_fp8_tiled(x, &w.wkv, wscratch, self.ds.kv, t)?;
        ops.rmsnorm(self.ds.kv, w.kv_norm, self.ds.kv, t, HEAD_DIM, eps)?;
        ops.rope_tail(self.ds.kv, s.pos, rope, t, 1, HEAD_DIM, false)?;
        KernelLaunch::new(ops.gpu, self.ds.k.attn)
            .grid([t as u32, (N_HEADS / 8) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(s.q)
            .arg_ptr(self.ds.rings[self.k])
            .arg_ptr(self.ds.wpos)
            .arg_ptr(self.ds.kv)
            .arg_ptr(self.ds.cidx)
            .arg_ptr(w.attn_sink)
            .arg_ptr(s.o)
            .arg_i32(t as i32)
            .arg_i32(N_HEADS as i32)
            .arg_i32(HEAD_DIM as i32)
            .arg_i32(WINDOW as i32)
            .arg_i32(B as i32)
            .arg_i32(RING as i32)
            .arg_i32(0)
            .arg_f32((HEAD_DIM as f32).powf(-0.5))
            .launch(ops.stream)?;
        attn_block::attn_output(ops, w, s, wscratch, rope, t, out, &Tap::off())
    }
}

/// The seed rows a spec step leaves in `fwd.dspark_seed`: one per verify row (L37-39 hc-means).
pub const SEED_COLS: usize = DSPARK_TARGET_LAYERS.len();
