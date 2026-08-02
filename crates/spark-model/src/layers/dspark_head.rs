// SPDX-License-Identifier: AGPL-3.0-only

//! DSpark block-drafter propose forward (docs/dspark_port.md).
//!
//! Executes the official `inference/model.py` DSpark algorithm with Atlas
//! ops: fuse the target's captured hiddens (`main_proj`/`main_norm`), write
//! one `main_kv` row per stage into a 128-entry ring, run a 5-row
//! bidirectional block (committed token + 4 noise rows) through the three
//! drafter stages (mHC + custom windowed attention + 256-expert MoE), then
//! the Markov-biased greedy chain + confidence head.
//!
//! Eager and correctness-first: sizes are tiny except the MoE and lm_head,
//! which reuse the target's batched paths. The oracle is the Python
//! reference driven by the same ATLAS_DSPARK_DUMP captures
//! (`bench/deepseek-v4/dspark_probe/`, 3.81 tok/step ungated) — the
//! engine-vs-reference draft diff is the acceptance test for this module.

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_loader::deepseek_v4::dspark::DsparkDrafterModule;
use crate::weight_map::DenseWeight;

/// Geometry constants fixed by the V4-Flash drafter checkpoint. Asserted at
/// build from the target config where they overlap.
const HEADS: u32 = 64;
const HEAD_DIM: u32 = 512;
const ROPE_DIM: u32 = 64;
const Q_LORA: u32 = 1024;
const O_GROUPS: u32 = 8;
const O_LORA: u32 = 1024;

pub struct DsparkDraftHead {
    pub module: DsparkDrafterModule,
    /// Drafter-side config: the target config with `num_experts` swapped to
    /// the drafter's 256. The MoE forward paths read expert count from
    /// `ctx.config`, so propose builds a derived ForwardContext around this.
    drafter_config: ModelConfig,
    /// Target's token embedding table (shared; BF16 `[vocab, h]`).
    embed: DenseWeight,
    /// Target's lm_head, dequanted/dense BF16 (`[vocab, h]`). The in-server
    /// build can instead pass the FP8 mirror through `lm_head_fp8`.
    lm_head_bf16: Option<DenseWeight>,
    lm_head_fp8: Option<crate::weight_map::Fp8DenseWeight>,
    vocab: u32,
    h: u32,
    hc_mult: u32,
    eps: f32,
    hc_eps: f32,
    sinkhorn_iters: u32,
    block: u32,

    // ── kernels ──
    k_gemm: KernelHandle,
    k_gemv: KernelHandle,
    k_gemv_fp8: KernelHandle,
    k_rms: KernelHandle, // rms_norm_vanilla — HF-exact weights
    k_residual_add: KernelHandle,
    k_hc_expand: KernelHandle,
    k_hc_pre: KernelHandle,
    k_hc_post: KernelHandle,
    k_hc_head: KernelHandle,
    k_argmax: KernelHandle,
    k_rope: KernelHandle,
    k_attn: KernelHandle,

    // ── state ──
    /// Per-stage `main_kv` ring `[window, HEAD_DIM]` BF16.
    rings: Vec<DevicePtr>,
    /// Rope table `[max_seq, ROPE_DIM/2, 2]` F32 (cos, sin), θ from config,
    /// NO YaRN (drafter is pure sliding-window attention).
    freqs: DevicePtr,

    // ── scratch (own allocations; propose interleaves with target scratch) ──
    concat_in: DevicePtr, // [3h]
    main_x: DevicePtr,    // [h]
    mkv: DevicePtr,       // [HEAD_DIM]
    x5: DevicePtr,        // [B, h]
    hc_a: DevicePtr,      // [B, hc, h] F32
    hc_b: DevicePtr,      // [B, hc, h] F32 (hc_post cannot run in place)
    y5: DevicePtr,        // [B, h]
    n5: DevicePtr,        // [B, h]
    post5: DevicePtr,     // [B, hc] F32
    comb5: DevicePtr,     // [B, hc, hc] F32
    q_lora5: DevicePtr,   // [B, Q_LORA]
    q5: DevicePtr,        // [B, HEADS*HEAD_DIM]
    kv5: DevicePtr,       // [B, HEAD_DIM]
    o5: DevicePtr,        // [B, HEADS*HEAD_DIM]
    ogrp: DevicePtr,      // [B, HEADS*HEAD_DIM/O_GROUPS]
    ogrp_out: DevicePtr,  // [B, O_LORA]
    o_lora5: DevicePtr,   // [B, O_GROUPS*O_LORA]
    attn5: DevicePtr,     // [B, h]
    h5: DevicePtr,        // [B, h]
    f5: DevicePtr,        // [B, h]
    logits5: DevicePtr,   // [B, vocab]
    ones_hd: DevicePtr,   // [HEAD_DIM] BF16 = 1.0 (parameterless per-head RMS)
    mstate: DevicePtr,    // [markov_rank]
    mbias: DevicePtr,     // [vocab]
    conf_in: DevicePtr,   // [h + markov_rank]
    scratch_u32: DevicePtr,
    /// The model's `[layers, rows, h]` hc-mean capture buffer
    /// (ATLAS_DSPARK_CAPTURE=1), installed by the factory after model build.
    capture_buf: DevicePtr,
    capture_rows: usize,
}

impl DsparkDraftHead {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        module: DsparkDrafterModule,
        target_config: &ModelConfig,
        embed: DenseWeight,
        lm_head_bf16: Option<DenseWeight>,
        lm_head_fp8: Option<crate::weight_map::Fp8DenseWeight>,
        drafter_num_experts: usize,
        max_seq_len: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let h = target_config.hidden_size as u32;
        let hc_mult = target_config.hc_mult as u32;
        let vocab = target_config.vocab_size as u32;
        anyhow::ensure!(h == 4096 && hc_mult > 0, "unexpected V4-Flash geometry");
        anyhow::ensure!(
            lm_head_bf16.is_some() || lm_head_fp8.is_some(),
            "DSpark head needs a shared lm_head"
        );
        let block = module.params.block_size as u32;
        let win = module.params.window;
        let mr = module.params.markov_rank;
        let mut drafter_config = target_config.clone();
        drafter_config.num_experts = drafter_num_experts;

        // Rope table: interleaved-pair (cos, sin) per (pos, j). Plain theta —
        // the reference disables YaRN for pure sliding-window attention.
        let theta = if target_config.rope_theta > 0.0 {
            target_config.rope_theta
        } else {
            10000.0
        };
        let half = (ROPE_DIM / 2) as usize;
        let mut tab = vec![0f32; max_seq_len * half * 2];
        for pos in 0..max_seq_len {
            for j in 0..half {
                let freq = 1.0f64 / (theta as f64).powf(2.0 * j as f64 / ROPE_DIM as f64);
                let ang = pos as f64 * freq;
                tab[(pos * half + j) * 2] = ang.cos() as f32;
                tab[(pos * half + j) * 2 + 1] = ang.sin() as f32;
            }
        }
        let freqs = gpu.alloc(tab.len() * 4)?;
        // SAFETY: `tab` is a live Vec<f32>; the byte view covers exactly its
        // allocation for the duration of the copy.
        let bytes = unsafe { std::slice::from_raw_parts(tab.as_ptr() as *const u8, tab.len() * 4) };
        gpu.copy_h2d(bytes, freqs)?;

        let bf16 = 2usize;
        let b = block as usize;
        let hu = h as usize;
        let alloc = |elems: usize| gpu.alloc(elems * bf16);
        let alloc_f32 = |elems: usize| gpu.alloc(elems * 4);

        let mut rings = Vec::new();
        for _ in 0..module.stages.len() {
            rings.push(alloc(win * HEAD_DIM as usize)?);
        }
        // Parameterless per-head q-RMS reuses rms_norm_vanilla with a ones
        // weight (BF16 1.0 = 0x3F80).
        let ones_hd = alloc(HEAD_DIM as usize)?;
        let one_bf16 = vec![0x3F80u16; HEAD_DIM as usize];
        // SAFETY: byte view of a live Vec<u16> for the copy duration.
        let ob = unsafe {
            std::slice::from_raw_parts(one_bf16.as_ptr() as *const u8, HEAD_DIM as usize * 2)
        };
        gpu.copy_h2d(ob, ones_hd)?;

        Ok(Self {
            drafter_config,
            embed,
            lm_head_bf16,
            lm_head_fp8,
            vocab,
            h,
            hc_mult,
            eps: target_config.rms_norm_eps as f32,
            hc_eps: target_config.hc_eps as f32,
            sinkhorn_iters: target_config.hc_sinkhorn_iters as u32,
            block,
            k_gemm: gpu.kernel("gemm", "dense_gemm_bf16")?,
            k_gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            k_gemv_fp8: gpu.kernel("gemv_fp8w", "dense_gemv_fp8w")?,
            k_rms: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            k_residual_add: gpu.kernel("residual_add", "bf16_residual_add")?,
            k_hc_expand: gpu.kernel("hyper_connection", "hc_expand")?,
            k_hc_pre: gpu.kernel("hyper_connection", "hc_pre")?,
            k_hc_post: gpu.kernel("hyper_connection", "hc_post")?,
            k_hc_head: gpu.kernel("hyper_connection", "hc_head")?,
            k_argmax: gpu.kernel("argmax", "argmax_bf16")?,
            k_rope: gpu.kernel("dspark_drafter", "dspark_rope")?,
            k_attn: gpu.kernel("dspark_drafter", "dspark_attn")?,
            rings,
            freqs,
            concat_in: alloc(3 * hu)?,
            main_x: alloc(hu)?,
            mkv: alloc(HEAD_DIM as usize)?,
            x5: alloc(b * hu)?,
            hc_a: alloc_f32(b * hc_mult as usize * hu)?,
            hc_b: alloc_f32(b * hc_mult as usize * hu)?,
            y5: alloc(b * hu)?,
            n5: alloc(b * hu)?,
            post5: alloc_f32(b * hc_mult as usize)?,
            comb5: alloc_f32(b * (hc_mult * hc_mult) as usize)?,
            q_lora5: alloc(b * Q_LORA as usize)?,
            q5: alloc(b * (HEADS * HEAD_DIM) as usize)?,
            kv5: alloc(b * HEAD_DIM as usize)?,
            o5: alloc(b * (HEADS * HEAD_DIM) as usize)?,
            ogrp: alloc(b * (HEADS * HEAD_DIM / O_GROUPS) as usize)?,
            ogrp_out: alloc(b * O_LORA as usize)?,
            o_lora5: alloc(b * (O_GROUPS * O_LORA) as usize)?,
            attn5: alloc(b * hu)?,
            h5: alloc(b * hu)?,
            f5: alloc(b * hu)?,
            logits5: alloc(b * vocab as usize)?,
            ones_hd,
            mstate: alloc(mr)?,
            mbias: alloc(vocab as usize)?,
            conf_in: alloc(hu + mr)?,
            scratch_u32: gpu.alloc(4)?,
            capture_buf: DevicePtr::NULL,
            capture_rows: 0,
            module,
        })
    }

    /// ATLAS_DSPARK_DEBUG=1: sync + print the L2 norm and first values of a
    /// BF16 buffer. Bisects the propose chain against the Python reference.
    fn dbg(&self, gpu: &dyn GpuBackend, label: &str, ptr: DevicePtr, n: usize, stream: u64) {
        if std::env::var("ATLAS_DSPARK_DEBUG").as_deref() != Ok("1") {
            return;
        }
        let _ = gpu.synchronize(stream);
        let mut b = vec![0u8; n * 2];
        if gpu.copy_d2h(ptr, &mut b).is_err() {
            return;
        }
        let v: Vec<f32> = b
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect();
        let norm = v
            .iter()
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            .sqrt();
        eprintln!(
            "DSPARK_DBG {label}: norm={norm:.4} first={:?}",
            &v[..4.min(v.len())]
        );
    }

    #[allow(dead_code)]
    fn dbg_f32(&self, gpu: &dyn GpuBackend, label: &str, ptr: DevicePtr, n: usize, stream: u64) {
        if std::env::var("ATLAS_DSPARK_DEBUG").as_deref() != Ok("1") {
            return;
        }
        let _ = gpu.synchronize(stream);
        let mut b = vec![0u8; n * 4];
        if gpu.copy_d2h(ptr, &mut b).is_err() {
            return;
        }
        let v: Vec<f32> = b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let norm = v
            .iter()
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            .sqrt();
        eprintln!(
            "DSPARK_DBG {label}: norm={norm:.4} first={:?}",
            &v[..4.min(v.len())]
        );
    }

    fn rope(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        rows: u32,
        heads: u32,
        head_dim: u32,
        pos_base: u32,
        pos_stride: u32,
        inverse: bool,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_rope)
            .grid([rows, heads, 1])
            .block([ROPE_DIM / 2, 1, 1])
            .arg_ptr(x)
            .arg_ptr(self.freqs)
            .arg_u32(heads)
            .arg_u32(head_dim)
            .arg_u32(ROPE_DIM)
            .arg_u32(pos_base)
            .arg_u32(pos_stride)
            .arg_u32(inverse as u32)
            .launch(stream)
    }

    /// Compute `main_x` from the three capture vectors and write each stage's
    /// `main_kv` into its ring at `pos % window`. Must be called exactly once
    /// per committed position, in order — prefill seeding, the propose site,
    /// and the multi-accept catchup all funnel here.
    pub fn seed_position(
        &self,
        gpu: &dyn GpuBackend,
        captures: [DevicePtr; 3],
        pos: usize,
        stream: u64,
    ) -> Result<()> {
        let h = self.h as usize;
        for (i, c) in captures.iter().enumerate() {
            gpu.copy_d2d_async(*c, self.concat_in.offset(i * h * 2), h * 2, stream)?;
        }
        ops::dense_gemv(
            gpu,
            self.k_gemv,
            self.concat_in,
            &self.module.main_proj,
            self.main_x,
            self.h,
            3 * self.h,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.k_rms,
            self.main_x,
            &self.module.main_norm,
            self.main_x,
            1,
            self.h,
            self.eps,
            stream,
        )?;
        let win = self.module.params.window;
        let slot = pos % win;
        for (s, stage) in self.module.stages.iter().enumerate() {
            ops::dense_gemv(
                gpu,
                self.k_gemv,
                self.main_x,
                &stage.wkv,
                self.mkv,
                HEAD_DIM,
                self.h,
                stream,
            )?;
            ops::rms_norm(
                gpu,
                self.k_rms,
                self.mkv,
                &stage.kv_norm,
                self.mkv,
                1,
                HEAD_DIM,
                self.eps,
                stream,
            )?;
            self.rope(gpu, self.mkv, 1, 1, HEAD_DIM, pos as u32, 0, false, stream)?;
            gpu.copy_d2d_async(
                self.mkv,
                self.rings[s].offset(slot * HEAD_DIM as usize * 2),
                HEAD_DIM as usize * 2,
                stream,
            )?;
        }
        Ok(())
    }

    /// Full propose: seed `pos`, then run the 5-row block and the
    /// Markov-biased greedy chain. Returns `block_size` drafts and their
    /// confidence-head sigmoids (ungated — the caller applies the policy).
    ///
    /// Contract (pinned by the probe alignment fix): `committed` is the token
    /// GENERATED from position `pos`; `captures` are the hc-mean hiddens OF
    /// position `pos`; draft j targets sequence position `pos + 2 + j`.
    #[allow(clippy::too_many_arguments)]
    pub fn propose_block(
        &self,
        gpu: &dyn GpuBackend,
        ctx: &ForwardContext,
        captures: [DevicePtr; 3],
        committed: u32,
        pos: usize,
        stream: u64,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        self.seed_position(gpu, captures, pos, stream)?;
        self.dbg(gpu, "main_x", self.main_x, self.h as usize, stream);
        self.dbg(
            gpu,
            "ring[0]@slot",
            self.rings[0].offset((pos % self.module.params.window) * HEAD_DIM as usize * 2),
            HEAD_DIM as usize,
            stream,
        );

        let b = self.block;
        let h = self.h;
        let hu = h as usize;
        let hc = self.hc_mult;
        let noise = self.module.params.noise_token_id;

        // Block rows: [committed, noise×(b-1)] embedded from the shared table.
        for r in 0..b as usize {
            let tok = if r == 0 { committed } else { noise } as usize;
            gpu.copy_d2d_async(
                self.embed.weight.offset(tok * hu * 2),
                self.x5.offset(r * hu * 2),
                hu * 2,
                stream,
            )?;
        }
        ops::hc_expand(gpu, self.k_hc_expand, self.x5, self.hc_a, b, h, hc, stream)?;
        self.dbg(gpu, "x5.row0", self.x5, hu, stream);

        // MoE runs against the DRAFTER config (256 experts) — the target ctx
        // would route over the wrong expert count. Rebuilt field-by-field
        // (`..*ctx` can't move the non-Copy capture field).
        let moe_ctx = ForwardContext {
            buffers: ctx.buffers,
            gpu: ctx.gpu,
            config: &self.drafter_config,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
        };

        let ring_vis = (pos + 1).min(self.module.params.window) as u32;
        let blk_pos = (pos + 1) as u32; // block rows at pos+1 .. pos+b
        let mut cur = self.hc_a;
        let mut nxt = self.hc_b;
        for (s, stage) in self.module.stages.iter().enumerate() {
            // ── attention site ──
            ops::hc_pre(
                gpu,
                self.k_hc_pre,
                cur,
                stage.hc_attn.hc_fn,
                stage.hc_attn.hc_scale,
                stage.hc_attn.hc_base,
                self.y5,
                self.post5,
                self.comb5,
                b,
                h,
                hc,
                self.sinkhorn_iters,
                self.eps,
                self.hc_eps,
                stream,
            )?;
            ops::rms_norm(
                gpu,
                self.k_rms,
                self.y5,
                &stage.attn_norm,
                self.n5,
                b,
                h,
                self.eps,
                stream,
            )?;
            self.dbg(gpu, &format!("s{s}.attn.n5"), self.n5, hu, stream);

            // Q path: wq_a → q_norm → wq_b → per-head parameterless RMS → rope.
            ops::dense_gemm(
                gpu,
                self.k_gemm,
                self.n5,
                &stage.wq_a,
                self.q_lora5,
                b,
                Q_LORA,
                h,
                stream,
            )?;
            ops::rms_norm(
                gpu,
                self.k_rms,
                self.q_lora5,
                &stage.q_norm,
                self.q_lora5,
                b,
                Q_LORA,
                self.eps,
                stream,
            )?;
            ops::dense_gemm(
                gpu,
                self.k_gemm,
                self.q_lora5,
                &stage.wq_b,
                self.q5,
                b,
                HEADS * HEAD_DIM,
                Q_LORA,
                stream,
            )?;
            let ones = DenseWeight {
                weight: self.ones_hd,
            };
            ops::rms_norm(
                gpu,
                self.k_rms,
                self.q5,
                &ones,
                self.q5,
                b * HEADS,
                HEAD_DIM,
                self.eps,
                stream,
            )?;
            self.rope(gpu, self.q5, b, HEADS, HEAD_DIM, blk_pos, 1, false, stream)?;
            self.dbg(
                gpu,
                &format!("s{s}.q5.h0"),
                self.q5,
                HEAD_DIM as usize,
                stream,
            );

            // KV path: wkv → kv_norm → rope (norm BEFORE rope, per reference).
            ops::dense_gemm(
                gpu,
                self.k_gemm,
                self.n5,
                &stage.wkv,
                self.kv5,
                b,
                HEAD_DIM,
                h,
                stream,
            )?;
            ops::rms_norm(
                gpu,
                self.k_rms,
                self.kv5,
                &stage.kv_norm,
                self.kv5,
                b,
                HEAD_DIM,
                self.eps,
                stream,
            )?;
            self.rope(gpu, self.kv5, b, 1, HEAD_DIM, blk_pos, 1, false, stream)?;
            self.dbg(
                gpu,
                &format!("s{s}.kv5.r0"),
                self.kv5,
                HEAD_DIM as usize,
                stream,
            );

            // Windowed bidirectional attention + MLA output de-rotation.
            KernelLaunch::new(gpu, self.k_attn)
                .grid([b, HEADS, 1])
                .block([128, 1, 1])
                .arg_ptr(self.q5)
                .arg_ptr(self.rings[s])
                .arg_ptr(self.kv5)
                .arg_ptr(stage.attn_sink.weight)
                .arg_ptr(self.o5)
                .arg_u32(b)
                .arg_u32(HEADS)
                .arg_u32(HEAD_DIM)
                .arg_u32(ring_vis)
                .arg_f32(1.0 / (HEAD_DIM as f32).sqrt())
                .launch(stream)?;
            self.rope(gpu, self.o5, b, HEADS, HEAD_DIM, blk_pos, 1, true, stream)?;
            self.dbg(
                gpu,
                &format!("s{s}.o5.h0"),
                self.o5,
                HEAD_DIM as usize,
                stream,
            );

            // Grouped wo_a (einsum bsgd,grd->bsgr) then wo_b.
            let group_in = (HEADS * HEAD_DIM / O_GROUPS) as usize; // 4096
            let row_o = (HEADS * HEAD_DIM) as usize;
            for g in 0..O_GROUPS as usize {
                for r in 0..b as usize {
                    gpu.copy_d2d_async(
                        self.o5.offset((r * row_o + g * group_in) * 2),
                        self.ogrp.offset(r * group_in * 2),
                        group_in * 2,
                        stream,
                    )?;
                }
                let wg = DenseWeight {
                    weight: stage.wo_a.weight.offset(g * O_LORA as usize * group_in * 2),
                };
                ops::dense_gemm(
                    gpu,
                    self.k_gemm,
                    self.ogrp,
                    &wg,
                    self.ogrp_out,
                    b,
                    O_LORA,
                    group_in as u32,
                    stream,
                )?;
                for r in 0..b as usize {
                    gpu.copy_d2d_async(
                        self.ogrp_out.offset(r * O_LORA as usize * 2),
                        self.o_lora5
                            .offset((r * (O_GROUPS * O_LORA) as usize + g * O_LORA as usize) * 2),
                        O_LORA as usize * 2,
                        stream,
                    )?;
                }
            }
            ops::dense_gemm(
                gpu,
                self.k_gemm,
                self.o_lora5,
                &stage.wo_b,
                self.attn5,
                b,
                h,
                O_GROUPS * O_LORA,
                stream,
            )?;
            self.dbg(gpu, &format!("s{s}.attn5.r0"), self.attn5, hu, stream);
            ops::hc_post(
                gpu,
                self.k_hc_post,
                self.attn5,
                cur,
                self.post5,
                self.comb5,
                nxt,
                b,
                h,
                hc,
                stream,
            )?;
            self.dbg_f32(gpu, &format!("s{s}.hc_post_attn"), nxt, 8, stream);
            std::mem::swap(&mut cur, &mut nxt);

            // ── FFN site ──
            ops::hc_pre(
                gpu,
                self.k_hc_pre,
                cur,
                stage.hc_ffn.hc_fn,
                stage.hc_ffn.hc_scale,
                stage.hc_ffn.hc_base,
                self.y5,
                self.post5,
                self.comb5,
                b,
                h,
                hc,
                self.sinkhorn_iters,
                self.eps,
                self.hc_eps,
                stream,
            )?;
            ops::rms_norm(
                gpu,
                self.k_rms,
                self.y5,
                &stage.ffn_norm,
                self.n5,
                b,
                h,
                self.eps,
                stream,
            )?;
            self.dbg(gpu, &format!("s{s}.ffn.n5"), self.n5, 8, stream);
            stage
                .moe
                .forward_kn(self.n5, b as usize, &moe_ctx, stream)
                .with_context(|| format!("DSpark stage {s} MoE"))?;
            let moe_out = ctx.buffers.moe_output();
            self.dbg(gpu, &format!("s{s}.moe.r0"), moe_out, hu, stream);
            ops::hc_post(
                gpu,
                self.k_hc_post,
                moe_out,
                cur,
                self.post5,
                self.comb5,
                nxt,
                b,
                h,
                hc,
                stream,
            )?;
            std::mem::swap(&mut cur, &mut nxt);
        }

        // ── head: hc collapse → final norm → shared lm_head ──
        let hh = self
            .module
            .hc_head
            .as_ref()
            .context("DSpark drafter has no hc_head")?;
        ops::hc_head(
            gpu,
            self.k_hc_head,
            cur,
            hh.hc_fn,
            hh.hc_scale,
            hh.hc_base,
            self.h5,
            b,
            h,
            hc,
            self.eps,
            self.hc_eps,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.k_rms,
            self.h5,
            &self.module.norm,
            self.f5,
            b,
            h,
            self.eps,
            stream,
        )?;
        self.dbg(gpu, "f5.r0", self.f5, hu, stream);
        for r in 0..b as usize {
            let row_in = self.f5.offset(r * hu * 2);
            let row_out = self.logits5.offset(r * self.vocab as usize * 2);
            if let Some(ref fp8) = self.lm_head_fp8 {
                ops::dense_gemv_fp8w(
                    gpu,
                    self.k_gemv_fp8,
                    row_in,
                    fp8,
                    row_out,
                    self.vocab,
                    h,
                    stream,
                )?;
            } else if let Some(ref bf) = self.lm_head_bf16 {
                ops::dense_gemv(gpu, self.k_gemv, row_in, bf, row_out, self.vocab, h, stream)?;
            }
        }

        // ── Markov-biased greedy chain + confidence ──
        let mr = self.module.params.markov_rank;
        let mut drafts = Vec::with_capacity(b as usize);
        let mut confs = Vec::with_capacity(b as usize);
        let mut prev = committed;
        for r in 0..b as usize {
            // markov_w1[prev] → state; logits_r += markov_w2 · state.
            gpu.copy_d2d_async(
                self.module.markov_w1.weight.offset(prev as usize * mr * 2),
                self.mstate,
                mr * 2,
                stream,
            )?;
            ops::dense_gemv(
                gpu,
                self.k_gemv,
                self.mstate,
                &self.module.markov_w2,
                self.mbias,
                self.vocab,
                mr as u32,
                stream,
            )?;
            let row = self.logits5.offset(r * self.vocab as usize * 2);
            ops::residual_add(
                gpu,
                self.k_residual_add,
                row,
                self.mbias,
                self.vocab,
                stream,
            )?;
            ops::argmax_bf16(
                gpu,
                self.k_argmax,
                row,
                self.scratch_u32,
                self.vocab,
                stream,
            )?;

            // Confidence: sigmoid(proj · [hidden_row | markov_state]).
            gpu.copy_d2d_async(self.f5.offset(r * hu * 2), self.conf_in, hu * 2, stream)?;
            gpu.copy_d2d_async(self.mstate, self.conf_in.offset(hu * 2), mr * 2, stream)?;
            ops::dense_gemv(
                gpu,
                self.k_gemv,
                self.conf_in,
                &self.module.confidence_proj,
                self.mbias,
                1,
                (hu + mr) as u32,
                stream,
            )?;

            gpu.synchronize(stream)?;
            let mut tb = [0u8; 4];
            gpu.copy_d2h(self.scratch_u32, &mut tb)?;
            let tok = u32::from_le_bytes(tb);
            let mut cb = [0u8; 2];
            gpu.copy_d2h(self.mbias, &mut cb)?;
            let logit = f32::from_bits((u16::from_le_bytes(cb) as u32) << 16);
            drafts.push(tok);
            confs.push(1.0 / (1.0 + (-logit).exp()));
            prev = tok;
        }
        Ok((drafts, confs))
    }
}

/// Per-sequence DSpark proposer state. The ring itself lives on the head
/// (single-stream serving); this tracks how far it has been seeded so the
/// propose site can catch up over verify-accepted positions from the
/// model's capture buffer.
pub struct DsparkProposerState {
    /// Highest sequence position whose `main_kv` is in the rings; -1 = none.
    pub last_seeded: i64,
}

impl crate::speculative::ProposerState for DsparkProposerState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl DsparkDraftHead {
    /// Capture-buffer geometry, installed post-model-build by the factory:
    /// `[num_capture_layers, rows, h]` BF16, rows at sequence positions.
    pub fn set_capture(&mut self, buf: DevicePtr, rows: usize) {
        self.capture_buf = buf;
        self.capture_rows = rows;
    }

    fn captures_at(&self, pos: usize) -> [DevicePtr; 3] {
        let h = self.h as usize;
        [
            self.capture_buf.offset(pos * h * 2),
            self.capture_buf.offset((self.capture_rows + pos) * h * 2),
            self.capture_buf.offset((2 * self.capture_rows + pos) * h * 2),
        ]
    }
}

impl crate::speculative::DraftProposer for DsparkDraftHead {
    fn alloc_state(
        &self,
        _gpu: &dyn GpuBackend,
    ) -> Result<Box<dyn crate::speculative::ProposerState>> {
        Ok(Box::new(DsparkProposerState { last_seeded: -1 }))
    }

    /// Contract at the propose site (matches the V4 MTP head's convention):
    /// `last_token` was just committed at sequence position `position`; the
    /// captures of the step that GENERATED it live at row `position - 1`.
    /// Ring catch-up walks every unseeded position up to there — verify
    /// captured all its rows, so multi-accept gaps are covered.
    fn propose(
        &self,
        last_token: u32,
        _target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn crate::speculative::ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        _draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        _target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        if grammar_bitmask.is_some() {
            // Grammar-constrained drafting is not wired for the block drafter
            // yet; declining to draft is lossless (plain decode).
            return Ok(vec![]);
        }
        let st = state
            .as_any_mut()
            .downcast_mut::<DsparkProposerState>()
            .context("DSpark propose: wrong state type")?;
        if self.capture_buf.is_null() || position == 0 {
            return Ok(vec![]);
        }
        let p = position - 1;
        if p >= self.capture_rows {
            return Ok(vec![]);
        }
        let gpu = ctx.gpu;
        // Catch-up: seed every position since the last propose (multi-accept
        // rows were captured by the verify forward).
        let from = (st.last_seeded + 1).max(0) as usize;
        for q in from..p {
            self.seed_position(gpu, self.captures_at(q), q, stream)?;
        }
        let (drafts, confs) = self.propose_block(gpu, ctx, self.captures_at(p), last_token, p, stream)?;
        st.last_seeded = p as i64;

        // Confidence gate (ATLAS_DSPARK_CONF, 0 = ungated). The offline
        // calibration (docs/dspark_port.md) shows 0.9 over-truncates on our
        // cost model; default ungated until tuned in-server.
        let thr: f32 = std::env::var("ATLAS_DSPARK_CONF")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0);
        let mut keep = drafts.len().min(num_drafts.max(1));
        if thr > 0.0 {
            let mut k = 0;
            while k < keep && confs[k] >= thr {
                k += 1;
            }
            keep = k;
        }
        Ok(drafts[..keep].to_vec())
    }

    /// Nothing to roll back: the rings hold only committed-position
    /// `main_kv` rows (draft rows are never persisted), and the next
    /// propose's catch-up walk seeds exactly the accepted positions.
    fn after_verify(
        &self,
        _num_accepted: usize,
        _state: &mut dyn crate::speculative::ProposerState,
        _stream: u64,
    ) -> Result<()> {
        Ok(())
    }

    /// Ring seeding over the prompt: the chunked-prefill capture wrote every
    /// prompt position's hc-mean at its sequence row.
    fn prefill_drafter(
        &self,
        prompt_tokens: &[u32],
        _hiddens: DevicePtr,
        state: &mut dyn crate::speculative::ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let st = state
            .as_any_mut()
            .downcast_mut::<DsparkProposerState>()
            .context("DSpark prefill: wrong state type")?;
        if self.capture_buf.is_null() {
            return Ok(0);
        }
        let n = prompt_tokens.len().min(self.capture_rows);
        // Only the last `window` positions can ever be attended; skipping the
        // rest keeps re-prefill O(window) at long prompts.
        let start = n.saturating_sub(self.module.params.window);
        for q in start..n {
            self.seed_position(ctx.gpu, self.captures_at(q), q, stream)?;
        }
        st.last_seeded = n as i64 - 1;
        Ok(n - start)
    }
}
