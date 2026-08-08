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
    /// M≤8 row-scaled FP8 tile. The lm_head tail used to run one GEMV per
    /// block row, re-streaming the whole `[vocab, h]` FP8 mirror `block`
    /// times; this does all `block` rows in one pass over the weight.
    /// `KernelHandle(0)` when the `w4a16` module is absent — the per-row
    /// GEMV loop is kept as the fallback.
    k_gemm_smallm: KernelHandle,
    k_rms: KernelHandle, // rms_norm_vanilla — HF-exact weights
    k_residual_add: KernelHandle,
    k_hc_expand: KernelHandle,
    k_hc_pre: KernelHandle,
    k_hc_post: KernelHandle,
    k_hc_head: KernelHandle,
    k_argmax: KernelHandle,
    /// Device-indexed row gather (`token_ids[i] → table[token_ids[i], :]`).
    /// Lets the Markov chain read `markov_w1[prev]` without `prev` ever
    /// touching the host. `KernelHandle(0)` → the host-routed fallback.
    k_batched_embed: KernelHandle,
    /// Per-row top-2 over `[block, vocab]` logits. Only the runner-up token
    /// and the top-1/top-2 margin are needed, and only when DDTree is armed
    /// (`ATLAS_DSPARK_TREE=1`); `KernelHandle(0)` disables tree payloads.
    k_top2: KernelHandle,
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
    /// `[1 + B]` u32 greedy chain: slot 0 is the committed token, slot `r+1`
    /// is row `r`'s argmax. Slot `r` is the gather index for row `r`, so the
    /// `prev → markov_w1[prev]` dependency stays device-side.
    tok_dev: DevicePtr,
    /// `[B]` BF16 confidence logits, one per row — lets all `B` rows be read
    /// back in a single D2H after one sync.
    conf_dev: DevicePtr,
    /// `[B, 4]` u32 top-2 quads `(idx1, bits(val1), idx2, bits(val2))`.
    top2_out: DevicePtr,
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
            // ATLAS_LMHEAD_EXACT=1: zero the handle so the block tail takes
            // the per-row GEMV — this GEMM casts BF16 activations to FP8 in-
            // kernel and its near-tie argmax can differ (see impl_a1's gate).
            k_gemm_smallm: if std::env::var("ATLAS_LMHEAD_EXACT").as_deref() == Ok("1") {
                spark_runtime::gpu::KernelHandle(0)
            } else {
                crate::layers::try_kernel(gpu, "w4a16", "fp8_gemm_t_row_scaled_mtile8")
            },
            k_rms: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            k_residual_add: gpu.kernel("residual_add", "bf16_residual_add")?,
            k_hc_expand: gpu.kernel("hyper_connection", "hc_expand")?,
            k_hc_pre: gpu.kernel("hyper_connection", "hc_pre")?,
            k_hc_post: gpu.kernel("hyper_connection", "hc_post")?,
            k_hc_head: gpu.kernel("hyper_connection", "hc_head")?,
            k_argmax: gpu.kernel("argmax", "argmax_bf16")?,
            k_batched_embed: crate::layers::try_kernel(gpu, "embed_from_argmax", "batched_embed"),
            k_top2: crate::layers::try_kernel(gpu, "argmax", "top2_bf16_rows"),
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
            tok_dev: gpu.alloc((b + 1) * 4)?,
            conf_dev: alloc(b)?,
            top2_out: gpu.alloc(b * 16)?,
            capture_buf: DevicePtr::NULL,
            capture_rows: 0,
            module,
        })
    }

    /// `ATLAS_DSPARK_DEBUG=1`, read ONCE.
    ///
    /// This gates 14 `dbg` probes inside `propose_block` plus the `==PROPOSE==`
    /// marker, and `propose` runs every speculative step on the hot path.
    /// `std::env::var` is not free — it takes the process-wide env lock and
    /// allocates a `String` per call — so reading it per probe put ~18 locked
    /// allocations in front of every propose for a flag that is off in
    /// production. Same read-once discipline as every other gate in this file.
    fn debug_enabled() -> bool {
        static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *CACHED.get_or_init(|| std::env::var("ATLAS_DSPARK_DEBUG").as_deref() == Ok("1"))
    }

    /// `ATLAS_DSPARK_RING_DUMP=1`, read once (per-propose host sync when on).
    fn ring_dump_enabled() -> bool {
        static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *CACHED.get_or_init(|| std::env::var("ATLAS_DSPARK_RING_DUMP").as_deref() == Ok("1"))
    }

    /// `ATLAS_DSPARK_PROBE_DUMP=<path>`, read once.
    fn probe_dump_path() -> Option<&'static str> {
        static CACHED: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        CACHED
            .get_or_init(|| std::env::var("ATLAS_DSPARK_PROBE_DUMP").ok())
            .as_deref()
    }

    /// `ATLAS_DSPARK_CONF` confidence threshold, read once (0 = ungated).
    fn conf_threshold() -> f32 {
        static CACHED: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
        *CACHED.get_or_init(|| {
            std::env::var("ATLAS_DSPARK_CONF")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0)
        })
    }

    /// ATLAS_DSPARK_DEBUG=1: sync + print the L2 norm and first values of a
    /// BF16 buffer. Bisects the propose chain against the Python reference.
    fn dbg(&self, gpu: &dyn GpuBackend, label: &str, ptr: DevicePtr, n: usize, stream: u64) {
        if !Self::debug_enabled() {
            return;
        }
        let _ = gpu.synchronize(stream);
        let mut b = vec![0u8; n * 2];
        if gpu.copy_d2h(ptr, &mut b).is_err() {
            return;
        }
        // bhash decides equality — 4-decimal norms hide byte drift (task #45:
        // the norm-based probes falsely cleared every stage before s0.o5).
        let mut fnv: u64 = 0xcbf29ce484222325;
        for &byte in b.iter() {
            fnv ^= byte as u64;
            fnv = fnv.wrapping_mul(0x100000001b3);
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
            "DSPARK_DBG {label}: bhash={fnv:016x} norm={norm:.4} first={:?}",
            &v[..4.min(v.len())]
        );
    }

    #[allow(dead_code)]
    fn dbg_f32(&self, gpu: &dyn GpuBackend, label: &str, ptr: DevicePtr, n: usize, stream: u64) {
        if !Self::debug_enabled() {
            return;
        }
        let _ = gpu.synchronize(stream);
        let mut b = vec![0u8; n * 4];
        if gpu.copy_d2h(ptr, &mut b).is_err() {
            return;
        }
        let mut fnv: u64 = 0xcbf29ce484222325;
        for &byte in b.iter() {
            fnv ^= byte as u64;
            fnv = fnv.wrapping_mul(0x100000001b3);
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
            "DSPARK_DBG {label}: bhash={fnv:016x} norm={norm:.4} first={:?}",
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
    /// Markov-biased greedy chain. Returns `block_size` drafts, their
    /// confidence-head sigmoids (ungated — the caller applies the policy),
    /// and the per-row top-2 quads when DDTree is armed (empty otherwise).
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
    ) -> Result<(Vec<u32>, Vec<f32>, Vec<u32>)> {
        if Self::debug_enabled() {
            eprintln!("DSPARK_DBG ==PROPOSE pos={pos} committed={committed}==");
        }
        self.seed_position(gpu, captures, pos, stream)?;
        self.dbg(gpu, "main_x", self.main_x, self.h as usize, stream);
        self.dbg(
            gpu,
            "ring[0]@slot",
            self.rings[0].offset((pos % self.module.params.window) * HEAD_DIM as usize * 2),
            HEAD_DIM as usize,
            stream,
        );
        // ATLAS_DSPARK_RING_DUMP=1 (task #45): FNV byte-hash of EVERY visible
        // ring slot (stage 0), one line per propose. The attention output was
        // proven to be the first divergent stage with byte-identical probed
        // inputs — the divergence must live in unprobed slots; this names them.
        // Hashes, not norms: 4-decimal norms hide byte drift (hard-won rule).
        if Self::ring_dump_enabled() {
            let vis = (pos + 1).min(self.module.params.window);
            let hd = HEAD_DIM as usize * 2;
            let _ = gpu.synchronize(stream);
            let mut line = format!("RINGHASH pos={pos} s0:");
            let mut buf = vec![0u8; hd];
            for slot in 0..vis {
                if gpu
                    .copy_d2h(self.rings[0].offset(slot * hd), &mut buf)
                    .is_err()
                {
                    break;
                }
                let mut fnv: u64 = 0xcbf29ce484222325;
                for &b in buf.iter() {
                    fnv ^= b as u64;
                    fnv = fnv.wrapping_mul(0x100000001b3);
                }
                line.push_str(&format!(" {slot}={:08x}", (fnv >> 32) as u32));
            }
            eprintln!("{line}");
        }

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

        // Block-row base position. The ds4 reference (Entrpi/ds4 ds4.c:30738)
        // places the committed token (block row 0) at the SAME position as the
        // target hidden — `positions[i+1] = pos + i`, so row 0 sits at `pos`,
        // drafts at pos+1... Our original `pos + 1` put row 0 one position ABOVE
        // the target hidden, the measured +1 draft offset that collapses online
        // acceptance (drafts predict position+1 while the verify checks position).
        // ATLAS_DSPARK_BLK_SHIFT A/Bs it: 0 = ds4-aligned (row0 at pos), 1 = old.
        let blk_shift: i64 = {
            static BS: std::sync::OnceLock<i64> = std::sync::OnceLock::new();
            *BS.get_or_init(|| {
                std::env::var("ATLAS_DSPARK_BLK_SHIFT")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1)
            })
        };
        let blk_pos = ((pos as i64 + blk_shift).max(0) as u32).min(u32::MAX);
        let ring_vis = (blk_pos).min(self.module.params.window as u32);
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

            // Probe prior ring slots + sink to localize the o5 divergence:
            // slot 0 already matches engine; check whether a specific PRIOR
            // slot is grossly wrong (fixable) or all are slightly off (numerics).
            {
                let win = self.module.params.window;
                for back in [1usize, 5, 10] {
                    if pos >= back {
                        let slot = (pos - back) % win;
                        self.dbg(
                            gpu,
                            &format!("s{s}.ring@pos-{back}"),
                            self.rings[s].offset(slot * HEAD_DIM as usize * 2),
                            HEAD_DIM as usize,
                            stream,
                        );
                    }
                }
                self.dbg(gpu, &format!("s{s}.sink"), stage.attn_sink.weight, HEADS as usize, stream);
            }
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
        // `f5` is `[b, h]` and `logits5` is `[b, vocab]`, both contiguous, so
        // the M≤8 tile can take all b rows in one pass over the FP8 mirror
        // instead of streaming `[vocab, h]` once per row. The tile's f32
        // accumulation order differs from the GEMV's, so the drafted tokens
        // can differ in the rare near-tie; served text does not, because
        // verify only accepts a draft that matches the target's own argmax.
        let lmhead_batched = self.k_gemm_smallm.0 != 0
            && b <= 8
            && h.is_multiple_of(32)
            && {
                static LB: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                *LB.get_or_init(|| {
                    std::env::var("ATLAS_DSPARK_LMHEAD_BATCH").as_deref() != Ok("0")
                })
            };
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::info!(
                    "DSpark lm_head: batched={} (kernel={:#x} b={} h={} fp8={} bf16={})",
                    lmhead_batched && self.lm_head_fp8.is_some(),
                    self.k_gemm_smallm.0,
                    b,
                    h,
                    self.lm_head_fp8.is_some(),
                    self.lm_head_bf16.is_some(),
                );
            });
        }
        // ATLAS_DSPARK_LMHEAD_BF16=1 (task #45 A/B, MEASURED NO-OP): forces the
        // engine probe's BF16 lm_head instead of the FP8 mirror. Tested
        // 2026-08-06: accepted 1.02 == FP8 baseline, output hash identical,
        // propose +18ms — the FP8 head does NOT cause the online acceptance
        // collapse (the near-tie flip hypothesis is disproven). Default OFF.
        let force_bf16 = self.lm_head_bf16.is_some() && {
            static FB: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *FB.get_or_init(|| {
                std::env::var("ATLAS_DSPARK_LMHEAD_BF16").as_deref() == Ok("1")
            })
        };
        let fp8_head = if force_bf16 { None } else { self.lm_head_fp8.as_ref() };
        match (fp8_head, lmhead_batched) {
            (Some(fp8), true) => ops::fp8_gemm_row_scaled_smallm(
                gpu,
                self.k_gemm_smallm,
                self.f5,
                fp8,
                self.logits5,
                b,
                self.vocab,
                h,
                stream,
            )?,
            _ => {
                for r in 0..b as usize {
                    let row_in = self.f5.offset(r * hu * 2);
                    let row_out = self.logits5.offset(r * self.vocab as usize * 2);
                    if let Some(fp8) = fp8_head {
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
            }
        }

        // ── Markov-biased greedy chain + confidence ──
        //
        // The chain is `prev → markov_w1[prev] → logits → argmax → prev`.
        // Indexing `markov_w1` on the host makes every row a full pipeline
        // flush: `block` syncs + 2·`block` D2H per propose. `batched_embed`
        // gathers the row from a DEVICE-resident index, so the whole chain
        // stays on the stream and only the final `block` tokens come back.
        // Same kernels in the same order — bit-exact with the host route.
        let mr = self.module.params.markov_rank;
        let chain_on_device = self.k_batched_embed.0 != 0 && {
            static CD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *CD.get_or_init(|| std::env::var("ATLAS_DSPARK_CHAIN_DEV").as_deref() != Ok("0"))
        };
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::info!(
                    "DSpark markov chain: on_device={} (kernel={:#x} block={} mr={})",
                    chain_on_device,
                    self.k_batched_embed.0,
                    b,
                    mr,
                );
            });
        }

        // DDTree (ATLAS_DSPARK_TREE=1): the runner-up token at each row is
        // the branch the tree verify would explore. `residual_add` folds the
        // Markov bias into `logits5` in place, so the top-2 must be taken
        // AFTER the chain loop — the same biased rows the argmax read.
        let tree_on = self.k_top2.0 != 0 && {
            static TR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *TR.get_or_init(|| std::env::var("ATLAS_DSPARK_TREE").as_deref() == Ok("1"))
        };

        let mut drafts = Vec::with_capacity(b as usize);
        let mut confs = Vec::with_capacity(b as usize);
        if chain_on_device {
            // Device-side write, NOT copy_h2d_async(&committed.to_le_bytes()):
            // that rvalue temporary dies at the end of the statement while the
            // async copy may still be queued (first sync is after the whole
            // chain loop), and a torn read here poisons tok_dev[0] — the
            // gather index every subsequent chain row derives from. This was a
            // measured source of greedy non-determinism in the DSpark path.
            gpu.memset_u32_async(self.tok_dev, committed, 1, stream)?;
        }
        let mut prev = committed;
        for r in 0..b as usize {
            // markov_w1[prev] → state; logits_r += markov_w2 · state.
            if chain_on_device {
                ops::batched_embed(
                    gpu,
                    self.k_batched_embed,
                    self.tok_dev.offset(r * 4),
                    self.module.markov_w1.weight,
                    self.mstate,
                    1,
                    mr as u32,
                    stream,
                )?;
            } else {
                gpu.copy_d2d_async(
                    self.module.markov_w1.weight.offset(prev as usize * mr * 2),
                    self.mstate,
                    mr * 2,
                    stream,
                )?;
            }
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
            // On-device: row `r`'s argmax lands in slot `r + 1`, which is the
            // next iteration's gather index. Host route keeps the single slot.
            let tok_out = if chain_on_device {
                self.tok_dev.offset((r + 1) * 4)
            } else {
                self.scratch_u32
            };
            ops::argmax_bf16(gpu, self.k_argmax, row, tok_out, self.vocab, stream)?;

            // Confidence: sigmoid(proj · [hidden_row | markov_state]).
            gpu.copy_d2d_async(self.f5.offset(r * hu * 2), self.conf_in, hu * 2, stream)?;
            gpu.copy_d2d_async(self.mstate, self.conf_in.offset(hu * 2), mr * 2, stream)?;
            let conf_out = if chain_on_device {
                self.conf_dev.offset(r * 2)
            } else {
                self.mbias
            };
            ops::dense_gemv(
                gpu,
                self.k_gemv,
                self.conf_in,
                &self.module.confidence_proj,
                conf_out,
                1,
                (hu + mr) as u32,
                stream,
            )?;

            if chain_on_device {
                continue;
            }
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
        // Queued before the flush so the top-2 rides the chain's single sync.
        if tree_on {
            ops::top2_bf16_rows(
                gpu,
                self.k_top2,
                self.logits5,
                self.top2_out,
                b,
                self.vocab,
                stream,
            )?;
        }
        if chain_on_device {
            // One flush for the whole chain instead of `block` of them.
            gpu.synchronize(stream)?;
            let mut tb = vec![0u8; b as usize * 4];
            gpu.copy_d2h(self.tok_dev.offset(4), &mut tb)?;
            let mut cb = vec![0u8; b as usize * 2];
            gpu.copy_d2h(self.conf_dev, &mut cb)?;
            for r in 0..b as usize {
                drafts.push(u32::from_le_bytes(tb[r * 4..r * 4 + 4].try_into().unwrap()));
                let bits = u16::from_le_bytes(cb[r * 2..r * 2 + 2].try_into().unwrap());
                let logit = f32::from_bits((bits as u32) << 16);
                confs.push(1.0 / (1.0 + (-logit).exp()));
            }
        }
        let top2 = if tree_on {
            if !chain_on_device {
                gpu.synchronize(stream)?;
            }
            let mut w = vec![0u8; b as usize * 16];
            gpu.copy_d2h(self.top2_out, &mut w)?;
            w.chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        } else {
            Vec::new()
        };
        let _ = prev;
        Ok((drafts, confs, top2))
    }
}

/// DDTree shape knobs, mirroring the DFlash names so a sweep can drive both
/// drafters: `(max_branches, tail_len, margin_gate)`.
///
/// The margin gate is the economics knob. A tree step forfeits the flat CUDA
/// graph and pays for its extra rows, so it only pays off where an early
/// death is likely — i.e. where the drafter's top-1 barely beat its top-2.
fn tree_shape() -> (usize, usize, f32) {
    static V: std::sync::OnceLock<(usize, usize, f32)> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let env = |k: &str, d: usize, hi: usize| {
            std::env::var(k)
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(d)
                .clamp(1, hi)
        };
        (
            env("ATLAS_DSPARK_TREE_BRANCHES", 1, 3),
            env("ATLAS_DSPARK_TREE_TAIL", 1, 3),
            std::env::var("ATLAS_DSPARK_TREE_MARGIN")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(2.5),
        )
    })
}

/// Per-sequence DSpark proposer state. The ring itself lives on the head
/// (single-stream serving); this tracks how far it has been seeded so the
/// propose site can catch up over verify-accepted positions from the
/// model's capture buffer.
pub struct DsparkProposerState {
    /// Highest sequence position whose `main_kv` is in the rings; -1 = none.
    pub last_seeded: i64,
    /// The first decode position (= prompt length), set by `prefill_drafter`.
    /// Task #45 byte-proof: this position's capture is written by the graphed
    /// BOOTSTRAP step and is poisoned — it was the ONLY ring byte-difference
    /// between the live server and the 2.38-tok/step engine probe (which
    /// leaves the slot zero), and it forked every draft chain. The propose
    /// path keeps this slot ZERO (skip seeding + post-propose memset), which
    /// took draft[0] to 9/9 engine parity and accepted 1.02 → 1.10.
    pub boundary_pos: i64,
    /// DDTree payload for the drafts just proposed (`ATLAS_DSPARK_TREE=1`).
    /// Consumed by `dflash_take_tree_payload`; `None` keeps the flat path.
    pub pending_tree_payload: Option<crate::layers::DDTreePayload>,
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
        Ok(Box::new(DsparkProposerState {
            last_seeded: -1,
            boundary_pos: -1,
            pending_tree_payload: None,
        }))
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
        // Block-position alignment (task #45 — THE online acceptance fix).
        // The official reference (`bench/.../dspark_probe/probe.py`) conditions
        // stage 0 on `main_hidden@p` with the block forwarded at position `p`,
        // so `forward_embed`'s committed row (the NEXT token, `tok@p+1`) lands
        // at block position `p+1` and the drafts predict `p+2…`. In-engine the
        // committed frontier is `position` and the last committed token is
        // `tok@(position-1)`; matching the reference's "row-0 token sits one
        // position below where we draft" means the block base must be
        // `pos = position-2`: then row 0 (last_token = tok@position-1) sits at
        // `pos+1 = position-1` (its TRUE position) and the drafts fill
        // `position…`, conditioned on `hidden@(position-2)` (which predicts
        // position-1 = last_token) — the reference's exact contract. The old
        // `position-1` shifted the whole block one slot high (row 0 at
        // `position`, drafts at `position+1`) AND read the bonus-poisoned
        // capture row, capping online acceptance at ~1.0 vs 3.8 offline.
        // `ATLAS_DSPARK_POS_BEHIND` A/Bs the base (2 = fix, 1 = old).
        let behind: usize = {
            static PB: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            *PB.get_or_init(|| {
                std::env::var("ATLAS_DSPARK_POS_BEHIND")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1)
            })
        };
        if self.capture_buf.is_null() || position < behind {
            return Ok(vec![]);
        }
        let p = position - behind;
        if p >= self.capture_rows {
            return Ok(vec![]);
        }
        let gpu = ctx.gpu;
        // Catch-up: seed every position since the last propose (multi-accept
        // rows were captured by the verify forward).
        //
        // ATLAS_DSPARK_FULL_RESEED=1 (task #45): rebuild the ENTIRE active
        // sliding window [p+1-window .. p) from the current capture buffer every
        // propose, instead of only the incremental [last_seeded+1 .. p). The
        // engine probe (dspark_engine_probe) hits 2.38 tok/step on the SAME
        // online captures but the live server gets ~1.0 — the only difference is
        // ring STATE. The incremental catch-up never refreshes positions already
        // seeded, so a ring slot that received a rejected-trajectory capture (or
        // draft-row pollution from a prior propose_block) stays stale inside the
        // window the drafter attends over. A full reseed from the committed
        // capture buffer makes the live ring match the clean engine-probe ring.
        static FULL_RESEED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let full_reseed =
            *FULL_RESEED.get_or_init(|| std::env::var("ATLAS_DSPARK_FULL_RESEED").as_deref() == Ok("1"));
        let win = self.module.params.window;
        let from = if full_reseed {
            (p + 1).saturating_sub(win)
        } else {
            (st.last_seeded + 1).max(0) as usize
        };
        // Boundary fix (task #45, byte-proven): never seed the bootstrap
        // position — its capture is poisoned and this slot must stay zero
        // (see `DsparkProposerState::boundary_pos`). ATLAS_DSPARK_BOUNDARY_FIX=0
        // restores the old behavior for A/B.
        let boundary_fix = st.boundary_pos >= 0 && {
            static BF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *BF.get_or_init(|| {
                std::env::var("ATLAS_DSPARK_BOUNDARY_FIX").as_deref() != Ok("0")
            })
        };
        for q in from..p {
            if boundary_fix && q as i64 == st.boundary_pos {
                continue;
            }
            self.seed_position(gpu, self.captures_at(q), q, stream)?;
        }
        // ATLAS_DSPARK_PROBE_DUMP=<path>: append the exact capture bytes the
        // drafter reads at `p` (the verify-capture the online path feeds
        // propose), so it can be diffed against the plain-decode probe dump at
        // the same sequence position. Debug-only host sync.
        if let Some(path) = Self::probe_dump_path() {
            use std::io::Write;
            let h = self.h as usize;
            let caps = self.captures_at(p);
            let mut host = vec![0u8; 3 * h * 2];
            gpu.synchronize(stream)?;
            for (i, c) in caps.iter().enumerate() {
                gpu.copy_d2h(*c, &mut host[i * h * 2..(i + 1) * h * 2])?;
            }
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                let _ = f.write_all(&(p as u32).to_le_bytes());
                let _ = f.write_all(&(last_token).to_le_bytes());
                let _ = f.write_all(&host);
            }
        }
        // ATLAS_DSPARK_CAP_SHIFT: A/B the capture-row alignment fed to the
        // drafter. The online pairing is (captures_at(p), last_token) where
        // last_token sits at index p+1 — the hidden is one position BEHIND
        // the chain token, a distribution the drafter may not have been
        // trained on (offline eval scores 3.69 tok/step; online ~1.0-1.4).
        // shift=-1 feeds an even older hidden; 0 = current; the off-by-one
        // family already produced the compressor-replay bug (task #45).
        let cap_shift: i64 = {
            static CS: std::sync::OnceLock<i64> = std::sync::OnceLock::new();
            *CS.get_or_init(|| {
                std::env::var("ATLAS_DSPARK_CAP_SHIFT")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0)
            })
        };
        let p_eff = ((p as i64 + cap_shift).max(0) as usize).min(self.capture_rows - 1);
        let (drafts, confs, top2) =
            self.propose_block(gpu, ctx, self.captures_at(p_eff), last_token, p, stream)?;
        // ATLAS_DSPARK_ZERO_SLOT=<abs pos> (task #45 equivalence test): after
        // each propose, zero that position's ring slot in every stage. The
        // engine probe's ring has a never-seeded (zero) hole at the first
        // decode position; the live ring holds a real capture there — the ONLY
        // byte difference between the two rings (RINGHASH proof). Zeroing it
        // makes the live ring byte-identical to the engine's, so live drafts
        // must reproduce the engine's 2.38 tok/step if the analysis is right.
        {
            static ZS: std::sync::OnceLock<i64> = std::sync::OnceLock::new();
            let zs = *ZS.get_or_init(|| {
                std::env::var("ATLAS_DSPARK_ZERO_SLOT")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(-1)
            });
            // Auto boundary fix: keep the bootstrap position's slot zero.
            // propose_block itself seeds `pos` when the propose lands on the
            // boundary, so the memset must follow it. Idempotent afterwards.
            let target = if zs >= 0 {
                zs
            } else if boundary_fix {
                st.boundary_pos
            } else {
                -1
            };
            if target >= 0 {
                let win = self.module.params.window;
                let slot = (target as usize) % win;
                let hd = HEAD_DIM as usize * 2;
                for ring in &self.rings {
                    gpu.memset_u32_async(ring.offset(slot * hd), 0, hd / 4, stream)?;
                }
            }
        }
        // ATLAS_DSPARK_DRAFT_LOG=1 (task #45): log the LIVE drafts per propose
        // position so they can be diffed against the engine probe's drafts on
        // the SAME committed captures — the first divergence at byte-exact
        // inputs pinpoints the propose_block execution-context bug.
        static DRAFT_LOG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *DRAFT_LOG.get_or_init(|| std::env::var("ATLAS_DSPARK_DRAFT_LOG").as_deref() == Ok("1")) {
            tracing::info!("DRAFTLOG pos={p} last_token={last_token} drafts={drafts:?}");
        }
        st.last_seeded = p as i64;

        // Confidence gate (ATLAS_DSPARK_CONF, 0 = ungated). The offline
        // calibration (docs/dspark_port.md) shows 0.9 over-truncates on our
        // cost model; default ungated until tuned in-server.
        let thr: f32 = Self::conf_threshold();
        let mut keep = drafts.len().min(num_drafts.max(1));
        if thr > 0.0 {
            let mut k = 0;
            while k < keep && confs[k] >= thr {
                k += 1;
            }
            keep = k;
        }
        let kept = drafts[..keep].to_vec();

        // ── DDTree payload (ATLAS_DSPARK_TREE=1) ──
        //
        // The spine is exactly the drafts the verify will see, which is what
        // `try_decode_verify_tree`'s staleness guard requires. Each cliff
        // adds a sibling row carrying the runner-up token plus a re-rooted
        // tail, so a step that dies at the cliff still commits the branch
        // instead of stopping. Chain acceptance decays 84.5% -> 43.1% by
        // position while verify costs only ~16ms per extra row, so width is
        // the cheap axis. Gate on the top-1/top-2 margin: on confident steps
        // the branch is wasted rows AND forfeits the flat CUDA graph.
        st.pending_tree_payload = None;
        if !top2.is_empty() && kept.len() >= 2 {
            // Transparency gate (ATLAS_DSPARK_TREE_DEGEN=1): force the fork
            // token to the spine draft at the same depth. The branch rows
            // then verify exactly what the spine rows verify, through the
            // scratch KV, so the committed text MUST be byte-identical to
            // the flat path. Any drift is a bug in the tree metadata, the
            // COW block table, or the per-layer branch re-seed — not a
            // modelling difference. Mirrors ATLAS_DFLASH_TREE_DEGEN.
            static DEGEN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let degen =
                *DEGEN.get_or_init(|| std::env::var("ATLAS_DSPARK_TREE_DEGEN").as_deref() == Ok("1"));
            let (max_branches, tail_cfg, margin_gate) = tree_shape();
            let mut cliffs: Vec<(usize, u32, f32)> = Vec::new();
            // Draft `di` came from logits row `di` (no row-0 drop here — the
            // DSpark block emits one draft per row, unlike DFlash).
            for di in 0..kept.len().saturating_sub(1) {
                let margin = f32::from_bits(top2[di * 4 + 1]) - f32::from_bits(top2[di * 4 + 3]);
                let fork = if degen { kept[di] } else { top2[di * 4 + 2] };
                if degen || (fork != kept[di] && margin < margin_gate) {
                    cliffs.push((di, fork, margin));
                }
            }
            cliffs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
            cliffs.truncate(max_branches);
            if !cliffs.is_empty() {
                let branches: Vec<crate::layers::dflash_head::ddtree::FreeSlotBranch> = cliffs
                    .iter()
                    .map(
                        |&(di, fork, _)| crate::layers::dflash_head::ddtree::FreeSlotBranch {
                            cliff_depth: di + 1,
                            fork_token: fork,
                            tail: kept[di + 1..(di + 1 + tail_cfg).min(kept.len())].to_vec(),
                        },
                    )
                    .collect();
                // K_t = bonus row + spine + Σ(fork + tail); the verify arena
                // holds 20 rows (verify_d_tree.rs TREE_MAX_ROWS).
                let want = kept.len() + branches.iter().map(|b| 1 + b.tail.len()).sum::<usize>();
                let payload = crate::layers::dflash_head::ddtree::build_free_slots_payload(
                    &kept,
                    &branches,
                    want.min(19),
                );
                st.pending_tree_payload = Some(payload);
            }
        }
        Ok(kept)
    }

    /// Nothing to roll back: the rings hold only committed-position
    /// `main_kv` rows (draft rows are never persisted), and the next
    /// propose's catch-up walk seeds exactly the accepted positions.
    fn after_verify(
        &self,
        _num_accepted: usize,
        state: &mut dyn crate::speculative::ProposerState,
        _stream: u64,
    ) -> Result<()> {
        // Task #45 acceptance fix: rewind the seed frontier so the next
        // propose RE-SEEDS the ring rows for every position the verify just
        // re-captured. Without this, ring rows seeded during earlier proposes
        // keep hiddens from REJECTED trajectories forever: the verify
        // overwrites the capture buffer rows with the corrected hiddens, but
        // `last_seeded` had already advanced past them, so the drafter's
        // attention window stays poisoned on every partial accept (~most
        // steps) — a mechanical cap on acceptance regardless of capture
        // fidelity. Re-seeding is idempotent (pure copy from the capture
        // buffer), so rewinding a couple of rows extra is harmless.
        if let Some(st) = state
            .as_any_mut()
            .downcast_mut::<DsparkProposerState>()
        {
            let rewind = self.block as i64 + 2;
            st.last_seeded = (st.last_seeded - rewind).max(-1);
        }
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
        // The first decode position's capture (written by the graphed
        // bootstrap step, position n) is poisoned — see `boundary_pos` docs.
        // Recording it here lets propose() keep that ring slot zero.
        st.boundary_pos = n as i64;
        Ok(n - start)
    }
}
