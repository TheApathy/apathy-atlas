// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 5-7 of `Qwen3SsmLayer::decode_batched_inner`: Conv1d + L2 norm +
//! GDN per-token (with intermediate checkpoints). Extracted from
//! `trait_decode_batched.rs` to keep the parent file under 500 LoC.
//! Dispatches one of the fused K=2/3/4/17 paths or the sequential
//! per-token fallback. All buffers + state are owned by the caller; this
//! function only mutates `ssm_state.h_state`, `ssm_state.conv_state`,
//! their intermediates, `conv_out_buf`, and `gdn_out_buf`.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::{Qwen3SsmLayer, SsmLayerState};
use crate::layer::ForwardContext;
use crate::layers::ops;

#[allow(clippy::too_many_arguments)]
pub(super) struct ConvGdnArgs {
    pub num_tokens: usize,
    pub deinterleaved: DevicePtr,
    pub gates_buf: DevicePtr,
    pub conv_out_buf: DevicePtr,
    pub gdn_out_buf: DevicePtr,
    pub h_bytes: usize,
    pub conv_bytes: usize,
    pub qkvz_size: usize,
    pub conv_dim: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub d_conv: usize,
    pub qk_ch: u32,
    pub nk: usize,
    pub nv: usize,
    pub kd: usize,
    pub vd: usize,
    pub bf16: usize,
    pub fp32: usize,
    pub stream: u64,
}

impl Qwen3SsmLayer {
    /// Run conv1d_update_l2norm + GDN over `num_tokens` (multi-token decode
    /// / MTP verify). Picks the K=2/3/4/17 fused WY path if available,
    /// otherwise falls back to the sequential per-token gdn_decode loop.
    pub(super) fn decode_batched_conv_gdn(
        &self,
        ssm_state: &mut SsmLayerState,
        ctx: &ForwardContext,
        args: &ConvGdnArgs,
    ) -> Result<()> {
        let ConvGdnArgs {
            num_tokens,
            deinterleaved,
            gates_buf,
            conv_out_buf,
            gdn_out_buf,
            h_bytes,
            conv_bytes,
            qkvz_size,
            conv_dim,
            key_dim,
            value_dim: _,
            d_conv,
            qk_ch,
            nk,
            nv,
            kd,
            vd,
            bf16,
            fp32,
            stream,
        } = *args;

        if num_tokens == 4 {
            // ── K=4 fused path: conv1d+L2norm sequential, GDN WY4 ──
            for t in 0..4u32 {
                let qkv_t = deinterleaved.offset(t as usize * qkvz_size * bf16);
                let conv_out_t = conv_out_buf.offset(t as usize * conv_dim * bf16);
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    self.conv1d_l2norm_k,
                    ssm_state.conv_state,
                    qkv_t,
                    &self.ssm.conv1d,
                    conv_out_t,
                    conv_dim as u32,
                    d_conv as u32,
                    1,
                    qk_ch,
                    kd as u32,
                    1e-6,
                    stream,
                )?;
                ctx.gpu.copy_d2d_async(
                    ssm_state.conv_state,
                    ssm_state.conv_state_intermediates[t as usize],
                    conv_bytes,
                    stream,
                )?;
            }

            // WY-chunkwise GDN: 2-pass algorithm for 4-token verification.
            let q_ptr = conv_out_buf;
            let k_ptr = conv_out_buf.offset(key_dim * bf16);
            let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
            let gate_ptr = gates_buf;
            let beta_ptr = gates_buf.offset(nv * fp32);
            ops::gdn_decode_wy4(
                ctx.gpu,
                self.gdn_wy4_k,
                ssm_state.h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_out_buf,
                ssm_state.h_state_intermediates[0],
                ssm_state.h_state_intermediates[1],
                ssm_state.h_state_intermediates[2],
                1, // batch_size
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32, // qk_stride
                conv_dim as u32, // v_stride
                (nv * 2) as u32, // gb_stride
                stream,
            )?;
        } else if num_tokens == 3 {
            // ── K=3 fused path: conv1d+L2norm per token, GDN WY3 ──
            for t in 0..3u32 {
                let qkv_t = deinterleaved.offset(t as usize * qkvz_size * bf16);
                let conv_out_t = conv_out_buf.offset(t as usize * conv_dim * bf16);
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    self.conv1d_l2norm_k,
                    ssm_state.conv_state,
                    qkv_t,
                    &self.ssm.conv1d,
                    conv_out_t,
                    conv_dim as u32,
                    d_conv as u32,
                    1,
                    qk_ch,
                    kd as u32,
                    1e-6,
                    stream,
                )?;
                ctx.gpu.copy_d2d_async(
                    ssm_state.conv_state,
                    ssm_state.conv_state_intermediates[t as usize],
                    conv_bytes,
                    stream,
                )?;
            }

            let q_ptr = conv_out_buf;
            let k_ptr = conv_out_buf.offset(key_dim * bf16);
            let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
            let gate_ptr = gates_buf;
            let beta_ptr = gates_buf.offset(nv * fp32);
            ops::gdn_decode_wy3(
                ctx.gpu,
                self.gdn_wy3_k,
                ssm_state.h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_out_buf,
                ssm_state.h_state_intermediates[0],
                ssm_state.h_state_intermediates[1],
                1, // batch_size
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32, // qk_stride
                conv_dim as u32, // v_stride
                (nv * 2) as u32, // gb_stride
                stream,
            )?;
        } else if num_tokens == 2 {
            // ── K=2 fused path: conv1d sequential, L2 norm sequential, GDN chunk2 ──
            let qkv_0 = deinterleaved;
            let conv_out_0 = conv_out_buf;
            ops::conv1d_update_l2norm(
                ctx.gpu,
                self.conv1d_l2norm_k,
                ssm_state.conv_state,
                qkv_0,
                &self.ssm.conv1d,
                conv_out_0,
                conv_dim as u32,
                d_conv as u32,
                1,
                qk_ch,
                kd as u32,
                1e-6,
                stream,
            )?;
            ctx.gpu.copy_d2d_async(
                ssm_state.conv_state,
                ssm_state.conv_state_intermediates[0],
                conv_bytes,
                stream,
            )?;

            let qkv_1 = deinterleaved.offset(qkvz_size * bf16);
            let conv_out_1 = conv_out_buf.offset(conv_dim * bf16);
            ops::conv1d_update_l2norm(
                ctx.gpu,
                self.conv1d_l2norm_k,
                ssm_state.conv_state,
                qkv_1,
                &self.ssm.conv1d,
                conv_out_1,
                conv_dim as u32,
                d_conv as u32,
                1,
                qk_ch,
                kd as u32,
                1e-6,
                stream,
            )?;
            ctx.gpu.copy_d2d_async(
                ssm_state.conv_state,
                ssm_state.conv_state_intermediates[1],
                conv_bytes,
                stream,
            )?;

            let q_ptr = conv_out_buf;
            let k_ptr = conv_out_buf.offset(key_dim * bf16);
            let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
            let gate_ptr = gates_buf;
            let beta_ptr = gates_buf.offset(nv * fp32);
            // ATLAS_GDN_CHUNK2=1: use the registered-but-unused
            // gated_delta_rule_chunk2 kernel (3-pass H_state vs wy2's 4-pass).
            // Atlas docs: "Reads H_0 once, computes both outputs and H_2 in
            // 3 passes." Identical signature to wy2 — swap is just kernel handle.
            let use_chunk2 = std::env::var("ATLAS_GDN_CHUNK2")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            let (k2_kernel, k2_op): (_, fn(_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _) -> _) =
                if use_chunk2 && self.gdn_chunk2_k.0 != 0 {
                    (self.gdn_chunk2_k, ops::gdn_decode_chunk2)
                } else {
                    (self.gdn_wy2_k, ops::gdn_decode_wy2)
                };
            k2_op(
                ctx.gpu,
                k2_kernel,
                ssm_state.h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_out_buf,
                ssm_state.h_state_intermediates[0],
                1, // batch_size
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32, // qk_stride
                conv_dim as u32, // v_stride
                (nv * 2) as u32, // gb_stride
                stream,
            )?;
        } else if let (Some(parent_ids_dev), true) = (
            ctx.ddtree_parent_ids_dev,
            self.gdn_tree_k.0 != 0
                && std::env::var("ATLAS_FORCE_WY17").ok().as_deref() != Some("1"),
        ) {
            static DISPATCH_DBG: std::sync::atomic::AtomicUsize =
                std::sync::atomic::AtomicUsize::new(0);
            let n = DISPATCH_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 3 {
                tracing::info!(
                    "M8A dispatch FIRES #{n}: num_tokens={} parent_ids_dev={:?} gdn_tree_k=non-null",
                    num_tokens, parent_ids_dev
                );
            }
            // ── M8A: DDTree tree-aware GDN verify ──
            // parent_ids_dev is a [num_tokens × i32] device tensor uploaded by
            // verify_d.rs from a.pending_tree_payload before the layer loop.
            // Each token's state load follows parent_ids[i] instead of i-1,
            // letting the verifier walk non-flat tree branches.
            for t in 0..(num_tokens as u32) {
                let qkv_t = deinterleaved.offset(t as usize * qkvz_size * bf16);
                let conv_out_t = conv_out_buf.offset(t as usize * conv_dim * bf16);
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    self.conv1d_l2norm_k,
                    ssm_state.conv_state,
                    qkv_t,
                    &self.ssm.conv1d,
                    conv_out_t,
                    conv_dim as u32,
                    d_conv as u32,
                    1,
                    qk_ch,
                    kd as u32,
                    1e-6,
                    stream,
                )?;
                ctx.gpu.copy_d2d_async(
                    ssm_state.conv_state,
                    ssm_state.conv_state_intermediates[t as usize],
                    conv_bytes,
                    stream,
                )?;
            }
            let q_ptr = conv_out_buf;
            let k_ptr = conv_out_buf.offset(key_dim * bf16);
            let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
            let gate_ptr = gates_buf;
            let beta_ptr = gates_buf.offset(nv * fp32);
            // Use the FIRST per-token intermediate slot as the base for the
            // tree kernel's contiguous `h_state_inter[T, ...]` output layout.
            // Atlas allocates intermediates[0..num_tokens] contiguously per
            // (b, vh) — the kernel writes inter[t] = base + t * hv directly.
            // ── M8A precision dump (one-shot, env-gated) ──
            // ATLAS_M8A_DUMP=1 + ATLAS_M8A_DUMP_LAYER=N writes:
            //   /tmp/m8a_dump_q.bin           [T*qk_stride] BF16
            //   /tmp/m8a_dump_k.bin           [T*qk_stride] BF16
            //   /tmp/m8a_dump_v.bin           [T*v_stride]  BF16
            //   /tmp/m8a_dump_gate.bin        [T*gb_stride] FP32
            //   /tmp/m8a_dump_beta.bin        [T*gb_stride] FP32
            //   /tmp/m8a_dump_parent_ids.bin  [T]           i32
            //   /tmp/m8a_dump_h_in.bin        [nv*kd*vd]    FP32 (h_state BEFORE kernel)
            //   /tmp/m8a_dump_h_out.bin       [T*nv*kd*vd]  FP32 (h_state_intermediates AFTER)
            //   /tmp/m8a_dump_output.bin      [T*nv*v_dim]  BF16
            //   /tmp/m8a_dump_meta.json       dims + strides for python ref
            // After kernel runs, set ATLAS_M8A_DUMP_DONE marker so subsequent
            // calls skip. Python ref consumes the dump.
            static DUMP_DONE: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            let dump_enabled = std::env::var("ATLAS_M8A_DUMP").ok().as_deref() == Some("1")
                && !DUMP_DONE.load(std::sync::atomic::Ordering::Relaxed);
            let dump_layer_match = std::env::var("ATLAS_M8A_DUMP_LAYER")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .map_or(true, |target| target == 0usize);
            let dump_now = dump_enabled && dump_layer_match;

            if dump_now {
                // Dump inputs BEFORE the kernel runs.
                let qk_bytes_per = qkvz_size * bf16;
                let v_bytes_per = qkvz_size * bf16; // q,k,v all same size in deinterleaved layout
                let _ = v_bytes_per;
                let q_total = (num_tokens as usize) * (conv_dim as usize) * bf16;
                let v_total = (num_tokens as usize) * (conv_dim as usize) * bf16;
                let gb_total = (num_tokens as usize) * nv * fp32;
                let h_total = nv * (kd as usize) * (vd as usize) * fp32;
                let h_inter_total = (num_tokens as usize) * h_total;

                let dump = |name: &str, ptr: spark_runtime::gpu::DevicePtr, n: usize| -> Result<()> {
                    let mut buf = vec![0u8; n];
                    ctx.gpu.synchronize(stream)?;
                    ctx.gpu.copy_d2h(ptr, &mut buf)?;
                    let path = format!("/tmp/m8a_dump_{name}.bin");
                    std::fs::write(&path, &buf)
                        .map_err(|e| anyhow::anyhow!("m8a dump write {path}: {e}"))?;
                    Ok(())
                };
                dump("q", q_ptr, q_total)?;
                dump("k", k_ptr, q_total)?;
                dump("v", v_ptr, v_total)?;
                dump("gate", gate_ptr, gb_total)?;
                dump("beta", beta_ptr, gb_total)?;
                // parent_ids has length num_tokens (verify K = γ+1). Now correctly
                // sized post-fix in set_ddtree_parent_ids (kernel-frame with leading -1).
                let parent_ids_bytes = (num_tokens as usize) * 4;
                dump("parent_ids", parent_ids_dev, parent_ids_bytes)?;
                dump("h_in", ssm_state.h_state, h_total)?;

                let meta = serde_json::json!({
                    "0usize": 0usize,
                    "num_tokens": num_tokens,
                    "batch_size": 1,
                    "num_k_heads": nk,
                    "num_v_heads": nv,
                    "k_dim": kd,
                    "v_dim": vd,
                    "qk_stride": conv_dim,
                    "v_stride": conv_dim,
                    "gb_stride": nv * 2,
                    "h_per_token_bytes": h_total,
                    "h_inter_total_bytes": h_inter_total,
                    "qk_bytes_per_token": qk_bytes_per,
                });
                std::fs::write("/tmp/m8a_dump_meta.json", serde_json::to_string_pretty(&meta)?)
                    .map_err(|e| anyhow::anyhow!("meta write: {e}"))?;
            }

            // wy17 uses inter_stride_floats = h_bytes/4 = nv*kd*vd. Match it
            // so post-verify commit reads from the right per-token slot.
            let inter_stride_floats = (h_bytes / 4) as u32;

            // ── Inline A/B: when M8A_VS_WY17=1, run wy17 first on same inputs
            // (only when num_tokens==17 and wy17 available), dump its output,
            // then restore h_state and run tree_wy normally. Lets us bit-diff
            // both kernels' outputs from IDENTICAL inputs.
            let ab_diff = dump_now
                && num_tokens == 17
                && self.gdn_wy17_k.0 != 0
                && std::env::var("ATLAS_M8A_VS_WY17").ok().as_deref() == Some("1");
            if ab_diff {
                // Backup h_state to host before wy17 mutates it.
                let h_total = nv * (kd as usize) * (vd as usize) * fp32;
                let mut h_backup = vec![0u8; h_total];
                ctx.gpu.synchronize(stream)?;
                ctx.gpu.copy_d2h(ssm_state.h_state, &mut h_backup)?;
                // Run wy17 — it will write to h_state_intermediates AND h_state.
                ops::gdn_decode_wy17(
                    ctx.gpu,
                    self.gdn_wy17_k,
                    ssm_state.h_state,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gate_ptr,
                    beta_ptr,
                    gdn_out_buf,
                    ssm_state.h_state_intermediates[0],
                    inter_stride_floats,
                    1, nk as u32, nv as u32, kd as u32, vd as u32,
                    conv_dim as u32, conv_dim as u32, (nv * 2) as u32,
                    stream,
                )?;
                // Dump wy17's intermediates + output.
                let h_inter_total = (num_tokens as usize) * h_total;
                let out_total = (num_tokens as usize) * nv * (vd as usize) * bf16;
                let dump_wy17 = |name: &str, ptr: spark_runtime::gpu::DevicePtr, n: usize| -> Result<()> {
                    let mut buf = vec![0u8; n];
                    ctx.gpu.synchronize(stream)?;
                    ctx.gpu.copy_d2h(ptr, &mut buf)?;
                    std::fs::write(format!("/tmp/wy17_dump_{name}.bin"), &buf)
                        .map_err(|e| anyhow::anyhow!("wy17 ab dump {name}: {e}"))?;
                    Ok(())
                };
                dump_wy17("h_out_inter", ssm_state.h_state_intermediates[0], h_inter_total)?;
                dump_wy17("output", gdn_out_buf, out_total)?;
                // Restore h_state from backup so tree_wy gets identical input.
                ctx.gpu.copy_h2d_async(&h_backup, ssm_state.h_state, stream)?;
                ctx.gpu.synchronize(stream)?;
                tracing::info!("M8A A/B: wy17 dumped to /tmp/wy17_dump_*.bin, h_state restored");
            }

            // Prefer M8A v2 (tree-aware WY-fused) when available — bit-
            // equivalent to wy17 on flat-chain payloads. Falls back to the
            // sequential per-token kernel (M8A v1) when WY kernel not loaded.
            if self.gdn_tree_wy_k.0 != 0 {
                ops::gdn_decode_tree_wy(
                    ctx.gpu,
                    self.gdn_tree_wy_k,
                    ssm_state.h_state,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gate_ptr,
                    beta_ptr,
                    parent_ids_dev,
                    gdn_out_buf,
                    ssm_state.h_state_intermediates[0],
                    inter_stride_floats,
                    num_tokens as u32,
                    1, // batch_size
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32, // qk_stride
                    conv_dim as u32, // v_stride
                    (nv * 2) as u32, // gb_stride
                    stream,
                )?;
            } else {
                ops::gdn_decode_tree(
                    ctx.gpu,
                    self.gdn_tree_k,
                    ssm_state.h_state,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gate_ptr,
                    beta_ptr,
                    parent_ids_dev,
                    gdn_out_buf,
                    ssm_state.h_state_intermediates[0],
                    inter_stride_floats,
                    num_tokens as u32,
                    1, // batch_size
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32, // qk_stride
                    conv_dim as u32, // v_stride
                    (nv * 2) as u32, // gb_stride
                    stream,
                )?;
            }

            if dump_now {
                let h_total = nv * (kd as usize) * (vd as usize) * fp32;
                let h_inter_total = (num_tokens as usize) * h_total;
                let out_total = (num_tokens as usize) * nv * (vd as usize) * bf16;
                let dump_after = |name: &str, ptr: spark_runtime::gpu::DevicePtr, n: usize| -> Result<()> {
                    let mut buf = vec![0u8; n];
                    ctx.gpu.synchronize(stream)?;
                    ctx.gpu.copy_d2h(ptr, &mut buf)?;
                    let path = format!("/tmp/m8a_dump_{name}.bin");
                    std::fs::write(&path, &buf)
                        .map_err(|e| anyhow::anyhow!("m8a dump write {path}: {e}"))?;
                    Ok(())
                };
                dump_after("h_out_inter", ssm_state.h_state_intermediates[0], h_inter_total)?;
                dump_after("output", gdn_out_buf, out_total)?;
                tracing::info!(
                    "M8A precision dump complete (layer={}, T={}, nv={}, kd={}, vd={}); see /tmp/m8a_dump_*.bin",
                    0usize, num_tokens, nv, kd, vd
                );
                DUMP_DONE.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        } else if num_tokens == 17 && self.gdn_wy17_k.0 != 0 {
            // ── K=17 (DFlash γ+1): fused WY-Chunkwise path ──
            for t in 0..(num_tokens as u32) {
                let qkv_t = deinterleaved.offset(t as usize * qkvz_size * bf16);
                let conv_out_t = conv_out_buf.offset(t as usize * conv_dim * bf16);
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    self.conv1d_l2norm_k,
                    ssm_state.conv_state,
                    qkv_t,
                    &self.ssm.conv1d,
                    conv_out_t,
                    conv_dim as u32,
                    d_conv as u32,
                    1,
                    qk_ch,
                    kd as u32,
                    1e-6,
                    stream,
                )?;
                ctx.gpu.copy_d2d_async(
                    ssm_state.conv_state,
                    ssm_state.conv_state_intermediates[t as usize],
                    conv_bytes,
                    stream,
                )?;
            }

            let q_ptr = conv_out_buf;
            let k_ptr = conv_out_buf.offset(key_dim * bf16);
            let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
            let gate_ptr = gates_buf;
            let beta_ptr = gates_buf.offset(nv * fp32);
            let inter_stride_floats = (h_bytes / 4) as u32;

            // ── wy17 precision dump (one-shot, env-gated) ──
            // ATLAS_WY17_DUMP=1 dumps inputs+output for the FIRST K=17 verify
            // call so we can bit-diff against M8A v2 tree_wy on the same chain.
            static WY17_DUMP_DONE: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            let wy17_dump = std::env::var("ATLAS_WY17_DUMP").ok().as_deref() == Some("1")
                && !WY17_DUMP_DONE.load(std::sync::atomic::Ordering::Relaxed);
            if wy17_dump {
                let q_total = (num_tokens as usize) * (conv_dim as usize) * bf16;
                let gb_total = (num_tokens as usize) * nv * fp32;
                let h_total = nv * (kd as usize) * (vd as usize) * fp32;
                let dump = |name: &str, ptr: spark_runtime::gpu::DevicePtr, n: usize| -> Result<()> {
                    let mut buf = vec![0u8; n];
                    ctx.gpu.synchronize(stream)?;
                    ctx.gpu.copy_d2h(ptr, &mut buf)?;
                    let path = format!("/tmp/wy17_dump_{name}.bin");
                    std::fs::write(&path, &buf)
                        .map_err(|e| anyhow::anyhow!("wy17 dump write {path}: {e}"))?;
                    Ok(())
                };
                dump("q", q_ptr, q_total)?;
                dump("k", k_ptr, q_total)?;
                dump("v", v_ptr, q_total)?;
                dump("gate", gate_ptr, gb_total)?;
                dump("beta", beta_ptr, gb_total)?;
                dump("h_in", ssm_state.h_state, h_total)?;
            }

            // ── Phase 2 profiling: wall-clock per wy17 call ──
            // Gated by ATLAS_SSM_KERNEL_PROFILE=1. Accumulates total ns spent in
            // wy17 across every SSM layer + every verify step. Sync-before to
            // start clock with empty stream; sync-after to capture GPU finish.
            // Adds ~2 host syncs per layer when active. Disabled by default.
            static SSM_PROFILE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);
            let profile = {
                let v = SSM_PROFILE.load(std::sync::atomic::Ordering::Relaxed);
                if v >= 0 {
                    v == 1
                } else {
                    let enabled =
                        std::env::var("ATLAS_SSM_KERNEL_PROFILE").ok().as_deref() == Some("1");
                    SSM_PROFILE.store(if enabled { 1 } else { 0 }, std::sync::atomic::Ordering::Relaxed);
                    enabled
                }
            };
            if profile {
                ctx.gpu.synchronize(stream)?;
                let t0 = std::time::Instant::now();
                ops::gdn_decode_wy17(
                    ctx.gpu,
                    self.gdn_wy17_k,
                    ssm_state.h_state,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gate_ptr,
                    beta_ptr,
                    gdn_out_buf,
                    ssm_state.h_state_intermediates[0],
                    inter_stride_floats,
                    1, // batch_size
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32, // qk_stride
                    conv_dim as u32, // v_stride
                    (nv * 2) as u32, // gb_stride
                    stream,
                )?;
                ctx.gpu.synchronize(stream)?;
                let ns = t0.elapsed().as_nanos() as u64;
                crate::layers::qwen3_ssm::ssm_profile_record(ns);
            } else {
                ops::gdn_decode_wy17(
                    ctx.gpu,
                    self.gdn_wy17_k,
                    ssm_state.h_state,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gate_ptr,
                    beta_ptr,
                    gdn_out_buf,
                    ssm_state.h_state_intermediates[0],
                    inter_stride_floats,
                    1, // batch_size
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32, // qk_stride
                    conv_dim as u32, // v_stride
                    (nv * 2) as u32, // gb_stride
                    stream,
                )?;
            }

            if wy17_dump {
                let h_total = nv * (kd as usize) * (vd as usize) * fp32;
                let h_inter_total = (num_tokens as usize) * h_total;
                let out_total = (num_tokens as usize) * nv * (vd as usize) * bf16;
                let dump_after = |name: &str, ptr: spark_runtime::gpu::DevicePtr, n: usize| -> Result<()> {
                    let mut buf = vec![0u8; n];
                    ctx.gpu.synchronize(stream)?;
                    ctx.gpu.copy_d2h(ptr, &mut buf)?;
                    let path = format!("/tmp/wy17_dump_{name}.bin");
                    std::fs::write(&path, &buf)
                        .map_err(|e| anyhow::anyhow!("wy17 dump write {path}: {e}"))?;
                    Ok(())
                };
                dump_after("h_out_inter", ssm_state.h_state_intermediates[0], h_inter_total)?;
                dump_after("output", gdn_out_buf, out_total)?;
                tracing::info!(
                    "wy17 precision dump complete (T={num_tokens}, nv={nv}, kd={kd}, vd={vd}); see /tmp/wy17_dump_*.bin"
                );
                WY17_DUMP_DONE.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        } else {
            // ── K∈{5..16}: chunked path using fused wy4/wy3/wy2 kernels ──
            //
            // The legacy per-token sequential path (`ops::gdn_decode` in a
            // loop) has a confirmed numerical bug that produces NaN at the
            // last 3 positions for K∈{5..16} (observed empirically at K=16:
            // positions 13–15 = NaN; symptom is target output collapsing to
            // `correct_first_token + !!!!!`). The fused wy4/wy3/wy2 kernels
            // handle the WY-chunkwise update correctly, so we chunk K into
            // sizes of 4 (or smaller for the tail) and run a fused kernel
            // per chunk. h_state flows naturally between chunks because each
            // wy_k kernel reads + writes it in place.
            //
            // wy_k writes K-1 explicit intermediates (slot 0..K-2 of the
            // chunk) plus h_state = state-after-(K-1) implicit. For
            // non-final chunks we must save h_state to the corresponding
            // global intermediate slot so partial-accept rollback can find
            // every per-position state.
            // Chunk-size selector: avoid chunk=1 (chunk=1 falls through to
            // `ops::gdn_decode` which expects FP32 q/k/v but our conv_out_buf
            // is BF16 — reading 2-byte BF16 elements as 4-byte FP32 produces
            // garbage that corrupted output on K=5/9/13/17%4==1. The wy_k
            // fused kernels (wy2/wy3/wy4) all accept BF16 properly, so we
            // strictly use {2, 3, 4}-sized chunks. The split is chosen so
            // each chunk handles a contiguous run of tokens with one
            // wy_k launch.
            fn pick_chunk(remaining: usize) -> usize {
                match remaining {
                    0 => 0,
                    1 => unreachable!("remaining=1 only happens if prior chunk took too many; handled below"),
                    2 => 2,
                    3 => 3,
                    4 => 4,
                    5 => 3,       // 3 + 2
                    _ => 4,       // 4 + ... rest
                }
            }
            let mut t_done: usize = 0;
            while t_done < num_tokens {
                let remaining = num_tokens - t_done;
                let chunk = pick_chunk(remaining);

                // ── 1. conv1d_l2norm + save conv_state intermediate per token ──
                for ct in 0..chunk {
                    let t_abs = t_done + ct;
                    let qkv_t = deinterleaved.offset(t_abs * qkvz_size * bf16);
                    let conv_out_t = conv_out_buf.offset(t_abs * conv_dim * bf16);
                    ops::conv1d_update_l2norm(
                        ctx.gpu,
                        self.conv1d_l2norm_k,
                        ssm_state.conv_state,
                        qkv_t,
                        &self.ssm.conv1d,
                        conv_out_t,
                        conv_dim as u32,
                        d_conv as u32,
                        1,
                        qk_ch,
                        kd as u32,
                        1e-6,
                        stream,
                    )?;
                    ctx.gpu.copy_d2d_async(
                        ssm_state.conv_state,
                        ssm_state.conv_state_intermediates[t_abs],
                        conv_bytes,
                        stream,
                    )?;
                }

                // ── 2. Run gdn fused kernel for the chunk ──
                let q_ptr = conv_out_buf.offset(t_done * conv_dim * bf16);
                let k_ptr = q_ptr.offset(key_dim * bf16);
                let v_ptr = q_ptr.offset(key_dim * 2 * bf16);
                let gate_beta_stride = nv * 2 * fp32;
                let gate_ptr = gates_buf.offset(t_done * gate_beta_stride);
                let beta_ptr = gate_ptr.offset(nv * fp32);
                let gdn_out_t = gdn_out_buf.offset(t_done * args.value_dim * bf16);

                match chunk {
                    4 => ops::gdn_decode_wy4(
                        ctx.gpu,
                        self.gdn_wy4_k,
                        ssm_state.h_state,
                        q_ptr, k_ptr, v_ptr,
                        gate_ptr, beta_ptr,
                        gdn_out_t,
                        ssm_state.h_state_intermediates[t_done],
                        ssm_state.h_state_intermediates[t_done + 1],
                        ssm_state.h_state_intermediates[t_done + 2],
                        1, nk as u32, nv as u32, kd as u32, vd as u32,
                        conv_dim as u32, conv_dim as u32, (nv * 2) as u32,
                        stream,
                    )?,
                    3 => ops::gdn_decode_wy3(
                        ctx.gpu,
                        self.gdn_wy3_k,
                        ssm_state.h_state,
                        q_ptr, k_ptr, v_ptr,
                        gate_ptr, beta_ptr,
                        gdn_out_t,
                        ssm_state.h_state_intermediates[t_done],
                        ssm_state.h_state_intermediates[t_done + 1],
                        1, nk as u32, nv as u32, kd as u32, vd as u32,
                        conv_dim as u32, conv_dim as u32, (nv * 2) as u32,
                        stream,
                    )?,
                    2 => ops::gdn_decode_wy2(
                        ctx.gpu,
                        self.gdn_wy2_k,
                        ssm_state.h_state,
                        q_ptr, k_ptr, v_ptr,
                        gate_ptr, beta_ptr,
                        gdn_out_t,
                        ssm_state.h_state_intermediates[t_done],
                        1, nk as u32, nv as u32, kd as u32, vd as u32,
                        conv_dim as u32, conv_dim as u32, (nv * 2) as u32,
                        stream,
                    )?,
                    // chunk=1 deliberately not handled here — pick_chunk()
                    // guarantees a size-1 remainder never appears (it
                    // pre-splits e.g. K=5 as 3+2 so every chunk is in
                    // {2, 3, 4}). The ops::gdn_decode single-token kernel
                    // expects FP32 q/k/v but conv_out_buf is BF16, which
                    // caused silent corruption on K=5 / K=9 / K=13 (any
                    // K with K%4==1) — confirmed by `def fibonacci(n):
                    // \n if n <= 0:\n... def/:\n\n\n` output at N=4 K=5.
                    _ => unreachable!("pick_chunk guarantees chunk in {{2, 3, 4}}; got {chunk}"),
                }

                // ── 3. Save h_state to chunk-end intermediate (non-final chunks only) ──
                // wy_k writes K-1 explicit intermediates; the K-th position
                // (= chunk-end) lives in h_state. For the LAST chunk the
                // final h_state IS the canonical post-step state and isn't
                // read via intermediates (full-accept rollback keeps
                // h_state as-is). For non-final chunks we need
                // intermediates[t_done+chunk-1] populated so partial
                // accepts that land mid-K can find the per-position state.
                if t_done + chunk < num_tokens {
                    ctx.gpu.copy_d2d_async(
                        ssm_state.h_state,
                        ssm_state.h_state_intermediates[t_done + chunk - 1],
                        h_bytes,
                        stream,
                    )?;
                }

                t_done += chunk;
            }
        }

        Ok(())
    }
}
