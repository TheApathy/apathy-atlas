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
