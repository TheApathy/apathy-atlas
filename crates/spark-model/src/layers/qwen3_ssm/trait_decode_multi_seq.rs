// SPDX-License-Identifier: AGPL-3.0-only

//! TransformerLayer::decode_multi_seq.

use super::*;

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    /// Multi-sequence decode (one token per sequence, independent SSM state).
    ///
    /// SSM decode has two kinds of work per layer:
    /// - **State-free outer ops** (input/output RMS norm + residual) —
    ///   token-independent; safe to batch across N sequences in a single
    ///   kernel launch.
    /// - **State-bearing inner ops** (`conv1d_update`, `gdn_decode`) — each
    ///   carries its own per-sequence recurrent state, so the kernels can't
    ///   trivially fan over multiple states in one grid.
    ///
    /// We batch the outer ops (`rms_norm_residual` + the
    /// `residual_add_rms_norm` after SSM) and run the inner ops in a
    /// per-sequence loop using disjoint slices of the per-pass scratch arena
    /// (sized for `max_batch_tokens` in `BufferSizes::from_config`).
    ///
    /// An earlier batched attempt (#6) wrote conv1d output to
    /// `ctx.buffers.attn_output()`, which is only sized for
    /// `m * mamba2_d_inner * bf16` — half of what `n * conv_dim * bf16`
    /// requires on Qwen3.5-A3B. The out-of-bounds writes were what produced
    /// the multilingual gibberish. The corrected layout writes conv and GDN
    /// output into the much larger `ssm_conv_out_f32`
    /// (`m * ssm_qkvz_size * 4`), mirroring the single-seq `ssm_forward`
    /// layout per slice.
    ///
    /// MoE forward writes to `moe_output[0..h]` and is therefore interleaved
    /// with the per-seq `residual_add` (same pattern as
    /// [`decode_batched_inner`] for non-fused K).
    ///
    /// If the FP32 conv1d / GDN / gated-RMS kernels aren't loaded on the
    /// active backend (e.g. Metal), fall back to per-sequence
    /// [`decode_inner`] — the BF16 path stores GDN output in `attn_output`,
    /// which the safe layout above can't accommodate.
    pub(super) fn decode_multi_seq_inner<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let use_batched = self.conv1d_l2norm_f32_k.0 != 0
            && self.gdn_f32_k.0 != 0
            && self.gated_rms_norm_f32_k.0 != 0;
        if !use_batched {
            return self.decode_multi_seq_per_seq_fallback(
                hidden,
                residual,
                num_seqs,
                states,
                kv_cache,
                seq_lens,
                block_tables,
                ctx,
                stream,
            );
        }

        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let bf16 = 2usize;
        let fp32 = 4usize;
        let residual_elem = if ctx.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        let n = num_seqs as u32;

        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = (key_dim * 2 + value_dim) as u32;
        let qk_channels = (key_dim * 2) as u32;
        let d_conv = ctx.config.linear_conv_kernel_dim as u32;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let ba_size = ctx.config.ssm_ba_size();

        // ── 1. Batched RMS norm + residual across all N sequences ──
        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            n,
            h as u32,
            eps,
            stream,
        )?;

        // ── 2-7. Per-seq SSM forward (state-bearing) ──
        //
        // Buffer layout per seq i:
        //   ssm_deinterleaved[i * qkvz_size * bf16 ..]            — [Q|K|V|Z]
        //   ssm_gates[i * nv * 2 * fp32 ..]                       — [gate(nv) | beta(nv)]
        //   ssm_conv_out_f32[i * qkvz_size * fp32 ..]             — conv1d FP32 out
        //     + (key_dim*2 + value_dim)*fp32                      — GDN FP32 out (in-slice)
        //   ssm_qkvz[i * qkvz_size * bf16 ..]                     — normed_out (gated_rms)
        //   moe_output[i * h * bf16 ..]                           — SSM output projection
        //
        // All strides match the per-token natural stride of the respective
        // arena buffer (see `BufferSizes::from_config`), so per-seq writes
        // stay in-bounds for any `num_seqs ≤ max_batch_tokens`.
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        let gates_buf = ctx.buffers.ssm_gates();
        let conv_out_f32 = ctx.buffers.ssm_conv_out_f32();
        let normed_out_buf = ctx.buffers.ssm_qkvz();
        let ssm_out_buf = ctx.buffers.moe_output();
        let gdn_local_offset = (key_dim * 2 + value_dim) * fp32;

        // ── ATLAS_SSM_MULTI_SEQ_KERNEL multi-seq state-advance dispatch ──
        //
        // Gated by `ssm_multi_seq_kernel_enabled()` AND the FP32 multi-seq
        // kernel handles AND num_seqs ≤ ssm_multi_seq_ptr_max. When ON:
        //   - QKVZ projection + BA + gates run per-seq (no shape change)
        //   - conv1d_update_l2norm + gdn_decode each collapse to a SINGLE
        //     launch via the FP32 multi-seq variants, fed a device-resident
        //     per-seq state pointer table uploaded once per layer.
        //   - gated_rms_norm + out_proj stay per-seq (their compute is
        //     state-free; the per-seq launch overhead is left untouched
        //     since they're not the bottleneck — conv1d+gdn alone is
        //     ~50 % of the per-seq loop cost on AEON-Q36-27B).
        //
        // Saves 2 launches per SSM layer per token at num_seqs ≥ 2. At
        // num_seqs=4 × 48 SSM layers × ~20 μs/launch = ~3.8 ms/token of
        // pure launch-overhead removed.
        let multi_seq_kernel_path = num_seqs >= 2
            && num_seqs <= self.ssm_multi_seq_ptr_max
            && self.conv1d_l2norm_f32_multi_seq_k.0 != 0
            && self.gdn_decode_f32_multi_seq_k.0 != 0
            && crate::layers::ssm_multi_seq_kernel_enabled();
        if multi_seq_kernel_path {
            // Collect per-seq h_state + conv_state device pointers and
            // upload to the layer's pre-allocated scratch
            // (`ssm_multi_seq_ptr_scratch`). Layout:
            //   [h_state_ptrs[N] u64, conv_state_ptrs[N] u64]
            // h_state_ptrs at offset 0, conv_state_ptrs at offset N*8.
            let mut ptr_buf: [u64; 64] = [0u64; 64];
            assert!(
                num_seqs * 2 <= ptr_buf.len(),
                "ssm_multi_seq_ptr_scratch capacity exceeded: {}*2 > {}",
                num_seqs,
                ptr_buf.len()
            );
            for (i, state) in states.iter_mut().enumerate().take(num_seqs) {
                let ssm_state = state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;
                ptr_buf[i] = ssm_state.h_state.0;
                ptr_buf[num_seqs + i] = ssm_state.conv_state.0;
            }
            let ptr_bytes = unsafe {
                std::slice::from_raw_parts(
                    ptr_buf.as_ptr() as *const u8,
                    num_seqs * 2 * std::mem::size_of::<u64>(),
                )
            };
            ctx.gpu.copy_h2d_async(
                ptr_bytes,
                self.ssm_multi_seq_ptr_scratch,
                stream,
            )?;
            let h_state_ptrs_dev = self.ssm_multi_seq_ptr_scratch;
            let conv_state_ptrs_dev = self
                .ssm_multi_seq_ptr_scratch
                .offset(num_seqs * std::mem::size_of::<u64>());

            // ── Step 2: QKVZ projection ──
            // Batched path: sequential_qkvz + nvfp4 → single w4a16_gemm at
            //   M=num_seqs. Input `normed` is contiguous (stride h BF16 =
            //   K), output `deinterleaved` is contiguous (stride qkvz_size
            //   BF16 = N). Matches the per-seq layout 1:1.
            // Fallback: per-seq loop (interleaved 80B / dense / non-nvfp4).
            let qkvz_batched_gemm = self.sequential_qkvz
                && self.qkvz_nvfp4.is_some()
                && self.w4a16_gemm_k.0 != 0;
            if qkvz_batched_gemm {
                let nvfp4 = self.qkvz_nvfp4.as_ref().unwrap();
                ops::w4a16_gemm(
                    ctx.gpu,
                    self.w4a16_gemm_k,
                    normed,
                    nvfp4,
                    deinterleaved,
                    num_seqs as u32,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                for i in 0..num_seqs {
                    let normed_i = normed.offset(i * h * bf16);
                    let deint_i = deinterleaved.offset(i * qkvz_size * bf16);
                    if self.sequential_qkvz {
                        if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                            ops::w4a16_gemv(
                                ctx.gpu,
                                self.w4a16_gemv_k,
                                normed_i,
                                nvfp4,
                                deint_i,
                                qkvz_size as u32,
                                h as u32,
                                stream,
                            )?;
                        } else {
                            ops::dense_gemv(
                                ctx.gpu,
                                self.dense_gemv_k,
                                normed_i,
                                &self.ssm.in_proj_qkvz,
                                deint_i,
                                qkvz_size as u32,
                                h as u32,
                                stream,
                            )?;
                        }
                    } else if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                        ops::w4a16_gemv_qkvz(
                            ctx.gpu,
                            self.w4a16_gemv_qkvz_k,
                            normed_i,
                            nvfp4,
                            deint_i,
                            qkvz_size as u32,
                            h as u32,
                            nk as u32,
                            kd as u32,
                            vpg as u32,
                            vd as u32,
                            stream,
                        )?;
                    } else {
                        ops::dense_gemv(
                            ctx.gpu,
                            self.dense_gemv_k,
                            normed_i,
                            &self.ssm.in_proj_qkvz,
                            deint_i,
                            qkvz_size as u32,
                            h as u32,
                            stream,
                        )?;
                        ops::deinterleave_qkvz(
                            ctx.gpu,
                            self.deinterleave_k,
                            deint_i,
                            deint_i,
                            1,
                            nk as u32,
                            kd as u32,
                            vpg as u32,
                            vd as u32,
                            stream,
                        )?;
                    }
                }
            }

            // ── Step 3: BA projection + GDN gates ──
            // Batched path: split the per-seq `dense_gemv_ba_gates` fusion
            //   into one batched dense_gemm (BA at M=num_seqs writing BF16
            //   [num_seqs, ba_size] into the head of `ssm_conv_out_f32` —
            //   safe because conv1d_update_l2norm writes that buffer LATER
            //   in step 4) + one `compute_gdn_gates_multi_seq` consuming
            //   the BA output.
            // Fallback: per-seq `dense_gemv_ba_gates` loop.
            let ba_batched_split =
                self.dense_gemm_k.0 != 0 && self.compute_gdn_gates_multi_seq_k.0 != 0;
            if ba_batched_split {
                // Scratch: `conv_out_f32` reinterpreted as BF16 head. We
                // need num_seqs * ba_size BF16 elements; the buffer is
                // sized num_seqs * qkvz_size FP32 = num_seqs * 2 *
                // qkvz_size BF16 ≫ num_seqs * ba_size BF16. Safe.
                let ba_scratch = conv_out_f32;
                ops::dense_gemm(
                    ctx.gpu,
                    self.dense_gemm_k,
                    normed,
                    &self.ssm.in_proj_ba,
                    ba_scratch,
                    num_seqs as u32,
                    ba_size as u32,
                    h as u32,
                    stream,
                )?;
                ops::compute_gdn_gates_multi_seq(
                    ctx.gpu,
                    self.compute_gdn_gates_multi_seq_k,
                    ba_scratch,
                    self.ssm.a_log.weight,
                    self.ssm.dt_bias.weight,
                    gates_buf,
                    gates_buf.offset(nv * fp32),
                    num_seqs as u32,
                    nv as u32,
                    nk as u32,
                    vpg as u32,
                    ba_size as u32,           // BF16 elements between seqs in ba_scratch
                    (nv * 2) as u32,          // FP32 elements between seqs in gates_buf
                    stream,
                )?;
            } else {
                for i in 0..num_seqs {
                    let normed_i = normed.offset(i * h * bf16);
                    let gate_i = gates_buf.offset(i * nv * 2 * fp32);
                    let beta_i = gate_i.offset(nv * fp32);
                    ops::dense_gemv_ba_gates(
                        ctx.gpu,
                        self.ba_gates_k,
                        normed_i,
                        &self.ssm.in_proj_ba,
                        self.ssm.a_log.weight,
                        self.ssm.dt_bias.weight,
                        gate_i,
                        beta_i,
                        ba_size as u32,
                        h as u32,
                        vpg as u32,
                        stream,
                    )?;
                }
            }

            // Step 4: ONE conv1d_update_l2norm_f32_multi_seq launch.
            // Input layout: deinterleaved [num_seqs, qkvz_size] BF16
            //   → input_stride = qkvz_size BF16 elements between seqs
            // Output layout: conv_out_f32 [num_seqs, qkvz_size] FP32
            //   → output_stride = qkvz_size FP32 elements between seqs
            ops::conv1d_update_l2norm_f32_multi_seq(
                ctx.gpu,
                self.conv1d_l2norm_f32_multi_seq_k,
                conv_state_ptrs_dev,
                deinterleaved,
                &self.ssm.conv1d,
                conv_out_f32,
                conv_dim,
                d_conv,
                num_seqs as u32,
                qk_channels,
                kd as u32,
                1e-6,
                qkvz_size as u32,
                qkvz_size as u32,
                stream,
            )?;

            // Step 5: ONE gdn_decode_f32_multi_seq launch.
            //   Q/K split: query at conv_out_f32 + offset 0
            //              key   at conv_out_f32 + key_dim FP32 elements
            //   V        : conv_out_f32 + 2*key_dim FP32 elements
            //   gate/beta: gates_buf, gate_beta_stride = 2*nv FP32 elements
            //   output   : conv_out_f32 + gdn_local_offset (Z region tail
            //              per ssm_forward.rs single-seq layout —
            //              re-uses the qkv region's tail in the FP32 slot)
            //   v_out_stride = qkvz_size FP32 elements between seqs
            // q/k/v strides are FP32 elements (kernel reads FP32 inputs
            // matching conv1d_update_l2norm_f32_multi_seq output layout).
            let qk_stride_fp32 = qkvz_size as u32;
            let v_in_stride_fp32 = qkvz_size as u32;
            let gdn_out = conv_out_f32.offset(gdn_local_offset);
            ops::gdn_decode_f32_multi_seq(
                ctx.gpu,
                self.gdn_decode_f32_multi_seq_k,
                h_state_ptrs_dev,
                conv_out_f32,
                conv_out_f32.offset(key_dim * fp32),
                conv_out_f32.offset(key_dim * 2 * fp32),
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out,
                num_seqs as u32,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                (nv * 2) as u32,
                qk_stride_fp32,
                v_in_stride_fp32,
                qkvz_size as u32,
                stream,
            )?;

            // ── Step 6: Gated RMS norm ──
            // Batched path: ONE `gated_rms_norm_f32_multi_seq` launch
            //   reading from `conv_out_f32 + gdn_local_offset` (stride
            //   qkvz_size FP32) + `deinterleaved + Z offset` (stride
            //   qkvz_size BF16) and WRITING into a value_dim-contig
            //   scratch at the head of `normed_out_buf` (= ssm_qkvz).
            //   That buffer is sized num_seqs * qkvz_size BF16 ≫ the
            //   num_seqs * value_dim BF16 we need, and it isn't read
            //   until step 7 (out_proj) which we batch directly out of
            //   the value_dim-contig layout we just wrote.
            //   NOTE: `deinterleaved` IS the gate source, so it cannot
            //   double as the compact output (writes would alias gate
            //   reads of later seqs — produces garbage).
            // Fallback: per-seq loop.
            let normed_out_compact = normed_out_buf; // value_dim-contig scratch in ssm_qkvz
            let z_base = deinterleaved.offset((key_dim * 2 + value_dim) * bf16);
            let rms_batched = self.gated_rms_norm_f32_multi_seq_k.0 != 0;
            if rms_batched {
                ops::gated_rms_norm_f32_multi_seq(
                    ctx.gpu,
                    self.gated_rms_norm_f32_multi_seq_k,
                    conv_out_f32.offset(gdn_local_offset),
                    z_base,
                    &self.ssm.norm,
                    normed_out_compact,
                    nv as u32,           // num_v_heads (32)
                    num_seqs as u32,
                    vd as u32,           // head_dim (128) — norm is per-head
                    eps,
                    qkvz_size as u32,    // FP32 elements between seqs (input)
                    qkvz_size as u32,    // BF16 elements between seqs (gate)
                    value_dim as u32,    // BF16 elements between seqs (output)
                    stream,
                )?;
            } else {
                for i in 0..num_seqs {
                    let deint_i = deinterleaved.offset(i * qkvz_size * bf16);
                    let conv_out_i = conv_out_f32.offset(i * qkvz_size * fp32);
                    let gdn_out_i = conv_out_i.offset(gdn_local_offset);
                    let normed_out_i = normed_out_buf.offset(i * qkvz_size * bf16);
                    let z_ptr = deint_i.offset((key_dim * 2 + value_dim) * bf16);
                    ops::gated_rms_norm(
                        ctx.gpu,
                        self.gated_rms_norm_f32_k,
                        gdn_out_i,
                        z_ptr,
                        &self.ssm.norm,
                        normed_out_i,
                        nv as u32,
                        vd as u32,
                        vd as u32,
                        eps,
                        vd as u32,
                        stream,
                    )?;
                }
            }

            // ── Step 7: Output projection ──
            // Batched path: ONE w4a16_gemm at M=num_seqs reading the
            //   value_dim-contig output from step 6 (`normed_out_compact`,
            //   stride value_dim BF16 = K) and writing into `ssm_out_buf`
            //   [num_seqs, h] BF16 contig (stride h BF16 = N). Falls back
            //   to dense_gemm M=num_seqs when no nvfp4 out_proj, else
            //   per-seq.
            if rms_batched {
                if let Some(ref dense_out) = self.out_proj_dense {
                    if self.dense_gemm_k.0 != 0 {
                        ops::dense_gemm(
                            ctx.gpu,
                            self.dense_gemm_k,
                            normed_out_compact,
                            dense_out,
                            ssm_out_buf,
                            num_seqs as u32,
                            h as u32,
                            value_dim as u32,
                            stream,
                        )?;
                    } else {
                        for i in 0..num_seqs {
                            let normed_out_i =
                                normed_out_compact.offset(i * value_dim * bf16);
                            let ssm_out_i = ssm_out_buf.offset(i * h * bf16);
                            ops::dense_gemv(
                                ctx.gpu,
                                self.dense_gemv_k,
                                normed_out_i,
                                dense_out,
                                ssm_out_i,
                                h as u32,
                                value_dim as u32,
                                stream,
                            )?;
                        }
                    }
                } else if self.w4a16_gemm_k.0 != 0 {
                    ops::w4a16_gemm(
                        ctx.gpu,
                        self.w4a16_gemm_k,
                        normed_out_compact,
                        &self.ssm.out_proj,
                        ssm_out_buf,
                        num_seqs as u32,
                        h as u32,
                        value_dim as u32,
                        stream,
                    )?;
                } else {
                    for i in 0..num_seqs {
                        let normed_out_i = normed_out_compact.offset(i * value_dim * bf16);
                        let ssm_out_i = ssm_out_buf.offset(i * h * bf16);
                        ops::w4a16_gemv(
                            ctx.gpu,
                            self.w4a16_gemv_k,
                            normed_out_i,
                            &self.ssm.out_proj,
                            ssm_out_i,
                            h as u32,
                            value_dim as u32,
                            stream,
                        )?;
                    }
                }
            } else {
                // RMS norm fallback wrote per-seq into normed_out_buf with
                // qkvz_size stride; reuse the original per-seq out_proj.
                for i in 0..num_seqs {
                    let normed_out_i = normed_out_buf.offset(i * qkvz_size * bf16);
                    let ssm_out_i = ssm_out_buf.offset(i * h * bf16);
                    if let Some(ref dense_out) = self.out_proj_dense {
                        ops::dense_gemv(
                            ctx.gpu,
                            self.dense_gemv_k,
                            normed_out_i,
                            dense_out,
                            ssm_out_i,
                            h as u32,
                            value_dim as u32,
                            stream,
                        )?;
                    } else {
                        ops::w4a16_gemv(
                            ctx.gpu,
                            self.w4a16_gemv_k,
                            normed_out_i,
                            &self.ssm.out_proj,
                            ssm_out_i,
                            h as u32,
                            value_dim as u32,
                            stream,
                        )?;
                    }
                }
            }

            // Skip the legacy per-seq loop below and continue with the
            // existing batched step 8 (residual_add_rms_norm) + step 9
            // (per-seq MoE) at the bottom of this function.
            // We did the equivalent of steps 2-7 above. Use a labeled
            // loop break by overwriting the entire per-seq block with a
            // sentinel.
        }

        for i in 0..num_seqs {
            if multi_seq_kernel_path {
                break; // already done above
            }
            let normed_i = normed.offset(i * h * bf16);
            let deint_i = deinterleaved.offset(i * qkvz_size * bf16);
            let gate_i = gates_buf.offset(i * nv * 2 * fp32);
            let beta_i = gate_i.offset(nv * fp32);
            let conv_out_i = conv_out_f32.offset(i * qkvz_size * fp32);
            let gdn_out_i = conv_out_i.offset(gdn_local_offset);
            let normed_out_i = normed_out_buf.offset(i * qkvz_size * bf16);
            let ssm_out_i = ssm_out_buf.offset(i * h * bf16);

            let ssm_state = states[i]
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;

            // ── 2. QKVZ projection (+ deinterleave if needed) ──
            if self.sequential_qkvz {
                if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                    ops::w4a16_gemv(
                        ctx.gpu,
                        self.w4a16_gemv_k,
                        normed_i,
                        nvfp4,
                        deint_i,
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?;
                } else {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed_i,
                        &self.ssm.in_proj_qkvz,
                        deint_i,
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?;
                }
            } else if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                ops::w4a16_gemv_qkvz(
                    ctx.gpu,
                    self.w4a16_gemv_qkvz_k,
                    normed_i,
                    nvfp4,
                    deint_i,
                    qkvz_size as u32,
                    h as u32,
                    nk as u32,
                    kd as u32,
                    vpg as u32,
                    vd as u32,
                    stream,
                )?;
            } else {
                // Dense fallback: interleaved GEMV then in-place deinterleave.
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed_i,
                    &self.ssm.in_proj_qkvz,
                    deint_i,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
                ops::deinterleave_qkvz(
                    ctx.gpu,
                    self.deinterleave_k,
                    deint_i,
                    deint_i,
                    1,
                    nk as u32,
                    kd as u32,
                    vpg as u32,
                    vd as u32,
                    stream,
                )?;
            }

            // ── 3. Fused BA projection + GDN gates ──
            ops::dense_gemv_ba_gates(
                ctx.gpu,
                self.ba_gates_k,
                normed_i,
                &self.ssm.in_proj_ba,
                self.ssm.a_log.weight,
                self.ssm.dt_bias.weight,
                gate_i,
                beta_i,
                ba_size as u32,
                h as u32,
                vpg as u32,
                stream,
            )?;

            // ── 4. Conv1d update + SiLU + L2 norm (FP32 output) ──
            ops::conv1d_update_l2norm(
                ctx.gpu,
                self.conv1d_l2norm_f32_k,
                ssm_state.conv_state,
                deint_i,
                &self.ssm.conv1d,
                conv_out_i,
                conv_dim,
                d_conv,
                1,
                qk_channels,
                kd as u32,
                1e-6,
                stream,
            )?;

            // ── 5. GDN decode (FP32 state + output) ──
            let q_conv = conv_out_i;
            let k_conv = conv_out_i.offset(key_dim * fp32);
            let v_conv = conv_out_i.offset(key_dim * 2 * fp32);
            ops::gdn_decode(
                ctx.gpu,
                self.gdn_f32_k,
                ssm_state.h_state,
                q_conv,
                k_conv,
                v_conv,
                gate_i,
                beta_i,
                gdn_out_i,
                1,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                stream,
            )?;

            // ── 6. Gated RMS norm (FP32 GDN input → BF16 normed output) ──
            let z_ptr = deint_i.offset((key_dim * 2 + value_dim) * bf16);
            ops::gated_rms_norm(
                ctx.gpu,
                self.gated_rms_norm_f32_k,
                gdn_out_i,
                z_ptr,
                &self.ssm.norm,
                normed_out_i,
                nv as u32,
                vd as u32,
                vd as u32,
                eps,
                vd as u32,
                stream,
            )?;

            // ── 7. Output projection [value_dim → hidden_size] ──
            if let Some(ref dense_out) = self.out_proj_dense {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed_out_i,
                    dense_out,
                    ssm_out_i,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else {
                ops::w4a16_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    normed_out_i,
                    &self.ssm.out_proj,
                    ssm_out_i,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            }
        }

        // ── 8. Batched residual + post-attn RMS norm across all N seqs ──
        // Reads SSM output from `ssm_out_buf[0..n*h*bf16]` (contiguous,
        // written in step 7) and produces post-norm input for the MoE.
        let normed2 = ctx.buffers.norm_output();
        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            ssm_out_buf,
            &self.post_attn_norm,
            normed2,
            residual,
            n,
            h as u32,
            eps,
            stream,
        )?;

        // ── 9. Per-seq MoE forward + residual_add (interleaved) ──
        //
        // MoE.forward writes its output into `moe_output[0..h*bf16]`
        // regardless of which sequence it's processing, so we add it back
        // into the hidden slot for seq i before invoking MoE for seq i+1.
        // Same pattern as `decode_batched_inner` for non-fused K.
        for i in 0..num_seqs {
            let normed2_i = normed2.offset(i * h * bf16);
            let moe_out = self.ffn.forward(normed2_i, ctx, stream)?;
            let hidden_i = hidden.offset(i * h * residual_elem);
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden_i,
                moe_out,
                h as u32,
                stream,
            )?;
        }

        Ok(())
    }

    /// Per-sequence single-decode fallback. Used when FP32 conv1d / GDN
    /// kernels aren't loaded (e.g. Metal backend) — the BF16 path uses
    /// `attn_output` for GDN output, which the batched layout above can't
    /// safely accommodate for N>1 without overflow.
    #[allow(clippy::too_many_arguments)]
    fn decode_multi_seq_per_seq_fallback<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let residual_elem = if ctx.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };

        let mut stub_disk = Vec::<u32>::new();
        let mut stub_last_offloaded = Vec::<u32>::new();
        for i in 0..num_seqs {
            let hidden_i = hidden.offset(i * h * residual_elem);
            let residual_i = residual.offset(i * h * residual_elem);
            self.decode(
                hidden_i,
                residual_i,
                states[i],
                kv_cache,
                seq_lens[i],
                &mut block_tables[i].clone(),
                &mut stub_disk,
                &mut stub_last_offloaded,
                ctx,
                stream,
            )?;
        }
        Ok(())
    }
}
