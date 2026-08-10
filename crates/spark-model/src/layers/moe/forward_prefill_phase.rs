// SPDX-License-Identifier: AGPL-3.0-only

//! Shared-expert phase of `MoeLayer::forward_prefill`.
//!
//! Hoisted from `forward_prefill.rs` to keep that file under the 500 LoC
//! cap. The single entry point [`MoeLayer::run_shared_expert_prefill`]
//! mirrors the original block 1:1 — same control flow, same kernel
//! launches, same buffer wiring.

use super::*;

/// Which W4A16 kernel serves the three shared-expert prefill GEMMs.
///
/// All arms consume the IDENTICAL weight tables (`shared_gate_t` /
/// `shared_up_t` / `shared_down_t`), see `shared_prefill_arm` for the layout
/// compatibility argument. Only the tiling/pipeline differs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum SharedPrefillArm {
    /// `w4a16_gemm_t` — M64 × N128, K_STEP=32. The historical default.
    /// 96 regs, 19584 B smem → 5 CTAs/SM.
    K32,
    /// `w4a16_gemm_t_v2` — K32 with B_fp8 re-banked to a 48 B row stride and
    /// the dequant published as 2× STS.128 instead of 16× STS.U16. RECOMMENDED
    /// first A/B: 2.6× fewer shared-memory transactions for one CTA/SM.
    /// 128 regs, 21632 B smem → 4 CTAs/SM.
    K32V2,
    /// `w4a16_gemm_t_k64` — M64 × N128, K_STEP=64.
    /// 128 regs, 39104 B smem → 2 CTAs/SM (occupancy risk).
    K64,
    /// `w4a16_gemm_t_k64_v2` — K64 plus the same vectorized dequant store.
    /// 128 regs, 39104 B smem → 2 CTAs/SM (occupancy risk).
    K64V2,
    /// `w4a16_gemm_t_m128` — M128 × N128, K_STEP=32 (2 M-chunks per CTA).
    /// 168 regs, 29824 B smem → 3 CTAs/SM. Halves B re-reads, but keeps the
    /// 8-way-conflicted STS.U16 dequant.
    M128,
}

impl MoeLayer {
    /// Resolve the shared-expert prefill GEMM arm from `ATLAS_MOE_SHARED_K64`.
    ///
    /// `unset`/`0` → K32 (unchanged behaviour, the default), `1` → K64,
    /// `2` → K64V2, `3` → M128, `4` → K32V2 (the recommended first A/B).
    /// Read once via `OnceLock` so a CUDA-graph capture and its replays cannot
    /// disagree (same discipline as `moe_gate_gemv`).
    ///
    /// ── WHY (mechanism, verified in SASS not guessed) ───────────────────
    /// `w4a16_gemm_t` is LSU-bound, not tensor-core bound and not DRAM
    /// bound. Its DEQUANT_T publishes the dequanted B tile into
    /// `smem_B_fp8[128][32]`; `cuobjdump -sass` shows nvcc emits 32
    /// `STS.U16` (it does NOT coalesce them), and with a 32-byte row stride
    /// the store address is `my_n*8 + kp/2` words — gcd(8,32)=4, so 32 lanes
    /// hit 4 banks at 4 addresses each: an 8-WAY bank conflict. The MMA-side
    /// reload `smem_B_fp8[nt*8+group_id][4*tid]` is 2-way conflicted for the
    /// same reason. Those two account for ~77% of the kernel's shared-memory
    /// transactions, which is exactly why a GEMM with ~4600 FLOP/byte of
    /// arithmetic intensity measures 35 TFLOP/s (~14% of dense-FP8 peak).
    ///
    /// Transactions per warp per K=64 of work (LUT / B_packed / dequant STS /
    /// MMA operand / A fragment):
    ///   K32   :  64 +  32 + 256 + 128 + 16 = 496
    ///   K32V2 :  64 +  32 +  16 +  64 + 16 = 192   (2.6×)
    ///   K64   :  64 +  32 + 128 +  64 + 16 = 304   (1.6×)
    ///   K64V2 :  64 +  32 +  16 +  64 + 16 = 192   (2.6×, but 2 CTAs/SM)
    ///
    /// ── LAYOUT COMPATIBILITY (verified, not assumed) ────────────────────
    /// The shared expert is **NVFP4 GROUP_SIZE=16 with FP8-E4M3 scale bytes
    /// and a per-tensor FP32 `scale2`** — NOT the MXFP4/E8M0 GS=32 format the
    /// routed experts use on DeepSeek-V4. `helpers_a.rs` builds every shared
    /// table with an explicit literal 16
    /// (`transpose_for_gemm_gs_inplace(gpu, shared_inter, h, 16)`), while the
    /// routed tables use `routed_gs` = 32 when
    /// `experts_scale_kind == Mxfp4E8m0`. That is why the shared expert must
    /// NOT be routed through the `*_e8m0` K64 entries in
    /// `moe_w4a16_grouped_gemm.cu` — those instantiate `<32, true>` and would
    /// index the scale table with the wrong stride. (The `<GROUP_SIZE,false>`
    /// NVFP4 instantiations there ARE numerically compatible, but they are
    /// pointer-table/`expert_offsets` kernels: using them would mean
    /// manufacturing a 1-expert ptr table and an identity `sorted_token_ids`
    /// for zero benefit over the dense entries below.)
    ///
    /// All four arms live in the SAME translation unit
    /// (`kernels/gb10/deepseek-v4-flash/nvfp4/w4a16_gemm.cu`) and share its
    /// `GROUP_SIZE 16` / `E2M1_LUT` / `__nv_fp8_e4m3` scale decode and the
    /// same `(A, B_packed, B_scale, scale2, C, M, N, K)` signature, with
    /// `B_packed[K/2, N]` and `B_scale[K/GROUP_SIZE, N]` — exactly what
    /// `QuantizedWeight::transpose_for_gemm*` produces. So the arms are
    /// interchangeable at the call site; only the launch grid differs
    /// (M128 tiles M by 128 instead of 64).
    pub(super) fn shared_prefill_arm() -> SharedPrefillArm {
        static ARM: std::sync::OnceLock<SharedPrefillArm> = std::sync::OnceLock::new();
        *ARM.get_or_init(|| {
            match std::env::var("ATLAS_MOE_SHARED_K64").ok().as_deref() {
                Some("1") => SharedPrefillArm::K64,
                Some("2") => SharedPrefillArm::K64V2,
                Some("3") => SharedPrefillArm::M128,
                Some("4") => SharedPrefillArm::K32V2,
                _ => SharedPrefillArm::K32,
            }
        })
    }

    /// One shared-expert prefill projection, honouring `shared_prefill_arm()`.
    ///
    /// Falls back to the K32 default whenever the requested kernel is absent
    /// from this model's kernel set, or when `k % 64 != 0` (the K64 bodies
    /// step K in units of 64 and have no tail loop). Both K=4096 (gate/up) and
    /// K=2048 (down) on DeepSeek-V4 satisfy the divisibility guard.
    #[allow(clippy::too_many_arguments)]
    fn shared_prefill_gemm(
        &self,
        input: DevicePtr,
        weight: &QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        aux: u64,
        ctx: &ForwardContext,
    ) -> Result<()> {
        let k64_ok = k.is_multiple_of(64);
        let handle = match Self::shared_prefill_arm() {
            SharedPrefillArm::K32V2 => self.w4a16_gemm_t_v2,
            SharedPrefillArm::K64 if k64_ok => self.w4a16_gemm_t_k64,
            SharedPrefillArm::K64V2 if k64_ok => self.w4a16_gemm_t_k64_v2,
            SharedPrefillArm::M128 if self.w4a16_gemm_t_m128.0 != 0 => {
                // Same kernel signature + weight layout; only the grid changes
                // (M tiled by 128 instead of 64, still 128 threads/CTA).
                return ops::w4a16_gemm_n128_m128(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128,
                    input,
                    weight,
                    output,
                    m,
                    n,
                    k,
                    aux,
                );
            }
            _ => self.w4a16_gemm_t,
        };
        let handle = if handle.0 == 0 {
            self.w4a16_gemm_t
        } else {
            handle
        };
        ops::w4a16_gemm_n128(ctx.gpu, handle, input, weight, output, m, n, k, aux)
    }
}

impl MoeLayer {
    /// Shared-expert path of the prefill pipeline (gate + up GEMM → SiLU →
    /// down GEMM). Runs sequentially on the supplied `aux` stream when
    /// `use_overlap == false`; otherwise issues an event so the routed
    /// path can wait on completion.
    ///
    /// Skips entirely when `shared_inter == 0` (e.g. Qwen3-VL-30B has no
    /// shared expert). Launching kernels with N=0 returns
    /// CUDA_ERROR_INVALID_VALUE (grid.x=0).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_shared_expert_prefill(
        &self,
        input: DevicePtr,
        n: u32,
        h: u32,
        shared_inter: u32,
        aux: u64,
        stream: u64,
        use_overlap: bool,
        ctx: &ForwardContext,
    ) -> Result<()> {
        if shared_inter == 0 {
            return Ok(());
        }
        if use_overlap {
            // Ensure secondary stream sees `input` (produced by prior default-stream work)
            ctx.gpu.record_event(self.event_a, stream)?;
            ctx.gpu.stream_wait_event(aux, self.event_a)?;
        }

        let shared_gate_out = ctx.buffers.ssm_deinterleaved();
        let shared_up_out = ctx.buffers.ssm_qkvz();
        let shared_down_out = ctx.buffers.attn_output();
        if self.run_bf16_shared_expert(
            input,
            n,
            h,
            shared_inter,
            shared_gate_out,
            shared_up_out,
            shared_down_out,
            ctx,
            aux,
        )? {
            if use_overlap {
                ctx.gpu.record_event(self.event_b, aux)?;
            }
            return Ok(());
        }

        // Shared gate + up GEMM on aux stream
        if let (Some(sg_fp8), Some(su_fp8)) = (self.shared_gate_fp8, self.shared_up_fp8) {
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                input,
                sg_fp8,
                shared_gate_out,
                n,
                shared_inter,
                h,
                aux,
            )?;
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                input,
                su_fp8,
                shared_up_out,
                n,
                shared_inter,
                h,
                aux,
            )?;
        } else if let (Some(sg), Some(su), Some(_sd)) =
            (&self.shared_gate_t, &self.shared_up_t, &self.shared_down_t)
        {
            // ATLAS_MOE_SHARED_K64 selects the tiling; default OFF = the
            // historical `w4a16_gemm_t`. See `shared_prefill_arm` for the
            // NVFP4-GS16 layout compatibility argument.
            self.shared_prefill_gemm(input, sg, shared_gate_out, n, shared_inter, h, aux, ctx)?;
            self.shared_prefill_gemm(input, su, shared_up_out, n, shared_inter, h, aux, ctx)?;
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm,
                input,
                &self.weights.shared_expert.gate_proj,
                shared_gate_out,
                n,
                shared_inter,
                h,
                aux,
            )?;
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm,
                input,
                &self.weights.shared_expert.up_proj,
                shared_up_out,
                n,
                shared_inter,
                h,
                aux,
            )?;
        }

        // Shared activation (SiLU or GeGLU) + down GEMM on aux stream
        ops::silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            shared_gate_out,
            shared_up_out,
            shared_gate_out,
            n * shared_inter,
            aux,
        )?;
        if let Some(sd_fp8) = self.shared_down_fp8 {
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                shared_gate_out,
                sd_fp8,
                shared_down_out,
                n,
                h,
                shared_inter,
                aux,
            )?;
        } else if let Some(sd) = &self.shared_down_t {
            self.shared_prefill_gemm(
                shared_gate_out,
                sd,
                shared_down_out,
                n,
                h,
                shared_inter,
                aux,
                ctx,
            )?;
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm,
                shared_gate_out,
                &self.weights.shared_expert.down_proj,
                shared_down_out,
                n,
                h,
                shared_inter,
                aux,
            )?;
        }

        if use_overlap {
            ctx.gpu.record_event(self.event_b, aux)?;
        }
        Ok(())
    }
}
