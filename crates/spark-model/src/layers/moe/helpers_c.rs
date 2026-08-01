// SPDX-License-Identifier: AGPL-3.0-only

//! Shared-expert precision setup, predequantization, and router input.

use super::*;

impl MoeLayer {
    /// Pre-dequant dense (non-expert) NVFP4 weights to FP8 for zero-overhead prefill.
    ///
    /// Only affects gate GEMM and shared expert GEMMs.  Expert weights stay NVFP4
    /// (they're bandwidth-bound so FP8 wouldn't help).
    pub fn predequant_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        stream: u64,
    ) -> Result<()> {
        let h = config.hidden_size;
        let shared_inter = config.shared_expert_intermediate_size;
        let num_experts = config.num_experts;
        let predequant_k = gpu.kernel("w4a16", "predequant_nvfp4_to_fp8")?;

        // Pre-dequant gate weight: [num_experts, H] → FP8 [num_experts, H]
        if let Some(ref nvfp4) = self.gate_nvfp4 {
            self.gate_fp8 =
                Some(nvfp4.predequant_to_fp8(gpu, predequant_k, num_experts, h, stream)?);
        }

        // A checkpoint-native BF16 shared expert is the authoritative copy.
        // Do not manufacture an FP8 prefill variant with different numerics.
        if self.bf16_shared_expert.is_none()
            && !self.weights.shared_expert.gate_proj.is_null()
            && shared_inter > 0
        {
            self.shared_gate_fp8 = Some(self.weights.shared_expert.gate_proj.predequant_to_fp8(
                gpu,
                predequant_k,
                shared_inter,
                h,
                stream,
            )?);
            self.shared_up_fp8 = Some(self.weights.shared_expert.up_proj.predequant_to_fp8(
                gpu,
                predequant_k,
                shared_inter,
                h,
                stream,
            )?);
            self.shared_down_fp8 = Some(self.weights.shared_expert.down_proj.predequant_to_fp8(
                gpu,
                predequant_k,
                h,
                shared_inter,
                stream,
            )?);
        }

        Ok(())
    }

    /// Set FP8 expert weights for native FP8 dispatch.
    ///
    /// Builds device-side pointer tables from FP8 expert weights so the
    /// fused FP8 MoE kernel can index by expert_id at dispatch time.
    /// Also stores the shared expert FP8 weights for direct pointer passing.
    pub fn set_fp8_experts(
        &mut self,
        experts: &[Fp8ExpertWeight],
        shared_expert: Fp8ExpertWeight,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        self.fp8_gate_weight_ptrs = Some(build_fp8_ptr_table(experts, |e| &e.gate_proj, gpu)?);
        self.fp8_up_weight_ptrs = Some(build_fp8_ptr_table(experts, |e| &e.up_proj, gpu)?);
        self.fp8_down_weight_ptrs = Some(build_fp8_ptr_table(experts, |e| &e.down_proj, gpu)?);
        self.fp8_shared_expert = Some(shared_expert);
        Ok(())
    }

    /// Set BF16 expert weights for the FP8-dequant-on-load MoE path.
    ///
    /// Activated by `ATLAS_FP8_DEQUANT_MOE_TO_BF16=1`. Eliminates the per-layer
    /// 0.989 FP8 cosine ceiling (measured in bench/fp8_dgx2_drift/cosine_run.py)
    /// by serving experts as BF16 throughout, matching vLLM-BF16 reference
    /// numerics. Memory cost: 2× expert weights vs native FP8.
    ///
    /// `shared_*` are the shared expert's BF16 gate/up/down DevicePtrs (or
    /// `DevicePtr::NULL` when the model has no shared expert).
    pub fn set_bf16_experts(
        &mut self,
        gate_experts: &[crate::weight_map::DenseWeight],
        up_experts: &[crate::weight_map::DenseWeight],
        down_experts: &[crate::weight_map::DenseWeight],
        shared_gate: DevicePtr,
        shared_up: DevicePtr,
        shared_down: DevicePtr,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        use super::build_bf16_ptr_table;
        self.bf16_gate_weight_ptrs = Some(build_bf16_ptr_table(gate_experts, gpu)?);
        self.bf16_up_weight_ptrs = Some(build_bf16_ptr_table(up_experts, gpu)?);
        self.bf16_down_weight_ptrs = Some(build_bf16_ptr_table(down_experts, gpu)?);
        if shared_gate.is_null() && shared_up.is_null() && shared_down.is_null() {
            self.bf16_shared_expert = None;
        } else {
            self.set_bf16_shared_expert(
                DenseWeight {
                    weight: shared_gate,
                },
                DenseWeight { weight: shared_up },
                DenseWeight {
                    weight: shared_down,
                },
            )?;
        }
        Ok(())
    }

    /// Install checkpoint-native BF16 shared-expert weights independently of
    /// routed-expert precision.
    pub fn set_bf16_shared_expert(
        &mut self,
        gate_proj: DenseWeight,
        up_proj: DenseWeight,
        down_proj: DenseWeight,
    ) -> Result<()> {
        self.bf16_shared_expert = Some(Bf16SharedExpert::new(gate_proj, up_proj, down_proj)?);
        Ok(())
    }

    /// Build the FP8-E4M3 row-scaled MIRROR of the BF16 shared expert
    /// (ATLAS_TARGET_SHARED_FP8=1). Returns the device bytes allocated, or 0
    /// when the mirror was not built.
    ///
    /// Mirrors `build_attn_fp8_mirrors` exactly, including its soft-fail
    /// contract: a missing kernel, an absent BF16 shared expert, or an
    /// allocation failure all fall back to the BF16 path and must NEVER abort
    /// model load.
    ///
    /// Shapes (checkpoint `[N, K]` row-major): gate/up `[shared_inter, h]`,
    /// down `[h, shared_inter]`.
    pub fn build_shared_fp8_mirror(
        &mut self,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        h: usize,
        shared_inter: usize,
    ) -> Result<usize> {
        use std::sync::Once;
        static WARN_KERNELS: Once = Once::new();
        static WARN_SOURCE: Once = Once::new();
        static WARN_ALLOC: Once = Once::new();

        if std::env::var("ATLAS_TARGET_SHARED_FP8").ok().as_deref() != Some("1") {
            return Ok(0);
        }
        let quantize_k = crate::layers::try_kernel(gpu, "gemv_fp8w", "quantize_bf16_to_fp8");
        if quantize_k.0 == 0 || self.dense_gemv_fp8w_k.0 == 0 {
            WARN_KERNELS.call_once(|| {
                tracing::warn!(
                    "ATLAS_TARGET_SHARED_FP8=1 but quantize_bf16_to_fp8 / dense_gemv_fp8w \
                     are absent from this kernel set — shared expert stays BF16"
                );
            });
            return Ok(0);
        }
        let Some(shared) = self.bf16_shared_expert else {
            WARN_SOURCE.call_once(|| {
                tracing::warn!(
                    "ATLAS_TARGET_SHARED_FP8=1 but this layer has no BF16 shared expert — \
                     nothing to mirror"
                );
            });
            return Ok(0);
        };
        anyhow::ensure!(
            h > 0 && shared_inter > 0,
            "shared-expert FP8 mirror requires non-zero hidden/intermediate dims"
        );

        let stream = gpu.default_stream();
        let mut built: Vec<crate::weight_map::Fp8DenseWeight> = Vec::with_capacity(3);
        let specs: [(&DenseWeight, usize, usize); 3] = [
            (&shared.gate_proj, shared_inter, h),
            (&shared.up_proj, shared_inter, h),
            (&shared.down_proj, h, shared_inter),
        ];
        for (src, n, k) in specs {
            match src.quantize_to_fp8(gpu, quantize_k, n, k, stream) {
                Ok(m) => built.push(m),
                Err(e) => {
                    for m in built {
                        let _ = gpu.free(m.weight);
                        let _ = gpu.free(m.row_scale);
                    }
                    WARN_ALLOC.call_once(|| {
                        tracing::warn!(
                            "ATLAS_TARGET_SHARED_FP8=1: shared-expert mirror quantization \
                             failed ({e:#}); this layer (and any later failures) stay BF16"
                        );
                    });
                    return Ok(0);
                }
            }
        }
        // Order matches `specs`: gate, up, down.
        let down_proj = built.pop().expect("down mirror");
        let up_proj = built.pop().expect("up mirror");
        let gate_proj = built.pop().expect("gate mirror");
        self.fp8_shared_expert_mirror = Some(Fp8SharedExpertMirror {
            gate_proj,
            up_proj,
            down_proj,
        });
        // Load-time proof the mirror is armed: log-grepping layer 0 is useless
        // here because layer 0 is dense (no shared expert), so the first MoE
        // layer to build a mirror announces it once for the whole model.
        {
            static ANNOUNCE: std::sync::Once = std::sync::Once::new();
            ANNOUNCE.call_once(|| {
                tracing::info!(
                    "ATLAS_TARGET_SHARED_FP8=1: FP8 row-scaled shared-expert mirrors ARMED \
                     (first MoE layer built; ~{:.1} MiB per MoE layer, decode GEMVs only)",
                    (2 * shared_inter * h + h * shared_inter) as f64 / (1024.0 * 1024.0)
                );
            });
        }
        let scale = std::mem::size_of::<f32>();
        Ok(2 * shared_inter * h + h * shared_inter + scale * (2 * shared_inter + h))
    }

    /// Whether a BF16 shared expert must overwrite the contribution produced
    /// by a quantized fused routed-expert kernel.
    pub(super) fn has_mixed_bf16_shared_expert(&self) -> bool {
        self.bf16_shared_expert.is_some() && self.bf16_gate_weight_ptrs.is_none()
    }

    /// ATLAS_MOE_SKIP_PLACEHOLDER_SHARED=1: in the mixed NVFP4-routed /
    /// BF16-shared config (Laguna S-2.1), drop the shared expert slot from the
    /// fused routed kernel's grid entirely.
    ///
    /// Rationale: when `has_mixed_bf16_shared_expert()` is true the NVFP4
    /// shared weights are load-time PLACEHOLDERS (weight_loader/laguna/
    /// load_layers.rs excludes the shared expert from NVFP4; BF16 is
    /// authoritative). The fused kernel computes them into
    /// shared_gate/up/out scratch, and `run_bf16_shared_expert` then
    /// unconditionally OVERWRITES all three from BF16. So the fused kernel's
    /// shared slot is pure waste: ~5.3 MB/layer x 47 layers = ~250 MB/token
    /// read and discarded (~1.0 ms/token at a 245 GB/s wall).
    ///
    /// The kernels select the shared expert solely via `blockIdx.y == top_k`
    /// (moe_shared_expert_fused.cu), so launching grid.y = top_k instead of
    /// top_k+1 removes exactly those blocks and touches no routed slot —
    /// BIT-EXACT by construction.
    ///
    /// The batch2/batch3 verify paths (forward_k2/forward_k3) already achieve
    /// this by passing NULL shared weights; only the M=1 serial decode path
    /// still pays for the placeholder.
    pub(super) fn skip_placeholder_shared(&self) -> bool {
        static SKIP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *SKIP.get_or_init(|| {
            std::env::var("ATLAS_MOE_SKIP_PLACEHOLDER_SHARED")
                .ok()
                .as_deref()
                == Some("1")
        }) && self.has_mixed_bf16_shared_expert()
    }

    /// ATLAS_MOE_GATE_GEMV=1 (default OFF): run the router gate through
    /// `dense_gemv_bf16_batchm` instead of `dense_gemm_bf16` at small `n`.
    ///
    /// Read once via OnceLock so the choice cannot differ between a CUDA-graph
    /// capture and its replays (same discipline as
    /// `dflash_propose_onegraph_enabled`). The caller additionally checks the
    /// kernel handle and the `n <= DENSE_GEMV_BATCHM_MAX_M` bound, so this is
    /// only the operator's opt-in, not the whole predicate.
    ///
    /// Default OFF because this one is NOT bit-exact — see the dispatch comment
    /// in `forward_kn.rs`.
    pub(super) fn moe_gate_gemv() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("ATLAS_MOE_GATE_GEMV").ok().as_deref() == Some("1"))
    }

    /// Evaluate a checkpoint-native BF16 shared expert into `down_out`.
    ///
    /// Callers supply scratch buffers because the safe aliases differ between
    /// decode and prefill. Returns `true` when BF16 weights were installed.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_bf16_shared_expert(
        &self,
        input: DevicePtr,
        num_tokens: u32,
        hidden_size: u32,
        shared_intermediate: u32,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        down_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(shared) = self.bf16_shared_expert else {
            return Ok(false);
        };
        anyhow::ensure!(
            num_tokens > 0 && shared_intermediate > 0,
            "BF16 shared expert requires non-zero token and intermediate dimensions"
        );

        // ATLAS_TARGET_SHARED_FP8=1: the M=1 decode GEMVs read the FP8-E4M3
        // row-scaled mirror instead of the BF16 original — half the weight
        // bytes (887 -> ~444 MB/token across 47 layers). Multi-token and
        // prefill paths below deliberately keep BF16: the mirror exists only
        // to relieve the decode bandwidth wall, and cuBLASLt BF16 already wins
        // at those shapes. NOT bit-exact (quantization) — quality-gated.
        let mirror = self.fp8_shared_expert_mirror;
        let project = |activation: DevicePtr,
                       weight: &DenseWeight,
                       fp8: Option<&crate::weight_map::Fp8DenseWeight>,
                       output: DevicePtr,
                       n: u32,
                       k: u32|
         -> Result<()> {
            if num_tokens == 1 {
                if let Some(w8) = fp8
                    && self.dense_gemv_fp8w_k.0 != 0
                {
                    return ops::dense_gemv_fp8w(
                        ctx.gpu,
                        self.dense_gemv_fp8w_k,
                        activation,
                        w8,
                        output,
                        n,
                        k,
                        stream,
                    );
                }
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv,
                    activation,
                    weight,
                    output,
                    n,
                    k,
                    stream,
                )
            } else if ops::cublas_gemm_enabled() {
                // Multi-token shared expert: cuBLASLt BF16 beats the hand-written
                // mma.sync GEMM. The single-token arm above stays on the GEMV —
                // decode-sized shapes do not repay cuBLAS heuristic overhead.
                ops::cublas_bf16_proj_dense(
                    activation,
                    weight.weight,
                    output,
                    num_tokens,
                    n,
                    k,
                    stream,
                )
            } else {
                ops::dense_gemm_prefill(
                    ctx.gpu,
                    self.dense_gemm,
                    self.dense_gemm_pipelined,
                    activation,
                    weight,
                    output,
                    num_tokens,
                    n,
                    k,
                    stream,
                )
            }
        };

        project(
            input,
            &shared.gate_proj,
            mirror.as_ref().map(|m| &m.gate_proj),
            gate_out,
            shared_intermediate,
            hidden_size,
        )?;
        project(
            input,
            &shared.up_proj,
            mirror.as_ref().map(|m| &m.up_proj),
            up_out,
            shared_intermediate,
            hidden_size,
        )?;
        ops::silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            gate_out,
            up_out,
            gate_out,
            num_tokens * shared_intermediate,
            stream,
        )?;
        project(
            gate_out,
            &shared.down_proj,
            mirror.as_ref().map(|m| &m.down_proj),
            down_out,
            hidden_size,
            shared_intermediate,
        )?;
        Ok(true)
    }

    /// Apply the router pre-normalization (Gemma-4 only) and return the
    /// pointer that should be fed into the gate GEMV. If the MoE has no
    /// router_pre_norm weight, this is a no-op and returns `input` unchanged.
    ///
    /// HF Gemma4TextRouter computes:
    ///   router_input = rms_norm(x) * scale * hidden_size^(-0.5)
    /// We fused `scale * root_size` into a single BF16 weight at load time
    /// so the existing rms_norm kernel applies both steps in one pass.
    ///
    /// The normed output is written to `ctx.buffers.qkv_output()` which is
    /// free at MoE time (the attention block already consumed qkv_output).
    pub(super) fn router_input(
        &self,
        input: DevicePtr,
        num_tokens: u32,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let Some(ref weight) = self.weights.router_pre_norm else {
            return Ok(input);
        };
        let eps = ctx.config.rms_norm_eps as f32;
        let normed = ctx.buffers.qkv_output();
        ops::rms_norm(
            ctx.gpu,
            self.pre_expert_norm_k,
            input,
            weight,
            normed,
            num_tokens,
            h,
            eps,
            stream,
        )?;
        Ok(normed)
    }
}
