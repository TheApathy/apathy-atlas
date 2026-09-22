// SPDX-License-Identifier: AGPL-3.0-only

//! `Qwen3AttentionLayer` prefill-side weight setup: transposed NVFP4 /
//! FP8 copies, FP8 weight installation, FP8 transpose for fast prefill,
//! and NVFP4→FP8 pre-dequant for zero-overhead prefill GEMMs. Also
//! hosts the W4A16 M=128 GEMM dispatcher (selects v1/v2/v3 by env).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::types::Qwen3AttentionLayer;
use crate::weight_map::{Fp8Weight, QuantWeight, QuantizedWeight};


/// Attention projections that actually executed on the cuBLASLt route. Read
/// from the server log as `ATTN_CUBLASLT_CALL n=<total>`; a total short of
/// (projections x layers x requests) means the fail-safe fallback ran and the
/// run must not be scored. A boolean "did it engage" cannot see partial
/// engagement — only a count against an expected count can.
static ATTN_CUBLASLT_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl Qwen3AttentionLayer {
    /// BF16 copy of `weight`, materialised once and cached by packed-weight
    /// pointer. Returns None if the kernel is missing or allocation fails, so
    /// the caller keeps the hand-written path.
    fn bf16_weight_for(
        &self,
        gpu: &dyn GpuBackend,
        weight: &crate::weight_map::QuantizedWeight,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Option<DevicePtr> {
        if self.dequant_nvfp4_to_bf16_k.0 == 0 || k % 16 != 0 {
            return None;
        }
        let key = weight.weight.0;
        let mut cache = self.bf16_weight_cache.lock().ok()?;
        if let Some(p) = cache.get(&key) {
            return Some(*p);
        }
        let bytes = n as usize * k as usize * 2;
        let ptr = gpu.alloc(bytes).ok()?;
        crate::layers::ops::dequant_nvfp4_to_bf16(
            gpu,
            self.dequant_nvfp4_to_bf16_k,
            weight.weight,
            weight.weight_scale,
            ptr,
            weight.weight_scale_2,
            n,
            k,
            stream,
        )
        .ok()?;
        tracing::info!(
            "ATLAS_ATTN_PROJ_CUBLASLT: materialised BF16 weight N={n} K={k} ({} MiB)",
            bytes / (1024 * 1024)
        );
        cache.insert(key, ptr);
        Some(ptr)
    }

    /// Exact original-layout NVFP4 prefill projection. Prefers the 128x128
    /// byte-exact shadow (`ATLAS_PREFILL_PROJ_PIPE_M128=1`), then the 64x64
    /// pipe shadow (`ATLAS_PREFILL_PROJ_PIPE=1`), then the baseline
    /// `w4a16_gemm`. All three produce bit-identical output; only speed differs.
    /// An explicitly requested eligible route with a missing symbol fails closed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn exact_prefill_projection(
        &self,
        gpu: &dyn GpuBackend,
        label: &'static str,
        input: DevicePtr,
        weight: &crate::weight_map::QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> anyhow::Result<()> {
        use crate::layers::PrefillProjectionPipeRoute as Route;
        // cuBLASLt BF16: identical operands to the W4A16 kernel (same
        // __float2bfloat16 dequant, BF16 activations untouched); only the
        // FP32 accumulation order differs. Fail-safe.
        if crate::layers::attn_proj_cublaslt_enabled()
            && let Some(w) = self.bf16_weight_for(gpu, weight, n, k, stream)
            && spark_runtime::cublaslt::bf16_gemm_act_weight_t_tuned(
                input.0, w.0, output.0, m, n, k, stream,
            )
            .is_ok()
        {
            let calls =
                ATTN_CUBLASLT_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            tracing::info!("ATTN_CUBLASLT_CALL n={calls} {label} M={m} N={n} K={k}");
            return Ok(());
        }
        match crate::layers::prefill_projection_pipe_m128_route(
            crate::layers::prefill_proj_pipe_m128_enabled(),
            n,
            k,
            self.w4a16_gemm_pipe_m128n128_k.0 != 0,
        ) {
            Route::Complete => {
                static SEEN: std::sync::Once = std::sync::Once::new();
                SEEN.call_once(|| {
                    tracing::info!("ENGAGED ATLAS_PREFILL_PROJ_PIPE_M128: {label} (first) M={m} N={n} K={k}");
                });
                return crate::layers::ops::w4a16_gemm_pipe_m128n128(
                    gpu, self.w4a16_gemm_pipe_m128n128_k, input, weight, output, m, n, k, stream,
                );
            }
            Route::Missing => anyhow::bail!(
                "ATLAS_PREFILL_PROJ_PIPE_M128=1 requires w4a16_gemm_pipe_m128n128 for {label}"
            ),
            Route::Disabled | Route::Ineligible => {}
        }
        match crate::layers::prefill_projection_pipe_route(
            crate::layers::prefill_proj_pipe_enabled(),
            k,
            self.w4a16_gemm_pipe_k.0 != 0,
        ) {
            Route::Complete => {
                static SEEN_PIPE: std::sync::Once = std::sync::Once::new();
                SEEN_PIPE.call_once(|| {
                    tracing::info!("ENGAGED ATLAS_PREFILL_PROJ_PIPE: {label} (first)");
                });
                crate::layers::ops::w4a16_gemm_pipe(
                    gpu, self.w4a16_gemm_pipe_k, input, weight, output, m, n, k, stream,
                )
            }
            Route::Missing => anyhow::bail!(
                "ATLAS_PREFILL_PROJ_PIPE=1 requires w4a16_gemm_pipe for {label}"
            ),
            Route::Disabled | Route::Ineligible => crate::layers::ops::w4a16_gemm(
                gpu, self.w4a16_gemm_k, input, weight, output, m, n, k, stream,
            ),
        }
    }

    /// Dispatch the M=128 W4A16 prefill GEMM. Routes to the v2 shadow
    /// kernel when available (MiniMax-only), otherwise to the v1 kernel.
    /// Args mirror [`crate::layers::ops::w4a16_gemm_n128_m128`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn w4a16_gemm_m128_dispatch(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &crate::weight_map::QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> anyhow::Result<()> {
        // ATLAS_W4A16_VARIANT env: "v1", "v2", "v3" — overrides auto.
        // Default: v2 (3 CTAs/SM, 8 warps). v3 (K_STEP=64, 1 CTA/SM) is
        // slower in practice; keep it available for A/B but don't default.
        static VARIANT: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
        let v =
            *VARIANT.get_or_init(
                || match std::env::var("ATLAS_W4A16_VARIANT").ok().as_deref() {
                    Some("v1") => 1,
                    Some("v2") => 2,
                    Some("v3") => 3,
                    _ => 0, // auto (prefer v2)
                },
            );
        {
            use crate::layers::PrefillProjectionPipeRoute as Route;
            match crate::layers::prefill_fp8_w8_route(
                crate::layers::prefill_fp8_w8_enabled(), m, n, k, self.w4a16_gemm_t_w8_k.0 != 0,
            ) {
                Route::Complete => {
                    static SEEN: std::sync::Once = std::sync::Once::new();
                    SEEN.call_once(|| tracing::info!("ENGAGED ATLAS_PREFILL_FP8_W8: attention w4a16_t M={m} N={n} K={k}"));
                    return crate::layers::ops::w4a16_gemm_t_w8(
                        gpu, self.w4a16_gemm_t_w8_k, input, weight, output, m, n, k, stream,
                    );
                }
                Route::Missing => anyhow::bail!("ATLAS_PREFILL_FP8_W8=1 requires w4a16_gemm_t_m128n128_w8 (attention)"),
                Route::Disabled | Route::Ineligible => {}
            }
        }
        if v == 3 && self.w4a16_gemm_t_m128_v3_k.0 != 0 {
            crate::layers::ops::w4a16_gemm_n128_m128_v3(
                gpu,
                self.w4a16_gemm_t_m128_v3_k,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            )
        } else if v != 1 && self.w4a16_gemm_t_m128_v2_k.0 != 0 {
            crate::layers::ops::w4a16_gemm_n128_m128_v2(
                gpu,
                self.w4a16_gemm_t_m128_v2_k,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            )
        } else {
            crate::layers::ops::w4a16_gemm_n128_m128(
                gpu,
                self.w4a16_gemm_t_m128_k,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            )
        }
    }

    /// Set transposed NVFP4 weight copies for prefill GEMM
    /// (`w4a16_gemm_t`, N_TILE=128).
    pub fn set_prefill_weights(
        &mut self,
        q_nvfp4_t: Option<QuantizedWeight>,
        k_nvfp4_t: Option<QuantizedWeight>,
        v_nvfp4_t: Option<QuantizedWeight>,
        o_nvfp4_t: Option<QuantizedWeight>,
    ) {
        self.q_nvfp4_t = q_nvfp4_t;
        self.k_nvfp4_t = k_nvfp4_t;
        self.v_nvfp4_t = v_nvfp4_t;
        self.o_nvfp4_t = o_nvfp4_t;
    }

    /// Eagerly allocate the FP32 split-K workspace for the K/V projections
    /// on the K=γ verify QKV path (`ATLAS_ATTN_QKV_SPLITK`). `n` is the
    /// largest split-K output dim (kv_dim = num_kv_heads*head_dim). Called
    /// at load time (pre-graph-capture) because `gpu.alloc` is illegal
    /// during CUDA graph capture. No-op when split-K is disabled or the
    /// split-K kernel symbols are missing. Sized `[max_splits, M_TILE=32, n]`
    /// FP32 so K and V (run back-to-back, each fully reduced before the
    /// next partial phase) can share the one scratch.
    pub fn alloc_qkv_splitk_workspace(&self, gpu: &dyn GpuBackend, n: u32) -> anyhow::Result<()> {
        if crate::layers::attn_qkv_splitk() == 0
            || self.w4a16_gemm_t_m32_n64_splitk_k.0 == 0
            || self.reduce_splitk_k.0 == 0
        {
            return Ok(());
        }
        let mut slot = self.qkv_splitk_workspace.lock().unwrap();
        if slot.is_none() {
            // 8 = max split clamp; 32 = M_TILE of the split-K kernel.
            let bytes = 8usize * 32 * n as usize * 4;
            *slot = Some(gpu.alloc(bytes)?);
        }
        Ok(())
    }

    /// Set native FP8 checkpoint weights for `w8a16_gemv` decode path.
    ///
    /// NOTE: Does NOT set `q_fp8`/`k_fp8`/`v_fp8`/`o_fp8` (raw FP8
    /// prefill pointers) because `fp8_gemm_t` doesn't apply block scales.
    /// Native FP8 block-scaled weights need per-block scale during GEMM,
    /// which `fp8_gemm_t` doesn't do. Prefill falls through to the
    /// NVFP4/BF16 dequant path instead.
    pub fn set_fp8_weights(
        &mut self,
        q: Option<Fp8Weight>,
        k: Option<Fp8Weight>,
        v: Option<Fp8Weight>,
        o: Option<Fp8Weight>,
    ) {
        // Overwrite decode weights with FP8 variant. Replaces any NVFP4
        // weights set during construction.
        if let Some(qw) = q {
            self.q_weight = Some(QuantWeight::Fp8(qw));
        }
        if let Some(kw) = k {
            self.k_weight = Some(QuantWeight::Fp8(kw));
        }
        if let Some(vw) = v {
            self.v_weight = Some(QuantWeight::Fp8(vw));
        }
        if let Some(ow) = o {
            self.o_weight = Some(QuantWeight::Fp8(ow));
        }
    }

    /// Transpose FP8 weights for fast prefill (`w8a16_gemm_t`: coalesced
    /// reads). Must be called after [`Self::set_fp8_weights`]. Allocates
    /// new GPU buffers.
    pub fn transpose_fp8_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> anyhow::Result<()> {
        if self.w8a16_gemm_t_k.0 == 0 {
            return Ok(()); // kernel not available
        }
        let transpose_k = gpu.kernel("w8a16_gemm_t", "transpose_fp8")?;
        let transpose_scale_k = gpu.kernel("w8a16_gemm_t", "transpose_block_scale")?;

        if let Some(w) = self.q_weight.as_ref().and_then(|w| w.as_fp8()) {
            self.q_fp8w_t =
                Some(w.transpose_for_gemm(gpu, transpose_k, transpose_scale_k, stream)?);
        }
        if let Some(w) = self.k_weight.as_ref().and_then(|w| w.as_fp8()) {
            self.k_fp8w_t =
                Some(w.transpose_for_gemm(gpu, transpose_k, transpose_scale_k, stream)?);
        }
        if let Some(w) = self.v_weight.as_ref().and_then(|w| w.as_fp8()) {
            self.v_fp8w_t =
                Some(w.transpose_for_gemm(gpu, transpose_k, transpose_scale_k, stream)?);
        }
        if let Some(w) = self.o_weight.as_ref().and_then(|w| w.as_fp8()) {
            self.o_fp8w_t =
                Some(w.transpose_for_gemm(gpu, transpose_k, transpose_scale_k, stream)?);
        }
        Ok(())
    }

    /// Pre-dequant NVFP4 → FP8 for Q/K/V/O transposed weights.
    pub fn predequant_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        stream: u64,
    ) -> Result<()> {
        let predequant_k = gpu.kernel("w4a16", "predequant_nvfp4_to_fp8")?;
        let h = config.hidden_size;
        let nq = config.num_attention_heads;
        let nkv = config.num_key_value_heads;
        let hd = config.head_dim;
        let q_dim = nq * hd;
        let q_proj_dim = if self.gated { q_dim * 2 } else { q_dim };
        let kv_dim = nkv * hd;

        // Use NON-transposed weights for predequant.
        // `predequant_nvfp4_to_fp8` assumes [N, K/2] input layout.
        if let Some(nvfp4) = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()) {
            self.q_fp8 = Some(nvfp4.predequant_to_fp8(gpu, predequant_k, q_proj_dim, h, stream)?);
        }
        if let Some(nvfp4) = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()) {
            self.k_fp8 = Some(nvfp4.predequant_to_fp8(gpu, predequant_k, kv_dim, h, stream)?);
        }
        if let Some(nvfp4) = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()) {
            self.v_fp8 = Some(nvfp4.predequant_to_fp8(gpu, predequant_k, kv_dim, h, stream)?);
        }
        // O proj: use attn.o_proj (non-transposed QuantizedWeight)
        if self.o_nvfp4_t.is_some() {
            self.o_fp8 =
                Some(
                    self.attn
                        .o_proj
                        .predequant_to_fp8(gpu, predequant_k, h, q_dim, stream)?,
                );
        }
        Ok(())
    }
}
