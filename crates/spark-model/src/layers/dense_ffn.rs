// SPDX-License-Identifier: AGPL-3.0-only

//! Dense SwiGLU FFN component for non-MoE models.
//!
//! Forward: gate = gate_proj(x), up = up_proj(x), out = down_proj(SiLU(gate) * up)
//! 2 fused kernel launches per decode token (dual GEMV + SiLU-fused down GEMV).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use std::sync::Mutex;

use crate::layer::ForwardContext;
use crate::layers::ffn_dual_tuned_enabled;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, QuantizedWeight};

/// Scratch buffers for the inline BF16 → NVFP4 activation prequant
/// (W4A4 `nvfp4_nvfp4_gemm` fast path). Lazily allocated on first prefill
/// call, then resized in-place if M or K grows.
///
/// Sizes (per row M, per col K):
///   - `a_packed`: M × K/2 bytes (E2M1 nibbles)
///   - `a_scale`:  M × K/16 bytes (FP8 E4M3 per-group scales)
///   - `a_max`:    4 bytes (FP32 per-tensor absmax scratch)
///
/// One arena per `DenseFfnLayer` (= per transformer layer). Reused across
/// gate / up / down GEMMs within a layer: gate/up share K=H, down has
/// K=Inter, so the arena is sized for `max(H, Inter)` × `max_M`.
struct E2m1Scratch {
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    a_max: DevicePtr,
    /// Current row capacity (M) the buffers can hold.
    cap_m: usize,
    /// Current column capacity (K) the buffers can hold (full K, not K/2).
    cap_k: usize,
}

pub struct DenseFfnWeights {
    pub gate_proj: QuantizedWeight,
    pub up_proj: QuantizedWeight,
    pub down_proj: QuantizedWeight,
}

/// BF16 dense MLP weights — alternative to NVFP4 for precision-sensitive
/// models (Gemma-4-31B). Each is `[N, K]` row-major BF16. When installed
/// on a `DenseFfnLayer` via `set_bf16_weights`, the forward paths
/// dispatch to `dense_gemv_bf16` / `dense_gemm_bf16` instead of the
/// w4a16 NVFP4 kernels. Costs ~3.4 GB extra GPU memory on Gemma-4-31B
/// (3 × hidden×intermediate × 2 bytes) vs NVFP4's 0.5 bytes/weight.
pub struct DenseFfnWeightsBf16 {
    pub gate_proj: DenseWeight,
    pub up_proj: DenseWeight,
    pub down_proj: DenseWeight,
}

/// Activation function for gated FFN (SiLU for Qwen/Llama, GELU for Gemma-4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnActivation {
    SiLU,
    GeLU,
}

pub struct DenseFfnLayer {
    pub weights: DenseFfnWeights,
    activation: FfnActivation,
    w4a16_gemv: KernelHandle,
    w4a16_gemv_dual: KernelHandle,
    w4a16_gemv_silu_input: KernelHandle,
    w4a16_gemv_dual_batch2: KernelHandle,
    w4a16_gemv_dual_batch3: KernelHandle,
    /// Tuned dual-batch3 variant: fuses gate+up into the SAME CTA so the
    /// 3-token activation vector is loaded once per CTA. Gated behind the
    /// `ATLAS_FFN_DUAL_TUNED=1` env var; falls back to the baseline kernel
    /// when off. Loaded via `try_kernel` so older built kernel caches that
    /// pre-date this symbol still link.
    w4a16_gemv_dual_batch3_tuned: KernelHandle,
    w4a16_gemv_batch2: KernelHandle,
    w4a16_gemv_batch3: KernelHandle,
    w4a16_gemm: KernelHandle,
    /// SiLU(gate)*up or GELU(gate)*up depending on activation.
    act_mul: KernelHandle,
    /// BF16 dense MLP weights — when `Some`, all forward paths use the
    /// `dense_gemv_bf16` / `dense_gemm_bf16` kernels instead of w4a16
    /// NVFP4. Falls back to the NVFP4 weights when `None`. Set via
    /// `set_bf16_weights`. Used by Gemma-4 dense to avoid the structural
    /// NVFP4 attention drift on greedy code generation (the fib test's
    /// broken-indentation pattern).
    bf16_weights: Option<DenseFfnWeightsBf16>,
    dense_gemv_bf16_k: KernelHandle,
    dense_gemm_bf16_k: KernelHandle,
    /// Transposed (`nvfp4_t` layout) FFN projections — populated only when
    /// `ATLAS_FFN_M16_TRANSPOSED=1` and the loader successfully built the
    /// transposed copies via `QuantizedWeight::transpose_for_gemm`. When
    /// `Some`, `forward_kgamma` routes gate/up/down through the
    /// `w4a16_gemm_n128_m16` (M_TILE=16) kernel which has near-zero MMA
    /// accumulator waste at M=γ+1 (typically 17). Falls back to the
    /// non-transposed `w4a16_gemm` (M_TILE=64) when `None` OR when the
    /// `w4a16_gemm_t_m16` kernel symbol is missing.
    gate_proj_t: Option<QuantizedWeight>,
    up_proj_t: Option<QuantizedWeight>,
    down_proj_t: Option<QuantizedWeight>,
    /// `w4a16_gemm_t_m16` — small-M (M_TILE=16) transposed-weight GEMM.
    /// Loaded via `try_kernel`; handle is 0 when the symbol is missing
    /// (older PTX caches), in which case `forward_kgamma` always uses the
    /// non-transposed fallback regardless of the transposed weights.
    w4a16_gemm_t_m16: KernelHandle,
    /// `w4a16_gemm_t_m16_n64` — small-M (M_TILE=16), small-N (N_TILE=64)
    /// transposed-weight GEMM, tuned for the K=3 MTP verify path on dense
    /// Qwen3.6-27B (M=3 padded to MMA-16). At intermediate=17408 the N=128
    /// parent only fields ~136 CTAs/projection (1.2 CTAs/SM on GB10) so
    /// the SMs are starved; N=64 doubles the grid to ~272 CTAs/projection
    /// at half the per-CTA work. Loaded via `try_kernel`; handle 0 falls
    /// back to the GEMV path silently.
    w4a16_gemm_t_m16_n64: KernelHandle,
    /// `w4a16_gemm_t_m128` — large-M (M_TILE=128, N_TILE=128)
    /// transposed-weight GEMM. Loaded via `try_kernel`; handle is 0 when
    /// the symbol is missing. Used by `forward_prefill` when
    /// `ATLAS_PREFILL_FFN_FAST=1` AND transposed weights are installed
    /// AND M >= 128. Mirrors the attention `w4a16_gemm_t_m128_k`
    /// dispatch in `qwen3_attention/prefill_weights.rs`. Designed for
    /// large-M prefill: kernel comment claims ~2x speedup over
    /// `w4a16_gemm_t` at ISL>128. For Qwen3.6-27B prefill at M=3575,
    /// N=17408 (gate/up dual): grid=(136, 28)=3808 CTAs vs the default
    /// (272, 56)=15232 CTAs — 4x fewer CTAs but 4x more work per CTA,
    /// and ~2x less weight DRAM traffic.
    w4a16_gemm_t_m128: KernelHandle,
    /// `w4a16_gemm_t_m32_n64` — DFlash K=17 verify specialization:
    /// single B read (one 32-row M-tile) × 272 CTAs (N_TILE=64).
    /// Loaded via `try_kernel`; 0 falls back to m128/m16.
    w4a16_gemm_t_m32_n64: KernelHandle,
    /// `w4a16_gemm_t_m128_v2` — 8-warp (blockDim 256) shadow of
    /// `w4a16_gemm_t_m128`. Same 2-stage cp.async pipeline + same SMEM
    /// footprint (~29.8KB → 3 CTAs/SM), but parallelizes chunk 0 and
    /// chunk 1 MMA computation across warps {0-3} and {4-7} instead of
    /// serializing both chunks on 4 warps. Yields 2× more warps/SM (768
    /// vs 384) → more MMA pipeline slots in flight. Originally a
    /// MiniMax-only kernel (kernels/gb10/minimax-m2-229b/nvfp4/
    /// w4a16_gemm_v2.cu) — copied verbatim into the qwen3.6-27b target
    /// so the FFN prefill GEMM can route through it. Gated by
    /// `ATLAS_FFN_M128_V2=1`. Loaded via `try_kernel`; handle 0 keeps
    /// the v1 path as silent fallback.
    w4a16_gemm_t_m128_v2: KernelHandle,
    /// `fp8_gemm_t_m128` — large-M FP8×FP8 GEMM kernel for pre-dequanted
    /// FFN weights. Loaded via `try_kernel`. When set together with
    /// `gate_fp8`/`up_fp8`/`down_fp8` (installed by
    /// `predequant_for_prefill`), the FFN prefill bypasses NVFP4 dequant
    /// entirely — saving 1 __syncthreads + the 16-iteration dequant
    /// phase per K-step inside `w4a16_gemm_t_m128`. Mirrors the
    /// attention `predequant_for_prefill` + `fp8_gemm_n128_m128`
    /// pattern (qwen3_attention/prefill_weights.rs:161).
    fp8_gemm_t_m128_k: KernelHandle,
    /// Pre-dequanted FP8 [N, K] gate weight. `Some` only when
    /// `ATLAS_FFN_PREDEQUANT_FP8=1` is set at startup and the
    /// `predequant_nvfp4_to_fp8` + `fp8_gemm_t_m128` kernels are present.
    /// Memory cost: N×K bytes per projection (e.g. 17408×5120 = 89 MB
    /// for gate/up, 5120×17408 = 89 MB for down → ~17 GB per Qwen3.6-27B
    /// run, ~270 MB per layer × 64 layers). Roughly DOUBLES the FFN
    /// weight footprint vs NVFP4-only — gate by intent.
    gate_fp8: Option<DevicePtr>,
    up_fp8: Option<DevicePtr>,
    down_fp8: Option<DevicePtr>,
    /// W4A4 NVFP4×NVFP4 native tensor-core GEMM
    /// (`nvfp4_cutlass::nvfp4_nvfp4_gemm_t_m64`). Loaded via `try_kernel`
    /// — handle 0 silently disables the `ATLAS_E2M1_GEMM` fast path.
    nvfp4_gemm_k: KernelHandle,
    /// `quantize_nvfp4::nvfp4_global_absmax` — per-tensor absmax scan
    /// used to derive the activation `scale2` for the W4A4 path.
    nvfp4_absmax_k: KernelHandle,
    /// `quantize_nvfp4::quantize_bf16_to_nvfp4` — per-row E2M1 quantizer
    /// used to convert BF16 activations to the W4A4 GEMM input layout.
    nvfp4_quantize_k: KernelHandle,
    /// Lazily allocated scratch arena for the W4A4 fast path. Holds the
    /// packed activation nibbles + per-group FP8 scales + absmax scratch.
    /// Resized in-place if M or K grows. `Mutex` because `forward_prefill`
    /// takes `&self`.
    e2m1_scratch: Mutex<Option<E2m1Scratch>>,
}

impl DenseFfnLayer {
    pub fn new(weights: DenseFfnWeights, gpu: &dyn GpuBackend) -> Result<Self> {
        Self::new_with_activation(weights, FfnActivation::SiLU, gpu)
    }

    pub fn new_with_activation(
        weights: DenseFfnWeights,
        activation: FfnActivation,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let act_mul = match activation {
            FfnActivation::SiLU => gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
            FfnActivation::GeLU => gpu.kernel("gelu", "gelu_mul")?,
        };
        // BF16 path kernels — optional (only loaded if available; gemma4
        // is the only consumer today). `try_kernel` returns
        // `KernelHandle(0)` on miss so we don't break NVFP4-only models
        // that were built without these kernels. Module names per
        // `kernels/gb10/{target}/nvfp4/KERNEL.toml`:
        //   `dense_gemv_bf16 = "gemv"`, `dense_gemm_bf16 = "gemm"`.
        let dense_gemv_bf16_k = super::try_kernel(gpu, "gemv", "dense_gemv_bf16");
        let dense_gemm_bf16_k = super::try_kernel(gpu, "gemm", "dense_gemm_bf16");

        Ok(Self {
            weights,
            activation,
            w4a16_gemv: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_dual: gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_dual")?,
            w4a16_gemv_silu_input: gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_silu_input")?,
            w4a16_gemv_dual_batch2: gpu.kernel("w4a16_gemv", "w4a16_gemv_dual_batch2")?,
            w4a16_gemv_dual_batch3: gpu.kernel("w4a16_gemv", "w4a16_gemv_dual_batch3")?,
            w4a16_gemv_dual_batch3_tuned: super::try_kernel(
                gpu,
                "w4a16_gemv",
                "w4a16_gemv_dual_batch3_tuned",
            ),
            w4a16_gemv_batch2: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?,
            w4a16_gemv_batch3: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3")?,
            w4a16_gemm: gpu.kernel("w4a16", "w4a16_gemm")?,
            act_mul,
            bf16_weights: None,
            dense_gemv_bf16_k,
            dense_gemm_bf16_k,
            gate_proj_t: None,
            up_proj_t: None,
            down_proj_t: None,
            // Optional small-M (M_TILE=16) transposed-weight GEMM.
            // Missing on older kernel caches — handle 0 disables routing
            // through this kernel regardless of the transposed weights.
            w4a16_gemm_t_m16: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m16"),
            // Optional small-M (M_TILE=16), small-N (N_TILE=64) variant for
            // K=3 MTP verify on dense Qwen3.6-27B. Missing on non-qwen3.6-27b
            // kernel caches — handle 0 falls back to GEMV silently.
            w4a16_gemm_t_m16_n64: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m16_n64"),
            // Optional large-M (M_TILE=128, N_TILE=128) transposed-weight
            // GEMM for prefill. Handle 0 disables the
            // `ATLAS_PREFILL_FFN_FAST` fast path silently.
            w4a16_gemm_t_m128: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m128"),
            w4a16_gemm_t_m32_n64: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m32_n64"),
            // Optional 8-warp shadow of the M=128 kernel. Handle 0
            // disables `ATLAS_FFN_M128_V2` silently.
            w4a16_gemm_t_m128_v2: super::try_kernel(gpu, "w4a16_v2", "w4a16_gemm_t_m128_v2"),
            // Optional FP8×FP8 M=128 GEMM for the predequant fast path.
            // Handle 0 disables `ATLAS_FFN_PREDEQUANT_FP8` silently.
            fp8_gemm_t_m128_k: super::try_kernel(gpu, "w4a16", "fp8_gemm_t_m128"),
            gate_fp8: None,
            up_fp8: None,
            down_fp8: None,
            // Optional W4A4 (NVFP4×NVFP4) native tensor-core GEMM. Loaded
            // via `try_kernel` — handle 0 disables the `ATLAS_E2M1_GEMM`
            // fast path silently. The companion absmax + quantize kernels
            // are already loaded by every NVFP4 weight loader; we re-fetch
            // them here so the dispatch site can launch without plumbing
            // them through the forward context.
            nvfp4_gemm_k: super::try_kernel(gpu, "nvfp4_cutlass", "nvfp4_nvfp4_gemm_t_m64"),
            nvfp4_absmax_k: super::try_kernel(gpu, "quantize_nvfp4", "nvfp4_global_absmax"),
            nvfp4_quantize_k: super::try_kernel(gpu, "quantize_nvfp4", "quantize_bf16_to_nvfp4"),
            e2m1_scratch: Mutex::new(None),
        })
    }

    /// Whether the W4A4 (NVFP4×NVFP4) native tensor-core FFN prefill path
    /// is wired up. Returns true when all three required kernel symbols
    /// are present in the loaded module. Used by `forward_prefill` to
    /// choose between the W4A4 fast path and the existing fp8/v2/m128
    /// fallbacks.
    pub fn has_e2m1_ffn(&self) -> bool {
        self.nvfp4_gemm_k.0 != 0
            && self.nvfp4_absmax_k.0 != 0
            && self.nvfp4_quantize_k.0 != 0
    }

    /// Ensure the W4A4 activation scratch arena has capacity for `m` rows
    /// and `k` columns. Reallocates in-place if either dimension grew.
    ///
    /// Returns `(a_packed, a_scale, a_max)`.
    fn ensure_e2m1_scratch(
        &self,
        gpu: &dyn GpuBackend,
        m: usize,
        k: usize,
    ) -> Result<(DevicePtr, DevicePtr, DevicePtr)> {
        let mut slot = self.e2m1_scratch.lock().unwrap();
        let needs_realloc = match slot.as_ref() {
            Some(s) => s.cap_m < m || s.cap_k < k,
            None => true,
        };
        if needs_realloc {
            // Free previous (if any) so we don't leak when M or K grows.
            if let Some(prev) = slot.take() {
                let _ = gpu.free(prev.a_packed);
                let _ = gpu.free(prev.a_scale);
                let _ = gpu.free(prev.a_max);
            }
            // Round capacity up so small M growth doesn't trigger a
            // realloc every chunk.
            let cap_m = m.max(128);
            let cap_k = k;
            let a_packed = gpu.alloc(cap_m * cap_k / 2)?;
            let a_scale = gpu.alloc(cap_m * cap_k / 16)?;
            let a_max = gpu.alloc(4)?;
            *slot = Some(E2m1Scratch {
                a_packed,
                a_scale,
                a_max,
                cap_m,
                cap_k,
            });
        }
        let s = slot.as_ref().unwrap();
        Ok((s.a_packed, s.a_scale, s.a_max))
    }

    /// Run the W4A4 fast path for one FFN projection: prequant BF16 input
    /// to NVFP4, then dispatch `nvfp4_nvfp4_gemm`.
    ///
    /// `weight` is the standard `[N, K/2]` row-major NVFP4 weight (NOT
    /// the transposed `nvfp4_t` layout) — the kernel reads B in the
    /// HuggingFace layout, matching `self.weights.gate_proj` etc.
    #[allow(clippy::too_many_arguments)]
    fn forward_e2m1_proj(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        weight: &QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        let (a_packed, a_scale, a_max) =
            self.ensure_e2m1_scratch(ctx.gpu, m as usize, k as usize)?;

        // Phase 1: per-tensor absmax over the BF16 [M, K] activation.
        // Caller-side memset to zero matches `quantize_to_nvfp4` (see
        // weight_map/loaders_fp8.rs:87).
        ctx.gpu.memset_async(a_max, 0, 4, stream)?;
        ops::nvfp4_global_absmax(
            ctx.gpu,
            self.nvfp4_absmax_k,
            input,
            a_max,
            m * k,
            stream,
        )?;

        // Phase 2: read absmax back, derive scale2.
        // We have to synchronize: scale2 is a kernel ARGUMENT (FP32 by-
        // value), not a device pointer, so we can't defer the D2H copy.
        // This adds 1 sync per GEMM (3 per FFN, 192 per Qwen3.6-27B
        // forward). For a 4K prefill the sync cost is dominated by the
        // GEMM itself; profile will show whether this needs to become a
        // device-resident scale2 + a kernel signature change.
        ctx.gpu.synchronize(stream)?;
        let mut bytes = [0u8; 4];
        ctx.gpu.copy_d2h(a_max, &mut bytes)?;
        let global_max = f32::from_le_bytes(bytes);
        let a_scale2 = if global_max > 0.0 {
            global_max / (6.0 * 448.0)
        } else {
            1.0
        };

        // Phase 3: per-row E2M1 quantization of the activation. Writes
        // `a_packed` [M, K/2] + `a_scale` [M, K/16] in place.
        ops::quantize_bf16_to_nvfp4(
            ctx.gpu,
            self.nvfp4_quantize_k,
            input,
            a_packed,
            a_scale,
            a_scale2,
            m,
            k,
            stream,
        )?;

        // Phase 4: native W4A4 tensor-core GEMM.
        let scale2_ab = a_scale2 * weight.weight_scale_2;
        ops::nvfp4_nvfp4_gemm(
            ctx.gpu,
            self.nvfp4_gemm_k,
            a_packed,
            a_scale,
            weight.weight,
            weight.weight_scale,
            scale2_ab,
            output,
            m,
            n,
            k,
            stream,
        )?;
        Ok(())
    }

    /// Install transposed (`nvfp4_t` layout) FFN projection weights for
    /// the `forward_kgamma` M_TILE=16 fast path. Called by the loader
    /// after `DenseFfnLayer::new` when `ATLAS_FFN_M16_TRANSPOSED=1`.
    /// Takes ownership of `gate_proj_t` / `up_proj_t` / `down_proj_t`
    /// (additional allocations alongside the standard `weights.*_proj`;
    /// the originals are kept for decode-side GEMV paths that target the
    /// HuggingFace `[N, K/2]` layout).
    pub fn set_transposed_weights(
        &mut self,
        gate_proj_t: QuantizedWeight,
        up_proj_t: QuantizedWeight,
        down_proj_t: QuantizedWeight,
    ) {
        self.gate_proj_t = Some(gate_proj_t);
        self.up_proj_t = Some(up_proj_t);
        self.down_proj_t = Some(down_proj_t);
    }

    /// Whether the M_TILE=16 transposed-weight path is wired up.
    /// Used by `forward_kgamma` for dispatch and by the loader log.
    pub fn has_transposed_ffn(&self) -> bool {
        self.gate_proj_t.is_some()
            && self.up_proj_t.is_some()
            && self.down_proj_t.is_some()
            && self.w4a16_gemm_t_m16.0 != 0
    }

    /// Whether the predequanted-FP8 FFN prefill path is wired up.
    /// Returns true when all three projections have FP8 buffers AND the
    /// `fp8_gemm_t_m128` kernel symbol is present. Used by
    /// `forward_prefill` to choose between the NVFP4 + dequant path and
    /// the predequant fast path.
    pub fn has_fp8_ffn(&self) -> bool {
        self.gate_fp8.is_some()
            && self.up_fp8.is_some()
            && self.down_fp8.is_some()
            && self.fp8_gemm_t_m128_k.0 != 0
    }

    /// Pre-dequant NVFP4 FFN weights to FP8 [N, K] for the predequant
    /// fast prefill path. Mirrors
    /// `Qwen3AttentionLayer::predequant_for_prefill` — uses the
    /// NON-transposed weights in `self.weights.gate_proj` etc. (the
    /// `predequant_nvfp4_to_fp8` kernel reads the original [N, K/2]
    /// NVFP4 layout). Allocates 3 × N×K bytes of GPU memory per layer.
    ///
    /// Called by the loader after `DenseFfnLayer::new` when the
    /// `ATLAS_FFN_PREDEQUANT_FP8` env var is set. Silently no-ops
    /// when the kernel symbol is missing.
    ///
    /// `inter` = intermediate_size (gate/up output dim; down input dim).
    /// `hidden` = hidden_size (gate/up input dim; down output dim).
    pub fn predequant_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        hidden: usize,
        inter: usize,
        stream: u64,
    ) -> Result<()> {
        if self.fp8_gemm_t_m128_k.0 == 0 {
            return Ok(()); // FP8 kernel not available — silently skip
        }
        let predequant_k = gpu.kernel("w4a16", "predequant_nvfp4_to_fp8")?;
        // gate_proj: [inter, hidden] NVFP4 → [inter, hidden] FP8
        self.gate_fp8 = Some(self.weights.gate_proj.predequant_to_fp8(
            gpu,
            predequant_k,
            inter,
            hidden,
            stream,
        )?);
        // up_proj: same shape as gate
        self.up_fp8 = Some(self.weights.up_proj.predequant_to_fp8(
            gpu,
            predequant_k,
            inter,
            hidden,
            stream,
        )?);
        // down_proj: [hidden, inter]
        self.down_fp8 = Some(self.weights.down_proj.predequant_to_fp8(
            gpu,
            predequant_k,
            hidden,
            inter,
            stream,
        )?);
        Ok(())
    }

    /// Install BF16 dense MLP weights. After this call, the forward paths
    /// dispatch to the BF16 GEMV/GEMM kernels instead of w4a16. The
    /// caller must ensure the BF16 kernels are loaded (see
    /// `dense_gemv_bf16_k` / `dense_gemm_bf16_k` checks). Spec-decode
    /// batched paths (`forward_k2`, `forward_k3`) are NOT supported on
    /// the BF16 path — Gemma-4 dense has no MTP so they're never called.
    pub fn set_bf16_weights(&mut self, gate: DenseWeight, up: DenseWeight, down: DenseWeight) {
        self.bf16_weights = Some(DenseFfnWeightsBf16 {
            gate_proj: gate,
            up_proj: up,
            down_proj: down,
        });
    }

    /// Single-token decode: 2-3 kernel launches depending on activation.
    /// SiLU: dual GEMV + SiLU-fused down GEMV (2 launches).
    /// GELU: dual GEMV + gelu_mul + down GEMV (3 launches, no fused GELU down kernel).
    pub fn forward(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // BF16 dispatch: per-projection GEMV via `dense_gemv_bf16`. We
        // don't have a fused dual-BF16-GEMV kernel today; two sequential
        // launches are still BF16-precision-correct and only ~10% slower
        // than the fused w4a16 path on Gemma-4-31B (the cost is dominated
        // by the bigger BF16 weight reads, not launch overhead).
        if let Some(ref bf16w) = self.bf16_weights {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                input,
                &bf16w.gate_proj,
                gate_out,
                inter,
                h,
                stream,
            )?;
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                input,
                &bf16w.up_proj,
                up_out,
                inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                gate_out,
                &bf16w.down_proj,
                output,
                h,
                inter,
                stream,
            )?;
            return Ok(output);
        }

        // Fused gate_proj + up_proj: [1, H] → [1, inter] × 2
        crate::kprof!(ctx.gpu, stream, "ffn_gate_up_dual_m1", {
            ops::w4a16_gemv_dual(
                ctx.gpu,
                self.w4a16_gemv_dual,
                input,
                &self.weights.gate_proj,
                gate_out,
                &self.weights.up_proj,
                up_out,
                inter,
                h,
                stream,
            )?;
            anyhow::Result::<()>::Ok(())
        })?;

        let output = ctx.buffers.moe_output();
        match self.activation {
            FfnActivation::SiLU => {
                // Fused SiLU(gate)*up + down_proj: [1, inter] → [1, H]
                crate::kprof!(ctx.gpu, stream, "ffn_down_silu_m1", {
                    ops::w4a16_gemv_silu_input(
                        ctx.gpu,
                        self.w4a16_gemv_silu_input,
                        gate_out,
                        up_out,
                        &self.weights.down_proj,
                        output,
                        h,
                        inter,
                        stream,
                    )?;
                    anyhow::Result::<()>::Ok(())
                })?;
            }
            FfnActivation::GeLU => {
                // GELU(gate)*up → gate_out, then down_proj GEMV
                crate::kprof!(ctx.gpu, stream, "ffn_silu_mul_m1", {
                    ops::silu_mul(
                        ctx.gpu,
                        self.act_mul,
                        gate_out,
                        up_out,
                        gate_out,
                        inter,
                        stream,
                    )?;
                    anyhow::Result::<()>::Ok(())
                })?;
                crate::kprof!(ctx.gpu, stream, "ffn_down_m1", {
                    ops::w4a16_gemv(
                        ctx.gpu,
                        self.w4a16_gemv,
                        gate_out,
                        &self.weights.down_proj,
                        output,
                        h,
                        inter,
                        stream,
                    )?;
                    anyhow::Result::<()>::Ok(())
                })?;
            }
        }

        Ok(output)
    }

    /// K=2 speculative: batched GEMV for 2 tokens.
    /// 3 launches: dual batch2 (gate+up) + silu_mul + batch2 (down).
    pub fn forward_k2(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // Fused gate+up for 2 tokens
        ops::w4a16_gemv_dual_batch2(
            ctx.gpu,
            self.w4a16_gemv_dual_batch2,
            input,
            &self.weights.gate_proj,
            gate_out,
            &self.weights.up_proj,
            up_out,
            inter,
            h,
            stream,
        )?;
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            2 * inter,
            stream,
        )?;
        let output = ctx.buffers.moe_output();
        ops::w4a16_gemv_batch2(
            ctx.gpu,
            self.w4a16_gemv_batch2,
            gate_out,
            &self.weights.down_proj,
            output,
            h,
            inter,
            stream,
        )?;

        Ok(())
    }

    /// K=3 speculative: batched GEMV for 3 tokens.
    /// 3 launches: dual batch3 (gate+up) + silu_mul + batch3 (down).
    pub fn forward_k3(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        // Tensor-core M=3 fast path (ATLAS_TC_NVFP4_K3=1):
        //
        // Routes through `w4a16_gemm_t_m16_n64` (small-M, N_TILE=64) for
        // gate / up / down. Dispatches 3 GEMM launches (gate, up, down)
        // instead of the GEMV path's 3 launches (dual gate+up, silu, down)
        // but each GEMM runs on tensor cores via m16n8k32 e4m3 MMA.
        //
        // The first attempt (routing to `forward_kgamma(n=3)` which used
        // `w4a16_gemm_t_m16` with N_TILE=128) measured −23% mean tok/s
        // on AEON-27B because the 136-CTA grid starved GB10's 110 SMs.
        // The N_TILE=64 variant fields 272 CTAs/projection (~2.5 CTAs/SM)
        // at half the per-CTA work — designed to keep the tensor-core
        // pipeline fed at M=3.
        //
        // Bounds: requires transposed weights + the n64 kernel symbol
        // loaded; falls through to the GEMV path otherwise.
        if crate::layers::tc_nvfp4_k3_enabled()
            && self.has_transposed_ffn()
            && self.w4a16_gemm_t_m16_n64.0 != 0
        {
            let h = ctx.config.hidden_size as u32;
            let inter = ctx.config.intermediate_size as u32;
            let gate_out_buf = ctx.buffers.expert_gate_out();
            let up_out_buf = ctx.buffers.expert_up_out();
            let gt = self.gate_proj_t.as_ref().unwrap();
            let ut = self.up_proj_t.as_ref().unwrap();
            let dt = self.down_proj_t.as_ref().unwrap();

            crate::kprof!(ctx.gpu, stream, "ffn_gate_up_dual_batch3", {
                ops::w4a16_gemm_n64_m16(
                    ctx.gpu, self.w4a16_gemm_t_m16_n64,
                    input, gt, gate_out_buf, 3, inter, h, stream,
                )?;
                ops::w4a16_gemm_n64_m16(
                    ctx.gpu, self.w4a16_gemm_t_m16_n64,
                    input, ut, up_out_buf, 3, inter, h, stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            crate::kprof!(ctx.gpu, stream, "ffn_silu_mul", {
                ops::silu_mul(
                    ctx.gpu, self.act_mul,
                    gate_out_buf, up_out_buf, gate_out_buf,
                    3 * inter, stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            let output = ctx.buffers.moe_output();
            crate::kprof!(ctx.gpu, stream, "ffn_down_batch3", {
                ops::w4a16_gemm_n64_m16(
                    ctx.gpu, self.w4a16_gemm_t_m16_n64,
                    gate_out_buf, dt, output, 3, h, inter, stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            return Ok(());
        }

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // Fused gate+up for 3 tokens. Tuned variant (gated by
        // `ATLAS_FFN_DUAL_TUNED=1`) fuses both projections into the SAME CTA
        // so the 3-token activation vector is loaded once per CTA instead of
        // twice. Falls back to the baseline kernel when the env var is unset
        // OR the tuned kernel symbol was not present in the loaded cache.
        let use_tuned =
            ffn_dual_tuned_enabled() && self.w4a16_gemv_dual_batch3_tuned.0 != 0;
        let dual_kernel = if use_tuned {
            self.w4a16_gemv_dual_batch3_tuned
        } else {
            self.w4a16_gemv_dual_batch3
        };
        crate::kprof!(ctx.gpu, stream, "ffn_gate_up_dual_batch3", {
            if use_tuned {
                ops::w4a16_gemv_dual_batch3_tuned(
                    ctx.gpu,
                    dual_kernel,
                    input,
                    &self.weights.gate_proj,
                    gate_out,
                    &self.weights.up_proj,
                    up_out,
                    inter,
                    h,
                    stream,
                )?;
            } else {
                ops::w4a16_gemv_dual_batch3(
                    ctx.gpu,
                    dual_kernel,
                    input,
                    &self.weights.gate_proj,
                    gate_out,
                    &self.weights.up_proj,
                    up_out,
                    inter,
                    h,
                    stream,
                )?;
            }
            anyhow::Result::<()>::Ok(())
        })?;
        crate::kprof!(ctx.gpu, stream, "ffn_silu_mul", {
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                3 * inter,
                stream,
            )?;
            anyhow::Result::<()>::Ok(())
        })?;
        let output = ctx.buffers.moe_output();
        crate::kprof!(ctx.gpu, stream, "ffn_down_batch3", {
            ops::w4a16_gemv_batch3(
                ctx.gpu,
                self.w4a16_gemv_batch3,
                gate_out,
                &self.weights.down_proj,
                output,
                h,
                inter,
                stream,
            )?;
            anyhow::Result::<()>::Ok(())
        })?;

        Ok(())
    }

    /// K=γ verify batch (DFlash γ ≥ 16, typical n=17 with γ=16).
    ///
    /// Replaces the per-token loop that calls `forward()` n times (n=γ+1).
    /// Each `forward()` call runs 2 M=1 GEMVs that re-read 134 MB of NVFP4
    /// FFN weights from LPDDR5X; per-step this costs `64 layers × 17 tokens
    /// × 134 MB = 145 GB` of redundant weight bandwidth.
    ///
    /// This path issues 3 `w4a16_gemm` calls with M=n. The standard
    /// (non-transposed) NVFP4 GEMM has M_TILE=64; M=17 fits inside a single
    /// CTA-row with some accumulator waste but loads the weight tile once
    /// per layer instead of n times. Expected: ~64 × 134 MB = 8.6 GB per
    /// step, an ~18× reduction in FFN-loop bandwidth.
    ///
    /// Sequence: gate_proj (GEMM M=n) → up_proj (GEMM M=n) → silu_mul
    /// (n × intermediate) → down_proj (GEMM M=n).
    ///
    /// Reuses `ctx.buffers.expert_gate_out` / `expert_up_out` /
    /// `moe_output`, which are sized for `max_batch_tokens × intermediate`
    /// (always ≥ n, see `buffers/sizes.rs`).
    ///
    /// Output is written to `ctx.buffers.moe_output()`; callers downstream
    /// already consume from there (see `ms_phase_ffn` for the n==3 branch
    /// which is the contract this matches at higher n).
    ///
    /// When `ATLAS_FFN_M16_TRANSPOSED=1` and the loader installed transposed
    /// (`nvfp4_t`) FFN weights via `set_transposed_weights`, this path
    /// dispatches gate/up/down through `w4a16_gemm_n128_m16` (M_TILE=16,
    /// near-zero MMA accumulator waste at M ≤ 32). Otherwise the
    /// non-transposed `w4a16_gemm` (M_TILE=64) is used as the fallback —
    /// layout-compatible with the standard HuggingFace `[N, K/2]` weights
    /// but discards ~73% of accumulator writes at M=17. The transposed
    /// fast path is the K=γ verify analogue of the existing SSM `qkvz` /
    /// `out_proj` and DFlash drafter routings.
    /// BF16-weight fallback is not supported on this path (Gemma-4 dense
    /// has no MTP / DFlash, so n is always 1 there).
    pub fn forward_kgamma(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        n: u32,
        stream: u64,
    ) -> Result<()> {
        debug_assert!(n > 1, "forward_kgamma is for batched verify; use forward() at n=1");
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // Route through M_TILE=16 + transposed weights when:
        //   - ATLAS_FFN_M16_TRANSPOSED=1 OR ATLAS_TC_NVFP4_M16=1
        //   - the loader installed transposed copies of all 3 FFN projections
        //   - the `w4a16_gemm_t_m16` kernel symbol is present (try_kernel)
        //   - n ≤ 32 (the kernel's intended small-M window)
        // The combined gate matches the SSM `qkvz` / drafter pattern.
        let m16_path = n <= 32
            && (crate::layers::ffn_m16_transposed_enabled()
                || crate::layers::tc_nvfp4_m16_enabled())
            && self.has_transposed_ffn();
        // m128 upgrade of the m16 path: ONE M-tile at n ≤ 128 → single
        // weight read (m16 re-reads B per 16-row tile: 2× traffic at
        // n=17 on a memory-bound GEMM). See ffn_kgamma_m128_enabled.
        let m128_path = m16_path
            && crate::layers::ffn_kgamma_m128_enabled()
            && self.w4a16_gemm_t_m128.0 != 0;
        // m32_n64: single B read AND full SM occupancy — strictly better
        // than m128 at n ≤ 32 when the kernel symbol is present.
        // ATLAS_FFN_KGAMMA_M32=0 opts out for bisection.
        let m32_path = m128_path
            && self.w4a16_gemm_t_m32_n64.0 != 0
            && std::env::var("ATLAS_FFN_KGAMMA_M32").ok().as_deref() != Some("0");

        // gate_proj GEMM: [n, H] → [n, inter]
        crate::kprof!(ctx.gpu, stream, "ffn_gate_kgamma", {
            if m32_path {
                let gt = self.gate_proj_t.as_ref().unwrap();
                ops::w4a16_gemm_n64_m32(
                    ctx.gpu,
                    self.w4a16_gemm_t_m32_n64,
                    input,
                    gt,
                    gate_out,
                    n,
                    inter,
                    h,
                    stream,
                )?;
            } else if m128_path {
                let gt = self.gate_proj_t.as_ref().unwrap();
                ops::w4a16_gemm_n128_m128(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128,
                    input,
                    gt,
                    gate_out,
                    n,
                    inter,
                    h,
                    stream,
                )?;
            } else if m16_path {
                let gt = self.gate_proj_t.as_ref().unwrap();
                ops::w4a16_gemm_n128_m16(
                    ctx.gpu,
                    self.w4a16_gemm_t_m16,
                    input,
                    gt,
                    gate_out,
                    n,
                    inter,
                    h,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm(
                    ctx.gpu,
                    self.w4a16_gemm,
                    input,
                    &self.weights.gate_proj,
                    gate_out,
                    n,
                    inter,
                    h,
                    stream,
                )?;
            }
            anyhow::Result::<()>::Ok(())
        })?;

        // up_proj GEMM: [n, H] → [n, inter]
        crate::kprof!(ctx.gpu, stream, "ffn_up_kgamma", {
            if m32_path {
                let ut = self.up_proj_t.as_ref().unwrap();
                ops::w4a16_gemm_n64_m32(
                    ctx.gpu,
                    self.w4a16_gemm_t_m32_n64,
                    input,
                    ut,
                    up_out,
                    n,
                    inter,
                    h,
                    stream,
                )?;
            } else if m128_path {
                let ut = self.up_proj_t.as_ref().unwrap();
                ops::w4a16_gemm_n128_m128(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128,
                    input,
                    ut,
                    up_out,
                    n,
                    inter,
                    h,
                    stream,
                )?;
            } else if m16_path {
                let ut = self.up_proj_t.as_ref().unwrap();
                ops::w4a16_gemm_n128_m16(
                    ctx.gpu,
                    self.w4a16_gemm_t_m16,
                    input,
                    ut,
                    up_out,
                    n,
                    inter,
                    h,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm(
                    ctx.gpu,
                    self.w4a16_gemm,
                    input,
                    &self.weights.up_proj,
                    up_out,
                    n,
                    inter,
                    h,
                    stream,
                )?;
            }
            anyhow::Result::<()>::Ok(())
        })?;

        // activation(gate) * up for n tokens
        crate::kprof!(ctx.gpu, stream, "ffn_silu_mul_kgamma", {
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                n * inter,
                stream,
            )?;
            anyhow::Result::<()>::Ok(())
        })?;

        // down_proj GEMM: [n, inter] → [n, H]
        let output = ctx.buffers.moe_output();
        crate::kprof!(ctx.gpu, stream, "ffn_down_kgamma", {
            if m32_path {
                let dt = self.down_proj_t.as_ref().unwrap();
                ops::w4a16_gemm_n64_m32(
                    ctx.gpu,
                    self.w4a16_gemm_t_m32_n64,
                    gate_out,
                    dt,
                    output,
                    n,
                    h,
                    inter,
                    stream,
                )?;
            } else if m128_path {
                let dt = self.down_proj_t.as_ref().unwrap();
                ops::w4a16_gemm_n128_m128(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128,
                    gate_out,
                    dt,
                    output,
                    n,
                    h,
                    inter,
                    stream,
                )?;
            } else if m16_path {
                let dt = self.down_proj_t.as_ref().unwrap();
                ops::w4a16_gemm_n128_m16(
                    ctx.gpu,
                    self.w4a16_gemm_t_m16,
                    gate_out,
                    dt,
                    output,
                    n,
                    h,
                    inter,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm(
                    ctx.gpu,
                    self.w4a16_gemm,
                    gate_out,
                    &self.weights.down_proj,
                    output,
                    n,
                    h,
                    inter,
                    stream,
                )?;
            }
            anyhow::Result::<()>::Ok(())
        })?;

        Ok(())
    }

    /// N-token prefill: GEMM for all projections.
    pub fn forward_prefill(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        let m = num_tokens as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // BF16 prefill dispatch: dense_gemm_bf16 for all three projections.
        // (Gemma-4 dense path — bypasses NVFP4 entirely; not affected by
        // ATLAS_PREFILL_FFN_FAST.)
        if let Some(ref bf16w) = self.bf16_weights {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_bf16_k,
                input,
                &bf16w.gate_proj,
                gate_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_bf16_k,
                input,
                &bf16w.up_proj,
                up_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                m * inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_bf16_k,
                gate_out,
                &bf16w.down_proj,
                output,
                m,
                h,
                inter,
                stream,
            )?;
            return Ok(());
        }

        // Large-M W4A4 native NVFP4×NVFP4 fast path: route through the
        // CUTLASS-style `nvfp4_nvfp4_gemm_t_m64` kernel (native E2M1
        // tensor-core MMA) when:
        //   - ATLAS_E2M1_GEMM=1 was set at startup
        //   - all three required kernel symbols loaded (nvfp4_cutlass +
        //     quantize_nvfp4)
        //   - M >= 128 (kernel's intended window — matches the other
        //     prefill fast paths)
        // Each GEMM prequantizes BF16 activations to NVFP4 inline
        // (absmax + per-group E2M1) before dispatching the native MMA.
        // Theoretical 2× MFU lift over `w4a16_gemm_t_m128` (BF16×NVFP4)
        // since we eliminate the inner-loop dequant AND halve activation
        // DRAM traffic (0.5 B/elt vs 2 B/elt BF16). Takes precedence
        // over the fp8/v2/m128 paths when active. Falls back silently
        // when the gate is off or kernels are missing.
        let e2m1_fast_path = m >= 128
            && crate::layers::prefill_ffn_e2m1_enabled()
            && self.has_e2m1_ffn();

        // Per-shape E2M1 dispatch: gate/up stay on `w4a16_gemm_t_m128`,
        // down_proj routes through native E2M1 hardware MMA. Per the
        // shape table at the `prefill_ffn_e2m1_down_only_enabled` doc
        // site: gate (K=5120,N=17408) + up (same) are faster on the
        // BF16×NVFP4 w4a16 m128 path, while down (K=17408,N=5120) is
        // 1.31× faster via E2M1 MMA — net ~30% down savings with no
        // gate/up regression. Mutually exclusive with the all-three
        // `e2m1_fast_path`.
        let e2m1_down_only_path = !e2m1_fast_path
            && m >= 128
            && crate::layers::prefill_ffn_e2m1_down_only_enabled()
            && self.has_e2m1_ffn()
            && self.has_transposed_ffn()
            && self.w4a16_gemm_t_m128.0 != 0;

        // Large-M FP8 predequant fast path: route through the
        // `fp8_gemm_t_m128` kernel (BF16 A × pre-dequanted FP8 B) when:
        //   - ATLAS_FFN_PREDEQUANT_FP8=1 was set at startup
        //   - `predequant_for_prefill` ran successfully (gate/up/down
        //     FP8 buffers installed)
        //   - the `fp8_gemm_t_m128` kernel symbol is present
        //   - M >= 128
        // Saves the entire DEQUANT phase + one __syncthreads per
        // K-step inside w4a16_gemm_t_m128. Empirically a REGRESSION
        // on Qwen3.6-27B (5.68s → 8.11s TTFT at ISL=3603) because the
        // 2× B DRAM traffic (FP8 1 byte vs NVFP4 0.5 byte/elt) at
        // K=5120 + N=17408 exceeds the dequant savings — the FFN GEMM
        // is memory-bound, not compute-bound. Falls back to the w4a16
        // m128 path when not active. Mirrors the attention
        // `predequant_for_prefill` + `fp8_gemm_n128_m128` pattern.
        let fp8_fast_path = !e2m1_fast_path
            && !e2m1_down_only_path
            && m >= 128
            && crate::layers::prefill_ffn_fp8_enabled()
            && self.has_fp8_ffn();

        // Large-M v2 fast path: route through the 8-warp shadow
        // `w4a16_gemm_t_m128_v2` kernel when:
        //   - ATLAS_FFN_M128_V2=1
        //   - transposed (`nvfp4_t`) FFN weights installed
        //   - the v2 kernel symbol is present
        //   - M >= 128
        // Same SMEM footprint + 2-stage pipeline as v1 but doubles the
        // active warp count (256 threads/CTA, 4 warps per chunk run
        // chunk 0 and chunk 1 MMAs in parallel instead of serial). Net
        // result on compute-bound GEMMs: ~10-20% kernel-time win. On
        // memory-bound GEMMs: roughly neutral. Falls back to v1 when
        // not active. Original kernel:
        // kernels/gb10/minimax-m2-229b/nvfp4/w4a16_gemm_v2.cu —
        // copied verbatim into qwen3.6-27b/.
        let v2_fast_path = !e2m1_fast_path
            && !e2m1_down_only_path
            && !fp8_fast_path
            && m >= 128
            && crate::layers::prefill_ffn_m128_v2_enabled()
            && self.has_transposed_ffn()
            && self.w4a16_gemm_t_m128_v2.0 != 0;

        // Large-M fast path: route through the transposed-weight
        // M_TILE=128 kernel (`w4a16_gemm_t_m128`) when:
        //   - ATLAS_PREFILL_FFN_FAST=1
        //   - transposed (`nvfp4_t`) FFN weights installed at load
        //     (requires ATLAS_FFN_M16_TRANSPOSED=1)
        //   - the kernel symbol is present in the loaded module
        //   - M >= 128 (kernel's intended window — small-M would waste
        //     128-row CTAs the same way the M_TILE=64 path wastes 64-row
        //     CTAs at M < 64)
        // Mirrors the attention `w4a16_gemm_m128_dispatch` pattern in
        // `qwen3_attention/prefill_weights.rs:14`. Falls back to the
        // standard M_TILE=64 `w4a16_gemm` when any condition fails.
        let fast_path = !e2m1_fast_path
            && !e2m1_down_only_path
            && !fp8_fast_path
            && !v2_fast_path
            && m >= 128
            && crate::layers::prefill_ffn_fast_enabled()
            && self.has_transposed_ffn()
            && self.w4a16_gemm_t_m128.0 != 0;

        // One-shot info log on first prefill so the bench harness can
        // verify which path is firing. The atomic ensures we log once
        // across all 64 layers per process. Uses both tracing::info! and
        // eprintln! — the latter bypasses tracing's BufWriter when the
        // server is launched under `nohup ... > file 2>&1` (stderr is
        // line-buffered to a regular file, so a single `\n`-terminated
        // write hits disk immediately, even before process teardown).
        static LOGGED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let has_t = self.has_transposed_ffn();
            let has_fp8 = self.has_fp8_ffn();
            let has_e2m1 = self.has_e2m1_ffn();
            let m128_ok = self.w4a16_gemm_t_m128.0 != 0;
            let m128_v2_ok = self.w4a16_gemm_t_m128_v2.0 != 0;
            let fp8_m128_ok = self.fp8_gemm_t_m128_k.0 != 0;
            let e2m1_ok = self.nvfp4_gemm_k.0 != 0;
            let gate_on =
                std::env::var("ATLAS_PREFILL_FFN_FAST").ok().as_deref() == Some("1");
            let fp8_gate_on =
                std::env::var("ATLAS_FFN_PREDEQUANT_FP8").ok().as_deref() == Some("1");
            let v2_gate_on =
                std::env::var("ATLAS_FFN_M128_V2").ok().as_deref() == Some("1");
            let e2m1_gate_on =
                std::env::var("ATLAS_E2M1_GEMM").ok().as_deref() == Some("1");
            let e2m1_down_only_gate_on =
                std::env::var("ATLAS_E2M1_GEMM_DOWN_ONLY").ok().as_deref() == Some("1");
            tracing::info!(
                m,
                inter,
                h,
                e2m1_fast_path,
                e2m1_down_only_path,
                fp8_fast_path,
                v2_fast_path,
                fast_path,
                has_e2m1_ffn = has_e2m1,
                has_fp8_ffn = has_fp8,
                has_transposed = has_t,
                e2m1_kernel = e2m1_ok,
                fp8_m128_kernel = fp8_m128_ok,
                m128_v2_kernel = m128_v2_ok,
                m128_kernel = m128_ok,
                gate = gate_on,
                fp8_gate = fp8_gate_on,
                v2_gate = v2_gate_on,
                e2m1_gate = e2m1_gate_on,
                e2m1_down_only_gate = e2m1_down_only_gate_on,
                "dense_ffn forward_prefill dispatch (one-shot)"
            );
            eprintln!(
                "[atlas-prefill-ffn] dispatch: M={m} inter={inter} h={h} \
                 e2m1_fast_path={e2m1_fast_path} \
                 e2m1_down_only_path={e2m1_down_only_path} \
                 fp8_fast_path={fp8_fast_path} v2_fast_path={v2_fast_path} \
                 fast_path={fast_path} has_e2m1={has_e2m1} has_fp8={has_fp8} \
                 has_transposed={has_t} e2m1_kernel={e2m1_ok} \
                 fp8_m128_kernel={fp8_m128_ok} \
                 m128_v2_kernel={m128_v2_ok} m128_kernel={m128_ok} \
                 e2m1_gate=ATLAS_E2M1_GEMM={e2m1_gate_on} \
                 e2m1_down_only_gate=ATLAS_E2M1_GEMM_DOWN_ONLY={e2m1_down_only_gate_on} \
                 gate=ATLAS_PREFILL_FFN_FAST={gate_on} \
                 fp8_gate=ATLAS_FFN_PREDEQUANT_FP8={fp8_gate_on} \
                 v2_gate=ATLAS_FFN_M128_V2={v2_gate_on}"
            );
        }

        if e2m1_fast_path {
            // Native W4A4 NVFP4×NVFP4 dispatch: prequant activations to
            // NVFP4 in-place, then issue the CUTLASS-style E2M1×E2M1 MMA
            // GEMM. Uses the standard `[N, K/2]` HuggingFace weight
            // layout (NOT the `nvfp4_t` transposed layout) — matches the
            // kernel's coalesced gmem read pattern.
            //
            // gate_proj: [M, H] BF16 → [M, inter] BF16
            self.forward_e2m1_proj(
                ctx,
                input,
                &self.weights.gate_proj,
                gate_out,
                m,
                inter,
                h,
                stream,
            )?;
            // up_proj: [M, H] BF16 → [M, inter] BF16
            self.forward_e2m1_proj(
                ctx,
                input,
                &self.weights.up_proj,
                up_out,
                m,
                inter,
                h,
                stream,
            )?;
            // SiLU/GELU(gate) * up for all M tokens
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                m * inter,
                stream,
            )?;
            // down_proj: [M, inter] BF16 → [M, H] BF16
            let output = ctx.buffers.moe_output();
            self.forward_e2m1_proj(
                ctx,
                gate_out,
                &self.weights.down_proj,
                output,
                m,
                h,
                inter,
                stream,
            )?;
            return Ok(());
        }

        if e2m1_down_only_path {
            // gate/up stay on the w4a16 m128 fast path; only down_proj
            // routes through E2M1 hardware MMA. Matches the shape table
            // documented at `prefill_ffn_e2m1_down_only_enabled` — net
            // ~30% down_proj savings, no gate/up regression vs the
            // standard `fast_path`.
            let gt = self.gate_proj_t.as_ref().unwrap();
            let ut = self.up_proj_t.as_ref().unwrap();
            ops::w4a16_gemm_n128_m128(
                ctx.gpu,
                self.w4a16_gemm_t_m128,
                input,
                gt,
                gate_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::w4a16_gemm_n128_m128(
                ctx.gpu,
                self.w4a16_gemm_t_m128,
                input,
                ut,
                up_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                m * inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            self.forward_e2m1_proj(
                ctx,
                gate_out,
                &self.weights.down_proj,
                output,
                m,
                h,
                inter,
                stream,
            )?;
            return Ok(());
        }

        if v2_fast_path {
            let gt = self.gate_proj_t.as_ref().unwrap();
            let ut = self.up_proj_t.as_ref().unwrap();
            let dt = self.down_proj_t.as_ref().unwrap();
            // gate_proj GEMM via 8-warp v2 kernel
            ops::w4a16_gemm_n128_m128_v2(
                ctx.gpu,
                self.w4a16_gemm_t_m128_v2,
                input,
                gt,
                gate_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::w4a16_gemm_n128_m128_v2(
                ctx.gpu,
                self.w4a16_gemm_t_m128_v2,
                input,
                ut,
                up_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                m * inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ops::w4a16_gemm_n128_m128_v2(
                ctx.gpu,
                self.w4a16_gemm_t_m128_v2,
                gate_out,
                dt,
                output,
                m,
                h,
                inter,
                stream,
            )?;
            return Ok(());
        }

        if fp8_fast_path {
            // Pre-dequanted FP8 weights — single sync per K-step in
            // the kernel, no DEQUANT phase. Same A×B GEMM math, just
            // bypassing the inner-loop NVFP4 → FP8 dequant.
            let gfp8 = self.gate_fp8.unwrap();
            let ufp8 = self.up_fp8.unwrap();
            let dfp8 = self.down_fp8.unwrap();

            // gate_proj GEMM: [M, H] BF16 × [inter, H] FP8 → [M, inter] BF16
            ops::fp8_gemm_n128_m128(
                ctx.gpu,
                self.fp8_gemm_t_m128_k,
                input,
                gfp8,
                gate_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::fp8_gemm_n128_m128(
                ctx.gpu,
                self.fp8_gemm_t_m128_k,
                input,
                ufp8,
                up_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                m * inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ops::fp8_gemm_n128_m128(
                ctx.gpu,
                self.fp8_gemm_t_m128_k,
                gate_out,
                dfp8,
                output,
                m,
                h,
                inter,
                stream,
            )?;
            return Ok(());
        }

        if fast_path {
            let gt = self.gate_proj_t.as_ref().unwrap();
            let ut = self.up_proj_t.as_ref().unwrap();
            let dt = self.down_proj_t.as_ref().unwrap();

            // gate_proj GEMM: [M, H] → [M, inter]
            ops::w4a16_gemm_n128_m128(
                ctx.gpu,
                self.w4a16_gemm_t_m128,
                input,
                gt,
                gate_out,
                m,
                inter,
                h,
                stream,
            )?;
            // up_proj GEMM: [M, H] → [M, inter]
            ops::w4a16_gemm_n128_m128(
                ctx.gpu,
                self.w4a16_gemm_t_m128,
                input,
                ut,
                up_out,
                m,
                inter,
                h,
                stream,
            )?;
            // activation(gate) * up for all M tokens (SiLU or GELU)
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                m * inter,
                stream,
            )?;
            // down_proj GEMM: [M, inter] → [M, H]
            let output = ctx.buffers.moe_output();
            ops::w4a16_gemm_n128_m128(
                ctx.gpu,
                self.w4a16_gemm_t_m128,
                gate_out,
                dt,
                output,
                m,
                h,
                inter,
                stream,
            )?;
            return Ok(());
        }

        // Baseline M_TILE=64 path (unchanged).
        // gate_proj GEMM: [M, H] → [M, inter]
        ops::w4a16_gemm(
            ctx.gpu,
            self.w4a16_gemm,
            input,
            &self.weights.gate_proj,
            gate_out,
            m,
            inter,
            h,
            stream,
        )?;

        // up_proj GEMM: [M, H] → [M, inter]
        ops::w4a16_gemm(
            ctx.gpu,
            self.w4a16_gemm,
            input,
            &self.weights.up_proj,
            up_out,
            m,
            inter,
            h,
            stream,
        )?;

        // activation(gate) * up for all M tokens (SiLU or GELU)
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            m * inter,
            stream,
        )?;

        // down_proj GEMM: [M, inter] → [M, H]
        let output = ctx.buffers.moe_output();
        ops::w4a16_gemm(
            ctx.gpu,
            self.w4a16_gemm,
            gate_out,
            &self.weights.down_proj,
            output,
            m,
            h,
            inter,
            stream,
        )?;

        Ok(())
    }

    /// Batched forward (per-token loop). Used by forward_batched in model loop.
    pub fn forward_batched(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_prefill(input, num_tokens, ctx, stream)
    }
}
