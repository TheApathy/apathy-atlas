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
    /// `w4a16_gemm_t_m32_n64_gateup_silu` — FUSED gate_proj + up_proj +
    /// SiLU·mul for the K=γ verify path. Loads the shared [M,K] input
    /// tile once, streams BOTH transposed weights (gate + up), and writes
    /// only the fused silu(gate)*up [M,N] activation in one launch (vs the
    /// baseline's two m32_n64 GEMMs + standalone `moe_silu_mul`). Gated by
    /// `ATLAS_FFN_FUSED_GATEUP`. Loaded via `try_kernel`; handle 0 keeps
    /// the split gate/up path.
    w4a16_gemm_t_m32_n64_gateup_silu: KernelHandle,
    /// `w4a16_gemm_t_m32_n64_gateup_silu_pipe` — DEQUANT-IN-REGISTERS fork of
    /// the fused gate+up+SiLU kernel (`ATLAS_DEQUANT_PIPE=1`). Byte-exact
    /// with `w4a16_gemm_t_m32_n64_gateup_silu` (same shape, accumulation
    /// order, and BF16 round-trips) but the NVFP4→FP8 dequant runs in
    /// registers immediately before each MMA instead of via a `smem_B_fp8`
    /// staging array — dropping the 2nd per-K-step `__syncthreads`, shrinking
    /// SMEM ~27% (10.9 KB vs 15.0 KB → higher occupancy), and using
    /// `cp.async.wait_group<1>` so the next tile's load overlaps the current
    /// dequant+MMA. Loaded via `try_kernel`; handle 0 keeps the staged fused
    /// kernel as silent fallback.
    w4a16_gemm_t_m32_n64_gateup_silu_pipe: KernelHandle,
    /// `w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64` — K_STEP=64 fork of the
    /// `_pipe` register-dequant kernel (`ATLAS_GATEUP_K64=1`). Doubles the
    /// K-tile from 32 to 64 elements: 80 K-loop iterations vs 160, halving
    /// sync count and loop overhead. Each step issues two m16n8k32 MMAs per
    /// accumulator (K[0..31] then K[32..63]) and 2× the cp.async volume
    /// (6 KB vs 3 KB per stage), so the background load overlaps more of the
    /// inline dequant+compute work. Takes priority over ATLAS_DEQUANT_PIPE.
    /// Requires K divisible by 64 (hidden_size 5120 qualifies). Handle 0
    /// silently falls back through _pipe → staged fused kernel.
    w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64: KernelHandle,
    /// `w4a16_gemm_t_m32_n64_splitk` — split-K variant of the above for
    /// the DFlash K=17 verify `down_proj` ([M=17,N=5120,K=16384]). The
    /// single-slice kernel fields only 80 CTAs (N=5120/64) and is
    /// occupancy-starved on the long K=16384 loop (~91 GB/s vs gate/up's
    /// ~163). Split-K multiplies CTAs by k_splits into an FP32 workspace,
    /// then `reduce_splitk_f32_to_bf16` sums + downcasts. Gated by
    /// `ATLAS_FFN_DOWN_SPLITK` (default 4; 0/1 disables). Loaded via
    /// `try_kernel` — handle 0 keeps the single-slice m32_n64 path.
    w4a16_gemm_t_m32_n64_splitk: KernelHandle,
    /// `reduce_splitk_f32_to_bf16` — companion reduce kernel for the
    /// split-K down_proj. Sums the k_splits FP32 partial bands → BF16.
    reduce_splitk_k: KernelHandle,
    /// Lazily-allocated FP32 split-K workspace [k_splits, M, N].
    /// `Mutex` because `forward_kgamma` takes `&self`.
    splitk_workspace: Mutex<Option<DevicePtr>>,
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
    /// `ffn_sparsity_measure` — TEAL-style activation-sparsity observer for
    /// the sparsity-drafted self-speculation feasibility gate
    /// (`ATLAS_MEASURE_FFN_SPARSITY=1`). Loaded via `try_kernel`; handle 0
    /// disables the measurement silently (older kernel caches).
    ffn_sparsity_measure_k: KernelHandle,
    /// Lazily-allocated device counter buffers for the sparsity measurement.
    /// `Mutex` because `forward` takes `&self`. Allocated on first measured
    /// `forward` call (gpu.alloc is illegal during graph capture, but the
    /// self-spec draft + measurement run EAGER — see the gate docs). None
    /// until the first measured forward.
    sparsity_meas: Mutex<Option<SparsityMeas>>,
    /// W3 (3-bit weight) FFN projections — mixed-precision byte-reduction
    /// lane. `Some` only when `ATLAS_FFN_W3_LAYERS` names this layer AND
    /// the repacked sidecar (`ATLAS_FFN_W3_SIDECAR`, built by
    /// `local/tools/repack_w3.py`) contained its tensors; installed by the
    /// loader via `set_w3_weights`. GEMV layout `[N, 3K/8]` — used by the
    /// single-token `forward` SiLU path (dual gate/up + fused SiLU down).
    /// Cuts packed FFN weight bytes 25% vs NVFP4 on a weight-bandwidth-
    /// bound decode. NOT md5-gated (weights differ from W4 by
    /// construction) — quality is protected by the ABBA eval gate; the
    /// default path (gate off / no sidecar) stays byte-identical.
    w3_weights: Option<DenseFfnWeights>,
    /// Transposed W3 copies (`[3K/8, N_pad64]`) for the K=γ verify GEMM
    /// path (`w3a16_gemm_t_m32_n64`). Built host-side by the sidecar
    /// loader. `forward_kgamma` routes gate/up/down through the W3 GEMM
    /// when set (superseding the W4 m32/fused/split-K variants on this
    /// layer). Other paths (prefill, K=2/3 batched GEMV) intentionally
    /// stay on the retained W4 weights — they are not weight-bandwidth-
    /// bound the same way and keep their higher-precision copies.
    w3_weights_t: Option<DenseFfnWeights>,
    /// `w3a16_gemv_dual` — W3 gate+up dual GEMV. `try_kernel`; handle 0
    /// (older PTX caches) disables the W3 GEMV path silently.
    w3a16_gemv_dual_k: KernelHandle,
    /// `w3a16_gemv_silu_input` — W3 fused SiLU-input down GEMV.
    w3a16_gemv_silu_input_k: KernelHandle,
    /// `w3a16_gemm_t_m32_n64` — W3 clone of the m32_n64 verify GEMM.
    w3a16_gemm_t_m32_n64_k: KernelHandle,
    /// `ffn_build_keep_chunks` — on-device keep-chunk selector for the SPARSE
    /// self-spec DRAFT path (`ATLAS_SELF_SPEC_SPARSE=1`). Handle 0 disables.
    ffn_build_keep_chunks_k: KernelHandle,
    /// `w4a16_gemv_sparse_cols` — column-sparse GEMV for the SPARSE draft
    /// path. Handle 0 disables the sparse draft (falls back to dense GEMV).
    w4a16_gemv_sparse_cols_k: KernelHandle,
    /// Lazily-allocated per-layer keep_idx / keep_len device scratch for the
    /// sparse draft path. `Mutex` because `forward_draft_sparse` takes `&self`.
    sparse_draft_scratch: Mutex<Option<SparseDraftScratch>>,
}

/// Per-layer device counter buffers for the FFN activation-sparsity
/// measurement. Two sites (gate/up input + down input), each with a
/// `NUM_THRESH`-entry u32 histogram (below-threshold counts, atomically
/// accumulated) and a 2-entry u32 `count` ([0]=rows seen, [1]=elements seen).
struct SparsityMeas {
    /// Histogram for site 0 (gate/up input, K=hidden). `NUM_THRESH` u32s.
    hist_gateup: DevicePtr,
    /// [rows, elements] u32 counter for site 0.
    count_gateup: DevicePtr,
    /// Histogram for site 1 (down input, K=intermediate). `NUM_THRESH` u32s.
    hist_down: DevicePtr,
    /// [rows, elements] u32 counter for site 1.
    count_down: DevicePtr,
    /// Dedicated BF16 scratch [1, intermediate] into which the observer
    /// recomputes `silu(gate)*up` for the DOWN-input site. This is a SEPARATE
    /// buffer from the token-stream's `gate_out`/`up_out` — the fused down
    /// GEMV (`w4a16_gemv_silu_input`) applies SiLU internally and never
    /// materialises a standalone silu'd vector, so the observer computes its
    /// own copy here WITHOUT touching the buffers the fused kernel reads.
    /// Keeps the measurement a pure reader (token stream byte-identical).
    meas_silu: DevicePtr,
    /// Number of measured `forward` calls since process start on this layer.
    /// Drives the periodic D2H dump cadence.
    steps: u64,
}

/// Per-layer device scratch for the SPARSE self-spec draft: `keep_idx`
/// (surviving k8-chunk indices, capacity `K_max/8`) + `keep_len` (1 u32).
struct SparseDraftScratch {
    /// Surviving k8-chunk index list, sized for the largest K this layer
    /// might sparsify (down input K=intermediate). `keep_idx.len == K/8`.
    keep_idx: DevicePtr,
    /// Single u32: number of surviving chunks written by
    /// `ffn_build_keep_chunks`.
    keep_len: DevicePtr,
    /// Capacity in k8 chunks (= K_max/8) the `keep_idx` buffer can hold.
    cap_chunks: usize,
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
            // Optional fused gate+up+silu kernel. Handle 0 keeps the split
            // gate/up path (ATLAS_FFN_FUSED_GATEUP).
            w4a16_gemm_t_m32_n64_gateup_silu: super::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m32_n64_gateup_silu",
            ),
            // Optional dequant-in-registers fork of the fused kernel. Handle
            // 0 keeps the staged fused kernel (ATLAS_DEQUANT_PIPE).
            w4a16_gemm_t_m32_n64_gateup_silu_pipe: super::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m32_n64_gateup_silu_pipe",
            ),
            // Optional K_STEP=64 register-dequant fork (ATLAS_GATEUP_K64).
            // Handle 0 falls back through _pipe → staged fused kernel.
            w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64: super::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64",
            ),
            // Optional split-K down_proj variant + reduce. Handle 0 keeps
            // the single-slice m32_n64 path (ATLAS_FFN_DOWN_SPLITK).
            w4a16_gemm_t_m32_n64_splitk: super::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m32_n64_splitk",
            ),
            reduce_splitk_k: super::try_kernel(gpu, "w4a16", "reduce_splitk_f32_to_bf16"),
            splitk_workspace: Mutex::new(None),
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
            // Sparsity-drafted self-speculation kernels (default-off features).
            // The .cu files live in kernels/gb10/common/ and register under
            // their file-stem module names. try_kernel → handle 0 disables the
            // feature silently on caches built before these kernels existed.
            // W3 (3-bit) FFN lane — weights installed later by the loader
            // (set_w3_weights) iff ATLAS_FFN_W3_LAYERS + sidecar match.
            // Kernels live in kernels/gb10/common/w3a16_gemv.cu /
            // w3a16_gemm.cu; try_kernel → handle 0 on caches built before
            // they existed (W3 then stays fully disabled).
            w3_weights: None,
            w3_weights_t: None,
            w3a16_gemv_dual_k: super::try_kernel(gpu, "w3a16_gemv", "w3a16_gemv_dual"),
            w3a16_gemv_silu_input_k: super::try_kernel(gpu, "w3a16_gemv", "w3a16_gemv_silu_input"),
            w3a16_gemm_t_m32_n64_k: super::try_kernel(gpu, "w3a16_gemm", "w3a16_gemm_t_m32_n64"),
            ffn_sparsity_measure_k: super::try_kernel(
                gpu,
                "ffn_sparsity_measure",
                "ffn_sparsity_measure",
            ),
            sparsity_meas: Mutex::new(None),
            ffn_build_keep_chunks_k: super::try_kernel(
                gpu,
                "w4a16_gemv_sparse_cols",
                "ffn_build_keep_chunks",
            ),
            w4a16_gemv_sparse_cols_k: super::try_kernel(
                gpu,
                "w4a16_gemv_sparse_cols",
                "w4a16_gemv_sparse_cols",
            ),
            sparse_draft_scratch: Mutex::new(None),
        })
    }

    /// Whether the W4A4 (NVFP4×NVFP4) native tensor-core FFN prefill path
    /// is wired up. Returns true when all three required kernel symbols
    /// are present in the loaded module. Used by `forward_prefill` to
    /// choose between the W4A4 fast path and the existing fp8/v2/m128
    /// fallbacks.
    pub fn has_e2m1_ffn(&self) -> bool {
        self.nvfp4_gemm_k.0 != 0 && self.nvfp4_absmax_k.0 != 0 && self.nvfp4_quantize_k.0 != 0
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
        ops::nvfp4_global_absmax(ctx.gpu, self.nvfp4_absmax_k, input, a_max, m * k, stream)?;

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

    /// Install W3 (3-bit) FFN weights for this layer — GEMV-layout copies
    /// (used by the single-token `forward` SiLU path) and transposed
    /// GEMM-layout copies (used by `forward_kgamma`). Called by the loader
    /// when `ATLAS_FFN_W3_LAYERS` names this layer and the sidecar tensors
    /// loaded cleanly (see `weight_map::w3_sidecar`). The original W4
    /// weights are RETAINED for the paths W3 does not cover (prefill,
    /// K=2/3 batched GEMV, GELU models, sparse draft).
    pub fn set_w3_weights(&mut self, gemv: DenseFfnWeights, gemm_t: DenseFfnWeights) {
        self.w3_weights = Some(gemv);
        self.w3_weights_t = Some(gemm_t);
    }

    /// Whether the W3 single-token GEMV path is fully wired: weights
    /// installed + both kernel symbols present + SiLU activation (the W3
    /// GEMV set has no GELU-fused down kernel; GELU models stay on W4).
    fn has_w3_gemv(&self) -> bool {
        self.w3_weights.is_some()
            && self.w3a16_gemv_dual_k.0 != 0
            && self.w3a16_gemv_silu_input_k.0 != 0
            && self.activation == FfnActivation::SiLU
    }

    /// Whether the W3 K=γ verify GEMM path is fully wired.
    fn has_w3_gemm(&self) -> bool {
        self.w3_weights_t.is_some() && self.w3a16_gemm_t_m32_n64_k.0 != 0
    }

    /// Whether ANY W3 routing is active on this layer (loader log helper).
    pub fn has_w3(&self) -> bool {
        self.has_w3_gemv() || self.has_w3_gemm()
    }

    /// Eagerly allocate the FP32 split-K workspace for the down_proj
    /// (`[k_splits, 32, n]` where n = hidden, M padded to the M_TILE=32
    /// of the split-K kernel). Called at load time (pre-graph-capture)
    /// because `gpu.alloc` is illegal during CUDA graph capture. No-op
    /// when split-K is disabled or the kernel symbols are missing.
    pub fn alloc_splitk_workspace(&self, gpu: &dyn GpuBackend, n: u32) -> Result<()> {
        // `n` is the largest split-K output dim this layer might use. The
        // down_proj path needs N=hidden; the gate/up path (when
        // `ATLAS_FFN_GATEUP_SPLITK` is on) needs N=intermediate. Callers pass
        // `max(hidden, intermediate)` so ONE FP32 workspace serves both.
        // gate and up run back-to-back on the same stream and each fully
        // reduces into its own output before the next partial phase, so they
        // can safely share this scratch.
        let down_splits = crate::layers::ffn_down_splitk();
        let gateup_splits = crate::layers::ffn_gateup_splitk();
        if (down_splits == 0 && gateup_splits == 0)
            || self.w4a16_gemm_t_m32_n64_splitk.0 == 0
            || self.reduce_splitk_k.0 == 0
        {
            return Ok(());
        }
        let mut slot = self.splitk_workspace.lock().unwrap();
        if slot.is_none() {
            // 8 = max split clamp; 32 = M_TILE of the split-K kernel.
            let bytes = 8usize * 32 * n as usize * 4;
            *slot = Some(gpu.alloc(bytes)?);
        }
        Ok(())
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

    /// Whether the FFN activation-sparsity MEASUREMENT harness is wired up:
    /// the env gate is on AND the measure kernel symbol is present.
    fn sparsity_measure_active(&self) -> bool {
        crate::layers::measure_ffn_sparsity_enabled() && self.ffn_sparsity_measure_k.0 != 0
    }

    /// Ensure the per-layer sparsity-measurement counter buffers exist and are
    /// zeroed on first allocation. Returns the four device pointers. Allocated
    /// lazily on the first measured `forward` (never during graph capture —
    /// the measured path runs eager).
    fn ensure_sparsity_meas(
        &self,
        gpu: &dyn GpuBackend,
        inter: usize,
    ) -> Result<(DevicePtr, DevicePtr, DevicePtr, DevicePtr, DevicePtr)> {
        let n_thresh = ops::SPARSITY_NUM_THRESH;
        let mut slot = self.sparsity_meas.lock().unwrap();
        if slot.is_none() {
            let hist_gateup = gpu.alloc(n_thresh * 4)?;
            let count_gateup = gpu.alloc(2 * 4)?;
            let hist_down = gpu.alloc(n_thresh * 4)?;
            let count_down = gpu.alloc(2 * 4)?;
            let meas_silu = gpu.alloc(inter * 2)?; // BF16 [1, intermediate]
            // Zero the accumulators up front (kernel uses atomicAdd).
            gpu.memset(hist_gateup, 0, n_thresh * 4)?;
            gpu.memset(count_gateup, 0, 2 * 4)?;
            gpu.memset(hist_down, 0, n_thresh * 4)?;
            gpu.memset(count_down, 0, 2 * 4)?;
            *slot = Some(SparsityMeas {
                hist_gateup,
                count_gateup,
                hist_down,
                count_down,
                meas_silu,
                steps: 0,
            });
        }
        let s = slot.as_ref().unwrap();
        Ok((
            s.hist_gateup,
            s.count_gateup,
            s.hist_down,
            s.count_down,
            s.meas_silu,
        ))
    }

    /// Observer: launch `ffn_sparsity_measure` on `input` at the given site.
    /// PURE READER — never mutates `input` or any token-stream buffer; writes
    /// only into the dedicated `hist`/`count` accumulators. Called from
    /// `forward` at the two FFN sites when the measurement gate is on.
    fn measure_sparsity_site(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        hist: DevicePtr,
        count: DevicePtr,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        ops::ffn_sparsity_measure(
            ctx.gpu,
            self.ffn_sparsity_measure_k,
            input,
            hist,
            count,
            k,
            stream,
        )
    }

    /// Periodic D2H dump of the accumulated per-site sparsity histograms,
    /// averaged over all rows measured since process start. Emits a
    /// `tracing::info` line every `measure_ffn_sparsity_dump_every` measured
    /// forwards. Bumps the per-layer step counter each call. No-op when the
    /// dump cadence has not been reached.
    ///
    /// The reported fraction for threshold t at a site is
    /// `hist[t] / elements_seen` — the mean below-threshold activation
    /// fraction, i.e. the UPPER BOUND on the column-skip weight-byte savings
    /// for that projection at that threshold. The go/no-go number is the
    /// down-input (K=intermediate) fraction at the 1% threshold.
    fn maybe_dump_sparsity(&self, ctx: &ForwardContext, layer_tag: &str) -> Result<()> {
        let every = crate::layers::measure_ffn_sparsity_dump_every();
        let (hist_gateup, count_gateup, hist_down, count_down, steps) = {
            let mut slot = self.sparsity_meas.lock().unwrap();
            let Some(s) = slot.as_mut() else {
                return Ok(());
            };
            s.steps += 1;
            if !s.steps.is_multiple_of(every) {
                return Ok(());
            }
            (
                s.hist_gateup,
                s.count_gateup,
                s.hist_down,
                s.count_down,
                s.steps,
            )
        };

        // Sync so the accumulators reflect all launched measurements, then
        // read the histograms + counts back to the host.
        ctx.gpu.synchronize(ctx.gpu.default_stream())?;
        let n_thresh = ops::SPARSITY_NUM_THRESH;
        let read_hist = |hist: DevicePtr| -> Result<Vec<u32>> {
            let mut bytes = vec![0u8; n_thresh * 4];
            ctx.gpu.copy_d2h(hist, &mut bytes)?;
            Ok(bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };
        let read_count = |count: DevicePtr| -> Result<(u64, u64)> {
            let mut bytes = vec![0u8; 2 * 4];
            ctx.gpu.copy_d2h(count, &mut bytes)?;
            let rows = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64;
            let elems = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as u64;
            Ok((rows, elems))
        };

        let fmt_site = |hist: &[u32], elems: u64| -> String {
            if elems == 0 {
                return "n/a".to_string();
            }
            ops::SPARSITY_TAU
                .iter()
                .zip(hist.iter())
                .map(|(tau, &cnt)| {
                    let frac = cnt as f64 / elems as f64;
                    format!("{:.1}%tau={:.1}%", tau * 100.0, frac * 100.0)
                })
                .collect::<Vec<_>>()
                .join(" ")
        };

        let hg = read_hist(hist_gateup)?;
        let hd = read_hist(hist_down)?;
        let (rows_g, elems_g) = read_count(count_gateup)?;
        let (rows_d, elems_d) = read_count(count_down)?;

        tracing::info!(
            "FFN_SPARSITY[{layer_tag}] steps={steps} \
             gateup_in(K=hidden rows={rows_g}): {} | \
             down_in(K=inter rows={rows_d}): {}",
            fmt_site(&hg, elems_g),
            fmt_site(&hd, elems_d),
        );
        Ok(())
    }

    /// Whether the column-sparse self-spec DRAFT FFN path is wired up: both
    /// kernel symbols present. The env gate (`ATLAS_SELF_SPEC_SPARSE`) is
    /// checked by the caller (`step_self_spec`) so this only reports capability.
    pub fn has_sparse_draft(&self) -> bool {
        self.ffn_build_keep_chunks_k.0 != 0 && self.w4a16_gemv_sparse_cols_k.0 != 0
    }

    /// Ensure the per-layer sparse-draft scratch (`keep_idx` + `keep_len`)
    /// exists with capacity for `k` (the largest K this layer will sparsify —
    /// the down input K=intermediate). Allocated lazily on the first sparse
    /// draft forward (eager path, no graph capture).
    fn ensure_sparse_draft_scratch(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
    ) -> Result<(DevicePtr, DevicePtr)> {
        let need_chunks = k / 8;
        let mut slot = self.sparse_draft_scratch.lock().unwrap();
        let realloc = match slot.as_ref() {
            Some(s) => s.cap_chunks < need_chunks,
            None => true,
        };
        if realloc {
            if let Some(prev) = slot.take() {
                let _ = gpu.free(prev.keep_idx);
                let _ = gpu.free(prev.keep_len);
            }
            let keep_idx = gpu.alloc(need_chunks * 4)?; // u32 per chunk
            let keep_len = gpu.alloc(4)?; // single u32
            *slot = Some(SparseDraftScratch {
                keep_idx,
                keep_len,
                cap_chunks: need_chunks,
            });
        }
        let s = slot.as_ref().unwrap();
        Ok((s.keep_idx, s.keep_len))
    }

    /// SPARSE self-spec DRAFT single-token FFN forward.
    ///
    /// Same gate/up GEMV shape as `forward` (gate/up input is the dense
    /// residual stream — low activation sparsity, per the TEAL analysis, so
    /// it stays a dense dual GEMV), then swaps the down_proj GEMV for the
    /// column-sparse path: `ffn_build_keep_chunks` thresholds the silu(gate)*up
    /// activation into a surviving-chunk list, then `w4a16_gemv_sparse_cols`
    /// reads only those weight columns. APPROXIMATE by design — the dense
    /// verify is the lossless oracle, so this only proposes.
    ///
    /// `thresh_frac` is the keep threshold as a fraction of per-row max-abs
    /// (e.g. 0.01 for 1%). Falls back to the exact `forward` dense path when
    /// the sparse kernels are missing (`has_sparse_draft` false), so callers
    /// can always invoke it safely.
    ///
    /// EAGER only (the self-spec draft never captures a CUDA graph): the
    /// `keep_len` scalar is read back D2H before the sparse GEMV launch.
    pub fn forward_draft_sparse(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        thresh_frac: f32,
        stream: u64,
    ) -> Result<DevicePtr> {
        // Capability + BF16-weight guard: the sparse kernels operate on the
        // NVFP4 `QuantizedWeight` layout only. Fall back to the exact dense
        // forward when sparse kernels are missing or BF16 weights are active.
        if !self.has_sparse_draft() || self.bf16_weights.is_some() {
            return self.forward(input, ctx, stream);
        }

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // gate/up stay DENSE (residual-stream input, low sparsity).
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

        // silu(gate)*up → gate_out (the down-proj input we sparsify).
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            inter,
            stream,
        )?;

        // Threshold the down-input into a surviving k8-chunk list.
        let (keep_idx, keep_len) = self.ensure_sparse_draft_scratch(ctx.gpu, inter as usize)?;
        ops::ffn_build_keep_chunks(
            ctx.gpu,
            self.ffn_build_keep_chunks_k,
            gate_out,
            thresh_frac,
            keep_idx,
            keep_len,
            inter,
            stream,
        )?;

        // Read back keep_len (scalar-by-value kernel arg). Sync is acceptable
        // on the eager draft path; it also bounds the sparse GEMV's loop.
        ctx.gpu.synchronize(stream)?;
        let mut kl_bytes = [0u8; 4];
        ctx.gpu.copy_d2h(keep_len, &mut kl_bytes)?;
        let keep_len_val = u32::from_le_bytes(kl_bytes);

        // Column-sparse down_proj GEMV over the surviving chunks only.
        let output = ctx.buffers.moe_output();
        ops::w4a16_gemv_sparse_cols(
            ctx.gpu,
            self.w4a16_gemv_sparse_cols_k,
            gate_out,
            &self.weights.down_proj,
            keep_idx,
            keep_len_val,
            output,
            h,
            inter,
            stream,
        )?;
        Ok(output)
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

        // ── FFN activation-sparsity MEASUREMENT (observer, default-off) ──
        // Runs ONLY when ATLAS_MEASURE_FFN_SPARSITY=1 and the kernel symbol is
        // present. Pure reader: measures `input` (gate/up in, K=hidden) and a
        // freshly-recomputed `silu(gate)*up` copy (down in, K=inter) into
        // dedicated counter buffers. It never touches `input`, `gate_out`,
        // `up_out`, or `output`, so the token stream stays byte-identical
        // whether the gate is on or off (counting-md5 constitution preserved).
        //
        // Skipped under CUDA graph capture: the observer lazily `gpu.alloc`s its
        // counter buffers on first use, which is illegal mid-capture. The
        // measurement run is intended for eager decode (the operator sets the
        // gate for a measurement window); skipping graphed steps only omits a
        // subset of rows from the average and never perturbs the token stream.
        if self.sparsity_measure_active() && !ctx.graph_capture {
            let (hist_g, count_g, hist_d, count_d, meas_silu) =
                self.ensure_sparsity_meas(ctx.gpu, inter as usize)?;
            // Site 0: gate/up input (residual stream, K=hidden).
            self.measure_sparsity_site(ctx, input, hist_g, count_g, h, stream)?;
            // Site 1: down input = silu(gate)*up (K=intermediate). Recompute
            // into the observer's OWN scratch so gate_out/up_out (read by the
            // fused down GEMV below) are untouched.
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                meas_silu,
                inter,
                stream,
            )?;
            self.measure_sparsity_site(ctx, meas_silu, hist_d, count_d, inter, stream)?;
            self.maybe_dump_sparsity(ctx, "dense_ffn")?;
        }

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
                    ctx.gpu,
                    self.w4a16_gemm_t_m16_n64,
                    input,
                    gt,
                    gate_out_buf,
                    3,
                    inter,
                    h,
                    stream,
                )?;
                ops::w4a16_gemm_n64_m16(
                    ctx.gpu,
                    self.w4a16_gemm_t_m16_n64,
                    input,
                    ut,
                    up_out_buf,
                    3,
                    inter,
                    h,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            crate::kprof!(ctx.gpu, stream, "ffn_silu_mul", {
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out_buf,
                    up_out_buf,
                    gate_out_buf,
                    3 * inter,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            let output = ctx.buffers.moe_output();
            crate::kprof!(ctx.gpu, stream, "ffn_down_batch3", {
                ops::w4a16_gemm_n64_m16(
                    ctx.gpu,
                    self.w4a16_gemm_t_m16_n64,
                    gate_out_buf,
                    dt,
                    output,
                    3,
                    h,
                    inter,
                    stream,
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
        let use_tuned = ffn_dual_tuned_enabled() && self.w4a16_gemv_dual_batch3_tuned.0 != 0;
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
        debug_assert!(
            n > 1,
            "forward_kgamma is for batched verify; use forward() at n=1"
        );
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
        // WIDE window (SASS audit 2026-07-08): at c>=2 batched verify,
        // M = 17c exceeds 32 and this whole transposed family used to
        // disengage — silent fallback to legacy `w4a16_gemm` (no cp.async,
        // scalar LDG.U8, ~4x sector overfetch, issue-capped ~47% of DRAM
        // BW): the measured cause of the concurrency ceiling. Route
        // 32 < n <= 256 through `w4a16_gemm_t_m128` (y-tiled; at M=136 two
        // weight reads still beat legacy's 3 sweeps x overfetch). Requires
        // the m128 kernel — the m16 kernel's small-M window is NOT widened.
        let wide_m128 = n > 32
            && n <= 256
            && crate::layers::ffn_kgamma_wide_enabled()
            && crate::layers::ffn_kgamma_m128_enabled()
            && self.w4a16_gemm_t_m128.0 != 0
            && (crate::layers::ffn_m16_transposed_enabled()
                || crate::layers::tc_nvfp4_m16_enabled())
            && self.has_transposed_ffn();
        let m16_path = (n <= 32
            && (crate::layers::ffn_m16_transposed_enabled()
                || crate::layers::tc_nvfp4_m16_enabled())
            && self.has_transposed_ffn())
            || wide_m128;
        // m128 upgrade of the m16 path: ONE M-tile at n ≤ 128 → single
        // weight read (m16 re-reads B per 16-row tile: 2× traffic at
        // n=17 on a memory-bound GEMM). See ffn_kgamma_m128_enabled.
        let m128_path =
            m16_path && crate::layers::ffn_kgamma_m128_enabled() && self.w4a16_gemm_t_m128.0 != 0;
        // m32_n64: single B read AND full SM occupancy — strictly better
        // than m128 at n ≤ 32 when the kernel symbol is present. The n<=32
        // bound keeps the WIDE window off m32/fused/split-K variants (their
        // M_TILE=32 shapes don't cover wide M).
        // ATLAS_FFN_KGAMMA_M32=0 opts out for bisection.
        let m32_path = m128_path
            && n <= 32
            && self.w4a16_gemm_t_m32_n64.0 != 0
            && std::env::var("ATLAS_FFN_KGAMMA_M32").ok().as_deref() != Some("0");

        // gate_proj GEMM: [n, H] → [n, inter]
        // Split-K [M=n, N=inter, K=h] when ATLAS_FFN_GATEUP_SPLITK is set —
        // slices K across gridDim.z into the shared FP32 workspace (lossless,
        // token-exact). Falls through to the single-slice m32_n64 path below.
        let gateup_splitk = crate::layers::ffn_gateup_splitk();
        let gateup_ws = if gateup_splitk > 0 {
            *self.splitk_workspace.lock().unwrap()
        } else {
            None
        };
        let gateup_splitk_ok = m32_path
            && gateup_splitk > 0
            && self.w4a16_gemm_t_m32_n64_splitk.0 != 0
            && self.reduce_splitk_k.0 != 0
            && gateup_ws.is_some();

        // FUSED gate+up+silu (ATLAS_FFN_FUSED_GATEUP=1): one launch reads
        // the shared [n,H] input once, streams both transposed weights, and
        // writes silu(gate)*up into `gate_out` (the same buffer moe_silu_mul
        // targets) — replacing the gate GEMM + up GEMM + silu_mul below.
        // Supersedes gateup split-K. Requires the m32 transposed path + the
        // fused kernel symbol; byte-exact (BF16 activation round-trip matched
        // in-kernel). Falls through to the split path otherwise.
        let fused_gateup = m32_path
            && !gateup_splitk_ok
            && crate::layers::ffn_fused_gateup_enabled()
            && self.w4a16_gemm_t_m32_n64_gateup_silu.0 != 0;
        if fused_gateup {
            let gt = self.gate_proj_t.as_ref().unwrap();
            let ut = self.up_proj_t.as_ref().unwrap();
            // Kernel selection priority (highest first):
            //   ATLAS_GATEUP_K64=1  → _pipe_k64 (K_STEP=64, reg-dequant)
            //   ATLAS_DEQUANT_PIPE=1 → _pipe    (K_STEP=32, reg-dequant)
            //   default              → staged fused (smem_B_fp8 staging)
            let fused_kernel = if crate::layers::gateup_k64_enabled()
                && self.w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64.0 != 0
                && h % 64 == 0  // K must be divisible by 64
            {
                self.w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64
            } else if crate::layers::dequant_pipe_enabled()
                && self.w4a16_gemm_t_m32_n64_gateup_silu_pipe.0 != 0
            {
                self.w4a16_gemm_t_m32_n64_gateup_silu_pipe
            } else {
                self.w4a16_gemm_t_m32_n64_gateup_silu
            };
            crate::kprof!(ctx.gpu, stream, "ffn_gateup_fused_kgamma", {
                ops::w4a16_gemm_n64_m32_gateup_silu(
                    ctx.gpu,
                    fused_kernel,
                    input,
                    gt,
                    ut,
                    gate_out,
                    n,
                    inter,
                    h,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
        } else {
            crate::kprof!(ctx.gpu, stream, "ffn_gate_kgamma", {
                if gateup_splitk_ok {
                    let gt = self.gate_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n64_m32_splitk(
                        ctx.gpu,
                        self.w4a16_gemm_t_m32_n64_splitk,
                        self.reduce_splitk_k,
                        input,
                        gt,
                        gate_out,
                        gateup_ws.unwrap(),
                        n,
                        inter,
                        h,
                        inter, // ldb == N for tightly-packed T-weight
                        gateup_splitk,
                        stream,
                    )?;
                } else if m32_path {
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

            // up_proj GEMM: [n, H] → [n, inter]. Same split-K treatment as gate;
            // reuses the shared workspace (gate's reduce already consumed it).
            crate::kprof!(ctx.gpu, stream, "ffn_up_kgamma", {
                if gateup_splitk_ok {
                    let ut = self.up_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n64_m32_splitk(
                        ctx.gpu,
                        self.w4a16_gemm_t_m32_n64_splitk,
                        self.reduce_splitk_k,
                        input,
                        ut,
                        up_out,
                        gateup_ws.unwrap(),
                        n,
                        inter,
                        h,
                        inter, // ldb == N for tightly-packed T-weight
                        gateup_splitk,
                        stream,
                    )?;
                } else if m32_path {
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
        } // end !fused_gateup

        // FFN activation-sparsity MEASUREMENT (ATLAS_MEASURE_FFN_SPARSITY).
        // This is the K=γ verify FFN — the path DFlash decode ACTUALLY uses
        // (forward() M=1 is not hit per decode step). Row 0 of the [n, *]
        // buffers is a representative decode activation. `input` is the
        // gate/up input (K=hidden); `gate_out` now holds silu(gate)*up, the
        // down_proj input (K=intermediate) — no recompute needed here (unlike
        // forward(), which consumes gate_out in its fused path). PURE READER:
        // measures row 0 into dedicated accumulators, never mutates the token
        // stream. Skipped under graph capture (lazy alloc illegal mid-capture)
        // → run with ATLAS_DFLASH_DEBUG_NO_GRAPH=1 to observe eager decode.
        if self.sparsity_measure_active() && !ctx.graph_capture {
            let (hist_g, count_g, hist_d, count_d, _meas) =
                self.ensure_sparsity_meas(ctx.gpu, inter as usize)?;
            self.measure_sparsity_site(ctx, input, hist_g, count_g, h, stream)?;
            self.measure_sparsity_site(ctx, gate_out, hist_d, count_d, inter, stream)?;
            self.maybe_dump_sparsity(ctx, "dense_ffn_kgamma")?;
        }

        // down_proj GEMM: [n, inter] → [n, H]
        let output = ctx.buffers.moe_output();
        crate::kprof!(ctx.gpu, stream, "ffn_down_kgamma", {
            // Split-K down_proj: [M=n, N=h, K=inter]. The single-slice
            // m32_n64 kernel is occupancy-starved here (N=h=5120 → 80 CTAs
            // vs gate/up's 256 at N=inter=16384) and grinds a long K-loop.
            // Split-K multiplies CTAs by `splits` into an FP32 workspace,
            // then reduces → BF16. Gated by ATLAS_FFN_DOWN_SPLITK; falls
            // through to the single-slice path when disabled or unallocated.
            let splitk = crate::layers::ffn_down_splitk();
            let ws = if splitk > 0 {
                *self.splitk_workspace.lock().unwrap()
            } else {
                None
            };
            if let (true, Some(ws)) = (
                m32_path
                    && splitk > 0
                    && self.w4a16_gemm_t_m32_n64_splitk.0 != 0
                    && self.reduce_splitk_k.0 != 0,
                ws,
            ) {
                let dt = self.down_proj_t.as_ref().unwrap();
                ops::w4a16_gemm_n64_m32_splitk(
                    ctx.gpu,
                    self.w4a16_gemm_t_m32_n64_splitk,
                    self.reduce_splitk_k,
                    gate_out,
                    dt,
                    output,
                    ws,
                    n,
                    h,
                    inter,
                    h, // ldb == N for tightly-packed T-weight
                    splitk,
                    stream,
                )?;
            } else if m32_path {
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
        let e2m1_fast_path =
            m >= 128 && crate::layers::prefill_ffn_e2m1_enabled() && self.has_e2m1_ffn();

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
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let has_t = self.has_transposed_ffn();
            let has_fp8 = self.has_fp8_ffn();
            let has_e2m1 = self.has_e2m1_ffn();
            let m128_ok = self.w4a16_gemm_t_m128.0 != 0;
            let m128_v2_ok = self.w4a16_gemm_t_m128_v2.0 != 0;
            let fp8_m128_ok = self.fp8_gemm_t_m128_k.0 != 0;
            let e2m1_ok = self.nvfp4_gemm_k.0 != 0;
            let gate_on = std::env::var("ATLAS_PREFILL_FFN_FAST").ok().as_deref() == Some("1");
            let fp8_gate_on =
                std::env::var("ATLAS_FFN_PREDEQUANT_FP8").ok().as_deref() == Some("1");
            let v2_gate_on = std::env::var("ATLAS_FFN_M128_V2").ok().as_deref() == Some("1");
            let e2m1_gate_on = std::env::var("ATLAS_E2M1_GEMM").ok().as_deref() == Some("1");
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
