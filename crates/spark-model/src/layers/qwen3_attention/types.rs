// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3 attention struct definitions: `MlaWeights` (latent attention
//! 2-step decode) and `Qwen3AttentionLayer` (full attention layer).

use spark_runtime::gpu::{DevicePtr, KernelHandle};
use spark_runtime::kv_cache::KvCacheDtype;

use crate::layers::FfnComponent;
use crate::layers::fp8_calibration::Fp8KvCalibration;
use crate::weight_map::{
    AttentionWeights, DenseWeight, Fp8DenseWeight, QuantWeight, QuantizedWeight,
};

pub use super::types_weights::{HcWeights, MlaWeights};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HeadGateActivation {
    Sigmoid,
    Softplus,
}

/// Qwen3-Next full attention layer (12 of 48 layers).
#[allow(dead_code)]
pub struct Qwen3AttentionLayer {
    pub(super) input_norm: DenseWeight,
    pub(crate) attn: AttentionWeights,
    pub(super) post_attn_norm: DenseWeight,
    pub(super) ffn: FfnComponent,
    pub(super) attn_layer_idx: usize,
    /// Startup-static LoRA adapter overlay for the K/V/O projections (v0;
    /// q_proj excluded — gated Q+gate interleave). Installed
    /// post-construction via `set_lora_weights`; `None` = base-only.
    /// M0: stored only — the compute-path reads land in M1.
    pub(super) lora: Option<crate::layers::ops::lora_delta::LoraAttnWeights>,
    /// Whether Q projection includes an output gate (Q+Gate interleaved).
    /// When true, q_proj output is 2× q_dim; attn output is gated by sigmoid.
    /// When false (e.g. Qwen3-VL), q_proj output is q_dim; no gating applied.
    pub(super) gated: bool,
    /// Whether this layer should apply MRoPE-interleaved instead of scalar
    /// RoPE. Set when `config.mrope_interleaved = true` (Qwen3.6).
    pub(crate) mrope_interleaved: bool,
    /// Per-layer dimension overrides for heterogeneous models (Gemma-4).
    pub(crate) head_dim_override: Option<usize>,
    pub(crate) num_q_heads_override: Option<usize>,
    pub(crate) num_kv_heads_override: Option<usize>,
    /// Per-layer sliding-window size for Gemma-4 hybrid attention.
    pub(crate) sliding_window: Option<u32>,
    /// Per-layer RoPE overrides for heterogeneous models (Gemma-4).
    pub(crate) rope_theta_override: Option<f32>,
    pub(crate) rotary_dim_override: Option<u32>,
    /// Proportional RoPE (Gemma-4 full-attention).
    pub(crate) rope_proportional: bool,
    /// Per-layer attention scale override (Gemma-4: 1.0 because QK-norm
    /// handles scaling). When None, uses the standard 1/sqrt(head_dim).
    pub(crate) attn_scale_override: Option<f32>,
    /// K=V mode: V comes from raw K projection output (no separate v_proj).
    pub(crate) k_eq_v: bool,
    /// Ones-filled BF16 weight buffer for the pure-RMSNorm v_norm path.
    pub(crate) v_norm_weight: Option<DenseWeight>,
    /// Per-head attention gate weight (Step 3.7 g_proj).
    /// Shape: [num_q_heads, hidden_size] BF16. Applied as:
    /// attn_out = attn_out * sigmoid(g_proj @ hidden_states)
    /// with broadcast over head_dim.
    pub(crate) head_gate_weight: Option<DenseWeight>,
    pub(crate) head_gate_activation: HeadGateActivation,
    /// Kernel handle for per-head sigmoid gate broadcast multiply.
    pub(super) sigmoid_gate_head_broadcast_k: KernelHandle,
    pub(super) softplus_gate_head_broadcast_k: KernelHandle,
    /// Optional YaRN frequencies for standard (non-MLA) attention.
    pub(crate) yarn_inv_freq: DevicePtr,
    pub(crate) yarn_attention_factor: f32,
    /// Post-attention output norm (Gemma-4).  
    pub(crate) post_attn_out_norm: Option<DenseWeight>,
    /// Post-FFN output norm (Gemma-4).
    pub(crate) post_ffn_out_norm: Option<DenseWeight>,
    /// Per-layer scalar (Gemma-4): hidden_states *= layer_scalar at end of forward.
    pub(crate) layer_scalar: Option<f32>,
    /// Secondary FFN (Gemma-4 26B MoE): runs in parallel with primary FFN (dense).
    pub(crate) moe_ffn: Option<FfnComponent>,
    /// Pre-norm for MoE input (pre_feedforward_layernorm_2).
    pub(crate) pre_moe_norm: Option<DenseWeight>,
    /// Post-norm for MoE output (post_feedforward_layernorm_2).
    pub(crate) post_moe_out_norm: Option<DenseWeight>,
    /// Post-norm for dense FFN output only (post_feedforward_layernorm_1).
    pub(crate) post_dense_ffn_norm: Option<DenseWeight>,
    pub(super) kv_dtype: KvCacheDtype,
    /// Turbo4 sparse-V pruning threshold (0.0 = disabled).
    pub(super) sparse_v_threshold: f32,
    // ── Decode weights (QuantWeight enum: Nvfp4 | Fp8 | Dense) ──
    pub(super) q_weight: Option<QuantWeight>,
    pub(super) k_weight: Option<QuantWeight>,
    pub(super) v_weight: Option<QuantWeight>,
    pub(super) o_weight: Option<QuantWeight>,
    /// BF16 dense fallback for the output projection. When `Some`, the
    /// decode/prefill o_proj GEMV uses this BF16 weight instead of the
    /// NVFP4 path (`attn.o_proj`). Used by Gemma-4 dense which honors
    /// Nvidia ModelOpt's official ignore list.
    pub(super) o_dense_bf16: Option<DenseWeight>,
    // ── FP8-E4M3 row-scaled MIRROR copies of the BF16 attention projections
    // (ATLAS_TARGET_ATTN_FP8_MIRROR=1; Laguna ships attention unquantized).
    // Built once at load time from the BF16 source weights and consumed ONLY
    // by the decode/verify GEMV/GEMM dispatch sites — prefill stays BF16
    // (cuBLASLt). `None` (the default) keeps every path byte-identical to
    // the BF16 baseline. Halves attention weight-read bandwidth on the hot
    // decode/verify path (qkv 20.3ms + oproj 22.6ms of a 112ms verify step
    // is pure BF16 weight bandwidth).
    pub(super) q_fp8_mirror: Option<Fp8DenseWeight>,
    pub(super) k_fp8_mirror: Option<Fp8DenseWeight>,
    pub(super) v_fp8_mirror: Option<Fp8DenseWeight>,
    pub(super) o_fp8_mirror: Option<Fp8DenseWeight>,
    /// M=1 GEMV against an FP8 mirror (`gemv_fp8w::dense_gemv_fp8w`);
    /// 0-handle when the kernel module is absent (mirrors never built then).
    pub(super) dense_gemv_fp8w_k: KernelHandle,
    /// Batched row-scaled FP8 GEMM (`w4a16::fp8_gemm_t_row_scaled`) for the
    /// M=n verify projections; 0-handle on miss.
    pub(super) fp8_gemm_row_scaled_k: KernelHandle,
    /// Single-warp M_TILE=16 sibling (`w4a16::fp8_gemm_t_row_scaled_m16`)
    /// used when M ≤ 16; 0-handle on miss (falls back to the M64 tile).
    pub(super) fp8_gemm_row_scaled_m16_k: KernelHandle,
    /// Weight-read-bound M ≤ 8 sibling (`w4a16::fp8_gemm_t_row_scaled_mtile8`,
    /// N_TILE=64, 4-stage cp.async ring) for the verify projections;
    /// 0-handle on miss (falls back to the _m16/M64 tiles).
    pub(super) fp8_gemm_row_scaled_mtile8_k: KernelHandle,
    /// N_TILE=32 sibling (`w4a16::fp8_gemm_t_row_scaled_mtile8_n32`) for the
    /// small-N/large-K mirror shapes (o_proj: N=3072, K=6144/9216) where the
    /// N_TILE=64 grid is only 48 CTAs = 1 CTA/SM; 0-handle on miss (falls
    /// back to the N_TILE=64 mtile8 kernel). Bit-identical accumulation.
    pub(super) fp8_gemm_row_scaled_mtile8_n32_k: KernelHandle,
    // ── MLA (Multi-head Latent Attention) — 2-step decode ──
    pub(crate) mla: Option<MlaWeights>,
    // ── Manifold-Constrained Hyper-Connections (mHC) — DeepSeek-V4 ──
    /// Per-block HC parameters. `Some` only for DeepSeek-V4 (`hc_mult > 0`),
    /// in which case the attn/ffn residual sites use `hc_pre`/`hc_post`
    /// against the `hc_streams` buffer instead of the standard residual add.
    pub(crate) hc: Option<HcWeights>,
    /// HC `hc_pre` kernel handle (NULL when HC disabled).
    pub(super) hc_pre_k: KernelHandle,
    /// HC `hc_pre_mix` / `hc_pre_finish` handles — the decode-only multi-block
    /// split of `hc_pre` (NULL when the kernel module predates the split).
    pub(super) hc_pre_mix_k: KernelHandle,
    pub(super) hc_pre_finish_k: KernelHandle,
    /// Prefill-width tiled mix (`hc_pre_mix_tiled`): both operands read
    /// ~once (fn per 32-token tile, x once). 2.39 vs 3.99 ms at T=2410,
    /// y cosine 1.0000000 vs hc_pre. 0 on miss.
    pub(super) hc_pre_mix_tiled_k: KernelHandle,
    /// Single-launch multi-block decode hc_pre (ATLAS_V4_DECODE_FUSED=1):
    /// row-parallel mix + last-block finish, T==1 only. Zero when absent.
    pub(super) hc_pre_fused_k: KernelHandle,
    /// HC `hc_post` kernel handle (NULL when HC disabled).
    pub(super) hc_post_k: KernelHandle,
    /// HC `hc_expand` kernel handle (NULL when HC disabled).
    pub(super) hc_expand_k: KernelHandle,
    /// HC `hc_head` kernel handle (NULL when HC disabled).
    pub(super) hc_head_k: KernelHandle,
    // ── Transposed weights for prefill GEMM ──
    pub(super) q_nvfp4_t: Option<QuantizedWeight>,
    pub(super) k_nvfp4_t: Option<QuantizedWeight>,
    pub(super) v_nvfp4_t: Option<QuantizedWeight>,
    pub(super) o_nvfp4_t: Option<QuantizedWeight>,
    pub(super) q_fp8w_t: Option<crate::weight_map::Fp8WeightTransposed>,
    pub(super) k_fp8w_t: Option<crate::weight_map::Fp8WeightTransposed>,
    pub(super) v_fp8w_t: Option<crate::weight_map::Fp8WeightTransposed>,
    pub(super) o_fp8w_t: Option<crate::weight_map::Fp8WeightTransposed>,
    pub(super) w8a16_gemm_t_k: KernelHandle,
    pub(super) w8a16_gemm_t_pipelined_k: KernelHandle,
    // Fast transposed FP8 prefill GEMM (128x128 / 8-warp / two-level FP32 fold).
    // Consumes the SAME B_t[K,N] + block_scale_t[K/128,N/128] that
    // transpose_fp8 / transpose_block_scale already produce. KernelHandle(0) on
    // miss → fall back to w8a16_gemm_t.
    pub(super) w8a16_gemm_t_m128_k: KernelHandle,
    // W8A8 + FP32 epilogue (vLLM-equivalent) — gated by ATLAS_FP8_W8A8=1.
    pub(super) per_token_group_quant_fp8_k: KernelHandle,
    pub(super) fp8_gemm_t_blockscaled_k: KernelHandle,
    // Kernels — decode (GEMV M=1)
    /// Offset-from-1 `rms_norm` (`out = x * (1 + w) / rms`). Used ONLY for the
    /// unweighted normalize (`norm_unit_w()` is zero-filled, so `1 + 0 = 1`).
    pub(super) rms_norm_k: KernelHandle,
    /// The norm kernel for every weight that comes from the CHECKPOINT.
    /// Same handle as `rms_norm_k` for offset-from-1 models; `rms_norm_vanilla`
    /// (`out = x * w / rms`) for models that ship HF-vanilla norm weights.
    pub(super) rms_norm_w_k: KernelHandle,
    /// Warp-per-row sibling of `rms_norm_w_k` for short per-head rows; 0 if absent.
    pub(super) rms_norm_w_warp_row_k: KernelHandle,
    /// True when `rms_norm_w_k` is the vanilla kernel — i.e. the checkpoint's
    /// norm weights are loaded exactly, with no `-1` pre-subtraction.
    pub(super) norm_vanilla: bool,
    pub(super) rms_norm_residual_k: KernelHandle,
    /// Gemma-4 FP32-input rms_norm (absolute formula).
    pub(super) rms_norm_f32_in_k: KernelHandle,
    pub(super) dense_gemv_k: KernelHandle,
    /// Small-M sibling of `dense_gemv_k` (`dense_gemv_bf16_batchm`); 0 if the
    /// target has no such kernel. Used for the DFlash verify head-gate
    /// projection, where M = gamma+1 is far too small to fill the prefill
    /// tensor-core GEMM's 16x64 tile.
    pub(super) dense_gemv_batchm_k: KernelHandle,
    pub(super) w4a16_gemv_k: KernelHandle,
    /// One-launch block-diagonal wo_a (`w4a16_gemv_grouped`): replaces the
    /// 8-per-layer per-group launches. Bit-identical per row; measured
    /// 153 -> 194 GB/s at the wo_a shape (grouped microtest, 2026-08-09).
    /// 0 if the target has no such kernel — dispatch falls back per group.
    pub(super) w4a16_gemv_grouped_k: KernelHandle,
    /// Batched (M<=8) sibling of `w4a16_gemv_grouped` whose PER-ROW math is
    /// byte-identical to single-row `w4a16_gemv` — the ATLAS_OPROJ_EXACT
    /// semantics at batch speed (3.07x the per-row cost in the grouped
    /// microtest). Serves BOTH verify o-projection phases: wo_a with
    /// rows_per_group=o_lora, wo_b with rows_per_group=N (single group).
    /// 0 when absent — verify falls back to the `_ld` kernels (K-order
    /// drift documented at the OPROJ_EXACT comment in multi_seq/mla.rs).
    pub(super) w4a16_gemv_grouped_batchm_k: KernelHandle,
    /// V2 data-movement rework of `w4a16_gemv_grouped_batchm`
    /// (ATLAS_VERIFY_GEMV_V2=1, default OFF): compile-time M entries
    /// [m4, m5, m6, m8], SASS-verified bit-identical per row (same FFMA
    /// sequence; only load widths / addressing / guards changed). Requires
    /// K % 32 == 0 — dispatch falls back to the incumbent otherwise.
    pub(super) w4a16_gemv_grouped_batchm_v2_k: [KernelHandle; 4],
    /// FP8 sibling of the exact batched GEMV (`w8a16_gemv_batchm_exact`,
    /// M<=8, strided): per-row byte-identical to single-row `w8a16_gemv`.
    /// With the w4 exact kernel this makes EVERY verify GEMV projection
    /// single-row-order under ATLAS_VERIFY_EXACT_GEMV=1. 0 on miss.
    pub(super) w8a16_gemv_batchm_exact_k: KernelHandle,
    /// V2 of `w8a16_gemv_batchm_exact` (ATLAS_VERIFY_GEMV_V2=1): same
    /// [m4, m5, m6, m8] compile-time-M scheme, same bit-identity proof.
    pub(super) w8a16_gemv_batchm_exact_v2_k: [KernelHandle; 4],
    pub(super) w8a16_gemv_k: KernelHandle,
    pub(super) w8a16_gemv_batch4_k: KernelHandle,
    pub(super) w8a16_gemv_batch4_ld_k: KernelHandle,
    pub(super) w4a16_gemv_batch4_ld_k: KernelHandle,
    /// M<=8 siblings of the `batch4` pair, for the DSpark block verify (γ=6).
    /// Without them `ms_mla_decode_v4_flash` has no batched path past n=4 and
    /// re-reads every projection once per verify row.
    pub(super) w8a16_gemv_batch8_k: KernelHandle,
    pub(super) w8a16_gemv_batch8_ld_k: KernelHandle,
    pub(super) w4a16_gemv_batch8_ld_k: KernelHandle,
    pub(super) w8a16_gemm_k: KernelHandle,
    pub(super) w8a16_gemm_pipelined_k: KernelHandle,
    /// FP8-native W8A8 prefill GEMM (`mma.m16n8k32.e4m3`) and its per-row
    /// activation quantizer. Both 0 unless `w8a8_gemm_pipelined.cu` loaded;
    /// the V4 projection dispatch falls back to `w8a16_gemm_pipelined`.
    pub(super) w8a8_gemm_pipelined_k: KernelHandle,
    pub(super) quantize_a_fp8_rows_k: KernelHandle,
    /// Strided-A/C siblings (`..._ld`) — let the block-diagonal wo_a run its
    /// groups in place, deleting 8 gather + 8 scatter copies per layer.
    pub(super) w8a16_gemm_pipelined_ld_k: KernelHandle,
    pub(super) dense_gemm_pipelined_ld_k: KernelHandle,
    pub(super) w4a16_gemv_dual_k: KernelHandle,
    pub(super) rope_k: KernelHandle,
    /// MRoPE-interleaved kernel.
    pub(super) rope_mrope_interleaved_k: KernelHandle,
    /// K-only MRoPE kernel used when Q RoPE is fused into Q deinterleave/norm.
    pub(super) rope_mrope_interleaved_k_only_k: KernelHandle,
    /// YaRN RoPE kernel using pre-computed inv_freq table (Mistral, etc.)
    pub(super) rope_yarn_k: KernelHandle,
    pub(super) rope_yarn_scaled_k: KernelHandle,
    /// Interleaved (GPT-J / is_neox_style=False) YaRN RoPE kernel — DeepSeek MLA.
    pub(super) rope_yarn_interleaved_k: KernelHandle,
    /// Conjugate (negated-sin) interleaved YaRN RoPE — DeepSeek-V4 attention
    /// output de-rotation (eq.26).
    pub(super) rope_yarn_interleaved_inv_k: KernelHandle,
    /// Proportional RoPE kernel (Gemma-4 full-attention layers).
    pub(super) rope_proportional_k: KernelHandle,
    pub(super) reshape_cache_k: KernelHandle,
    /// Fused k_norm + RoPE + paged BF16 cache write — eliminates two
    /// intermediate BF16 rounding steps that cause the documented L35-L39
    /// cliff in chunked-prefill BF16 KV mode (memory:
    /// `project_qwen36_phase2b_softmax_expf.md`).
    pub(super) fused_k_norm_rope_cache_write_bf16_k: KernelHandle,
    /// MRoPE-interleaved variant of the above. Same precision regime.
    /// Dispatched when `mrope_interleaved` is true.
    pub(super) fused_k_norm_rope_mrope_cache_write_bf16_k: KernelHandle,
    /// V-only paged cache write. Used alongside the fused K-path so the
    /// K side of the cache stays single-rounded.
    pub(super) reshape_and_cache_flash_v_only_k: KernelHandle,
    /// Fused multi-seq verify epilogue (ATLAS_FUSED_ELEMWISE=1): per-head q/k
    /// rms_norm + yarn-scaled RoPE + paged BF16 K/V cache write in ONE launch,
    /// bit-identical to the unfused per-row chain. 0 when the module is
    /// absent — dispatch sites guard on `.0 != 0`.
    pub(super) fused_qkv_norm_rope_cache_k: KernelHandle,
    /// WHT kernel for turbo KV cache.
    pub(super) wht_bf16_k: KernelHandle,
    /// Inverse WHT. With TQ_PLUS_SIGNS off this aliases the forward kernel
    /// (plain WHT is self-inverse); with TQ+ signs the inverse reverses the
    /// signs1/signs2 order, which is required because (S2·H·S1)·(S2·H·S1) ≠ I.
    pub(super) wht_bf16_k_inv: KernelHandle,
    /// InnerQ application kernels (Q pre-WHT scale_inv, K post-WHT scale).
    /// Returns 0 handle when InnerQ kernel module isn't loaded — caller should
    /// guard launches with `.0 != 0`.
    pub(super) innerq_apply_q_k: KernelHandle,
    pub(super) innerq_apply_k_k: KernelHandle,
    pub(super) paged_decode_k: KernelHandle,
    /// HDIM=512 paged decode kernel for Gemma-4 full-attention layers
    pub(super) paged_decode_512_k: KernelHandle,
    /// MLA absorbed paged decode kernel (HDIM=320).
    pub(super) paged_decode_mla_k: KernelHandle,
    /// MLA paged decode kernel for DeepSeek-V4-Flash (compressed KV cache: 576 dims)
    pub(super) mla_paged_decode_k: KernelHandle,
    /// MLA paged decode kernel for DeepSeek-V4-Flash with FP8 KV cache
    pub(super) mla_paged_decode_fp8_k: KernelHandle,
    /// Same kernel with the V load elided (V4 MLA writes V == K byte-for-byte, so
    /// the V pool is pure redundant DRAM traffic). Selected only when the host has
    /// verified `k_scale == v_scale`; see `mla_paged_decode_fp8.cu`'s KV_ALIAS note.
    pub(super) mla_paged_decode_fp8_kvalias_k: KernelHandle,
    /// MLA batched GEMV for Q absorption and V extraction.
    pub(super) mla_batched_gemv_k: KernelHandle,
    /// MLA fused kernels — decode.
    pub(super) mla_q_rope_scatter_k: KernelHandle,
    pub(super) mla_q_rope_writeback_k: KernelHandle,
    pub(super) mla_cache_assemble_k: KernelHandle,
    /// MLA fused kernels — prefill.
    pub(super) mla_q_rope_extract_batched_k: KernelHandle,
    pub(super) mla_q_rope_writeback_batched_k: KernelHandle,
    pub(super) mla_kv_assemble_batched_k: KernelHandle,
    pub(super) mla_cache_assemble_batched_k: KernelHandle,
    /// V4 M=1 decode glue fusion (ATLAS_V4_DECODE_FUSED=1): in-place fused
    /// Q+K interleaved-YaRN rope (replaces extract×2 + rope + writeback×2)
    /// and fused MLA cache assemble + FP8 paged write (replaces
    /// cache_assemble + reshape_and_cache_flash_fp8). Zero handle when the
    /// model's `mla_absorbed` module doesn't ship them (non-V4 targets).
    pub(super) v4_decode_rope_fused_k: KernelHandle,
    pub(super) v4_decode_cache_fused_fp8_k: KernelHandle,
    /// MLA absorbed prefill flash attention (HDIM=320, GQA 32:1)
    pub(super) prefill_attn_mla320_k: KernelHandle,
    /// Grouped GEMM for MLA Q absorption + V extraction.
    pub(super) grouped_gemm_mla_k: KernelHandle,
    /// Q_final assembly: [absorbed|rope] per head.
    pub(super) mla_q_final_assemble_k: KernelHandle,
    /// Fused MLA prefill: Q_absorb + attention + V_extract in one kernel.
    pub(super) mla_fused_prefill_k: KernelHandle,
    /// Split-K GEMM for skinny prefill matrices (M < 64).
    pub(super) gemm_splitk_partial_k: KernelHandle,
    pub(super) gemm_splitk_reduce_k: KernelHandle,
    /// Tensor-core BF16 GEMM (m16n8k16 MMA).
    pub(super) dense_gemm_tc_k: KernelHandle,
    pub(super) paged_decode_splitk_k: Option<KernelHandle>,
    pub(super) paged_decode_reduce_k: Option<KernelHandle>,
    pub(super) residual_add_k: KernelHandle,
    pub(super) sigmoid_gate_mul_k: KernelHandle,
    pub(super) deinterleave_qg_k: KernelHandle,
    pub(super) w4a16_gemv_qg_k: KernelHandle,
    pub(super) residual_add_rms_norm_k: KernelHandle,
    /// Dual-output (bf16 + f32) MoE-input norm for ATLAS_FP32_ROUTING. Zero if absent.
    pub(super) residual_add_rms_norm_gatef32_k: KernelHandle,
    // Kernels — batch2 (K=2 verify)
    pub(super) w4a16_gemv_qg_batch2_k: KernelHandle,
    pub(super) w4a16_gemv_dual_batch2_k: KernelHandle,
    pub(super) w4a16_gemv_batch2_k: KernelHandle,
    // Kernels — batch3 (K=3 verify)
    pub(super) w4a16_gemv_qg_batch3_k: KernelHandle,
    pub(super) w4a16_gemv_dual_batch3_k: KernelHandle,
    pub(super) w4a16_gemv_batch3_k: KernelHandle,
    /// M<=4 batched GEMV (K=4 verify q/k/v/o); 0-handle when absent.
    pub(super) w4a16_gemv_batch4_k: KernelHandle,
    /// M<=8 sibling for the DSpark block verify; 0-handle when absent.
    pub(super) w4a16_gemv_batch8_k: KernelHandle,
    // Kernels — prefill (GEMM M=N + Flash Attention)
    pub(super) w4a16_gemm_k: KernelHandle,
    pub(super) w4a16_gemm_t_k: KernelHandle,
    pub(super) w4a16_gemm_t_k64_k: KernelHandle,
    pub(super) w4a16_gemm_t_m128_k: KernelHandle,
    /// LOSSLESS BF16-TC variant of t_m128 for QKV/o projection prefill (FP4→BF16
    /// dequant + BF16 MMA, no FP8 activation crush). Opt-in via ATLAS_BF16_TC_PROJ
    /// (default off → t_m128 path unchanged). KernelHandle(0) on miss.
    pub(super) w4a16_gemm_t_m128_bf16_k: KernelHandle,
    /// MiniMax-only shadow kernel.
    pub(super) w4a16_gemm_t_m128_v2_k: KernelHandle,
    /// v3 variant: K_STEP=64.
    pub(super) w4a16_gemm_t_m128_v3_k: KernelHandle,
    pub(super) dense_gemm_k: KernelHandle,
    /// Tensor-core pipelined BF16 GEMM (mma.sync + cp.async, 128×128 tile) —
    /// ~40× the scalar `dense_gemm_k` on large-M prefill projections, same math
    /// (cosine 1.0). Used for the BF16-fallback Q/K/V/O projections (Holo's
    /// native-FP8-dequant-to-BF16 attention path).
    pub(super) dense_gemm_pipelined_k: KernelHandle,
    pub(super) prefill_attn_k: KernelHandle,
    /// HDIM=512 contiguous prefill for Gemma-4 full-attention layers
    pub(super) prefill_attn_512_k: KernelHandle,
    /// DeepSeek-V4 CSA compressor: window softmax-gated KV compression.
    pub(super) csa_compress_k: KernelHandle,
    /// DeepSeek-V4 CSA prefill attention over [raw | compressed] KV + sink.
    pub(super) prefill_attn_compressed_k: KernelHandle,
    /// Tensor-core sibling (m16n8k16; head_dim=512 only). Default when
    /// present; `ATLAS_V4_PREFILL_TC=0` opts back into the scalar kernel.
    /// Oracle: prefill_attn_tc_microtest — cos 0.9999975 vs scalar,
    /// 7.12 -> 1.85 ms/call at S=896 (2026-08-09). 0 on miss.
    pub(super) prefill_attn_compressed_tc_k: KernelHandle,
    /// Round-2 sibling of the above: identical semantics and arithmetic, pure
    /// data-movement rewrite (natural-K staging, one aliased K/V tile,
    /// ldmatrix B operands, P kept in registers, 20,992 B smem = 3 CTAs/SM
    /// where tc gets 2). OPT-IN while it is being A/B'd on hardware:
    /// `ATLAS_V4_PREFILL_TC2=1`. 0 on miss.
    pub(super) prefill_attn_compressed_tc2_k: KernelHandle,
    /// 4b: # compressed blocks prefill wrote to `mla.compressor.pool` for the
    /// active sequence (= prefill_len / ratio). Decode's compressed arm attends
    /// blocks `[0, this)`. AtomicU32 for interior mutability under prefill's
    /// `&self`; V4 serves max_batch=1 so one counter suffices (inc-3: per-seq
    /// tracking + decode-time append will grow this each boundary crossing).
    pub(super) v4_comp_pool_filled: std::sync::atomic::AtomicU32,
    /// 4b: DEVICE mirror of `v4_comp_pool_filled` — a single u32 the graphed
    /// decode/verify `mla_paged_decode_fp8` kernels read at execution time. A
    /// by-value launch arg froze at graph capture, so a captured γ-verify replay
    /// never saw the compressor's per-step growth and rejected every draft (the
    /// DSpark acceptance bug). Written on-stream via `memset_u32_async` at every
    /// `v4_compress_append` and at per-request init. Allocated ONLY when this
    /// layer owns a compressor (else `DevicePtr::NULL` → kernel reads 0).
    pub(super) v4_comp_count_dev: DevicePtr,
    /// 4b inc-3 decode-append state (V4 serves max_batch=1 → scalar per layer).
    /// `prev_valid`: the CSA `prev_win` ring holds a real previous decode window
    /// (false until the first decode append, and reset each prefill) — when false
    /// the CSA append masks Ca (window-0 semantics). `decode_started`/`first_pos`:
    /// the absolute position of the first decode token this sequence, used to skip
    /// any prefill/decode straddle window whose ring slots aren't all decode-written
    /// (that one block is left as prefill/zero — a documented seam, not corruption).
    pub(super) v4_comp_prev_valid: std::sync::atomic::AtomicBool,
    pub(super) v4_decode_started: std::sync::atomic::AtomicBool,
    pub(super) v4_decode_first_pos: std::sync::atomic::AtomicU32,
    /// 4b inc-3 γ-verify catch-up: per-layer BF16 scratch `[MAX_VERIFY_ROWS × h]`
    /// capturing `c.normed` (the compressor input) for every verify row, so the
    /// post-accept `v4_compress_catchup` can replay `v4_compress_append` for the
    /// committed positions and advance the compressed pool the `pos:None` batched
    /// verify path skips. Allocated ONLY when this layer owns a compressor
    /// (else `DevicePtr::NULL`); MAX_VERIFY_ROWS=8 covers γ≤7 (kt≤8).
    pub(super) verify_comp_normed: DevicePtr,
    /// γ-verify compressor frontier snapshot (the ds4 `spec_frontier_snapshot`
    /// analogue). `v4_compress_speculate` runs the compressor forward over ALL
    /// γ rows *before* the verify attention so each row's causally-visible
    /// compressed blocks actually exist; a partial accept must then rewind
    /// every frontier to the committed prefix. These hold the pre-speculation
    /// scalars — the device buffers go to `CompressorWeights::ring_snap` /
    /// `prev_win_snap`.
    ///
    /// `spec_saved_filled`: `v4_comp_pool_filled` before speculation.
    /// `spec_saved_prev_valid`: `v4_comp_prev_valid` before speculation.
    /// `spec_rows`: how many rows were speculated (0 = speculation did not run
    /// this step, so the rollback is a no-op). `spec_base_pos`: absolute
    /// position of verify row 0, so the rollback can rebuild the ring slot map.
    ///
    /// Pool BYTES past the committed prefix are deliberately NOT restored:
    /// blocks beyond the rewound count are unreachable (the kernel clamps every
    /// row to `seq_len/ratio`) and are overwritten by the next real append.
    /// This mirrors ds4's "invisible garbage" note — only the frontiers and
    /// counters have to be exact.
    pub(super) spec_saved_filled: std::sync::atomic::AtomicU32,
    pub(super) spec_saved_prev_valid: std::sync::atomic::AtomicBool,
    pub(super) spec_rows: std::sync::atomic::AtomicU32,
    pub(super) spec_base_pos: std::sync::atomic::AtomicU32,
    /// First/last pool block saved into `CompressorWeights::pool_snap` by the
    /// γ-speculation snapshot (u32::MAX = nothing saved). Restored with the
    /// frontiers: the CSA boundary append rewrites block w-1, and a rejected
    /// row's rewrite lands INSIDE the committed range, so the "invisible
    /// garbage" rule for un-counted blocks does not cover it.
    pub(super) spec_pool_w_lo: std::sync::atomic::AtomicU32,
    pub(super) spec_pool_w_hi: std::sync::atomic::AtomicU32,
    /// HDIM=512 paged prefill (BF16 KV) for Gemma-4 chunked long-context prefill
    pub(super) prefill_attn_paged_512_k: KernelHandle,
    pub(super) prefill_attn_64_k: KernelHandle,
    pub(super) prefill_attn_paged_k: KernelHandle,
    pub(super) prefill_attn_paged_fp8_k: KernelHandle,
    pub(super) prefill_attn_paged_nvfp4_k: KernelHandle,
    pub(super) prefill_attn_paged_turbo4_k: KernelHandle,
    // BR=64 variants for long-context prefill (q_len >= 256)
    pub(super) prefill_attn_paged_64_k: KernelHandle,
    pub(super) prefill_attn_paged_fp8_64_k: KernelHandle,
    pub(super) prefill_attn_paged_nvfp4_64_k: KernelHandle,
    pub(super) prefill_attn_paged_turbo2_64_k: KernelHandle,
    pub(super) prefill_attn_paged_turbo3_64_k: KernelHandle,
    pub(super) prefill_attn_paged_turbo4_64_k: KernelHandle,
    pub(super) prefill_attn_paged_turbo8_64_k: KernelHandle,
    // ── TurboQuant+ asymmetric BR=64 prefill kernels ──
    // Combined-dtype kernels that read K and V with different on-disk layouts.
    // Currently: Bf16K + Turbo3V (safer-asym variant — K kept at bf16 precision,
    // V aggressively compressed to 3-bit Lloyd-Max + FP8 group scale).
    pub(super) prefill_attn_paged_bf16k_turbo3v_64_k: KernelHandle,
    pub(super) prefill_attn_paged_bf16k_turbo4v_64_k: KernelHandle,
    pub(super) prefill_attn_paged_bf16k_turbo2v_64_k: KernelHandle,
    // Fp8K + TurboNV variants — same shape as bf16k_turbo*v_64 but threads
    // the FP8 K-side per-tensor `k_scale` through to the dequant in
    // LOAD_K_TILE. Targets FP8-attention models (Qwen3.6-35B-FP8 etc.).
    pub(super) prefill_attn_paged_fp8k_turbo3v_64_k: KernelHandle,
    pub(super) prefill_attn_paged_fp8k_turbo4v_64_k: KernelHandle,
    pub(super) prefill_attn_paged_fp8k_turbo2v_64_k: KernelHandle,
    // Both-sides-quantized TurboQuant+ asym (K and V both turbo, separate
    // pool strides). K-side WHT bookend + Q WHT both fire because K is turbo.
    pub(super) prefill_attn_paged_turbo4k_turbo3v_64_k: KernelHandle,
    pub(super) prefill_attn_paged_turbo4k_turbo8v_64_k: KernelHandle,
    pub(super) prefill_attn_paged_turbo3k_turbo8v_64_k: KernelHandle,
    // ── Q12 Phase 3: same-chunk-len batched paged-prefill kernels ──
    // Each takes `const int* const* block_table_ptrs` + per-batch Q/O
    // offsets. Used by `Qwen3AttentionLayer::prefill_batched` when N≥2
    // streams share the same chunk_len. Null on targets that don't
    // carry the corresponding kernel (e.g. CPU backend).
    pub(super) prefill_attn_paged_batched_k: KernelHandle,
    pub(super) prefill_attn_paged_fp8_batched_k: KernelHandle,
    pub(super) prefill_attn_paged_nvfp4_batched_k: KernelHandle,
    pub(super) prefill_attn_paged_batched_64_k: KernelHandle,
    pub(super) prefill_attn_paged_fp8_batched_64_k: KernelHandle,
    pub(super) prefill_attn_paged_nvfp4_batched_64_k: KernelHandle,
    // Batched prefill kernels
    pub(super) deinterleave_qg_split_k: KernelHandle,
    pub(super) deinterleave_qg_split_qnorm_k: KernelHandle,
    pub(super) deinterleave_qg_split_qnorm_mrope_k: KernelHandle,
    pub(super) sigmoid_gate_mul_batched_k: KernelHandle,
    // Pre-dequanted FP8 weights for zero-overhead prefill GEMMs
    pub(super) q_fp8: Option<DevicePtr>,
    pub(super) k_fp8: Option<DevicePtr>,
    pub(super) v_fp8: Option<DevicePtr>,
    pub(super) o_fp8: Option<DevicePtr>,
    pub(super) fp8_gemm_k: KernelHandle,
    // FP8×FP8 GEMM
    pub(super) bf16_to_fp8_k: KernelHandle,
    pub(super) fp8_fp8_gemm_k: KernelHandle,
    // M128 variants
    pub(super) fp8_gemm_t_m128_k: KernelHandle,
    pub(super) fp8_fp8_gemm_t_m128_k: KernelHandle,
    // Native FP4 prefill (mxf4nvf4): present only for models whose kernel dir
    // ships w4a4_gemm_mfast (try_kernel returns 0 elsewhere).
    pub(super) w4a4_gemm_k: KernelHandle,
    pub(super) quantize_nvfp4_k: KernelHandle,
    /// Online FP8 KV scale calibration.
    pub(super) fp8_calibration: Option<Fp8KvCalibration>,
}
