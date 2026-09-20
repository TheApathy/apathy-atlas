// SPDX-License-Identifier: AGPL-3.0-only

//! Per-walk working buffers, carved from the arena's transient region.
//!
//! # This does not come out of the transient arena region
//!
//! `GLM53_T1_TRANSIENT_BYTES` is 1,139,200 bytes and begins with the
//! 90,624-byte schedule workspace, leaving exactly **1,048,576** — which is
//! precisely `262,144 pools x 4 bytes`, the DSA score buffer at the full 1M
//! context. The transient region was sized as *workspace plus max scores* and
//! has no room at all for FFN/KDA/DSA working buffers.
//!
//! That is not a defect in the arena: `Glm53ArenaPlan` documents itself as an
//! "exact checked lower-bound" that "excludes device allocator overhead and
//! effectful ABI deltas, so it is not a runnable peak". Working scratch is one
//! of the things it excludes. So this module binds a **caller-supplied
//! allocation** sized from its own tables, rather than carving the arena, and
//! the model constructor is responsible for allocating it.
//!
//! Every consumer (MoE FFN, dense FFN, KDA attention, DSA attention) is placed
//! from **one** cursor here rather than each carving the region independently,
//! so two consumers cannot overlap by construction. Overlap is the corruption
//! class that yields fluent, wrong output instead of a crash. The regions are
//! deliberately NOT time-multiplexed by liveness. Layer-major prompt scratch
//! is large because DSA retains one full-context score row per prompt token,
//! but disjointness that holds by construction is worth more than an unsafe
//! alias in a 107 GiB resident model.
//!
//! `Glm53SerialMoeKernels::execute` validates every buffer's extent exactly and
//! refuses any pairwise overlap, so the placement here is checked twice: once
//! by the binder (bounds, alignment, tiling) and again by the op on each call.
//! The binder's job is to make the second check unreachable.
//!
//! The FFN input and output are deliberately **not** carved from here. They are
//! the workspace's `collapsed` (the normalized block input written by
//! `hc_pre` + `ffn_norm`) and `hidden_b` (the block output `PostFfn` folds back
//! into the streams), so the FFN reads and writes the same buffers the rest of
//! the walk uses rather than a private copy that would have to be shuttled.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

use crate::layers::Glm53SerialMoeBuffers;
use crate::layers::ops::{GLM53_EXL3_RECONSTRUCT_MAX_BYTES, 
    GLM53_EXL3_LOCK_BYTES, GLM53_EXL3_MAX_INPUT_F16_BYTES, GLM53_EXL3_MAX_OUTPUT_F16_BYTES,
    GLM53_EXL3_MAX_WIDE_INPUT_F16_BYTES, GLM53_EXL3_MAX_WIDE_OUTPUT_F16_BYTES,
    GLM53_EXL3_MAX_WIDE_ROWS, GLM53_EXL3_MOE_LOCK_BYTES, GgmlIqBuffer, Glm53Exl3Buffer,
    Glm53Exl3MoeScratch, Glm53Exl3ProjectionScratch, Glm53Exl3RoutePolicy,
};

use super::arena::{GLM53_T1_TRANSIENT_BYTES, GLM53_T1_WORKSPACE_BYTES};

const ALIGNMENT: u64 = 256;
const MAX_WIDE_ROWS: u32 = GLM53_EXL3_MAX_WIDE_ROWS as u32;
const MAX_WIDE_ROWS_U64: u64 = GLM53_EXL3_MAX_WIDE_ROWS as u64;

/// One DSA layer's raw index tail: TAIL_CAPACITY rows of INDEX_DIM bf16.
const DSA_TAIL_BYTES: u64 = spark_runtime::kv_cache::GLM53_DSA_TAIL_CAPACITY as u64 * 128 * 2;
/// A chunk can begin with three carried tail rows, so M2048 completes at most
/// floor((2048 + 3) / 4) = 512 fresh k-pools.
const DSA_MAX_WIDE_COMPLETE_POOLS: u64 = (MAX_WIDE_ROWS_U64 + 3) / 4;

/// Cumulative offsets of each table inside the concatenated `bound[..]`.
///
/// Derived, never written by hand. `bind` maps the chained tables positionally,
/// so a literal `bound[16 + index]` silently rebinds every KDA and DSA buffer to
/// a neighbour the moment MOE_SCRATCH or DENSE_SCRATCH grows. That failure is
/// worse than an out-of-range panic: it produces plausible wrong data instead of
/// stopping. Deriving them makes the drift impossible rather than unlikely.
const KDA_OFFSET: usize = MOE_SCRATCH.len() + DENSE_SCRATCH.len();
const DSA_OFFSET: usize = KDA_OFFSET + KDA_SCRATCH.len();
const ROUTER_DIAG_OFFSET: usize = DSA_OFFSET + DSA_SCRATCH.len();
const GROUPED_MOE_OFFSET: usize = ROUTER_DIAG_OFFSET + ROUTER_DIAG_SCRATCH.len();
const EXL3_OFFSET: usize = GROUPED_MOE_OFFSET + GROUPED_MOE_SCRATCH.len();
const EXL3_MOE_OFFSET: usize = EXL3_OFFSET + EXL3_SCRATCH.len();
const EXL3_WIDE_OFFSET: usize = EXL3_MOE_OFFSET + EXL3_MOE_SCRATCH.len();
const EXL3_WIDE_DENSE_OFFSET: usize = EXL3_WIDE_OFFSET + EXL3_WIDE_SCRATCH.len();
const EXL3_WIDE_ROUTER_MOE_OFFSET: usize = EXL3_WIDE_DENSE_OFFSET + EXL3_WIDE_DENSE_SCRATCH.len();
const EXL3_WIDE_KDA_OFFSET: usize =
    EXL3_WIDE_ROUTER_MOE_OFFSET + EXL3_WIDE_ROUTER_MOE_SCRATCH.len();
const EXL3_WIDE_DSA_OFFSET: usize = EXL3_WIDE_KDA_OFFSET + EXL3_WIDE_KDA_SCRATCH.len();
const EXL3_MOE_STAGED_OFFSET: usize = EXL3_WIDE_DSA_OFFSET + EXL3_WIDE_DSA_SCRATCH.len();
const EXL3_RECONSTRUCT_OFFSET: usize = EXL3_MOE_STAGED_OFFSET + EXL3_MOE_STAGED_SCRATCH.len();

// Documents the invariant and fails the build loudly if anyone reintroduces a
// literal, or grows a table without noticing what it shifts.
const _: () = {
    assert!(KDA_OFFSET == 16);
    assert!(DSA_OFFSET == 32);
    assert!(ROUTER_DIAG_OFFSET == 52);
    assert!(ROUTER_DIAG_OFFSET + ROUTER_DIAG_SCRATCH.len() == 54);
    assert!(GROUPED_MOE_OFFSET == 54);
    assert!(GROUPED_MOE_OFFSET + GROUPED_MOE_SCRATCH.len() == 58);
    assert!(EXL3_OFFSET == 58);
    assert!(EXL3_OFFSET + EXL3_SCRATCH.len() == 62);
    assert!(EXL3_MOE_OFFSET == 62);
    assert!(EXL3_MOE_OFFSET + EXL3_MOE_SCRATCH.len() == 72);
    assert!(EXL3_WIDE_OFFSET == 72);
    assert!(EXL3_WIDE_DENSE_OFFSET == 76);
    assert!(EXL3_WIDE_ROUTER_MOE_OFFSET == 79);
    assert!(EXL3_WIDE_ROUTER_MOE_OFFSET + EXL3_WIDE_ROUTER_MOE_SCRATCH.len() == 89);
    assert!(EXL3_WIDE_KDA_OFFSET == 89);
    assert!(EXL3_WIDE_KDA_OFFSET + EXL3_WIDE_KDA_SCRATCH.len() == 106);
    assert!(EXL3_WIDE_DSA_OFFSET == 106);
    assert!(EXL3_WIDE_DSA_OFFSET + EXL3_WIDE_DSA_SCRATCH.len() == 132);
    assert!(EXL3_MOE_STAGED_OFFSET == 132);
    assert!(EXL3_MOE_STAGED_OFFSET + EXL3_MOE_STAGED_SCRATCH.len() == 138);
    assert!(EXL3_RECONSTRUCT_OFFSET == 138);
};

/// Prompt-scope original-basis F16 weight staging (appended; nothing moves).
const EXL3_RECONSTRUCT_SCRATCH: [(&str, u64); 3] = [
    (
        "exl3_reconstruct_weight_f16",
        GLM53_EXL3_RECONSTRUCT_MAX_BYTES as u64,
    ),
    // Prompt-scope F32 activation staging (router logits GEMM input; mHC
    // mixing GEMM output): 2048 x 4096 f32.
    ("exl3_prompt_f32_scratch", MAX_WIDE_ROWS_U64 * 4_096 * 4),
    // Second reconstruct staging slot so reconstruct(i+1) can overlap GEMM(i).
    (
        "exl3_reconstruct_weight_f16_b",
        GLM53_EXL3_RECONSTRUCT_MAX_BYTES as u64,
    ),
];

/// Exact extents `Glm53SerialMoeKernels::execute` requires, in placement order.
/// Names match the op's buffer struct so a mismatch is greppable.
const MOE_SCRATCH: [(&str, u64); 12] = [
    // One f32 logit per expert, consumed by the top-k sigmoid.
    ("router_logits_f32", 1_152),
    ("route_ids_u32", 32),
    ("route_weights_f32", 32),
    ("q8_activation", 4_608),
    ("expert_gate_bf16", 4_096),
    ("expert_up_bf16", 4_096),
    ("expert_swiglu_bf16", 4_096),
    ("routed_bf16", 65_536),
    ("shared_gate_bf16", 4_096),
    ("shared_up_bf16", 4_096),
    ("shared_swiglu_bf16", 4_096),
    ("shared_bf16", 8_192),
];

/// Dense-FFN extents. Layers 0..3 use a 12,288-wide intermediate (versus the
/// MoE experts' 2,048), and the down projection quantizes a 12,288-long
/// activation: `(12288 / 128) * 144 = 13,824` bytes.
const DENSE_SCRATCH: [(&str, u64); 4] = [
    ("dense_q8_activation", 13_824),
    ("dense_gate_bf16", 24_576),
    ("dense_up_bf16", 24_576),
    ("dense_swiglu_bf16", 24_576),
];

/// KDA attention extents for one token. The projections are `n_head * head_dim`
/// = 64 * 128 = 8,192 wide in BF16; the low-rank gate stages are 128 wide. The
/// Q8 activation is sized for the widest matmul in the layer, `attn_output`
/// with an 8,192-long inner: `(8192 / 128) * 144 = 9,216` bytes.
const KDA_SCRATCH: [(&str, u64); 16] = [
    ("kda_q_proj_bf16", 16_384),
    ("kda_k_proj_bf16", 16_384),
    ("kda_v_proj_bf16", 16_384),
    ("kda_q_conv_bf16", 16_384),
    ("kda_k_conv_bf16", 16_384),
    ("kda_v_conv_bf16", 16_384),
    ("kda_f_a_bf16", 256),
    ("kda_f_b_bf16", 16_384),
    ("kda_log_decay_f32", 32_768),
    ("kda_beta_proj_bf16", 128),
    ("kda_beta_bf16", 128),
    ("kda_g_a_bf16", 256),
    ("kda_g_b_bf16", 16_384),
    ("kda_recurrent_out_bf16", 16_384),
    ("kda_gated_bf16", 16_384),
    ("kda_q8_activation", 9_216),
];

/// DSA attention extents for one token.
///
/// `scores_f32` dominates and is context-proportional: one f32 per k-pool, and
/// at the full 1,048,576-token context with `kpool = 4` that is 262,144 pools,
/// so 1 MiB. It is sized for the maximum here so a long sequence cannot outgrow
/// its buffer mid-generation.
const DSA_SCRATCH: [(&str, u64); 20] = [
    ("dsa_qr_bf16", 3_072),
    ("dsa_qr_norm_bf16", 3_072),
    ("dsa_q_b_bf16", 32_768),
    ("dsa_absorbed_q_bf16", 65_536),
    ("dsa_kv_cmpr_bf16", 1_024),
    ("dsa_kv_cmpr_norm_bf16", 1_024),
    ("dsa_index_k_bf16", 256),
    ("dsa_index_k_norm_bf16", 256),
    ("dsa_index_g_bf16", 256),
    ("dsa_index_q_bf16", 8_192),
    ("dsa_head_weights_bf16", 64),
    ("dsa_scores_f32", 1_048_576),
    ("dsa_selected_indices_i32", 8_204),
    ("dsa_weighted_latent_bf16", 65_536),
    ("dsa_unabsorbed_bf16", 32_768),
    ("dsa_pool_keys_bf16", 256),
    ("dsa_pool_validity_u8", 256),
    // Derived from the single tail-capacity definition, not the old literal 768
    // (= 3 x INDEX_DIM x 2). A hardcoded size here would stay short while the
    // constant moved, which is the whole failure this change is fixing.
    ("dsa_tail_keys_bf16", DSA_TAIL_BYTES),
    ("dsa_tail_gates_bf16", DSA_TAIL_BYTES),
    // attn_output has a 16,384-long inner: (16384 / 128) * 144.
    ("dsa_q8_activation", 18_432),
];

/// Router diagnostics: the full per-expert score vectors, 288 x f32 each.
///
/// Chained LAST, after DSA. `bind` maps the concatenated tables positionally
/// onto `bound[..]`, so adding these anywhere earlier would silently rebind
/// every slot after them -- route_ids_u32 included -- to the wrong buffer.
/// Appending is what keeps the existing indices meaning what they say.
/// Extents the GROUPED MoE path needs, chained LAST so that adding them shifts
/// no existing `bound[]` index.
///
/// Appending to MOE_SCRATCH instead would move DENSE, KDA, DSA and the router
/// diagnostics by four slots each, silently rebinding every one of them to a
/// neighbour. That is the failure the derived offsets above exist to prevent,
/// and a new table is how you avoid it rather than merely detect it.
///
/// The serial seam sizes gate/up/swiglu for ONE expert and reuses them per
/// slot; the grouped path runs all TOP_K at once and needs TOP_K of each.
/// EXPERT_INTERMEDIATE is 2048, so 8 * 2048 * 2 = 32,768 bytes apiece.
///
/// The down projection is the one that is NOT shared: each slot consumes its
/// own swiglu row, so its quantized activation is TOP_K blocks back to back.
/// Down's inner extent is EXPERT_INTERMEDIATE, and the MMQ activation formula
/// this file already uses for the dense path is (inner / 128) * 144, giving
/// (2048 / 128) * 144 = 2,304 per block and 8 * 2,304 = 18,432.
const GROUPED_MOE_SCRATCH: [(&str, u64); 4] = [
    ("grouped_gate_bf16", 32_768),
    ("grouped_up_bf16", 32_768),
    ("grouped_swiglu_bf16", 32_768),
    ("grouped_down_q8", 18_432),
];

const ROUTER_DIAG_SCRATCH: [(&str, u64); 2] =
    [("router_probs_f32", 1_152), ("router_biased_f32", 1_152)];

/// Shared across every EXL3 projection in one ordered target walk.
const EXL3_SCRATCH: [(&str, u64); 4] = [
    ("exl3_input_f16", GLM53_EXL3_MAX_INPUT_F16_BYTES as u64),
    ("exl3_output_f16", GLM53_EXL3_MAX_OUTPUT_F16_BYTES as u64),
    ("exl3_locks_i32", GLM53_EXL3_LOCK_BYTES as u64),
    (
        "exl3_input_hadamard_f16",
        GLM53_EXL3_MAX_INPUT_F16_BYTES as u64,
    ),
];

/// Max-M2048 device-routed EXL3 MoE scratch. Appended after every established
/// slot and prefix-sliced to the active row count by the executor. DFlash K8
/// consumes the same first-eight-row prefixes as before.
const EXL3_MOE_MAX_PAIRS: u64 = MAX_WIDE_ROWS_U64 * 8;
const EXL3_MOE_MAX_CHUNKS: u64 = EXL3_MOE_MAX_PAIRS.div_ceil(16) + 288;
const EXL3_MOE_SCRATCH: [(&str, u64); 10] = [
    ("exl3_moe_expert_count_i64", 2_312),
    ("exl3_moe_token_sorted_i64", MAX_WIDE_ROWS_U64 * 8 * 8),
    ("exl3_moe_weight_sorted_f16", MAX_WIDE_ROWS_U64 * 8 * 2),
    (
        "exl3_moe_temp_state_g_f16",
        8 * MAX_WIDE_ROWS_U64 * 4_096 * 2,
    ),
    (
        "exl3_moe_temp_state_u_f16",
        8 * MAX_WIDE_ROWS_U64 * 4_096 * 2,
    ),
    (
        "exl3_moe_temp_intermediate_g_f16",
        8 * MAX_WIDE_ROWS_U64 * 2_048 * 2,
    ),
    (
        "exl3_moe_temp_intermediate_u_f16",
        8 * MAX_WIDE_ROWS_U64 * 2_048 * 2,
    ),
    ("exl3_moe_output_f32", MAX_WIDE_ROWS_U64 * 4_096 * 4),
    ("exl3_moe_route_status_u32", 4),
    ("exl3_moe_locks_i32", GLM53_EXL3_MOE_LOCK_BYTES as u64),
];

/// Staged expert-major descriptors append after all established scratch slots,
/// so enabling the experiment cannot silently rebind an existing consumer.
const EXL3_MOE_STAGED_SCRATCH: [(&str, u64); 6] = [
    ("exl3_moe_pair_expert_u32", EXL3_MOE_MAX_PAIRS * 4),
    ("exl3_moe_chunk_expert_u32", EXL3_MOE_MAX_CHUNKS * 4),
    ("exl3_moe_chunk_start_u32", EXL3_MOE_MAX_CHUNKS * 4),
    ("exl3_moe_chunk_rows_u32", EXL3_MOE_MAX_CHUNKS * 4),
    ("exl3_moe_chunk_count_u32", 4),
    // Legacy K8 extent; layout() substitutes the latched prefill extent only
    // for this final slot, preserving every preceding consumer's address.
    (
        "exl3_moe_route_private_f32",
        Glm53Exl3RoutePolicy::legacy().scratch_private_bytes(),
    ),
];

/// Separate max-M2048 projection scratch, appended so no established T1 slot
/// moves. Locks retain the upstream fixed ABI; activation storage scales by M.
const EXL3_WIDE_SCRATCH: [(&str, u64); 4] = [
    (
        "exl3_wide_input_f16",
        GLM53_EXL3_MAX_WIDE_INPUT_F16_BYTES as u64,
    ),
    (
        "exl3_wide_output_f16",
        GLM53_EXL3_MAX_WIDE_OUTPUT_F16_BYTES as u64,
    ),
    ("exl3_wide_locks_i32", GLM53_EXL3_LOCK_BYTES as u64),
    (
        "exl3_wide_input_hadamard_f16",
        GLM53_EXL3_MAX_WIDE_INPUT_F16_BYTES as u64,
    ),
];

/// EXL3 dense intermediates for the maximum prompt chunk. The Q8 slot is absent:
/// compressed EXL3 projections consume the shared projection scratch above.
const EXL3_WIDE_DENSE_SCRATCH: [(&str, u64); 3] = [
    ("exl3_wide_dense_gate_bf16", MAX_WIDE_ROWS_U64 * 12_288 * 2),
    ("exl3_wide_dense_up_bf16", MAX_WIDE_ROWS_U64 * 12_288 * 2),
    (
        "exl3_wide_dense_swiglu_bf16",
        MAX_WIDE_ROWS_U64 * 12_288 * 2,
    ),
];

/// Router and shared-expert buffers for the maximum prompt chunk. Routed-expert
/// intermediates live in `EXL3_MOE_SCRATCH`; only the fused path uses these.
const EXL3_WIDE_ROUTER_MOE_SCRATCH: [(&str, u64); 10] = [
    ("exl3_wide_router_logits_f32", MAX_WIDE_ROWS_U64 * 288 * 4),
    ("exl3_wide_router_probs_f32", MAX_WIDE_ROWS_U64 * 288 * 4),
    ("exl3_wide_router_biased_f32", MAX_WIDE_ROWS_U64 * 288 * 4),
    ("exl3_wide_route_ids_u32", MAX_WIDE_ROWS_U64 * 8 * 4),
    ("exl3_wide_route_weights_f32", MAX_WIDE_ROWS_U64 * 8 * 4),
    ("exl3_wide_shared_gate_bf16", MAX_WIDE_ROWS_U64 * 2_048 * 2),
    ("exl3_wide_shared_up_bf16", MAX_WIDE_ROWS_U64 * 2_048 * 2),
    (
        "exl3_wide_shared_swiglu_bf16",
        MAX_WIDE_ROWS_U64 * 2_048 * 2,
    ),
    ("exl3_wide_shared_bf16", MAX_WIDE_ROWS_U64 * 4_096 * 2),
    ("exl3_wide_unused_q8", 1),
];

/// KDA intermediates for one causal prompt chunk.
const EXL3_WIDE_KDA_SCRATCH: [(&str, u64); 17] = [
    ("exl3_wide_kda_q_proj_bf16", MAX_WIDE_ROWS_U64 * 16_384),
    ("exl3_wide_kda_k_proj_bf16", MAX_WIDE_ROWS_U64 * 16_384),
    ("exl3_wide_kda_v_proj_bf16", MAX_WIDE_ROWS_U64 * 16_384),
    ("exl3_wide_kda_q_conv_bf16", MAX_WIDE_ROWS_U64 * 16_384),
    ("exl3_wide_kda_k_conv_bf16", MAX_WIDE_ROWS_U64 * 16_384),
    ("exl3_wide_kda_v_conv_bf16", MAX_WIDE_ROWS_U64 * 16_384),
    ("exl3_wide_kda_f_a_bf16", MAX_WIDE_ROWS_U64 * 256),
    ("exl3_wide_kda_f_b_bf16", MAX_WIDE_ROWS_U64 * 16_384),
    ("exl3_wide_kda_log_decay_f32", MAX_WIDE_ROWS_U64 * 32_768),
    ("exl3_wide_kda_beta_proj_bf16", MAX_WIDE_ROWS_U64 * 128),
    ("exl3_wide_kda_beta_bf16", MAX_WIDE_ROWS_U64 * 128),
    ("exl3_wide_kda_g_a_bf16", MAX_WIDE_ROWS_U64 * 256),
    ("exl3_wide_kda_g_b_bf16", MAX_WIDE_ROWS_U64 * 16_384),
    (
        "exl3_wide_kda_recurrent_out_bf16",
        MAX_WIDE_ROWS_U64 * 16_384,
    ),
    ("exl3_wide_kda_gated_bf16", MAX_WIDE_ROWS_U64 * 16_384),
    ("exl3_wide_kda_unused_q8", 1),
    (
        "exl3_wide_kda_combined_qkv_bf16",
        MAX_WIDE_ROWS_U64 * 3 * 16_384,
    ),
];

/// DSA verifier-chunk intermediates, including row/head transposition and
/// query metadata that the persistent one-row cache ABI cannot hold.
const EXL3_WIDE_DSA_SCRATCH: [(&str, u64); 26] = [
    ("exl3_wide_dsa_qr_bf16", MAX_WIDE_ROWS_U64 * 3_072),
    ("exl3_wide_dsa_qr_norm_bf16", MAX_WIDE_ROWS_U64 * 3_072),
    ("exl3_wide_dsa_q_b_bf16", MAX_WIDE_ROWS_U64 * 32_768),
    ("exl3_wide_dsa_absorbed_q_bf16", MAX_WIDE_ROWS_U64 * 65_536),
    ("exl3_wide_dsa_kv_cmpr_bf16", MAX_WIDE_ROWS_U64 * 1_024),
    ("exl3_wide_dsa_kv_cmpr_norm_bf16", MAX_WIDE_ROWS_U64 * 1_024),
    ("exl3_wide_dsa_index_k_bf16", MAX_WIDE_ROWS_U64 * 256),
    ("exl3_wide_dsa_index_k_norm_bf16", MAX_WIDE_ROWS_U64 * 256),
    ("exl3_wide_dsa_index_g_bf16", MAX_WIDE_ROWS_U64 * 256),
    ("exl3_wide_dsa_index_q_bf16", MAX_WIDE_ROWS_U64 * 8_192),
    ("exl3_wide_dsa_head_weights_bf16", MAX_WIDE_ROWS_U64 * 64),
    ("exl3_wide_dsa_scores_f32", MAX_WIDE_ROWS_U64 * 1_048_576),
    (
        "exl3_wide_dsa_selected_indices_i32",
        MAX_WIDE_ROWS_U64 * 8_204,
    ),
    (
        "exl3_wide_dsa_weighted_latent_bf16",
        MAX_WIDE_ROWS_U64 * 65_536,
    ),
    ("exl3_wide_dsa_unabsorbed_bf16", MAX_WIDE_ROWS_U64 * 32_768),
    (
        "exl3_wide_dsa_pool_keys_bf16",
        DSA_MAX_WIDE_COMPLETE_POOLS * 128 * 2,
    ),
    (
        "exl3_wide_dsa_pool_validity_u8",
        DSA_MAX_WIDE_COMPLETE_POOLS,
    ),
    ("exl3_wide_dsa_tail_keys_bf16", DSA_TAIL_BYTES),
    ("exl3_wide_dsa_tail_gates_bf16", DSA_TAIL_BYTES),
    ("exl3_wide_dsa_unused_q8", 1),
    ("exl3_wide_dsa_head_major_large", MAX_WIDE_ROWS_U64 * 65_536),
    ("exl3_wide_dsa_head_major_small", MAX_WIDE_ROWS_U64 * 32_768),
    ("exl3_wide_dsa_sequence_length_u32", 4),
    ("exl3_wide_dsa_query_positions_u32", MAX_WIDE_ROWS_U64 * 4),
    ("exl3_wide_dsa_query_validity_u8", MAX_WIDE_ROWS_U64),
    ("exl3_wide_dsa_tail_validity_u8", 3),
];

/// The largest k-pool count the score buffer is sized for: full 1M / kpool 4.
pub const GLM53_DSA_MAX_POOLS: u64 = 262_144;

const _: () = {
    // The score buffer must cover the maximum pool count exactly.
    assert!(DSA_SCRATCH[11].1 == GLM53_DSA_MAX_POOLS * 4);
};

/// The eleven MoE scratch buffers, bound once per walk.
#[derive(Debug, Clone, Copy)]
pub struct Glm53WalkScratch {
    exl3_route_policy: Glm53Exl3RoutePolicy,
    router_logits_f32: GgmlIqBuffer,
    route_ids_u32: GgmlIqBuffer,
    route_weights_f32: GgmlIqBuffer,
    q8_activation: GgmlIqBuffer,
    expert_gate_bf16: GgmlIqBuffer,
    expert_up_bf16: GgmlIqBuffer,
    expert_swiglu_bf16: GgmlIqBuffer,
    grouped_gate_bf16: GgmlIqBuffer,
    grouped_up_bf16: GgmlIqBuffer,
    grouped_swiglu_bf16: GgmlIqBuffer,
    grouped_down_q8: GgmlIqBuffer,
    routed_bf16: GgmlIqBuffer,
    shared_gate_bf16: GgmlIqBuffer,
    shared_up_bf16: GgmlIqBuffer,
    shared_swiglu_bf16: GgmlIqBuffer,
    shared_bf16: GgmlIqBuffer,
    dense_q8_activation: GgmlIqBuffer,
    dense_gate_bf16: GgmlIqBuffer,
    dense_up_bf16: GgmlIqBuffer,
    dense_swiglu_bf16: GgmlIqBuffer,
    kda: [GgmlIqBuffer; 16],
    dsa: [GgmlIqBuffer; 20],
    router_probs_f32: GgmlIqBuffer,
    router_biased_f32: GgmlIqBuffer,
    exl3: [GgmlIqBuffer; 4],
    exl3_moe: [GgmlIqBuffer; 10],
    exl3_wide: [GgmlIqBuffer; 4],
    exl3_wide_dense: [GgmlIqBuffer; 3],
    exl3_wide_router_moe: [GgmlIqBuffer; 10],
    exl3_wide_kda: [GgmlIqBuffer; 17],
    exl3_wide_dsa: [GgmlIqBuffer; 26],
    exl3_moe_staged: [GgmlIqBuffer; 6],
    exl3_reconstruct: [GgmlIqBuffer; 3],
}

/// The KDA attention working buffers for one layer.
#[derive(Debug, Clone, Copy)]
pub struct Glm53KdaScratchBuffers {
    pub combined_qkv_bf16: GgmlIqBuffer,
    pub q_proj_bf16: GgmlIqBuffer,
    pub k_proj_bf16: GgmlIqBuffer,
    pub v_proj_bf16: GgmlIqBuffer,
    pub q_conv_bf16: GgmlIqBuffer,
    pub k_conv_bf16: GgmlIqBuffer,
    pub v_conv_bf16: GgmlIqBuffer,
    pub f_a_bf16: GgmlIqBuffer,
    pub f_b_bf16: GgmlIqBuffer,
    pub log_decay_f32: GgmlIqBuffer,
    pub beta_proj_bf16: GgmlIqBuffer,
    pub beta_bf16: GgmlIqBuffer,
    pub g_a_bf16: GgmlIqBuffer,
    pub g_b_bf16: GgmlIqBuffer,
    pub recurrent_out_bf16: GgmlIqBuffer,
    pub gated_bf16: GgmlIqBuffer,
    pub q8_activation: GgmlIqBuffer,
}

/// The DSA attention working buffers for one layer.
#[derive(Debug, Clone, Copy)]
pub struct Glm53DsaScratchBuffers {
    pub qr_bf16: GgmlIqBuffer,
    pub qr_norm_bf16: GgmlIqBuffer,
    pub q_b_bf16: GgmlIqBuffer,
    pub absorbed_q_bf16: GgmlIqBuffer,
    pub kv_cmpr_bf16: GgmlIqBuffer,
    pub kv_cmpr_norm_bf16: GgmlIqBuffer,
    pub index_k_bf16: GgmlIqBuffer,
    pub index_k_norm_bf16: GgmlIqBuffer,
    pub index_g_bf16: GgmlIqBuffer,
    pub index_q_bf16: GgmlIqBuffer,
    pub head_weights_bf16: GgmlIqBuffer,
    pub scores_f32: GgmlIqBuffer,
    pub selected_indices_i32: GgmlIqBuffer,
    pub weighted_latent_bf16: GgmlIqBuffer,
    pub unabsorbed_bf16: GgmlIqBuffer,
    pub pool_keys_bf16: GgmlIqBuffer,
    pub pool_validity_u8: GgmlIqBuffer,
    pub tail_keys_bf16: GgmlIqBuffer,
    pub tail_gates_bf16: GgmlIqBuffer,
    pub q8_activation: GgmlIqBuffer,
    pub head_major_large_bf16: GgmlIqBuffer,
    pub head_major_small_bf16: GgmlIqBuffer,
    pub sequence_length_u32: GgmlIqBuffer,
    pub query_positions_u32: GgmlIqBuffer,
    pub query_validity_u8: GgmlIqBuffer,
    pub tail_validity_u8: GgmlIqBuffer,
}

/// The four dense-FFN working buffers.
#[derive(Debug, Clone, Copy)]
pub struct Glm53DenseFfnBuffers {
    pub q8_activation: GgmlIqBuffer,
    pub gate_bf16: GgmlIqBuffer,
    pub up_bf16: GgmlIqBuffer,
    pub swiglu_bf16: GgmlIqBuffer,
}

fn align_up(value: u64) -> u64 {
    (value + ALIGNMENT - 1) & !(ALIGNMENT - 1)
}

fn layout(policy: Glm53Exl3RoutePolicy) -> impl Iterator<Item = (&'static str, u64)> {
    MOE_SCRATCH
        .iter()
        .chain(DENSE_SCRATCH.iter())
        .chain(KDA_SCRATCH.iter())
        .chain(DSA_SCRATCH.iter())
        .chain(ROUTER_DIAG_SCRATCH.iter())
        .chain(GROUPED_MOE_SCRATCH.iter())
        .chain(EXL3_SCRATCH.iter())
        .chain(EXL3_MOE_SCRATCH.iter())
        .chain(EXL3_WIDE_SCRATCH.iter())
        .chain(EXL3_WIDE_DENSE_SCRATCH.iter())
        .chain(EXL3_WIDE_ROUTER_MOE_SCRATCH.iter())
        .chain(EXL3_WIDE_KDA_SCRATCH.iter())
        .chain(EXL3_WIDE_DSA_SCRATCH.iter())
        .chain(EXL3_MOE_STAGED_SCRATCH.iter())
        .chain(EXL3_RECONSTRUCT_SCRATCH.iter())
        .copied()
        .enumerate()
        .map(move |(index, (name, bytes))| {
            let bytes = if index == EXL3_MOE_STAGED_OFFSET + EXL3_MOE_STAGED_SCRATCH.len() - 1 {
                policy.scratch_private_bytes()
            } else {
                bytes
            };
            (name, bytes)
        })
}

impl Glm53WalkScratch {
    /// Bind scratch from a dedicated allocation.
    ///
    /// `base` must be a 256-byte-aligned device allocation of at least
    /// [`Glm53WalkScratch::required_bytes`] bytes, distinct from the arena.
    pub fn bind(base: DevicePtr, bytes: u64) -> Result<Self> {
        Self::bind_with_route_policy(base, bytes, Glm53Exl3RoutePolicy::legacy())
    }

    pub fn bind_with_route_policy(
        base: DevicePtr,
        bytes: u64,
        policy: Glm53Exl3RoutePolicy,
    ) -> Result<Self> {
        ensure!(!base.is_null(), "GLM walk scratch base is NULL");
        let required = Self::required_bytes_with_route_policy(policy);
        ensure!(
            bytes >= required,
            "GLM walk scratch allocation is {bytes} bytes, need {required}"
        );
        let start = base.0;
        start
            .checked_add(required)
            .context("GLM walk scratch allocation address overflow")?;
        ensure!(
            start.is_multiple_of(ALIGNMENT),
            "GLM walk scratch base is {ALIGNMENT}-byte misaligned"
        );

        let bytes_available = bytes;
        let mut cursor = 0u64;
        let mut bound = Vec::with_capacity(
            MOE_SCRATCH.len() + DENSE_SCRATCH.len() + KDA_SCRATCH.len() + DSA_SCRATCH.len(),
        );
        // MoE and dense buffers are placed disjointly rather than aliased: a
        // layer is one or the other, but overlapping them would make a
        // scheduling bug silently read another layer's activations instead of
        // failing, and there is a megabyte to spare.
        for (name, bytes) in layout(policy) {
            let offset = align_up(cursor);
            let end = offset
                .checked_add(bytes)
                .with_context(|| format!("GLM walk scratch {name} extent overflow"))?;
            ensure!(
                end <= bytes_available,
                "GLM walk scratch {name} ends at {end}, past the {bytes_available}-byte allocation"
            );
            bound.push(GgmlIqBuffer {
                ptr: DevicePtr(
                    start
                        .checked_add(offset)
                        .with_context(|| format!("GLM walk scratch {name} address overflow"))?,
                ),
                bytes: usize::try_from(bytes)?,
            });
            cursor = end;
        }

        Ok(Self {
            exl3_route_policy: policy,
            router_logits_f32: bound[0],
            route_ids_u32: bound[1],
            route_weights_f32: bound[2],
            q8_activation: bound[3],
            expert_gate_bf16: bound[4],
            expert_up_bf16: bound[5],
            expert_swiglu_bf16: bound[6],
            grouped_gate_bf16: bound[GROUPED_MOE_OFFSET],
            grouped_up_bf16: bound[GROUPED_MOE_OFFSET + 1],
            grouped_swiglu_bf16: bound[GROUPED_MOE_OFFSET + 2],
            grouped_down_q8: bound[GROUPED_MOE_OFFSET + 3],
            routed_bf16: bound[7],
            shared_gate_bf16: bound[8],
            shared_up_bf16: bound[9],
            shared_swiglu_bf16: bound[10],
            shared_bf16: bound[11],
            dense_q8_activation: bound[12],
            dense_gate_bf16: bound[13],
            dense_up_bf16: bound[14],
            dense_swiglu_bf16: bound[15],
            kda: core::array::from_fn(|index| bound[KDA_OFFSET + index]),
            dsa: core::array::from_fn(|index| bound[DSA_OFFSET + index]),
            router_probs_f32: bound[ROUTER_DIAG_OFFSET],
            router_biased_f32: bound[ROUTER_DIAG_OFFSET + 1],
            exl3: core::array::from_fn(|index| bound[EXL3_OFFSET + index]),
            exl3_moe: core::array::from_fn(|index| bound[EXL3_MOE_OFFSET + index]),
            exl3_wide: core::array::from_fn(|index| bound[EXL3_WIDE_OFFSET + index]),
            exl3_wide_dense: core::array::from_fn(|index| bound[EXL3_WIDE_DENSE_OFFSET + index]),
            exl3_wide_router_moe: core::array::from_fn(|index| {
                bound[EXL3_WIDE_ROUTER_MOE_OFFSET + index]
            }),
            exl3_wide_kda: core::array::from_fn(|index| bound[EXL3_WIDE_KDA_OFFSET + index]),
            exl3_wide_dsa: core::array::from_fn(|index| bound[EXL3_WIDE_DSA_OFFSET + index]),
            exl3_moe_staged: core::array::from_fn(|index| bound[EXL3_MOE_STAGED_OFFSET + index]),
            exl3_reconstruct: core::array::from_fn(|index| bound[EXL3_RECONSTRUCT_OFFSET + index]),
        })
    }

    /// Assemble the op's buffer set around a walk-owned input and output.
    ///
    /// `input` is the normalized block input and `output` the block output;
    /// both come from the workspace, not from scratch.
    pub fn moe_buffers(&self, input: GgmlIqBuffer, output: GgmlIqBuffer) -> Glm53SerialMoeBuffers {
        Glm53SerialMoeBuffers {
            input_bf16: input,
            route_ids_u32: self.route_ids_u32,
            route_weights_f32: self.route_weights_f32,
            q8_activation: self.q8_activation,
            expert_gate_bf16: self.expert_gate_bf16,
            expert_up_bf16: self.expert_up_bf16,
            expert_swiglu_bf16: self.expert_swiglu_bf16,
            routed_bf16: self.routed_bf16,
            shared_gate_bf16: self.shared_gate_bf16,
            shared_up_bf16: self.shared_up_bf16,
            shared_swiglu_bf16: self.shared_swiglu_bf16,
            shared_bf16: self.shared_bf16,
            output_bf16: output,
        }
    }

    /// The four extra buffers the grouped MoE path needs.
    pub fn grouped_moe_scratch(&self) -> crate::layers::Glm53GroupedMoeScratch {
        crate::layers::Glm53GroupedMoeScratch {
            gate_bf16: self.grouped_gate_bf16,
            up_bf16: self.grouped_up_bf16,
            swiglu_bf16: self.grouped_swiglu_bf16,
            down_q8: self.grouped_down_q8,
        }
    }

    /// The router's scratch: expert logits plus the decoded route.
    pub fn router_scratch(&self) -> (GgmlIqBuffer, GgmlIqBuffer, GgmlIqBuffer) {
        (
            self.router_logits_f32,
            self.route_ids_u32,
            self.route_weights_f32,
        )
    }

    pub fn router_scratch_rows(
        &self,
        rows: u32,
    ) -> Result<(GgmlIqBuffer, GgmlIqBuffer, GgmlIqBuffer)> {
        ensure!(
            (1..=MAX_WIDE_ROWS).contains(&rows),
            "GLM router scratch rows must be 1..={MAX_WIDE_ROWS}"
        );
        if rows == 1 {
            return Ok(self.router_scratch());
        }
        let rows = usize::try_from(rows)?;
        Ok((
            prefix_buffer(self.exl3_wide_router_moe[0], rows * 288 * 4)?,
            prefix_buffer(self.exl3_wide_router_moe[3], rows * 8 * 4)?,
            prefix_buffer(self.exl3_wide_router_moe[4], rows * 8 * 4)?,
        ))
    }

    /// The router's full per-expert score vectors: `(pre_bias, post_bias)`.
    ///
    /// GLM ranks on the post-bias score but emits the pre-bias sigmoid as the
    /// weight, so both are exposed by name. Inferring one from the other is
    /// what produced a phantom "8th kept expert" that was not in the top-8.
    pub fn router_scores(&self) -> (GgmlIqBuffer, GgmlIqBuffer) {
        (self.router_probs_f32, self.router_biased_f32)
    }

    pub fn router_scores_rows(&self, rows: u32) -> Result<(GgmlIqBuffer, GgmlIqBuffer)> {
        ensure!(
            (1..=MAX_WIDE_ROWS).contains(&rows),
            "GLM router score rows must be 1..={MAX_WIDE_ROWS}"
        );
        if rows == 1 {
            return Ok(self.router_scores());
        }
        let bytes = usize::try_from(rows)? * 288 * 4;
        Ok((
            prefix_buffer(self.exl3_wide_router_moe[1], bytes)?,
            prefix_buffer(self.exl3_wide_router_moe[2], bytes)?,
        ))
    }

    pub fn moe_buffers_rows(
        &self,
        rows: u32,
        input: GgmlIqBuffer,
        output: GgmlIqBuffer,
    ) -> Result<Glm53SerialMoeBuffers> {
        ensure!(
            (1..=MAX_WIDE_ROWS).contains(&rows),
            "GLM MoE scratch rows must be 1..={MAX_WIDE_ROWS}"
        );
        if rows == 1 {
            return Ok(self.moe_buffers(input, output));
        }
        let rows = usize::try_from(rows)?;
        let unused = self.exl3_wide_router_moe[9];
        Ok(Glm53SerialMoeBuffers {
            input_bf16: input,
            route_ids_u32: prefix_buffer(self.exl3_wide_router_moe[3], rows * 8 * 4)?,
            route_weights_f32: prefix_buffer(self.exl3_wide_router_moe[4], rows * 8 * 4)?,
            q8_activation: unused,
            expert_gate_bf16: unused,
            expert_up_bf16: unused,
            expert_swiglu_bf16: unused,
            routed_bf16: unused,
            shared_gate_bf16: prefix_buffer(self.exl3_wide_router_moe[5], rows * 2_048 * 2)?,
            shared_up_bf16: prefix_buffer(self.exl3_wide_router_moe[6], rows * 2_048 * 2)?,
            shared_swiglu_bf16: prefix_buffer(self.exl3_wide_router_moe[7], rows * 2_048 * 2)?,
            shared_bf16: prefix_buffer(self.exl3_wide_router_moe[8], rows * 4_096 * 2)?,
            output_bf16: output,
        })
    }

    /// One reusable EXL3 workspace. The target walk is strictly ordered.
    pub fn exl3_projection_scratch(&self) -> Glm53Exl3ProjectionScratch {
        let buffer = |buffer: GgmlIqBuffer| Glm53Exl3Buffer {
            ptr: buffer.ptr,
            bytes: buffer.bytes,
        };
        Glm53Exl3ProjectionScratch {
            input_f16: buffer(self.exl3[0]),
            output_f16: buffer(self.exl3[1]),
            locks_i32: buffer(self.exl3[2]),
            input_hadamard_f16: buffer(self.exl3[3]),
            reconstruct_f16: Glm53Exl3Buffer {
                ptr: DevicePtr::NULL,
                bytes: 0,
            },
            reconstruct_f16_b: Glm53Exl3Buffer {
                ptr: DevicePtr::NULL,
                bytes: 0,
            },
        }
    }

    /// Prompt-scope F32 staging (see `EXL3_RECONSTRUCT_SCRATCH`); transient
    /// within one launch sequence, never carried across events.
    pub fn prompt_f32_scratch(&self) -> GgmlIqBuffer {
        self.exl3_reconstruct[1]
    }

    /// Select the immutable T1 layout or the appended max-M2048 layout.
    pub fn exl3_projection_scratch_rows(&self, rows: u32) -> Result<Glm53Exl3ProjectionScratch> {
        ensure!(
            (1..=MAX_WIDE_ROWS).contains(&rows),
            "GLM EXL3 scratch rows must be 1..={MAX_WIDE_ROWS}"
        );
        if rows == 1 {
            return Ok(self.exl3_projection_scratch());
        }
        let buffer = |buffer: GgmlIqBuffer| Glm53Exl3Buffer {
            ptr: buffer.ptr,
            bytes: buffer.bytes,
        };
        Ok(Glm53Exl3ProjectionScratch {
            input_f16: buffer(self.exl3_wide[0]),
            output_f16: buffer(self.exl3_wide[1]),
            locks_i32: buffer(self.exl3_wide[2]),
            input_hadamard_f16: buffer(self.exl3_wide[3]),
            reconstruct_f16: buffer(self.exl3_reconstruct[0]),
            reconstruct_f16_b: buffer(self.exl3_reconstruct[2]),
        })
    }

    pub fn exl3_moe_scratch(&self) -> Glm53Exl3MoeScratch {
        let buffer = |buffer: GgmlIqBuffer| Glm53Exl3Buffer {
            ptr: buffer.ptr,
            bytes: buffer.bytes,
        };
        Glm53Exl3MoeScratch {
            expert_count_i64: buffer(self.exl3_moe[0]),
            token_sorted_i64: buffer(self.exl3_moe[1]),
            weight_sorted_f16: buffer(self.exl3_moe[2]),
            temp_state_g_f16: buffer(self.exl3_moe[3]),
            temp_state_u_f16: buffer(self.exl3_moe[4]),
            temp_intermediate_g_f16: buffer(self.exl3_moe[5]),
            temp_intermediate_u_f16: buffer(self.exl3_moe[6]),
            output_f32: buffer(self.exl3_moe[7]),
            route_private_f32: buffer(self.exl3_moe_staged[5]),
            route_status_u32: buffer(self.exl3_moe[8]),
            locks_i32: buffer(self.exl3_moe[9]),
            pair_expert_u32: buffer(self.exl3_moe_staged[0]),
            chunk_expert_u32: buffer(self.exl3_moe_staged[1]),
            chunk_start_u32: buffer(self.exl3_moe_staged[2]),
            chunk_rows_u32: buffer(self.exl3_moe_staged[3]),
            chunk_count_u32: buffer(self.exl3_moe_staged[4]),
        }
    }

    /// The dense-FFN working buffers.
    pub fn dense_buffers(&self) -> Glm53DenseFfnBuffers {
        Glm53DenseFfnBuffers {
            q8_activation: self.dense_q8_activation,
            gate_bf16: self.dense_gate_bf16,
            up_bf16: self.dense_up_bf16,
            swiglu_bf16: self.dense_swiglu_bf16,
        }
    }

    pub fn dense_buffers_rows(&self, rows: u32) -> Result<Glm53DenseFfnBuffers> {
        ensure!(
            (1..=MAX_WIDE_ROWS).contains(&rows),
            "GLM dense scratch rows must be 1..={MAX_WIDE_ROWS}"
        );
        if rows == 1 {
            return Ok(self.dense_buffers());
        }
        Ok(Glm53DenseFfnBuffers {
            // EXL3 does not consume Q8 scratch, but retain a non-null sentinel
            // so the common typed buffer contract remains fail-closed.
            q8_activation: self.exl3_wide_router_moe[9],
            gate_bf16: prefix_buffer(self.exl3_wide_dense[0], rows as usize * 12_288 * 2)?,
            up_bf16: prefix_buffer(self.exl3_wide_dense[1], rows as usize * 12_288 * 2)?,
            swiglu_bf16: prefix_buffer(self.exl3_wide_dense[2], rows as usize * 12_288 * 2)?,
        })
    }

    /// The KDA attention working buffers.
    pub fn kda_buffers(&self) -> Glm53KdaScratchBuffers {
        Glm53KdaScratchBuffers {
            combined_qkv_bf16: GgmlIqBuffer {
                ptr: self.kda[0].ptr,
                bytes: self.kda[0].bytes + self.kda[1].bytes + self.kda[2].bytes,
            },
            q_proj_bf16: self.kda[0],
            k_proj_bf16: self.kda[1],
            v_proj_bf16: self.kda[2],
            q_conv_bf16: self.kda[3],
            k_conv_bf16: self.kda[4],
            v_conv_bf16: self.kda[5],
            f_a_bf16: self.kda[6],
            f_b_bf16: self.kda[7],
            log_decay_f32: self.kda[8],
            beta_proj_bf16: self.kda[9],
            beta_bf16: self.kda[10],
            g_a_bf16: self.kda[11],
            g_b_bf16: self.kda[12],
            recurrent_out_bf16: self.kda[13],
            gated_bf16: self.kda[14],
            q8_activation: self.kda[15],
        }
    }

    pub fn kda_buffers_rows(&self, rows: u32) -> Result<Glm53KdaScratchBuffers> {
        ensure!(
            (1..=MAX_WIDE_ROWS).contains(&rows),
            "GLM KDA scratch rows must be 1..={MAX_WIDE_ROWS}"
        );
        if rows == 1 {
            return Ok(self.kda_buffers());
        }
        let rows = usize::try_from(rows)?;
        let b = |index, per_row| prefix_buffer(self.exl3_wide_kda[index], rows * per_row);
        Ok(Glm53KdaScratchBuffers {
            combined_qkv_bf16: b(16, 3 * 16_384)?,
            q_proj_bf16: b(0, 16_384)?,
            k_proj_bf16: b(1, 16_384)?,
            v_proj_bf16: b(2, 16_384)?,
            q_conv_bf16: b(3, 16_384)?,
            k_conv_bf16: b(4, 16_384)?,
            v_conv_bf16: b(5, 16_384)?,
            f_a_bf16: b(6, 256)?,
            f_b_bf16: b(7, 16_384)?,
            log_decay_f32: b(8, 32_768)?,
            beta_proj_bf16: b(9, 128)?,
            beta_bf16: b(10, 128)?,
            g_a_bf16: b(11, 256)?,
            g_b_bf16: b(12, 16_384)?,
            recurrent_out_bf16: b(13, 16_384)?,
            gated_bf16: b(14, 16_384)?,
            q8_activation: self.exl3_wide_kda[15],
        })
    }

    /// The DSA attention working buffers.
    pub fn dsa_buffers(&self) -> Glm53DsaScratchBuffers {
        Glm53DsaScratchBuffers {
            qr_bf16: self.dsa[0],
            qr_norm_bf16: self.dsa[1],
            q_b_bf16: self.dsa[2],
            absorbed_q_bf16: self.dsa[3],
            kv_cmpr_bf16: self.dsa[4],
            kv_cmpr_norm_bf16: self.dsa[5],
            index_k_bf16: self.dsa[6],
            index_k_norm_bf16: self.dsa[7],
            index_g_bf16: self.dsa[8],
            index_q_bf16: self.dsa[9],
            head_weights_bf16: self.dsa[10],
            scores_f32: self.dsa[11],
            selected_indices_i32: self.dsa[12],
            weighted_latent_bf16: self.dsa[13],
            unabsorbed_bf16: self.dsa[14],
            pool_keys_bf16: self.dsa[15],
            pool_validity_u8: self.dsa[16],
            tail_keys_bf16: self.dsa[17],
            tail_gates_bf16: self.dsa[18],
            q8_activation: self.dsa[19],
            head_major_large_bf16: self.dsa[13],
            head_major_small_bf16: self.dsa[14],
            sequence_length_u32: GgmlIqBuffer {
                ptr: DevicePtr::NULL,
                bytes: 0,
            },
            query_positions_u32: GgmlIqBuffer {
                ptr: DevicePtr::NULL,
                bytes: 0,
            },
            query_validity_u8: GgmlIqBuffer {
                ptr: DevicePtr::NULL,
                bytes: 0,
            },
            tail_validity_u8: GgmlIqBuffer {
                ptr: DevicePtr::NULL,
                bytes: 0,
            },
        }
    }

    pub fn dsa_buffers_rows(&self, rows: u32) -> Result<Glm53DsaScratchBuffers> {
        ensure!(
            (1..=MAX_WIDE_ROWS).contains(&rows),
            "GLM DSA scratch rows must be 1..={MAX_WIDE_ROWS}"
        );
        if rows == 1 {
            return Ok(self.dsa_buffers());
        }
        let rows = usize::try_from(rows)?;
        let b = |index, per_row| prefix_buffer(self.exl3_wide_dsa[index], rows * per_row);
        Ok(Glm53DsaScratchBuffers {
            qr_bf16: b(0, 3_072)?,
            qr_norm_bf16: b(1, 3_072)?,
            q_b_bf16: b(2, 32_768)?,
            absorbed_q_bf16: b(3, 65_536)?,
            kv_cmpr_bf16: b(4, 1_024)?,
            kv_cmpr_norm_bf16: b(5, 1_024)?,
            index_k_bf16: b(6, 256)?,
            index_k_norm_bf16: b(7, 256)?,
            index_g_bf16: b(8, 256)?,
            index_q_bf16: b(9, 8_192)?,
            head_weights_bf16: b(10, 64)?,
            scores_f32: b(11, 1_048_576)?,
            selected_indices_i32: b(12, 8_204)?,
            weighted_latent_bf16: b(13, 65_536)?,
            unabsorbed_bf16: b(14, 32_768)?,
            pool_keys_bf16: self.exl3_wide_dsa[15],
            pool_validity_u8: self.exl3_wide_dsa[16],
            tail_keys_bf16: self.exl3_wide_dsa[17],
            tail_gates_bf16: self.exl3_wide_dsa[18],
            q8_activation: self.exl3_wide_dsa[19],
            head_major_large_bf16: b(20, 65_536)?,
            head_major_small_bf16: b(21, 32_768)?,
            sequence_length_u32: self.exl3_wide_dsa[22],
            query_positions_u32: b(23, 4)?,
            query_validity_u8: b(24, 1)?,
            tail_validity_u8: self.exl3_wide_dsa[25],
        })
    }

    /// The allocation size a walk needs. Callers allocate exactly this.
    pub fn required_bytes() -> u64 {
        Self::used_bytes()
    }

    pub fn exl3_route_policy(&self) -> Glm53Exl3RoutePolicy {
        self.exl3_route_policy
    }

    /// Total scratch actually consumed, for residency accounting and tests.
    pub fn used_bytes() -> u64 {
        Self::required_bytes_with_route_policy(Glm53Exl3RoutePolicy::legacy())
    }

    pub fn required_bytes_with_route_policy(policy: Glm53Exl3RoutePolicy) -> u64 {
        let mut cursor = 0u64;
        for (_, bytes) in layout(policy) {
            cursor = align_up(cursor) + bytes;
        }
        cursor
    }
}

fn prefix_buffer(buffer: GgmlIqBuffer, bytes: usize) -> Result<GgmlIqBuffer> {
    ensure!(
        buffer.ptr != DevicePtr::NULL && buffer.bytes >= bytes,
        "GLM wide scratch buffer is null or too small"
    );
    buffer
        .ptr
        .0
        .checked_add(u64::try_from(bytes)?)
        .context("GLM wide scratch address overflow")?;
    Ok(GgmlIqBuffer {
        ptr: buffer.ptr,
        bytes,
    })
}

#[cfg(test)]
#[path = "walk_scratch_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "walk_scratch_route_policy_tests.rs"]
mod route_policy_tests;
