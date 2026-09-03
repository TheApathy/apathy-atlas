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
//! deliberately NOT time-multiplexed by liveness: the whole set is under 2 MiB
//! against a 107 GiB resident model, and disjointness that holds by
//! construction is worth far more than the bytes.
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
use crate::layers::ops::GgmlIqBuffer;

use super::arena::{GLM53_T1_TRANSIENT_BYTES, GLM53_T1_WORKSPACE_BYTES};

const ALIGNMENT: u64 = 256;

/// One DSA layer's raw index tail: TAIL_CAPACITY rows of INDEX_DIM bf16.
const DSA_TAIL_BYTES: u64 =
    spark_runtime::kv_cache::GLM53_DSA_TAIL_CAPACITY as u64 * 128 * 2;

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

// Documents the invariant and fails the build loudly if anyone reintroduces a
// literal, or grows a table without noticing what it shifts.
const _: () = {
    assert!(KDA_OFFSET == 16);
    assert!(DSA_OFFSET == 32);
    assert!(ROUTER_DIAG_OFFSET == 52);
    assert!(ROUTER_DIAG_OFFSET + ROUTER_DIAG_SCRATCH.len() == 54);
    assert!(GROUPED_MOE_OFFSET == 54);
    assert!(GROUPED_MOE_OFFSET + GROUPED_MOE_SCRATCH.len() == 58);
};

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

const ROUTER_DIAG_SCRATCH: [(&str, u64); 2] = [
    ("router_probs_f32", 1_152),
    ("router_biased_f32", 1_152),
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
}

/// The KDA attention working buffers for one layer.
#[derive(Debug, Clone, Copy)]
pub struct Glm53KdaScratchBuffers {
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

impl Glm53WalkScratch {
    /// Bind scratch from a dedicated allocation.
    ///
    /// `base` must be a 256-byte-aligned device allocation of at least
    /// [`Glm53WalkScratch::required_bytes`] bytes, distinct from the arena.
    pub fn bind(base: DevicePtr, bytes: u64) -> Result<Self> {
        ensure!(!base.is_null(), "GLM walk scratch base is NULL");
        let required = Self::required_bytes();
        ensure!(
            bytes >= required,
            "GLM walk scratch allocation is {bytes} bytes, need {required}"
        );
        let start = base.0;
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
        for (name, bytes) in MOE_SCRATCH
            .iter()
            .chain(DENSE_SCRATCH.iter())
            .chain(KDA_SCRATCH.iter())
            .chain(DSA_SCRATCH.iter())
            .chain(ROUTER_DIAG_SCRATCH.iter())
            .chain(GROUPED_MOE_SCRATCH.iter())
            .copied()
        {
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

    /// The router's scratch: expert logits plus the decoded route.
    pub fn router_scratch(&self) -> (GgmlIqBuffer, GgmlIqBuffer, GgmlIqBuffer) {
        (
            self.router_logits_f32,
            self.route_ids_u32,
            self.route_weights_f32,
        )
    }

    /// The router's full per-expert score vectors: `(pre_bias, post_bias)`.
    ///
    /// GLM ranks on the post-bias score but emits the pre-bias sigmoid as the
    /// weight, so both are exposed by name. Inferring one from the other is
    /// what produced a phantom "8th kept expert" that was not in the top-8.
    pub fn router_scores(&self) -> (GgmlIqBuffer, GgmlIqBuffer) {
        (self.router_probs_f32, self.router_biased_f32)
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

    /// The KDA attention working buffers.
    pub fn kda_buffers(&self) -> Glm53KdaScratchBuffers {
        Glm53KdaScratchBuffers {
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
        }
    }

    /// The allocation size a walk needs. Callers allocate exactly this.
    pub fn required_bytes() -> u64 {
        Self::used_bytes()
    }

    /// Total scratch actually consumed, for residency accounting and tests.
    pub fn used_bytes() -> u64 {
        let mut cursor = 0u64;
        for (_, bytes) in MOE_SCRATCH
            .iter()
            .chain(DENSE_SCRATCH.iter())
            .chain(KDA_SCRATCH.iter())
            .chain(DSA_SCRATCH.iter())
            .chain(ROUTER_DIAG_SCRATCH.iter())
            .chain(GROUPED_MOE_SCRATCH.iter())
        {
            cursor = align_up(cursor) + bytes;
        }
        cursor
    }
}

#[cfg(test)]
#[path = "walk_scratch_tests.rs"]
mod tests;
