// SPDX-License-Identifier: AGPL-3.0-only

//! The DeepSeek-V4.1 MoE expert FFN: **CB3 -> bf16 -> cuBLASLt**.
//!
//! This is the other half of the loader's old hard stop — "the MoE GEMM that consumes the
//! decoded e2m1 tiles". It is the harness at
//! `kernels/gb10/deepseek-v4.1/cb3/harness/cb3_gemm.cu` wired into the engine: the same
//! reconstruct kernel, then `cublaslt::bf16_gemm_act_weight_t`, which builds exactly the
//! `y[M,N] = x[M,K] @ W[N,K]^T` plan the harness hand-rolled (bf16 in, bf16 out, fp32
//! compute, weight packed `[N,K]`). The harness measured rel_l2 2.58e-07 against the
//! Python reference with a wrong-K-order control at 1.41.
//!
//! ## Reconstruct-then-GEMM, not fused
//! Chosen because it is what won on this box twice (Flash-Next 227 -> 1068 tok/s, GLM EXL3
//! 516 -> 744) and cuBLASLt measured 3.5-4.8x our own kernels at these shapes. There is
//! also NO accuracy argument for fusing: a CB3 value is an e2m1 grid point times a
//! power-of-two scale, exactly representable in bf16, so the format contributes zero error
//! and fusion could only ever match this, never beat it numerically.
//!
//! ## The cost this DOES pay, stated plainly
//! One expert reconstructs to 3 x 23.6 MB of bf16 scratch (`w1`/`w3` are `[2304, 5120]`,
//! `w2` is `[5120, 2304]`). At decode M=1 that is ~70 MB of writes plus ~70 MB of reads
//! per expert per token, against ~14.5 MB of packed bytes read — a ~10x write
//! amplification that a fused decode+MMA kernel would not pay. It is the bandwidth
//! argument the format note in `CB3_FORMAT.md` says is the only one available. This path
//! is therefore the CORRECTNESS baseline and the thing a fused kernel must be measured
//! against; it is **not** claimed to be the fast one, and nothing here has been timed.

use anyhow::{Context, Result, ensure};

use atlas_core::config::{Cb3Tensor, ModelConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::cb3_arena::{CB3_RECONSTRUCT_FN, CB3_RECONSTRUCT_MODULE, Cb3LayerResidency};

/// Threads per block for the reconstruct launch. 256 is the shape the harness measured
/// with; changing it changes nothing numerically (the kernel is elementwise over N*K) but
/// the measured figure is quoted for this one.
const RECONSTRUCT_BLOCK: u32 = 256;

/// The three CB3 planes that make up one packed weight matrix, plus its bf16 extent.
#[derive(Clone, Copy, Debug)]
pub struct Cb3Matrix {
    pub lo: Cb3Tensor,
    pub hi: Cb3Tensor,
    pub cb: Cb3Tensor,
    pub scale: Cb3Tensor,
    /// Output rows. The GEMM's N.
    pub rows: usize,
    /// Reduction width. The GEMM's K.
    pub cols: usize,
}

/// Gate (`w1`), up (`w3`) and down (`w2`) for a V4.1 expert.
///
/// Shapes are read off [`Cb3Tensor::shape`] rather than restated, so a pack whose geometry
/// changes fails at construction instead of silently reading the wrong extent.
pub fn expert_matrices(config: &ModelConfig) -> Result<[Cb3Matrix; 3]> {
    let inter = config.moe_intermediate_size;
    let hidden = config.hidden_size;

    // `shape()` is (rows, bytes_per_row) of the LO plane's tensor; lo packs 4 weights per
    // byte, so cols = bytes_per_row * 4. Deriving cols this way rather than asserting
    // `hidden` keeps the pack's geometry authoritative.
    let matrix = |lo: Cb3Tensor,
                  hi: Cb3Tensor,
                  cb: Cb3Tensor,
                  scale: Cb3Tensor,
                  expect_rows: usize,
                  expect_cols: usize|
     -> Result<Cb3Matrix> {
        let (rows, lo_bytes) = lo.shape();
        let cols = lo_bytes * 4;
        ensure!(
            rows == expect_rows && cols == expect_cols,
            "CB3 {} is [{rows}, {cols}], but the config implies [{expect_rows}, {expect_cols}]",
            lo.name()
        );
        // The other three planes must agree with that K, or the kernel indexes past a row.
        ensure!(hi.shape() == (rows, cols / 8), "CB3 {} disagrees on K", hi.name());
        ensure!(cb.shape() == (rows, 8), "CB3 {} is not an 8-entry codebook", cb.name());
        ensure!(
            scale.shape() == (rows, cols / 32),
            "CB3 {} disagrees on the 32-wide scale grouping",
            scale.name()
        );
        Ok(Cb3Matrix { lo, hi, cb, scale, rows, cols })
    };

    Ok([
        // gate: [moe_intermediate, hidden]
        matrix(
            Cb3Tensor::W1Lo,
            Cb3Tensor::W1Hi,
            Cb3Tensor::W1Cb,
            Cb3Tensor::S1,
            inter,
            hidden,
        )?,
        // up: [moe_intermediate, hidden]
        matrix(
            Cb3Tensor::W3Lo,
            Cb3Tensor::W3Hi,
            Cb3Tensor::W3Cb,
            Cb3Tensor::S3,
            inter,
            hidden,
        )?,
        // down: [hidden, moe_intermediate]
        matrix(
            Cb3Tensor::W2Lo,
            Cb3Tensor::W2Hi,
            Cb3Tensor::W2Cb,
            Cb3Tensor::S2,
            hidden,
            inter,
        )?,
    ])
}

/// Cached kernel handle for the reconstruct, looked up once per layer.
#[derive(Clone, Copy, Debug)]
pub struct Cb3Reconstruct {
    kernel: KernelHandle,
}

impl Cb3Reconstruct {
    pub fn new(gpu: &dyn GpuBackend) -> Result<Self> {
        let kernel = gpu
            .kernel(CB3_RECONSTRUCT_MODULE, CB3_RECONSTRUCT_FN)
            .with_context(|| {
                format!(
                    "CB3 reconstruct kernel {CB3_RECONSTRUCT_MODULE}::{CB3_RECONSTRUCT_FN} is \
                     not in the compiled PTX. It is built only for the \
                     (gb10, deepseek-v4.1, cb3) target — check ATLAS_TARGET_MODEL / \
                     ATLAS_TARGET_QUANT, and that the 'compiled N kernels' line names it."
                )
            })?;
        Ok(Self { kernel })
    }

    /// Reconstruct one expert's matrix into `out`, which must hold `rows * cols` bf16.
    ///
    /// Everything here is per-expert: `slot` addresses the arena's resident prefix, NOT the
    /// 384-space routed id. Callers resolve that through `Cb3ExpertArena::slot_of`, which
    /// is a hard error for a non-resident id — never a modulo, never a clamp.
    pub fn run(
        &self,
        residency: &Cb3LayerResidency,
        matrix: Cb3Matrix,
        slot: usize,
        packed_keep: usize,
        out: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let lo = residency.plane_ptr(matrix.lo, slot, packed_keep)?;
        let hi = residency.plane_ptr(matrix.hi, slot, packed_keep)?;
        let cb = residency.plane_ptr(matrix.cb, slot, packed_keep)?;
        let scale = residency.plane_ptr(matrix.scale, slot, packed_keep)?;

        let total = matrix
            .rows
            .checked_mul(matrix.cols)
            .context("CB3 reconstruct extent overflow")?;
        let grid = total.div_ceil(RECONSTRUCT_BLOCK as usize) as u32;

        let rows = matrix.rows as i32;
        let cols = matrix.cols as i32;
        let mut lo_p = lo.0;
        let mut hi_p = hi.0;
        let mut cb_p = cb.0;
        let mut sc_p = scale.0;
        let mut out_p = out.0;
        let mut rows_p = rows;
        let mut cols_p = cols;
        let mut params: [*mut std::ffi::c_void; 7] = [
            &mut lo_p as *mut u64 as *mut _,
            &mut hi_p as *mut u64 as *mut _,
            &mut cb_p as *mut u64 as *mut _,
            &mut sc_p as *mut u64 as *mut _,
            &mut out_p as *mut u64 as *mut _,
            &mut rows_p as *mut i32 as *mut _,
            &mut cols_p as *mut i32 as *mut _,
        ];
        gpu.launch(
            self.kernel,
            [grid, 1, 1],
            [RECONSTRUCT_BLOCK, 1, 1],
            0,
            stream,
            &mut params,
        )
    }
}

/// `y[m, n] = x[m, k] @ W[n, k]^T`, bf16 in and out, fp32 accumulate.
///
/// A thin named wrapper so the one place the GEMM contract is stated is here, next to the
/// note that `bf16_gemm_act_weight_t` wants the weight packed `[N, K]` — which is exactly
/// how CB3 reconstructs it, with no transpose anywhere on this path.
pub fn gemm_weight_t(
    act: DevicePtr,
    weight_bf16: DevicePtr,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    spark_runtime::cublaslt::bf16_gemm_act_weight_t(
        act.0,
        weight_bf16.0,
        out.0,
        m as u32,
        n as u32,
        k as u32,
        stream,
    )
}

/// [`gemm_weight_t`] with an fp32 output — the accumulator is NOT rounded to bf16.
///
/// The engine keeps gate/up in fp32 through the SwiGLU and keeps each expert's down
/// projection in fp32 until the six picks are summed. Rounding either to bf16 here adds
/// error the reference does not have; see [`COMBINE_MODULE`].
pub fn gemm_weight_t_f32out(
    act: DevicePtr,
    weight_bf16: DevicePtr,
    out_f32: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    use spark_runtime::cublaslt::{GemmDtype, gemm_act_weight_t_typed};
    gemm_act_weight_t_typed(
        act.0,
        k as u32,
        weight_bf16.0,
        out_f32.0,
        n as u32,
        m as u32,
        n as u32,
        k as u32,
        GemmDtype::Bf16,
        GemmDtype::F32,
        stream,
    )
}

/// Module of the V4.1 routed epilogues (`kernels/gb10/deepseek-v4.1/cb3/dsv41_moe_combine.cu`),
/// which round to bf16 at the engine's two points and nowhere else.
pub const COMBINE_MODULE: &str = "dsv41_moe_combine";
/// `h = bf16(silu(min(g,L)) * clamp(u,±L) * w_row)` from fp32 gate/up.
pub const SWIGLU_WEIGHTED_FN: &str = "dsv41_swiglu_weighted";
/// `out = bf16(sum_k expert_out[token_to_perm[t,k]])` over fp32, already-weighted rows.
pub const UNPERMUTE_SUM_FN: &str = "dsv41_unpermute_sum_f32";
/// Router softplus/sqrt + bias (text or vision per row) + residency mask + top-k + weights.
pub const ROUTE_TOPK_FN: &str = "dsv41_route_topk";

/// Module of the fused grouped CB3 GEMM (`cb3/cb3_moe_gemm.cu`): weights decoded in shared
/// memory, never written to DRAM.
pub const FUSED_GEMM_MODULE: &str = "cb3_moe_gemm";
/// Gate + up + SwiGLU x route weight -> bf16 h.
pub const FUSED_GATE_UP_FN: &str = "cb3_moe_gate_up";
/// Down projection -> fp32 rows.
pub const FUSED_DOWN_FN: &str = "cb3_moe_down";
/// Rows per M tile of the fused kernels (`BM` in the .cu). N must divide by
/// [`FUSED_TILE_N`], K by 64.
pub const FUSED_TILE_M: usize = 128;
pub const FUSED_TILE_N: usize = 64;
/// Mainloop shared-memory stages of the fused CB3 kernels (`CB3_MOE_STAGES` in the .cu).
pub const FUSED_STAGES: u32 = 1;
/// Dynamic shared memory per launch: a stage is 32 KB for gate/up, 24 KB for down.
pub const FUSED_GATE_UP_SMEM: u32 = FUSED_STAGES * 32_768;
pub const FUSED_DOWN_SMEM: u32 = FUSED_STAGES * 24_576;


/// Bytes of bf16 scratch one expert's reconstruct needs, for all three matrices at once.
pub fn scratch_bytes(config: &ModelConfig) -> Result<usize> {
    let matrices = expert_matrices(config)?;
    matrices
        .iter()
        .try_fold(0usize, |total, matrix| {
            matrix
                .rows
                .checked_mul(matrix.cols)
                .and_then(|elements| elements.checked_mul(2))
                .and_then(|bytes| total.checked_add(bytes))
                .context("CB3 scratch extent overflow")
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v41_config() -> ModelConfig {
        let raw = std::fs::read_to_string(
            "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K/config.json",
        )
        .expect("config present");
        atlas_core::config::parse_config(&raw).expect("V4.1 config parses")
    }

    /// The pack's geometry and the config must agree, checked against the real checkpoint.
    #[test]
    fn expert_matrix_shapes_match_the_pack_and_the_config() {
        if !std::path::Path::new(
            "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K/config.json",
        )
        .exists()
        {
            eprintln!("skipping: checkpoint not present");
            return;
        }
        let config = v41_config();
        assert_eq!(config.hidden_size, 5120);
        assert_eq!(config.moe_intermediate_size, 2304);

        let [gate, up, down] = expert_matrices(&config).expect("shapes agree");
        // gate/up reduce over hidden and emit moe_intermediate; down is the transpose pair.
        assert_eq!((gate.rows, gate.cols), (2304, 5120));
        assert_eq!((up.rows, up.cols), (2304, 5120));
        assert_eq!((down.rows, down.cols), (5120, 2304));

        // ~70.8 MB of bf16 scratch per expert. Stated so the write amplification in the
        // module note is a number in the test, not only in prose.
        let scratch = scratch_bytes(&config).unwrap();
        assert_eq!(scratch, (2304 * 5120 + 2304 * 5120 + 5120 * 2304) * 2);
        assert!((70e6..72e6).contains(&(scratch as f64)));
    }

    /// NEGATIVE CONTROL for the shape check: a config that disagrees with the pack must be
    /// REFUSED. Without this, `expert_matrices` is a gate on a structurally-guaranteed
    /// input and proves nothing.
    #[test]
    fn a_config_that_disagrees_with_the_pack_is_refused() {
        if !std::path::Path::new(
            "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K/config.json",
        )
        .exists()
        {
            eprintln!("skipping: checkpoint not present");
            return;
        }
        let mut config = v41_config();
        // V4-Flash-0731's moe_intermediate_size. The CB3 pack is 2304-wide, so this must
        // fail rather than index 2048 rows of a 2304-row plane.
        config.moe_intermediate_size = 2048;
        let err = expert_matrices(&config)
            .expect_err("a 2048-wide config must not be accepted against a 2304-wide pack")
            .to_string();
        assert!(err.contains("2304"), "the refusal must name the real extent: {err}");

        // And the hidden size, independently.
        let mut config = v41_config();
        config.hidden_size = 4096;
        assert!(expert_matrices(&config).is_err());
    }
}

// =====================================================================================
// COMBINE — grouping tokens by expert, then the weighted sum
// =====================================================================================

/// Tokens routed to one expert, and their routing weights.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertGroup {
    /// 384-space routed id.
    pub expert_id: u32,
    /// Resident slot in the arena, resolved once.
    pub slot: usize,
    /// Indices of the tokens routed here.
    pub tokens: Vec<u32>,
    /// Their weights, parallel to `tokens`.
    pub weights: Vec<f32>,
}

/// Group a routing decision by EXPERT rather than by token.
///
/// ## Why this shape, and what it costs
/// The obvious loop is per (token, expert) pair, which reconstructs three CB3 matrices —
/// ~70.8 MB of bf16 — for every pair. Grouping means each expert is reconstructed **once
/// per forward** and applied to all its tokens as one batched GEMM, so the reconstruct
/// amortises over the group.
///
/// The amortisation is everything, and it is wildly different between the two regimes:
///
/// ```text
///   prefill, 2048 tokens x 6 / 124 experts  ~= 99 tokens per expert -> ~99x amortised
///   decode,  1 token x 6 experts            ==  1 token per expert  -> NONE
/// ```
///
/// So at decode this path reconstructs ~425 MB of bf16 to multiply one 5120-wide vector.
/// That is the bandwidth argument `CB3_FORMAT.md` says is the only argument for a fused
/// decode+MMA kernel, and it is where one would pay off. Stated here rather than
/// discovered later: grouping fixes prefill and does nothing for decode.
///
/// Returned groups are sorted by `expert_id` so the reconstruct order is deterministic —
/// fp32 accumulation is not associative, and a nondeterministic expert order would make
/// run-to-run output differ for no reason.
pub fn group_by_expert(
    indices: &[i64],
    weights: &[f32],
    num_tokens: usize,
    k: usize,
    slot_of: impl Fn(u32) -> Result<i32>,
) -> Result<Vec<ExpertGroup>> {
    ensure!(
        indices.len() == num_tokens * k && weights.len() == indices.len(),
        "routing is {} indices / {} weights, expected {num_tokens} x {k}",
        indices.len(),
        weights.len()
    );

    let mut groups: std::collections::BTreeMap<u32, ExpertGroup> = std::collections::BTreeMap::new();
    for token in 0..num_tokens {
        for pick in 0..k {
            let flat = token * k + pick;
            let expert_id = u32::try_from(indices[flat])
                .with_context(|| format!("negative expert id {} at token {token}", indices[flat]))?;
            let slot = slot_of(expert_id)?;
            ensure!(
                slot >= 0,
                "token {token} routed to expert {expert_id}, which is NOT resident. Routing \
                 must apply the arena's residency mask BEFORE top-k; reaching here means the \
                 mask was skipped, and there is no correct fallback — a modulo or clamp would \
                 silently substitute a different, valid-looking expert."
            );
            let entry = groups.entry(expert_id).or_insert_with(|| ExpertGroup {
                expert_id,
                slot: slot as usize,
                tokens: Vec::new(),
                weights: Vec::new(),
            });
            entry.tokens.push(token as u32);
            entry.weights.push(weights[flat]);
        }
    }
    Ok(groups.into_values().collect())
}

/// Reconstruct-cost accounting for one forward, in bytes of bf16 written.
///
/// Exposed so the decode/prefill asymmetry above is a number a caller can log or assert
/// on, not just a comment.
pub fn reconstruct_bytes_for(groups: &[ExpertGroup], config: &ModelConfig) -> Result<usize> {
    let per_expert = scratch_bytes(config)?;
    per_expert
        .checked_mul(groups.len())
        .context("reconstruct byte accounting overflow")
}

#[cfg(test)]
mod combine_tests {
    use super::*;

    fn all_resident(id: u32) -> Result<i32> {
        Ok(id as i32)
    }

    /// Grouping must invert the routing exactly: every (token, expert) pair lands once,
    /// with its own weight.
    #[test]
    fn grouping_preserves_every_token_expert_pair() {
        // 3 tokens, k=2. Token 0 -> {5, 9}, token 1 -> {5, 7}, token 2 -> {9, 5}.
        let indices: Vec<i64> = vec![5, 9, 5, 7, 9, 5];
        let weights: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let groups = group_by_expert(&indices, &weights, 3, 2, all_resident).unwrap();

        // Sorted by expert id, so the reconstruct order is deterministic.
        assert_eq!(
            groups.iter().map(|g| g.expert_id).collect::<Vec<_>>(),
            vec![5, 7, 9]
        );
        let five = &groups[0];
        assert_eq!(five.tokens, vec![0, 1, 2]);
        assert_eq!(five.weights, vec![0.1, 0.3, 0.6]);
        assert_eq!(groups[1].tokens, vec![1]);
        assert_eq!(groups[2].tokens, vec![0, 2]);
        assert_eq!(groups[2].weights, vec![0.2, 0.5]);

        // Every pair is accounted for exactly once — no drops, no duplicates.
        let placed: usize = groups.iter().map(|g| g.tokens.len()).sum();
        assert_eq!(placed, indices.len());
    }

    /// A non-resident expert must be a HARD ERROR, never a substitution.
    ///
    /// This is the last line of defence behind the routing mask. If it ever silently
    /// clamped, the model would run on a different-but-valid expert and produce fluent,
    /// wrong output — the failure this whole port is organised against.
    #[test]
    fn a_non_resident_expert_is_refused_not_substituted() {
        let indices: Vec<i64> = vec![3, 11];
        let weights: Vec<f32> = vec![0.5, 0.5];
        // Expert 11 is not resident.
        let slot_of = |id: u32| Ok(if id == 11 { -1 } else { id as i32 });
        let err = group_by_expert(&indices, &weights, 1, 2, slot_of)
            .expect_err("a non-resident expert must not be grouped")
            .to_string();
        assert!(err.contains("NOT resident"), "{err}");
        assert!(err.contains("no correct fallback"), "{err}");

        // NEGATIVE CONTROL: the same call with 11 resident must SUCCEED, so the refusal
        // above is about residency and not about some unrelated shape error.
        let groups = group_by_expert(&indices, &weights, 1, 2, all_resident).unwrap();
        assert_eq!(groups.len(), 2);
    }

    /// The prefill/decode amortisation asymmetry, as numbers rather than prose.
    #[test]
    fn grouping_amortises_at_prefill_and_not_at_decode() {
        let Ok(raw) = std::fs::read_to_string(
            "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K/config.json",
        ) else {
            eprintln!("skipping: checkpoint not present");
            return;
        };
        let config = atlas_core::config::parse_config(&raw).unwrap();
        let k = 6usize;

        // DECODE: 1 token, 6 distinct experts -> 6 groups of 1. No amortisation.
        let decode_idx: Vec<i64> = (0..k as i64).collect();
        let decode_w = vec![1.0f32 / k as f32; k];
        let decode = group_by_expert(&decode_idx, &decode_w, 1, k, all_resident).unwrap();
        assert_eq!(decode.len(), 6);
        assert!(decode.iter().all(|g| g.tokens.len() == 1));
        let decode_bytes = reconstruct_bytes_for(&decode, &config).unwrap();
        // ~425 MB of bf16 reconstructed to multiply ONE 5120-wide vector.
        assert!(
            (420e6..430e6).contains(&(decode_bytes as f64)),
            "decode reconstructs {:.0} MB, expected ~425",
            decode_bytes as f64 / 1e6
        );

        // PREFILL: 256 tokens over 124 experts. Groups are bounded by the expert count,
        // so the reconstruct cost stops growing with tokens — that IS the amortisation.
        let mut prefill_idx: Vec<i64> = Vec::new();
        for token in 0..256usize {
            for pick in 0..k {
                prefill_idx.push(((token * k + pick) % 124) as i64);
            }
        }
        let prefill_w = vec![1.0f32 / k as f32; prefill_idx.len()];
        let prefill = group_by_expert(&prefill_idx, &prefill_w, 256, k, all_resident).unwrap();
        assert_eq!(prefill.len(), 124, "groups are capped by the resident expert count");
        let prefill_bytes = reconstruct_bytes_for(&prefill, &config).unwrap();

        // 256x the tokens for only ~20x the reconstruct bytes.
        let ratio = prefill_bytes as f64 / decode_bytes as f64;
        assert!(
            (20.0..21.0).contains(&ratio),
            "expected ~20.7x reconstruct for 256x the tokens, got {ratio:.1}x"
        );
    }
}

/// The permutation that turns token-major routing into expert-major rows.
///
/// ## The trap this exists to avoid
/// `kernels/gb10/common/moe_permute.cu` ships TWO unpermute kernels and they assume
/// different layouts:
///
/// - `moe_unpermute_reduce` hardcodes `perm_row = token * topk + k`. That is TOKEN-MAJOR:
///   a token's `topk` rows are contiguous, and an expert's rows are scattered.
/// - `moe_unpermute_reduce_indexed` takes an explicit `token_to_perm[token, k]` map and
///   imposes no layout at all.
///
/// [`group_by_expert`] produces EXPERT-major order, because that is what lets one expert be
/// reconstructed once and applied as a single batched GEMM. Pairing expert-major rows with
/// the token-major kernel would read whichever rows happened to sit at `token * topk + k` —
/// real expert outputs, correctly shaped, belonging to the wrong tokens. No error, no NaN,
/// fluent wrong output.
///
/// So this plan carries BOTH index arrays and [`Cb3Permutation::verify_round_trip`] checks
/// they invert each other, and the doc names `moe_unpermute_reduce_indexed` as the only
/// correct consumer.
pub struct Cb3Permutation {
    /// `[total_expanded]` — permuted row -> original token. Feeds `moe_permute_tokens`.
    pub sorted_token_ids: Vec<i32>,
    /// `[num_tokens, topk]` — (token, pick) -> permuted row. Feeds
    /// `moe_unpermute_reduce_indexed`. **Not** `moe_unpermute_reduce`.
    pub token_to_perm: Vec<i32>,
    /// `[num_tokens, topk]` routing weights, in the caller's original order.
    pub weights: Vec<f32>,
    /// Row ranges into the permuted buffer, one per expert, in `groups` order.
    pub group_rows: Vec<(usize, usize)>,
    pub total_expanded: usize,
}

/// Module holding [`PERMUTE_KERNEL`] and [`UNPERMUTE_KERNEL`].
///
/// NOT the file stem: `kernels/gb10/common/moe_permute.cu` is renamed to `moe` by the
/// `[modules]` table in `kernels/gb10/deepseek-v4.1/cb3/KERNEL.toml`, so a lookup of
/// `"moe_permute"` fails at `gpu.kernel()`. Pinned to that file by a test.
pub const MOE_PERMUTE_MODULE: &str = "moe";
/// Module holding `moe_silu_mul` (the swiglu_limit-clamped routed SwiGLU). Not renamed, so
/// the stem is the module name.
pub const SILU_MUL_MODULE: &str = "moe_silu_mul";
/// The unpermute kernel this plan is valid for. Spelled once.
pub const UNPERMUTE_KERNEL: &str = "moe_unpermute_reduce_indexed";
/// The gather kernel.
pub const PERMUTE_KERNEL: &str = "moe_permute_tokens";

impl Cb3Permutation {
    /// Build the expert-major permutation from grouped routing.
    pub fn build(groups: &[ExpertGroup], num_tokens: usize, k: usize) -> Result<Self> {
        let total_expanded = num_tokens * k;
        let mut sorted_token_ids = vec![-1i32; total_expanded];
        let mut token_to_perm = vec![-1i32; total_expanded];
        let mut weights = vec![0.0f32; total_expanded];
        let mut group_rows = Vec::with_capacity(groups.len());
        // How many picks of each token have been placed, so a token routed to several
        // experts gets distinct (token, pick) slots.
        let mut placed = vec![0usize; num_tokens];

        let mut row = 0usize;
        for group in groups {
            let begin = row;
            for (token, weight) in group.tokens.iter().zip(&group.weights) {
                let token = *token as usize;
                ensure!(token < num_tokens, "token {token} is outside {num_tokens}");
                let pick = placed[token];
                ensure!(
                    pick < k,
                    "token {token} routed to more than {k} experts — routing produced \
                     duplicate picks"
                );
                sorted_token_ids[row] = token as i32;
                token_to_perm[token * k + pick] = row as i32;
                weights[token * k + pick] = *weight;
                placed[token] = pick + 1;
                row += 1;
            }
            group_rows.push((begin, row));
        }

        ensure!(
            row == total_expanded,
            "permutation placed {row} rows, expected {total_expanded}"
        );
        ensure!(
            placed.iter().all(|count| *count == k),
            "some token was not routed to exactly {k} experts"
        );
        let plan = Self {
            sorted_token_ids,
            token_to_perm,
            weights,
            group_rows,
            total_expanded,
        };
        plan.verify_round_trip(num_tokens, k)?;
        Ok(plan)
    }

    /// The two index arrays must invert each other.
    ///
    /// Cheap (O(num_tokens * k)) and run on every build, because the failure it catches is
    /// silent: a permutation that is internally inconsistent still produces finite output.
    pub fn verify_round_trip(&self, num_tokens: usize, k: usize) -> Result<()> {
        for token in 0..num_tokens {
            for pick in 0..k {
                let row = self.token_to_perm[token * k + pick];
                ensure!(row >= 0, "token {token} pick {pick} has no permuted row");
                let back = self.sorted_token_ids[row as usize];
                ensure!(
                    back == token as i32,
                    "permutation is inconsistent: token {token} pick {pick} -> row {row} -> \
                     token {back}. The forward and reverse maps disagree, which would make \
                     the unpermute read another token's expert output."
                );
            }
        }
        ensure!(
            self.sorted_token_ids.iter().all(|t| *t >= 0),
            "a permuted row was never assigned a token"
        );
        Ok(())
    }
}

#[cfg(test)]
mod permutation_tests {
    use super::*;

    fn resident(id: u32) -> Result<i32> {
        Ok(id as i32)
    }

    /// The plan must round-trip, and each expert's rows must be CONTIGUOUS — that
    /// contiguity is the entire reason for grouping.
    #[test]
    fn the_permutation_is_expert_major_and_round_trips() {
        // 3 tokens, k=2: t0 -> {5,9}, t1 -> {5,7}, t2 -> {9,5}
        let indices: Vec<i64> = vec![5, 9, 5, 7, 9, 5];
        let weights: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let groups = group_by_expert(&indices, &weights, 3, 2, resident).unwrap();
        let plan = Cb3Permutation::build(&groups, 3, 2).unwrap();

        assert_eq!(plan.total_expanded, 6);
        // Expert 5 has three tokens, then 7 has one, then 9 has two — contiguous ranges.
        assert_eq!(plan.group_rows, vec![(0, 3), (3, 4), (4, 6)]);
        assert_eq!(plan.sorted_token_ids, vec![0, 1, 2, 1, 0, 2]);
        // Round-trip is checked inside build(); assert it independently too.
        plan.verify_round_trip(3, 2).unwrap();

        // Every weight must land with its own (token, pick) slot.
        for token in 0..3usize {
            let mut got: Vec<f32> = (0..2).map(|p| plan.weights[token * 2 + p]).collect();
            let mut want: Vec<f32> = (0..2).map(|p| weights[token * 2 + p]).collect();
            got.sort_by(|a, b| a.partial_cmp(b).unwrap());
            want.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(got, want, "token {token} lost or duplicated a weight");
        }
    }

    /// THE TRAP, as a test: expert-major rows do NOT satisfy the token-major kernel's
    /// assumption, so pairing them would silently read the wrong rows.
    ///
    /// `moe_unpermute_reduce` assumes `perm_row == token * topk + k`. If that happened to
    /// hold for expert-major order, the two kernels would be interchangeable and this
    /// distinction would not matter. It does not hold — asserted here so the constraint is
    /// demonstrated rather than described.
    #[test]
    fn expert_major_rows_violate_the_token_major_kernels_assumption() {
        let indices: Vec<i64> = vec![5, 9, 5, 7, 9, 5];
        let weights: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let groups = group_by_expert(&indices, &weights, 3, 2, resident).unwrap();
        let plan = Cb3Permutation::build(&groups, 3, 2).unwrap();

        let token_major_would_read: Vec<i32> = (0..3)
            .flat_map(|t| (0..2).map(move |k| (t * 2 + k) as i32))
            .collect();
        assert_ne!(
            plan.token_to_perm, token_major_would_read,
            "if these matched, moe_unpermute_reduce would be safe here and the _indexed \
             variant unnecessary — the whole reason for UNPERMUTE_KERNEL would be gone"
        );
        // And name the kernel that IS correct, so the constant cannot drift from the doc.
        assert_eq!(UNPERMUTE_KERNEL, "moe_unpermute_reduce_indexed");
    }

    /// The module names must be the ones the cb3 target's `KERNEL.toml` actually produces.
    ///
    /// The previous agent's first GPU run failed at `gpu.kernel("moe_permute", ..)` because
    /// the target renames that stem. This ties the constants to the file, so the next rename
    /// fails here rather than at load time on the GPU.
    #[test]
    fn moe_module_names_match_the_cb3_kernel_toml() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .expect("workspace root")
            .to_path_buf();
        let toml = std::fs::read_to_string(root.join("kernels/gb10/deepseek-v4.1/cb3/KERNEL.toml"))
            .expect("cb3 KERNEL.toml readable");
        let renamed_to = |stem: &str| -> Option<String> {
            toml.lines()
                .map(str::trim)
                .filter(|line| !line.starts_with('#'))
                .find_map(|line| {
                    let (key, value) = line.split_once('=')?;
                    (key.trim() == stem).then(|| value.trim().trim_matches('"').to_string())
                })
        };
        // The stem IS renamed, so the stem is the wrong lookup name — the bug this pins.
        assert_eq!(renamed_to("moe_permute").as_deref(), Some(MOE_PERMUTE_MODULE));
        assert_ne!(MOE_PERMUTE_MODULE, "moe_permute");
        // silu_mul is NOT renamed, so its stem is its module name.
        assert_eq!(renamed_to(SILU_MUL_MODULE), None);
        let common = root.join("kernels/gb10/common");
        for (file, kernel) in [
            ("moe_permute.cu", PERMUTE_KERNEL),
            ("moe_permute.cu", UNPERMUTE_KERNEL),
            ("moe_silu_mul.cu", "moe_silu_mul"),
        ] {
            let source = std::fs::read_to_string(common.join(file)).expect("kernel source");
            assert!(
                source.contains(&format!("__global__ void {kernel}(")),
                "{kernel} must be defined in {file}"
            );
        }
    }

    /// The combine kernels must exist under the names the build gives them (file stem, no
    /// KERNEL.toml override).
    #[test]
    fn combine_kernel_names_match_the_source() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .expect("workspace root")
            .to_path_buf();
        let cu = root
            .join("kernels/gb10/deepseek-v4.1/cb3")
            .join(format!("{COMBINE_MODULE}.cu"));
        let source = std::fs::read_to_string(&cu).expect("combine kernel source");
        for kernel in [SWIGLU_WEIGHTED_FN, UNPERMUTE_SUM_FN, ROUTE_TOPK_FN] {
            // `__launch_bounds__(...)` may sit between `__global__` and the return type.
            let declared = source.lines().any(|line| {
                line.starts_with("extern \"C\" __global__") && line.contains(&format!(" {kernel}("))
            });
            assert!(declared, "{kernel} must be a C-linkage kernel in {}", cu.display());
        }
    }

    /// An inconsistent permutation must be REFUSED, not silently used.
    #[test]
    fn a_broken_round_trip_is_caught() {
        let indices: Vec<i64> = vec![5, 9];
        let weights: Vec<f32> = vec![0.5, 0.5];
        let groups = group_by_expert(&indices, &weights, 1, 2, resident).unwrap();
        let mut plan = Cb3Permutation::build(&groups, 1, 2).unwrap();
        plan.verify_round_trip(1, 2).expect("the built plan is consistent");

        // Corrupt the reverse map so it points at the other expert's row.
        plan.token_to_perm[0] = 1;
        plan.sorted_token_ids[1] = 99;
        let err = plan
            .verify_round_trip(1, 2)
            .expect_err("an inconsistent permutation must be refused")
            .to_string();
        assert!(err.contains("inconsistent"), "{err}");
        assert!(
            err.contains("another token's expert output"),
            "the error must name the consequence: {err}"
        );
    }
}
