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
