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
