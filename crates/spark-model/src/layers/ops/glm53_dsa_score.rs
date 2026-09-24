// SPDX-License-Identifier: AGPL-3.0-only

//! Exact pooled-key score primitive for the GLM-5.3 DSA indexer.
//! `head_weights_bf16` is the raw BF16 `weights_proj` result. The kernel
//! converts it to F32 and applies the official `32^-0.5` scale; no sigmoid is
//! present in the current upstream GLM5Next scorer. Visibility and top-k are
//! deliberately outside this primitive.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::{GLM53_EXL3_MAX_WIDE_ROWS, GgmlIqBuffer};

const HEADS: u32 = 32;
const INDEX_DIM: u32 = 128;
const MAX_QUERIES: u32 = GLM53_EXL3_MAX_WIDE_ROWS as u32;
const MAX_POOLS: u32 = 262_144;
const THREADS: u32 = INDEX_DIM;
const MAX_GRID_YZ: u64 = 65_535;
const ROWS_THREADS: u32 = 256;

/// `ATLAS_GLM53_DSA_SCORE_ROWS=1`: row-shared pooled-key scoring, one CTA per
/// query row. Bit-identical to the per-(row, pool) reference kernel.
fn score_rows() -> Result<bool> {
    use std::sync::OnceLock;
    static ON: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ON.get_or_init(|| match std::env::var("ATLAS_GLM53_DSA_SCORE_ROWS") {
        Ok(v) if v == "1" => Ok(true),
        Ok(v) if v == "0" => Ok(false),
        Ok(other) => Err(format!(
            "ATLAS_GLM53_DSA_SCORE_ROWS must be 0 or 1, got {other:?}"
        )),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(e) => Err(format!("ATLAS_GLM53_DSA_SCORE_ROWS: {e}")),
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53DsaScorePlan {
    pub batch: u32,
    pub queries: u32,
    pub pools: u32,
    pub heads: u32,
    pub index_dim: u32,
    pub grid_x: u32,
    pub grid_y: u32,
    pub grid_z: u32,
    pub query_bytes: usize,
    pub head_weight_bytes: usize,
    pub pool_key_bytes: usize,
    pub pool_validity_bytes: usize,
    pub output_bytes: usize,
}

impl Glm53DsaScorePlan {
    pub fn new(batch: u32, queries: u32, pools: u32, heads: u32, index_dim: u32) -> Result<Self> {
        if batch == 0 || !(1..=MAX_QUERIES).contains(&queries) {
            bail!("GLM DSA scoring requires batch>0 and Q in 1..={MAX_QUERIES}");
        }
        if !(1..=MAX_POOLS).contains(&pools) || heads != HEADS || index_dim != INDEX_DIM {
            bail!("GLM DSA scoring requires P in 1..=262,144, heads32 and dim128");
        }
        let rows = u64::from(batch)
            .checked_mul(u64::from(queries))
            .context("GLM DSA score row overflow")?;
        let grid_y = rows.min(MAX_GRID_YZ);
        let grid_z = rows
            .checked_add(grid_y - 1)
            .context("GLM DSA score grid rounding overflow")?
            / grid_y;
        if grid_z > MAX_GRID_YZ {
            bail!("GLM DSA score row grid exceeds CUDA y/z capacity");
        }
        let pool_rows = u64::from(batch)
            .checked_mul(u64::from(pools))
            .context("GLM DSA score pool-row overflow")?;
        let output_elements = rows
            .checked_mul(u64::from(pools))
            .context("GLM DSA score output-element overflow")?;
        Ok(Self {
            batch,
            queries,
            pools,
            heads,
            index_dim,
            grid_x: pools,
            grid_y: u32::try_from(grid_y)?,
            grid_z: u32::try_from(grid_z)?,
            query_bytes: extent(rows, u64::from(HEADS) * u64::from(INDEX_DIM), 2)?,
            head_weight_bytes: extent(rows, u64::from(HEADS), 2)?,
            pool_key_bytes: extent(pool_rows, u64::from(INDEX_DIM), 2)?,
            pool_validity_bytes: usize::try_from(pool_rows)?,
            output_bytes: extent(output_elements, 1, 4)?,
        })
    }

    pub fn validate(self) -> Result<()> {
        if Self::new(
            self.batch,
            self.queries,
            self.pools,
            self.heads,
            self.index_dim,
        )? != self
        {
            bail!("forged GLM DSA score plan");
        }
        Ok(())
    }
}

fn extent(rows: u64, columns: u64, element_bytes: u64) -> Result<usize> {
    usize::try_from(
        rows.checked_mul(columns)
            .and_then(|elements| elements.checked_mul(element_bytes))
            .context("GLM DSA score byte-extent overflow")?,
    )
    .context("GLM DSA score extent does not fit usize")
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53DsaScoreBuffers {
    pub queries_bf16: GgmlIqBuffer,
    pub head_weights_bf16: GgmlIqBuffer,
    pub pool_keys_bf16: GgmlIqBuffer,
    pub pool_validity_u8: GgmlIqBuffer,
    pub output_scores_f32: GgmlIqBuffer,
}

pub struct Glm53DsaScoreKernel {
    score: KernelHandle,
    rows_score: KernelHandle,
}

impl Glm53DsaScoreKernel {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            score: gpu.kernel("glm53_dsa_score", "atlas_glm53_dsa_score_bf16")?,
            rows_score: gpu.kernel("glm53_dsa_score", "atlas_glm53_dsa_score_rows_bf16")?,
        })
    }

    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53DsaScorePlan,
        buffers: Glm53DsaScoreBuffers,
        stream: u64,
    ) -> Result<()> {
        plan.validate()?;
        validate_buffers(plan, buffers)?;
        let (kernel, grid, block) = if score_rows()? {
            // One CTA per query row; bit-identical (see the .cu header).
            let rows = plan
                .batch
                .checked_mul(plan.queries)
                .context("GLM DSA score row overflow")?;
            (self.rows_score, [rows, 1, 1], ROWS_THREADS)
        } else {
            (self.score, [plan.grid_x, plan.grid_y, plan.grid_z], THREADS)
        };
        KernelLaunch::new(gpu, kernel)
            .grid(grid)
            .block([block, 1, 1])
            .arg_ptr(buffers.queries_bf16.ptr)
            .arg_ptr(buffers.head_weights_bf16.ptr)
            .arg_ptr(buffers.pool_keys_bf16.ptr)
            .arg_ptr(buffers.pool_validity_u8.ptr)
            .arg_ptr(buffers.output_scores_f32.ptr)
            .arg_u32(plan.batch)
            .arg_u32(plan.queries)
            .arg_u32(plan.pools)
            .launch(stream)
    }
}

fn validate_buffers(plan: Glm53DsaScorePlan, buffers: Glm53DsaScoreBuffers) -> Result<()> {
    let named = [
        ("queries", buffers.queries_bf16, plan.query_bytes),
        (
            "head weights",
            buffers.head_weights_bf16,
            plan.head_weight_bytes,
        ),
        ("pool keys", buffers.pool_keys_bf16, plan.pool_key_bytes),
        (
            "pool validity",
            buffers.pool_validity_u8,
            plan.pool_validity_bytes,
        ),
        (
            "output scores",
            buffers.output_scores_f32,
            plan.output_bytes,
        ),
    ];
    // Sized from the table it mirrors, never a literal: a hand-written
    // length silently goes out of bounds the moment the table grows.
    let mut ranges = Vec::with_capacity(named.len());
    for (name, buffer, expected) in named.iter().copied() {
        if buffer.bytes != expected || buffer.ptr == DevicePtr::NULL {
            bail!("GLM DSA score {name} buffer is null or has the wrong extent");
        }
        ranges.push((
            buffer.ptr.0,
            buffer
                .ptr
                .0
                .checked_add(u64::try_from(expected)?)
                .with_context(|| format!("GLM DSA score {name} address overflow"))?,
        ));
    }
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].0 < ranges[right].1 && ranges[right].0 < ranges[left].1 {
                bail!("GLM DSA score device buffers overlap");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use spark_runtime::gpu::mock::MockGpuBackend;

    use super::*;

    fn rounded(value: f32) -> f32 {
        bf16::from_f32(value).to_f32()
    }

    fn reference(
        queries: &[[f32; 128]; 32],
        raw_weights: &[f32; 32],
        key: &[f32; 128],
        valid: bool,
    ) -> f32 {
        if !valid {
            return f32::MIN;
        }
        let mut score = 0.0f32;
        for head in 0..32 {
            let mut partial = [0.0f32; 128];
            for channel in 0..128 {
                partial[channel] = rounded(queries[head][channel]) * rounded(key[channel]);
            }
            let mut stride = 64;
            while stride != 0 {
                for channel in 0..stride {
                    partial[channel] += partial[channel + stride];
                }
                stride >>= 1;
            }
            let head_score = (partial[0] * (128.0f32).sqrt().recip()).max(0.0);
            let weight = rounded(raw_weights[head]) * (32.0f32).sqrt().recip();
            score += weight * head_score;
        }
        score
    }

    #[test]
    fn plan_pins_bootstrap_geometry_extents_and_grid_split() {
        const CUDA: &str =
            include_str!("../../../../../kernels/gb10/glm5.3-flash/iq3/glm53_dsa_score.cu");
        assert!(CUDA.contains("#define GLM53_DSA_MAX_QUERIES 8192U"));
        let plan = Glm53DsaScorePlan::new(1, MAX_QUERIES, 262_144, 32, 128).unwrap();
        assert_eq!((plan.grid_x, plan.grid_y, plan.grid_z), (262_144, 2_048, 1));
        assert_eq!(plan.query_bytes, 16_777_216);
        assert_eq!(plan.head_weight_bytes, 131_072);
        assert_eq!(plan.pool_key_bytes, 67_108_864);
        assert_eq!(plan.pool_validity_bytes, 262_144);
        assert_eq!(plan.output_bytes, 2_147_483_648);
        let split = Glm53DsaScorePlan::new(70_000, 1, 1, 32, 128).unwrap();
        assert_eq!((split.grid_y, split.grid_z), (65_535, 2));
        assert!(Glm53DsaScorePlan::new(0, 1, 1, 32, 128).is_err());
        assert!(Glm53DsaScorePlan::new(1, 0, 1, 32, 128).is_err());
        assert!(Glm53DsaScorePlan::new(1, MAX_QUERIES + 1, 1, 32, 128).is_err());
        assert!(Glm53DsaScorePlan::new(1, 1, 0, 32, 128).is_err());
        assert!(Glm53DsaScorePlan::new(1, 1, 262_145, 32, 128).is_err());
        assert!(Glm53DsaScorePlan::new(1, 1, 1, 31, 128).is_err());
        assert!(Glm53DsaScorePlan::new(1, 1, 1, 32, 127).is_err());
        assert!(Glm53DsaScorePlan::new(u32::MAX, MAX_QUERIES, 1, 32, 128).is_err());
    }

    #[test]
    fn reference_pins_relu_raw_weight_scale_and_order() {
        let mut queries = [[0.0f32; 128]; 32];
        let mut key = [0.0f32; 128];
        let mut weights = [0.0f32; 32];
        key[..4].fill(1.0);
        queries[0][..4].copy_from_slice(&[1.0e20, 1.0, -1.0e20, 1.0]);
        queries[1][0] = -1.0;
        weights[0] = 1.0;
        weights[1] = 1.0;
        let tree_expected = 2.0 * (128.0f32).sqrt().recip() * (32.0f32).sqrt().recip();
        assert_eq!(
            reference(&queries, &weights, &key, true).to_bits(),
            tree_expected.to_bits()
        );
        assert_eq!(reference(&queries, &weights, &key, false), f32::MIN);
        queries = [[0.0; 128]; 32];
        key = [0.0; 128];
        key[0] = 1.0;
        for head in 0..4 {
            queries[head][0] = 1.0;
        }
        weights[..4].copy_from_slice(&[1.0e20, 1.0, -1.0e20, 1.0]);
        let ordered = reference(&queries, &weights, &key, true);
        weights[..4].copy_from_slice(&[1.0, -1.0e20, 1.0, 1.0e20]);
        let reordered = reference(&queries, &weights, &key, true);
        assert_ne!(ordered.to_bits(), reordered.to_bits());
    }

    #[test]
    fn forged_extent_or_alias_fails_before_launch() {
        let gpu = MockGpuBackend::new();
        let kernel = Glm53DsaScoreKernel::load(&gpu).unwrap();
        let plan = Glm53DsaScorePlan::new(1, 1, 1, 32, 128).unwrap();
        let mut address = 0x10_0000u64;
        let mut next = |bytes: usize| {
            let buffer = GgmlIqBuffer {
                ptr: DevicePtr(address),
                bytes,
            };
            address += u64::try_from(bytes).unwrap() + 0x1000;
            buffer
        };
        let valid = Glm53DsaScoreBuffers {
            queries_bf16: next(plan.query_bytes),
            head_weights_bf16: next(plan.head_weight_bytes),
            pool_keys_bf16: next(plan.pool_key_bytes),
            pool_validity_u8: next(plan.pool_validity_bytes),
            output_scores_f32: next(plan.output_bytes),
        };
        let mut forged = plan;
        forged.grid_z += 1;
        assert!(kernel.launch(&gpu, forged, valid, 0).is_err());
        let wrong_extent = Glm53DsaScoreBuffers {
            pool_validity_u8: GgmlIqBuffer {
                ptr: valid.pool_validity_u8.ptr,
                bytes: plan.pool_validity_bytes + 1,
            },
            ..valid
        };
        assert!(kernel.launch(&gpu, plan, wrong_extent, 0).is_err());
        let overflowing = Glm53DsaScoreBuffers {
            output_scores_f32: GgmlIqBuffer {
                ptr: DevicePtr(u64::MAX - 1),
                bytes: plan.output_bytes,
            },
            ..valid
        };
        assert!(kernel.launch(&gpu, plan, overflowing, 0).is_err());
        let alias = Glm53DsaScoreBuffers {
            output_scores_f32: GgmlIqBuffer {
                ptr: valid.queries_bf16.ptr,
                bytes: plan.output_bytes,
            },
            ..valid
        };
        assert!(kernel.launch(&gpu, plan, alias, 0).is_err());
        assert_eq!(gpu.launch_count(), 0);
        kernel.launch(&gpu, plan, valid, 0).unwrap();
        assert_eq!(gpu.launch_count(), 1);
    }
}
