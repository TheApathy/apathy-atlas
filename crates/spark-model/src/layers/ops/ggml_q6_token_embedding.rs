// SPDX-License-Identifier: AGPL-3.0-only

//! Direct Q6_K token-row gather for the pinned GLM-5.3 IQ3 embedding.
//!
//! The caller must copy the same IDs passed to [`GgmlQ6TokenEmbeddingPlan::new`]
//! into `token_ids_device`; construction validates every ID before any launch.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::GgmlIqBuffer;

const VOCAB: u32 = 154_880;
const HIDDEN: u32 = 4_096;
const Q6_BLOCK_VALUES: usize = 256;
const Q6_BLOCK_BYTES: usize = 210;
const MAX_TOKENS: usize = 65_520;
const THREADS: u32 = 256;

/// Exact extents for one validated, bounded gather.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgmlQ6TokenEmbeddingPlan {
    token_count: u32,
    source_bytes: usize,
    token_id_bytes: usize,
    output_bytes: usize,
}

impl GgmlQ6TokenEmbeddingPlan {
    pub fn new(token_ids: &[u32]) -> Result<Self> {
        if token_ids.is_empty() || token_ids.len() > MAX_TOKENS {
            bail!("GLM Q6_K token gather requires 1..={MAX_TOKENS} token IDs");
        }
        if let Some((index, token)) = token_ids
            .iter()
            .copied()
            .enumerate()
            .find(|(_, token)| *token >= VOCAB)
        {
            bail!("GLM Q6_K token ID {token} at index {index} exceeds vocabulary");
        }
        let token_count = u32::try_from(token_ids.len())?;
        let source_elements = usize::try_from(VOCAB)?
            .checked_mul(usize::try_from(HIDDEN)?)
            .context("GLM Q6_K embedding element count overflow")?;
        let source_bytes = source_elements
            .checked_div(Q6_BLOCK_VALUES)
            .and_then(|blocks| blocks.checked_mul(Q6_BLOCK_BYTES))
            .context("GLM Q6_K embedding source byte count overflow")?;
        let token_id_bytes = token_ids
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .context("GLM Q6_K token ID byte count overflow")?;
        let output_bytes = token_ids
            .len()
            .checked_mul(usize::try_from(HIDDEN)?)
            .and_then(|elements| elements.checked_mul(2))
            .context("GLM Q6_K token output byte count overflow")?;
        Ok(Self {
            token_count,
            source_bytes,
            token_id_bytes,
            output_bytes,
        })
    }

    pub const fn token_count(self) -> u32 {
        self.token_count
    }

    pub const fn source_bytes(self) -> usize {
        self.source_bytes
    }

    pub const fn token_id_bytes(self) -> usize {
        self.token_id_bytes
    }

    pub const fn output_bytes(self) -> usize {
        self.output_bytes
    }
}

pub struct GgmlQ6TokenEmbeddingKernel {
    gather: KernelHandle,
}

impl GgmlQ6TokenEmbeddingKernel {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            gather: gpu.kernel("ggml_q6_token_embedding", "atlas_q6_k_token_embedding_bf16")?,
        })
    }

    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        plan: GgmlQ6TokenEmbeddingPlan,
        source: GgmlIqBuffer,
        token_ids_device: GgmlIqBuffer,
        output: GgmlIqBuffer,
        stream: u64,
    ) -> Result<()> {
        if source.bytes != plan.source_bytes
            || token_ids_device.bytes != plan.token_id_bytes
            || output.bytes != plan.output_bytes
        {
            bail!("GLM Q6_K token gather device buffer extent mismatch");
        }
        if source.ptr == DevicePtr::NULL
            || token_ids_device.ptr == DevicePtr::NULL
            || output.ptr == DevicePtr::NULL
        {
            bail!("GLM Q6_K token gather requires nonnull device buffers");
        }
        if source.ptr.0 % 2 != 0 || token_ids_device.ptr.0 % 4 != 0 || output.ptr.0 % 2 != 0 {
            bail!("GLM Q6_K token gather device buffer alignment mismatch");
        }
        let source_end = checked_end(source, "source")?;
        let token_ids_end = checked_end(token_ids_device, "token IDs")?;
        let output_end = checked_end(output, "output")?;
        if overlaps(
            source.ptr.0,
            source_end,
            token_ids_device.ptr.0,
            token_ids_end,
        ) || overlaps(source.ptr.0, source_end, output.ptr.0, output_end)
            || overlaps(
                token_ids_device.ptr.0,
                token_ids_end,
                output.ptr.0,
                output_end,
            )
        {
            bail!("GLM Q6_K token gather device buffers overlap");
        }
        KernelLaunch::new(gpu, self.gather)
            .grid([plan.token_count, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(source.ptr)
            .arg_ptr(token_ids_device.ptr)
            .arg_ptr(output.ptr)
            .arg_u32(plan.token_count)
            .launch(stream)
    }
}

fn checked_end(buffer: GgmlIqBuffer, label: &str) -> Result<u64> {
    buffer
        .ptr
        .0
        .checked_add(u64::try_from(buffer.bytes)?)
        .with_context(|| format!("GLM Q6_K token gather {label} address overflow"))
}

const fn overlaps(a_start: u64, a_end: u64, b_start: u64, b_end: u64) -> bool {
    a_start < b_end && b_start < a_end
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    fn buffer(ptr: u64, bytes: usize) -> GgmlIqBuffer {
        GgmlIqBuffer {
            ptr: DevicePtr(ptr),
            bytes,
        }
    }

    #[test]
    fn exact_glm_geometry_and_token_bounds_are_pinned() {
        let one = GgmlQ6TokenEmbeddingPlan::new(&[0]).unwrap();
        assert_eq!(one.token_count(), 1);
        assert_eq!(one.source_bytes(), 520_396_800);
        assert_eq!(one.token_id_bytes(), 4);
        assert_eq!(one.output_bytes(), 8_192);
        let max_ids = vec![VOCAB - 1; MAX_TOKENS];
        let max = GgmlQ6TokenEmbeddingPlan::new(&max_ids).unwrap();
        assert_eq!(max.token_count(), 65_520);
        assert_eq!(max.token_id_bytes(), 262_080);
        assert_eq!(max.output_bytes(), 536_739_840);
        assert!(GgmlQ6TokenEmbeddingPlan::new(&[]).is_err());
        assert!(GgmlQ6TokenEmbeddingPlan::new(&vec![0; MAX_TOKENS + 1]).is_err());
        assert!(GgmlQ6TokenEmbeddingPlan::new(&[0, VOCAB - 1, VOCAB]).is_err());
    }

    #[test]
    fn invalid_ids_and_extents_fail_before_launch() {
        let gpu = MockGpuBackend::new();
        let kernel = GgmlQ6TokenEmbeddingKernel::load(&gpu).unwrap();
        assert!(GgmlQ6TokenEmbeddingPlan::new(&[VOCAB]).is_err());
        assert_eq!(gpu.launch_count(), 0);
        let plan = GgmlQ6TokenEmbeddingPlan::new(&[7, 11]).unwrap();
        let source = buffer(0x1000, plan.source_bytes());
        let ids = buffer(0x4000_0000, plan.token_id_bytes());
        let output = buffer(0x5000_0000, plan.output_bytes());
        for bad in [
            (buffer(source.ptr.0, source.bytes - 1), ids, output),
            (source, buffer(ids.ptr.0, ids.bytes - 1), output),
            (source, ids, buffer(output.ptr.0, output.bytes - 1)),
        ] {
            assert!(kernel.launch(&gpu, plan, bad.0, bad.1, bad.2, 9).is_err());
        }
        assert_eq!(gpu.launch_count(), 0);
    }

    #[test]
    fn null_overflow_alignment_and_every_alias_pair_fail_closed() {
        let gpu = MockGpuBackend::new();
        let kernel = GgmlQ6TokenEmbeddingKernel::load(&gpu).unwrap();
        let plan = GgmlQ6TokenEmbeddingPlan::new(&[3]).unwrap();
        let source = buffer(0x1000, plan.source_bytes());
        let ids = buffer(0x4000_0000, plan.token_id_bytes());
        let output = buffer(0x5000_0000, plan.output_bytes());
        let hostile = [
            (buffer(0, source.bytes), ids, output),
            (source, buffer(0, ids.bytes), output),
            (source, ids, buffer(0, output.bytes)),
            (
                buffer(u64::MAX - source.bytes as u64 + 1, source.bytes),
                ids,
                output,
            ),
            (
                source,
                buffer(u64::MAX - ids.bytes as u64 + 1, ids.bytes),
                output,
            ),
            (
                source,
                ids,
                buffer(u64::MAX - output.bytes as u64 + 1, output.bytes),
            ),
            (source, buffer(ids.ptr.0 + 1, ids.bytes), output),
            (source, buffer(source.ptr.0 + 4, ids.bytes), output),
            (source, ids, buffer(source.ptr.0 + 2, output.bytes)),
            (source, ids, buffer(ids.ptr.0 + 2, output.bytes)),
        ];
        for (bad_source, bad_ids, bad_output) in hostile {
            assert!(
                kernel
                    .launch(&gpu, plan, bad_source, bad_ids, bad_output, 9)
                    .is_err()
            );
        }
        assert_eq!(gpu.launch_count(), 0);
        kernel.launch(&gpu, plan, source, ids, output, 9).unwrap();
        assert_eq!(gpu.launch_count(), 1);
    }

    struct ReferenceQ6Block {
        delta: f32,
        low: [u8; 128],
        high: [u8; 64],
        scales: [i8; 16],
    }

    fn reference_q6(block: &ReferenceQ6Block, column: usize) -> f32 {
        let half = column / 128;
        let segment = (column % 128) / 32;
        let lane = column % 32;
        let low_index = 64 * half + lane + 32 * (segment & 1);
        let low = (block.low[low_index] >> (4 * (segment / 2))) & 0x0f;
        let high = (block.high[32 * half + lane] >> (2 * segment)) & 0x03;
        let quant = i32::from(low | (high << 4)) - 32;
        let scale_index = 8 * half + lane / 16 + 2 * segment;
        block.delta * f32::from(block.scales[scale_index]) * quant as f32
    }

    #[test]
    fn reference_pins_q6_low_high_scale_and_half_chronology() {
        let mut block = ReferenceQ6Block {
            delta: 0.5,
            low: [0; 128],
            high: [0; 64],
            scales: [0; 16],
        };
        block.low[0] = 0xb5;
        block.high[0] = 0x12;
        block.scales[0] = -3;
        block.scales[4] = 2;
        block.low[17] = 3;
        block.scales[1] = 4;
        block.low[64] = 7;
        block.high[32] = 0x83;
        block.scales[8] = -1;
        block.low[96] = 0xc0;
        block.scales[14] = 3;
        assert_eq!(reference_q6(&block, 0), -7.5);
        assert_eq!(reference_q6(&block, 17), -58.0);
        assert_eq!(reference_q6(&block, 64), -5.0);
        assert_eq!(reference_q6(&block, 128), -11.5);
        assert_eq!(reference_q6(&block, 224), 18.0);
    }
}
