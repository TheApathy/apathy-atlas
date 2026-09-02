// SPDX-License-Identifier: AGPL-3.0-only

//! Direct Q5_K token-row gather for the pinned GLM-5.3 embedding table.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::GgmlIqBuffer;

pub const GLM53_Q5_EMBEDDING_VOCAB: u32 = 154_880;
pub const GLM53_Q5_EMBEDDING_HIDDEN: u32 = 4_096;
pub const GLM53_Q5_EMBEDDING_MAX_TOKENS: u32 = 65_520;

const Q5_BLOCK_VALUES: usize = 256;
const Q5_BLOCK_BYTES: usize = 176;
const BF16_BYTES: usize = 2;
const U32_BYTES: usize = 4;
const THREADS: u32 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgmlQ5EmbeddingPlan {
    pub tokens: u32,
    pub vocab: u32,
    pub hidden: u32,
    pub source_bytes: usize,
    pub token_ids_bytes: usize,
    pub destination_bytes: usize,
}

impl GgmlQ5EmbeddingPlan {
    pub fn new(tokens: u32, vocab: u32, hidden: u32) -> Result<Self> {
        if tokens == 0 || tokens > GLM53_Q5_EMBEDDING_MAX_TOKENS {
            bail!("GLM Q5_K embedding tokens must be in 1..=65,520");
        }
        if vocab != GLM53_Q5_EMBEDDING_VOCAB || hidden != GLM53_Q5_EMBEDDING_HIDDEN {
            bail!("GLM Q5_K embedding requires exact vocab154880/hidden4096 geometry");
        }
        let vocab = usize::try_from(vocab)?;
        let hidden = usize::try_from(hidden)?;
        let tokens = usize::try_from(tokens)?;
        if !hidden.is_multiple_of(Q5_BLOCK_VALUES) {
            bail!("GLM Q5_K embedding hidden width must be divisible by 256");
        }
        let source_blocks = vocab
            .checked_mul(hidden / Q5_BLOCK_VALUES)
            .context("GLM Q5_K embedding source block count overflow")?;
        let source_bytes = source_blocks
            .checked_mul(Q5_BLOCK_BYTES)
            .context("GLM Q5_K embedding source byte count overflow")?;
        let token_ids_bytes = tokens
            .checked_mul(U32_BYTES)
            .context("GLM Q5_K embedding token byte count overflow")?;
        let destination_bytes = tokens
            .checked_mul(hidden)
            .and_then(|elements| elements.checked_mul(BF16_BYTES))
            .context("GLM Q5_K embedding destination byte count overflow")?;
        Ok(Self {
            tokens: u32::try_from(tokens)?,
            vocab: u32::try_from(vocab)?,
            hidden: u32::try_from(hidden)?,
            source_bytes,
            token_ids_bytes,
            destination_bytes,
        })
    }

    fn validate(self) -> Result<()> {
        let expected = Self::new(self.tokens, self.vocab, self.hidden)?;
        if self != expected {
            bail!("GLM Q5_K embedding plan fields were forged");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GgmlQ5EmbeddingBuffers {
    /// Exact `[154880, 4096]` Q5_K table (176 bytes per 256 values).
    pub source_q5_k: GgmlIqBuffer,
    /// Caller-validated device token IDs, exact `[tokens]` `u32`.
    pub token_ids_u32: GgmlIqBuffer,
    /// Exact `[tokens, 4096]` BF16 output; no persistent expanded table.
    pub destination_bf16: GgmlIqBuffer,
}

pub struct GgmlQ5EmbeddingKernel {
    gather: KernelHandle,
}

impl GgmlQ5EmbeddingKernel {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            gather: gpu.kernel("ggml_q5_embedding", "atlas_q5_k_embedding_gather_bf16")?,
        })
    }

    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        plan: GgmlQ5EmbeddingPlan,
        buffers: GgmlQ5EmbeddingBuffers,
        stream: u64,
    ) -> Result<()> {
        plan.validate()?;
        validate_buffers(plan, buffers)?;
        KernelLaunch::new(gpu, self.gather)
            .grid([plan.tokens, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.source_q5_k.ptr)
            .arg_ptr(buffers.token_ids_u32.ptr)
            .arg_ptr(buffers.destination_bf16.ptr)
            .arg_u32(plan.tokens)
            .arg_u32(plan.vocab)
            .arg_u32(plan.hidden)
            .launch(stream)
    }
}

fn validate_buffers(plan: GgmlQ5EmbeddingPlan, buffers: GgmlQ5EmbeddingBuffers) -> Result<()> {
    let named = [
        ("Q5_K source", buffers.source_q5_k, plan.source_bytes, 4u64),
        (
            "token IDs",
            buffers.token_ids_u32,
            plan.token_ids_bytes,
            4u64,
        ),
        (
            "BF16 destination",
            buffers.destination_bf16,
            plan.destination_bytes,
            2u64,
        ),
    ];
    // Sized from the table it mirrors, never a literal: a hand-written
    // length silently goes out of bounds the moment the table grows.
    let mut ranges = Vec::with_capacity(named.len());
    for (name, buffer, expected, alignment) in named.iter().copied() {
        if buffer.ptr == DevicePtr::NULL || buffer.bytes != expected {
            bail!("GLM Q5_K embedding {name} is null or has the wrong extent");
        }
        if buffer.ptr.0 % alignment != 0 {
            bail!("GLM Q5_K embedding {name} address is misaligned");
        }
        let end = buffer
            .ptr
            .0
            .checked_add(u64::try_from(buffer.bytes)?)
            .with_context(|| format!("GLM Q5_K embedding {name} address overflow"))?;
        ranges.push((buffer.ptr.0, end));
    }
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].0 < ranges[right].1 && ranges[right].0 < ranges[left].1 {
                bail!("GLM Q5_K embedding device buffers overlap");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use spark_runtime::gpu::mock::MockGpuBackend;

    use super::*;

    const CUDA_SOURCE: &str =
        include_str!("../../../../../kernels/gb10/glm5.3-flash/iq3/ggml_q5_embedding.cu");

    fn buffer(address: u64, bytes: usize) -> GgmlIqBuffer {
        GgmlIqBuffer {
            ptr: DevicePtr(address),
            bytes,
        }
    }

    fn valid_buffers(plan: GgmlQ5EmbeddingPlan) -> GgmlQ5EmbeddingBuffers {
        GgmlQ5EmbeddingBuffers {
            source_q5_k: buffer(0x10_0000, plan.source_bytes),
            token_ids_u32: buffer(0x2000_0000, plan.token_ids_bytes),
            destination_bf16: buffer(0x3000_0000, plan.destination_bytes),
        }
    }

    #[test]
    fn exact_glm_geometry_has_no_persistent_bf16_expansion() {
        let one = GgmlQ5EmbeddingPlan::new(1, 154_880, 4_096).unwrap();
        assert_eq!(one.source_bytes, 436_142_080);
        assert_eq!(one.token_ids_bytes, 4);
        assert_eq!(one.destination_bytes, 8_192);
        let max = GgmlQ5EmbeddingPlan::new(65_520, 154_880, 4_096).unwrap();
        assert_eq!(max.source_bytes, one.source_bytes);
        assert_eq!(max.token_ids_bytes, 262_080);
        assert_eq!(max.destination_bytes, 536_739_840);
        assert!(GgmlQ5EmbeddingPlan::new(0, 154_880, 4_096).is_err());
        assert!(GgmlQ5EmbeddingPlan::new(65_521, 154_880, 4_096).is_err());
        assert!(GgmlQ5EmbeddingPlan::new(1, 154_879, 4_096).is_err());
        assert!(GgmlQ5EmbeddingPlan::new(1, 154_880, 4_095).is_err());
    }

    fn pack_scales(scales: [u8; 8], mins: [u8; 8]) -> [u8; 12] {
        let mut packed = [0u8; 12];
        for group in 0..8 {
            assert!(scales[group] < 64 && mins[group] < 64);
            if group < 4 {
                packed[group] = scales[group];
                packed[group + 4] = mins[group];
            } else {
                packed[group + 4] = (scales[group] & 0x0f) | ((mins[group] & 0x0f) << 4);
                packed[group - 4] |= (scales[group] >> 4) << 6;
                packed[group] |= (mins[group] >> 4) << 6;
            }
        }
        packed
    }

    fn unpack_scale_min(group: usize, packed: &[u8; 12]) -> (u8, u8) {
        if group < 4 {
            (packed[group] & 63, packed[group + 4] & 63)
        } else {
            (
                (packed[group + 4] & 0x0f) | ((packed[group - 4] >> 6) << 4),
                (packed[group + 4] >> 4) | ((packed[group] >> 6) << 4),
            )
        }
    }

    fn reference_dequant(block: &[u8; 176]) -> [bf16; 256] {
        let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let dmin = f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
        let packed: &[u8; 12] = block[4..16].try_into().unwrap();
        let qh = &block[16..48];
        let qs = &block[48..176];
        let mut output = [bf16::from_bits(0); 256];
        for column in 0..256usize {
            let chunk = column / 64;
            let half = (column / 32) & 1;
            let lane = column & 31;
            let group = 2 * chunk + half;
            let (scale, min) = unpack_scale_min(group, packed);
            let nibble = if half == 0 {
                qs[chunk * 32 + lane] & 0x0f
            } else {
                qs[chunk * 32 + lane] >> 4
            };
            let high_mask = (1u8 << half) << (2 * chunk);
            let quant = u32::from(nibble) + if qh[lane] & high_mask != 0 { 16 } else { 0 };
            let scaled = d * f32::from(scale);
            let minimum = dmin * f32::from(min);
            output[column] = bf16::from_f32(scaled * quant as f32 - minimum);
        }
        output
    }

    #[test]
    fn reference_pins_scale_min_high_bit_and_nibble_chronology() {
        let scales = [1u8, 2, 3, 4, 17, 34, 51, 63];
        let mins = [5u8, 6, 7, 8, 21, 38, 55, 62];
        let mut block = [0u8; 176];
        block[0..2].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&f16::from_f32(0.25).to_bits().to_le_bytes());
        block[4..16].copy_from_slice(&pack_scales(scales, mins));
        let mut logical_quants = [0u8; 256];
        for chunk in 0..4usize {
            for lane in 0..32usize {
                let low = u8::try_from((3 * chunk + lane) % 16).unwrap();
                let high = u8::try_from((5 * chunk + 2 * lane + 1) % 16).unwrap();
                logical_quants[64 * chunk + lane] = low;
                logical_quants[64 * chunk + 32 + lane] = high;
                block[48 + 32 * chunk + lane] = low | (high << 4);
                if (chunk + lane) % 3 == 0 {
                    block[16 + lane] |= 1u8 << (2 * chunk);
                    logical_quants[64 * chunk + lane] += 16;
                }
                if (2 * chunk + lane) % 5 == 0 {
                    block[16 + lane] |= 2u8 << (2 * chunk);
                    logical_quants[64 * chunk + 32 + lane] += 16;
                }
            }
        }
        let actual = reference_dequant(&block);
        for column in 0..256usize {
            let group = column / 32;
            let expected = bf16::from_f32(
                0.5 * f32::from(scales[group]) * f32::from(logical_quants[column])
                    - 0.25 * f32::from(mins[group]),
            );
            assert_eq!(
                actual[column].to_bits(),
                expected.to_bits(),
                "column {column}"
            );
        }
        assert_ne!(actual[0].to_bits(), actual[32].to_bits());
        assert_ne!(actual[127].to_bits(), actual[128].to_bits());
        assert_ne!(actual[223].to_bits(), actual[224].to_bits());
    }

    #[test]
    fn cuda_source_is_a_direct_row_gather_with_the_pinned_equation() {
        for required in [
            "ca3d5a3e10d53f7ea672cb9b6178faca3e2807bc",
            "atlas_q5_k_embedding_gather_bf16",
            "const unsigned int token = token_ids[output_row]",
            "(unsigned long long) token * blocks_per_row + column / QK_K",
            "block->qs[chunk * 32u + lane]",
            "(1u << half) << (2u * chunk)",
            "block->qh[lane] & high_mask",
            "block->data.d",
            "block->data.dmin",
            "scaled * (float) quant - offset",
            "__float2bfloat16_rn(value)",
        ] {
            assert!(
                CUDA_SOURCE.contains(required),
                "missing CUDA contract: {required}"
            );
        }
        assert!(CUDA_SOURCE.contains("if (token >= vocab)"));
        assert!(CUDA_SOURCE.contains("output[column] = __float2bfloat16_rn(0.0f)"));
    }

    #[test]
    fn launch_hostiles_fail_before_enqueue() {
        let gpu = MockGpuBackend::new();
        let kernel = GgmlQ5EmbeddingKernel::load(&gpu).unwrap();
        let plan = GgmlQ5EmbeddingPlan::new(2, 154_880, 4_096).unwrap();

        let mut forged = plan;
        forged.destination_bytes -= 2;
        assert!(kernel.launch(&gpu, forged, valid_buffers(plan), 7).is_err());

        let mut cases = Vec::new();
        let mut buffers = valid_buffers(plan);
        buffers.source_q5_k.bytes -= 1;
        cases.push(buffers);
        let mut buffers = valid_buffers(plan);
        buffers.token_ids_u32.bytes -= 1;
        cases.push(buffers);
        let mut buffers = valid_buffers(plan);
        buffers.destination_bf16.bytes -= 2;
        cases.push(buffers);
        let mut buffers = valid_buffers(plan);
        buffers.source_q5_k.ptr = DevicePtr::NULL;
        cases.push(buffers);
        let mut buffers = valid_buffers(plan);
        buffers.token_ids_u32.ptr.0 += 2;
        cases.push(buffers);
        let mut buffers = valid_buffers(plan);
        buffers.destination_bf16.ptr =
            DevicePtr(u64::MAX - u64::try_from(plan.destination_bytes).unwrap() + 1);
        cases.push(buffers);
        let mut buffers = valid_buffers(plan);
        buffers.token_ids_u32.ptr = DevicePtr(buffers.source_q5_k.ptr.0 + 4);
        cases.push(buffers);
        let mut buffers = valid_buffers(plan);
        buffers.destination_bf16.ptr = buffers.token_ids_u32.ptr;
        cases.push(buffers);
        let mut buffers = valid_buffers(plan);
        buffers.destination_bf16.ptr = DevicePtr(buffers.source_q5_k.ptr.0 + 8);
        cases.push(buffers);
        for buffers in cases {
            assert!(kernel.launch(&gpu, plan, buffers, 7).is_err());
        }
        assert_eq!(gpu.launch_count(), 0);
        kernel.launch(&gpu, plan, valid_buffers(plan), 7).unwrap();
        assert_eq!(gpu.launch_count(), 1);
    }
}
