// SPDX-License-Identifier: AGPL-3.0-only

//! Direct Q4_K token-row gather for the UD-IQ2_XXS GLM-5.3 embedding table.
//!
//! UD-Q2_K_XL holds `token_embd.weight` at Q5_K, which [`super::ggml_q5_embedding`]
//! covers; UD-IQ2_XXS drops it to Q4_K. Without this the IQ2_XXS profile — the
//! only recipe that fits one Spark once resident — cannot embed a token at all.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::GgmlIqBuffer;

pub const GLM53_Q4_EMBEDDING_VOCAB: u32 = 154_880;
pub const GLM53_Q4_EMBEDDING_HIDDEN: u32 = 4_096;
pub const GLM53_Q4_EMBEDDING_MAX_TOKENS: u32 = 65_520;

const Q4_BLOCK_VALUES: usize = 256;
const Q4_BLOCK_BYTES: usize = 144;
const BF16_BYTES: usize = 2;
const U32_BYTES: usize = 4;
const THREADS: u32 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgmlQ4EmbeddingPlan {
    pub tokens: u32,
    pub vocab: u32,
    pub hidden: u32,
    pub source_bytes: usize,
    pub token_ids_bytes: usize,
    pub destination_bytes: usize,
}

impl GgmlQ4EmbeddingPlan {
    pub fn new(tokens: u32, vocab: u32, hidden: u32) -> Result<Self> {
        if tokens == 0 || tokens > GLM53_Q4_EMBEDDING_MAX_TOKENS {
            bail!("GLM Q4_K embedding tokens must be in 1..=65,520");
        }
        if vocab != GLM53_Q4_EMBEDDING_VOCAB || hidden != GLM53_Q4_EMBEDDING_HIDDEN {
            bail!("GLM Q4_K embedding requires exact vocab154880/hidden4096 geometry");
        }
        let vocab = usize::try_from(vocab)?;
        let hidden = usize::try_from(hidden)?;
        let tokens = usize::try_from(tokens)?;
        if !hidden.is_multiple_of(Q4_BLOCK_VALUES) {
            bail!("GLM Q4_K embedding hidden width must be divisible by 256");
        }
        let source_blocks = vocab
            .checked_mul(hidden / Q4_BLOCK_VALUES)
            .context("GLM Q4_K embedding source block count overflow")?;
        let source_bytes = source_blocks
            .checked_mul(Q4_BLOCK_BYTES)
            .context("GLM Q4_K embedding source byte count overflow")?;
        let token_ids_bytes = tokens
            .checked_mul(U32_BYTES)
            .context("GLM Q4_K embedding token byte count overflow")?;
        let destination_bytes = tokens
            .checked_mul(hidden)
            .and_then(|elements| elements.checked_mul(BF16_BYTES))
            .context("GLM Q4_K embedding destination byte count overflow")?;
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
            bail!("GLM Q4_K embedding plan fields were forged");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GgmlQ4EmbeddingBuffers {
    /// Exact `[154880, 4096]` Q4_K table (144 bytes per 256 values).
    pub source_q4_k: GgmlIqBuffer,
    /// Caller-validated device token IDs, exact `[tokens]` `u32`.
    pub token_ids_u32: GgmlIqBuffer,
    /// Exact `[tokens, 4096]` BF16 output; no persistent expanded table.
    pub destination_bf16: GgmlIqBuffer,
}

pub struct GgmlQ4EmbeddingKernel {
    gather: KernelHandle,
}

impl GgmlQ4EmbeddingKernel {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            gather: gpu.kernel("ggml_q4_embedding", "atlas_q4_k_embedding_gather_bf16")?,
        })
    }

    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        plan: GgmlQ4EmbeddingPlan,
        buffers: GgmlQ4EmbeddingBuffers,
        stream: u64,
    ) -> Result<()> {
        plan.validate()?;
        validate_buffers(plan, buffers)?;
        KernelLaunch::new(gpu, self.gather)
            .grid([plan.tokens, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.source_q4_k.ptr)
            .arg_ptr(buffers.token_ids_u32.ptr)
            .arg_ptr(buffers.destination_bf16.ptr)
            .arg_u32(plan.tokens)
            .arg_u32(plan.vocab)
            .arg_u32(plan.hidden)
            .launch(stream)
    }
}

fn validate_buffers(plan: GgmlQ4EmbeddingPlan, buffers: GgmlQ4EmbeddingBuffers) -> Result<()> {
    let named = [
        ("Q4_K source", buffers.source_q4_k, plan.source_bytes, 4u64),
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
            bail!("GLM Q4_K embedding {name} is null or has the wrong extent");
        }
        if buffer.ptr.0 % alignment != 0 {
            bail!("GLM Q4_K embedding {name} address is misaligned");
        }
        let end = buffer
            .ptr
            .0
            .checked_add(u64::try_from(buffer.bytes)?)
            .with_context(|| format!("GLM Q4_K embedding {name} address overflow"))?;
        ranges.push((buffer.ptr.0, end));
    }
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].0 < ranges[right].1 && ranges[right].0 < ranges[left].1 {
                bail!("GLM Q4_K embedding device buffers overlap");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::{bf16, f16};
    use spark_runtime::gpu::mock::MockGpuBackend;

    const CUDA_SOURCE: &str =
        include_str!("../../../../../kernels/gb10/glm5.3-flash/iq3/ggml_q4_embedding.cu");

    fn buffer(address: u64, bytes: usize) -> GgmlIqBuffer {
        GgmlIqBuffer {
            ptr: DevicePtr(address),
            bytes,
        }
    }

    fn valid_buffers(plan: GgmlQ4EmbeddingPlan) -> GgmlQ4EmbeddingBuffers {
        GgmlQ4EmbeddingBuffers {
            source_q4_k: buffer(0x10_0000, plan.source_bytes),
            token_ids_u32: buffer(0x2000_0000, plan.token_ids_bytes),
            destination_bf16: buffer(0x3000_0000, plan.destination_bytes),
        }
    }

    /// Q4_K packs eight 6-bit scale/min pairs into twelve bytes, exactly as
    /// Q5_K does. Mirrors `get_scale_min_k4`.
    fn pack_scales(scales: [u8; 8], mins: [u8; 8]) -> [u8; 12] {
        let mut packed = [0u8; 12];
        for group in 0..4usize {
            packed[group] = scales[group] & 63;
            packed[group + 4] = mins[group] & 63;
        }
        for group in 4..8usize {
            packed[group + 4] = (scales[group] & 0x0f) | ((mins[group] & 0x0f) << 4);
            packed[group - 4] |= (scales[group] >> 4) << 6;
            packed[group] |= (mins[group] >> 4) << 6;
        }
        packed
    }

    /// Host mirror of `dequantize_row_q4_K`. Q4_K is Q5_K without the `qh`
    /// plane: same 6-bit scale/min packing, same chunk/half/lane walk, and the
    /// quantized value is the bare nibble.
    fn reference_dequant(block: &[u8; 144]) -> [bf16; 256] {
        let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let dmin = f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
        let packed: [u8; 12] = block[4..16].try_into().unwrap();
        let mut output = [bf16::ZERO; 256];
        for column in 0..256usize {
            let chunk = column / 64;
            let half = (column / 32) & 1;
            let lane = column & 31;
            let group = 2 * chunk + half;
            let (scale, minimum) = if group < 4 {
                (packed[group] & 63, packed[group + 4] & 63)
            } else {
                (
                    (packed[group + 4] & 0x0f) | ((packed[group - 4] >> 6) << 4),
                    (packed[group + 4] >> 4) | ((packed[group] >> 6) << 4),
                )
            };
            // Q4_K quants start at byte 16; Q5_K's `qh` plane occupies 16..48
            // and pushes its quants to 48. Using the wrong base silently reads
            // the neighbouring plane.
            let packed_quant = block[16 + 32 * chunk + lane];
            let quant = if half == 0 {
                packed_quant & 0x0f
            } else {
                packed_quant >> 4
            };
            output[column] =
                bf16::from_f32(d * f32::from(scale) * f32::from(quant) - dmin * f32::from(minimum));
        }
        output
    }

    #[test]
    fn exact_glm_geometry_has_no_persistent_bf16_expansion() {
        let one = GgmlQ4EmbeddingPlan::new(1, 154_880, 4_096).unwrap();
        // 154,880 rows x 16 blocks x 144 bytes. This is also exactly the
        // `output.weight` census entry for Q4_K in the pinned schema.
        assert_eq!(one.source_bytes, 356_843_520);
        assert_eq!(one.token_ids_bytes, 4);
        assert_eq!(one.destination_bytes, 8_192);
        let max = GgmlQ4EmbeddingPlan::new(65_520, 154_880, 4_096).unwrap();
        assert_eq!(max.source_bytes, one.source_bytes);
        assert_eq!(max.token_ids_bytes, 262_080);
        assert_eq!(max.destination_bytes, 536_739_840);
        assert!(GgmlQ4EmbeddingPlan::new(0, 154_880, 4_096).is_err());
        assert!(GgmlQ4EmbeddingPlan::new(1, 154_879, 4_096).is_err());
        assert!(GgmlQ4EmbeddingPlan::new(1, 154_880, 4_095).is_err());
    }

    #[test]
    fn reference_pins_scale_min_and_nibble_chronology() {
        let scales = [1u8, 2, 3, 4, 17, 34, 51, 63];
        let mins = [5u8, 6, 7, 8, 21, 38, 55, 62];
        let mut block = [0u8; 144];
        block[0..2].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&f16::from_f32(0.25).to_bits().to_le_bytes());
        block[4..16].copy_from_slice(&pack_scales(scales, mins));
        let mut logical = [0u8; 256];
        for chunk in 0..4usize {
            for lane in 0..32usize {
                let low = u8::try_from((3 * chunk + lane) % 16).unwrap();
                let high = u8::try_from((5 * chunk + 2 * lane + 1) % 16).unwrap();
                logical[64 * chunk + lane] = low;
                logical[64 * chunk + 32 + lane] = high;
                block[16 + 32 * chunk + lane] = low | (high << 4);
            }
        }
        let actual = reference_dequant(&block);
        for column in 0..256usize {
            let group = column / 32;
            let expected = bf16::from_f32(
                0.5 * f32::from(scales[group]) * f32::from(logical[column])
                    - 0.25 * f32::from(mins[group]),
            );
            assert_eq!(
                actual[column].to_bits(),
                expected.to_bits(),
                "column {column}"
            );
        }
        // Low and high nibbles of the same byte must land 32 columns apart, and
        // group boundaries must actually change the scale.
        assert_ne!(actual[0].to_bits(), actual[32].to_bits());
        assert_ne!(actual[127].to_bits(), actual[128].to_bits());
        assert_ne!(actual[223].to_bits(), actual[224].to_bits());
    }

    #[test]
    fn cuda_source_is_a_direct_row_gather_with_the_pinned_equation() {
        for required in [
            "atlas_q4_k_embedding_gather_bf16",
            "sizeof(block_q4_K) == 144",
            "const unsigned int token = token_ids[output_row]",
            "(unsigned long long) token * blocks_per_row + column / QK_K",
            "block->qs[chunk * 32u + lane]",
            "block->dm.x",
            "block->dm.y",
            "d * (float) scale * (float) quant - dmin * (float) minimum",
            "__float2bfloat16_rn(value)",
        ] {
            assert!(
                CUDA_SOURCE.contains(required),
                "missing CUDA contract: {required}"
            );
        }
        // Q4_K has no high-bit plane; referencing one would read Q5_K layout.
        assert!(!CUDA_SOURCE.contains("qh["));
        assert!(CUDA_SOURCE.contains("if (token >= vocab)"));
        assert!(CUDA_SOURCE.contains("output[column] = __float2bfloat16_rn(0.0f)"));
    }

    #[test]
    fn launch_hostiles_fail_before_enqueue() {
        let gpu = MockGpuBackend::new();
        let kernel = GgmlQ4EmbeddingKernel::load(&gpu).unwrap();
        let plan = GgmlQ4EmbeddingPlan::new(1, 154_880, 4_096).unwrap();

        let mut buffers = valid_buffers(plan);
        buffers.source_q4_k.bytes -= 1;
        assert!(kernel.launch(&gpu, plan, buffers, 0).is_err());

        let mut buffers = valid_buffers(plan);
        buffers.source_q4_k.ptr = DevicePtr::NULL;
        assert!(kernel.launch(&gpu, plan, buffers, 0).is_err());

        // Overlap between the table and its outputs is silent corruption.
        let mut buffers = valid_buffers(plan);
        buffers.token_ids_u32.ptr = DevicePtr(buffers.source_q4_k.ptr.0 + 4);
        assert!(kernel.launch(&gpu, plan, buffers, 0).is_err());

        let mut buffers = valid_buffers(plan);
        buffers.destination_bf16.ptr = DevicePtr(buffers.source_q4_k.ptr.0 + 8);
        assert!(kernel.launch(&gpu, plan, buffers, 0).is_err());

        assert!(kernel.launch(&gpu, plan, valid_buffers(plan), 0).is_ok());
    }
}
