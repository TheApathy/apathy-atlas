// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 mHC contraction into the five DFlash2 target-capture slots.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::GgmlIqBuffer;

const HIDDEN: u32 = 4096;
const HC_STREAMS: u32 = 4;
const CAPTURE_SLOTS: u32 = 5;
const THREADS: u32 = 256;
const MAX_GRID_X: u64 = 2_147_483_647;
// Zero-based walk indices for checkpoint IDs [5,14,24,33,42].
const TARGET_LAYERS: [u32; 5] = [4, 13, 23, 32, 41];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53Dflash2CapturePlan {
    pub batch: u32,
    pub tokens: u32,
    pub rows: u32,
    pub streams_bytes: usize,
    pub captures_bytes: usize,
}

impl Glm53Dflash2CapturePlan {
    pub fn new(batch: u32, tokens: u32, hidden: u32, hc_streams: u32) -> Result<Self> {
        if batch == 0 || tokens == 0 {
            bail!("GLM DFlash2 target capture requires nonzero batch and tokens");
        }
        if hidden != HIDDEN || hc_streams != HC_STREAMS {
            bail!("GLM DFlash2 target capture requires exact H4096/hc4 geometry");
        }
        let rows = u64::from(batch)
            .checked_mul(u64::from(tokens))
            .context("GLM DFlash2 target-capture row overflow")?;
        if rows > MAX_GRID_X {
            bail!("GLM DFlash2 target-capture grid exceeds the CUDA limit");
        }
        let rows_u32 = u32::try_from(rows).context("GLM DFlash2 target-capture grid overflow")?;
        let bytes = |width: u64| -> Result<usize> {
            let elements = rows
                .checked_mul(width)
                .context("GLM DFlash2 target-capture element overflow")?;
            usize::try_from(elements)?
                .checked_mul(2)
                .context("GLM DFlash2 target-capture byte overflow")
        };
        Ok(Self {
            batch,
            tokens,
            rows: rows_u32,
            streams_bytes: bytes(u64::from(HC_STREAMS) * u64::from(HIDDEN))?,
            captures_bytes: bytes(u64::from(CAPTURE_SLOTS) * u64::from(HIDDEN))?,
        })
    }

    fn validate(self) -> Result<()> {
        let expected = Self::new(self.batch, self.tokens, HIDDEN, HC_STREAMS)?;
        if self != expected {
            bail!("GLM DFlash2 target-capture plan fields were forged");
        }
        Ok(())
    }

    pub fn slot_for_post_layer(self, post_layer: u32) -> Result<u32> {
        let slot = TARGET_LAYERS
            .iter()
            .position(|&layer| layer == post_layer)
            .context("GLM DFlash2 target capture is not a configured post-layer")?;
        Ok(u32::try_from(slot)?)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53Dflash2CaptureBuffers {
    /// `[batch, tokens, 4, 4096]` BF16 post-layer mHC state.
    pub streams_bf16: GgmlIqBuffer,
    /// `[batch, tokens, 5, 4096]` BF16, populated one capture slot at a time.
    pub captures_bf16: GgmlIqBuffer,
}

pub struct Glm53Dflash2CaptureKernel {
    contract_mean: KernelHandle,
}

impl Glm53Dflash2CaptureKernel {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            contract_mean: gpu
                .kernel("glm53_dflash2_capture", "atlas_glm53_dflash2_capture_mean")?,
        })
    }

    pub fn capture_post_layer(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53Dflash2CapturePlan,
        post_layer: u32,
        buffers: Glm53Dflash2CaptureBuffers,
        stream: u64,
    ) -> Result<()> {
        plan.validate()?;
        let slot = plan.slot_for_post_layer(post_layer)?;
        validate_buffers(plan, buffers)?;
        KernelLaunch::new(gpu, self.contract_mean)
            .grid([plan.rows, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.streams_bf16.ptr)
            .arg_ptr(buffers.captures_bf16.ptr)
            .arg_u32(plan.batch)
            .arg_u32(plan.tokens)
            .arg_u32(HIDDEN)
            .arg_u32(HC_STREAMS)
            .arg_u32(post_layer)
            .arg_u32(slot)
            .launch(stream)
    }
}

fn validate_buffers(
    plan: Glm53Dflash2CapturePlan,
    buffers: Glm53Dflash2CaptureBuffers,
) -> Result<()> {
    let named = [
        ("mHC streams", buffers.streams_bf16, plan.streams_bytes),
        ("captures", buffers.captures_bf16, plan.captures_bytes),
    ];
    // Sized from the table it mirrors, never a literal: a hand-written
    // length silently goes out of bounds the moment the table grows.
    let mut ranges = Vec::with_capacity(named.len());
    for (name, buffer, expected) in named.iter().copied() {
        if buffer.ptr == DevicePtr::NULL || buffer.bytes != expected {
            bail!("GLM DFlash2 target-capture {name} is null or has the wrong extent");
        }
        let end = buffer
            .ptr
            .0
            .checked_add(u64::try_from(buffer.bytes)?)
            .with_context(|| format!("GLM DFlash2 target-capture {name} address overflow"))?;
        ranges.push((buffer.ptr.0, end));
    }
    if ranges[0].0 < ranges[1].1 && ranges[1].0 < ranges[0].1 {
        bail!("GLM DFlash2 target-capture input and output overlap");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use spark_runtime::gpu::mock::MockGpuBackend;

    use super::*;

    fn ordered_mean(values: [bf16; 4]) -> bf16 {
        let mut sum = 0.0f32;
        for value in values {
            sum += value.to_f32();
        }
        bf16::from_f32(sum * 0.25)
    }

    #[test]
    fn exact_geometry_layer_mapping_and_mean_order_are_pinned() {
        let plan = Glm53Dflash2CapturePlan::new(2, 8, 4096, 4).unwrap();
        assert_eq!(plan.rows, 16);
        assert_eq!(plan.streams_bytes, 2 * 8 * 4 * 4096 * 2);
        assert_eq!(plan.captures_bytes, 2 * 8 * 5 * 4096 * 2);
        for (slot, layer) in TARGET_LAYERS.into_iter().enumerate() {
            assert_eq!(
                plan.slot_for_post_layer(layer).unwrap(),
                u32::try_from(slot).unwrap()
            );
        }
        for layer in [0, 3, 5, 12, 14, 22, 24, 31, 33, 40, 42] {
            assert!(plan.slot_for_post_layer(layer).is_err());
        }
        assert!(Glm53Dflash2CapturePlan::new(0, 1, 4096, 4).is_err());
        assert!(Glm53Dflash2CapturePlan::new(1, 1, 4096, 1).is_err());
        assert!(Glm53Dflash2CapturePlan::new(1, u32::MAX, 4096, 4).is_err());

        let values = [
            bf16::from_bits(0x31e0),
            bf16::from_bits(0xd79d),
            bf16::from_bits(0x6313),
            bf16::from_bits(0x6380),
        ];
        let balanced = bf16::from_f32(
            ((values[0].to_f32() + values[1].to_f32()) + (values[2].to_f32() + values[3].to_f32()))
                * 0.25,
        );
        assert_eq!(ordered_mean(values).to_bits(), 0x62ca);
        assert_eq!(balanced.to_bits(), 0x62c9);
    }

    #[test]
    fn all_five_layers_launch_and_faults_are_rejected_before_effect() {
        let gpu = MockGpuBackend::new();
        let kernel = Glm53Dflash2CaptureKernel::load(&gpu).unwrap();
        let plan = Glm53Dflash2CapturePlan::new(1, 8, 4096, 4).unwrap();
        let streams = GgmlIqBuffer {
            ptr: DevicePtr(0x10_0000),
            bytes: plan.streams_bytes,
        };
        let captures = GgmlIqBuffer {
            ptr: DevicePtr(0x20_0000),
            bytes: plan.captures_bytes,
        };
        let buffers = Glm53Dflash2CaptureBuffers {
            streams_bf16: streams,
            captures_bf16: captures,
        };
        assert!(
            kernel
                .capture_post_layer(
                    &gpu,
                    Glm53Dflash2CapturePlan { rows: 1, ..plan },
                    4,
                    buffers,
                    0,
                )
                .is_err()
        );
        assert!(
            kernel
                .capture_post_layer(&gpu, plan, 14, buffers, 0)
                .is_err()
        );
        assert!(
            kernel
                .capture_post_layer(
                    &gpu,
                    plan,
                    4,
                    Glm53Dflash2CaptureBuffers {
                        captures_bf16: GgmlIqBuffer {
                            ptr: streams.ptr,
                            bytes: plan.captures_bytes,
                        },
                        ..buffers
                    },
                    0,
                )
                .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);
        for layer in TARGET_LAYERS {
            kernel
                .capture_post_layer(&gpu, plan, layer, buffers, 0)
                .unwrap();
        }
        assert_eq!(gpu.launch_count(), 5);
    }
}
