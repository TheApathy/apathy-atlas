// SPDX-License-Identifier: AGPL-3.0-only

use super::geometry::Geometry;
use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

pub(super) struct WeightSpec {
    pub name: String,
    pub shape: Vec<usize>,
}

pub(super) fn validate_tensor(spec: &WeightSpec, tensor: &WeightTensor) -> Result<()> {
    ensure!(
        tensor.dtype == WeightDtype::BF16 && tensor.shape == spec.shape && !tensor.ptr.is_null(),
        "{}: expected native BF16 {:?}, got {:?} {:?} at {}",
        spec.name,
        spec.shape,
        tensor.dtype,
        tensor.shape,
        tensor.ptr
    );
    Ok(())
}

#[derive(Clone, Copy)]
pub(super) struct Linear {
    pub weight: DevicePtr,
    pub bias: DevicePtr,
    pub n: usize,
    pub k: usize,
}

pub(super) struct Block {
    pub norm1: DevicePtr,
    pub qkv: Linear,
    pub proj: Linear,
    pub norm2: DevicePtr,
    pub fc1: Linear,
    pub fc2: Linear,
}

pub(super) struct Weights {
    pub patch: Linear,
    pub blocks: Vec<Block>,
    pub norm: DevicePtr,
    pub align1: Linear,
    pub align2: Linear,
    pub special: [DevicePtr; 5],
}

impl Weights {
    pub fn load(store: &WeightStore, g: &Geometry, depth: usize) -> Result<Self> {
        // No allocation, reinterpretation, dequantization, or data mutation.
        let tensor = |name: &str, shape: &[usize]| -> Result<DevicePtr> {
            let spec = WeightSpec {
                name: name.into(),
                shape: shape.into(),
            };
            let value = store
                .get(name)
                .with_context(|| format!("missing DeepSeek vision tensor {name}"))?;
            validate_tensor(&spec, value)?;
            Ok(value.ptr)
        };
        let linear = |prefix: &str, n: usize, k: usize, bias: bool| -> Result<Linear> {
            Ok(Linear {
                weight: tensor(&format!("{prefix}.weight"), &[n, k])?,
                bias: if bias {
                    tensor(&format!("{prefix}.bias"), &[n])?
                } else {
                    DevicePtr(0)
                },
                n,
                k,
            })
        };
        let mut blocks = Vec::with_capacity(depth);
        for layer in 0..depth {
            let p = format!("vision.blocks.{layer}");
            blocks.push(Block {
                norm1: tensor(&format!("{p}.norm1.weight"), &[g.hidden])?,
                qkv: linear(&format!("{p}.attn.wqkv"), 3 * g.hidden, g.hidden, true)?,
                proj: linear(&format!("{p}.attn.wo"), g.hidden, g.hidden, true)?,
                norm2: tensor(&format!("{p}.norm2.weight"), &[g.hidden])?,
                fc1: linear(&format!("{p}.mlp.w1"), 2 * g.intermediate, g.hidden, false)?,
                fc2: linear(&format!("{p}.mlp.w2"), g.hidden, g.intermediate, false)?,
            });
        }
        let pad = tensor("image_pad", &[g.text_hidden])?;
        Ok(Self {
            patch: linear("vision.patch_embed.proj", g.hidden, g.patch_dim, true)?,
            blocks,
            norm: tensor("vision.norm.weight", &[g.hidden])?,
            align1: linear(
                "aligner.w1",
                g.text_hidden,
                g.hidden * g.ratio * g.ratio,
                true,
            )?,
            align2: linear("aligner.w2", g.text_hidden, g.text_hidden, true)?,
            special: [
                tensor("image_start", &[g.text_hidden])?,
                pad,
                pad,
                tensor("image_newline", &[g.text_hidden])?,
                tensor("image_end", &[g.text_hidden])?,
            ],
        })
    }
}
