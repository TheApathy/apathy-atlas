// SPDX-License-Identifier: AGPL-3.0-only

//! Exact manifest contract for GLM-5.3-Flash's EXL3 vision tower.
//!
//! This validates the distinct GLM vision ABI before any device execution is
//! allowed. It deliberately includes the redundant native fused QKV tensors:
//! the packed split Q/K/V projections are canonical for EXL3 execution, but a
//! missing or changed redundant copy still means the pinned checkpoint drifted.

use std::collections::BTreeSet;

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::{Glm53Exl3DeviceStore, Glm53Exl3Dtype, Glm53Exl3Files, Glm53Exl3Linear};

pub const GLM53_EXL3_VISION_BLOCKS: usize = 24;
pub const GLM53_EXL3_VISION_LINEAR_COUNT: usize = 172;
pub const GLM53_EXL3_VISION_RAW_COUNT: usize = 319;
pub const GLM53_EXL3_VISION_PHYSICAL_TENSOR_COUNT: usize =
    GLM53_EXL3_VISION_LINEAR_COUNT * 4 + GLM53_EXL3_VISION_RAW_COUNT;

#[derive(Clone, Debug, PartialEq, Eq)]
struct LinearSpec {
    name: String,
    input: u64,
    output: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RawSpec {
    name: String,
    dtype: Glm53Exl3Dtype,
    shape: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Glm53Exl3VisionRawTensor {
    name: String,
    ptr: DevicePtr,
    dtype: Glm53Exl3Dtype,
    shape: Vec<u64>,
    bytes: usize,
}

impl Glm53Exl3VisionRawTensor {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn ptr(&self) -> DevicePtr {
        self.ptr
    }

    pub fn dtype(&self) -> Glm53Exl3Dtype {
        self.dtype
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// Complete non-owning vision view. The source device store must outlive it.
pub struct Glm53Exl3VisionCatalog {
    linears: std::collections::BTreeMap<String, Glm53Exl3Linear>,
    raw: std::collections::BTreeMap<String, Glm53Exl3VisionRawTensor>,
}

impl Glm53Exl3VisionCatalog {
    pub fn new(files: &Glm53Exl3Files, store: &Glm53Exl3DeviceStore) -> Result<Self> {
        validate_glm53_exl3_vision_manifest(files)?;
        let (linears, raw) = vision_specs();
        let mut bound_linears = std::collections::BTreeMap::new();
        for spec in linears {
            let linear = Glm53Exl3Linear::bind(store, &spec.name)
                .with_context(|| format!("bind GLM EXL3 vision projection {}", spec.name))?;
            ensure!(
                u64::from(linear.size_k()) == spec.input
                    && u64::from(linear.size_n()) == spec.output,
                "GLM EXL3 vision projection geometry drift at {}",
                spec.name
            );
            ensure!(
                bound_linears.insert(spec.name, linear).is_none(),
                "duplicate GLM EXL3 vision projection"
            );
        }
        let mut bound_raw = std::collections::BTreeMap::new();
        for spec in raw {
            let tensor = store
                .get(&spec.name)
                .with_context(|| format!("missing GLM EXL3 vision tensor {}", spec.name))?;
            ensure!(
                tensor.ptr != DevicePtr::NULL
                    && tensor.dtype == spec.dtype
                    && tensor.shape == spec.shape,
                "GLM EXL3 vision tensor pointer, dtype, or shape drift at {}",
                spec.name
            );
            let view = Glm53Exl3VisionRawTensor {
                name: spec.name.clone(),
                ptr: tensor.ptr,
                dtype: tensor.dtype,
                shape: tensor.shape.clone(),
                bytes: tensor.byte_len,
            };
            ensure!(
                bound_raw.insert(spec.name, view).is_none(),
                "duplicate GLM EXL3 vision raw tensor"
            );
        }
        Ok(Self {
            linears: bound_linears,
            raw: bound_raw,
        })
    }

    pub fn linear(&self, name: &str) -> Option<&Glm53Exl3Linear> {
        self.linears.get(name)
    }

    pub fn raw(&self, name: &str) -> Option<&Glm53Exl3VisionRawTensor> {
        self.raw.get(name)
    }

    pub fn linear_count(&self) -> usize {
        self.linears.len()
    }

    pub fn raw_count(&self) -> usize {
        self.raw.len()
    }
}

pub fn validate_glm53_exl3_vision_manifest(files: &Glm53Exl3Files) -> Result<()> {
    let (linears, raw) = vision_specs();
    ensure!(
        linears.len() == GLM53_EXL3_VISION_LINEAR_COUNT && raw.len() == GLM53_EXL3_VISION_RAW_COUNT,
        "GLM EXL3 vision semantic census drift"
    );

    let mut expected = BTreeSet::new();
    for spec in &linears {
        for (suffix, dtype, shape) in [
            ("suh", Glm53Exl3Dtype::F16, vec![spec.input]),
            ("svh", Glm53Exl3Dtype::F16, vec![spec.output]),
            ("mul1", Glm53Exl3Dtype::I32, Vec::new()),
            (
                "trellis",
                Glm53Exl3Dtype::I16,
                vec![spec.input / 16, spec.output / 16, 80],
            ),
        ] {
            let name = format!("{}.{suffix}", spec.name);
            let tensor = files
                .tensor(&name)
                .with_context(|| format!("missing GLM EXL3 vision tensor {name}"))?;
            ensure!(
                tensor.dtype == dtype && tensor.shape == shape,
                "GLM EXL3 vision dtype/shape drift at {name}"
            );
            ensure!(expected.insert(name), "duplicate GLM EXL3 vision tensor");
        }
    }
    for spec in &raw {
        let tensor = files
            .tensor(&spec.name)
            .with_context(|| format!("missing GLM EXL3 vision tensor {}", spec.name))?;
        ensure!(
            tensor.dtype == spec.dtype && tensor.shape == spec.shape,
            "GLM EXL3 vision dtype/shape drift at {}",
            spec.name
        );
        ensure!(
            expected.insert(spec.name.clone()),
            "duplicate GLM EXL3 vision raw tensor"
        );
    }

    ensure!(
        expected.len() == GLM53_EXL3_VISION_PHYSICAL_TENSOR_COUNT,
        "GLM EXL3 vision physical tensor census drift"
    );
    let actual = files
        .tensor_names()
        .filter(|name| name.starts_with("model.visual."))
        .collect::<BTreeSet<_>>();
    ensure!(
        actual.len() == expected.len()
            && expected.iter().all(|name| actual.contains(name.as_str())),
        "GLM EXL3 vision manifest contains unexpected or missing tensors"
    );
    Ok(())
}

fn vision_specs() -> (Vec<LinearSpec>, Vec<RawSpec>) {
    let mut linears = Vec::with_capacity(GLM53_EXL3_VISION_LINEAR_COUNT);
    let mut raw = Vec::with_capacity(GLM53_EXL3_VISION_RAW_COUNT);
    for block in 0..GLM53_EXL3_VISION_BLOCKS {
        let base = format!("model.visual.blocks.{block}");
        for name in ["q_proj", "k_proj", "v_proj", "proj"] {
            linear(&mut linears, format!("{base}.attn.{name}"), 1024, 1024);
            raw_tensor(
                &mut raw,
                format!("{base}.attn.{name}.bias"),
                Glm53Exl3Dtype::F16,
                &[1024],
            );
        }
        for name in ["q_norm", "k_norm"] {
            raw_tensor(
                &mut raw,
                format!("{base}.attn.{name}.weight"),
                Glm53Exl3Dtype::Bf16,
                &[64],
            );
        }
        raw_tensor(
            &mut raw,
            format!("{base}.attn.qkv.weight"),
            Glm53Exl3Dtype::Bf16,
            &[3072, 1024],
        );
        raw_tensor(
            &mut raw,
            format!("{base}.attn.qkv.bias"),
            Glm53Exl3Dtype::Bf16,
            &[3072],
        );

        for (name, input, output) in [
            ("gate_proj", 1024, 4096),
            ("up_proj", 1024, 4096),
            ("down_proj", 4096, 1024),
        ] {
            linear(&mut linears, format!("{base}.mlp.{name}"), input, output);
            raw_tensor(
                &mut raw,
                format!("{base}.mlp.{name}.bias"),
                Glm53Exl3Dtype::F16,
                &[output],
            );
        }
        for name in ["norm1", "norm2"] {
            raw_tensor(
                &mut raw,
                format!("{base}.{name}.weight"),
                Glm53Exl3Dtype::Bf16,
                &[1024],
            );
        }
    }

    for (name, input, output) in [
        ("proj", 4096, 4096),
        ("gate_proj", 4096, 10240),
        ("up_proj", 4096, 10240),
        ("down_proj", 10240, 4096),
    ] {
        linear(
            &mut linears,
            format!("model.visual.merger.{name}"),
            input,
            output,
        );
    }
    for (name, dtype, shape) in [
        (
            "model.visual.downsample.bias",
            Glm53Exl3Dtype::F16,
            vec![4096],
        ),
        (
            "model.visual.downsample.weight",
            Glm53Exl3Dtype::F16,
            vec![4096, 1024, 2, 2],
        ),
        (
            "model.visual.merger.post_projection_norm.bias",
            Glm53Exl3Dtype::F16,
            vec![4096],
        ),
        (
            "model.visual.merger.post_projection_norm.weight",
            Glm53Exl3Dtype::F16,
            vec![4096],
        ),
        (
            "model.visual.patch_embed.proj.bias",
            Glm53Exl3Dtype::F16,
            vec![1024],
        ),
        (
            "model.visual.patch_embed.proj.weight",
            Glm53Exl3Dtype::F16,
            vec![1024, 3, 2, 14, 14],
        ),
        (
            "model.visual.post_layernorm.weight",
            Glm53Exl3Dtype::Bf16,
            vec![1024],
        ),
    ] {
        raw_tensor(&mut raw, name, dtype, &shape);
    }
    (linears, raw)
}

fn linear(specs: &mut Vec<LinearSpec>, name: String, input: u64, output: u64) {
    specs.push(LinearSpec {
        name,
        input,
        output,
    });
}

fn raw_tensor(
    specs: &mut Vec<RawSpec>,
    name: impl Into<String>,
    dtype: Glm53Exl3Dtype,
    shape: &[u64],
) {
    specs.push(RawSpec {
        name: name.into(),
        dtype,
        shape: shape.to_vec(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_vision_generator_is_unique() {
        let (linears, raw) = vision_specs();
        assert_eq!(linears.len(), GLM53_EXL3_VISION_LINEAR_COUNT);
        assert_eq!(raw.len(), GLM53_EXL3_VISION_RAW_COUNT);
        let names = linears
            .iter()
            .map(|spec| spec.name.as_str())
            .chain(raw.iter().map(|spec| spec.name.as_str()))
            .collect::<BTreeSet<_>>();
        assert_eq!(names.len(), linears.len() + raw.len());
        assert!(names.contains("model.visual.blocks.23.attn.q_proj"));
        assert!(names.contains("model.visual.merger.down_proj"));
    }

    #[test]
    #[ignore = "requires the pinned 85 GB one-Spark checkpoint metadata"]
    fn pinned_checkpoint_has_exact_vision_manifest() {
        let root = std::env::var_os("ATLAS_GLM53_EXL3_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .expect("set ATLAS_GLM53_EXL3_CHECKPOINT");
        let files = super::super::admit_glm53_exl3_files(&root).unwrap();
        validate_glm53_exl3_vision_manifest(&files).unwrap();
    }
}
