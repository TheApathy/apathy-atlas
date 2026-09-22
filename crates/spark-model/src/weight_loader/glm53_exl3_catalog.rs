// SPDX-License-Identifier: AGPL-3.0-only

//! Exact semantic catalog for the target-language portion of GLM-5.3 EXL3.
//!
//! Layer 45 is the native NextN/MTP block and is deliberately excluded. The
//! DFlash2 path consumes target layers 0..44 plus the shared embedding, final
//! norm, and LM head.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::DevicePtr;

use super::{Glm53Exl3DeviceStore, Glm53Exl3Dtype, Glm53Exl3Files, Glm53Exl3Linear};

pub const GLM53_EXL3_TARGET_LINEAR_COUNT: usize = 36_547;
pub const GLM53_EXL3_TARGET_RAW_COUNT: usize = 851;
pub const GLM53_EXL3_TARGET_PHYSICAL_TENSOR_COUNT: usize =
    GLM53_EXL3_TARGET_LINEAR_COUNT * 4 + GLM53_EXL3_TARGET_RAW_COUNT;

const TARGET_LAYERS: u32 = 45;
const EXPERTS: u32 = 288;

#[derive(Clone, Debug, PartialEq, Eq)]
struct LinearSpec {
    name: String,
    input: u32,
    output: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RawSpec {
    name: String,
    dtype: Glm53Exl3Dtype,
    shape: Vec<u64>,
}

/// Non-owning typed view of one native target tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Glm53Exl3RawTensor {
    name: String,
    ptr: DevicePtr,
    dtype: Glm53Exl3Dtype,
    shape: Vec<u64>,
    bytes: usize,
}

impl Glm53Exl3RawTensor {
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

/// Complete target-only device catalog. The source store must outlive it.
pub struct Glm53Exl3TargetCatalog {
    linears: BTreeMap<String, Glm53Exl3Linear>,
    raw: BTreeMap<String, Glm53Exl3RawTensor>,
}

impl Glm53Exl3TargetCatalog {
    /// Admit the exact file manifest and bind every target device view.
    pub fn new(files: &Glm53Exl3Files, store: &Glm53Exl3DeviceStore) -> Result<Self> {
        let (linears, raw) = target_specs()?;
        validate_specs(files, &linears, &raw)?;

        let mut bound_linears = BTreeMap::new();
        for spec in linears {
            let linear = Glm53Exl3Linear::bind(store, &spec.name)
                .with_context(|| format!("bind target EXL3 projection {}", spec.name))?;
            ensure!(
                linear.size_k() == spec.input && linear.size_n() == spec.output,
                "GLM EXL3 target projection geometry drift at {}",
                spec.name
            );
            ensure!(
                bound_linears.insert(spec.name, linear).is_none(),
                "duplicate GLM EXL3 target projection"
            );
        }

        let mut bound_raw = BTreeMap::new();
        for spec in raw {
            let tensor = store
                .get(&spec.name)
                .with_context(|| format!("missing target EXL3 tensor {}", spec.name))?;
            ensure!(
                tensor.ptr != DevicePtr::NULL
                    && tensor.dtype == spec.dtype
                    && tensor.shape == spec.shape,
                "GLM EXL3 target tensor pointer, dtype, or shape drift at {}",
                spec.name
            );
            let view = Glm53Exl3RawTensor {
                name: spec.name.clone(),
                ptr: tensor.ptr,
                dtype: tensor.dtype,
                shape: tensor.shape.clone(),
                bytes: tensor.byte_len,
            };
            ensure!(
                bound_raw.insert(spec.name, view).is_none(),
                "duplicate GLM EXL3 target native tensor"
            );
        }
        Ok(Self {
            linears: bound_linears,
            raw: bound_raw,
        })
    }

    pub fn linear(&self, logical_name: &str) -> Option<&Glm53Exl3Linear> {
        self.linears.get(logical_name)
    }

    pub fn raw(&self, name: &str) -> Option<&Glm53Exl3RawTensor> {
        self.raw.get(name)
    }

    pub fn raw_names(&self) -> impl Iterator<Item = &str> {
        self.raw.keys().map(String::as_str)
    }

    pub fn linear_count(&self) -> usize {
        self.linears.len()
    }

    pub fn raw_count(&self) -> usize {
        self.raw.len()
    }
}

/// Validate all target-language names/dtypes/shapes without allocating a GPU store.
pub fn validate_glm53_exl3_target_manifest(files: &Glm53Exl3Files) -> Result<()> {
    let (linears, raw) = target_specs()?;
    validate_specs(files, &linears, &raw)
}

fn validate_specs(files: &Glm53Exl3Files, linears: &[LinearSpec], raw: &[RawSpec]) -> Result<()> {
    ensure!(
        linears.len() == GLM53_EXL3_TARGET_LINEAR_COUNT && raw.len() == GLM53_EXL3_TARGET_RAW_COUNT,
        "GLM EXL3 target semantic census drift"
    );
    let mut expected = BTreeSet::new();
    for spec in linears {
        for (suffix, dtype, shape) in [
            ("suh", Glm53Exl3Dtype::F16, vec![u64::from(spec.input)]),
            ("svh", Glm53Exl3Dtype::F16, vec![u64::from(spec.output)]),
            ("mul1", Glm53Exl3Dtype::I32, vec![]),
        ] {
            let name = format!("{}.{suffix}", spec.name);
            let tensor = files
                .tensor(&name)
                .with_context(|| format!("missing target EXL3 manifest tensor {name}"))?;
            ensure!(
                tensor.dtype == dtype && tensor.shape == shape,
                "GLM EXL3 target manifest dtype/shape drift at {name}"
            );
            ensure!(expected.insert(name), "duplicate target EXL3 manifest name");
        }
        let trellis = format!("{}.trellis", spec.name);
        let tensor = files
            .tensor(&trellis)
            .with_context(|| format!("missing target EXL3 manifest tensor {trellis}"))?;
        ensure!(
            tensor.dtype == Glm53Exl3Dtype::I16
                && tensor.shape.len() == 3
                && tensor.shape[0] == u64::from(spec.input / 16)
                && tensor.shape[1] == u64::from(spec.output / 16)
                && matches!(tensor.shape[2], 32 | 48 | 64 | 80),
            "GLM EXL3 target trellis geometry drift at {trellis}"
        );
        ensure!(
            expected.insert(trellis),
            "duplicate target EXL3 trellis name"
        );
    }
    for spec in raw {
        let tensor = files
            .tensor(&spec.name)
            .with_context(|| format!("missing target EXL3 manifest tensor {}", spec.name))?;
        ensure!(
            tensor.dtype == spec.dtype && tensor.shape == spec.shape,
            "GLM EXL3 target native dtype/shape drift at {}",
            spec.name
        );
        ensure!(
            expected.insert(spec.name.clone()),
            "duplicate target EXL3 native manifest name"
        );
    }
    ensure!(
        expected.len() == GLM53_EXL3_TARGET_PHYSICAL_TENSOR_COUNT,
        "GLM EXL3 target physical tensor census drift"
    );

    let actual = files
        .tensor_names()
        .filter(|name| is_target_tensor(name))
        .collect::<BTreeSet<_>>();
    ensure!(
        actual.len() == expected.len(),
        "GLM EXL3 target manifest contains unexpected or missing tensors"
    );
    if let Some(name) = expected.iter().find(|name| !actual.contains(name.as_str())) {
        bail!("GLM EXL3 target manifest is missing expected tensor {name}");
    }
    if let Some(name) = actual.iter().find(|name| !expected.contains(**name)) {
        bail!("GLM EXL3 target manifest has unexpected tensor {name}");
    }
    Ok(())
}

fn is_target_tensor(name: &str) -> bool {
    if matches!(
        name,
        "model.language_model.embed_tokens.weight" | "model.language_model.norm.weight"
    ) || name.starts_with("lm_head.")
    {
        return true;
    }
    let Some(rest) = name.strip_prefix("model.language_model.layers.") else {
        return false;
    };
    let Some((layer, _)) = rest.split_once('.') else {
        return false;
    };
    layer
        .parse::<u32>()
        .is_ok_and(|layer| layer < TARGET_LAYERS)
}

fn target_specs() -> Result<(Vec<LinearSpec>, Vec<RawSpec>)> {
    let mut linears = Vec::with_capacity(GLM53_EXL3_TARGET_LINEAR_COUNT);
    let mut raw = Vec::with_capacity(GLM53_EXL3_TARGET_RAW_COUNT);
    raw_spec(
        &mut raw,
        "model.language_model.embed_tokens.weight",
        Glm53Exl3Dtype::Bf16,
        &[154_880, 4_096],
    );
    raw_spec(
        &mut raw,
        "model.language_model.norm.weight",
        Glm53Exl3Dtype::Bf16,
        &[4_096],
    );
    linear_spec(&mut linears, "lm_head", 4_096, 154_880);

    for layer in 0..TARGET_LAYERS {
        let root = format!("model.language_model.layers.{layer}");
        raw_spec(
            &mut raw,
            format!("{root}.input_layernorm.weight"),
            Glm53Exl3Dtype::Bf16,
            &[4_096],
        );
        raw_spec(
            &mut raw,
            format!("{root}.post_attention_layernorm.weight"),
            Glm53Exl3Dtype::Bf16,
            &[4_096],
        );
        for branch in ["attn", "ffn"] {
            raw_spec(
                &mut raw,
                format!("{root}.hc_{branch}_base"),
                Glm53Exl3Dtype::F32,
                &[24],
            );
            raw_spec(
                &mut raw,
                format!("{root}.hc_{branch}_fn"),
                Glm53Exl3Dtype::F32,
                &[24, 16_384],
            );
            raw_spec(
                &mut raw,
                format!("{root}.hc_{branch}_scale"),
                Glm53Exl3Dtype::F32,
                &[3],
            );
        }

        if layer % 4 == 3 {
            dsa_specs(&root, &mut linears, &mut raw);
        } else {
            kda_specs(&root, &mut linears, &mut raw);
        }
        if layer < 3 {
            dense_ffn_specs(&root, &mut linears);
        } else {
            moe_specs(&root, &mut linears, &mut raw);
        }
    }

    ensure!(
        linears.len() == GLM53_EXL3_TARGET_LINEAR_COUNT && raw.len() == GLM53_EXL3_TARGET_RAW_COUNT,
        "GLM EXL3 target catalog generator count drift"
    );
    Ok((linears, raw))
}

fn kda_specs(root: &str, linears: &mut Vec<LinearSpec>, raw: &mut Vec<RawSpec>) {
    let attn = format!("{root}.self_attn");
    linear_spec(linears, format!("{attn}.qkv_proj"), 4_096, 24_576);
    linear_spec(linears, format!("{attn}.o_proj"), 8_192, 4_096);
    for (name, shape, dtype) in [
        ("A_log", vec![64], Glm53Exl3Dtype::F32),
        ("b_proj.weight", vec![64, 4_096], Glm53Exl3Dtype::F16),
        ("conv1d.weight", vec![24_576, 1, 4], Glm53Exl3Dtype::Bf16),
        ("dt_bias", vec![8_192], Glm53Exl3Dtype::F32),
        ("f_a_proj.weight", vec![128, 4_096], Glm53Exl3Dtype::F16),
        ("f_b_proj.weight", vec![8_192, 128], Glm53Exl3Dtype::F16),
        ("g_a_proj.weight", vec![128, 4_096], Glm53Exl3Dtype::F16),
        ("g_b_proj.weight", vec![8_192, 128], Glm53Exl3Dtype::F16),
        ("o_norm.weight", vec![128], Glm53Exl3Dtype::Bf16),
    ] {
        raw_spec(raw, format!("{attn}.{name}"), dtype, &shape);
    }
}

fn dsa_specs(root: &str, linears: &mut Vec<LinearSpec>, raw: &mut Vec<RawSpec>) {
    let attn = format!("{root}.self_attn");
    for (name, input, output) in [
        ("q_a_proj", 4_096, 1_536),
        ("q_b_proj", 1_536, 16_384),
        ("kv_a_proj_with_mqa", 4_096, 512),
        ("o_proj", 16_384, 4_096),
        ("indexer.wq_b", 1_536, 4_096),
    ] {
        linear_spec(linears, format!("{attn}.{name}"), input, output);
    }
    for (name, shape, dtype) in [
        ("q_a_layernorm.weight", vec![1_536], Glm53Exl3Dtype::Bf16),
        ("kv_a_layernorm.weight", vec![512], Glm53Exl3Dtype::Bf16),
        ("kv_b_proj.weight", vec![32_768, 512], Glm53Exl3Dtype::F16),
        (
            "indexer.weights_proj.weight",
            vec![32, 4_096],
            Glm53Exl3Dtype::F16,
        ),
        ("indexer.wk.weight", vec![128, 4_096], Glm53Exl3Dtype::F16),
        ("indexer.k_norm.weight", vec![128], Glm53Exl3Dtype::F16),
        ("indexer.k_norm.bias", vec![128], Glm53Exl3Dtype::F16),
        (
            "indexer.index_kpool_compress_ape",
            vec![4, 128],
            Glm53Exl3Dtype::F32,
        ),
        (
            "indexer.index_kpool_compress_gate",
            vec![128, 4_096],
            Glm53Exl3Dtype::F16,
        ),
    ] {
        raw_spec(raw, format!("{attn}.{name}"), dtype, &shape);
    }
}

fn dense_ffn_specs(root: &str, linears: &mut Vec<LinearSpec>) {
    let mlp = format!("{root}.mlp");
    linear_spec(linears, format!("{mlp}.gate_proj"), 4_096, 12_288);
    linear_spec(linears, format!("{mlp}.up_proj"), 4_096, 12_288);
    linear_spec(linears, format!("{mlp}.down_proj"), 12_288, 4_096);
}

fn moe_specs(root: &str, linears: &mut Vec<LinearSpec>, raw: &mut Vec<RawSpec>) {
    let mlp = format!("{root}.mlp");
    raw_spec(
        raw,
        format!("{mlp}.gate.weight"),
        Glm53Exl3Dtype::F16,
        &[288, 4_096],
    );
    raw_spec(
        raw,
        format!("{mlp}.gate.e_score_correction_bias"),
        Glm53Exl3Dtype::F32,
        &[288],
    );
    for expert in 0..EXPERTS {
        let base = format!("{mlp}.experts.{expert}");
        linear_spec(linears, format!("{base}.gate_proj"), 4_096, 2_048);
        linear_spec(linears, format!("{base}.up_proj"), 4_096, 2_048);
        linear_spec(linears, format!("{base}.down_proj"), 2_048, 4_096);
    }
    let shared = format!("{mlp}.shared_experts");
    linear_spec(linears, format!("{shared}.gate_proj"), 4_096, 2_048);
    linear_spec(linears, format!("{shared}.up_proj"), 4_096, 2_048);
    linear_spec(linears, format!("{shared}.down_proj"), 2_048, 4_096);
}

fn linear_spec(specs: &mut Vec<LinearSpec>, name: impl Into<String>, input: u32, output: u32) {
    specs.push(LinearSpec {
        name: name.into(),
        input,
        output,
    });
}

fn raw_spec(
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
    fn exact_target_generator_is_unique_and_excludes_nextn() {
        let (linears, raw) = target_specs().unwrap();
        assert_eq!(linears.len(), GLM53_EXL3_TARGET_LINEAR_COUNT);
        assert_eq!(raw.len(), GLM53_EXL3_TARGET_RAW_COUNT);
        let names = linears
            .iter()
            .map(|spec| spec.name.as_str())
            .chain(raw.iter().map(|spec| spec.name.as_str()))
            .collect::<BTreeSet<_>>();
        assert_eq!(names.len(), linears.len() + raw.len());
        assert!(names.contains("model.language_model.layers.3.self_attn.q_a_proj"));
        assert!(names.contains("model.language_model.layers.44.self_attn.qkv_proj"));
        assert!(!names.iter().any(|name| name.contains("layers.45.")));
    }

    #[test]
    fn target_boundary_is_exact() {
        assert!(is_target_tensor("model.language_model.layers.0.x"));
        assert!(is_target_tensor("model.language_model.layers.44.x"));
        assert!(!is_target_tensor("model.language_model.layers.45.x"));
        assert!(is_target_tensor("lm_head.trellis"));
        assert!(!is_target_tensor("model.visual.layers.0.x"));
    }

    #[test]
    #[ignore = "requires the pinned 85 GB one-Spark checkpoint metadata"]
    fn pinned_checkpoint_has_exact_target_manifest() {
        let root = std::env::var_os("ATLAS_GLM53_EXL3_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .expect("set ATLAS_GLM53_EXL3_CHECKPOINT");
        let files = super::super::admit_glm53_exl3_files(&root).unwrap();
        validate_glm53_exl3_target_manifest(&files).unwrap();
    }
}
