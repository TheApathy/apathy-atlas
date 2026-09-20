// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed admission for the pinned one-Spark GLM-5.3 EXL3 checkpoint.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;

pub const GLM53_EXL3_TENSOR_COUNT: usize = 151_554;
pub const GLM53_EXL3_DATA_BYTES: u64 = 85_128_745_176;
pub const GLM53_EXL3_LEDGER_ENTRIES: usize = 37_032;
pub const GLM53_EXL3_QUANTIZED_ENTRIES: usize = 36_547;

const SHARDS: usize = 12;
const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
const MUL1_MULTIPLIER: u64 = 2_212_286_765;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Glm53Exl3Dtype {
    Bf16,
    F16,
    F32,
    I16,
    I32,
}

impl Glm53Exl3Dtype {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "BF16" => Ok(Self::Bf16),
            "F16" => Ok(Self::F16),
            "F32" => Ok(Self::F32),
            "I16" => Ok(Self::I16),
            "I32" => Ok(Self::I32),
            other => bail!("GLM EXL3 unsupported physical dtype {other}"),
        }
    }

    pub const fn byte_size(self) -> u64 {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::Bf16 | Self::F16 | Self::I16 => 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Glm53Exl3Admission {
    pub tensor_count: usize,
    pub data_bytes: u64,
    pub ledger_entries: usize,
    pub quantized_entries: usize,
    pub dtype_counts: BTreeMap<Glm53Exl3Dtype, usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Glm53Exl3TensorInfo {
    pub dtype: Glm53Exl3Dtype,
    pub shape: Vec<u64>,
    pub shard: usize,
    pub data_offset: u64,
    pub byte_len: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Glm53Exl3ShardInfo {
    pub file_name: String,
    pub data_start: u64,
    pub data_bytes: u64,
}

/// Exact admitted checkpoint metadata retained for raw device loading.
pub struct Glm53Exl3Files {
    pub(super) root: PathBuf,
    pub(super) admission: Glm53Exl3Admission,
    pub(super) shards: Vec<Glm53Exl3ShardInfo>,
    pub(super) tensors: BTreeMap<String, Glm53Exl3TensorInfo>,
}

impl Glm53Exl3Files {
    pub fn admission(&self) -> &Glm53Exl3Admission {
        &self.admission
    }

    pub fn shards(&self) -> &[Glm53Exl3ShardInfo] {
        &self.shards
    }

    pub fn tensor(&self, name: &str) -> Option<&Glm53Exl3TensorInfo> {
        self.tensors.get(name)
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
}

#[derive(serde::Deserialize)]
struct SafetensorsIndex {
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug)]
struct HeaderTensor {
    name: String,
    dtype: Glm53Exl3Dtype,
    shape: Vec<u64>,
    start: u64,
    end: u64,
    elements: u64,
}

#[derive(Debug)]
struct ParsedShard {
    data_start: u64,
    data_bytes: u64,
    tensors: Vec<HeaderTensor>,
}

/// Admit only the exact EXL3 2.05 ABI used by the published one-Spark run.
pub fn admit_glm53_exl3_checkpoint(root: &Path) -> Result<Glm53Exl3Admission> {
    Ok(admit_glm53_exl3_files(root)?.admission)
}

/// Admit and retain the exact file/tensor routes needed for byte-preserving loading.
pub fn admit_glm53_exl3_files(root: &Path) -> Result<Glm53Exl3Files> {
    let quant_path = root.join("quantization_config.json");
    let quant: Value = serde_json::from_reader(
        File::open(&quant_path).with_context(|| format!("open {}", quant_path.display()))?,
    )
    .with_context(|| format!("parse {}", quant_path.display()))?;
    let (ledger_entries, quantized_entries) = validate_quantization(&quant)?;

    let index_path = root.join("model.safetensors.index.json");
    let index: SafetensorsIndex = serde_json::from_reader(
        File::open(&index_path).with_context(|| format!("open {}", index_path.display()))?,
    )
    .with_context(|| format!("parse {}", index_path.display()))?;
    ensure!(
        index.weight_map.len() == GLM53_EXL3_TENSOR_COUNT,
        "GLM EXL3 index tensor census drift"
    );

    let expected_shards = (1..=SHARDS)
        .map(|part| format!("model-{part:05}-of-{SHARDS:05}.safetensors"))
        .collect::<BTreeSet<_>>();
    let indexed_shards = index.weight_map.values().cloned().collect::<BTreeSet<_>>();
    ensure!(
        indexed_shards == expected_shards,
        "GLM EXL3 shard-name census drift"
    );

    let mut seen = BTreeSet::new();
    let mut dtype_counts = BTreeMap::new();
    let mut suffix_counts = BTreeMap::<String, usize>::new();
    let mut data_bytes = 0u64;
    let mut admitted_shards = Vec::with_capacity(SHARDS);
    let mut admitted_tensors = BTreeMap::new();
    for (shard_index, shard) in expected_shards.into_iter().enumerate() {
        let path = root.join(&shard);
        let parsed = read_header(&path)?;
        admitted_shards.push(Glm53Exl3ShardInfo {
            file_name: shard.clone(),
            data_start: parsed.data_start,
            data_bytes: parsed.data_bytes,
        });
        for tensor in parsed.tensors {
            ensure!(
                seen.insert(tensor.name.clone()),
                "duplicate EXL3 tensor name"
            );
            ensure!(
                index.weight_map.get(&tensor.name) == Some(&shard),
                "GLM EXL3 index/header route mismatch for {}",
                tensor.name
            );
            let bytes = tensor
                .elements
                .checked_mul(tensor.dtype.byte_size())
                .context("GLM EXL3 tensor byte overflow")?;
            ensure!(
                tensor.end - tensor.start == bytes,
                "GLM EXL3 tensor extent/dtype mismatch for {}",
                tensor.name
            );
            data_bytes = data_bytes
                .checked_add(bytes)
                .context("GLM EXL3 checkpoint byte overflow")?;
            *dtype_counts.entry(tensor.dtype).or_default() += 1;
            if let Some(suffix) = tensor.name.rsplit('.').next() {
                *suffix_counts.entry(suffix.to_owned()).or_default() += 1;
            }
            admitted_tensors.insert(
                tensor.name,
                Glm53Exl3TensorInfo {
                    dtype: tensor.dtype,
                    shape: tensor.shape,
                    shard: shard_index,
                    data_offset: tensor.start,
                    byte_len: bytes,
                },
            );
        }
    }
    ensure!(
        seen.len() == index.weight_map.len(),
        "GLM EXL3 header/index census drift"
    );
    ensure!(
        data_bytes == GLM53_EXL3_DATA_BYTES,
        "GLM EXL3 data-byte census drift"
    );
    ensure!(
        dtype_counts == expected_dtype_counts(),
        "GLM EXL3 dtype census drift"
    );
    for suffix in ["trellis", "suh", "svh", "mul1"] {
        ensure!(
            suffix_counts.get(suffix) == Some(&37_592),
            "GLM EXL3 {suffix} tensor census drift"
        );
    }
    Ok(Glm53Exl3Files {
        root: root.to_path_buf(),
        admission: Glm53Exl3Admission {
            tensor_count: seen.len(),
            data_bytes,
            ledger_entries,
            quantized_entries,
            dtype_counts,
        },
        shards: admitted_shards,
        tensors: admitted_tensors,
    })
}

fn validate_quantization(value: &Value) -> Result<(usize, usize)> {
    ensure!(value["quant_method"] == "exl3", "GLM target is not EXL3");
    ensure!(
        value["bits"].as_f64() == Some(2.05),
        "GLM EXL3 bit rate drift"
    );
    ensure!(value["head_bits"] == 5, "GLM EXL3 head bit rate drift");
    ensure!(value["mtp_bits"] == 2, "GLM EXL3 MTP bit rate drift");
    ensure!(value["vision_bits"] == 5, "GLM EXL3 vision bit rate drift");
    ensure!(value["codebook"] == "mul1", "GLM EXL3 codebook drift");
    ensure!(value["version"] == "1.4.4", "GLM EXL3 version drift");
    ensure!(
        value["out_scales"] == "always",
        "GLM EXL3 output-scale policy drift"
    );
    ensure!(
        value["calibration"]["rows"] == 250 && value["calibration"]["cols"] == 2048,
        "GLM EXL3 calibration geometry drift"
    );
    let ledger = value["tensor_storage"]
        .as_object()
        .context("GLM EXL3 tensor_storage must be an object")?;
    ensure!(
        ledger.len() == GLM53_EXL3_LEDGER_ENTRIES,
        "GLM EXL3 ledger census drift"
    );
    let mut quantized = 0usize;
    for (logical, entry) in ledger {
        if entry["quant_format"] != "exl3" {
            continue;
        }
        quantized += 1;
        ensure!(
            entry["mul1_multiplier"].as_u64() == Some(MUL1_MULTIPLIER),
            "GLM EXL3 mul1 multiplier drift at {logical}"
        );
        let stored = entry["stored_tensors"]
            .as_object()
            .with_context(|| format!("GLM EXL3 stored_tensors missing at {logical}"))?;
        ensure!(
            stored.len() == 4,
            "GLM EXL3 packed tensor count drift at {logical}"
        );
        for (suffix, dtype) in [
            ("trellis", "torch.int16"),
            ("suh", "torch.float16"),
            ("svh", "torch.float16"),
            ("mul1", "torch.int32"),
        ] {
            let (_, tensor) = stored
                .iter()
                .find(|(name, _)| name.rsplit('.').next() == Some(suffix))
                .with_context(|| format!("GLM EXL3 missing {suffix} at {logical}"))?;
            ensure!(
                tensor["dtype"] == dtype,
                "GLM EXL3 {suffix} dtype drift at {logical}"
            );
        }
    }
    ensure!(
        quantized == GLM53_EXL3_QUANTIZED_ENTRIES,
        "GLM EXL3 quantized ledger drift"
    );
    Ok((ledger.len(), quantized))
}

fn read_header(path: &Path) -> Result<ParsedShard> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let file_bytes = file.metadata()?.len();
    let mut size = [0u8; 8];
    file.read_exact(&mut size)?;
    let header_bytes = u64::from_le_bytes(size);
    ensure!(
        header_bytes <= MAX_HEADER_BYTES,
        "GLM EXL3 safetensors header too large"
    );
    let mut raw = vec![0u8; usize::try_from(header_bytes)?];
    file.read_exact(&mut raw)?;
    let header: Value = serde_json::from_slice(&raw)?;
    let object = header
        .as_object()
        .context("GLM EXL3 safetensors header is not an object")?;
    let mut tensors = Vec::with_capacity(object.len());
    for (name, info) in object {
        if name == "__metadata__" {
            continue;
        }
        let dtype = Glm53Exl3Dtype::parse(
            info["dtype"]
                .as_str()
                .context("GLM EXL3 tensor dtype missing")?,
        )?;
        let shape = info["shape"]
            .as_array()
            .context("GLM EXL3 tensor shape missing")?
            .iter()
            .map(|dimension| dimension.as_u64().context("GLM EXL3 dimension is not u64"))
            .collect::<Result<Vec<_>>>()?;
        let elements = shape.iter().try_fold(1u64, |count, dimension| {
            count
                .checked_mul(*dimension)
                .context("GLM EXL3 element count overflow")
        })?;
        let offsets = info["data_offsets"]
            .as_array()
            .context("GLM EXL3 tensor offsets missing")?;
        ensure!(offsets.len() == 2, "GLM EXL3 tensor offset arity drift");
        let start = offsets[0]
            .as_u64()
            .context("GLM EXL3 start offset is not u64")?;
        let end = offsets[1]
            .as_u64()
            .context("GLM EXL3 end offset is not u64")?;
        ensure!(end >= start, "GLM EXL3 tensor extent is reversed");
        tensors.push(HeaderTensor {
            name: name.clone(),
            dtype,
            shape,
            start,
            end,
            elements,
        });
    }
    tensors.sort_by_key(|tensor| tensor.start);
    let mut cursor = 0u64;
    for tensor in &tensors {
        ensure!(
            tensor.start == cursor,
            "GLM EXL3 shard tensor extents have a gap or overlap"
        );
        cursor = tensor.end;
    }
    ensure!(
        8 + header_bytes + cursor == file_bytes,
        "GLM EXL3 shard terminal extent does not match file size"
    );
    Ok(ParsedShard {
        data_start: 8 + header_bytes,
        data_bytes: cursor,
        tensors,
    })
}

fn expected_dtype_counts() -> BTreeMap<Glm53Exl3Dtype, usize> {
    BTreeMap::from([
        (Glm53Exl3Dtype::Bf16, 334),
        (Glm53Exl3Dtype::F16, 75_643),
        (Glm53Exl3Dtype::F32, 393),
        (Glm53Exl3Dtype::I16, 37_592),
        (Glm53Exl3Dtype::I32, 37_592),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_dtypes_keep_exl3_storage_semantics() {
        assert_eq!(Glm53Exl3Dtype::parse("F16").unwrap().byte_size(), 2);
        assert_eq!(Glm53Exl3Dtype::parse("I16").unwrap().byte_size(), 2);
        assert_eq!(Glm53Exl3Dtype::parse("I32").unwrap().byte_size(), 4);
        assert!(Glm53Exl3Dtype::parse("BF16_CONVERTED_FROM_F16").is_err());
        assert_eq!(
            expected_dtype_counts().values().sum::<usize>(),
            GLM53_EXL3_TENSOR_COUNT
        );
    }

    #[test]
    fn quant_identity_rejects_nearby_formats_before_ledger_use() {
        let mut value = serde_json::json!({"quant_method":"nvfp4"});
        assert!(validate_quantization(&value).is_err());
        value["quant_method"] = Value::String("exl3".into());
        value["bits"] = Value::from(2.0);
        assert!(validate_quantization(&value).is_err());
    }

    #[test]
    #[ignore = "requires the pinned 85 GB one-Spark checkpoint"]
    fn pinned_local_checkpoint_passes_complete_admission() {
        let root = std::env::var_os("ATLAS_GLM53_EXL3_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .expect("set ATLAS_GLM53_EXL3_CHECKPOINT");
        let admitted = admit_glm53_exl3_checkpoint(&root).unwrap();
        assert_eq!(admitted.tensor_count, GLM53_EXL3_TENSOR_COUNT);
        assert_eq!(admitted.data_bytes, GLM53_EXL3_DATA_BYTES);
    }
}
