// SPDX-License-Identifier: AGPL-3.0-only

//! Materialized native operands for the existing GLM-5.3 execution graph.
//!
//! EXL3 projection quartets stay in their compressed store. Only checkpoint
//! tensors that are native are converted, and only when the graph's existing
//! kernel ABI requires a different 16/32-bit floating representation.

use std::collections::BTreeMap;
use std::fmt;

use anyhow::{Context, Result, anyhow, ensure};
use half::{bf16, f16};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{Glm53Exl3Dtype, Glm53Exl3RawTensor, Glm53Exl3TargetCatalog};

const ALIGNMENT: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Glm53Exl3NativeDtype {
    Bf16,
    F32,
}

impl Glm53Exl3NativeDtype {
    const fn byte_size(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Glm53Exl3NativeTensor {
    name: String,
    ptr: DevicePtr,
    dtype: Glm53Exl3NativeDtype,
    shape: Vec<u64>,
    bytes: usize,
    materialized: bool,
}

impl Glm53Exl3NativeTensor {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn ptr(&self) -> DevicePtr {
        self.ptr
    }

    pub fn dtype(&self) -> Glm53Exl3NativeDtype {
        self.dtype
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn is_materialized(&self) -> bool {
        self.materialized
    }

    /// A checked contiguous row view of a rank-2 BF16 matrix.
    ///
    /// This owns no device memory. Callers build and retain these views during
    /// model construction so token-time dispatch performs no name lookup or
    /// allocation.
    pub fn bf16_matrix_rows(&self, first_row: u64, rows: u64) -> Result<Self> {
        ensure!(
            self.dtype == Glm53Exl3NativeDtype::Bf16 && self.shape.len() == 2,
            "GLM EXL3 native row view requires a rank-2 BF16 matrix"
        );
        ensure!(rows != 0, "GLM EXL3 native row view must be non-empty");
        let end_row = first_row
            .checked_add(rows)
            .context("GLM EXL3 native row-view range overflow")?;
        ensure!(
            end_row <= self.shape[0],
            "GLM EXL3 native row view ends at {end_row}, past {} rows",
            self.shape[0]
        );
        let row_bytes = usize::try_from(self.shape[1])?
            .checked_mul(Glm53Exl3NativeDtype::Bf16.byte_size())
            .context("GLM EXL3 native row byte count overflow")?;
        let offset = usize::try_from(first_row)?
            .checked_mul(row_bytes)
            .context("GLM EXL3 native row offset overflow")?;
        let bytes = usize::try_from(rows)?
            .checked_mul(row_bytes)
            .context("GLM EXL3 native row extent overflow")?;
        let end = offset
            .checked_add(bytes)
            .context("GLM EXL3 native row-view extent overflow")?;
        ensure!(
            end <= self.bytes,
            "GLM EXL3 native row view exceeds the backing tensor"
        );
        Ok(Self {
            name: self.name.clone(),
            ptr: DevicePtr(
                self.ptr
                    .0
                    .checked_add(u64::try_from(offset)?)
                    .context("GLM EXL3 native row address overflow")?,
            ),
            dtype: self.dtype,
            shape: vec![rows, self.shape[1]],
            bytes,
            materialized: self.materialized,
        })
    }
}

#[must_use = "materialized EXL3 native allocations require explicit free"]
pub struct Glm53Exl3NativeStore {
    tensors: BTreeMap<String, Glm53Exl3NativeTensor>,
    slab: DevicePtr,
    slab_bytes: usize,
}

#[must_use = "failed EXL3 native materialization may retain a device allocation"]
pub struct Glm53Exl3NativeLoadError {
    primary: anyhow::Error,
    retained: Option<(DevicePtr, usize)>,
}

impl Glm53Exl3NativeStore {
    pub fn get(&self, name: &str) -> Option<&Glm53Exl3NativeTensor> {
        self.tensors.get(name)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn materialized_bytes(&self) -> usize {
        self.slab_bytes
    }

    pub fn materialized_count(&self) -> usize {
        self.tensors
            .values()
            .filter(|tensor| tensor.materialized)
            .count()
    }

    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        if self.slab == DevicePtr::NULL {
            return Ok(());
        }
        gpu.free(self.slab)
            .context("free GLM EXL3 native materialization slab")
    }
}

impl Glm53Exl3NativeLoadError {
    pub fn failure(&self) -> &anyhow::Error {
        &self.primary
    }

    pub fn retained_bytes(&self) -> usize {
        self.retained.map_or(0, |(_, bytes)| bytes)
    }

    pub fn retry_cleanup(self, gpu: &dyn GpuBackend) -> std::result::Result<anyhow::Error, Self> {
        let Some((ptr, bytes)) = self.retained else {
            return Ok(self.primary);
        };
        match gpu.free(ptr) {
            Ok(()) => Ok(self.primary),
            Err(cleanup) => Err(Self {
                primary: self
                    .primary
                    .context(format!("GLM EXL3 native cleanup also failed: {cleanup:#}")),
                retained: Some((ptr, bytes)),
            }),
        }
    }
}

impl fmt::Debug for Glm53Exl3NativeLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Glm53Exl3NativeLoadError")
            .field("primary", &self.primary)
            .field("retained_bytes", &self.retained_bytes())
            .finish()
    }
}

impl fmt::Display for Glm53Exl3NativeLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "GLM EXL3 native materialization failed: {:#}",
            self.primary
        )
    }
}

/// Build graph-compatible views, numerically converting only mismatched dtypes.
pub fn materialize_glm53_exl3_native(
    catalog: &Glm53Exl3TargetCatalog,
    gpu: &dyn GpuBackend,
) -> std::result::Result<Glm53Exl3NativeStore, Glm53Exl3NativeLoadError> {
    match materialize(catalog, gpu) {
        Ok(store) => Ok(store),
        Err((primary, retained)) => Err(Glm53Exl3NativeLoadError { primary, retained }),
    }
}

fn materialize(
    catalog: &Glm53Exl3TargetCatalog,
    gpu: &dyn GpuBackend,
) -> std::result::Result<Glm53Exl3NativeStore, (anyhow::Error, Option<(DevicePtr, usize)>)> {
    let mut layout = BTreeMap::new();
    let mut cursor = 0usize;
    for name in catalog.raw_names() {
        let tensor = catalog.raw(name).expect("catalog name must resolve");
        let target = target_dtype(tensor);
        if source_matches(tensor.dtype(), target) && !requires_folded_kda_a_log(tensor.name()) {
            continue;
        }
        cursor = align_up(cursor).map_err(|error| (error, None))?;
        let bytes = element_count(tensor)
            .and_then(|elements| {
                elements
                    .checked_mul(target.byte_size())
                    .context("GLM EXL3 native output byte overflow")
            })
            .map_err(|error| (error, None))?;
        layout.insert(name.to_owned(), (cursor, bytes, target));
        cursor = cursor
            .checked_add(bytes)
            .ok_or_else(|| (anyhow!("GLM EXL3 native slab overflow"), None))?;
    }

    let slab = if cursor == 0 {
        DevicePtr::NULL
    } else {
        gpu.alloc(cursor).map_err(|error| (error, None))?
    };
    let retained = || (slab != DevicePtr::NULL).then_some((slab, cursor));
    let mut tensors = BTreeMap::new();
    for name in catalog.raw_names() {
        let source = catalog.raw(name).expect("catalog name must resolve");
        let target = target_dtype(source);
        let (ptr, bytes, materialized) = if let Some(&(offset, bytes, _)) = layout.get(name) {
            let ptr = DevicePtr(
                slab.0
                    .checked_add(u64::try_from(offset).map_err(|error| {
                        (anyhow!(error).context("GLM EXL3 native offset"), retained())
                    })?)
                    .ok_or_else(|| (anyhow!("GLM EXL3 native address overflow"), retained()))?,
            );
            let mut input = vec![0u8; source.bytes()];
            gpu.copy_d2h(source.ptr(), &mut input)
                .map_err(|error| (error, retained()))?;
            let output = convert_for_graph(name, &input, source.dtype(), target)
                .map_err(|error| (error, retained()))?;
            if output.len() != bytes {
                return Err((
                    anyhow!("GLM EXL3 native conversion extent drift at {name}"),
                    retained(),
                ));
            }
            gpu.copy_h2d(&output, ptr)
                .map_err(|error| (error, retained()))?;
            (ptr, bytes, true)
        } else {
            (source.ptr(), source.bytes(), false)
        };
        let view = Glm53Exl3NativeTensor {
            name: name.to_owned(),
            ptr,
            dtype: target,
            shape: source.shape().to_vec(),
            bytes,
            materialized,
        };
        if tensors.insert(name.to_owned(), view).is_some() {
            return Err((anyhow!("duplicate GLM EXL3 native tensor"), retained()));
        }
    }
    if tensors.len() != super::GLM53_EXL3_TARGET_RAW_COUNT {
        return Err((
            anyhow!("GLM EXL3 native materialization tensor census drift"),
            retained(),
        ));
    }
    Ok(Glm53Exl3NativeStore {
        tensors,
        slab,
        slab_bytes: cursor,
    })
}

fn target_dtype(tensor: &Glm53Exl3RawTensor) -> Glm53Exl3NativeDtype {
    if tensor.name() == "model.language_model.embed_tokens.weight" {
        return Glm53Exl3NativeDtype::Bf16;
    }
    if tensor.dtype() == Glm53Exl3Dtype::F32 || requires_f32(tensor.name()) {
        Glm53Exl3NativeDtype::F32
    } else {
        Glm53Exl3NativeDtype::Bf16
    }
}

fn requires_f32(name: &str) -> bool {
    name == "model.language_model.norm.weight"
        || name.ends_with("input_layernorm.weight")
        || name.ends_with("post_attention_layernorm.weight")
        || name.ends_with("o_norm.weight")
        || name.ends_with("q_a_layernorm.weight")
        || name.ends_with("kv_a_layernorm.weight")
        || name.ends_with("indexer.k_norm.weight")
        || name.ends_with("indexer.k_norm.bias")
        || name.ends_with("indexer.weights_proj.weight")
        || name.ends_with("conv1d.weight")
        || name.ends_with("mlp.gate.weight")
}

fn requires_folded_kda_a_log(name: &str) -> bool {
    name.starts_with("model.language_model.layers.") && name.ends_with(".self_attn.A_log")
}

fn source_matches(source: Glm53Exl3Dtype, target: Glm53Exl3NativeDtype) -> bool {
    matches!(
        (source, target),
        (Glm53Exl3Dtype::Bf16, Glm53Exl3NativeDtype::Bf16)
            | (Glm53Exl3Dtype::F32, Glm53Exl3NativeDtype::F32)
    )
}

fn align_up(value: usize) -> Result<usize> {
    value
        .checked_add(ALIGNMENT - 1)
        .map(|value| value & !(ALIGNMENT - 1))
        .context("GLM EXL3 native alignment overflow")
}

fn element_count(tensor: &Glm53Exl3RawTensor) -> Result<usize> {
    tensor.shape().iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(usize::try_from(*dimension)?)
            .context("GLM EXL3 native element count overflow")
    })
}

fn convert(input: &[u8], source: Glm53Exl3Dtype, target: Glm53Exl3NativeDtype) -> Result<Vec<u8>> {
    let source_width = usize::try_from(source.byte_size())?;
    ensure!(
        input.len() % source_width == 0,
        "GLM EXL3 native input extent is not element aligned"
    );
    let mut output = Vec::with_capacity(input.len() / source_width * target.byte_size());
    for chunk in input.chunks_exact(source_width) {
        let value = match source {
            Glm53Exl3Dtype::Bf16 => {
                bf16::from_bits(u16::from_le_bytes(chunk.try_into().expect("two bytes"))).to_f32()
            }
            Glm53Exl3Dtype::F16 => {
                f16::from_bits(u16::from_le_bytes(chunk.try_into().expect("two bytes"))).to_f32()
            }
            Glm53Exl3Dtype::F32 => f32::from_le_bytes(chunk.try_into().expect("four bytes")),
            Glm53Exl3Dtype::I16 | Glm53Exl3Dtype::I32 => {
                return Err(anyhow!("integer EXL3 tensor cannot be a native operand"));
            }
        };
        match target {
            Glm53Exl3NativeDtype::Bf16 => {
                output.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes())
            }
            Glm53Exl3NativeDtype::F32 => output.extend_from_slice(&value.to_le_bytes()),
        }
    }
    Ok(output)
}

fn convert_for_graph(
    name: &str,
    input: &[u8],
    source: Glm53Exl3Dtype,
    target: Glm53Exl3NativeDtype,
) -> Result<Vec<u8>> {
    let mut output = convert(input, source, target)?;
    if !requires_folded_kda_a_log(name) {
        return Ok(output);
    }
    ensure!(
        source == Glm53Exl3Dtype::F32 && target == Glm53Exl3NativeDtype::F32,
        "GLM EXL3 KDA A_log folding requires F32 input and output"
    );
    for word in output.chunks_exact_mut(4) {
        let raw = f32::from_le_bytes(word.try_into().expect("four bytes"));
        word.copy_from_slice(&(-raw.exp()).to_le_bytes());
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_conversion_is_not_a_bitcast() {
        let f16_input = [f16::from_f32(1.5).to_bits().to_le_bytes()].concat();
        let bf16_output =
            convert(&f16_input, Glm53Exl3Dtype::F16, Glm53Exl3NativeDtype::Bf16).unwrap();
        assert_eq!(
            u16::from_le_bytes(bf16_output.as_slice().try_into().unwrap()),
            bf16::from_f32(1.5).to_bits()
        );
        assert_ne!(f16_input, bf16_output);

        let f32_output =
            convert(&f16_input, Glm53Exl3Dtype::F16, Glm53Exl3NativeDtype::F32).unwrap();
        assert_eq!(f32::from_le_bytes(f32_output.try_into().unwrap()), 1.5);
    }

    #[test]
    fn policy_keeps_only_embedding_bf16_and_expands_graph_f32_operands() {
        assert!(!requires_f32("model.language_model.embed_tokens.weight"));
        assert!(requires_f32(
            "model.language_model.layers.3.mlp.gate.weight"
        ));
        assert!(requires_f32(
            "model.language_model.layers.0.self_attn.conv1d.weight"
        ));
        assert!(requires_f32(
            "model.language_model.layers.3.self_attn.indexer.k_norm.bias"
        ));
        assert!(requires_f32(
            "model.language_model.layers.3.self_attn.indexer.weights_proj.weight"
        ));
        assert!(!requires_f32(
            "model.language_model.layers.0.self_attn.f_a_proj.weight"
        ));
        assert!(requires_folded_kda_a_log(
            "model.language_model.layers.0.self_attn.A_log"
        ));
        assert!(!requires_folded_kda_a_log(
            "model.language_model.layers.3.self_attn.indexer.weights_proj.weight"
        ));
    }

    #[test]
    fn raw_kda_a_log_is_folded_to_the_shared_kernel_representation() {
        let raw = 2.0f32.ln().to_le_bytes();
        let output = convert_for_graph(
            "model.language_model.layers.0.self_attn.A_log",
            &raw,
            Glm53Exl3Dtype::F32,
            Glm53Exl3NativeDtype::F32,
        )
        .unwrap();
        assert_eq!(f32::from_le_bytes(output.try_into().unwrap()), -2.0);

        let untouched = convert_for_graph(
            "model.language_model.layers.3.self_attn.indexer.weights_proj.weight",
            &raw,
            Glm53Exl3Dtype::F32,
            Glm53Exl3NativeDtype::F32,
        )
        .unwrap();
        assert_eq!(
            f32::from_le_bytes(untouched.try_into().unwrap()),
            2.0f32.ln()
        );
    }

    #[test]
    fn alignment_and_integer_refusal_are_fail_closed() {
        assert_eq!(align_up(0).unwrap(), 0);
        assert_eq!(align_up(1).unwrap(), 256);
        assert_eq!(align_up(257).unwrap(), 512);
        assert!(convert(&[0, 0], Glm53Exl3Dtype::I16, Glm53Exl3NativeDtype::Bf16).is_err());
    }
}
