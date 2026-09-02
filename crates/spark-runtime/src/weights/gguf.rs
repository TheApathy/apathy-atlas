// SPDX-License-Identifier: AGPL-3.0-only

//! Bounded GGUF-v3 directory types and reader. Tensor payloads are never read here.

use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::fmt;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1_M,
    BF16,
}

impl GgmlType {
    pub(super) fn from_raw(raw: u32) -> Result<Self> {
        Ok(match raw {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1_M,
            30 => Self::BF16,
            _ => bail!("unsupported GGML tensor type {raw}"),
        })
    }

    fn layout(self) -> (u64, u64) {
        match self {
            Self::F32 | Self::I32 => (1, 4),
            Self::F16 | Self::BF16 | Self::I16 => (1, 2),
            Self::I8 => (1, 1),
            Self::I64 | Self::F64 => (1, 8),
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 36),
            Self::Q2_K => (256, 84),
            Self::Q3_K | Self::IQ3_S => (256, 110),
            Self::Q4_K => (256, 144),
            Self::Q5_K => (256, 176),
            Self::Q6_K => (256, 210),
            Self::Q8_K => (256, 292),
            Self::IQ2_XXS => (256, 66),
            Self::IQ2_XS => (256, 74),
            Self::IQ3_XXS => (256, 98),
            Self::IQ1_S => (256, 50),
            Self::IQ4_NL => (32, 18),
            Self::IQ2_S => (256, 82),
            Self::IQ4_XS => (256, 136),
            Self::IQ1_M => (256, 56),
        }
    }

    pub(super) fn block_size(self) -> u64 {
        self.layout().0
    }

    pub(super) fn byte_len(self, elements: u64) -> Result<u64> {
        let (block, bytes) = self.layout();
        if !elements.is_multiple_of(block) {
            bail!("GGML {self:?} tensor element count {elements} violates block size {block}");
        }
        elements
            .checked_div(block)
            .and_then(|blocks| blocks.checked_mul(bytes))
            .context("GGUF tensor byte length overflow")
    }
}

#[derive(Clone, PartialEq)]
pub enum GgufValue {
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Array { element_type: u32, len: u64 },
}

impl fmt::Debug for GgufValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsigned(value) => formatter.debug_tuple("Unsigned").field(value).finish(),
            Self::Signed(value) => formatter.debug_tuple("Signed").field(value).finish(),
            Self::Float(value) => formatter.debug_tuple("Float").field(value).finish(),
            Self::Bool(value) => formatter.debug_tuple("Bool").field(value).finish(),
            Self::String(value) => write!(formatter, "String(<{} bytes>)", value.len()),
            Self::Array { element_type, len } => formatter
                .debug_struct("Array")
                .field("element_type", element_type)
                .field("len", len)
                .finish(),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: GgmlType,
    pub offset: u64,
    pub byte_len: u64,
}

impl fmt::Debug for GgufTensorInfo {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GgufTensorInfo")
            .field("name_bytes", &self.name.len())
            .field("dimensions", &self.dimensions)
            .field("ggml_type", &self.ggml_type)
            .field("offset", &self.offset)
            .field("byte_len", &self.byte_len)
            .finish()
    }
}

#[derive(Clone, PartialEq)]
pub struct GgufHeader {
    pub version: u32,
    pub alignment: u64,
    pub data_offset: u64,
    pub file_len: u64,
    pub metadata: BTreeMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
}

impl fmt::Debug for GgufHeader {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GgufHeader")
            .field("version", &self.version)
            .field("alignment", &self.alignment)
            .field("data_offset", &self.data_offset)
            .field("file_len", &self.file_len)
            .field("metadata_entries", &self.metadata.len())
            .field("tensors", &self.tensors.len())
            .finish()
    }
}

mod decode;
mod device;
mod glm53;
mod payload;
mod sha256;
mod shards;
mod values;
pub use decode::read_gguf_header;
pub use device::{
    GgufDeviceLoadError, GgufDeviceStore, GgufDeviceStoreFreeError, GgufDeviceTensor,
    load_glm53_iq3_store, load_glm53_iq3_tensor, load_glm53_store, load_glm53_tensor,
};
pub use glm53::{
    GLM53_GGUF_REVISION, GLM53_GGUF_VOCAB_SIZE, Glm53GgufSummary, Glm53Iq3Summary, Glm53QuantProfile, open_glm53_files,
    open_glm53_iq3_files, validate_glm53_files, validate_glm53_iq3_files,
};
pub use payload::{Glm53GgufFiles, Glm53Iq3Files};
pub use shards::{GgufDirectory, GgufShardRef, LocatedTensor, assemble_split_shards};

#[cfg(test)]
#[path = "gguf_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "gguf_glm53_tests.rs"]
mod glm53_tests;

#[cfg(all(test, target_os = "linux"))]
#[path = "gguf_payload_tests.rs"]
mod payload_tests;

#[cfg(all(test, target_os = "linux"))]
#[path = "gguf_device_tests.rs"]
mod device_tests;
