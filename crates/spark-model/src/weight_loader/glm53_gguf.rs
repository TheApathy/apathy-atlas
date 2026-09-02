// SPDX-License-Identifier: AGPL-3.0-only

//! Checked zero-copy views over admitted GLM-5.3 GGUF device tensors.
//!
//! Pointer extraction is crate-private and trusted to remain within a borrow of
//! the owning runtime. A lifetime-branded executor facade is still required
//! before these raw views can form a public execution API.

use std::fmt;

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::weights::gguf::{GgmlType, GgufDeviceTensor};

use crate::layers::ops::{GgmlIqBuffer, GgmlIqMmqPlan};

/// One `[K, N]` GGUF tensor ready for the dense IQ-MMQ primitive.
pub struct Glm53GgufMatrix {
    buffer: GgmlIqBuffer,
    kind: GgmlType,
    inner: u32,
    columns: u32,
}

impl fmt::Debug for Glm53GgufMatrix {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Glm53GgufMatrix")
            .field("kind", &self.kind)
            .field("inner", &self.inner)
            .field("columns", &self.columns)
            .field("bytes", &self.buffer.bytes)
            .finish()
    }
}

impl Glm53GgufMatrix {
    pub(crate) fn new(tensor: &GgufDeviceTensor) -> Result<Self> {
        if tensor.dimensions.len() != 2 {
            bail!("GLM GGUF matrix must have exact [K, N] rank");
        }
        let inner =
            u32::try_from(tensor.dimensions[0]).context("GLM GGUF matrix K exceeds kernel ABI")?;
        let columns =
            u32::try_from(tensor.dimensions[1]).context("GLM GGUF matrix N exceeds kernel ABI")?;
        Self::from_parts(
            tensor.ptr,
            tensor.byte_len,
            tensor.ggml_type,
            inner,
            columns,
        )
    }

    fn from_parts(
        ptr: DevicePtr,
        bytes: usize,
        kind: GgmlType,
        inner: u32,
        columns: u32,
    ) -> Result<Self> {
        if ptr == DevicePtr::NULL {
            bail!("GLM GGUF matrix has a null device pointer");
        }
        let plan = GgmlIqMmqPlan::new(kind, 1, columns, inner)?;
        if bytes != plan.weight_bytes {
            bail!(
                "GLM GGUF matrix byte extent mismatch: expected {}, got {bytes}",
                plan.weight_bytes
            );
        }
        ptr.0
            .checked_add(u64::try_from(bytes)?)
            .context("GLM GGUF matrix device address overflow")?;
        Ok(Self {
            buffer: GgmlIqBuffer { ptr, bytes },
            kind,
            inner,
            columns,
        })
    }

    pub fn plan(&self, rows: u32) -> Result<GgmlIqMmqPlan> {
        let plan = GgmlIqMmqPlan::new(self.kind, rows, self.columns, self.inner)?;
        if plan.weight_bytes != self.buffer.bytes {
            bail!("GLM GGUF matrix changed extent while planning");
        }
        Ok(plan)
    }

    pub(crate) fn buffer(&self) -> GgmlIqBuffer {
        self.buffer
    }

    pub fn kind(&self) -> GgmlType {
        self.kind
    }

    pub fn inner(&self) -> u32 {
        self.inner
    }

    pub fn columns(&self) -> u32 {
        self.columns
    }
}

/// One contiguous `[K, N, E]` GGUF tensor with checked per-expert views.
pub struct Glm53GgufExperts {
    base: DevicePtr,
    kind: GgmlType,
    inner: u32,
    columns: u32,
    experts: u32,
    expert_bytes: usize,
    total_bytes: usize,
}

impl fmt::Debug for Glm53GgufExperts {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Glm53GgufExperts")
            .field("kind", &self.kind)
            .field("inner", &self.inner)
            .field("columns", &self.columns)
            .field("experts", &self.experts)
            .field("expert_bytes", &self.expert_bytes)
            .field("total_bytes", &self.total_bytes)
            .finish()
    }
}

impl Glm53GgufExperts {
    pub(crate) fn new(tensor: &GgufDeviceTensor) -> Result<Self> {
        if tensor.dimensions.len() != 3 {
            bail!("GLM GGUF experts must have exact [K, N, E] rank");
        }
        if tensor.ptr == DevicePtr::NULL {
            bail!("GLM GGUF experts have a null device pointer");
        }
        let inner =
            u32::try_from(tensor.dimensions[0]).context("GLM GGUF expert K exceeds kernel ABI")?;
        let columns =
            u32::try_from(tensor.dimensions[1]).context("GLM GGUF expert N exceeds kernel ABI")?;
        let experts = u32::try_from(tensor.dimensions[2])
            .context("GLM GGUF expert count exceeds kernel ABI")?;
        if experts == 0 {
            bail!("GLM GGUF experts require at least one expert");
        }
        let expert_bytes = GgmlIqMmqPlan::new(tensor.ggml_type, 1, columns, inner)?.weight_bytes;
        let total_bytes = expert_bytes
            .checked_mul(usize::try_from(experts)?)
            .context("GLM GGUF packed expert byte count overflow")?;
        if tensor.byte_len != total_bytes {
            bail!(
                "GLM GGUF packed expert byte extent mismatch: expected {total_bytes}, got {}",
                tensor.byte_len
            );
        }
        tensor
            .ptr
            .0
            .checked_add(u64::try_from(total_bytes)?)
            .context("GLM GGUF packed expert device address overflow")?;
        Ok(Self {
            base: tensor.ptr,
            kind: tensor.ggml_type,
            inner,
            columns,
            experts,
            expert_bytes,
            total_bytes,
        })
    }

    pub(crate) fn expert(&self, index: u32) -> Result<Glm53GgufMatrix> {
        if index >= self.experts {
            bail!("GLM GGUF expert index {index} is out of range");
        }
        let offset = usize::try_from(index)?
            .checked_mul(self.expert_bytes)
            .context("GLM GGUF expert offset overflow")?;
        let end = offset
            .checked_add(self.expert_bytes)
            .context("GLM GGUF expert end overflow")?;
        if end > self.total_bytes {
            bail!("GLM GGUF expert slice exceeds packed tensor");
        }
        let address = self
            .base
            .0
            .checked_add(u64::try_from(offset)?)
            .context("GLM GGUF expert device address overflow")?;
        Glm53GgufMatrix::from_parts(
            DevicePtr(address),
            self.expert_bytes,
            self.kind,
            self.inner,
            self.columns,
        )
    }

    pub fn len(&self) -> u32 {
        self.experts
    }

    pub fn is_empty(&self) -> bool {
        self.experts == 0
    }
}

/// The same physical `[K, N, B]` layout used for DSA's 64 per-head matrices.
pub type Glm53GgufMatrixBank = Glm53GgufExperts;

/// Exact-shape view over one unquantized GGUF F32 tensor.
#[derive(PartialEq, Eq)]
pub struct Glm53GgufF32 {
    ptr: DevicePtr,
    elements: usize,
    bytes: usize,
}

impl fmt::Debug for Glm53GgufF32 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Glm53GgufF32")
            .field("elements", &self.elements)
            .field("bytes", &self.bytes)
            .finish()
    }
}

impl Glm53GgufF32 {
    pub(crate) fn new(tensor: &GgufDeviceTensor, expected_dimensions: &[u64]) -> Result<Self> {
        if expected_dimensions.is_empty()
            || tensor.dimensions.as_slice() != expected_dimensions
            || tensor.ggml_type != GgmlType::F32
        {
            bail!("GLM GGUF F32 tensor shape or type mismatch");
        }
        if tensor.ptr == DevicePtr::NULL {
            bail!("GLM GGUF F32 tensor has a null device pointer");
        }
        let elements = expected_dimensions
            .iter()
            .try_fold(1usize, |total, &dimension| {
                if dimension == 0 {
                    bail!("GLM GGUF F32 tensor has a zero dimension");
                }
                total
                    .checked_mul(usize::try_from(dimension)?)
                    .context("GLM GGUF F32 element count overflow")
            })?;
        let bytes = elements
            .checked_mul(4)
            .context("GLM GGUF F32 byte count overflow")?;
        if tensor.byte_len != bytes {
            bail!("GLM GGUF F32 tensor byte extent mismatch");
        }
        tensor
            .ptr
            .0
            .checked_add(u64::try_from(bytes)?)
            .context("GLM GGUF F32 device address overflow")?;
        Ok(Self {
            ptr: tensor.ptr,
            elements,
            bytes,
        })
    }

    pub(crate) fn ptr(&self) -> DevicePtr {
        self.ptr
    }

    pub fn elements(&self) -> usize {
        self.elements
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
}
