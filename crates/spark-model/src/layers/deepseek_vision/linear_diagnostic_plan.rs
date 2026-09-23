// SPDX-License-Identifier: AGPL-3.0-only
//! Pure diagnostic linear planning from already-admitted encoder geometry.
//! No image/checkpoint validation, I/O, CUDA initialization or device allocation.
use std::ffi::OsStr;

pub const WORKSPACE_BYTES: usize = 8_519_680;
const CALL_METADATA_CAP: usize = 64 * 1024;
const COPY_METADATA_CAP: usize = 128 * 1024;
const OPERAND_BYTE_CAP: usize = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendMode {
    Scalar,
    Default,
    Full,
}

impl BackendMode {
    pub fn parse(value: Option<&OsStr>) -> Result<Self, String> {
        match value.and_then(OsStr::to_str) {
            Some("scalar") => Ok(Self::Scalar),
            Some("default") => Ok(Self::Default),
            Some("full") => Ok(Self::Full),
            _ => Err("explicit diagnostic mode must be scalar/default/full".into()),
        }
    }
    pub fn math_mode(self) -> Option<i32> {
        match self {
            Self::Scalar => None,
            Self::Default => Some(0),
            Self::Full => Some(16),
        }
    }
    pub fn receipt_label(self) -> &'static str {
        match self {
            Self::Scalar => "native-wmma",
            Self::Default => "gemmex-default",
            Self::Full => "gemmex-full",
        }
    }
}

/// Copy these values from admitted Geometry and the loaded block count. The
/// model/config loader remains the authority for the actual architecture.
#[derive(Clone, Copy, Debug)]
pub struct Dimensions {
    pub hidden: usize,
    pub intermediate: usize,
    pub patch_dim: usize,
    pub ratio: usize,
    pub text_hidden: usize,
    pub depth: usize,
    pub max_patches: usize,
    pub max_rows: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Patch,
    Qkv,
    Projection,
    Fc1,
    Fc2,
    Align1,
    Align2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActualLinear {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub ldc: usize,
    pub has_bias: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Span {
    pointer: u64,
    bytes: usize,
    end: u64,
}

impl Span {
    pub fn new(pointer: u64, bytes: usize) -> Result<Self, String> {
        if pointer == 0 || bytes == 0 {
            return Err("null or empty device span".into());
        }
        let length = u64::try_from(bytes).map_err(|_| "device extent overflow")?;
        let end = pointer.checked_add(length).ok_or("device end overflow")?;
        Ok(Self {
            pointer,
            bytes,
            end,
        })
    }
    fn require(self, bytes: usize, alignment: u64) -> Result<(), String> {
        if self.bytes < bytes || self.pointer % alignment != 0 {
            return Err("device extent or alignment mismatch".into());
        }
        Ok(())
    }
    fn overlaps(self, other: Self) -> bool {
        self.pointer < other.end && other.pointer < self.end
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Buffers {
    pub input: Span,
    pub weight: Span,
    pub bias: Option<Span>,
    pub output: Span,
    pub workspace: Option<Span>,
}

#[derive(Clone, Debug)]
pub struct LinearSpec {
    family: Family,
    layer: Option<usize>,
    actual: ActualLinear,
    bytes: [usize; 3],
}

impl LinearSpec {
    fn new(
        family: Family,
        layer: Option<usize>,
        m: usize,
        n: usize,
        k: usize,
        has_bias: bool,
    ) -> Result<Self, String> {
        for dim in [m, n, k] {
            if dim == 0 || i32::try_from(dim).is_err() {
                return Err("invalid GemmEx dimension".into());
            }
        }
        let bytes = [bf16_bytes(m, k)?, bf16_bytes(n, k)?, bf16_bytes(m, n)?];
        let copy_bytes = m
            .checked_mul(size_of::<BiasCopy>())
            .ok_or("copy metadata overflow")?;
        if has_bias && copy_bytes > COPY_METADATA_CAP {
            return Err("bias copy plan exceeds bounded host metadata".into());
        }
        Ok(Self {
            family,
            layer,
            actual: ActualLinear {
                m,
                n,
                k,
                ldc: n,
                has_bias,
            },
            bytes,
        })
    }
    pub fn family(&self) -> Family {
        self.family
    }
    pub fn layer(&self) -> Option<usize> {
        self.layer
    }
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.actual.m, self.actual.n, self.actual.k)
    }
    pub fn has_bias(&self) -> bool {
        self.actual.has_bias
    }
    pub fn input_bytes(&self) -> usize {
        self.bytes[0]
    }
    pub fn weight_bytes(&self) -> usize {
        self.bytes[1]
    }
    pub fn output_bytes(&self) -> usize {
        self.bytes[2]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GemmCall {
    pub transa: i32,
    pub transb: i32,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub lda: i32,
    pub ldb: i32,
    pub ldc: i32,
    pub a: u64,
    pub b: u64,
    pub c: u64,
    pub a_type: i32,
    pub b_type: i32,
    pub c_type: i32,
    pub compute_type: i32,
    pub algorithm: i32,
    pub alpha: f32,
    pub beta: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BiasCopy {
    pub src: u64,
    pub dst: u64,
    pub bytes: usize,
}

#[derive(Debug)]
pub struct BoundLinear {
    gemm: Option<GemmCall>,
    copies: Vec<BiasCopy>,
}

impl BoundLinear {
    pub fn gemm_call(&self) -> Option<GemmCall> {
        self.gemm
    }
    pub fn bias_copies(&self) -> &[BiasCopy] {
        &self.copies
    }
}

#[derive(Debug)]
pub struct EncoderPlan {
    calls: Vec<LinearSpec>,
    mode: BackendMode,
}

impl EncoderPlan {
    pub fn new(
        d: Dimensions,
        patches: usize,
        aligned_rows: usize,
        mode: BackendMode,
    ) -> Result<Self, String> {
        if [
            d.hidden,
            d.intermediate,
            d.patch_dim,
            d.ratio,
            d.text_hidden,
            d.depth,
            d.max_patches,
            d.max_rows,
            patches,
            aligned_rows,
        ]
        .contains(&0)
            || patches > d.max_patches
            || aligned_rows > d.max_rows
        {
            return Err("empty dimensions or rows outside admitted scratch bounds".into());
        }
        // Derive from admitted depth, never reproduce the loader's 32-layer guard.
        let count = d
            .depth
            .checked_mul(4)
            .and_then(|x| x.checked_add(3))
            .ok_or("linear call count overflow")?;
        let metadata = count
            .checked_mul(size_of::<LinearSpec>())
            .ok_or("call metadata overflow")?;
        if metadata > CALL_METADATA_CAP {
            return Err("call plan exceeds host metadata cap".into());
        }
        let qkv = d.hidden.checked_mul(3).ok_or("QKV width overflow")?;
        let wide = d.intermediate.checked_mul(2).ok_or("FC1 width overflow")?;
        let merge = d
            .hidden
            .checked_mul(d.ratio)
            .and_then(|x| x.checked_mul(d.ratio))
            .ok_or("aligner width overflow")?;
        let mut calls = Vec::new();
        calls
            .try_reserve_exact(count)
            .map_err(|_| "call metadata allocation failed")?;
        calls.push(LinearSpec::new(
            Family::Patch,
            None,
            patches,
            d.hidden,
            d.patch_dim,
            true,
        )?);
        for layer in 0..d.depth {
            for (family, n, k, bias) in [
                (Family::Qkv, qkv, d.hidden, true),
                (Family::Projection, d.hidden, d.hidden, true),
                (Family::Fc1, wide, d.hidden, false),
                (Family::Fc2, d.hidden, d.intermediate, false),
            ] {
                calls.push(LinearSpec::new(family, Some(layer), patches, n, k, bias)?);
            }
        }
        calls.push(LinearSpec::new(
            Family::Align1,
            None,
            aligned_rows,
            d.text_hidden,
            merge,
            true,
        )?);
        calls.push(LinearSpec::new(
            Family::Align2,
            None,
            aligned_rows,
            d.text_hidden,
            d.text_hidden,
            true,
        )?);
        Ok(Self { calls, mode })
    }

    pub fn calls(&self) -> &[LinearSpec] {
        &self.calls
    }
    pub fn mode(&self) -> BackendMode {
        self.mode
    }

    /// A span proves arithmetic bounds, not that a pointer is allocated: the
    /// example owner must bind these subviews to its actual allocation receipts.
    pub fn bind(
        &self,
        slot: usize,
        actual: ActualLinear,
        buffers: Buffers,
    ) -> Result<BoundLinear, String> {
        let spec = self
            .calls
            .get(slot)
            .ok_or("linear call slot outside plan")?;
        if actual != spec.actual {
            return Err("actual linear call differs from planned slot".into());
        }
        let Buffers {
            input,
            weight,
            bias,
            output,
            workspace,
        } = buffers;
        input.require(spec.input_bytes(), 2)?;
        weight.require(spec.weight_bytes(), 2)?;
        output.require(spec.output_bytes(), 2)?;
        match (spec.has_bias(), bias) {
            (true, Some(span)) => span.require(bf16_bytes(1, actual.n)?, 2)?,
            (false, None) => (),
            _ => return Err("bias presence differs from loaded linear contract".into()),
        }
        match (self.mode, workspace) {
            (BackendMode::Scalar, None) => (),
            (BackendMode::Scalar, Some(_)) => {
                return Err("native mode must not allocate GemmEx workspace".into());
            }
            (_, Some(span)) if span.bytes == WORKSPACE_BYTES => {
                span.require(WORKSPACE_BYTES, 256)?
            }
            _ => return Err("selected GemmEx workspace is absent or has wrong extent".into()),
        }
        let spans = [Some(input), Some(weight), bias, Some(output), workspace];
        for (i, left) in spans.iter().enumerate() {
            for right in &spans[i + 1..] {
                if let (Some(a), Some(b)) = (left, right) {
                    if a.overlaps(*b) {
                        return Err("linear operand/workspace spans alias".into());
                    }
                }
            }
        }
        if self.mode == BackendMode::Scalar {
            return Ok(BoundLinear {
                gemm: None,
                copies: Vec::new(),
            });
        }
        let mut copies = Vec::new();
        if let Some(bias) = bias {
            let row_bytes = bf16_bytes(1, actual.n)?;
            copies
                .try_reserve_exact(actual.m)
                .map_err(|_| "copy metadata allocation failed")?;
            for row in 0..actual.m {
                let offset = row
                    .checked_mul(row_bytes)
                    .ok_or("bias row offset overflow")?;
                let dst = output
                    .pointer
                    .checked_add(u64::try_from(offset).map_err(|_| "bias offset overflow")?)
                    .ok_or("bias destination overflow")?;
                let end = dst
                    .checked_add(row_bytes as u64)
                    .ok_or("bias destination end overflow")?;
                if end > output.end {
                    return Err("bias copy exceeds output span".into());
                }
                copies.push(BiasCopy {
                    src: bias.pointer,
                    dst,
                    bytes: row_bytes,
                });
            }
        }
        // Dimensions were checked for i32 at LinearSpec construction.
        let (m, n, k) = (actual.m as i32, actual.n as i32, actual.k as i32);
        Ok(BoundLinear {
            gemm: Some(GemmCall {
                transa: 1,
                transb: 0,
                m: n,
                n: m,
                k,
                lda: k,
                ldb: k,
                ldc: n,
                a: weight.pointer,
                b: input.pointer,
                c: output.pointer,
                a_type: 14,
                b_type: 14,
                c_type: 14,
                compute_type: 68,
                algorithm: 99,
                alpha: 1.0,
                beta: if spec.has_bias() { 1.0 } else { 0.0 },
            }),
            copies,
        })
    }
}

fn bf16_bytes(rows: usize, columns: usize) -> Result<usize, String> {
    let bytes = rows
        .checked_mul(columns)
        .and_then(|x| x.checked_mul(2))
        .ok_or("BF16 byte extent overflow")?;
    if bytes == 0 || bytes > OPERAND_BYTE_CAP {
        return Err("linear operand exceeds diagnostic byte cap".into());
    }
    Ok(bytes)
}
