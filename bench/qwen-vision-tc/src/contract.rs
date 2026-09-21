// SPDX-License-Identifier: AGPL-3.0-only

//! Pure launch admission; image payloads remain subject to ImageLayout.
use std::ffi::OsStr;

#[path = "contract_launch.rs"]
mod launch;
pub use launch::{Arg, BoundPlan, Buffers, Handles, Launch, Region};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Scalar,
    UpstreamSeparate,
    FusedBias,
}

impl Mode {
    pub fn parse(value: Option<&OsStr>) -> Result<Self, String> {
        match value.and_then(OsStr::to_str) {
            Some("scalar") => Ok(Self::Scalar),
            Some("upstream-separate") => Ok(Self::UpstreamSeparate),
            Some("fused-bias") => Ok(Self::FusedBias),
            _ => Err("explicit mode must be scalar, upstream-separate, or fused-bias".into()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Patch,
    Qkv,
    AttentionOutput,
    Fc1,
    Fc2,
    MergerFc1,
    MergerFc2,
}

#[derive(Debug)]
pub struct Plan {
    m: u32,
    n: u32,
    k: u32,
    mode: Mode,
    bytes: [usize; 4],
    total: usize,
}

impl Plan {
    /// `rows` is GEMM M, not image pixels or pre-merge patch count.
    pub fn new(
        family: Family,
        rows: usize,
        output_width: usize,
        mode: Mode,
    ) -> Result<Self, String> {
        if !matches!(output_width, 2560 | 5120) {
            return Err("unsupported Qwen vision output width".into());
        }
        let (n, k, cap) = match family {
            Family::Patch => (1152, 1536, 6400),
            Family::Qkv => (3456, 1152, 6400),
            Family::AttentionOutput => (1152, 1152, 6400),
            Family::Fc1 => (4304, 1152, 6400),
            Family::Fc2 => (1152, 4304, 6400),
            Family::MergerFc1 => (4608, 4608, 1600),
            Family::MergerFc2 => (output_width, 4608, 1600),
        };
        if rows == 0 || rows > cap {
            return Err("projection row count exceeds its contract".into());
        }
        let bytes = [
            bf16_bytes(rows, k)?,
            bf16_bytes(n, k)?,
            bf16_bytes(1, n)?,
            bf16_bytes(rows, n)?,
        ];
        let total = bytes
            .iter()
            .try_fold(0usize, |sum, x| sum.checked_add(*x))
            .ok_or("projection byte total overflow")?;
        if total >= 96 * 1024 * 1024 {
            return Err("projection exceeds 96 MiB data cap".into());
        }
        let m = u32::try_from(rows).map_err(|_| "M overflow")?;
        let n = u32::try_from(n).map_err(|_| "N overflow")?;
        let k = u32::try_from(k).map_err(|_| "K overflow")?;
        m.checked_mul(n).ok_or("bias element count overflow")?;
        Ok(Self {
            m,
            n,
            k,
            mode,
            bytes,
            total,
        })
    }

    pub fn shape(&self) -> (u32, u32, u32) {
        (self.m, self.n, self.k)
    }
    pub fn input_bytes(&self) -> usize {
        self.bytes[0]
    }
    pub fn weight_bytes(&self) -> usize {
        self.bytes[1]
    }
    pub fn bias_bytes(&self) -> usize {
        self.bytes[2]
    }
    pub fn output_bytes(&self) -> usize {
        self.bytes[3]
    }
    pub fn total_bytes(&self) -> usize {
        self.total
    }

    /// Receipts are immutable and cannot be constructed without admission.
    /// Read-only operands may alias; the full output region may not alias any.
    pub fn bind(&self, buffers: Buffers, handles: Handles) -> Result<BoundPlan, String> {
        let Buffers { a, b, bias, c } = buffers;
        let alignment = if self.mode == Mode::Scalar { 2 } else { 16 };
        a.require(self.input_bytes(), alignment)?;
        b.require(self.weight_bytes(), alignment)?;
        bias.require(self.bias_bytes(), 2)?;
        c.require(self.output_bytes(), 2)?;
        if self.mode != Mode::Scalar && self.k % 8 != 0 {
            return Err("tensor-core K row stride must be a multiple of 16 bytes".into());
        }
        if [a, b, bias].iter().any(|r| c.overlaps(*r)) {
            return Err("output aliases an immutable operand".into());
        }
        let kernel = match self.mode {
            Mode::Scalar => handles.scalar,
            Mode::UpstreamSeparate => handles.pipelined,
            Mode::FusedBias => handles.fused_bias,
        };
        if kernel == 0 || (self.mode == Mode::UpstreamSeparate && handles.add_bias == 0) {
            return Err("selected kernel handle is missing; fallback is forbidden".into());
        }
        let mut args = vec![Arg::Ptr(a.address), Arg::Ptr(b.address)];
        if self.mode != Mode::UpstreamSeparate {
            args.push(Arg::Ptr(bias.address));
        }
        args.extend([
            Arg::Ptr(c.address),
            Arg::U32(self.m),
            Arg::U32(self.n),
            Arg::U32(self.k),
        ]);
        let (tile, block) = if self.mode == Mode::Scalar {
            (32, [32, 32, 1])
        } else {
            (128, [256, 1, 1])
        };
        let mut launches = vec![Launch {
            kernel,
            grid: [self.n.div_ceil(tile), self.m.div_ceil(tile), 1],
            block,
            args,
        }];
        if self.mode == Mode::UpstreamSeparate {
            let elements = self
                .m
                .checked_mul(self.n)
                .ok_or("bias element count overflow")?;
            launches.push(Launch {
                kernel: handles.add_bias,
                grid: [elements.div_ceil(256), 1, 1],
                block: [256, 1, 1],
                args: vec![
                    Arg::Ptr(c.address),
                    Arg::Ptr(bias.address),
                    Arg::U32(self.m),
                    Arg::U32(self.n),
                ],
            });
        }
        Ok(BoundPlan {
            launches,
            output_elements: self.output_bytes() / 2,
        })
    }
}

fn bf16_bytes(rows: usize, columns: usize) -> Result<usize, String> {
    rows.checked_mul(columns)
        .and_then(|x| x.checked_mul(2))
        .ok_or_else(|| "BF16 extent overflow".into())
}
