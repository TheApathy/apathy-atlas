// SPDX-License-Identifier: AGPL-3.0-only

//! Checked native BF16 M1 row spans. This module never initializes a context.

pub const WORKSPACE_BYTES: usize = 64 * 1024 * 1024;
const BF16_BYTES: usize = 2;
const CUDA_R_16BF: i32 = 14;

#[path = "serial_rows_request.rs"]
mod request;
pub use request::{ByteSpan, Orientation, SerialRowsRequest};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatrixLayout {
    pub dtype: i32,
    pub rows: u64,
    pub cols: u64,
    pub ld: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowCall {
    pub act: u64,
    pub weight: u64,
    pub out: u64,
    pub workspace: u64,
    pub stream: u64,
    pub workspace_bytes: usize,
    pub alpha_bits: u32,
    pub beta_bits: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SerialRowsPlan {
    request: SerialRowsRequest,
    layouts: [MatrixLayout; 3],
    act_stride: u64,
    out_stride: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundSerialRowsPlan {
    plan: SerialRowsPlan,
    workspace: ByteSpan,
}

fn bf16_bytes(rows: u32, width: u32) -> Result<usize, &'static str> {
    usize::try_from(rows)
        .ok()
        .and_then(|rows| {
            usize::try_from(width)
                .ok()
                .and_then(|width| rows.checked_mul(width))
        })
        .and_then(|elements| elements.checked_mul(BF16_BYTES))
        .ok_or("native serial-row BF16 byte extent overflow")
}

fn span_end(span: ByteSpan, required: usize, alignment: u64) -> Result<u64, &'static str> {
    if span.address == 0 || span.address % alignment != 0 || span.bytes < required {
        return Err("native serial-row null, misaligned, or short span");
    }
    let bytes = u64::try_from(span.bytes).map_err(|_| "native serial-row span exceeds u64")?;
    span.address
        .checked_add(bytes)
        .ok_or("native serial-row declared address span wraps")
}

fn overlaps(a: ByteSpan, a_end: u64, b: ByteSpan, b_end: u64) -> bool {
    a.address < b_end && b.address < a_end
}

impl SerialRowsPlan {
    /// Perform this input-only admission before even the cold ctx() call.
    pub fn new(request: SerialRowsRequest) -> Result<Self, &'static str> {
        if !(2..=8).contains(&request.rows) {
            return Err("native serial-row helper requires 2..=8 rows");
        }
        if request.n == 0
            || request.k == 0
            || request.n > i32::MAX as u32
            || request.k > i32::MAX as u32
        {
            return Err("native serial-row dimensions must be positive i32 values");
        }
        let spans = [request.act, request.weight, request.out];
        let required = [
            bf16_bytes(request.rows, request.k)?,
            bf16_bytes(request.n, request.k)?,
            bf16_bytes(request.rows, request.n)?,
        ];
        let mut ends = [0; 3];
        for index in 0..spans.len() {
            ends[index] = span_end(spans[index], required[index], BF16_BYTES as u64)?;
        }
        for left in 0..spans.len() {
            for right in left + 1..spans.len() {
                if overlaps(spans[left], ends[left], spans[right], ends[right]) {
                    return Err("native serial-row operand spans overlap");
                }
            }
        }
        let (a_rows, a_cols, a_ld) = match request.orientation {
            Orientation::Nk => (request.k, request.n, request.k),
            Orientation::Kn => (request.n, request.k, request.n),
        };
        let layout = |rows: u32, cols: u32, ld: u32| MatrixLayout {
            dtype: CUDA_R_16BF,
            rows: u64::from(rows),
            cols: u64::from(cols),
            ld: i64::from(ld),
        };
        Ok(Self {
            request,
            layouts: [
                layout(a_rows, a_cols, a_ld),
                layout(request.k, 1, request.k),
                layout(request.n, 1, request.n),
            ],
            act_stride: u64::try_from(bf16_bytes(1, request.k)?)
                .map_err(|_| "native serial-row input stride exceeds u64")?,
            out_stride: u64::try_from(bf16_bytes(1, request.n)?)
                .map_err(|_| "native serial-row output stride exceeds u64")?,
        })
    }

    pub fn rows(&self) -> u32 {
        self.request.rows
    }

    pub fn trans_a(&self) -> i32 {
        match self.request.orientation {
            Orientation::Nk => 1,
            Orientation::Kn => 0,
        }
    }

    pub fn layouts(&self) -> [MatrixLayout; 3] {
        self.layouts
    }

    /// Bind the actual existing context workspace after input admission.
    pub fn bind_workspace(&self, workspace: ByteSpan) -> Result<BoundSerialRowsPlan, &'static str> {
        if workspace.bytes != WORKSPACE_BYTES {
            return Err("native serial-row workspace must be exactly 64 MiB");
        }
        let workspace_end = span_end(workspace, WORKSPACE_BYTES, 256)?;
        for operand in [self.request.act, self.request.weight, self.request.out] {
            let operand_end = span_end(operand, BF16_BYTES, BF16_BYTES as u64)?;
            if overlaps(workspace, workspace_end, operand, operand_end) {
                return Err("native serial-row workspace overlaps an operand");
            }
        }
        Ok(BoundSerialRowsPlan {
            plan: *self,
            workspace,
        })
    }
}

impl BoundSerialRowsPlan {
    pub fn row(&self, index: u32) -> Result<RowCall, &'static str> {
        if index >= self.plan.rows() {
            return Err("native serial-row index is outside the admitted span");
        }
        let pointer = |base: u64, stride: u64| {
            u64::from(index)
                .checked_mul(stride)
                .and_then(|offset| base.checked_add(offset))
                .ok_or("native serial-row pointer offset overflow")
        };
        let request = self.plan.request;
        Ok(RowCall {
            act: pointer(request.act.address, self.plan.act_stride)?,
            weight: request.weight.address,
            out: pointer(request.out.address, self.plan.out_stride)?,
            workspace: self.workspace.address,
            stream: request.stream,
            workspace_bytes: self.workspace.bytes,
            alpha_bits: 1.0f32.to_bits(),
            beta_bits: 0.0f32.to_bits(),
        })
    }
}
