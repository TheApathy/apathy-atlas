// SPDX-License-Identifier: AGPL-3.0-only

//! Shared request types for CUDA execution and non-CUDA compile-only stubs.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    /// Row-major weight is [N,K]; column-major A is transposed.
    Nk,
    /// Row-major weight is [K,N]; column-major A is not transposed.
    Kn,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteSpan {
    pub address: u64,
    pub bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SerialRowsRequest {
    pub rows: u32,
    pub n: u32,
    pub k: u32,
    pub orientation: Orientation,
    pub act: ByteSpan,
    pub weight: ByteSpan,
    pub out: ByteSpan,
    pub stream: u64,
}
