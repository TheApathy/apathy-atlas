// SPDX-License-Identifier: AGPL-3.0-only

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureManifest {
    pub run_id: String,
    pub verify_step: u64,
    pub pre_verify_len: usize,
    pub tokens: Vec<u32>,
    pub absolute_seq_lens: Vec<usize>,
    pub family: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FirstDivergence {
    pub stage: String,
    pub row: usize,
    pub first_byte: usize,
    pub serial_hash: u64,
    pub batch_hash: u64,
    pub mismatch_rows: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageReport {
    pub manifest: CaptureManifest,
    pub stages: usize,
    pub terminal_stage: String,
    pub logits_compared: bool,
    pub first: Option<FirstDivergence>,
}

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}
