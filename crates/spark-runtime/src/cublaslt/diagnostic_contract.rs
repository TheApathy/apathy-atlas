// SPDX-License-Identifier: AGPL-3.0-only
//! Pure CUDA13 ABI and explicit experimental arithmetic admission.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReductionPolicy {
    /// Leave the production preference unchanged, including its library default.
    Baseline,
    /// Allow unsplit algorithms or FP32-compute-type split reductions only.
    ComputeTypeOnly,
}
impl ReductionPolicy {
    pub fn preference_mask(self) -> Option<u32> {
        match self {
            Self::Baseline => None,
            Self::ComputeTypeOnly => Some(2),
        }
    }
    pub fn validate_selected(self, split: i32, reduction: u32) -> Result<(), &'static str> {
        if split < 0 || !matches!(reduction, 0 | 1 | 2 | 4) {
            return Err("invalid selected split-K or reduction scheme");
        }
        if split > 1 && reduction == 0 {
            return Err("split-K lacks a reduction scheme");
        }
        if self == Self::ComputeTypeOnly && !matches!(reduction, 0 | 2) {
            return Err("selected algorithm violates compute-type-only reduction");
        }
        Ok(())
    }
}

/// cublasLtMatmulHeuristicResult_t, verified against the installed CUDA13 header.
/// The previous byte array had no guaranteed alignment for the opaque uint64s.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct HeuristicResult {
    pub algo: [u64; 8],
    pub workspace_bytes: usize,
    pub state: i32,
    pub waves: f32,
    reserved: [i32; 4],
}
impl HeuristicResult {
    pub fn admit(&self, returned: i32, available: usize) -> Result<(), &'static str> {
        if returned != 1 || self.state != 0 {
            return Err("heuristic did not return one successful algorithm");
        }
        if self.workspace_bytes > available || !self.waves.is_finite() || self.waves < 0.0 {
            return Err("invalid heuristic workspace or wave count");
        }
        Ok(())
    }
}
