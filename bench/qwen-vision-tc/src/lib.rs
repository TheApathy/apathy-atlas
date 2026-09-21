// SPDX-License-Identifier: AGPL-3.0-only

//! Benchmark-only contracts: formatted/split snapshot passed 15 CPU tests.
//! Formatting and the 250-line module cap are checked separately from GPU gates.
//! No production encoder, CUDA code, GPU initialization, or runnable probe yet.

pub mod contract;
pub mod execution;
pub mod numerics;
