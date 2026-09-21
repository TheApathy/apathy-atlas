// SPDX-License-Identifier: AGPL-3.0-only

//! Standalone numerical diagnostics. No CUDA access occurs during CPU tests.

pub mod admission;
pub mod contract;
pub mod cuda_abi;
pub mod driver;
pub mod fc2;
mod host_angles;
pub mod io;
pub mod lt;
pub mod lt_abi;
pub mod pins;
pub mod rope;
pub mod weight;
