// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1 engram: n-gram hashing and the sparse row gather behind it.
//!
//! The tables are 768M rows over ~190 GB and can never be resident on a 119.7 GB
//! box, so `spark_storage::engram_tier` pulls each token's rows off NVMe. This
//! module owns the *which rows* half: see [`hash`].

pub mod hash;

pub use hash::{DEAD, EngramHashState, EngramLayout};
