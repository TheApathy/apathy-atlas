// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1 engram: n-gram hashing and the sparse row gather behind it.
//!
//! The tables are 768M rows over ~190 GB and can never be resident on a 119.7 GB
//! box, so `spark_storage::engram_tier` pulls each token's rows off NVMe. This
//! module owns the *which rows* half: see [`hash`].

pub mod dead_heads;
pub mod gather;
pub mod hash;
#[cfg(feature = "engram-tokenizer")]
pub mod token_map;

pub use dead_heads::{IMAGE_PAD_ID, IMAGE_SENTINEL_ID, apply_dead_mask, engram_dead_heads};
pub use gather::{EngramGather, EngramLoader, register};
pub use hash::{DEAD, EngramHashState, EngramLayout};
#[cfg(feature = "engram-tokenizer")]
pub use token_map::build_compressed_token_map;
