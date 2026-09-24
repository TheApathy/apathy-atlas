// SPDX-License-Identifier: AGPL-3.0-only

//! Conversation context checkpoints on NVMe.
//!
//! A checkpoint is the complete model state of a sequence after prefilling
//! a prompt of `C` tokens (KV pages, recurrent state, speculative rings),
//! plus those token ids. When a later prompt starts with the same `C`
//! tokens, the state is restored and only the suffix is prefilled, which
//! turns a multi-minute re-read of a long conversation into a disk read.
//!
//! This module is model-agnostic: models describe their state as named
//! sections (`Model::ctx_capture` / `Model::ctx_restore`), and this store
//! handles the file format, integrity, identity, indexing and disk budget.

mod aligned;
mod format;
mod identity;
mod store;

pub use aligned::{ALIGN, AlignedBuf};
pub use format::{CtxSection, CtxSnapshot, FORMAT_VERSION};
pub use identity::{
    KeyBuilder, dir_fingerprint, engine_digest, file_digest, hex, prefix_hash, recipe_env,
};
pub use store::{CtxStore, Entry};
