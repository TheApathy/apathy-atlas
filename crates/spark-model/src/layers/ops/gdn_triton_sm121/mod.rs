// SPDX-License-Identifier: AGPL-3.0-only

//! Unregistered preparation for the frozen SGLang c143 Triton GDN oracle.
//!
//! This directory is deliberately absent from `ops.rs` and every production
//! route. `PreparedArtifacts::preflight` is CPU-only; both unsafe methods cross
//! the CUDA boundary and require a later reviewed integration and reservation.

mod authority;
mod digest;
mod ffi;
mod launch;
mod loader;
mod manifest;
mod types;

pub use launch::{MetadataContract, PreparedLaunch, prepare_launch};
pub use loader::{LoadedFamily, PreparedArtifacts};
pub use manifest::{
    ARTIFACT_ATTESTATION_SHA256, EXECUTOR_SOURCE_BUNDLE_SHA256, GATE_SOURCE_BUNDLE_SHA256,
    MANIFEST_SHA256, SAME_STREAM_SEQUENCE,
};
pub use types::{Buffers, Region, Stream, WorkspaceLayout, workspace_layout};
