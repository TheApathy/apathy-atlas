// SPDX-License-Identifier: AGPL-3.0-only

//! Single-use verifier handoff without retaining or moving a host sequence.

use super::verify_policy_transaction::VerifyRequest;
use anyhow::{Result, ensure};
use std::sync::Arc;

/// All live target coordinates must agree again before staging device work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifyFrame {
    pub generation: u64,
    pub nonce: u64,
    pub position: usize,
    pub capacity: usize,
    pub vocab: usize,
    pub stream: u64,
}

/// The model owns this identity for its entire lifetime. A retained binding
/// keeps the identity allocation alive, not the model or its device buffers.
pub struct VerifyBindingOwner(Arc<()>);

#[must_use = "consume against the live model before staging any verifier work"]
pub struct BoundVerifyRequest {
    owner: Arc<()>,
    frame: VerifyFrame,
    request: VerifyRequest,
}

impl BoundVerifyRequest {
    pub(super) fn stream(&self) -> u64 {
        self.frame.stream
    }
}

impl VerifyBindingOwner {
    pub fn new() -> Self {
        Self(Arc::new(()))
    }

    pub fn bind(
        &self,
        frame: VerifyFrame,
        prefix: &[u32],
        inputs: &[u32],
    ) -> Result<BoundVerifyRequest> {
        ensure!(
            prefix.len() == frame.position,
            "GLM binding host position mismatch"
        );
        ensure!(
            prefix.iter().all(|&v| (v as usize) < frame.vocab),
            "GLM binding host prefix token is outside vocabulary"
        );
        let mut request = VerifyRequest::new(frame.position, frame.capacity, frame.vocab, inputs)?;
        request.bind_host_prefix(prefix);
        Ok(BoundVerifyRequest {
            owner: self.0.clone(),
            frame,
            request,
        })
    }

    /// The caller advances its live nonce before releasing the transaction
    /// lock or doing I/O. This value cannot be cloned or consumed twice.
    pub fn consume(
        &self,
        bound: BoundVerifyRequest,
        current: VerifyFrame,
    ) -> Result<VerifyRequest> {
        ensure!(
            Arc::ptr_eq(&self.0, &bound.owner),
            "GLM verify binding belongs to another model"
        );
        ensure!(
            bound.frame == current,
            "GLM verify binding is stale or its stream changed"
        );
        Ok(bound.request)
    }
}
