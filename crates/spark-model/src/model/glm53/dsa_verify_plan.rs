// SPDX-License-Identifier: AGPL-3.0-only

//! Bounded DSA bytes overwritten by one ordinary, non-prefill verifier chunk.

use anyhow::{Context, Result, ensure};

use crate::layers::glm53_dsa_t1_transaction::GLM53_DSA_T1_LAYERS;
use crate::layers::ops::Glm53DsaPoolPlan;

pub(super) const MAX_BACKUP_BYTES: usize = 118_272;
pub(super) const DSA_LAYERS: usize = GLM53_DSA_T1_LAYERS.len();
const ALIGNMENT: usize = 256;
const MAX_ROWS: u32 = 8;
const MAX_POSITIONS: u32 = 1_048_576;

#[derive(Debug, Clone, Copy)]
pub(super) struct Region {
    pub offset: usize,
    pub bytes: usize,
}

#[derive(Debug)]
pub(super) struct DsaVerifyPlan {
    regions: [Region; 6],
    backup_offsets: [usize; 6],
    layer_stride: usize,
    payload_bytes: usize,
    backup_bytes: usize,
}

impl DsaVerifyPlan {
    pub fn new(position: u32, rows: u32, capacity: u32) -> Result<Self> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows),
            "GLM DSA snapshot requires 1..=8 rows"
        );
        ensure!(
            (1..=MAX_POSITIONS).contains(&capacity),
            "GLM DSA snapshot capacity is invalid"
        );
        ensure!(
            position
                .checked_add(rows)
                .is_some_and(|end| end <= capacity),
            "GLM DSA snapshot exceeds sequence capacity"
        );
        let pool = Glm53DsaPoolPlan::new(1, rows, position, 128, 4, MAX_POSITIONS)?;
        let regions = [
            Region {
                offset: usize::try_from(position)?
                    .checked_mul(1024)
                    .context("DSA latent offset overflow")?,
                bytes: usize::try_from(rows)? * 1024,
            },
            Region {
                offset: usize::try_from(position / 4)?
                    .checked_mul(256)
                    .context("DSA pool offset overflow")?,
                bytes: pool.pool_vector_bytes,
            },
            Region {
                offset: usize::try_from(position / 4)?,
                bytes: pool.pool_validity_bytes,
            },
            Region {
                offset: 0,
                bytes: pool.tail_vector_bytes,
            },
            Region {
                offset: 0,
                bytes: pool.tail_vector_bytes,
            },
            Region {
                offset: 0,
                bytes: pool.tail_validity_bytes,
            },
        ];
        let mut backup_offsets = [0; 6];
        let mut layer_stride = 0usize;
        let mut payload = 0usize;
        for (slot, region) in regions.iter().enumerate() {
            backup_offsets[slot] = layer_stride;
            let padded = region
                .bytes
                .checked_add(ALIGNMENT - 1)
                .context("DSA snapshot alignment overflow")?
                & !(ALIGNMENT - 1);
            layer_stride = layer_stride
                .checked_add(padded)
                .context("DSA snapshot stride overflow")?;
            payload = payload
                .checked_add(region.bytes)
                .context("DSA snapshot payload overflow")?;
        }
        let backup_bytes = layer_stride
            .checked_mul(DSA_LAYERS)
            .context("DSA snapshot total overflow")?;
        ensure!(
            backup_bytes <= MAX_BACKUP_BYTES,
            "DSA snapshot exceeds bounded allocation"
        );
        Ok(Self {
            regions,
            backup_offsets,
            layer_stride,
            payload_bytes: payload
                .checked_mul(DSA_LAYERS)
                .context("DSA payload total overflow")?,
            backup_bytes,
        })
    }

    pub fn source_regions(&self) -> [Region; 6] {
        self.regions
    }
    pub fn backup_bytes(&self) -> usize {
        self.backup_bytes
    }
    pub fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }

    pub fn backup_offset(&self, layer: usize, slot: usize) -> Result<usize> {
        ensure!(
            layer < DSA_LAYERS && slot < self.regions.len(),
            "DSA snapshot ordinal is invalid"
        );
        layer
            .checked_mul(self.layer_stride)
            .and_then(|offset| offset.checked_add(self.backup_offsets[slot]))
            .context("DSA snapshot destination offset overflow")
    }
}

/// Check the whole possible commit before target effects, not after KDA commit.
pub(super) fn validate_capture_advance(
    position: u32,
    context_tokens: u32,
    rows: u32,
    limit: u32,
) -> Result<()> {
    ensure!(
        (1..=MAX_ROWS).contains(&rows),
        "GLM DFlash2 capture advance requires 1..=8 rows"
    );
    ensure!(
        position == context_tokens,
        "GLM DFlash2 target/capture context is out of order"
    );
    ensure!(
        position.checked_add(rows).is_some_and(|end| end <= limit),
        "GLM DFlash2 capture advance exceeds its visible context capacity"
    );
    Ok(())
}
