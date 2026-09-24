// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM53_PREFIX_COMMIT=1` (default off): commit an accepted prefix of a
//! wide exact verify without replaying it through the target.
//!
//! The exact verifier already computes every row with the one-row arithmetic:
//! the KDA recurrence is chained row by row and the DSA stage loops over rows.
//! What a partial acceptance lacks is the persistent STATE as of the accepted
//! row, which the shipping path recovers by running the accepted prefix through
//! the target a second time (`PartialReplay`). This module keeps that state
//! instead: the KDA recurrence writes the state after row `r` to snapshot `r`
//! (the last row keeps writing the staged buffer, so full acceptance is
//! untouched), and the DSA loop saves the bounded carry regions after each row
//! with the same region plan the verify snapshot uses. A prefix commit is then
//! copies plus a re-stage of the tiny conv shift register.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops::GgmlIqBuffer;

use super::dsa_attention::Glm53DsaCacheSlots;
use super::dsa_verify_plan::DsaVerifyPlan;

pub(super) const KDA_ORDINALS: usize = 34;
pub(super) const DSA_ORDINALS: usize = 11;
/// Recurrent state of one KDA layer: 64 heads x 128 x 128 f32.
pub(super) const KDA_ROW_BYTES: usize = 4 * 1024 * 1024;
/// Rows 0..=6 of an 8-row pass get a snapshot; the last row lands in staged.
pub(super) const KDA_SNAPSHOT_ROWS: usize = 7;

/// `ATLAS_GLM53_PREFIX_COMMIT_ROWS=<1..=7>` (default 7): KDA snapshot rows to
/// allocate, i.e. the widest verify minus one. 7 rows cost 952 MiB; a
/// `--glm-dflash-max-drafts 3` server needs 3 (408 MiB). A wider verify is refused.
pub(super) fn kda_snapshot_rows_from(value: Option<&str>) -> Result<usize> {
    let Some(value) = value else {
        return Ok(KDA_SNAPSHOT_ROWS);
    };
    let rows: usize = value
        .parse()
        .ok()
        .filter(|rows| (1..=KDA_SNAPSHOT_ROWS).contains(rows))
        .with_context(|| format!("ATLAS_GLM53_PREFIX_COMMIT_ROWS must be 1..=7, got {value:?}"))?;
    Ok(rows)
}
pub(super) const DSA_SNAPSHOT_ROWS: usize = 8;
/// One row's DSA carry (latent rows, published pools, tail) packed 256-aligned.
pub(super) const DSA_ROW_SLOT_BYTES: usize = 12_288;
/// One row of one KDA conv input plane (q, k or v projection): 8192 x bf16.
pub(super) const CONV_INPUT_ROW_BYTES: usize = 16_384;
/// Conv input rows kept per plane: a verify is at most 8 rows.
pub(super) const CONV_INPUT_ROWS: usize = 8;
const ALIGNMENT: usize = 256;

pub(super) fn prefix_commit_from(value: Option<&str>) -> bool {
    value == Some("1")
}

pub(super) fn prefix_commit_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        prefix_commit_from(std::env::var("ATLAS_GLM53_PREFIX_COMMIT").ok().as_deref())
    })
}

#[derive(Clone, Copy, Debug)]
pub struct PrefixCommitBuffers {
    kda_rows: usize,
    kda: DevicePtr,
    dsa: DevicePtr,
    conv: DevicePtr,
}

impl PrefixCommitBuffers {
    pub(super) const fn kda_bytes(kda_rows: usize) -> usize {
        KDA_ORDINALS * kda_rows * KDA_ROW_BYTES
    }
    pub(super) const DSA_BYTES: usize = DSA_ORDINALS * DSA_SNAPSHOT_ROWS * DSA_ROW_SLOT_BYTES;
    /// Each KDA layer's q/k/v conv inputs for the verify rows. The verify
    /// scratch is shared by all 34 layers, so the per-layer inputs a prefix
    /// re-stage of the conv shift register needs must be kept here.
    pub(super) const CONV_BYTES: usize =
        KDA_ORDINALS * 3 * CONV_INPUT_ROWS * CONV_INPUT_ROW_BYTES;
    pub(super) const fn bytes(kda_rows: usize) -> usize {
        Self::kda_bytes(kda_rows) + Self::DSA_BYTES + Self::CONV_BYTES
    }

    pub(super) fn bind(base: DevicePtr, kda_rows: usize) -> Result<Self> {
        ensure!(
            (1..=KDA_SNAPSHOT_ROWS).contains(&kda_rows),
            "prefix commit KDA snapshot rows must be 1..=7"
        );
        ensure!(base != DevicePtr::NULL, "prefix commit buffers are null");
        ensure!(
            base.0 % ALIGNMENT as u64 == 0,
            "prefix commit buffers must be 256-byte aligned"
        );
        let kda_bytes = Self::kda_bytes(kda_rows);
        Ok(Self {
            kda_rows,
            kda: base,
            dsa: DevicePtr(base.0 + kda_bytes as u64),
            conv: DevicePtr(base.0 + (kda_bytes + Self::DSA_BYTES) as u64),
        })
    }

    pub(super) fn kda_row(&self, ordinal: usize, row: usize) -> Result<GgmlIqBuffer> {
        ensure!(
            ordinal < KDA_ORDINALS && row < self.kda_rows,
            "prefix commit KDA snapshot ordinal {ordinal}/row {row} out of range \
             (ATLAS_GLM53_PREFIX_COMMIT_ROWS={}; it must cover --glm-dflash-max-drafts)",
            self.kda_rows
        );
        Ok(GgmlIqBuffer {
            ptr: DevicePtr(self.kda.0 + ((ordinal * self.kda_rows + row) * KDA_ROW_BYTES) as u64),
            bytes: KDA_ROW_BYTES,
        })
    }

    /// Saved conv input plane `plane` (0 = q, 1 = k, 2 = v) of KDA layer
    /// `ordinal`, the first `rows` rows.
    pub(super) fn conv_input(&self, ordinal: usize, plane: usize, rows: usize) -> Result<GgmlIqBuffer> {
        ensure!(
            ordinal < KDA_ORDINALS && plane < 3 && (1..=CONV_INPUT_ROWS).contains(&rows),
            "prefix commit conv input ordinal {ordinal}/plane {plane}/rows {rows} out of range"
        );
        let offset = (ordinal * 3 + plane) * CONV_INPUT_ROWS * CONV_INPUT_ROW_BYTES;
        Ok(GgmlIqBuffer {
            ptr: DevicePtr(self.conv.0 + offset as u64),
            bytes: rows * CONV_INPUT_ROW_BYTES,
        })
    }

    pub(super) fn dsa_layer(&self, ordinal: usize) -> Result<GgmlIqBuffer> {
        ensure!(
            ordinal < DSA_ORDINALS,
            "prefix commit DSA snapshot ordinal {ordinal} out of range"
        );
        Ok(GgmlIqBuffer {
            ptr: DevicePtr(self.dsa.0 + (ordinal * DSA_SNAPSHOT_ROWS * DSA_ROW_SLOT_BYTES) as u64),
            bytes: DSA_SNAPSHOT_ROWS * DSA_ROW_SLOT_BYTES,
        })
    }
}

/// Row slot `row` inside one layer's DSA snapshot area.
pub(super) fn dsa_row_slot(layer: GgmlIqBuffer, row: usize) -> Result<GgmlIqBuffer> {
    ensure!(
        row < DSA_SNAPSHOT_ROWS && layer.bytes == DSA_SNAPSHOT_ROWS * DSA_ROW_SLOT_BYTES,
        "prefix commit DSA row slot {row} out of range"
    );
    Ok(GgmlIqBuffer {
        ptr: layer.ptr.offset(row * DSA_ROW_SLOT_BYTES),
        bytes: DSA_ROW_SLOT_BYTES,
    })
}

/// One copy between a cache buffer and the packed row slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DsaSpan {
    /// Index into [`dsa_cache_buffers`].
    pub slot: usize,
    pub cache_offset: usize,
    pub snapshot_offset: usize,
    pub bytes: usize,
}

/// The bounded carry a `rows`-row chunk starting at `position` writes, in the
/// verify snapshot's own region order, packed into one row slot.
pub(super) fn dsa_row_spans(position: u32, rows: u32, capacity: u32) -> Result<Vec<DsaSpan>> {
    let plan = DsaVerifyPlan::new(position, rows, capacity)?;
    let mut cursor = 0usize;
    let mut spans = Vec::with_capacity(6);
    for (slot, region) in plan.source_regions().into_iter().enumerate() {
        if region.bytes == 0 {
            continue;
        }
        spans.push(DsaSpan {
            slot,
            cache_offset: region.offset,
            snapshot_offset: cursor,
            bytes: region.bytes,
        });
        cursor = cursor
            .checked_add((region.bytes + ALIGNMENT - 1) & !(ALIGNMENT - 1))
            .context("prefix commit DSA row slot overflow")?;
    }
    ensure!(
        cursor <= DSA_ROW_SLOT_BYTES,
        "prefix commit DSA row carry ({cursor} bytes) exceeds its {DSA_ROW_SLOT_BYTES}-byte slot"
    );
    Ok(spans)
}

/// The six persistent DSA buffers in the verify snapshot's region order.
pub(super) fn dsa_cache_buffers(cache: &Glm53DsaCacheSlots) -> [GgmlIqBuffer; 6] {
    [
        cache.latent_cache_bf16,
        cache.pool_keys_bf16,
        cache.pool_validity_u8,
        cache.prior_tail_keys_bf16,
        cache.prior_tail_gates_bf16,
        cache.prior_tail_validity_u8,
    ]
}

/// Stream-ordered copy of the carry after `rows` rows between the cache and
/// the row slot: `to_snapshot` saves, otherwise restores.
pub(super) fn dsa_snapshot_copy(
    gpu: &dyn GpuBackend,
    cache: &Glm53DsaCacheSlots,
    slot: GgmlIqBuffer,
    position: u32,
    rows: u32,
    capacity: u32,
    to_snapshot: bool,
    stream: u64,
) -> Result<()> {
    ensure!(
        slot.bytes == DSA_ROW_SLOT_BYTES,
        "prefix commit DSA row slot extent drift"
    );
    let buffers = dsa_cache_buffers(cache);
    for span in dsa_row_spans(position, rows, capacity)? {
        let buffer = buffers[span.slot];
        ensure!(
            span.cache_offset + span.bytes <= buffer.bytes,
            "prefix commit DSA span exceeds its cache buffer"
        );
        let cache_ptr = buffer.ptr.offset(span.cache_offset);
        let snapshot_ptr = slot.ptr.offset(span.snapshot_offset);
        let (source, destination) = if to_snapshot {
            (cache_ptr, snapshot_ptr)
        } else {
            (snapshot_ptr, cache_ptr)
        };
        gpu.copy_d2d_async(source, destination, span.bytes, stream)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_is_strict_and_default_off() {
        assert!(!prefix_commit_from(None));
        assert!(!prefix_commit_from(Some("0")));
        assert!(!prefix_commit_from(Some("true")));
        assert!(prefix_commit_from(Some("1")));
    }

    #[test]
    fn layout_is_disjoint_and_bounded() {
        assert_eq!(PrefixCommitBuffers::kda_bytes(7), 34 * 7 * 4 * 1024 * 1024);
        let buffers = PrefixCommitBuffers::bind(DevicePtr(0x1000), 7).unwrap();
        assert!(PrefixCommitBuffers::bind(DevicePtr(0x1010), 7).is_err());
        assert!(PrefixCommitBuffers::bind(DevicePtr(0x1000), 0).is_err());
        assert!(PrefixCommitBuffers::bind(DevicePtr(0x1000), 8).is_err());
        let narrow = PrefixCommitBuffers::bind(DevicePtr(0x1000), 3).unwrap();
        assert!(narrow.kda_row(0, 2).is_ok() && narrow.kda_row(0, 3).is_err());
        assert_eq!(kda_snapshot_rows_from(None).unwrap(), 7);
        assert_eq!(kda_snapshot_rows_from(Some("3")).unwrap(), 3);
        assert!(kda_snapshot_rows_from(Some("0")).is_err());
        assert!(kda_snapshot_rows_from(Some("8")).is_err());
        let mut seen = Vec::new();
        for ordinal in 0..KDA_ORDINALS {
            for row in 0..KDA_SNAPSHOT_ROWS {
                let b = buffers.kda_row(ordinal, row).unwrap();
                assert_eq!(b.bytes, KDA_ROW_BYTES);
                seen.push((b.ptr.0, b.ptr.0 + b.bytes as u64));
            }
        }
        for ordinal in 0..DSA_ORDINALS {
            let layer = buffers.dsa_layer(ordinal).unwrap();
            for row in 0..DSA_SNAPSHOT_ROWS {
                let b = dsa_row_slot(layer, row).unwrap();
                seen.push((b.ptr.0, b.ptr.0 + b.bytes as u64));
            }
        }
        for ordinal in 0..KDA_ORDINALS {
            for plane in 0..3 {
                let b = buffers.conv_input(ordinal, plane, CONV_INPUT_ROWS).unwrap();
                seen.push((b.ptr.0, b.ptr.0 + b.bytes as u64));
            }
        }
        seen.sort_unstable();
        for pair in seen.windows(2) {
            assert!(pair[0].1 <= pair[1].0, "prefix commit snapshot slots overlap");
        }
        assert_eq!(
            seen.last().unwrap().1 - 0x1000,
            PrefixCommitBuffers::bytes(7) as u64
        );
        assert!(buffers.kda_row(34, 0).is_err());
        assert!(buffers.kda_row(0, 7).is_err());
        assert!(buffers.dsa_layer(11).is_err());
    }

    #[test]
    fn every_row_carry_fits_its_slot_and_matches_the_verify_plan_order() {
        for position in [0u32, 1, 2, 3, 4, 7, 8, 1_000, 2_039] {
            for rows in 1..=8u32 {
                let spans = dsa_row_spans(position, rows, 2_048).unwrap();
                assert_eq!(spans[0].slot, 0);
                assert_eq!(spans[0].cache_offset, position as usize * 1024);
                assert_eq!(spans[0].bytes, rows as usize * 1024);
                let mut end = 0;
                for span in &spans {
                    assert!(span.snapshot_offset >= end);
                    assert_eq!(span.snapshot_offset % ALIGNMENT, 0);
                    end = span.snapshot_offset + span.bytes;
                }
                assert!(end <= DSA_ROW_SLOT_BYTES);
            }
        }
    }
}
