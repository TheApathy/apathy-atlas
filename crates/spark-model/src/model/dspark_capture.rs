// SPDX-License-Identifier: AGPL-3.0-only

//! Addressing for the DSpark hidden-state capture history.
//!
//! Serving only needs the drafter's bounded attention window, so its capture
//! history is circular. Offline dumps remain linear because their on-disk
//! records need absolute sequence positions.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CaptureSpan {
    pub(crate) src_row: usize,
    pub(crate) dst_row: usize,
    pub(crate) rows: usize,
}

/// Byte addresses for a FP32 HC-highway input and a BF16 capture output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CaptureAddress {
    pub(crate) source_offset: usize,
    pub(crate) source_bytes: usize,
    pub(crate) destination_offset: usize,
    pub(crate) destination_bytes: usize,
    pub(crate) rows: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DsparkCaptureLayout {
    rows: usize,
    ring: bool,
}

impl DsparkCaptureLayout {
    pub(crate) fn new(rows: usize, ring: bool) -> Self {
        debug_assert!(rows > 0);
        Self { rows, ring }
    }

    pub(crate) fn row(self, position: usize) -> Option<usize> {
        if self.ring {
            Some(position % self.rows)
        } else {
            (position < self.rows).then_some(position)
        }
    }

    /// Return the one or two contiguous copies needed to store an input
    /// interval. If a ring input is wider than the history, only its newest
    /// `rows` entries survive, matching circular-buffer semantics.
    pub(crate) fn spans(self, start: usize, count: usize) -> Vec<CaptureSpan> {
        if count == 0 {
            return Vec::new();
        }
        if !self.ring {
            if start >= self.rows {
                return Vec::new();
            }
            return vec![CaptureSpan {
                src_row: 0,
                dst_row: start,
                rows: count.min(self.rows - start),
            }];
        }

        let kept = count.min(self.rows);
        let src_row = count - kept;
        let kept_start = start + src_row;
        let dst_row = kept_start % self.rows;
        let first = kept.min(self.rows - dst_row);
        let mut spans = vec![CaptureSpan {
            src_row,
            dst_row,
            rows: first,
        }];
        if first < kept {
            spans.push(CaptureSpan {
                src_row: src_row + first,
                dst_row: 0,
                rows: kept - first,
            });
        }
        spans
    }

    /// Validate the entire capture before its first launch. Source row strides
    /// use FP32 storage, even though each resulting mean row is stored as BF16.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn addresses(
        self,
        start: usize,
        count: usize,
        slot: usize,
        layers: usize,
        source_capacity_rows: usize,
        hidden: usize,
        hc_mult: usize,
    ) -> Result<Vec<CaptureAddress>, &'static str> {
        if self.rows == 0 || layers == 0 || slot >= layers || hidden == 0 || hc_mult == 0 {
            return Err("invalid DSpark capture geometry");
        }
        if count > source_capacity_rows {
            return Err("DSpark capture input exceeds the HC arena");
        }
        for extent in [count, hidden, hc_mult] {
            u32::try_from(extent).map_err(|_| "DSpark capture exceeds kernel ABI")?;
        }
        start
            .checked_add(count)
            .ok_or("DSpark capture position overflow")?;
        let source_stride = hidden
            .checked_mul(hc_mult)
            .and_then(|n| n.checked_mul(4))
            .ok_or("DSpark FP32 source stride overflow")?;
        let destination_stride = hidden
            .checked_mul(2)
            .ok_or("DSpark BF16 destination stride overflow")?;
        let source_capacity = source_capacity_rows
            .checked_mul(source_stride)
            .ok_or("DSpark source capacity overflow")?;
        let destination_capacity = layers
            .checked_mul(self.rows)
            .and_then(|n| n.checked_mul(destination_stride))
            .ok_or("DSpark destination capacity overflow")?;
        let mut addresses = Vec::new();
        for span in self.spans(start, count) {
            let source_offset = span
                .src_row
                .checked_mul(source_stride)
                .ok_or("DSpark source offset overflow")?;
            let source_bytes = span
                .rows
                .checked_mul(source_stride)
                .ok_or("DSpark source byte count overflow")?;
            let destination_offset = slot
                .checked_mul(self.rows)
                .and_then(|n| n.checked_add(span.dst_row))
                .and_then(|n| n.checked_mul(destination_stride))
                .ok_or("DSpark destination offset overflow")?;
            let destination_bytes = span
                .rows
                .checked_mul(destination_stride)
                .ok_or("DSpark destination byte count overflow")?;
            if source_offset
                .checked_add(source_bytes)
                .is_none_or(|n| n > source_capacity)
                || destination_offset
                    .checked_add(destination_bytes)
                    .is_none_or(|n| n > destination_capacity)
            {
                return Err("DSpark capture span exceeds allocation capacity");
            }
            addresses.push(CaptureAddress {
                source_offset,
                source_bytes,
                destination_offset,
                destination_bytes,
                rows: u32::try_from(span.rows).map_err(|_| "DSpark capture row overflow")?,
            });
        }
        Ok(addresses)
    }
}

pub(crate) fn position_in_window(position: usize, newest: usize, window: usize) -> bool {
    position <= newest && newest - position < window
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_layout_clips_at_capacity() {
        let layout = DsparkCaptureLayout::new(8, false);
        assert_eq!(layout.row(7), Some(7));
        assert_eq!(layout.row(8), None);
        assert_eq!(
            layout.spans(6, 4),
            vec![CaptureSpan {
                src_row: 0,
                dst_row: 6,
                rows: 2,
            }]
        );
    }

    #[test]
    fn ring_layout_wraps_absolute_positions() {
        let layout = DsparkCaptureLayout::new(8, true);
        assert_eq!(layout.row(10), Some(2));
        assert_eq!(
            layout.spans(6, 4),
            vec![
                CaptureSpan {
                    src_row: 0,
                    dst_row: 6,
                    rows: 2,
                },
                CaptureSpan {
                    src_row: 2,
                    dst_row: 0,
                    rows: 2,
                },
            ]
        );
    }

    #[test]
    fn ring_layout_keeps_only_newest_full_history() {
        let layout = DsparkCaptureLayout::new(4, true);
        assert_eq!(
            layout.spans(2, 7),
            vec![
                CaptureSpan {
                    src_row: 3,
                    dst_row: 1,
                    rows: 3,
                },
                CaptureSpan {
                    src_row: 6,
                    dst_row: 0,
                    rows: 1,
                },
            ]
        );
    }

    #[test]
    fn expired_boundary_slot_is_not_treated_as_active() {
        assert!(position_in_window(128, 128, 128));
        assert!(position_in_window(128, 255, 128));
        assert!(!position_in_window(128, 256, 128));
        assert!(!position_in_window(129, 128, 128));
    }
}
