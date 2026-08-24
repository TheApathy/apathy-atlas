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
