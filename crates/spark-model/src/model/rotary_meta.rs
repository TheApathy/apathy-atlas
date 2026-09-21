// SPDX-License-Identifier: AGPL-3.0-only

//! Host metadata for scalar target replay. Slot +8 and length +16 stay physical;
//! prompt replay's independent H/W coordinates occupy unused +20/+24 bytes.

use crate::traits::RotaryPositions;
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, HostToDeviceCopy};

pub(super) struct BatchRotary {
    pub positions: Vec<u32>,
    heights: Vec<u8>,
    widths: Vec<u8>,
    shared: bool,
}

impl BatchRotary {
    pub fn new(rows: &[(&RotaryPositions, usize)], padded: usize) -> Result<Self> {
        anyhow::ensure!(
            rows.len() <= padded && padded <= 64,
            "rotary batch exceeds fixed metadata extent"
        );
        // Graphs bind pointer topology, not merely today's equal axis values.
        // Keep image sequences three-axis even after entering the scalar tail.
        let shared = rows.iter().all(|(map, _)| map.is_identity());
        anyhow::ensure!(
            shared || padded * 12 <= 256,
            "three-axis rotary batch exceeds position metadata region"
        );
        let mut result = Self {
            positions: Vec::with_capacity(padded),
            heights: Vec::new(),
            widths: Vec::new(),
            shared,
        };
        for &(map, physical) in rows {
            let [t, h, w] = map.position(physical)?;
            result.positions.push(t);
            result.heights.extend_from_slice(&h.to_le_bytes());
            result.widths.extend_from_slice(&w.to_le_bytes());
        }
        result.positions.resize(padded, 0);
        result.heights.resize(padded * 4, 0);
        result.widths.resize(padded * 4, 0);
        Ok(result)
    }

    pub fn upload_axes(
        &self,
        gpu: &dyn GpuBackend,
        base: DevicePtr,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr)> {
        if self.shared {
            return Ok((base, base));
        }
        anyhow::ensure!(
            base.0
                .checked_add((self.positions.len() * 12) as u64)
                .is_some(),
            "rotary batch device range overflow"
        );
        let height = base.offset(self.positions.len() * 4);
        let width = base.offset(self.positions.len() * 8);
        gpu.copy_h2d_group_on_stream(
            &[
                HostToDeviceCopy::new(&self.heights, height),
                HostToDeviceCopy::new(&self.widths, width),
            ],
            stream,
        )?;
        Ok((height, width))
    }
}

pub(super) struct SingleRotary {
    pub position: u32,
    height: [u8; 4],
    width: [u8; 4],
    shared: bool,
}

impl SingleRotary {
    pub fn new(map: &RotaryPositions, physical: usize) -> Result<Self> {
        let [t, h, w] = map.position(physical)?;
        Ok(Self {
            position: t,
            height: h.to_le_bytes(),
            width: w.to_le_bytes(),
            shared: map.is_identity(),
        })
    }

    pub fn upload_axes(
        &self,
        gpu: &dyn GpuBackend,
        base: DevicePtr,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr)> {
        if self.shared {
            return Ok((base, base));
        }
        anyhow::ensure!(
            base.0.checked_add(28).is_some(),
            "rotary device range overflow"
        );
        let height = base.offset(20);
        let width = base.offset(24);
        gpu.copy_h2d_group_on_stream(
            &[
                HostToDeviceCopy::new(&self.height, height),
                HostToDeviceCopy::new(&self.width, width),
            ],
            stream,
        )?;
        Ok((height, width))
    }
}

#[cfg(test)]
#[path = "rotary_meta_io_tests.rs"]
mod io_tests;

#[cfg(test)]
#[path = "rotary_wiring_tests.rs"]
mod wiring_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ImageSpan;

    #[test]
    fn prompt_replay_axes_do_not_overlap_slot_or_length_fields() {
        let map = RotaryPositions::from_image_spans(
            6,
            32,
            &[ImageSpan {
                start: 1,
                height: 2,
                width: 2,
            }],
        )
        .unwrap();
        let axes = SingleRotary::new(&map, 2).unwrap();
        assert_eq!(axes.position, 1);
        assert_eq!(axes.height, 1u32.to_le_bytes());
        assert_eq!(axes.width, 2u32.to_le_bytes());
        assert!(!axes.shared);
        assert!(!SingleRotary::new(&map, 1).unwrap().shared);
        assert!(!SingleRotary::new(&map, 6).unwrap().shared);
        assert!(
            SingleRotary::new(&RotaryPositions::identity(), 2)
                .unwrap()
                .shared
        );
    }

    #[test]
    fn batch_padding_and_tail_alias_are_explicit_and_bounded() {
        let map = RotaryPositions::from_image_spans(
            6,
            32,
            &[ImageSpan {
                start: 1,
                height: 2,
                width: 2,
            }],
        )
        .unwrap();
        let batch = BatchRotary::new(&[(&map, 2)], 4).unwrap();
        assert_eq!(batch.positions, [1, 0, 0, 0]);
        assert_eq!(&batch.widths[..4], &2u32.to_le_bytes());
        assert!(batch.widths[4..].iter().all(|b| *b == 0));
        assert!(!batch.shared);
        assert!(!BatchRotary::new(&[(&map, 6)], 4).unwrap().shared);
        assert!(
            BatchRotary::new(&[(&RotaryPositions::identity(), 6)], 32)
                .unwrap()
                .shared
        );
        assert!(BatchRotary::new(&[(&map, 2)], 32).is_err());
        assert!(BatchRotary::new(&[(&map, 32)], 1).is_err());
    }
}
