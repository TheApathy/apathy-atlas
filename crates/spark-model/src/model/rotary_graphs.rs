// SPDX-License-Identifier: AGPL-3.0-only

use super::types::TransformerModel;
use crate::traits::{Model, RotaryPositions};
use anyhow::Result;

pub(super) fn layout_changes(old: &RotaryPositions, next: &RotaryPositions) -> bool {
    old.is_identity() != next.is_identity()
}

impl TransformerModel {
    /// Cached scalar/batched graphs bind H/W pointers. Destroy affected graphs
    /// before publishing a new prompt with a different pointer layout.
    pub(super) fn invalidate_rotary_graphs(&self, slot: usize) -> Result<()> {
        let batch_keys: Vec<_> = self
            .batch_decode_graphs
            .lock()
            .keys()
            .filter(|(slots, _)| slots.is_empty() || slots.contains(&slot))
            .cloned()
            .collect();
        let has_scalar = self.decode_graph.lock().contains_key(&slot);
        if !has_scalar && batch_keys.is_empty() {
            return Ok(());
        }
        self.sync_secondary()?;
        self.gpu.synchronize(self.gpu.default_stream())?;
        if let Some(graph) = self.decode_graph.lock().remove(&slot) {
            self.gpu.destroy_graph(graph)?;
        }
        for key in batch_keys {
            if let Some(graph) = self.batch_decode_graphs.lock().remove(&key) {
                self.gpu.destroy_graph(graph)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ImageSpan;

    #[test]
    fn only_identity_image_transition_changes_pointer_topology() {
        let text = RotaryPositions::identity();
        let first = RotaryPositions::from_image_spans(
            6,
            64,
            &[ImageSpan {
                start: 1,
                height: 2,
                width: 2,
            }],
        )
        .unwrap();
        let second = RotaryPositions::from_image_spans(
            8,
            64,
            &[ImageSpan {
                start: 1,
                height: 2,
                width: 3,
            }],
        )
        .unwrap();
        assert!(layout_changes(&text, &first));
        assert!(layout_changes(&first, &text));
        assert!(!layout_changes(&first, &second));
        assert!(!layout_changes(&text, &text));
        // Moving the physical cursor does not change map/topology.
        assert_eq!(first.position(2).unwrap(), [1, 1, 2]);
        assert_eq!(first.position(6).unwrap(), [4; 3]);
        assert!(!layout_changes(&first, &first));
    }
}
