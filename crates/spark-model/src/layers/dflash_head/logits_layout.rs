// SPDX-License-Identifier: AGPL-3.0-only

//! Checked layout arithmetic for the compact DFlash logits scratch buffer.

use anyhow::{Context, Result, ensure};

const BF16_BYTES: usize = 2;
// `noise_pass`'s opt-in diagnostic reads this many values from row zero.
const FIXED_DEBUG_PROBE_ELEMENTS: usize = 10;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LogitsLayout {
    configured_rows: usize,
    vocab: usize,
    allocation_bytes: usize,
}

impl LogitsLayout {
    /// Construct the allocation layout shared by every logits writer/reader.
    pub(super) fn new(
        configured_rows: usize,
        target_vocab: usize,
        drafter_vocab: usize,
    ) -> Result<Self> {
        ensure!(configured_rows > 0, "DFlash logits rows must be non-zero");
        ensure!(target_vocab > 0, "DFlash target vocab must be non-zero");
        ensure!(drafter_vocab > 0, "DFlash drafter vocab must be non-zero");

        let vocab = target_vocab.min(drafter_vocab);
        ensure!(
            configured_rows <= u32::MAX as usize,
            "DFlash logits rows exceed the u32 kernel ABI: {configured_rows}"
        );
        ensure!(
            vocab <= u32::MAX as usize,
            "DFlash logits vocab exceeds the u32 kernel ABI: {vocab}"
        );

        let elements = configured_rows
            .checked_mul(vocab)
            .context("DFlash logits element count overflow")?;
        // Preserve the existing fixed-width debug probe without widening the
        // allocation. Real model vocabularies exceed this by several orders
        // of magnitude; tiny synthetic shapes fail closed.
        ensure!(
            elements >= FIXED_DEBUG_PROBE_ELEMENTS,
            "DFlash logits layout has {elements} elements, fewer than the fixed \
             {FIXED_DEBUG_PROBE_ELEMENTS}-element debug probe"
        );
        let allocation_bytes = elements
            .checked_mul(BF16_BYTES)
            .context("DFlash logits BF16 byte size overflow")?;

        Ok(Self {
            configured_rows,
            vocab,
            allocation_bytes,
        })
    }

    pub(super) fn allocation_bytes(self) -> usize {
        self.allocation_bytes
    }

    /// Return the exact active-row prefix that must be cleared this step.
    pub(super) fn active_bytes(self, active_rows: usize) -> Result<usize> {
        ensure!(
            active_rows > 0,
            "DFlash active logits rows must be non-zero"
        );
        ensure!(
            active_rows <= self.configured_rows,
            "DFlash active logits rows {active_rows} exceed configured rows {}",
            self.configured_rows
        );
        active_rows
            .checked_mul(self.vocab)
            .and_then(|elements| elements.checked_mul(BF16_BYTES))
            .context("DFlash active logits BF16 byte size overflow")
    }
}

#[cfg(test)]
mod tests {
    use super::LogitsLayout;

    #[test]
    fn prime_target_vocab_and_active_prefix_are_exact() {
        let layout = LogitsLayout::new(17, 248_077, 248_320).unwrap();

        assert_eq!(layout.vocab, 248_077);
        assert_eq!(layout.allocation_bytes(), 17 * 248_077 * 2);
        assert_eq!(layout.active_bytes(13).unwrap(), 13 * 248_077 * 2);
    }

    #[test]
    fn smaller_drafter_vocab_sets_the_compact_stride() {
        let layout = LogitsLayout::new(13, 248_320, 248_077).unwrap();

        assert_eq!(layout.vocab, 248_077);
        assert_eq!(layout.allocation_bytes(), 13 * 248_077 * 2);
    }

    #[test]
    fn zero_and_out_of_range_dimensions_fail_closed() {
        assert!(LogitsLayout::new(0, 248_077, 248_320).is_err());
        assert!(LogitsLayout::new(13, 0, 248_320).is_err());
        assert!(LogitsLayout::new(13, 248_077, 0).is_err());

        let layout = LogitsLayout::new(13, 248_077, 248_320).unwrap();
        assert!(layout.active_bytes(0).is_err());
        assert!(layout.active_bytes(14).is_err());
    }

    #[test]
    fn tiny_layout_cannot_underrun_the_fixed_debug_probe() {
        let err = LogitsLayout::new(3, 3, 5).unwrap_err();
        assert!(err.to_string().contains("10-element debug probe"));
    }

    #[test]
    fn kernel_abi_and_bf16_byte_overflow_fail_closed() {
        if usize::BITS > 32 {
            let too_wide = u32::MAX as usize + 1;
            assert!(LogitsLayout::new(10, too_wide, too_wide).is_err());

            let err = LogitsLayout::new(u32::MAX as usize, u32::MAX as usize, u32::MAX as usize)
                .unwrap_err();
            assert!(err.to_string().contains("BF16 byte size overflow"));
        }
    }
}
