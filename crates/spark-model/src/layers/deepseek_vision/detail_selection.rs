// SPDX-License-Identifier: AGPL-3.0-only

//! Validated diagnostic detail selection; never changes encoder arithmetic.
#[derive(Clone, Copy)]
pub struct DetailSelection {
    block: usize,
}

impl DetailSelection {
    pub fn new(block: usize, actual_blocks: usize) -> Result<Self, &'static str> {
        if block >= actual_blocks {
            return Err("DeepSeek vision detail block is outside the loaded encoder");
        }
        Ok(Self { block })
    }

    pub fn is_selected(self, layer: usize) -> bool {
        layer == self.block
    }

    /// Called only when a diagnostic observer is present for this block.
    pub fn stage_name(self, suffix: &str) -> String {
        format!("block-{:02}-{suffix}", self.block)
    }
}
