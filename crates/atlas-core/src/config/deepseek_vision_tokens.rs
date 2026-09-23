// SPDX-License-Identifier: AGPL-3.0-only

//! Validated sentinel spans shared by admission, embedding and attention.
//! Bounds use the pinned official Vision-Exp raw-arm mask; compressed KV
//! remains causal and must never consume these expanded bounds.

use anyhow::{Result, ensure};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepSeekImageSpan {
    /// Position of IMAGE_START, excluding compression pads before it.
    pub start: usize,
    /// Inclusive position of IMAGE_END.
    pub end: usize,
}

impl DeepSeekImageSpan {
    /// Exclusive upper raw-key bound. Outside this span this is the ordinary
    /// causal window. Callers validate positive window/max_tokens first.
    pub fn raw_bounds(&self, row: usize, window: usize, max_tokens: usize) -> (usize, usize) {
        let (left, right) = if (self.start..=self.end).contains(&row) {
            (
                (row - self.start).min(max_tokens.saturating_sub(1)),
                (self.end - row).min(max_tokens),
            )
        } else {
            (0, 0)
        };
        let lookback = window.saturating_sub(1).max(left);
        (
            row.saturating_sub(lookback),
            row.saturating_add(right).saturating_add(1),
        )
    }
}

/// Reject malformed sentinel streams before ordinary embedding/hash lookup.
/// The full prompt is validated once; callers additionally require every image
/// END to lie in the first prefill chunk and a matching prepared pixel input.
pub fn validate_deepseek_image_tokens(
    tokens: &[u32],
    vocab_size: u32,
    max_tokens: usize,
) -> Result<Vec<DeepSeekImageSpan>> {
    let limit = vocab_size
        .checked_add(5)
        .ok_or_else(|| anyhow::anyhow!("DeepSeek image token IDs overflow"))?;
    ensure!(
        (8..=384).contains(&max_tokens),
        "Unsupported image token budget"
    );
    let mut spans = Vec::new();
    let mut active: Option<(usize, usize)> = None;
    let mut pad_start: Option<usize> = None;
    for (position, &token) in tokens.iter().enumerate() {
        ensure!(
            token < limit,
            "Invalid DeepSeek token ID at position{position}"
        );
        if token < vocab_size {
            ensure!(
                active.is_none() && pad_start.is_none(),
                "Text splits a DeepSeek image sentinel block at position{position}"
            );
            continue;
        }
        match token - vocab_size {
            0 => {
                ensure!(active.is_none(), "Nested DeepSeek image START");
                ensure!(
                    position % 4 == 3,
                    "DeepSeek image START must align to3 mod4"
                );
                let block_start = pad_start.take().unwrap_or(position);
                ensure!(
                    position - block_start <= 3,
                    "Excess DeepSeek image compression padding"
                );
                active = Some((position, block_start));
            }
            1 if active.is_none() => {
                let start = *pad_start.get_or_insert(position);
                ensure!(
                    position - start < 3,
                    "Excess DeepSeek image compression padding"
                );
            }
            1..=3 => {
                ensure!(
                    active.is_some(),
                    "DeepSeek visual token outside an image block"
                );
            }
            4 => {
                let (start, block_start) = active
                    .take()
                    .ok_or_else(|| anyhow::anyhow!("Unmatched DeepSeek image END"))?;
                ensure!(
                    position - block_start < max_tokens,
                    "DeepSeek image block exceeds token budget"
                );
                ensure!(
                    tokens[start + 1..position].contains(&(vocab_size + 2)),
                    "DeepSeek image block has no visual patches"
                );
                spans.push(DeepSeekImageSpan {
                    start,
                    end: position,
                });
            }
            _ => unreachable!("ID range checked before match"),
        }
    }
    ensure!(
        active.is_none() && pad_start.is_none(),
        "Incomplete DeepSeek image block"
    );
    Ok(spans)
}
