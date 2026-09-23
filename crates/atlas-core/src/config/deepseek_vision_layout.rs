// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, ensure};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum ImageTokenType {
    Start = 0,
    Pad = 1,
    Image = 2,
    NewLine = 3,
    End = 4,
}

#[derive(Debug, PartialEq)]
pub struct ImageBlock {
    pub start_pos: usize,
    pub types: Vec<ImageTokenType>,
    pub token_ids: Vec<u32>,
    /// For each IMAGE sentinel in final order, the original aligner grid row.
    pub aligner_permutation: Vec<usize>,
}

pub fn build_deepseek_image_block(
    h: usize,
    w: usize,
    start_pos: usize,
    vocab: u32,
) -> Result<ImageBlock> {
    use ImageTokenType::{End, Image, NewLine, Pad, Start};
    ensure!(h > 0 && w > 0, "DeepSeek aligner grid must be nonempty");
    vocab
        .checked_add(End as u32)
        .context("DeepSeek visual sentinel ID overflow")?;
    let rows = h
        .checked_add(h % 2)
        .context("DeepSeek padded grid height overflow")?;
    let row_len = w
        .checked_add(1)
        .context("DeepSeek aligner row length overflow")?;
    let total = deepseek_image_block_len(h, w, start_pos)?;
    let pad_last = (rows / 2 * row_len) % 2 * 2;
    let leading = 3 - start_pos % 4;
    ensure!(
        total <= 384,
        "DeepSeek image block exceeds supported 384-token budget"
    );
    start_pos
        .checked_add(total)
        .context("DeepSeek image position overflow")?;
    let mut types = vec![Pad; leading];
    types.push(Start);
    let mut aligner_permutation = Vec::with_capacity(h * w);
    for pair in 0..rows / 2 {
        for col in 0..row_len {
            for row in [2 * pair, 2 * pair + 1] {
                if row >= h {
                    types.push(Pad);
                } else if col == w {
                    types.push(NewLine);
                } else {
                    types.push(Image);
                    aligner_permutation.push(row * w + col);
                }
            }
        }
    }
    types.extend(std::iter::repeat_n(Pad, pad_last));
    types.push(End);
    let token_ids = types.iter().map(|kind| vocab + *kind as u32).collect();
    Ok(ImageBlock {
        start_pos,
        types,
        token_ids,
        aligner_permutation,
    })
}

/// Exact N-layout extent including absolute-position compression padding.
/// Used by both the resize budget solver and the executable sentinel builder.
pub fn deepseek_image_block_len(h: usize, w: usize, start_pos: usize) -> Result<usize> {
    ensure!(h > 0 && w > 0, "DeepSeek aligner grid must be nonempty");
    let rows = h
        .checked_add(h % 2)
        .context("DeepSeek grid height overflow")?;
    let width = w.checked_add(1).context("DeepSeek grid width overflow")?;
    let body = rows
        .checked_mul(width)
        .context("DeepSeek grid extent overflow")?;
    let tail = (body / 2) % 2 * 2;
    body.checked_add(tail + 3 - start_pos % 4 + 2)
        .context("DeepSeek image extent overflow")
}
