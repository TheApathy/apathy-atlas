// SPDX-License-Identifier: AGPL-3.0-only

//! One full-prompt scan owns image-span validation, global rows and MRoPE.

use anyhow::{Result, ensure};

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct CopyRun {
    pub source_row: usize,
    pub dest_row: usize,
    pub rows: usize,
}

pub(crate) struct PromptPlan {
    pub copies: Vec<CopyRun>,
    pub positions: [Vec<u32>; 3],
    pub image_spans: Vec<crate::traits::ImageSpan>,
}

pub(super) fn plan(
    tokens: &[u32],
    pad: u32,
    grids: &[(usize, usize)],
    start: usize,
    len: usize,
) -> Result<PromptPlan> {
    let end = start
        .checked_add(len)
        .ok_or_else(|| anyhow::anyhow!("vision chunk range overflow"))?;
    ensure!(end <= tokens.len(), "vision chunk range exceeds prompt");
    ensure!(
        tokens.len() <= u32::MAX as usize,
        "vision prompt position overflow"
    );
    let mut result = PromptPlan {
        copies: Vec::new(),
        positions: std::array::from_fn(|_| Vec::with_capacity(len)),
        image_spans: Vec::with_capacity(grids.len()),
    };
    let (mut index, mut image, mut source_row) = (0usize, 0usize, 0usize);
    while index < tokens.len() {
        if tokens[index] != pad {
            index += 1;
            continue;
        }
        let &(gh, gw) = grids
            .get(image)
            .ok_or_else(|| anyhow::anyhow!("image pads without prepared image rows"))?;
        ensure!(gh > 0 && gw > 0, "invalid published vision grid");
        let rows = gh
            .checked_mul(gw)
            .ok_or_else(|| anyhow::anyhow!("vision span size overflow"))?;
        let stop = index
            .checked_add(rows)
            .ok_or_else(|| anyhow::anyhow!("vision span end overflow"))?;
        ensure!(
            stop <= tokens.len() && tokens[index..stop].iter().all(|t| *t == pad),
            "image pad run does not match the prepared grid"
        );
        result.image_spans.push(crate::traits::ImageSpan {
            start: index,
            height: gh,
            width: gw,
        });
        let lo = index.max(start);
        let hi = stop.min(end);
        if lo < hi {
            result.copies.push(CopyRun {
                source_row: source_row + lo - index,
                dest_row: lo - start,
                rows: hi - lo,
            });
        }
        index = stop;
        source_row += rows;
        image += 1;
    }
    ensure!(
        image == grids.len(),
        "prepared images have missing prompt pad spans"
    );
    // The persistent sequence map is also the prefill arithmetic SSOT.
    // No second image-position formula may drift between chunking and decode.
    let rotary = crate::traits::RotaryPositions::from_image_spans(
        tokens.len(),
        u32::MAX as usize,
        &result.image_spans,
    )?;
    for [t, h, w] in rotary.range(start, len)? {
        result.positions[0].push(t);
        result.positions[1].push(h);
        result.positions[2].push(w);
    }
    ensure!(
        result.positions.iter().all(|p| p.len() == len),
        "vision position count mismatch"
    );
    Ok(result)
}
