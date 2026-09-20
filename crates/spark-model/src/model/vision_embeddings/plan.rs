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
    };
    let (mut index, mut image, mut source_row, mut position) = (0usize, 0usize, 0usize, 0u32);
    while index < tokens.len() {
        if tokens[index] != pad {
            if (start..end).contains(&index) {
                for axis in &mut result.positions {
                    axis.push(position);
                }
            }
            index += 1;
            position = position
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("MRoPE position overflow"))?;
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
        let lo = index.max(start);
        let hi = stop.min(end);
        if lo < hi {
            result.copies.push(CopyRun {
                source_row: source_row + lo - index,
                dest_row: lo - start,
                rows: hi - lo,
            });
            for token in lo..hi {
                let k = token - index;
                result.positions[0].push(position);
                result.positions[1].push(
                    position
                        .checked_add(u32::try_from(k / gw)?)
                        .ok_or_else(|| anyhow::anyhow!("MRoPE height overflow"))?,
                );
                result.positions[2].push(
                    position
                        .checked_add(u32::try_from(k % gw)?)
                        .ok_or_else(|| anyhow::anyhow!("MRoPE width overflow"))?,
                );
            }
        }
        position = position
            .checked_add(u32::try_from(gh.max(gw))?)
            .ok_or_else(|| anyhow::anyhow!("MRoPE position overflow"))?;
        index = stop;
        source_row += rows;
        image += 1;
    }
    ensure!(
        image == grids.len(),
        "prepared images have missing prompt pad spans"
    );
    ensure!(
        result.positions.iter().all(|p| p.len() == len),
        "vision position count mismatch"
    );
    Ok(result)
}
