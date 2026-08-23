// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::atomic::Ordering;

use anyhow::{Result, bail};

use super::{completed_flag, enabled};

pub fn parse_selector(raw: Option<&str>) -> Result<usize> {
    let raw = raw.ok_or_else(|| {
        anyhow::anyhow!("ATLAS_DFLASH_K1_STAGE_SEQ_LEN is required when stage diag is enabled")
    })?;
    if raw.is_empty() || raw.bytes().any(|byte| !byte.is_ascii_digit()) {
        bail!("ATLAS_DFLASH_K1_STAGE_SEQ_LEN must be an unsigned decimal integer");
    }
    raw.parse()
        .map_err(|_| anyhow::anyhow!("ATLAS_DFLASH_K1_STAGE_SEQ_LEN is out of range"))
}

pub fn parse_tokens_selector(raw: Option<&str>) -> Result<Vec<u32>> {
    let raw = raw.ok_or_else(|| {
        anyhow::anyhow!("ATLAS_DFLASH_K1_STAGE_TOKENS is required when stage diag is enabled")
    })?;
    if raw.is_empty() {
        bail!("ATLAS_DFLASH_K1_STAGE_TOKENS must be a nonempty comma-separated u32 list");
    }
    raw.split(',')
        .map(|part| {
            if part.is_empty() || part.bytes().any(|byte| !byte.is_ascii_digit()) {
                bail!("ATLAS_DFLASH_K1_STAGE_TOKENS must be a comma-separated u32 list");
            }
            part.parse::<u32>().map_err(|_| {
                anyhow::anyhow!("ATLAS_DFLASH_K1_STAGE_TOKENS contains an out-of-range token")
            })
        })
        .collect()
}

pub fn selector_matches(
    selected: usize,
    selected_tokens: &[u32],
    pre_verify_len: usize,
    tokens: &[u32],
    completed: bool,
) -> bool {
    selected == pre_verify_len && selected_tokens == tokens && !completed
}

pub fn requested_at(pre_verify_len: usize, tokens: &[u32]) -> Result<bool> {
    if !enabled() {
        return Ok(false);
    }
    let selected = parse_selector(
        std::env::var("ATLAS_DFLASH_K1_STAGE_SEQ_LEN")
            .ok()
            .as_deref(),
    )?;
    let selected_tokens = parse_tokens_selector(
        std::env::var("ATLAS_DFLASH_K1_STAGE_TOKENS")
            .ok()
            .as_deref(),
    )?;
    Ok(selector_matches(
        selected,
        &selected_tokens,
        pre_verify_len,
        tokens,
        completed_flag().load(Ordering::Relaxed),
    ))
}
