// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed planning for exact M16 QSA and causal attention batching.

use anyhow::{Context, Result, bail, ensure};
use atlas_core::config::ModelConfig;

pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_ATTN_CORE16";
pub(crate) const DEVICE_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_ATTN_DEVICE16";
pub(crate) const HC_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_ATTN_HC16";
pub(crate) const CORE32_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_ATTN_CORE32";
pub(crate) const TILE_ROWS: usize = 16;
pub(crate) const TILE32_ROWS: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Plan {
    pub(crate) full_tiles: usize,
    pub(crate) tail_rows: usize,
    pub(crate) block_table_words: usize,
    pub(crate) seq_lens_offset: usize,
    pub(crate) scratch_bytes: usize,
}

fn parse(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("{SELECTOR} must be absent, 0, or 1; got {other:?}"),
    }
}

fn parse_device(value: Option<&str>, core_selected: bool) -> Result<bool> {
    let selected = match value {
        None | Some("0") => false,
        Some("1") => true,
        Some(other) => bail!("{DEVICE_SELECTOR} must be absent, 0, or 1; got {other:?}"),
    };
    ensure!(
        !selected || core_selected,
        "{DEVICE_SELECTOR}=1 requires {SELECTOR}=1"
    );
    Ok(selected)
}

fn parse_hc(value: Option<&str>, core_selected: bool, device_selected: bool) -> Result<bool> {
    let selected = match value {
        None | Some("0") => false,
        Some("1") => true,
        Some(other) => bail!("{HC_SELECTOR} must be absent, 0, or 1; got {other:?}"),
    };
    ensure!(
        !selected || (core_selected && device_selected),
        "{HC_SELECTOR}=1 requires {SELECTOR}=1 and {DEVICE_SELECTOR}=1"
    );
    Ok(selected)
}

fn parse_core32(
    value: Option<&str>,
    core_selected: bool,
    device_selected: bool,
    hc_selected: bool,
) -> Result<bool> {
    let selected = match value {
        None | Some("0") => false,
        Some("1") => true,
        Some(other) => bail!("{CORE32_SELECTOR} must be absent, 0, or 1; got {other:?}"),
    };
    ensure!(
        !selected || (core_selected && device_selected && hc_selected),
        "{CORE32_SELECTOR}=1 requires {SELECTOR}=1, {DEVICE_SELECTOR}=1, and {HC_SELECTOR}=1"
    );
    Ok(selected)
}

/// Turns the whole fast-prefill family off on this thread until the guard
/// drops (see [`super::family_suppressed`]). A prefill that `admit_surface`
/// rejects (a prompt longer than one chunk, a warm continuation, vision,
/// high-speed-swap) takes the ordinary prefill path instead of failing the
/// request; prompts the family admits are untouched.
#[must_use]
pub(crate) struct SuppressGuard(bool);

impl Drop for SuppressGuard {
    fn drop(&mut self) {
        super::set_family_suppressed(self.0);
    }
}

fn suppressed() -> bool {
    super::family_suppressed()
}

/// Admit this prefill surface to the family, or suppress the family for the
/// rest of the caller's scope (returned guard) when it cannot serve it.
pub(crate) fn admit_or_suppress(
    has_vision: bool,
    sequence_start: usize,
    chunk_start: usize,
    chunk_rows: usize,
    total_rows: usize,
    high_speed_swap: bool,
) -> Result<Option<SuppressGuard>> {
    if !selected()? {
        return Ok(None);
    }
    match admit_surface(has_vision, sequence_start, chunk_start, chunk_rows, total_rows, high_speed_swap) {
        Ok(()) => Ok(None),
        Err(reason) => {
            static LOGGED: std::sync::Once = std::sync::Once::new();
            LOGGED.call_once(|| {
                tracing::warn!(
                    "{SELECTOR}: fast-prefill family suppressed for this prefill ({reason:#}); using the ordinary prefill path (logged once)"
                )
            });
            Ok(Some(SuppressGuard(super::set_family_suppressed(true))))
        }
    }
}

pub(crate) fn selected() -> Result<bool> {
    if suppressed() {
        return Ok(false);
    }
    match std::env::var(SELECTOR) {
        Ok(value) => parse(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse(None),
        Err(error) => Err(error).with_context(|| format!("invalid {SELECTOR}")),
    }
}

pub(crate) fn device_selected() -> Result<bool> {
    if suppressed() {
        return Ok(false);
    }
    let core = selected()?;
    match std::env::var(DEVICE_SELECTOR) {
        Ok(value) => parse_device(Some(&value), core),
        Err(std::env::VarError::NotPresent) => parse_device(None, core),
        Err(error) => Err(error).with_context(|| format!("invalid {DEVICE_SELECTOR}")),
    }
}

pub(crate) fn hc_selected() -> Result<bool> {
    if suppressed() {
        return Ok(false);
    }
    let core = selected()?;
    let device = device_selected()?;
    match std::env::var(HC_SELECTOR) {
        Ok(value) => parse_hc(Some(&value), core, device),
        Err(std::env::VarError::NotPresent) => parse_hc(None, core, device),
        Err(error) => Err(error).with_context(|| format!("invalid {HC_SELECTOR}")),
    }
}

pub(crate) fn core32_selected() -> Result<bool> {
    if suppressed() {
        return Ok(false);
    }
    let core = selected()?;
    let device = device_selected()?;
    let hc = hc_selected()?;
    match std::env::var(CORE32_SELECTOR) {
        Ok(value) => parse_core32(Some(&value), core, device, hc),
        Err(std::env::VarError::NotPresent) => parse_core32(None, core, device, hc),
        Err(error) => Err(error).with_context(|| format!("invalid {CORE32_SELECTOR}")),
    }
}

pub(crate) fn admit_surface(
    has_vision: bool,
    sequence_start: usize,
    chunk_start: usize,
    chunk_rows: usize,
    total_rows: usize,
    high_speed_swap: bool,
) -> Result<()> {
    ensure!(
        !has_vision,
        "{SELECTOR} does not yet support vision prompts"
    );
    ensure!(
        sequence_start == 0 && chunk_start == 0,
        "{SELECTOR} requires a cold initial sequence"
    );
    ensure!(
        chunk_rows == total_rows,
        "{SELECTOR} requires one complete prefill chunk"
    );
    ensure!(
        !high_speed_swap,
        "{SELECTOR} is incompatible with high-speed-swap"
    );
    Ok(())
}

pub(crate) fn admit_request(
    config: &ModelConfig,
    rows: usize,
    start: usize,
    moe_selected: bool,
    exact_hyper: bool,
    exact_qkv16: bool,
    exact_o16: bool,
) -> Result<()> {
    core32_selected()?;
    if !selected()? {
        return Ok(());
    }
    ensure!(
        moe_selected && exact_hyper && exact_qkv16 && exact_o16,
        "{SELECTOR} requires exact MoE, HC, QKV16, and O16 prefill"
    );
    ensure!(
        config.is_qwen4_exp()
            && config.hidden_size == 2560
            && config.num_attention_heads == 24
            && config.num_key_value_heads == 2
            && config.head_dim == 256,
        "{SELECTOR} requires canonical Flash-Next attention geometry"
    );
    ensure!(
        start == 0 && rows >= TILE_ROWS && rows <= 2048,
        "{SELECTOR} requires initial 16..=2048-row prefill"
    );
    for name in [
        "ATLAS_QWEN4_QSA_PREFILL_GEMM",
        "ATLAS_QWEN4_ATTN_PREFILL_BATCH",
        "ATLAS_QWEN4_PREFILL_SSM_GEMM",
        "ATLAS_PAGED_DECODE_SPLITK",
    ] {
        match std::env::var(name) {
            Err(std::env::VarError::NotPresent) => {}
            Ok(value) if value == "0" => {}
            Ok(value) => bail!("{SELECTOR} rejects {name}={value:?}"),
            Err(error) => return Err(error).with_context(|| format!("invalid {name}")),
        }
    }
    Ok(())
}

pub(crate) fn plan(rows: usize, start: usize, max_blocks: usize, arena: usize) -> Result<Plan> {
    plan_for_tile(rows, start, max_blocks, arena, TILE_ROWS)
}

pub(crate) fn plan_for_tile(
    rows: usize,
    start: usize,
    max_blocks: usize,
    arena: usize,
    tile_rows: usize,
) -> Result<Plan> {
    ensure!(start == 0, "attention16 requires an initial prefill");
    ensure!(
        tile_rows == TILE_ROWS || tile_rows == TILE32_ROWS,
        "attention tile must be 16 or 32 rows"
    );
    ensure!(
        rows >= tile_rows && rows <= 2048,
        "attention16 row range is invalid"
    );
    ensure!(
        max_blocks > 0,
        "attention16 requires a complete block table"
    );
    let block_table_words = tile_rows
        .checked_mul(max_blocks)
        .ok_or_else(|| anyhow::anyhow!("attention16 block-table overflow"))?;
    let block_table_bytes = block_table_words
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| anyhow::anyhow!("attention16 block-table byte overflow"))?;
    let seq_lens_offset = block_table_bytes.next_multiple_of(8);
    let scratch_bytes = seq_lens_offset
        .checked_add(tile_rows * std::mem::size_of::<u32>())
        .ok_or_else(|| anyhow::anyhow!("attention16 metadata overflow"))?;
    ensure!(
        scratch_bytes <= arena,
        "attention16 metadata exceeds scratch arena"
    );
    Ok(Plan {
        full_tiles: rows / tile_rows,
        tail_rows: rows % tile_rows,
        block_table_words,
        seq_lens_offset,
        scratch_bytes,
    })
}

#[cfg(test)]
#[path = "attn16_tests.rs"]
mod tests;
