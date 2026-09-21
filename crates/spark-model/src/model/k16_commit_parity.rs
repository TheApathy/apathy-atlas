// SPDX-License-Identifier: AGPL-3.0-only

//! One-shot exact committed-state receipt for native Qwen4 K=16 verification.
//!
//! This module is deliberately reached only from the already-slow serial
//! oracle.  It is not consulted by ordinary decode or speculative timing.

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail, ensure};
use atlas_core::config::{DflashCaptureMode, LayerType};

use super::types::TransformerModel;
use crate::layer::SsmLayerState;
use crate::traits::SequenceState;

pub const ENV: &str = "ATLAS_DFLASH_K16_COMMIT_PARITY";
pub const SEQ_LEN_ENV: &str = "ATLAS_DFLASH_K16_COMMIT_PARITY_SEQ_LEN";
pub const TOKENS_ENV: &str = "ATLAS_DFLASH_K16_COMMIT_PARITY_TOKENS";

const SCHEMA: u32 = 1;
const K16: usize = 16;
const QWEN4_LAYERS: usize = 48;
const QWEN4_GDN_LAYERS: usize = 36;
const QWEN4_HIDDEN: usize = 2560;
const QWEN4_RESIDUAL: usize = 10240;
const QWEN38_FLASH_NEXT_PHYSICAL_VOCAB: usize = 248320;
const QWEN38_FLASH_NEXT_LOGICAL_VOCAB: usize = 248077;
const QWEN4_TAPS: [usize; 8] = [1, 7, 13, 20, 26, 33, 39, 46];

static COMPLETED: AtomicBool = AtomicBool::new(false);

pub(super) fn is_qwen38_flash_next_target_vocab(vocab_size: usize) -> bool {
    matches!(
        vocab_size,
        QWEN38_FLASH_NEXT_PHYSICAL_VOCAB | QWEN38_FLASH_NEXT_LOGICAL_VOCAB
    )
}

#[derive(Debug)]
pub struct CanonicalStateSnapshot {
    schema: u32,
    slot_idx: usize,
    gdn: Vec<CanonicalGdnState>,
    ple: CanonicalPleState,
}

#[derive(Debug)]
struct CanonicalGdnState {
    layer_idx: usize,
    h_bytes: usize,
    conv_bytes: usize,
    h: Vec<u8>,
    conv: Vec<u8>,
}

#[derive(Debug)]
struct CanonicalPleState {
    stride: usize,
    conv: Vec<u8>,
}

#[derive(Debug)]
struct CommittedStateSnapshot {
    schema: u32,
    slot_idx: usize,
    gdn: Vec<CommittedGdnState>,
    ple: CommittedPleState,
}

#[derive(Debug)]
struct CommittedGdnState {
    layer_idx: usize,
    h_bytes: usize,
    conv_bytes: usize,
    h_live: Vec<u8>,
    h_checkpoint: Vec<u8>,
    conv_live: Vec<u8>,
    conv_checkpoint: Vec<u8>,
}

#[derive(Debug)]
struct CommittedPleState {
    stride: usize,
    conv_live: Vec<u8>,
    conv_checkpoint: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ComparisonReceipt {
    compared_buffers: usize,
    compared_bytes: usize,
    state_fnv1a64: u64,
}

fn optional_env(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => bail!("{name} is not valid Unicode"),
    }
}

fn canonical_usize(name: &str, raw: &str) -> Result<usize> {
    let value = raw
        .parse::<usize>()
        .map_err(|_| anyhow::anyhow!("{name} must be a canonical decimal usize"))?;
    ensure!(
        value.to_string() == raw,
        "{name} must be a canonical decimal usize"
    );
    Ok(value)
}

fn canonical_u32_csv(name: &str, raw: &str) -> Result<Vec<u32>> {
    ensure!(
        !raw.is_empty()
            && !raw
                .bytes()
                .any(|byte| !(byte.is_ascii_digit() || byte == b',')),
        "{name} must be a nonempty comma-separated canonical u32 vector"
    );
    raw.split(',')
        .map(|item| {
            let value = item.parse::<u32>().map_err(|_| {
                anyhow::anyhow!("{name} must be a nonempty comma-separated canonical u32 vector")
            })?;
            ensure!(
                value.to_string() == item,
                "{name} must be a nonempty comma-separated canonical u32 vector"
            );
            Ok(value)
        })
        .collect()
}

fn selector_values_match(
    enabled: Option<&str>,
    seq_len: Option<&str>,
    selected_tokens: Option<&str>,
    pre_verify_len: usize,
    tokens: &[u32],
    completed: bool,
) -> Result<bool> {
    let seq_len = seq_len.filter(|value| !value.is_empty());
    let selected_tokens = selected_tokens.filter(|value| !value.is_empty());
    match enabled {
        None | Some("0") => {
            ensure!(
                seq_len.is_none() && selected_tokens.is_none(),
                "K16 commit-parity selector variables require {ENV}=1"
            );
            Ok(false)
        }
        Some("1") => {
            let selected_seq_len = canonical_usize(
                SEQ_LEN_ENV,
                seq_len.ok_or_else(|| anyhow::anyhow!("{ENV}=1 requires {SEQ_LEN_ENV}"))?,
            )?;
            let selected_tokens = canonical_u32_csv(
                TOKENS_ENV,
                selected_tokens.ok_or_else(|| anyhow::anyhow!("{ENV}=1 requires {TOKENS_ENV}"))?,
            )?;
            ensure!(
                selected_tokens.len() == K16,
                "{TOKENS_ENV} must contain exactly 16 K16 inputs"
            );
            Ok(!completed && selected_seq_len == pre_verify_len && selected_tokens == tokens)
        }
        Some(_) => bail!("{ENV} must be exactly 0 or 1"),
    }
}

pub(super) fn requested_at(
    model: &TransformerModel,
    pre_verify_len: usize,
    tokens: &[u32],
) -> Result<bool> {
    let requested = selector_values_match(
        optional_env(ENV)?.as_deref(),
        optional_env(SEQ_LEN_ENV)?.as_deref(),
        optional_env(TOKENS_ENV)?.as_deref(),
        pre_verify_len,
        tokens,
        COMPLETED.load(Ordering::Acquire),
    )?;
    if !requested {
        return Ok(false);
    }
    validate_exact_model(model)?;
    ensure!(
        tokens.len() == K16,
        "K16 commit parity requires exactly 16 inputs"
    );
    ensure!(
        optional_env("ATLAS_QWEN4_K16_BATCHED_VERIFY")?.as_deref() == Some("1"),
        "K16 commit parity requires ATLAS_QWEN4_K16_BATCHED_VERIFY=1"
    );
    ensure!(
        optional_env("ATLAS_DFLASH_SERIAL_COMMIT")?.as_deref() == Some("1")
            && optional_env("ATLAS_DFLASH_SKIP_REPROPOSE")?.as_deref() == Some("1"),
        "K16 commit parity requires the serial-commit/skip-repropose oracle"
    );
    Ok(true)
}

fn validate_exact_model(model: &TransformerModel) -> Result<()> {
    let config = &model.config;
    ensure!(
        config.is_qwen4_exp(),
        "K16 commit parity requires Qwen4-Exp"
    );
    ensure!(
        config.hidden_size == QWEN4_HIDDEN
            && config.residual_width() == QWEN4_RESIDUAL
            && is_qwen38_flash_next_target_vocab(config.vocab_size)
            && config.num_hidden_layers == QWEN4_LAYERS,
        "K16 commit parity requires exact Qwen4 H2560/R10240/V248320-or-248077/T48 geometry"
    );
    ensure!(
        model.dflash_capture_width == QWEN4_HIDDEN
            && model.dflash_capture_offset == 0
            && model.dflash_capture_mode == DflashCaptureMode::Qwen4HyperProjected
            && model.dflash_capture_layers == QWEN4_TAPS,
        "K16 commit parity requires exact native Qwen4 hyper-projected capture"
    );
    let proposer = model
        .active_proposer()
        .context("K16 commit parity requires an active native DFlash proposer")?;
    ensure!(
        proposer.is_dflash()
            && proposer.physical_verify_k() == Some(K16)
            && model.ddtree_parent_ids_capacity == K16
            && model.ssm_pool.num_intermediates == K16,
        "K16 commit parity requires exact native DFlash physical K16 capacity"
    );
    ensure!(
        model.qwen4_ple.is_some(),
        "K16 commit parity requires Qwen4 PLE state"
    );
    let gdn_layers = (0..config.num_hidden_layers)
        .filter(|&layer| config.layer_type(layer) == LayerType::LinearAttention)
        .count();
    ensure!(
        gdn_layers == QWEN4_GDN_LAYERS,
        "K16 commit parity requires exactly 36 GDN layers, got {gdn_layers}"
    );
    Ok(())
}

fn state_sizes(model: &TransformerModel) -> Result<(usize, usize)> {
    let config = &model.config;
    let h_elements = config
        .linear_num_value_heads
        .checked_mul(config.linear_value_head_dim)
        .and_then(|value| value.checked_mul(config.linear_key_head_dim))
        .context("K16 commit parity h-state extent overflow")?;
    let conv_dim = config
        .linear_num_key_heads
        .checked_mul(config.linear_key_head_dim)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| {
            config
                .linear_num_value_heads
                .checked_mul(config.linear_value_head_dim)
                .and_then(|tail| value.checked_add(tail))
        })
        .context("K16 commit parity conv-state extent overflow")?;
    let h_bytes = h_elements
        .checked_mul(4)
        .context("K16 commit parity h-state byte extent overflow")?;
    let conv_bytes = conv_dim
        .checked_mul(config.linear_conv_kernel_dim)
        .and_then(|value| value.checked_mul(4))
        .context("K16 commit parity conv-state byte extent overflow")?;
    ensure!(
        h_bytes > 0 && conv_bytes > 0,
        "K16 commit parity empty GDN extent"
    );
    Ok((h_bytes, conv_bytes))
}

fn validate_sequence(model: &TransformerModel, seq: &SequenceState) -> Result<(usize, usize)> {
    validate_exact_model(model)?;
    ensure!(
        seq.layer_states.len() == QWEN4_LAYERS,
        "K16 commit parity sequence has {} layer states, expected 48",
        seq.layer_states.len()
    );
    ensure!(
        seq.slot_idx < model.ssm_pool.max_slots,
        "K16 commit parity slot {} exceeds SSM pool capacity {}",
        seq.slot_idx,
        model.ssm_pool.max_slots
    );
    state_sizes(model)
}

fn gdn_state<'a>(
    model: &TransformerModel,
    seq: &'a SequenceState,
    layer_idx: usize,
) -> Result<&'a SsmLayerState> {
    ensure!(
        model.config.layer_type(layer_idx) == LayerType::LinearAttention,
        "K16 commit parity layer {layer_idx} is not GDN"
    );
    let state = seq.layer_states[layer_idx]
        .as_any()
        .downcast_ref::<SsmLayerState>()
        .ok_or_else(|| {
            anyhow::anyhow!("K16 commit parity missing SsmLayerState at layer {layer_idx}")
        })?;
    ensure!(
        !state.h_is_f16,
        "K16 commit parity requires FP32 h-state at layer {layer_idx}"
    );
    ensure!(
        !state.h_state.is_null() && !state.conv_state.is_null(),
        "K16 commit parity null live state at layer {layer_idx}"
    );
    Ok(state)
}

pub(super) fn capture_live(
    model: &TransformerModel,
    seq: &SequenceState,
) -> Result<CanonicalStateSnapshot> {
    let (h_bytes, conv_bytes) = validate_sequence(model, seq)?;
    let stream = model.gpu.default_stream();
    model
        .gpu
        .synchronize(stream)
        .context("K16 commit parity drain live-state producer stream")?;
    let mut gdn = Vec::with_capacity(QWEN4_GDN_LAYERS);
    for layer_idx in 0..QWEN4_LAYERS {
        if model.config.layer_type(layer_idx) != LayerType::LinearAttention {
            continue;
        }
        let state = gdn_state(model, seq, layer_idx)?;
        let mut h = vec![0u8; h_bytes];
        let mut conv = vec![0u8; conv_bytes];
        model.gpu.copy_d2h(state.h_state, &mut h)?;
        model.gpu.copy_d2h(state.conv_state, &mut conv)?;
        gdn.push(CanonicalGdnState {
            layer_idx,
            h_bytes,
            conv_bytes,
            h,
            conv,
        });
    }
    ensure!(
        gdn.len() == QWEN4_GDN_LAYERS,
        "K16 commit parity captured {} GDN layers, expected 36",
        gdn.len()
    );
    let ple = model
        .qwen4_ple
        .as_ref()
        .context("K16 commit parity missing PLE")?;
    let (stride, conv, _) = ple.parity_snapshot(seq.slot_idx, model.gpu.as_ref(), stream)?;
    let snapshot = CanonicalStateSnapshot {
        schema: SCHEMA,
        slot_idx: seq.slot_idx,
        gdn,
        ple: CanonicalPleState { stride, conv },
    };
    validate_canonical(&snapshot)?;
    Ok(snapshot)
}

pub(super) fn restore_live(
    model: &TransformerModel,
    seq: &SequenceState,
    snapshot: &CanonicalStateSnapshot,
) -> Result<()> {
    let (h_bytes, conv_bytes) = validate_sequence(model, seq)?;
    validate_canonical(snapshot)?;
    ensure!(
        snapshot.slot_idx == seq.slot_idx,
        "K16 commit parity restore slot mismatch"
    );
    let mut cursor = 0usize;
    for layer_idx in 0..QWEN4_LAYERS {
        if model.config.layer_type(layer_idx) != LayerType::LinearAttention {
            continue;
        }
        let expected = snapshot
            .gdn
            .get(cursor)
            .context("K16 commit parity restore missing ordered GDN layer")?;
        ensure!(
            expected.layer_idx == layer_idx
                && expected.h_bytes == h_bytes
                && expected.conv_bytes == conv_bytes,
            "K16 commit parity restore GDN order/extent mismatch at layer {layer_idx}"
        );
        let state = gdn_state(model, seq, layer_idx)?;
        model.gpu.copy_h2d(&expected.h, state.h_state)?;
        model.gpu.copy_h2d(&expected.conv, state.conv_state)?;
        cursor += 1;
    }
    ensure!(
        cursor == snapshot.gdn.len(),
        "K16 commit parity restore trailing GDN state"
    );
    model
        .qwen4_ple
        .as_ref()
        .context("K16 commit parity restore missing PLE")?
        .parity_restore_live(
            seq.slot_idx,
            snapshot.ple.stride,
            &snapshot.ple.conv,
            model.gpu.as_ref(),
        )?;
    Ok(())
}

pub(super) fn drain_default(model: &TransformerModel) -> Result<()> {
    model
        .gpu
        .synchronize(model.gpu.default_stream())
        .context("K16 commit parity drain restored default-stream state")
}

fn capture_committed(
    model: &TransformerModel,
    seq: &SequenceState,
) -> Result<CommittedStateSnapshot> {
    let (h_bytes, conv_bytes) = validate_sequence(model, seq)?;
    let default_stream = model.gpu.default_stream();
    ensure!(
        model.secondary_event != 0,
        "K16 commit parity missing secondary event"
    );
    ensure!(
        model.secondary_stream != default_stream,
        "K16 commit parity secondary/default stream identity collision"
    );
    model
        .gpu
        .stream_wait_event(default_stream, model.secondary_event)
        .context("K16 commit parity post-commit event wait enqueue")?;
    model
        .gpu
        .synchronize(default_stream)
        .context("K16 commit parity post-commit event wait completion")?;

    let mut gdn = Vec::with_capacity(QWEN4_GDN_LAYERS);
    for layer_idx in 0..QWEN4_LAYERS {
        if model.config.layer_type(layer_idx) != LayerType::LinearAttention {
            continue;
        }
        let state = gdn_state(model, seq, layer_idx)?;
        let h_checkpoint = state
            .h_state_checkpoint
            .filter(|ptr| !ptr.is_null())
            .with_context(|| {
                format!("K16 commit parity missing h checkpoint at layer {layer_idx}")
            })?;
        let conv_checkpoint = state
            .conv_state_checkpoint
            .filter(|ptr| !ptr.is_null())
            .with_context(|| {
                format!("K16 commit parity missing conv checkpoint at layer {layer_idx}")
            })?;
        let mut h_live = vec![0u8; h_bytes];
        let mut h_checkpoint_bytes = vec![0u8; h_bytes];
        let mut conv_live = vec![0u8; conv_bytes];
        let mut conv_checkpoint_bytes = vec![0u8; conv_bytes];
        model.gpu.copy_d2h(state.h_state, &mut h_live)?;
        model.gpu.copy_d2h(h_checkpoint, &mut h_checkpoint_bytes)?;
        model.gpu.copy_d2h(state.conv_state, &mut conv_live)?;
        model
            .gpu
            .copy_d2h(conv_checkpoint, &mut conv_checkpoint_bytes)?;
        gdn.push(CommittedGdnState {
            layer_idx,
            h_bytes,
            conv_bytes,
            h_live,
            h_checkpoint: h_checkpoint_bytes,
            conv_live,
            conv_checkpoint: conv_checkpoint_bytes,
        });
    }
    let ple = model
        .qwen4_ple
        .as_ref()
        .context("K16 commit parity missing PLE")?;
    let (stride, conv_live, conv_checkpoint) =
        ple.parity_snapshot(seq.slot_idx, model.gpu.as_ref(), default_stream)?;
    Ok(CommittedStateSnapshot {
        schema: SCHEMA,
        slot_idx: seq.slot_idx,
        gdn,
        ple: CommittedPleState {
            stride,
            conv_live,
            conv_checkpoint,
        },
    })
}

fn validate_canonical(snapshot: &CanonicalStateSnapshot) -> Result<()> {
    ensure!(
        snapshot.schema == SCHEMA,
        "K16 commit parity canonical schema mismatch"
    );
    ensure!(
        snapshot.gdn.len() == QWEN4_GDN_LAYERS,
        "K16 commit parity canonical GDN census mismatch"
    );
    let mut previous = None;
    for state in &snapshot.gdn {
        ensure!(
            previous.is_none_or(|layer| state.layer_idx > layer),
            "K16 commit parity canonical GDN layer order mismatch"
        );
        ensure!(
            state.h_bytes > 0
                && state.conv_bytes > 0
                && state.h.len() == state.h_bytes
                && state.conv.len() == state.conv_bytes,
            "K16 commit parity canonical GDN extent mismatch at layer {}",
            state.layer_idx
        );
        previous = Some(state.layer_idx);
    }
    ensure!(
        snapshot.ple.stride > 0 && snapshot.ple.conv.len() == snapshot.ple.stride,
        "K16 commit parity canonical PLE stride mismatch"
    );
    Ok(())
}

fn fnv1a64_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn first_mismatch(expected: &[u8], actual: &[u8]) -> Option<usize> {
    expected
        .iter()
        .zip(actual)
        .position(|(left, right)| left != right)
        .or_else(|| (expected.len() != actual.len()).then_some(expected.len().min(actual.len())))
}

fn compare_buffer(label: &str, expected: &[u8], actual: &[u8]) -> Result<()> {
    if let Some(byte) = first_mismatch(expected, actual) {
        let expected_hash = fnv1a64_update(0xcbf29ce484222325, expected);
        let actual_hash = fnv1a64_update(0xcbf29ce484222325, actual);
        bail!(
            "K16 commit parity byte mismatch at {label} byte={byte} expected_len={} actual_len={} expected_fnv1a64={expected_hash:016x} actual_fnv1a64={actual_hash:016x}",
            expected.len(),
            actual.len()
        );
    }
    Ok(())
}

fn compare_snapshots(
    expected: &CanonicalStateSnapshot,
    actual: &CommittedStateSnapshot,
) -> Result<ComparisonReceipt> {
    validate_canonical(expected)?;
    ensure!(
        actual.schema == SCHEMA,
        "K16 commit parity committed schema mismatch"
    );
    ensure!(
        expected.slot_idx == actual.slot_idx,
        "K16 commit parity committed slot mismatch"
    );
    ensure!(
        actual.gdn.len() == QWEN4_GDN_LAYERS,
        "K16 commit parity committed GDN census mismatch"
    );
    let mut compared_buffers = 0usize;
    let mut compared_bytes = 0usize;
    let mut hash = 0xcbf29ce484222325;
    for (expected_state, actual_state) in expected.gdn.iter().zip(&actual.gdn) {
        ensure!(
            expected_state.layer_idx == actual_state.layer_idx,
            "K16 commit parity committed GDN order mismatch: expected layer {}, got {}",
            expected_state.layer_idx,
            actual_state.layer_idx
        );
        ensure!(
            expected_state.h_bytes == actual_state.h_bytes
                && expected_state.conv_bytes == actual_state.conv_bytes,
            "K16 commit parity committed GDN extent metadata mismatch at layer {}",
            expected_state.layer_idx
        );
        for (name, expected_bytes, actual_bytes) in [
            (
                "h_live",
                expected_state.h.as_slice(),
                actual_state.h_live.as_slice(),
            ),
            (
                "h_checkpoint",
                expected_state.h.as_slice(),
                actual_state.h_checkpoint.as_slice(),
            ),
            (
                "conv_live",
                expected_state.conv.as_slice(),
                actual_state.conv_live.as_slice(),
            ),
            (
                "conv_checkpoint",
                expected_state.conv.as_slice(),
                actual_state.conv_checkpoint.as_slice(),
            ),
        ] {
            compare_buffer(
                &format!("gdn.layer_{:02}.{name}", expected_state.layer_idx),
                expected_bytes,
                actual_bytes,
            )?;
            compared_buffers += 1;
            compared_bytes = compared_bytes
                .checked_add(expected_bytes.len())
                .context("K16 commit parity compared-byte overflow")?;
        }
        hash = fnv1a64_update(hash, &(expected_state.layer_idx as u64).to_le_bytes());
        hash = fnv1a64_update(hash, &expected_state.h);
        hash = fnv1a64_update(hash, &expected_state.conv);
    }
    ensure!(
        expected.ple.stride == actual.ple.stride,
        "K16 commit parity committed PLE stride mismatch"
    );
    for (name, actual_bytes) in [
        ("ple.conv_live", actual.ple.conv_live.as_slice()),
        ("ple.conv_checkpoint", actual.ple.conv_checkpoint.as_slice()),
    ] {
        compare_buffer(name, &expected.ple.conv, actual_bytes)?;
        compared_buffers += 1;
        compared_bytes = compared_bytes
            .checked_add(expected.ple.conv.len())
            .context("K16 commit parity compared-byte overflow")?;
    }
    hash = fnv1a64_update(hash, &expected.ple.conv);
    ensure!(
        compared_buffers == QWEN4_GDN_LAYERS * 4 + 2,
        "K16 commit parity compared-buffer census mismatch"
    );
    Ok(ComparisonReceipt {
        compared_buffers,
        compared_bytes,
        state_fnv1a64: hash,
    })
}

fn validate_frame(total_accepted: usize, k: usize, last_inter_slot: usize) -> Result<()> {
    ensure!(k == K16, "K16 commit parity expected k=16, got {k}");
    ensure!(
        (1..=k).contains(&total_accepted),
        "K16 commit parity total_accepted must be in 1..=16, got {total_accepted}"
    );
    ensure!(
        last_inter_slot == total_accepted - 1,
        "K16 commit parity requires flat row order: last_inter_slot={last_inter_slot}, total_accepted={total_accepted}"
    );
    Ok(())
}

fn canonical_tokens(tokens: &[u32]) -> String {
    tokens
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

pub(super) fn compare_committed(
    model: &TransformerModel,
    seq: &SequenceState,
    expected: &CanonicalStateSnapshot,
    pre_verify_len: usize,
    tokens: &[u32],
    total_accepted: usize,
    k: usize,
    last_inter_slot: usize,
) -> Result<()> {
    ensure!(
        tokens.len() == K16,
        "K16 commit parity frame token census mismatch"
    );
    validate_frame(total_accepted, k, last_inter_slot)?;
    let actual = capture_committed(model, seq)?;
    let receipt = compare_snapshots(expected, &actual)?;
    COMPLETED.store(true, Ordering::Release);
    tracing::info!(
        "DFLASH_K16_COMMIT_PARITY schema={} route=native_qwen4_dflash_physical_k16_batched pre_verify_len={} tokens={} total_accepted={} k={} last_inter_slot={} slot={} gdn_layers={} compared_buffers={} compared_bytes={} state_fnv1a64={:016x} match=true",
        SCHEMA,
        pre_verify_len,
        canonical_tokens(tokens),
        total_accepted,
        k,
        last_inter_slot,
        seq.slot_idx,
        QWEN4_GDN_LAYERS,
        receipt.compared_buffers,
        receipt.compared_bytes,
        receipt.state_fnv1a64,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flash_next_target_vocab_accepts_only_physical_or_canonical_logical() {
        for admitted in [248320, 248077] {
            assert!(is_qwen38_flash_next_target_vocab(admitted));
        }
        for rejected in [0, 248076, 248078, 248319, 248321, usize::MAX] {
            assert!(
                !is_qwen38_flash_next_target_vocab(rejected),
                "hostile target vocabulary unexpectedly admitted: {rejected}"
            );
        }
    }

    fn canonical() -> CanonicalStateSnapshot {
        CanonicalStateSnapshot {
            schema: SCHEMA,
            slot_idx: 3,
            gdn: (0..QWEN4_GDN_LAYERS)
                .map(|layer_idx| CanonicalGdnState {
                    layer_idx,
                    h_bytes: 2,
                    conv_bytes: 3,
                    h: vec![layer_idx as u8, 1],
                    conv: vec![layer_idx as u8, 2, 3],
                })
                .collect(),
            ple: CanonicalPleState {
                stride: 4,
                conv: vec![4, 5, 6, 7],
            },
        }
    }

    fn committed(expected: &CanonicalStateSnapshot) -> CommittedStateSnapshot {
        CommittedStateSnapshot {
            schema: SCHEMA,
            slot_idx: expected.slot_idx,
            gdn: expected
                .gdn
                .iter()
                .map(|state| CommittedGdnState {
                    layer_idx: state.layer_idx,
                    h_bytes: state.h_bytes,
                    conv_bytes: state.conv_bytes,
                    h_live: state.h.clone(),
                    h_checkpoint: state.h.clone(),
                    conv_live: state.conv.clone(),
                    conv_checkpoint: state.conv.clone(),
                })
                .collect(),
            ple: CommittedPleState {
                stride: expected.ple.stride,
                conv_live: expected.ple.conv.clone(),
                conv_checkpoint: expected.ple.conv.clone(),
            },
        }
    }

    #[test]
    fn selector_is_exact_default_off_and_one_shot() {
        let tokens: Vec<u32> = (1..=16).collect();
        assert!(!selector_values_match(None, None, None, 7, &tokens, false).unwrap());
        assert!(!selector_values_match(Some("0"), None, None, 7, &tokens, false).unwrap());
        assert!(selector_values_match(Some("0"), Some("7"), None, 7, &tokens, false).is_err());
        assert!(selector_values_match(Some("2"), Some("7"), Some("1"), 7, &tokens, false).is_err());
        assert!(selector_values_match(Some("1"), None, Some("1"), 7, &tokens, false).is_err());
        assert!(
            selector_values_match(Some("1"), Some("07"), Some("1"), 7, &tokens, false).is_err()
        );
        assert!(
            selector_values_match(Some("1"), Some("7"), Some("01"), 7, &tokens, false).is_err()
        );
        let csv = canonical_tokens(&tokens);
        assert!(
            selector_values_match(Some("1"), Some("7"), Some(&csv), 7, &tokens, false).unwrap()
        );
        assert!(
            !selector_values_match(Some("1"), Some("7"), Some(&csv), 8, &tokens, false).unwrap()
        );
        assert!(
            !selector_values_match(Some("1"), Some("7"), Some(&csv), 7, &tokens, true).unwrap()
        );
    }

    #[test]
    fn exact_match_counts_all_live_and_checkpoint_buffers() {
        let expected = canonical();
        let actual = committed(&expected);
        let receipt = compare_snapshots(&expected, &actual).unwrap();
        assert_eq!(receipt.compared_buffers, 146);
        assert_eq!(
            receipt.compared_bytes,
            QWEN4_GDN_LAYERS * (2 * 2 + 3 * 2) + 4 * 2
        );
        assert_ne!(receipt.state_fnv1a64, 0);
    }

    #[test]
    fn rejects_missing_reordered_or_shape_drifted_state() {
        let expected = canonical();
        let mut missing = committed(&expected);
        missing.gdn.pop();
        assert!(compare_snapshots(&expected, &missing).is_err());

        let mut reordered = committed(&expected);
        reordered.gdn.swap(0, 1);
        assert!(compare_snapshots(&expected, &reordered).is_err());

        let mut extent = committed(&expected);
        extent.gdn[0].h_bytes += 1;
        assert!(compare_snapshots(&expected, &extent).is_err());

        let mut ple = committed(&expected);
        ple.ple.stride += 1;
        assert!(compare_snapshots(&expected, &ple).is_err());
    }

    #[test]
    fn rejects_any_live_or_checkpoint_byte_drift() {
        let expected = canonical();
        for mutate in 0..6 {
            let mut actual = committed(&expected);
            match mutate {
                0 => actual.gdn[0].h_live[0] ^= 1,
                1 => actual.gdn[0].h_checkpoint[0] ^= 1,
                2 => actual.gdn[0].conv_live[0] ^= 1,
                3 => actual.gdn[0].conv_checkpoint[0] ^= 1,
                4 => actual.ple.conv_live[0] ^= 1,
                _ => actual.ple.conv_checkpoint[0] ^= 1,
            }
            assert!(compare_snapshots(&expected, &actual).is_err());
        }
    }

    #[test]
    fn frame_contract_rejects_wrong_width_bounds_or_order() {
        validate_frame(1, 16, 0).unwrap();
        validate_frame(16, 16, 15).unwrap();
        assert!(validate_frame(0, 16, 0).is_err());
        assert!(validate_frame(17, 16, 16).is_err());
        assert!(validate_frame(8, 15, 7).is_err());
        assert!(validate_frame(8, 16, 8).is_err());
    }

    #[test]
    fn static_contract_records_commit_event_after_every_state_write() {
        let source = include_str!("trait_impl/async_chkpt.rs");
        let commit = source
            .split_once("pub(super) fn commit_verify_state_async_dispatch")
            .unwrap()
            .1;
        let ple_write = commit
            .rfind("ple.rollback_and_checkpoint")
            .expect("commit must write PLE live/checkpoint state");
        let event = commit
            .rfind("record_event(self.secondary_event, stream)")
            .expect("commit must record its completion event");
        assert!(
            ple_write < event,
            "commit event must follow all recurrent writes"
        );
    }
}
