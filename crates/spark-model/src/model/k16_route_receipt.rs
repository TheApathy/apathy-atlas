// SPDX-License-Identifier: AGPL-3.0-only

//! One-frame route census for the qualification-only native K16 fixture.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use anyhow::{Result, bail, ensure};

const K16: usize = 16;
const EXPECTED_BATCHED_ENTRIES: usize = 1;
const EXPECTED_QKV_LAYERS: usize = 12;
const EXPECTED_SSM_LAYERS: usize = 36;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct RouteCounts {
    batched_entries: usize,
    qkv_layers: usize,
    ssm_layers: usize,
}

#[derive(Debug, Eq, PartialEq)]
pub(super) struct RouteReceipt {
    pub(super) pre_verify_len: usize,
    pub(super) tokens: Vec<u32>,
    pub(super) batched_entries: usize,
    pub(super) qkv_layers: usize,
    pub(super) ssm_layers: usize,
}

#[derive(Debug, Default)]
enum ReceiptState {
    #[default]
    Idle,
    Active {
        pre_verify_len: usize,
        tokens: Vec<u32>,
        counts: RouteCounts,
    },
}

#[derive(Clone, Copy)]
enum Route {
    BatchedEntry,
    QkvLayer,
    SsmLayer,
}

fn state() -> &'static Mutex<ReceiptState> {
    static STATE: OnceLock<Mutex<ReceiptState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(ReceiptState::Idle))
}

fn active() -> &'static AtomicBool {
    static ACTIVE: AtomicBool = AtomicBool::new(false);
    &ACTIVE
}

fn validate_counts(counts: RouteCounts) -> Result<()> {
    ensure!(
        counts.batched_entries == EXPECTED_BATCHED_ENTRIES,
        "K16 fixture requires exactly one production batched entry"
    );
    ensure!(
        counts.qkv_layers == EXPECTED_QKV_LAYERS,
        "K16 fixture requires exact QKV in all 12 attention layers"
    );
    ensure!(
        counts.ssm_layers == EXPECTED_SSM_LAYERS,
        "K16 fixture requires exact SSM in all 36 recurrent layers"
    );
    Ok(())
}

pub(super) fn begin(pre_verify_len: usize, tokens: &[u32]) -> Result<()> {
    ensure!(tokens.len() == K16, "K16 route receipt requires 16 inputs");
    let mut state = state().lock().expect("K16 route receipt mutex");
    ensure!(
        matches!(*state, ReceiptState::Idle),
        "K16 route receipt overlap"
    );
    *state = ReceiptState::Active {
        pre_verify_len,
        tokens: tokens.to_vec(),
        counts: RouteCounts::default(),
    };
    active().store(true, Ordering::Release);
    Ok(())
}

fn mark(route: Route, frame: Option<(usize, &[u32])>) -> Result<()> {
    if !active().load(Ordering::Acquire) {
        return Ok(());
    }
    let mut state = state().lock().expect("K16 route receipt mutex");
    let ReceiptState::Active {
        pre_verify_len,
        tokens,
        counts,
    } = &mut *state
    else {
        bail!("K16 route receipt active flag lacks an active frame");
    };
    if let Some((observed_len, observed_tokens)) = frame {
        ensure!(
            *pre_verify_len == observed_len && tokens == observed_tokens,
            "K16 batched route frame identity mismatch"
        );
    }
    let count = match route {
        Route::BatchedEntry => &mut counts.batched_entries,
        Route::QkvLayer => &mut counts.qkv_layers,
        Route::SsmLayer => &mut counts.ssm_layers,
    };
    *count = count
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("K16 route receipt counter overflow"))?;
    Ok(())
}

pub(crate) fn mark_batched_entry(pre_verify_len: usize, tokens: &[u32]) -> Result<()> {
    mark(Route::BatchedEntry, Some((pre_verify_len, tokens)))
}

pub(crate) fn mark_qkv_layer() -> Result<()> {
    mark(Route::QkvLayer, None)
}

pub(crate) fn mark_ssm_layer() -> Result<()> {
    mark(Route::SsmLayer, None)
}

pub(super) fn take_exact(pre_verify_len: usize, tokens: &[u32]) -> Result<RouteReceipt> {
    active().store(false, Ordering::Release);
    let mut state = state().lock().expect("K16 route receipt mutex");
    let ReceiptState::Active {
        pre_verify_len: recorded_len,
        tokens: recorded_tokens,
        counts,
    } = std::mem::take(&mut *state)
    else {
        bail!("K16 fixture lacks an active route receipt");
    };
    ensure!(
        recorded_len == pre_verify_len && recorded_tokens == tokens,
        "K16 completed route frame identity mismatch"
    );
    validate_counts(counts)?;
    Ok(RouteReceipt {
        pre_verify_len: recorded_len,
        tokens: recorded_tokens,
        batched_entries: counts.batched_entries,
        qkv_layers: counts.qkv_layers,
        ssm_layers: counts.ssm_layers,
    })
}

pub(super) fn abort() {
    active().store(false, Ordering::Release);
    *state().lock().expect("K16 route receipt mutex") = ReceiptState::Idle;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_census_is_the_only_admitted_route_receipt() {
        let exact = RouteCounts {
            batched_entries: 1,
            qkv_layers: 12,
            ssm_layers: 36,
        };
        validate_counts(exact).unwrap();
        for hostile in [
            RouteCounts {
                batched_entries: 0,
                ..exact
            },
            RouteCounts {
                batched_entries: 2,
                ..exact
            },
            RouteCounts {
                qkv_layers: 11,
                ..exact
            },
            RouteCounts {
                qkv_layers: 13,
                ..exact
            },
            RouteCounts {
                ssm_layers: 35,
                ..exact
            },
            RouteCounts {
                ssm_layers: 37,
                ..exact
            },
        ] {
            assert!(validate_counts(hostile).is_err());
        }
    }

    #[test]
    fn receipt_is_one_shot_and_binds_the_production_entry_frame() {
        abort();
        let tokens: Vec<u32> = (0..16).collect();
        begin(40, &tokens).unwrap();
        mark_batched_entry(40, &tokens).unwrap();
        for _ in 0..12 {
            mark_qkv_layer().unwrap();
        }
        for _ in 0..36 {
            mark_ssm_layer().unwrap();
        }
        let receipt = take_exact(40, &tokens).unwrap();
        assert_eq!(receipt.pre_verify_len, 40);
        assert_eq!(receipt.tokens, tokens);
        assert!(take_exact(40, &receipt.tokens).is_err());

        begin(40, &receipt.tokens).unwrap();
        let mut wrong = receipt.tokens.clone();
        wrong[15] += 1;
        assert!(mark_batched_entry(40, &wrong).is_err());
        abort();
    }
}
