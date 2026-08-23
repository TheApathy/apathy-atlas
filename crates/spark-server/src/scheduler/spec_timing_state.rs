// SPDX-License-Identifier: AGPL-3.0-only

//! Cached diagnostic gate and allocation-free C=1 request identity tracker.

use std::cell::Cell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Gate {
    pub(super) enabled: bool,
}

impl Gate {
    pub(super) fn resolve(
        flag: Option<&str>,
        async_flag: Option<&str>,
        max_batch: usize,
    ) -> Result<Self, &'static str> {
        match flag {
            None | Some("0") => Ok(Self { enabled: false }),
            Some("1") if max_batch != 1 => Err("spec-cycle v2 requires max_batch_size=1"),
            Some("1") if async_flag != Some("0") => {
                Err("spec-cycle v2 requires explicit ATLAS_DFLASH_ASYNC=0")
            }
            Some("1") => Ok(Self { enabled: true }),
            Some(_) => Err("ATLAS_DFLASH_SPEC_CYCLE_V2 must be 0 or 1"),
        }
    }

    pub(super) fn enabled() -> bool {
        GATE.get().is_some_and(|gate| gate.enabled)
    }
}

static GATE: OnceLock<Gate> = OnceLock::new();

pub(super) fn configure(max_batch: usize) -> Result<bool, &'static str> {
    let flag = match std::env::var("ATLAS_DFLASH_SPEC_CYCLE_V2") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err("ATLAS_DFLASH_SPEC_CYCLE_V2 is not valid Unicode");
        }
    };
    let async_flag = if flag.as_deref() == Some("1") {
        match std::env::var("ATLAS_DFLASH_ASYNC") {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err("ATLAS_DFLASH_ASYNC is not valid Unicode");
            }
        }
    } else {
        None
    };
    let gate = Gate::resolve(flag.as_deref(), async_flag.as_deref(), max_batch)?;
    match GATE.set(gate) {
        Ok(()) => Ok(gate.enabled),
        Err(_) if GATE.get() == Some(&gate) => Ok(gate.enabled),
        Err(_) => Err("spec-cycle v2 scheduler configuration changed after initialization"),
    }
}

#[derive(Clone, Copy)]
struct RequestState {
    id: u64,
    expected_pre: u64,
}

thread_local! {
    static REQUEST: Cell<RequestState> = const { Cell::new(RequestState { id: 0, expected_pre: 0 }) };
}
static NEXT_REQUEST: AtomicU64 = AtomicU64::new(1);

pub(super) fn begin_request(pre: u64) -> Option<u64> {
    REQUEST.with(|slot| {
        let mut state = slot.get();
        if state.id == 0 {
            state = RequestState {
                id: NEXT_REQUEST.fetch_add(1, Ordering::Relaxed),
                expected_pre: pre,
            };
            slot.set(state);
        }
        (state.expected_pre == pre).then_some(state.id)
    })
}

pub(super) fn finish_request(id: u64, pre: u64, next_pre: u64, terminal: bool) -> bool {
    REQUEST.with(|slot| {
        let mut state = slot.get();
        if state.id != id || state.expected_pre != pre || next_pre < pre {
            return false;
        }
        if terminal {
            state = RequestState {
                id: 0,
                expected_pre: 0,
            };
        } else {
            state.expected_pre = next_pre;
        }
        slot.set(state);
        true
    })
}

pub(super) fn abandon_request(id: u64) {
    REQUEST.with(|slot| {
        if slot.get().id == id {
            slot.set(RequestState {
                id: 0,
                expected_pre: 0,
            });
        }
    });
}
