// SPDX-License-Identifier: AGPL-3.0-only

//! Verification-purpose arithmetic scope, separate from prompt publication.
//!
//! Enter only around an admitted speculative target forward. This scope must
//! never enable prompt-only persistent KDA/conv writes, and must end before
//! oracle selection, accepted-state publication, or rejected-prefix replay.
//! It does not read the environment or perform GPU I/O. Numerical qualification
//! is incomplete, so the caller must explicitly opt into this experiment.

use std::cell::Cell;

pub(crate) fn parse_exact_verify_flag(value: Option<&str>) -> Result<bool, &'static str> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err("ATLAS_GLM53_EXACT_VERIFY must be absent, 0, or 1"),
    }
}

thread_local! {
    static EXACT_VERIFY_DEPTH: Cell<u32> = const { Cell::new(0) };
}

struct ExactVerifyGuard;

impl Drop for ExactVerifyGuard {
    fn drop(&mut self) {
        EXACT_VERIFY_DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
}

pub(crate) fn glm53_exact_verify_active() -> bool {
    EXACT_VERIFY_DEPTH.with(|depth| depth.get() != 0)
}

pub(crate) fn with_glm53_exact_verify<T>(operation: impl FnOnce() -> T) -> T {
    EXACT_VERIFY_DEPTH.with(|depth| {
        depth.set(
            depth
                .get()
                .checked_add(1)
                .expect("GLM exact-verification scope overflow"),
        )
    });
    let _guard = ExactVerifyGuard;
    operation()
}
