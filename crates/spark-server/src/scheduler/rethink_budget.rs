// SPDX-License-Identifier: AGPL-3.0-only

//! Shared spontaneous-thinking re-entry budget policy.

/// Resolve the content budget for a spontaneous `<think>` re-entry.
///
/// Each watchdog close halves the base budget, capped at four halvings. The
/// floor keeps the watchdog functional even for tiny or zero configured bases.
#[inline]
pub(in crate::scheduler) fn resolve_rethink_budget(base: u32, watchdog_fires: u32) -> u32 {
    (base >> watchdog_fires.min(4)).max(8)
}

#[cfg(test)]
#[path = "rethink_budget_tests.rs"]
mod tests;
