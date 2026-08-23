// SPDX-License-Identifier: AGPL-3.0-only

//! Exact schema-2 record formatting, separated from the hot-path collector.

use std::fmt;

use super::ActiveCycle;

pub(in crate::scheduler) struct CompleteRecord {
    pub(super) active: ActiveCycle,
    pub(super) accepted: u32,
    pub(super) emitted: u32,
    pub(super) total: u64,
}

impl fmt::Display for CompleteRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let p = self.active.phases;
        write!(
            f,
            "SPEC_CYCLE_V2 schema=2 req={:016x} pre={} k={} gamma={} accepted={} emitted={} c=1 async_requested=0 async_engaged=0 secondary_wait_enqueue_host_ns={} setup_host_ns={} verify_complete_host_ns={} accept_host_ns={} commit_enqueue_host_ns={} post_commit_enqueue_host_ns={} proposer_state_host_ns={} propose_complete_host_ns={} finalize_host_ns={} total_host_ns={} syncs_added=0",
            self.active.req,
            self.active.pre,
            self.active.gamma + 1,
            self.active.gamma,
            self.accepted,
            self.emitted,
            p[0],
            p[1],
            p[2],
            p[3],
            p[4],
            p[5],
            p[6],
            p[7],
            p[8],
            self.total,
        )
    }
}

pub(in crate::scheduler) struct TerminalRecord {
    pub(super) active: ActiveCycle,
    pub(super) verifier_accepted: u32,
    pub(super) accepted_emitted: u32,
    pub(super) bonus_emitted: u32,
    pub(super) emitted: u32,
    pub(super) branch: &'static str,
}

impl fmt::Display for TerminalRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SPEC_CYCLE_V2_TERMINAL schema=2 req={:016x} pre={} k={} gamma={} verifier_accepted={} accepted_emitted={} bonus_emitted={} emitted={} terminal_branch={} c=1 async_requested=0 async_engaged=0 syncs_added=0",
            self.active.req,
            self.active.pre,
            self.active.gamma + 1,
            self.active.gamma,
            self.verifier_accepted,
            self.accepted_emitted,
            self.bonus_emitted,
            self.emitted,
            self.branch,
        )
    }
}
