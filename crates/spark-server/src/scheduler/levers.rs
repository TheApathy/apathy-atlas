// SPDX-License-Identifier: AGPL-3.0-only

//! Run-scoped scheduler levers the dashboard can toggle.
//!
//! MINIMAL PORT BY DESIGN. Upstream's `levers.rs` is 364 lines carrying ~20
//! decode/verify/speculation flags resolved from `AVAROK_*` at run start. That
//! whole configuration regime belongs to upstream's scheduler, not ours, and
//! porting it would import twenty knobs this engine's decode path does not read
//! — each of which would then be a lever the Ops pane could toggle to no effect.
//!
//! What the dashboard actually touches is one field: the loop watchdog, which
//! `/watchdog on|off` flips while the server is serving. Upstream makes exactly
//! that field an `AtomicBool` inside the carried struct for the reason stated in
//! its own module docs — the mutation is real, so it is modelled, but it stays
//! INSIDE the run's state rather than becoming a process global that outlives
//! the run whose flags it encodes. That reasoning is worth keeping even when the
//! struct around it has one field.

use std::sync::atomic::{AtomicBool, Ordering};

/// Levers for one run. Published to the dashboard when the scheduler starts and
/// replaced on a hot-swap, so `/watchdog` always addresses the live run.
pub struct SchedLevers {
    loop_watchdog: AtomicBool,
}

impl SchedLevers {
    /// Ships DISARMED, matching upstream, whose `defaults()` and `from_env()`
    /// both construct `AtomicBool::new(false)`.
    ///
    /// An earlier draft of this module shipped it armed with a confident
    /// rationale about detector thresholds. That was invented: nothing in
    /// either tree supports booting it on, and the Ops toggle is what arms it.
    pub fn defaults() -> Self {
        Self {
            loop_watchdog: AtomicBool::new(false),
        }
    }

    /// Build them as a run would, honouring an operator override.
    ///
    /// `ATLAS_LOOP_WATCHDOG=1` boots with it armed. Off otherwise, as upstream
    /// does — the Ops pane's `/watchdog on` is the normal way to arm it, and the
    /// env read exists so a box that needs it from the first token does not have
    /// to be toggled by hand after every restart.
    pub fn from_env() -> Self {
        let armed = matches!(
            std::env::var("ATLAS_LOOP_WATCHDOG").as_deref(),
            Ok("1") | Ok("true")
        );
        Self {
            loop_watchdog: AtomicBool::new(armed),
        }
    }

    /// Is the loop watchdog armed?
    pub fn loop_watchdog(&self) -> bool {
        self.loop_watchdog.load(Ordering::Relaxed)
    }

    /// Arm or disarm it. `SeqCst` on the write and `Relaxed` on the read: the
    /// operator's toggle is rare and must be visible promptly, while the read
    /// sits in the decode loop.
    pub fn set_loop_watchdog(&self, on: bool) {
        self.loop_watchdog.store(on, Ordering::SeqCst);
    }
}

impl Default for SchedLevers {
    fn default() -> Self {
        Self::defaults()
    }
}
