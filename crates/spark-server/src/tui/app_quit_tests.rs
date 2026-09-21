// SPDX-License-Identifier: AGPL-3.0-only

//! The quit guard's knowledge of in-flight model loads.
//!
//! In its own mount (not `app_keys_tests.rs`, which is at the per-file cap)
//! because these cases are about what [`App::work_in_flight`] names, not
//! about key routing.

use super::*;
use crossterm::event::{KeyCode, KeyEvent};

fn app() -> App {
    App::new(clap::Parser::parse_from(["spark", "org/m"]))
}

fn press(a: &mut App, c: char) {
    a.on_key(KeyEvent::from(KeyCode::Char(c)));
}

/// A load is minutes of shard reading the user cannot resume, and `q` used to
/// tear it down with no confirmation — the one long-running job the guard
/// did not know about.
#[test]
fn q_asks_first_while_the_boot_load_is_still_running() {
    let mut a = app(); // argv names a model; `ready` has not flipped yet
    assert_eq!(a.work_in_flight(), Some("a model is still loading"));
    press(&mut a, 'q');
    assert!(a.confirm_quit, "the first press asks");
    assert!(!a.should_quit);

    // Once serving, the same press quits clean.
    let mut a = app();
    a.progress.ready = true;
    press(&mut a, 'q');
    assert!(a.should_quit);
    assert!(!a.confirm_quit);
}

// REMOVED: describes a state this engine cannot enter.
//
// Upstream allows `spark serve` with NO model — the dashboard is the front
// door and you pick a recipe from the Library tab. Our `ServeArgs` marks the
// MODEL positional `required_unless_present = "model_from_path"`, so one of
// the two must always be given and there is no awaiting-model state. clap
// EXITS THE PROCESS on the missing argument, which aborted the whole test
// binary rather than failing one test.
//
// Restore this WITH the model-less serve path, which is the same commit that
// wires `LibState::launch` — until a recipe can be started from the
// dashboard, a server with no model has nothing it could ever load.
// (was: an_awaiting_model_boot_quits_without_the_loading_prompt)

