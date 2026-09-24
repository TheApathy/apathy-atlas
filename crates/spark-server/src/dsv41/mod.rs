// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1 (`model_type = "deepseek_v41"`) request/response layer, ported
//! from the production Python server so Atlas serves the same prompts and parses
//! the same tool calls.
//!
//! - [`request`]: `resolve_thinking` + `build_chat_prompt` (server/app.py)
//! - [`encoding`]: the checkpoint's `encoding/encoding.py` renderer
//! - [`parse`]: output routing, strict + tolerant DSML tool-call parsing
//! - [`grammar`]: the DSML tool-call EBNF (server/tool_grammar.py)
//!
//! Fixtures in `tests/fixtures/dsv41/` are generated from the Python code by
//! `scripts/dsv41_parity/gen_fixtures.py`; `tests` compares against them.

pub(crate) mod bicubic;
pub mod encoding;
pub mod grammar;
pub mod parse;
pub(crate) use crate::pyjson;
pub mod repetition;
pub mod request;
pub mod turbojpeg;
pub mod vision;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_grammar;

use std::sync::atomic::{AtomicBool, Ordering};

// Set on every model load (a swap away turns it off again).
static SERVING: AtomicBool = AtomicBool::new(false);

/// Record whether the loaded model is deepseek_v41. The scheduler reads
/// [`serving`] to switch off the generic generation heuristics the Python
/// engine does not have (reflection suppression, think-loop watchdog, EOS
/// suppression, forced `</think>`, catastrophic-loop stop, ...).
pub fn set_serving(on: bool) {
    SERVING.store(on, Ordering::Relaxed);
}

/// Whether the loaded model is deepseek_v41 (see [`set_serving`]).
pub fn serving() -> bool {
    SERVING.load(Ordering::Relaxed)
}
