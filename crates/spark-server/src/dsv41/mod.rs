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

pub mod encoding;
pub mod grammar;
pub mod parse;
pub mod pyjson;
pub mod repetition;
pub mod request;
pub mod vision;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_grammar;
