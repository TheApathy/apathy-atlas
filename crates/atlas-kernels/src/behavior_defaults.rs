// SPDX-License-Identifier: AGPL-3.0-only
//
// `[behavior]` defaults shared by the library and its build script. The
// build script cannot import the library it is building, so this file is both
// a normal module and an `include!` at the MODEL.toml parser boundary.

/// Default cap on free-text tokens between successive tool calls.
///
/// The previous build-time default (384) truncated legitimate agent plans;
/// repeating wander is already covered by the independent loop watchdogs.
pub const DEFAULT_MAX_INTER_TOOL_PROSE: u32 = 3072;
