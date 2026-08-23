// SPDX-License-Identifier: AGPL-3.0-only
//
// Resolve `(enable_thinking, thinking_budget)` for a single
// request. Precedence (highest wins):
//   1. `--disable-thinking` CLI flag (forces OFF for every request)
//   2. Request body (`reasoning_effort`, `thinking.budget_tokens`, …)
//   3. MODEL.toml `[behavior].thinking_default`
//
// Lifted out of `chat::chat_completions_inner` (wave 4g).

use std::sync::Arc;

use crate::AppState;
use crate::openai::ChatCompletionRequest;

use super::super::failures::recent_message_is_tool_error;

pub(super) fn resolve_thinking(
    state: &Arc<AppState>,
    req: &ChatCompletionRequest,
    tools_active: bool,
) -> (bool, Option<u32>) {
    if state.disable_thinking {
        return (false, None);
    }
    let (et, tb) = req.resolve_thinking(state.behavior.thinking_default);
    let mt = u32::try_from(generation_max_tokens(
        req.max_tokens,
        tools_active,
        state.tool_max_tokens,
    ))
    .unwrap_or(u32::MAX);
    let max_budget = state.behavior.max_thinking_budget;
    // `thinking_in_tools=false` is the MODEL.toml DEFAULT for tool-
    // active turns: it suppresses thinking when the client is silent.
    let et = if tools_active
        && !state.behavior.thinking_in_tools
        && !req.thinking_explicitly_requested()
    {
        false
    } else {
        et
    };
    // F28: auto-disable thinking on turns following a tool error.
    let et = if et && recent_message_is_tool_error(&req.messages) {
        tracing::info!("F28: disabling thinking on this turn (most recent message is tool error)");
        false
    } else {
        et
    };
    let budget = if et {
        Some(cap_thinking_budget(
            tb,
            max_budget,
            mt,
            tools_active,
            state.behavior.thinking_in_tools,
        ))
    } else {
        None
    };
    (et, budget)
}

fn cap_thinking_budget(
    requested: Option<u32>,
    model_max: u32,
    generation_max: u32,
    tools_active: bool,
    thinking_in_tools: bool,
) -> u32 {
    let safety_cap_pct = if tools_active && thinking_in_tools {
        7
    } else {
        9
    };
    let safety_max =
        ((generation_max as u64 * safety_cap_pct as u64) / 10).clamp(1, u32::MAX as u64) as u32;
    requested.unwrap_or(model_max).min(safety_max)
}

/// Generation allowance after the tools-active server cap. Thinking-budget
/// resolution and sampling must consume this same ceiling: resolving from the
/// raw client request first can grant reasoning more tokens than the whole
/// tool turn is allowed to emit.
pub(super) fn generation_max_tokens(
    max_tokens: usize,
    tools_active: bool,
    tool_max_tokens: usize,
) -> usize {
    if tools_active {
        max_tokens.min(tool_max_tokens)
    } else {
        max_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::{cap_thinking_budget, generation_max_tokens};

    #[test]
    fn tool_cap_is_the_thinking_and_generation_ceiling() {
        let ceiling = generation_max_tokens(2048, true, 256);
        assert_eq!(ceiling, 256);
        // The normal 90% thinking safety cap now leaves room in the actual
        // tool-turn allowance: 230, not 90% of the raw 2048 (=1843).
        assert_eq!(
            cap_thinking_budget(None, 2048, ceiling as u32, true, false),
            230
        );
        assert_eq!(generation_max_tokens(128, true, 256), 128);
        assert_eq!(generation_max_tokens(2048, false, 256), 2048);
    }
}
