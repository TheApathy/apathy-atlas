// SPDX-License-Identifier: AGPL-3.0-only

//! The Python engine's repetition controls that Atlas has no equivalent for
//! (`Penalties` in engine/v41_engine.py). Presence and frequency penalties
//! already exist on the shared sampler.
//!
//! - **Cycle breaker, ON in production** (`DSV41_CYCLE_BREAK`, default 1).
//!   When the generated history ends in an exact block of period p <= 16,
//!   repeated 4 times, the token that would continue the cycle is banned for
//!   one step. Greedy decoding on this checkpoint can lock into such an
//!   attractor with no way out.
//! - **No-repeat-ngram** (`DSV41_NO_REPEAT_NGRAM`, default 0, so OFF in
//!   production). Uses transformers' NoRepeatNGramLogitsProcessor semantics.
//!
//! Both see only generated tokens, not the prompt, as in Python.

use std::sync::RwLock;

pub const CYCLE_REPEATS: usize = 4;
pub const CYCLE_MAX_PERIOD: usize = 16;

/// `Penalties._cycle_token`: the token that would continue an exact cycle.
pub fn cycle_token(history: &[u32]) -> Option<u32> {
    let len = history.len();
    if len < CYCLE_REPEATS.max(4) {
        return None;
    }
    for p in 1..=CYCLE_MAX_PERIOD.min(len / CYCLE_REPEATS) {
        let block = &history[len - p..];
        if (1..CYCLE_REPEATS).all(|i| &history[len - p * (i + 1)..len - p * i] == block) {
            return Some(block[0]);
        }
    }
    None
}

/// `Penalties._banned_ngram_tokens`: every token that followed an earlier
/// occurrence of the last n-1 tokens. Sorted, deduplicated.
pub fn banned_ngram_tokens(history: &[u32], n: usize) -> Vec<u32> {
    let len = history.len();
    if n == 0 || len < n {
        return Vec::new();
    }
    // Python `history[-(n - 1):]`; for n == 1 that slice is the whole list.
    let prefix = if n == 1 {
        history
    } else {
        &history[len - (n - 1)..]
    };
    let mut banned: Vec<u32> = (0..=len - n)
        .filter(|&i| &history[i..i + n - 1] == prefix)
        .map(|i| history[i + n - 1])
        .collect();
    banned.sort_unstable();
    banned.dedup();
    banned
}

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub cycle_break: bool,
    pub no_repeat_ngram: usize,
}

const OFF: Config = Config {
    cycle_break: false,
    no_repeat_ngram: 0,
};

// Not a OnceLock: a model swap re-runs `configure`, and a server that swaps
// from deepseek_v41 to another model must stop banning its tokens.
static CONFIG: RwLock<Config> = RwLock::new(OFF);

/// Called on every model load with whether the served model is deepseek_v41.
/// Reads the same environment variables as the Python engine.
pub fn configure(model_is_dsv41: bool) {
    let cfg = Config {
        cycle_break: model_is_dsv41
            && std::env::var("DSV41_CYCLE_BREAK").as_deref().unwrap_or("1") == "1",
        no_repeat_ngram: if model_is_dsv41 {
            std::env::var("DSV41_NO_REPEAT_NGRAM")
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(0)
        } else {
            0
        },
    };
    *CONFIG.write().unwrap_or_else(|e| e.into_inner()) = cfg;
    if model_is_dsv41 {
        tracing::info!(
            "deepseek_v41 repetition controls: cycle_break={} no_repeat_ngram={}",
            cfg.cycle_break,
            cfg.no_repeat_ngram
        );
    }
}

/// Whether [`apply`] can ban anything. The scheduler's GPU-argmax fast path
/// never sees host logits, so while this is true it must take the host path.
pub fn active() -> bool {
    let cfg = *CONFIG.read().unwrap_or_else(|e| e.into_inner());
    cfg.cycle_break || cfg.no_repeat_ngram > 0
}

/// Ban, in place, what the Python engine would ban at this step. Returns
/// true when anything was banned. A no-op unless [`configure`] enabled it.
pub fn apply(logits: &mut [f32], history: &[u32]) -> bool {
    let cfg = *CONFIG.read().unwrap_or_else(|e| e.into_inner());
    if !cfg.cycle_break && cfg.no_repeat_ngram == 0 {
        return false;
    }
    let mut hit = false;
    let mut ban = |t: u32| {
        if let Some(l) = logits.get_mut(t as usize) {
            *l = f32::NEG_INFINITY;
            hit = true;
        }
    };
    for t in banned_ngram_tokens(history, cfg.no_repeat_ngram) {
        ban(t);
    }
    if cfg.cycle_break
        && let Some(t) = cycle_token(history)
    {
        ban(t);
    }
    hit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_python_penalties() {
        let path = format!(
            "{}/tests/fixtures/dsv41/repetition.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let fx: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let cases = fx["cases"].as_array().unwrap();
        let (mut cycles, mut bans) = (0, 0);
        for c in cases {
            let h: Vec<u32> = c["history"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect();
            let want = c["cycle"].as_u64().map(|x| x as u32);
            assert_eq!(cycle_token(&h), want, "cycle {h:?}");
            cycles += usize::from(want.is_some());
            for n in [0usize, 2, 3, 4] {
                let want: Vec<u32> = c[format!("ngram{n}")]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as u32)
                    .collect();
                assert_eq!(banned_ngram_tokens(&h, n), want, "ngram{n} {h:?}");
                bans += usize::from(!want.is_empty());
            }
        }
        // Both answers occur: the comparison can fail either way.
        assert!(
            cycles > 10 && cycles < cases.len() && bans > 10,
            "{cycles} cycles, {bans} bans"
        );
    }

    /// NEGATIVE CONTROL: three repeats instead of four must disagree with Python.
    #[test]
    fn a_three_repeat_breaker_is_caught() {
        let three = |h: &[u32]| -> Option<u32> {
            let len = h.len();
            (1..=16.min(len / 3)).find_map(|p| {
                let b = &h[len - p..];
                (1..3)
                    .all(|i| &h[len - p * (i + 1)..len - p * i] == b)
                    .then(|| b[0])
            })
        };
        let h = [9, 1, 2, 1, 2, 1, 2];
        assert_ne!(three(&h), cycle_token(&h));
    }
}
