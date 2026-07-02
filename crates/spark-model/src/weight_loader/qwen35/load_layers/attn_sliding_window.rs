// SPDX-License-Identifier: AGPL-3.0-only
//
// Opt-in sliding-window override for Qwen3.5/3.6 full-attention layers.
//
// AEON-Q36-27B's full_attention layers are GLOBAL (no trained
// `sliding_window` in the config) — every full-attn layer reads the entire
// KV cache each step. At long context that KV read grows linearly and adds
// decode latency. Capping it to the last `W` tokens (a sliding window) cuts
// the per-step KV read to a constant, trading a small, model-dependent PPL
// drift (the model was NOT trained for a window) for faster long-context
// decode. This is the "lever 5" tradeoff knob.
//
// Default OFF (`None`) → kernels see window=0 → full global attention →
// byte-identical to the un-gated build. Enable with
// `ATLAS_ATTN_SLIDING_WINDOW=<tokens>` (e.g. 1024). A value of 0 or an
// unparseable value is treated as OFF.

use std::sync::OnceLock;

static WINDOW: OnceLock<Option<u32>> = OnceLock::new();

/// Per-layer sliding-window override for full-attention layers, read once
/// from `ATLAS_ATTN_SLIDING_WINDOW`. `None` = global attention (default).
pub fn sliding_window_override() -> Option<u32> {
    *WINDOW.get_or_init(|| {
        let w = std::env::var("ATLAS_ATTN_SLIDING_WINDOW")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .filter(|&w| w > 0);
        if let Some(w) = w {
            tracing::warn!(
                "ATLAS_ATTN_SLIDING_WINDOW={w}: full-attention layers capped to last {w} \
                 tokens (LOSSY — model not trained for a window; verify PPL drift)"
            );
        }
        w
    })
}
