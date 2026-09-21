// SPDX-License-Identifier: AGPL-3.0-only

//! Prometheus metrics for Atlas Spark.

use lazy_static::lazy_static;
use prometheus::{
    Histogram, IntCounter, IntCounterVec, IntGauge, register_histogram, register_int_counter,
    register_int_counter_vec, register_int_gauge,
};

lazy_static! {
    pub static ref REQUESTS_TOTAL: IntCounter =
        register_int_counter!("atlas_requests_total", "Total requests processed").unwrap();
    pub static ref REQUESTS_ACTIVE: IntGauge =
        register_int_gauge!("atlas_requests_active", "Currently active requests").unwrap();
    pub static ref TTFT_SECONDS: Histogram = register_histogram!(
        "atlas_time_to_first_token_seconds",
        "Time to first token",
        vec![0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
    )
    .unwrap();
    /// Tokens counted AS THEY ARE PRODUCED, in the streaming handler.
    ///
    /// Distinct from `GENERATION_TOKENS_TOTAL`, which only moves when a
    /// request finishes: sampling that alone reads as 0 tok/s for the whole
    /// generation and then one spike, which is not a rate. The dashboard takes
    /// the larger of the two deltas, so blocking requests — which never stream
    /// and legitimately arrive as a lump — still register.
    pub static ref DECODED_TOKENS_TOTAL: IntCounter =
        register_int_counter!("atlas_decoded_tokens_total", "Tokens counted as decoded, per token").unwrap();
    pub static ref GENERATION_TOKENS_TOTAL: IntCounter =
        register_int_counter!("atlas_generation_tokens_total", "Total tokens generated").unwrap();
    // ── HTTP byte accounting (Atlas TUI Server Stats) ──
    //
    // Request side counts body bytes as received by the byte-count
    // middleware; response side counts bytes actually written through the
    // wrapped body (streaming/SSE included, where Content-Length lies).
    pub static ref HTTP_BYTES_IN: IntCounter =
        register_int_counter!("atlas_http_bytes_in_total", "Total HTTP request body bytes")
            .unwrap();
    pub static ref HTTP_BYTES_OUT: IntCounter =
        register_int_counter!("atlas_http_bytes_out_total", "Total HTTP response body bytes")
            .unwrap();
    pub static ref PROMPT_TOKENS_TOTAL: IntCounter =
        register_int_counter!("atlas_prompt_tokens_total", "Total prompt tokens processed")
            .unwrap();

    // ── Loop-detector telemetry (P5.2, 2026-04-25) ──
    //
    // Track the verdict distribution emitted by `loop_detector::detect`
    // so we can tune thresholds against production traffic instead of
    // single dump fixtures. Labels:
    //   - verdict ∈ {none, hint, suppress}
    //   - channel ∈ {text, tools, combined, n/a (None verdict)}
    //   - spinning ∈ {0, 1} — was Layer-2 spinning detection also active
    pub static ref LOOP_DETECTOR_VERDICTS: IntCounterVec =
        register_int_counter_vec!(
            "atlas_loop_detector_verdicts_total",
            "Loop detector verdicts emitted, by verdict + channel + spinning flag",
            &["verdict", "channel", "spinning"]
        ).unwrap();

    // ── Server-side intervention telemetry (P5.2) ──
    //
    // Track how often the goal-pin reminder + observation-masking
    // fire, so we can correlate intervention frequency with outcome
    // metrics (TTFT, completion length, finish_reason).
    pub static ref TASK_PIN_INJECTIONS: IntCounter =
        register_int_counter!(
            "atlas_task_pin_injections_total",
            "Times the verbatim-goal reminder was injected into a request"
        ).unwrap();
    pub static ref OBSERVATION_MASK_ELIDED_BODIES: IntCounter =
        register_int_counter!(
            "atlas_observation_mask_elided_bodies_total",
            "Stale tool-failure bodies replaced with one-line summaries"
        ).unwrap();

    // ── Anthropic translation-drift counter (P5.1) ──
    //
    // Increments whenever the Anthropic→OpenAI translator produces a
    // round-trip diff against the original Anthropic shape. Diffs
    // indicate translation bugs that compound across long agentic
    // sessions. Logging the actual diff is gated behind the
    // ATLAS_DEBUG_TRANSLATION_DRIFT env var (anthropic.rs).
    pub static ref ANTHROPIC_TRANSLATION_DRIFTS: IntCounter =
        register_int_counter!(
            "atlas_anthropic_translation_drifts_total",
            "Anthropic ↔ OpenAI translator round-trip mismatches detected"
        ).unwrap();

    // ── Speculative-decode telemetry (A.2 EASD scaffolding) ──
    //
    // Per-K acceptance counters. Enables measuring baseline accept
    // rates across MTP K-paths so we can decide whether EASD
    // activation (per-step D2H of verify logits + entropy gating,
    // arXiv:2512.23765) is worth its cost. EASD itself is gated
    // behind future activation once these baselines are measured.
    pub static ref SPEC_DECODE_VERIFY: IntCounterVec =
        register_int_counter_vec!(
            "atlas_spec_decode_verify_total",
            "MTP draft verify outcomes by K and result",
            &["k", "outcome"]
        ).unwrap();

    // ── Tool-call telemetry ──
    //
    // Total successful tool calls emitted by the API layer (sum across
    // streaming + blocking). Paired with the "Tool call: name(args)"
    // info log so operators can both grep logs and graph rates.
    // Unlabeled (no `name` label) — high-cardinality tool names would
    // blow up Prometheus cardinality.
    pub static ref TOOL_CALLS_TOTAL: IntCounter =
        register_int_counter!(
            "atlas_tool_calls_total",
            "Total successful tool calls emitted by the server"
        ).unwrap();
}

/// Holds one count on [`REQUESTS_ACTIVE`] for as long as it is alive.
///
/// THE BUG THIS FIXES: `chat::completions` incremented the gauge and then had
/// TEN early returns — input validation, image admission, tool-prompt
/// prepending — none of which decremented. Every rejected request leaked a
/// count permanently, so the gauge only ever grew on bad input, and
/// `shutdown::drain_in_flight` waits on exactly that gauge: a server that had
/// seen malformed requests could never finish draining.
///
/// One `inc()` against six `dec()`s spread over four modules is not a thing
/// anyone can keep right by reading it, which is why this is a guard.
pub struct ActiveRequestGuard {
    released: bool,
}

impl ActiveRequestGuard {
    pub fn new() -> Self {
        REQUESTS_ACTIVE.inc();
        Self { released: false }
    }

    /// Hand the count to a path that decrements it ITSELF.
    ///
    /// The streaming and blocking dispatchers outlive this scope and already
    /// own their own `dec()` — a guard that also fired would double-decrement,
    /// and a gauge driven negative makes the drain wait forever on a count
    /// that can never reach zero. Consuming `self` means the transfer is a
    /// statement at the call site, not a comment someone has to notice.
    pub fn release(mut self) {
        self.released = true;
    }
}

impl Drop for ActiveRequestGuard {
    fn drop(&mut self) {
        if !self.released {
            REQUESTS_ACTIVE.dec();
        }
    }
}
