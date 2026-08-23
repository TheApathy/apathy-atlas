// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the pure-data eligibility predicate in
//! `batch_kernel.rs`. Kept in a sibling file to keep `batch_kernel.rs`
//! itself under the 500-LoC file-size-cap.

use super::batch_kernel::check_kernel_batched_eligible;

/// (chunk_len, chunk_start, is_last_chunk, cached_prefix_tokens, marconi_skip_to)
fn s(chunk_len: usize, chunk_start: usize, is_last: bool) -> (usize, usize, bool, usize, usize) {
    (chunk_len, chunk_start, is_last, 0, 0)
}

fn replay(
    chunk_len: usize,
    chunk_start: usize,
    cached: usize,
    skip: usize,
) -> (usize, usize, bool, usize, usize) {
    (chunk_len, chunk_start, false, cached, skip)
}

#[test]
fn rejects_under_two_streams() {
    assert!(!check_kernel_batched_eligible(
        std::iter::empty(),
        0,
        8192,
        "qwen3_next",
        256
    ));
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 0, false)],
        1,
        8192,
        "qwen3_next",
        256
    ));
}

#[test]
fn rejects_first_chunk_before_effects() {
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 0, false), s(4096, 0, false)],
        2,
        8192,
        "qwen3_next",
        256,
    ));
}

#[test]
fn accepts_uniform_n_2_after_cold_entry() {
    assert!(check_kernel_batched_eligible(
        vec![s(4096, 4096, false), s(4096, 4096, false)],
        2,
        8192,
        "qwen3_next",
        256,
    ));
}

#[test]
fn rejects_partial_paired_hit_before_effects() {
    assert!(!check_kernel_batched_eligible(
        vec![replay(64, 64, 32, 16), replay(64, 64, 32, 16)],
        2,
        8192,
        "qwen3_next",
        256,
    ));
}

#[test]
fn rejects_prefix_replay_on_later_stream_before_offset_setup() {
    assert!(!check_kernel_batched_eligible(
        vec![s(64, 64, false), replay(64, 64, 32, 16)],
        2,
        8192,
        "qwen3_next",
        256,
    ));
}

#[test]
fn rejects_cached_exact_bypass_and_inherited_skip_independently() {
    for streams in [
        vec![replay(64, 64, 32, 0), replay(64, 64, 32, 0)],
        vec![replay(64, 64, 0, 16), replay(64, 64, 0, 16)],
    ] {
        assert!(!check_kernel_batched_eligible(
            streams,
            2,
            8192,
            "qwen3_next",
            256,
        ));
    }
}

#[test]
fn post_effect_mismatch_and_layer_errors_propagate_without_sequential_retry() {
    let batch: String = include_str!("batch.rs")
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect();
    let direct = "returnself.prefill_batch_chunk_kernel_batched(streams,stream);";
    assert!(batch.contains(direct));
    assert_eq!(
        batch
            .matches("self.prefill_batch_chunk_kernel_batched(")
            .count(),
        1
    );
    assert!(
        batch.find(direct).unwrap() < batch.find("letmutkv_cache=self.kv_cache.lock();").unwrap()
    );
    assert!(
        !batch
            .replace(
                direct,
                "self.prefill_batch_chunk_kernel_batched(streams,stream);"
            )
            .contains(direct)
    );
}

#[test]
fn rejects_mismatched_chunk_len() {
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 0, false), s(2048, 0, false)],
        2,
        16384,
        "qwen3_next",
        256,
    ));
}

#[test]
fn rejects_mismatched_chunk_start() {
    // Scheduler stream-desync case observed 2026-05-11:
    // stream 0 at chunk_start=12288, stream 1 at chunk_start=4096.
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 12288, false), s(4096, 4096, false)],
        2,
        16384,
        "qwen3_next",
        256,
    ));
}

#[test]
fn rejects_mismatched_is_last() {
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 0, false), s(4096, 0, true)],
        2,
        8192,
        "qwen3_next",
        256,
    ));
}

#[test]
fn rejects_arena_overflow() {
    // N=2 × 4096 = 8192 > 4100 arena → reject.
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 0, false), s(4096, 0, false)],
        2,
        4100,
        "qwen3_next",
        256,
    ));
}

#[test]
fn rejects_mla_model() {
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 0, false), s(4096, 0, false)],
        2,
        8192,
        "mistral",
        128,
    ));
}

#[test]
fn rejects_large_head_dim() {
    // Gemma-4 long-attention head_dim=512 → reject.
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 0, false), s(4096, 0, false)],
        2,
        8192,
        "gemma4",
        512,
    ));
}

#[test]
fn accepts_n_4_uniform() {
    assert!(check_kernel_batched_eligible(
        vec![s(2048, 2048, false); 4],
        4,
        8192,
        "qwen3_next",
        256,
    ));
}
