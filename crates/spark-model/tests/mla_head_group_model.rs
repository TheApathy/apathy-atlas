// SPDX-License-Identifier: AGPL-3.0-only

//! CPU and source contracts for the isolated DeepSeek-V4 MLA heads8 experiment.
//!
//! The CUDA component is deliberately not registered with the serving build.

use half::bf16;

const HEADS: usize = 64;
const HEADS_PER_GROUP: usize = 8;
const DIM: usize = 512;
const KERNEL: &str =
    include_str!("../../../kernels/gb10/experiments/mla_paged_decode_fp8_heads8.cu");
const GATE: &str = include_str!("../../../scripts/check-mla-head-group-sass.sh");
const HOST: &str = include_str!("../src/layers/qwen3_attention/decode/run_paged_decode.rs");

fn scalar_online_softmax(q: &[f32], rows: &[Vec<f32>], sink: Option<f32>) -> Vec<f32> {
    let mut maximum = f32::NEG_INFINITY;
    let mut denominator = 0.0f32;
    let mut output = vec![0.0f32; DIM];
    for row in rows {
        let score = q
            .iter()
            .zip(row)
            .fold(0.0f32, |sum, (&qv, &kv)| sum + qv * kv);
        let next_maximum = maximum.max(score);
        let old_scale = (maximum - next_maximum).exp();
        let row_scale = (score - next_maximum).exp();
        denominator = denominator * old_scale + row_scale;
        for (accumulator, &value) in output.iter_mut().zip(row) {
            *accumulator = *accumulator * old_scale + value * row_scale;
        }
        maximum = next_maximum;
    }
    if let Some(sink) = sink {
        denominator += (sink - maximum).exp();
    }
    output
        .into_iter()
        .map(|value| bf16::from_f32(value / denominator).to_f32())
        .collect()
}

fn grouped_online_softmax(queries: &[Vec<f32>], rows: &[Vec<f32>], sinks: &[f32]) -> Vec<Vec<f32>> {
    let mut output = vec![Vec::new(); HEADS];
    for group in 0..HEADS / HEADS_PER_GROUP {
        for warp in 0..HEADS_PER_GROUP {
            let head = group * HEADS_PER_GROUP + warp;
            output[head] = scalar_online_softmax(&queries[head], rows, Some(sinks[head]));
        }
    }
    output
}

fn visible_row_ids(seq_len: usize, comp_count: usize, comp_ratio: usize) -> Vec<(bool, usize)> {
    let raw_start = seq_len.saturating_sub(128);
    let mut rows: Vec<_> = (raw_start..seq_len).map(|row| (false, row)).collect();
    let comp_visible = if comp_ratio == 0 {
        comp_count
    } else {
        comp_count.min(seq_len / comp_ratio)
    };
    rows.extend((0..comp_visible).map(|row| (true, row)));
    rows
}

fn tiled_online_softmax(scores: &[f32], values: &[f32]) -> f32 {
    let mut maximum = f32::NEG_INFINITY;
    let mut denominator = 0.0f32;
    let mut output = 0.0f32;
    for (score_tile, value_tile) in scores.chunks(4).zip(values.chunks(4)) {
        let tile_maximum = score_tile
            .iter()
            .copied()
            .fold(maximum, |current, score| current.max(score));
        let old_scale = (maximum - tile_maximum).exp();
        denominator *= old_scale;
        output *= old_scale;
        for (&score, &value) in score_tile.iter().zip(value_tile) {
            let factor = (score - tile_maximum).exp();
            denominator += factor;
            output += factor * value;
        }
        maximum = tile_maximum;
    }
    output / denominator
}

fn scalar_score_softmax(scores: &[f32], values: &[f32]) -> f32 {
    let mut maximum = f32::NEG_INFINITY;
    let mut denominator = 0.0f32;
    let mut output = 0.0f32;
    for (&score, &value) in scores.iter().zip(values) {
        let next_maximum = maximum.max(score);
        let old_scale = (maximum - next_maximum).exp();
        let factor = (score - next_maximum).exp();
        denominator = denominator * old_scale + factor;
        output = output * old_scale + factor * value;
        maximum = next_maximum;
    }
    output / denominator
}

fn buggy_deferred_factors(scores: &[f32], values: &[f32]) -> f32 {
    let mut maximum = f32::NEG_INFINITY;
    let mut denominator = 0.0f32;
    let mut output = 0.0f32;
    for (score_tile, value_tile) in scores.chunks(4).zip(values.chunks(4)) {
        let mut factors = Vec::new();
        for &score in score_tile {
            let next_maximum = maximum.max(score);
            let old_scale = (maximum - next_maximum).exp();
            let factor = (score - next_maximum).exp();
            denominator = denominator * old_scale + factor;
            output *= old_scale;
            maximum = next_maximum;
            factors.push(factor);
        }
        for (&factor, &value) in factors.iter().zip(value_tile) {
            output += factor * value;
        }
    }
    output / denominator
}

#[test]
fn eight_groups_own_all_heads_exactly_once() {
    let mut owners = [0u8; HEADS];
    for group in 0..HEADS / HEADS_PER_GROUP {
        for warp in 0..HEADS_PER_GROUP {
            owners[group * HEADS_PER_GROUP + warp] += 1;
        }
    }
    assert!(owners.into_iter().all(|count| count == 1));
}

#[test]
fn grouping_preserves_per_head_scalar_online_softmax_and_sink_semantics() {
    let rows: Vec<Vec<f32>> = (0..9)
        .map(|row| {
            (0..DIM)
                .map(|dim| ((row * 17 + dim * 13) % 101) as f32 / 128.0 - 0.4)
                .collect()
        })
        .collect();
    let queries: Vec<Vec<f32>> = (0..HEADS)
        .map(|head| {
            (0..DIM)
                .map(|dim| ((head * 7 + dim * 5) % 97) as f32 / 8192.0 - 0.004)
                .collect()
        })
        .collect();
    let sinks: Vec<f32> = (0..HEADS).map(|head| head as f32 / 64.0 - 0.5).collect();
    let grouped = grouped_online_softmax(&queries, &rows, &sinks);
    for head in 0..HEADS {
        let scalar = scalar_online_softmax(&queries[head], &rows, Some(sinks[head]));
        assert_eq!(scalar, grouped[head], "head={head}");
        assert_ne!(
            grouped[head],
            scalar_online_softmax(&queries[head], &rows, None)
        );
    }
}

#[test]
fn raw_then_compressed_visibility_matches_v4_causal_contract() {
    let rows = visible_row_ids(131, 40, 4);
    assert_eq!(&rows[..3], &[(false, 3), (false, 4), (false, 5)]);
    assert_eq!(rows[127], (false, 130));
    assert_eq!(rows[128], (true, 0));
    assert_eq!(rows.last(), Some(&(true, 31)));
    assert_eq!(rows.len(), 128 + 32);

    let ratio_zero = visible_row_ids(2, 3, 0);
    assert_eq!(
        ratio_zero,
        [(false, 0), (false, 1), (true, 0), (true, 1), (true, 2)]
    );
}

#[test]
fn tile_batch_max_rescales_all_factors_before_deferred_v_accumulation() {
    let scores = [-24.0f32, -8.0, 8.0, 24.0, 25.0, 26.0];
    let values = [1000.0f32, -100.0, 10.0, 1.0, -3.0, 7.0];
    let scalar = bf16::from_f32(scalar_score_softmax(&scores, &values)).to_f32();
    let tiled = bf16::from_f32(tiled_online_softmax(&scores, &values)).to_f32();
    let buggy = bf16::from_f32(buggy_deferred_factors(&scores, &values)).to_f32();
    assert_eq!(tiled, scalar);
    assert_ne!(buggy, scalar, "fixture must catch stale deferred factors");

    assert_eq!(KERNEL.matches("float tile_maximum = maximum;").count(), 2);
    assert_eq!(KERNEL.matches("maximum = tile_maximum;").count(), 2);
    assert_eq!(KERNEL.matches("scores[row] - tile_maximum").count(), 2);
    assert_eq!(KERNEL.matches("output[i] *= old_scale;").count(), 2);
}

#[test]
fn source_is_exact_shape_fail_closed_and_keeps_alias_entry() {
    for contract in [
        "MLA_HG_HEADS 8",
        "MLA_HG_ROWS 4",
        "num_q_heads != 64",
        "num_kv_heads != 1",
        "q_head_dim != 512",
        "kv_cache_dim != 576",
        "block_size != 16",
        "sliding_window != 128",
        "gridDim.x != 8",
        "blockDim.x != 256",
        "seq_len > (unsigned long long)max_blocks_per_seq * MLA_HG_BLOCK_SIZE",
        "const unsigned int q_head = blockIdx.x * MLA_HG_HEADS + warp_id",
        "if constexpr (KV_ALIAS)",
        "mla_paged_decode_fp8_heads8",
        "mla_paged_decode_fp8_heads8_kvalias",
    ] {
        assert!(
            KERNEL.contains(contract),
            "missing source contract: {contract}"
        );
    }
    assert!(KERNEL.contains("comp_visible = seq_len / comp_ratio"));
    assert!(KERNEL.contains("sinks[q_head]"));
    assert!(KERNEL.contains("COMP_BLOCK_DIM 512"));
    assert_eq!(
        KERNEL
            .matches("const unsigned int source_dim = dim < 448 ? dim : 512 + dim - 448;")
            .count(),
        2,
        "raw K and V must remap rope[512:576] onto Q/output[448:512]"
    );
    assert!(KERNEL.contains("(unsigned long long)(tile + row) * COMP_BLOCK_DIM"));
    assert!(KERNEL.contains("__float_as_uint(k_scale) != __float_as_uint(v_scale)"));
    assert!(KERNEL.contains("__shared__ float kv_tile[MLA_HG_ROWS][MLA_HG_DIM]"));
}

#[test]
fn experiment_is_not_reachable_from_serving_and_gate_targets_sm121a() {
    assert!(!HOST.contains("mla_paged_decode_fp8_heads8"));
    assert!(GATE.contains("-arch=sm_121a"));
    assert!(GATE.contains("mla_paged_decode_fp8_heads8_kvalias"));
    assert!(GATE.contains("REG"));
    assert!(GATE.contains("STACK"));
    assert!(GATE.contains("LOCAL"));
    assert!(GATE.contains("SHARED"));
    assert!(GATE.contains("spill stores"));
    assert!(GATE.contains("LDL|STL"));
}
