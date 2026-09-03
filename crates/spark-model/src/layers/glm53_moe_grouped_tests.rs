// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

const HIDDEN: u32 = 4_096;
const EXPERT_INTERMEDIATE: u32 = 1_024;
const EXPERTS: u32 = 288;
const TOP_K: u32 = 8;
const BASE: DevicePtr = DevicePtr(0x1000_0000_0000);

fn buf(bytes: usize) -> GgmlIqBuffer {
    GgmlIqBuffer { ptr: BASE, bytes }
}

fn bank(kind: GgmlType, inner: u32, columns: u32) -> Glm53GroupedBank {
    let per = GgmlIqMmqPlan::new(kind, 1, columns, inner).unwrap().weight_bytes;
    Glm53GroupedBank::new(buf(per * EXPERTS as usize), kind, inner, columns, EXPERTS, per).unwrap()
}

/// The uniform-stride guard, PROVEN TO FIRE rather than merely present.
///
/// The grouped kernel's entire premise is `base + id*expert_bytes`. Every way
/// a bank can stop satisfying that is a way to read a neighbouring expert's
/// weights and emit fluent, wrong output, which is the failure mode this
/// project has been chasing all session. So each is a separate case here.
#[test]
fn the_uniform_stride_guard_rejects_every_non_uniform_bank() {
    let kind = GgmlType::Q4_K;
    let per = GgmlIqMmqPlan::new(kind, 1, EXPERT_INTERMEDIATE, HIDDEN)
        .unwrap()
        .weight_bytes;
    let total = per * EXPERTS as usize;

    // The honest bank is accepted.
    let good =
        Glm53GroupedBank::new(buf(total), kind, HIDDEN, EXPERT_INTERMEDIATE, EXPERTS, per).unwrap();
    assert_eq!(good.expert_bytes, per);
    assert_eq!(good.total_bytes, total);

    // A stride that is not the type's own plan -- a per-expert alignment pad
    // is the realistic way this appears, and it is silently wrong, not a
    // crash: expert 1 onward would read shifted weights.
    let padded = Glm53GroupedBank::new(
        buf((per + 256) * EXPERTS as usize),
        kind,
        HIDDEN,
        EXPERT_INTERMEDIATE,
        EXPERTS,
        per + 256,
    )
    .unwrap_err()
    .to_string();
    assert!(padded.contains("stride is not uniform"), "{padded}");

    // A correct stride over a bank that is not densely packed: a header, a
    // trailing index, a bank sized for a different expert count.
    for total_bytes in [total + 1, total - per, total + per] {
        let error = Glm53GroupedBank::new(
            buf(total_bytes),
            kind,
            HIDDEN,
            EXPERT_INTERMEDIATE,
            EXPERTS,
            per,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("not densely packed"), "{total_bytes}: {error}");
    }

    // Degenerate geometry and a null base fail closed.
    assert!(
        Glm53GroupedBank::new(
            GgmlIqBuffer { ptr: DevicePtr::NULL, bytes: total },
            kind,
            HIDDEN,
            EXPERT_INTERMEDIATE,
            EXPERTS,
            per,
        )
        .is_err()
    );
    assert!(Glm53GroupedBank::new(buf(0), kind, HIDDEN, EXPERT_INTERMEDIATE, 0, per).is_err());
}

/// The pair grid is (rows, top_k) and NOT slot-only.
///
/// Grouping over slots alone would fix decode and leave prefill and
/// speculative verify exactly where they are, which is the reason this exists.
#[test]
fn the_plan_groups_over_rows_and_slots_and_bounds_both() {
    let bank = bank(GgmlType::Q4_K, HIDDEN, EXPERT_INTERMEDIATE);

    let decode = Glm53GroupedMoePlan::new(bank, 1, TOP_K).unwrap();
    assert_eq!(decode.pairs, 8, "decode is one row by top_k");
    assert_eq!(decode.route_bytes, 8 * 4);
    assert_eq!(decode.output_bytes, decode.tile.output_bytes * TOP_K as usize);

    let verify = Glm53GroupedMoePlan::new(bank, 16, TOP_K).unwrap();
    assert_eq!(verify.pairs, 128, "16 rows by top_k, the speculative shape");
    assert_eq!(verify.route_bytes, 128 * 4);
    // Rows multiply PAIRS, never the tile: the tile stays one row and the
    // output grows because there are more pairs.
    assert_eq!(verify.tile.rows, 1);
    assert_eq!(verify.tile.activation_bytes, decode.tile.activation_bytes);

    assert!(Glm53GroupedMoePlan::new(bank, 0, TOP_K).is_err());
    assert!(Glm53GroupedMoePlan::new(bank, 1, 0).is_err());
    // A top_k wider than the bank cannot be routed.
    assert!(Glm53GroupedMoePlan::new(bank, 1, EXPERTS + 1).is_err());
}

/// The gate DEFAULTS to the reference path and refuses a typo.
///
/// A misspelled value that quietly selected the slow correct path would be
/// scored as a grouped-path measurement, which is the one outcome that must
/// not be possible.
#[test]
fn the_moe_gate_defaults_to_the_reference_and_rejects_anything_else() {
    assert_eq!(
        Glm53MoePath::parse("serial-reference").unwrap(),
        Glm53MoePath::SerialReference
    );
    assert_eq!(Glm53MoePath::parse("grouped").unwrap(), Glm53MoePath::Grouped);
    for typo in ["", "Grouped", "group", "serial", "serial_reference", "1", "on"] {
        let error = Glm53MoePath::parse(typo).unwrap_err().to_string();
        assert!(error.contains(Glm53MoePath::ENV), "{typo}: {error}");
    }
}

/// Only the types the grouped kernel is DEFINED for resolve.
///
/// The eleven types here are the `ATLAS_DEFINE_GROUPED_MOE_MMQ` invocations,
/// cross-checked against the 22 `_moe_grouped_[nw]c` entries in the compiled
/// t3 PTX. The serial path additionally handles Q3_K; asking the grouped path
/// for it must name the reference path rather than resolve a kernel symbol
/// that was never emitted.
#[test]
fn grouped_type_coverage_is_a_subset_of_the_serial_path_and_says_so() {
    for kind in [
        GgmlType::Q2_K,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
        GgmlType::Q8_0,
        GgmlType::IQ2_XXS,
        GgmlType::IQ2_XS,
        GgmlType::IQ2_S,
        GgmlType::IQ3_XXS,
        GgmlType::IQ3_S,
        GgmlType::IQ4_XS,
    ] {
        assert!(grouped_tag(kind).is_ok(), "{kind:?}");
        assert!(quantize_name(kind).is_ok(), "{kind:?}");
    }
    // Q3_K is the ONE type the serial path has and the grouped path does not.
    for kind in [GgmlType::Q3_K] {
        let error = grouped_tag(kind).unwrap_err().to_string();
        assert!(error.contains("serial-reference"), "{kind:?}: {error}");
    }
}

/// The DOWN projection must be expressible, or the grouped path buys NOTHING.
///
/// Gate and up share one activation: all top_k experts consume the identical
/// token hidden state. Down does not — each slot consumes its own swiglu
/// output. If the grouped kernel could only express the shared case, down would
/// stay serial, the host would still need the route ids on the CPU to compute
/// `base + id*expert_bytes`, and the 3.91 ms drain that is 91.5% of decode wall
/// time would survive. Two of three matmuls is not two thirds of the win.
#[test]
fn per_pair_activations_are_expressible_and_shared_is_still_the_default() {
    let bank = bank(GgmlType::Q4_K, HIDDEN, EXPERT_INTERMEDIATE);

    let shared = Glm53GroupedMoePlan::new(bank, 1, TOP_K).unwrap();
    assert_eq!(shared.activations, Glm53GroupedActivations::Shared);
    // Shared indexes by ROW: one block for the single row, reused by all slots.
    assert_eq!(shared.activation_blocks, 1);
    assert_eq!(shared.activation_bytes, shared.tile.activation_bytes);

    let down = Glm53GroupedMoePlan::with_activations(
        bank,
        1,
        TOP_K,
        Glm53GroupedActivations::PerPair,
    )
    .unwrap();
    // One quantized block per pair, back to back, addressed in i32 words.
    assert_eq!(down.activation_block_ints as usize * 4, down.tile.activation_bytes);
    assert_eq!(down.activation_blocks, down.pairs);
    assert_eq!(down.activation_bytes, down.tile.activation_bytes * down.pairs as usize);
    assert_eq!(down.pairs, TOP_K);

    // Rows multiply pairs in both modes, so speculative verify strides too.
    let verify = Glm53GroupedMoePlan::with_activations(
        bank,
        16,
        TOP_K,
        Glm53GroupedActivations::PerPair,
    )
    .unwrap();
    assert_eq!(verify.pairs, 128);
    assert_eq!(verify.activation_blocks, 128);
    assert_eq!(verify.activation_bytes, verify.tile.activation_bytes * 128);

    // AND THE ROWS>1 TRAP THIS CAUGHT: at rows>1 a SHARED plan holds one block
    // per ROW, not one per pair, because `pair` already enumerates rows. A flat
    // per-pair stride would have handed row 1's experts row 8's activations.
    // At rows == 1 both forms give index 0 and neither can be wrong.
    let shared_batch = Glm53GroupedMoePlan::new(bank, 16, TOP_K).unwrap();
    assert_eq!(shared_batch.pairs, 128);
    assert_eq!(shared_batch.activation_blocks, 16, "one block per row, not per pair");
    assert_eq!(shared_batch.tile.rows, 1, "the tile is always one row per pair");
}

/// The quantize plan and the tile's per-pair addressing describe ONE buffer.
///
/// Two plans over one buffer is where a silent disagreement lives: it yields
/// plausible wrong numbers rather than an error. The quantize runs at
/// rows = pairs, the tile at rows per pair, and this pins that the derived
/// quantize plan lands exactly on the extent the tile strides through.
#[test]
fn the_quantize_plan_and_the_per_pair_tile_describe_the_same_buffer() {
    let bank = bank(GgmlType::Q4_K, EXPERT_INTERMEDIATE, HIDDEN);
    let down =
        Glm53GroupedMoePlan::with_activations(bank, 1, TOP_K, Glm53GroupedActivations::PerPair)
            .unwrap();
    let quantize = down.quantize_plan().unwrap();

    assert_eq!(quantize.rows, down.activation_blocks, "quantize must cover every block");
    assert_eq!(quantize.activation_bytes, down.activation_bytes);
    assert_eq!(quantize.inner_padded, down.tile.inner_padded);
    // Exactly linear in rows, so pair p starts at p * stride and the stride is
    // one whole quantized block.
    assert_eq!(
        quantize.activation_bytes,
        down.tile.activation_bytes * down.pairs as usize
    );
    assert_eq!(down.activation_block_ints as usize * 4, down.tile.activation_bytes);

    // A shared plan quantizes one block per row and hands it to every slot.
    let shared = Glm53GroupedMoePlan::new(bank, 1, TOP_K).unwrap();
    assert_eq!(shared.quantize_plan().unwrap().rows, shared.activation_blocks);
    assert_eq!(shared.activation_blocks, 1);
}
