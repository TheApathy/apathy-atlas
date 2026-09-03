// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

const HIDDEN: u32 = 4_096;
const EXPERT_INTERMEDIATE: u32 = 1_024;
const EXPERTS: u32 = 288;
const TOP_K: u32 = 8;
const BASE: DevicePtr = DevicePtr(0x1000_0000_0000);

fn bank(kind: GgmlType, inner: u32, columns: u32) -> Glm53GroupedBank {
    let per = GgmlIqMmqPlan::new(kind, 1, columns, inner).unwrap().weight_bytes;
    Glm53GroupedBank::new(
        BASE,
        kind,
        inner,
        columns,
        EXPERTS,
        per,
        per * EXPERTS as usize,
    )
    .unwrap()
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
        Glm53GroupedBank::new(BASE, kind, HIDDEN, EXPERT_INTERMEDIATE, EXPERTS, per, total).unwrap();
    assert_eq!(good.expert_bytes, per);
    assert_eq!(good.total_bytes, total);

    // A stride that is not the type's own plan -- a per-expert alignment pad
    // is the realistic way this appears, and it is silently wrong, not a
    // crash: expert 1 onward would read shifted weights.
    let padded = Glm53GroupedBank::new(
        BASE,
        kind,
        HIDDEN,
        EXPERT_INTERMEDIATE,
        EXPERTS,
        per + 256,
        (per + 256) * EXPERTS as usize,
    )
    .unwrap_err()
    .to_string();
    assert!(padded.contains("stride is not uniform"), "{padded}");

    // A correct stride over a bank that is not densely packed: a header, a
    // trailing index, a bank sized for a different expert count.
    for total_bytes in [total + 1, total - per, total + per] {
        let error =
            Glm53GroupedBank::new(BASE, kind, HIDDEN, EXPERT_INTERMEDIATE, EXPERTS, per, total_bytes)
                .unwrap_err()
                .to_string();
        assert!(error.contains("not densely packed"), "{total_bytes}: {error}");
    }

    // Degenerate geometry and a null base fail closed.
    assert!(
        Glm53GroupedBank::new(DevicePtr::NULL, kind, HIDDEN, EXPERT_INTERMEDIATE, EXPERTS, per, total)
            .is_err()
    );
    assert!(Glm53GroupedBank::new(BASE, kind, HIDDEN, EXPERT_INTERMEDIATE, 0, per, 0).is_err());
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
    // Rows enter the MMQ tile, not the pair index: the activation buffer grows
    // with rows while the per-pair output slice does not.
    assert!(verify.tile.activation_bytes > decode.tile.activation_bytes);

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
