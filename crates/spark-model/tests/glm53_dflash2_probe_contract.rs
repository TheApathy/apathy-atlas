// SPDX-License-Identifier: AGPL-3.0-only
//! RED: typed, bounded raw observations for actual full-versus-cached proposals.

#[path = "../src/model/glm53/dflash2_probe_contract.rs"]
#[allow(dead_code)]
mod contract;
use contract::{Dflash2ProbeMode, MAX_PROBE_BYTES, ProbeLayout, ProbeStage};

fn actual() -> ProbeLayout {
    ProbeLayout::new(
        5,
        129 * 16 * 8 * 128 * 2,
        8 * 4096 * 2,
        7 * 4096 * 2,
        7 * 154_880 * 2,
        7 * 4,
        154_880,
    )
    .unwrap()
}

#[test]
fn exact_actual_eighteen_stages_and_extents_derive_from_admitted_geometry() {
    let layout = actual();
    let stages = layout.stages();
    assert_eq!(stages.len(), 18);
    for layer in 0..5 {
        assert_eq!(stages[layer * 3], ProbeStage::KeyCache(layer));
        assert_eq!(stages[layer * 3 + 1], ProbeStage::ValueCache(layer));
        assert_eq!(stages[layer * 3 + 2], ProbeStage::Attention(layer));
        assert_eq!(
            layout.bytes(ProbeStage::KeyCache(layer)).unwrap(),
            4_227_072
        );
    }
    assert_eq!(
        &stages[15..],
        &[
            ProbeStage::SelectedHidden,
            ProbeStage::HeadLogits,
            ProbeStage::DraftIds
        ]
    );
    assert_eq!(layout.total_bytes(), 44_824_092);
    assert!(layout.total_bytes() < MAX_PROBE_BYTES);
    assert_eq!(layout.bytes(ProbeStage::Attention(0)).unwrap(), 65_536);
    assert_eq!(layout.bytes(ProbeStage::SelectedHidden).unwrap(), 57_344);
    assert_eq!(layout.bytes(ProbeStage::HeadLogits).unwrap(), 2_168_320);
    assert_eq!(layout.bytes(ProbeStage::DraftIds).unwrap(), 28);
    assert!(layout.bytes(ProbeStage::KeyCache(5)).is_err());
}

#[test]
fn geometry_bounds_and_arithmetic_fail_before_allocation_or_io() {
    assert!(ProbeLayout::new(0, 32, 16, 16, 32, 4, 99).is_err());
    assert!(ProbeLayout::new(1, 0, 16, 16, 32, 4, 99).is_err());
    assert!(ProbeLayout::new(1, 31, 16, 16, 32, 4, 99).is_err());
    assert!(ProbeLayout::new(1, 32, 16, 16, 32, 6, 99).is_err());
    assert!(ProbeLayout::new(1, 32, 16, 16, 32, 4, 0).is_err());
    assert!(ProbeLayout::new(usize::MAX, 32, 16, 16, 32, 4, 99).is_err());
    assert!(ProbeLayout::new(1, usize::MAX - 1, 16, 16, 32, 4, 99).is_err());
    assert!(ProbeLayout::new(1, MAX_PROBE_BYTES, 16, 16, 32, 4, 99).is_err());
}

#[test]
fn modes_are_explicit_and_never_alias_reference_to_cached_adapter() {
    assert_eq!(Dflash2ProbeMode::FullRecompute.as_str(), "full-recompute");
    assert_eq!(Dflash2ProbeMode::CachedPrefix.as_str(), "cached-prefix");
    assert_ne!(
        Dflash2ProbeMode::FullRecompute,
        Dflash2ProbeMode::CachedPrefix
    );
}

#[test]
fn nhd_identity_table_regions_separate_committed_noise_and_unused_pool() {
    let layout = actual();
    for context in [1, 15, 16, 17, 2039, 2047] {
        let spans = layout.kv_regions(context, 8, 2048).unwrap();
        let committed = context as usize * 2048;
        assert_eq!(spans.committed, 0..committed);
        assert_eq!(spans.noise, committed..committed + 16_384);
        assert_eq!(spans.unused, committed + 16_384..4_227_072);
    }
    assert!(layout.kv_regions(0, 8, 2048).is_err());
    assert!(layout.kv_regions(2047, 0, 2048).is_err());
    assert!(layout.kv_regions(2047, 8, 0).is_err());
    assert!(layout.kv_regions(2047, 8, 2047).is_err());
    assert!(layout.kv_regions(u32::MAX, 8, usize::MAX - 1).is_err());
    assert!(layout.kv_regions(2064, 8, 2048).is_err());
}
