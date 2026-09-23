// SPDX-License-Identifier: AGPL-3.0-only
//! Actual no-readback timing observer and counterbalanced order contract.
#[path = "../examples/glm53_dflash2_kv_parity/timing_observer.rs"]
mod observer;
#[path = "../examples/glm53_dflash2_kv_parity/timing_plan.rs"]
mod plan;
use observer::TimingObserver;
use spark_model::model::glm53::{
    Dflash2ProbeMode as Mode, Glm53Dflash2ProbeObserver, ProbeLayout, ProbeStage,
};

fn layout() -> ProbeLayout {
    ProbeLayout::new(2, 64, 16, 28, 56, 28, 100).unwrap()
}

#[test]
fn observer_requires_every_stage_once_and_complete() {
    let layout = layout();
    let mut observer = TimingObserver::new(layout.clone(), 17, 3).unwrap();
    assert!(!observer.wants_projected_target());
    observer.admit(&layout, 17, 3).unwrap();
    assert!(observer.finish().is_err());
    for &stage in layout.stages() {
        observer
            .record(stage, 4096, layout.bytes(stage).unwrap(), 3)
            .unwrap();
    }
    assert_eq!(observer.finish().unwrap(), layout.stages().len());
    assert!(observer.record(ProbeStage::DraftIds, 4096, 28, 3).is_err());
    assert!(observer.admit(&layout, 17, 3).is_err());
}

#[test]
fn admission_rejects_context_stream_and_layout_drift() {
    let observer = TimingObserver::new(layout(), 17, 3).unwrap();
    assert!(observer.admit(&layout(), 18, 3).is_err());
    assert!(observer.admit(&layout(), 17, 4).is_err());
    let changed = ProbeLayout::new(2, 64, 16, 28, 58, 28, 100).unwrap();
    assert!(observer.admit(&changed, 17, 3).is_err());
    let changed = ProbeLayout::new(2, 64, 16, 28, 56, 28, 101).unwrap();
    assert!(observer.admit(&changed, 17, 3).is_err());
    assert!(TimingObserver::new(layout(), 0, 3).is_err());
    assert!(TimingObserver::new(layout().with_projected_target(17, 4).unwrap(), 17, 3).is_err());
}

#[test]
fn invalid_callback_does_not_advance_stage_or_touch_memory() {
    let mut observer = TimingObserver::new(layout(), 17, 3).unwrap();
    for (stage, ptr, bytes, stream) in [
        (ProbeStage::ValueCache(0), 4096, 64, 3),
        (ProbeStage::KeyCache(0), 0, 64, 3),
        (ProbeStage::KeyCache(0), 4097, 64, 3),
        (ProbeStage::KeyCache(0), u64::MAX - 1, 64, 3),
        (ProbeStage::KeyCache(0), 4096, 62, 3),
        (ProbeStage::KeyCache(0), 4096, 64, 4),
    ] {
        assert!(observer.record(stage, ptr, bytes, stream).is_err());
    }
    observer
        .record(ProbeStage::KeyCache(0), 4096, 64, 3)
        .unwrap();
    observer
        .record(ProbeStage::ValueCache(0), 4096, 64, 3)
        .unwrap();
    assert!(observer.finish().is_err());
}

#[test]
fn schedule_contains_all_six_orders_and_balances_every_position() {
    let modes = [
        Mode::FullRecompute,
        Mode::StableFullProjection,
        Mode::StableCachedProjection,
    ];
    let schedules = (0..6).map(plan::order).collect::<Vec<_>>();
    for (i, row) in schedules.iter().enumerate() {
        assert!(!schedules[..i].contains(row));
        for mode in modes {
            assert_eq!(row.iter().filter(|&&v| v == mode).count(), 1);
        }
    }
    for position in 0..3 {
        for mode in modes {
            assert_eq!(
                schedules.iter().filter(|row| row[position] == mode).count(),
                2
            );
        }
    }
    assert_eq!(plan::order(6), plan::order(0));
}

#[test]
fn cache_history_labels_repeat_as_one_row_not_zero_or_new_context() {
    assert_eq!(plan::workload(0, 17).unwrap(), ("initial-full-context", 17));
    assert_eq!(plan::workload(9, 17).unwrap(), ("after-advance", 8));
    assert_eq!(plan::workload(17, 17).unwrap(), ("repeat-one-row", 1));
    assert!(plan::workload(18, 17).is_err());
    assert!(plan::workload(0, 0).is_err());
}
