// SPDX-License-Identifier: AGPL-3.0-only
#[path = "../examples/glm53_dflash2_kv_parity/projection_choice.rs"]
mod projection_choice;
#[path = "../examples/glm53_dflash2_kv_parity/timing_plan.rs"]
#[allow(dead_code)] // This test consumes scheduling; workload accounting has its own tests.
mod timing_plan;
use projection_choice::ProjectionChoice;
use spark_model::model::glm53::Dflash2ProbeMode as Mode;

#[test]
fn explicit_selection_preserves_original_and_tc_and_rejects_conflicts() {
    assert_eq!(
        ProjectionChoice::select(false, false, false, false).unwrap(),
        None
    );
    assert_eq!(
        ProjectionChoice::select(true, false, false, false).unwrap(),
        Some(ProjectionChoice::Tc)
    );
    assert_eq!(
        ProjectionChoice::select(false, false, true, false).unwrap(),
        Some(ProjectionChoice::Tc)
    );
    for timing in [false, true] {
        assert_eq!(
            ProjectionChoice::select(false, true, timing, false).unwrap(),
            Some(ProjectionChoice::Gemv)
        );
        assert!(ProjectionChoice::select(true, true, timing, false).is_err());
    }
    assert!(ProjectionChoice::select(false, true, true, true).is_err());
    assert!(ProjectionChoice::select(true, false, true, false).is_err());
    assert_eq!(
        ProjectionChoice::select(false, true, false, true).unwrap(),
        Some(ProjectionChoice::Gemv)
    );
}

#[test]
fn each_family_has_exactly_one_original_full_and_its_own_full_cached_modes() {
    let tc = ProjectionChoice::Tc.modes();
    assert_eq!(
        tc,
        [
            Mode::FullRecompute,
            Mode::StableFullProjection,
            Mode::StableCachedProjection
        ]
    );
    let gemv = ProjectionChoice::Gemv.modes();
    assert_eq!(
        gemv.map(Mode::as_str),
        [
            "full-recompute",
            "stable-gemv-full-projection",
            "stable-gemv-cached-projection"
        ]
    );
    assert_eq!(ProjectionChoice::Tc.name(), "stable-tc");
    assert_eq!(ProjectionChoice::Gemv.name(), "stable-gemv");
    for choice in [ProjectionChoice::Tc, ProjectionChoice::Gemv] {
        let orders = (0..6)
            .map(|i| choice.map_order(timing_plan::order(i)).unwrap())
            .collect::<Vec<_>>();
        for mode in choice.modes() {
            for position in 0..3 {
                assert_eq!(orders.iter().filter(|o| o[position] == mode).count(), 2);
            }
        }
        assert!(choice.map_order([Mode::FullRecompute; 3]).is_err());
        assert!(
            choice
                .map_order([Mode::CachedPrefix, tc[1], tc[2]])
                .is_err()
        );
    }
}

#[test]
fn cli_raw_and_timing_reports_bind_the_selected_family() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    let main = std::fs::read_to_string(root.join("glm53_dflash2_kv_parity.rs")).unwrap();
    assert!(main.contains("--gemv-projection"));
    assert!(main.contains("ProjectionChoice::select("));
    assert!(main.matches("\"projection_family\"").count() >= 2);
    for file in ["three_way.rs", "timing.rs"] {
        let source =
            std::fs::read_to_string(root.join("glm53_dflash2_kv_parity").join(file)).unwrap();
        assert!(
            source.contains("choice.modes()"),
            "missing actual mode selection in {file}"
        );
        assert!(
            source.contains("\"projection_family\""),
            "missing family receipt in {file}"
        );
    }
}
