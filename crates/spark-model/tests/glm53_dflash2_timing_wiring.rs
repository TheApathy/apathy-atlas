// SPDX-License-Identifier: AGPL-3.0-only
//! Protect the timing boundaries and prevent raw-dump speed misattribution.
#[test]
fn timing_observer_has_no_gpu_or_host_io() {
    let source = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/glm53_dflash2_kv_parity/timing_observer.rs"
    ))
    .unwrap();
    for forbidden in [
        "copy_d2h",
        "copy_h2d",
        "synchronize(",
        "std::fs",
        "unsafe",
        "ProbeCapture",
    ] {
        assert!(!source.contains(forbidden), "observer contains {forbidden}");
    }
    assert!(source.contains("_gpu: &dyn GpuBackend"));
    assert!(source.contains("self.record(stage, source.ptr.0, source.bytes, stream)"));
}

#[test]
fn timing_setup_and_artifact_io_stay_outside_measured_call() {
    let source = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/glm53_dflash2_kv_parity/timing.rs"
    ))
    .unwrap();
    let start = source.find("let started = Instant::now();").unwrap();
    let stop = source[start..]
        .find("let elapsed = started.elapsed();")
        .unwrap()
        + start;
    let timed = &source[start..stop];
    assert!(timed.contains("propose_installed_diagnostic_mode"));
    assert!(timed.contains("propose_diagnostic"));
    for forbidden in [
        "artifacts::",
        "session.advance",
        "session.anchor",
        "TimingObserver::new",
    ] {
        assert!(
            !timed.contains(forbidden),
            "measured setup/I/O: {forbidden}"
        );
    }
    assert!(source[..start].contains("synchronize(session.stream)?"));
    // Completion errors are retained in JSON before propagation, not discarded
    // by an early question-mark return before the sample receipt is written.
    assert!(source[stop..].contains("let observed = observer.finish();"));
    assert!(source[stop..].contains("\"observer_error\":observed.as_ref().err()"));
    assert!(source.contains("\"raw_exact_qualified\":false"));
    assert!(source.contains("\"decode_speed_qualified\":false"));
}

#[test]
fn timing_cli_cannot_masquerade_as_a_raw_exact_run() {
    let source = include_str!("../examples/glm53_dflash2_kv_parity.rs");
    assert!(source.contains("\"--proposal-timing\""));
    assert!(source.contains("!proposal_timing || (!stable_projection && !with_projected_target)"));
    assert!(source.contains("let raw_exact = !proposal_timing && matches!(executed, Ok(true));"));
    assert!(source.contains("\"TIMING_COMPLETE\""));
}
