// SPDX-License-Identifier: AGPL-3.0-only

const ENTRY: &str = include_str!("../qwen38_flashinfer_dynamic_quantizer_microgate.rs");
const CONTRACT: &str = include_str!("contract.rs");
const FIXTURES: &str = include_str!("fixtures.rs");
const GATE: &str = include_str!("gate.rs");
const GUARDED: &str = include_str!("guarded.rs");
const INVALID: &str = include_str!("invalid.rs");
const LAUNCH: &str = include_str!("launch.rs");
const PROVENANCE: &str = include_str!("provenance.rs");
const TESTS: &str = include_str!("tests.rs");
const VALID: &str = include_str!("valid.rs");

#[test]
fn every_source_respects_file_cap() {
    for (name, source) in sources() {
        assert!(source.lines().count() <= 250, "{name} exceeds 250 lines");
        assert!(source.starts_with("// SPDX-License-Identifier: AGPL-3.0-only\n"));
    }
}

#[test]
fn aggregate_gate_contract_is_retained() {
    let source = sources()
        .into_iter()
        .map(|(_, source)| source)
        .collect::<String>();
    for required in [
        "ROWS: [u32; 2] = [2_079, 8_192]",
        "COLS: [u32; 2] = [5_120, 6_144]",
        "Fixture::ALL",
        "parity_arms=28",
        "invalid=6",
        "cross-stream output accepted",
        "packed E2M1",
        "physical 128x4 E4M3",
        "device scale2 bits differ",
        "device combined alpha bits differ",
        "deterministic packed",
        "redzones=PASS immutable=PASS deterministic=PASS",
        "invalid-scales",
        "invalid-maximum",
        "invalid-scale2",
        "invalid-alpha",
        "no_production_route=true",
    ] {
        assert!(
            source.contains(required),
            "missing gate contract {required}"
        );
    }
}

fn sources() -> [(&'static str, &'static str); 10] {
    [
        ("entry", ENTRY),
        ("contract", CONTRACT),
        ("fixtures", FIXTURES),
        ("gate", GATE),
        ("guarded", GUARDED),
        ("invalid", INVALID),
        ("launch", LAUNCH),
        ("provenance", PROVENANCE),
        ("tests", TESTS),
        ("valid", VALID),
    ]
}
