// SPDX-License-Identifier: AGPL-3.0-only

// Install as crates/spark-model/tests/glm53_dsa_verify_precompute.rs.
// Authored before the proposed production helper; root owns RED/GREEN execution.
#[path = "../src/model/glm53/dsa_verify_precompute.rs"]
mod candidate;

const SOURCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/model/glm53/dsa_attention.rs"
));

#[test]
fn missing_and_zero_preserve_all_existing_scopes() {
    for value in [None, Some("0")] {
        for rows in [1, 2, 4, 8, 2048] {
            for verify in [false, true] {
                for prefill in [false, true] {
                    assert!(!candidate::select(value, rows, verify, prefill, false).unwrap());
                }
            }
        }
    }
}

#[test]
fn malformed_flags_never_become_implicit_defaults() {
    for value in [
        "", "2", "01", "true", "false", "+1", " 1", "1 ", "1\n", "\0", "１",
    ] {
        for verify in [false, true] {
            assert!(candidate::select(Some(value), 4, verify, false, true).is_err());
        }
    }
}

#[test]
fn opt_in_admits_only_exact_rowwise_verifier_chunks() {
    for rows in 2..=8 {
        assert!(candidate::select(Some("1"), rows, true, false, true).unwrap());
        assert!(candidate::select(Some("1"), rows, true, false, false).is_err());
    }
    for rows in [0, 9, 2048, u32::MAX] {
        assert!(candidate::select(Some("1"), rows, true, false, true).is_err());
    }
    assert!(!candidate::select(Some("1"), 1, true, false, false).unwrap());
}

#[test]
fn prefill_and_plain_work_are_not_enabled_by_the_new_flag() {
    for rows in [1, 2, 8, 2048] {
        assert!(!candidate::select(Some("1"), rows, false, false, false).unwrap());
        assert!(!candidate::select(Some("1"), rows, false, true, false).unwrap());
    }
    assert!(candidate::select(Some("1"), 4, true, true, true).is_err());
}

fn between<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start = source
        .find(start)
        .unwrap_or_else(|| panic!("missing {start}"));
    let source = &source[start..];
    &source[..source.find(end).unwrap_or_else(|| panic!("missing {end}"))]
}

fn compact(source: &str) -> String {
    source.chars().filter(|c| !c.is_whitespace()).collect()
}

#[test]
fn production_uses_strict_selection_before_precompute_or_causal_effects() {
    let stage = compact(between(
        SOURCE,
        "pub fn stage_exl3_rows(",
        "fn stage_inner(",
    ));
    let flag = stage
        .find("std::env::var(\"ATLAS_GLM53_DSA_VERIFY_PRECOMPUTE\")")
        .expect("explicit verifier-only flag read");
    let selection = stage
        .find("dsa_verify_precompute::select(")
        .expect("production selector");
    let effect = stage
        .find("self.precompute_wide_exl3_rows(")
        .expect("precompute effect");
    assert!(flag < selection && selection < effect);
    assert!(stage[flag..selection].contains("Err(std::env::VarError::NotPresent)=>None"));
    assert!(stage[flag..selection].contains("Err(error)=>{"));
    assert!(stage[flag..selection].contains("returnErr(error).context("));
    assert!(stage.contains("letprecompute=ifexact_verify{verify_precompute}else{"));
    assert!(stage.contains("ATLAS_GLM53_EXACT_WIDE_DSA_PRECOMPUTE"));
    assert!(stage.contains("forrowin0..rowsasusize"));
    assert!(stage.contains("DsaStatelessInputs::PrecomputedWide"));
}

#[test]
fn precompute_widens_only_scope_and_retains_exact_inputs() {
    let precompute = compact(between(
        SOURCE,
        "fn precompute_wide_exl3_rows(",
        "/// Stage one DSA layer.",
    ));
    assert!(
        precompute.contains(
            "letexact_wide=glm53_exact_wide_prefill_active()||glm53_exact_verify_active();"
        )
    );
    assert!(precompute.contains("ATLAS_GLM53_EXACT_WIDE_ROWEXACT"));
    for operation in [
        "weights_ref.q_a()",
        "weights_ref.q_b()",
        "weights_ref.kv_a()",
        "weights_ref.indexer_k()",
        "weights_ref.compressor_gate()",
        "weights_ref.indexer_q_b()",
        "self.absorb_exl3_bank(",
        "self.selected.transpose_heads(",
    ] {
        assert!(precompute.contains(operation), "missing {operation}");
    }
    let native_rows = &precompute[precompute.find("forrowin0..rowsasusize").unwrap()..];
    assert!(native_rows.contains("self.linear_dsa_rows(gpu,1,weights_ref.indexer_k()"));
    assert!(native_rows.contains("self.linear_dsa_rows(gpu,1,weights_ref.compressor_gate()"));
    for forbidden in [
        "self.pool.launch(",
        "self.topk.launch(",
        "copy_d2d_async(",
        "cache.",
    ] {
        assert!(
            !precompute.contains(forbidden),
            "causal effect moved into precompute: {forbidden}"
        );
    }
}
