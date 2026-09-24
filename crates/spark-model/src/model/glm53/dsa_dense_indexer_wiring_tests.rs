// SPDX-License-Identifier: AGPL-3.0-only
// Standalone std-only source RED; root may run rustc --test without Cargo/CUDA.
const SOURCE: &str = include_str!("dsa_attention.rs");
const SELECTED: &str = include_str!("../../layers/ops/glm53_dsa_selected_attention.rs");

fn between<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let source = &source[source
        .find(start)
        .unwrap_or_else(|| panic!("missing {start}"))..];
    &source[..source.find(end).unwrap_or_else(|| panic!("missing {end}"))]
}
fn compact(source: &str) -> String {
    source.chars().filter(|c| !c.is_whitespace()).collect()
}
fn gated_block<'a>(source: &'a str, marker: &str) -> &'a str {
    let tail = &source[source
        .find(marker)
        .unwrap_or_else(|| panic!("missing {marker}"))..];
    let start = tail.find('{').expect("gate brace");
    let mut depth = 0usize;
    for (offset, c) in tail[start..].char_indices() {
        if c == '{' {
            depth += 1;
        }
        if c == '}' {
            depth -= 1;
            if depth == 0 {
                return &tail[start..=start + offset];
            }
        }
    }
    panic!("unterminated gate")
}

#[test]
fn admission_parses_strict_flag_and_builds_plan_before_precompute() {
    let stage = between(SOURCE, "pub fn stage_exl3_rows(", "fn stage_inner(");
    let stage = compact(stage);
    let flag = stage
        .find("ATLAS_GLM53_DSA_DENSE_INDEXER_SKIP")
        .expect("explicit diagnostic opt-in");
    let parse = stage
        .find("parse_dense_indexer_skip(")
        .expect("strict shared flag parser");
    let plan = stage
        .find("Glm53DsaDenseIndexerPlan::new(")
        .expect("validated query-work plan");
    let effect = stage
        .find("self.precompute_wide_exl3_rows(")
        .expect("actual precompute effect");
    assert!(flag < parse && parse < plan && plan < effect);
    assert!(stage[plan..effect].contains("geometry.position"));
    assert!(stage[plan..effect].contains("geometry.capacity"));
}

#[test]
fn non_utf8_environment_is_an_error_not_a_missing_flag() {
    let stage = compact(between(
        SOURCE,
        "pub fn stage_exl3_rows(",
        "fn stage_inner(",
    ));
    let start = stage
        .find("std::env::var(\"ATLAS_GLM53_DSA_DENSE_INDEXER_SKIP\")")
        .expect("explicit fallible UTF8 environment read");
    let end = stage[start..]
        .find("Glm53DsaDenseIndexerPlan::new(")
        .map(|offset| start + offset)
        .expect("pre-effect plan admission");
    let flag = &stage[start..end];
    assert!(flag.contains("Err(std::env::VarError::NotPresent)=>parse_dense_indexer_skip(None)?"));
    assert!(flag.contains("Err(error)=>returnErr(error).context("));
    assert!(!flag.contains(".ok()"));
    assert!(!flag.contains("unwrap_or"));
}

#[test]
fn precompute_gate_contains_only_query_projection_and_head_weights() {
    let precompute = between(
        SOURCE,
        "fn precompute_wide_exl3_rows(",
        "/// Stage one DSA layer.",
    );
    let precompute = compact(precompute);
    let gate = gated_block(&precompute, "if!dense_indexer.skip_indexer_query()");
    assert_eq!(gate.matches("self.linear_dsa_rows(").count(), 1);
    assert!(gate.contains("weights_ref.indexer_q_b()"));
    assert_eq!(gate.matches("self.norms.launch(").count(), 1);
    assert!(gate.contains("Glm53DsaNormProjectionKind::F32IndexProjection"));
    for required in [
        "weights_ref.q_a()",
        "weights_ref.q_b()",
        "weights_ref.kv_a()",
        "weights_ref.indexer_k()",
        "weights_ref.compressor_gate()",
        "self.absorb_exl3_bank(",
    ] {
        assert!(
            precompute.contains(required),
            "required precompute operation removed: {required}"
        );
        assert!(
            !gate.contains(required),
            "persistent/attention dependency was gated: {required}"
        );
    }
    assert!(
        precompute.contains("dense_indexer.indexer_launches()"),
        "launch receipt must account for skipped work"
    );
}

#[test]
fn density_is_shared_and_sparse_consumers_and_persistent_publications_survive() {
    let inner = compact(&SOURCE[SOURCE.find("fn stage_inner(").unwrap()..]);
    assert!(
        inner.contains("dense_full_coverage(rows,geometry.position,geometry.capacity,"),
        "density must use the shared checked predicate"
    );
    assert!(
        !inner.contains("&&sequence_length<=SELECTED"),
        "do not duplicate the density equation"
    );
    for required in [
        "self.pool.launch(",
        "input_keys_bf16:buffers.index_k_norm_bf16",
        "input_gates_bf16:buffers.index_g_bf16",
        "cache.pool_keys_bf16.ptr.0+key_offset",
        "cache.pool_validity_u8.ptr.0+pool_row",
        "buffers.kv_cmpr_norm_bf16.ptr,destination",
        "buffers.tail_keys_bf16.ptr,cache.prior_tail_keys_bf16.ptr",
        "buffers.tail_gates_bf16.ptr,cache.prior_tail_gates_bf16.ptr",
        "cache.out_tail_validity_u8.ptr,cache.prior_tail_validity_u8.ptr",
        "self.score.launch(",
        "queries_bf16:rows_from(buffers.index_q_bf16,",
        "head_weights_bf16:rows_from(buffers.head_weights_bf16,",
        "self.topk.launch(",
        "self.selected.launch(gpu,selected_plan,selected_buffers,stream)",
    ] {
        assert!(
            inner.contains(required),
            "required sparse/state operation removed: {required}"
        );
    }
    let dense = compact(between(
        SELECTED,
        "pub fn launch_dense_causal(",
        "pub fn launch(",
    ));
    assert!(!dense.contains(".arg_ptr(buffers.selected_indices_i32.ptr)"));
    assert!(!dense.contains("index_q_bf16") && !dense.contains("head_weights_bf16"));
}
