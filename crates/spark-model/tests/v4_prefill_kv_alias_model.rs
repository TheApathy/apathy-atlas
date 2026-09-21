// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only source and arithmetic contracts for the default-off V4 prefill
//! K/V scratch-alias experiment.

use std::fs;
use std::path::PathBuf;

const SOURCE: &str = "crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source() -> String {
    fs::read_to_string(root().join(SOURCE)).expect("read V4 prefill source")
}

fn compact(text: &str) -> String {
    text.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn identifier_count(text: &str, name: &str) -> usize {
    text.split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
        .filter(|identifier| *identifier == name)
        .count()
}

fn section<'a>(text: &'a str, name: &str) -> &'a str {
    let begin = format!("// BEGIN {name}");
    let end = format!("// END {name}");
    assert_eq!(text.matches(&begin).count(), 1, "duplicate {begin}");
    assert_eq!(text.matches(&end).count(), 1, "duplicate {end}");
    let start = text.find(&begin).unwrap() + begin.len();
    let finish = text[start..].find(&end).unwrap() + start;
    &text[start..finish]
}

fn eligible(gate: bool, nkv: u32, kv_lora: u32, hd_mla: u32, graph: bool, diag: bool) -> bool {
    gate && nkv == 1 && kv_lora == 512 && hd_mla == 512 && !graph && !diag
}

#[test]
fn gate_is_cached_strict_opt_in_and_default_off() {
    let source = source();
    let gate = compact(section(&source, "V4 prefill K/V alias gate"));
    for contract in [
        "fnv4_prefill_kv_alias_enabled()->bool",
        "staticON:std::sync::OnceLock<bool>",
        "std::env::var(\"ATLAS_V4_PREFILL_KV_ALIAS\").as_deref()==Ok(\"1\")",
    ] {
        assert!(gate.contains(contract), "gate omits {contract}");
    }
    assert!(!gate.contains("!=Ok(\"0\")"));
}

#[test]
fn eligibility_is_exact_shape_and_excludes_graphs_and_diagnostics() {
    let source = source();
    let alias = compact(section(&source, "V4 prefill K/V alias decision"));
    for contract in [
        "v4_prefill_kv_alias_enabled()",
        "nkv==1",
        "kv_lora==512",
        "hd_mla==512",
        "!ctx.graph_capture",
        "!diag_this",
    ] {
        assert!(alias.contains(contract), "eligibility omits {contract}");
    }

    assert!(eligible(true, 1, 512, 512, false, false));
    for rejected in [
        eligible(false, 1, 512, 512, false, false),
        eligible(true, 2, 512, 512, false, false),
        eligible(true, 1, 511, 512, false, false),
        eligible(true, 1, 512, 576, false, false),
        eligible(true, 1, 512, 512, true, false),
        eligible(true, 1, 512, 512, false, true),
    ] {
        assert!(!rejected);
    }
}

#[test]
fn only_dead_k_to_v_copy_is_skipped_and_fallback_is_unchanged() {
    let source = source();
    let alias = compact(section(&source, "V4 prefill K/V alias decision"));
    assert!(alias.contains("copy_d2d_async(kv_latent,k_out"));
    assert!(alias.contains("if!kv_alias"));
    assert!(alias.contains("copy_d2d_async(k_out,v_out,(n*kv_dim)asusize*2,stream)"));
    assert!(alias.contains("ifdiag_this"));
    assert!(alias.contains("diag_norm(ctx.gpu,v_out"));

    let rope = source.find("// ── 3. RoPE on Q and K").unwrap();
    let alias_end = source.find("// END V4 prefill K/V alias decision").unwrap();
    assert!(alias_end < rope, "copy fallback moved after K mutation");
}

#[test]
fn v_out_has_no_consumer_after_projection_and_attention_alias_stays_k_out() {
    let source = source();
    let after_projection = source.split("aprof!(\"2_kv_proj\");").nth(1).unwrap();
    assert_eq!(identifier_count(after_projection, "v_out"), 0);

    let attention = compact(
        source
            .split("// ── 4. Core attention ──")
            .nth(1)
            .unwrap()
            .split("// ── 5. Assemble KV cache")
            .next()
            .unwrap(),
    );
    let raw_launches: Vec<_> = attention
        .split(".arg_ptr(q_full)")
        .skip(1)
        .map(|tail| tail.split(".launch(stream)").next().unwrap())
        .collect();
    assert_eq!(raw_launches.len(), 2);
    assert_eq!(raw_launches[0].matches(".arg_ptr(k_out)").count(), 2);
    assert_eq!(raw_launches[1].matches(".arg_ptr(k_out)").count(), 4);
    assert!(attention.contains(
        "prefill_attention_512_sink(ctx.gpu,self.prefill_attn_512_k,q_full,k_out,k_out,attn_out"
    ));
    assert_eq!(identifier_count(&attention, "v_out"), 0);
    assert!(!attention.contains("v_attn"));
}

#[test]
fn kv_latent_remains_live_read_only_through_cache_assembly() {
    let source = source();
    let after_norm = source.split("// Copy kv_latent → k_out").nth(1).unwrap();
    let before_output_projection = after_norm
        .split("// ── 6. Grouped low-rank O projection")
        .next()
        .unwrap();
    assert!(!before_output_projection.contains("expert_gate_out()"));

    let cache_call = source
        .split("ops::mla_cache_assemble_batched(")
        .nth(1)
        .unwrap()
        .split(")?;")
        .next()
        .unwrap();
    assert!(cache_call.contains("kv_latent"));
    assert!(cache_call.contains("k_rope_tmp"));
}

#[test]
fn exact_copy_traffic_and_enqueue_savings_are_frozen() {
    const TOKENS: u64 = 2_410;
    const DIMS: u64 = 512;
    const BF16_BYTES: u64 = 2;
    const LAYERS: u64 = 43;
    let payload = TOKENS * DIMS * BF16_BYTES;
    let logical_traffic = payload * 2;
    assert_eq!(payload, 2_467_840);
    assert_eq!(logical_traffic, 4_935_680);
    assert_eq!(logical_traffic * LAYERS, 212_234_240);
    assert_eq!(LAYERS, 43);

    let source = source();
    for evidence in [
        "4,935,680 logical bytes/layer",
        "212,234,240 logical bytes/pass",
        "43 D2D enqueues/pass",
    ] {
        assert!(
            source.contains(evidence),
            "missing arithmetic evidence {evidence}"
        );
    }
}
