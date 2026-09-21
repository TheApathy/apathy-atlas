// SPDX-License-Identifier: AGPL-3.0-only

#[test]
fn vision_l0_capture_is_only_in_hc_prefill_and_brackets_exact_operations() {
    let source = include_str!("../src/layers/qwen3_attention/trait_impl/prefill_inner.rs");
    let (generic, hc) = source.split_once("fn prefill_inner_hc(").unwrap();
    assert!(!generic.contains("Capture::begin("));
    let ordered = [
        "Capture::begin(",
        "Stage::Embed",
        "ops::hc_expand(",
        "Stage::HcExpanded",
        "Stage::HcPreAttn",
        "Stage::PostAttn",
        "Stage::CombAttn",
        "Stage::NormAttn",
        "self.prefill_attention_with_cache_skip(",
        "Stage::AttentionOut",
        "Stage::HcPostAttn",
        "Stage::HcPreFfn",
        "Stage::PostFfn",
        "Stage::CombFfn",
        "Stage::NormFfn",
        ".forward_prefill_observed(normed2,",
        "Stage::MoeOut",
        "Stage::HcPostFfn",
        ".finish()?",
    ];
    let mut cursor = 0;
    for needle in ordered {
        cursor += hc[cursor..]
            .find(needle)
            .unwrap_or_else(|| panic!("missing ordered hook {needle}"))
            + needle.len();
    }
    let decode = include_str!("../src/layers/qwen3_attention/trait_impl/decode_inner.rs");
    assert!(!decode.contains("vision_l0_dump"));
}

#[test]
fn vision_l0_capture_admission_is_default_off_and_has_no_gpu_allocations() {
    let source = include_str!("../src/layers/qwen3_attention/trait_impl/vision_l0_dump.rs");
    for contract in [
        "if layer != 0",
        "std::env::var_os(\"ATLAS_VISION_L0_DUMP\")",
        "c.deepseek_vision.is_some()",
        "start == 0 && write_start == 0",
        "c.ep_world_size <= 1",
        "c.tp_world_size <= 1",
        "ctx.comm.is_none()",
        "meta.num_seqs == 1",
        "!ctx.graph_capture",
        "c.hidden_size == 4096",
        "c.hc_mult == 4",
        "c.vocab_size == VOCAB as usize",
        "self.rows == 12",
        "id < VOCAB",
        "gpu.synchronize(stream)?;",
        "gpu.copy_d2h(ptr, &mut data)?;",
        "self.next == STAGES.len()",
    ] {
        assert!(
            source.contains(contract),
            "missing admission/I/O contract {contract}"
        );
    }
    let filesystem = include_str!("../src/layers/vision_capture_files.rs");
    assert!(source.contains("write_new(&self.root, name, bytes)"));
    assert!(filesystem.contains("create_new(true)"));
    assert!(
        source.find("std::env::var_os(").unwrap() < source.find("let c = ctx.config;").unwrap()
    );
    assert!(!source.contains("gpu.alloc("));
    assert!(!source.contains("copy_h2d("));
    assert!(!source.contains("gpu.launch("));
}
