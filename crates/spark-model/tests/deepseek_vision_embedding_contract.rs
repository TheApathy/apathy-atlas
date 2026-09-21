// SPDX-License-Identifier: AGPL-3.0-only

#[test]
fn deepseek_vision_splice_precedes_both_ordinary_embedding_paths() {
    for source in [
        include_str!("../src/model/trait_impl/prefill_a.rs"),
        include_str!("../src/model/trait_impl/prefill_b/embed_chunk.rs"),
    ] {
        let splice = source
            .find("self.embed_deepseek_chunk(")
            .expect("DeepSeek splice gate");
        let ordinary = source
            .find("ops::batched_embed(")
            .expect("ordinary embedding");
        assert!(splice < ordinary);
    }
}

#[test]
fn deepseek_vision_factory_and_cache_safety_are_explicit() {
    let factory = include_str!("../src/factory/build.rs");
    assert!(factory.contains("install_deepseek_vision"));
    let cache = include_str!("../src/model/impl_a2.rs");
    let helper = cache
        .split("fn tokens_have_vision_pad")
        .nth(1)
        .unwrap()
        .split("/// Free")
        .next()
        .unwrap();
    assert!(helper.contains("deepseek_vision.is_some()"));
    assert!(helper.contains("token as usize >= self.config.vocab_size"));
}

#[test]
fn deepseek_vision_direct_decode_cannot_index_sentinel_weights() {
    let source = include_str!("../src/model/impl_a3.rs");
    let embed = source
        .split("fn embed(")
        .nth(1)
        .unwrap()
        .split("///")
        .next()
        .unwrap();
    assert!(embed.find("deepseek_vision.is_none()").unwrap() < embed.find(".offset(").unwrap());
}
