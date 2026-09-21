// SPDX-License-Identifier: AGPL-3.0-only

fn compact(source: &str) -> String {
    source.split_whitespace().collect()
}

#[test]
fn mixed_image_admission_precedes_even_the_sequential_fallback() {
    let source = compact(include_str!("trait_impl/decode_b.rs"));
    let guard = source
        .find("!self.vision_prompt_present(prefill_tokens)")
        .unwrap();
    assert!(guard < source.find("self.decode_batch(").unwrap());
    assert!(guard < source.find("self.embed(").unwrap());
    assert!(source.contains("prefill_seq.rotary_positions.is_identity()"));
    assert!(source.contains("decode_seqs.iter().all(|seq|seq.rotary_positions.is_identity())"));
}

#[test]
fn batched_image_decode_is_rejected_before_ssm_conversion() {
    let source = compact(include_str!("trait_impl/decode_a2.rs"));
    let guard = source
        .find("n<=1||seqs.iter().all(|seq|seq.rotary_positions.is_identity())")
        .unwrap();
    assert!(guard < source.find("self.ssm_h_to_f16_dispatch(").unwrap());
}

#[test]
fn target_verify_validates_scalar_tail_before_any_snapshot_or_embedding() {
    for source in [
        include_str!("trait_impl/verify_b.rs"),
        include_str!("trait_impl/verify_c.rs"),
        include_str!("trait_impl/verify_c2.rs"),
    ] {
        let source = compact(source);
        let guard = source.find("seq.rotary_positions.verify_tail(").unwrap();
        assert!(guard < source.find("self.pre_verify_copy_async(").unwrap());
        assert!(guard < source.find("self.embed(").unwrap());
    }
    let source = compact(include_str!("trait_impl/verify_d.rs"));
    assert!(
        source.find("seq.rotary_positions.tail_scalar(").unwrap()
            < source.find("self.ssm_h_to_f16_dispatch(").unwrap()
    );
}

#[test]
fn native_mtp_uses_registered_interleaved_compunit_and_preserves_physical_slots() {
    let constructors = compact(include_str!("../layers/mtp_head/new.rs"));
    assert_eq!(
        constructors
            .matches("gpu.kernel(\"rope_mrope_interleaved\",\"rope_forward_mrope_interleaved\")")
            .count(),
        2
    );
    let forward = compact(include_str!("../layers/mtp_head/forward.rs"));
    assert!(forward.contains("state.block_table[state.seq_len/bs]"));
    assert!(
        forward.contains("self.apply_rotary(state,ctx,q_out,k_out,meta_base,nq,nkv,hd,stream)?")
    );
}
