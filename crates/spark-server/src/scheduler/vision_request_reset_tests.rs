// SPDX-License-Identifier: AGPL-3.0-only

#[test]
fn both_prefill_entrypoints_invalidate_images_on_text_only_requests() {
    for source in [
        include_str!("prefill_a_step.rs"),
        include_str!("prefill_b_step.rs"),
    ] {
        assert_eq!(
            source
                .matches("model.prepare_vision_embed(&image_pixels)?;")
                .count(),
            1
        );
        assert!(!source.contains("if !image_pixels.is_empty()"));
        assert!(
            source.find("model.prepare_vision_embed(").unwrap()
                < source.find("model.ep_broadcast_cmd(").unwrap()
        );
    }
}
