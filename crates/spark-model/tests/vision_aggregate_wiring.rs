// SPDX-License-Identifier: AGPL-3.0-only

//! Integration guards complement the byte-bearing production-state unit tests.
//! These require no CUDA, dependencies, or model construction.

const PREPARE: &str = include_str!("../src/model/trait_impl/prefill_a.rs");
const CHUNK: &str = include_str!("../src/model/trait_impl/prefill_b/embed_chunk.rs");
const TWOPHASE: &str = include_str!("../src/model/trait_impl/prefill_c.rs");
const STATE: &str = include_str!("../src/model/types.rs");

#[test]
fn prepare_cannot_sum_overwritten_scratch_as_concatenated_images() {
    assert!(
        !PREPARE.contains("total_patches += p;"),
        "each encoder result must be copied before the next image overwrites scratch"
    );
    assert!(PREPARE.contains("prepare_vision_aggregate(images)"));
}

#[test]
fn every_prefill_path_uses_the_validated_owned_aggregate() {
    for (name, source) in [
        ("ordinary", PREPARE),
        ("chunked", CHUNK),
        ("twophase", TWOPHASE),
    ] {
        assert!(
            source.contains("splice_vision_embeddings("),
            "{name} prefill must splice the same validated aggregate"
        );
        assert!(
            !source.contains("ve.buf_out.offset(img_idx"),
            "{name} prefill must not consume encoder scratch"
        );
    }
}

#[test]
fn cache_identity_must_not_skip_pixel_values() {
    assert!(
        !PREPARE.contains("pixels.chunks(64)"),
        "changing any pixel must invalidate cached image features"
    );
}

#[test]
fn published_state_is_a_single_coherent_owner() {
    assert!(STATE.contains("vision_embeddings: Mutex<"));
    for obsolete in [
        "vision_embed_patches:",
        "vision_cache_fp:",
        "vision_cache_buf:",
    ] {
        assert!(!STATE.contains(obsolete), "obsolete split state {obsolete}");
    }
}
