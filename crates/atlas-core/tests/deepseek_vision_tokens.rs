// SPDX-License-Identifier: AGPL-3.0-only

use atlas_core::config::validate_deepseek_image_tokens;

const V: u32 = 129280;

#[test]
fn deepseek_vision_visibility_is_bidirectional_only_inside_image() {
    let tokens = [7, V + 1, V + 1, V, V + 2, V + 2, V + 3, V + 1, V + 4, 8];
    let spans = validate_deepseek_image_tokens(&tokens, V, 384).unwrap();
    assert_eq!(spans.len(), 1);
    assert_eq!((spans[0].start, spans[0].end), (3, 8));
    assert_eq!(spans[0].raw_bounds(2, 128, 384), (0, 3));
    for row in 3..=8 {
        assert_eq!(spans[0].raw_bounds(row, 128, 384), (0, 9));
    }
    assert_eq!(spans[0].raw_bounds(9, 128, 384), (0, 10));
}

#[test]
fn deepseek_vision_rejects_unmatched_and_unbounded_sentinels() {
    for tokens in [
        vec![V + 2],
        vec![V + 4],
        vec![V + 1],
        vec![V + 5],
        vec![1, 2, 3, V, V + 2, 7, V + 4],
        vec![1, 2, 3, V, V],
        vec![1, 2, 3, V, V + 2],
        vec![V, V + 2, V + 4],
    ] {
        assert!(
            validate_deepseek_image_tokens(&tokens, V, 384).is_err(),
            "{tokens:?}"
        );
    }
    assert!(validate_deepseek_image_tokens(&[1, 2, 3, V, V + 2, V + 4], V, 2).is_err());
    assert!(validate_deepseek_image_tokens(&[1], u32::MAX, 384).is_err());
}

#[test]
fn deepseek_vision_long_span_keeps_original_text_window() {
    let mut tokens = vec![42; 515];
    tokens.push(V);
    tokens.extend(std::iter::repeat_n(V + 2, 378));
    tokens.push(V + 4);
    tokens.push(4);
    let span = validate_deepseek_image_tokens(&tokens, V, 384).unwrap()[0];
    assert_eq!(span.raw_bounds(515, 128, 384), (388, 895));
    assert_eq!(span.raw_bounds(700, 128, 384), (515, 895));
    assert_eq!(span.raw_bounds(894, 128, 384), (515, 895));
    assert_eq!(span.raw_bounds(895, 128, 384), (768, 896));
}
