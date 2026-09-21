// SPDX-License-Identifier: AGPL-3.0-only

const CHUNK: &str = include_str!("../src/model/trait_impl/prefill_b/embed_chunk.rs");

#[test]
fn qwen4_vision_must_preserve_hyper_stream_expansion_from_owned_rows() {
    assert!(CHUNK.contains("self.config.residual_width() * 2"));
    assert!(CHUNK.contains("self.embed(token, hidden_dst.offset(row * row_bytes), stream)?;"));
    assert!(CHUNK.contains("self.splice_vision_embeddings("));
    assert!(!CHUNK.contains("ve.buf_out.offset(img_idx"));
}
