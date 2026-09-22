// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5-Next architecture details that do not belong in generic model fields.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm5NextMlpType {
    Dense,
    Sparse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm5NextIndexerType {
    Full,
    Shared,
}

#[derive(Debug, Clone)]
pub struct Glm5NextConfig {
    pub mlp_layer_types: Vec<Glm5NextMlpType>,
    pub indexer_types: Vec<Glm5NextIndexerType>,
    pub eos_token_ids: Vec<u32>,
    pub index_head_dim: usize,
    pub index_n_heads: usize,
    pub index_topk: usize,
    pub index_kpool: usize,
    pub index_kpool_always_select_tail: bool,
    pub index_kpool_compress: bool,
    pub index_share_for_mtp_iteration: bool,
    pub indexer_rope_interleave: bool,
    pub linear_lower_bound: f32,
    pub swiglu_limit: f32,
    pub mla_use_nope: bool,
}
