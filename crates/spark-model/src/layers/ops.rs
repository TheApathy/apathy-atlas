// SPDX-License-Identifier: AGPL-3.0-only

//! Shared kernel dispatch operations.
//!
//! Freestanding functions wrapping CUDA kernel launches via `KernelLaunch`.
//! Layer implementations compose these to build forward passes.
//!
//! Each function's parameters exactly match the corresponding CUDA kernel
//! signature. Grid/block dimensions are computed from the problem size.
//!
//! Refactor wave 4a (2026-05-03): split into `ops/` sub-modules with thematic
//! groupings. All public functions remain available at this path via re-export.

#[path = "ops/activations.rs"]
mod activations;
#[path = "ops/dflash2_conv.rs"]
mod dflash2_conv;
#[path = "ops/embeddings.rs"]
mod embeddings;
#[cfg(all(feature = "cuda", target_os = "linux"))]
#[path = "ops/flashinfer_sm121.rs"]
pub mod flashinfer_sm121;
#[path = "ops/fp8_moe.rs"]
mod fp8_moe;
#[path = "ops/fp8_moe_batch_a.rs"]
mod fp8_moe_batch_a;
#[path = "ops/fp8_moe_batch_b.rs"]
mod fp8_moe_batch_b;
#[cfg(all(feature = "cuda", target_os = "linux"))]
#[path = "ops/gdn_c143_sm121.rs"]
pub mod gdn_c143_sm121;
#[path = "ops/gemm_dense.rs"]
mod gemm_dense;
#[path = "ops/gemm_quant.rs"]
mod gemm_quant;
#[path = "ops/gemm_w3.rs"]
mod gemm_w3;
#[path = "ops/gemv_exact_attention.rs"]
mod gemv_exact_attention;
#[path = "ops/gemv_exact_ffn.rs"]
mod gemv_exact_ffn;
#[path = "ops/gemv_exact_ffn_lowreg_m16.rs"]
mod gemv_exact_ffn_lowreg_m16;
#[path = "ops/gemv_exact_lm_head.rs"]
mod gemv_exact_lm_head;
#[path = "ops/gemv_sw.rs"]
mod gemv_sw;
pub(crate) mod ggml_iq_mmq;
mod ggml_q4_embedding;
mod ggml_q5_embedding;
mod ggml_q6_embedding;
mod ggml_q6_token_embedding;
mod ggml_q8_f32;
mod glm53_activations;
mod glm53_dflash2_attention;
mod glm53_dflash2_capture;
mod glm53_dflash2_conv;
mod glm53_dflash2_selector;
mod glm53_dflash2_topk;
mod glm53_dsa_current_visibility;
mod glm53_dsa_index_commit;
mod glm53_dsa_latent_append;
mod glm53_dsa_latent_commit;
mod glm53_dsa_norm_projection;
mod glm53_dsa_pool;
mod glm53_dsa_score;
mod glm53_dsa_selected_attention;
mod glm53_dsa_topk;
mod glm53_exl3;
mod glm53_exl3_cast;
mod glm53_exl3_moe;
mod glm53_exl3_projection;
mod glm53_exl3_reconstruct;
pub(crate) mod glm53_exl3_route_policy;
pub use glm53_exl3_route_policy::Glm53Exl3RoutePolicy;
mod glm53_hyper;
mod glm53_kda;
mod glm53_kda_conv;
mod glm53_kda_conv_fused;
mod glm53_kda_decode;
mod glm53_kda_prefill;
mod glm53_router;
mod glm53_verify_scope;
mod glm53_vision;
#[path = "ops/kv_cache.rs"]
mod kv_cache;
#[path = "ops/kv_cache_fp8k.rs"]
mod kv_cache_fp8k;
#[path = "ops/kv_cache_turbok.rs"]
mod kv_cache_turbok;
#[path = "ops/moe_expert.rs"]
mod moe_expert;
#[path = "ops/moe_expert_more.rs"]
mod moe_expert_more;
#[path = "ops/moe_gate.rs"]
mod moe_gate;
#[path = "ops/moe_grouped_a.rs"]
mod moe_grouped_a;
#[path = "ops/moe_grouped_b.rs"]
mod moe_grouped_b;
#[path = "ops/moe_prefill.rs"]
mod moe_prefill;
#[path = "ops/moe_worklist.rs"]
mod moe_worklist;
#[path = "ops/norm.rs"]
mod norm;
#[path = "ops/nvfp4_dynamic_scale.rs"]
pub mod nvfp4_dynamic_scale;
#[path = "ops/paged_decode_tree.rs"]
mod paged_decode_tree;
#[path = "ops/prefill_attn_a.rs"]
mod prefill_attn_a;
#[path = "ops/prefill_attn_b.rs"]
mod prefill_attn_b;
#[path = "ops/prefill_attn_batched.rs"]
mod prefill_attn_batched;
#[path = "ops/prefill_attn_fp8k.rs"]
mod prefill_attn_fp8k;
#[path = "ops/prefill_attn_main_a.rs"]
mod prefill_attn_main_a;
#[path = "ops/prefill_attn_main_b.rs"]
mod prefill_attn_main_b;
#[path = "ops/prefill_attn_turbok.rs"]
mod prefill_attn_turbok;
#[path = "ops/quant_dispatch.rs"]
mod quant_dispatch;
#[path = "ops/sampling.rs"]
mod sampling;
#[path = "ops/sparsity.rs"]
mod sparsity;
#[path = "ops/ssm_gdn_a.rs"]
mod ssm_gdn_a;
#[path = "ops/ssm_gdn_b.rs"]
mod ssm_gdn_b;
#[path = "ops/ssm_gdn_batched.rs"]
mod ssm_gdn_batched;
#[path = "ops/ssm_mamba.rs"]
mod ssm_mamba;
#[path = "ops/ssm_preproc.rs"]
mod ssm_preproc;

pub use activations::*;
pub use dflash2_conv::*;
pub use embeddings::*;
pub use fp8_moe::*;
pub use fp8_moe_batch_a::*;
pub use fp8_moe_batch_b::*;
pub use gemm_dense::*;
pub use gemm_quant::*;
pub use gemm_w3::*;
pub use gemv_exact_attention::*;
pub use gemv_exact_ffn::*;
pub use gemv_exact_ffn_lowreg_m16::*;
pub use gemv_exact_lm_head::*;
pub use gemv_sw::*;
pub use ggml_iq_mmq::*;
pub use ggml_q4_embedding::*;
pub use ggml_q5_embedding::*;
pub use ggml_q6_embedding::*;
pub use ggml_q6_token_embedding::*;
pub use ggml_q8_f32::*;
pub use glm53_activations::*;
pub use glm53_dflash2_attention::*;
pub use glm53_dflash2_capture::*;
pub use glm53_dflash2_conv::*;
pub use glm53_dflash2_selector::*;
pub use glm53_dflash2_topk::*;
pub use glm53_dsa_current_visibility::*;
pub use glm53_dsa_index_commit::*;
pub use glm53_dsa_latent_append::*;
pub use glm53_dsa_latent_commit::*;
pub use glm53_dsa_norm_projection::*;
pub use glm53_dsa_pool::*;
pub use glm53_dsa_score::*;
pub use glm53_dsa_selected_attention::*;
pub use glm53_dsa_topk::*;
pub use glm53_exl3::*;
pub use glm53_exl3_cast::*;
pub use glm53_exl3_moe::*;
pub use glm53_exl3_projection::*;
pub use glm53_exl3_reconstruct::*;
pub use glm53_hyper::*;
pub use glm53_kda::*;
pub use glm53_kda_conv::*;
pub use glm53_kda_decode::*;
pub use glm53_kda_prefill::*;
pub use glm53_router::*;
pub(crate) use glm53_verify_scope::*;
pub use glm53_vision::*;
pub use kv_cache::*;
pub use kv_cache_fp8k::*;
pub use kv_cache_turbok::*;
pub use moe_expert::*;
pub use moe_expert_more::*;
pub use moe_gate::*;
pub use moe_grouped_a::*;
#[allow(unused_imports)]
pub(crate) use moe_grouped_b::*;
pub use moe_prefill::*;
pub use moe_worklist::*;
pub use norm::*;
pub use paged_decode_tree::*;
pub use prefill_attn_a::*;
pub use prefill_attn_b::*;
pub use prefill_attn_batched::*;
pub use prefill_attn_fp8k::*;
pub use prefill_attn_main_a::*;
pub use prefill_attn_main_b::*;
pub use prefill_attn_turbok::*;
pub use quant_dispatch::*;
pub use sampling::*;
pub use sparsity::*;
pub use ssm_gdn_a::*;
pub use ssm_gdn_b::*;
pub use ssm_gdn_batched::*;
pub use ssm_mamba::*;
pub use ssm_preproc::*;

#[cfg(test)]
#[path = "ops/kernel_call_site_tests.rs"]
mod kernel_call_site_tests;
