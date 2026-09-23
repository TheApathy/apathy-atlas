// SPDX-License-Identifier: AGPL-3.0-only

//! Per-model-family JSON parsers, split out of `config.rs` for file-size
//! budget.

mod deepseek_v4;
mod deepseek_v41;
mod gemma4;
mod glm5_next;
mod minimax;
mod mistral;
mod quantization;
mod qwen_yarn;
mod vision;

pub(crate) use deepseek_v4::parse_deepseek_v4;
pub(crate) use deepseek_v41::parse_deepseek_v41;
pub(crate) use gemma4::parse_gemma4_params;
pub(crate) use glm5_next::parse_glm5_next;
pub(crate) use minimax::parse_minimax_m2;
pub use mistral::parse_mistral_params;
pub use quantization::parse_quantization_config;
pub(crate) use qwen_yarn::parse_qwen_rope_parameters;
pub(crate) use quantization::parse_quantization_config_checked;
pub(crate) use vision::parse_vision_config;
