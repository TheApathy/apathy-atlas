// SPDX-License-Identifier: AGPL-3.0-only

//! Sub-modules of `main.rs`, factored out to keep the binary entry-point file ≤500 LoC.

pub(crate) mod app_state;
pub(crate) mod byte_count;
mod context_extension;
pub(crate) mod glm53_gguf_resolver;
pub(crate) mod kv_dtypes;
pub(crate) mod middleware;
pub(crate) mod model_host;
pub(crate) mod model_profile;
pub(crate) mod model_swap;
// Process-scoped state carried across a model swap. Minimal port: the `Carried`
// struct only, not upstream's 1350-line loader. See the module docs.
pub(crate) mod serve;
pub(crate) mod serve_load;
pub(crate) mod serve_phases;
mod serve_router;
mod serve_shutdown;
pub(crate) mod swap_env;

#[cfg(test)]
mod tests;

pub(crate) use app_state::AppState;
pub(crate) use kv_dtypes::{build_layer_kv_dtypes, build_layer_kv_dtypes_from_set};
pub(crate) use serve::serve;
