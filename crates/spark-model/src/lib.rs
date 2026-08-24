// SPDX-License-Identifier: AGPL-3.0-only

#![deny(warnings)]
#![deny(clippy::all)]
// The PR #74 merge added a few diagnostic / upstream-only helpers our
// DFlash path doesn't call. Earlier this was a crate-level
// `#![allow(dead_code)]` shotgun, which masked any NEW dead code we
// introduced anywhere in spark-model. The shotgun is now scoped to
// the 4 individual files that legitimately need it (env-gated debug
// dumpers + upstream-only prefill helpers); deny(warnings) again
// catches new dead code in our own additions.
//
// Empirically validated: removing `unused_imports` + `unused_variables`
// shotguns produces ZERO new warnings; only `dead_code` had real items
// to silence, which are now narrowed to file-level allows.
// Kernel-launch helpers and trait-impl wide signatures legitimately exceed
// clippy's 7-argument default. The same goes for the indexing-loop patterns
// that mirror the kernel grids we dispatch.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
// Some FP/integer special-case branches return the same value but have
// distinct semantic meanings (NaN vs zero, etc.). Audit shows these are
// intentional.
#![allow(clippy::if_same_then_else)]
// The HSS / disk-spill plumbing threads `Vec<u32>` through trait methods so
// callers can grow them in place; converting to slices breaks the contract.
#![allow(clippy::ptr_arg)]
// HF safetensors index tuples are wide on purpose.
#![allow(clippy::type_complexity)]

pub mod direction_vector;
pub mod engine;
pub mod factory;
pub mod full_profile;
pub mod layer;
pub mod layers;
pub mod mistral_loader;
pub mod model;
pub mod precision_schedule;
pub mod preflight;
pub mod quant_format;
pub mod speculative;
pub mod tp_shard;
pub mod traits;
pub mod vision_preprocess;
pub mod weight_loader;
pub mod weight_map;
