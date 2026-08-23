// SPDX-License-Identifier: AGPL-3.0-only

//! REST-style retrieval draft store.
//!
//! Implements the datastore half of REST (*Retrieval-Based Speculative
//! Decoding*, NAACL'24): a large corpus of code is tokenized once with
//! the **target** tokenizer, concatenated into a single token stream,
//! and indexed with a suffix array. At decode time the last few context
//! tokens are looked up by longest-suffix-match and the verbatim
//! continuations that followed that suffix in the corpus become draft
//! tokens — a frequency-weighted trie, verified in one pass by the
//! existing tree-verify path.
//!
//! The store is entirely CPU-side and costs no GPU work to propose:
//! that is the whole point. A draft that the target rejects costs only
//! the verify slots it occupied, and verify width is already paid for.
//!
//! # Layout
//!
//! - [`boilerplate`] — boilerplate spans, for decontaminating an eval surface.
//! - [`format`] — on-disk header, section offsets, tokenizer fingerprint.
//! - [`sa`] — suffix array construction over a `u32` token alphabet.
//! - [`jsonl`] — generation-row schema shared by the builder and the eval.
//! - [`build`] — corpus walk, tokenization, holdout, assembly.
//! - [`serialize`] — writing the indexed corpus to disk.
//! - [`store`] — mmap reader and [`store::RestStore::longest_suffix_match`].
//! - [`trie`] — frequency-weighted continuation trie ([`trie::DraftTree`]).
//! - [`symbols`] — LSP-style symbol harvest from Rust source text.
//! - [`symparse`] — text primitives backing the harvest.
//! - [`symgen`] — synthesis of draft-shaped text from harvested symbols.

#![deny(warnings)]
#![deny(clippy::all)]

pub mod boilerplate;
pub mod build;
pub mod format;
pub mod jsonl;
pub mod sa;
pub mod sam;
pub mod serialize;
pub mod store;
pub mod symbols;
pub mod symgen;
pub mod symparse;
pub mod trie;

pub use format::{CACHE_FORMAT_VERSION, StoreHeader, tokenizer_fingerprint};
pub use jsonl::{GenerationRow, Holdout, IngestStats, load_rows};
pub use store::{MatchSet, RestStore};
pub use symbols::{FileSymbols, FnSig, TypeDef, harvest};
pub use symgen::{emit_file, module_path_for};
pub use trie::{DraftNode, DraftTree, TrieParams, build_draft_trie};

/// Default cap on how many context tokens participate in a suffix match.
///
/// REST's own ablation puts the useful ceiling around 16: longer suffixes
/// almost never occur verbatim in a corpus, so the extra binary searches
/// are wasted. Matches the `max_match` of the existing prompt-lookup
/// proposer in `spark-server::ngram`.
pub const DEFAULT_MAX_K: usize = 16;

/// Default engage threshold — below this match length the continuation is
/// noise and drafting it wastes verify slots.
pub const DEFAULT_MIN_MATCH: usize = 8;

/// Default continuation depth (max draft chain length below the root).
pub const DEFAULT_DEPTH: usize = 16;

/// Default node cap for a proposed tree, matching `ATLAS_DDTREE_MAX_NODES`
/// defaults on the verify side.
pub const DEFAULT_MAX_NODES: usize = 16;

/// Default cap on suffix-array occurrences scanned per lookup.
///
/// A short suffix can occur tens of thousands of times; the trie only
/// needs enough samples to rank continuations, and an unbounded scan
/// would put a corpus-size-dependent tail on the decode step.
pub const DEFAULT_MAX_OCCURRENCES: usize = 64;
