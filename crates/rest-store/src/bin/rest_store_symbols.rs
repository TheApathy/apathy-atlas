// SPDX-License-Identifier: AGPL-3.0-only

//! `rest-store-symbols` — harvest an LSP-style symbol index from a Rust
//! repo and emit it as draft-shaped text for `rest-store-build`.
//!
//! It also owns the FILE-LEVEL holdout used to measure the result
//! honestly: a deterministic path-hash partition selects files that are
//! excluded from BOTH the staged source corpus and the symbol harvest,
//! so a held-out file's own identifiers cannot reach the store by either
//! route.
//!
//! ```text
//! rest-store-symbols --root src --out /tmp/sym/symbols.rs \
//!   --stage-dir /tmp/kept --holdout-frac 0.15 --holdout-seed 7 \
//!   --holdout-list /tmp/heldout.txt
//! ```

#![deny(warnings)]
#![deny(clippy::all)]

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use clap::Parser;
use rest_store::symgen::{emit_file, module_path_for};

#[derive(Parser, Debug)]
#[command(name = "rest-store-symbols", about = "Harvest Rust symbols into draft-shaped text")]
struct Args {
    /// Repo roots to harvest `.rs` files from.
    #[arg(long)]
    root: Vec<PathBuf>,

    /// Directory for the synthesized symbol text. One shard is written
    /// per harvested source file, because `rest-store-build` makes each
    /// file a separate document and continuations never cross a document
    /// separator — one giant shard would let a draft run out of one
    /// symbol's forms and into an unrelated symbol's.
    ///
    /// Pass this directory to `rest-store-build` as the symbols corpus root.
    #[arg(long)]
    out: PathBuf,

    /// Copy every KEPT (non-held-out) source file here, preserving the
    /// path below its root. This is the source-only corpus arm, and
    /// staging is what guarantees held-out files never reach the store.
    #[arg(long)]
    stage_dir: Option<PathBuf>,

    /// Fraction of files to hold out, by deterministic path hash.
    #[arg(long, default_value_t = 0.0)]
    holdout_frac: f64,

    /// Seed mixed into the path hash.
    #[arg(long, default_value_t = 0)]
    holdout_seed: u64,

    /// Write the held-out file paths here, one per line, sorted.
    #[arg(long)]
    holdout_list: Option<PathBuf>,

    /// Directory names never descended into.
    #[arg(long, value_delimiter = ',', default_value = ".git,target,node_modules,.venv")]
    skip_dir: Vec<String>,
}

/// FNV-1a over the seed then the path, so the partition is stable across
/// machines and reproducible from `(frac, seed)` alone.
fn path_hash(rel: &str, seed: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in seed.to_le_bytes().iter().chain(rel.as_bytes()) {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn is_held_out(rel: &str, frac: f64, seed: u64) -> bool {
    if frac <= 0.0 {
        return false;
    }
    (path_hash(rel, seed) % 1_000_000) < (frac * 1_000_000.0) as u64
}

/// Walk one root and return `(relative_path, absolute_path)` for every
/// `.rs` file, sorted, skipping the configured directories.
fn collect_rs(root: &Path, skip: &[String]) -> Result<Vec<(String, PathBuf)>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))? {
            let entry = entry?;
            let path = entry.path();
            let ft = entry.file_type()?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if ft.is_dir() {
                if !skip.contains(&name) {
                    stack.push(path);
                }
            } else if ft.is_file() && name.ends_with(".rs") {
                let rel = path.strip_prefix(root).unwrap_or(&path).to_string_lossy().into_owned();
                out.push((rel, path));
            }
        }
    }
    out.sort();
    Ok(out)
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.root.is_empty() {
        anyhow::bail!("pass at least one --root");
    }
    if !(0.0..1.0).contains(&args.holdout_frac) {
        anyhow::bail!("--holdout-frac must be in [0.0, 1.0)");
    }
    std::fs::create_dir_all(&args.out)?;

    let mut emitted_bytes = 0usize;
    let mut shard = 0usize;
    let mut held: Vec<PathBuf> = Vec::new();
    let (mut n_kept, mut n_parse_fail) = (0usize, 0usize);
    let (mut n_fns, mut n_types, mut n_uses) = (0usize, 0usize, 0usize);

    for root in &args.root {
        let root_name = root.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
        for (rel, abs) in collect_rs(root, &args.skip_dir)? {
            // Hash the root-qualified path so two roots with the same
            // internal layout do not receive an identical partition.
            let key = format!("{root_name}/{rel}");
            if is_held_out(&key, args.holdout_frac, args.holdout_seed) {
                held.push(abs);
                continue;
            }
            n_kept += 1;
            let Ok(text) = std::fs::read_to_string(&abs) else {
                n_parse_fail += 1;
                continue;
            };
            if let Some(stage) = &args.stage_dir {
                let dest = stage.join(&root_name).join(&rel);
                if let Some(p) = dest.parent() {
                    std::fs::create_dir_all(p)?;
                }
                std::fs::write(&dest, &text)?;
            }
            // Derive the module path from the ROOT-RELATIVE path: an
            // absolute path may contain an unrelated `src` component
            // (this workspace lives under `atlas/src`) and would yield
            // module names like `atlas::crates::atlas_core::build`.
            let (krate, module) = module_path_for(Path::new(&rel)).unwrap_or_default();
            let sym = rest_store::symbols::harvest(&text, &module, &krate);
            n_fns += sym.fns.len();
            n_types += sym.structs.len() + sym.enums.len();
            n_uses += sym.uses.len();
            let mut doc = String::new();
            emit_file(&sym, &mut doc);
            if !doc.trim().is_empty() {
                emitted_bytes += doc.len();
                std::fs::write(args.out.join(format!("sym-{shard:05}.rs")), &doc)?;
                shard += 1;
            }
        }
    }

    held.sort();
    if let Some(list) = &args.holdout_list {
        let body: String = held.iter().map(|p| format!("{}\n", p.display())).collect();
        std::fs::write(list, body)?;
    }

    println!("rest-store-symbols");
    println!("  files harvested  {n_kept}  ({} held out, {n_parse_fail} unreadable)", held.len());
    println!("  symbols          {n_fns} fns, {n_types} types, {n_uses} use lines");
    println!(
        "  emitted          {shard} shards in {} -> {:.2} MiB",
        args.out.display(),
        emitted_bytes as f64 / (1024.0 * 1024.0)
    );
    if let Some(s) = &args.stage_dir {
        println!("  staged source    {}", s.display());
    }
    if let Some(l) = &args.holdout_list {
        println!("  holdout list     {} ({} files)", l.display(), held.len());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holdout_is_deterministic_and_roughly_the_requested_size() {
        let paths: Vec<String> = (0..4000).map(|i| format!("crates/c{}/src/m{i}.rs", i % 17)).collect();
        let picked = |seed: u64| -> Vec<&String> {
            paths.iter().filter(|p| is_held_out(p, 0.15, seed)).collect()
        };
        let a = picked(7);
        assert_eq!(a, picked(7), "same seed must reproduce the partition exactly");
        let frac = a.len() as f64 / paths.len() as f64;
        assert!((0.13..0.17).contains(&frac), "got {frac}");
        assert_ne!(a.len(), picked(8).len(), "a different seed must move the split");
    }

    #[test]
    fn zero_frac_holds_out_nothing() {
        assert!(!is_held_out("any/path.rs", 0.0, 3));
    }
}
