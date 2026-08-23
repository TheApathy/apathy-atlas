// SPDX-License-Identifier: AGPL-3.0-only

//! Symbol harvest: what an LSP knows about a Rust crate, recovered from
//! source text without a language server.
//!
//! This is the *parse* half. It walks a `.rs` file line by line and
//! recovers the declarations a completion engine would index: `use`
//! paths, function signatures, struct fields, enum variants, `impl`
//! headers, traits, and associated consts. [`crate::symgen`] turns those
//! into draft-shaped text.
//!
//! # Why not `syn`, and why not rustdoc JSON
//!
//! `syn` is in the workspace lock file only as a proc-macro transitive;
//! taking it as a direct dependency would enable `syn/full` and force a
//! rebuild of every derive macro in the tree. rustdoc JSON is the *right*
//! source for dependency signatures but requires a nightly toolchain to
//! successfully build the whole dependency graph, which is not a given on
//! a CUDA-heavy workspace. So this parser is deliberately a heuristic
//! line scanner over text, which works identically on first-party crates
//! and on vendored dependency sources in the cargo registry.
//!
//! # Known imprecision (documented, not hidden)
//!
//! * Brace depth is counted after stripping `//` comments only. A brace
//!   inside a string literal desynchronises the `impl` owner stack; the
//!   effect is a mis-attributed method owner, never a crash.
//! * `macro_rules!` bodies, `#[cfg]`-gated duplicates and generated code
//!   are harvested like any other text.
//! * Multi-line `where` clauses are truncated at the clause.

/// Everything harvested from one source file.
#[derive(Debug, Default, Clone)]
pub struct FileSymbols {
    /// Rust module path, e.g. `rest_store::symbols`. Empty if it could
    /// not be derived from the file's location.
    pub module_path: String,
    /// Crate name in Rust identifier form, e.g. `rest_store`.
    pub crate_name: String,
    /// Normalised, single-line `use ...;` statements.
    pub uses: Vec<String>,
    pub fns: Vec<FnSig>,
    pub structs: Vec<TypeDef>,
    pub enums: Vec<TypeDef>,
    /// `impl` headers, normalised, without the trailing brace.
    pub impls: Vec<String>,
    pub traits: Vec<String>,
    /// `(name, type)` for associated and free consts and statics.
    pub consts: Vec<(String, String)>,
}

/// One function or method signature.
#[derive(Debug, Clone)]
pub struct FnSig {
    pub name: String,
    /// Normalised declaration without the trailing `{` or `;`, e.g.
    /// `pub fn open(path: &Path) -> Result<Self>`.
    pub decl: String,
    /// Parameter names, `self` excluded.
    pub param_names: Vec<String>,
    pub takes_self: bool,
    /// Type named by the enclosing `impl` block, if any.
    pub owner: Option<String>,
    /// True when the item sits inside an `impl`/`trait` body.
    pub nested: bool,
}

/// A struct or enum with its members.
#[derive(Debug, Clone)]
pub struct TypeDef {
    pub name: String,
    /// Struct fields as `(name, type)`; enum variants as
    /// `(variant, payload)` where payload may be empty.
    pub members: Vec<(String, String)>,
}

use crate::symparse::{
    brace_delta, gather_header, header_body, impl_owner, name_after, parse_fn, parse_members,
    squeeze, strip_comment,
};

/// Harvest every indexable declaration from one file's text.
pub fn harvest(text: &str, module_path: &str, crate_name: &str) -> FileSymbols {
    let lines: Vec<&str> = text.lines().collect();
    let mut sym = FileSymbols {
        module_path: module_path.to_string(),
        crate_name: crate_name.to_string(),
        ..Default::default()
    };
    // Stack of (brace depth at which the impl opened, owner type name).
    let mut owners: Vec<(i32, String)> = Vec::new();
    let mut depth = 0i32;
    let mut i = 0;
    while i < lines.len() {
        let raw = strip_comment(lines[i]);
        let t = raw.trim();
        let t = t
            .trim_start_matches("pub(crate) ")
            .trim_start_matches("pub ");
        let owner = owners.last().map(|(_, o)| o.clone());
        let nested = !owners.is_empty();
        let mut consumed = i + 1;

        if t.starts_with("use ") {
            let (h, j) = gather_header(&lines, i);
            let stmt = h.split_once(';').map_or(h.as_str(), |(a, _)| a);
            sym.uses.push(squeeze(stmt) + ";");
            consumed = j;
        } else if t.starts_with("fn ")
            || t.starts_with("async fn ")
            || t.starts_with("unsafe fn ")
            || t.starts_with("const fn ")
            || t.starts_with("extern ") && t.contains("fn ")
        {
            let (h, j) = gather_header(&lines, i);
            if let Some(f) = parse_fn(&h, owner, nested) {
                sym.fns.push(f);
            }
            consumed = j;
        } else if t.starts_with("struct ") {
            let (h, j) = gather_header(&lines, i);
            if let Some(name) = name_after(&h, "struct ") {
                let (members, k) = if h.contains('{') {
                    parse_members(&lines, j - 1, false)
                } else {
                    (Vec::new(), j)
                };
                sym.structs.push(TypeDef { name, members });
                consumed = k.max(j);
            } else {
                consumed = j;
            }
        } else if t.starts_with("enum ") {
            let (h, j) = gather_header(&lines, i);
            if let Some(name) = name_after(&h, "enum ") {
                let (members, k) = parse_members(&lines, j - 1, true);
                sym.enums.push(TypeDef { name, members });
                consumed = k.max(j);
            } else {
                consumed = j;
            }
        } else if t.starts_with("trait ") || t.starts_with("unsafe trait ") {
            let (h, _) = gather_header(&lines, i);
            if let Some(name) = name_after(&h, "trait ") {
                sym.traits.push(name);
            }
        } else if t.starts_with("impl") {
            let (h, _) = gather_header(&lines, i);
            sym.impls.push(header_body(&h).to_string());
        } else if t.starts_with("const ") || t.starts_with("static ") {
            let (h, j) = gather_header(&lines, i);
            let kw = if t.starts_with("const ") {
                "const "
            } else {
                "static "
            };
            if let Some(name) = name_after(&h, kw)
                && let Some(rest) = h.split_once(':')
            {
                let ty = rest.1.split('=').next().unwrap_or("").trim().to_string();
                sym.consts.push((name, ty));
            }
            consumed = j;
        }

        // Owner-stack maintenance over every line the item consumed.
        for l in lines.iter().take(consumed).skip(i) {
            let l = strip_comment(l);
            let opening = l
                .trim()
                .trim_start_matches("pub(crate) ")
                .trim_start_matches("pub ");
            let opens_body = l.contains('{');
            let owner_here = if opening.starts_with("impl") {
                impl_owner(opening)
            } else if opening.starts_with("trait ") || opening.starts_with("unsafe trait ") {
                name_after(opening, "trait ")
            } else {
                None
            };
            let before = depth;
            depth += brace_delta(l);
            if opens_body
                && depth > before
                && let Some(o) = owner_here
            {
                owners.push((before, o));
            }
            while owners.last().is_some_and(|(d, _)| depth <= *d) {
                owners.pop();
            }
        }
        i = consumed.max(i + 1);
    }
    sym
}

/// Owner type named by an `impl` header, for grouping emitted methods.
pub fn impl_owner_of(header: &str) -> Option<String> {
    crate::symparse::impl_owner(header)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symparse::impl_owner;

    const SRC: &str = r#"
use std::path::{Path, PathBuf};
use anyhow::Result;

pub struct RestStore {
    path: PathBuf,
    n_tokens: u64,
}

pub enum MatchSet {
    Empty,
    Hit(u32),
}

impl RestStore {
    pub fn open(path: &Path, fingerprint: Option<u64>) -> Result<Self> {
        todo!()
    }
    pub fn tokens(&self) -> &[u32] {
        &[]
    }
}

pub const DEFAULT_MAX_K: usize = 16;

pub fn free_function(a: u32, b: &str) -> bool { true }
"#;

    #[test]
    fn harvests_uses_and_items() {
        let s = harvest(SRC, "rest_store::store", "rest_store");
        assert_eq!(
            s.uses,
            vec!["use std::path::{Path, PathBuf};", "use anyhow::Result;"]
        );
        assert_eq!(s.structs.len(), 1);
        assert_eq!(s.structs[0].name, "RestStore");
        assert_eq!(
            s.structs[0].members,
            vec![
                ("path".to_string(), "PathBuf".to_string()),
                ("n_tokens".to_string(), "u64".to_string())
            ]
        );
        assert_eq!(s.enums[0].name, "MatchSet");
        assert_eq!(
            s.enums[0]
                .members
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            ["Empty", "Hit"]
        );
        assert_eq!(
            s.consts,
            vec![("DEFAULT_MAX_K".to_string(), "usize".to_string())]
        );
        assert_eq!(s.impls, vec!["impl RestStore"]);
    }

    #[test]
    fn attributes_methods_to_their_impl_owner() {
        let s = harvest(SRC, "rest_store::store", "rest_store");
        let open = s.fns.iter().find(|f| f.name == "open").unwrap();
        assert_eq!(open.owner.as_deref(), Some("RestStore"));
        assert!(!open.takes_self);
        assert_eq!(open.param_names, ["path", "fingerprint"]);
        assert_eq!(
            open.decl,
            "pub fn open(path: &Path, fingerprint: Option<u64>) -> Result<Self>"
        );

        let tokens = s.fns.iter().find(|f| f.name == "tokens").unwrap();
        assert!(tokens.takes_self);
        assert_eq!(tokens.owner.as_deref(), Some("RestStore"));

        let free = s.fns.iter().find(|f| f.name == "free_function").unwrap();
        assert_eq!(
            free.owner, None,
            "a free fn must not inherit the impl owner"
        );
        assert!(!free.nested);
    }

    #[test]
    fn gathers_multiline_signatures() {
        let src = "impl Foo {\n    pub fn wide(\n        alpha: u32,\n        beta: &str,\n    ) -> Result<()> {\n        Ok(())\n    }\n}\n";
        let s = harvest(src, "m", "m");
        let f = &s.fns[0];
        assert_eq!(f.name, "wide");
        assert_eq!(f.param_names, ["alpha", "beta"]);
        assert_eq!(
            f.decl,
            "pub fn wide( alpha: u32, beta: &str, ) -> Result<()>"
        );
        assert_eq!(f.owner.as_deref(), Some("Foo"));
    }

    #[test]
    fn impl_trait_for_type_owner_is_the_type() {
        assert_eq!(
            impl_owner("impl<T> fmt::Display for Wrapper<T>").as_deref(),
            Some("Wrapper")
        );
        assert_eq!(
            impl_owner("impl Default for CorpusOptions").as_deref(),
            Some("CorpusOptions")
        );
        assert_eq!(impl_owner("impl RestStore").as_deref(), Some("RestStore"));
        assert_eq!(
            impl_owner("impl<T: Copy> Buffer<T>").as_deref(),
            Some("Buffer")
        );
        assert_eq!(
            impl_owner("impl<'a> Iterator for Cursor<'a>").as_deref(),
            Some("Cursor")
        );
    }

    #[test]
    fn comments_do_not_confuse_brace_depth() {
        let src = "impl A {\n    // }\n    pub fn m(&self) {}\n}\npub fn after() {}\n";
        let s = harvest(src, "m", "m");
        assert_eq!(
            s.fns
                .iter()
                .find(|f| f.name == "m")
                .unwrap()
                .owner
                .as_deref(),
            Some("A")
        );
        assert_eq!(
            s.fns.iter().find(|f| f.name == "after").unwrap().owner,
            None
        );
    }
}
