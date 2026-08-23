// SPDX-License-Identifier: AGPL-3.0-only

//! Draft-text synthesis from harvested symbols.
//!
//! A suffix-array draft store matches on **token runs**, so a bare list
//! of identifiers is worthless: engagement needs the ~10 tokens that
//! *precede* the identifier to also be in the corpus. This module
//! therefore emits, for every symbol, the SHAPES a model actually types
//! around it — the full `use` line, the rustfmt-wrapped signature, the
//! `self.method(args)` call, the `Type::Variant =>` match arm — so that a
//! real completion has a chance of sharing a long prefix with the store.
//!
//! Emission is grouped per source file and per type so that related lines
//! sit adjacent: a continuation that runs off the end of one line then
//! continues into a plausible next line, which is where the long draft
//! chains come from.

use std::fmt::Write as _;
use std::path::Path;

use crate::symbols::{FileSymbols, FnSig, TypeDef};

/// Signature width at which rustfmt breaks a declaration over lines.
const RUSTFMT_MAX_WIDTH: usize = 100;

/// Derive `(crate_name, module_path)` for a `.rs` file under a root.
///
/// `crates/rest-store/src/store.rs` yields `("rest_store",
/// "rest_store::store")`; a registry checkout `tokenizers-0.23.0/src/
/// tokenizer/mod.rs` yields `("tokenizers", "tokenizers::tokenizer")`.
/// Returns `None` when the file is not under a recognisable `src/`.
pub fn module_path_for(file: &Path) -> Option<(String, String)> {
    let parts: Vec<&str> = file.iter().filter_map(|p| p.to_str()).collect();
    let src_at = parts.iter().rposition(|p| *p == "src")?;
    let krate_dir = parts.get(src_at.checked_sub(1)?)?;
    // Strip a trailing `-1.2.3` version suffix from registry checkouts.
    let krate = krate_dir
        .rsplit_once('-')
        .filter(|(_, v)| v.starts_with(|c: char| c.is_ascii_digit()))
        .map_or(*krate_dir, |(n, _)| n)
        .replace('-', "_");
    let mut segs = vec![krate.clone()];
    for p in &parts[src_at + 1..] {
        let stem = p.strip_suffix(".rs").unwrap_or(p);
        if matches!(stem, "lib" | "main" | "mod") {
            continue;
        }
        segs.push(stem.replace('-', "_"));
    }
    Some((krate, segs.join("::")))
}

/// `RestStore` -> `rest_store`; used to name a plausible receiver variable.
fn snake(name: &str) -> String {
    let mut out = String::new();
    for (i, c) in name.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.extend(c.to_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// Expand `use a::b::{C, D};` into the braced original plus one flat line
/// per leaf, because both forms occur in real code.
fn expand_use(line: &str, out: &mut String) {
    let _ = writeln!(out, "{line}");
    let (Some(open), Some(close)) = (line.find('{'), line.rfind('}')) else {
        return;
    };
    let prefix = &line[..open];
    for leaf in line[open + 1..close].split(',') {
        let leaf = leaf.trim();
        if leaf.is_empty() || leaf.contains('{') || leaf == "self" {
            continue;
        }
        let _ = writeln!(out, "{prefix}{leaf};");
    }
}

/// Re-wrap a declaration the way rustfmt would when it exceeds the line
/// budget, so that long signatures match the source form as typed.
fn wrapped_decl(decl: &str, indent: &str, out: &mut String) {
    let Some((open, close)) = crate::symparse::param_span(decl) else {
        return;
    };
    let _ = writeln!(out, "{indent}{}(", &decl[..open]);
    for p in crate::symparse::split_params(&decl[open + 1..close]) {
        let _ = writeln!(out, "{indent}    {p},");
    }
    let _ = writeln!(out, "{indent}){} {{", &decl[close + 1..]);
}

/// Emit one function in definition and call-site forms.
fn emit_fn(f: &FnSig, module: &str, out: &mut String) {
    let indent = if f.nested { "    " } else { "" };
    let _ = writeln!(out, "{indent}{} {{", f.decl);
    if f.decl.len() + indent.len() > RUSTFMT_MAX_WIDTH {
        wrapped_decl(&f.decl, indent, out);
    }
    let args = f.param_names.join(", ");
    match (&f.owner, f.takes_self) {
        (Some(owner), true) => {
            let recv = snake(owner);
            let _ = writeln!(out, "        self.{}({args})", f.name);
            let _ = writeln!(out, "        let _ = self.{}({args});", f.name);
            let _ = writeln!(out, "        {recv}.{}({args})", f.name);
            let _ = writeln!(out, "        {recv}.{}({args})?;", f.name);
        }
        (Some(owner), false) => {
            let _ = writeln!(out, "        {owner}::{}({args})", f.name);
            let _ = writeln!(out, "        let _ = {owner}::{}({args})?;", f.name);
        }
        (None, _) => {
            let _ = writeln!(out, "        {}({args})", f.name);
            if !module.is_empty() {
                let _ = writeln!(out, "        {module}::{}({args})", f.name);
            }
        }
    }
}

/// Emit a struct definition plus field-access and literal-construction forms.
fn emit_struct(t: &TypeDef, out: &mut String) {
    let _ = writeln!(out, "pub struct {} {{", t.name);
    for (nm, ty) in &t.members {
        let _ = writeln!(out, "    pub {nm}: {ty},");
    }
    let _ = writeln!(out, "}}");
    for (nm, _) in &t.members {
        let _ = writeln!(out, "        self.{nm}");
        let _ = writeln!(out, "        let {nm} = self.{nm};");
    }
    let _ = writeln!(out, "    let {} = {} {{", snake(&t.name), t.name);
    for (nm, _) in &t.members {
        let _ = writeln!(out, "        {nm},");
    }
    let _ = writeln!(out, "    }};");
}

/// Emit an enum definition plus path and match-arm forms.
fn emit_enum(t: &TypeDef, out: &mut String) {
    let _ = writeln!(out, "pub enum {} {{", t.name);
    for (nm, payload) in &t.members {
        let _ = writeln!(out, "    {nm}{payload},");
    }
    let _ = writeln!(out, "}}");
    let _ = writeln!(out, "    match {} {{", snake(&t.name));
    for (nm, payload) in &t.members {
        let binder = if payload.starts_with('(') { "(v)" } else { "" };
        let _ = writeln!(out, "        {}::{nm}{binder} => {{}}", t.name);
    }
    let _ = writeln!(out, "    }}");
    for (nm, _) in &t.members {
        let _ = writeln!(out, "        {}::{nm}", t.name);
    }
}

/// Render the full synthesized document for one harvested file.
pub fn emit_file(sym: &FileSymbols, out: &mut String) {
    let _ = writeln!(out, "// module {}", sym.module_path);
    for u in &sym.uses {
        expand_use(u, out);
    }
    // Import lines a caller in another crate/module would type to reach
    // the items defined HERE. This is the part a plain source index
    // cannot know: the path is a property of the file's location, not of
    // any text inside it.
    let named = sym
        .structs
        .iter()
        .chain(sym.enums.iter())
        .map(|t| t.name.as_str())
        .chain(sym.traits.iter().map(String::as_str));
    for name in named {
        if !sym.module_path.is_empty() {
            let _ = writeln!(out, "use {}::{name};", sym.module_path);
            if let Some(inner) = sym.module_path.strip_prefix(&sym.crate_name) {
                let inner = inner.trim_start_matches("::");
                if !inner.is_empty() {
                    let _ = writeln!(out, "use crate::{inner}::{name};");
                }
            }
        }
    }
    for t in &sym.structs {
        emit_struct(t, out);
    }
    for t in &sym.enums {
        emit_enum(t, out);
    }
    // Each `impl` header is followed by ITS OWN methods, so a draft that
    // runs off the end of the header continues into a method that really
    // belongs to that type rather than into an unrelated free function.
    for h in &sym.impls {
        let _ = writeln!(out, "{h} {{");
        let owner = crate::symbols::impl_owner_of(h);
        for f in sym
            .fns
            .iter()
            .filter(|f| f.owner == owner && owner.is_some())
        {
            emit_fn(f, &sym.module_path, out);
        }
    }
    for f in sym.fns.iter().filter(|f| f.owner.is_none()) {
        emit_fn(f, &sym.module_path, out);
    }
    for (name, ty) in &sym.consts {
        let _ = writeln!(out, "pub const {name}: {ty} =");
        let _ = writeln!(out, "        {name}");
        if !sym.module_path.is_empty() {
            let _ = writeln!(out, "        {}::{name}", sym.module_path);
        }
    }
    out.push('\n');
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn module_paths_for_workspace_and_registry_layouts() {
        let f = PathBuf::from("/a/src/crates/rest-store/src/store.rs");
        assert_eq!(
            module_path_for(&f).unwrap(),
            ("rest_store".into(), "rest_store::store".into())
        );
        let f = PathBuf::from("/a/crates/rest-store/src/lib.rs");
        assert_eq!(
            module_path_for(&f).unwrap(),
            ("rest_store".into(), "rest_store".into())
        );
        let f = PathBuf::from("/r/tokenizers-0.23.0/src/tokenizer/mod.rs");
        assert_eq!(
            module_path_for(&f).unwrap(),
            ("tokenizers".into(), "tokenizers::tokenizer".into())
        );
        assert_eq!(
            module_path_for(&PathBuf::from("/no/src/here.rs")),
            Some(("no".into(), "no::here".into()))
        );
        assert_eq!(module_path_for(&PathBuf::from("bare.rs")), None);
    }

    #[test]
    fn use_expansion_keeps_both_forms() {
        let mut s = String::new();
        expand_use("use std::path::{Path, PathBuf};", &mut s);
        assert!(s.contains("use std::path::{Path, PathBuf};"));
        assert!(s.contains("use std::path::Path;"));
        assert!(s.contains("use std::path::PathBuf;"));
    }

    #[test]
    fn method_emission_produces_call_sites() {
        let sym = crate::symbols::harvest(
            "impl RestStore {\n    pub fn tokens(&self) -> &[u32] { &[] }\n}\n",
            "rest_store::store",
            "rest_store",
        );
        let mut s = String::new();
        emit_file(&sym, &mut s);
        assert!(s.contains("    pub fn tokens(&self) -> &[u32] {"), "{s}");
        assert!(s.contains("self.tokens()"), "{s}");
        assert!(s.contains("rest_store.tokens()"), "{s}");
        assert!(s.contains("impl RestStore {"), "{s}");
    }

    #[test]
    fn struct_and_enum_forms_include_paths_and_arms() {
        let sym = crate::symbols::harvest(
            "pub struct Cfg {\n    pub depth: usize,\n}\npub enum E { A, B(u8) }\n",
            "k::m",
            "k",
        );
        let mut s = String::new();
        emit_file(&sym, &mut s);
        assert!(s.contains("use k::m::Cfg;"), "{s}");
        assert!(s.contains("self.depth"), "{s}");
        assert!(s.contains("E::B(v) => {}"), "{s}");
        assert!(s.contains("E::A"), "{s}");
    }

    #[test]
    fn snake_case_receiver_names() {
        assert_eq!(snake("RestStore"), "rest_store");
        assert_eq!(snake("Cfg"), "cfg");
    }
}
