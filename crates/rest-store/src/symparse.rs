// SPDX-License-Identifier: AGPL-3.0-only

//! Text-level primitives for the Rust symbol harvest in [`crate::symbols`].
//!
//! Everything here works on raw source text: comment stripping, brace
//! accounting, multi-line item reassembly, and the small extractors that
//! pull a name or a parameter list out of a reassembled item header.
//! Split out of `symbols.rs` purely to keep both files under the 500-line
//! ceiling; the two are one logical unit.

use crate::symbols::FnSig;

/// Strip a trailing `//` line comment, leaving string literals alone
/// well enough for brace counting.
pub(crate) fn strip_comment(line: &str) -> &str {
    let b = line.as_bytes();
    let mut in_str = false;
    let mut i = 0;
    while i + 1 < b.len() {
        match b[i] {
            b'\\' if in_str => i += 1,
            b'"' => in_str = !in_str,
            b'/' if !in_str && b[i + 1] == b'/' => return &line[..i],
            _ => {}
        }
        i += 1;
    }
    line
}

/// Collapse all whitespace runs to single spaces.
pub(crate) fn squeeze(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Net brace delta of a comment-stripped line.
pub(crate) fn brace_delta(line: &str) -> i32 {
    line.bytes()
        .map(|c| match c {
            b'{' => 1,
            b'}' => -1,
            _ => 0,
        })
        .sum()
}

/// Accumulate lines from `i` until the item's header is complete: all
/// `(`/`<` opened by the header are closed AND a `{`, `;` or `where` has
/// been reached. Returns the squeezed header and the index after it.
pub(crate) fn gather_header(lines: &[&str], i: usize) -> (String, usize) {
    let mut acc = String::new();
    let mut paren = 0i32;
    let mut j = i;
    while j < lines.len() && j < i + 64 {
        let l = strip_comment(lines[j]);
        for c in l.bytes() {
            match c {
                b'(' | b'[' => paren += 1,
                b')' | b']' => paren -= 1,
                _ => {}
            }
        }
        if !acc.is_empty() {
            acc.push(' ');
        }
        acc.push_str(l.trim());
        j += 1;
        if paren <= 0
            && (acc.contains('{') || acc.trim_end().ends_with(';') || acc.contains(" where"))
        {
            break;
        }
    }
    (squeeze(&acc), j)
}

/// Cut a gathered header at the first `{`, `;` or ` where`.
pub(crate) fn header_body(h: &str) -> &str {
    let end = [h.find('{'), h.find(';'), h.find(" where ")]
        .into_iter()
        .flatten()
        .min()
        .unwrap_or(h.len());
    h[..end].trim_end()
}

/// Extract the type an `impl` block is for: `impl<T> Trait for Foo<T>`
/// yields `Foo`, `impl Foo` yields `Foo`.
pub fn impl_owner_of(header: &str) -> Option<String> {
    impl_owner(header)
}

pub(crate) fn impl_owner(header: &str) -> Option<String> {
    let body = header_body(header);
    let after = body.rsplit(" for ").next().unwrap_or(body);
    let after = after.strip_prefix("impl").unwrap_or(after);
    // Skip the generic parameter list: `impl<T> Foo<T>` is an impl of
    // `Foo`, not of `T`.
    let mut depth = 0i32;
    let mut rest = after;
    for (i, c) in after.char_indices() {
        match c {
            '<' => depth += 1,
            '>' => depth -= 1,
            _ if depth == 0 && !c.is_whitespace() => {
                rest = &after[i..];
                break;
            }
            _ => {}
        }
        rest = &after[i + c.len_utf8()..];
    }
    let after = rest;
    let mut ident = String::new();
    for c in after.chars() {
        if c.is_alphabetic() || c == '_' {
            ident.push(c);
        } else if ident.is_empty() {
            continue;
        } else if c.is_numeric() {
            ident.push(c);
        } else {
            break;
        }
    }
    (!ident.is_empty()).then_some(ident)
}

/// Name following a keyword: `pub struct Foo<T> {` with `struct` yields `Foo`.
pub(crate) fn name_after(header: &str, keyword: &str) -> Option<String> {
    let idx = header.find(keyword)? + keyword.len();
    let rest = header[idx..].trim_start();
    let ident: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    (!ident.is_empty()).then_some(ident)
}

/// Split a parameter list on top-level commas. Public because
/// [`crate::symgen`] re-wraps long signatures the way rustfmt would.
pub fn split_params(params: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in params.chars() {
        match c {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
                continue;
            }
            _ => {}
        }
        cur.push(c);
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// Byte span of the parameter list: the first `(` and ITS match.
///
/// `rfind(')')` is wrong here — `fn f(x: u32) -> Result<()>` ends with a
/// paren that belongs to the return type, and cutting there truncates
/// the signature mid-generic.
pub fn param_span(decl: &str) -> Option<(usize, usize)> {
    let open = decl.find('(')?;
    let mut depth = 0i32;
    for (i, c) in decl[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some((open, open + i));
                }
            }
            _ => {}
        }
    }
    None
}

/// Parse a gathered `fn` header into a signature.
pub(crate) fn parse_fn(header: &str, owner: Option<String>, nested: bool) -> Option<FnSig> {
    let decl = header_body(header).to_string();
    let name = name_after(&decl, "fn ")?;
    let (open, close) = param_span(&decl)?;
    let params = split_params(&decl[open + 1..close]);
    let mut takes_self = false;
    let mut param_names = Vec::new();
    for p in params {
        let head = p.trim_start_matches(['&', '\'']).trim();
        if head == "self" || head.starts_with("self ") || head.starts_with("mut self") {
            takes_self = true;
            continue;
        }
        let nm = p
            .split(':')
            .next()
            .unwrap_or("")
            .trim()
            .trim_start_matches("mut ")
            .trim();
        if !nm.is_empty() && nm.chars().all(|c| c.is_alphanumeric() || c == '_') {
            param_names.push(nm.to_string());
        }
    }
    Some(FnSig {
        name,
        decl,
        param_names,
        takes_self,
        owner,
        nested,
    })
}

/// Collect members from the braced body that starts at or after line `i`.
///
/// Works for both the inline form (`enum E { A, B(u8) }`) and the
/// multi-line form, because it reassembles the body text first and only
/// then splits it on top-level commas. Attribute and doc lines are
/// dropped; everything else is treated as one member entry.
pub(crate) fn parse_members(
    lines: &[&str],
    i: usize,
    enum_like: bool,
) -> (Vec<(String, String)>, usize) {
    let mut body = String::new();
    let mut depth = 0i32;
    let mut started = false;
    let mut j = i;
    while j < lines.len() {
        let l = strip_comment(lines[j]);
        let t = l.trim();
        if !t.starts_with('#') && !t.starts_with("///") {
            body.push(' ');
            body.push_str(t);
        }
        let before = depth;
        depth += brace_delta(l);
        if before == 0 && depth > 0 {
            started = true;
        }
        j += 1;
        if started && depth <= 0 {
            break;
        }
    }
    let (Some(open), Some(close)) = (body.find('{'), body.rfind('}')) else {
        return (Vec::new(), j);
    };
    let mut out = Vec::new();
    for entry in split_params(&body[open + 1..close]) {
        let e = entry
            .trim()
            .trim_start_matches("pub(crate)")
            .trim_start_matches("pub")
            .trim();
        if e.is_empty() {
            continue;
        }
        if enum_like {
            let nm: String = e
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if nm.is_empty() || !nm.starts_with(char::is_uppercase) {
                continue;
            }
            let payload = e[nm.len()..].trim().to_string();
            out.push((nm, payload));
        } else if let Some((nm, ty)) = e.split_once(':') {
            let nm = nm.trim();
            if !nm.is_empty() && nm.chars().all(|c| c.is_alphanumeric() || c == '_') {
                out.push((nm.to_string(), squeeze(ty)));
            }
        }
    }
    (out, j)
}
