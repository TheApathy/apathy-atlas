// SPDX-License-Identifier: AGPL-3.0-only

//! Boilerplate spans in a Rust source file, for decontaminating an
//! evaluation surface.
//!
//! Held-out-file replay flatters a repo-source draft store: a held-out
//! file opens with the same SPDX header, the same `use` preamble and the
//! same `#[derive(...)]` lines as the 800 files still in the corpus, and
//! those steps are trivially draftable for reasons that have nothing to
//! do with the model writing novel code. This module marks those byte
//! ranges so the eval can EXCLUDE them from scoring.
//!
//! # Excluded from scoring, NOT from the context
//!
//! The boilerplate tokens stay in the replayed stream — a real
//! generation emits them, and the tokens that follow are conditioned on
//! them. Only the decode steps *at* those positions stop counting. The
//! store is not rebuilt and its corpus is unchanged; the difference
//! between the two arms is purely which steps are scored.
//!
//! # What counts as boilerplate here
//!
//! * the leading `//` comment block (SPDX / licence / copyright), but
//!   NOT `///` or `//!` doc comments — module docs are prose someone
//!   actually wrote for that file, and removing them would bias the
//!   result toward the answer this measurement exists to test;
//! * file-scope `use` statements and inner attributes (`#![...]`);
//! * any attribute line (`#[derive(...)]`, `#[arg(...)]`, `#[test]`);
//! * whole `#[cfg(test)]` modules, brace-matched.

/// Half-open byte ranges of `text` considered boilerplate, sorted and
/// non-overlapping.
pub fn boilerplate_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let lines: Vec<(usize, &str)> = line_offsets(text);
    let mut in_header = true;
    let mut i = 0;
    while i < lines.len() {
        let (off, line) = lines[i];
        let t = line.trim_start();
        let end_of = |k: usize| lines[k].0 + lines[k].1.len();

        if in_header {
            if t.is_empty() {
                i += 1;
                continue;
            }
            if t.starts_with("//") && !t.starts_with("///") && !t.starts_with("//!") {
                spans.push((off, end_of(i)));
                i += 1;
                continue;
            }
            in_header = false;
        }

        // File-scope `use` statements: zero indentation only, so that a
        // `use super::*;` inside a test module is not double-counted.
        if (line.starts_with("use ")
            || line.starts_with("pub use ")
            || line.starts_with("pub(crate) use "))
            && let Some(j) = scan_to(&lines, i, |l| l.trim_end().ends_with(';'))
        {
            spans.push((off, end_of(j)));
            i = j + 1;
            continue;
        }

        // Attributes. `#[cfg(test)]` additionally swallows the module it
        // guards, brace-matched from the module's opening brace.
        if t.starts_with("#[") || t.starts_with("#![") {
            // Walk forward to the line where the attribute's brackets
            // close: `#[command(\n ... \n)]` spans several lines.
            let Some(j) = (i..lines.len().min(i + 64)).find(|k| balanced_brackets(&lines[i..=*k]))
            else {
                i += 1;
                continue;
            };
            let is_cfg_test = lines[i..=j].iter().any(|(_, l)| l.contains("cfg(test)"));
            let mut end = end_of(j);
            let mut next = j + 1;
            if is_cfg_test
                && let Some((k, mod_line)) = lines.get(j + 1).map(|(_, l)| (j + 1, l.trim_start()))
                && mod_line.starts_with("mod ")
                && let Some(close) = match_block(&lines, k)
            {
                end = end_of(close);
                next = close + 1;
            }
            spans.push((off, end));
            i = next;
            continue;
        }
        i += 1;
    }
    merge(spans)
}

/// `(byte offset, line text without its terminator)` for every line.
fn line_offsets(text: &str) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    let mut off = 0;
    for line in text.split_inclusive('\n') {
        out.push((off, line.trim_end_matches(['\n', '\r'])));
        off += line.len();
    }
    out
}

/// First index at or after `i` whose line satisfies `pred`, within a
/// bounded window — an unterminated item must not consume the file.
fn scan_to(lines: &[(usize, &str)], i: usize, pred: impl Fn(&str) -> bool) -> Option<usize> {
    (i..lines.len().min(i + 64)).find(|k| pred(lines[*k].1))
}

/// True when `[`/`]` balance across the slice, i.e. the attribute closed.
fn balanced_brackets(lines: &[(usize, &str)]) -> bool {
    let mut d = 0i32;
    for (_, l) in lines {
        for c in strip_line_comment(l).bytes() {
            match c {
                b'[' => d += 1,
                b']' => d -= 1,
                _ => {}
            }
        }
    }
    d == 0
}

/// Index of the line closing the brace block that opens at or after `i`.
fn match_block(lines: &[(usize, &str)], i: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut opened = false;
    for (k, (_, l)) in lines.iter().enumerate().skip(i) {
        for c in strip_line_comment(l).bytes() {
            match c {
                b'{' => {
                    depth += 1;
                    opened = true;
                }
                b'}' => depth -= 1,
                _ => {}
            }
        }
        if opened && depth <= 0 {
            return Some(k);
        }
    }
    None
}

/// Drop a `//` comment tail and the contents of string literals, so that
/// brace and bracket counting is not thrown off by a `"{"` in a test.
///
/// Char literals are handled by explicit lookahead rather than by
/// treating `'` as a quote: a `'` in Rust is far more often a lifetime
/// (`impl<'a>`) than a literal, and quoting on it would swallow the rest
/// of the line.
fn strip_line_comment(line: &str) -> String {
    let b = line.as_bytes();
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    let mut in_str = false;
    while i < b.len() {
        let c = b[i];
        if in_str {
            if c == b'\\' {
                i += 2;
                continue;
            }
            if c == b'"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        match c {
            b'/' if b.get(i + 1) == Some(&b'/') => break,
            b'"' => in_str = true,
            b'\'' if b.get(i + 2) == Some(&b'\'') => i += 2,
            b'\'' if b.get(i + 1) == Some(&b'\\') && b.get(i + 3) == Some(&b'\'') => i += 3,
            _ => out.push(c as char),
        }
        i += 1;
    }
    out
}

fn merge(mut spans: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    spans.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
    for (s, e) in spans {
        match out.last_mut() {
            Some(last) if s <= last.1 => last.1 = last.1.max(e),
            _ => out.push((s, e)),
        }
    }
    out
}

/// One flag per token: `true` where the token's byte span overlaps a
/// boilerplate range and the step must not be scored.
pub fn token_skip_mask(offsets: &[(usize, usize)], spans: &[(usize, usize)]) -> Vec<bool> {
    offsets
        .iter()
        .map(|(s, e)| {
            let (s, e) = (*s, (*e).max(s + 1));
            spans
                .binary_search_by(|(bs, be)| {
                    if *be <= s {
                        std::cmp::Ordering::Less
                    } else if *bs >= e {
                        std::cmp::Ordering::Greater
                    } else {
                        std::cmp::Ordering::Equal
                    }
                })
                .is_ok()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stripped(text: &str) -> String {
        let spans = boilerplate_spans(text);
        let mut out = String::new();
        let mut cur = 0;
        for (s, e) in spans {
            out.push_str(&text[cur..s]);
            cur = e;
        }
        out.push_str(&text[cur..]);
        out
    }

    #[test]
    fn strips_header_uses_and_attributes_but_keeps_doc_comments() {
        let src = "// SPDX-License-Identifier: AGPL-3.0-only\n\n//! Module docs stay.\n\nuse std::path::Path;\nuse anyhow::{Result, Context};\n\n#[derive(Debug, Clone)]\npub struct Cfg {\n    pub depth: usize,\n}\n";
        let out = stripped(src);
        assert!(!out.contains("SPDX"), "{out:?}");
        assert!(!out.contains("use std::path"), "{out:?}");
        assert!(!out.contains("use anyhow"), "{out:?}");
        assert!(!out.contains("#[derive"), "{out:?}");
        assert!(out.contains("//! Module docs stay."), "{out:?}");
        assert!(out.contains("pub struct Cfg {"), "{out:?}");
        assert!(out.contains("pub depth: usize,"), "{out:?}");
    }

    #[test]
    fn strips_whole_cfg_test_module_brace_matched() {
        let src = "pub fn keep() {}\n\n#[cfg(test)]\nmod tests {\n    use super::*;\n    #[test]\n    fn t() {\n        assert_eq!(fmt(\"{\"), \"}\");\n    }\n}\n\npub fn also_keep() {}\n";
        let out = stripped(src);
        assert!(out.contains("pub fn keep() {}"), "{out:?}");
        assert!(out.contains("pub fn also_keep() {}"), "{out:?}");
        assert!(!out.contains("mod tests"), "{out:?}");
        assert!(!out.contains("assert_eq!"), "{out:?}");
    }

    #[test]
    fn multiline_use_and_multiline_attribute_are_consumed_whole() {
        let src = "use std::{\n    path::Path,\n    fs,\n};\n#[command(\n    name = \"x\",\n)]\npub fn f() {}\n";
        let out = stripped(src);
        assert_eq!(out.trim(), "pub fn f() {}");
    }

    #[test]
    fn inner_attributes_go_but_a_lone_comment_mid_file_stays() {
        let src = "#![deny(warnings)]\npub fn a() {}\n// an explanatory comment\npub fn b() {}\n";
        let out = stripped(src);
        assert!(!out.contains("deny(warnings)"), "{out:?}");
        assert!(out.contains("// an explanatory comment"), "{out:?}");
    }

    #[test]
    fn lifetimes_and_char_literals_do_not_desync_brace_matching() {
        let src = "#[cfg(test)]\nmod tests {\n    fn f<'a>(x: &'a str) -> char { '}' }\n    fn g() { let s = \"{{\"; }\n}\npub fn kept() {}\n";
        let out = stripped(src);
        assert_eq!(out.trim(), "pub fn kept() {}", "{out:?}");
    }

    #[test]
    fn mask_marks_exactly_the_tokens_inside_a_span() {
        // Token spans: [0,2) [2,6) [6,9); boilerplate covers bytes 2..6.
        let offsets = [(0usize, 2usize), (2, 6), (6, 9)];
        let mask = token_skip_mask(&offsets, &[(2, 6)]);
        assert_eq!(mask, vec![false, true, false]);
    }

    #[test]
    fn spans_are_sorted_non_overlapping_and_in_bounds() {
        let src = std::fs::read_to_string("src/boilerplate.rs").unwrap();
        let spans = boilerplate_spans(&src);
        assert!(!spans.is_empty());
        for w in spans.windows(2) {
            assert!(w[0].1 <= w[1].0, "overlap: {:?}", w);
        }
        for (s, e) in &spans {
            assert!(s < e && *e <= src.len());
            assert!(src.is_char_boundary(*s) && src.is_char_boundary(*e));
        }
    }
}
