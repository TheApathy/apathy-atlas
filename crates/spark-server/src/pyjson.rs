// SPDX-License-Identifier: AGPL-3.0-only

//! Python `json.dumps` byte parity, and Python `str()` for the handful of
//! places the DeepSeek-V4.1 encoder formats a non-string.
//!
//! Two users: the DeepSeek-V4.1 encoder (`crate::dsv41`), which serialises
//! tool schemas and arguments with `json.dumps(x, ensure_ascii=False)`, and
//! the chat-template `tojson` filter, which HF transformers defines as
//! `json.dumps(x, ensure_ascii=False, indent=None, separators=None,
//! sort_keys=False)` (`chat_template_utils._compile_jinja_template`).
//! Keys keep insertion order (serde_json `preserve_order`), separators are
//! Python's (`", "`/`": "`, or `","`/`": "` with an indent), non-ASCII is raw
//! unless `ensure_ascii`, control characters are escaped as Python does, and
//! floats follow Python `repr` (`1e+16`, `1e-05`, `1.0`).

use serde_json::Value;

/// `json.dumps` keyword arguments.
#[derive(Debug, Clone)]
pub struct DumpOpts {
    pub ensure_ascii: bool,
    /// `indent`: `None` = one line; `Some(s)` = newline + `s` per level
    /// (Python turns an int n into n spaces).
    pub indent: Option<String>,
    /// `separators`; `None` = Python's default for the indent mode.
    pub separators: Option<(String, String)>,
    pub sort_keys: bool,
}

impl Default for DumpOpts {
    fn default() -> Self {
        Self {
            ensure_ascii: false,
            indent: None,
            separators: None,
            sort_keys: false,
        }
    }
}

/// `json.dumps(v, ensure_ascii=False)`.
pub fn dumps(v: &Value) -> String {
    dumps_with(v, &DumpOpts::default())
}

/// `json.dumps(v, **opts)`.
pub fn dumps_with(v: &Value, opts: &DumpOpts) -> String {
    let (item, key) = match &opts.separators {
        Some((i, k)) => (i.clone(), k.clone()),
        None if opts.indent.is_some() => (",".to_string(), ": ".to_string()),
        None => (", ".to_string(), ": ".to_string()),
    };
    let mut w = Writer {
        out: String::new(),
        opts,
        item,
        key,
    };
    w.value(v, 0);
    w.out
}

struct Writer<'a> {
    out: String,
    opts: &'a DumpOpts,
    item: String,
    key: String,
}

impl Writer<'_> {
    fn newline(&mut self, level: usize) {
        if let Some(ind) = &self.opts.indent {
            self.out.push('\n');
            for _ in 0..level {
                self.out.push_str(ind);
            }
        }
    }

    fn value(&mut self, v: &Value, level: usize) {
        match v {
            Value::Null => self.out.push_str("null"),
            Value::Bool(b) => self.out.push_str(if *b { "true" } else { "false" }),
            Value::Number(n) => self.out.push_str(&number_repr(n)),
            Value::String(s) => write_str(&mut self.out, s, self.opts.ensure_ascii),
            Value::Array(a) if a.is_empty() => self.out.push_str("[]"),
            Value::Array(a) => {
                self.out.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        let sep = self.item.clone();
                        self.out.push_str(&sep);
                    }
                    self.newline(level + 1);
                    self.value(x, level + 1);
                }
                self.newline(level);
                self.out.push(']');
            }
            Value::Object(m) if m.is_empty() => self.out.push_str("{}"),
            Value::Object(m) => {
                let mut entries: Vec<(&String, &Value)> = m.iter().collect();
                if self.opts.sort_keys {
                    entries.sort_by(|a, b| a.0.cmp(b.0));
                }
                self.out.push('{');
                for (i, (k, x)) in entries.into_iter().enumerate() {
                    if i > 0 {
                        let sep = self.item.clone();
                        self.out.push_str(&sep);
                    }
                    self.newline(level + 1);
                    write_str(&mut self.out, k, self.opts.ensure_ascii);
                    let sep = self.key.clone();
                    self.out.push_str(&sep);
                    self.value(x, level + 1);
                }
                self.newline(level);
                self.out.push('}');
            }
        }
    }
}

fn write_str(out: &mut String, s: &str, ensure_ascii: bool) {
    // Python escapes `"` `\\` and C0 controls (`\b \f \n \r \t` short, the
    // rest `\u00xx`, lowercase hex); with ensure_ascii every non-ASCII char
    // too, as `\uxxxx` (UTF-16 surrogate pairs above the BMP) and DEL.
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ensure_ascii && !(' '..='~').contains(&c) => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", unit));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn number_repr(n: &serde_json::Number) -> String {
    if n.is_i64() || n.is_u64() {
        return n.to_string();
    }
    match n.as_f64() {
        Some(f) => float_repr(f),
        None => n.to_string(),
    }
}

/// Python `float.__repr__`: shortest round-trip digits; scientific notation
/// when the decimal exponent is < -4 or >= 16, with a signed, at-least-two
/// digit exponent; otherwise fixed with a trailing `.0` for integral values.
pub fn float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        };
    }
    // `{:e}` gives the shortest round-trip mantissa, e.g. "1.5e20", "1e-5".
    let sci = format!("{f:e}");
    let (mant, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    if !(-4..16).contains(&exp) {
        let sign = if exp < 0 { '-' } else { '+' };
        return format!("{mant}e{sign}{:02}", exp.abs());
    }
    let s = format!("{f}");
    if s.contains('.') { s } else { format!("{s}.0") }
}

/// Python `str(x)` for values the encoder string-formats directly
/// (`"{content}".format(content=None)` renders `None`).
pub fn py_str(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => (if *b { "True" } else { "False" }).into(),
        Value::String(s) => s.clone(),
        Value::Number(n) => number_repr(n),
        // Python would print a dict/list repr here (single quotes); nothing a
        // well-formed OpenAI request reaches, so JSON is close enough.
        other => dumps(other),
    }
}

/// Python truthiness for a JSON value (`if x:`).
pub fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(m)) => !m.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn matches_python_json_dumps() {
        // Expected strings produced by CPython 3 json.dumps(..., ensure_ascii=False).
        let v = json!({"b": [1, 2.5, true, null], "a": "é\n\t\"\\\u{1}", "c": {}});
        assert_eq!(
            dumps(&v),
            r#"{"b": [1, 2.5, true, null], "a": "é\n\t\"\\\u0001", "c": {}}"#
        );
        for (f, want) in [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (1.5e20, "1.5e+20"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (-2.5e-7, "-2.5e-07"),
            (123456.789, "123456.789"),
        ] {
            assert_eq!(float_repr(f), want, "{f}");
        }
    }

    #[test]
    fn dumps_with_matches_python_options() {
        // Expected strings produced by CPython 3.12 json.dumps with these kwargs.
        let v = json!({"b": [1, {"x": "é😀"}], "a": {}, "c": []});
        let o = |ensure_ascii, indent: Option<&str>, sort_keys| DumpOpts {
            ensure_ascii,
            indent: indent.map(str::to_string),
            separators: None,
            sort_keys,
        };
        // json.dumps(v, ensure_ascii=True)
        assert_eq!(
            dumps_with(&v, &o(true, None, false)),
            r#"{"b": [1, {"x": "\u00e9\ud83d\ude00"}], "a": {}, "c": []}"#
        );
        // json.dumps(v, ensure_ascii=False, indent=2, sort_keys=True)
        assert_eq!(
            dumps_with(&v, &o(false, Some("  "), true)),
            "{\n  \"a\": {},\n  \"b\": [\n    1,\n    {\n      \"x\": \"é😀\"\n    }\n  ],\n  \"c\": []\n}"
        );
        // json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        let compact = DumpOpts {
            separators: Some((",".into(), ":".into())),
            ..DumpOpts::default()
        };
        assert_eq!(dumps_with(&v, &compact), r#"{"b":[1,{"x":"é😀"}],"a":{},"c":[]}"#);
        // DEL is raw unless ensure_ascii
        assert_eq!(dumps(&json!("\u{7f}")), "\"\u{7f}\"");
        assert_eq!(dumps_with(&json!("\u{7f}"), &o(true, None, false)), r#""\u007f""#);
    }
}
