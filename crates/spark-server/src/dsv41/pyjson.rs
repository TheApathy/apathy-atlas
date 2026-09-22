// SPDX-License-Identifier: AGPL-3.0-only

//! Python `json.dumps(x, ensure_ascii=False)` byte parity, and Python `str()`
//! for the handful of places the DeepSeek-V4.1 encoder formats a non-string.
//!
//! The checkpoint's encoder serialises tool schemas, JSON arguments and the
//! response-format schema with `json.dumps`, and those bytes land in the
//! prompt. Separators are `", "` / `": "`, keys keep insertion order
//! (serde_json `preserve_order`), non-ASCII is emitted raw, control characters
//! are escaped exactly as serde_json does (the same set Python escapes), and
//! floats follow Python `repr` (`1e+16`, `1e-05`, `1.0`).

use serde_json::Value;

pub fn dumps(v: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, v);
    out
}

fn write_value(out: &mut String, v: &Value) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&number_repr(n)),
        Value::String(s) => write_str(out, s),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_value(out, x);
            }
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_str(out, k);
                out.push_str(": ");
                write_value(out, x);
            }
            out.push('}');
        }
    }
}

fn write_str(out: &mut String, s: &str) {
    // serde_json's string escaping is Python's for ensure_ascii=False:
    // `"` `\\` `\b` `\f` `\n` `\r` `\t`, other C0 controls as \u00XX.
    out.push_str(&serde_json::to_string(s).unwrap_or_default());
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
}
