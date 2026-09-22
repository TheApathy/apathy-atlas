// SPDX-License-Identifier: AGPL-3.0-only

use super::python_canonical;

#[test]
fn ignores_layout_whitespace_only() {
    let compact = "scores=(unary+call(\"br,bkr->bk\",left,right))";
    let layout = "scores = (\n unary +\n call ( \"br,bkr->bk\" , left , right )\n)";
    assert_eq!(python_canonical(compact), python_canonical(layout));
    assert_eq!(
        python_canonical("for x in y:\n  use(x)"),
        python_canonical("for x in y:\n    use ( x )")
    );
}

#[test]
fn statement_indent_and_comment_boundaries_are_authoritative() {
    for (valid, hostile) in [
        ("for x in y:\n  use(x)", "for x in y:\nuse(x)"),
        ("x = 1 #\ny = 2", "x = 1\n# y = 2"),
        ("x = 1\ny = 2", "x = 1 y = 2"),
    ] {
        assert_ne!(python_canonical(valid), python_canonical(hostile));
    }
}

#[test]
#[should_panic(expected = "indentation does not follow a suite")]
fn malformed_continuation_fails_closed() {
    let _ = python_canonical("value = left +\n  right");
}

#[test]
#[should_panic(expected = "unmatched delimiter")]
fn mismatched_delimiter_fails_closed() {
    let _ = python_canonical("call([value))");
}

#[test]
fn punctuation_boundaries_and_string_bytes_are_authoritative() {
    for (valid, hostile) in [
        ("offset == 0", "offset = = 0"),
        ("load(...) ", "load(. . .)"),
        ("\"br,bkr->bk\"", "\"br,bkr- >bk\""),
        ("value := 1", "value : = 1"),
        ("1.25e-3", "1 . 25e - 3"),
        (r#""a\\\"b""#, r#""a\"b""#),
        ("left", "right"),
    ] {
        assert_ne!(python_canonical(valid), python_canonical(hostile));
    }
}

#[test]
#[should_panic(expected = "unterminated quoted string")]
fn unterminated_quote_fails_closed() {
    let _ = python_canonical("call(\"unterminated)");
}

#[test]
#[should_panic(expected = "official Python receipt must be ASCII")]
fn non_ascii_fails_closed() {
    let _ = python_canonical("π");
}
