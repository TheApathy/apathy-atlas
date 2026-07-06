"""Code-extraction tests: fenced python, bare fenced, bare code, stitching."""
from extract import extract_code, complete_function


def test_fenced_python_block():
    text = "Here is the solution:\n```python\ndef f(x):\n    return x + 1\n```\nDone."
    code = extract_code(text)
    assert code == "def f(x):\n    return x + 1"


def test_fenced_untagged_block():
    text = "```\ndef g():\n    return 42\n```"
    code = extract_code(text)
    assert "def g():" in code
    assert "42" in code


def test_prefers_python_over_other_lang():
    text = "```json\n{\"a\":1}\n```\nand\n```python\ndef h():\n    return 7\n```"
    code = extract_code(text)
    assert "def h():" in code
    assert "json" not in code


def test_bare_code_no_fence():
    text = "def sq(x):\n    return x * x\n"
    code = extract_code(text)
    assert code.strip() == "def sq(x):\n    return x * x"


def test_bare_strips_leading_language_echo():
    text = "python\ndef k():\n    return 1\n"
    code = extract_code(text)
    assert code.startswith("def k():")


def test_complete_function_standalone_redef():
    prompt = "def foo(x):\n    \"\"\"doc\"\"\"\n"
    completion = "```python\ndef foo(x):\n    return x * 2\n```"
    out = complete_function(prompt, completion)
    # Standalone redefinition used directly, prompt not duplicated.
    assert out.count("def foo(") == 1
    assert "return x * 2" in out


def test_complete_function_stitches_bare_body():
    prompt = "def bar(x):\n    \"\"\"doc\"\"\"\n"
    completion = "    return x + 100\n"
    out = complete_function(prompt, completion)
    assert out.startswith("def bar(x):")
    assert "return x + 100" in out


def test_complete_function_stops_at_test_harness():
    prompt = "def baz(x):\n    \"\"\"doc\"\"\"\n"
    completion = "    return x\n\nassert baz(1) == 1\n"
    out = complete_function(prompt, completion)
    assert "return x" in out
    assert "assert baz" not in out
