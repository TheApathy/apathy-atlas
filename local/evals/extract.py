"""Extract runnable Python code from a model completion.

Models return code in several shapes:
  1. Fenced ```python ... ``` (or ``` ... ```) blocks.
  2. Bare code (completion mode: the prompt ended mid-function and the model
     just continues the body).
  3. Prose + a fenced block + more prose.

Strategy:
  - If fenced blocks exist, prefer the FIRST python-tagged block, else the
    first fenced block of any kind.
  - Otherwise return the text as-is (bare completion), lightly trimmed of any
    trailing prose after an obvious end.

For HumanEval completion mode we also expose `complete_function`, which stitches
the model's continuation onto the given prompt prefix (the prompt already
contains the signature + docstring).
"""

from __future__ import annotations

import re

_FENCE_RE = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_+-]*)\r?\n(?P<body>.*?)```",
    re.DOTALL,
)


def extract_code(text: str) -> str:
    """Return the best-guess Python code from a completion.

    Prefers a ```python fenced block; falls back to the first fenced block;
    falls back to the raw text (bare code).
    """
    blocks = list(_FENCE_RE.finditer(text))
    if blocks:
        # Prefer an explicitly python-tagged block.
        for m in blocks:
            lang = (m.group("lang") or "").lower()
            if lang in ("python", "py", "python3"):
                return m.group("body").rstrip()
        # Otherwise first fenced block of any language tag.
        return blocks[0].group("body").rstrip()

    # No fences: treat as bare code.
    return _trim_bare(text)


def _trim_bare(text: str) -> str:
    """Strip a leading language echo and obvious trailing prose from bare code."""
    s = text
    # Unclosed fence (completion truncated before the closing ```): the fence
    # regex found no block, but a leading ```python line would poison the code.
    s = re.sub(r"^\s*```[a-zA-Z0-9_+-]*\r?\n", "", s, count=1)
    # Some models prefix a stray "python" line.
    s = re.sub(r"^\s*python\s*\n", "", s, count=1)
    return s.rstrip()


def complete_function(prompt: str, completion: str) -> str:
    """HumanEval completion mode: attach model continuation to the prompt prefix.

    HumanEval prompts end with a signature + docstring, expecting the model to
    emit the function body. If the completion itself contains a full redefinition
    (fenced block with `def`), prefer that standalone code; otherwise stitch.
    """
    code = extract_code(completion)

    # If the extracted code already redefines the entry function (contains a
    # top-level `def`), it is standalone — use it directly.
    if re.search(r"^\s*def\s+\w+\s*\(", code, re.MULTILINE):
        return code

    # Otherwise treat `code` as the raw body continuation and stitch onto prompt.
    body = _stop_at_next_def(completion)
    return prompt + body


def _stop_at_next_def(completion: str) -> str:
    """Cut a bare-body continuation at the start of the next top-level def/class.

    HumanEval solutions occasionally ramble into a second function or into the
    test harness; truncate at the first line that starts a new top-level block.
    """
    lines = completion.splitlines(keepends=True)
    out: list[str] = []
    for i, ln in enumerate(lines):
        # A new top-level def/class (no indentation) after we've emitted body.
        if i > 0 and re.match(r"^(def|class)\s", ln):
            break
        # Common stop markers models emit.
        if re.match(r"^(#\s*Test|if __name__|assert\s)", ln) and out:
            break
        out.append(ln)
    return "".join(out)
