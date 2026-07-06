"""BFCL-style tool/function-calling accuracy (TERTIARY). SCAFFOLD.

The Atlas engine has grammar-enforced tool calls. This is a tiny AST-based
accuracy check: given a user query + a tool schema, does the model emit a tool
call selecting the right function with correct argument values?

Scope is deliberately small — a handful of single-function cases. It scores
"exact match" on (function name, args dict) against a gold answer, comparing
values structurally (order-insensitive for dicts). It does NOT execute tools.

Two prompting modes:
  - "native": pass `tools=[...]` to /v1/chat/completions and read the returned
    tool_calls (requires the server's grammar-enforced tool path). Preferred.
  - "text":   ask the model to emit a JSON call in the content and parse it.
              Fallback when native tool_calls aren't wired for a given build.

Left as a scaffold: wire the `native` extraction once the exact response shape
of the grammar-enforced path is confirmed against the live server.
"""

from __future__ import annotations

import json
import re

from client import AtlasClient

CASES = [
    {
        "id": "weather_1",
        "query": "What's the weather in Paris in celsius?",
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["city"],
                },
            },
        }],
        "gold": {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}},
    },
    {
        "id": "add_1",
        "query": "Add 17 and 25.",
        "tools": [{
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            },
        }],
        "gold": {"name": "add", "arguments": {"a": 17, "b": 25}},
    },
]


def _args_match(gold: dict, got: dict) -> bool:
    """Order-insensitive structural match on required gold keys."""
    for k, v in gold.items():
        if k not in got:
            return False
        gv = got[k]
        # Loose numeric/string comparison.
        if isinstance(v, str) and isinstance(gv, str):
            if v.strip().lower() != gv.strip().lower():
                return False
        elif v != gv:
            # Allow "17" == 17.
            if str(v) != str(gv):
                return False
    return True


def _extract_native(comp):
    """Read tool_calls from a chat response, if the server returned them."""
    choices = comp.raw.get("choices", [{}])
    msg = choices[0].get("message", {})
    calls = msg.get("tool_calls")
    if not calls:
        return None
    fn = calls[0].get("function", {})
    name = fn.get("name")
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except json.JSONDecodeError:
        args = {}
    return {"name": name, "arguments": args}


def _extract_text(comp):
    """Parse a JSON call the model wrote into content (fallback mode)."""
    m = re.search(r"\{.*\}", comp.text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    name = obj.get("name") or obj.get("function")
    args = obj.get("arguments") or obj.get("args") or {}
    return {"name": name, "arguments": args}


def run(base_url="http://127.0.0.1:8890", model="aeon-27b-dflash",
        mode="native", out_path=None):
    client = AtlasClient(base_url=base_url, model=model)
    records = []
    correct = 0
    for c in CASES:
        # NOTE: AtlasClient.chat doesn't forward `tools` yet; native mode needs
        # that wired once the server response shape is confirmed. Text mode
        # works today.
        if mode == "text":
            prompt = (c["query"] + "\n\nRespond ONLY with a JSON object "
                      '{"name": <fn>, "arguments": {...}} calling the correct '
                      "tool. Tools: " + json.dumps(c["tools"]))
            comp = client.chat([{"role": "user", "content": prompt}],
                               max_tokens=128, temperature=0.0)
            got = _extract_text(comp)
        else:
            # Placeholder path; requires tools forwarding in the client.
            got = None
        gold = c["gold"]
        ok = bool(got and got.get("name") == gold["name"]
                  and _args_match(gold["arguments"], got.get("arguments", {})))
        correct += int(ok)
        records.append({"id": c["id"], "gold": gold, "got": got, "correct": ok})
    acc = correct / len(CASES) if CASES else 0.0
    result = {"model": model, "mode": mode, "accuracy": acc, "records": records}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    print(f"[bfcl] mode={mode} accuracy={acc:.3f} ({correct}/{len(CASES)})")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="BFCL-style tool-calling (scaffold)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8890")
    ap.add_argument("--model", default="aeon-27b-dflash")
    ap.add_argument("--mode", choices=["native", "text"], default="text")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.base_url, a.model, a.mode, a.out)
