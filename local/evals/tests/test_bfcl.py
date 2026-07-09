"""GPU-free tests for bfcl.py native tool-call wiring.

Uses canned server responses in the EXACT shape the Atlas server emits
(openai/chat_response.rs ChatCompletionResponse::with_tool_calls):
  choices[0].message.tool_calls = [{"id","type":"function",
      "function": {"name": str, "arguments": "<json string>"}}]
  finish_reason == "tool_calls"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bfcl  # noqa: E402
from client import Completion  # noqa: E402


def _canned_tool_response(name: str, arguments: dict | str,
                          content: str | None = None) -> Completion:
    args_str = arguments if isinstance(arguments, str) else json.dumps(arguments)
    raw = {
        "id": "chatcmpl-fixture",
        "object": "chat.completion",
        "model": "aeon-27b-dflash",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": [{
                    "id": "call_0001",
                    "type": "function",
                    "function": {"name": name, "arguments": args_str},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"completion_tokens": 20},
    }
    return Completion(text=content or "", raw=raw)


def _canned_plain_response(content: str) -> Completion:
    raw = {
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content,
                        "tool_calls": None},
            "finish_reason": "stop",
        }],
    }
    return Completion(text=content, raw=raw)


class FixtureClient:
    """Stub AtlasClient returning canned responses keyed by case query."""

    def __init__(self, responses: dict[str, Completion]):
        self.responses = responses
        self.requests: list[dict] = []

    def chat(self, messages, *, tools=None, **kw):
        self.requests.append({"messages": messages, "tools": tools, **kw})
        return self.responses[messages[-1]["content"]]


# ── _extract_native on the server's exact shape ─────────────────────────

def test_extract_native_parses_server_shape():
    comp = _canned_tool_response("get_weather",
                                 {"city": "Paris", "unit": "celsius"})
    got, meta = bfcl._extract_native(comp)
    assert got == {"name": "get_weather",
                   "arguments": {"city": "Paris", "unit": "celsius"}}
    assert meta["returned_call"] is True
    assert meta["args_json_ok"] is True
    assert meta["finish_reason"] == "tool_calls"


def test_extract_native_no_call():
    got, meta = bfcl._extract_native(_canned_plain_response("Shakespeare."))
    assert got is None
    assert meta["returned_call"] is False
    assert meta["finish_reason"] == "stop"


def test_extract_native_unparseable_arguments():
    comp = _canned_tool_response("add", '{"a": 17, "b": ')  # truncated JSON
    got, meta = bfcl._extract_native(comp)
    assert got == {"name": "add", "arguments": {}}
    assert meta["returned_call"] is True
    assert meta["args_json_ok"] is False


# ── matching: nested / loose ───────────────────────────────────────────

def test_args_match_nested_and_loose():
    gold = {"customer": {"name": "John Doe", "address": "5 Main St"},
            "items": [{"sku": "ABC-123", "qty": 2}]}
    got = {"customer": {"name": "john doe", "address": "5 main st"},
           "items": [{"sku": "abc-123", "qty": "2"}],
           "extra": "ignored"}
    assert bfcl._args_match(gold, got)
    # wrong list length fails
    assert not bfcl._args_match(gold, {**got, "items": []})
    # bool mismatch fails
    assert not bfcl._args_match({"prefs": {"email": False}},
                                {"prefs": {"email": True}})
    assert bfcl._args_match({"prefs": {"email": False}},
                            {"prefs": {"email": False}})


def test_score_no_tool_case():
    case = {"gold": None}
    assert bfcl._score_case(case, None)
    assert not bfcl._score_case(case, {"name": "get_weather", "arguments": {}})


# ── end-to-end run() against the fixture client ────────────────────────

def test_run_native_with_fixture_client(tmp_path):
    cases = bfcl.CASES
    responses = {}
    for c in cases:
        if c["gold"] is None:
            responses[c["query"]] = _canned_plain_response("A direct answer.")
        else:
            responses[c["query"]] = _canned_tool_response(
                c["gold"]["name"], c["gold"]["arguments"])
    fx = FixtureClient(responses)
    out = tmp_path / "bfcl.json"
    result = bfcl.run(mode="native", client=fx, out_path=str(out))
    assert result["accuracy"] == 1.0
    assert result["n_cases"] == len(cases)
    assert result["native_format"]["unparseable_arguments"] == 0
    # tools were forwarded on every request
    assert all(r["tools"] for r in fx.requests)
    # results file round-trips
    saved = json.loads(out.read_text())
    assert saved["accuracy"] == 1.0
    # category coverage: all 5 categories present
    assert set(saved["by_category"]) == {
        "simple", "multi", "no_tool", "required_params", "nested"}


def test_run_native_scores_wrong_tool_as_fail():
    # Server picks the WRONG function on multi cases -> those fail.
    responses = {}
    for c in bfcl.CASES:
        if c["gold"] is None:
            responses[c["query"]] = _canned_tool_response("get_weather",
                                                          {"city": "X"})
        else:
            responses[c["query"]] = _canned_tool_response(
                "totally_wrong_fn", c["gold"]["arguments"])
    fx = FixtureClient(responses)
    result = bfcl.run(mode="native", client=fx)
    assert result["accuracy"] == 0.0


def test_client_chat_body_includes_tools():
    """AtlasClient.chat must forward tools + tool_choice into the POST body."""
    from client import AtlasClient
    captured = {}

    client = AtlasClient(base_url="http://fixture")
    def fake_post(path, body):
        captured["path"] = path
        captured["body"] = body
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}
    client._post = fake_post

    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    client.chat([{"role": "user", "content": "q"}], tools=tools,
                tool_choice="auto", enable_thinking=True)
    assert captured["path"] == "/v1/chat/completions"
    assert captured["body"]["tools"] == tools
    assert captured["body"]["tool_choice"] == "auto"
    assert captured["body"]["enable_thinking"] is True
