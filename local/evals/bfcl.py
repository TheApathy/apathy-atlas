"""BFCL-style tool/function-calling accuracy.

The Atlas engine has grammar-enforced tool calls (qwen3_coder XML dialect,
parsed server-side into OpenAI `message.tool_calls`). This is an AST-based
accuracy check: given a user query + tool schemas, does the model emit a tool
call selecting the right function with correct argument values? It does NOT
execute tools.

Server shape (confirmed against crates/spark-server/src/openai/chat_response.rs
+ api/chat_blocking.rs, 2026-07-08):
  request : {"tools": [{"type":"function","function":{name,description,
             parameters}}], "tool_choice": "auto"|"required"|{...}}
  response: choices[0].message.tool_calls = [{"id", "type":"function",
             "function": {"name": str, "arguments": "<json string>"}}]
            finish_reason == "tool_calls" when a call fired.

Two prompting modes:
  - "native": pass `tools=[...]` to /v1/chat/completions and read the returned
    tool_calls (grammar-enforced tool path). Preferred.
  - "text":   ask the model to emit a JSON call in the content and parse it.
              Fallback for builds where native tool_calls aren't wired.

Scoring: correct = (function name matches gold) AND (all gold args present and
structurally equal, recursive for nested objects/arrays, case-insensitive for
strings). "no_tool" cases score correct iff NO tool call was emitted.

Format-validity is tracked separately in native mode: a returned tool_call
whose `arguments` fail json.loads counts as unparseable (the grammar should
make this impossible — the eval verifies that claim).
"""

from __future__ import annotations

import json
import re

from client import AtlasClient

# ── Eval set: 20 cases across 5 categories ─────────────────────────────
# simple          : one tool, obvious call
# multi           : choose the right tool among 3
# no_tool         : tools offered but the query needs a direct answer
# required_params : all required params must be extracted
# nested          : nested object / array arguments

def _fn(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props,
                       "required": required}}}


_WEATHER = _fn("get_weather", "Get current weather for a city.",
               {"city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
               ["city"])
_ADD = _fn("add", "Add two integers.",
           {"a": {"type": "integer"}, "b": {"type": "integer"}}, ["a", "b"])
_SUB = _fn("subtract", "Subtract b from a.",
           {"a": {"type": "integer"}, "b": {"type": "integer"}}, ["a", "b"])
_MUL = _fn("multiply", "Multiply two integers.",
           {"a": {"type": "integer"}, "b": {"type": "integer"}}, ["a", "b"])

CASES = [
    # -- simple ----------------------------------------------------------
    {"id": "weather_1", "category": "simple",
     "query": "What's the weather in Paris in celsius?",
     "tools": [_WEATHER],
     "gold": {"name": "get_weather",
              "arguments": {"city": "Paris", "unit": "celsius"}}},
    {"id": "add_1", "category": "simple",
     "query": "Add 17 and 25.",
     "tools": [_ADD],
     "gold": {"name": "add", "arguments": {"a": 17, "b": 25}}},
    {"id": "stock_1", "category": "simple",
     "query": "What is the current stock price of AAPL?",
     "tools": [_fn("get_stock_price", "Get the latest stock price for a ticker.",
                   {"ticker": {"type": "string"}}, ["ticker"])],
     "gold": {"name": "get_stock_price", "arguments": {"ticker": "AAPL"}}},
    {"id": "timer_1", "category": "simple",
     "query": "Set a timer for 15 minutes.",
     "tools": [_fn("set_timer", "Set a countdown timer.",
                   {"minutes": {"type": "integer"}}, ["minutes"])],
     "gold": {"name": "set_timer", "arguments": {"minutes": 15}}},
    {"id": "currency_1", "category": "simple",
     "query": "Convert 250 USD to EUR.",
     "tools": [_fn("convert_currency", "Convert an amount between currencies.",
                   {"amount": {"type": "number"},
                    "from_currency": {"type": "string", "enum": ["USD", "EUR", "GBP", "JPY"]},
                    "to_currency": {"type": "string", "enum": ["USD", "EUR", "GBP", "JPY"]}},
                   ["amount", "from_currency", "to_currency"])],
     "gold": {"name": "convert_currency",
              "arguments": {"amount": 250, "from_currency": "USD",
                            "to_currency": "EUR"}}},
    {"id": "email_1", "category": "simple",
     "query": 'Send an email to bob@example.com with the subject "Lunch".',
     "tools": [_fn("send_email", "Send an email.",
                   {"to": {"type": "string"}, "subject": {"type": "string"},
                    "body": {"type": "string"}}, ["to", "subject"])],
     "gold": {"name": "send_email",
              "arguments": {"to": "bob@example.com", "subject": "Lunch"}}},

    # -- multi: choose among 3 tools --------------------------------------
    {"id": "multi_time_1", "category": "multi",
     "query": "What time is it right now in Tokyo?",
     "tools": [_WEATHER,
               _fn("get_news", "Get top news headlines for a topic.",
                   {"topic": {"type": "string"}}, ["topic"]),
               _fn("get_time", "Get the current local time in a city.",
                   {"city": {"type": "string"}}, ["city"])],
     "gold": {"name": "get_time", "arguments": {"city": "Tokyo"}}},
    {"id": "multi_hotel_1", "category": "multi",
     "query": "Book me a hotel in Berlin for 3 nights.",
     "tools": [_fn("book_flight", "Book a flight.",
                   {"origin": {"type": "string"}, "destination": {"type": "string"}},
                   ["origin", "destination"]),
               _fn("book_hotel", "Book a hotel stay.",
                   {"city": {"type": "string"}, "nights": {"type": "integer"}},
                   ["city", "nights"]),
               _fn("rent_car", "Rent a car.",
                   {"city": {"type": "string"}, "days": {"type": "integer"}},
                   ["city", "days"])],
     "gold": {"name": "book_hotel",
              "arguments": {"city": "Berlin", "nights": 3}}},
    {"id": "multi_mul_1", "category": "multi",
     "query": "What is 8 times 9?",
     "tools": [_ADD, _SUB, _MUL],
     "gold": {"name": "multiply", "arguments": {"a": 8, "b": 9}}},
    {"id": "multi_delete_1", "category": "multi",
     "query": "Delete the file /tmp/old.log",
     "tools": [_fn("create_file", "Create a new file.",
                   {"path": {"type": "string"}, "content": {"type": "string"}},
                   ["path"]),
               _fn("delete_file", "Delete a file.",
                   {"path": {"type": "string"}}, ["path"]),
               _fn("read_file", "Read a file's contents.",
                   {"path": {"type": "string"}}, ["path"])],
     "gold": {"name": "delete_file", "arguments": {"path": "/tmp/old.log"}}},
    {"id": "multi_count_1", "category": "multi",
     "query": "How many words are in the text 'the quick brown fox'?",
     "tools": [_fn("translate_text", "Translate text to a target language.",
                   {"text": {"type": "string"}, "target_lang": {"type": "string"}},
                   ["text", "target_lang"]),
               _fn("summarize_text", "Summarize text.",
                   {"text": {"type": "string"}}, ["text"]),
               _fn("count_words", "Count the words in a text.",
                   {"text": {"type": "string"}}, ["text"])],
     "gold": {"name": "count_words",
              "arguments": {"text": "the quick brown fox"}}},

    # -- no_tool: tools offered but irrelevant -----------------------------
    {"id": "notool_1", "category": "no_tool",
     "query": "Who wrote the play Hamlet?",
     "tools": [_WEATHER],
     "gold": None},
    {"id": "notool_2", "category": "no_tool",
     "query": "What is the capital of France? Just answer in words.",
     "tools": [_ADD],
     "gold": None},
    {"id": "notool_3", "category": "no_tool",
     "query": "Tell me a fun fact about cats.",
     "tools": [_fn("book_flight", "Book a flight.",
                   {"origin": {"type": "string"}, "destination": {"type": "string"}},
                   ["origin", "destination"])],
     "gold": None},

    # -- required_params: several required args to extract ------------------
    {"id": "req_user_1", "category": "required_params",
     "query": "Create a user account with username 'jdoe', email jdoe@example.com, age 34.",
     "tools": [_fn("create_user", "Create a user account.",
                   {"username": {"type": "string"}, "email": {"type": "string"},
                    "age": {"type": "integer"}}, ["username", "email", "age"])],
     "gold": {"name": "create_user",
              "arguments": {"username": "jdoe", "email": "jdoe@example.com",
                            "age": 34}}},
    {"id": "req_meeting_1", "category": "required_params",
     "query": ("Schedule a meeting titled 'Sync' on 2026-07-10 at 14:00 "
               "for 30 minutes."),
     "tools": [_fn("schedule_meeting", "Schedule a calendar meeting.",
                   {"title": {"type": "string"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "time": {"type": "string", "description": "HH:MM 24h"},
                    "duration_minutes": {"type": "integer"}},
                   ["title", "date", "time", "duration_minutes"])],
     "gold": {"name": "schedule_meeting",
              "arguments": {"title": "Sync", "date": "2026-07-10",
                            "time": "14:00", "duration_minutes": 30}}},
    {"id": "req_rect_1", "category": "required_params",
     "query": "Compute the area of a rectangle 7 units wide and 12 units tall.",
     "tools": [_fn("calculate_area", "Area of a rectangle.",
                   {"width": {"type": "number"}, "height": {"type": "number"}},
                   ["width", "height"])],
     "gold": {"name": "calculate_area",
              "arguments": {"width": 7, "height": 12}}},

    # -- nested: nested objects / arrays -----------------------------------
    {"id": "nested_order_1", "category": "nested",
     "query": ("Place an order for customer John Doe at address '5 Main St' "
               "with 2 units of SKU ABC-123."),
     "tools": [_fn("create_order", "Create a purchase order.",
                   {"customer": {"type": "object", "properties": {
                        "name": {"type": "string"},
                        "address": {"type": "string"}},
                        "required": ["name", "address"]},
                    "items": {"type": "array", "items": {
                        "type": "object", "properties": {
                            "sku": {"type": "string"},
                            "qty": {"type": "integer"}},
                        "required": ["sku", "qty"]}}},
                   ["customer", "items"])],
     "gold": {"name": "create_order",
              "arguments": {"customer": {"name": "John Doe",
                                         "address": "5 Main St"},
                            "items": [{"sku": "ABC-123", "qty": 2}]}}},
    {"id": "nested_settings_1", "category": "nested",
     "query": "Set my theme to dark and turn OFF email notifications.",
     "tools": [_fn("update_settings", "Update user preference settings.",
                   {"preferences": {"type": "object", "properties": {
                        "theme": {"type": "string", "enum": ["light", "dark"]},
                        "notifications": {"type": "object", "properties": {
                            "email": {"type": "boolean"}},
                            "required": ["email"]}},
                        "required": ["theme", "notifications"]}},
                   ["preferences"])],
     "gold": {"name": "update_settings",
              "arguments": {"preferences": {
                  "theme": "dark",
                  "notifications": {"email": False}}}}},
    {"id": "nested_tags_1", "category": "nested",
     "query": "Tag item 42 with the tags 'urgent' and 'review'.",
     "tools": [_fn("add_tags", "Add tags to an item.",
                   {"item_id": {"type": "integer"},
                    "tags": {"type": "array", "items": {"type": "string"}}},
                   ["item_id", "tags"])],
     "gold": {"name": "add_tags",
              "arguments": {"item_id": 42, "tags": ["urgent", "review"]}}},
    {"id": "nested_points_1", "category": "nested",
     "query": ("Move the robot along the waypoints (0,0) then (3,4) then "
               "(6,8), in that order."),
     "tools": [_fn("move_robot", "Move a robot through waypoints.",
                   {"waypoints": {"type": "array", "items": {
                       "type": "object", "properties": {
                           "x": {"type": "number"}, "y": {"type": "number"}},
                       "required": ["x", "y"]}}},
                   ["waypoints"])],
     "gold": {"name": "move_robot",
              "arguments": {"waypoints": [{"x": 0, "y": 0},
                                          {"x": 3, "y": 4},
                                          {"x": 6, "y": 8}]}}},
]


# ── Matching ────────────────────────────────────────────────────────────

def _value_match(gold, got) -> bool:
    """Recursive structural match: dicts (gold keys ⊆ got), lists (elementwise),
    strings case-insensitive, numerics loose ("17" == 17, 7 == 7.0)."""
    if isinstance(gold, dict):
        if not isinstance(got, dict):
            return False
        return all(k in got and _value_match(v, got[k]) for k, v in gold.items())
    if isinstance(gold, list):
        if not isinstance(got, list) or len(gold) != len(got):
            return False
        return all(_value_match(g, o) for g, o in zip(gold, got))
    if isinstance(gold, bool) or isinstance(got, bool):
        # Don't let bool/int cross-typing (True == 1) or "false" strings slip.
        if isinstance(gold, bool) and isinstance(got, bool):
            return gold == got
        if isinstance(got, str):
            return str(gold).lower() == got.strip().lower()
        return gold == got
    if isinstance(gold, str) and isinstance(got, str):
        return gold.strip().lower() == got.strip().lower()
    if gold == got:
        return True
    # Loose numeric / stringified comparison ("17" == 17, 7 == 7.0).
    try:
        return float(gold) == float(got)
    except (TypeError, ValueError):
        return str(gold) == str(got)


def _args_match(gold: dict, got: dict) -> bool:
    """Order-insensitive structural match on required gold keys (recursive)."""
    return _value_match(gold, got)


# ── Extraction ──────────────────────────────────────────────────────────

def _extract_native(comp):
    """Read tool_calls from a chat response, if the server returned them.

    Returns (call_dict | None, meta) where meta records format validity:
      returned_call  : server emitted message.tool_calls
      args_json_ok   : the arguments string parsed as JSON
      finish_reason  : the choice's finish_reason
    """
    choices = comp.raw.get("choices", [{}])
    msg = choices[0].get("message", {}) or {}
    meta = {"returned_call": False, "args_json_ok": None,
            "finish_reason": choices[0].get("finish_reason")}
    calls = msg.get("tool_calls")
    if not calls:
        return None, meta
    meta["returned_call"] = True
    fn = calls[0].get("function", {}) or {}
    name = fn.get("name")
    raw_args = fn.get("arguments")
    if isinstance(raw_args, dict):  # tolerate pre-parsed dict
        args, meta["args_json_ok"] = raw_args, True
    else:
        try:
            args = json.loads(raw_args or "{}")
            meta["args_json_ok"] = True
        except json.JSONDecodeError:
            args, meta["args_json_ok"] = {}, False
    return {"name": name, "arguments": args}, meta


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


def _score_case(case: dict, got: dict | None) -> bool:
    gold = case["gold"]
    if gold is None:
        return got is None  # no-tool case: any call is wrong
    return bool(got and got.get("name") == gold["name"]
                and _args_match(gold["arguments"], got.get("arguments", {})))


# ── Runner ──────────────────────────────────────────────────────────────

def run(base_url="http://127.0.0.1:8890", model="aeon-27b-dflash",
        mode="native", out_path=None, *, thinking=False, temperature=0.0,
        max_tokens=1024, limit=None, label=None, client=None):
    client = client or AtlasClient(base_url=base_url, model=model)
    cases = CASES[:limit] if limit else CASES
    records = []
    correct = 0
    n_returned = 0          # native: tool_calls present in response
    n_unparseable = 0       # native: arguments failed json.loads
    by_cat: dict[str, list[bool]] = {}
    for c in cases:
        meta = {}
        if mode == "text":
            prompt = (c["query"] + "\n\nRespond ONLY with a JSON object "
                      '{"name": <fn>, "arguments": {...}} calling the correct '
                      "tool, or the literal string NONE if no tool applies. "
                      "Tools: " + json.dumps(c["tools"]))
            comp = client.chat([{"role": "user", "content": prompt}],
                               max_tokens=max_tokens, temperature=temperature,
                               enable_thinking=thinking)
            got = _extract_text(comp)
        else:
            comp = client.chat([{"role": "user", "content": c["query"]}],
                               tools=c["tools"], max_tokens=max_tokens,
                               temperature=temperature,
                               enable_thinking=thinking)
            got, meta = _extract_native(comp)
            n_returned += int(meta.get("returned_call", False))
            n_unparseable += int(meta.get("args_json_ok") is False)
        ok = _score_case(c, got)
        correct += int(ok)
        by_cat.setdefault(c["category"], []).append(ok)
        records.append({
            "id": c["id"], "category": c["category"], "gold": c["gold"],
            "got": got, "correct": ok, "meta": meta,
            "wall_s": round(getattr(comp, "wall_s", 0.0), 2),
            "content_tail": (comp.text or "")[-300:],
        })
        mark = "PASS" if ok else "FAIL"
        print(f"[bfcl] {c['id']:20s} {mark} got={json.dumps(got)[:120]}",
              flush=True)
    acc = correct / len(cases) if cases else 0.0
    result = {
        "label": label or mode, "model": model, "mode": mode,
        "thinking": thinking, "temperature": temperature,
        "accuracy": acc, "n_cases": len(cases), "n_correct": correct,
        "by_category": {k: {"correct": sum(v), "total": len(v),
                            "acc": sum(v) / len(v)}
                        for k, v in sorted(by_cat.items())},
        "native_format": {
            "tool_calls_returned": n_returned,
            "unparseable_arguments": n_unparseable,
        } if mode == "native" else None,
        "records": records,
    }
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    print(f"[bfcl] mode={mode} thinking={thinking} "
          f"accuracy={acc:.3f} ({correct}/{len(cases)}) "
          f"cats=" + json.dumps({k: f"{v['correct']}/{v['total']}"
                                 for k, v in result['by_category'].items()}))
    if mode == "native":
        print(f"[bfcl] format: tool_calls returned on {n_returned}/{len(cases)}"
              f" cases, unparseable arguments: {n_unparseable}")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="BFCL-style tool-calling eval")
    ap.add_argument("--base-url", default="http://127.0.0.1:8890")
    ap.add_argument("--model", default="aeon-27b-dflash")
    ap.add_argument("--mode", choices=["native", "text"], default="native")
    ap.add_argument("--thinking", action="store_true",
                    help="chat with enable_thinking=true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.base_url, a.model, a.mode, a.out, thinking=a.thinking,
        temperature=a.temperature, max_tokens=a.max_tokens,
        limit=a.limit, label=a.label)
