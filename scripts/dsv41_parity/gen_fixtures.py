#!/usr/bin/env python3
"""Generate the DeepSeek-V4.1 parity fixtures from the PRODUCTION Python server code.

The oracle is the code production actually runs, imported read-only:
  * `server/app.py` (resolve_thinking, build_chat_prompt, _parse_tool_calls_tolerant)
    from the dsv41-prefill-work tree, and
  * the checkpoint's own `encoding/encoding.py` (render + strict completion parse), and
  * `server/tool_grammar.py::build_tool_grammar` (the EBNF text).

Output: crates/spark-server/tests/fixtures/dsv41/{render,parse,grammar}.json.
CPU only, no GPU, no weights beyond tokenizer.json.

    python3 scripts/dsv41_parity/gen_fixtures.py
"""

from __future__ import annotations

import copy
import json
import os
import sys

PY_TREE = os.environ.get("DSV41_PY_TREE", "/home/flocka/atlas/dsv41-prefill-work")
MODEL_DIR = os.environ.get(
    "DSV41_MODEL_DIR", "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "crates", "spark-server", "tests", "fixtures", "dsv41")

sys.path.insert(0, os.path.join(PY_TREE, "server"))
sys.path.insert(0, PY_TREE)
import app  # noqa: E402
import tool_grammar  # noqa: E402

TOK = app.Tok(MODEL_DIR)
ENC = app.load_encoding_module(MODEL_DIR)

PNG_1PX = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
           "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the weather for a city.\nUse \"metric\" units.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "days": {"type": "integer"},
        "units": {"type": "string", "enum": ["metric", "imperial"]},
    }, "required": ["city"]}}}
SEARCH = {"type": "function", "function": {
    "name": "web_search", "description": "Search the web — 検索",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string"}, "max_results": {"type": "number"},
        "filters": {"type": "object"}, "tags": {"type": "array", "items": {"type": "string"}},
        "ids": {"type": "array", "items": {"type": "integer"}}, "flag": {"type": "boolean"},
        "anything": {},
    }, "required": ["query", "flag"]}}}
NOPROPS = {"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}
EMPTYPROPS = {"type": "function", "function": {
    "name": "now", "description": "", "parameters": {"type": "object", "properties": {}}}}
NS_TOOL = {"type": "function", "namespace": "fs", "function": {
    "name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}
NS_QUAL = {"type": "function", "function": {
    "name": "git::status", "parameters": {"type": "object", "properties": {}}}}
BAD_NAME = {"type": "function", "function": {
    "name": "bad\"name", "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}}}

U = lambda c: {"role": "user", "content": c}  # noqa: E731
S = lambda c: {"role": "system", "content": c}  # noqa: E731


def tc(i, name, args):
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": args}}


TOOL_CONV = [
    S("You are a helpful agent."),
    U("Weather in Paris and Tokyo? Also search news."),
    {"role": "assistant", "content": "", "reasoning_content": "Need three calls.",
     "tool_calls": [tc("a", "get_weather", '{"city": "Paris", "days": 2}'),
                    tc("b", "get_weather", {"city": "東京", "units": "metric"}),
                    tc("c", "web_search", json.dumps(json.dumps(
                        {"query": "news", "flag": True, "filters": {"lang": ["en", "fr"]}, "max_results": 3.5})))]},
    # results arrive out of call order: encoding sorts them back
    {"role": "tool", "tool_call_id": "call_c", "content": "headline 1\nheadline 2"},
    {"role": "tool", "tool_call_id": "call_a", "content": "18C"},
    {"role": "tool", "tool_call_id": "call_b", "content": [{"type": "text", "text": "22C"},
                                                     {"type": "text", "text": "humid"}]},
]

# (name, body) -- body is an OpenAI chat request. Thinking/effort resolve through app.py.
RENDER_CASES = [
    ("plain_chat", {"messages": [U("Hello!")]}),
    ("plain_think_default", {"messages": [U("Hello!")], "enable_thinking": True}),
    ("system_think_int", {"messages": [S("Be terse."), U("2+2?")], "reasoning_effort": 33}),
    ("effort_low_is_off", {"messages": [U("hi")], "reasoning_effort": "low"}),
    ("effort_medium", {"messages": [U("hi")], "reasoning_effort": "medium"}),
    ("effort_xhigh", {"messages": [U("hi")], "reasoning": {"effort": "xhigh"}}),
    ("effort_max_ctk", {"messages": [U("hi")], "chat_template_kwargs": {"reasoning_effort": "max"}}),
    ("effort_digit_string", {"messages": [U("hi")], "reasoning_effort": " 7 "}),
    ("effort_none", {"messages": [U("hi")], "reasoning_effort": "none"}),
    ("ctk_thinking_beats_low", {"messages": [U("hi")], "reasoning_effort": "low",
                                "chat_template_kwargs": {"thinking": True}}),
    ("enable_false_beats_high", {"messages": [U("hi")], "reasoning_effort": "high", "enable_thinking": False}),
    ("ctk_enable_thinking", {"messages": [U("hi")], "chat_template_kwargs": {"enable_thinking": True}}),
    ("multiturn_drop_thinking", {"enable_thinking": True, "messages": [
        S("sys"), U("q1"), {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        U("q2"), {"role": "assistant", "content": "a2", "reasoning_content": "r2"}, U("q3")]}),
    ("multiturn_chat", {"messages": [
        U("q1"), {"role": "assistant", "content": "a1", "reasoning_content": "r1"}, U("q2")]}),
    ("tools_chat", {"messages": copy.deepcopy(TOOL_CONV), "tools": [WEATHER, SEARCH]}),
    ("tools_think", {"messages": copy.deepcopy(TOOL_CONV), "tools": [WEATHER, SEARCH], "reasoning_effort": 90}),
    ("tools_no_system", {"messages": [U("ping it")], "tools": [NOPROPS, EMPTYPROPS]}),
    ("tool_choice_none", {"messages": [U("hi")], "tools": [WEATHER], "tool_choice": "none"}),
    ("namespaces", {"messages": [
        U("read"), {"role": "assistant", "content": None,
                    "tool_calls": [tc("x", "fs::read_file", {"path": "/a b"}), tc("y", "git::status", "{}")]},
        {"role": "tool", "tool_call_id": "x", "content": "data"},
        {"role": "tool", "tool_call_id": "y", "content": "clean"}], "tools": [NS_TOOL, NS_QUAL]}),
    ("args_non_dict", {"messages": [
        U("x"), {"role": "assistant", "content": "ok", "tool_calls": [tc("z", "ping", "[1, 2]"),
                                                                       tc("w", "ping", "not json")]},
        {"role": "tool", "tool_call_id": "z", "content": "r"}], "tools": [NOPROPS]}),
    ("mid_system", {"enable_thinking": True, "messages": [
        U("q1"), {"role": "assistant", "content": "a1", "reasoning_content": "r1"}, S("New rule: be brief.")]}),
    ("mid_system_then_user", {"enable_thinking": True, "messages": [
        S("s0"), U("q1"), {"role": "assistant", "content": "a1"}, S("s1"), U("q2")]}),
    ("response_format", {"messages": [U("Give JSON")], "response_format": {
        "type": "json_schema", "json_schema": {"name": "x", "schema": {
            "type": "object", "properties": {"a": {"type": "integer"}, "é": {"type": "string"}}}}}}),
    ("response_format_json_object_ignored", {"messages": [U("Give JSON")],
                                             "response_format": {"type": "json_object"}}),
    ("content_list", {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]}]}),
    ("image_blocks", {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "What is this?"}, {"type": "image_url", "image_url": {"url": PNG_1PX}},
        {"type": "text", "text": "and"}, {"type": "image_url", "image_url": PNG_1PX}]}]}),
    ("image_in_tool_result", {"messages": [
        U("look"), {"role": "assistant", "content": "", "tool_calls": [tc("i", "ping", {})]},
        {"role": "tool", "tool_call_id": "i", "content": [
            {"type": "text", "text": "screenshot:"}, {"type": "image_url", "image_url": {"url": PNG_1PX}}]}],
        "tools": [NOPROPS]}),
    ("unicode", {"messages": [S("Réponds en français 🙂"), U("日本語\ttab\r\nCRLF \u0000 nul \"q\" \\ back")]}),
    ("latest_reminder", {"messages": [U("q"), {"role": "latest_reminder", "content": "today is Monday"}]}),
    ("assistant_last", {"messages": [U("q"), {"role": "assistant", "content": "partial"}]}),
    ("assistant_last_think", {"enable_thinking": True,
                              "messages": [U("q"), {"role": "assistant", "content": "a", "reasoning_content": "r"}]}),
    ("user_user_merge", {"messages": [U("first"), U("second")]}),
    ("tool_after_user", {"messages": [U("q"), {"role": "tool", "tool_call_id": "q", "content": "orphan"}]}),
    # errors (the Rust side must refuse too; messages need not match)
    ("err_empty", {"messages": []}),
    ("err_effort_zero", {"messages": [U("hi")], "reasoning_effort": 0}),
    ("err_effort_unknown", {"messages": [U("hi")], "reasoning_effort": "extreme"}),
    ("err_effort_bool", {"messages": [U("hi")], "reasoning_effort": True}),
    ("err_thinking_not_bool", {"messages": [U("hi")], "enable_thinking": "yes"}),
    ("err_placeholder_in_text", {"messages": [U("see <｜deepseek_image｜> here")]}),
    ("err_unknown_role", {"messages": [{"role": "developer", "content": "x"}]}),
    ("err_tools_not_list", {"messages": [U("hi")], "tools": {"a": 1}}),
    ("err_no_role", {"messages": [{"content": "x"}]}),
]


def render_case(name, body):
    try:
        thinking, effort = app.resolve_thinking(copy.deepcopy(body), False, 75)
        # build_chat_prompt decodes images with PIL (engine.vision); keep that, it is production.
        prompt, ids, tools, images = app.build_chat_prompt(copy.deepcopy(body), ENC, TOK, thinking, effort)
    except app.APIError as e:
        return {"name": name, "body": body, "error": {"status": e.status, "message": e.message}}
    return {"name": name, "body": body, "thinking": thinking, "effort": effort,
            "prompt": prompt, "ids": ids, "n_images": len(images),
            "grammar_tools": tools}


# ---------------------------------------------------------------- completion parsing
D = "｜DSML｜"


def block(*invokes):
    return f"\n\n<{D} calls>\n" + "\n".join(invokes) + f"\n</{D} calls>"


def inv(name, *params):
    return f'<{D} invoke name="{name}">\n' + "".join(p + "\n" for p in params) + f"</{D} invoke>"


def par(k, v, s="true"):
    return f'<{D} parameter name="{k}" string="{s}">{v}</{D} parameter>'


PARSE_CASES = [
    ("content_only", "chat", "Hello there."),
    ("think_content", "thinking", "I think.</think>Answer."),
    ("one_call", "chat", "Sure." + block(inv("get_weather", par("city", "Paris"), par("days", "2", "false")))),
    ("two_calls_think", "thinking", "plan</think>" + block(
        inv("get_weather", par("city", "東京")),
        inv("web_search", par("query", "a <b> & c\nline2"), par("filters", '{"x": [1, 2]}', "false"),
            par("flag", "true", "false")))),
    ("no_params", "chat", block(inv("now"))),
    ("namespaced", "chat", block(inv("fs::read_file", par("path", "/tmp/x")))),
    ("empty_content_call", "chat", block(inv("ping", par("a", "")))),
    # drift the tolerant parser exists for
    ("value_in_attr", "chat", "Let me search." + f'\n\n<{D} calls>\n<{D} invoke name="web_search">\n'
                                                 f'<{D} parameter name="query" string="rust async book">\n'
                                                 f'</{D} invoke>\n</{D} calls>'),
    ("prose_after_block", "chat", "x" + block(inv("ping", par("a", "1"))) + "\nDone, anything else?"),
    ("dup_param", "chat", block(inv("ping", par("a", "1"), par("a", "2")))),
    ("unterminated", "chat", "x" + f'\n\n<{D} calls>\n<{D} invoke name="ping">\n' + par("a", "1")),
    ("bad_json_false", "chat", block(inv("ping", par("n", "{oops", "false")))),
    ("missing_newline", "chat", f'\n\n<{D} calls><{D} invoke name="ping">\n</{D} invoke>\n</{D} calls>'),
    ("think_missing_end", "thinking", "never closes"),
    ("special_in_content", "chat", "a <think> b"),
    ("v4_spelling_is_not_v41", "chat", f'\n\n<{D}tool_calls>\n<{D}invoke name="ping">\n</{D}invoke>\n</{D}tool_calls>'),
]


class _R:  # the OutputRouter surface _parse_tool_calls reads
    pass


def parse_case(name, mode, text):
    thinking = mode == "thinking"
    out = {"name": name, "mode": mode, "text": text}
    full = text if text.endswith(ENC.eos_token) else text + ENC.eos_token
    try:
        p = ENC.parse_message_from_completion_text(full, thinking_mode=mode)
        out["strict"] = {"content": p["content"], "reasoning_content": p["reasoning_content"],
                         "tool_calls": [{"name": t["function"]["name"], "arguments": t["function"]["arguments"],
                                         **({"namespace": t["namespace"]} if t.get("namespace") else {})}
                                        for t in p["tool_calls"]]}
    except Exception as e:  # noqa: BLE001
        out["strict_error"] = f"{type(e).__name__}: {e}"
    out["tolerant"] = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                       for c in app.State._parse_tool_calls_tolerant(full)]
    # the full server pipeline: OutputRouter split + _parse_tool_calls (strict, tolerant, give-up)
    router = app.OutputRouter(thinking, [], True)
    router.feed(text)
    router.finish()
    st = _R()
    st.enc = ENC
    st._parse_tool_calls_tolerant = app.State._parse_tool_calls_tolerant
    calls = app.State._parse_tool_calls(st, router, thinking)
    out["server"] = {"reasoning": router.reasoning, "content": router.content,
                     "tool_calls": [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                                    for c in calls]}
    return out


GRAMMAR_CASES = [
    ("weather", [WEATHER]),
    ("weather_search", [WEATHER, SEARCH]),
    ("noprops_emptyprops", [NOPROPS, EMPTYPROPS]),
    ("namespaced_flat", ENC.tools_from_openai_format(copy.deepcopy([NS_TOOL, NS_QUAL]))),
    ("bad_name_dropped", [BAD_NAME, WEATHER]),
    ("all_bad", [BAD_NAME]),
    ("enum_untyped", [{"type": "function", "function": {"name": "mode", "parameters": {
        "type": "object", "properties": {"m": {"enum": ["a\"b", "c"]}, "n": {"enum": ["x", 1]},
                                         "arr": {"type": "array", "items": {"type": "object"}},
                                         "t": {"type": " integer "}}, "required": ["m"]}}}]),
]


# Token streams walked through the compiled grammar (Python xgrammar, as the production gate
# compiles it). At every step: how many tokens the mask allows, a digest of the allowed set,
# and whether the next token was accepted. The Rust matcher must reproduce all three.
MASK_STREAMS = {
    "weather": [
        block(inv("get_weather", par("city", "Paris"), par("days", "3", "false"))) + ENC.eos_token,
        block(inv("get_weather", par("city", "a ｜ b"))),          # U+FF5C inside a value: refused
        block(inv("get_weather", par("days", "3", "false"))),      # required `city` missing: refused
        block(inv("get_weather", par("city", "x"), par("units", "kelvin"))),  # enum violation
        # U+FF0C (fullwidth comma) is legal DSML value text, but xgrammar 0.1.32's UTF-8 range
        # split loses U+F000..U+FF3F from [\u0000-\uFF5B]: Python REJECTS it. Rust must match.
        block(inv("get_weather", par("city", "東京，大阪"))),
    ],
    "weather_search": [
        block(inv("web_search", par("query", "q <b>\n{\"x\": 1}"), par("tags", '["a", "b"]', "false"),
                  par("flag", "false", "false"))) + ENC.eos_token,
        block(inv("web_search", par("query", "q"), par("flag", "maybe", "false"))),  # bad bool
    ],
    "noprops_emptyprops": [
        block(inv("ping", par("anything", "goes"), par("n", "1", "false")), inv("now")) + ENC.eos_token,
    ],
}


def mask_trace(ebnf, text):
    import torch  # noqa: F401 - xgrammar's bitmask helpers need it
    import xgrammar as xgr
    f = _FACTORY[0]
    cg = f.compile(ebnf)
    m = xgr.GrammarMatcher(cg, override_stop_tokens=[1], max_rollback_tokens=16)
    ids = TOK.encode(text)
    bm = xgr.allocate_token_bitmask(1, f.vocab_size)
    steps = []
    for t in ids:
        m.fill_next_token_bitmask(bm, 0)
        words = bm[0].tolist()
        allowed = [i for i in range(f.vocab_size) if (words[i >> 5] >> (i & 31)) & 1]
        ok = bool(m.accept_token(t))
        steps.append({"n_allowed": len(allowed),
                      "digest": fnv1a64(",".join(map(str, allowed)).encode()),
                      "token": t, "accepted": ok})
        if not ok:
            break
    return {"text": text, "ids": ids, "steps": steps, "terminated": bool(m.is_terminated())}


_FACTORY = []


def fnv1a64(data: bytes) -> str:
    h = 0xcbf29ce484222325
    for b in data:
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def load_penalties_class():
    """`Penalties` from engine/v41_engine.py, extracted by AST so the engine's
    GPU imports never run. The class needs only `os` and `torch`."""
    import ast
    import torch
    src = open(os.path.join(PY_TREE, "engine", "v41_engine.py")).read()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "Penalties")
    ns = {"os": os, "torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "v41_engine.py", "exec"), ns)
    return ns["Penalties"]


def repetition_cases():
    import random
    Penalties = load_penalties_class()
    rng = random.Random(41)
    hists = [[], [5], [7, 7, 7], [7, 7, 7, 7], [1, 2, 1, 2, 1, 2, 1, 2], [9, 1, 2, 1, 2, 1, 2, 1, 2],
             [3, 1, 2, 3, 1, 2, 3, 1, 2], [3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2], list(range(40)) * 4,
             list(range(17)) * 4, list(range(16)) * 4, [4, 4, 4, 5, 4, 4, 4, 5, 4, 4, 4, 5, 4, 4, 4, 5]]
    for _ in range(60):
        base = [rng.randrange(6) for _ in range(rng.randrange(1, 6))]
        h = [rng.randrange(6) for _ in range(rng.randrange(0, 8))] + base * rng.randrange(1, 6)
        hists.append(h)
    out = []
    for h in hists:
        row = {"history": h}
        for n in (0, 2, 3, 4):
            p = Penalties(no_repeat_ngram=n, enabled=True)
            row[f"ngram{n}"] = sorted(p._banned_ngram_tokens(h))
        row["cycle"] = Penalties(enabled=True)._cycle_token(h)
        out.append(row)
    return out


def _png(img, fmt="PNG", **kw):
    import base64
    import io
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def _pattern(w, h, mode="RGB"):
    import numpy as np
    from PIL import Image
    y, x = np.mgrid[0:h, 0:w]
    arr = np.stack([(x * 7 + y * 3) % 256, (x * 11 ^ y * 5) % 256, ((x // 9 + y // 7) * 37) % 256], -1)
    return Image.fromarray(arr.astype("uint8"), "RGB").convert(mode)


def vision_cases():
    import numpy as np
    from PIL import Image
    sys.path.insert(0, os.path.join(PY_TREE, "engine"))
    from engine import vision as V
    cfg = V.VisionConfig.from_mapping(json.load(open(os.path.join(MODEL_DIR, "config.json"))))
    exif_img = _pattern(90, 60)
    exif = exif_img.getexif()
    exif[0x0112] = 6  # orientation: rotate 90 CW on display
    images = [
        ("gradient_640x480", _png(_pattern(640, 480))),
        ("small_upscaled_200x100", _png(_pattern(200, 100))),
        ("large_solver_3000x2000", _png(_pattern(3000, 2000))),
        ("wide_5000x300", _png(_pattern(5000, 300))),
        ("tall_120x2500", _png(_pattern(120, 2500))),
        ("tiny_1x1", _png(_pattern(1, 1))),
        ("jpeg_q90", _png(_pattern(333, 222), "JPEG", quality=90)),
        ("jpeg_exif_orient6", _png(exif_img, "JPEG", quality=95, exif=exif)),
        ("rgba_png", _png(_pattern(150, 170, "RGBA"))),
        ("gray_png", _png(_pattern(150, 170, "L"))),
        ("palette_png", _png(_pattern(160, 90).convert("P"))),
    ]
    out = []
    for name, uri in images:
        rec = {"name": name, "uri": uri}
        try:
            img = V.decode_image_record({"type": "image", "url": uri})
            prep = V.preprocess_image(img, cfg)
        except ValueError as e:
            rec["error"] = str(e)
            out.append(rec)
            continue
        bits = prep.patches.contiguous().view(torch_int16()).numpy().astype("<u2").tobytes()
        rec.update(decoded_size=list(img.size), vit_h=prep.vit_h, vit_w=prep.vit_w, llm_h=prep.llm_h,
                   llm_w=prep.llm_w, types=prep.types.tolist(), patches_shape=list(prep.patches.shape),
                   patches_fnv=fnv1a64(bits), patches_head=[float(v) for v in prep.patches.flatten()[:48]])
        out.append(rec)
    for name, uri in [("remote_url", "https://example.com/a.png"), ("gif", _png(_pattern(8, 8), "GIF").replace("image/jpeg", "image/gif")),
                      ("too_many_pixels", _png(_pattern(8000, 5001))), ("not_base64", "data:image/png;base64,@@@@")]:
        try:
            V.decode_image_record({"type": "image", "url": uri})
            out.append({"name": name, "uri": uri, "error": None})
        except ValueError as e:
            # the 8000x5001 payload is ~MBs; the Rust test builds its own oversized image
            out.append({"name": name, "uri": uri if len(uri) < 4096 else None, "error": str(e)})
    types = [V.image_token_types(2, 3), V.image_token_types(1, 1)]
    expand = []
    for ids in ([5, 129264, 7, 129264], [129264, 129264], [5, 6, 129264, 8, 9, 10, 129264, 11]):
        o, spans = V.expand_image_placeholders(ids, types)
        expand.append({"ids": ids, "out": o, "spans": [[p.start, p.pad, p.types.numel()] for p in spans]})
    dead_ids = [5, 129265, 129264, 129264, 7, 8, 9, 10, 129264, 11]
    import torch
    dead = V.engram_dead_heads(torch.tensor(dead_ids)).int().tolist()
    return {"images": out, "types": [t.tolist() for t in types], "expand": expand,
            "dead_heads": {"ids": dead_ids, "dead": dead}}


def torch_int16():
    import torch
    return torch.int16


def main():
    os.makedirs(OUT, exist_ok=True)
    render = [render_case(n, b) for n, b in RENDER_CASES]
    parse = [parse_case(*c) for c in PARSE_CASES]
    _FACTORY.append(tool_grammar.ToolGrammarFactory(TOK, eos_id=1))
    grammar = []
    for n, t in GRAMMAR_CASES:
        ebnf = tool_grammar.build_tool_grammar(t)
        grammar.append({"name": n, "tools": t, "ebnf": ebnf,
                        "streams": [mask_trace(ebnf, x) for x in MASK_STREAMS.get(n, [])]})
    with open(os.path.join(OUT, "vision.json"), "w") as f:
        json.dump(vision_cases(), f)
    with open(os.path.join(OUT, "repetition.json"), "w") as f:
        json.dump({"cases": repetition_cases()}, f)
    meta = {"generator": "scripts/dsv41_parity/gen_fixtures.py", "py_tree": PY_TREE, "model_dir": MODEL_DIR}
    for fname, data in (("render.json", render), ("parse.json", parse), ("grammar.json", grammar)):
        with open(os.path.join(OUT, fname), "w") as f:
            json.dump({"meta": meta, "cases": data}, f, ensure_ascii=False, indent=1)
    n_err = sum("error" in r for r in render)
    print(f"render {len(render)} ({n_err} errors), parse {len(parse)}, grammar {len(grammar)} -> {OUT}")
    for r in render:
        print(f"  {r['name']:<34} {'ERR ' + str(r['error']['status']) if 'error' in r else len(r['ids'])}")


if __name__ == "__main__":
    main()
