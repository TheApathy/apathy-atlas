"""Thin OpenAI-compatible client for the Atlas server.

Mirrors the request patterns in local/bench_wave2.py (completions) and
/tmp/vb35/think_bench.py (chat + enable_thinking). Uses stdlib urllib so the
harness has no hard dependency on `requests`.

Server: http://127.0.0.1:8890
  POST /v1/completions       -> {"choices":[{"text": ...}], "usage": {...}}
  POST /v1/chat/completions   -> {"choices":[{"message":{"content","reasoning_content"}}]}
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field

DEFAULT_BASE = "http://127.0.0.1:8890"
DEFAULT_MODEL = "aeon-27b-dflash"


@dataclass(frozen=True)
class Completion:
    text: str
    reasoning: str = ""
    completion_tokens: int = 0
    wall_s: float = 0.0
    raw: dict = field(default_factory=dict)


class AtlasClient:
    def __init__(self, base_url: str = DEFAULT_BASE, model: str = DEFAULT_MODEL,
                 timeout: float = 900.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            self.base_url + path, data=data,
            headers={"Content-Type": "application/json"},
        )
        import time
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            d = json.loads(r.read())
        d["_wall_s"] = time.time() - t0
        return d

    def complete(self, prompt: str, *, max_tokens: int = 1024,
                 temperature: float = 0.0, seed: int | None = None,
                 stop: list[str] | None = None) -> Completion:
        body = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if seed is not None:
            body["seed"] = seed
        if stop:
            body["stop"] = stop
        d = self._post("/v1/completions", body)
        usage = d.get("usage", {})
        return Completion(
            text=d["choices"][0].get("text", ""),
            completion_tokens=usage.get("completion_tokens", 0),
            wall_s=d.get("_wall_s", 0.0),
            raw=d,
        )

    def chat(self, messages: list[dict], *, max_tokens: int = 1024,
             temperature: float = 0.0, seed: int | None = None,
             enable_thinking: bool = False,
             thinking_budget: int | None = None) -> Completion:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if enable_thinking:
            body["enable_thinking"] = True
        if thinking_budget:
            body["thinking"] = {"budget_tokens": thinking_budget}
        if seed is not None:
            body["seed"] = seed
        d = self._post("/v1/chat/completions", body)
        msg = d["choices"][0].get("message", {})
        usage = d.get("usage", {})
        return Completion(
            text=msg.get("content") or "",
            reasoning=msg.get("reasoning_content") or "",
            completion_tokens=usage.get("completion_tokens", 0),
            wall_s=d.get("_wall_s", 0.0),
            raw=d,
        )
