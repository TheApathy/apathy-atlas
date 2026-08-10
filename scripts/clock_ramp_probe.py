#!/usr/bin/env python3
"""Cold-vs-hot prefill TTFT with SM-clock sampling.

Sequence: idle settle → COLD prefill (clocks sampled) → heavy 300-token
generation (heats clocks) → HOT prefill immediately (clocks sampled).
The delta isolates the SM clock-ramp share of TTFT.

Usage: python3 scripts/clock_ramp_probe.py [port]
"""
import json
import subprocess
import sys
import threading
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8977
BASE = f"http://127.0.0.1:{PORT}"

words = " ".join(
    f"Fact {i}: division {i} reported revenue of {1000 + i * 7} units at margin "
    f"{10 + i % 20} percent." for i in range(120)
)
PREFILL_PROMPT = "Summarize in one sentence: " + words


def model_id() -> str:
    with urllib.request.urlopen(f"{BASE}/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def chat(model, content, max_tokens, stream=True):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    t0 = time.time()
    first = None
    for line in urllib.request.urlopen(req, timeout=600):
        line = line.decode().strip()
        if line.startswith("data:") and line != "data: [DONE]":
            d = json.loads(line[5:])
            if first is None and d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
                first = time.time()
    return (first or time.time()) - t0


class ClockSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2,
                ).stdout.strip()
                self.samples.append(int(out))
            except Exception:
                pass
            time.sleep(0.15)


def measured(label, model):
    s = ClockSampler()
    s.start()
    ttft = chat(model, PREFILL_PROMPT, 4)
    s.stop_flag = True
    s.join(timeout=1)
    clk = s.samples or [0]
    print(
        f"{label}: TTFT {ttft:.2f}s -> {2410 / ttft:.0f} tok/s | "
        f"SM clocks min {min(clk)} / median {sorted(clk)[len(clk) // 2]} / max {max(clk)} MHz "
        f"({len(clk)} samples)"
    )
    return ttft


def main():
    model = model_id()
    chat(model, "hi", 4)          # session warmup (weights touched, clocks idle again after)
    time.sleep(20)                # let clocks fall back to idle
    measured("COLD (after 20s idle)", model)
    print("heating: 300-token generation ...")
    chat(model, "Write a detailed essay about the history of navigation.", 300)
    measured("HOT  (immediately after)", model)


if __name__ == "__main__":
    main()
