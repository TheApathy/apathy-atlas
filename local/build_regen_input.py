#!/usr/bin/env python3
"""Assemble the SpecForge regeneration input JSONL for the AEON drafter
retrain (task #10).

Mix (36K conversations, single-turn, user->assistant):
  - 12K creative-writing prompts (euclaise/writingprompts) — the cat-story
    workload class where DFlash acceptance is weakest (1-3/16)
  - 12K coding tasks (sahil2801/CodeAlpaca-20k) — mid-acceptance class
  -  6K general chat (HuggingFaceH4/ultrachat_200k train_sft subset)
  -  6K synthetic structured prompts (counting, lists, fizzbuzz variants)
    — keeps the high-acceptance regime represented so the finetune does
    not regress it

Output lines: {"conversations": [{"role": "user", "content": ...},
                                 {"role": "assistant", "content": ""}]}
The assistant turn is a placeholder; regenerate_train_data.py replaces it
with the AEON target's own sampled response (temperature 0.8).
"""
from __future__ import annotations

import json
import random
import sys

from datasets import load_dataset

OUT = sys.argv[1] if len(sys.argv) > 1 else "/path/to/dflash-retrain/regen_input.jsonl"
random.seed(42)

rows: list[dict] = []


def add(prompt: str) -> None:
    prompt = prompt.strip()
    if not (20 <= len(prompt) <= 4000):
        return
    rows.append(
        {
            "conversations": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ""},
            ]
        }
    )


print("loading writingprompts...", flush=True)
wp = load_dataset("euclaise/writingprompts", split="train", streaming=True)
n = 0
for ex in wp:
    add("Write a short story based on this prompt: " + ex["prompt"])
    n += 1
    if n >= 12000:
        break

print("loading CodeAlpaca...", flush=True)
ca = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
picks = random.sample(range(len(ca)), min(12000, len(ca)))
for i in picks:
    ex = ca[i]
    prompt = ex["instruction"]
    if ex.get("input"):
        prompt += "\n\n" + ex["input"]
    add(prompt)

print("loading ultrachat subset...", flush=True)
uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
n = 0
for ex in uc:
    msgs = ex.get("messages", [])
    if msgs and msgs[0]["role"] == "user":
        add(msgs[0]["content"])
        n += 1
    if n >= 6000:
        break

print("synthesizing structured prompts...", flush=True)
templates = [
    "Count from {a} to {b}, comma separated, numbers only.",
    "List the even numbers from {a} to {b}.",
    "Print FizzBuzz from 1 to {b}.",
    "Write out the {n} times table up to 20 entries.",
    "List the squares of the integers from {a} to {b}.",
    "Spell out the numbers from {a} to {b} in English words, one per line.",
]
for _ in range(6000):
    t = random.choice(templates)
    a = random.randint(1, 400)
    add(t.format(a=a, b=a + random.randint(20, 120), n=random.randint(2, 19)))

random.shuffle(rows)
with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} conversations to {OUT}")
