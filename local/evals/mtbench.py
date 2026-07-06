"""MT-Bench-style quality eval (SECONDARY, needs a judge). SCAFFOLD.

This is intentionally minimal and pluggable: it generates answers from the
Atlas server for a small set of open-ended prompts, then asks a JUDGE to rate
each on a 1-10 scale. The judge is pluggable:

  - "server": reuse the same Atlas server as an LLM judge (single-model self-
    or peer-grading). Cheapest, no external dep.
  - "none":   skip judging; just dump answers for manual review.
  - (future)  an external API judge — add a callable to JUDGES.

NOT the ship gate. Coding pass@1 + ABBA is the gate. This gives a soft,
subjective read on prose/reasoning quality drift for a lever. Do not block on it.

Judge bias caveat: an LLM grading its own output is optimistic and noisy; treat
scores as directional only. For a real MT-Bench number use a strong external
judge (documented in the recipe), not the model under test.
"""

from __future__ import annotations

import json
import re

from client import AtlasClient

# Tiny bundled question set (single-turn subset, MT-Bench-flavored categories).
QUESTIONS = [
    {"id": "writing_1", "category": "writing",
     "q": "Write a concise, vivid 4-sentence description of a thunderstorm rolling over a coastal town."},
    {"id": "reasoning_1", "category": "reasoning",
     "q": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Explain."},
    {"id": "math_1", "category": "math",
     "q": "What is the sum of the first 50 positive even integers? Show the reasoning."},
    {"id": "extraction_1", "category": "extraction",
     "q": "From 'The meeting is on 2026-07-06 at 14:30 in room B12 with Dana and Omar', extract date, time, room, and attendees as JSON."},
]

JUDGE_TMPL = (
    "You are an impartial evaluator. Rate the assistant's answer to the user "
    "question on a scale of 1 to 10 for overall quality (accuracy, helpfulness, "
    "clarity). Respond with ONLY the integer rating on the first line.\n\n"
    "[Question]\n{q}\n\n[Answer]\n{a}\n\n[Rating]"
)


def generate_answers(client: AtlasClient, *, max_tokens=512, temperature=0.0):
    answers = []
    for item in QUESTIONS:
        comp = client.chat(
            [{"role": "user", "content": item["q"]}],
            max_tokens=max_tokens, temperature=temperature,
        )
        answers.append({**item, "answer": comp.text})
    return answers


def judge_server(client: AtlasClient, answers, *, max_tokens=64):
    scored = []
    for a in answers:
        comp = client.chat(
            [{"role": "user",
              "content": JUDGE_TMPL.format(q=a["q"], a=a["answer"])}],
            max_tokens=max_tokens, temperature=0.0,
        )
        scored.append({**a, "score": _parse_score(comp.text)})
    return scored


def _parse_score(text: str):
    m = re.search(r"\b(10|[1-9])\b", text)
    return int(m.group(1)) if m else None


JUDGES = {"server": judge_server, "none": None}


def run(base_url="http://127.0.0.1:8890", model="aeon-27b-dflash",
        judge="server", out_path=None):
    client = AtlasClient(base_url=base_url, model=model)
    answers = generate_answers(client)
    if JUDGES.get(judge):
        scored = JUDGES[judge](client, answers)
        valid = [s["score"] for s in scored if s["score"] is not None]
        mean = sum(valid) / len(valid) if valid else None
    else:
        scored, mean = answers, None
    result = {"model": model, "judge": judge, "mean_score": mean,
              "records": scored}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    print(f"[mtbench] judge={judge} mean_score={mean}")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="MT-Bench-style quality (scaffold)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8890")
    ap.add_argument("--model", default="aeon-27b-dflash")
    ap.add_argument("--judge", choices=list(JUDGES), default="server")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.base_url, a.model, a.judge, a.out)
