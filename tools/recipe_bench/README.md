# recipe_bench

Measures a built-in recipe (`crates/spark-server/src/recipe/builtin/*.yaml`)
the way the TUI actually launches it, not a hand-transcribed guess at its
flags. Every "the recipe is slow" finding on 2026-09-23 turned out to be a
hand-transcription bug (missing env vars, wrong port, a shell/python
variable collision), not a real regression — this tool exists so that
doesn't happen again.

## Files

- `generate.py` — reads a recipe YAML, emits `env` (KEY=VAL lines) or `argv`
  (one token per line) exactly as `main_modules::model_profile` /
  `recipe::schema` would build them. `generate_test.py` covers it,
  including the presence-flag hyphenation bug (`qwen4_qsa` vs `qwen4-qsa`)
  that a hand-written version of this generator shipped with initially.
- `run.sh` — GB10-lock-safe harness: queues itself, takes the lock
  exclusively, preflights, launches the server via `generate.py`'s output
  (never hand-typed), drops the first prefill/decode rep and reports the
  warm median, and writes one `result.json` per model. Guaranteed cleanup
  via `trap cleanup EXIT INT TERM ERR` — kills the server's whole process
  group (not a bare PID) and waits for it to actually exit before logging
  the queue's `window END` line, and the server's own launch closes fd 9
  (`9>&-`) so it can never hold the flock open even if the trap somehow
  doesn't run.
- `prompts/coding.json`, `prompts/chat.json` — the two decode probes (256
  tokens each, temperature 0).

## Usage

```
tools/recipe_bench/run.sh <lane> <label> <recipe.yaml> <model_dir> <model_name> <bin> \
  <prefill_req_2048.json> <prefill_req_8192.json> [port=8897]
```

The two prefill request files are pre-tokenized `/v1/completions` bodies
(`prompt_token_ids`, `max_tokens: 1`) — model/tokenizer-specific, so this
tool doesn't generate them. Point it at existing ones under
`allmodels-bench/req-*.json` or build fresh ones with the target model's
tokenizer.

Output: `tools/recipe_bench/results/<label>/result.json` — prefill tok/s at
2048 and 8192 (warm median of reps 2-4), decode tok/s + output hash on the
coding and chat prompts, and `env_proof_match` (whether
`/proc/<pid>/environ` on the live server exactly matched what `generate.py`
produced — `true` unless something outside this tool's control is
overriding the launch).

## Tests

```
cd tools/recipe_bench && python3 -m pytest generate_test.py -v
```
CPU-only, no GPU, no server. Run these before trusting a new recipe or a
change to the generator's renaming/presence-flag logic.
