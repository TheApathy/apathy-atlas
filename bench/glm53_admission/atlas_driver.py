#!/usr/bin/env python3
"""Serve GLM-5.3 through its built-in recipe and capture logits for the admission gate.

Run inside gpu_window.sh (which owns the lock, preflight and watchdog):

  atlas_driver.py --bin <spark> --out <dir> --prefill prompts.json [--env K=V ...]
  atlas_driver.py --bin <spark> --out <dir> --decode prompts.json \
      --decode-prefix 512 --decode-tokens 65 [--env K=V ...]

The server's argv/env come from tools/recipe_bench/generate.py applied to the
recipe YAML, never typed by hand; --env only ADDS the logits dump and arm
switches. The live server's /proc environ is checked against that set. Writes
<out>/logits/ (the raw dumps), <out>/run.json and, for decode,
<out>/sequences.json (prompt + generated ids, for the reference run).
"""
import argparse
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

import numpy as np

WORKTREE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GENERATE = os.path.join(WORKTREE, "tools/recipe_bench/generate.py")
RECIPE = os.path.join(WORKTREE, "crates/spark-server/src/recipe/builtin/glm-5.3-flash-exl3-local.yaml")
MODEL_DIR = "/home/flocka/models/GLM-5.3-Flash-exl3-2.05bpw"
MODEL_NAME = "GLM-5.3-Flash-exl3-2.05bpw"
V = 154_880
ESCAPE = "ATLAS_GLM53_UNVALIDATED_BRINGUP"


def generate(kind, *extra):
    out = subprocess.run([sys.executable, GENERATE, RECIPE, MODEL_DIR, MODEL_NAME, kind, *extra],
                         check=True, capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line]


def post(port, body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                 json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def dumps(d):
    out = []
    for f in glob.glob(os.path.join(d, "logits-*.bf16")):
        m = re.fullmatch(r"logits-(\d+)(?:-r(\d+))?\.bf16", os.path.basename(f))
        out.append((int(m.group(1)), int(m.group(2) or 1), f))
    return sorted(out)


def last_row_argmax(path, rows):
    """The server's greedy pick on this path: the device argmax keeps the FIRST
    exact maximum (measured on a bf16 tie at decode step 9 of code_rust; the
    host Iterator::max_by path keeps the last). The text check below catches a
    wrong rule."""
    u = np.fromfile(path, dtype=np.uint16)[(rows - 1) * V:rows * V]
    x = (u.astype(np.uint32) << 16).view(np.float32)
    return int(np.argmax(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8894)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--argv-override", action="append", default=[],
                    help="recipe default override passed to generate.py, e.g. max_model_len=9216")
    ap.add_argument("--prefill")
    ap.add_argument("--decode")
    ap.add_argument("--decode-prefix", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=65)
    args = ap.parse_args()
    if bool(args.prefill) == bool(args.decode):
        sys.exit("exactly one of --prefill / --decode")

    os.makedirs(args.out, exist_ok=False)
    logits_dir = os.path.join(args.out, "logits")
    env_lines = generate("env")
    argv = generate("argv", f"port={args.port}", *args.argv_override)
    extra = list(args.env) + [f"ATLAS_GLM53_LOGITS_DUMP={logits_dir}", "ATLAS_GLM53_LOGITS_DUMP_ALL=1",
                              "ATLAS_GLM53_EXL3_LAST_ROW_HEAD=0"]
    env = {"HOME": os.environ["HOME"], "LANG": "C.UTF-8",
           "PATH": "/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:/usr/bin:/bin",
           "LD_LIBRARY_PATH": "/usr/local/cuda-13.0/lib64:/usr/local/cuda/lib64", "RUST_LOG": "info"}
    for line in env_lines + extra:
        k, v = line.split("=", 1)
        env[k] = v
    escape = env.get(ESCAPE) == "1"
    log_path = os.path.join(args.out, "server.log")
    with open(os.path.join(args.out, "launch.json"), "w") as f:
        json.dump({"bin": args.bin, "argv": argv, "env": sorted(k + "=" + v for k, v in env.items()
                                                                if k.startswith("ATLAS_"))}, f, indent=2)
    server = subprocess.Popen([args.bin, *argv], env=env, stdout=open(log_path, "w"),
                              stderr=subprocess.STDOUT)
    run = {"escape_in_env": escape, "extra_env": list(args.env), "argv_override": list(args.argv_override)}
    try:
        for _ in range(900):
            if server.poll() is not None:
                sys.exit(f"server exited rc={server.returncode} during load; see {log_path}")
            try:
                if b"ready" in urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2).read():
                    break
            except OSError:
                pass
            time.sleep(2)
        else:
            sys.exit("server never became ready")
        live = dict(l.split("=", 1) for l in open(f"/proc/{server.pid}/environ").read().split("\0") if "=" in l)
        run["environ_matches_launch"] = all(live.get(k) == v for k, v in env.items())
        run["escape_in_live_environ"] = live.get(ESCAPE) == "1"

        if args.prefill:
            prompts = json.load(open(args.prefill))
            for p in prompts:
                r = post(args.port, {"model": MODEL_NAME, "prompt": p["ids"], "max_tokens": 1,
                                     "temperature": 0, "stream": False})
                print(f"prefill {p['name']}: prompt_tokens={r['usage']['prompt_tokens']}", flush=True)
            run["prefill_prompts"] = args.prefill
        else:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
            prompts = json.load(open(args.decode))
            seqs, seen = [], 0
            for p in prompts:
                prefix = p["ids"][:args.decode_prefix]
                r = post(args.port, {"model": MODEL_NAME, "prompt": prefix, "max_tokens": args.decode_tokens,
                                     "temperature": 0, "stream": False})
                new = dumps(logits_dir)[seen:]
                seen += len(new)
                text = r["choices"][0]["text"]
                generated = [last_row_argmax(f, rows) for _, rows, f in new]
                # The prefill's first dump row covers the prompt; the rest are one per walk.
                # The server's text drops the continuation's leading whitespace.
                same_text = tok.decode(generated).lstrip() == text.lstrip()
                ok = (len(new) == r["usage"]["completion_tokens"]
                      and new[0][1] == len(prefix) and all(rows == 1 for _, rows, _ in new[1:])
                      and same_text)
                print(f"decode {p['name']}: dumps={len(new)} completion_tokens={r['usage']['completion_tokens']} "
                      f"ids_reproduce_text={same_text}", flush=True)
                if not ok:
                    sys.exit(f"decode capture for {p['name']} does not account for the response: "
                             f"{[(n, rows) for n, rows, _ in new][:4]}... text={text[:80]!r}")
                seqs.append({"name": p["name"], "prompt_len": len(prefix),
                             "ids": prefix + generated[:-1], "text": text,
                             "atlas_files": [f for _, _, f in new], "atlas_rows": [rows for _, rows, _ in new]})
            json.dump(seqs, open(os.path.join(args.out, "sequences.json"), "w"))
        log = open(log_path, errors="replace").read()
        run["escape_warning_in_log"] = "UNVALIDATED_BRINGUP=1 with its admission" in log
        run["target_only_admission_in_log"] = "EXL3 target-only admission open" in log
    finally:
        if server.poll() is None:
            server.send_signal(signal.SIGINT)
            try:
                server.wait(60)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    json.dump(run, open(os.path.join(args.out, "run.json"), "w"), indent=2)
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
