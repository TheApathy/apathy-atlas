"""APP-BUILD SMOKE: end-to-end 'build a small real app' eval.

Prompts the model to build a Flask TODO API as 3 files (models.py, app.py,
test_app.py), extracts the files, and runs the model's OWN tests plus a small
fixed harness (import + syntax) in a sandboxed subprocess (untrusted-code
discipline mirrors sandbox.py: separate process, rlimits, scrubbed env,
process-group kill, throwaway cwd).

Two build modes:
  - "single"   : one conversation, model emits all files in one response.
  - "parallel" : one request per file, issued concurrently (self-batching
                 pattern); each prompt carries a shared interface contract.

Score per mode: files_extracted, files_parse (compile()), app_imports,
tests_pass (model's own pytest), plus fixed smoke-tests pass, and wall-clock
for the whole build (generation + execution) = effective app-building
throughput.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from client import AtlasClient
from sandbox import _make_preexec  # reuse the rlimit discipline

EXPECTED_FILES = ("models.py", "app.py", "test_app.py")

_CONTRACT = """\
Interface contract (all files MUST follow it exactly):
- models.py: defines `class TodoStore` with methods:
    add(title: str) -> dict            # returns {"id": int, "title": str, "done": bool}
    list_all() -> list[dict]
    get(todo_id: int) -> dict | None
    update(todo_id: int, title=None, done=None) -> dict | None
    delete(todo_id: int) -> bool
  In-memory storage only (a dict). IDs are increasing integers starting at 1.
- app.py: `from models import TodoStore`; defines `def create_app()` returning a
  Flask app with routes:
    GET    /todos            -> 200, JSON list of todos
    POST   /todos            -> 201, JSON of created todo; body {"title": str};
                                400 if title missing/empty
    GET    /todos/<int:id>   -> 200 or 404
    PUT    /todos/<int:id>   -> 200 with updated todo, or 404; body may set
                                "title" and/or "done"
    DELETE /todos/<int:id>   -> 204, or 404
  Also `app = create_app()` at module level.
- test_app.py: pytest tests using `from app import create_app` and Flask's
  test_client(); at least 5 tests covering create, list, get-404, update, delete.
Use ONLY flask + stdlib (+pytest in tests). No database, no extensions.
"""

_SINGLE_PROMPT = f"""Build a small Flask TODO API as exactly 3 files.

{_CONTRACT}

Output ALL THREE files, each as:

### FILE: <filename>
```python
<complete file content>
```

No other prose. Every file must be complete and runnable.
"""

_PARALLEL_PROMPTS = {
    "models.py": f"""You are writing ONE file of a 3-file Flask TODO API.

{_CONTRACT}

Write ONLY models.py, complete and runnable. Output a single ```python fence,
no prose.""",
    "app.py": f"""You are writing ONE file of a 3-file Flask TODO API.

{_CONTRACT}

Write ONLY app.py, complete and runnable (assume models.py exists per the
contract). Output a single ```python fence, no prose.""",
    "test_app.py": f"""You are writing ONE file of a 3-file Flask TODO API.

{_CONTRACT}

Write ONLY test_app.py, complete and runnable (assume models.py and app.py
exist per the contract). Output a single ```python fence, no prose.""",
}

# Fixed smoke-tests (OUR harness, not the model's) — written into the app dir.
_FIXED_TESTS = '''\
"""Fixed harness tests (not model-authored)."""
from app import create_app


def _client():
    app = create_app()
    app.testing = True
    return app.test_client()


def test_fixed_create_and_list():
    c = _client()
    r = c.post("/todos", json={"title": "buy milk"})
    assert r.status_code == 201
    todo = r.get_json()
    assert todo["title"] == "buy milk" and todo["done"] is False
    r = c.get("/todos")
    assert r.status_code == 200
    assert any(t["title"] == "buy milk" for t in r.get_json())


def test_fixed_missing_title_400():
    c = _client()
    assert c.post("/todos", json={}).status_code == 400


def test_fixed_get_update_delete_cycle():
    c = _client()
    tid = c.post("/todos", json={"title": "x"}).get_json()["id"]
    assert c.get(f"/todos/{tid}").status_code == 200
    r = c.put(f"/todos/{tid}", json={"done": True})
    assert r.status_code == 200 and r.get_json()["done"] is True
    assert c.delete(f"/todos/{tid}").status_code == 204
    assert c.get(f"/todos/{tid}").status_code == 404
'''


# ── extraction ──────────────────────────────────────────────────────────

_FILE_HDR = re.compile(
    r"^[#*\s]*(?:###\s*)?FILE:\s*[`\"']?([\w./-]+\.py)[`\"']?\s*$",
    re.IGNORECASE | re.MULTILINE)
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_files(text: str) -> dict[str, str]:
    """Extract {filename: content} from a multi-file response."""
    files: dict[str, str] = {}
    headers = list(_FILE_HDR.finditer(text))
    for i, h in enumerate(headers):
        seg_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        seg = text[h.end():seg_end]
        m = _FENCE.search(seg)
        if m:
            files[os.path.basename(h.group(1))] = m.group(1)
    if not files:
        # No headers: maybe fences labelled inline (```python # models.py)
        for m in _FENCE.finditer(text):
            first = m.group(1).lstrip().splitlines()[:1]
            if first:
                fm = re.match(r"#\s*([\w./-]+\.py)", first[0])
                if fm:
                    files[os.path.basename(fm.group(1))] = m.group(1)
    return files


def extract_single_file(text: str) -> str | None:
    """Extract the one fenced python block from a per-file response."""
    m = _FENCE.search(text)
    if m:
        return m.group(1)
    # Bare code fallback: looks like python if it has def/class/import.
    if re.search(r"^(from|import|def|class)\s", text, re.MULTILINE):
        return text
    return None


# ── sandboxed execution ─────────────────────────────────────────────────

def _site_pythonpath() -> str:
    """Site dirs of THIS interpreter (incl. user site, where flask/pytest may
    live) so the sandbox child — which gets HOME=scratch — can import them."""
    import site
    # User site FIRST: it may carry newer versions (e.g. typing_extensions)
    # that user-installed packages require; system dist-packages would shadow
    # them otherwise.
    dirs = []
    us = site.getusersitepackages()
    if us:
        dirs.append(us)
    dirs.extend(site.getsitepackages())
    return os.pathsep.join(d for d in dirs if os.path.isdir(d))


def _run_sandboxed(cmd: list[str], cwd: str, timeout: float = 60.0):
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "EVALS_SANDBOX": "1",
        "http_proxy": "", "https_proxy": "", "HTTP_PROXY": "", "HTTPS_PROXY": "",
        # pytest/flask may live in the USER site (~/.local); HOME is pointed
        # into the scratch dir so nothing user-owned leaks, which would hide
        # them — surface the harness's own site dirs explicitly instead.
        "PYTHONPATH": _site_pythonpath(),
        # Deterministic pytest: don't autoload whatever plugins the host user
        # happens to have installed (they drag in unrelated imports).
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "HOME": cwd,
    }
    preexec = _make_preexec(cpu_seconds=45, mem_bytes=1024 * 1024 * 1024,
                            fsize_bytes=32 * 1024 * 1024, nproc=128)
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True,
                           text=True, timeout=timeout, preexec_fn=preexec)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return None, "", "TIMEOUT"


def evaluate_files(files: dict[str, str]) -> dict:
    """Write files to a scratch dir; check parse, import, model tests, fixed tests."""
    out = {
        "files_extracted": sorted(files),
        "missing_files": [f for f in EXPECTED_FILES if f not in files],
        "parse_ok": {}, "app_imports": False,
        "model_tests": None, "fixed_tests": None,
    }
    for name, src in files.items():
        try:
            compile(src, name, "exec")
            out["parse_ok"][name] = True
        except SyntaxError as e:
            out["parse_ok"][name] = f"SyntaxError: {e}"
    with tempfile.TemporaryDirectory(prefix="appbuild_") as tmp:
        for name, src in files.items():
            with open(os.path.join(tmp, name), "w", encoding="utf-8") as f:
                f.write(src)
        with open(os.path.join(tmp, "test_fixed_harness.py"), "w",
                  encoding="utf-8") as f:
            f.write(_FIXED_TESTS)
        py = sys.executable
        rc, _, err = _run_sandboxed([py, "-B", "-c", "import app"], tmp)
        out["app_imports"] = rc == 0
        if not out["app_imports"]:
            out["import_error"] = err[-500:]
        if "test_app.py" in files:
            rc, so, se = _run_sandboxed(
                [py, "-B", "-m", "pytest", "-x", "-q", "test_app.py"], tmp)
            out["model_tests"] = {"passed": rc == 0,
                                  "tail": (so + se)[-600:]}
        rc, so, se = _run_sandboxed(
            [py, "-B", "-m", "pytest", "-q", "test_fixed_harness.py"], tmp)
        out["fixed_tests"] = {"passed": rc == 0, "tail": (so + se)[-600:]}
    return out


# ── build modes ─────────────────────────────────────────────────────────

def build_single(client, *, max_tokens=6144, thinking=False) -> tuple[dict, dict]:
    t0 = time.time()
    comp = client.chat([{"role": "user", "content": _SINGLE_PROMPT}],
                       max_tokens=max_tokens, temperature=0.0,
                       enable_thinking=thinking)
    gen_s = time.time() - t0
    files = extract_files(comp.text)
    meta = {"gen_wall_s": round(gen_s, 1),
            "completion_tokens": comp.completion_tokens,
            "response_tail": comp.text[-400:]}
    return files, meta


def build_parallel(client, *, max_tokens=3072, thinking=False) -> tuple[dict, dict]:
    t0 = time.time()

    def one(name):
        c = client.chat([{"role": "user", "content": _PARALLEL_PROMPTS[name]}],
                        max_tokens=max_tokens, temperature=0.0,
                        enable_thinking=thinking)
        return name, c

    files, toks = {}, 0
    with ThreadPoolExecutor(max_workers=3) as ex:
        for name, comp in ex.map(one, EXPECTED_FILES):
            code = extract_single_file(comp.text)
            if code:
                files[name] = code
            toks += comp.completion_tokens
    gen_s = time.time() - t0
    return files, {"gen_wall_s": round(gen_s, 1), "completion_tokens": toks}


def run(base_url="http://127.0.0.1:8890", model="aeon-27b-dflash",
        modes=("single", "parallel"), thinking=False, out_path=None):
    client = AtlasClient(base_url=base_url, model=model)
    results = {}
    for mode in modes:
        print(f"[appbuild] mode={mode} generating ...", flush=True)
        t0 = time.time()
        if mode == "single":
            files, meta = build_single(client, thinking=thinking)
        else:
            files, meta = build_parallel(client, thinking=thinking)
        ev = evaluate_files(files)
        total_s = time.time() - t0
        results[mode] = {**meta, **ev,
                         "total_wall_s": round(total_s, 1),
                         "file_sizes": {k: len(v) for k, v in files.items()},
                         "files": files}
        print(f"[appbuild] {mode}: files={sorted(files)} "
              f"parse={ev['parse_ok']} imports={ev['app_imports']} "
              f"model_tests={ev['model_tests'] and ev['model_tests']['passed']} "
              f"fixed_tests={ev['fixed_tests'] and ev['fixed_tests']['passed']} "
              f"gen={meta['gen_wall_s']}s total={round(total_s,1)}s", flush=True)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="App-build smoke eval")
    ap.add_argument("--base-url", default="http://127.0.0.1:8890")
    ap.add_argument("--model", default="aeon-27b-dflash")
    ap.add_argument("--modes", default="single,parallel")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.base_url, a.model, modes=tuple(a.modes.split(",")),
        thinking=a.thinking, out_path=a.out)
