#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only

"""Launch and qualify the exact N=2410 DeepSeek V4 max-prefill profile.

This is a GB10 runner, not an offline performance estimator.  It derives the
environment from ``exl3-prefill-max.sh``, adopts that launch into an active
benchmark receipt before inference, measures twenty unique zero-cache prompts,
and requires every strict production-arm engagement marker.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import shlex
import signal
import stat
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
import benchmark_receipt  # noqa: E402


def load_prefill_probe():
    """Load the generic probe without letting its legacy port argv see ours."""
    path = SCRIPT_DIR / "prefill_probe.py"
    spec = importlib.util.spec_from_file_location("atlas_prefill_probe_reuse", path)
    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    try:
        sys.argv = [str(path)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv
    return module


prefill_probe = load_prefill_probe()


SCHEMA = "atlas-exl3-prefill-max-qualification-v1"
TARGET_TOKENS = 2410
MEASURED_RUNS = 20
TARGET_TOK_S = 2000.0
TARGET_TTFT_SECONDS = TARGET_TOKENS / TARGET_TOK_S
MODEL = Path("/home/flocka/models/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1")
BINARY = REPO / "target/release/spark"
LAUNCHER = SCRIPT_DIR / "exl3-prefill-max.sh"
BUILD_RECEIPT = REPO / "target/release/spark.prefill-max-build-receipt"
MAX_CAPTURE_BYTES = 1 << 20
MAX_SERVER_LOG_BYTES = 64 << 20

REQUIRED_PROFILE_VALUES = {
    # exl3-serve.sh always prepends these two values.
    "ATLAS_EXL3_PREFILL_CHUNK": "1",
    "ATLAS_KV_OVERCOMMIT": "0",
    # exl3-prefill-max.sh owns every remaining value. Keep this exact so a
    # launcher edit cannot silently weaken or extend a qualification profile.
    "ATLAS_PREFILL_MAX_REQUIRE_ARMS": "1",
    "ATLAS_V4_PREFILL_CUBLASLT": "1",
    "ATLAS_V4_ATTN_RELEASE_BF16": "0",
    "ATLAS_V4_PREFILL_HC_RMS_FUSED": "1",
    "ATLAS_V4_ATTN_NVFP4": "0",
    "ATLAS_V4_PREFILL_TC": "1",
    "ATLAS_V4_PREFILL_TC2": "1",
    "ATLAS_V4_PREFILL_TC2_WARP0": "0",
    "ATLAS_V4_PREFILL_QB_ROPE_FUSED": "0",
    "ATLAS_V4_COMP_GEMM_TC": "1",
    "ATLAS_V4_KV_PIPELINED": "1",
    "ATLAS_V4_WOA_INPLACE": "1",
    "ATLAS_HC_TILED": "1",
    "ATLAS_V4_PREFILL_KV_ALIAS": "1",
    "ATLAS_V4_PREFILL_INVERSE_ROPE_FUSED": "1",
    "ATLAS_EXL3_PREFILL_DIRECT": "1",
    "ATLAS_EXL3_PREFILL_PERSISTENT": "1",
    "ATLAS_EXL3_PREFILL_FIXED_K2": "1",
    "ATLAS_EXL3_PREFILL_FIXED_SHAPE": "1",
    "ATLAS_EXL3_PREFILL_FUSED_POST": "1",
    "ATLAS_EXL3_PREFILL_DUAL_PRE": "1",
    "ATLAS_EXL3_PREFILL_W2A8": "1",
    "ATLAS_EXL3_PREFILL_W2A8_FUSED_GU_DOWN": "1",
    "ATLAS_EXL3_PREFILL_W2A8_FUSED_GU_DOWN_N256": "0",
    "ATLAS_EXL3_PREFILL_W2A8_N256_DOWN": "1",
    "ATLAS_EXL3_PREFILL_FUSED_UNPERMUTE": "1",
    "ATLAS_EXL3_HROW_FIXED_SHAPE": "1",
    "ATLAS_EXL3_PREFILL_FUSED_BLEND": "0",
    "ATLAS_MOE_SHARED_K64": "4",
    "ATLAS_EXL3_PREFILL_M128": "0",
    "ATLAS_EXL3_PREFILL_K64": "0",
    "ATLAS_EXL3_PREFILL_N128": "0",
    "ATLAS_EXL3_PREFILL_N256": "0",
}

FORBIDDEN_PROCESS_ENV = frozenset(
    {
        "GAMMA",
        "DSPARK_TOKENS",
        "DRAFTER",
        "DFLASH_TRAIN_DUMP",
        "FP8_KV_CALIBRATION_TOKENS",
        "PRINT_CONFIG_ONLY",
    }
)

ENGAGEMENT_PATTERNS = {
    "w2a8_core": re.compile(
        r"ATLAS_PREFILL_MAX_ARMS_RECEIPT core=w2a8 fused_gu=n128 "
        r"down=n256 n_tokens=2410 total_expanded=14460 experts=256 top_k=6(?:\s|$)"
    ),
    "w2a8_fused_unpermute": re.compile(
        r"ATLAS_PREFILL_MAX_ARMS_RECEIPT tail=fused_unpermute "
        r"n_tokens=2410 hidden=4096 top_k=6(?:\s|$)"
    ),
    "v4_hc_rms_attention": re.compile(
        r"V4_PREFILL_MAX_ARM_ENGAGED arm=hc_pre_finish_rms_fused "
        r"site=attention layer=\d+ n=2410(?:\s|$)"
    ),
    "v4_hc_rms_ffn": re.compile(
        r"V4_PREFILL_MAX_ARM_ENGAGED arm=hc_pre_finish_rms_fused "
        r"site=ffn layer=\d+ n=2410(?:\s|$)"
    ),
    "v4_kv_alias": re.compile(
        r"V4_PREFILL_MAX_ARM_ENGAGED arm=kv_alias layer=\d+ "
        r"n=2410 nq=64 nkv=1 hd_mla=512(?:\s|$)"
    ),
    "v4_inverse_rope": re.compile(
        r"V4_PREFILL_MAX_ARM_ENGAGED arm=inverse_rope layer=\d+ "
        r"n=2410 nq=64 nkv=1 hd_mla=512(?:\s|$)"
    ),
}


def load_receipt_binding(path: Path, base_url: str) -> dict:
    """Reuse the generic probe's live receipt verifier at this runner's port."""
    previous_base = prefill_probe.BASE
    try:
        prefill_probe.BASE = base_url
        return prefill_probe.load_receipt_binding(str(path))
    finally:
        prefill_probe.BASE = previous_base


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_receipt_digest(path: Path = BUILD_RECEIPT) -> str:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ValueError("max-prefill build receipt is missing") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or path.is_symlink()
        or metadata.st_size > 4096
    ):
        raise ValueError("max-prefill build receipt is unsafe or oversized")
    return sha256_file(path)


def require_unchanged_build_receipt(path: Path, expected_digest: str) -> None:
    if build_receipt_digest(path) != expected_digest:
        raise ValueError("max-prefill build receipt changed during qualification")


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def request_json(base_url: str, endpoint: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{base_url}{endpoint}",
        json.dumps(payload, separators=(",", ":")).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with benchmark_receipt.urlopen_no_redirect(request, timeout=30) as response:
            value = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise ValueError(f"Atlas {endpoint} request failed") from error
    if type(value) is not dict:
        raise ValueError(f"Atlas {endpoint} response is malformed")
    return value


def tokenize(base_url: str, text: str) -> list[int]:
    value = request_json(base_url, "/tokenize", {"prompt": text})
    tokens = value.get("tokens")
    count = value.get("count")
    if (
        type(tokens) is not list
        or type(count) is not int
        or count != len(tokens)
        or any(type(token) is not int or token < 0 for token in tokens)
    ):
        raise ValueError("Atlas /tokenize token IDs or count are malformed")
    return tokens


def build_exact_prompt(base_url: str, nonce: str, target_tokens: int) -> list[int]:
    if not re.fullmatch(r"[0-9a-f]{64}", nonce):
        raise ValueError("measurement nonce must be 64 lowercase hex characters")
    if type(target_tokens) is not int or target_tokens <= 0:
        raise ValueError("target token count must be a positive integer")
    prefix = tokenize(
        base_url,
        f"nonce={nonce}\nSummarize the following Atlas benchmark facts in one sentence:\n",
    )
    filler = tokenize(base_url, " atlas")
    if not prefix or not filler or len(prefix) >= target_tokens:
        raise ValueError(
            "live tokenizer cannot construct the exact qualification prompt"
        )
    # Legacy completions accepts token IDs directly.  Appending one known-valid
    # filler ID makes the submitted vector exact without retokenization drift.
    return prefix + [filler[-1]] * (target_tokens - len(prefix))


def validate_usage(usage: dict) -> tuple[int, int]:
    if type(usage) is not dict:
        raise ValueError("final completion usage is missing or malformed")
    prompt_tokens = usage.get("prompt_tokens")
    if type(prompt_tokens) is not int or prompt_tokens != TARGET_TOKENS:
        raise ValueError(
            f"qualification requires exactly {TARGET_TOKENS} API-reported prompt tokens"
        )
    details = usage.get("prompt_tokens_details")
    if type(details) is not dict or type(details.get("cached_tokens")) is not int:
        raise ValueError("final completion cached-token accounting is missing")
    cached_tokens = details["cached_tokens"]
    if cached_tokens != 0:
        raise ValueError("qualification requires zero cached prompt tokens")
    return prompt_tokens, prompt_tokens


def validate_terminal_usage(usage: dict, finish_reason: str | None) -> int:
    if finish_reason not in ("length", "stop"):
        raise ValueError("completion terminal finish_reason is missing or unsupported")
    completion_tokens = usage.get("completion_tokens") if type(usage) is dict else None
    if type(completion_tokens) is not int or not 1 <= completion_tokens <= 4:
        raise ValueError("completion terminal completion_tokens is malformed")
    if finish_reason == "length" and completion_tokens != 4:
        raise ValueError("length terminal must account for all four completion tokens")
    return completion_tokens


def run_exact_completion(base_url: str, model: str, prompt_tokens: list[int]) -> dict:
    if type(model) is not str or not model:
        raise ValueError("completion model identity is missing")
    if (
        type(prompt_tokens) is not list
        or len(prompt_tokens) != TARGET_TOKENS
        or any(type(token) is not int or token < 0 for token in prompt_tokens)
    ):
        raise ValueError(
            f"submitted prompt must contain exactly {TARGET_TOKENS} valid token IDs"
        )
    payload = {
        "model": model,
        "prompt": prompt_tokens,
        "temperature": 0,
        "max_tokens": 4,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        json.dumps(payload, separators=(",", ":")).encode(),
        {"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    ttft = None
    final_usage = None
    finish_reason = None
    terminal_seen = False
    usage_seen = False
    done_seen = False
    output_parts = []
    try:
        response = benchmark_receipt.urlopen_no_redirect(request, timeout=1800)
        with response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                if line == "data: [DONE]":
                    if done_seen or not terminal_seen or not usage_seen:
                        raise ValueError(
                            "completion SSE has malformed terminal ordering"
                        )
                    done_seen = True
                    continue
                if done_seen:
                    raise ValueError("completion SSE emitted data after [DONE]")
                event = json.loads(line[5:])
                if type(event) is not dict:
                    raise ValueError("completion SSE event is malformed")
                if "error" in event:
                    raise ValueError(f"completion SSE error: {event['error']}")
                choices = event.get("choices")
                if type(choices) is not list:
                    raise ValueError("completion SSE choices are malformed")
                if choices == []:
                    if not terminal_seen or usage_seen or event.get("usage") is None:
                        raise ValueError(
                            "completion SSE has malformed or repeated usage"
                        )
                    final_usage = event["usage"]
                    usage_seen = True
                    continue
                if usage_seen or terminal_seen:
                    raise ValueError("completion SSE emitted content after terminal")
                if len(choices) != 1 or type(choices[0]) is not dict:
                    raise ValueError("completion SSE must contain exactly one choice")
                if event.get("usage") is not None:
                    raise ValueError("completion SSE attached usage before terminal")
                choice = choices[0]
                text = choice.get("text")
                if type(text) is not str:
                    raise ValueError("completion SSE choice text is malformed")
                if text:
                    output_parts.append(text)
                    if ttft is None:
                        ttft = time.perf_counter() - started
                terminal = choice.get("finish_reason")
                if terminal is not None:
                    if type(terminal) is not str:
                        raise ValueError(
                            "completion terminal finish_reason is malformed"
                        )
                    finish_reason = terminal
                    terminal_seen = True
    except (
        OSError,
        UnicodeError,
        urllib.error.URLError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError("streaming completion request failed") from error
    wall = time.perf_counter() - started
    if not done_seen:
        raise ValueError("completion SSE is missing [DONE]")
    total, fresh = validate_usage(final_usage)
    completion_tokens = validate_terminal_usage(final_usage, finish_reason)
    if ttft is None or ttft <= 0.0:
        raise ValueError("first streamed completion timestamp is missing")
    record = prefill_probe.sample_record((total, fresh, ttft, wall))
    record["prompt_sha256"] = canonical_sha256(prompt_tokens)
    record["output_sha256"] = hashlib.sha256("".join(output_parts).encode()).hexdigest()
    record["finish_reason"] = finish_reason
    record["completion_tokens"] = completion_tokens
    return record


def measure_exact_prefill(
    base_url: str, model: str, measured_runs: int = MEASURED_RUNS
) -> dict:
    if type(measured_runs) is not int or measured_runs <= 0:
        raise ValueError("measured run count must be positive")
    prompts = set()

    def one() -> dict:
        prompt = build_exact_prompt(base_url, secrets.token_hex(32), TARGET_TOKENS)
        fingerprint = canonical_sha256(prompt)
        if fingerprint in prompts:
            raise ValueError("qualification prompt nonce repeated")
        prompts.add(fingerprint)
        record = run_exact_completion(base_url, model, prompt)
        if record["prompt_sha256"] != fingerprint:
            raise ValueError("qualification prompt fingerprint changed")
        return record

    warmup = one()
    samples = [one() for _ in range(measured_runs)]
    return {"warmup": warmup, "samples": samples}


def parse_profile_config(output: str) -> dict[str, str]:
    if len(output.encode()) > MAX_CAPTURE_BYTES:
        raise ValueError("max-profile config output exceeds the bounded size")
    lines = [
        line.removeprefix("env  : ")
        for line in output.splitlines()
        if line.startswith("env  : ")
    ]
    if len(lines) != 1:
        raise ValueError("max-profile config must contain exactly one environment line")
    profile = {}
    for assignment in shlex.split(lines[0]):
        if not re.fullmatch(r"ATLAS_[A-Z0-9_]+=[^\s]*", assignment):
            raise ValueError(
                f"max-profile config contains an invalid assignment: {assignment}"
            )
        key, value = assignment.split("=", 1)
        if key in profile:
            raise ValueError(f"max-profile config repeats an assignment: {key}")
        profile[key] = value
    missing = sorted(set(REQUIRED_PROFILE_VALUES) - set(profile))
    if missing:
        raise ValueError(f"max-profile config omits required value: {missing[0]}")
    wrong = sorted(
        key
        for key, expected in REQUIRED_PROFILE_VALUES.items()
        if profile.get(key) != expected
    )
    if wrong:
        key = wrong[0]
        raise ValueError(
            f"max-profile config has wrong value for {key}: "
            f"expected {REQUIRED_PROFILE_VALUES[key]!r}, got {profile[key]!r}"
        )
    unexpected = sorted(set(profile) - set(REQUIRED_PROFILE_VALUES))
    if unexpected:
        raise ValueError(f"max-profile config has unexpected value: {unexpected[0]}")
    return profile


def validate_process_profile(expected: dict[str, str], actual: dict[str, str]) -> None:
    actual_profile = {
        key: value for key, value in actual.items() if key.startswith("ATLAS_")
    }
    forbidden = sorted(key for key in FORBIDDEN_PROCESS_ENV if key in actual)
    if actual_profile != expected or forbidden:
        detail = forbidden[0] if forbidden else "ATLAS_*"
        raise ValueError(f"max-profile process environment mismatch: {detail}")


def expected_argv(port: int) -> list[str]:
    return [
        str(BINARY),
        "serve",
        str(MODEL),
        "--port",
        str(port),
        "--kv-cache-dtype",
        "fp8",
        "--lm-head-dtype",
        "fp8",
        "--gpu-memory-utilization",
        "0.96",
        "--max-seq-len",
        "4096",
        "--max-num-seqs",
        "1",
        "--max-batch-size",
        "1",
        "--max-prefill-tokens",
        "4096",
        "--oom-guard-mb",
        "2048",
    ]


def launcher_environment(port: int, log: Path, *, print_only: bool) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("ATLAS_") and key not in FORBIDDEN_PROCESS_ENV
    }
    environment.update({"PORT": str(port), "LOG": str(log)})
    if print_only:
        environment["PRINT_CONFIG_ONLY"] = "1"
    return environment


def run_config_preflight(output_dir: Path, port: int) -> dict[str, str]:
    result = subprocess.run(
        [str(LAUNCHER)],
        cwd=REPO,
        env=launcher_environment(port, output_dir / "config.log", print_only=True),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(
            f"max-profile config preflight failed: {result.stderr[-1000:]}"
        )
    return parse_profile_config(result.stdout)


def parse_launch_output(output: str, expected_log: Path) -> tuple[int, dict]:
    if len(output.encode()) > MAX_CAPTURE_BYTES:
        raise ValueError("max-profile launcher output exceeds the bounded size")
    matches = re.findall(r"^pid=([0-9]+) log=(\S+)$", output, flags=re.MULTILINE)
    if len(matches) != 1 or Path(matches[0][1]) != expected_log:
        raise ValueError(
            "max-profile launcher did not return one exact process identity"
        )
    preflights = []
    for line in output.splitlines():
        if line.startswith("{") and len(line.encode()) <= MAX_CAPTURE_BYTES:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                type(value) is dict
                and value.get("schema") == "atlas-exl3-prefill-max-model-v1"
            ):
                preflights.append(value)
    if len(preflights) != 1 or preflights[0].get("status") != "ok":
        raise ValueError("max-profile launcher omitted the exact model preflight")
    return int(matches[0][0]), preflights[0]


def launch_server(output_dir: Path, port: int) -> tuple[int, Path, dict]:
    log = output_dir / "server.log"
    result = subprocess.run(
        [str(LAUNCHER)],
        cwd=REPO,
        env=launcher_environment(port, log, print_only=False),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=600,
        check=False,
    )
    (output_dir / "launcher.stdout").write_text(result.stdout, encoding="utf-8")
    (output_dir / "launcher.stderr").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise ValueError(f"max-profile launcher failed: {result.stderr[-1000:]}")
    pid, preflight = parse_launch_output(result.stdout, log)
    return pid, log, preflight


def wait_for_process_identity(
    pid: int, port: int, expected_profile: dict[str, str], timeout: float = 10.0
) -> dict[str, str]:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            environment = benchmark_receipt.read_process_environment(pid)
            validate_process_profile(expected_profile, environment)
            if benchmark_receipt.process_argv(pid) != expected_argv(port):
                raise ValueError(
                    "process argv has not reached the exact server command"
                )
            executable, _ = benchmark_receipt.process_executable(pid)
            if Path(executable).resolve() != BINARY.resolve():
                raise ValueError(
                    "process executable has not reached the release binary"
                )
            return environment
        except ValueError as error:
            last_error = error
            try:
                os.kill(pid, 0)
            except OSError as process_error:
                raise ValueError(
                    f"max-profile process exited before identity binding: pid={pid}"
                ) from process_error
            time.sleep(0.05)
    raise ValueError(f"max-profile process identity timed out: {last_error}")


def create_active_receipt(
    output_dir: Path,
    pid: int,
    port: int,
    expected_profile: dict[str, str],
    ready_timeout: float,
) -> Path:
    process_environment = wait_for_process_identity(pid, port, expected_profile)
    argv = benchmark_receipt.process_argv(pid)

    planned = output_dir / "benchmark.planned.receipt.json"
    active = output_dir / "benchmark.active.receipt.json"
    helper = SCRIPT_DIR / "benchmark_receipt.py"
    plan_command = [
        sys.executable,
        str(helper),
        "--repo",
        str(REPO),
        "--binary",
        str(BINARY),
        "--model",
        str(MODEL),
        "--argv-json",
        json.dumps(argv, separators=(",", ":")),
        "--environment-json",
        json.dumps(expected_profile, sort_keys=True, separators=(",", ":")),
        "--require-string",
        "deepseek_v4",
        "--require-string",
        "ATLAS_PREFILL_MAX_ARMS_RECEIPT",
        "--require-string",
        "V4_PREFILL_MAX_ARM_ENGAGED",
        "--output",
        str(planned),
    ]
    result = subprocess.run(
        plan_command,
        cwd=REPO,
        env=process_environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=1800,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"benchmark receipt planning failed: {result.stderr[-1000:]}")
    envelope = benchmark_receipt.read_envelope(planned)
    if envelope["manifest"][
        "full_environment_sha256"
    ] != benchmark_receipt.environment_sha256(process_environment):
        raise ValueError("planned receipt does not bind the full server environment")
    activate_command = [
        sys.executable,
        str(helper),
        "--activate-receipt",
        str(planned),
        "--pid",
        str(pid),
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--port",
        str(port),
        "--ready-timeout",
        str(ready_timeout),
        "--output",
        str(active),
    ]
    result = subprocess.run(
        activate_command,
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=ready_timeout + 1800,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(
            f"benchmark receipt activation failed: {result.stderr[-1000:]}"
        )
    envelope = benchmark_receipt.read_envelope(active)
    if envelope.get("activation", {}).get("state") != "ACTIVE_VERIFIED":
        raise ValueError("benchmark receipt did not reach ACTIVE_VERIFIED")
    return active


def validate_engagement_text(text: str) -> dict[str, int]:
    counts = {
        name: len(pattern.findall(text))
        for name, pattern in ENGAGEMENT_PATTERNS.items()
    }
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"strict engagement log is missing exact arm: {missing[0]}")
    return counts


def validate_engagement_log(path: Path) -> dict[str, int]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise ValueError("server log must be a regular non-symlink file")
    if metadata.st_size > MAX_SERVER_LOG_BYTES:
        raise ValueError("server log exceeds the bounded qualification size")
    return validate_engagement_text(path.read_text(encoding="utf-8"))


def validate_sample(sample: dict) -> None:
    ttft = sample.get("ttft_seconds") if type(sample) is dict else None
    wall = sample.get("wall_seconds") if type(sample) is dict else None
    rate = sample.get("prefill_tok_s") if type(sample) is dict else None
    if (
        type(sample) is not dict
        or sample.get("total_tokens") != TARGET_TOKENS
        or sample.get("fresh_tokens") != TARGET_TOKENS
        or type(ttft) not in (int, float)
        or type(wall) not in (int, float)
        or type(rate) not in (int, float)
        or not all(math.isfinite(value) and value > 0 for value in (ttft, wall, rate))
        or wall < ttft
        or not math.isclose(rate, TARGET_TOKENS / ttft, rel_tol=1e-12)
        or not re.fullmatch(r"[0-9a-f]{64}", str(sample.get("prompt_sha256")))
        or not re.fullmatch(r"[0-9a-f]{64}", str(sample.get("output_sha256")))
        or sample.get("finish_reason") not in ("length", "stop")
        or type(sample.get("completion_tokens")) is not int
        or not 1 <= sample["completion_tokens"] <= 4
        or (sample["finish_reason"] == "length" and sample["completion_tokens"] != 4)
    ):
        raise ValueError("qualification contains a malformed or non-exact sample")


def build_report(
    *,
    measurement: dict,
    receipt_binding: dict,
    engagement_counts: dict[str, int],
    build_receipt_sha256: str,
    model_preflight_sha256: str,
) -> dict:
    samples = measurement.get("samples") if type(measurement) is dict else None
    warmup = measurement.get("warmup") if type(measurement) is dict else None
    if type(samples) is not list or len(samples) != MEASURED_RUNS:
        raise ValueError(
            f"qualification requires exactly {MEASURED_RUNS} measured samples"
        )
    validate_sample(warmup)
    for sample in samples:
        validate_sample(sample)
    fingerprints = [sample["prompt_sha256"] for sample in [warmup, *samples]]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("qualification prompt fingerprints are not unique")
    if receipt_binding.get("classification") != "ACTIVE_VERIFIED":
        raise ValueError("qualification requires an ACTIVE_VERIFIED benchmark receipt")
    receipt_digest = receipt_binding.get("benchmark_receipt_sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", str(receipt_digest)):
        raise ValueError("active benchmark receipt digest is malformed")
    if (
        type(receipt_binding.get("gpu_identity")) is not dict
        or type(receipt_binding.get("gpu_activation_state")) is not dict
    ):
        raise ValueError("active benchmark GPU provenance is malformed")
    if set(engagement_counts) != set(ENGAGEMENT_PATTERNS) or any(
        type(count) is not int or count <= 0 for count in engagement_counts.values()
    ):
        raise ValueError("qualification engagement counts are incomplete")
    for label, digest in (
        ("build receipt", build_receipt_sha256),
        ("model preflight", model_preflight_sha256),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{label} digest is malformed")
    rates = [sample["prefill_tok_s"] for sample in samples]
    ttfts = [sample["ttft_seconds"] for sample in samples]
    median_rate = statistics.median(rates)
    median_ttft = statistics.median(ttfts)
    return {
        "schema": SCHEMA,
        "classification": "ACTIVE_VERIFIED",
        "geometry": {"prompt_tokens": TARGET_TOKENS, "cached_tokens": 0},
        "measured_runs": len(samples),
        "warmup": warmup,
        "samples": samples,
        "median_prefill_tok_s": round(median_rate, 6),
        "min_prefill_tok_s": round(min(rates), 6),
        "max_prefill_tok_s": round(max(rates), 6),
        "median_ttft_seconds": round(median_ttft, 6),
        "target_2000_tok_s_met": median_rate >= TARGET_TOK_S
        and median_ttft <= TARGET_TTFT_SECONDS,
        "benchmark_receipt_sha256": receipt_digest,
        "gpu_identity": receipt_binding["gpu_identity"],
        "gpu_activation_state": receipt_binding["gpu_activation_state"],
        "max_build_receipt_sha256": build_receipt_sha256,
        "model_preflight_sha256": model_preflight_sha256,
        "engagement_counts": dict(sorted(engagement_counts.items())),
    }


def original_process_running(pid: int, start_ticks: int) -> bool:
    try:
        observed_ticks = benchmark_receipt.process_start_ticks(pid)
    except ValueError as error:
        if "unavailable" not in str(error):
            raise
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        raise ValueError(
            "max-profile process identity is unavailable while its PID remains live"
        ) from error
    return observed_ticks == start_ticks


def stop_server(
    pid: int,
    start_ticks: int,
    *,
    graceful_timeout: float = 15.0,
    kill_timeout: float = 5.0,
) -> None:
    if not original_process_running(pid, start_ticks):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + graceful_timeout
    while time.monotonic() < deadline:
        if not original_process_running(pid, start_ticks):
            return
        time.sleep(0.1)
    if original_process_running(pid, start_ticks):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    deadline = time.monotonic() + kill_timeout
    while time.monotonic() < deadline:
        if not original_process_running(pid, start_ticks):
            return
        time.sleep(0.05)
    if original_process_running(pid, start_ticks):
        raise ValueError("max-profile server remained live after SIGKILL")


def create_output_dir(path: Path) -> Path:
    path = path.resolve()
    if path == REPO or REPO in path.parents or path == MODEL or MODEL in path.parents:
        raise ValueError(
            "qualification output directory must be outside the source and model repositories"
        )
    try:
        path.mkdir(mode=0o700)
    except FileExistsError as error:
        raise ValueError(
            f"qualification output directory already exists: {path}"
        ) from error
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8977)
    parser.add_argument("--ready-timeout", type=float, default=900.0)
    args = parser.parse_args()
    if not 0 < args.port <= 65535:
        raise SystemExit("port must be in 1..65535")
    if args.ready_timeout <= 0:
        raise SystemExit("ready timeout must be positive")
    output_dir = create_output_dir(args.output_dir)
    pid = None
    start_ticks = None
    report = None
    try:
        expected_profile = run_config_preflight(output_dir, args.port)
        build_receipt_before = build_receipt_digest()
        pid, server_log, model_preflight = launch_server(output_dir, args.port)
        start_ticks = benchmark_receipt.process_start_ticks(pid)
        active_receipt = create_active_receipt(
            output_dir, pid, args.port, expected_profile, args.ready_timeout
        )
        active_envelope = benchmark_receipt.read_envelope(active_receipt)
        activation = active_envelope["activation"]
        initial_binding = {
            "classification": activation["state"],
            "benchmark_receipt_sha256": benchmark_receipt.receipt_digest(
                active_envelope
            ),
            "gpu_identity": activation["gpu_identity"],
            "gpu_activation_state": activation["gpu_state"],
        }
        measurement = measure_exact_prefill(
            f"http://127.0.0.1:{args.port}", activation["model_id"]
        )
        require_unchanged_build_receipt(BUILD_RECEIPT, build_receipt_before)
        final_binding = load_receipt_binding(
            active_receipt, f"http://127.0.0.1:{args.port}"
        )
        if final_binding != initial_binding:
            raise ValueError("benchmark receipt binding changed during qualification")
        engagement_counts = validate_engagement_log(server_log)
        require_unchanged_build_receipt(BUILD_RECEIPT, build_receipt_before)
        report = build_report(
            measurement=measurement,
            receipt_binding=final_binding,
            engagement_counts=engagement_counts,
            build_receipt_sha256=build_receipt_before,
            model_preflight_sha256=canonical_sha256(model_preflight),
        )
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        raise SystemExit(str(error)) from error
    finally:
        if pid is not None and start_ticks is not None:
            try:
                stop_server(pid, start_ticks)
            except (OSError, ValueError):
                if report is not None:
                    raise
    if report is None:
        raise SystemExit("qualification did not produce a report")
    report["server_stopped_after_measurement"] = True
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result_path = output_dir / "qualification.json"
    with result_path.open("x", encoding="utf-8") as output:
        output.write(encoded + "\n")
        output.flush()
        os.fsync(output.fileno())
    print("RESULT_JSON=" + encoded)
    print(f"retained_artifacts={output_dir}")


if __name__ == "__main__":
    main()
