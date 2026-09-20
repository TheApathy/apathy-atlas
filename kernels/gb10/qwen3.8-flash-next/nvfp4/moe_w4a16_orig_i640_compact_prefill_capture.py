# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from moe_w4a16_orig_i640_compact_prefill_capture_support import (
    HEX,
    decode_kv,
    encode_kv,
    read_immutable,
    validate_model,
    write_immutable,
)
from moe_w4a16_orig_i640_compact_prefill_capture_run import hash_bytes, run_shape
from moe_w4a16_orig_i640_compact_prefill_provenance import (
    hook_receipt,
    server_receipt,
    source_receipt,
)

PLAN_KEYS = {
    "schema",
    "model",
    "model_manifest",
    "server",
    "server_receipt",
    "hook",
    "hook_compile_receipt",
    "output",
    "health_url",
    "request_url",
    "base_argv",
    "base_env",
    "nonce",
    "shapes",
}


def validate_plan(plan: dict) -> None:
    if set(plan) != PLAN_KEYS or plan["schema"] != "oi640-capture-plan-v1":
        raise ValueError("capture plan schema/keys")
    for key in (
        "model",
        "model_manifest",
        "server",
        "server_receipt",
        "hook",
        "hook_compile_receipt",
        "output",
    ):
        if not Path(plan[key]).is_absolute():
            raise ValueError(f"absolute path required: {key}")
    if not HEX.fullmatch(plan["nonce"]) or len(set(plan["nonce"])) < 8:
        raise ValueError("source bundle/nonce")
    if set(plan["shapes"]) != {"2013", "8192"}:
        raise ValueError("both exact shapes required")
    if (
        not isinstance(plan["base_argv"], list)
        or plan["base_argv"][0] != plan["server"]
    ):
        raise ValueError("exact server argv")
    argv = plan["base_argv"]
    required = (
        ("--kernel-target", "qwen3.8-flash-next"),
        ("--model-from-path", plan["model"]),
        ("--max-num-seqs", "1"),
    )
    for flag, value in required:
        if argv.count(flag) != 1 or argv[argv.index(flag) + 1] != value:
            raise ValueError(f"missing exact argv {flag}")
    health, request = urlparse(plan["health_url"]), urlparse(plan["request_url"])
    endpoint = ("http", "127.0.0.1", health.port)
    if (
        health.port is None
        or not 1024 <= health.port <= 65535
        or (health.scheme, health.hostname, health.port) != endpoint
        or (request.scheme, request.hostname, request.port) != endpoint
        or health.netloc != f"127.0.0.1:{health.port}"
        or request.netloc != health.netloc
        or health.path != "/health"
        or request.path != "/v1/chat/completions"
        or health.query
        or request.query
        or health.fragment
        or request.fragment
        or argv.count("--port") != 1
        or argv[argv.index("--port") + 1] != str(health.port)
    ):
        raise ValueError("request endpoint must be the launched local listener")
    if not isinstance(plan["base_env"], dict) or any(
        "\0" in k + v for k, v in plan["base_env"].items()
    ):
        raise ValueError("environment")
    required_env = {
        "ATLAS_DUMP_EXPERT_IDS": "1",
        "ATLAS_QWEN4_ATTN_PREFILL_BATCH": "1",
        "ATLAS_QWEN4_SSM_PREFILL_BATCH": "0",
    }
    if any(plan["base_env"].get(key) != value for key, value in required_env.items()):
        raise ValueError("exact route evidence environment")
    for rows in (2013, 8192):
        shape = plan["shapes"][str(rows)]
        if (
            set(shape) != {"request", "request_sha256"}
            or not Path(shape["request"]).is_absolute()
            or not HEX.fullmatch(shape["request_sha256"])
        ):
            raise ValueError("request plan")


def execute(plan_path: Path) -> None:
    plan_bytes = read_immutable(plan_path, 1 << 20)
    plan = json.loads(plan_bytes)
    validate_plan(plan)
    output = Path(plan["output"])
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    sources = source_receipt(Path(__file__))
    producer_path = output / "producer_sources.kv"
    producer_bytes = encode_kv(sources)
    write_immutable(producer_path, producer_bytes)
    model_manifest_path = Path(plan["model_manifest"])
    model_bytes = read_immutable(model_manifest_path)
    model_fields = decode_kv(model_bytes)
    validate_model(model_fields, Path(plan["model"]))
    hook_path = Path(plan["hook_compile_receipt"])
    hook_bytes, hook_id = hook_receipt(hook_path, Path(plan["hook"]), sources)
    server_path = Path(plan["server_receipt"])
    server_bytes, server_id = server_receipt(server_path, Path(plan["server"]))
    results = {
        rows: run_shape(plan, rows, output, server_id, hook_id) for rows in (2013, 8192)
    }
    validate_model(model_fields, Path(plan["model"]))
    if (
        source_receipt(Path(__file__)) != sources
        or hook_receipt(hook_path, Path(plan["hook"]), sources) != (hook_bytes, hook_id)
        or server_receipt(server_path, Path(plan["server"]))
        != (server_bytes, server_id)
    ):
        raise ValueError("producer/hook/server provenance drift")
    fields = {
        "schema": "oi640-capture-v1",
        "hidden": "2560",
        "intermediate": "640",
        "experts": "512",
        "top_k": "10",
        "capture.profile": "release",
        "source.bundle_sha256": sources["source.bundle_sha256"],
        "producer.bundle_sha256": sources["producer.bundle_sha256"],
        "producer.receipt.path": str(producer_path.resolve()),
        "producer.receipt.sha256": hash_bytes(producer_bytes),
        "hook.source.sha256": decode_kv(hook_bytes)["source.sha256"],
        "hook.compile_receipt.path": str(hook_path.resolve()),
        "hook.compile_receipt.sha256": hash_bytes(hook_bytes),
        "model.manifest.sha256": hash_bytes(model_bytes),
        "model.config.sha256": model_fields["config.sha256"],
        "model.index.sha256": model_fields["index.sha256"],
        "model.shard.count": model_fields["shard.count"],
        "server.build_receipt.path": str(server_path.resolve()),
        "server.build_receipt.sha256": hash_bytes(server_bytes),
        "capture.nonce": plan["nonce"],
        "capture.plan.path": str(plan_path.resolve()),
        "capture.plan.sha256": hash_bytes(plan_bytes),
    }
    fields.update({f"capture.binary.{key}": value for key, value in hook_id.items()})
    fields.update({f"server.binary.{key}": value for key, value in server_id.items()})
    for rows, result in results.items():
        fields.update({f"m{rows}.{key}": value for key, value in result.items()})
    write_immutable(output / "capture_receipt.kv", encode_kv(fields))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture authentic I640 offsets from real Atlas routes"
    )
    parser.add_argument("--plan", type=Path, required=True)
    execute(parser.parse_args().plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
