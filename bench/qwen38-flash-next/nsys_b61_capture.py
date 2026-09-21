#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Capture one warmed b61 request with Nsight; never measure throughput."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import nsys_b61_capture_cleanup as cleanup_support
import nsys_b61_capture_control as control_support
import nsys_b61_capture_logs as log_support
import nsys_b61_capture_model as model_support
import nsys_b61_capture_process as process_support
import nsys_b61_capture_support as support
import nsys_b61_capture_trace as trace_support

REPO, NSYS, PORT = support.REPO, support.NSYS, support.PORT
server_argv = support.server_argv
launch_command = support.launch_command
start_command = support.start_command


def main() -> int:
    args = support.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    receipt: dict[str, Any] = {
        "schema": "atlas-b61-nsys-split-capture-v2",
        "qualified": False,
        "performance_claim_allowed": False,
        "gpu_activity_attribution_only": True,
        "reservation_nonce_sha256": hashlib.sha256(
            args.gpu_reservation_nonce.encode()
        ).hexdigest(),
    }
    launcher = None
    target_pid = None
    before = None
    collection = False
    candidate = False
    events: list[dict[str, Any]] = []
    owned_descendants: list[dict[str, int]] = []
    server_log = (output / "server.log").open("xb")
    control_log = (output / "nsys-control.log").open("xb")
    session = f"atlas_b61_nsys_{os.getpid()}"
    trace_base = output / "trace"
    try:
        if process_support.listener_inodes(PORT):
            raise RuntimeError(f"port {PORT} already has a listener")
        receipt["locked_files"] = support.attest_locked_files()
        receipt["model_pre"] = model_support.attest_model()
        receipt["nsys"] = {
            "path": str(NSYS),
            "sha256": support.sha_file(NSYS),
            "capability_sha256": support.nsys_capabilities(NSYS),
        }
        build_id = subprocess.check_output(
            ["/usr/bin/readelf", "-n", str(support.BINARY)], text=True
        )
        expected_build_id = "85ed951204ccc167771ee3defcb7345fdac3db2a"
        if expected_build_id not in build_id:
            raise RuntimeError("sealed binary build-id drift")
        receipt["build_id"] = expected_build_id
        command = launch_command(session)
        receipt["launch"] = {
            "argv": command,
            "argv_sha256": hashlib.sha256(support.canonical_bytes(command)).hexdigest(),
            "target_environment": support.TARGET_ENV,
            "session": session,
        }
        launcher = subprocess.Popen(
            command,
            cwd=REPO,
            env=support.TARGET_ENV,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        target_pid, before = process_support.wait_target(
            launcher.pid, process_support.attest_target
        )
        owned_descendants = process_support.process_snapshot(
            process_support.descendants(launcher.pid)
        )
        receipt["process_pre"] = before
        process_support.wait_http()
        guard = process_support.CaptureGuard()
        warm_body = support.request_body(64)
        warm_response, warm_hash = log_support.http_json(support.ENDPOINT, warm_body)
        warm_semantic = log_support.validate_response(warm_response, 64)
        guard.warmup_done(warm_body, warm_semantic)
        receipt["warmup"] = {
            "request_sha256": hashlib.sha256(
                support.canonical_bytes(warm_body)
            ).hexdigest(),
            "response_sha256": warm_hash,
            "semantic": warm_semantic,
        }
        log_support.run_control(
            "start", start_command(session, trace_base), control_log, events
        )
        collection = True
        guard.collection_started()
        traced_body = support.request_body(400)
        guard.traced_request(traced_body)
        traced_response, traced_hash = log_support.http_json(
            support.ENDPOINT, traced_body
        )
        traced_semantic = log_support.validate_response(traced_response, 400)
        log_support.run_control(
            "stop", [str(NSYS), "stop", f"--session={session}"], control_log, events
        )
        collection = False
        guard.collection_stopped()
        receipt["request"] = {
            "request_sha256": hashlib.sha256(
                support.canonical_bytes(traced_body)
            ).hexdigest(),
            "response_sha256": traced_hash,
            "semantic": traced_semantic,
        }
        report, sqlite = (
            trace_base.with_suffix(".nsys-rep"),
            trace_base.with_suffix(".sqlite"),
        )
        reports = (
            ("cuda_gpu_trace", "cuda-trace"),
            ("cuda_gpu_kern_gb_sum", "cuda-summary"),
        )
        for report_name, output_name in reports:
            log_support.run_control(
                report_name,
                [
                    str(NSYS),
                    "stats",
                    "--report",
                    report_name,
                    "--format",
                    "csv",
                    "--output",
                    str(output / output_name),
                    str(report),
                ],
                control_log,
                events,
            )
        trace_csv = output / "cuda-trace_cuda_gpu_trace.csv"
        summary_csv = output / "cuda-summary_cuda_gpu_kern_gb_sum.csv"
        receipt["trace_analysis"] = trace_support.validate_trace(trace_csv, sqlite)
        after = process_support.attest_target(target_pid)
        if after != before:
            raise RuntimeError("process/listener identity drift during capture")
        receipt["process_post"] = after
        receipt["locked_files_post"] = support.attest_locked_files()
        if receipt["locked_files_post"] != receipt["locked_files"]:
            raise RuntimeError("sealed build artifacts drift during capture")
        receipt["model_post"] = model_support.attest_model()
        if receipt["model_post"] != receipt["model_pre"]:
            raise RuntimeError("model inventory drift during capture")
        artifacts = [report, sqlite, trace_csv, summary_csv]
        receipt["artifacts"] = {
            path.name: log_support.freeze(path) for path in artifacts
        }
        modules = (
            support,
            model_support,
            process_support,
            cleanup_support,
            control_support,
            log_support,
            trace_support,
        )
        receipt["harness_sha256"] = {
            Path(__file__).name: support.sha_file(Path(__file__)),
            **{
                Path(module.__file__).name: support.sha_file(Path(module.__file__))
                for module in modules
            },
        }
        candidate = True
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        receipt["qualified"] = cleanup_support.finalize(
            output=output,
            receipt=receipt,
            candidate=candidate,
            launcher=launcher,
            target_pid=target_pid,
            before=before,
            collection=collection,
            session=session,
            server_log=server_log,
            control_log=control_log,
            events=events,
            owned_descendants=owned_descendants,
        )
        print(output / "capture-receipt.json")
    return 0 if receipt["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
