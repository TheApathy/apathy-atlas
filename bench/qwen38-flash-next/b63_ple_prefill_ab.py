# SPDX-License-Identifier: AGPL-3.0-only
"""Run sealed b63 serial-control/whole-prompt-PLE M2013 qualification once."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import b63_ple_prefill_ab_authority as authority
import b63_ple_prefill_ab_contract as contract
import b63_ple_prefill_ab_http as http_io
import b63_ple_prefill_ab_identity as identity
import b63_ple_prefill_ab_inventory as inventory
import b63_ple_prefill_ab_model as model_identity
import b63_ple_prefill_ab_process as process_identity
import b63_ple_prefill_ab_validate as validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", required=True, choices=[contract.EXECUTE])
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        parser.error("--output-dir must be absolute")
    return args


def _run_requests(
    arm: str,
    events: Any,
    process_pre: dict[str, Any],
    reservation_check: Callable[[], None],
    inventory_check: Callable[[set[int]], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model_endpoint = http_io.wait_models()
    reservation_check()
    if process_identity.attest_target(process_pre["pid"], arm) != process_pre:
        raise RuntimeError("server identity drift across readiness request")
    inventory_check({process_pre["pid"]})
    specs = contract.request_specs()
    schedule = [("canary", 0)] + [("m2013", index) for index in range(5)]
    runs = []
    for kind, index in schedule:
        reservation_check()
        before = process_identity.attest_target(process_pre["pid"], arm)
        inventory_check({process_pre["pid"]})
        if before != process_pre:
            raise RuntimeError("server identity drift before request")
        spec = specs[kind]
        response, response_sha256 = http_io.http_json(
            "POST", "/v1/chat/completions", spec.wire
        )
        semantic = http_io.validate_response(response, spec)
        after = process_identity.attest_target(process_pre["pid"], arm)
        inventory_check({process_pre["pid"]})
        reservation_check()
        if after != before:
            raise RuntimeError("server identity drift across request")
        event = {
            "arm": arm,
            "kind": kind,
            "index": index,
            "request_sha256": spec.wire_sha256,
            "response_sha256": response_sha256,
            "semantic": semantic,
        }
        persisted = {
            **event,
            "semantic": http_io.persistent_semantic(semantic),
        }
        identity.write_json_line(events, persisted)
        runs.append(event)
    return runs, model_endpoint


def run_arm(
    arm: str,
    output: Path,
    reservation_check: Callable[[], None],
    inventory_check: Callable[[set[int]], dict[str, Any]],
) -> dict[str, Any]:
    reservation_check()
    inventory_check(set())
    process_identity.assert_port_free()
    server_path = output / f"{arm}-server.log"
    event_path = output / f"{arm}-http.jsonl"
    process: subprocess.Popen[Any] | None = None
    process_pre = None
    starttime = None
    candidate = False
    runs: list[dict[str, Any]] = []
    endpoint: dict[str, Any] = {}
    cleanup: dict[str, Any] | None = None
    target_inventory: dict[str, Any] | None = None
    post_inventory: dict[str, Any] | None = None
    error: Exception | None = None
    with server_path.open("xb") as server_log, event_path.open("xb") as events:
        try:
            process = subprocess.Popen(
                process_identity.server_argv(contract.MODEL),
                cwd=process_identity.REPO,
                env=process_identity.arm_environment(arm),
                stdin=subprocess.DEVNULL,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            starttime = process_identity.starttime_ticks(process.pid)
            process_pre = process_identity.wait_target(process, arm)
            if process_pre["starttime_ticks"] != starttime:
                raise RuntimeError("server PID reuse during startup")
            target_inventory = inventory_check({process.pid})
            runs, endpoint = _run_requests(
                arm, events, process_pre, reservation_check, inventory_check
            )
            if process_identity.attest_target(process.pid, arm) != process_pre:
                raise RuntimeError("server identity drift before teardown")
            candidate = True
        except Exception as caught:
            error = caught
        finally:
            try:
                if process is not None and starttime is not None:
                    cleanup = process_identity.terminate_owned(process, starttime)
                elif process is not None:
                    cleanup = process_identity.terminate_unattested(process)
            except Exception as caught:
                error = error or caught
            try:
                reservation_check()
                post_inventory = inventory_check(set())
            except Exception as caught:
                error = error or caught
            server_log.flush()
    artifacts = {
        server_path.name: identity.freeze_file(server_path),
        event_path.name: identity.freeze_file(event_path),
    }
    if error is not None:
        raise RuntimeError(f"{arm} failed: {error}") from error
    if not candidate or cleanup is None or cleanup["returncode"] != 0:
        raise RuntimeError(f"{arm} did not shut down cleanly")
    log_census = validate.validate_server_log(server_path, arm)
    if len(runs) != 6 or [run["kind"] for run in runs] != ["canary"] + ["m2013"] * 5:
        raise RuntimeError("arm request schedule drift")
    return {
        "arm": arm,
        "process": process_pre,
        "model_endpoint": endpoint,
        "runs": runs,
        "metrics": validate.metrics(runs[1:]),
        "log_census": log_census,
        "cleanup": cleanup,
        "target_inventory": target_inventory,
        "post_inventory": post_inventory,
        "artifacts": artifacts,
    }


def main() -> int:
    args = parse_args()
    output = args.output_dir
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    qualified, inventory_authorized = False, False
    error = None
    completed: list[str] = []
    arms: list[dict[str, Any]] = []
    performance_gate = None
    campaign_inventory_pre = campaign_inventory_post = None
    try:
        parity = validate.attest_raw_parity()
        manifest = model_identity.load_manifest(
            contract.MODEL_MANIFEST, contract.MODEL_MANIFEST_SHA256
        )
        inputs_pre = authority.input_recheck(manifest, parity, True)
        inventory_authorized = True

        def inventory_check(expected_pids: set[int]) -> dict[str, Any]:
            return inventory.attest_inventory(inputs_pre["reservation"], expected_pids)

        def reservation_check() -> None:
            if authority.load_reservation() != inputs_pre["reservation"]:
                raise RuntimeError("root GPU authorization identity drift")

        campaign_inventory_pre = inventory_check(set())
        for arm_name in contract.ARM_ORDER:
            authority.input_recheck(manifest, parity, False)
            arms.append(run_arm(arm_name, output, reservation_check, inventory_check))
            completed.append(arm_name)
            authority.input_recheck(manifest, parity, False)
        inputs_post = authority.input_recheck(manifest, parity, True)
        campaign_inventory_post = inventory_check(set())
        if inputs_post != inputs_pre:
            raise RuntimeError("campaign input identity drift")
        performance_gate = validate.campaign_gate(arms)
        qualified = True
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
        if inventory_authorized:
            try:
                campaign_inventory_post = inventory_check(set())
            except Exception as cleanup_error:
                error += f"; inventory cleanup: {type(cleanup_error).__name__}: {cleanup_error}"
    if qualified:
        document = {
            "schema": "atlas-b63-ple-prefill-ab-qualification-v3",
            "qualified": True,
            "performance_claim_allowed": True,
            "binary_sha256": contract.BINARY_SHA256,
            "input_identity": inputs_pre,
            "campaign_inventory_pre": campaign_inventory_pre,
            "campaign_inventory_post": campaign_inventory_post,
            "performance_gate": performance_gate,
            "arms": arms,
        }
        result = output / "qualification.json"
    else:
        try:
            authority.purge_unqualified(output)
        except Exception as purge_error:
            error += f"; artifact purge: {type(purge_error).__name__}: {purge_error}"
        document = {
            "schema": "atlas-b63-ple-prefill-ab-failure-v2",
            "qualified": False,
            "performance_claim_allowed": False,
            "binary_sha256": contract.BINARY_SHA256,
            "completed_arms": completed,
            "error": error,
        }
        result = output / "failure.json"
    identity.publish_json(result, document)
    print(result)
    return 0 if qualified else 2


if __name__ == "__main__":
    sys.exit(main())
