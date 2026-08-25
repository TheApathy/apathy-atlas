#!/usr/bin/env python3
"""Monitor Vast credit and optionally stop one named instance at a floor."""

import argparse
import json
import pathlib
import subprocess
import time


def read_credit(vastai: pathlib.Path) -> float:
    result = subprocess.run(
        [str(vastai), "--raw", "show", "user"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    credit = payload.get("credit")
    if not isinstance(credit, (int, float)):
        raise RuntimeError("Vast user response has no numeric credit")
    return float(credit)


def stop_instance(vastai: pathlib.Path, instance_id: int) -> None:
    subprocess.run(
        [str(vastai), "stop", "instance", str(instance_id)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("instance_id", type=int)
    parser.add_argument("--floor", type=float, required=True)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--vastai", type=pathlib.Path, default=pathlib.Path("vastai"))
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--arm",
        action="store_true",
        help="actually stop the instance at the floor; otherwise only report",
    )
    args = parser.parse_args()
    if args.instance_id <= 0:
        raise RuntimeError("instance_id must be positive")
    if args.floor < 0:
        raise RuntimeError("floor must be non-negative")
    if args.interval < 10 and not args.once:
        raise RuntimeError("interval must be at least 10 seconds")

    while True:
        credit = read_credit(args.vastai)
        print(
            f"Vast credit=${credit:.2f} floor=${args.floor:.2f} "
            f"instance={args.instance_id} armed={args.arm}",
            flush=True,
        )
        if credit <= args.floor:
            if not args.arm:
                raise SystemExit(2)
            stop_instance(args.vastai, args.instance_id)
            print(f"stopped Vast instance {args.instance_id} at credit floor", flush=True)
            return
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
