#!/usr/bin/env python3
"""Project strict Track 2 runtime from completed, real V15 rollout epochs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PROGRESS = re.compile(
    r"Generating Rollout Epochs:\s+(?P<pct>\d+)%.*?\|\s+(?P<done>\d+)/(?P<total>\d+)\s+"
    r"\[(?P<elapsed>[^<]+)<[^,]+,\s*(?P<seconds>[0-9.]+)s/it\]"
)


def count(path: Path, token: str) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(token in line for line in handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher-log", required=True, type=Path)
    parser.add_argument("--service-log", required=True, type=Path)
    parser.add_argument("--bridge-log", required=True, type=Path)
    parser.add_argument("--formal-steps", default=1000, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    text = ANSI.sub("", args.launcher_log.read_text(encoding="utf-8", errors="replace")).replace("\r", "\n")
    matches = list(PROGRESS.finditer(text))
    completed = [m for m in matches if int(m.group("done")) > 0]
    if not completed:
        raise SystemExit("no completed rollout epoch is available")
    latest = completed[-1]
    done = int(latest.group("done"))
    total = int(latest.group("total"))
    seconds_per_epoch = float(latest.group("seconds"))
    rollout_seconds_per_update = seconds_per_epoch * total
    rollout_lower_bound_days = rollout_seconds_per_update * args.formal_steps / 86400.0
    service_success = count(args.service_log, "POST /v1/predict HTTP/1.1\" 200 OK")
    bridge_success = count(args.bridge_log, "POST /chunk_step HTTP/1.1\" 200 OK")
    report = {
        "format": "strict-track2-runtime-projection-v1",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "source": {
            "launcher_log": str(args.launcher_log),
            "service_log": str(args.service_log),
            "bridge_log": str(args.bridge_log),
        },
        "observed": {
            "completed_rollout_epochs": done,
            "rollout_epochs_per_grpo_update": total,
            "mean_seconds_per_completed_epoch_from_tqdm": seconds_per_epoch,
            "successful_predict_requests": service_success,
            "successful_chunk_steps": bridge_success,
            "predict_requests_per_observed_chunk": (
                service_success / bridge_success if bridge_success else None
            ),
        },
        "formal_projection": {
            "public_full_reference_grpo_steps": args.formal_steps,
            "rollout_seconds_per_grpo_step": rollout_seconds_per_update,
            "rollout_only_lower_bound_days": rollout_lower_bound_days,
            "continuous_operation_required": True,
            "checkpoint_interval_steps": 10,
            "rollout_only_hours_between_recovery_points": rollout_seconds_per_update * 10 / 3600.0,
        },
        "limitations": [
            "This is a lower bound from completed rollout epochs, not a promised finish time.",
            "Actor optimization, weight synchronization, checkpoint I/O, service acceptance and real RoboTwin evaluation are excluded.",
            "The projection does not authorize changing the byte-verified public full-reference 1000-step configuration.",
            "WorldArena 2.0 does not publish the organizer's held-out sample counts, so this projection is not an organizer-certified schedule.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
