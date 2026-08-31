#!/usr/bin/env python3
"""Fail closed before any expensive strict-Track2 RL candidate run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_gate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("gate must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("gate must be LABEL=PATH")
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--absolute-gate", action="append", type=parse_gate, required=True)
    parser.add_argument("--causal-gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    candidate = json.loads(args.candidate_manifest.read_text())
    causal = json.loads(args.causal_gate.read_text())
    absolute = {
        label: {"path": str(path), "report": json.loads(path.read_text())}
        for label, path in args.absolute_gate
    }
    guards = candidate.get("guards", {})
    authorization = causal.get("authorization", {})
    checks = {
        "all_absolute_gates_passed": bool(absolute)
        and all(item["report"].get("passed") is True for item in absolute.values()),
        "causal_gate_passed": causal.get("passed") is True,
        "intended_effect_observed": authorization.get("intended_effect_observed") is True,
        "right_terminal_ranking": authorization.get("right_terminal_ranking") is True,
        "left_non_regression": authorization.get("left_non_regression") is True,
        "no_waivers_or_exclusions": authorization.get("waivers_or_excluded_failed_checks") is False,
        "world_model_only": guards.get("participant_component") == "world-model RGB predictor only",
        "policy_unmodified": guards.get("policy_modified") is False,
        "official_reward_unmodified": guards.get("official_reward_modified") is False,
        "official_rl_fixed": guards.get("official_rl_algorithm_or_budget_modified") is False,
        "no_hidden_or_final_data": guards.get("hidden_or_final_data") is False,
        "no_real_submission": guards.get("real_submission") is False,
    }
    passed = all(checks.values())
    report = {
        "format": "strict-track2-expensive-rl-authorization-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_manifest": {
            "path": str(args.candidate_manifest),
            "sha256": sha256(args.candidate_manifest),
        },
        "absolute_gates": {
            label: {
                "path": item["path"],
                "sha256": sha256(Path(item["path"])),
                "passed": item["report"].get("passed") is True,
            }
            for label, item in absolute.items()
        },
        "causal_gate": {
            "path": str(args.causal_gate),
            "sha256": sha256(args.causal_gate),
            "passed": causal.get("passed") is True,
        },
        "checks": checks,
        "passed": passed,
        "launch_permission": passed,
        "rule": "full RL launch wrapper must require this exact report and hash",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
