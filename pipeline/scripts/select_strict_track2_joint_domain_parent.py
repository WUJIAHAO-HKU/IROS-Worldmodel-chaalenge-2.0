#!/usr/bin/env python3
"""Select a domain-general parent using preregistered paired validation gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_reports(values: list[str]) -> dict[int, Path]:
    reports: dict[int, Path] = {}
    for value in values:
        step_text, separator, path_text = value.partition(":")
        if not separator or not step_text.isdigit():
            raise ValueError(f"report must be STEP:PATH, got {value!r}")
        step = int(step_text)
        path = Path(path_text).resolve()
        if step in reports:
            raise ValueError(f"duplicate report step {step}")
        reports[step] = path
    return reports


def metric(report: dict, group: str, name: str) -> float:
    return float(report["improvement"][group][f"{name}_improvement_percent"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-report", action="append", required=True)
    parser.add_argument("--onpolicy-report", action="append", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    official = parse_reports(args.official_report)
    onpolicy = parse_reports(args.onpolicy_report)
    if set(official) != set(onpolicy):
        raise ValueError("official and on-policy reports must cover identical steps")
    preregistration = Path(args.preregistration).resolve()
    prereg = json.loads(preregistration.read_text())
    expected = set(int(value) for value in prereg["selection"]["checkpoint_candidates"])
    if set(official) != expected:
        raise ValueError(f"evaluated steps {sorted(official)} do not match preregistered {sorted(expected)}")

    candidates = []
    for step in sorted(official):
        official_report = json.loads(official[step].read_text())
        onpolicy_report = json.loads(onpolicy[step].read_text())
        values = {
            "official_overall_rgb": metric(official_report, "overall", "rgb_mae"),
            "official_left_rgb": metric(official_report, "left", "rgb_mae"),
            "official_right_rgb": metric(official_report, "right", "rgb_mae"),
            "official_overall_contact": metric(official_report, "overall", "contact_rgb_mae"),
            "official_texture": metric(official_report, "overall", "texture_mae"),
            "official_temporal": metric(official_report, "overall", "temporal_delta_mae"),
            "onpolicy_overall_rgb": metric(onpolicy_report, "overall", "rgb_mae"),
            "onpolicy_left_rgb": metric(onpolicy_report, "left", "rgb_mae"),
            "onpolicy_right_rgb": metric(onpolicy_report, "right", "rgb_mae"),
            "onpolicy_overall_contact": metric(onpolicy_report, "overall", "contact_rgb_mae"),
            "onpolicy_texture": metric(onpolicy_report, "overall", "texture_mae"),
            "onpolicy_temporal": metric(onpolicy_report, "overall", "temporal_delta_mae"),
        }
        thresholds = prereg["selection"]["required_parent_gates_percent"]
        checks = {
            name: {"value_percent": value, "minimum_percent": float(thresholds[name]),
                   "passed": value >= float(thresholds[name])}
            for name, value in values.items()
        }
        checkpoint = (
            Path(args.checkpoint_root).resolve()
            / "checkpoints"
            / f"checkpoint_step_{step:06d}"
        )
        score_components = [
            values["official_left_rgb"], values["official_right_rgb"],
            values["onpolicy_left_rgb"], values["onpolicy_right_rgb"],
        ]
        candidates.append({
            "step": step,
            "checkpoint": str(checkpoint),
            "official_report": str(official[step]),
            "official_report_sha256": sha256(official[step]),
            "onpolicy_report": str(onpolicy[step]),
            "onpolicy_report_sha256": sha256(onpolicy[step]),
            "checks": checks,
            "passed": all(item["passed"] for item in checks.values()),
            "selection_score": min(score_components) + 0.001 * sum(score_components),
        })

    eligible = [candidate for candidate in candidates if candidate["passed"]]
    selected = max(eligible, key=lambda item: item["selection_score"]) if eligible else None
    result = {
        "format": "strict-track2-joint-domain-parent-selection-v1",
        "preregistration": str(preregistration),
        "preregistration_sha256": sha256(preregistration),
        "selection_rule": prereg["selection"]["primary"],
        "passed": selected is not None,
        "selected": selected,
        "candidates": candidates,
        "fallback_required": selected is None,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
