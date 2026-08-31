#!/usr/bin/env python3
"""Persist a training-only v318/v308 comparison through metric step 3."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
OFF = ROOT / "artifacts/strict_track2_official_20260810"
V318 = OFF / "runs/v318_v317_authorized_official_h200_r4_step10_lr2e5_beta001_seed1471_20260822"
V308 = OFF / "runs/v308_v301_rtx5090_fresh_official_h200_r4_step10_lr2e5_beta001_seed1471_20260821"


def values(report: dict, tag: str) -> list[float]:
    return [float(report["metrics"][tag][str(step)]) for step in range(4)]


def main() -> None:
    output = V318 / "audit/v318_v308_through_step3_comparison.json"
    if output.exists():
        raise FileExistsError(output)
    current_path = V318 / "audit/v318_training_prefix_step3.json"
    baseline_path = V308 / "audit/v308_training_prefix_step3.json"
    current = json.loads(current_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    cur_success = values(current, "env/success_once")
    old_success = values(baseline, "env/success_once")
    cur_return = values(current, "env/return")
    old_return = values(baseline, "env/return")
    report = {
        "format": "strict-track2-v318-v308-training-only-through-step3-comparison-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "v318": {
            "success_once": cur_success,
            "success_count_equivalent": [round(value * 128) for value in cur_success],
            "return": cur_return,
            "first4_success_count_equivalent_sum": round(sum(cur_success) * 128),
            "first4_return_mean": sum(cur_return) / 4,
        },
        "rejected_v308": {
            "success_once": old_success,
            "success_count_equivalent": [round(value * 128) for value in old_success],
            "return": old_return,
            "first4_success_count_equivalent_sum": round(sum(old_success) * 128),
            "first4_return_mean": sum(old_return) / 4,
        },
        "observations": {
            "v318_step3_success_rebounded_above_steps0_1_2": cur_success[3] > max(cur_success[:3]),
            "v318_step3_return_rebounded_above_steps0_1_2": cur_return[3] > max(cur_return[:3]),
            "v318_step3_return_above_v308_step3": cur_return[3] > old_return[3],
            "v318_step3_success_above_v308_step3": cur_success[3] > old_success[3],
            "v318_first4_coverage_above_v308": sum(cur_success) > sum(old_success),
            "v318_first4_return_mean_above_v308": sum(cur_return) > sum(old_return),
            "training_proxy_proves_public_or_final_success": False,
        },
        "inputs": {"v318": str(current_path), "v308": str(baseline_path)},
        "selection_inputs": "training-only metrics; no public112, final128, hidden, or contest outcomes",
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
