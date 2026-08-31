#!/usr/bin/env python3
"""Apply the preregistered v218 route-aware service promotion gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads((args.registry / "preregistration.json").read_text())
    v217 = json.loads(Path(prereg["inputs"]["v217_report"]).read_text())
    route = json.loads(Path(prereg["inputs"]["route_diagnostic"]).read_text())
    real = json.loads((args.run / "audit/real_context_determinism.json").read_text())
    fixed = prereg["fixed_audit"]
    checks = {
        "v217_success_recall": v217["success"]["hit_rate"] >= fixed["success_hit_rate_min"],
        "v217_failure_consistency": v217["failure"]["hit_rate"] <= fixed["failure_hit_rate_max"],
        "v217_margin": v217["success"]["hit_rate"] - v217["failure"]["hit_rate"] >= fixed["hit_rate_margin_min"],
        "official_http_acceptance": v217["checks"]["service_contract"],
        "route_mismatch_explains_large_difference": route["groups"]["public_failure_mismatched"]["count"] > 0,
        "route_matched_batch_difference_bounded": max(
            route["groups"]["public_success_matched"]["max_abs"],
            route["groups"]["public_failure_matched"]["max_abs"],
        ) <= fixed["route_matched_first_chunk_max_abs"],
        "real_context_repeat_pixel_exact": real["repeat_pixel_exact"],
        "real_context_action_sensitive": real["action_permutation_changed_count"] >= fixed["action_permutation_changed_count_min"],
    }
    report = {
        "format": "strict-track2-v218-route-aware-service-promotion-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "v217_online_success": v217["success"],
        "v217_online_failure": v217["failure"],
        "real_context": real,
        "model_pixels_changed_from_v217": False,
        "policy_training": False,
        "hidden_or_final_data": False,
        "real_submission": False,
        "final_128_used": False,
    }
    (args.run / "audit/v218_route_aware_promotion_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (args.run / ("V218_SERVICE_PROMOTED" if report["passed"] else "V218_SERVICE_REJECTED")).touch()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
