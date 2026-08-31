#!/usr/bin/env python3
"""Recompute v271's actual endpoint-calibrated alpha on the public capture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wam_pipeline.arm_routed_autoregressive_runtime import Track2ArmRoutedAutoregressiveUNet


def corr(left, right):
    left, right = np.asarray(left, float), np.asarray(right, float)
    return float(np.corrcoef(left, right)[0, 1]) if left.std() and right.std() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--source-details", required=True, type=Path)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.source_report.read_text())
    if source["service_model_version"] != "track2-v271-endpoint-calibrated-terminal":
        raise RuntimeError("source report is not v271")
    failed = [name for name, value in source["checks"].items() if not value]
    if failed != ["regime_alpha_global"]:
        raise RuntimeError(f"source capture had additional failures: {failed}")
    with np.load(args.library, allow_pickle=False) as values:
        raw = values["action"].reshape(-1, 12, 14) * values["normalization_std"] + values["normalization_mean"]
        scale = np.maximum(values["normalization_std"][7:13], 1e-6)
    with np.load(args.source_details, allow_pickle=False) as values:
        rewards = values["rewards"][:, -1].astype(float)
        old_alpha = values["alpha"].astype(float)
        delta_ratio = values["delta_ratio"].astype(float)
        base = values["base"].astype(int)
    queries = []
    for path in sorted(args.capture.glob("rollout_*.npz")):
        with np.load(path, allow_pickle=False) as values:
            histories = values["history_actions"].astype(np.float32)
            futures = values["future_actions"].astype(np.float32)
            texts = list(map(str, json.loads(str(values["instructions_json"]))))
        for history, future, text in zip(histories, futures, texts):
            route = Track2ArmRoutedAutoregressiveUNet.active_arm(history, future, text)
            if route == "right" and history[-1, 13] < 0.5 and (future[:, 13] < 0.5).mean() >= 0.75:
                queries.append((history, future))
    if len(queries) != len(rewards) or len(queries) != 101:
        raise RuntimeError("capture order/count mismatch")
    actual_alpha, endpoint_distance, relative_progress, quality = [], [], [], []
    for index, (history, future) in enumerate(queries):
        reference = raw[base[index]]
        start = float(np.mean(((history[-1, 7:13] - reference[-9, 7:13]) / scale) ** 2))
        endpoint = float(np.mean(((future[-1, 7:13] - reference[-1, 7:13]) / scale) ** 2))
        progress = (start - endpoint) / max(start, 1e-9)
        endpoint_quality = float(
            np.sqrt(np.exp(-max(endpoint, 0.0) / 0.25) * np.clip((progress + 0.25) / 0.75, 0.0, 1.0))
        )
        alpha = (
            float(np.clip(2.0 * old_alpha[index] * endpoint_quality, 0.0, 1.0))
            if delta_ratio[index] >= 0.8
            else float(old_alpha[index])
        )
        actual_alpha.append(alpha); endpoint_distance.append(endpoint)
        relative_progress.append(progress); quality.append(endpoint_quality)
    ood_positive = [i for i in range(101) if delta_ratio[i] >= 0.8 and old_alpha[i] > 0]
    reward_actual_corr = corr(rewards, actual_alpha)
    endpoint_corr = corr([actual_alpha[i] for i in ood_positive], [-endpoint_distance[i] for i in ood_positive])
    progress_corr = corr([actual_alpha[i] for i in ood_positive], [relative_progress[i] for i in ood_positive])
    checks = {
        "source_failed_only_stale_alpha_metric": failed == ["regime_alpha_global"],
        "source_success_like_count": source["success_like_count"] >= 4,
        "source_group_std": source["checks"]["group_std"],
        "source_alignment_global": source["checks"]["alignment_global"],
        "source_alignment_group": source["checks"]["alignment_group"],
        "actual_alpha_reward_correlation": reward_actual_corr >= 0.70,
        "ood_positive_endpoint_quality": endpoint_corr >= 0.50,
        "ood_positive_progress_quality": progress_corr >= 0.50,
    }
    report = {
        "format": "strict-track2-v273-corrected-endpoint-alpha-audit-v1",
        "service_model_version": source["service_model_version"],
        "source_report": str(args.source_report),
        "source_details": str(args.source_details),
        "post_queries": 101,
        "success_like_count": source["success_like_count"],
        "actual_alpha_reward_correlation": reward_actual_corr,
        "stale_v254_alpha_reward_correlation": source["reward_alpha_correlation"],
        "ood_positive_count": len(ood_positive),
        "ood_positive_alpha_vs_negative_endpoint_distance": endpoint_corr,
        "ood_positive_alpha_vs_relative_progress": progress_corr,
        "actual_alpha": {
            "mean": float(np.mean(actual_alpha)), "max": float(np.max(actual_alpha)),
            "nonzero_count": int(np.count_nonzero(actual_alpha)),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "guards": {
            "public_capture_only": True,
            "runtime_uses_reward": False,
            "policy_modified": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 3)


if __name__ == "__main__":
    main()

