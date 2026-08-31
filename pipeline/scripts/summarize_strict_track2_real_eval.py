#!/usr/bin/env python3
"""Aggregate clean and read-only-instrumented RoboTwin Track 2 evaluations."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch


METRIC_RE = re.compile(
    r"'eval/(?P<key>[a-zA-Z0-9_]+)':\s*(?:array\()?(?P<value>[0-9.eE+-]+)"
)
COUNT_RE = re.compile(r"'eval/num_trajectories':\s*(?P<count>[0-9]+)")
EXPECTED_COUNTS = {
    "00": 8,
    "01": 16,
    "02": 16,
    "03": 16,
    "04": 16,
    "05": 16,
    "06": 16,
    "07": 16,
    "08": 8,
}
VARIANTS = ("baseline", "trained_global_step_1")


def parse_log(path: Path) -> dict | None:
    if not path.exists():
        return None
    lines = [line for line in path.read_text(errors="replace").splitlines() if "'eval/" in line]
    if not lines:
        return None
    line = lines[-1]
    count_match = COUNT_RE.search(line)
    if count_match is None:
        return None
    metrics = {match.group("key"): float(match.group("value")) for match in METRIC_RE.finditer(line)}
    metrics["num_trajectories"] = int(count_match.group("count"))
    metrics["log_path"] = str(path)
    return metrics


def clean_log(eval_root: Path, variant: str, batch: str) -> Path:
    if batch == "00":
        return eval_root / variant / "shard_00" / "launcher.log"
    return eval_root / variant / f"batch16_{batch}" / "launcher.log"


def instrumented_log(eval_root: Path, variant: str, batch: str) -> Path:
    return eval_root / "instrumented_metrics" / variant / f"batch16_{batch}" / "launcher.log"


def batch_seeds(eval_root: Path, batch: str) -> list[int]:
    path = (
        eval_root / "seed_shards" / "shard_00.json"
        if batch == "00"
        else eval_root / "seed_batches16" / f"batch_{batch}.json"
    )
    data = json.loads(path.read_text())
    seeds = torch.as_tensor(data["adjust_bottle"]["success_seeds"], dtype=torch.long)
    generator = torch.Generator()
    generator.manual_seed(0)
    order = torch.randperm(seeds.numel(), generator=generator)
    return [int(seed) for seed in seeds[order].tolist()]


def successes(metrics: dict) -> int:
    return round(metrics["success_once"] * metrics["num_trajectories"])


def wilson(success_count: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p = success_count / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half_width = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [center - half_width, center + half_width]


def aggregate_primary(eval_root: Path, variant: str) -> dict:
    rows = []
    total_success = 0
    total_count = 0
    for batch, expected in EXPECTED_COUNTS.items():
        clean = parse_log(clean_log(eval_root, variant, batch))
        instrumented = parse_log(instrumented_log(eval_root, variant, batch))
        selected = instrumented or clean
        if selected is None:
            raise RuntimeError(f"missing completed primary result for {variant} batch {batch}")
        if selected["num_trajectories"] != expected:
            raise RuntimeError(f"unexpected trajectory count for {variant} batch {batch}")
        batch_success = successes(selected)
        total_success += batch_success
        total_count += expected
        rows.append(
            {
                "batch": batch,
                "count": expected,
                "successes": batch_success,
                "success_rate": batch_success / expected,
                "source": (
                    "deterministic_instrumented_read_only"
                    if instrumented is not None
                    else "unseeded_clean_fallback"
                ),
                "log_path": selected["log_path"],
            }
        )
    return {
        "successes": total_success,
        "count": total_count,
        "success_rate": total_success / total_count,
        "success_rate_wilson_95": wilson(total_success, total_count),
        "batches": rows,
    }


def aggregate_instrumented(eval_root: Path, variant: str) -> dict:
    sums = {
        "success_once": 0,
        "grasp_once": 0,
        "arm_right": 0,
        "arm_left": 0,
        "right_success": 0,
        "left_success": 0,
        "right_grasp": 0,
        "left_grasp": 0,
    }
    batch_rows = []
    seed_results = []
    total = 0
    for batch, expected in EXPECTED_COUNTS.items():
        metrics = parse_log(instrumented_log(eval_root, variant, batch))
        if metrics is None:
            raise RuntimeError(f"missing instrumented result for {variant} batch {batch}")
        if metrics["num_trajectories"] != expected:
            raise RuntimeError(f"unexpected instrumented count for {variant} batch {batch}")
        row_counts = {}
        for key in sums:
            if key not in metrics:
                raise RuntimeError(f"instrumented metric {key} missing in {variant} batch {batch}")
            value = round(metrics[key] * expected)
            sums[key] += value
            row_counts[key] = value
        packed = {}
        for key in ("success_bitmask", "grasp_bitmask", "arm_right_bitmask"):
            if key not in metrics:
                raise RuntimeError(f"instrumented metric {key} missing in {variant} batch {batch}")
            packed[key] = round(metrics[key] * expected)
        seeds = batch_seeds(eval_root, batch)
        if len(seeds) != expected:
            raise RuntimeError(f"seed count mismatch in batch {batch}")
        decoded = []
        for index, seed in enumerate(seeds):
            result = {
                "seed": seed,
                "arm": "right" if (packed["arm_right_bitmask"] >> index) & 1 else "left",
                "success": bool((packed["success_bitmask"] >> index) & 1),
                "grasp": bool((packed["grasp_bitmask"] >> index) & 1),
            }
            decoded.append(result)
            seed_results.append(result)
        if sum(row["success"] for row in decoded) != row_counts["success_once"]:
            raise RuntimeError(f"success bitmask mismatch in {variant} batch {batch}")
        if sum(row["grasp"] for row in decoded) != row_counts["grasp_once"]:
            raise RuntimeError(f"grasp bitmask mismatch in {variant} batch {batch}")
        if any(row["success"] and not row["grasp"] for row in decoded):
            raise RuntimeError(f"official success without grasp evidence in {variant} batch {batch}")
        batch_rows.append({"batch": batch, "count": expected, **row_counts})
        total += expected

    right = sums["arm_right"]
    left = sums["arm_left"]
    assert right + left == total
    assert sums["right_success"] + sums["left_success"] == sums["success_once"]
    return {
        "count": total,
        "successes": sums["success_once"],
        "success_rate": sums["success_once"] / total,
        "grasp_completions": sums["grasp_once"],
        "grasp_completion_rate": sums["grasp_once"] / total,
        "right": {
            "count": right,
            "successes": sums["right_success"],
            "success_rate": sums["right_success"] / right,
            "grasp_completions": sums["right_grasp"],
            "grasp_completion_rate": sums["right_grasp"] / right,
        },
        "left": {
            "count": left,
            "successes": sums["left_success"],
            "success_rate": sums["left_success"] / left,
            "grasp_completions": sums["left_grasp"],
            "grasp_completion_rate": sums["left_grasp"] / left,
        },
        "batches": batch_rows,
        "seed_results": seed_results,
    }


def exact_mcnemar(
    baseline: list[dict], trained: list[dict], arm: str | None = None, field: str = "success"
) -> dict:
    baseline_by_seed = {row["seed"]: row for row in baseline}
    trained_by_seed = {row["seed"]: row for row in trained}
    if set(baseline_by_seed) != set(trained_by_seed):
        raise RuntimeError("paired seed sets differ")
    pairs = []
    for seed in sorted(baseline_by_seed):
        base = baseline_by_seed[seed]
        candidate = trained_by_seed[seed]
        if base["arm"] != candidate["arm"]:
            raise RuntimeError(f"arm label changed for seed {seed}")
        if arm is None or base["arm"] == arm:
            pairs.append((base[field], candidate[field]))
    gained = sum((not base) and candidate for base, candidate in pairs)
    lost = sum(base and (not candidate) for base, candidate in pairs)
    both_success = sum(base and candidate for base, candidate in pairs)
    both_fail = sum((not base) and (not candidate) for base, candidate in pairs)
    discordant = gained + lost
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(min(gained, lost) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    return {
        "count": len(pairs),
        "both_success": both_success,
        "both_fail": both_fail,
        "baseline_fail_trained_success": gained,
        "baseline_success_trained_fail": lost,
        "exact_two_sided_p": p_value,
    }


def rate_delta(baseline: float, trained: float) -> dict:
    absolute_pp = (trained - baseline) * 100.0
    if baseline:
        relative_pct = ((trained / baseline) - 1.0) * 100.0
    else:
        relative_pct = math.inf if trained > 0.0 else 0.0
    return {
        "baseline": baseline,
        "trained": trained,
        "absolute_percentage_points": absolute_pp,
        "relative_percent": relative_pct,
        "absolute_at_least_3pp": absolute_pp >= 3.0,
        "relative_at_least_3_percent": relative_pct >= 3.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True, type=Path)
    args = parser.parse_args()

    primary = {variant: aggregate_primary(args.eval_root, variant) for variant in VARIANTS}
    instrumented = {variant: aggregate_instrumented(args.eval_root, variant) for variant in VARIANTS}
    paired = {
        "all": exact_mcnemar(
            instrumented["baseline"]["seed_results"],
            instrumented["trained_global_step_1"]["seed_results"],
        ),
        "left": exact_mcnemar(
            instrumented["baseline"]["seed_results"],
            instrumented["trained_global_step_1"]["seed_results"],
            arm="left",
        ),
        "right": exact_mcnemar(
            instrumented["baseline"]["seed_results"],
            instrumented["trained_global_step_1"]["seed_results"],
            arm="right",
        ),
    }
    paired_grasp = {
        "all": exact_mcnemar(
            instrumented["baseline"]["seed_results"],
            instrumented["trained_global_step_1"]["seed_results"],
            field="grasp",
        ),
        "left": exact_mcnemar(
            instrumented["baseline"]["seed_results"],
            instrumented["trained_global_step_1"]["seed_results"],
            arm="left",
            field="grasp",
        ),
        "right": exact_mcnemar(
            instrumented["baseline"]["seed_results"],
            instrumented["trained_global_step_1"]["seed_results"],
            arm="right",
            field="grasp",
        ),
    }

    equivalence = []
    for variant in VARIANTS:
        for batch in EXPECTED_COUNTS:
            clean = parse_log(clean_log(args.eval_root, variant, batch))
            auxiliary = parse_log(instrumented_log(args.eval_root, variant, batch))
            if clean is not None and auxiliary is not None:
                equivalence.append(
                    {
                        "variant": variant,
                        "batch": batch,
                        "clean_successes": successes(clean),
                        "instrumented_successes": successes(auxiliary),
                        "identical": successes(clean) == successes(auxiliary),
                    }
                )

    baseline_rate = primary["baseline"]["success_rate"]
    trained_rate = primary["trained_global_step_1"]["success_rate"]
    absolute_pp = (trained_rate - baseline_rate) * 100.0
    relative_pct = ((trained_rate / baseline_rate) - 1.0) * 100.0 if baseline_rate else math.inf
    arm_improvement = {
        arm: rate_delta(
            instrumented["baseline"][arm]["success_rate"],
            instrumented["trained_global_step_1"][arm]["success_rate"],
        )
        for arm in ("left", "right")
    }
    grasp_improvement = {
        "all": rate_delta(
            instrumented["baseline"]["grasp_completion_rate"],
            instrumented["trained_global_step_1"]["grasp_completion_rate"],
        ),
        **{
            arm: rate_delta(
                instrumented["baseline"][arm]["grasp_completion_rate"],
                instrumented["trained_global_step_1"][arm]["grasp_completion_rate"],
            )
            for arm in ("left", "right")
        },
    }
    report = {
        "format": "strict-track2-real-robotwin-eval-v1",
        "task": "adjust_bottle",
        "official_seed_count": sum(EXPECTED_COUNTS.values()),
        "primary_official_success": primary,
        "auxiliary_read_only_metrics": instrumented,
        "paired_success_transitions": paired,
        "paired_grasp_transitions": paired_grasp,
        "unseeded_clean_overlap_diagnostic": {
            "overlap": equivalence,
            "all_counts_identical": bool(equivalence) and all(row["identical"] for row in equivalence),
            "note": "Clean pilots did not seed rollout diffusion RNG; deterministic overlay results are primary.",
        },
        "improvement": {
            "absolute_percentage_points": absolute_pp,
            "relative_percent": relative_pct,
            "absolute_at_least_3pp": absolute_pp >= 3.0,
            "relative_at_least_3_percent": relative_pct >= 3.0,
        },
        "success_improvement_by_arm": arm_improvement,
        "grasp_improvement": grasp_improvement,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
