#!/usr/bin/env python3
"""Compare paired Track 2 closed-loop JSON results and apply promotion gates."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def _episode_id(path: str) -> int:
    match = re.search(r"episode(\d+)", path)
    if match is None:
        raise ValueError(f"cannot parse episode id from {path}")
    return int(match.group(1))


def _infer_side(reset_path: str) -> tuple[str, float, float]:
    frames = np.load(reset_path, allow_pickle=True)
    actions = np.stack([np.asarray(frame["abs_action"]) for frame in frames])
    left_motion = float(np.linalg.norm(np.diff(actions[:, :7], axis=0)))
    right_motion = float(np.linalg.norm(np.diff(actions[:, 7:], axis=0)))
    side = "left" if left_motion > right_motion else "right"
    return side, left_motion, right_motion


def compare(base: dict, candidate: dict, promotion_threshold_pct: float) -> dict:
    base_episodes = base["episodes"]
    candidate_episodes = candidate["episodes"]
    if len(base_episodes) != len(candidate_episodes):
        raise ValueError("base and candidate episode counts differ")

    rows = []
    for base_episode, candidate_episode in zip(base_episodes, candidate_episodes, strict=True):
        if base_episode["reset"] != candidate_episode["reset"]:
            raise ValueError("paired reset paths differ")
        base_scores = np.asarray(
            [item["official_reward_mean"] for item in base_episode["rounds"]], dtype=float
        )
        candidate_scores = np.asarray(
            [item["official_reward_mean"] for item in candidate_episode["rounds"]],
            dtype=float,
        )
        if base_scores.shape != candidate_scores.shape:
            raise ValueError("paired round counts differ")
        side, left_motion, right_motion = _infer_side(base_episode["reset"])
        rows.append(
            {
                "episode": _episode_id(base_episode["reset"]),
                "side": side,
                "instruction": base_episode["instruction"],
                "left_reset_motion": left_motion,
                "right_reset_motion": right_motion,
                "base_scores": base_scores,
                "candidate_scores": candidate_scores,
            }
        )

    rounds = len(rows[0]["base_scores"])
    if any(len(row["base_scores"]) != rounds for row in rows):
        raise ValueError("all episodes must have the same number of rounds")
    segments = {"full": (0, rounds), "early_r1_r2": (0, min(2, rounds))}
    if rounds >= 8:
        segments["middle_r3_r8"] = (2, 8)
    if rounds >= 16:
        segments["late_r9_r16"] = (8, 16)

    segment_results = {}
    for segment, (start, stop) in segments.items():
        side_results = {}
        for side in ("all", "left", "right"):
            selected = rows if side == "all" else [row for row in rows if row["side"] == side]
            base_values = np.concatenate([row["base_scores"][start:stop] for row in selected])
            candidate_values = np.concatenate(
                [row["candidate_scores"][start:stop] for row in selected]
            )
            base_mean = float(base_values.mean())
            candidate_mean = float(candidate_values.mean())
            episode_wins = sum(
                float(row["candidate_scores"][start:stop].mean())
                > float(row["base_scores"][start:stop].mean())
                for row in selected
            )
            side_results[side] = {
                "episodes": len(selected),
                "round_values": int(base_values.size),
                "base_mean": base_mean,
                "candidate_mean": candidate_mean,
                "absolute_delta": candidate_mean - base_mean,
                "relative_delta_pct": (candidate_mean / base_mean - 1.0) * 100.0,
                "episode_wins": episode_wins,
            }
        segment_results[segment] = side_results

    episode_results = []
    for row in rows:
        base_mean = float(row["base_scores"].mean())
        candidate_mean = float(row["candidate_scores"].mean())
        episode_results.append(
            {
                "episode": row["episode"],
                "side": row["side"],
                "instruction": row["instruction"],
                "base_mean": base_mean,
                "candidate_mean": candidate_mean,
                "absolute_delta": candidate_mean - base_mean,
                "relative_delta_pct": (candidate_mean / base_mean - 1.0) * 100.0,
            }
        )

    full = segment_results["full"]
    passed = all(
        full[side]["relative_delta_pct"] >= promotion_threshold_pct
        for side in ("all", "left", "right")
    )
    return {
        "format": "track2-paired-closed-loop-comparison-v1",
        "rounds_per_episode": rounds,
        "segments": segment_results,
        "episodes": episode_results,
        "promotion_gate": {
            "threshold_pct": promotion_threshold_pct,
            "requires": ["all", "left", "right"],
            "passed": passed,
            "decision": "promote" if passed else "reject",
        },
    }


def _markdown(result: dict) -> str:
    lines = [
        "# Track 2 Paired Closed-Loop Evaluation",
        "",
        f"Decision: **{result['promotion_gate']['decision'].upper()}** ",
        f"(required improvement: {result['promotion_gate']['threshold_pct']:.1f}% for all, left, and right).",
        "",
        "## Aggregate metrics",
        "",
        "| Segment | Side | Episodes | Base | Candidate | Relative delta | Wins |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for segment, side_results in result["segments"].items():
        for side, metrics in side_results.items():
            lines.append(
                f"| {segment} | {side} | {metrics['episodes']} | "
                f"{metrics['base_mean']:.8f} | {metrics['candidate_mean']:.8f} | "
                f"{metrics['relative_delta_pct']:+.4f}% | "
                f"{metrics['episode_wins']}/{metrics['episodes']} |"
            )
    lines.extend(
        [
            "",
            "## Per-episode full-horizon metrics",
            "",
            "| Episode | Side | Base | Candidate | Relative delta | Instruction |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for episode in result["episodes"]:
        lines.append(
            f"| {episode['episode']} | {episode['side']} | {episode['base_mean']:.8f} | "
            f"{episode['candidate_mean']:.8f} | {episode['relative_delta_pct']:+.4f}% | "
            f"{episode['instruction']} |"
        )
    lines.extend(
        [
            "",
            "Side is inferred from the larger reset-action motion norm across the two 7-DoF arms.",
            "Scores are public-data local diagnostics, not organizer hidden-test results.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--promotion-threshold-pct", type=float, default=3.0)
    args = parser.parse_args()

    result = compare(json.loads(args.base.read_text()), json.loads(args.candidate.read_text()), args.promotion_threshold_pct)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    args.output_md.write_text(_markdown(result))
    print(json.dumps(result["promotion_gate"]))


if __name__ == "__main__":
    main()
