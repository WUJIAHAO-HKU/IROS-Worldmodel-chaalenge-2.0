#!/usr/bin/env python3
"""Diagnose public v214 kNN false positives without changing the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def episode(path: str) -> int:
    return int(Path(path).stem.split("_")[0][7:])


def parse_name(name: str) -> tuple[int, int]:
    stem = Path(name).stem
    episode_text, start_text = stem.split("_")
    return int(episode_text[7:]), int(start_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--query-windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.run / "audit/public_failure_baseline.npz", allow_pickle=False) as values:
        paths = values["path"].astype(str)
        right = values["arm_right"].astype(bool)
    rewards = json.loads((args.run / "audit/public_failure_reward.json").read_text())
    scores = np.asarray(rewards["raw_scores"]["candidate"], dtype=np.float64)
    target_scores = np.asarray(rewards["raw_scores"]["target"], dtype=np.float64)
    manifest = json.loads((args.run / "audit/public_failure_candidate.manifest.json").read_text())
    selections = manifest["selections"]
    index = {parse_name(path.name): path for path in args.query_windows.glob("episode*_*.npz")}
    rows = []
    for query_index in np.flatnonzero(right):
        name = paths[query_index]
        query_episode, start = parse_name(name)
        actions = []
        for chunk in range(16):
            with np.load(index[(query_episode, start + 8 * chunk)], allow_pickle=False) as values:
                actions.append(values["future_actions"].astype(np.float32))
        actions = np.concatenate(actions, 0)
        selected = selections[query_index]
        selected_keys = [parse_name(Path(item["library_path"]).name) for item in selected]
        continuation_fraction = float(
            np.mean(
                [
                    selected_keys[index][0] == selected_keys[index - 1][0]
                    and selected_keys[index][1] == selected_keys[index - 1][1] + 8
                    for index in range(1, len(selected_keys))
                ]
            )
        )
        selected_success = []
        for item in selected:
            with np.load(item["library_path"], allow_pickle=False) as values:
                # The declared public demonstration pack predates this audit
                # field; those episode-offset windows are all successful
                # demonstrations, matching MultiChunkWindows' documented
                # neutral/default interpretation.
                selected_success.append(
                    bool(values["capture_success"])
                    if "capture_success" in values.files
                    else episode(item["library_path"]) >= 20000
                )
        peak = float(scores[query_index].max())
        peak_frame = int(scores[query_index].argmax())
        peak_chunk = peak_frame // 8
        rows.append(
            {
                "path": name,
                "peak_reward": peak,
                "peak_frame": peak_frame,
                "target_peak_reward": float(target_scores[query_index].max()),
                "target_threshold_hit": bool(target_scores[query_index].max() >= 0.9),
                "false_positive": peak >= 0.9,
                "right_close_fraction": float((actions[:, 13] < 0.5).mean()),
                "right_gripper_min": float(actions[:, 13].min()),
                "right_gripper_terminal": float(actions[-1, 13]),
                "selected_public_demo_fraction": float(np.mean([episode(item["library_path"]) >= 20000 for item in selected])),
                "selected_capture_success_fraction": float(np.mean(selected_success)),
                "selected_action_distance_mean": float(np.mean([item["action_distance"] for item in selected])),
                "selected_visual_distance_mean": float(np.mean([item["visual_distance"] for item in selected])),
                "retrieval_continuation_fraction": continuation_fraction,
                "peak_library_path": selected[peak_chunk]["library_path"],
                "peak_library_capture_success": selected_success[peak_chunk],
                "selected_paths": [item["library_path"] for item in selected],
            }
        )
    report = {
        "format": "strict-track2-v214-public-false-positive-diagnostic-v1",
        "rows": rows,
        "aggregate": {},
        "public_data_only": True,
        "model_selection": False,
        "hidden_or_final_data": False,
    }
    for key, subset in (("false_positive", [row for row in rows if row["false_positive"]]), ("true_negative", [row for row in rows if not row["false_positive"]])):
        report["aggregate"][key] = {
            "count": len(subset),
            "right_close_fraction_mean": float(np.mean([row["right_close_fraction"] for row in subset])),
            "selected_public_demo_fraction_mean": float(np.mean([row["selected_public_demo_fraction"] for row in subset])),
            "selected_capture_success_fraction_mean": float(np.mean([row["selected_capture_success_fraction"] for row in subset])),
            "selected_action_distance_mean": float(np.mean([row["selected_action_distance_mean"] for row in subset])),
            "selected_visual_distance_mean": float(np.mean([row["selected_visual_distance_mean"] for row in subset])),
            "retrieval_continuation_fraction_mean": float(np.mean([row["retrieval_continuation_fraction"] for row in subset])),
            "target_threshold_hit_rate": float(np.mean([row["target_threshold_hit"] for row in subset])),
            "peak_library_capture_success_rate": float(np.mean([row["peak_library_capture_success"] for row in subset])),
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
