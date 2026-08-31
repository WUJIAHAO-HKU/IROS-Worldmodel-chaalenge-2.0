#!/usr/bin/env python3
"""Measure public-library visual OOD drift under bridge-faithful v326 replay.

This is a read-only training-data diagnostic.  It evaluates all eight chunk
alignments of the four frozen public right-arm holdout episodes and records the
visual/action routes seen with teacher and recursively generated contexts.
No public evaluation outcomes, hidden data, or reward values are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_v310_full_mirror_causal_gate import RIGHT_EPISODES
from wam_pipeline.v216_public_knn_blend_runtime import _visual_descriptor
from wam_pipeline.v324_phase_guarded_terminal_runtime import ACTION_PROBABILITY_MIN
from wam_pipeline.v326_blended_phase_terminal_runtime import Track2V326BlendedPhaseTerminal


def seed_for(path: Path) -> int:
    return int.from_bytes(hashlib.sha256(path.name.encode()).digest()[:8], "little") % (2**31)


def route(runtime, context, history, future) -> dict:
    query_visual = _visual_descriptor(context[-1])
    visual_distance = ((runtime.visual[runtime.clean_rows] - query_visual) ** 2).mean(axis=1)
    vlocal = int(np.argmin(visual_distance))
    vrow = int(runtime.clean_rows[vlocal])

    actions = np.concatenate((history, future), axis=0).astype(np.float32)
    query_action = ((actions - runtime.action_mean) / runtime.action_std).reshape(-1)
    action_distance = ((runtime.action[runtime.clean_rows] - query_action) ** 2).mean(axis=1)
    alocal = int(np.argmin(action_distance))
    arow = int(runtime.clean_rows[alocal])

    combined, _ = runtime._nearest_clean(context, history, future)
    probability = float(runtime._probability(history, future))
    signature = runtime._signature(history, future, probability)

    def phase(row: int) -> dict:
        ready, episode, start, onset = runtime._phase(row)
        return {
            "row": row,
            "episode": episode,
            "start": start,
            "onset": onset,
            "offset": start - onset,
            "ready": bool(ready),
            "terminal_eligible": bool(runtime.eligible_episode[episode]),
        }

    return {
        "visual_min": float(visual_distance[vlocal]),
        "visual_median": float(np.median(visual_distance)),
        "visual_ratio": float(visual_distance[vlocal] / max(float(np.median(visual_distance)), 1e-9)),
        "action_min": float(action_distance[alocal]),
        "action_median": float(np.median(action_distance)),
        "action_ratio": float(action_distance[alocal] / max(float(np.median(action_distance)), 1e-9)),
        "probability": probability,
        "signature": signature,
        "post_grasp": bool(runtime._post_grasp(history, future)),
        "probability_gate": probability >= ACTION_PROBABILITY_MIN,
        "visual_phase": phase(vrow),
        "action_phase": phase(arow),
        "combined_phase": phase(combined),
    }


def stats(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(a)),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p90": float(np.quantile(a, 0.90)),
        "p95": float(np.quantile(a, 0.95)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(a.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("checkpoint-dir", "library-index", "action-gate", "phase-gate", "windows", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    runtime = Track2V326BlendedPhaseTerminal(
        args.checkpoint_dir, args.library_index, args.device, args.action_gate, args.phase_gate
    )
    states = []
    for split, episodes in RIGHT_EPISODES.items():
        for episode in episodes:
            available = {
                int(path.stem.split("_")[1]): path
                for path in args.windows.glob(f"episode{episode}_*.npz")
            }
            for alignment in range(8):
                if alignment not in available:
                    continue
                with np.load(available[alignment], allow_pickle=False) as values:
                    initial = values["context_frames"].astype(np.uint8)
                states.append({
                    "split": split,
                    "episode": episode,
                    "alignment": alignment,
                    "start": alignment,
                    "available": available,
                    "recursive_context": initial,
                })

    rows = []
    while True:
        active = [state for state in states if state["start"] in state["available"]]
        if not active:
            break
        requests = []
        for state in active:
            path = state["available"][state["start"]]
            with np.load(path, allow_pickle=False) as values:
                teacher_context = values["context_frames"].astype(np.uint8)
                history = values["history_actions"].astype(np.float32)
                future = values["future_actions"].astype(np.float32)
            requests.append({
                "state": state,
                "path": path,
                "teacher_context": teacher_context,
                "history": history,
                "future": future,
                "teacher_route": route(runtime, teacher_context, history, future),
                "recursive_route": route(runtime, state["recursive_context"], history, future),
            })
        predictions = []
        for begin in range(0, len(requests), args.batch_size):
            batch = requests[begin : begin + args.batch_size]
            predictions.extend(runtime.predict_batch(
                np.stack([item["state"]["recursive_context"] for item in batch]),
                np.stack([item["history"] for item in batch]),
                np.stack([item["future"] for item in batch]),
                np.asarray([seed_for(item["path"]) for item in batch], dtype=np.int64),
                ["Adjust bottle" for _ in batch],
            ))
        for request, prediction in zip(requests, predictions, strict=True):
            state = request["state"]
            rows.append({
                "key": f"{state['split']}/episode{state['episode']}/a{state['alignment']}/s{state['start']:05d}",
                "split": state["split"],
                "episode": state["episode"],
                "alignment": state["alignment"],
                "start": state["start"],
                "teacher": request["teacher_route"],
                "recursive": request["recursive_route"],
            })
            state["recursive_context"] = prediction[-5:].copy()
            state["start"] += 8

    summary = {}
    for split in RIGHT_EPISODES:
        selected = [row for row in rows if row["split"] == split]
        summary[split] = {
            context: {
                metric: stats([row[context][metric] for row in selected])
                for metric in ("visual_min", "visual_ratio", "action_min", "action_ratio")
            }
            for context in ("teacher", "recursive")
        }
    report = {
        "format": "strict-track2-v336-recursive-context-ood-diagnostic-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "declared_public_world_model_data_only": True,
        "public_evaluation_outcomes_read": False,
        "hidden_or_final_data": False,
        "real_submission": False,
        "rows_expected": 512,
        "rows_found": len(rows),
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "summary": summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
