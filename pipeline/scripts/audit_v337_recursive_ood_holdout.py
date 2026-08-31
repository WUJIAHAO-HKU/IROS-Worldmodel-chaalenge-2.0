#!/usr/bin/env python3
"""Validate the train-only recursive OOD gate on frozen public holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_v310_full_mirror_causal_gate import RIGHT_EPISODES, sha256
from wam_pipeline.v326_blended_phase_terminal_runtime import Track2V326BlendedPhaseTerminal
from wam_pipeline.v337_public_recursive_ood_gate import PublicRecursiveOODGate


CORRUPTION_MAE_FLOOR = 15.0


def seed_for(path: Path) -> int:
    return int.from_bytes(hashlib.sha256(path.name.encode()).digest()[:8], "little") % (2**31)


def metrics(rows: list[dict], split: str) -> dict:
    selected = [row for row in rows if row["split"] == split]
    teacher_predictions = np.asarray([row["teacher_probability"] >= row["threshold"] for row in selected])
    recursive_labels = np.asarray([row["recursive_label"] for row in selected], dtype=bool)
    recursive_predictions = np.asarray([row["recursive_probability"] >= row["threshold"] for row in selected])
    negative = ~recursive_labels
    return {
        "rows": len(selected),
        "corrupt_contexts": int(recursive_labels.sum()),
        "teacher_specificity": float((~teacher_predictions).mean()),
        "recursive_negative_specificity": float((~recursive_predictions[negative]).mean()),
        "corrupt_recall": float(recursive_predictions[recursive_labels].mean()),
        "teacher_probability_p99": float(np.quantile([row["teacher_probability"] for row in selected], 0.99)),
        "corrupt_probability_p10": float(np.quantile([row["recursive_probability"] for row in selected if row["recursive_label"]], 0.10)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "checkpoint-dir", "library-index", "action-gate", "phase-gate", "recursive-ood-gate",
        "windows", "training-report", "preregistration", "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v337-recursive-ood-holdout-preregistration-v1":
        raise RuntimeError("wrong v337 holdout preregistration")
    training = json.loads(args.training_report.read_text())
    if training.get("passed") is not True:
        raise RuntimeError("v337 train-only gate did not pass")
    gate = PublicRecursiveOODGate(args.recursive_ood_gate)
    if gate.corruption_mae_floor != CORRUPTION_MAE_FLOOR:
        raise RuntimeError("corruption floor drift")
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
                with np.load(available[alignment], allow_pickle=False) as payload:
                    initial = payload["context_frames"].astype(np.uint8)
                states.append({
                    "split": split, "episode": episode, "alignment": alignment,
                    "start": alignment, "available": available, "recursive_context": initial,
                })
    rows = []
    while True:
        active = [state for state in states if state["start"] in state["available"]]
        if not active:
            break
        requests = []
        for state in active:
            path = state["available"][state["start"]]
            with np.load(path, allow_pickle=False) as payload:
                teacher = payload["context_frames"].astype(np.uint8)
                history = payload["history_actions"].astype(np.float32)
                future = payload["future_actions"].astype(np.float32)
            recursive = state["recursive_context"]
            error = float(np.abs(recursive.astype(np.int16) - teacher.astype(np.int16)).mean())
            rows.append({
                "key": f"{state['split']}/episode{state['episode']}/a{state['alignment']}/s{state['start']:05d}",
                "split": state["split"], "episode": state["episode"],
                "alignment": state["alignment"], "start": state["start"],
                "recursive_context_rgb_mae": error,
                "recursive_label": bool(error >= CORRUPTION_MAE_FLOOR),
                "teacher_probability": gate.probability(teacher),
                "recursive_probability": gate.probability(recursive),
                "threshold": gate.threshold,
            })
            requests.append((state, path, history, future))
        predictions = []
        for begin in range(0, len(requests), args.batch_size):
            batch = requests[begin : begin + args.batch_size]
            predictions.extend(runtime.predict_batch(
                np.stack([item[0]["recursive_context"] for item in batch]),
                np.stack([item[2] for item in batch]),
                np.stack([item[3] for item in batch]),
                np.asarray([seed_for(item[1]) for item in batch], dtype=np.int64),
                ["Adjust bottle" for _ in batch],
            ))
        for (state, _, _, _), prediction in zip(requests, predictions, strict=True):
            state["recursive_context"] = prediction[-5:].copy()
            state["start"] += 8

    aggregate = {split: metrics(rows, split) for split in RIGHT_EPISODES}
    checks = {
        "exact_512_rows": len(rows) == 512,
        "validation_corrupt_contexts_ge_50": aggregate["validation"]["corrupt_contexts"] >= 50,
        "local_corrupt_contexts_ge_50": aggregate["local_test"]["corrupt_contexts"] >= 50,
        "validation_teacher_specificity_ge_0p98": aggregate["validation"]["teacher_specificity"] >= 0.98,
        "local_teacher_specificity_ge_0p98": aggregate["local_test"]["teacher_specificity"] >= 0.98,
        "validation_recursive_negative_specificity_ge_0p93": aggregate["validation"]["recursive_negative_specificity"] >= 0.93,
        "local_recursive_negative_specificity_ge_0p93": aggregate["local_test"]["recursive_negative_specificity"] >= 0.93,
        "validation_corrupt_recall_ge_0p65": aggregate["validation"]["corrupt_recall"] >= 0.65,
        "local_corrupt_recall_ge_0p65": aggregate["local_test"]["corrupt_recall"] >= 0.65,
    }
    report = {
        "format": "strict-track2-v337-recursive-ood-holdout-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "aggregates": aggregate,
        "checks": checks,
        "passed": all(checks.values()),
        "authorizes_candidate_design": all(checks.values()),
        "evidence_sha256": {
            "preregistration": sha256(args.preregistration),
            "training_report": sha256(args.training_report),
            "recursive_ood_gate": sha256(args.recursive_ood_gate),
        },
        "guards": {
            "public_world_model_holdout_only": True,
            "reward_or_success_outcomes_used": False,
            "official_batch16_outcomes_used": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"aggregates": aggregate, "checks": checks, "passed": report["passed"]}, indent=2), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
