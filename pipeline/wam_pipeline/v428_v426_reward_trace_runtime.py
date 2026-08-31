"""Exact v426 output plus frozen-v209/candidate final-frame telemetry."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .v426_motion_guarded_right_runtime import Track2V426MotionGuardedRight


class Track2V428V426RewardTrace(Track2V426MotionGuardedRight):
    def __init__(self, release_dir: str | Path, device: str = "cuda") -> None:
        root = Path(release_dir)
        manifest = json.loads((root / "motion_guarded_right_manifest.json").read_text())
        super().__init__(
            root / manifest["left_expert"],
            root / manifest["learned_right_expert"],
            root / manifest["frozen_right_expert"],
            device,
            float(manifest["right_normalized_motion_max"]),
        )
        value = os.environ.get("WAM_V428_TRACE_DIR", "")
        if not value:
            raise RuntimeError("WAM_V428_TRACE_DIR required")
        self.trace_dir = Path(value)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        if any(self.trace_dir.glob("batch_*")):
            raise RuntimeError("v428 trace directory is not empty")
        self.route_path = self.trace_dir / "route_trace.jsonl"
        self._batch_index = 0

    def _record(self, history: np.ndarray, future: np.ndarray, instruction: str) -> dict:
        arm = self.active_arm(history, future, instruction)
        motion = self.right_motion(history, future) if arm == "right" else None
        route = "left" if arm == "left" else "learned_right" if motion <= self.right_motion_max else "frozen_right"
        return {
            "arm": arm,
            "route": route,
            "learned_right_route": route == "learned_right",
            "right_normalized_motion": motion,
            "right_motion_max": self.right_motion_max,
        }

    def _flush(self, baseline: np.ndarray, candidate: np.ndarray, instructions: list[str], records: list[dict]) -> None:
        if len(baseline) != len(candidate) or len(records) != len(candidate):
            raise RuntimeError("v428 trace alignment failure")
        stem = f"batch_{self._batch_index:05d}"
        np.save(self.trace_dir / f"{stem}_v209.npy", np.asarray(baseline[:, -1], dtype=np.uint8), allow_pickle=False)
        np.save(self.trace_dir / f"{stem}_v426.npy", np.asarray(candidate[:, -1], dtype=np.uint8), allow_pickle=False)
        (self.trace_dir / f"{stem}_prompts.json").write_text(json.dumps(list(instructions), ensure_ascii=False) + "\n")
        with self.route_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"batch": records}, separators=(",", ":")) + "\n")
        self._batch_index += 1

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        record = self._record(history_actions, future_actions, instruction)
        candidate = super().predict(context_frames, history_actions, future_actions, seed, instruction)
        baseline = (
            self.frozen_right.predict(context_frames, history_actions, future_actions, seed, instruction)
            if record["learned_right_route"]
            else candidate.copy()
        )
        self._flush(baseline[None], candidate[None], [instruction], [record])
        return candidate

    def predict_batch(self, context_frames, history_actions, future_actions, seeds, instructions):
        records = [
            self._record(history, future, instruction)
            for history, future, instruction in zip(history_actions, future_actions, instructions)
        ]
        candidate = super().predict_batch(context_frames, history_actions, future_actions, seeds, instructions)
        baseline = candidate.copy()
        mask = np.asarray([record["learned_right_route"] for record in records], dtype=np.bool_)
        if mask.any():
            selected = np.flatnonzero(mask)
            baseline[mask] = self.frozen_right.predict_batch(
                context_frames[mask], history_actions[mask], future_actions[mask],
                np.asarray(seeds)[mask], [instructions[index] for index in selected],
            )
        self._flush(baseline, candidate, instructions, records)
        return candidate
