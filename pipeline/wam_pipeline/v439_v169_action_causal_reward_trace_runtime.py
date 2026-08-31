"""Outcome-free paired RGB telemetry for the v439 hybrid and v169 baseline."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .v439_v169_action_causal_projection_runtime import (
    Track2V439V169ActionCausalProjection,
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class Track2V439V169ActionCausalRewardTrace:
    """Return hybrid RGB while recording its same-request v169 counterfactual."""

    def __init__(self, release_dir: str | Path, device: str = "cuda") -> None:
        self.runtime = Track2V439V169ActionCausalProjection(release_dir, device)
        trace_value = os.environ.get("WAM_V439_TRACE_DIR", "")
        if not trace_value:
            raise RuntimeError("WAM_V439_TRACE_DIR required")
        self.trace_dir = Path(trace_value)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        if any(self.trace_dir.iterdir()):
            raise RuntimeError("v439 trace directory is not empty")
        self.route_path = self.trace_dir / "route_trace.jsonl"
        self.index = 0

    def _flush(self, baseline, hybrid, prompts, decisions) -> None:
        baseline_array = np.asarray(baseline, dtype=np.uint8)
        hybrid_array = np.asarray(hybrid, dtype=np.uint8)
        if (
            baseline_array.shape != hybrid_array.shape
            or baseline_array.ndim != 5
            or baseline_array.shape[1:] != (8, 256, 256, 3)
            or len(prompts) != len(baseline_array)
            or len(decisions) != len(baseline_array)
        ):
            raise RuntimeError("v439 paired trace alignment failure")
        stem = f"batch_{self.index:05d}"
        np.save(
            self.trace_dir / f"{stem}_v169.npy",
            baseline_array[:, -1],
            allow_pickle=False,
        )
        np.save(
            self.trace_dir / f"{stem}_hybrid.npy",
            hybrid_array[:, -1],
            allow_pickle=False,
        )
        (self.trace_dir / f"{stem}_prompts.json").write_text(
            json.dumps(list(prompts), ensure_ascii=False) + "\n"
        )
        with self.route_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"batch": [_jsonable(value) for value in decisions]},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
        self.index += 1

    def predict_batch(
        self, context_frames, history_actions, future_actions, seeds, instructions
    ):
        baseline, hybrid, decisions = self.runtime.predict_batch_with_baseline(
            context_frames,
            history_actions,
            future_actions,
            seeds,
            instructions,
        )
        self._flush(baseline, hybrid, list(instructions), decisions)
        return hybrid

    def predict(
        self, context_frames, history_actions, future_actions, seed, instruction
    ):
        baseline, hybrid, decision = self.runtime.predict_with_baseline(
            context_frames,
            history_actions,
            future_actions,
            seed,
            instruction,
        )
        self._flush(
            np.asarray(baseline)[None],
            np.asarray(hybrid)[None],
            [instruction],
            [decision],
        )
        return hybrid
