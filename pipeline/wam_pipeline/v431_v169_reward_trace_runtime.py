"""Bit-exact v169 output with outcome-free final-frame and arm-route telemetry."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .v169_arm_routed_runtime import Track2V169ArmRoutedRuntime


class Track2V431V169RewardTrace(Track2V169ArmRoutedRuntime):
    """Trace the original v169 backend without changing any predicted pixels."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        library_dir: str | Path,
        device: str = "cuda",
    ) -> None:
        super().__init__(checkpoint_dir, library_dir, device)
        value = os.environ.get("WAM_V431_TRACE_DIR", "")
        if not value:
            raise RuntimeError("WAM_V431_TRACE_DIR required")
        self.trace_dir = Path(value)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        if any(self.trace_dir.iterdir()):
            raise RuntimeError("v431 trace directory is not empty")
        self.route_path = self.trace_dir / "route_trace.jsonl"
        self._batch_index = 0

    def _record(
        self,
        context_frames: np.ndarray,
        history_actions: np.ndarray,
        future_actions: np.ndarray,
        instruction: str | None,
    ) -> dict:
        right_probability = self.router.probability_right(
            context_frames, history_actions, future_actions
        )
        explicit = self.router.arm_from_instruction(instruction)
        routed_right = int(explicit) if explicit is not None else int(right_probability >= 0.5)
        return {
            "arm": "right" if routed_right else "left",
            "route": "student_blend" if routed_right else "base",
            "explicit_instruction_route": explicit is not None,
            "right_probability": float(right_probability),
        }

    def _flush(
        self,
        candidate: np.ndarray,
        instructions: list[str | None],
        records: list[dict],
    ) -> None:
        if len(records) != len(candidate) or len(instructions) != len(candidate):
            raise RuntimeError("v431 trace alignment failure")
        stem = f"batch_{self._batch_index:05d}"
        np.save(
            self.trace_dir / f"{stem}_v169.npy",
            np.asarray(candidate[:, -1], dtype=np.uint8),
            allow_pickle=False,
        )
        (self.trace_dir / f"{stem}_prompts.json").write_text(
            json.dumps(list(instructions), ensure_ascii=False) + "\n"
        )
        with self.route_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"batch": records}, separators=(",", ":")) + "\n")
        self._batch_index += 1

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        record = self._record(context_frames, history_actions, future_actions, instruction)
        candidate = super().predict(
            context_frames, history_actions, future_actions, seed, instruction
        )
        self._flush(candidate[None], [instruction], [record])
        return candidate

    def predict_batch(
        self, context_frames, history_actions, future_actions, seeds, instructions
    ):
        records = [
            self._record(context, history, future, instruction)
            for context, history, future, instruction in zip(
                context_frames, history_actions, future_actions, instructions
            )
        ]
        candidate = super().predict_batch(
            context_frames, history_actions, future_actions, seeds, instructions
        )
        self._flush(candidate, list(instructions), records)
        return candidate
