"""Exact v420 output plus outcome-free v400/v420 final-frame telemetry."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .v245_clean_progressive_successor_runtime import (
    Track2V245CleanProgressiveSuccessor,
)
from .v400_supported_posterior_blend_runtime import (
    Track2V400SupportedPosteriorBlend,
)
from .v409_half_contracted_progressive_runtime import CONTRACTION_SCALE
from .v420_action_phase_progressive_runtime import (
    Track2V420ActionPhaseProgressive,
)


class Track2V421V420RewardTrace(Track2V420ActionPhaseProgressive):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        value = os.environ.get("WAM_V421_TRACE_DIR", "")
        if not value:
            raise RuntimeError("WAM_V421_TRACE_DIR required")
        self.trace_dir = Path(value)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        if any(self.trace_dir.glob("batch_*")):
            raise RuntimeError("v421 trace directory is not empty")
        self.route_path = self.trace_dir / "route_trace.jsonl"
        self._batch_index = 0
        self._trace_records = []
        self._pre_terminal_frames = []

    def _apply(self, baseline, context, history, future, arm):
        terminal, terminal_route, source_probability, phase_row, episode = (
            Track2V400SupportedPosteriorBlend._apply(
                self, baseline, context, history, future, arm
            )
        )
        self._pre_terminal_frames.append(terminal[-1].copy())
        self.last_v406_terminal_precedence = bool(terminal_route)
        self.last_v406_progressive_route = False
        self.last_v406_progressive_alpha = 0.0
        base = target = None
        contracted = 0.0
        action_probability = float(self._probability(history, future))
        signature = self._signature(history, future, action_probability)
        if terminal_route:
            result = (terminal, terminal_route, source_probability, phase_row, episode)
        elif (
            arm != "right"
            or source_probability < self.source_threshold
            or not self._post_grasp(history, future)
            or signature is not None
        ):
            result = (terminal, False, source_probability, phase_row, episode)
        else:
            base = self._action_phase_base(history, future)
            actions = np.concatenate((history, future), axis=0).astype(np.float32)
            query = ((actions - self.action_mean) / self.action_std).reshape(-1)
            distance = float(((self.action[base] - query) ** 2).mean())
            target = self._progressive_row(base)
            alpha = float(
                Track2V245CleanProgressiveSuccessor._alpha(
                    self, history, future, base, distance
                )
            )
            contracted = alpha * CONTRACTION_SCALE
            self.last_v406_progressive_route = bool(contracted > 0.0)
            self.last_v406_progressive_alpha = contracted
            self.last_base_index = base
            self.last_progressive_index = target
            self.last_progressive_alpha = contracted
            self.last_retrieval_index = target
            output = self._contracted_blend(
                terminal, self._target(target), future, alpha
            )
            result = (
                output,
                bool(contracted > 0.0),
                source_probability,
                base,
                int(self.row_episode[target]),
            )
        self._trace_records.append(
            {
                "arm": arm,
                "source_probability": float(source_probability),
                "post_grasp": bool(self._post_grasp(history, future)),
                "action_probability": action_probability,
                "failure_signature": signature,
                "terminal_precedence": bool(terminal_route),
                "progressive_route": bool(contracted > 0.0),
                "contracted_alpha": float(contracted),
                "phase_base_row": int(base) if base is not None else None,
                "phase_base_start": int(self.row_start[base]) if base is not None else None,
                "progressive_target_row": int(target) if target is not None else None,
                "progressive_target_advance": (
                    int(self.row_start[target] - self.row_start[base])
                    if base is not None and target is not None else None
                ),
            }
        )
        return result

    def _flush(self, output, instructions):
        if len(self._trace_records) != len(output) or len(self._pre_terminal_frames) != len(output):
            raise RuntimeError("v421 trace alignment failure")
        stem = f"batch_{self._batch_index:05d}"
        np.save(self.trace_dir / f"{stem}_v400.npy", np.stack(self._pre_terminal_frames).astype(np.uint8), allow_pickle=False)
        np.save(self.trace_dir / f"{stem}_v420.npy", np.asarray(output[:, -1], dtype=np.uint8), allow_pickle=False)
        (self.trace_dir / f"{stem}_prompts.json").write_text(json.dumps(list(instructions), ensure_ascii=False) + "\n")
        with self.route_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"batch": self._trace_records}, separators=(",", ":")) + "\n")
        self._batch_index += 1

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        self._trace_records = []
        self._pre_terminal_frames = []
        output = super().predict(context_frames, history_actions, future_actions, seed, instruction)
        self._flush(output[None], [instruction])
        return output

    def predict_batch(self, context_frames, history_actions, future_actions, seeds, instructions):
        self._trace_records = []
        self._pre_terminal_frames = []
        output = super().predict_batch(context_frames, history_actions, future_actions, seeds, instructions)
        self._flush(output, instructions)
        return output
