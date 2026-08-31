"""Exact v409 output plus outcome-free v400/v409 final-frame telemetry."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .v245_clean_progressive_successor_runtime import (
    Track2V245CleanProgressiveSuccessor,
)
from .v324_phase_guarded_terminal_runtime import ACTION_PROBABILITY_MIN
from .v400_supported_posterior_blend_runtime import (
    Track2V400SupportedPosteriorBlend,
)
from .v409_half_contracted_progressive_runtime import (
    CONTRACTION_SCALE,
    Track2V409HalfContractedProgressive,
)


class Track2V414V409RewardTrace(Track2V409HalfContractedProgressive):
    """Capture reward-model inputs without reading reward or outcomes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        value = os.environ.get("WAM_V414_TRACE_DIR", "")
        if not value:
            raise RuntimeError("WAM_V414_TRACE_DIR required")
        self.trace_dir = Path(value)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        if any(self.trace_dir.glob("batch_*")):
            raise RuntimeError("v414 trace directory is not empty")
        self.route_path = self.trace_dir / "route_trace.jsonl"
        self._batch_index = 0
        self._trace_records: list[dict] = []
        self._pre_terminal_frames: list[np.ndarray] = []

    def _route(self, context, history, future, arm):
        source_probability = float(self.source_gate.probability(context))
        right = arm == "right"
        source_ready = source_probability >= self.source_threshold
        post_grasp = bool(self._post_grasp(history, future))
        action_probability = float(self._probability(history, future))
        probability_ready = action_probability >= ACTION_PROBABILITY_MIN
        signature = self._signature(history, future, action_probability)
        broad_eligible = bool(right and source_ready and post_grasp and signature is None)
        strict_eligible = bool(broad_eligible and probability_ready)
        phase_row = self._action_phase_base(history, future) if strict_eligible else None
        phase_probability = (
            float(self.continuous_phase_gate.probability(context, history, future))
            if strict_eligible
            else None
        )
        hard_phase_ready = False
        if strict_eligible:
            hard_phase_ready, _, _, _ = self._phase(phase_row)
        self.last_continuous_phase_probability = phase_probability
        self.last_hard_phase_ready = bool(hard_phase_ready and strict_eligible)
        supported = bool(
            strict_eligible
            and phase_probability is not None
            and phase_probability >= self.support_probability_min
        )
        self._trace_records.append(
            {
                "arm": arm,
                "right": right,
                "source_probability": source_probability,
                "source_ready": source_ready,
                "post_grasp": post_grasp,
                "action_probability": action_probability,
                "probability_ready": probability_ready,
                "failure_signature": signature,
                "broad_eligible": broad_eligible,
                "strict_eligible": strict_eligible,
                "phase_probability": phase_probability,
                "hard_phase_ready": bool(hard_phase_ready),
                "supported_v400_route": supported,
            }
        )
        continuous = bool(
            strict_eligible
            and phase_probability is not None
            and phase_probability >= self.continuous_phase_gate.threshold
        )
        return continuous, source_probability, phase_row

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
        base = None
        target = None
        contracted = 0.0
        if terminal_route:
            output = terminal
            result = (output, terminal_route, source_probability, phase_row, episode)
        elif (
            arm != "right"
            or source_probability < self.source_threshold
            or not self._post_grasp(history, future)
        ):
            output = terminal
            result = (output, False, source_probability, phase_row, episode)
        else:
            action_probability = float(self._probability(history, future))
            if self._signature(history, future, action_probability) is not None:
                output = terminal
                result = (output, False, source_probability, phase_row, episode)
            else:
                base, distance = self._nearest_clean(context, history, future)
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
        record = self._trace_records[-1]
        record.update(
            {
                "terminal_precedence": bool(terminal_route),
                "progressive_route": bool(contracted > 0.0),
                "contracted_alpha": float(contracted),
                "progressive_base_row": int(base) if base is not None else None,
                "progressive_target_row": int(target) if target is not None else None,
                "progressive_target_advance": (
                    int(self.row_start[target] - self.row_start[base])
                    if base is not None and target is not None
                    else None
                ),
            }
        )
        return result

    def _flush(self, output, instructions):
        if len(self._trace_records) != len(output) or len(self._pre_terminal_frames) != len(output):
            raise RuntimeError("v414 trace alignment failure")
        stem = f"batch_{self._batch_index:05d}"
        np.save(
            self.trace_dir / f"{stem}_v400.npy",
            np.stack(self._pre_terminal_frames).astype(np.uint8),
            allow_pickle=False,
        )
        np.save(
            self.trace_dir / f"{stem}_v409.npy",
            np.asarray(output[:, -1], dtype=np.uint8),
            allow_pickle=False,
        )
        (self.trace_dir / f"{stem}_prompts.json").write_text(
            json.dumps(list(instructions), ensure_ascii=False) + "\n"
        )
        with self.route_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"batch": self._trace_records}, separators=(",", ":")) + "\n")
        self._batch_index += 1

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        self._trace_records = []
        self._pre_terminal_frames = []
        output = super().predict(
            context_frames, history_actions, future_actions, seed, instruction
        )
        self._flush(output[None], [instruction])
        return output

    def predict_batch(self, context_frames, history_actions, future_actions, seeds, instructions):
        self._trace_records = []
        self._pre_terminal_frames = []
        output = super().predict_batch(
            context_frames, history_actions, future_actions, seeds, instructions
        )
        self._flush(output, instructions)
        return output
