"""Use Cartesian phase correction only on visually corrupted recursive contexts.

The default path is the frozen v355 arm-routed parametric world model.  The
public-train v339 image-quality classifier is evaluated solely on the five RGB
context frames.  Only a right-arm request classified as recursively corrupted
is routed through the frozen v375 Cartesian phase renderer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .v337_public_recursive_ood_gate import PublicRecursiveOODGate
from .v375_bounded_cartesian_phase_runtime import Track2V375BoundedCartesianPhase


MANIFEST_FORMAT = "strict-track2-v376-ood-routed-cartesian-phase-release-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Track2V376OODRoutedCartesianPhase(Track2V375BoundedCartesianPhase):
    """Preserve v355 exactly unless a frozen visual-only OOD gate fires."""

    def __init__(self, checkpoint_dir, library_index, device="cuda"):
        super().__init__(checkpoint_dir, library_index, device)
        manifest_path = self.root / "ood_routed_cartesian_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != MANIFEST_FORMAT:
            raise RuntimeError("unsupported v376 OOD routing manifest")
        gate_path = self.root / manifest["recursive_ood_gate"]
        if not gate_path.is_file() or _sha256(gate_path) != manifest["recursive_ood_gate_sha256"]:
            raise RuntimeError("v376 recursive OOD gate hash mismatch")
        if manifest.get("reward_or_outcomes_used") is not False:
            raise RuntimeError("v376 routing must not use reward or outcomes")
        self.recursive_ood_gate = PublicRecursiveOODGate(gate_path)
        expected = float(manifest["threshold"])
        if abs(self.recursive_ood_gate.threshold - expected) > 1e-7:
            raise RuntimeError("v376 recursive OOD threshold drift")
        self.last_recursive_ood_probability = None
        self.last_cartesian_route = False
        self.last_batch_ood_probabilities: list[float] = []
        self.last_batch_cartesian_routes: list[bool] = []

    def _route(self, context: np.ndarray, arm: str) -> tuple[bool, float]:
        probability = float(self.recursive_ood_gate.probability(context))
        accepted = bool(
            arm == "right" and probability >= self.recursive_ood_gate.threshold
        )
        return accepted, probability

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        baseline = self.parent.predict(
            context_frames, history_actions, future_actions, seed, instruction
        )
        arm = self.parent.active_arm(history_actions, future_actions, instruction or "")
        accepted, probability = self._route(context_frames, arm)
        self.last_arm_route = arm
        self.last_recursive_ood_probability = probability
        self.last_cartesian_route = accepted
        self.last_batch_ood_probabilities = [probability]
        self.last_batch_cartesian_routes = [accepted]
        if not accepted:
            self.last_base_index = None
            self.last_selected_frames = None
            return baseline
        return self._right_prediction(
            baseline, context_frames, history_actions, future_actions
        )

    def predict_batch(
        self, context_frames, history_actions, future_actions, seeds, instructions
    ):
        baseline = self.parent.predict_batch(
            context_frames, history_actions, future_actions, seeds, instructions
        )
        arms = [
            self.parent.active_arm(history, future, instruction or "")
            for history, future, instruction in zip(
                history_actions, future_actions, instructions, strict=True
            )
        ]
        decisions = [
            self._route(context, arm)
            for context, arm in zip(context_frames, arms, strict=True)
        ]
        output = baseline.copy()
        for index, ((accepted, _), arm) in enumerate(
            zip(decisions, arms, strict=True)
        ):
            if accepted:
                output[index] = self._right_prediction(
                    baseline[index],
                    context_frames[index],
                    history_actions[index],
                    future_actions[index],
                )
        self.last_arm_route = "mixed" if len(set(arms)) > 1 else arms[0]
        self.last_batch_ood_probabilities = [value[1] for value in decisions]
        self.last_batch_cartesian_routes = [value[0] for value in decisions]
        self.last_recursive_ood_probability = (
            decisions[0][1] if len(decisions) == 1 else None
        )
        self.last_cartesian_route = (
            decisions[0][0] if len(decisions) == 1 else any(value[0] for value in decisions)
        )
        return output
