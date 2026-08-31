"""Frozen source-routed, blended Cartesian phase world model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .v337_public_recursive_ood_gate import PublicRecursiveOODGate
from .v375_bounded_cartesian_phase_runtime import Track2V375BoundedCartesianPhase


MANIFEST_FORMAT = "strict-track2-v378-source-routed-blended-cartesian-release-v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Track2V378SourceRoutedBlendedCartesian(Track2V375BoundedCartesianPhase):
    """Return v355 on real-looking contexts; blend v375 on generated contexts."""

    def __init__(self, checkpoint_dir, library_index, device="cuda"):
        super().__init__(checkpoint_dir, library_index, device)
        manifest = json.loads((self.root / "source_routed_blend_manifest.json").read_text())
        if manifest.get("format") != MANIFEST_FORMAT:
            raise RuntimeError("unsupported v378 source routing manifest")
        gate_path = self.root / manifest["source_gate"]
        if not gate_path.is_file() or _sha256(gate_path) != manifest["source_gate_sha256"]:
            raise RuntimeError("v378 source gate hash mismatch")
        if manifest.get("reward_or_outcomes_used") is not False:
            raise RuntimeError("v378 routing must not use reward or outcomes")
        self.source_gate = PublicRecursiveOODGate(gate_path)
        self.source_threshold = float(manifest["source_threshold"])
        self.cartesian_alpha = float(manifest["cartesian_alpha"])
        if not 0.0 < self.source_threshold < 1.0:
            raise RuntimeError("invalid v378 source threshold")
        if not 0.0 < self.cartesian_alpha <= 1.0:
            raise RuntimeError("invalid v378 Cartesian alpha")
        self.last_source_probability = None
        self.last_cartesian_route = False
        self.last_batch_source_probabilities = []
        self.last_batch_cartesian_routes = []

    def _route(self, context, arm):
        probability = float(self.source_gate.probability(context))
        return bool(arm == "right" and probability >= self.source_threshold), probability

    def _blend_cartesian(self, baseline, context, history, future):
        cartesian = self._right_prediction(baseline, context, history, future)
        return np.clip(np.rint(
            baseline.astype(np.float32)
            + self.cartesian_alpha * (cartesian.astype(np.float32) - baseline.astype(np.float32))
        ), 0, 255).astype(np.uint8)

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        baseline = self.parent.predict(context_frames, history_actions, future_actions, seed, instruction)
        arm = self.parent.active_arm(history_actions, future_actions, instruction or "")
        accepted, probability = self._route(context_frames, arm)
        self.last_arm_route = arm
        self.last_source_probability = probability
        self.last_cartesian_route = accepted
        self.last_batch_source_probabilities = [probability]
        self.last_batch_cartesian_routes = [accepted]
        if not accepted:
            self.last_base_index = None; self.last_selected_frames = None
            return baseline
        return self._blend_cartesian(baseline, context_frames, history_actions, future_actions)

    def predict_batch(self, context_frames, history_actions, future_actions, seeds, instructions):
        baseline = self.parent.predict_batch(context_frames, history_actions, future_actions, seeds, instructions)
        arms = [self.parent.active_arm(h, f, t or "") for h, f, t in zip(history_actions, future_actions, instructions, strict=True)]
        decisions = [self._route(c, a) for c, a in zip(context_frames, arms, strict=True)]
        output = baseline.copy()
        for i, (accepted, _) in enumerate(decisions):
            if accepted:
                output[i] = self._blend_cartesian(baseline[i], context_frames[i], history_actions[i], future_actions[i])
        self.last_arm_route = "mixed" if len(set(arms)) > 1 else arms[0]
        self.last_batch_source_probabilities = [x[1] for x in decisions]
        self.last_batch_cartesian_routes = [x[0] for x in decisions]
        self.last_source_probability = decisions[0][1] if len(decisions) == 1 else None
        self.last_cartesian_route = decisions[0][0] if len(decisions) == 1 else any(x[0] for x in decisions)
        return output
