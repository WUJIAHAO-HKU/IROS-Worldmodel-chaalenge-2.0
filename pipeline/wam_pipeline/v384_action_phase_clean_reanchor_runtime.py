"""Action-phase-qualified clean terminal reanchor for generated right contexts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .v326_blended_phase_terminal_runtime import Track2V326BlendedPhaseTerminal
from .v324_phase_guarded_terminal_runtime import ACTION_PROBABILITY_MIN
from .v337_public_recursive_ood_gate import PublicRecursiveOODGate


FORMAT = "strict-track2-v384-action-phase-clean-reanchor-release-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Track2V384ActionPhaseCleanReanchor(Track2V326BlendedPhaseTerminal):
    """Reanchor only causal, post-grasp, generated-context right-arm requests."""

    def __init__(self, checkpoint_dir, library_index, device="cuda"):
        super().__init__(checkpoint_dir, library_index, device)
        manifest = json.loads((self.root / "action_phase_clean_reanchor_manifest.json").read_text())
        if manifest.get("format") != FORMAT:
            raise RuntimeError("unsupported v384 manifest")
        gate_path = self.root / manifest["source_gate"]
        if not gate_path.is_file() or sha256(gate_path) != manifest["source_gate_sha256"]:
            raise RuntimeError("v384 source gate hash mismatch")
        if manifest.get("reward_or_outcomes_used") is not False:
            raise RuntimeError("v384 runtime data boundary violation")
        self.source_gate = PublicRecursiveOODGate(gate_path)
        self.source_threshold = float(manifest["source_threshold"])
        self.last_source_probability = None
        self.last_clean_reanchor = False
        self.last_action_phase_row = None
        self.last_clean_terminal_episode = None

    def _action_phase_base(self, history: np.ndarray, future: np.ndarray) -> int:
        actions = np.concatenate((history, future), axis=0).astype(np.float32)
        query = ((actions - self.action_mean) / self.action_std).reshape(-1)
        rows = self.clean_rows
        distance = ((self.action[rows] - query) ** 2).mean(axis=1)
        return int(rows[int(np.argmin(distance))])

    def _endpoint_terminal(self, future: np.ndarray) -> tuple[int, int]:
        scale = np.maximum(self.action_std[7:13], 1e-6)
        distance = (((self.success_terminal_endpoints - future[-1, 7:13]) / scale) ** 2).mean(axis=1)
        index = int(np.argmin(distance))
        return int(self.success_terminal_rows[index]), int(self.success_episodes[index])

    def _route(self, context, history, future, arm):
        source_probability = float(self.source_gate.probability(context))
        if arm != "right" or source_probability < self.source_threshold:
            return False, source_probability, None
        probability = float(self._probability(history, future))
        if not self._post_grasp(history, future) or probability < ACTION_PROBABILITY_MIN:
            return False, source_probability, None
        if self._signature(history, future, probability) is not None:
            return False, source_probability, None
        base = self._action_phase_base(history, future)
        phase_ready, _, _, _ = self._phase(base)
        return bool(phase_ready), source_probability, base

    def predict(self, context, history, future, seed, instruction):
        baseline = super().predict(context, history, future, seed, instruction)
        arm = self.parent.active_arm(history, future, instruction or "")
        route, source_probability, phase_row = self._route(context, history, future, arm)
        self.last_source_probability = source_probability
        self.last_clean_reanchor = route
        self.last_action_phase_row = phase_row
        self.last_clean_terminal_episode = None
        if not route:
            return baseline
        target, episode = self._endpoint_terminal(future)
        self.last_clean_terminal_episode = episode
        self.last_retrieval_index = target
        return self._target(target).copy()

    def predict_batch(self, contexts, histories, futures, seeds, instructions):
        return np.stack([
            self.predict(context, history, future, int(seed), instruction)
            for context, history, future, seed, instruction in zip(
                contexts, histories, futures, seeds, instructions, strict=True
            )
        ], axis=0)
