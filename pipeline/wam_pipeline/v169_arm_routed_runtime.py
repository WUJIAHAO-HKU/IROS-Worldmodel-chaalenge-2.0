"""V16.9 terminal-protected blend with a validated visual/action arm router."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .arm_router import LinearVisualActionArmRouter
from .v168_terminal_protected_runtime import Track2V168TerminalProtectedRuntime


FORMAT = "track2-v16.9-arm-routed-terminal-protected-release-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Track2V169ArmRoutedRuntime:
    """Route a fixed world-model expert; never score, select, or alter actions."""

    def __init__(self, checkpoint_dir: str | Path, library_dir: str | Path, device="cuda") -> None:
        root = Path(checkpoint_dir).resolve()
        manifest = json.loads(
            (root / "v169_arm_routed_manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("format") != FORMAT:
            raise RuntimeError("unsupported V16.9 release")
        for relative, expected in manifest["sha256"].items():
            path = (root / relative).resolve()
            if not path.is_file() or _sha256(path) != expected:
                raise RuntimeError(f"V16.9 release artifact hash mismatch: {path}")
        self.composite = Track2V168TerminalProtectedRuntime(
            root / manifest["v168_release"], library_dir, device
        )
        self.router = LinearVisualActionArmRouter(root / manifest["arm_router"])

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        baseline = self.composite.base.predict(
            context_frames, history_actions, future_actions, seed, instruction
        )
        if self.router.predict(context_frames, history_actions, future_actions, instruction) == 0:
            return baseline
        student = self.composite.student.predict(
            context_frames, history_actions, future_actions, seed, instruction
        )
        return self.composite._blend(baseline, student)

    def predict_batch(self, context_frames, history_actions, future_actions, seeds, instructions):
        baseline = self.composite.base.predict_batch(
            context_frames, history_actions, future_actions, seeds, instructions
        )
        route = np.asarray(
            [
                self.router.predict(
                    context_frames[index], history_actions[index], future_actions[index],
                    instructions[index],
                )
                for index in range(len(baseline))
            ],
            dtype=bool,
        )
        if not route.any():
            return baseline
        selected = np.flatnonzero(route)
        student = self.composite.student.predict_batch(
            context_frames[route],
            history_actions[route],
            future_actions[route],
            np.asarray(seeds)[route],
            [instructions[index] for index in selected],
        )
        output = baseline.copy()
        output[route] = np.stack(
            [self.composite._blend(base, candidate) for base, candidate in zip(baseline[route], student)]
        )
        return output
