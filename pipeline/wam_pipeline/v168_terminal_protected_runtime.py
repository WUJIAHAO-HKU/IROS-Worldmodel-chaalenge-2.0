"""Fixed V15.7/V16.6 world-model blend selected without policy action search."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .multisource_flow_unet_runtime import Track2MultiSourceFlowUNet
from .v15_gated_runtime import Track2V15GatedRuntime
from .v15_runtime import _active_arm


FORMAT = "track2-v16.8-terminal-protected-release-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Track2V168TerminalProtectedRuntime:
    """Keep V15.7 everywhere except a smooth right-arm t+3..t+6 residual.

    The blend profile is immutable and depends only on which half of the
    submitted action tensor is active.  It never generates, scores, selects,
    rescales, or otherwise changes a policy action.
    """

    def __init__(self, checkpoint_dir: str | Path, library_dir: str | Path, device: str = "cuda") -> None:
        root = Path(checkpoint_dir).resolve()
        manifest_path = root / "v168_terminal_protected_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT:
            raise RuntimeError("unsupported V16.8 terminal-protected release")
        profile = manifest.get("profile", {})
        expected = [0.0, 0.0, 0.03661165, 0.125, 0.125, 0.03661165, 0.0, 0.0]
        actual = [float(value) for value in profile.get("right_horizon_alpha", [])]
        if len(actual) != 8 or not np.allclose(actual, expected, atol=1e-8, rtol=0.0):
            raise RuntimeError("V16.8 release does not contain the admitted fixed horizon profile")
        if float(profile.get("left_alpha", -1.0)) != 0.0:
            raise RuntimeError("V16.8 release must preserve the complete left-arm V15.7 prediction")

        base = (root / manifest["base_release"]).resolve()
        student = (root / manifest["student_release"]).resolve()
        for relative, expected_hash in manifest["sha256"].items():
            path = (root / relative).resolve()
            if not path.is_file() or _sha256(path) != expected_hash:
                raise RuntimeError(f"V16.8 release artifact hash mismatch: {path}")
        self.base = Track2V15GatedRuntime(base, library_dir, device)
        self.student = Track2MultiSourceFlowUNet(student, device)
        self.alpha = np.asarray(actual, dtype=np.float32).reshape(8, 1, 1, 1)

    def _blend(self, baseline: np.ndarray, student: np.ndarray) -> np.ndarray:
        mixed = baseline.astype(np.float32) + self.alpha * (
            student.astype(np.float32) - baseline.astype(np.float32)
        )
        return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        baseline = self.base.predict(
            context_frames, history_actions, future_actions, seed, instruction
        )
        arm, _ = _active_arm(history_actions, future_actions)
        if arm == 0:
            return baseline
        student = self.student.predict(
            context_frames, history_actions, future_actions, seed, instruction
        )
        return self._blend(baseline, student)

    def predict_batch(self, context_frames, history_actions, future_actions, seeds, instructions):
        baseline = self.base.predict_batch(
            context_frames, history_actions, future_actions, seeds, instructions
        )
        arms = np.asarray(
            [
                _active_arm(history_actions[index], future_actions[index])[0]
                for index in range(len(baseline))
            ],
            dtype=np.int64,
        )
        right = arms == 1
        if not right.any():
            return baseline
        student = self.student.predict_batch(
            context_frames[right],
            history_actions[right],
            future_actions[right],
            np.asarray(seeds)[right],
            [instructions[index] for index in np.flatnonzero(right)],
        )
        output = baseline.copy()
        output[right] = np.stack(
            [self._blend(base, candidate) for base, candidate in zip(baseline[right], student)]
        )
        return output
