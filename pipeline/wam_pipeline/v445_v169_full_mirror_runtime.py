"""V445: exact full-coordinate mirror adapter around frozen v169."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .v169_arm_routed_runtime import Track2V169ArmRoutedRuntime


FORMAT = "track2-v445-v169-full-mirror-release-v1"
MIRROR_SIGN = np.asarray([-1, 1, 1, 1, -1, -1, 1], dtype=np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def explicit_right(instruction: str | None) -> bool:
    text = str(instruction or "").lower()
    return "right arm" in text and "left arm" not in text


def mirror_prompt(instruction: str) -> str:
    if not explicit_right(instruction):
        raise ValueError("v445 only mirrors explicit-right prompts")
    return re.sub(r"right arm", "left arm", str(instruction), flags=re.IGNORECASE)


def mirror_joint14(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values)
    if source.shape[-1] != 14:
        raise ValueError(f"v445 expected joint14, got {source.shape}")
    output = np.empty_like(source, dtype=np.float32)
    output[..., :7] = source[..., 7:14] * MIRROR_SIGN
    output[..., 7:14] = source[..., :7] * MIRROR_SIGN
    return output


def mirror_rgb(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values)
    if source.ndim < 3 or source.shape[-1] != 3:
        raise ValueError(f"v445 expected channel-last RGB, got {source.shape}")
    return np.flip(source, axis=-2).copy()


class Track2V445V169FullMirror:
    def __init__(self, release_dir: str | Path, device: str = "cuda") -> None:
        root = Path(release_dir).resolve()
        manifest_path = root / "v445_full_mirror_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != FORMAT or manifest.get("official_reward_runtime_used") is not False:
            raise RuntimeError("unsupported or reward-coupled v445 release")
        v169_manifest = root / manifest["v169_release"] / "v169_arm_routed_manifest.json"
        if not v169_manifest.is_file() or _sha256(v169_manifest) != manifest["sha256"]["v169_manifest"]:
            raise RuntimeError("v445 v169 manifest hash mismatch")
        self.v169 = Track2V169ArmRoutedRuntime(
            root / manifest["v169_release"], root / manifest["v169_library"], device
        )
        self.last_decisions: list[dict] = []

    @staticmethod
    def gate_decision(history_actions, future_actions, instruction):
        del history_actions, future_actions
        gate = explicit_right(instruction)
        return {"gate": gate, "reason": "explicit_right_full_mirror" if gate else "v169_bitexact_fallback"}

    def predict_batch_with_baseline(self, context_frames, history_actions, future_actions, seeds, instructions):
        context = np.asarray(context_frames)
        history = np.asarray(history_actions)
        future = np.asarray(future_actions)
        baseline = self.v169.predict_batch(context, history, future, seeds, instructions)
        enabled = np.asarray([explicit_right(text) for text in instructions], dtype=bool)
        output = baseline.copy()
        if enabled.any():
            indices = np.flatnonzero(enabled)
            mirrored = self.v169.predict_batch(
                mirror_rgb(context[enabled]),
                mirror_joint14(history[enabled]),
                mirror_joint14(future[enabled]),
                np.asarray(seeds)[enabled],
                [mirror_prompt(instructions[index]) for index in indices],
            )
            output[enabled] = mirror_rgb(mirrored)
        self.last_decisions = [self.gate_decision(h, f, text) for h, f, text in zip(history, future, instructions)]
        return baseline, output, self.last_decisions

    def predict_batch(self, context_frames, history_actions, future_actions, seeds, instructions):
        return self.predict_batch_with_baseline(context_frames, history_actions, future_actions, seeds, instructions)[1]

    def predict_with_baseline(self, context_frames, history_actions, future_actions, seed, instruction):
        baseline, output, decisions = self.predict_batch_with_baseline(
            np.asarray(context_frames)[None], np.asarray(history_actions)[None], np.asarray(future_actions)[None],
            np.asarray([seed]), [instruction],
        )
        return baseline[0], output[0], decisions[0]

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        return self.predict_with_baseline(context_frames, history_actions, future_actions, seed, instruction)[1]
