"""V444 frozen-v169 direct residual learner and runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .v169_arm_routed_runtime import Track2V169ArmRoutedRuntime
from .v442_v169_close_aligned_projection_runtime import gate_decision


FORMAT = "track2-v444-v169-direct-residual-release-v1"
PROTECTED_FRAMES = (0, 1, 6, 7)
ACTION_DIM = 14
HISTORY = 4
FUTURE = 8
COND_DIM = (HISTORY + FUTURE) * ACTION_DIM + 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def instruction_tokens(instructions: list[str] | tuple[str, ...]) -> np.ndarray:
    rows = []
    for instruction in instructions:
        text = str(instruction or "").lower()
        rows.append((float("right arm" in text), float("left arm" in text)))
    return np.asarray(rows, dtype=np.float32)


class DirectResidualHead(nn.Module):
    """Small RGB residual head with explicit joint-sequence and arm-token FiLM."""

    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.channels = int(channels)
        self.stem = nn.Sequential(
            nn.Conv2d(6, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
        )
        self.condition = nn.Sequential(
            nn.Linear(COND_DIM, 64), nn.SiLU(), nn.Linear(64, FUTURE * channels * 2),
        )
        self.output = nn.Conv2d(channels, 3, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("protected_mask", self._protected_mask(), persistent=True)

    @staticmethod
    def _protected_mask() -> torch.Tensor:
        mask = torch.ones((1, FUTURE, 1, 1, 1), dtype=torch.float32)
        mask[:, list(PROTECTED_FRAMES)] = 0.0
        return mask

    def forward(
        self,
        baseline: torch.Tensor,
        context_last: torch.Tensor,
        normalized_actions: torch.Tensor,
        arm_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if baseline.ndim != 5 or baseline.shape[1] != FUTURE or baseline.shape[2] != 3:
            raise ValueError(f"v444 baseline must be [B,8,3,H,W], got {tuple(baseline.shape)}")
        batch, _, _, height, width = baseline.shape
        if context_last.shape != (batch, 3, height, width):
            raise ValueError("v444 context_last shape mismatch")
        if normalized_actions.shape != (batch, HISTORY + FUTURE, ACTION_DIM):
            raise ValueError("v444 normalized action shape mismatch")
        if arm_tokens.shape != (batch, 2):
            raise ValueError("v444 instruction token shape mismatch")
        context = context_last[:, None].expand(-1, FUTURE, -1, -1, -1)
        image = torch.cat((baseline, context), dim=2).reshape(batch * FUTURE, 6, height, width)
        features = self.stem(image).reshape(batch, FUTURE, self.channels, height, width)
        cond_input = torch.cat((normalized_actions.flatten(1), arm_tokens), dim=1)
        film = self.condition(cond_input).reshape(batch, FUTURE, 2, self.channels)
        gamma = 0.1 * torch.tanh(film[:, :, 0])[:, :, :, None, None]
        bias = film[:, :, 1, :, None, None]
        features = features * (1.0 + gamma) + bias
        residual = self.output(features.reshape(batch * FUTURE, self.channels, height, width))
        residual = 8.0 * torch.tanh(residual).reshape(batch, FUTURE, 3, height, width)
        return residual * self.protected_mask


def apply_direct_residual(baseline: np.ndarray, residual: np.ndarray | None) -> np.ndarray:
    baseline = np.asarray(baseline)
    if baseline.dtype != np.uint8 or baseline.ndim != 4 or baseline.shape[0] != 8 or baseline.shape[-1] != 3:
        raise ValueError("v444 baseline must be uint8 [8,H,W,3]")
    if residual is None:
        return baseline.copy()
    value = np.asarray(residual, dtype=np.float32)
    if value.shape != baseline.shape or not np.isfinite(value).all() or np.abs(value).max(initial=0.0) > 8.00001:
        raise ValueError("v444 residual must match baseline and be bounded by 8")
    if np.any(value[list(PROTECTED_FRAMES)] != 0.0):
        raise ValueError("v444 protected residual frames must be exactly zero")
    return np.clip(np.rint(baseline.astype(np.float32) + value), 0, 255).astype(np.uint8)


class Track2V444V169DirectResidual:
    def __init__(self, release_dir: str | Path, device: str = "cuda") -> None:
        root = Path(release_dir).resolve()
        manifest_path = root / "v444_direct_residual_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != FORMAT or manifest.get("official_reward_runtime_used") is not False:
            raise RuntimeError("unsupported or reward-coupled v444 release")
        checkpoint_path = root / manifest["checkpoint"]
        v169_manifest = root / manifest["v169_release"] / "v169_arm_routed_manifest.json"
        for key, path in (("checkpoint", checkpoint_path), ("v169_manifest", v169_manifest)):
            if not path.is_file() or _sha256(path) != manifest["sha256"][key]:
                raise RuntimeError(f"v444 release hash mismatch: {key}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "strict-track2-v444-direct-residual-checkpoint-v1" or checkpoint.get("step") != 25:
            raise RuntimeError("invalid v444 checkpoint")
        self.device = torch.device(device)
        self.model = DirectResidualHead(channels=int(checkpoint["channels"]))
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()
        self.action_mean = torch.as_tensor(checkpoint["action_mean"], dtype=torch.float32, device=self.device)
        self.action_std = torch.as_tensor(checkpoint["action_std"], dtype=torch.float32, device=self.device)
        if self.action_mean.shape != (14,) or self.action_std.shape != (14,) or torch.any(self.action_std <= 0):
            raise RuntimeError("invalid v444 action normalization")
        self.v169 = Track2V169ArmRoutedRuntime(root / manifest["v169_release"], root / manifest["v169_library"], device)
        self.last_decisions: list[dict] = []

    @staticmethod
    def gate_decision(history_actions, future_actions, instruction):
        return gate_decision(history_actions, future_actions, instruction)

    @torch.inference_mode()
    def predict_batch_with_baseline(self, context_frames, history_actions, future_actions, seeds, instructions):
        baseline = self.v169.predict_batch(context_frames, history_actions, future_actions, seeds, instructions)
        decisions = [gate_decision(h, f, t) for h, f, t in zip(history_actions, future_actions, instructions)]
        enabled = np.asarray([row["gate"] for row in decisions], dtype=bool)
        output = baseline.copy()
        if enabled.any():
            actions = np.concatenate((np.asarray(history_actions)[enabled], np.asarray(future_actions)[enabled]), axis=1)
            normalized = (torch.as_tensor(actions, dtype=torch.float32, device=self.device) - self.action_mean) / self.action_std
            base = torch.as_tensor(np.asarray(baseline)[enabled], dtype=torch.float32, device=self.device).permute(0, 1, 4, 2, 3) / 255.0
            context = torch.as_tensor(np.asarray(context_frames)[enabled, -1], dtype=torch.float32, device=self.device).permute(0, 3, 1, 2) / 255.0
            tokens = torch.as_tensor(instruction_tokens([instructions[i] for i in np.flatnonzero(enabled)]), device=self.device)
            residual = self.model(base, context, normalized, tokens).permute(0, 1, 3, 4, 2).cpu().numpy()
            for local, index in enumerate(np.flatnonzero(enabled)):
                output[index] = apply_direct_residual(baseline[index], residual[local])
        self.last_decisions = decisions
        return baseline, output, decisions

    def predict_batch(self, context_frames, history_actions, future_actions, seeds, instructions):
        return self.predict_batch_with_baseline(context_frames, history_actions, future_actions, seeds, instructions)[1]

    def predict_with_baseline(self, context_frames, history_actions, future_actions, seed, instruction):
        baseline, output, decisions = self.predict_batch_with_baseline(
            np.asarray(context_frames)[None], np.asarray(history_actions)[None], np.asarray(future_actions)[None], np.asarray([seed]), [instruction]
        )
        return baseline[0], output[0], decisions[0]

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        return self.predict_with_baseline(context_frames, history_actions, future_actions, seed, instruction)[1]
