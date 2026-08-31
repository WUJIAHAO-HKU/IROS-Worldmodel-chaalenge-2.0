"""V446 frozen-v169 128-resolution action-contrastive residual U-Net runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .v169_arm_routed_runtime import Track2V169ArmRoutedRuntime
from .v442_v169_close_aligned_projection_runtime import gate_decision


FORMAT = "track2-v446-v169-contrastive-residual-unet-release-v1"
CHECKPOINT_FORMAT = "strict-track2-v446-contrastive-residual-unet-checkpoint-v1"
WORKING_RESOLUTION = 128
RESIDUAL_CAP = 4.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def instruction_tokens(instructions) -> np.ndarray:
    rows = []
    for instruction in instructions:
        text = str(instruction or "").lower()
        rows.append((float("right arm" in text), float("left arm" in text)))
    return np.asarray(rows, dtype=np.float32)


class ResidualUNet128FiLM(nn.Module):
    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.channels = int(channels)
        self.enc1 = nn.Sequential(nn.Conv2d(6, channels, 3, padding=1), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU())
        self.enc2 = nn.Sequential(nn.Conv2d(channels, 2 * channels, 3, stride=2, padding=1), nn.SiLU(), nn.Conv2d(2 * channels, 2 * channels, 3, padding=1), nn.SiLU())
        self.condition = nn.Sequential(nn.Linear(12 * 14 + 2, 128), nn.SiLU(), nn.Linear(128, 8 * 4 * channels))
        self.up = nn.ConvTranspose2d(2 * channels, channels, 4, stride=2, padding=1)
        self.decode = nn.Sequential(nn.Conv2d(2 * channels, channels, 3, padding=1), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU())
        self.output = nn.Conv2d(channels, 3, 3, padding=1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, baseline: torch.Tensor, context_last: torch.Tensor, normalized_actions: torch.Tensor, arm_tokens: torch.Tensor) -> torch.Tensor:
        if baseline.ndim != 5 or baseline.shape[1:3] != (8, 3):
            raise ValueError(f"v446 baseline must be [B,8,3,H,W], got {tuple(baseline.shape)}")
        batch, _, _, height, width = baseline.shape
        if context_last.shape != (batch, 3, height, width) or normalized_actions.shape != (batch, 12, 14) or arm_tokens.shape != (batch, 2):
            raise ValueError("v446 context/action/token shape mismatch")
        base128 = F.interpolate(baseline.flatten(0, 1), size=(WORKING_RESOLUTION, WORKING_RESOLUTION), mode="bilinear", align_corners=False).reshape(batch, 8, 3, WORKING_RESOLUTION, WORKING_RESOLUTION)
        context128 = F.interpolate(context_last, size=(WORKING_RESOLUTION, WORKING_RESOLUTION), mode="bilinear", align_corners=False)
        context128 = context128[:, None].expand(-1, 8, -1, -1, -1)
        image = torch.cat((base128, context128), dim=2).reshape(batch * 8, 6, WORKING_RESOLUTION, WORKING_RESOLUTION)
        skip = self.enc1(image)
        encoded = self.enc2(skip).reshape(batch, 8, 2 * self.channels, 64, 64)
        film = self.condition(torch.cat((normalized_actions.flatten(1), arm_tokens), dim=1)).reshape(batch, 8, 2, 2 * self.channels)
        gamma = 0.1 * torch.tanh(film[:, :, 0])[:, :, :, None, None]
        bias = film[:, :, 1, :, None, None]
        encoded = (encoded * (1.0 + gamma) + bias).reshape(batch * 8, 2 * self.channels, 64, 64)
        decoded = self.up(encoded)
        decoded = self.decode(torch.cat((decoded, skip), dim=1))
        residual128 = RESIDUAL_CAP * torch.tanh(self.output(decoded)).reshape(batch, 8, 3, WORKING_RESOLUTION, WORKING_RESOLUTION)
        residual = F.interpolate(residual128.flatten(0, 1), size=(height, width), mode="bilinear", align_corners=False).reshape(batch, 8, 3, height, width)
        return residual.clamp(-RESIDUAL_CAP, RESIDUAL_CAP)


def apply_residual(baseline: np.ndarray, residual: np.ndarray | None) -> np.ndarray:
    baseline = np.asarray(baseline)
    if baseline.dtype != np.uint8 or baseline.ndim != 4 or baseline.shape[0] != 8 or baseline.shape[-1] != 3:
        raise ValueError("v446 baseline must be uint8 [8,H,W,3]")
    if residual is None:
        return baseline.copy()
    value = np.asarray(residual, dtype=np.float32)
    if value.shape != baseline.shape or not np.isfinite(value).all() or np.abs(value).max(initial=0.0) > RESIDUAL_CAP + 1e-5:
        raise ValueError("v446 residual shape/finite/cap contract failed")
    return np.clip(np.rint(baseline.astype(np.float32) + value), 0, 255).astype(np.uint8)


class Track2V446V169ContrastiveResidualUNet:
    def __init__(self, release_dir: str | Path, device: str = "cuda") -> None:
        root = Path(release_dir).resolve()
        manifest = json.loads((root / "v446_contrastive_residual_manifest.json").read_text())
        if manifest.get("format") != FORMAT or manifest.get("official_reward_runtime_used") is not False:
            raise RuntimeError("unsupported or reward-coupled v446 release")
        checkpoint_path = root / manifest["checkpoint"]
        v169_manifest = root / manifest["v169_release"] / "v169_arm_routed_manifest.json"
        for key, path in (("checkpoint", checkpoint_path), ("v169_manifest", v169_manifest)):
            if not path.is_file() or _sha256(path) != manifest["sha256"][key]:
                raise RuntimeError(f"v446 hash mismatch: {key}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != CHECKPOINT_FORMAT or checkpoint.get("step") != 50 or checkpoint.get("training_scope") != "all15":
            raise RuntimeError("invalid v446 final checkpoint")
        self.device = torch.device(device)
        self.model = ResidualUNet128FiLM(int(checkpoint["channels"])).to(self.device)
        self.model.load_state_dict(checkpoint["model"]); self.model.eval()
        self.action_mean = torch.as_tensor(checkpoint["action_mean"], dtype=torch.float32, device=self.device)
        self.action_std = torch.as_tensor(checkpoint["action_std"], dtype=torch.float32, device=self.device)
        self.v169 = Track2V169ArmRoutedRuntime(root / manifest["v169_release"], root / manifest["v169_library"], device)
        self.last_decisions = []

    @staticmethod
    def gate_decision(history_actions, future_actions, instruction):
        return gate_decision(history_actions, future_actions, instruction)

    @torch.inference_mode()
    def predict_batch_with_baseline(self, context_frames, history_actions, future_actions, seeds, instructions):
        context = np.asarray(context_frames); history = np.asarray(history_actions); future = np.asarray(future_actions)
        baseline = self.v169.predict_batch(context, history, future, seeds, instructions)
        decisions = [gate_decision(h, f, text) for h, f, text in zip(history, future, instructions)]
        enabled = np.asarray([row["gate"] for row in decisions], dtype=bool)
        output = baseline.copy()
        if enabled.any():
            actions = np.concatenate((history[enabled], future[enabled]), axis=1)
            normalized = (torch.as_tensor(actions, dtype=torch.float32, device=self.device) - self.action_mean) / self.action_std
            base = torch.as_tensor(baseline[enabled], dtype=torch.float32, device=self.device).permute(0, 1, 4, 2, 3) / 255.0
            last = torch.as_tensor(context[enabled, -1], dtype=torch.float32, device=self.device).permute(0, 3, 1, 2) / 255.0
            tokens = torch.as_tensor(instruction_tokens([instructions[i] for i in np.flatnonzero(enabled)]), device=self.device)
            residual = self.model(base, last, normalized, tokens).permute(0, 1, 3, 4, 2).cpu().numpy()
            for local, index in enumerate(np.flatnonzero(enabled)):
                output[index] = apply_residual(baseline[index], residual[local])
        self.last_decisions = decisions
        return baseline, output, decisions

    def predict_batch(self, context_frames, history_actions, future_actions, seeds, instructions):
        return self.predict_batch_with_baseline(context_frames, history_actions, future_actions, seeds, instructions)[1]

    def predict_with_baseline(self, context_frames, history_actions, future_actions, seed, instruction):
        baseline, output, decisions = self.predict_batch_with_baseline(np.asarray(context_frames)[None], np.asarray(history_actions)[None], np.asarray(future_actions)[None], np.asarray([seed]), [instruction])
        return baseline[0], output[0], decisions[0]

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        return self.predict_with_baseline(context_frames, history_actions, future_actions, seed, instruction)[1]
