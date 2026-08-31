"""Frozen complete V15 followed by an audited, bounded residual correction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .post_v15_residual import PostV15ResidualUNet
from .v15_runtime import Track2V15Runtime, _active_arm, _sha256
from .visual_source_gate_runtime import Track2VisualSourceGate


FORMAT = "track2-v15-post-residual-v1"


class Track2V15PostResidualRuntime(Track2V15Runtime):
    """Apply the preregistered residual only after the accepted full V15 chain.

    Routing uses only request inputs: the last context image selects the visual
    domain and action deltas select the active arm.  Targets and rewards are
    never available at inference time.
    """

    def __init__(
        self, checkpoint_dir: str | Path, library_dir: str | Path, device: str = "cuda"
    ) -> None:
        super().__init__(checkpoint_dir, library_dir, device)
        adaptation = self.root / "post_v15_residual"
        manifest_path = adaptation / "adaptation_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != FORMAT:
            raise RuntimeError("unsupported post-V15 residual release format")
        if manifest.get("base_release_manifest_sha256") != _sha256(
            self.root / "release_manifest.json"
        ):
            raise RuntimeError("post-V15 residual was packaged against another base release")
        files = manifest.get("sha256", {})
        if not isinstance(files, dict) or not files:
            raise RuntimeError("post-V15 residual manifest has no artifacts")
        for relative, expected in files.items():
            path = adaptation / relative
            if not path.is_file() or _sha256(path) != expected:
                raise RuntimeError(f"post-V15 residual artifact hash mismatch: {path}")

        state = torch.load(
            adaptation / "post_v15_residual.pt", map_location="cpu", weights_only=True
        )
        if state.get("format") != "strict-track2-post-v15-residual-v1":
            raise RuntimeError("unsupported post-V15 residual checkpoint")
        config = state["config"]
        self.residual = PostV15ResidualUNet(
            int(config["base_channels"]), float(config["maximum_residual_255"])
        )
        self.residual.load_state_dict(state["state_dict"], strict=True)
        self.residual = self.residual.requires_grad_(False).to(self.device).eval()
        # Keep Track2V15Runtime.action_mean/std intact: the V10 stage uses the
        # full 14-D action normalization, while this residual consumes only
        # the selected arm's 7-D action.
        self.residual_action_mean = state["active_action_mean"].to(self.device).float()
        self.residual_action_std = state["active_action_std"].to(self.device).float()
        self.source_gate = Track2VisualSourceGate(adaptation / "source_gate.pt", device)
        with np.load(adaptation / "advantage_mask.npz", allow_pickle=False) as values:
            if str(values["format"]) != "strict-track2-residual-advantage-mask-v1":
                raise RuntimeError("unsupported post-V15 residual advantage mask")
            mask = values["mask"].astype(np.float32)
        if mask.shape != (2, 2, 8, 256, 256):
            raise RuntimeError(f"unexpected post-V15 residual mask shape: {mask.shape}")
        self.advantage_mask = torch.from_numpy(mask).to(self.device)
        self.output_strength = float(manifest["output_strength"])
        self.maximum_deployed_residual = (
            float(manifest["maximum_deployed_residual_255"]) / 255.0
        )
        if self.output_strength <= 0 or self.maximum_deployed_residual <= 0:
            raise RuntimeError("invalid post-V15 residual deployment bounds")

    @torch.inference_mode()
    def _post_residual(
        self,
        baseline_uint8: np.ndarray,
        context_frames: np.ndarray,
        history_actions: np.ndarray,
        future_actions: np.ndarray,
    ) -> np.ndarray:
        arm, active = _active_arm(history_actions, future_actions)
        baseline = (
            torch.from_numpy(np.ascontiguousarray(baseline_uint8))
            .permute(0, 3, 1, 2)
            .to(self.device)
            .float()
            .div(255)
        )
        context = (
            torch.from_numpy(np.ascontiguousarray(context_frames[-1]))
            .permute(2, 0, 1)
            .to(self.device)
            .float()
            .div(255)
        )
        actions = torch.from_numpy(np.ascontiguousarray(active)).to(self.device).float()
        normalized = (actions - self.residual_action_mean) / self.residual_action_std
        horizons = torch.arange(1, 9, device=self.device, dtype=baseline.dtype).div(8)
        arm_right = torch.full((8,), arm, device=self.device, dtype=torch.long)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            raw, _ = self.residual(
                baseline,
                context[None].expand(8, -1, -1, -1),
                normalized,
                horizons,
                arm_right,
            )
        correction = self.output_strength * (raw.float() - baseline)
        correction = correction.clamp(
            -self.maximum_deployed_residual, self.maximum_deployed_residual
        )
        source, _ = self.source_gate.source_index(context_frames[-1])
        mask = self.advantage_mask[source, arm, :, None]
        candidate = baseline + mask * correction
        return (
            candidate.mul(255)
            .round()
            .clamp(0, 255)
            .byte()
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
            .copy()
        )

    def predict(
        self,
        context_frames,
        history_actions,
        future_actions,
        seed: int,
        instruction: str | None,
    ):
        baseline = super().predict(
            context_frames, history_actions, future_actions, seed, instruction
        )
        return self._post_residual(
            baseline, context_frames, history_actions, future_actions
        )
