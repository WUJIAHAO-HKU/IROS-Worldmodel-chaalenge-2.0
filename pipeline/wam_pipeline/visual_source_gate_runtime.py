"""Standalone frozen visual-domain gate using only the last request frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class Track2VisualSourceGate:
    def __init__(self, checkpoint: str | Path, device: str = "cuda") -> None:
        self.device = torch.device(device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state.get("format") != "strict-track2-visual-source-gate-v1":
            raise ValueError("unsupported visual source gate")
        self.pool_size = int(state["pool_size"])
        self.feature_mean = state["feature_mean"].to(self.device).float()
        self.feature_std = state["feature_std"].to(self.device).float()
        self.weight = state["weight"].to(self.device).float()
        self.bias = state["bias"].to(self.device).float()
        self.threshold = float(state["threshold"])

    def probability_synthetic(self, frame: np.ndarray) -> float:
        value = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)
        value = value.to(self.device).float().div(255).unsqueeze(0)
        pooled = F.adaptive_avg_pool2d(value, (self.pool_size, self.pool_size)).flatten(1)
        feature = torch.cat((pooled, value.mean((2, 3)), value.std((2, 3), unbiased=False)), 1)
        normalized = (feature - self.feature_mean) / self.feature_std
        with torch.inference_mode():
            return float(torch.sigmoid(F.linear(normalized, self.weight, self.bias))[0, 0])

    def source_index(self, frame: np.ndarray) -> tuple[int, float]:
        probability = self.probability_synthetic(frame)
        return int(probability >= self.threshold), probability
