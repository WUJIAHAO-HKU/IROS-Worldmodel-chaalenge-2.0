"""Geometry-guided hard texture routing for the data-closed v17.2 experiment.

The renderer never synthesizes RGB.  It constructs candidates by warping one
of the five observed frames with the action-predicted end-effector geometry,
then predicts whether that observed pixel is visible and better than the frozen
parent.  A hard parent/candidate decision preserves source-frequency detail and
prevents alpha mixtures at the gripper/bottle boundary.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

try:
    from .canonical_arm_texture_v11 import REGIONS, _observed_beam_mask
    from .contact_structure_v135 import structure_masks
    from .object_geometry_v170 import affine_from_landmarks, warp_layer
except ImportError:  # Standalone mirror used by tests.
    from canonical_arm_texture_v11 import REGIONS, _observed_beam_mask
    from contact_structure_v135 import structure_masks
    from object_geometry_v170 import affine_from_landmarks, warp_layer


CANDIDATE_COUNT = 11  # five rigid transports, five static bottle views, one cleanup.
SUPPORT_CHANNELS = 5  # beam, black, grey, bottle, stale-cleanup.


def full_layer_masks(frame: np.ndarray, side: str) -> np.ndarray:
    """Observable five-channel semantic support for one RGB frame."""
    output = np.zeros((SUPPORT_CHANNELS, *frame.shape[:2]), dtype=np.uint8)
    y0, y1, x0, x1 = REGIONS[side]
    beam = _observed_beam_mask(frame[y0:y1, x0:x1])
    beam = cv2.morphologyEx(beam, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    output[0, y0:y1, x0:x1] = cv2.dilate(beam, np.ones((3, 3), np.uint8))
    bottle, black, grey = structure_masks(frame)
    output[1] = black; output[2] = grey; output[3] = bottle
    return output


def _inpaint_candidate(parent: np.ndarray, stale: np.ndarray) -> np.ndarray:
    if not stale.any():
        return parent.copy()
    mask = cv2.dilate(stale.astype(np.uint8), np.ones((7, 7), np.uint8))
    # Telea uses only pixels from this supplied frame; it introduces no data or
    # learned external prior.  The learned router can always retain the parent.
    return cv2.inpaint(parent, mask * 255, 4.0, cv2.INPAINT_TELEA)


def build_candidates(
    context: np.ndarray,
    parent: np.ndarray,
    source_geometry: np.ndarray,
    target_geometry: np.ndarray,
    arm: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return RGB candidates, semantic supports, and compact conditions.

    Shapes are ``[11,H,W,3]``, ``[11,5,H,W]`` and ``[11,5]``.  Conditions are
    normalized translation, affine scale change, source age and hypothesis id.
    Everything is computable from request inputs and the frozen pose projector.
    """
    if context.shape != (5, *parent.shape):
        raise ValueError(f"context/parent shapes differ: {context.shape}, {parent.shape}")
    side = "left" if int(arm) == 0 else "right"
    rgbs: list[np.ndarray] = []; supports: list[np.ndarray] = []; conditions: list[np.ndarray] = []
    moved_union = np.zeros(parent.shape[:2], dtype=np.uint8)
    source_layers = [full_layer_masks(frame, side) for frame in context]
    for source_index, (frame, layers) in enumerate(zip(context, source_layers)):
        matrix = affine_from_landmarks(source_geometry[source_index], target_geometry)
        rgb = warp_layer(frame, matrix, cv2.INTER_CUBIC)
        support = np.stack([warp_layer(layer, matrix, cv2.INTER_NEAREST) for layer in layers])
        moved_union |= support[:4].max(0).astype(np.uint8)
        linear = matrix[:, :2]
        scale = float(np.sqrt(max(abs(np.linalg.det(linear)), 1e-8)))
        rgbs.append(rgb); supports.append(support)
        conditions.append(np.asarray((matrix[0, 2] / 64, matrix[1, 2] / 64,
                                      np.log(scale + 1e-8), source_index / 4, 0), np.float32))

    identity = np.asarray([[1, 0, 0], [0, 1, 0]], np.float32)
    for source_index, (frame, layers) in enumerate(zip(context, source_layers)):
        support = np.zeros_like(layers); support[3] = layers[3]
        rgbs.append(warp_layer(frame, identity, cv2.INTER_CUBIC)); supports.append(support)
        conditions.append(np.asarray((0, 0, 0, source_index / 4, 1), np.float32))

    source_structure = source_layers[-1][1:3].max(0)
    stale = source_structure & ~cv2.dilate(moved_union, np.ones((5, 5), np.uint8)).astype(bool)
    cleanup_support = np.zeros_like(source_layers[-1]); cleanup_support[4] = stale
    rgbs.append(_inpaint_candidate(parent, stale)); supports.append(cleanup_support)
    conditions.append(np.asarray((0, 0, 0, 1, 2), np.float32))
    return np.stack(rgbs), np.stack(supports), np.stack(conditions)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(4, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.silu(value + self.body(value))


class GeometryVisibilityRouterV172(nn.Module):
    """Shared per-candidate gain scorer with native-resolution skip paths."""

    input_channels = 23  # parent/candidate/delta RGB, support, xy, and seven conditions.

    def __init__(self, base_channels: int = 24) -> None:
        super().__init__(); c = base_channels
        self.enc0 = nn.Sequential(nn.Conv2d(self.input_channels, c, 3, padding=1),
                                  nn.GroupNorm(4, c), nn.SiLU(), ResidualBlock(c))
        self.enc1 = nn.Sequential(nn.Conv2d(c, 2 * c, 3, stride=2, padding=1),
                                  nn.GroupNorm(8, 2 * c), nn.SiLU(), ResidualBlock(2 * c))
        self.enc2 = nn.Sequential(nn.Conv2d(2 * c, 3 * c, 3, stride=2, padding=1),
                                  nn.GroupNorm(8, 3 * c), nn.SiLU(),
                                  ResidualBlock(3 * c, 2), ResidualBlock(3 * c, 4))
        self.dec1 = nn.Sequential(nn.Conv2d(5 * c, 2 * c, 3, padding=1),
                                  nn.GroupNorm(8, 2 * c), nn.SiLU(), ResidualBlock(2 * c))
        self.dec0 = nn.Sequential(nn.Conv2d(3 * c, c, 3, padding=1),
                                  nn.GroupNorm(4, c), nn.SiLU(), ResidualBlock(c))
        self.head = nn.Conv2d(c, 1, 1)
        nn.init.zeros_(self.head.weight); nn.init.constant_(self.head.bias, -2.0)

    @staticmethod
    def features(parent: torch.Tensor, candidates: torch.Tensor, supports: torch.Tensor,
                 conditions: torch.Tensor, horizon: torch.Tensor, arm: torch.Tensor) -> torch.Tensor:
        """Build observable features for ``[B,K,...]`` candidates."""
        batch, count, _, height, width = candidates.shape
        parent = parent[:, None].expand(-1, count, -1, -1, -1)
        delta = candidates - parent
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, height, device=parent.device, dtype=parent.dtype),
                                torch.linspace(-1, 1, width, device=parent.device, dtype=parent.dtype),
                                indexing="ij")
        xy = torch.stack((xx, yy))[None, None].expand(batch, count, -1, -1, -1)
        scalar = torch.cat((conditions, horizon.reshape(batch, 1, 1).expand(-1, count, -1),
                            arm.reshape(batch, 1, 1).expand(-1, count, -1)), 2)
        scalar = scalar[:, :, :, None, None].expand(-1, -1, -1, height, width)
        return torch.cat((parent, candidates, delta, supports, xy, scalar), 2)

    def forward(self, parent: torch.Tensor, candidates: torch.Tensor, supports: torch.Tensor,
                conditions: torch.Tensor, horizon: torch.Tensor, arm: torch.Tensor) -> torch.Tensor:
        batch, count = candidates.shape[:2]
        value = self.features(parent, candidates, supports, conditions, horizon, arm).flatten(0, 1)
        e0 = self.enc0(value); e1 = self.enc1(e0); e2 = self.enc2(e1)
        d1 = F.interpolate(e2, e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat((d1, e1), 1))
        d0 = F.interpolate(d1, e0.shape[-2:], mode="bilinear", align_corners=False)
        score = self.head(self.dec0(torch.cat((d0, e0), 1)))
        return score.unflatten(0, (batch, count))


def oracle_labels(parent: torch.Tensor, candidates: torch.Tensor, supports: torch.Tensor,
                  margin: float = 0.25 / 255.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Training-only target labels are built separately from observable inputs."""
    raise RuntimeError("oracle_labels requires target; use routing_targets")


def routing_targets(parent: torch.Tensor, candidates: torch.Tensor, supports: torch.Tensor,
                    target: torch.Tensor, margin: float = 0.25 / 255.0
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    parent_error = (parent - target).abs().mean(1)
    candidate_error = (candidates - target[:, None]).abs().mean(2)
    valid = supports.max(2).values > 0.5
    gain = (parent_error[:, None] - candidate_error).masked_fill(~valid, -1.0)
    best_gain, best = gain.max(1)
    labels = best + 1
    labels = torch.where(best_gain > margin, labels, torch.zeros_like(labels))
    focus = valid.any(1)
    return labels, focus, gain


def hard_render(parent: torch.Tensor, candidates: torch.Tensor, supports: torch.Tensor,
                scores: torch.Tensor, threshold: float = 0.0
                ) -> tuple[torch.Tensor, torch.Tensor]:
    valid = supports.max(2).values > 0.5
    scores = scores[:, :, 0].masked_fill(~valid, -1e4)
    best_score, best = scores.max(1)
    use = best_score > threshold
    gather = best[:, None, None, :, :].expand(-1, 1, candidates.shape[2], -1, -1)
    selected = candidates.gather(1, gather).squeeze(1)
    output = torch.where(use[:, None], selected, parent)
    return output, torch.where(use, best + 1, torch.zeros_like(best))


def parameter_count(base_channels: int = 24) -> int:
    return sum(value.numel() for value in GeometryVisibilityRouterV172(base_channels).parameters())
