"""Online runtime for the validated Track-2 v15 composite.

The validation experiment was assembled from cached stages.  This module
replays the same stages for arbitrary 5-frame contexts and 8-action chunks so
the model can be used by the HTTP submission service and RLinf.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .candidate_risk_router_v101 import CandidateRiskRouterV101
from .canonical_arm_texture_v11 import REGIONS, _observed_beam_mask, _observed_logo_mask
from .contact_occlusion_head_v13 import CONTACT_REGION
from .contact_structure_v135 import structure_semantic_mask
from .dual_tiny_experts_v150 import TinyBlackGripperExpert, TinyGlyphMotionExpert, render_black_gripper
from .multisource_flow_unet import MultiSourceActionFlowUNet
from .profile import ACTION_DIM, CONTEXT_FRAMES, PREDICTION_FRAMES
from .protected_layered_flow_v91 import ProtectedLayeredFlowV91
from .structure_refiner_runtime import Track2StructureGatedLocalFusion


FORMAT = "track2-v15-online-release-v1"
SIZE = 96


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode(name: str) -> int:
    match = re.search(r"episode(\d+)", name)
    if not match:
        raise ValueError(f"window has no episode number: {name}")
    return int(match.group(1))


def _active_arm(history: np.ndarray, future: np.ndarray) -> tuple[int, np.ndarray]:
    actions = np.concatenate((history, future), axis=0)
    delta = np.abs(np.diff(actions, axis=0))
    activity = np.asarray((delta[:, :7].mean(), delta[:, 7:].mean()))
    arm = int(activity.argmax())
    return arm, future[:, arm * 7 : (arm + 1) * 7].copy()


def _frames(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value)).permute(0, 3, 1, 2).to(device).float().div(255.0)


def _resize_rgb(value: np.ndarray) -> np.ndarray:
    if value.ndim == 3:
        return cv2.resize(value, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return np.stack([_resize_rgb(frame) for frame in value])


def _resize_mask(value: np.ndarray) -> np.ndarray:
    if value.ndim == 2:
        return cv2.resize(value.astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    return np.stack([_resize_mask(frame) for frame in value])


def _rgb(value: np.ndarray, device: torch.device) -> torch.Tensor:
    axes = (2, 0, 1) if value.ndim == 3 else (0, 3, 1, 2)
    return torch.from_numpy(value.transpose(axes).copy()).to(device).float().div(255.0)


def _mask(value: np.ndarray, device: torch.device) -> torch.Tensor:
    value = value[None] if value.ndim == 2 else value[:, None]
    return torch.from_numpy(value.astype(np.float32)).to(device)


def _descriptor(frame: np.ndarray) -> np.ndarray:
    y0, y1, x0, x1 = CONTACT_REGION
    value = cv2.resize(frame[y0:y1, x0:x1], (24, 20), interpolation=cv2.INTER_AREA)
    return value.astype(np.float32).reshape(-1) / 255.0


def _motion_descriptor(action: np.ndarray) -> np.ndarray:
    return np.concatenate(((action - action[:1]).reshape(-1), np.diff(action, axis=0).reshape(-1)))


def _align(candidate: np.ndarray, query: np.ndarray) -> tuple[float, np.ndarray]:
    y0, y1, x0, x1 = CONTACT_REGION
    candidate_gray = cv2.cvtColor(candidate[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    query_gray = cv2.cvtColor(query[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        score, warp = cv2.findTransformECC(
            query_gray,
            candidate_gray,
            warp,
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5),
            None,
            3,
        )
    except cv2.error:
        score = -1.0
    return float(score), warp


def _warp_future(frames: np.ndarray, warp: np.ndarray) -> np.ndarray:
    y0, y1, x0, x1 = CONTACT_REGION
    return np.stack(
        [
            cv2.warpAffine(
                frame[y0:y1, x0:x1],
                warp,
                (x1 - x0, y1 - y0),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT,
            )
            for frame in frames
        ]
    )


def _photometric_calibration(candidate_source: np.ndarray, query_source: np.ndarray, warp: np.ndarray):
    y0, y1, x0, x1 = CONTACT_REGION
    aligned = _warp_future(candidate_source[None], warp)[0].astype(np.float32)
    query = query_source[y0:y1, x0:x1].astype(np.float32)
    support = (aligned.mean(2) > 170) & (query.mean(2) > 170)
    gains, biases = [], []
    for channel in range(3):
        left, right = aligned[..., channel][support], query[..., channel][support]
        if len(left) < 100:
            gain, bias = 1.0, 0.0
        elif left.std() < 1.0 or right.std() < 1.0:
            gain, bias = 1.0, float(np.clip(right.mean() - left.mean(), -18, 18))
        else:
            gain = float(np.clip(right.std() / max(left.std(), 1.0), 0.85, 1.15))
            bias = float(np.clip(right.mean() - gain * left.mean(), -18, 18))
        gains.append(gain)
        biases.append(bias)
    return np.asarray(gains, np.float32), np.asarray(biases, np.float32)


def _feather(height: int, width: int, radius: int = 16) -> np.ndarray:
    y, x = np.ogrid[:height, :width]
    distance = np.minimum.reduce(
        (
            np.broadcast_to(y, (height, width)),
            np.broadcast_to(x, (height, width)),
            np.broadcast_to(height - 1 - y, (height, width)),
            np.broadcast_to(width - 1 - x, (height, width)),
        )
    )
    return np.clip(distance.astype(np.float32) / radius, 0, 1)


class Track2V15Runtime:
    """Full v8 -> v10.1 -> v14.1 -> v15 inference chain."""

    def __init__(self, checkpoint_dir: str | Path, library_dir: str | Path, device: str = "cuda") -> None:
        self.root = Path(checkpoint_dir).resolve()
        self.library_dir = Path(library_dir).resolve()
        self.device = torch.device(device)
        manifest = json.loads((self.root / "release_manifest.json").read_text())
        if manifest.get("format") != FORMAT:
            raise RuntimeError("unsupported v15 release format")
        for relative, expected in manifest["sha256"].items():
            path = self.root / relative
            if not path.is_file() or _sha256(path) != expected:
                raise RuntimeError(f"v15 release artifact hash mismatch: {path}")

        self.parent = Track2StructureGatedLocalFusion(self.root / "v8", device)
        self._load_v10()
        self._load_v15()
        self._load_library(manifest)

    def _load_v10(self) -> None:
        base_dir = self.root / "v10" / "flow_base"
        with np.load(base_dir / "track2_multisource_flow_unet_config.npz", allow_pickle=False) as config:
            base = MultiSourceActionFlowUNet(int(config["base_channels"]))
        base.load_state_dict(torch.load(base_dir / "model.pt", map_location="cpu", weights_only=True)["state_dict"], strict=True)
        head = torch.load(self.root / "v10" / "flow_head.pt", map_location="cpu", weights_only=True)
        self.flow = ProtectedLayeredFlowV91(base, int(head["base_channels"]), float(head["max_residual_flow"]))
        incompatible = self.flow.load_state_dict(head["state_dict"], strict=False)
        missing = [key for key in incompatible.missing_keys if not key.startswith("base_flow_model.")]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(f"invalid v10 flow head: {missing} {incompatible.unexpected_keys}")
        self.flow = self.flow.requires_grad_(False).to(self.device).eval()
        self.active_mean = head["active_mean"].to(self.device)
        self.active_std = head["active_std"].to(self.device)
        with np.load(base_dir / "action_normalization.npz", allow_pickle=False) as values:
            self.action_mean = torch.from_numpy(values["mean"]).to(self.device)
            self.action_std = torch.from_numpy(values["std"]).to(self.device)
        checkpoint = torch.load(self.root / "v10" / "risk_router.pt", map_location="cpu", weights_only=False)
        if checkpoint.get("router_version") != "v10.1":
            raise RuntimeError("v15 release risk router is not v10.1")
        self.router = CandidateRiskRouterV101(int(checkpoint["base_channels"]))
        self.router.load_state_dict(checkpoint["state_dict"], strict=True)
        self.router = self.router.requires_grad_(False).to(self.device).eval()

    def _load_v15(self) -> None:
        # Glyph strength is zero in the accepted deployment, but instantiate it
        # to verify architecture/checkpoint compatibility for reproducibility.
        glyph = torch.load(self.root / "v15" / "glyph.pt", map_location="cpu", weights_only=False)
        glyph_model = TinyGlyphMotionExpert(int(glyph["base_channels"]))
        glyph_model.load_state_dict(glyph["state_dict"], strict=True)
        gripper = torch.load(self.root / "v15" / "gripper.pt", map_location="cpu", weights_only=False)
        self.gripper = TinyBlackGripperExpert(int(gripper["base_channels"]))
        self.gripper.load_state_dict(gripper["state_dict"], strict=True)
        self.gripper = self.gripper.requires_grad_(False).to(self.device).eval()
        self.gripper_mean = gripper["action_mean"].numpy()
        self.gripper_std = gripper["action_std"].numpy()

    def _load_library(self, manifest: dict) -> None:
        retrieval = manifest["retrieval"]
        split_path = self.library_dir / retrieval["split_manifest"]
        split = json.loads(split_path.read_text())
        train_episodes = {int(value) for value in split["train_episodes"]}
        window_dir = self.library_dir / retrieval["windows_directory"]
        paths = sorted(path for path in window_dir.glob("episode*_*.npz") if _episode(path.name) in train_episodes)
        if not paths:
            raise RuntimeError("v15 retrieval library has no training windows")
        cache_sources = os.environ.get("WAM_RETRIEVAL_SOURCE_CACHE", "0") == "1"
        cache_targets = os.environ.get("WAM_RETRIEVAL_TARGET_CACHE", "0") == "1"
        visual, motion, arms, valid, sources = [], [], [], [], []
        for path in paths:
            with np.load(path, allow_pickle=False) as window:
                source = window["context_frames"][-1]
                arm, action = _active_arm(window["history_actions"], window["future_actions"])
            visual.append(_descriptor(source))
            motion.append(_motion_descriptor(action))
            arms.append(arm)
            valid.append(path)
            if cache_sources:
                # _load_library already decodes this exact source frame to form
                # its visual descriptor.  Retaining an owned uint8 copy avoids
                # reopening and decompressing up to 24 NPZ members per sample;
                # it does not alter retrieval scores, ECC, routing, or pixels.
                sources.append(source.copy())
        self.library_visual = np.stack(visual)
        self.library_motion = np.stack(motion)
        self.library_arms = np.asarray(arms)
        self.library_paths = valid
        self.library_sources = sources if cache_sources else None
        # Target windows are substantially larger than source frames.  Keep a
        # lazy, candidate-indexed cache so long rollouts avoid repeatedly
        # decompressing hot retrieval matches without paying a multi-gigabyte
        # startup cost.  Values are owned uint8 copies of the exact NPZ arrays.
        self.library_targets = {} if cache_targets else None

    def _library_source(self, candidate: int) -> np.ndarray:
        if self.library_sources is not None:
            return self.library_sources[candidate]
        with np.load(self.library_paths[candidate], allow_pickle=False) as window:
            return window["context_frames"][-1]

    def _library_target(self, candidate: int) -> np.ndarray:
        if self.library_targets is not None:
            cached = self.library_targets.get(candidate)
            if cached is not None:
                return cached
            with np.load(self.library_paths[candidate], allow_pickle=False) as window:
                cached = window["target_frames"].copy()
            self.library_targets[candidate] = cached
            return cached
        with np.load(self.library_paths[candidate], allow_pickle=False) as window:
            return window["target_frames"]

    @torch.inference_mode()
    def _v10(self, parent: np.ndarray, context: np.ndarray, history: np.ndarray, future: np.ndarray) -> np.ndarray:
        context_t = _frames(context, self.device)[None]
        parent_t = _frames(parent, self.device)[None]
        actions = torch.from_numpy(np.concatenate((history, future), axis=0)[None]).to(self.device).float()
        active, arm = self.flow.active_arm_actions(actions)
        flow = self.flow(
            context_t,
            (actions - self.action_mean) / self.action_std,
            (active - self.active_mean) / self.active_std,
            arm,
            parent_t,
        )
        candidates = flow["candidates"]
        risk = self.router(
            context_t,
            parent_t,
            candidates[:, :, 1:],
            flow["refined_flow"],
            flow["visibility_logits"],
            active,
            arm,
        )
        choice, _ = self.router.safe_choice(risk["advantage_mean"], risk["advantage_uncertainty"], 1.0, 0.25)
        full = choice.repeat_interleave(4, 2).repeat_interleave(4, 3)
        routed = candidates.gather(2, full[:, :, None, None].expand(-1, -1, 1, 3, -1, -1)).squeeze(2)
        highpass_parent = parent_t - F.avg_pool2d(parent_t.flatten(0, 1), 5, 1, 2, count_include_pad=False).unflatten(0, parent_t.shape[:2])
        highpass_routed = routed - F.avg_pool2d(routed.flatten(0, 1), 5, 1, 2, count_include_pad=False).unflatten(0, routed.shape[:2])
        routed = (routed + 0.5 * (highpass_parent - highpass_routed)).clamp(0, 1)
        return routed.mul(255).round().byte()[0].permute(0, 2, 3, 1).cpu().numpy().copy()

    @torch.inference_mode()
    def _v10_batch(
        self,
        parent: np.ndarray,
        context: np.ndarray,
        history: np.ndarray,
        future: np.ndarray,
    ) -> np.ndarray:
        context_t = torch.from_numpy(np.ascontiguousarray(context)).permute(0, 1, 4, 2, 3).to(self.device).float().div(255.0)
        parent_t = torch.from_numpy(np.ascontiguousarray(parent)).permute(0, 1, 4, 2, 3).to(self.device).float().div(255.0)
        actions = torch.from_numpy(np.ascontiguousarray(np.concatenate((history, future), axis=1))).to(self.device).float()
        active, arm = self.flow.active_arm_actions(actions)
        flow = self.flow(
            context_t,
            (actions - self.action_mean) / self.action_std,
            (active - self.active_mean) / self.active_std,
            arm,
            parent_t,
        )
        candidates = flow["candidates"]
        risk = self.router(
            context_t,
            parent_t,
            candidates[:, :, 1:],
            flow["refined_flow"],
            flow["visibility_logits"],
            active,
            arm,
        )
        choice, _ = self.router.safe_choice(
            risk["advantage_mean"], risk["advantage_uncertainty"], 1.0, 0.25
        )
        full = choice.repeat_interleave(4, 2).repeat_interleave(4, 3)
        routed = candidates.gather(
            2, full[:, :, None, None].expand(-1, -1, 1, 3, -1, -1)
        ).squeeze(2)
        highpass_parent = parent_t - F.avg_pool2d(
            parent_t.flatten(0, 1), 5, 1, 2, count_include_pad=False
        ).unflatten(0, parent_t.shape[:2])
        highpass_routed = routed - F.avg_pool2d(
            routed.flatten(0, 1), 5, 1, 2, count_include_pad=False
        ).unflatten(0, routed.shape[:2])
        routed = (routed + 0.5 * (highpass_parent - highpass_routed)).clamp(0, 1)
        return routed.mul(255).round().byte().permute(0, 1, 3, 4, 2).cpu().numpy().copy()

    def _v141(self, parent: np.ndarray, context: np.ndarray, history: np.ndarray, future: np.ndarray) -> np.ndarray:
        output = parent.copy()
        arm, action = _active_arm(history, future)
        # The accepted v14.1 route is right-arm only.  Previously the runtime
        # still searched and ECC-aligned 24 library candidates before this
        # immutable gate rejected every left-arm sample.  Returning here is
        # therefore byte-equivalent while removing all of that dead work.
        if arm != 1:
            return output

        # These structure predicates are also required by the final route.
        # Evaluate them before retrieval so samples that cannot possibly use a
        # retrieved target do not perform descriptor search or ECC alignment.
        y0, y1, x0, x1 = CONTACT_REGION
        source_label = structure_semantic_mask(context[-1:, y0:y1, x0:x1])[0]
        parent_label = structure_semantic_mask(parent[:, y0:y1, x0:x1])
        source_area = max(int(np.isin(source_label, (2, 3)).sum()), 1)
        ratios = np.isin(parent_label, (2, 3)).sum(axis=(1, 2)) / source_area
        decay = np.flatnonzero(ratios < 0.88)
        start = max(int(decay[0]) - 1, 0) if len(decay) else None
        intact_prefix = bool(start is not None and start >= 3 and np.all(ratios[:start] >= 0.95))
        if not (intact_prefix and ratios[-1] >= 0.65):
            return output

        candidates = np.flatnonzero(self.library_arms == arm)
        query_visual = _descriptor(context[-1])
        query_motion = _motion_descriptor(action)
        visual_distance = ((self.library_visual[candidates] - query_visual) ** 2).mean(1)
        motion_distance = ((self.library_motion[candidates] - query_motion) ** 2).mean(1)
        combined = visual_distance / max(float(np.median(visual_distance)), 1e-9)
        combined += 2.5 * motion_distance / max(float(np.median(motion_distance)), 1e-12)
        shortlist = candidates[np.argsort(combined)[:24]]
        aligned = []
        for candidate in shortlist:
            candidate_source = self._library_source(candidate)
            ecc, warp = _align(candidate_source, context[-1])
            if ecc >= 0.95:
                aligned.append((float(((self.library_motion[candidate] - query_motion) ** 2).mean()), candidate, -ecc, warp))
        if not aligned:
            return output
        _, candidate, negative_ecc, warp = min(aligned)
        selected_future = self._library_target(candidate)
        candidate_source = self._library_source(candidate)
        warped = _warp_future(selected_future, warp).astype(np.float32)
        gain, bias = _photometric_calibration(candidate_source, context[-1], warp)
        warped = np.clip(warped * gain + bias, 0, 255)
        ecc = -negative_ecc
        alpha = max(0.35, min(1.0, (ecc - 0.95) / 0.04))
        gate = np.zeros(8, np.float32)
        gate[start:] = 1.0
        alpha_map = gate[:, None, None, None] * alpha * _feather(y1 - y0, x1 - x0)[None, ..., None]
        parent_crop = parent[:, y0:y1, x0:x1].astype(np.float32)
        output[:, y0:y1, x0:x1] = np.clip(np.round(parent_crop * (1 - alpha_map) + warped * alpha_map), 0, 255).astype(np.uint8)
        # v14.1's text branch is accepted only before the structure switch.
        from .canonical_arm_texture_v11 import _gray, _warp
        from .v15_texture_reprojection import reproject_observed_glyph

        for time in range(min(start, 2)):
            highpass_strength = 0.55 if time == 0 else 0.0
            output[time], _ = reproject_observed_glyph(
                output[time], context[-1], _gray, _warp, highpass_strength
            )
        return output

    @torch.inference_mode()
    def _v15(self, parent: np.ndarray, context: np.ndarray, history: np.ndarray, future: np.ndarray) -> np.ndarray:
        arm, action = _active_arm(history, future)
        if arm != 1:
            return parent
        cy0, cy1, cx0, cx1 = CONTACT_REGION
        source = context[-1, cy0:cy1, cx0:cx1]
        parent_crop = parent[:, cy0:cy1, cx0:cx1]
        source_label = structure_semantic_mask(source[None])[0]
        parent_label = structure_semantic_mask(parent_crop)
        normalized = (action - self.gripper_mean[arm]) / self.gripper_std[arm]
        logits, residual = self.gripper(
            _rgb(_resize_rgb(source), self.device)[None],
            _rgb(_resize_rgb(parent_crop), self.device)[None],
            torch.from_numpy(normalized[None]).to(self.device).float(),
            torch.tensor([arm], device=self.device),
            _mask(_resize_mask(source_label == 2), self.device)[None],
            _mask(_resize_mask(parent_label == 2), self.device)[None],
            _mask(_resize_mask(parent_label == 1), self.device)[None],
        )
        height, width = parent_crop.shape[1:3]
        logits = F.interpolate(logits.flatten(0, 1), (height, width), mode="bilinear", align_corners=False).unflatten(0, (1, 8))
        residual = F.interpolate(residual.flatten(0, 1), (height, width), mode="bilinear", align_corners=False).unflatten(0, (1, 8))
        rendered, _, _ = render_black_gripper(
            _rgb(parent_crop, self.device)[None],
            _mask(source_label == 2, self.device)[None],
            _mask(parent_label == 2, self.device)[None],
            logits,
            residual,
            1.0,
        )
        rendered = rendered[0].permute(0, 2, 3, 1).mul(255).cpu().numpy()
        output = parent.copy()
        output[2:, cy0:cy1, cx0:cx1] = np.clip(np.round(rendered[2:]), 0, 255).astype(np.uint8)
        return output

    def predict(self, context_frames, history_actions, future_actions, seed: int, instruction: str | None):
        if context_frames.shape != (CONTEXT_FRAMES, 256, 256, 3) or context_frames.dtype != np.uint8:
            raise ValueError("context_frames must be [5,256,256,3] uint8")
        if history_actions.shape != (4, ACTION_DIM) or future_actions.shape != (PREDICTION_FRAMES, ACTION_DIM):
            raise ValueError("actions must be [4,14] history and [8,14] future")
        parent = self.parent.predict(context_frames, history_actions, future_actions, seed, instruction)
        parent = self._v10(parent, context_frames, history_actions, future_actions)
        parent = self._v141(parent, context_frames, history_actions, future_actions)
        return self._v15(parent, context_frames, history_actions, future_actions)

    def predict_batch(
        self,
        context_frames,
        history_actions,
        future_actions,
        seeds,
        instructions,
    ):
        batch = int(context_frames.shape[0])
        if context_frames.shape != (batch, CONTEXT_FRAMES, 256, 256, 3) or context_frames.dtype != np.uint8:
            raise ValueError("context_frames must be [B,5,256,256,3] uint8")
        if history_actions.shape != (batch, 4, ACTION_DIM) or future_actions.shape != (batch, PREDICTION_FRAMES, ACTION_DIM):
            raise ValueError("actions must be [B,4,14] history and [B,8,14] future")
        if len(seeds) != batch or len(instructions) != batch:
            raise ValueError("batched request metadata has inconsistent length")
        parent = self.parent.predict_batch(
            context_frames, history_actions, future_actions, seeds, instructions
        )
        parent = self._v10_batch(parent, context_frames, history_actions, future_actions)
        # Retrieval/ECC and the contact expert contain data-dependent routing.
        # Keep those lightweight stages per sample while batching every dense
        # neural stage above.
        output = np.empty_like(parent)
        for index in range(batch):
            refined = self._v141(
                parent[index],
                context_frames[index],
                history_actions[index],
                future_actions[index],
            )
            output[index] = self._v15(
                refined,
                context_frames[index],
                history_actions[index],
                future_actions[index],
            )
        return output
