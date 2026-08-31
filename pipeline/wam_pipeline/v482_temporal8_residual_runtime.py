"""Causal temporal-8 residual runtime over a frozen scalar-v169 comparator."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from wam_pipeline.v169_arm_routed_runtime import Track2V169ArmRoutedRuntime


FORMAT = "track2-v482-temporal8-residual-release-v1"
CHECKPOINT_FORMAT = "strict-track2-v482-temporal8-residual-checkpoint-v1"
WORKING_RESOLUTION = 128
ACTION_FEATURE_DIM = 178
FEATURE_SCHEMA = {
    "dim": ACTION_FEATURE_DIM,
    "normalized_history": 56,
    "causal_normalized_future": 112,
    "prefix_mask": 8,
    "prefix_fraction": 1,
    "prefix_postclose": 1,
    "future_after_prefix_exact_zero": True,
    "normalization": "immutable selection lower/upper to [-1,1]",
}


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


def directory_target_sha(path):
    path = Path(path).resolve(); items = []
    for parent, dirs, files in os.walk(path, followlinks=False):
        for name in sorted(dirs + files):
            item = Path(parent) / name; relative = str(item.relative_to(path))
            if item.is_symlink(): items.append(["link", relative, os.readlink(item)])
            elif item.is_file(): items.append(["file", relative, sha(item)])
            else: items.append(["dir", relative, None])
    return hashlib.sha256(json.dumps(items, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def release_inventory(release, joint_root):
    release = Path(release).absolute(); joint_root = Path(joint_root).resolve()
    items, seen_dirs, seen_files, active = [], set(), set(), set()
    def visit(logical, relative):
        resolved = logical.resolve()
        if not inside(resolved, joint_root) or not resolved.is_dir(): raise RuntimeError("v482 inventory directory escape")
        dir_key = (resolved.stat().st_dev, resolved.stat().st_ino)
        if dir_key in active or dir_key in seen_dirs: raise RuntimeError("v482 inventory directory cycle/duplicate")
        active.add(dir_key); seen_dirs.add(dir_key)
        for child in sorted(logical.iterdir(), key=lambda x: x.name):
            rel, target = relative / child.name, child.resolve()
            if not inside(target, joint_root): raise RuntimeError("v482 inventory target escape")
            if child.is_symlink():
                items.append(("directory_symlink" if target.is_dir() else "file_symlink", "v169_release_link", str(rel), str(child.absolute()), os.readlink(child), str(target)))
            if target.is_dir(): visit(child, rel)
            elif target.is_file():
                key = (target.stat().st_dev, target.stat().st_ino)
                if key in seen_files: raise RuntimeError("v482 inventory duplicate file target")
                seen_files.add(key)
                if not child.is_symlink():
                    items.append(("file", "v169_release_target" if not inside(target, release.resolve()) else "v169_release", str(rel), str(child.absolute()), None, str(target)))
            else: raise RuntimeError("v482 inventory unsupported target")
        active.remove(dir_key)
    visit(release, Path("."))
    return sorted(items)


def verify_v169(value):
    if not isinstance(value, dict): raise RuntimeError("v482 v169 closure")
    release, library = Path(value["release"]).absolute(), Path(value["library"]).resolve()
    joint_root = release.resolve().parent
    roots = {"v169_release": release, "v169_release_target": release, "v169_release_link": release,
             "v169_base_release": (release / "v168_release/base_release").resolve(), "v169_library": library}
    seen = set()
    for records in (value["release_files"], value["library_files"]):
        for row in records:
            if set(row) != {"record_type", "base_kind", "relative", "lexical_path", "link_target", "resolved_path", "target_sha"} or row["base_kind"] not in roots:
                raise RuntimeError("v482 closure schema")
            base = roots[row["base_kind"]]; lexical = (base / row["relative"]).absolute(); resolved = lexical.resolve()
            key = (row["record_type"], row["base_kind"], row["relative"], row["resolved_path"])
            if key in seen: raise RuntimeError("v482 duplicate closure record")
            seen.add(key); allowed = library if row["base_kind"] == "v169_library" else joint_root
            if os.path.commonpath((str(base), str(lexical))) != str(base) or not inside(resolved, allowed) or str(lexical) != row["lexical_path"] or str(resolved) != row["resolved_path"]:
                raise RuntimeError("v482 closure path drift")
            is_link = lexical.is_symlink()
            if row["link_target"] != (os.readlink(lexical) if is_link else None): raise RuntimeError("v482 symlink text drift")
            if row["record_type"] == "directory_symlink":
                if not is_link or not resolved.is_dir() or directory_target_sha(resolved) != row["target_sha"]: raise RuntimeError("v482 directory symlink drift")
            elif not resolved.is_file() or sha(resolved) != row["target_sha"]: raise RuntimeError("v482 file closure drift")
    declared = sorted((r["record_type"], r["base_kind"], r["relative"], r["lexical_path"], r["link_target"], r["resolved_path"]) for r in value["release_files"])
    if declared != release_inventory(release, joint_root) or sum(r["record_type"] == "directory_symlink" for r in value["release_files"]) != 4 or sum(r["record_type"] == "file_symlink" for r in value["release_files"]) != 2 or len({r["resolved_path"] for r in value["release_files"]}) != len(value["release_files"]):
        raise RuntimeError("v482 exact release inventory")
    for key, path in (("release_manifest_sha256", release / "v169_arm_routed_manifest.json"), ("library_manifest_sha256", Path(value["library_manifest"]))):
        if not path.is_file() or sha(path) != value[key]: raise RuntimeError(f"v482 {key}")
    if not Path(value["runtime_source"]["path"]).is_file() or sha(value["runtime_source"]["path"]) != value["runtime_source"]["sha256"]:
        raise RuntimeError("v482 v169 runtime source")
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def explicit_right(text: str) -> bool:
    value = str(text).lower()
    return "right arm" in value and "left arm" not in value


def request_gate(history: np.ndarray, future: np.ndarray, instruction: str) -> dict:
    history = np.asarray(history)
    future = np.asarray(future)
    enabled = (
        history.shape == (4, 14)
        and future.shape == (8, 14)
        and np.isfinite(history).all()
        and np.isfinite(future).all()
        and explicit_right(instruction)
        and float(history[-1, 13]) < 0.5
        and bool(np.all(future[:, 13] < 0.5))
    )
    return {
        "gate": bool(enabled),
        "explicit_right": bool(explicit_right(instruction)),
        "phase": "postclose" if enabled else "g0",
    }


def zero_span_observation_counts(history, future, lower, upper):
    values = np.concatenate((np.asarray(history, np.float32), np.asarray(future, np.float32)), axis=0)
    lower = np.asarray(lower, np.float32)
    zero = np.asarray(upper, np.float32) == lower
    return {
        str(index): {
            "below": int(np.sum(values[:, index] < lower[index])),
            "equal": int(np.sum(values[:, index] == lower[index])),
            "above": int(np.sum(values[:, index] > lower[index])),
        }
        for index in np.flatnonzero(zero)
    }


def causal_action_features(
    history: np.ndarray,
    future: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    mode: str = "action",
    strict_source: bool = False,
) -> np.ndarray:
    """Return N x 8 x 178 request-only features; frame k sees future[:k+1]."""
    history = np.asarray(history, np.float32)
    future = np.asarray(future, np.float32)
    lower = np.asarray(lower, np.float32)
    upper = np.asarray(upper, np.float32)
    if history.ndim == 2:
        history = history[None]
        future = future[None]
    n = len(history)
    if history.shape != (n, 4, 14) or future.shape != (n, 8, 14):
        raise ValueError("v482 action shape contract")
    if lower.shape != (14,) or upper.shape != (14,) or np.any(upper < lower):
        raise ValueError("v482 normalization contract")
    span = upper - lower
    zero = span == 0
    if strict_source and (np.any(history[..., zero] != lower[zero]) or np.any(future[..., zero] != lower[zero])):
        raise ValueError("v482 zero-span action dimension drift")
    normalized_h = np.zeros_like(history, dtype=np.float32)
    normalized_f = np.zeros_like(future, dtype=np.float32)
    nonzero = ~zero
    normalized_h[..., nonzero] = np.float32(
        2.0 * ((np.clip(history[..., nonzero], lower[nonzero], upper[nonzero]) - lower[nonzero]) / span[nonzero]) - 1.0
    )
    normalized_f[..., nonzero] = np.float32(
        2.0 * ((np.clip(future[..., nonzero], lower[nonzero], upper[nonzero]) - lower[nonzero]) / span[nonzero]) - 1.0
    )
    output = np.zeros((n, 8, ACTION_FEATURE_DIM), np.float32)
    for frame in range(8):
        causal_future = np.zeros((n, 8, 14), np.float32)
        causal_future[:, : frame + 1] = normalized_f[:, : frame + 1]
        values = np.concatenate((normalized_h, causal_future), axis=1).reshape(n, 168)
        if mode == "context_only":
            values.fill(0)
        elif mode != "action":
            raise ValueError(f"unsupported v482 feature mode {mode!r}")
        mask = np.zeros((n, 8), np.float32)
        mask[:, : frame + 1] = 1
        fraction = np.full((n, 1), (frame + 1) / 8.0, np.float32)
        postclose = (
            (history[:, -1, 13] < 0.5)
            & np.all(future[:, : frame + 1, 13] < 0.5, axis=1)
        ).astype(np.float32)[:, None]
        output[:, frame] = np.concatenate((values, mask, fraction, postclose), axis=1)
    return output


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = min(8, int(channels))
        self.layers = nn.Sequential(
            nn.GroupNorm(groups, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value):
        return value + self.layers(value)


class FuseBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.project = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.residual = ResidualBlock(output_channels)

    def forward(self, value):
        return self.residual(self.project(value))


class TemporalResidualUNet128FiLM(nn.Module):
    def __init__(self, channels: int = 16):
        super().__init__()
        self.channels = int(channels)
        if channels != 16:
            raise ValueError("v482 frozen channels must equal 16")
        self.stem = nn.Conv2d(9, 16, 3, padding=1)
        self.enc16 = ResidualBlock(16)
        self.down32 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.enc32 = ResidualBlock(32)
        self.down64 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.enc64 = ResidualBlock(64)
        self.down96 = nn.Conv2d(64, 96, 3, stride=2, padding=1)
        self.bottleneck1 = ResidualBlock(96)
        self.bottleneck2 = ResidualBlock(96)
        self.condition = nn.Sequential(
            nn.Linear(ACTION_FEATURE_DIM, 256),
            nn.SiLU(),
            nn.Linear(256, 192),
        )
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)
        self.up64 = nn.Conv2d(96, 64, 3, padding=1)
        self.fuse64 = FuseBlock(128, 64)
        self.up32 = nn.Conv2d(64, 32, 3, padding=1)
        self.fuse32 = FuseBlock(64, 32)
        self.up16 = nn.Conv2d(32, 16, 3, padding=1)
        self.fuse16 = FuseBlock(32, 16)
        self.output = nn.Conv2d(16, 3, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if sum(parameter.numel() for parameter in self.parameters()) > 1_200_000:
            raise RuntimeError("v482 parameter cap exceeded")

    def forward(self, baseline, context_last, features):
        batch, _, height, width = baseline.shape
        if (
            context_last.shape != (batch, 3, height, width)
            or features.shape != (batch, ACTION_FEATURE_DIM)
        ):
            raise ValueError("v482 tensor shape contract")
        if (height, width) != (256, 256):
            raise ValueError("v482 exact 256px visual contract")
        base = F.avg_pool2d(baseline, kernel_size=2, stride=2)
        context = F.avg_pool2d(context_last, kernel_size=2, stride=2)
        visual = torch.cat((context, base, base - context), dim=1)
        skip16 = self.enc16(self.stem(visual))
        skip32 = self.enc32(self.down32(skip16))
        skip64 = self.enc64(self.down64(skip32))
        encoded = self.bottleneck1(self.down96(skip64))
        film = self.condition(features).reshape(batch, 2, 96)
        gamma = film[:, 0].to(encoded.dtype)[:, :, None, None]
        bias = film[:, 1].to(encoded.dtype)[:, :, None, None]
        encoded = self.bottleneck2(encoded * (1 + gamma) + bias)
        decoded64 = F.interpolate(encoded, scale_factor=2, mode="nearest")
        decoded64 = self.fuse64(torch.cat((self.up64(decoded64), skip64), dim=1))
        decoded32 = F.interpolate(decoded64, scale_factor=2, mode="nearest")
        decoded32 = self.fuse32(torch.cat((self.up32(decoded32), skip32), dim=1))
        decoded16 = F.interpolate(decoded32, scale_factor=2, mode="nearest")
        decoded16 = self.fuse16(torch.cat((self.up16(decoded16), skip16), dim=1))
        return torch.tanh(self.output(decoded16))


class Track2V482Temporal8ResidualRuntime:
    """Ordered scalar-v169 baseline plus a causal shared-frame residual head."""

    def __init__(self, release_dir: str | Path, device: str = "cuda"):
        root = Path(release_dir).resolve()
        manifest_path = root / "v482_temporal8_residual_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        checkpoint = (root / manifest["checkpoint"]).resolve()
        v169_release = Path(manifest["v169_release"]).resolve()
        v169_library = Path(manifest["v169_library"]).resolve()
        v169_closure_path = (root / manifest["v169_closure"]).resolve()
        if (
            manifest.get("format") != FORMAT
            or manifest.get("endpoint_only") is not False
            or manifest.get("ordered_scalar_v169") is not True
            or manifest.get("reward_or_outcome_used") is not False
            or root not in checkpoint.parents
            or sha(Path(__file__)) != manifest["sha256"]["runtime_source"]
            or sha(checkpoint) != manifest["sha256"]["checkpoint"]
            or root not in v169_closure_path.parents
            or sha(v169_closure_path) != manifest["sha256"]["v169_closure"]
        ):
            raise RuntimeError("bad v482 release")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            state.get("format") != CHECKPOINT_FORMAT
            or state.get("training_scope") != "all200-action"
            or state.get("precision") != "bf16"
            or state.get("feature_schema") != FEATURE_SCHEMA
            or state.get("runtime_sha256") != manifest["sha256"]["runtime_source"]
            or state.get("closure_digest") != manifest["closure_digest"]
            or not all(bool(torch.isfinite(value).all()) for value in state.get("model", {}).values())
        ):
            raise RuntimeError("bad v482 checkpoint")
        self.device = torch.device(device)
        self.model = TemporalResidualUNet128FiLM(state["channels"]).to(self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.lower = np.asarray(state["action_lower"], np.float32)
        self.upper = np.asarray(state["action_upper"], np.float32)
        if (
            self.lower.shape != (14,)
            or self.upper.shape != (14,)
            or not np.isfinite(self.lower).all()
            or not np.isfinite(self.upper).all()
            or np.any(self.upper < self.lower)
            or self.lower[13] != np.float32(0)
            or self.upper[13] != np.float32(0)
        ):
            raise RuntimeError("bad v482 action normalization")
        for key, path in (
            ("v169_release_manifest", v169_release / "v169_arm_routed_manifest.json"),
            ("v169_library_manifest", Path(manifest["v169_library_manifest"])),
        ):
            if not path.is_file() or sha(path) != manifest["sha256"][key]:
                raise RuntimeError(f"bad {key}")
        closure = json.loads(v169_closure_path.read_text())
        if verify_v169(closure) != manifest["v169_closure_digest"]:
            raise RuntimeError("bad v482 v169 closure digest")
        self.v169 = Track2V169ArmRoutedRuntime(v169_release, v169_library, device)

    def _scalar_baseline(self, context, history, future, seed, instruction):
        value = self.v169.predict(context, history, future, int(seed), str(instruction))
        value = np.asarray(value)
        if value.shape != (8, 256, 256, 3) or value.dtype != np.uint8:
            raise RuntimeError("v482 scalar v169 output contract")
        return value

    @torch.inference_mode()
    def predict_one_with_baseline(self, context, history, future, seed, instruction):
        context = np.asarray(context)
        history = np.asarray(history)
        future = np.asarray(future)
        if (
            context.shape != (5, 256, 256, 3)
            or context.dtype != np.uint8
            or history.shape != (4, 14)
            or future.shape != (8, 14)
            or not np.issubdtype(history.dtype, np.number)
            or not np.issubdtype(future.dtype, np.number)
            or not np.isfinite(history).all()
            or not np.isfinite(future).all()
        ):
            raise ValueError("v482 scalar request contract")
        seed_array = np.asarray(seed)
        if seed_array.ndim != 0:
            raise ValueError("v482 scalar seed contract")
        canonical_seed = int(seed_array.item())
        if isinstance(seed_array.item(), (float, np.floating)) and float(canonical_seed) != float(seed_array.item()):
            raise ValueError("v482 lossy scalar seed")
        canonical_instruction = str(instruction)
        baseline = self._scalar_baseline(context, history, future, canonical_seed, canonical_instruction)
        decision = request_gate(history, future, canonical_instruction)
        decision["zero_span_observation_counts"] = zero_span_observation_counts(
            history, future, self.lower, self.upper
        )
        if not decision["gate"]:
            return baseline, baseline, decision
        features = causal_action_features(
            history, future, self.lower, self.upper, strict_source=False
        ).reshape(8, ACTION_FEATURE_DIM)
        base = torch.as_tensor(baseline, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2) / 255.0
        last = torch.as_tensor(
            np.repeat(context[-1][None], 8, axis=0), dtype=torch.float32, device=self.device
        ).permute(0, 3, 1, 2) / 255.0
        feat = torch.as_tensor(features, device=self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            residual = self.model(base, last, feat)
        if not bool(torch.isfinite(residual).all()):
            raise RuntimeError("v482 nonfinite runtime residual")
        residual = residual.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        candidate = (255.0 * base + 255.0 * residual).clamp(0, 255)
        if not bool(torch.isfinite(candidate).all()):
            raise RuntimeError("v482 nonfinite runtime candidate")
        output = np.clip(
            np.rint(candidate.permute(0, 2, 3, 1).float().cpu().numpy()), 0, 255
        ).astype(np.uint8)
        return baseline, output, decision

    def predict_one(self, context, history, future, seed, instruction):
        return self.predict_one_with_baseline(context, history, future, seed, instruction)[1]

    def predict_batch_with_baseline(self, context, history, future, seeds, instructions):
        context = np.asarray(context)
        history = np.asarray(history)
        future = np.asarray(future)
        seeds = np.asarray(seeds)
        n = len(context)
        if (
            context.shape != (n, 5, 256, 256, 3)
            or history.shape != (n, 4, 14)
            or future.shape != (n, 8, 14)
            or seeds.shape != (n,)
            or len(instructions) != n
        ):
            raise ValueError("v482 batch request contract")
        baseline, output, decisions = [], [], []
        for index in range(n):
            b0, prediction, decision = self.predict_one_with_baseline(
                context[index], history[index], future[index], seeds[index], instructions[index]
            )
            baseline.append(b0)
            output.append(prediction)
            decisions.append(decision)
        return np.stack(baseline), np.stack(output), decisions

    def predict_batch(self, *args):
        return self.predict_batch_with_baseline(*args)[1]
