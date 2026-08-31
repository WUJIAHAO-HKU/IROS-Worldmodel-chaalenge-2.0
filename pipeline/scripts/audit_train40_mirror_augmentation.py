#!/usr/bin/env python3
"""Audit a left-to-right Aloha mirror using only the fixed public train40 split."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


MIRROR_SIGN = np.asarray([-1, 1, 1, 1, -1, -1, 1], dtype=np.float32)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def resample(values: np.ndarray, length: int = 128) -> np.ndarray:
    position = np.linspace(0, len(values) - 1, length)
    lo = np.floor(position).astype(int)
    hi = np.minimum(lo + 1, len(values) - 1)
    weight = position - lo
    return values[lo] * (1 - weight[:, None]) + values[hi] * weight[:, None]


def decode(value: np.bytes_) -> np.ndarray:
    with Image.open(io.BytesIO(value.tobytes())) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def classify(actions: np.ndarray) -> int:
    paths = [
        float(np.linalg.norm(np.diff(actions[:, start : start + 6], axis=0), axis=1).sum())
        for start in (0, 7)
    ]
    return int(paths[1] > paths[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    split = json.loads(args.split.read_text())
    train = [int(x) for x in split["train_episodes"]]
    forbidden = set(split["validation_episodes"]) | set(split["local_test_episodes"])
    assert len(train) == 40 and not (set(train) & forbidden)

    left, right = [], []
    left_images, right_images = [], []
    episodes = []
    involution_error = 0.0
    for episode in train:
        path = args.source / "data" / f"episode{episode}.hdf5"
        with h5py.File(path, "r") as item:
            actions = np.asarray(item["joint_action/vector"], dtype=np.float32)
            images = item["observation/head_camera/rgb"]
            arm = classify(actions)
            active = actions[:, 7:14] if arm else actions[:, 0:7]
            (right if arm else left).append(resample(active))
            for fraction in (0.2, 0.5, 0.8):
                index = min(len(images) - 1, int(round(fraction * (len(images) - 1))))
                image = decode(images[index])
                (right_images if arm else left_images).append(image)
            mirrored_twice = (active * MIRROR_SIGN) * MIRROR_SIGN
            involution_error = max(involution_error, float(np.max(np.abs(active - mirrored_twice))))
            episodes.append({"episode": episode, "arm": "right" if arm else "left"})

    assert len(left) == 25 and len(right) == 15
    left_mean = np.mean(left, axis=0)
    right_mean = np.mean(right, axis=0)

    def error(prediction: np.ndarray) -> dict:
        delta = prediction - right_mean
        return {
            "mae": float(np.abs(delta).mean()),
            "rmse": float(np.sqrt(np.square(delta).mean())),
            "mae_per_joint": np.abs(delta).mean(0).tolist(),
        }

    left_image_mean = np.mean(left_images, axis=0)
    right_image_mean = np.mean(right_images, axis=0)
    plain_image_mae = float(np.abs(left_image_mean - right_image_mean).mean())
    flipped_image_mae = float(np.abs(left_image_mean[:, ::-1] - right_image_mean).mean())
    trajectory = {
        "plain_swap": error(left_mean),
        "signed_mirror": error(left_mean * MIRROR_SIGN),
    }
    checks = {
        "fixed_train40_only": len(train) == 40 and not (set(train) & forbidden),
        "expected_arm_counts": len(left) == 25 and len(right) == 15,
        "action_mirror_is_exact_involution": involution_error == 0.0,
        "signed_mirror_improves_action_rmse": (
            trajectory["signed_mirror"]["rmse"] < trajectory["plain_swap"]["rmse"]
        ),
        "horizontal_flip_improves_mean_image_mae": flipped_image_mae < plain_image_mae,
    }
    report = {
        "format": "strict-track2-public-train40-mirror-audit-v1",
        "source": str(args.source.resolve()),
        "split": str(args.split.resolve()),
        "split_sha256": digest(args.split),
        "train_episodes": train,
        "forbidden_episodes_read": [],
        "left_episode_count": len(left),
        "right_episode_count": len(right),
        "mirror_sign": MIRROR_SIGN.tolist(),
        "action_involution_max_abs_error": involution_error,
        "trajectory_alignment_error": trajectory,
        "mean_image_alignment": {
            "plain_mae": plain_image_mae,
            "horizontal_flip_mae": flipped_image_mae,
        },
        "episodes": episodes,
        "checks": checks,
        "accepted": all(checks.values()),
        "reserved_or_hidden_evaluation_access": False,
        "real_competition_submission": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["accepted"] else 5)


if __name__ == "__main__":
    main()
