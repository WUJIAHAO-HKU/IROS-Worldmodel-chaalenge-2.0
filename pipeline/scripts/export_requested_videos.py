#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from wam_pipeline.backends import build_backend


ROOT = Path("/tmp/iros_track2")
OUT = ROOT / "artifacts/strict_track2_official_20260810/video_exports/final_bundle"
WINDOWS = ROOT / "artifacts/adjust_bottle_windows_full"
RELEASE = ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v169_instruction_arm_routed_release"
RL_CAPTURE = ROOT / "artifacts/strict_track2_official_20260810/video_exports/success_capture"


def start(path: Path) -> int:
    return int(re.search(r"_(\d+)\.npz$", path.name).group(1))


def annotate(frame: np.ndarray, text: str) -> np.ndarray:
    frame = np.ascontiguousarray(frame.copy())
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(frame, text, (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def write_video(path: Path, frames: list[np.ndarray], fps: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=8)


def validation_video(backend) -> dict:
    paths = sorted(WINDOWS.glob("episode16_*.npz"), key=start)
    by_start = {start(p): p for p in paths}
    selected = [by_start[s] for s in sorted(by_start) if (s - min(by_start)) % 8 == 0]
    frames: list[np.ndarray] = []
    for chunk_index, path in enumerate(selected):
        with np.load(path, allow_pickle=False) as z:
            context = z["context_frames"]
            history = z["history_actions"]
            actions = z["future_actions"]
            truth = z["target_frames"]
        prediction = backend.predict(context, history, actions, 160000 + chunk_index, "adjust_bottle")
        for horizon, (pred, target) in enumerate(zip(prediction, truth), 1):
            left = annotate(target, f"validation truth | episode16 frame {start(path)+4+horizon}")
            right = annotate(pred, f"V16.9 world model | predicted t+{horizon}")
            frames.append(np.concatenate([left, right], axis=1))
    output = OUT / "01_world_model_validation_full_episode.mp4"
    write_video(output, frames)
    return {"path": str(output), "episode": 16, "frames": len(frames), "prediction_chunks": len(selected)}


def rl_actions_world_model_video(backend) -> dict:
    action_paths = sorted((RL_CAPTURE / "actions").glob("policy_action_chunk_*.npy"))
    if not action_paths:
        raise RuntimeError("no captured RL policy actions")
    reader = imageio.get_reader(RL_CAPTURE / "video/seed_0/0.mp4")
    real_frames = [cv2.resize(np.asarray(frame)[..., :3], (256, 256), interpolation=cv2.INTER_AREA) for frame in reader]
    reader.close()
    if len(real_frames) < 5:
        raise RuntimeError("RL trajectory video has fewer than five context frames")
    context = np.stack(real_frames[:5]).astype(np.uint8)
    history = np.zeros((4, 14), dtype=np.float32)
    output_frames: list[np.ndarray] = []
    for chunk_index, action_path in enumerate(action_paths):
        actions = np.load(action_path, allow_pickle=False)
        actions = np.asarray(actions).reshape(-1, actions.shape[-1])[:8].astype(np.float32)
        prediction = backend.predict(context, history, actions, 430000 + chunk_index, "adjust_bottle")
        for horizon, frame in enumerate(prediction, 1):
            output_frames.append(
                annotate(frame, f"RL global_step_4 actions -> V16.9 world model | chunk {chunk_index+1} t+{horizon}")
            )
        context = prediction[-5:].copy()
        history = actions[-4:].copy()
    output = OUT / "04_rl_actions_to_world_model_full_prediction.mp4"
    write_video(output, output_frames)
    np.save(OUT / "rl_policy_action_sequence.npy", np.concatenate([
        np.load(path, allow_pickle=False).reshape(-1, 14)[:8] for path in action_paths
    ], axis=0))
    return {
        "path": str(output),
        "source_policy": "official Pi0.5 RL global_step_4",
        "source_seed": 100100043,
        "action_chunks": len(action_paths),
        "predicted_frames": len(output_frames),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    backend = build_backend(
        "v169-arm-routed",
        str(RELEASE),
        "cuda",
        v15_library_dir=ROOT / "artifacts",
    )
    report = {
        "format": "track2-requested-video-bundle-v1",
        "world_model_validation": validation_video(backend),
        "rl_actions_world_model": rl_actions_world_model_video(backend),
    }
    (OUT / "generation_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
