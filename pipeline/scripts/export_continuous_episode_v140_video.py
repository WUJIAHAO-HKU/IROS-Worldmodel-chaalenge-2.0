#!/usr/bin/env python3
"""Export full-episode, deduplicated v14.0 evaluation videos and metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


CONTACT_REGION = (72, 232, 30, 210)


def parse_start(name: str) -> int:
    match = re.search(r"_(\d+)\.npz$", name)
    if match is None:
        raise ValueError(f"cannot parse window start from {name!r}")
    return int(match.group(1))


def verify_timeline(names: np.ndarray, target: np.ndarray) -> np.ndarray:
    starts = np.asarray([parse_start(name) for name in names], dtype=np.int64)
    if len(starts) > 1 and not np.all(np.diff(starts) == 1):
        raise ValueError("windows must be ordered and consecutive")
    for index in range(1, len(target)):
        if not np.array_equal(target[index - 1, 1:], target[index, :-1]):
            raise ValueError(f"target timeline discontinuity before {names[index]}")
    return starts


def chunked_open_loop_indices(starts: np.ndarray, horizon: int) -> list[tuple[int, int, bool]]:
    """Cover every global target time once with non-overlapping rollouts.

    Returns ``(window_index, horizon_index, is_reacquisition_boundary)``.
    """
    first, last = int(starts[0]), int(starts[-1])
    by_start = {int(start): index for index, start in enumerate(starts)}
    output: list[tuple[int, int, bool]] = []
    global_time = first + 5
    final_time = last + 5 + horizon - 1
    while global_time <= final_time:
        preferred_start = first + ((global_time - (first + 5)) // horizon) * horizon
        start = min(preferred_start, last)
        horizon_index = global_time - (start + 5)
        if not 0 <= horizon_index < horizon or start not in by_start:
            # Near the tail, use the newest available window that contains the
            # requested global target while retaining the largest valid horizon.
            start = min(last, global_time - 5)
            horizon_index = global_time - (start + 5)
        output.append((by_start[start], int(horizon_index), not output or horizon_index == 0))
        global_time += 1
    return output


def hardest_horizon_indices(starts: np.ndarray, horizon: int) -> list[tuple[int, int, bool]]:
    return [(index, horizon - 1, index == 0) for index in range(len(starts))]


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(prediction.astype(np.float32) - target.astype(np.float32)).mean())


def metric_block(parent: np.ndarray, refined: np.ndarray, target: np.ndarray,
                 version_key: str = "v140") -> dict:
    y0, y1, x0, x1 = CONTACT_REGION
    before = np.abs(parent.astype(np.float32) - target.astype(np.float32))
    after = np.abs(refined.astype(np.float32) - target.astype(np.float32))
    before_contact = before[..., y0:y1, x0:x1, :]
    after_contact = after[..., y0:y1, x0:x1, :]

    def values(a: np.ndarray, b: np.ndarray) -> dict:
        left, right = float(a.mean()), float(b.mean())
        return {
            "parent_mae": left,
            f"{version_key}_mae": right,
            "improvement_percent": 100.0 * (left - right) / max(left, 1e-12),
        }

    return {"rgb": values(before, after), "contact_rgb": values(before_contact, after_contact)}


def transition_metric(prediction: np.ndarray, target: np.ndarray, boundaries: np.ndarray) -> dict:
    pred_delta = np.diff(prediction.astype(np.float32), axis=0)
    target_delta = np.diff(target.astype(np.float32), axis=0)
    error = np.abs(pred_delta - target_delta).mean(axis=(1, 2, 3))
    boundary_transitions = boundaries[1:]
    return {
        "all_transition_delta_mae": float(error.mean()),
        "reacquisition_boundary_delta_mae": float(error[boundary_transitions].mean())
        if boundary_transitions.any() else None,
        "within_rollout_delta_mae": float(error[~boundary_transitions].mean())
        if (~boundary_transitions).any() else None,
    }


def panel(frame: np.ndarray, title: str, subtitle: str, size: int = 320) -> np.ndarray:
    canvas = Image.new("RGB", (size, size + 48), "white")
    canvas.paste(Image.fromarray(cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)), (0, 48))
    draw = ImageDraw.Draw(canvas)
    draw.text((7, 7), title, fill="black")
    draw.text((7, 25), subtitle, fill=(65, 65, 65))
    return np.asarray(canvas)


def zoom_panel(parent: np.ndarray, refined: np.ndarray, target: np.ndarray,
               version_label: str, size: int = 320) -> np.ndarray:
    y0, y1, x0, x1 = CONTACT_REGION
    canvas = Image.new("RGB", (size, size + 48), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((7, 7), "contact zoom", fill="black")
    draw.text((7, 25), f"parent / {version_label} / truth", fill=(65, 65, 65))
    height = size // 3
    for row, (frame, label) in enumerate(((parent, "parent"), (refined, version_label), (target, "truth"))):
        crop = cv2.resize(frame[y0:y1, x0:x1], (size, height), interpolation=cv2.INTER_AREA)
        canvas.paste(Image.fromarray(crop), (0, 48 + row * height))
        draw.text((5, 51 + row * height), label, fill=(255, 40, 40))
    return np.asarray(canvas)


def render_frames(
    entries: list[tuple[int, int, bool]], starts: np.ndarray, parent: np.ndarray,
    refined: np.ndarray, target: np.ndarray, accepted: set[str], names: np.ndarray,
    version_label: str,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames, p_values, r_values, t_values, boundaries = [], [], [], [], []
    for sequence_index, (window_index, horizon_index, boundary) in enumerate(entries):
        p, r, t = (value[window_index, horizon_index] for value in (parent, refined, target))
        global_time = int(starts[window_index]) + 5 + horizon_index
        changed = names[window_index] in accepted or not np.array_equal(p, r)
        label = "retrieval ON" if changed else "safe fallback"
        frames.append(np.concatenate((
            panel(t, "ground truth", f"episode frame {global_time}"),
            panel(p, "parent pipeline", f"MAE {mae(p, t):.3f}"),
            panel(r, f"final {version_label} pipeline", f"MAE {mae(r, t):.3f} | {label}"),
            zoom_panel(p, r, t, version_label),
        ), axis=1))
        p_values.append(p); r_values.append(r); t_values.append(t); boundaries.append(boundary)
    return frames, np.stack(p_values), np.stack(r_values), np.stack(t_values), np.asarray(boundaries)


def save_video_set(output: Path, stem: str, frames: list[np.ndarray], fps: int) -> None:
    imageio.mimsave(output / f"{stem}.mp4", frames, fps=fps, codec="libx264", quality=8)
    preview = frames[:: max(1, len(frames) // 36)]
    imageio.mimsave(output / f"{stem}_preview.gif", preview, fps=max(1, fps // 2), loop=0)
    sheet_indices = np.linspace(0, len(frames) - 1, min(12, len(frames)), dtype=np.int64)
    sheet = np.concatenate([frames[int(index)] for index in sheet_indices], axis=0)
    Image.fromarray(sheet).save(output / f"{stem}_sheet.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--retrieval-cache", required=True)
    parser.add_argument("--retrieval-report")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--version-label", default="v14.0")
    args = parser.parse_args()

    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent, target = cache["prediction"], cache["target"]
        names = cache["windows"].astype(str)
    with np.load(args.retrieval_cache, allow_pickle=False) as cache:
        refined, refined_target = cache["prediction"], cache["target"]
        refined_names = cache["windows"].astype(str)
    if not np.array_equal(names, refined_names) or not np.array_equal(target, refined_target):
        raise ValueError("parent and retrieval caches do not share an identical timeline")
    starts = verify_timeline(names, target)
    accepted: set[str] = set()
    if args.retrieval_report:
        report = json.loads(Path(args.retrieval_report).read_text())
        accepted = {value["window"] for value in report.get("windows", []) if value.get("accepted")}

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    version_slug = args.version_label.lower().replace(".", "").replace(" ", "_")
    modes = {
        "open_loop_full_episode": chunked_open_loop_indices(starts, parent.shape[1]),
        "fixed_t8_full_episode": hardest_horizon_indices(starts, parent.shape[1]),
    }
    result = {
        "format": f"track2-{args.version_label}-continuous-episode-evaluation",
        "episode": names[0].split("_")[0],
        "window_count": len(names),
        "accepted_window_count": len(accepted),
        "accepted_windows": sorted(accepted),
        "all_windows": metric_block(parent, refined, target, version_slug),
        "horizons": [],
        "modes": {},
    }
    for horizon_index in range(parent.shape[1]):
        result["horizons"].append({
            "horizon": horizon_index + 1,
            **metric_block(parent[:, horizon_index], refined[:, horizon_index], target[:, horizon_index], version_slug),
        })
    for stem, entries in modes.items():
        frames, p_values, r_values, t_values, boundaries = render_frames(
            entries, starts, parent, refined, target, accepted, names, args.version_label
        )
        save_video_set(output, f"best_{version_slug}_{stem}", frames, args.fps)
        result["modes"][stem] = {
            "frame_count": len(frames),
            "first_global_frame": int(starts[entries[0][0]]) + 5 + entries[0][1],
            "last_global_frame": int(starts[entries[-1][0]]) + 5 + entries[-1][1],
            "metrics": metric_block(p_values, r_values, t_values, version_slug),
            "parent_temporal": transition_metric(p_values, t_values, boundaries),
            f"{version_slug}_temporal": transition_metric(r_values, t_values, boundaries),
        }
    (output / f"best_{version_slug}_continuous_episode.metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
