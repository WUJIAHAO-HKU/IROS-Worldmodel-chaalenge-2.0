#!/usr/bin/env python3
"""Episode-disjoint retrieval of real eight-frame contact motion templates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.contact_occlusion_head_v13 import CONTACT_REGION
from wam_pipeline.contact_structure_v135 import structure_semantic_mask
from wam_pipeline.canonical_arm_texture_v11 import (
    REGIONS as ARM_TEXTURE_REGIONS,
    _gray,
    _observed_beam_mask,
    _observed_logo_mask,
    _warp,
)
from train_contact_occlusion_head_v13 import episode
from train_contact_occlusion_head_v131 import active_arm


def descriptor(frame: np.ndarray) -> np.ndarray:
    y0, y1, x0, x1 = CONTACT_REGION
    value = cv2.resize(frame[y0:y1, x0:x1], (24, 20), interpolation=cv2.INTER_AREA)
    return value.astype(np.float32).reshape(-1) / 255


def motion_descriptor(action: np.ndarray) -> np.ndarray:
    return np.concatenate(((action - action[:1]).reshape(-1), np.diff(action, axis=0).reshape(-1)))


def align(candidate: np.ndarray, query: np.ndarray) -> tuple[float, np.ndarray]:
    y0, y1, x0, x1 = CONTACT_REGION
    candidate = cv2.cvtColor(candidate[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    query = cv2.cvtColor(query[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        score, warp = cv2.findTransformECC(query, candidate, warp, cv2.MOTION_AFFINE,
                                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5),
                                           None, 3)
    except cv2.error:
        score = -1.0
    return float(score), warp


def warp_future(frames: np.ndarray, warp: np.ndarray) -> np.ndarray:
    y0, y1, x0, x1 = CONTACT_REGION; values = []
    for frame in frames:
        values.append(cv2.warpAffine(frame[y0:y1, x0:x1], warp, (x1 - x0, y1 - y0),
                                     flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_REFLECT))
    return np.stack(values)


def photometric_calibration(candidate_source: np.ndarray, query_source: np.ndarray,
                            warp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match bright static background without learning from future targets."""
    y0, y1, x0, x1 = CONTACT_REGION
    aligned = warp_future(candidate_source[None], warp)[0].astype(np.float32)
    query = query_source[y0:y1, x0:x1].astype(np.float32)
    support = (aligned.mean(axis=2) > 170) & (query.mean(axis=2) > 170)
    gains, biases = [], []
    for channel in range(3):
        left, right = aligned[..., channel][support], query[..., channel][support]
        if len(left) < 100:
            gain, bias = 1.0, 0.0
        elif left.std() < 1.0 or right.std() < 1.0:
            gain = 1.0
            bias = float(np.clip(right.mean() - left.mean(), -18, 18))
        else:
            gain = float(np.clip(right.std() / max(left.std(), 1.0), 0.85, 1.15))
            bias = float(np.clip(right.mean() - gain * left.mean(), -18, 18))
        gains.append(gain); biases.append(bias)
    return np.asarray(gains, np.float32), np.asarray(biases, np.float32)


def feather(height: int, width: int, radius: int = 16) -> np.ndarray:
    y, x = np.ogrid[:height, :width]
    distance = np.minimum.reduce((np.broadcast_to(y, (height, width)),
                                  np.broadcast_to(x, (height, width)),
                                  np.broadcast_to(height - 1 - y, (height, width)),
                                  np.broadcast_to(width - 1 - x, (height, width))))
    return np.clip(distance.astype(np.float32) / radius, 0, 1)


def right_texture_mask(frame: np.ndarray) -> np.ndarray:
    """Conservative metric-only mask for right-arm glyph texture."""
    output = np.zeros(frame.shape[:2], dtype=bool)
    y0, y1, x0, x1 = ARM_TEXTURE_REGIONS["right"]
    crop = frame[y0:y1, x0:x1]
    mask = _observed_logo_mask(crop)
    if mask.any():
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        output[y0:y1, x0:x1] = mask > 0
    return output


def align_observed_beam(query: np.ndarray, source: np.ndarray) -> tuple[float, np.ndarray] | None:
    """Frame-local affine alignment constrained to the horizontal logo beam.

    The returned matrix follows OpenCV's WARP_INVERSE_MAP convention and maps
    query coordinates to source coordinates.  This is deliberately separate
    from the contact-crop transform used by v14.0: contact geometry and glyph
    texture are allowed to move independently.
    """
    query_beam = _observed_beam_mask(query)
    source_beam = _observed_beam_mask(source)
    if query_beam.sum() < 180 or source_beam.sum() < 180:
        return None
    query_xy = np.argwhere(query_beam)[:, ::-1].astype(np.float64)
    source_xy = np.argwhere(source_beam)[:, ::-1].astype(np.float64)
    query_center, source_center = query_xy.mean(0), source_xy.mean(0)
    query_e, query_v = np.linalg.eigh(np.cov((query_xy - query_center).T) + np.eye(2))
    source_e, source_v = np.linalg.eigh(np.cov((source_xy - source_center).T) + np.eye(2))
    query_order, source_order = np.argsort(query_e)[::-1], np.argsort(source_e)[::-1]
    query_e, query_v = query_e[query_order], query_v[:, query_order]
    source_e, source_v = source_e[source_order], source_v[:, source_order]
    for axis in range(2):
        if np.dot(query_v[:, axis], source_v[:, axis]) < 0:
            source_v[:, axis] *= -1
    linear = source_v @ np.diag(np.sqrt(np.clip(source_e / query_e, 0.50, 2.0))) @ query_v.T
    matrix = np.concatenate((linear, (source_center - linear @ query_center)[:, None]), axis=1)
    matrix = matrix.astype(np.float32); fallback = matrix.copy()
    try:
        correlation, matrix = cv2.findTransformECC(
            cv2.GaussianBlur(_gray(query), (0, 0), 1.8),
            cv2.GaussianBlur(_gray(source), (0, 0), 1.8),
            matrix,
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6),
            inputMask=query_beam * 255,
            gaussFiltSize=5,
        )
    except cv2.error:
        correlation, matrix = 0.55, fallback
    singular = np.linalg.svd(matrix[:, :2], compute_uv=False)
    if np.linalg.det(matrix[:, :2]) <= 0 or singular.min() < 0.65 or singular.max() > 1.50:
        return None
    return float(correlation), matrix


def reproject_observed_glyph(
    image: np.ndarray,
    observed: np.ndarray,
    minimum_ecc: float = 0.55,
    texture_alpha: float = 0.90,
    highpass_strength: float = 0.55,
) -> tuple[np.ndarray, dict]:
    """Remove stale glyphs and transfer only locally aligned glyph texture."""
    y0, y1, x0, x1 = ARM_TEXTURE_REGIONS["right"]
    query = image[y0:y1, x0:x1]
    source = observed[y0:y1, x0:x1]
    source_mask = _observed_logo_mask(source)
    rejected = {"accepted": False, "ecc": 0.0, "beam_overlap": 0.0,
                "source_glyph_pixels": int(source_mask.sum()), "edited_fraction": 0.0}
    if source_mask.sum() < 8:
        rejected["reason"] = "no_observed_glyph"
        return image, rejected
    aligned = align_observed_beam(query, source)
    if aligned is None:
        rejected["reason"] = "beam_alignment_failed"
        return image, rejected
    correlation, matrix = aligned
    query_beam = _observed_beam_mask(query) > 0
    source_beam = _observed_beam_mask(source) > 0
    warped_beam = _warp(source_beam.astype(np.uint8), matrix, cv2.INTER_NEAREST) > 0
    overlap = float(np.logical_and(query_beam, warped_beam).sum() /
                    max(np.logical_or(query_beam, warped_beam).sum(), 1))
    rejected.update({"ecc": correlation, "beam_overlap": overlap})
    if correlation < minimum_ecc or overlap < 0.20:
        rejected["reason"] = "low_alignment_confidence"
        return image, rejected

    aligned_source = _warp(source.astype(np.float32) / 255.0, matrix, cv2.INTER_LINEAR)
    aligned_mask = _warp(source_mask.astype(np.float32), matrix, cv2.INTER_LINEAR)
    aligned_mask *= query_beam.astype(np.float32)
    aligned_mask = cv2.GaussianBlur(aligned_mask, (0, 0), 0.45)
    aligned_mask = np.clip(aligned_mask, 0.0, 1.0)

    # Remove only already-detected glyph highlights inside the dark beam. This
    # prevents a faint old word from surviving next to the newly aligned one.
    existing = _observed_logo_mask(query)
    clear_mask = cv2.dilate(existing, np.ones((3, 3), np.uint8)) > 0
    clear_mask &= query_beam
    cleaned = (cv2.inpaint(query, clear_mask.astype(np.uint8) * 255, 2.0, cv2.INPAINT_TELEA)
               if clear_mask.any() else query.copy())
    cleaned = cleaned.astype(np.float32) / 255.0

    blend = np.clip(texture_alpha * aligned_mask[..., None], 0.0, 1.0)
    rendered = cleaned * (1.0 - blend) + aligned_source * blend
    low = cv2.GaussianBlur(aligned_source, (0, 0), 1.0)
    positive_highpass = np.maximum(aligned_source - low, 0.0)
    rendered += highpass_strength * aligned_mask[..., None] * positive_highpass
    output = image.copy()
    output[y0:y1, x0:x1] = np.round(np.clip(rendered, 0.0, 1.0) * 255).astype(np.uint8)
    return output, {**rejected, "accepted": True,
                    "cleared_glyph_pixels": int(clear_mask.sum()),
                    "edited_fraction": float((aligned_mask > 0.05).mean())}


def temporal_ramp(length: int, start: int, transition_frames: int) -> np.ndarray:
    """Return a deterministic late-branch gate without changing the trigger."""
    gate = np.zeros(length, np.float32)
    if transition_frames <= 1:
        gate[start:] = 1.0
    else:
        count = min(transition_frames, length - start)
        gate[start:start + count] = np.linspace(1.0 / count, 1.0, count, dtype=np.float32)
        gate[start + count:] = 1.0
    return gate


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    y0, y1, x0, x1 = CONTACT_REGION
    error = np.abs(prediction.astype(np.float32) - target.astype(np.float32))
    crop = error[:, :, y0:y1, x0:x1]
    labels = np.stack([structure_semantic_mask(value[:, y0:y1, x0:x1]) for value in target])
    mask = np.repeat(np.isin(labels, (2, 3))[..., None], 3, axis=4)
    structure_error = float(crop[mask].mean()) if mask.any() else float("nan")
    texture = np.stack([[right_texture_mask(frame) for frame in sequence] for sequence in target])
    texture_rgb = np.repeat(texture[..., None], 3, axis=4)
    texture_error = float(error[texture_rgb].mean()) if texture_rgb.any() else float("nan")
    frame_texture = []
    for time in range(target.shape[1]):
        time_mask = texture_rgb[:, time]
        frame_texture.append(float(error[:, time][time_mask].mean()) if time_mask.any() else float("nan"))
    return {"rgb_mae": float(error.mean()), "contact_rgb_mae": float(crop.mean()),
            "structure_rgb_mae": structure_error, "target_texture_rgb_mae": texture_error,
            "frame_rgb_mae": error.mean(axis=(0, 2, 3, 4)).tolist(),
            "frame_contact_rgb_mae": crop.mean(axis=(0, 2, 3, 4)).tolist(),
            "frame_target_texture_rgb_mae": frame_texture}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True); parser.add_argument("--library-cache", required=True)
    parser.add_argument("--parent-cache", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--output-cache", required=True); parser.add_argument("--preselect", type=int, default=24)
    parser.add_argument("--minimum-ecc", type=float, default=0.95)
    parser.add_argument("--minimum-decay", type=float, default=0.88)
    parser.add_argument("--minimum-recovery", type=float, default=0.65)
    parser.add_argument("--minimum-alpha", type=float, default=0.35)
    parser.add_argument("--transition-frames", type=int, default=1)
    parser.add_argument("--reproject-observed-text", action="store_true")
    parser.add_argument("--text-minimum-ecc", type=float, default=0.55)
    parser.add_argument("--text-alpha", type=float, default=0.90)
    parser.add_argument("--text-highpass-strength", type=float, default=0.55)
    parser.add_argument("--text-prefix-frames", type=int, default=2)
    parser.add_argument("--expand-library-episodes", action="store_true")
    parser.add_argument("--only-window")
    args = parser.parse_args(); windows = Path(args.windows)
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent, target, context = cache["prediction"], cache["target"], cache["context"]
        names = cache["windows"].astype(str)
    if args.only_window:
        selected = np.flatnonzero(names == args.only_window)
        if len(selected) != 1: raise KeyError(args.only_window)
        parent, target, context, names = (value[selected] for value in (parent, target, context, names))
    with np.load(args.library_cache, allow_pickle=False) as cache:
        library_names = cache["windows"].astype(str)
    if args.expand_library_episodes:
        training_episodes = {episode(name) for name in library_names}
        library_names = np.asarray(sorted(
            path.name for path in windows.glob("*.npz") if episode(path.name) in training_episodes
        ))
    # Build the inference library once. Future frames are loaded only for the
    # selected neighbour, keeping RAM bounded and the split auditable.
    visual, motion, arms, valid_names = [], [], [], []
    for name in library_names:
        path = windows / name
        if not path.exists(): continue
        with np.load(path, allow_pickle=False) as window:
            source = window["context_frames"][-1].copy()
            arm, action = active_arm(window["history_actions"], window["future_actions"])
        visual.append(descriptor(source)); motion.append(motion_descriptor(action))
        arms.append(arm); valid_names.append(name)
    visual = np.stack(visual); motion = np.stack(motion)
    arms = np.asarray(arms); valid_names = np.asarray(valid_names)

    y0, y1, x0, x1 = CONTACT_REGION; output = parent.copy(); query_arms = []; diagnostics = []
    for index, name in enumerate(names):
        with np.load(windows / name, allow_pickle=False) as query:
            arm, action = active_arm(query["history_actions"], query["future_actions"])
        query_arms.append(arm); query_episode = episode(name)
        allowed = (arms == arm) & np.asarray([episode(value) != query_episode for value in valid_names])
        candidates = np.flatnonzero(allowed)
        query_visual, query_motion = descriptor(context[index, -1]), motion_descriptor(action)
        visual_distance = ((visual[candidates] - query_visual) ** 2).mean(axis=1)
        motion_distance = ((motion[candidates] - query_motion) ** 2).mean(axis=1)
        visual_scale = max(float(np.median(visual_distance)), 1e-9)
        motion_scale = max(float(np.median(motion_distance)), 1e-12)
        combined = visual_distance / visual_scale + 2.5 * motion_distance / motion_scale
        shortlist = candidates[np.argsort(combined)[:args.preselect]]
        aligned = []
        for candidate in shortlist:
            with np.load(windows / valid_names[candidate], allow_pickle=False) as candidate_window:
                candidate_source = candidate_window["context_frames"][-1]
            ecc, warp = align(candidate_source, context[index, -1])
            if ecc >= args.minimum_ecc:
                local = int(np.flatnonzero(candidates == candidate)[0])
                aligned.append((motion_distance[local], visual_distance[local], -ecc, candidate, warp))
        source_label = structure_semantic_mask(context[index, -1:, y0:y1, x0:x1])[0]
        parent_label = structure_semantic_mask(parent[index, :, y0:y1, x0:x1])
        source_area = max(int(np.isin(source_label, (2, 3)).sum()), 1)
        ratios = np.isin(parent_label, (2, 3)).sum(axis=(1, 2)) / source_area
        decay_frames = np.flatnonzero(ratios < args.minimum_decay)
        start_frame = max(int(decay_frames[0]) - 1, 0) if len(decay_frames) else None
        # The validated retrieval regime is a late, transient right-arm
        # collapse: early frames remain structurally intact, then the parent
        # loses support. Other regimes retain the parent pixel-for-pixel.
        intact_prefix = bool(start_frame is not None and start_frame >= 3 and
                             np.all(ratios[:start_frame] >= 0.95))
        accepted = bool(aligned and arm == 1 and intact_prefix and
                        ratios[-1] >= args.minimum_recovery)
        detail = {"window": name, "arm": arm, "query_episode": query_episode,
                  "accepted": accepted, "start_frame": start_frame,
                  "parent_structure_area_ratios": ratios.tolist()}
        if accepted:
            _, _, negative_ecc, candidate, warp = min(aligned)
            candidate_name = str(valid_names[candidate]); candidate_episode = episode(candidate_name)
            if candidate_episode == query_episode: raise AssertionError("retrieval episode leakage")
            with np.load(windows / candidate_name, allow_pickle=False) as selected:
                future = selected["target_frames"]
                candidate_source = selected["context_frames"][-1]
            warped = warp_future(future, warp).astype(np.float32); ecc = -negative_ecc
            gain, bias = photometric_calibration(candidate_source, context[index, -1], warp)
            warped = np.clip(warped * gain + bias, 0, 255)
            alpha = max(args.minimum_alpha, min(1.0, (ecc - args.minimum_ecc) /
                                                max(0.99 - args.minimum_ecc, 1e-6)))
            parent_crop = parent[index, :, y0:y1, x0:x1].astype(np.float32)
            blended = parent_crop.copy()
            gate = temporal_ramp(len(blended), start_frame, args.transition_frames)
            alpha_map = (gate[:, None, None, None] * alpha
                         * feather(y1 - y0, x1 - x0)[None, ..., None])
            blended = parent_crop * (1 - alpha_map) + warped * alpha_map
            output[index, :, y0:y1, x0:x1] = np.clip(np.round(blended), 0, 255).astype(np.uint8)
            text_diagnostics = []
            if args.reproject_observed_text:
                # The real-motion template already retains sharp glyphs after
                # the branch switch. Correct only the unreliable parent prefix
                # so that two competing text transports never create doubles.
                text_end = min(start_frame, args.text_prefix_frames, len(output[index]))
                for time in range(text_end):
                    hp_scale = 1.0 if text_end == 1 else max(1.0 - time / (text_end - 1), 0.0)
                    output[index, time], text_detail = reproject_observed_glyph(
                        output[index, time], context[index, -1], args.text_minimum_ecc,
                        args.text_alpha, args.text_highpass_strength * hp_scale,
                    )
                    text_detail["frame"] = time + 1; text_diagnostics.append(text_detail)
            detail.update({"candidate": candidate_name, "candidate_episode": candidate_episode,
                           "ecc": ecc, "alpha": alpha, "photometric_gain": gain.tolist(),
                           "photometric_bias": bias.tolist(), "feather_radius": 16,
                           "temporal_gate": gate.tolist(),
                           "text_reprojection": text_diagnostics})
        diagnostics.append(detail)
    query_arms = np.asarray(query_arms)
    version = "v14.1" if args.reproject_observed_text else "v14.0"
    report = {"format": f"track2-contact-motion-retrieval-{version}", "sample_count": len(parent),
              "library_size": len(valid_names), "accepted_count": int(sum(x["accepted"] for x in diagnostics)),
              "parameters": {"preselect": args.preselect, "minimum_ecc": args.minimum_ecc,
                             "minimum_decay": args.minimum_decay,
                             "minimum_recovery": args.minimum_recovery,
                             "minimum_alpha": args.minimum_alpha,
                             "transition_frames": args.transition_frames,
                             "reproject_observed_text": args.reproject_observed_text,
                             "text_minimum_ecc": args.text_minimum_ecc,
                             "text_alpha": args.text_alpha,
                             "text_highpass_strength": args.text_highpass_strength,
                             "text_prefix_frames": args.text_prefix_frames},
              "overall": {"parent": metrics(parent, target), "retrieval": metrics(output, target)},
              "arms": {}, "windows": []}
    for arm in (0, 1):
        indices = np.flatnonzero(query_arms == arm)
        if not len(indices):
            report["arms"][f"arm{arm}"] = {"sample_count": 0}
            continue
        report["arms"][f"arm{arm}"] = {"sample_count": len(indices),
                                          "parent": metrics(parent[indices], target[indices]),
                                          "retrieval": metrics(output[indices], target[indices])}
    for index, detail in enumerate(diagnostics):
        detail["parent"] = metrics(parent[index:index + 1], target[index:index + 1])
        detail["retrieval"] = metrics(output[index:index + 1], target[index:index + 1])
        report["windows"].append(detail)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    cache_path = Path(args.output_cache); cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, prediction=output, target=target, context=context,
                        windows=names, arm_id=query_arms)
    print(json.dumps({key: value for key, value in report.items() if key != "windows"}, indent=2))


if __name__ == "__main__": main()
