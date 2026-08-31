"""Sequence-latched contact tracking and topology-preserving reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .contact_layer_v12 import CONTACT_REGION, bottle_mask, gripper_mask


@dataclass
class ContactTrackParameters:
    minimum_source_area: int = 700
    maximum_source_area: int = 1500
    minimum_query_area: int = 45
    latch_area_ratio: float = 0.65
    target_area_scale: float = 1.0
    maximum_expansion_radius: int = 6
    contact_support_radius: int = 14
    green_purity_ratio: float = 0.62


def _expand_connected_mask(
    seed: np.ndarray, support: np.ndarray, target_area: int, maximum_radius: int
) -> np.ndarray:
    output = cv2.morphologyEx(seed.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if int(output.sum()) >= target_area:
        return output
    previous = output
    for radius in range(1, maximum_radius + 1):
        diameter = 2 * radius + 1
        candidate = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter)))
        candidate = np.logical_and(candidate > 0, support > 0).astype(np.uint8)
        previous = candidate
        if int(candidate.sum()) >= target_area:
            break
    return cv2.morphologyEx(previous, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


class ContactTrackV13:
    def render_sequence(
        self,
        prediction: np.ndarray,
        history: np.ndarray,
        parameters: ContactTrackParameters,
    ) -> tuple[np.ndarray, dict]:
        y0, y1, x0, x1 = CONTACT_REGION
        source = history[-1, y0:y1, x0:x1]
        source_bottle = bottle_mask(source)
        source_gripper = gripper_mask(source, source_bottle)
        source_area = int(source_gripper.sum())
        rejected = {
            "accepted": False,
            "source_gripper_area": source_area,
            "accepted_frame_fraction": 0.0,
        }
        if not parameters.minimum_source_area <= source_area <= parameters.maximum_source_area:
            rejected["rejection_reason"] = "invalid_source_contact_instance"
            return prediction, rejected

        query_bottles = []
        query_grippers = []
        ratios = []
        for frame in prediction:
            crop = frame[y0:y1, x0:x1]
            bottle = bottle_mask(crop)
            gripper = gripper_mask(crop, bottle)
            query_bottles.append(bottle)
            query_grippers.append(gripper)
            ratios.append(float(gripper.sum() / max(source_area, 1)))
        valid = [mask.sum() >= parameters.minimum_query_area for mask in query_grippers]
        if not any(valid) or min(ratio for ratio, ok in zip(ratios, valid) if ok) >= parameters.latch_area_ratio:
            rejected.update(
                {
                    "rejection_reason": "no_sequence_dissolution",
                    "query_source_area_ratios": ratios,
                }
            )
            return prediction, rejected

        output = prediction.copy()
        target_area = int(round(source_area * parameters.target_area_scale))
        material = np.median(source[source_gripper > 0].astype(np.float32), axis=0)
        frame_details = []
        previous_mask = None
        for frame_index, (bottle, gripper, valid_frame) in enumerate(
            zip(query_bottles, query_grippers, valid)
        ):
            crop = output[frame_index, y0:y1, x0:x1]
            if not valid_frame:
                # A fully missing frame inherits the last connected support;
                # this path is rare but keeps the contact identity latched.
                if previous_mask is None:
                    frame_details.append({"accepted": False, "reason": "no_track_seed"})
                    continue
                track_mask = previous_mask
            else:
                support = cv2.dilate(
                    bottle,
                    np.ones((2 * parameters.contact_support_radius + 1,) * 2, np.uint8),
                )
                track_mask = _expand_connected_mask(
                    gripper, support, target_area, parameters.maximum_expansion_radius
                )
            previous_mask = track_mask
            rendered = crop.astype(np.float32).copy()
            core = track_mask > 0
            ring = np.logical_and(cv2.dilate(track_mask, np.ones((3, 3), np.uint8)) > 0, ~core)
            value = crop.astype(np.float32)
            red, green, blue = value[..., 0], value[..., 1], value[..., 2]
            green_side = ring & (green > np.maximum(red, blue) + 7.0)
            black_side = ring & ~green_side & (value.mean(axis=2) < 118.0)
            rendered[core | black_side] = material
            purified = rendered[green_side]
            if len(purified):
                purified[:, 0] = np.minimum(
                    purified[:, 0], parameters.green_purity_ratio * purified[:, 1]
                )
                purified[:, 2] = np.minimum(
                    purified[:, 2], parameters.green_purity_ratio * purified[:, 1]
                )
                rendered[green_side] = purified
            output[frame_index, y0:y1, x0:x1] = np.round(
                np.clip(rendered, 0.0, 255.0)
            ).astype(np.uint8)
            frame_details.append(
                {
                    "accepted": True,
                    "input_area": int(gripper.sum()),
                    "output_area": int(track_mask.sum()),
                    "target_area": target_area,
                    "input_source_area_ratio": ratios[frame_index],
                }
            )
        return output, {
            **rejected,
            "accepted": True,
            "accepted_frame_fraction": float(np.mean([value.get("accepted", False) for value in frame_details])),
            "query_source_area_ratios": ratios,
            "target_area": target_area,
            "material_rgb": material.tolist(),
            "frames": frame_details,
        }

