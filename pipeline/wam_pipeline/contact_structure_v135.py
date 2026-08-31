"""Explicit black-pad and grey-jaw labels for full contact-structure modelling."""

from __future__ import annotations

import cv2
import numpy as np

from .contact_layer_v12 import bottle_mask


def _components_touching(candidate: np.ndarray, anchor: np.ndarray,
                         minimum_area: int = 10) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8
    )
    output = np.zeros_like(candidate, dtype=bool)
    for index in range(1, count):
        component = labels == index
        if stats[index, cv2.CC_STAT_AREA] >= minimum_area and np.logical_and(component, anchor).any():
            output |= component
    return output


def structure_masks(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bottle, near-black pad/arm and grey jaw/support masks.

    Unlike the old three-class label, the black layer is not dilated for
    scoring and the neutral mid-tone jaw is retained as its own instance.
    Both structure layers must connect to the grasp neighbourhood, preventing
    table shadows from becoming robot labels.
    """
    value = frame.astype(np.float32)
    bottle = bottle_mask(frame) > 0
    near = cv2.dilate(bottle.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    touching = cv2.dilate(bottle.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    mean = value.mean(axis=2)
    chroma = value.max(axis=2) - value.min(axis=2)
    neutral = chroma < 42.0

    black_candidate = neutral & (mean < 82.0) & near
    # Preserve every connected black part of the assembly that reaches the
    # bottle/contact pad, including the vertical wrist above the old v13 crop.
    black_anchor = cv2.dilate(touching.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    black = _components_touching(black_candidate, black_anchor, minimum_area=12)

    grey_candidate = neutral & (mean >= 82.0) & (mean < 205.0) & near
    grey_anchor = cv2.dilate(black.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    grey_anchor |= touching
    grey = _components_touching(grey_candidate, grey_anchor, minimum_area=10)
    grey &= ~black
    return bottle, black, grey


def structure_semantic_mask(frames: np.ndarray) -> np.ndarray:
    labels = []
    for frame in frames:
        bottle, black, grey = structure_masks(frame)
        label = bottle.astype(np.uint8)
        label[black] = 2
        label[grey] = 3
        labels.append(label)
    return np.stack(labels)
