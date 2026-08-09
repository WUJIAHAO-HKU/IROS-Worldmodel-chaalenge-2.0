"""Validated v14.1 observed-glyph reprojection used by the v15 runtime."""

from __future__ import annotations

import cv2
import numpy as np

from .canonical_arm_texture_v11 import REGIONS, _observed_beam_mask, _observed_logo_mask


def _align(query: np.ndarray, source: np.ndarray, gray) -> tuple[float, np.ndarray] | None:
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
    matrix = np.concatenate((linear, (source_center - linear @ query_center)[:, None]), axis=1).astype(np.float32)
    fallback = matrix.copy()
    try:
        correlation, matrix = cv2.findTransformECC(
            cv2.GaussianBlur(gray(query), (0, 0), 1.8),
            cv2.GaussianBlur(gray(source), (0, 0), 1.8),
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
    image: np.ndarray, observed: np.ndarray, gray, warp, highpass_strength: float = 0.55
):
    y0, y1, x0, x1 = REGIONS["right"]
    query, source = image[y0:y1, x0:x1], observed[y0:y1, x0:x1]
    source_mask = _observed_logo_mask(source)
    detail = {"accepted": False, "source_glyph_pixels": int(source_mask.sum())}
    if source_mask.sum() < 8:
        return image, detail
    aligned = _align(query, source, gray)
    if aligned is None:
        return image, detail
    correlation, matrix = aligned
    query_beam = _observed_beam_mask(query) > 0
    warped_beam = warp((_observed_beam_mask(source) > 0).astype(np.uint8), matrix, cv2.INTER_NEAREST) > 0
    overlap = float(np.logical_and(query_beam, warped_beam).sum() / max(np.logical_or(query_beam, warped_beam).sum(), 1))
    if correlation < 0.55 or overlap < 0.20:
        return image, detail
    aligned_source = warp(source.astype(np.float32) / 255.0, matrix, cv2.INTER_LINEAR)
    aligned_mask = warp(source_mask.astype(np.float32), matrix, cv2.INTER_LINEAR) * query_beam
    aligned_mask = np.clip(cv2.GaussianBlur(aligned_mask, (0, 0), 0.45), 0, 1)
    existing = _observed_logo_mask(query)
    clear = (cv2.dilate(existing, np.ones((3, 3), np.uint8)) > 0) & query_beam
    cleaned = cv2.inpaint(query, clear.astype(np.uint8) * 255, 2.0, cv2.INPAINT_TELEA) if clear.any() else query.copy()
    cleaned = cleaned.astype(np.float32) / 255.0
    blend = np.clip(0.90 * aligned_mask[..., None], 0, 1)
    rendered = cleaned * (1 - blend) + aligned_source * blend
    rendered += highpass_strength * aligned_mask[..., None] * np.maximum(
        aligned_source - cv2.GaussianBlur(aligned_source, (0, 0), 1.0), 0
    )
    output = image.copy()
    output[y0:y1, x0:x1] = np.round(np.clip(rendered, 0, 1) * 255).astype(np.uint8)
    return output, {"accepted": True, "ecc": correlation, "beam_overlap": overlap}
