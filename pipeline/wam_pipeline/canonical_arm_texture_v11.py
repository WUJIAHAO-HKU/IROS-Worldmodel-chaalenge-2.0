"""Training-only canonical texture retrieval and geometric rendering for robot arms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


REGIONS = {
    "left": (0, 128, 0, 128),
    "right": (0, 128, 128, 256),
}


def geometry_descriptor(frame: np.ndarray, side: str) -> np.ndarray:
    y0, y1, x0, x1 = REGIONS[side]
    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    low = cv2.resize(cv2.GaussianBlur(gray, (0, 0), 4.0), (24, 24), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(low, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(low, cv2.CV_32F, 0, 1, ksize=3)
    value = np.stack((low, np.sqrt(gx * gx + gy * gy)), axis=0)
    value = value - value.mean(axis=(1, 2), keepdims=True)
    return value / (value.std(axis=(1, 2), keepdims=True) + 1e-6)


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def _beam_support(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), 2.5)
    dark = (blur < 0.43).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    support = np.zeros_like(dark)
    for index in range(1, count):
        _, _, width, height, area = stats[index]
        center_y = centroids[index][1]
        if area >= 180 and width >= 34 and width >= 1.05 * height and center_y < 85:
            support[labels == index] = 1
    return cv2.dilate(support, np.ones((9, 9), np.uint8))


def _warp(value: np.ndarray, matrix: np.ndarray, interpolation: int) -> np.ndarray:
    return cv2.warpAffine(
        value,
        matrix,
        (value.shape[1], value.shape[0]),
        flags=interpolation | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


@dataclass
class RenderParameters:
    top_k: int = 8
    texture_alpha: float = 1.0
    minimum_ecc: float = 0.55
    minimum_overlap: float = 0.20
    maximum_descriptor_distance: float = 2.5
    mask_dilation: int = 2
    mask_blur: float = 0.8
    highpass_sigma: float = 1.4
    require_disocclusion: bool = True
    texture_mode: str = "highpass"
    inward_shift: float = 0.0
    vertical_shift: float = 0.0
    render_sides: str = "both"
    prefer_observed_history: bool = False
    observed_history_alpha: float | None = None


def _observed_logo_mask(crop: np.ndarray) -> np.ndarray:
    """Find small white/blue glyphs that are actually visible in history."""
    gray = _gray(crop)
    blur = cv2.GaussianBlur(gray, (0, 0), 3.0)
    neighbourhood = cv2.GaussianBlur(gray, (0, 0), 5.0)
    value = crop.astype(np.float32) / 255.0
    # The beam can touch the vertical wrist at this pose, so the conservative
    # component-based _beam_support rejects it.  Local dark surround still
    # separates bright lettering from the white robot shell/background.
    white = (gray > 0.42) & ((gray - blur) > 0.06) & (neighbourhood < 0.60)
    blue = (
        (value[..., 2] > 0.28)
        & (value[..., 2] > np.maximum(value[..., 0], value[..., 1]) + 0.07)
        & (neighbourhood < 0.60)
    )
    candidate = (white | blue).astype(np.uint8)
    candidate[:12] = 0
    candidate[112:] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    components = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if 2 <= area <= 120 and width <= 28 and height <= 24:
            components.append((index, x, y, width, height, area))
    if not components:
        return np.zeros_like(candidate)
    # Letters form a horizontal run of several small components; isolated
    # highlights on the white wrist do not.  Group nearby components and retain
    # only the strongest text-like run.
    groups: list[list[tuple[int, int, int, int, int, int]]] = []
    remaining = set(range(len(components)))
    while remaining:
        queue = [remaining.pop()]; group = []
        while queue:
            current = queue.pop(); component = components[current]; group.append(component)
            _, x, y, width, height, _ = component
            cy = y + 0.5 * height
            for other in list(remaining):
                _, ox, oy, ow, oh, _ = components[other]
                ocy = oy + 0.5 * oh
                horizontal_gap = max(ox - (x + width), x - (ox + ow), 0)
                if abs(cy - ocy) <= 14 and horizontal_gap <= 18:
                    remaining.remove(other); queue.append(other)
        groups.append(group)
    eligible = []
    for group in groups:
        left = min(value[1] for value in group)
        right = max(value[1] + value[3] for value in group)
        if len(group) >= 3 and right - left >= 20:
            score = sum(value[5] for value in group) + 15 * len(group) + (right - left)
            eligible.append((score, group))
    if not eligible:
        return np.zeros_like(candidate)
    group = max(eligible, key=lambda value: value[0])[1]
    accepted = np.zeros_like(candidate)
    for index, *_ in group:
        accepted[labels == index] = 1
    return accepted


def _observed_beam_mask(crop: np.ndarray) -> np.ndarray:
    """Isolate the right horizontal logo beam from the connected wrist."""
    gray = _gray(crop)
    dark = (cv2.GaussianBlur(gray, (0, 0), 2.0) < 0.50).astype(np.uint8)
    dark[:30] = 0; dark[118:] = 0; dark[:, :35] = 0
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    candidates = [index for index in range(1, count)
                  if stats[index, cv2.CC_STAT_AREA] >= 100]
    if not candidates:
        return np.zeros_like(dark)
    index = max(candidates, key=lambda value: stats[value, cv2.CC_STAT_AREA])
    return (labels == index).astype(np.uint8)


class CanonicalArmTextureV11:
    def __init__(self, atlas_path: str | Path) -> None:
        with np.load(atlas_path, allow_pickle=False) as atlas:
            self.frames = atlas["frames"]
            self.descriptors = atlas["descriptors"].astype(np.float32)
            self.text_masks = atlas["text_masks"]
            self.scores = atlas["scores"]
            self.windows = atlas["windows"].astype(str)
            self.frame_indices = atlas["frame_indices"]
            self.sides = atlas["sides"].astype(str)

    def reset_observed_state(self) -> None:
        # Kept as an explicit sequence boundary for evaluator compatibility.
        # Beam alignment itself is frame-local and cannot accumulate drift.
        return None

    def _align(self, query: np.ndarray, source: np.ndarray) -> tuple[float, np.ndarray] | None:
        query_gray = cv2.GaussianBlur(_gray(query), (0, 0), 1.8)
        source_gray = cv2.GaussianBlur(_gray(source), (0, 0), 1.8)
        # The background is almost white; retaining non-white pixels makes ECC
        # follow robot geometry instead of table illumination.
        input_mask = _beam_support(query_gray) * 255
        if int((input_mask > 0).sum()) < 300:
            return None
        matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
        try:
            correlation, matrix = cv2.findTransformECC(
                query_gray,
                source_gray,
                matrix,
                cv2.MOTION_AFFINE,
                criteria,
                inputMask=input_mask,
                gaussFiltSize=5,
            )
        except cv2.error:
            return None
        linear = matrix[:, :2]
        determinant = float(np.linalg.det(linear))
        singular = np.linalg.svd(linear, compute_uv=False)
        if determinant <= 0 or singular.min() < 0.72 or singular.max() > 1.38:
            return None
        return float(correlation), matrix

    def _align_observed(self, query: np.ndarray, source: np.ndarray,
                        initial_matrix: np.ndarray | None = None) -> tuple[float, np.ndarray] | None:
        query_gray = cv2.GaussianBlur(_gray(query), (0, 0), 1.8)
        source_gray = cv2.GaussianBlur(_gray(source), (0, 0), 1.8)
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
        matrix = matrix.astype(np.float32)
        fallback_matrix = matrix.copy()
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6)
        try:
            correlation, matrix = cv2.findTransformECC(
                query_gray, source_gray, matrix, cv2.MOTION_AFFINE, criteria,
                inputMask=query_beam * 255, gaussFiltSize=5
            )
        except cv2.error:
            correlation, matrix = 0.55, fallback_matrix
        singular = np.linalg.svd(matrix[:, :2], compute_uv=False)
        if np.linalg.det(matrix[:, :2]) <= 0 or singular.min() < 0.65 or singular.max() > 1.50:
            return None
        return float(correlation), matrix

    def render_observed_side(self, image: np.ndarray, observed: np.ndarray, side: str,
                             parameters: RenderParameters) -> tuple[np.ndarray, dict]:
        """Reproject input-visible text before falling back to the training atlas."""
        y0, y1, x0, x1 = REGIONS[side]
        query = image[y0:y1, x0:x1]
        source = observed[y0:y1, x0:x1]
        source_mask = _observed_logo_mask(source)
        rejected = {
            "side": side, "accepted": False, "source": "observed_history",
            "ecc": 0.0, "dark_overlap": 0.0, "descriptor_distance": 0.0,
            "edited_fraction": 0.0, "observed_glyph_pixels": int(source_mask.sum()),
        }
        if source_mask.sum() < 8:
            rejected["rejection_reason"] = "no_observed_glyph"
            return image, rejected
        aligned = self._align_observed(query, source)
        if aligned is None:
            rejected["rejection_reason"] = "observed_alignment_failed"
            return image, rejected
        correlation, matrix = aligned
        query_dark = _observed_beam_mask(query) > 0
        source_dark = _observed_beam_mask(source) > 0
        warped_dark = _warp(source_dark.astype(np.uint8), matrix, cv2.INTER_NEAREST) > 0
        overlap = float(np.logical_and(warped_dark, query_dark).sum() /
                        max(np.logical_or(warped_dark, query_dark).sum(), 1))
        rejected.update({"ecc": correlation, "dark_overlap": overlap})
        if correlation < parameters.minimum_ecc or overlap < parameters.minimum_overlap:
            rejected["rejection_reason"] = "low_observed_alignment_confidence"
            return image, rejected
        aligned_source = _warp(source.astype(np.float32) / 255.0, matrix, cv2.INTER_LINEAR)
        aligned_mask = _warp(source_mask.astype(np.float32), matrix, cv2.INTER_LINEAR)
        aligned_mask *= query_dark.astype(np.float32)
        if parameters.mask_blur:
            aligned_mask = cv2.GaussianBlur(aligned_mask, (0, 0), parameters.mask_blur)
        maximum_alpha = (parameters.observed_history_alpha
                         if parameters.observed_history_alpha is not None
                         else parameters.texture_alpha)
        alignment_confidence = np.clip((correlation - 0.80) / 0.15, 0.0, 1.0)
        overlap_confidence = np.clip((overlap - 0.20) / 0.45, 0.0, 1.0)
        observed_alpha = parameters.texture_alpha + (
            maximum_alpha - parameters.texture_alpha
        ) * alignment_confidence * overlap_confidence
        blend = np.clip(observed_alpha * aligned_mask[..., None], 0.0, 1.0)
        query_float = query.astype(np.float32) / 255.0
        rendered = query_float * (1.0 - blend) + aligned_source * blend
        output = image.copy()
        output[y0:y1, x0:x1] = np.round(np.clip(rendered, 0.0, 1.0) * 255.0).astype(np.uint8)
        return output, {
            **rejected, "accepted": True, "source": "observed_history",
            "observed_alpha": float(observed_alpha),
            "edited_fraction": float((aligned_mask > 0.05).mean()),
        }

    def render_side(
        self,
        image: np.ndarray,
        side: str,
        parameters: RenderParameters,
    ) -> tuple[np.ndarray, dict]:
        y0, y1, x0, x1 = REGIONS[side]
        query = image[y0:y1, x0:x1]
        query_descriptor = geometry_descriptor(image, side)
        indices = np.flatnonzero(self.sides == side)
        distances = ((self.descriptors[indices] - query_descriptor[None]) ** 2).mean(axis=(1, 2, 3))
        order = indices[np.argsort(distances)[: parameters.top_k]]
        distance_by_index = {int(index): float(distance) for index, distance in zip(indices, distances)}
        query_gray = _gray(query)
        query_dark = _beam_support(query_gray)
        best = None
        for index in order:
            source = self.frames[index, y0:y1, x0:x1]
            aligned = self._align(query, source)
            if aligned is None:
                continue
            correlation, matrix = aligned
            source_gray = _gray(source)
            source_dark = _beam_support(source_gray)
            warped_dark = _warp(source_dark, matrix, cv2.INTER_NEAREST) > 0
            intersection = float(np.logical_and(warped_dark, query_dark > 0).sum())
            union = float(np.logical_or(warped_dark, query_dark > 0).sum())
            overlap = intersection / max(union, 1.0)
            distance = distance_by_index[int(index)]
            rank_score = correlation + 0.35 * overlap - 0.08 * distance
            if best is None or rank_score > best[0]:
                best = (rank_score, int(index), correlation, overlap, distance, matrix)

        rejected = {
            "side": side,
            "accepted": False,
            "atlas_index": -1,
            "ecc": 0.0,
            "dark_overlap": 0.0,
            "descriptor_distance": float(distances.min()) if len(distances) else float("inf"),
            "edited_fraction": 0.0,
        }
        if best is None:
            return image, rejected
        _, index, correlation, overlap, distance, matrix = best
        if (
            correlation < parameters.minimum_ecc
            or overlap < parameters.minimum_overlap
            or distance > parameters.maximum_descriptor_distance
        ):
            rejected.update(
                {
                    "atlas_index": index,
                    "ecc": correlation,
                    "dark_overlap": overlap,
                    "descriptor_distance": distance,
                }
            )
            return image, rejected

        source = self.frames[index, y0:y1, x0:x1]
        source_float = source.astype(np.float32) / 255.0
        aligned_source = _warp(source_float, matrix, cv2.INTER_LINEAR)
        source_mask = self.text_masks[index]
        if parameters.mask_dilation:
            diameter = 2 * parameters.mask_dilation + 1
            source_mask = cv2.dilate(source_mask, np.ones((diameter, diameter), np.uint8))
        aligned_mask = _warp(source_mask.astype(np.float32), matrix, cv2.INTER_LINEAR)
        shift_x = parameters.inward_shift if side == "left" else -parameters.inward_shift
        if shift_x or parameters.vertical_shift:
            shift = np.asarray(
                [[1.0, 0.0, shift_x], [0.0, 1.0, parameters.vertical_shift]], dtype=np.float32
            )
            aligned_source = cv2.warpAffine(
                aligned_source,
                shift,
                (aligned_source.shape[1], aligned_source.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            aligned_mask = cv2.warpAffine(
                aligned_mask,
                shift,
                (aligned_mask.shape[1], aligned_mask.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        aligned_mask *= query_dark.astype(np.float32)
        if parameters.mask_blur:
            aligned_mask = cv2.GaussianBlur(aligned_mask, (0, 0), parameters.mask_blur)
        aligned_mask = np.clip(aligned_mask, 0.0, 1.0)[..., None]
        query_float = query.astype(np.float32) / 255.0
        if parameters.texture_mode == "direct":
            blend = np.clip(parameters.texture_alpha * aligned_mask, 0.0, 1.0)
            rendered = query_float * (1.0 - blend) + aligned_source * blend
        elif parameters.texture_mode == "semantic":
            # A low-alpha RGB copy retains the atlas frame's blur and looks like
            # a watermark. Reconstruct canonical material colours inside the
            # training-derived glyph mask while retaining its warped geometry.
            blue_source = (
                (source_float[..., 2] > source_float[..., 0] + 0.10)
                & (source_float[..., 2] > source_float[..., 1] + 0.04)
                & (source_float[..., 2] > 0.28)
            ).astype(np.float32)
            aligned_blue = _warp(blue_source, matrix, cv2.INTER_LINEAR)
            if shift_x or parameters.vertical_shift:
                aligned_blue = cv2.warpAffine(
                    aligned_blue,
                    shift,
                    (aligned_blue.shape[1], aligned_blue.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
            aligned_blue = np.clip(aligned_blue, 0.0, 1.0)[..., None]
            white = np.asarray((0.92, 0.92, 0.92), dtype=np.float32)
            blue = np.asarray((0.12, 0.34, 0.95), dtype=np.float32)
            canonical_colour = white[None, None, :] * (1.0 - aligned_blue) + blue[None, None, :] * aligned_blue
            blend = np.clip(parameters.texture_alpha * aligned_mask, 0.0, 1.0)
            rendered = query_float * (1.0 - blend) + canonical_colour * blend
        elif parameters.texture_mode in ("highpass", "positive"):
            low = cv2.GaussianBlur(aligned_source, (0, 0), parameters.highpass_sigma)
            residual = aligned_source - low
            if parameters.texture_mode == "positive":
                residual = np.maximum(residual, 0.0)
            rendered = np.clip(
                query_float + parameters.texture_alpha * aligned_mask * residual,
                0.0,
                1.0,
            )
        else:
            raise ValueError(f"unknown texture mode: {parameters.texture_mode}")
        output = image.copy()
        output[y0:y1, x0:x1] = np.round(255.0 * rendered).astype(np.uint8)
        diagnostics = {
            "side": side,
            "accepted": True,
            "atlas_index": index,
            "atlas_window": self.windows[index],
            "atlas_frame": int(self.frame_indices[index]),
            "ecc": correlation,
            "dark_overlap": overlap,
            "descriptor_distance": distance,
            "edited_fraction": float((aligned_mask[..., 0] > 0.05).mean()),
        }
        return output, diagnostics

    def render(
        self,
        image: np.ndarray,
        parameters: RenderParameters,
        history: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[dict]]:
        output = image
        diagnostics = []
        for side in REGIONS:
            if parameters.render_sides != "both" and side != parameters.render_sides:
                diagnostics.append(
                    {
                        "side": side,
                        "accepted": False,
                        "rejection_reason": "side_disabled",
                        "atlas_index": -1,
                        "ecc": 0.0,
                        "dark_overlap": 0.0,
                        "descriptor_distance": float("inf"),
                        "edited_fraction": 0.0,
                    }
                )
                continue
            if parameters.require_disocclusion and history is not None:
                y0, y1, x0, x1 = REGIONS[side]
                history_beam_area = max(
                    int(_beam_support(_gray(frame[y0:y1, x0:x1])).sum()) for frame in history
                )
                if history_beam_area > 0:
                    diagnostics.append(
                        {
                            "side": side,
                            "accepted": False,
                            "rejection_reason": "beam_visible_in_history",
                            "history_beam_area": history_beam_area,
                            "atlas_index": -1,
                            "ecc": 0.0,
                            "dark_overlap": 0.0,
                            "descriptor_distance": float("inf"),
                            "edited_fraction": 0.0,
                        }
                    )
                    continue
            if parameters.prefer_observed_history and history is not None:
                observed_output, observed_diagnostics = self.render_observed_side(
                    output, history[-1], side, parameters
                )
                if observed_diagnostics["accepted"]:
                    diagnostics.append(observed_diagnostics)
                    output = observed_output
                    continue
            output, side_diagnostics = self.render_side(output, side, parameters)
            side_diagnostics["history_beam_area"] = 0
            diagnostics.append(side_diagnostics)
        return output, diagnostics
