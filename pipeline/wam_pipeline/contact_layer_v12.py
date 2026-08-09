"""Contact-aware rigid bottle/gripper layer reconstruction.

The renderer uses only the last observation and a parent prediction.  Once a
green bottle is grasped, its motion supplies a rigid transform for both the
bottle and the contacting gripper.  Layers are composited bottle first and
gripper second so RGB averaging cannot dissolve the contact boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


CONTACT_REGION = (72, 232, 30, 210)


def _largest_component(mask: np.ndarray, minimum_area: int = 80) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if int(stats[index, cv2.CC_STAT_AREA]) < minimum_area:
        return np.zeros_like(mask, dtype=np.uint8)
    return (labels == index).astype(np.uint8)


def bottle_probability(image: np.ndarray) -> np.ndarray:
    value = image.astype(np.float32) / 255.0
    red, green, blue = value[..., 0], value[..., 1], value[..., 2]
    chroma = green - np.maximum(red, blue)
    probability = np.clip((chroma - 0.025) / 0.20, 0.0, 1.0)
    probability *= np.clip((green - 0.12) / 0.35, 0.0, 1.0)
    return cv2.GaussianBlur(probability.astype(np.float32), (0, 0), 0.8)


def bottle_mask(image: np.ndarray) -> np.ndarray:
    seed = (bottle_probability(image) > 0.16).astype(np.uint8)
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    seed = _largest_component(seed)
    if not seed.any():
        return seed
    contours, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    support = np.zeros_like(seed)
    cv2.fillConvexPoly(support, hull, 1)
    return cv2.dilate(support, np.ones((3, 3), np.uint8))


def gripper_mask(image: np.ndarray, bottle: np.ndarray) -> np.ndarray:
    value = image.astype(np.float32) / 255.0
    maximum = value.max(axis=2)
    minimum = value.min(axis=2)
    # Use the near-black contact pad only. A broader threshold connects it to
    # the grey parallel jaw and eventually the entire arm, which defeats the
    # purpose of instance-level compositing.
    neutral_dark = (value.mean(axis=2) < 0.30) & ((maximum - minimum) < 0.18)
    near = cv2.dilate(bottle, np.ones((17, 17), np.uint8)) > 0
    touching = cv2.dilate(bottle, np.ones((7, 7), np.uint8)) > 0
    candidate = (neutral_dark & near).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    output = np.zeros_like(candidate)
    for index in range(1, count):
        component = labels == index
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area >= 15 and np.logical_and(component, touching).any():
            output[component] = 1
    return cv2.dilate(output, np.ones((3, 3), np.uint8))


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
class ContactParameters:
    minimum_bottle_area: int = 220
    minimum_gripper_area: int = 45
    minimum_query_gripper_area: int = 45
    maximum_gripper_area: int = 100000
    minimum_ecc: float = 0.50
    minimum_bottle_overlap: float = 0.42
    bottle_alpha: float = 0.72
    gripper_alpha: float = 0.88
    edge_blur: float = 0.55
    completion_only: bool = True
    gripper_alignment: str = "top_right"
    maximum_gripper_area_ratio: float = 0.78
    reconstruction_mode: str = "query_sharpen"
    structure_dilation: int = 0
    trimap_radius: int = 1
    green_purity_ratio: float = 0.62


class ContactLayerV12:
    def _align_bottle(
        self, source_probability: np.ndarray, query_probability: np.ndarray, query_mask: np.ndarray
    ) -> tuple[float, np.ndarray] | None:
        source_points = np.argwhere(source_probability > 0.12)
        query_points = np.argwhere(query_probability > 0.12)
        if len(source_points) < 80 or len(query_points) < 80:
            return None
        source_center = source_points.mean(axis=0)[::-1]
        query_center = query_points.mean(axis=0)[::-1]
        matrix = np.eye(2, 3, dtype=np.float32)
        # WARP_INVERSE_MAP expects output(query) -> input(source).
        matrix[:, 2] = source_center - query_center
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6)
        try:
            correlation, matrix = cv2.findTransformECC(
                query_probability,
                source_probability,
                matrix,
                cv2.MOTION_AFFINE,
                criteria,
                inputMask=(cv2.dilate(query_mask, np.ones((15, 15), np.uint8)) * 255),
                gaussFiltSize=5,
            )
        except cv2.error:
            return None
        singular = np.linalg.svd(matrix[:, :2], compute_uv=False)
        determinant = float(np.linalg.det(matrix[:, :2]))
        if determinant <= 0.0 or singular.min() < 0.65 or singular.max() > 1.45:
            return None
        return float(correlation), matrix

    def render(
        self, prediction: np.ndarray, history: np.ndarray, parameters: ContactParameters
    ) -> tuple[np.ndarray, dict]:
        y0, y1, x0, x1 = CONTACT_REGION
        source = history[-1, y0:y1, x0:x1]
        query = prediction[y0:y1, x0:x1]
        source_bottle = bottle_mask(source)
        query_bottle = bottle_mask(query)
        source_gripper = gripper_mask(source, source_bottle)
        query_gripper = gripper_mask(query, query_bottle)
        rejected = {
            "accepted": False,
            "ecc": 0.0,
            "bottle_overlap": 0.0,
            "source_bottle_area": int(source_bottle.sum()),
            "query_bottle_area": int(query_bottle.sum()),
            "source_gripper_area": int(source_gripper.sum()),
            "query_gripper_area": int(query_gripper.sum()),
            "edited_fraction": 0.0,
        }
        if (
            source_bottle.sum() < parameters.minimum_bottle_area
            or query_bottle.sum() < parameters.minimum_bottle_area
            or source_gripper.sum() < parameters.minimum_gripper_area
            or source_gripper.sum() > parameters.maximum_gripper_area
            or query_gripper.sum() < parameters.minimum_query_gripper_area
        ):
            rejected["rejection_reason"] = "missing_contact_layers"
            return prediction, rejected
        gripper_area_ratio = float(query_gripper.sum() / max(source_gripper.sum(), 1))
        rejected["query_source_gripper_area_ratio"] = gripper_area_ratio
        if gripper_area_ratio >= parameters.maximum_gripper_area_ratio:
            rejected["rejection_reason"] = "gripper_structure_already_present"
            return prediction, rejected
        if parameters.reconstruction_mode in ("query_sharpen", "hard_trimap"):
            # The parent's contact-pad support is geometrically more reliable
            # than an old RGB patch; only its contrast and filled area decay.
            # Restore that current-frame support toward the observed black
            # material, avoiding any cross-frame positioning error.
            structure = query_gripper
            if parameters.structure_dilation:
                diameter = 2 * parameters.structure_dilation + 1
                structure = cv2.dilate(structure, np.ones((diameter, diameter), np.uint8))
            material = np.median(source[source_gripper > 0].astype(np.float32), axis=0)
            if parameters.reconstruction_mode == "query_sharpen":
                alpha = structure.astype(np.float32) * parameters.gripper_alpha
                if parameters.edge_blur:
                    alpha = cv2.GaussianBlur(alpha, (0, 0), parameters.edge_blur)
                alpha = np.clip(alpha, 0.0, 1.0)[..., None]
                rendered = query.astype(np.float32) * (1.0 - alpha) + material[None, None, :] * alpha
            else:
                # Convert the soft RGB contact boundary into mutually exclusive
                # semantic labels. The detected gripper core is opaque black;
                # its one-pixel ring is assigned either to black foreground or
                # to purified green bottle material, never an interpolation.
                rendered = query.astype(np.float32).copy()
                core = structure > 0
                diameter = 2 * parameters.trimap_radius + 1
                expanded = cv2.dilate(structure, np.ones((diameter, diameter), np.uint8)) > 0
                ring = np.logical_and(expanded, ~core)
                value = query.astype(np.float32)
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
                alpha = np.logical_or(core, np.logical_or(black_side, green_side)).astype(np.float32)[..., None]
            output = prediction.copy()
            output[y0:y1, x0:x1] = np.round(np.clip(rendered, 0.0, 255.0)).astype(np.uint8)
            return output, {
                **rejected,
                "accepted": True,
                "edited_fraction": float((alpha[..., 0] > 0.05).mean()),
                "material_rgb": material.tolist(),
            }
        if parameters.reconstruction_mode != "rigid_reprojection":
            raise ValueError(f"unknown reconstruction mode: {parameters.reconstruction_mode}")
        aligned = self._align_bottle(bottle_probability(source), bottle_probability(query), query_bottle)
        if aligned is None:
            rejected["rejection_reason"] = "alignment_failed"
            return prediction, rejected
        correlation, matrix = aligned
        warped_bottle = _warp(source_bottle.astype(np.float32), matrix, cv2.INTER_LINEAR)
        warped_source_bottle = _warp(source.astype(np.float32) / 255.0, matrix, cv2.INTER_LINEAR)
        # The bottle is almost stationary after contact, while the wrist keeps
        # rotating.  Using the bottle transform for the gripper therefore leaves
        # a crisp but static jaw.  The blurred parent still preserves a reliable
        # centroid of the dark contact assembly, so use that observable motion
        # for an independent rigid gripper translation.
        gripper_matrix = np.eye(2, 3, dtype=np.float32)
        source_points = np.argwhere(source_gripper > 0)
        query_points = np.argwhere(query_gripper > 0)
        if parameters.gripper_alignment == "centroid":
            source_anchor = source_points.mean(axis=0)[::-1]
            query_anchor = query_points.mean(axis=0)[::-1]
        elif parameters.gripper_alignment == "top_right":
            # The parent preserves the top and outer (right) edges of the pad;
            # dissolution primarily removes its lower/inner portion. Anchoring
            # stable edges prevents the restored block drifting into the bottle.
            source_anchor = np.asarray((source_points[:, 1].max(), source_points[:, 0].min()))
            query_anchor = np.asarray((query_points[:, 1].max(), query_points[:, 0].min()))
        else:
            raise ValueError(f"unknown gripper alignment: {parameters.gripper_alignment}")
        gripper_matrix[:, 2] = source_anchor - query_anchor
        warped_gripper = _warp(source_gripper.astype(np.float32), gripper_matrix, cv2.INTER_LINEAR)
        warped_source_gripper = _warp(
            source.astype(np.float32) / 255.0, gripper_matrix, cv2.INTER_LINEAR
        )
        intersection = np.logical_and(warped_bottle > 0.25, query_bottle > 0).sum()
        union = np.logical_or(warped_bottle > 0.25, query_bottle > 0).sum()
        overlap = float(intersection / max(union, 1))
        rejected.update({"ecc": correlation, "bottle_overlap": overlap})
        if correlation < parameters.minimum_ecc or overlap < parameters.minimum_bottle_overlap:
            rejected["rejection_reason"] = "low_alignment_confidence"
            return prediction, rejected

        # Repaint the rigid bottle layer first to remove arm-colour bleeding,
        # then place the contacting gripper above it with a near-binary alpha.
        bottle_alpha = warped_bottle * parameters.bottle_alpha
        gripper_alpha = warped_gripper * parameters.gripper_alpha
        if parameters.completion_only:
            query_float = query.astype(np.float32) / 255.0
            query_green = bottle_probability(query)
            query_dark = np.clip((0.38 - query_float.mean(axis=2)) / 0.28, 0.0, 1.0)
            # Fill only texture/structure missing from the parent. Existing
            # confident green and black pixels remain bit-identical.
            bottle_alpha *= 1.0 - query_green
            gripper_alpha *= 1.0 - query_dark
        if parameters.edge_blur:
            bottle_alpha = cv2.GaussianBlur(bottle_alpha, (0, 0), parameters.edge_blur)
            gripper_alpha = cv2.GaussianBlur(gripper_alpha, (0, 0), parameters.edge_blur)
        bottle_alpha = np.clip(bottle_alpha, 0.0, 1.0)[..., None]
        gripper_alpha = np.clip(gripper_alpha, 0.0, 1.0)[..., None]
        rendered = query.astype(np.float32) / 255.0
        rendered = rendered * (1.0 - bottle_alpha) + warped_source_bottle * bottle_alpha
        rendered = rendered * (1.0 - gripper_alpha) + warped_source_gripper * gripper_alpha
        output = prediction.copy()
        output[y0:y1, x0:x1] = np.round(np.clip(rendered, 0.0, 1.0) * 255.0).astype(np.uint8)
        return output, {
            **rejected,
            "accepted": True,
            "edited_fraction": float(np.logical_or(bottle_alpha[..., 0] > 0.05, gripper_alpha[..., 0] > 0.05).mean()),
            "matrix": matrix.tolist(),
            "gripper_matrix": gripper_matrix.tolist(),
        }
