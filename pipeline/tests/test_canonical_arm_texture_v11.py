import cv2
import numpy as np

from wam_pipeline.canonical_arm_texture_v11 import (
    CanonicalArmTextureV11,
    RenderParameters,
    _observed_beam_mask,
    _observed_logo_mask,
    geometry_descriptor,
)


def synthetic_arm(with_texture: bool) -> tuple[np.ndarray, np.ndarray]:
    frame = np.full((256, 256, 3), 238, dtype=np.uint8)
    frame[38:84, 160:256] = (24, 24, 24)
    # Fixed low-frequency geometry makes ECC alignment deterministic.
    frame[38:84, 160:164] = (65, 65, 65)
    frame[80:84, 160:256] = (55, 55, 55)
    mask = np.zeros((128, 128), dtype=np.uint8)
    if with_texture:
        for start in (46, 56, 66, 76, 86):
            frame[52:64, 128 + start : 128 + start + 3] = (225, 225, 225)
            mask[52:64, start : start + 3] = 1
        frame[53:63, 224:228] = (45, 70, 225)
        mask[53:63, 96:100] = 1
    return frame, mask


def build_atlas(path) -> None:
    source, mask = synthetic_arm(True)
    np.savez_compressed(
        path,
        frames=source[None],
        descriptors=geometry_descriptor(source, "right")[None].astype(np.float16),
        text_masks=mask[None],
        scores=np.asarray([100.0], dtype=np.float32),
        windows=np.asarray(["train_episode_00000.npz"]),
        frame_indices=np.asarray([0], dtype=np.int16),
        sides=np.asarray(["right"]),
    )


def test_positive_texture_render_and_history_gate(tmp_path):
    atlas_path = tmp_path / "atlas.npz"
    build_atlas(atlas_path)
    renderer = CanonicalArmTextureV11(atlas_path)
    renderer._align = lambda query, source: (1.0, np.eye(2, 3, dtype=np.float32))
    query, _ = synthetic_arm(False)
    history = np.full((5, 256, 256, 3), 238, dtype=np.uint8)
    parameters = RenderParameters(
        top_k=1,
        texture_alpha=0.75,
        minimum_ecc=0.0,
        minimum_overlap=0.0,
        maximum_descriptor_distance=10.0,
        mask_dilation=0,
        mask_blur=0.0,
        texture_mode="positive",
        render_sides="right",
    )
    output, diagnostics = renderer.render(query, parameters, history)
    assert diagnostics[0]["rejection_reason"] == "side_disabled"
    assert diagnostics[1]["accepted"]
    assert np.abs(output.astype(np.int16) - query.astype(np.int16)).sum() > 0
    assert output[58, 175].mean() >= query[58, 175].mean()

    visible_history = history.copy()
    visible_history[-1] = query
    gated, diagnostics = renderer.render(query, parameters, visible_history)
    assert np.array_equal(gated, query)
    assert diagnostics[1]["rejection_reason"] == "beam_visible_in_history"


def test_beam_detector_rejects_background():
    from wam_pipeline.canonical_arm_texture_v11 import _beam_support

    blank = np.ones((128, 128), dtype=np.float32)
    assert int(_beam_support(blank).sum()) == 0
    beam = blank.copy()
    beam[35:75, 30:128] = 0.1
    assert int(_beam_support(cv2.GaussianBlur(beam, (0, 0), 0.5)).sum()) > 0


def test_semantic_mode_reconstructs_readable_white_glyph(tmp_path):
    atlas_path = tmp_path / "atlas.npz"
    build_atlas(atlas_path)
    renderer = CanonicalArmTextureV11(atlas_path)
    renderer._align = lambda query, source: (1.0, np.eye(2, 3, dtype=np.float32))
    query, _ = synthetic_arm(False)
    output, diagnostics = renderer.render_side(
        query,
        "right",
        RenderParameters(
            top_k=1,
            texture_alpha=0.75,
            minimum_ecc=0.0,
            minimum_overlap=0.0,
            maximum_descriptor_distance=10.0,
            mask_dilation=0,
            mask_blur=0.0,
            texture_mode="semantic",
        ),
    )
    assert diagnostics["accepted"]
    assert output[58, 175].mean() > query[58, 175].mean() + 80


def test_observed_history_reprojects_visible_glyph_without_atlas_guess(tmp_path):
    atlas_path = tmp_path / "atlas.npz"
    build_atlas(atlas_path)
    observed, _ = synthetic_arm(True)
    query, _ = synthetic_arm(False)
    assert _observed_logo_mask(observed[:128, 128:256]).sum() > 20
    assert _observed_beam_mask(query[:128, 128:256]).sum() > 100
    renderer = CanonicalArmTextureV11(atlas_path)
    output, detail = renderer.render_observed_side(
        query,
        observed,
        "right",
        RenderParameters(
            texture_alpha=0.5,
            minimum_ecc=0.0,
            minimum_overlap=0.0,
            mask_blur=0.0,
            prefer_observed_history=True,
        ),
    )
    assert detail["accepted"]
    assert detail["source"] == "observed_history"
    assert output[58, 175].mean() > query[58, 175].mean() + 30
