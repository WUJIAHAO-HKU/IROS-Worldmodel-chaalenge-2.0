from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from diagnose_track2_window_alignment import audit_windows  # noqa: E402
from train_autoregressive_unet import apply_arm_sampling  # noqa: E402


def test_consensus_sampling_never_boosts_mismatched_metadata():
    weights = torch.ones(4, dtype=torch.float64)
    metadata_right = np.asarray([False, True, True, False])
    left_motion = np.asarray([2.0, 1.0, 3.0, 1.0])
    right_motion = np.asarray([1.0, 3.0, 1.0, 1.05])
    adjusted, report = apply_arm_sampling(
        weights,
        metadata_right,
        4.0,
        left_motion,
        right_motion,
        source="consensus",
        dominance_margin=1.10,
    )
    assert adjusted.tolist() == [1.0, 4.0, 1.0, 1.0]
    assert report["boosted_right_window_count"] == 1
    assert report["action_ambiguous_count"] == 1
    assert report["metadata_right_action_right_fraction"] == 0.5


def _window(path: Path, *, label_right: bool, right_motion: bool) -> None:
    actions = np.zeros((12, 14), dtype=np.float32)
    channel = 7 if right_motion else 0
    actions[:, channel] = np.arange(12, dtype=np.float32)
    np.savez(
        path,
        history_actions=actions[:4],
        future_actions=actions[4:],
        arm_right=np.asarray(label_right),
    )


def test_full_audit_reports_label_action_mismatch(tmp_path: Path):
    _window(tmp_path / "episode10000_00000.npz", label_right=False, right_motion=False)
    _window(tmp_path / "episode10001_00000.npz", label_right=True, right_motion=False)
    report = audit_windows(tmp_path, dominance_margin=1.10)
    episodes = report["episode_summary"]["all"]
    assert episodes["count"] == 2
    assert episodes["label_action_agreement_all"] == 0.5
    assert episodes["dominant_left"] == 2
    assert report["outcome_labels_used_for_arm_inference"] is False
