#!/usr/bin/env python3
"""Package left/right reward-aligned experts behind the audited V15 source gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


FORMAT = "track2-v15-instruction-gated-arm-temporal-adaptation-v3"
AR_FILES = (
    "model.pt",
    "action_normalization.npz",
    "track2_autoregressive_unet_config.npz",
    "training_manifest.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-release", required=True)
    parser.add_argument("--left-autoregressive", required=True)
    parser.add_argument("--right-autoregressive", required=True)
    parser.add_argument("--source-gate", required=True)
    parser.add_argument("--screen-report", required=True)
    parser.add_argument("--reward-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-version", default="track2-v15.2-arm-routed-reward-aligned")
    parser.add_argument("--left-candidate-blend-schedule")
    parser.add_argument("--right-candidate-blend-schedule")
    parser.add_argument("--instruction-arm-router", action="store_true")
    args = parser.parse_args()
    base = Path(args.base_release).resolve()
    left = Path(args.left_autoregressive).resolve()
    right = Path(args.right_autoregressive).resolve()
    source_gate = Path(args.source_gate).resolve()
    screen = Path(args.screen_report).resolve()
    reward = Path(args.reward_report).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite packaged release: {output}")
    if json.loads(screen.read_text()).get("passed") is not True:
        raise RuntimeError("parent candidate did not pass the preregistered screen")
    shutil.copytree(base, output, copy_function=os.link)
    root = output / "onpolicy_adaptation"
    old_adaptation = root
    if old_adaptation.exists():
        shutil.rmtree(old_adaptation)
    for arm, checkpoint in (("", left), ("_right", right)):
        for filename in AR_FILES:
            link(checkpoint / filename, root / f"adapted_autoregressive{arm}" / filename)
    link(source_gate / "source_gate.pt", root / "source_gate.pt")
    link(source_gate / "training_manifest.json", root / "source_gate_training_manifest.json")
    link(screen, root / "parent_screen_report.json")
    link(reward, root / "reward_alignment_report.json")
    relative_files = [
        *(f"adapted_autoregressive/{name}" for name in AR_FILES),
        *(f"adapted_autoregressive_right/{name}" for name in AR_FILES),
        "source_gate.pt",
        "source_gate_training_manifest.json",
        "parent_screen_report.json",
        "reward_alignment_report.json",
    ]
    manifest = {
        "format": FORMAT,
        "model_version": args.model_version,
        "base_release": str(base),
        "base_release_manifest_sha256": sha256(output / "release_manifest.json"),
        "left_autoregressive": str(left),
        "right_autoregressive": str(right),
        "arm_router": (
            "instruction_left_right_token_with_raw_action_delta_fallback"
            if args.instruction_arm_router
            else "raw_action_delta_argmax"
        ),
        "instruction_arm_router": bool(args.instruction_arm_router),
        "router_uses_only_request_inputs": True,
        "source_gate": str(source_gate),
        "screen_report": str(screen),
        "reward_report": str(reward),
        "hard_route_threshold": .5,
        "candidate_blend": 1.0,
        "sha256": {relative: sha256(root / relative) for relative in relative_files},
    }
    if args.left_candidate_blend_schedule or args.right_candidate_blend_schedule:
        if not args.left_candidate_blend_schedule or not args.right_candidate_blend_schedule:
            raise ValueError("both left and right blend schedules are required")
        left_schedule = json.loads(args.left_candidate_blend_schedule)
        right_schedule = json.loads(args.right_candidate_blend_schedule)
        for name, schedule in (("left", left_schedule), ("right", right_schedule)):
            if len(schedule) != 8 or any(not 0.0 <= float(value) <= 1.0 for value in schedule):
                raise ValueError(f"invalid {name} candidate blend schedule")
        manifest["left_candidate_blend_schedule"] = left_schedule
        manifest["right_candidate_blend_schedule"] = right_schedule
    (root / "adaptation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
