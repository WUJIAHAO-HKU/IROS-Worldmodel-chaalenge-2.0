#!/usr/bin/env python3
"""Package a preregistered V15 action-gated on-policy parent candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


FORMAT = "track2-v15-action-gated-onpolicy-adaptation-v1"
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
    parser.add_argument("--adapted-autoregressive", required=True)
    parser.add_argument("--action-gate", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--screen-report")
    parser.add_argument("--candidate-blend", type=float, default=1.0)
    parser.add_argument("--blend-selection-report")
    parser.add_argument("--model-version", default="track2-v15.5-action-gated-onpolicy")
    args = parser.parse_args()

    base = Path(args.base_release).resolve()
    adapted = Path(args.adapted_autoregressive).resolve()
    action_gate = Path(args.action_gate).resolve()
    preregistration = Path(args.preregistration).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not 0.0 < args.candidate_blend <= 1.0:
        raise ValueError("--candidate-blend must be in (0, 1]")
    prereg = json.loads(preregistration.read_text())
    if prereg.get("format") not in {
        "strict-track2-action-domain-gate-preregistration-v1",
        "strict-track2-v157-hybrid-action-gate-preregistration-v1",
    }:
        raise RuntimeError("unexpected action-gate preregistration")
    screen = Path(args.screen_report).resolve() if args.screen_report else None
    if screen is not None and json.loads(screen.read_text()).get("passed") is not True:
        raise RuntimeError("formal parent screen did not pass")
    blend_selection = (
        Path(args.blend_selection_report).resolve() if args.blend_selection_report else None
    )
    if blend_selection is not None:
        selection = json.loads(blend_selection.read_text())
        if selection.get("passed") is not True or selection.get("selected") is None:
            raise RuntimeError("blend selection report did not pass")
        if abs(float(selection["selected"]["strength"]) - args.candidate_blend) > 1e-12:
            raise RuntimeError("candidate blend does not match the selected strength")
        blend_preregistration = Path(selection["preregistration"]).resolve()
        blend_prereg = json.loads(blend_preregistration.read_text())
        if Path(blend_prereg["candidate_checkpoint"]).resolve() != adapted:
            raise RuntimeError("adapted checkpoint does not match blend preregistration")

    shutil.copytree(base, output, copy_function=os.link)
    root = output / "onpolicy_adaptation"
    for filename in AR_FILES:
        link(adapted / filename, root / "adapted_autoregressive" / filename)
    link(action_gate / "action_source_gate.pt", root / "source_gate.pt")
    link(action_gate / "training_manifest.json", root / "source_gate_training_manifest.json")
    link(preregistration, root / "preregistration.json")
    relative_files = [
        *(f"adapted_autoregressive/{filename}" for filename in AR_FILES),
        "source_gate.pt",
        "source_gate_training_manifest.json",
        "preregistration.json",
    ]
    if screen is not None:
        link(screen, root / "parent_screen_report.json")
        relative_files.append("parent_screen_report.json")
    if blend_selection is not None:
        link(blend_selection, root / "blend_selection_report.json")
        link(blend_preregistration, root / "blend_preregistration.json")
        relative_files.extend(("blend_selection_report.json", "blend_preregistration.json"))
    manifest = {
        "format": FORMAT,
        "model_version": args.model_version,
        "candidate_stage": "formal" if screen is not None else "evaluation",
        "base_release": str(base),
        "base_release_manifest_sha256": sha256(output / "release_manifest.json"),
        "adapted_autoregressive": str(adapted),
        "candidate_blend": args.candidate_blend,
        "blend_selection_report": str(blend_selection) if blend_selection is not None else None,
        "routing": "request history and future actions only; policy actions are forwarded unchanged",
        "action_selection": False,
        "mpc": False,
        "screen_report": str(screen) if screen is not None else None,
        "sha256": {relative: sha256(root / relative) for relative in relative_files},
    }
    (root / "adaptation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
