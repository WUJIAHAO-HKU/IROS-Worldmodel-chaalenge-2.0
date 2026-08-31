#!/usr/bin/env python3
"""Package the two small checkpoints used by the observable-motion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


FORMAT = "track2-autoregressive-direct-flow-motion-gated-v1"
COMPONENTS = {
    "autoregressive": (
        "model.pt",
        "training_manifest.json",
        "action_normalization.npz",
        "track2_autoregressive_unet_config.npz",
    ),
    "direct_flow": (
        "model.pt",
        "training_manifest.json",
        "action_normalization.npz",
        "track2_direct_flow_unet_config.npz",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_component(source: Path, destination: Path, files: tuple[str, ...]) -> dict[str, str]:
    if not source.is_dir():
        raise ValueError(f"component directory is missing: {source}")
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for filename in files:
        original = source / filename
        if not original.is_file():
            raise ValueError(f"component artifact is missing: {original}")
        copied = destination / filename
        shutil.copy2(original, copied)
        hashes[filename] = sha256(original)
        if sha256(copied) != hashes[filename]:
            raise RuntimeError(f"copy integrity check failed: {original}")
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoregressive-checkpoint", required=True)
    parser.add_argument("--direct-flow-checkpoint", required=True)
    parser.add_argument("--selection-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    autoregressive = Path(args.autoregressive_checkpoint).resolve()
    direct_flow = Path(args.direct_flow_checkpoint).resolve()
    selection_report = Path(args.selection_report).resolve()
    if not selection_report.is_file():
        raise SystemExit(f"selection report is missing: {selection_report}")
    selection = json.loads(selection_report.read_text())
    if selection.get("format") != "track2-crossvalidated-motion-gated-blend-v1":
        raise SystemExit("selection report has the wrong format")
    if not selection.get("promote_to_backend"):
        raise SystemExit("selection report did not promote the motion gate")
    chosen = selection.get("selected")
    if not isinstance(chosen, dict) or chosen.get("context_motion_statistic") != "last":
        raise SystemExit("selection report did not select the supported last-transition gate")
    threshold = float(chosen["context_motion_threshold"])
    autoregressive_weight = float(chosen["low_motion_first_weight"])
    direct_flow_weight = 1.0 - autoregressive_weight
    if threshold < 0 or not 0 <= autoregressive_weight <= 1:
        raise SystemExit("selection report contains an invalid threshold or weight")

    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing ensemble directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        autoregressive_hashes = copy_component(
            autoregressive, stage / "autoregressive", COMPONENTS["autoregressive"]
        )
        direct_flow_hashes = copy_component(
            direct_flow, stage / "direct_flow", COMPONENTS["direct_flow"]
        )
        config = {
            "format": FORMAT,
            "context_motion_statistic": "last",
            "context_motion_threshold": threshold,
            "high_motion_policy": "autoregressive-only",
            "low_motion_autoregressive_weight": autoregressive_weight,
            "low_motion_direct_flow_weight": direct_flow_weight,
            "selection_report": {
                "source": str(selection_report),
                "sha256": sha256(selection_report),
                "holdout_relative_improvement": selection["holdout_relative_improvement"],
                "holdout_high_motion_relative_improvement": selection[
                    "holdout_high_motion_relative_improvement"
                ],
            },
            "autoregressive": {
                "directory": "autoregressive",
                "source_checkpoint": str(autoregressive),
                "sha256": autoregressive_hashes,
            },
            "direct_flow": {
                "directory": "direct_flow",
                "source_checkpoint": str(direct_flow),
                "sha256": direct_flow_hashes,
            },
        }
        (stage / "ensemble_config.json").write_text(json.dumps(config, indent=2) + "\n")
        os.replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(json.dumps({"status": "packaged", "checkpoint_dir": str(output), **config}, indent=2))


if __name__ == "__main__":
    main()
