#!/usr/bin/env python3
"""Package a same-domain direct arm-routed parent into the frozen V15 shell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


FORMAT = "track2-v15-arm-routed-initial-window-v3"
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
    parser.add_argument("--screen-report", required=True)
    parser.add_argument("--reward-report", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-version", default="track2-v15.3-initial-window-joint")
    args = parser.parse_args()
    paths = {name: Path(value).resolve() for name, value in {
        "base": args.base_release,
        "left": args.left_autoregressive,
        "right": args.right_autoregressive,
        "screen": args.screen_report,
        "reward": args.reward_report,
        "preregistration": args.preregistration,
        "output": args.output,
    }.items()}
    if paths["output"].exists():
        raise FileExistsError(f"refusing to overwrite packaged release: {paths['output']}")
    if json.loads(paths["screen"].read_text()).get("passed") is not True:
        raise RuntimeError("parent candidate did not pass the preregistered screen")
    shutil.copytree(paths["base"], paths["output"], copy_function=os.link)
    adaptation = paths["output"] / "onpolicy_adaptation"
    if adaptation.exists():
        shutil.rmtree(adaptation)
    for suffix, checkpoint in (("", paths["left"]), ("_right", paths["right"])):
        for filename in AR_FILES:
            link(checkpoint / filename, adaptation / f"adapted_autoregressive{suffix}" / filename)
    link(paths["screen"], adaptation / "parent_screen_report.json")
    link(paths["reward"], adaptation / "reward_alignment_report.json")
    link(paths["preregistration"], adaptation / "preregistration.json")
    relative_files = [
        *(f"adapted_autoregressive/{name}" for name in AR_FILES),
        *(f"adapted_autoregressive_right/{name}" for name in AR_FILES),
        "parent_screen_report.json",
        "reward_alignment_report.json",
        "preregistration.json",
    ]
    manifest = {
        "format": FORMAT,
        "model_version": args.model_version,
        "base_release": str(paths["base"]),
        "base_release_manifest_sha256": sha256(paths["output"] / "release_manifest.json"),
        "left_autoregressive": str(paths["left"]),
        "right_autoregressive": str(paths["right"]),
        "routing": "always use adapted expert; select left/right from raw request action delta",
        "router_uses_only_request_inputs": True,
        "source_gate": None,
        "screen_report": str(paths["screen"]),
        "reward_report": str(paths["reward"]),
        "preregistration": str(paths["preregistration"]),
        "sha256": {relative: sha256(adaptation / relative) for relative in relative_files},
    }
    (adaptation / "adaptation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
