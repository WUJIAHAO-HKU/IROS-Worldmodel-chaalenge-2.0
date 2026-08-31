#!/usr/bin/env python3
"""Package the admitted fixed V16.8 terminal-protected world model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", required=True, type=Path)
    parser.add_argument("--base-release", required=True, type=Path)
    parser.add_argument("--student-release", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite release: {args.output}")
    search = json.loads(args.search.read_text(encoding="utf-8"))
    if search.get("format") != "strict-track2-v168-terminal-protected-blend-search-v1":
        raise SystemExit("unexpected V16.8 search report")
    tag = search.get("selected_profile")
    selected = next((row for row in search.get("profiles", []) if row.get("tag") == tag), None)
    if selected is None or not selected.get("passed") or not all(selected.get("checks", {}).values()):
        raise SystemExit("V16.8 search has no fully admitted profile")
    profile = selected["profile"]
    if profile != {
        "left_alpha": 0.0,
        "right_first_two_alpha": 0.0,
        "right_middle_shape": "cosine",
        "right_middle_t3_t6_alpha": 0.25,
        "right_terminal_t7_t8_alpha": 0.0,
    }:
        raise SystemExit("V16.8 selected profile differs from the audited fixed profile")

    base = args.base_release.resolve()
    student = args.student_release.resolve()
    required = {
        "base_release/release_manifest.json": base / "release_manifest.json",
        "student_release/model.pt": student / "model.pt",
        "student_release/action_normalization.npz": student / "action_normalization.npz",
        "student_release/track2_multisource_flow_unet_config.npz": student / "track2_multisource_flow_unet_config.npz",
        "search_result.json": args.search.resolve(),
    }
    for path in required.values():
        if not path.is_file():
            raise SystemExit(f"missing package input: {path}")
    args.output.mkdir(parents=True)
    os.symlink(base, args.output / "base_release", target_is_directory=True)
    os.symlink(student, args.output / "student_release", target_is_directory=True)
    os.symlink(args.search.resolve(), args.output / "search_result.json")
    manifest = {
        "format": "track2-v16.8-terminal-protected-release-v1",
        "classification": "fixed world-model composition; never performs policy action selection",
        "base_release": "base_release",
        "student_release": "student_release",
        "profile": {
            "left_alpha": 0.0,
            "right_horizon_alpha": [0.0, 0.0, 0.03661165, 0.125, 0.125, 0.03661165, 0.0, 0.0],
        },
        "sha256": {relative: sha256(path) for relative, path in required.items()},
        "promotion_guard": "This release permits an official one-update diagnostic only; real RoboTwin evidence is still required.",
    }
    (args.output / "v168_terminal_protected_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
