#!/usr/bin/env python3
"""Package the V16.9 routed, terminal-protected world-model release."""

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
    parser.add_argument("--v168-release", required=True, type=Path)
    parser.add_argument("--arm-router", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite release: {args.output}")
    search = json.loads(args.search.read_text(encoding="utf-8"))
    if search.get("format") != "strict-track2-v169-routed-terminal-protected-blend-search-v1":
        raise SystemExit("unexpected V16.9 search report")
    tag = search.get("selected_profile")
    selected = next((row for row in search["profiles"] if row["tag"] == tag), None)
    if selected is None or not selected["passed"] or not all(selected["checks"].values()):
        raise SystemExit("V16.9 search has no fully admitted profile")
    if selected["profile"] != {
        "left_alpha": 0.0,
        "right_first_two_alpha": 0.0,
        "right_middle_shape": "cosine",
        "right_middle_t3_t6_alpha": 0.25,
        "right_terminal_t7_t8_alpha": 0.0,
    }:
        raise SystemExit("V16.9 profile is not identical to the packaged fixed blend")
    router_manifest = json.loads(
        (args.arm_router / "arm_router_manifest.json").read_text(encoding="utf-8")
    )
    if router_manifest["validation"]["learned_window_accuracy"] < 0.90:
        raise SystemExit("arm router failed its 90% validation gate")
    if router_manifest["validation"]["episode_majority_accuracy"] < 1.0:
        raise SystemExit("arm router failed its episode-level validation gate")
    required = {
        "v168_release/v168_terminal_protected_manifest.json": args.v168_release / "v168_terminal_protected_manifest.json",
        "arm_router/arm_router.npz": args.arm_router / "arm_router.npz",
        "arm_router/arm_router_manifest.json": args.arm_router / "arm_router_manifest.json",
        "search_result.json": args.search,
    }
    for path in required.values():
        if not path.is_file():
            raise SystemExit(f"missing package input: {path}")
    args.output.mkdir(parents=True)
    os.symlink(args.v168_release.resolve(), args.output / "v168_release", target_is_directory=True)
    os.symlink(args.arm_router.resolve(), args.output / "arm_router", target_is_directory=True)
    os.symlink(args.search.resolve(), args.output / "search_result.json")
    manifest = {
        "format": "track2-v16.9-arm-routed-terminal-protected-release-v1",
        "classification": "fixed world-model expert routing; never scores, selects, or changes policy actions",
        "v168_release": "v168_release",
        "arm_router": "arm_router",
        "selected_profile": selected["profile"],
        "route_audit": search["route_audit"],
        "sha256": {relative: sha256(path.resolve()) for relative, path in required.items()},
        "promotion_guard": "Official one-update diagnostic only until paired real RoboTwin acceptance.",
    }
    (args.output / "v169_arm_routed_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
