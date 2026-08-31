#!/usr/bin/env python3
"""Package every learned component needed by the online v15 runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, target: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--output", default="artifacts/releases/track2_v15_best")
    args = parser.parse_args()
    artifacts, output = Path(args.artifacts), Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing release: {output}")

    v8_source = artifacts / "checkpoints/autoregressive-direct-flow-local-motion-texture-structure-refiner-v8"
    shutil.copytree(v8_source, output / "v8")
    flow_base = artifacts / "checkpoints/multisource-flow-unet-track2-formal-v1/best"
    for name in (
        "model.pt",
        "action_normalization.npz",
        "track2_multisource_flow_unet_config.npz",
        "training_manifest.json",
    ):
        copy_file(flow_base / name, output / "v10/flow_base" / name)
    copy_file(
        artifacts / "checkpoints/protected-layered-flow-v91-diverse1200-flow2000/latest.pt",
        output / "v10/flow_head.pt",
    )
    copy_file(
        artifacts / "checkpoints/candidate-risk-router-v101-motion-cont1500/latest.pt",
        output / "v10/risk_router.pt",
    )
    copy_file(artifacts / "dual_tiny_experts_v150/glyph/best.pt", output / "v15/glyph.pt")
    copy_file(artifacts / "dual_tiny_experts_v150/gripper/best.pt", output / "v15/gripper.pt")
    for name in ("deployment_manifest.json", "validation64_temporal_safe_report.json"):
        copy_file(artifacts / "dual_tiny_experts_v150" / name, output / "evidence" / name)
    copy_file(
        artifacts / "contact_motion_retrieval_v141/deployment_manifest.json",
        output / "evidence/v14.1_deployment_manifest.json",
    )
    copy_file(
        artifacts / "candidate_risk_router_v101_deployment_manifest.json",
        output / "evidence/v10.1_deployment_manifest.json",
    )

    hashes = {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "format": "track2-v15-online-release-v1",
        "model_version": "track2-v15.0-best",
        "parent_chain": ["v8", "v10.1", "v14.1", "v15.0"],
        "accepted_routing": {
            "glyph_strength": 0.0,
            "arm0_gripper_strength": 0.0,
            "arm1_gripper_strength": 1.0,
            "arm1_gripper_start_frame_zero_based": 2,
        },
        "retrieval": {
            "windows_directory": "adjust_bottle_windows_full",
            "split_manifest": "splits/adjust_bottle_50episodes_full.json",
            "data_policy": "train episodes only; public 50-episode data",
        },
        "validation64": {
            "sample_count": 64,
            "rgb_mae_0_255": 6.120042324066162,
            "contact_rgb_mae_0_255": 7.913723468780518,
            "structure_rgb_mae_0_255": 17.84128761291504,
        },
        "sha256": hashes,
    }
    (output / "release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output.resolve()), "files": len(hashes), "bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file())}, indent=2))


if __name__ == "__main__":
    main()
