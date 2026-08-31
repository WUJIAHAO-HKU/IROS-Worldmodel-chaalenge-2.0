#!/usr/bin/env python3
"""Atomically package the single preregistered v464 all-200 endpoint parent."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

FORMAT = "track2-v464-v169-endpoint-residual-parent-release-v1"
CHECKPOINT_FORMAT = "strict-track2-v464-endpoint-residual-checkpoint-v1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_fsync(source: Path, destination: Path) -> None:
    shutil.copy2(source, destination)
    with destination.open("rb") as handle:
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("preregistration", "training-dir", "runtime", "trainer", "v169-release", "v169-library", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads(args.preregistration.read_text())
    report_path = args.training_dir / "training_report.json"
    s0_path = args.training_dir / "s0_report.json"
    checkpoint_path = args.training_dir / "all200_action_step50.pt"
    report = json.loads(report_path.read_text())
    s0 = json.loads(s0_path.read_text())
    if prereg.get("format") != "strict-track2-v464-endpoint-residual-5fold-preregistration-v2":
        raise RuntimeError("bad v464 preregistration")
    if s0.get("format") != "strict-track2-v464-step50-three-head-5fold-s0-receipt-v1" or s0.get("passed") is not True:
        raise RuntimeError("v464 S0 did not pass")
    if report.get("format") != "strict-track2-v464-step50-three-head-5fold-killgate-v2" or report.get("passed") is not True or report.get("all200_training_performed") is not True:
        raise RuntimeError("v464 final training report did not pass")
    if report.get("s0_report_sha256") != sha(s0_path) or report.get("all200_checkpoint_sha256") != sha(checkpoint_path):
        raise RuntimeError("v464 report artifact hash drift")
    if sha(args.runtime) != prereg["source"]["runtime_sha256"] or sha(args.trainer) != prereg["source"]["trainer_sha256"]:
        raise RuntimeError("v464 source closure drift")
    if str(args.v169_release.resolve()) != prereg["v169"]["release"] or str(args.v169_library.resolve()) != prereg["v169"]["library"]:
        raise RuntimeError("v464 v169 canonical path drift")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    closure_digest = hashlib.sha256(json.dumps({"v169": prereg["v169"], "source": prereg["source"], "v461_files": prereg["v461"]["files"]}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    expected_schema = {"dim": 223, "normalized_actions": 168, "right6_endpoint_delta": 6, "right6_anchor_relative_path": 48, "postclose": 1}
    if state.get("format") != CHECKPOINT_FORMAT or state.get("training_scope") != "all200" or state.get("step") != 50 or state.get("precision") != "bf16" or state.get("feature_schema") != expected_schema:
        raise RuntimeError("bad v464 all200 checkpoint contract")
    if state.get("closure_digest") != closure_digest or state.get("runtime_sha256") != sha(args.runtime) or state.get("trainer_sha256") != sha(args.trainer) or state.get("preregistration_sha256") != sha(args.preregistration) or state.get("schedule_sha256") != prereg["all200_schedule_sha256"]:
        raise RuntimeError("v464 all200 checkpoint closure drift")
    output = args.output.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise RuntimeError("refusing to overwrite v464 release/partial")
    partial.mkdir(parents=True)
    copy_fsync(checkpoint_path, partial / "all200_action_step50.pt")
    copy_fsync(args.runtime, partial / "v464_v169_endpoint_residual_runtime.py")
    copy_fsync(args.preregistration, partial / "preregistration.json")
    copy_fsync(report_path, partial / "training_report.json")
    copy_fsync(s0_path, partial / "s0_report.json")
    library_manifest = Path(prereg["v169"]["library_manifest"]).resolve()
    try:
        library_manifest_relative = str(library_manifest.relative_to(args.v169_library.resolve()))
    except ValueError as exc:
        raise RuntimeError("v169 library manifest escapes canonical library") from exc
    manifest = {
        "format": FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": "all200_action_step50.pt",
        "runtime_source": "v464_v169_endpoint_residual_runtime.py",
        "v169_release": str(args.v169_release.resolve()),
        "v169_library": str(args.v169_library.resolve()),
        "canonical_v169_release": str(args.v169_release.resolve()),
        "canonical_v169_library": str(args.v169_library.resolve()),
        "v169_library_manifest": library_manifest_relative,
        "closure_digest": closure_digest,
        "precision": "bf16",
        "feature_schema": expected_schema,
        "right_postclose_endpoint_only": True,
        "left_g0_frames0_to6_bitexact": True,
        "official_reward_runtime_used": False,
        "policy_modified": False,
        "rl_authorized": False,
        "sha256": {
            "checkpoint": sha(partial / "all200_action_step50.pt"),
            "runtime_source": sha(partial / "v464_v169_endpoint_residual_runtime.py"),
            "preregistration": sha(partial / "preregistration.json"),
            "training_report": sha(partial / "training_report.json"),
            "s0_report": sha(partial / "s0_report.json"),
            "trainer_source": sha(args.trainer),
            "v169_release_manifest": prereg["v169"]["release_manifest_sha256"],
            "v169_library_manifest": prereg["v169"]["library_manifest_sha256"],
        },
    }
    manifest_path = partial / "v464_endpoint_residual_manifest.json"
    with manifest_path.open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    descriptor = os.open(partial, os.O_RDONLY)
    os.fsync(descriptor)
    os.close(descriptor)
    os.replace(partial, output)
    descriptor = os.open(output.parent, os.O_RDONLY)
    os.fsync(descriptor)
    os.close(descriptor)
    print(json.dumps({"release": str(output), "manifest_sha256": sha(output / "v464_endpoint_residual_manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
