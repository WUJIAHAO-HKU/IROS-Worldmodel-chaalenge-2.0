#!/usr/bin/env python3
"""Package the single preregistered v482 all200 checkpoint after independent S0 PASS."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch


FORMAT = "track2-v482-temporal8-residual-release-v1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser()
    for name in ("preregistration", "contract", "result-dir", "audit", "runtime", "trainer",
                 "prepare", "auditor", "release-auditor", "v169-release", "v169-library", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    pre = json.loads(args.preregistration.read_text())
    report = json.loads((args.result_dir / "s0_report.json").read_text())
    audit = json.loads(args.audit.read_text())
    final = args.output_dir.resolve(); partial = final.with_name(final.name + ".partial")
    if final.exists() or partial.exists():
        raise FileExistsError(final)
    if (
        sha(Path(__file__)) != pre["source"]["packager_sha256"]
        or sha(args.contract) != pre["source"]["contract_sha256"]
        or sha(args.runtime) != pre["source"]["runtime_sha256"]
        or sha(args.trainer) != pre["source"]["trainer_sha256"]
        or sha(args.prepare) != pre["source"]["prepare_sha256"]
        or sha(args.auditor) != pre["source"]["auditor_sha256"]
        or sha(args.release_auditor) != pre["source"]["release_auditor_sha256"]
        or args.v169_release.resolve() != Path(pre["v169"]["release_path"]).resolve()
        or args.v169_library.resolve() != Path(pre["v169"]["library_path"]).resolve()
        or report.get("passed") is not True
        or report.get("all200_training_performed") is not True
        or audit.get("passed") is not True
        or audit.get("candidate_passed") is not True
    ):
        raise RuntimeError("v482 package unauthorized")
    checkpoint = args.result_dir / "all200_action.pt"
    if sha(checkpoint) != report["all200_checkpoint_sha256"]:
        raise RuntimeError("v482 checkpoint drift")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not all(bool(torch.isfinite(value).all()) for value in state.get("model", {}).values()):
        raise RuntimeError("v482 nonfinite checkpoint")
    partial.mkdir(parents=True)
    sources = {
        "v482_temporal8_residual_runtime.py": args.runtime,
        "all200_action.pt": checkpoint,
        "v482_design_contract.json": args.contract,
        "v482_execution_preregistration.json": args.preregistration,
        "v482_s0_report.json": args.result_dir / "s0_report.json",
        "v482_s0_audit.json": args.audit,
        "train_v482_temporal8_residual_5fold.py": args.trainer,
        "prepare_v482_temporal8_residual_s0.py": args.prepare,
        "audit_v482_temporal8_residual_s0.py": args.auditor,
        "package_v482_temporal8_residual_release.py": Path(__file__),
        "audit_v482_temporal8_residual_release.py": args.release_auditor,
        "v169_closure.json": Path(pre["v169"]["closure_path"]),
    }
    for name, source in sources.items():
        shutil.copy2(source, partial / name)
    inventory = [
        {"relative": name, "sha256": sha(partial / name), "bytes": (partial / name).stat().st_size}
        for name in sorted(sources)
    ]
    manifest = {
        "format": FORMAT,
        "endpoint_only": False,
        "ordered_scalar_v169": True,
        "reward_or_outcome_used": False,
        "checkpoint": "all200_action.pt",
        "v169_release": str(args.v169_release.resolve()),
        "v169_library": str(args.v169_library.resolve()),
        "v169_library_manifest": pre["v169"]["library_manifest_path"],
        "v169_closure": "v169_closure.json",
        "v169_closure_digest": pre["v169"]["closure_digest"],
        "closure_digest": state["closure_digest"],
        "architecture": "v482_prefix_causal_film_residual_unet128",
        "normalization": "frozen selection float32 minmax; zero-span runtime clip then +0",
        "film_formula": "x*(1+gamma)+beta",
        "runtime_gate": "explicit-right and full8 postclose",
        "inventory_excluding_manifest": inventory,
        "inventory_digest_sha256": hashlib.sha256(
            json.dumps([[x["relative"], x["sha256"]] for x in inventory], separators=(",", ":")).encode()
        ).hexdigest(),
        "sha256": {
            "runtime_source": sha(args.runtime),
            "checkpoint": sha(checkpoint),
            "v169_release_manifest": pre["v169"]["release_manifest_sha256"],
            "v169_library_manifest": pre["v169"]["library_manifest_sha256"],
            "v169_closure": pre["v169"]["closure_sha256"],
            "contract": sha(args.contract),
            "preregistration": sha(args.preregistration),
            "report": sha(args.result_dir / "s0_report.json"),
            "audit": sha(args.audit),
            "packager": sha(Path(__file__)),
        },
        "guards": {"s1_authorized": False, "zero_update_authorized": False, "policy_updates": 0, "rl_authorized": False},
    }
    atomic_json(partial / "v482_temporal8_residual_manifest.json", manifest)
    for path in partial.iterdir():
        with path.open("rb") as stream: os.fsync(stream.fileno())
    descriptor = os.open(str(partial), os.O_RDONLY); os.fsync(descriptor); os.close(descriptor)
    os.replace(partial, final)
    descriptor = os.open(str(final.parent), os.O_RDONLY); os.fsync(descriptor); os.close(descriptor)
    print(json.dumps({"release": str(final), "manifest_sha256": sha(final / "v482_temporal8_residual_manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
