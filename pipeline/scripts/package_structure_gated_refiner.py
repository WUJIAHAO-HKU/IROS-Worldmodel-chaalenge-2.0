#!/usr/bin/env python3
"""Package the accepted baseline, explicit-structure parent, and local refiner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


FILES = {
    "structure_parent": ("model.pt", "training_manifest.json", "action_normalization.npz", "track2_autoregressive_structure_unet_config.npz"),
    "refiner": ("model.pt", "training_manifest.json", "action_normalization.npz", "structure_refiner_config.npz"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    from wam_pipeline.local_fusion_runtime import ensemble_artifact_sha256

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--structure-parent", required=True)
    parser.add_argument("--refiner", required=True)
    parser.add_argument("--validation-report", required=True)
    parser.add_argument("--local-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite package: {output}")
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    sources = {"structure_parent": Path(args.structure_parent).resolve(), "refiner": Path(args.refiner).resolve()}
    baseline = Path(args.baseline).resolve()
    try:
        temporary.mkdir(parents=True)
        shutil.copytree(baseline, temporary / "baseline")
        config = {
            "format": "track2-structure-gated-local-fusion-ensemble-v1",
            "baseline": {"directory": "baseline", "source_checkpoint": str(baseline), "artifact_sha256": ensemble_artifact_sha256(baseline)},
            "selection_reports": {"validation": str(Path(args.validation_report).resolve()), "local_test": str(Path(args.local_report).resolve())},
        }
        for name, filenames in FILES.items():
            destination = temporary / name
            destination.mkdir()
            hashes = {}
            for filename in filenames:
                source = sources[name] / filename
                if not source.is_file():
                    raise FileNotFoundError(source)
                shutil.copy2(source, destination / filename)
                hashes[filename] = sha256(destination / filename)
            config[name] = {"directory": name, "source_checkpoint": str(sources[name]), "sha256": hashes}
        (temporary / "ensemble_config.json").write_text(json.dumps(config, indent=2) + "\n")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"output": str(output.resolve()), "format": config["format"]}, indent=2))


if __name__ == "__main__":
    main()
