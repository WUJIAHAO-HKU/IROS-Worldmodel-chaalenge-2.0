#!/usr/bin/env python3
"""Create a self-contained, hash-audited local-fusion deployment artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


FILES = {
    "autoregressive": ("model.pt", "training_manifest.json", "action_normalization.npz", "track2_autoregressive_unet_config.npz"),
    "direct_flow": ("model.pt", "training_manifest.json", "action_normalization.npz", "track2_direct_flow_unet_config.npz"),
    "fusion": ("model.pt", "training_manifest.json", "action_normalization.npz", "local_fusion_config.npz"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoregressive", required=True)
    parser.add_argument("--direct-flow", required=True)
    parser.add_argument("--fusion", required=True)
    parser.add_argument("--validation-report", required=True)
    parser.add_argument("--local-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite package: {output}")
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    sources = {"autoregressive": Path(args.autoregressive).resolve(), "direct_flow": Path(args.direct_flow).resolve(), "fusion": Path(args.fusion).resolve()}
    try:
        temporary.mkdir(parents=True)
        config = {"format": "track2-local-motion-texture-fusion-ensemble-v1", "selection_reports": {"validation": str(Path(args.validation_report).resolve()), "local_test": str(Path(args.local_report).resolve())}}
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
