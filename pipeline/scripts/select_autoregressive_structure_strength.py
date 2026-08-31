#!/usr/bin/env python3
"""Materialize a held-out-selected structure coupling as an immutable checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--sweep-report", required=True)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.checkpoint_dir), Path(args.output)
    report = json.loads(Path(args.sweep_report).read_text())
    matches = [record for record in report["strengths"] if abs(float(record["strength"]) - args.strength) < 1e-12]
    if len(matches) != 1:
        raise ValueError("selected strength is not present exactly once in the sweep")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("action_normalization.npz", "track2_autoregressive_structure_unet_config.npz", "training_manifest.json"):
        shutil.copy2(source / name, output / name)
    state = torch.load(source / "model.pt", map_location="cpu", weights_only=True)
    if state.get("format") not in {"track2-autoregressive-structure-unet-v2", "track2-autoregressive-structure-unet-v3"}:
        raise ValueError("selected checkpoint is not a coupled structure model")
    state["state_dict"]["structure_strength"].fill_(args.strength)
    temporary = output / f"model.pt.tmp.{os.getpid()}"
    torch.save(state, temporary)
    os.replace(temporary, output / "model.pt")
    selection = {
        "format": "track2-autoregressive-structure-strength-selection-v1",
        "source_checkpoint": str(source.resolve()),
        "sweep_report": str(Path(args.sweep_report).resolve()),
        "selected_strength": args.strength,
        "held_out_metrics": matches[0],
    }
    (output / "structure_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps({"output": str(output), **selection}, indent=2))


if __name__ == "__main__":
    main()
