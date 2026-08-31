#!/usr/bin/env python3
"""Render public-training proxy terminals for spatial error diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sweep_v329_train_only_terminal_bridge import blend, seed
from wam_pipeline.v271_endpoint_calibrated_terminal_runtime import Track2V271EndpointCalibratedTerminal
from wam_pipeline.v290_right_closed_mirror_runtime import mirror_actions, mirror_prompt
from wam_pipeline.v326_blended_phase_terminal_runtime import REPAIR_ALPHA
from wam_pipeline.v328_coherent_phase_trajectory_runtime import Track2V328CoherentPhaseTrajectory


def label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (255, 25), (0, 0, 0), -1)
    cv2.putText(result, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (0, 255, 255), 1, cv2.LINE_AA)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("checkpoint-dir", "library-index", "action-gate", "phase-gate", "windows", "instruction-map", "sweep-report", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    sweep = json.loads(args.sweep_report.read_text())
    if sweep.get("scope") != "declared public-training right-arm phase proxy only":
        raise RuntimeError("visualization input is not training-only")
    mapping = json.loads(args.instruction_map.read_text())
    rows = sweep["rows"]
    # Deterministic coverage of early/middle/late entries without reward-based selection.
    indices = np.linspace(0, len(rows) - 1, 8).round().astype(int)
    runtime = Track2V328CoherentPhaseTrajectory(
        args.checkpoint_dir, args.library_index, args.device, args.action_gate, args.phase_gate
    )
    output_rows = []
    for index in indices:
        record = rows[int(index)]
        path = args.windows / record["window"]
        with np.load(path, allow_pickle=False) as values:
            context = values["context_frames"].astype(np.uint8)
            history = values["history_actions"].astype(np.float32)
            future = values["future_actions"].astype(np.float32)
            target = values["target_frames"].astype(np.uint8)
        prompt = mapping["episode_to_instruction"][str(record["episode"])]
        parent = runtime.parent.predict(context, history, future, seed(path), prompt)
        repair_row, _ = runtime._success_terminal_for_action_context(context, future)
        repaired = runtime._blend(parent, runtime._target(repair_row), future, REPAIR_ALPHA)
        mirrored = Track2V271EndpointCalibratedTerminal.predict(
            runtime,
            np.ascontiguousarray(context[:, :, ::-1, :]),
            mirror_actions(history),
            mirror_actions(future),
            seed(path),
            mirror_prompt(prompt),
        )
        mirrored = np.ascontiguousarray(mirrored[:, :, ::-1, :])
        mixed = blend(mirrored[-1], repaired[-1], 0.85)
        difference = np.abs(repaired[-1].astype(np.int16) - mirrored[-1].astype(np.int16)).astype(np.uint8)
        difference = cv2.applyColorMap(cv2.cvtColor(difference, cv2.COLOR_RGB2GRAY), cv2.COLORMAP_TURBO)
        name = record["window"]
        output_rows.append(np.concatenate((
            label(context[-1], f"{name} context"),
            label(mirrored[-1], "mirrored terminal"),
            label(repaired[-1], "v326 repaired"),
            label(mixed, "lambda 0.85"),
            label(target[-1], "public GT"),
            label(difference, "repair-mirror diff"),
        ), axis=1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), np.concatenate(output_rows, axis=0)):
        raise RuntimeError("failed to write visualization")
    print(json.dumps({"training_only": True, "rows": [rows[int(i)]["window"] for i in indices], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
