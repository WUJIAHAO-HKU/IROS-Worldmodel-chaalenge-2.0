#!/usr/bin/env python3
"""Evaluate sequence-level v13 contact tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.contact_track_v13 import ContactTrackParameters, ContactTrackV13
from scripts.evaluate_contact_layer_v12 import metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-cache")
    parser.add_argument("--minimum-source-area", type=int, default=700)
    parser.add_argument("--maximum-source-area", type=int, default=1500)
    parser.add_argument("--latch-area-ratio", type=float, default=0.65)
    parser.add_argument("--target-area-scale", type=float, default=1.0)
    parser.add_argument("--maximum-expansion-radius", type=int, default=6)
    parser.add_argument("--contact-support-radius", type=int, default=14)
    parser.add_argument("--green-purity-ratio", type=float, default=0.62)
    args = parser.parse_args()
    with np.load(args.prediction_cache, allow_pickle=False) as cache:
        prediction, target, context = cache["prediction"], cache["target"], cache["context"]
        names = cache["windows"].astype(str)
    parameters = ContactTrackParameters(
        minimum_source_area=args.minimum_source_area,
        maximum_source_area=args.maximum_source_area,
        latch_area_ratio=args.latch_area_ratio,
        target_area_scale=args.target_area_scale,
        maximum_expansion_radius=args.maximum_expansion_radius,
        contact_support_radius=args.contact_support_radius,
        green_purity_ratio=args.green_purity_ratio,
    )
    renderer = ContactTrackV13()
    rendered = np.empty_like(prediction)
    diagnostics = []
    for index in range(len(prediction)):
        rendered[index], detail = renderer.render_sequence(prediction[index], context[index], parameters)
        diagnostics.append(detail)
    baseline_parts = [metrics(prediction[i], target[i]) for i in range(len(prediction))]
    rendered_parts = [metrics(rendered[i], target[i]) for i in range(len(prediction))]
    baseline = {key: float(np.mean([part[key] for part in baseline_parts])) for key in baseline_parts[0]}
    result_metrics = {key: float(np.mean([part[key] for part in rendered_parts])) for key in rendered_parts[0]}
    result = {
        "format": "track2-contact-track-v13-eval",
        "sample_count": len(prediction),
        "parameters": vars(parameters),
        "baseline": baseline,
        "rendered": result_metrics,
        "delta": {key: result_metrics[key] - baseline[key] for key in baseline},
        "accepted_sequence_fraction": float(np.mean([value["accepted"] for value in diagnostics])),
        "accepted_frame_fraction": float(np.mean([value["accepted_frame_fraction"] for value in diagnostics])),
        "windows": [
            {"window": str(name), "diagnostics": detail} for name, detail in zip(names, diagnostics)
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if args.output_cache:
        np.savez_compressed(
            args.output_cache, prediction=rendered, target=target, context=context, windows=names
        )
    print(json.dumps({key: value for key, value in result.items() if key != "windows"}, indent=2))


if __name__ == "__main__":
    main()

