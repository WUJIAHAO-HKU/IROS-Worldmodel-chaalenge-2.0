#!/usr/bin/env python3
"""Apply frame-local observed glyph correction to a frozen v14.0 cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_contact_motion_retrieval_v140 import metrics, reproject_observed_glyph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-cache", required=True,
                        help="Parent cache containing context, target and ordered window names")
    parser.add_argument("--v140-cache", required=True)
    parser.add_argument("--v140-report", required=True)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--text-alpha", type=float, default=0.90)
    parser.add_argument("--text-highpass-strength", type=float, default=0.55)
    parser.add_argument("--text-minimum-ecc", type=float, default=0.55)
    parser.add_argument("--text-prefix-frames", type=int, default=2)
    args = parser.parse_args()

    with np.load(args.parent_cache, allow_pickle=False) as cache:
        target = cache["target"]
        context = cache["context"]
        names = cache["windows"].astype(str)
    with np.load(args.v140_cache, allow_pickle=False) as cache:
        prediction = cache["prediction"].copy()
        v140_names = cache["windows"].astype(str)
        arm_id = cache["arm_id"].copy() if "arm_id" in cache.files else None
    if not np.array_equal(names, v140_names):
        raise ValueError("parent and v14.0 caches have different window order")

    source_report = json.loads(Path(args.v140_report).read_text())
    details_by_name = {value["window"]: value for value in source_report["windows"]}
    output = prediction.copy(); diagnostics = []
    for index, name in enumerate(names):
        source_detail = details_by_name.get(name)
        if not source_detail or not source_detail.get("accepted") or source_detail.get("arm") != 1:
            continue
        start_frame = int(source_detail["start_frame"])
        text_end = min(start_frame, args.text_prefix_frames, output.shape[1])
        frame_details = []
        for time in range(text_end):
            hp_scale = 1.0 if text_end == 1 else max(1.0 - time / (text_end - 1), 0.0)
            output[index, time], detail = reproject_observed_glyph(
                output[index, time], context[index, -1], args.text_minimum_ecc,
                args.text_alpha, args.text_highpass_strength * hp_scale,
            )
            detail["frame"] = time + 1; frame_details.append(detail)
        diagnostics.append({"window": name, "arm": 1, "start_frame": start_frame,
                            "text_prefix_frames": text_end, "frames": frame_details})

    before = metrics(prediction, target); after = metrics(output, target)
    report = {
        "format": "track2-contact-motion-retrieval-v14.1-frozen-v14-text",
        "parent": "frozen v14.0 output cache",
        "sample_count": len(names),
        "accepted_window_count": len(diagnostics),
        "parameters": {
            "text_alpha": args.text_alpha,
            "text_highpass_strength": args.text_highpass_strength,
            "text_minimum_ecc": args.text_minimum_ecc,
            "text_prefix_frames": args.text_prefix_frames,
        },
        "overall": {"v14.0": before, "v14.1": after},
        "unchanged_pixel_fraction": float((output == prediction).mean()),
        "windows": diagnostics,
    }
    if arm_id is not None:
        report["arms"] = {}
        for arm in (0, 1):
            selected = np.flatnonzero(arm_id == arm)
            arm_report = {"sample_count": len(selected)}
            if len(selected):
                arm_report.update({
                    "v14.0": metrics(prediction[selected], target[selected]),
                    "v14.1": metrics(output[selected], target[selected]),
                })
            report["arms"][f"arm{arm}"] = arm_report

    output_cache = Path(args.output_cache); output_cache.parent.mkdir(parents=True, exist_ok=True)
    values = {"prediction": output, "target": target, "context": context,
              "windows": names}
    if arm_id is not None:
        values["arm_id"] = arm_id
    np.savez_compressed(output_cache, **values)
    output_report = Path(args.output_report); output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "windows"}, indent=2))


if __name__ == "__main__":
    main()
