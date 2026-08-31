#!/usr/bin/env python3
"""Require V16.6 non-regression on balanced official and synthetic validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads(args.preregistration.read_text())
    cache = json.loads(args.cache_manifest.read_text())
    visual = json.loads(args.visual.read_text())
    if prereg.get("format") != "strict-track2-v166-mixed-visual-screen-preregistration-v1":
        raise SystemExit("unexpected mixed visual screen preregistration")
    if cache.get("format") != "strict-track2-v157-balanced-mixed-visual-cache-v1":
        raise SystemExit("unexpected mixed V15.7 cache manifest")
    if visual.get("format") != "strict-track2-v166-singlepass-student-against-v157-cache-v1":
        raise SystemExit("unexpected V16.6 visual report")
    if visual["v157_cache_sha256"] != cache["cache_sha256"]:
        raise SystemExit("visual report does not use the frozen mixed V15.7 cache")
    if Path(visual["student"]).resolve() != args.candidate.resolve():
        raise SystemExit("visual report does not use the selected V16.6 candidate")
    gain = visual["improvement"]
    required_counts = prereg["required_counts"]
    counts_ok = {
        name: name in gain and int(gain[name]["windows"]) == int(count)
        for name, count in required_counts.items()
    }
    gate = prereg["gates"]
    checks = {
        **{f"count_{name}": passed for name, passed in counts_ok.items()},
        "overall_rgb": gain["overall"]["rgb_mae_improvement_percent"] >= gate["overall_rgb_improvement_percent_min"],
        "official_rgb": gain["official"]["rgb_mae_improvement_percent"] >= gate["official_rgb_improvement_percent_min"],
        "official_left_rgb": gain["official_left"]["rgb_mae_improvement_percent"] >= gate["official_left_rgb_improvement_percent_min"],
        "official_right_rgb": gain["official_right"]["rgb_mae_improvement_percent"] >= gate["official_right_rgb_improvement_percent_min"],
        "official_texture": gain["official"]["texture_mae_improvement_percent"] >= gate["official_texture_improvement_percent_min"],
        "official_temporal": gain["official"]["temporal_delta_mae_improvement_percent"] >= gate["official_temporal_improvement_percent_min"],
        "official_right_contact": gain["official_right"]["contact_rgb_mae_improvement_percent"] >= gate["official_right_contact_improvement_percent_min"],
        "synthetic_rgb": gain["synthetic"]["rgb_mae_improvement_percent"] >= gate["synthetic_rgb_improvement_percent_min"],
    }
    report = {
        "format": "strict-track2-v166-balanced-mixed-visual-screen-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "candidate": str(args.candidate.resolve()),
        "selected_improvements": {
            name: gain[name] for name in ("overall", "official", "official_left", "official_right", "synthetic")
        },
        "evidence": {
            "preregistration": {"path": str(args.preregistration.resolve()), "sha256": sha256(args.preregistration)},
            "cache_manifest": {"path": str(args.cache_manifest.resolve()), "sha256": sha256(args.cache_manifest)},
            "visual": {"path": str(args.visual.resolve()), "sha256": sha256(args.visual)},
        },
        "interpretation_guard": prereg["promotion_guard"],
    }
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
