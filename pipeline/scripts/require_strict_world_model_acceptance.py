#!/usr/bin/env python3
"""Allow MBRL only for the exact checkpoint that passed full Track 2 validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_window_names(windows: Path, split_manifest: Path) -> list[str]:
    split = json.loads(split_manifest.read_text())
    names = [
        path.name
        for episode in split["validation_episodes"]
        for path in sorted(windows.glob(f"episode{episode}_*.npz"))
    ]
    if not names:
        raise ValueError("the validation split has no windows")
    return names


def checkpoint_weights_path(checkpoint_dir: Path, backend: str) -> Path:
    """Resolve the actual deployable model artifact for each supported backend."""
    if backend == "track2-wan":
        return checkpoint_dir / "track2_wan_lora.pt"
    return checkpoint_dir / "model.pt"


def checkpoint_artifact_sha256(checkpoint_dir: Path, backend: str) -> str:
    """Hash the exact deployable artifact, including every packaged ensemble member."""
    if backend == "wan-flow-ensemble":
        from wam_pipeline.wan_flow_ensemble_runtime import ensemble_artifact_sha256

        return ensemble_artifact_sha256(checkpoint_dir)
    if backend == "autoregressive-flow-ensemble":
        from wam_pipeline.autoregressive_flow_ensemble_runtime import ensemble_artifact_sha256

        return ensemble_artifact_sha256(checkpoint_dir)
    if backend == "local-motion-texture-fusion":
        from wam_pipeline.local_fusion_runtime import ensemble_artifact_sha256

        return ensemble_artifact_sha256(checkpoint_dir)
    model = checkpoint_weights_path(checkpoint_dir, backend)
    if not model.is_file():
        raise ValueError(f"world-model weight artifact is missing: {model}")
    return sha256(model)


def validate_acceptance(
    report: dict,
    expected_names: list[str],
    checkpoint_dir: Path,
    backend: str,
    accept_mae: float,
    wan_base_model: str | None = None,
) -> dict:
    """Validate both the held-out coverage and identity of the evaluated model."""
    if report.get("format") != "track2-open-loop-eval-v1" or report.get("split") != "validation":
        raise ValueError("acceptance report is not a Track 2 validation evaluation")
    if report.get("backend") != backend:
        raise ValueError("acceptance report backend does not match the requested world model")
    if Path(str(report.get("checkpoint_dir", ""))).resolve() != checkpoint_dir.resolve():
        raise ValueError("acceptance report checkpoint directory does not match the requested world model")
    if backend == "wan-flow-ensemble":
        if not wan_base_model:
            raise ValueError("Wan/direct-flow acceptance requires --wan-base-model")
        from wam_pipeline.wan_flow_ensemble_runtime import load_ensemble_config, wan_base_identity

        ensemble = report.get("ensemble")
        active_base = wan_base_identity(wan_base_model)
        packaged_base = load_ensemble_config(checkpoint_dir)["wan"]["base_model"]
        if active_base != packaged_base:
            raise ValueError("active Wan base identity does not match the packaged ensemble")
        if not isinstance(ensemble, dict) or ensemble.get("wan_base_model") != active_base:
            raise ValueError("acceptance report Wan base identity does not match the active base model")
    if report.get("sample_count") != len(expected_names):
        raise ValueError("acceptance report did not evaluate every validation window")
    hard = report.get("hard_acceptance")
    if not isinstance(hard, dict):
        raise ValueError("acceptance report is missing hard_acceptance")
    threshold = float(hard.get("threshold_mae_0_255", float("inf")))
    if (
        not math.isfinite(threshold)
        or threshold <= 0
        or threshold > accept_mae
        or hard.get("window_count") != len(expected_names)
        or not bool(hard.get("all_windows_and_frames_pass"))
    ):
        raise ValueError("world model has not passed the required all-window, all-frame MAE gate")
    windows = report.get("window_acceptance")
    if not isinstance(windows, list) or len(windows) != len(expected_names):
        raise ValueError("acceptance report has incomplete per-window evidence")
    actual_names = [entry.get("window") for entry in windows]
    if actual_names != expected_names:
        raise ValueError("acceptance report window order or coverage does not match the fixed validation split")
    for entry in windows:
        frame_maes = entry.get("mae_by_prediction_frame")
        peak = float(entry.get("max_prediction_frame_mae", float("inf")))
        if (
            not bool(entry.get("passes_all_prediction_frames"))
            or not isinstance(frame_maes, list)
            or len(frame_maes) != 8
            or not math.isfinite(peak)
            or peak >= accept_mae
            or any(not math.isfinite(float(value)) or float(value) >= accept_mae for value in frame_maes)
        ):
            raise ValueError("at least one validation window exceeds the required frame MAE")
    reported_hash = report.get("checkpoint_model_sha256")
    if not isinstance(reported_hash, str) or not reported_hash:
        raise ValueError("acceptance report does not identify its evaluated model weights")
    actual_hash = checkpoint_artifact_sha256(checkpoint_dir, backend)
    if actual_hash != reported_hash:
        raise ValueError("world-model weights changed after the accepted evaluation")
    return {
        "format": "track2-strict-world-model-acceptance-v1",
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "checkpoint_model_sha256": actual_hash,
        "backend": backend,
        "validation_window_count": len(expected_names),
        "accept_mae_0_255": accept_mae,
        "status": "accepted",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Require an exact full-validation Track 2 world-model pass.")
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--backend", default="multisource-flow-unet")
    parser.add_argument("--wan-base-model", help="Required when validating --backend wan-flow-ensemble.")
    parser.add_argument(
        "--accept-mae",
        type=float,
        default=2.5,
        help="Required maximum RGB MAE on the 0--255 scale for every held-out future frame.",
    )
    parser.add_argument("--output", help="Optional acceptance record written only after a pass.")
    args = parser.parse_args()
    if args.accept_mae <= 0:
        raise SystemExit("--accept-mae must be positive")
    try:
        report = json.loads(Path(args.evaluation).read_text())
        result = validate_acceptance(
            report,
            expected_window_names(Path(args.windows), Path(args.split_manifest)),
            Path(args.checkpoint_dir),
            args.backend,
            args.accept_mae,
            args.wan_base_model,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"strict world-model acceptance required: {exc}") from exc
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
