#!/usr/bin/env python3
"""Streaming, mmap-backed parameter displacement audit for frozen policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def component(key: str) -> str:
    if "vision_tower" in key:
        return "vision_tower"
    if "gemma_expert" in key or ".expert" in key:
        return "action_expert"
    if "paligemma" in key:
        return "paligemma_language"
    if "time_mlp" in key or "action_in_proj" in key or "action_out_proj" in key:
        return "action_head"
    return key.split(".", 1)[0]


def empty_stats() -> dict:
    return {
        "tensors": 0,
        "changed_tensors": 0,
        "numel": 0,
        "changed_elements": 0,
        "base_sq": 0.0,
        "delta_sq": 0.0,
        "delta_abs": 0.0,
        "max_abs_delta": 0.0,
    }


def finalize(stats: dict) -> dict:
    numel = max(int(stats["numel"]), 1)
    base_sq = float(stats["base_sq"])
    delta_sq = float(stats["delta_sq"])
    return {
        "tensors": int(stats["tensors"]),
        "changed_tensors": int(stats["changed_tensors"]),
        "numel": int(stats["numel"]),
        "changed_elements": int(stats["changed_elements"]),
        "changed_element_fraction": float(stats["changed_elements"] / numel),
        "base_l2": math.sqrt(base_sq),
        "delta_l2": math.sqrt(delta_sq),
        "relative_l2": math.sqrt(delta_sq / max(base_sq, 1e-30)),
        "delta_rms": math.sqrt(delta_sq / numel),
        "mean_abs_delta": float(stats["delta_abs"] / numel),
        "max_abs_delta": float(stats["max_abs_delta"]),
    }


def compare(base: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor], chunk: int) -> dict:
    if list(base) != list(candidate):
        raise RuntimeError("state-dict key/order mismatch")
    global_stats = empty_stats()
    components: dict[str, dict] = {}
    rows = []
    for key in base:
        left = base[key]
        right = candidate[key]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise TypeError(f"non-tensor state entry: {key}")
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(f"tensor metadata mismatch: {key}")
        if not left.is_floating_point():
            if not torch.equal(left, right):
                raise RuntimeError(f"changed non-floating tensor: {key}")
            continue
        local = empty_stats()
        local["tensors"] = 1
        flat_left = left.reshape(-1)
        flat_right = right.reshape(-1)
        for begin in range(0, flat_left.numel(), chunk):
            end = min(begin + chunk, flat_left.numel())
            a = flat_left[begin:end].float()
            b = flat_right[begin:end].float()
            delta = b - a
            local["numel"] += int(delta.numel())
            local["changed_elements"] += int(torch.count_nonzero(delta).item())
            local["base_sq"] += float(torch.sum(a.double() * a.double()).item())
            local["delta_sq"] += float(torch.sum(delta.double() * delta.double()).item())
            local["delta_abs"] += float(torch.sum(delta.abs().double()).item())
            if delta.numel():
                local["max_abs_delta"] = max(
                    float(local["max_abs_delta"]), float(delta.abs().max().item())
                )
        local["changed_tensors"] = int(local["changed_elements"] > 0)
        for name in ("tensors", "changed_tensors", "numel", "changed_elements"):
            global_stats[name] += local[name]
        for name in ("base_sq", "delta_sq", "delta_abs"):
            global_stats[name] += local[name]
        global_stats["max_abs_delta"] = max(
            global_stats["max_abs_delta"], local["max_abs_delta"]
        )
        group = components.setdefault(component(key), empty_stats())
        for name in ("tensors", "changed_tensors", "numel", "changed_elements"):
            group[name] += local[name]
        for name in ("base_sq", "delta_sq", "delta_abs"):
            group[name] += local[name]
        group["max_abs_delta"] = max(group["max_abs_delta"], local["max_abs_delta"])
        row = finalize(local)
        row["key"] = key
        rows.append(row)
    return {
        "global": finalize(global_stats),
        "components": {name: finalize(value) for name, value in sorted(components.items())},
        "top_delta_l2": sorted(rows, key=lambda row: row["delta_l2"], reverse=True)[:20],
        "top_relative_l2_numel_ge1024": sorted(
            (row for row in rows if row["numel"] >= 1024),
            key=lambda row: row["relative_l2"],
            reverse=True,
        )[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v169-step5", required=True, type=Path)
    parser.add_argument("--v169-step6", required=True, type=Path)
    parser.add_argument("--v383", required=True, type=Path)
    parser.add_argument("--v411", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-elements", type=int, default=1 << 20)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = {
        "v169_step5": args.v169_step5,
        "v169_step6": args.v169_step6,
        "v383": args.v383,
        "v411": args.v411,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    states = {
        name: torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        for name, path in paths.items()
    }
    report = {
        "format": "strict-track2-v413-policy-update-delta-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tensor_count": len(states["v169_step5"]),
        "comparisons": {
            "v169_step5_to_step6": compare(states["v169_step5"], states["v169_step6"], args.chunk_elements),
            "v169_step5_to_v383": compare(states["v169_step5"], states["v383"], args.chunk_elements),
            "v169_step5_to_v411": compare(states["v169_step5"], states["v411"], args.chunk_elements),
            "v383_to_v411": compare(states["v383"], states["v411"], args.chunk_elements),
        },
        "evidence": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "guards": {
            "read_only": True,
            "evaluation_outcomes_used": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "global": {name: value["global"] for name, value in report["comparisons"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
