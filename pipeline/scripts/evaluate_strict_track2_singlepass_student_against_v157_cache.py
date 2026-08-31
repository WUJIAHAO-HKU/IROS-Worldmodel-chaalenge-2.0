#!/usr/bin/env python3
"""Screen a single-pass student against an immutable cached V15.7 reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from evaluate_strict_track2_autoregressive_candidate import Accumulator, gains, metric_rows
from wam_pipeline.multisource_flow_unet_runtime import Track2MultiSourceFlowUNet


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_frames(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(value.copy()).permute(0, 3, 1, 2).float().div(255).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--v157-cache", type=Path, required=True)
    parser.add_argument("--v157-cache-sha256", required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-cache", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--benchmark-total", type=int, default=32)
    parser.add_argument("--benchmark-micro-batch", type=int, default=8)
    args = parser.parse_args()
    if args.benchmark_total < 1 or not 1 <= args.benchmark_micro_batch <= 8:
        raise SystemExit("benchmark total must be positive and micro-batch must be in [1,8]")
    if sha256(args.v157_cache) != args.v157_cache_sha256:
        raise SystemExit("V15.7 cache SHA256 mismatch")
    with np.load(args.v157_cache, allow_pickle=False) as cache:
        cached = {name: cache[name].copy() for name in cache.files}
    required = {
        "context_last", "target", "candidate", "arm_right", "capture_success",
        "path", "synthetic_seed", "start",
    }
    if not required.issubset(cached):
        raise SystemExit("V15.7 cache is missing required arrays")
    count = len(cached["path"])
    student = Track2MultiSourceFlowUNet(args.student, args.device)
    before, after = Accumulator(), Accumulator()
    candidate_outputs, latencies = [], []
    benchmark_inputs = []
    for index, name in enumerate(cached["path"]):
        path = args.windows / str(name)
        if not path.is_file():
            raise SystemExit(f"source window is missing: {path}")
        with np.load(path, allow_pickle=False) as values:
            context = values["context_frames"].copy()
            history = values["history_actions"].copy()
            future = values["future_actions"].copy()
            target = values["target_frames"].copy()
        if not np.array_equal(context[-1], cached["context_last"][index]):
            raise SystemExit(f"cached context mismatch: {path}")
        if not np.array_equal(target, cached["target"][index]):
            raise SystemExit(f"cached target mismatch: {path}")
        started = time.perf_counter()
        prediction = student.predict(context, history, future, 0, None)
        latencies.append(time.perf_counter() - started)
        candidate_outputs.append(prediction)
        if len(benchmark_inputs) < args.benchmark_total:
            benchmark_inputs.append((context, history, future))
        target_tensor = tensor_frames(target)
        context_tensor = tensor_frames(context)[0]
        reference_rows = metric_rows(
            tensor_frames(cached["candidate"][index]), target_tensor, context_tensor[-1:]
        )
        student_rows = metric_rows(
            tensor_frames(prediction), target_tensor, context_tensor[-1:]
        )
        arm_right = bool(cached["arm_right"][index])
        success = bool(cached["capture_success"][index])
        synthetic = int(cached["synthetic_seed"][index]) >= 0
        masks = {
            "overall": torch.ones(1, dtype=torch.bool),
            "left": torch.tensor([not arm_right]),
            "right": torch.tensor([arm_right]),
            "capture_success": torch.tensor([success]),
            "capture_failure": torch.tensor([not success]),
            "official": torch.tensor([not synthetic]),
            "synthetic": torch.tensor([synthetic]),
            "official_left": torch.tensor([not synthetic and not arm_right]),
            "official_right": torch.tensor([not synthetic and arm_right]),
            "synthetic_left": torch.tensor([synthetic and not arm_right]),
            "synthetic_right": torch.tensor([synthetic and arm_right]),
        }
        for group, mask in masks.items():
            before.add(group, reference_rows, mask)
            after.add(group, student_rows, mask)
        if index == 0 or (index + 1) % 8 == 0 or index + 1 == count:
            print(json.dumps({"evaluated": index + 1, "total": count}), flush=True)
    reference, candidate = before.result(), after.result()
    steady = latencies[1:] if len(latencies) > 1 else latencies
    benchmark_started = time.perf_counter()
    if student.device.type == "cuda":
        torch.cuda.synchronize(student.device)
        torch.cuda.reset_peak_memory_stats(student.device)
        benchmark_started = time.perf_counter()
    batched_outputs = []
    for start in range(0, len(benchmark_inputs), args.benchmark_micro_batch):
        batch = benchmark_inputs[start : start + args.benchmark_micro_batch]
        batched_outputs.extend(student.predict_batch(
            np.stack([value[0] for value in batch]),
            np.stack([value[1] for value in batch]),
            np.stack([value[2] for value in batch]),
            [0] * len(batch),
            [None] * len(batch),
        ))
    if student.device.type == "cuda":
        torch.cuda.synchronize(student.device)
    benchmark_seconds = time.perf_counter() - benchmark_started
    cuda_memory = None
    if student.device.type == "cuda":
        cuda_memory = {
            "peak_allocated_mib": float(torch.cuda.max_memory_allocated(student.device) / 2**20),
            "peak_reserved_mib": float(torch.cuda.max_memory_reserved(student.device) / 2**20),
        }
    serial_benchmark = np.stack(candidate_outputs[: len(batched_outputs)]).astype(np.int16)
    batched_benchmark = np.stack(batched_outputs).astype(np.int16)
    batch_difference = np.abs(serial_benchmark - batched_benchmark)
    report = {
        "format": "strict-track2-v166-singlepass-student-against-v157-cache-v1",
        "windows": str(args.windows.resolve()),
        "v157_cache": str(args.v157_cache.resolve()),
        "v157_cache_sha256": args.v157_cache_sha256,
        "student": str(args.student.resolve()),
        "samples": count,
        "v157": reference,
        "student_metrics": candidate,
        "improvement": gains(reference, candidate),
        "latency_seconds": {
            "first": latencies[0],
            "steady_mean": float(np.mean(steady)),
            "steady_median": float(np.median(steady)),
            "steady_p95": float(np.quantile(steady, 0.95)),
            "batch32_sharded_max8": benchmark_seconds,
            "benchmark_samples": len(benchmark_inputs),
            "benchmark_micro_batch": args.benchmark_micro_batch,
        },
        "cuda_memory": cuda_memory,
        "native_batch_equivalence": {
            "compared_values": int(batch_difference.size),
            "different_values": int(np.count_nonzero(batch_difference)),
            "different_fraction": float(np.count_nonzero(batch_difference) / batch_difference.size),
            "uint8_mae": float(batch_difference.mean()),
            "uint8_max_absolute_error": int(batch_difference.max()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.prediction_cache:
        if args.prediction_cache.exists():
            raise SystemExit(f"refusing to overwrite {args.prediction_cache}")
        args.prediction_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.prediction_cache,
            context_last=cached["context_last"],
            target=cached["target"],
            baseline=cached["candidate"],
            candidate=np.stack(candidate_outputs),
            arm_right=cached["arm_right"],
            capture_success=cached["capture_success"],
            path=cached["path"],
            synthetic_seed=cached["synthetic_seed"],
            start=cached["start"],
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
