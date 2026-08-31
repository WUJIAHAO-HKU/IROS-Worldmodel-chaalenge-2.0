#!/usr/bin/env python3
"""Episode-held-out temporal-8 residual S0 over immutable public-train simulator data."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wam_pipeline.v169_arm_routed_runtime import Track2V169ArmRoutedRuntime
from wam_pipeline.v482_temporal8_residual_runtime import (
    ACTION_FEATURE_DIM,
    CHECKPOINT_FORMAT,
    FEATURE_SCHEMA,
    TemporalResidualUNet128FiLM,
    causal_action_features,
    request_gate,
    verify_v169,
)


SEED = 1624
CHANNELS = 16
BATCH_SEQUENCES = 2
BRANCHES = ("factual", "no_transport", "scale_0p4", "scale_1p25", "reverse_direction_0p4")
MODES = ("action", "context_only", "phase_shuffle")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def arrsha(value):
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, default=json_default)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(str(path.parent), os.O_RDONLY)
    os.fsync(descriptor)
    os.close(descriptor)


def stable_seed(episode, start):
    payload = f"v482-v169-cache/seed1624/episode{int(episode)}/start{int(start)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31)


def load_data(args, pre):
    if sha(Path(__file__)) != pre["source"]["trainer_sha256"]:
        raise RuntimeError("v482 trainer source drift")
    if sha(pre["source"]["runtime_path"]) != pre["source"]["runtime_sha256"]:
        raise RuntimeError("v482 runtime source drift")
    if sha(args.contract) != pre["source"]["contract_sha256"]:
        raise RuntimeError("v482 contract drift")
    if (
        args.v169_release.resolve() != Path(pre["v169"]["release_path"]).resolve()
        or args.v169_library.resolve() != Path(pre["v169"]["library_path"]).resolve()
        or sha(pre["v169"]["release_manifest_path"]) != pre["v169"]["release_manifest_sha256"]
        or sha(pre["v169"]["library_manifest_path"]) != pre["v169"]["library_manifest_sha256"]
        or pre["v169"].get("required_closure_digest") != "5fc181bf1049b0e763ef47716f5e3356445de8eb6b2eb3de07dd2d70cd4346f3"
        or sha(pre["v169"]["closure_path"]) != pre["v169"]["closure_sha256"]
        or verify_v169(json.loads(Path(pre["v169"]["closure_path"]).read_text())) != pre["v169"]["closure_digest"]
    ):
        raise RuntimeError("v482 v169 path/manifest closure drift")
    final_report = json.loads(Path(pre["dataset"]["final_report"]["path"]).read_text())
    final_audit = json.loads(Path(pre["dataset"]["final_audit"]["path"]).read_text())
    if (
        sha(pre["dataset"]["final_report"]["path"]) != pre["dataset"]["final_report"]["sha256"]
        or sha(pre["dataset"]["final_audit"]["path"]) != pre["dataset"]["final_audit"]["sha256"]
        or final_report.get("passed") is not True
        or final_audit.get("passed") is not True
        or final_report.get("collection_integrity_passed") is not True
        or final_report.get("technical_effect_diagnostic_qualification_passed") is not True
    ):
        raise RuntimeError("v482 blocked: temporal200 final PASS required")
    selection_path = Path(pre["dataset"]["selection"]["path"])
    if sha(selection_path) != pre["dataset"]["selection"]["sha256"]:
        raise RuntimeError("v482 selection drift")
    selection = json.loads(selection_path.read_text())
    specs = {(int(row["episode"]), int(row["start"])): row for row in selection["contexts"]}
    root = Path(pre["dataset"]["root"]).resolve()
    declared = {row["relative"]: row["sha256"] for row in pre["dataset"]["files"]}
    actual = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("v482 dataset symlink forbidden")
        if path.is_file():
            actual[path.relative_to(root).as_posix()] = sha(path)
    if actual != declared:
        raise RuntimeError("v482 exact dataset tree drift")
    contexts, samples, canonical_actions = [], [], []
    context_manifest = []
    for cid, spec in enumerate(selection["contexts"]):
        batch = int(spec["batch_id"])
        rowdir = root / f"batch_{batch:03d}" / "rows" / f"episode{int(spec['episode'])}_start{int(spec['start']):05d}"
        npz_path = rowdir / "temporal.npz"
        receipt_path = rowdir / "receipt.json"
        receipt = json.loads((rowdir / "receipt.json").read_text())
        with np.load(npz_path, allow_pickle=False) as data:
            episode = int(data["episode"])
            start = int(data["start"])
            variants = list(np.asarray(data["variants"]).astype("U"))
            history = np.asarray(data["history_actions"])
            futures = np.asarray(data["future_actions"])
            stored_pre_future_context = np.asarray(data["pre_future_context_rgb"])
            targets6 = np.asarray(data["temporal_rgb"])
            if (
                (episode, start) != (int(spec["episode"]), int(spec["start"]))
                or variants != list(BRANCHES) + ["factual_duplicate"]
                or history.shape != (4, 14)
                or history.dtype != np.float32
                or futures.shape != (6, 8, 14)
                or futures.dtype != np.float32
                or stored_pre_future_context.shape != (6, 8, 256, 256, 3)
                or stored_pre_future_context.dtype != np.uint8
                or not stored_pre_future_context.flags.c_contiguous
                or targets6.shape != (6, 8, 256, 256, 3)
                or targets6.dtype != np.uint8
            ):
                raise RuntimeError("v482 temporal row schema drift")
            redundancy_count = sum(
                np.array_equal(stored_pre_future_context[branch, frame], stored_pre_future_context[0, 0])
                for branch in range(6) for frame in range(8)
            )
            if (
                redundancy_count != 48
                or not np.array_equal(
                    stored_pre_future_context,
                    np.broadcast_to(stored_pre_future_context[0, 0], stored_pre_future_context.shape),
                )
                or not np.array_equal(futures[0], futures[5])
                or not np.array_equal(targets6[0], targets6[5])
                or receipt.get("factual_duplicate_temporal_rgb_state_pose_bottle_bitexact") is not True
            ):
                raise RuntimeError("v482 context/duplicate identity drift")
            pre_future_context = np.ascontiguousarray(stored_pre_future_context[0, 0])
            constructed_context = np.repeat(pre_future_context[None, :, :, :], 5, axis=0)
            frozen_context = pre["temporal_contexts"][cid]
            observed_context = {
                "selection_order": cid,
                "episode": episode,
                "start": start,
                "batch_id": batch,
                "row_npz_path": str(npz_path.resolve()),
                "row_npz_sha256": sha(npz_path),
                "row_receipt_path": str(receipt_path.resolve()),
                "row_receipt_sha256": sha(receipt_path),
                "redundancy_identity_gate": True,
                "redundancy_equality_count": redundancy_count,
                "stored_pre_future_context_rgb_sha256": arrsha(stored_pre_future_context),
                "pre_future_context_rgb_sha256": arrsha(pre_future_context),
                "constructed_repeat5_context_sha256": arrsha(constructed_context),
            }
            if observed_context != frozen_context or not np.array_equal(
                constructed_context, np.broadcast_to(pre_future_context, constructed_context.shape)
            ):
                raise RuntimeError("v482 frozen repeat-five context manifest drift")
            context_manifest.append(observed_context)
            if arrsha(history) != spec["history_action_sha256"]:
                raise RuntimeError("v482 history action drift")
            for branch, name in enumerate(BRANCHES):
                if arrsha(futures[branch]) != spec["branch_action_sha256"][name]:
                    raise RuntimeError("v482 branch action drift")
        cid = len(contexts)
        canonical_actions.append(np.concatenate([history] + [futures[branch] for branch in range(5)], axis=0))
        contexts.append(
            {
                "episode": episode,
                "start": start,
                "fold": int(spec["fold"]),
                "phase": int(spec["phase_bin"]),
                "instruction": spec["instruction"],
                "context": pre_future_context.copy(),
                "history": history.copy(),
            }
        )
        for branch in range(5):
            samples.append(
                {
                    "context_id": cid,
                    "episode": episode,
                    "fold": int(spec["fold"]),
                    "phase": int(spec["phase_bin"]),
                    "branch": branch,
                    "future": futures[branch].copy(),
                    "target": targets6[branch].copy(),
                }
            )
    if len(contexts) != 200 or len(samples) != 1000:
        raise RuntimeError("v482 exact 200x5 data contract")
    context_manifest_digest = hashlib.sha256(
        json.dumps(context_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if context_manifest_digest != pre["v169_cache"]["context_manifest_sha256"]:
        raise RuntimeError("v482 repeat-five ordered context manifest drift")
    if pre["v169_cache"].get("redundancy_rows_passed") != 200 or pre["v169_cache"].get("redundancy_equality_count") != 9600:
        raise RuntimeError("v482 redundancy aggregate drift")
    source_actions = np.ascontiguousarray(np.stack(canonical_actions), np.float32)
    lower = source_actions.min(axis=(0, 1)).astype(np.float32)
    upper = source_actions.max(axis=(0, 1)).astype(np.float32)
    if (
        source_actions.shape != (200, 44, 14)
        or arrsha(source_actions) != pre["action_bounds"]["source_array_sha256"]
        or not np.array_equal(lower, np.asarray(pre["action_bounds"]["lower"], np.float32))
        or not np.array_equal(upper, np.asarray(pre["action_bounds"]["upper"], np.float32))
        or arrsha(np.concatenate((lower, upper))) != pre["action_bounds"]["lower_upper_sha256"]
    ):
        raise RuntimeError("v482 canonical action-bound derivation drift")
    return contexts, samples


def build_baseline_cache(args, pre, contexts, samples, work):
    cache_root = Path("/dev/shm/v482_v169_temporal_cache_seed1624")
    cache_partial = cache_root.with_name(cache_root.name + ".partial")
    if cache_root.exists() or cache_partial.exists():
        raise RuntimeError("v482 scalar cache path must be absent")
    cache_partial.mkdir()
    cache = cache_partial / "scalar_v169_temporal_cache.npz"
    repeated_context_hashes = [
        arrsha(np.repeat(context["context"][None], 5, axis=0)) for context in contexts
    ]
    repeated_context_digest = hashlib.sha256("".join(repeated_context_hashes).encode()).hexdigest()
    if repeated_context_digest != pre["v169_cache"]["context_repeat5_digest_sha256"]:
        raise RuntimeError("v482 frozen repeat5 temporal context drift")
    v169 = Track2V169ArmRoutedRuntime(args.v169_release, args.v169_library, args.device)
    baselines, request_hashes, output_hashes, seeds = [], [], [], []
    for sample in samples:
        context = contexts[sample["context_id"]]
        request_context = np.repeat(context["context"][None], 5, axis=0)
        seed = stable_seed(sample["episode"], context["start"])
        request_digest = hashlib.sha256()
        request_digest.update(np.ascontiguousarray(request_context).view(np.uint8))
        request_digest.update(np.ascontiguousarray(context["history"], np.float32).view(np.uint8))
        request_digest.update(np.ascontiguousarray(sample["future"], np.float32).view(np.uint8))
        request_digest.update(np.asarray([seed], dtype="<i8").view(np.uint8))
        request_digest.update(str(context["instruction"]).encode("utf-8"))
        request_hashes.append(request_digest.hexdigest())
        prediction = np.asarray(
            v169.predict(
                request_context,
                context["history"],
                sample["future"],
                seed,
                context["instruction"],
            )
        )
        if prediction.shape != (8, 256, 256, 3) or prediction.dtype != np.uint8:
            raise RuntimeError("v482 scalar v169 cache contract")
        baselines.append(prediction)
        output_hashes.append(arrsha(prediction))
        seeds.append(seed)
    del v169
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    baseline_array = np.stack(baselines).reshape(200, 5, 8, 256, 256, 3)
    payload = {
        "baseline": baseline_array,
        "seed": np.asarray(seeds, np.int64).reshape(200, 5),
        "request_sha256": np.asarray(request_hashes).reshape(200, 5),
        "output_sha256": np.asarray(output_hashes).reshape(200, 5),
        "sample_id": np.arange(1000, dtype=np.int64).reshape(200, 5),
    }
    temporary = cache.with_name(cache.name + ".tmp")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, cache)
    manifest = {
        "format": "strict-track2-v482-scalar-v169-temporal-cache-v1",
        "array_sha256": sha(cache),
        "ordered_scalar_requests": 1000,
        "request_digest_sha256": hashlib.sha256("".join(request_hashes).encode()).hexdigest(),
        "output_digest_sha256": hashlib.sha256("".join(output_hashes).encode()).hexdigest(),
        "seed_digest_sha256": arrsha(payload["seed"]),
        "context_repeat5_digest_sha256": repeated_context_digest,
        "context_repeat5_sha256": repeated_context_hashes,
    }
    atomic_json(cache_partial / "manifest.json", manifest)
    descriptor = os.open(str(cache_partial), os.O_RDONLY)
    os.fsync(descriptor)
    os.close(descriptor)
    os.replace(cache_partial, cache_root)
    descriptor = os.open(str(cache_root.parent), os.O_RDONLY); os.fsync(descriptor); os.close(descriptor)
    return baseline_array.reshape(1000, 8, 256, 256, 3), {
        **manifest,
        "path": str(cache_root / "scalar_v169_temporal_cache.npz"),
        "manifest_path": str(cache_root / "manifest.json"),
        "manifest_sha256": sha(cache_root / "manifest.json"),
    }


def schedule(ids, seed):
    ids = np.asarray(ids, np.int64)
    rng = np.random.default_rng(seed)
    ordered = ids[rng.permutation(len(ids))]
    if len(ordered) % BATCH_SEQUENCES:
        raise RuntimeError("v482 exact epoch batch divisibility")
    return ordered.reshape(-1, BATCH_SEQUENCES)


def schedule_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value, np.int64).view(np.uint8)).hexdigest()


def state_sha(state):
    """Canonical state digest independent of torch's container serialization."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def donor_map(contexts, samples, fit_ids, seed, query_ids=None):
    fit_set = set(map(int, fit_ids))
    query_ids = np.asarray(fit_ids if query_ids is None else query_ids, np.int64)
    del seed
    groups = {}
    for index in fit_ids:
        sample = samples[int(index)]
        groups.setdefault(sample["phase"], set()).add(sample["context_id"])
    result = {}
    ordered_groups = {
        phase: sorted(context_ids, key=lambda cid: (contexts[cid]["episode"], contexts[cid]["start"], cid))
        for phase, context_ids in groups.items()
    }
    for index in query_ids:
        index = int(index)
        sample = samples[index]
        ordered = ordered_groups[sample["phase"]]
        query_context = contexts[sample["context_id"]]
        insertion = next(
            (position for position, cid in enumerate(ordered) if (contexts[cid]["episode"], contexts[cid]["start"], cid) > (query_context["episode"], query_context["start"], sample["context_id"])),
            0,
        )
        donor_context = None
        for offset in range(len(ordered)):
            candidate = ordered[(insertion + offset) % len(ordered)]
            if contexts[candidate]["episode"] != query_context["episode"]:
                donor_context = candidate
                break
        if donor_context is None:
            raise RuntimeError("v482 no different-episode phase donor")
        donor = 5 * donor_context + sample["branch"]
        if donor not in fit_set or samples[donor]["phase"] != sample["phase"] or samples[donor]["branch"] != sample["branch"]:
            raise RuntimeError("v482 donor closure")
        result[index] = donor
    if set(result) != set(map(int, query_ids)):
        raise RuntimeError("v482 incomplete donor map")
    return result


def tensors(contexts, samples, baselines, ids, feature_ids, lower, upper, mode, device):
    rows = [samples[int(index)] for index in ids]
    feature_rows = [samples[int(index)] for index in feature_ids]
    query_histories = np.stack([contexts[row["context_id"]]["history"] for row in rows])
    query_futures = np.stack([row["future"] for row in rows])
    features = causal_action_features(
        query_histories,
        query_futures,
        lower,
        upper,
        "context_only" if mode == "context_only" else "action",
        strict_source=True,
    )
    if mode == "phase_shuffle":
        donor_histories = np.stack([contexts[row["context_id"]]["history"] for row in feature_rows])
        donor_futures = np.stack([row["future"] for row in feature_rows])
        donor_features = causal_action_features(
            donor_histories, donor_futures, lower, upper, "action", strict_source=True
        )
        features[:, :, :168] = donor_features[:, :, :168]
    base_np = np.stack([baselines[int(index)] for index in ids])
    target_np = np.stack([row["target"] for row in rows])
    context_np = np.stack([contexts[row["context_id"]]["context"] for row in rows])
    base = torch.as_tensor(base_np, dtype=torch.float32, device=device).permute(0, 1, 4, 2, 3) / 255.0
    context = torch.as_tensor(context_np, dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
    context = context[:, None].expand(-1, 8, -1, -1, -1)
    target = torch.as_tensor(target_np, dtype=torch.float32, device=device).permute(0, 1, 4, 2, 3) / 255.0
    target = F.avg_pool2d(target.reshape(-1, 3, 256, 256), 2, 2).reshape(-1, 8, 3, 128, 128)
    base_target = F.avg_pool2d(base.reshape(-1, 3, 256, 256), 2, 2).reshape(-1, 8, 3, 128, 128)
    return (
        rows,
        base.reshape(-1, 3, 256, 256),
        context.reshape(-1, 3, 256, 256),
        torch.as_tensor(features.reshape(-1, ACTION_FEATURE_DIM), device=device),
        target,
        base_target,
    )


def loss_value(prediction, base, target):
    prediction = (base + prediction.reshape(-1, 8, 3, 128, 128).float()).clamp(0, 1)
    target = target.float()
    epsilon2 = (1.0 / 255.0) ** 2
    charbonnier = torch.sqrt((prediction - target).square() + epsilon2).mean()
    pooled = 0.5 * sum(
        F.l1_loss(
            F.avg_pool2d(prediction.reshape(-1, 3, 128, 128), factor, factor),
            F.avg_pool2d(target.reshape(-1, 3, 128, 128), factor, factor),
        )
        for factor in (2, 4)
    )
    temporal_error = (prediction[:, 1:] - prediction[:, :-1]) - (target[:, 1:] - target[:, :-1])
    temporal = torch.sqrt(temporal_error.square() + epsilon2).mean()
    return charbonnier + 0.10 * pooled + 0.05 * temporal


def train_head(initial, contexts, samples, baselines, ids, epoch, donors, lower, upper, mode, device):
    model = TemporalResidualUNet128FiLM(CHANNELS).to(device)
    model.load_state_dict(copy.deepcopy(initial))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-4)
    losses = []
    for batch in epoch:
        feature_ids = [donors[int(index)] for index in batch] if mode == "phase_shuffle" else batch
        _, base, context, features, target, base_target = tensors(
            contexts, samples, baselines, batch, feature_ids, lower, upper, mode, device
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(base, context, features)
            loss = loss_value(prediction, base_target, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError("v482 nonfinite loss or gradient")
        optimizer.step()
        if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise RuntimeError("v482 nonfinite parameter after optimizer step")
        losses.append(float(loss.detach().cpu()))
    return model.eval(), losses


@torch.inference_mode()
def predict(model, contexts, samples, baselines, ids, feature_ids, lower, upper, mode, device):
    continuous, uint8 = [], []
    for begin in range(len(ids)):
        batch = np.asarray(ids[begin : begin + 1])
        fids = np.asarray(feature_ids[begin : begin + 1])
        rows, base, context, features, _, _ = tensors(
            contexts, samples, baselines, batch, fids, lower, upper, mode, device
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            residual128 = model(base, context, features)
        if not bool(torch.isfinite(residual128).all()):
            raise RuntimeError("v482 nonfinite prediction")
        residual = residual128.repeat_interleave(2, 2).repeat_interleave(2, 3)
        candidate = (255.0 * base + 255.0 * residual).clamp(0, 255).reshape(-1, 8, 3, 256, 256)
        if not bool(torch.isfinite(candidate).all()):
            raise RuntimeError("v482 nonfinite continuous candidate")
        candidate = candidate.permute(0, 1, 3, 4, 2).float().cpu().numpy()
        for local, row in enumerate(rows):
            context_row = contexts[row["context_id"]]
            if not request_gate(context_row["history"], row["future"], context_row["instruction"])["gate"]:
                candidate[local] = baselines[int(batch[local])]
        continuous.append(candidate)
        uint8.append(np.clip(np.rint(candidate), 0, 255).astype(np.uint8))
    return np.concatenate(continuous), np.concatenate(uint8)


def mae(prediction, target):
    return np.abs(prediction.astype(np.float64) - target.astype(np.float64)).mean((2, 3, 4))


def metrics(contexts, samples, ids, baseline, action, action_uint, controls):
    numeric = [baseline, action, action_uint, *controls.values()]
    if any(not np.isfinite(np.asarray(value)).all() for value in numeric):
        raise RuntimeError("v482 nonfinite metric input")
    target = np.stack([samples[int(index)]["target"] for index in ids])
    baseline_mae = mae(baseline, target)
    action_mae = mae(action, target)
    uint_mae = mae(action_uint, target)
    enabled = np.asarray(
        [
            request_gate(
                contexts[samples[int(index)]["context_id"]]["history"],
                samples[int(index)]["future"],
                contexts[samples[int(index)]["context_id"]]["instruction"],
            )["gate"]
            for index in ids
        ],
        bool,
    )
    if not enabled.any():
        raise RuntimeError("v482 empty enabled holdout")
    def ratio(numerator, denominator, mask=enabled):
        return float(numerator[mask].mean() / max(denominator[mask].mean(), 1e-12))
    branch = {}
    for branch_id, name in enumerate(BRANCHES):
        mask = enabled & np.asarray([samples[int(index)]["branch"] == branch_id for index in ids])
        if not mask.any():
            raise RuntimeError("v482 empty enabled branch")
        branch[name] = ratio(action_mae, baseline_mae, mask)
    prefix = [ratio(action_mae[:, frame], baseline_mae[:, frame]) for frame in range(8)]
    context_mae = mae(controls["context_only"], target)
    phase_mae = mae(controls["phase_shuffle"], target)
    protected = ~enabled
    protected_mismatch = np.zeros(len(ids), np.int64)
    for local in np.flatnonzero(protected):
        protected_mismatch[local] = int(np.count_nonzero(action_uint[local] != baseline[local]))
    protected_exact = bool(np.all(protected_mismatch[protected] == 0))
    cosine, nonzero, cosine_dot, target_norm, prediction_norm, delta_prediction_norm = [], [], [], [], [], []
    cosine_fold, nonzero_fold = [], []
    position = {int(index): local for local, index in enumerate(ids)}
    for cid in sorted({samples[int(index)]["context_id"] for index in ids}):
        group = [5 * cid + branch_id for branch_id in range(5)]
        if not all(index in position for index in group):
            continue
        p = np.stack([action[position[index]] for index in group]).astype(np.float64)
        b = np.stack([baseline[position[index]] for index in group]).astype(np.float64)
        t = np.stack([target[position[index]] for index in group]).astype(np.float64)
        for branch_id in range(1, 5):
            pd = ((p[branch_id] - b[branch_id]) - (p[0] - b[0])).reshape(-1)
            td = (
                (t[branch_id] - b[branch_id])
                - (t[0] - b[0])
            ).reshape(-1)
            dot = float(np.dot(pd, td))
            pn = float(np.linalg.norm(pd))
            tn = float(np.linalg.norm(td))
            cosine.append(tn > 1e-8 and dot > 0)
            cosine_dot.append(dot)
            target_norm.append(tn)
            prediction_norm.append(pn)
            cosine_fold.append(int(contexts[cid]["fold"]))
            pn_all = float(np.linalg.norm(pd))
            nonzero.append(pn_all > 1e-8)
            delta_prediction_norm.append(pn_all)
            nonzero_fold.append(int(contexts[cid]["fold"]))
    episodes = sorted({samples[int(index)]["episode"] for index in ids})
    improved = sum(
        action_mae[[samples[int(index)]["episode"] == episode for index in ids]].mean()
        < baseline_mae[[samples[int(index)]["episode"] == episode for index in ids]].mean()
        for episode in episodes
    )
    episode_uint8_ratios = {
        str(episode): float(
            uint_mae[[samples[int(index)]["episode"] == episode for index in ids]].sum()
            / max(baseline_mae[[samples[int(index)]["episode"] == episode for index in ids]].sum(), 1e-12)
        )
        for episode in episodes
    }
    cosine_eligible = np.asarray(target_norm, np.float64) > 1e-8
    if not cosine_eligible.any():
        raise RuntimeError("v482 empty eligible causal-cosine set")
    result = {
        "enabled_sequences": int(enabled.sum()),
        "protected_sequences": int(protected.sum()),
        "continuous_ratio": ratio(action_mae, baseline_mae),
        "uint8_ratio": ratio(uint_mae, baseline_mae),
        "branch_ratio": branch,
        "prefix_ratio": prefix,
        "first7_ratio": float(action_mae[enabled, :7].mean() / baseline_mae[enabled, :7].mean()),
        "action_over_context_only": ratio(action_mae, context_mae),
        "action_over_phase_shuffle": ratio(action_mae, phase_mae),
        "positive_delta_cosine_fraction": float(
            np.mean(np.asarray(cosine_dot, np.float64)[cosine_eligible] > 0)
        ),
        "nonzero_prediction_delta_fraction": float(np.mean(nonzero)),
        "improved_episodes": int(improved),
        "episode_uint8_ratio": episode_uint8_ratios,
        "episode_uint8_ratio_max": max(episode_uint8_ratios.values()),
        "protected_uint8_bitexact": protected_exact,
    }
    if any(
        not np.isfinite(value)
        for value in (
            result["continuous_ratio"], result["uint8_ratio"], result["first7_ratio"],
            result["action_over_context_only"], result["action_over_phase_shuffle"],
            result["positive_delta_cosine_fraction"], result["nonzero_prediction_delta_fraction"],
            result["episode_uint8_ratio_max"], *branch.values(), *prefix,
        )
    ):
        raise RuntimeError("v482 nonfinite metric")
    result["passed"] = bool(
        result["continuous_ratio"] <= 0.95
        and result["uint8_ratio"] <= 1.0
        and branch["factual"] <= 1.0
        and all(branch[name] <= 0.98 for name in BRANCHES[1:])
        and max(prefix) <= 1.0
        and result["first7_ratio"] <= 0.98
        and result["action_over_context_only"] <= 0.95
        and result["action_over_phase_shuffle"] <= 0.97
        and result["positive_delta_cosine_fraction"] >= 0.60
        and result["nonzero_prediction_delta_fraction"] >= 0.95
        and result["episode_uint8_ratio_max"] <= 1.05
        and protected_exact
    )
    evidence = {
        "baseline_mae": baseline_mae,
        "action_mae": action_mae,
        "uint_mae": uint_mae,
        "context_mae": context_mae,
        "phase_mae": phase_mae,
        "enabled": enabled,
        "protected_pixel_mismatch_count": protected_mismatch,
        "cosine_positive": np.asarray(cosine, bool),
        "cosine_dot": np.asarray(cosine_dot, np.float64),
        "cosine_target_norm": np.asarray(target_norm, np.float64),
        "cosine_prediction_norm": np.asarray(prediction_norm, np.float64),
        "cosine_fold": np.asarray(cosine_fold, np.int64),
        "delta_nonzero": np.asarray(nonzero, bool),
        "delta_prediction_norm": np.asarray(delta_prediction_norm, np.float64),
        "delta_nonzero_fold": np.asarray(nonzero_fold, np.int64),
    }
    return result, evidence


def main():
    parser = argparse.ArgumentParser()
    for name in ("preregistration", "contract", "v169-release", "v169-library", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    pre = json.loads(args.preregistration.read_text())
    final = args.output_dir.resolve()
    work = final.with_name(final.name + ".partial")
    if (
        pre.get("format") != "strict-track2-v482-temporal8-residual-preregistration-v1"
        or final.exists()
        or work.exists()
    ):
        raise RuntimeError("v482 prereg/output contract")
    work.mkdir(parents=True)
    contexts, samples = load_data(args, pre)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    lower = np.asarray(pre["action_bounds"]["lower"], np.float32)
    upper = np.asarray(pre["action_bounds"]["upper"], np.float32)
    if lower.shape != (14,) or upper.shape != (14,) or np.any(upper < lower) or lower[13] != 0 or upper[13] != 0:
        raise RuntimeError("v482 frozen selection action bounds")
    baselines, cache_receipt = build_baseline_cache(args, pre, contexts, samples, work)
    atomic_json(work / "scalar_v169_temporal_cache_receipt.json", cache_receipt)
    closure_digest = hashlib.sha256(
        json.dumps(
            {"dataset": pre["dataset"], "v169": pre["v169"], "source": pre["source"]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    fold_results, failed = [], 0
    oof = {
        name: np.full((1000, 8), np.nan, np.float64)
        for name in ("baseline_mae", "action_mae", "uint_mae", "context_mae", "phase_mae")
    }
    oof["enabled"] = np.zeros(1000, bool)
    oof["protected_pixel_mismatch_count"] = np.full(1000, np.nan, np.float64)
    oof["episode"] = np.asarray([row["episode"] for row in samples], np.int64)
    oof["branch"] = np.asarray([row["branch"] for row in samples], np.int64)
    oof["fold"] = np.asarray([row["fold"] for row in samples], np.int64)
    oof_cosine, oof_nonzero = [], []
    oof_cosine_dot, oof_cosine_target_norm, oof_cosine_prediction_norm = [], [], []
    oof_cosine_fold, oof_nonzero_fold, oof_delta_prediction_norm = [], [], []
    for fold in range(5):
        fit_ids = np.asarray([index for index, row in enumerate(samples) if row["fold"] != fold], np.int64)
        hold_ids = np.asarray([index for index, row in enumerate(samples) if row["fold"] == fold], np.int64)
        epoch = schedule(fit_ids, SEED + fold)
        if schedule_sha(epoch) != pre["schedules"][fold]["sha256"]:
            raise RuntimeError("v482 frozen schedule drift")
        donors = donor_map(contexts, samples, fit_ids, SEED + fold)
        donor_digest = hashlib.sha256(
            json.dumps(donors, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if donor_digest != pre["phase_shuffle"][fold]["fit_sha256"]:
            raise RuntimeError("v482 phase donor drift")
        torch.manual_seed(SEED + fold)
        initial = copy.deepcopy(TemporalResidualUNet128FiLM(CHANNELS).state_dict())
        initial_digest = state_sha(initial)
        if initial_digest != pre["initial_state_sha256"][fold]:
            raise RuntimeError("v482 frozen initial state drift")
        models, losses = {}, {}
        for mode in MODES:
            models[mode], losses[mode] = train_head(
                initial, contexts, samples, baselines, fit_ids, epoch, donors, lower, upper, mode, device
            )
        predictions, uints = {}, {}
        hold_donors = donor_map(contexts, samples, fit_ids, SEED + 100 + fold, hold_ids)
        hold_donor_digest = hashlib.sha256(
            json.dumps(hold_donors, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if hold_donor_digest != pre["phase_shuffle"][fold]["holdout_sha256"]:
            raise RuntimeError("v482 holdout phase donor drift")
        for mode in MODES:
            feature_ids = np.asarray(
                [hold_donors[int(index)] for index in hold_ids] if mode == "phase_shuffle" else hold_ids
            )
            predictions[mode], uints[mode] = predict(
                models[mode], contexts, samples, baselines, hold_ids, feature_ids, lower, upper, mode, device
            )
        fold_metric, fold_evidence = metrics(
            contexts,
            samples,
            hold_ids,
            baselines[hold_ids],
            predictions["action"],
            uints["action"],
            {name: predictions[name] for name in ("context_only", "phase_shuffle")},
        )
        fold_metric.update(
            {
                "fold": fold,
                "fit_sequences": len(fit_ids),
                "holdout_sequences": len(hold_ids),
                "schedule_sha256": schedule_sha(epoch),
                "phase_shuffle_fit_sha256": donor_digest,
                "phase_shuffle_holdout_sha256": hold_donor_digest,
                "initial_state_sha256": initial_digest,
                "loss_first_last": {mode: [losses[mode][0], losses[mode][-1]] for mode in MODES},
                "all_losses_finite": all(np.isfinite(losses[mode]).all() for mode in MODES),
            }
        )
        fold_results.append(fold_metric)
        for name in ("baseline_mae", "action_mae", "uint_mae", "context_mae", "phase_mae"):
            oof[name][hold_ids] = fold_evidence[name]
        oof["enabled"][hold_ids] = fold_evidence["enabled"]
        oof["protected_pixel_mismatch_count"][hold_ids] = fold_evidence["protected_pixel_mismatch_count"]
        oof_cosine.extend(fold_evidence["cosine_positive"].tolist())
        oof_cosine_dot.extend(fold_evidence["cosine_dot"].tolist())
        oof_cosine_target_norm.extend(fold_evidence["cosine_target_norm"].tolist())
        oof_cosine_prediction_norm.extend(fold_evidence["cosine_prediction_norm"].tolist())
        oof_cosine_fold.extend(fold_evidence["cosine_fold"].tolist())
        oof_nonzero.extend(fold_evidence["delta_nonzero"].tolist())
        oof_delta_prediction_norm.extend(fold_evidence["delta_prediction_norm"].tolist())
        oof_nonzero_fold.extend(fold_evidence["delta_nonzero_fold"].tolist())
        if not fold_metric["passed"]:
            failed += 1
        common = {
            "format": CHECKPOINT_FORMAT,
            "channels": CHANNELS,
            "precision": "bf16",
            "feature_schema": FEATURE_SCHEMA,
            "closure_digest": closure_digest,
            "trainer_sha256": pre["source"]["trainer_sha256"],
            "runtime_sha256": pre["source"]["runtime_sha256"],
            "preregistration_sha256": sha(args.preregistration),
            "action_lower": lower,
            "action_upper": upper,
        }
        for mode, model in models.items():
            if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
                raise RuntimeError("v482 nonfinite fold checkpoint")
            torch.save(
                {**common, "training_scope": f"fold{fold}-{mode}", "model": model.state_dict()},
                work / f"fold{fold}_{mode}.pt",
            )
        atomic_json(work / f"fold{fold}_receipt.json", fold_metric)
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if failed >= 2:
            break
    early = failed >= 2
    passed = False
    aggregate = None
    if not early and len(fold_results) == 5:
        if any(not np.isfinite(oof[name]).all() for name in ("baseline_mae", "action_mae", "uint_mae", "context_mae", "phase_mae")):
            raise RuntimeError("v482 incomplete OOF scalar evidence")
        enabled = oof["enabled"]
        def ratio(numerator, denominator, mask=enabled):
            return float(numerator[mask].mean() / max(denominator[mask].mean(), 1e-12))
        branch_ratio = {}
        for branch_id, name in enumerate(BRANCHES):
            mask = enabled & np.asarray([row["branch"] == branch_id for row in samples])
            branch_ratio[name] = ratio(oof["action_mae"], oof["baseline_mae"], mask)
        prefix_ratio = [ratio(oof["action_mae"][:, frame], oof["baseline_mae"][:, frame]) for frame in range(8)]
        improved_episodes = sum(
            oof["action_mae"][[row["episode"] == episode for row in samples]].mean()
            < oof["baseline_mae"][[row["episode"] == episode for row in samples]].mean()
            for episode in sorted({row["episode"] for row in samples})
        )
        aggregate_cosine_eligible = np.asarray(oof_cosine_target_norm, np.float64) > 1e-8
        if not aggregate_cosine_eligible.any():
            raise RuntimeError("v482 empty aggregate eligible causal-cosine set")
        aggregate = {
            "passing_folds": sum(row["passed"] for row in fold_results),
            "enabled_sequences": int(enabled.sum()),
            "protected_sequences": int((~enabled).sum()),
            "continuous_ratio": ratio(oof["action_mae"], oof["baseline_mae"]),
            "uint8_ratio": ratio(oof["uint_mae"], oof["baseline_mae"]),
            "branch_ratio": branch_ratio,
            "prefix_ratio": prefix_ratio,
            "first7_ratio": float(oof["action_mae"][enabled, :7].mean() / oof["baseline_mae"][enabled, :7].mean()),
            "action_over_context_only": ratio(oof["action_mae"], oof["context_mae"]),
            "action_over_phase_shuffle": ratio(oof["action_mae"], oof["phase_mae"]),
            "positive_delta_cosine_fraction": float(
                np.mean(
                    np.asarray(oof_cosine_dot, np.float64)[aggregate_cosine_eligible]
                    > 0
                )
            ),
            "nonzero_prediction_delta_fraction": float(np.mean(oof_nonzero)),
            "improved_episodes": int(improved_episodes),
            "episode_uint8_ratio": {
                str(episode): float(
                    oof["uint_mae"][[row["episode"] == episode for row in samples]].sum()
                    / max(oof["baseline_mae"][[row["episode"] == episode for row in samples]].sum(), 1e-12)
                )
                for episode in sorted({row["episode"] for row in samples})
            },
            "protected_uint8_bitexact": bool(
                np.all(oof["protected_pixel_mismatch_count"][~enabled] == 0)
            ),
        }
        aggregate["episode_uint8_ratio_max"] = max(aggregate["episode_uint8_ratio"].values())
        passed = bool(
            aggregate["passing_folds"] >= 4
            and aggregate["continuous_ratio"] <= 0.95
            and aggregate["uint8_ratio"] <= 1.0
            and aggregate["branch_ratio"]["factual"] <= 1.0
            and all(aggregate["branch_ratio"][name] <= 0.98 for name in BRANCHES[1:])
            and max(aggregate["prefix_ratio"]) <= 1.0
            and aggregate["first7_ratio"] <= 0.98
            and aggregate["action_over_context_only"] <= 0.95
            and aggregate["action_over_phase_shuffle"] <= 0.97
            and aggregate["positive_delta_cosine_fraction"] >= 0.60
            and aggregate["nonzero_prediction_delta_fraction"] >= 0.95
            and aggregate["improved_episodes"] >= 12
            and aggregate["episode_uint8_ratio_max"] <= 1.05
            and aggregate["protected_uint8_bitexact"]
        )
        if passed:
            all_ids = np.arange(1000, dtype=np.int64)
            epoch = schedule(all_ids, SEED + 1000)
            if schedule_sha(epoch) != pre["all200_schedule_sha256"]:
                raise RuntimeError("v482 all200 schedule drift")
            torch.manual_seed(SEED + 1000)
            initial = copy.deepcopy(TemporalResidualUNet128FiLM(CHANNELS).state_dict())
            if state_sha(initial) != pre["all200_initial_state_sha256"]:
                raise RuntimeError("v482 frozen all200 initial state drift")
            model, _ = train_head(
                initial, contexts, samples, baselines, all_ids, epoch, {}, lower, upper, "action", device
            )
            if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
                raise RuntimeError("v482 nonfinite all200 checkpoint")
            torch.save(
                {
                    "format": CHECKPOINT_FORMAT,
                    "training_scope": "all200-action",
                    "channels": CHANNELS,
                    "precision": "bf16",
                    "feature_schema": FEATURE_SCHEMA,
                    "closure_digest": closure_digest,
                    "trainer_sha256": pre["source"]["trainer_sha256"],
                    "runtime_sha256": pre["source"]["runtime_sha256"],
                    "preregistration_sha256": sha(args.preregistration),
                    "action_lower": lower,
                    "action_upper": upper,
                    "model": model.state_dict(),
                },
                work / "all200_action.pt",
            )
    oof_path = work / "oof_scalar_evidence.npz"
    with oof_path.with_name(oof_path.name + ".tmp").open("xb") as stream:
        np.savez_compressed(
            stream,
            **oof,
            cosine_positive=np.asarray(oof_cosine, bool),
            cosine_dot=np.asarray(oof_cosine_dot, np.float64),
            cosine_target_norm=np.asarray(oof_cosine_target_norm, np.float64),
            cosine_prediction_norm=np.asarray(oof_cosine_prediction_norm, np.float64),
            cosine_fold=np.asarray(oof_cosine_fold, np.int64),
            delta_nonzero=np.asarray(oof_nonzero, bool),
            delta_prediction_norm=np.asarray(oof_delta_prediction_norm, np.float64),
            delta_nonzero_fold=np.asarray(oof_nonzero_fold, np.int64),
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(oof_path.with_name(oof_path.name + ".tmp"), oof_path)
    report = {
        "format": "strict-track2-v482-temporal8-residual-s0-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": sha(args.preregistration),
        "contract_sha256": sha(args.contract),
        "trainer_sha256": sha(Path(__file__)),
        "runtime_sha256": sha(pre["source"]["runtime_path"]),
        "passed": passed,
        "early_stop_mathematically_unreachable": early,
        "completed_folds": len(fold_results),
        "failed_folds": failed,
        "folds": fold_results,
        "aggregate": aggregate,
        "scalar_v169_cache": cache_receipt,
        "oof_scalar_evidence_sha256": sha(oof_path),
        "all200_training_performed": bool(passed),
        "all200_checkpoint_sha256": sha(work / "all200_action.pt") if passed else None,
        "guards": {
            "public_train_only": True,
            "factual_duplicate_training_samples": 0,
            "reward_loaded": False,
            "outcome_loaded": False,
            "development_or_final_used": False,
            "training_launch_required_full200_final_pass": True,
            "s1_authorized": False,
            "zero_update_authorized": False,
            "policy_updates": 0,
            "rl_authorized": False,
        },
        "normalization_clip_counts": {
            "below_lower": [0] * 14,
            "above_upper": [0] * 14,
            "zero_span_dimensions": np.flatnonzero(upper == lower).tolist(),
        },
    }
    atomic_json(work / "s0_report.json", report)
    if not passed:
        atomic_json(
            work / "failure_receipt.json",
            {
                "passed": False,
                "completed_folds": len(fold_results),
                "failed_folds": failed,
                "all200_training_performed": False,
                "retry_authorized": False,
            },
        )
    for path in work.iterdir():
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    os.replace(work, final)
    descriptor = os.open(str(final.parent), os.O_RDONLY)
    os.fsync(descriptor)
    os.close(descriptor)
    print(json.dumps(report, indent=2, default=json_default))
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as error:
        if "--output-dir" in sys.argv:
            try:
                output = Path(sys.argv[sys.argv.index("--output-dir") + 1]).resolve()
                partial = output.with_name(output.name + ".partial")
                receipt = partial / "hard_failure_receipt.json"
                if partial.is_dir() and not receipt.exists() and not receipt.with_name(receipt.name + ".tmp").exists():
                    atomic_json(
                        receipt,
                        {
                            "format": "strict-track2-v482-temporal8-residual-hard-failure-v1",
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "training_retry_authorized": False,
                            "all200_training_performed": False,
                            "s1_authorized": False,
                            "zero_update_authorized": False,
                            "policy_updates": 0,
                            "rl_authorized": False,
                        },
                    )
            except Exception:
                pass
        raise
