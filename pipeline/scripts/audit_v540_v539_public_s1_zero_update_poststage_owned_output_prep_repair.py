#!/usr/bin/env python3
"""Independent data-only audit of staged v540 public-S1/zero-update evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8<<20),b""): digest.update(block)
    return digest.hexdigest()


def audit_staged(root: Path, manifest: dict, baseline_model: dict, rng_entry: dict) -> dict:
    expected_preterminal={"attempt_intent.json","public_s1_events.ndjson","public_s1_metrics.json","source_manifest.json","zero_update_receipt.json","zero_update_trace.json"}
    if root.is_symlink() or not root.is_dir() or {x.name for x in root.iterdir()} != expected_preterminal:
        raise RuntimeError("auditor staged exact6")
    events=[]
    with (root/"public_s1_events.ndjson").open("r",encoding="utf-8") as stream:
        for ordinal,line in enumerate(stream):
            row=json.loads(line)
            expected={"absolute_error_sum_uint64","branch","branch_index","episode","event_canonical_sha256","isolated_torch_seed","model_snapshot_sha256","ordinal","pixel_count_uint64","prediction_sha256","rng_snapshot_sha256","start","target_sha256","torch_rng_isolation"}
            if (
                set(row)!=expected
                or type(row["ordinal"]) is not int
                or row["ordinal"]!=ordinal
                or row["torch_rng_isolation"]!="fork_rng_all_cuda_devices_per_public_row"
                or type(row["isolated_torch_seed"]) is not int
                or row["isolated_torch_seed"]!=1671*1000+ordinal
                or canonical_sha({k:v for k,v in row.items() if k!="event_canonical_sha256"})!=row["event_canonical_sha256"]
            ):
                raise RuntimeError("auditor event schema/order")
            events.append(row)
    if len(events)!=44 or [x["branch"] for x in events] != ["right_recursive32"]*32+["left_first8"]*12:
        raise RuntimeError("auditor public branch partition")
    metrics=json.loads((root/"public_s1_metrics.json").read_text()); trace=json.loads((root/"zero_update_trace.json").read_text()); zero=json.loads((root/"zero_update_receipt.json").read_text())
    error=sum(x["absolute_error_sum_uint64"] for x in events); pixels=sum(x["pixel_count_uint64"] for x in events)
    checks={
        "public_rows_exact44":metrics.get("rows")==44 and metrics.get("right_rows")==32 and metrics.get("left_rows")==12,
        "metrics_recomputed":metrics.get("absolute_error_sum_uint64")==error and metrics.get("pixel_count_uint64")==pixels and metrics.get("mean_absolute_error")==error/pixels,
        "ordered_events_digest":metrics.get("ordered_events_sha256")==canonical_sha([x["event_canonical_sha256"] for x in events]),
        "public_label_only":manifest["public_s1_input_contract"]["public_label_keys"]==["target_frames"] and manifest["public_s1_input_contract"]["aliases_authorized"] is False,
        "zero_trace_schema":trace.get("format")=="strict-track2-v540-v539-zero-update-trace-v1" and type(trace.get("stage_count")) is int and trace["stage_count"]==len(trace.get("stages",[]))==46,
        "all_intermediate_model_snapshots_bitexact":trace.get("all_stages_bitexact") is True and all(x.get("model")==baseline_model for x in trace["stages"]),
        "all_intermediate_rng_snapshots_bitexact":trace.get("rng_all_stages_bitexact") is True and trace.get("rng_entry")==rng_entry and trace.get("rng_exit")==rng_entry and all(x.get("rng")==rng_entry for x in trace["stages"]),
        "parameter_and_buffer_digests_event_bound":all(x["model_snapshot_sha256"]==canonical_sha(baseline_model) for x in events),
        "rng_event_bound":all(x["rng_snapshot_sha256"]==canonical_sha(rng_entry) for x in events),
        "zero_update_counts":zero.get("passed") is True and all(type(zero.get(key)) is int and zero[key]==0 for key in ("parameter_updates","buffer_updates","optimizer_instances","scheduler_instances","optimizer_steps","optimizer_zero_grads","scheduler_steps","backward_calls","model_writes","cache_writes","reward_reads","hidden_private_final_reads")),
        "mutation_restore_rejected":zero.get("mutation_then_restore_accepted") is False and zero.get("all_intermediate_stages_bitexact") is True,
        "trace_hash_bound":zero.get("trace_sha256")==sha(root/"zero_update_trace.json"),
        "output_contract_exact8":manifest["public_s1_output_contract"]["exact_count"]==8,
        "no_reward_hidden_private":manifest["hidden_input_denial_contract"]["hidden_private_final_reward_inputs_authorized"] is False,
    }
    if any(value is not True for value in checks.values()): raise RuntimeError("independent public-S1/zero-update audit")
    return {"format":"strict-track2-v540-v539-public-s1-zero-update-independent-audit-v1","status":"passed_independent_public_s1_zero_update_audit","passed":True,"checks":checks,"check_keys":sorted(checks),"checks_sha256":canonical_sha(checks),"events_sha256":sha(root/"public_s1_events.ndjson"),"metrics_sha256":sha(root/"public_s1_metrics.json"),"zero_update_trace_sha256":sha(root/"zero_update_trace.json"),"models_loaded_by_auditor":False,"reward_hidden_private_read_by_auditor":False,"training_or_update_by_auditor":False}


if __name__=="__main__":
    raise SystemExit("library-only independent auditor; direct execution forbidden")
