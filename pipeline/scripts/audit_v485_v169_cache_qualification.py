#!/usr/bin/env python3
"""Independent pre-cleanup and final auditor for v485 two-process cache qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


BASELINE_SHAPE=(1000,8,256,256,3)
ALLOWED_CATEGORY="UserWarning"
ALLOWED_MESSAGE=("median CUDA with indices output does not have a deterministic implementation, but you set "
"'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues "
"to help us prioritize adding deterministic support for this operation. (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:93.)")
BRANCHES=("factual","no_transport","scale_0p4","scale_1p25","reverse_direction_0p4")
CONTRACT_SHA="8c4a583aa19de6927cc1ef787920b65317893e27ad01371d0b95b622f260cf64"


def sha(path):
    d=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):d.update(b)
    return d.hexdigest()


def array_raw_sha(array):
    d=hashlib.sha256();flat=np.asarray(array).reshape(-1)
    step=64<<20
    for start in range(0,flat.size,step): d.update(np.ascontiguousarray(flat[start:start+step]).tobytes())
    return d.hexdigest()
def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
def validate_completion(root,base,role,intent,worker_receipt):
    completion_path=base/"completion_receipt.json";completion=json.loads(completion_path.read_text())
    exact={"format","status","role","intent_path","intent_sha256","attempt_nonce","identity_present","identity_path","identity_sha256","worker_receipt_present","worker_receipt_path","worker_receipt_sha256","exit_type","exit_code","signal","wait_seconds","process_log_path","process_log_sha256","process_log_logical_bytes","child_completed","worker_passed","authorities"}
    authorities={"cache_reuse_authorized":False,"phase_b_preregistration_authorized":False,"training_authorized":False,"folds_authorized":0,"policy_updates":0,"s1_authorized":False,"zero_update_authorized":False,"rl_authorized":False,"submission_authorized":False,"retry_authorized":False}
    if set(completion)!=exact or completion.get("format")!="strict-track2-v485-v482-v169-cache-phase-a-driver-completion-v1" or completion.get("status")!="passed" or completion.get("role")!=role or completion.get("intent_path")!=str((root/"attempt_intent.json").resolve()) or completion.get("intent_sha256")!=sha(root/"attempt_intent.json") or completion.get("attempt_nonce")!=intent["attempt_nonce"] or completion.get("identity_present") is not True or completion.get("identity_path")!=str((base/"identity.json").resolve()) or completion.get("identity_sha256")!=sha(base/"identity.json") or completion.get("worker_receipt_present") is not True or completion.get("worker_receipt_path")!=str((base/"receipt.json").resolve()) or completion.get("worker_receipt_sha256")!=sha(base/"receipt.json") or completion.get("exit_type")!="clean_exit" or completion.get("exit_code")!=0 or completion.get("signal") is not None or completion.get("process_log_path")!=str((base/"process.log").resolve()) or completion.get("process_log_sha256")!=sha(base/"process.log") or completion.get("process_log_logical_bytes")!=(base/"process.log").stat().st_size or completion.get("child_completed") is not True or completion.get("worker_passed") is not True or not isinstance(completion.get("wait_seconds"),float) or not math.isfinite(completion["wait_seconds"]) or completion["wait_seconds"]<0 or completion.get("authorities")!=authorities or worker_receipt.get("passed") is not True:return False
    return True
def final_source_closure(worker_receipt):
    ancestry=worker_receipt["source_and_ancestry"]
    for key,value in ancestry.items():
        if key=="interpreter":continue
        path=Path(value["path"])
        if path.is_symlink() or not path.is_file() or sha(path)!=value["sha256"]:raise RuntimeError(f"final ancestry drift {key}")
    interpreter=ancestry["interpreter"];resolved=Path(interpreter["resolved_path"])
    if resolved.is_symlink() or not resolved.is_file() or sha(resolved)!=interpreter["resolved_sha256"] or resolved.stat().st_size!=interpreter["resolved_bytes"]:raise RuntimeError("final interpreter")
    formal_path=Path(ancestry["formal"]["path"]);contract_path=Path(ancestry["contract"]["path"]);authority_path=Path(ancestry["authority_receipt"]["path"]);parent_path=Path(ancestry["parent_preregistration"]["path"])
    formal=json.loads(formal_path.read_text());contract=json.loads(contract_path.read_text());authority=json.loads(authority_path.read_text());parent=json.loads(parent_path.read_text())
    guards=formal.get("authorization",{})
    if formal.get("format")!="strict-track2-v485-v482-v169-cache-determinism-qualification-preregistration-v1" or guards.get("phase_a_cache_qualification_authorized") is not False or guards.get("attempts_authorized")!=0 or guards.get("training_authorized") is not False or guards.get("policy_updates")!=0:raise RuntimeError("final formal")
    if contract.get("format")!="strict-track2-v485-v482-v169-cache-determinism-qualification-design-contract-v7" or ancestry["contract"]["sha256"]!=CONTRACT_SHA:raise RuntimeError("final contract")
    if authority.get("passed") is not True or authority.get("phase_a_cache_qualification_authorized") is not True or authority.get("training_authorized") is not False or authority.get("preregistration_sha256")!=ancestry["formal"]["sha256"] or authority.get("contract_sha256")!=CONTRACT_SHA:raise RuntimeError("final authority")
    exact_roles=contract["phase_a_output_and_receipt_schema"]["source_closure_contract"]["exact_roles"];records=[]
    if set(formal.get("execution_sources",{}))!=set(exact_roles):raise RuntimeError("final exact seven roles")
    for role in exact_roles:
        frozen=formal["execution_sources"][role];path=Path(frozen["path"])
        if path.is_symlink() or not path.is_file() or sha(path)!=frozen["sha256"] or path.stat().st_size!=frozen["logical_bytes"]:raise RuntimeError(f"final source {role}")
        records.append({"role":role,**frozen})
    digest=canonical_sha(records)
    if records!=formal.get("execution_source_records") or digest!=formal.get("execution_sources_digest_sha256"):raise RuntimeError("final source digest")
    runtime_path=Path(parent["source"]["runtime_path"])
    if runtime_path.is_symlink() or not runtime_path.is_file() or sha(runtime_path)!=parent["source"]["runtime_sha256"] or ancestry["runtime"]!={"path":str(runtime_path.resolve()),"sha256":parent["source"]["runtime_sha256"]}:raise RuntimeError("final parent runtime")
    return {"formal":{"path":str(formal_path.resolve()),"sha256":sha(formal_path)},"contract":{"path":str(contract_path.resolve()),"sha256":sha(contract_path)},"authority_receipt":{"path":str(authority_path.resolve()),"sha256":sha(authority_path)},"parent_preregistration":{"path":str(parent_path.resolve()),"sha256":sha(parent_path)},"parent_runtime":{"path":str(runtime_path.resolve()),"sha256":sha(runtime_path)},"actual_execution_source_records":records,"actual_execution_sources_digest_sha256":digest}
def tree_evidence(root,exclude=()):
    root=Path(root);excluded={Path(item).resolve() for item in exclude};inventory=[]
    for path in sorted(root.rglob("*"),key=lambda value:value.relative_to(root).as_posix()):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):raise RuntimeError("tree nonregular")
        if path.is_file() and path.resolve() not in excluded:inventory.append([path.relative_to(root).as_posix(),sha(path),path.stat().st_size])
    lines="".join(f"{digest}  {relative}\n" for relative,digest,_ in inventory).encode()
    return {"file_count":len(inventory),"logical_file_bytes":sum(row[2] for row in inventory),"sha256sum_lines_digest_sha256":hashlib.sha256(lines).hexdigest(),"canonical_json_digest_sha256":canonical_sha(inventory),"inventory":inventory}
def tree_fields(prefix,evidence):return {f"{prefix}_file_count":evidence["file_count"],f"{prefix}_logical_file_bytes":evidence["logical_file_bytes"],f"{prefix}_sha256sum_lines_digest_sha256":evidence["sha256sum_lines_digest_sha256"],f"{prefix}_canonical_json_digest_sha256":evidence["canonical_json_digest_sha256"],f"{prefix}_inventory":evidence["inventory"]}
def evidence_from_inventory(inventory):
    if inventory!=sorted(inventory,key=lambda row:row[0]) or len({row[0] for row in inventory})!=len(inventory):raise RuntimeError("tree inventory order")
    lines="".join(f"{digest}  {relative}\n" for relative,digest,_ in inventory).encode()
    return {"file_count":len(inventory),"logical_file_bytes":sum(row[2] for row in inventory),"sha256sum_lines_digest_sha256":hashlib.sha256(lines).hexdigest(),"canonical_json_digest_sha256":canonical_sha(inventory),"inventory":inventory}
def validate_tree_fields(payload,prefix):
    evidence=evidence_from_inventory(payload[f"{prefix}_inventory"])
    if any(payload[f"{prefix}_{key}"]!=evidence[key] for key in ("file_count","logical_file_bytes","sha256sum_lines_digest_sha256","canonical_json_digest_sha256")):raise RuntimeError(f"stored tree fields {prefix}")
    return evidence
def stable_seed(episode,start):
    value=f"v482-v169-cache/seed1624/episode{int(episode)}/start{int(start)}".encode();return int.from_bytes(hashlib.sha256(value).digest()[:8],"little")%(2**31)


def rebuild_requests(parent):
    selection_path=Path(parent["dataset"]["selection"]["path"])
    if sha(selection_path)!=parent["dataset"]["selection"]["sha256"]:raise RuntimeError("audit selection drift")
    selection=json.loads(selection_path.read_text());result=[]
    if len(selection.get("contexts",[]))!=200 or len(parent.get("temporal_contexts",[]))!=200:raise RuntimeError("audit exact200")
    for index,(spec,frozen) in enumerate(zip(selection["contexts"],parent["temporal_contexts"])):
        if int(spec["episode"])!=int(frozen["episode"]) or int(spec["start"])!=int(frozen["start"]):raise RuntimeError("selection/row pairing")
        row=Path(frozen["row_npz_path"]);receipt=Path(frozen["row_receipt_path"])
        if sha(row)!=frozen["row_npz_sha256"] or sha(receipt)!=frozen["row_receipt_sha256"]:raise RuntimeError("audit row drift")
        with np.load(row,allow_pickle=False) as data:
            history=np.asarray(data["history_actions"]);future=np.asarray(data["future_actions"]);stored=np.asarray(data["pre_future_context_rgb"])
        if history.shape!=(4,14) or history.dtype!=np.float32 or not history.flags.c_contiguous or future.shape!=(6,8,14) or future.dtype!=np.float32 or not future.flags.c_contiguous or stored.shape!=(6,8,256,256,3) or stored.dtype!=np.uint8 or not stored.flags.c_contiguous:raise RuntimeError("audit source schema")
        context=np.repeat(np.ascontiguousarray(stored[0,0])[None],5,axis=0)
        if not np.array_equal(stored,np.broadcast_to(stored[0,0],stored.shape)) or array_raw_sha(context)!=frozen["constructed_repeat5_context_sha256"] or array_raw_sha(history)!=spec["history_action_sha256"]:raise RuntimeError("audit source identity")
        for branch,name in enumerate(BRANCHES):
            branch_future=np.ascontiguousarray(future[branch])
            if array_raw_sha(branch_future)!=spec["branch_action_sha256"][name]:raise RuntimeError("audit future identity")
            seed=stable_seed(spec["episode"],spec["start"]);instruction=str(spec["instruction"]);digest=hashlib.sha256();digest.update(context.view(np.uint8));digest.update(history.view(np.uint8));digest.update(branch_future.view(np.uint8));digest.update(np.asarray([seed],dtype="<i8").view(np.uint8));digest.update(instruction.encode())
            result.append({"sample_id":index*5+branch,"episode":int(spec["episode"]),"start":int(spec["start"]),"branch":branch,"seed":seed,"instruction":instruction,"context_sha256":array_raw_sha(context),"history_sha256":array_raw_sha(history),"future_sha256":array_raw_sha(branch_future),"seed_sha256":array_raw_sha(np.asarray([seed],dtype="<i8")),"instruction_sha256":hashlib.sha256(instruction.encode()).hexdigest(),"request_sha256":digest.hexdigest()})
    if [x["sample_id"] for x in result]!=list(range(1000)):raise RuntimeError("audit identities")
    return result


def atomic_json(path,payload):
    path=Path(path);tmp=path.with_name(path.name+".tmp")
    if path.exists() or tmp.exists():raise FileExistsError(path)
    with tmp.open("x") as f:json.dump(payload,f,sort_keys=True,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);fd=os.open(str(path.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)


def events(path,role):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
    order=list(range(1000)) if role=="A" else list(reversed(range(1000)))
    strict={"enabled":True,"warn_only":False};compat={"enabled":True,"warn_only":True}
    if len(rows)!=1000 or [x.get("ordinal") for x in rows]!=list(range(1000)) or [x.get("sample_id") for x in rows]!=order:raise RuntimeError("event identity/order")
    expected_keys={"ordinal","sample_id","episode","start","branch","seed","request_sha256","output_sha256","context_sha256","history_sha256","future_sha256","seed_sha256","instruction_sha256","warning_category","warning_full_message","deterministic_before","warn_only_before","deterministic_inside","warn_only_inside","deterministic_after","warn_only_after","cpu_rng_before_sha256","cpu_rng_after_sha256","cuda_rng_before_sha256_by_device","cuda_rng_after_sha256_by_device","rng_unchanged"}
    for row in rows:
        if set(row)!=expected_keys:raise RuntimeError("event exact schema")
        if (row["deterministic_before"],row["warn_only_before"],row["deterministic_inside"],row["warn_only_inside"],row["deterministic_after"],row["warn_only_after"])!=(True,False,True,True,True,False) or row["warning_category"]!=ALLOWED_CATEGORY or row["warning_full_message"]!=ALLOWED_MESSAGE or row["rng_unchanged"] is not True or row["cpu_rng_before_sha256"]!=row["cpu_rng_after_sha256"] or row["cuda_rng_before_sha256_by_device"]!=row["cuda_rng_after_sha256_by_device"]:raise RuntimeError("event guard/warning/RNG")
        if not isinstance(row["cuda_rng_before_sha256_by_device"],list):raise RuntimeError("RNG schema")
    return rows
def progress(path,role,event_rows):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
    order=list(range(1000)) if role=="A" else list(reversed(range(1000)))
    if len(rows)!=1002 or rows[0].get("event")!="start" or rows[-1]!={"event":"process_complete","role":role,"calls":1000}:raise RuntimeError("progress boundaries")
    for ordinal,(row,event,sample_id) in enumerate(zip(rows[1:-1],event_rows,order)):
        if row!={"event":"call_committed","ordinal":ordinal,"sample_id":sample_id,"request_sha256":event["request_sha256"],"output_sha256":event["output_sha256"]}:raise RuntimeError("progress exact call")
    return rows


def no_symlinks(root):
    for path in Path(root).rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):raise RuntimeError("qualification nonregular")


def snapshot(root,exclude=()):
    root=Path(root);excluded={Path(x).resolve() for x in exclude};rows=[]
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):raise RuntimeError("snapshot nonregular")
        if path.is_file() and path.resolve() not in excluded:rows.append({"relative":path.relative_to(root).as_posix(),"sha256":sha(path),"bytes":path.stat().st_size})
    return rows


def validate_precleanup(root):
    root=Path(root);a=root/"process_a";b=root/"process_b";no_symlinks(root)
    expected_relative={"attempt_intent.json","process_a/identity.json","process_a/cache/scalar_v169_temporal_cache.npz","process_a/cache/manifest.json","process_a/call_events.ndjson","process_a/progress.ndjson","process_a/process.log","process_a/receipt.json","process_a/completion_receipt.json","process_b/identity.json","process_b/baseline_canonical.memmap","process_b/call_events.ndjson","process_b/progress.ndjson","process_b/process.log","process_b/receipt.json","process_b/completion_receipt.json"}
    actual_relative={path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual_relative!=expected_relative or (root/"process_a_failure_receipt.json").exists() or (root/"process_b_failure_receipt.json").exists():raise RuntimeError("precleanup exact tree")
    ar=json.loads((a/"receipt.json").read_text());br=json.loads((b/"receipt.json").read_text());ai=json.loads((a/"identity.json").read_text());bi=json.loads((b/"identity.json").read_text());intent_path=root/"attempt_intent.json";intent=json.loads(intent_path.read_text());intent_sha=sha(intent_path)
    identity_fields={"intent_path","intent_sha256","attempt_nonce","role","expected_identity_relative_path","linux_boot_id","pid","proc_self_stat_starttime_ticks","sys_executable_resolved_path","sys_executable_sha256","worker_source_sha256","identity_schema_version"}
    if set(ai)!=identity_fields or set(bi)!=identity_fields or ai.get("role")!="process_a" or bi.get("role")!="process_b" or ai.get("expected_identity_relative_path")!="process_a/identity.json" or bi.get("expected_identity_relative_path")!="process_b/identity.json" or ar.get("fresh_process_identity")!=ai or br.get("fresh_process_identity")!=bi:raise RuntimeError("fresh identity schema")
    for identity in (ai,bi):
        if identity["intent_path"]!=str(intent_path.resolve()) or identity["intent_sha256"]!=intent_sha or identity["attempt_nonce"]!=intent["attempt_nonce"] or identity["worker_source_sha256"]!=intent["worker_source_sha256"] or identity["identity_schema_version"]!=intent["identity_schema_version"]:raise RuntimeError("identity intent reverse binding")
    process_tuple=("linux_boot_id","pid","proc_self_stat_starttime_ticks","sys_executable_resolved_path","sys_executable_sha256","worker_source_sha256")
    if tuple(ai[key] for key in process_tuple)==tuple(bi[key] for key in process_tuple):raise RuntimeError("fresh identity tuple")
    if ar.get("passed") is not True or br.get("passed") is not True or ar.get("role")!="A" or br.get("role")!="B":raise RuntimeError("fresh process receipts")
    for identity in (ai,bi):
        executable=Path(identity["sys_executable_resolved_path"])
        if not executable.is_file() or executable.is_symlink() or sha(executable)!=identity["sys_executable_sha256"] or identity["linux_boot_id"]!=Path("/proc/sys/kernel/random/boot_id").read_text().strip() or not isinstance(identity["pid"],int) or not str(identity["proc_self_stat_starttime_ticks"]).isdigit():raise RuntimeError("identity executable")
        proc=Path(f"/proc/{identity['pid']}/stat")
        if proc.exists() and proc.read_text().split()[21]==str(identity["proc_self_stat_starttime_ticks"]):raise RuntimeError("worker still live")
    ae=events(a/"call_events.ndjson","A");be=events(b/"call_events.ndjson","B");ap=progress(a/"progress.ndjson","A",ae);bp=progress(b/"progress.ndjson","B",be)
    if ap[0]!={"event":"start","role":"A","identity":ai} or bp[0]!={"event":"start","role":"B","identity":bi}:raise RuntimeError("progress identity start")
    npz=a/"cache/scalar_v169_temporal_cache.npz";manifest_path=a/"cache/manifest.json";memmap_path=b/"baseline_canonical.memmap"
    manifest=json.loads(manifest_path.read_text())
    if manifest.get("cache_npz_sha256")!=sha(npz) or manifest.get("ordered_scalar_requests")!=1000:raise RuntimeError("A manifest")
    if ar.get("call_events_sha256")!=sha(a/"call_events.ndjson") or br.get("call_events_sha256")!=sha(b/"call_events.ndjson") or ar.get("call_events_bytes")!=(a/"call_events.ndjson").stat().st_size or br.get("call_events_bytes")!=(b/"call_events.ndjson").stat().st_size or ar.get("call_events_count")!=1000 or br.get("call_events_count")!=1000 or not (ar.get("calls")==br.get("calls")==1000) or not (ar.get("warning_count")==br.get("warning_count")==1000) or not (ar.get("other_warning_count")==br.get("other_warning_count")==0) or not (ar.get("rng_unchanged_count")==br.get("rng_unchanged_count")==1000):raise RuntimeError("receipt counters")
    if ar.get("source_and_ancestry")!=br.get("source_and_ancestry") or canonical_sha(ar.get("source_and_ancestry"))!=ar.get("source_and_ancestry_sha256") or not (ar.get("source_and_ancestry_sha256")==br.get("source_and_ancestry_sha256")==manifest.get("source_and_ancestry_sha256")):raise RuntimeError("source ancestry")
    ancestry=ar["source_and_ancestry"]
    for key,value in ancestry.items():
        if key=="interpreter":continue
        path=Path(value["path"])
        if not path.is_file() or path.is_symlink() or sha(path)!=value["sha256"]:raise RuntimeError(f"ancestry drift {key}")
    interpreter=ancestry["interpreter"];resolved=Path(interpreter["resolved_path"])
    if not resolved.is_file() or resolved.is_symlink() or sha(resolved)!=interpreter["resolved_sha256"] or resolved.stat().st_size!=interpreter["resolved_bytes"]:raise RuntimeError("interpreter closure")
    formal=json.loads(Path(ancestry["formal"]["path"]).read_text());authority=json.loads(Path(ancestry["authority_receipt"]["path"]).read_text());contract=json.loads(Path(ancestry["contract"]["path"]).read_text())
    guards=formal.get("authorization",{})
    if formal.get("format")!="strict-track2-v485-v482-v169-cache-determinism-qualification-preregistration-v1" or guards.get("phase_a_cache_qualification_authorized") is not False or guards.get("attempts_authorized")!=0 or guards.get("training_authorized") is not False or guards.get("folds_authorized")!=0 or guards.get("policy_updates")!=0:raise RuntimeError("formal boundary")
    if authority.get("passed") is not True or authority.get("phase_a_cache_qualification_authorized") is not True or authority.get("training_authorized") is not False or authority.get("preregistration_sha256")!=ancestry["formal"]["sha256"] or authority.get("contract_sha256")!=CONTRACT_SHA:raise RuntimeError("authority boundary")
    if ancestry["contract"]["sha256"]!=CONTRACT_SHA or formal.get("contract")!={"path":ancestry["contract"]["path"],"sha256":CONTRACT_SHA}:raise RuntimeError("contract closure")
    if set(ai)!=set(formal["fresh_process_identity_contract"]["fields"]) or set(bi)!=set(formal["fresh_process_identity_contract"]["fields"]) or any(key in intent for key in formal["fresh_process_identity_contract"]["intent_future_identity_fields_forbidden"]):raise RuntimeError("formal identity contract")
    if intent.get("roles")!={"process_a":"canonical_0_to_999","process_b":"reverse_999_to_0"} or intent.get("expected_identity_relative_paths")!={"process_a":"process_a/identity.json","process_b":"process_b/identity.json"} or intent.get("qualification_output_root")!=str(root.resolve()) or intent.get("preregistration_sha256")!=ancestry["formal"]["sha256"] or intent.get("contract_sha256")!=CONTRACT_SHA or intent.get("authority_receipt_sha256")!=ancestry["authority_receipt"]["sha256"] or intent.get("execution_source_records")!=formal["execution_source_records"] or intent.get("execution_sources_digest_sha256")!=formal["execution_sources_digest_sha256"]:raise RuntimeError("intent independent closure")
    schema=contract["cache_artifact_and_manifest_contract"]
    if contract.get("format")!="strict-track2-v485-v482-v169-cache-determinism-qualification-design-contract-v7" or set(manifest)!=set(schema["manifest_exact_fields"]) or manifest["npz_key_order"]!=schema["npz_exact_keys"] or manifest["npz_array_schema"]!=schema["npz_exact_array_contract"]:raise RuntimeError("manifest contract")
    payload=tree_evidence(a,exclude=(manifest_path,a/"process.log",a/"completion_receipt.json"))
    for field,key in (("file_count","process_a_payload_tree_file_count"),("logical_file_bytes","process_a_payload_tree_logical_file_bytes"),("sha256sum_lines_digest_sha256","process_a_payload_tree_sha256sum_lines_digest_sha256"),("canonical_json_digest_sha256","process_a_payload_tree_canonical_json_digest_sha256")):
        if manifest[key]!=payload[field]:raise RuntimeError("process A payload tree")
    if manifest["process_a_worker_receipt_path"]!=str((a/"receipt.json").resolve()) or manifest["process_a_worker_receipt_sha256"]!=sha(a/"receipt.json") or manifest["phase_a_preregistration_path"]!=ancestry["formal"]["path"] or manifest["phase_a_preregistration_sha256"]!=ancestry["formal"]["sha256"] or manifest["phase_a_formal_path"]!=ancestry["formal"]["path"] or manifest["phase_a_formal_sha256"]!=ancestry["formal"]["sha256"]:raise RuntimeError("manifest formal/receipt")
    if Path(formal["qualification_output_root"]).resolve()!=root.resolve() or Path(authority["qualification_output_root"]).resolve()!=root.resolve():raise RuntimeError("formal output root")
    expected_sources={"scope_source":"phase_a_cache_scope_helper","driver_source":"phase_a_driver","worker_source":"phase_a_process_worker","auditor_source":"phase_a_independent_auditor"}
    for ancestry_key,formal_key in expected_sources.items():
        source=formal["execution_sources"][formal_key]
        if ancestry[ancestry_key]["path"]!=source["path"] or ancestry[ancestry_key]["sha256"]!=source["sha256"]:raise RuntimeError(f"formal source binding {formal_key}")
    for role,source in formal.get("execution_sources",{}).items():
        path=Path(source["path"])
        if not path.is_file() or path.is_symlink() or sha(path)!=source["sha256"] or path.stat().st_size!=source["logical_bytes"]:raise RuntimeError(f"full formal source closure {role}")
    exact_roles=contract["phase_a_output_and_receipt_schema"]["source_closure_contract"]["exact_roles"]
    if formal.get("execution_source_records")!= [{"role":role,**formal["execution_sources"][role]} for role in exact_roles] or formal.get("execution_sources_digest_sha256")!=canonical_sha(formal["execution_source_records"]):raise RuntimeError("source record digest")
    frozen_interpreter=formal["execution_interpreter"]
    for key in ("resolved_path","resolved_sha256","resolved_bytes","python_version","numpy_version","torch_version"):
        if interpreter.get(key)!=frozen_interpreter.get(key):raise RuntimeError("worker interpreter/formal")
    parent=json.loads(Path(ancestry["parent_preregistration"]["path"]).read_text());runtime_path=Path(parent["source"]["runtime_path"])
    if not runtime_path.is_file() or runtime_path.is_symlink() or sha(runtime_path)!=parent["source"]["runtime_sha256"] or ancestry["runtime"]!={"path":str(runtime_path.resolve()),"sha256":parent["source"]["runtime_sha256"]}:raise RuntimeError("independent parent runtime frozen SHA")
    rebuilt=rebuild_requests(parent);expected={x["sample_id"]:x for x in rebuilt}
    for collection in (ae,be):
        for row in collection:
            target=expected[row["sample_id"]]
            for key in ("sample_id","episode","start","branch","seed","request_sha256","context_sha256","history_sha256","future_sha256","seed_sha256","instruction_sha256"):
                if row.get(key)!=target[key]:raise RuntimeError(f"independent request rebuild {key}")
    expected_request_digest=hashlib.sha256("".join(row["request_sha256"] for row in rebuilt).encode()).hexdigest()
    expected_seed_digest=array_raw_sha(np.asarray([row["seed"] for row in rebuilt],dtype="<i8").reshape(200,5))
    expected_contexts=[rebuilt[index*5]["context_sha256"] for index in range(200)]
    expected_context_digest=hashlib.sha256("".join(expected_contexts).encode()).hexdigest()
    if manifest.get("request_digest_sha256")!=expected_request_digest or manifest.get("seed_digest_sha256")!=expected_seed_digest or manifest.get("context_repeat5_sha256")!=expected_contexts or manifest.get("context_repeat5_digest_sha256")!=expected_context_digest:raise RuntimeError("independent request manifests")
    for role_name,receipt,base,identity,progress_rows in (("process_a",ar,a,ai,ap),("process_b",br,b,bi,bp)):
        if receipt.get("identity_path")!=str((base/"identity.json").resolve()) or receipt.get("identity_sha256")!=sha(base/"identity.json") or receipt.get("progress_path")!=str((base/"progress.ndjson").resolve()) or receipt.get("progress_sha256")!=sha(base/"progress.ndjson") or receipt.get("progress_bytes")!=(base/"progress.ndjson").stat().st_size or receipt.get("progress_count")!=len(progress_rows) or receipt.get("process_log_future_owned_by_driver") is not True or receipt.get("driver_completion_receipt_future_owned") is not True or any(key in receipt for key in ("process_log_path","process_log_sha256","process_log_bytes","exit_code")):raise RuntimeError("identity/progress worker receipt")
        if not validate_completion(root,base,role_name,intent,receipt):raise RuntimeError("driver completion receipt")
    if ar.get("artifact",{}).get("cache_npz_path")!=str(npz.resolve()) or ar.get("artifact",{}).get("cache_npz_sha256")!=sha(npz) or ar.get("artifact",{}).get("cache_npz_bytes")!=npz.stat().st_size or manifest["cache_npz_path"]!=str(npz.resolve()) or manifest["cache_npz_logical_bytes"]!=npz.stat().st_size or br.get("artifact",{}).get("memmap_path")!=str(memmap_path.resolve()) or br.get("artifact",{}).get("memmap_retained") is not True or br.get("artifact",{}).get("cleanup_authorized") is not False:raise RuntimeError("artifact receipts")
    with np.load(npz,allow_pickle=False) as data:
        if data.files != ["baseline","seed","request_sha256","output_sha256","sample_id"]:raise RuntimeError("NPZ key order")
        baseline=np.asarray(data["baseline"]);seed=np.asarray(data["seed"]);requests=np.asarray(data["request_sha256"]);outputs=np.asarray(data["output_sha256"]);sample=np.asarray(data["sample_id"])
        arrays=(baseline,seed,requests,outputs,sample)
        if baseline.shape!=(200,5,8,256,256,3) or baseline.dtype!=np.uint8 or seed.shape!=(200,5) or seed.dtype!=np.dtype("<i8") or requests.shape!=(200,5) or requests.dtype!=np.dtype("<U64") or outputs.shape!=(200,5) or outputs.dtype!=np.dtype("<U64") or sample.shape!=(200,5) or sample.dtype!=np.dtype("<i8") or not all(value.flags.c_contiguous for value in arrays) or not np.array_equal(sample,np.arange(1000,dtype=np.int64).reshape(200,5)):raise RuntimeError("NPZ schema")
        flat=baseline.reshape(BASELINE_SHAPE);raw_a=array_raw_sha(flat)
        if raw_a!=manifest["baseline_raw_array_sha256"]:raise RuntimeError("A raw hash")
        if hashlib.sha256("".join(requests.reshape(-1)).encode()).hexdigest()!=manifest["request_digest_sha256"] or hashlib.sha256("".join(outputs.reshape(-1)).encode()).hexdigest()!=manifest["output_digest_sha256"] or array_raw_sha(seed)!=manifest["seed_digest_sha256"] or array_raw_sha(requests)!=manifest["request_sha256_array_digest_sha256"] or array_raw_sha(outputs)!=manifest["output_sha256_array_digest_sha256"] or array_raw_sha(sample)!=manifest["sample_id_sha256"]:raise RuntimeError("A ordered manifests")
        mmap=np.memmap(memmap_path,dtype=np.uint8,mode="r",shape=BASELINE_SHAPE)
        if memmap_path.stat().st_size!=int(np.prod(BASELINE_SHAPE)) or sha(memmap_path)!=raw_a:raise RuntimeError("B memmap hash")
        for begin in range(0,1000,8):
            if not np.array_equal(flat[begin:begin+8],mmap[begin:begin+8]):raise RuntimeError("A/B bytes")
        del mmap
        by_a={int(x["sample_id"]):x for x in ae};by_b={int(x["sample_id"]):x for x in be}
        for i in range(1000):
            if not (by_a[i]["request_sha256"]==by_b[i]["request_sha256"]==requests.reshape(-1)[i]):raise RuntimeError("request comparison")
            if not (by_a[i]["output_sha256"]==by_b[i]["output_sha256"]==outputs.reshape(-1)[i]) or array_raw_sha(flat[i])!=outputs.reshape(-1)[i]:raise RuntimeError("output comparison")
            if not (int(by_a[i]["seed"])==int(by_b[i]["seed"])==int(seed.reshape(-1)[i])):raise RuntimeError("seed comparison")
        guards=[{"ordinal":x["ordinal"],"sample_id":x["sample_id"],"deterministic_before":x["deterministic_before"],"warn_only_before":x["warn_only_before"],"deterministic_inside":x["deterministic_inside"],"warn_only_inside":x["warn_only_inside"],"deterministic_after":x["deterministic_after"],"warn_only_after":x["warn_only_after"],"warning_category":x["warning_category"],"warning_full_message":x["warning_full_message"]} for x in ae]
        rng=[{"ordinal":x["ordinal"],"sample_id":x["sample_id"],"cpu_rng_before_sha256":x["cpu_rng_before_sha256"],"cpu_rng_after_sha256":x["cpu_rng_after_sha256"],"cuda_rng_before_sha256_by_device":x["cuda_rng_before_sha256_by_device"],"cuda_rng_after_sha256_by_device":x["cuda_rng_after_sha256_by_device"],"rng_unchanged":x["rng_unchanged"]} for x in ae]
        if canonical_sha(guards)!=manifest["guard_event_manifest_sha256"] or canonical_sha(rng)!=manifest["rng_event_manifest_sha256"]:raise RuntimeError("guard/RNG manifests")
        guards_b=[{"ordinal":x["ordinal"],"sample_id":x["sample_id"],"deterministic_before":x["deterministic_before"],"warn_only_before":x["warn_only_before"],"deterministic_inside":x["deterministic_inside"],"warn_only_inside":x["warn_only_inside"],"deterministic_after":x["deterministic_after"],"warn_only_after":x["warn_only_after"],"warning_category":x["warning_category"],"warning_full_message":x["warning_full_message"]} for x in be]
        rng_b=[{"ordinal":x["ordinal"],"sample_id":x["sample_id"],"cpu_rng_before_sha256":x["cpu_rng_before_sha256"],"cpu_rng_after_sha256":x["cpu_rng_after_sha256"],"cuda_rng_before_sha256_by_device":x["cuda_rng_before_sha256_by_device"],"cuda_rng_after_sha256_by_device":x["cuda_rng_after_sha256_by_device"],"rng_unchanged":x["rng_unchanged"]} for x in be]
        if canonical_sha(guards_b)!=br.get("guard_event_manifest_sha256") or canonical_sha(rng_b)!=br.get("rng_event_manifest_sha256"):raise RuntimeError("B guard/RNG manifests")
        for key in ("request_digest_sha256","output_digest_sha256","seed_digest_sha256","request_sha256_array_digest_sha256","output_sha256_array_digest_sha256","sample_id_sha256","context_repeat5_digest_sha256"):
            if ar.get(key)!=manifest.get(key) or br.get(key)!=manifest.get(key):raise RuntimeError(f"A/B receipt manifest {key}")
        for key in ("guard_event_manifest_sha256","rng_event_manifest_sha256"):
            if ar.get(key)!=manifest.get(key):raise RuntimeError(f"A receipt manifest {key}")
    retained=snapshot(root,exclude=(memmap_path,));stage2=tree_evidence(root)
    return {"A_receipt_sha256":sha(a/"receipt.json"),"B_receipt_sha256":sha(b/"receipt.json"),"A_npz_sha256":sha(npz),"A_manifest_sha256":sha(manifest_path),"A_events_sha256":sha(a/"call_events.ndjson"),"B_memmap_path":str(memmap_path.resolve()),"B_memmap_sha256":sha(memmap_path),"B_memmap_bytes":memmap_path.stat().st_size,"B_events_sha256":sha(b/"call_events.ndjson"),"baseline_raw_array_sha256":raw_a,"retained_precleanup_snapshot":retained,"precleanup_input_tree":stage2}


def precleanup(root):
    root=Path(root);audit_path=root/"independent_precleanup_audit.json";cleanup_path=root/"process_b_cleanup_receipt.json"
    if audit_path.exists() or cleanup_path.exists():raise FileExistsError(audit_path)
    evidence=validate_precleanup(root)
    audit={"format":"strict-track2-v485-v169-cache-independent-precleanup-audit-v1","passed":True,"checks":{"immutable_closure":True,"exact_1000_each":True,"fresh_processes":True,"warning_category_message_count":True,"guard_states":True,"rng_states_unchanged":True,"A_NPZ_schema_hashes":True,"B_memmap_schema_hash":True,"identity_sorted_request_output_equal":True,"A_B_all1000_byteexact":True},"evidence":evidence,**tree_fields("precleanup_input_tree",evidence["precleanup_input_tree"]),"training_authorized":False,"cache_reuse_authorized":False,"folds":0,"policy_updates":0,"rl_authorized":False}
    atomic_json(audit_path,audit);audit_sha=sha(audit_path);memmap=Path(evidence["B_memmap_path"]);stage3=tree_evidence(root)
    if stage3["inventory"]!=sorted(evidence["precleanup_input_tree"]["inventory"]+[["independent_precleanup_audit.json",audit_sha,audit_path.stat().st_size]],key=lambda row:row[0]):raise RuntimeError("stage2->stage3 tree")
    if sha(memmap)!=evidence["B_memmap_sha256"] or memmap.stat().st_size!=evidence["B_memmap_bytes"]:raise RuntimeError("B changed before cleanup")
    memmap.unlink();fd=os.open(str(memmap.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
    cleanup={"format":"strict-track2-v485-v169-cache-B-cleanup-receipt-v1","passed":True,"precleanup_audit_path":str(audit_path.resolve()),"precleanup_audit_sha256":audit_sha,"former_memmap_path":str(memmap),"former_memmap_sha256":evidence["B_memmap_sha256"],"former_memmap_bytes":evidence["B_memmap_bytes"],**tree_fields("pre_cleanup_state_tree",stage3),"unlink_succeeded":not memmap.exists(),"parent_directory_fsynced":True,"training_authorized":False,"cache_reuse_authorized":False,"policy_updates":0,"rl_authorized":False}
    atomic_json(cleanup_path,cleanup);print(json.dumps(audit,sort_keys=True));return 0


def final(root):
    root=Path(root);out=root/"independent_final_audit.json"
    if out.exists():raise FileExistsError(out)
    pre_path=root/"independent_precleanup_audit.json";cleanup_path=root/"process_b_cleanup_receipt.json";pre=json.loads(pre_path.read_text());cleanup=json.loads(cleanup_path.read_text())
    memmap=Path(cleanup["former_memmap_path"]);pre_evidence=pre["evidence"]
    if pre.get("passed") is not True or cleanup.get("passed") is not True or cleanup.get("precleanup_audit_sha256")!=sha(pre_path) or cleanup.get("former_memmap_path")!=pre_evidence["B_memmap_path"] or cleanup.get("former_memmap_sha256")!=pre_evidence["B_memmap_sha256"] or cleanup.get("former_memmap_bytes")!=pre_evidence["B_memmap_bytes"] or memmap.exists():raise RuntimeError("cleanup exact former memmap closure")
    validate_tree_fields(pre,"precleanup_input_tree");validate_tree_fields(cleanup,"pre_cleanup_state_tree")
    expected=pre["evidence"]["retained_precleanup_snapshot"];actual=snapshot(root,exclude=(pre_path,cleanup_path,out))
    if actual!=expected:raise RuntimeError("retained byte drift")
    ar_path=root/"process_a/receipt.json";br_path=root/"process_b/receipt.json";npz=root/"process_a/cache/scalar_v169_temporal_cache.npz";manifest_path=root/"process_a/cache/manifest.json"
    if sha(ar_path)!=pre["evidence"]["A_receipt_sha256"] or sha(br_path)!=pre["evidence"]["B_receipt_sha256"] or sha(npz)!=pre["evidence"]["A_npz_sha256"] or sha(manifest_path)!=pre["evidence"]["A_manifest_sha256"]:raise RuntimeError("retained evidence SHA")
    ar=json.loads(ar_path.read_text());br=json.loads(br_path.read_text());intent=json.loads((root/"attempt_intent.json").read_text())
    if not validate_completion(root,root/"process_a","process_a",intent,ar) or not validate_completion(root,root/"process_b","process_b",intent,br):raise RuntimeError("retained completion receipts")
    ae=events(root/"process_a/call_events.ndjson","A");be=events(root/"process_b/call_events.ndjson","B");progress(root/"process_a/progress.ndjson","A",ae);progress(root/"process_b/progress.ndjson","B",be)
    if len(ae)!=1000 or len(be)!=1000 or sha(root/"process_a/call_events.ndjson")!=pre["evidence"]["A_events_sha256"] or sha(root/"process_b/call_events.ndjson")!=pre["evidence"]["B_events_sha256"]:raise RuntimeError("retained events")
    with np.load(npz,allow_pickle=False) as data:
        arrays=[np.asarray(data[name]) for name in ("baseline","seed","request_sha256","output_sha256","sample_id")]
        if data.files != ["baseline","seed","request_sha256","output_sha256","sample_id"] or not all(array.flags.c_contiguous for array in arrays) or arrays[0].dtype!=np.uint8 or arrays[1].dtype!=np.dtype("<i8") or arrays[2].dtype!=np.dtype("<U64") or arrays[3].dtype!=np.dtype("<U64") or arrays[4].dtype!=np.dtype("<i8") or array_raw_sha(arrays[0])!=pre["evidence"]["baseline_raw_array_sha256"]:raise RuntimeError("retained NPZ schema/order")
    closure=final_source_closure(ar);terminal_base=tree_evidence(root)
    stage3_inventory=cleanup["pre_cleanup_state_tree_inventory"];expected_terminal_base=[row for row in stage3_inventory if row[0]!="process_b/baseline_canonical.memmap"]+[["process_b_cleanup_receipt.json",sha(cleanup_path),cleanup_path.stat().st_size]]
    if terminal_base["inventory"]!=sorted(expected_terminal_base,key=lambda row:row[0]):raise RuntimeError("stage3->stage4 tree")
    receipt={"format":"strict-track2-v485-v169-cache-independent-final-audit-v1","passed":True,"checks":{"precleanup_audit":True,"cleanup_receipt":True,"B_memmap_absent":True,"retained_outputs_byteexact":True,"retained_receipts_events_NPZ_revalidated":True,"final_formal_contract_authority_exact7_parent_runtime_rehashed":True,"A_cache_only":True},"precleanup_audit_sha256":sha(pre_path),"cleanup_receipt_sha256":sha(cleanup_path),"final_immutable_closure":closure,"actual_execution_source_records":closure["actual_execution_source_records"],"actual_execution_sources_digest_sha256":closure["actual_execution_sources_digest_sha256"],**tree_fields("terminal_base_tree",terminal_base),"qualified_cache_reconciliation_review_authorized":True,"training_authorized":False,"cache_reuse_authorized":False,"folds":0,"policy_updates":0,"rl_authorized":False}
    atomic_json(out,receipt);print(json.dumps(receipt,sort_keys=True));return 0


def synthetic():
    a=np.arange(96,dtype=np.uint8).reshape(4,2,4,3);b=np.asarray(a).copy();return bool(array_raw_sha(a)==array_raw_sha(b) and all(np.array_equal(a[i:i+1],b[i:i+1]) for i in range(4)))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--preregistration",type=Path);p.add_argument("--preregistration-sha");p.add_argument("--authority-receipt",type=Path);p.add_argument("--authority-receipt-sha");p.add_argument("--stage",choices=("precleanup","final"));p.add_argument("--synthetic-self-test",action="store_true");args=p.parse_args()
    if args.synthetic_self_test: print(json.dumps({"passed":synthetic()}));raise SystemExit(0 if synthetic() else 3)
    if args.preregistration is None or args.preregistration_sha is None or args.authority_receipt is None or args.authority_receipt_sha is None or args.stage is None:raise SystemExit(78)
    if sha(args.preregistration)!=args.preregistration_sha or sha(args.authority_receipt)!=args.authority_receipt_sha:raise RuntimeError("auditor formal/authority drift")
    pre=json.loads(args.preregistration.read_text());authority=json.loads(args.authority_receipt.read_text());root=Path(pre["qualification_output_root"])
    if not root.is_absolute() or root.resolve()!=root or Path(authority.get("qualification_output_root","")).resolve()!=root:raise RuntimeError("auditor output root")
    raise SystemExit(precleanup(root) if args.stage=="precleanup" else final(root))
