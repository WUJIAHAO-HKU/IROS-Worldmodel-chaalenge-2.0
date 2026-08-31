#!/usr/bin/env python3
"""Fresh-process v522 A/B worker with per-call RNG isolation; never trains."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


PARENT_PRE_SHA = "426d4f7a774520458f6af41d20660cf31202791f99603dacec114f771da6a881"
PARENT_FAILURE_SHA = "dbe3e10d5e7d47d9690c42ebf7f8658fcb10b5aafcd9f6bd19cbf4accb381106"
CONTRACT_SHA = "8c4a583aa19de6927cc1ef787920b65317893e27ad01371d0b95b622f260cf64"
BRANCHES = ("factual", "no_transport", "scale_0p4", "scale_1p25", "reverse_direction_0p4")
BASELINE_SHAPE = (1000, 8, 256, 256, 3)
FAILURE_CONTEXT=None
PROGRESS={"current_call":None,"completed_calls":0,"last_warning":None,"last_rng":None}
AUTHORIZATION_KEYS={"phase_a_cache_qualification_launcher_authorized","launcher_invocations_authorized","launcher_invocations_consumed","nested_phase_a_driver_invocations_authorized","nested_phase_a_driver_only_via_launcher","direct_phase_a_driver_authorized","phase_a_worker_invocations_authorized","rng_proxy_required_for_every_runtime_delegate","runtime_delegate_calls_authorized","runtime_delegate_calls_consumed","worker_environment_exact","service_environment_overrides_authorized","retry_authorized","cache_reuse_authorized","training_authorized","reward_read_authorized","dev_hidden_final_outcome_read_authorized","submission_authorized"}
EXPECTED_WORKER_ENV={"PYTHONPATH":"/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline","CUBLAS_WORKSPACE_CONFIG":":4096:8"}
PER_CALL_RNG_CONTRACT={"format":"strict-track2-v520-per-call-rng-isolation-evidence-v1","expected_branches":["A","B"],"calls_per_branch":1000,"total_runtime_delegate_calls":2000,"required_rng_stage_keys":["entry_external","inside_before_delegate","internal_after_delegate","exit_restored"],"delegate_invocations_started_per_call":1,"delegate_invocations_completed_per_successful_call":1,"fork_rng_devices_exact_all_cuda_indices":True,"python_all_four_stages_equal":True,"numpy_all_four_stages_equal":True,"torch_cpu_exit_restored":True,"torch_cuda_exit_restored":True,"torch_internal_change_allowed":True,"raw_warning_order_preserved":True,"output_schema_recorded_per_call":True,"canonical_event_digest_recorded_per_call":True,"sample0_evidence_is_ancestry_not_runtime_substitute":True}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""): digest.update(block)
    return digest.hexdigest()


def raw_sha(value): return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()
def canonical_sha(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def tree_evidence(root,exclude=()):
    root=Path(root);excluded={Path(item).resolve() for item in exclude};inventory=[]
    for path in sorted(root.rglob("*"),key=lambda value:value.relative_to(root).as_posix()):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):raise RuntimeError("payload tree nonregular")
        if path.is_file() and path.resolve() not in excluded:inventory.append([path.relative_to(root).as_posix(),sha(path),path.stat().st_size])
    lines="".join(f"{digest}  {relative}\n" for relative,digest,_ in inventory).encode()
    return {"file_count":len(inventory),"logical_file_bytes":sum(row[2] for row in inventory),"sha256sum_lines_digest_sha256":hashlib.sha256(lines).hexdigest(),"canonical_json_digest_sha256":canonical_sha(inventory),"inventory":inventory}
def stable_seed(episode, start):
    value = f"v482-v169-cache/seed1624/episode{int(episode)}/start{int(start)}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**31)


def atomic_json(path, payload):
    path=Path(path);tmp=path.with_name(path.name+".tmp")
    if path.exists() or tmp.exists(): raise FileExistsError(path)
    with tmp.open("x") as stream:
        json.dump(payload,stream,sort_keys=True,indent=2);stream.write("\n");stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)
    fd=os.open(str(path.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)


def append_event(stream,row):
    stream.write(json.dumps(row,sort_keys=True,separators=(",", ":"))+"\n");stream.flush();os.fsync(stream.fileno())
def durable_stdout_event(row):
    print(json.dumps(row,sort_keys=True,separators=(",", ":")),flush=True);os.fsync(sys.stdout.fileno())


def load_module_exact(path,name,want_sha):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise RuntimeError(f"module nonregular {name}")
    source=path.read_bytes()
    if hashlib.sha256(source).hexdigest()!=want_sha:raise RuntimeError(f"module drift {name}")
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);exec(compile(source,str(path),"exec"),module.__dict__);return module

def validate_v520_authorization(authority):
    authorization=authority.get("authorization",{})
    if set(authorization)!=AUTHORIZATION_KEYS:raise RuntimeError("v520 authorization exact keys")
    exact={"phase_a_cache_qualification_launcher_authorized":True,"launcher_invocations_authorized":1,"launcher_invocations_consumed":0,"nested_phase_a_driver_invocations_authorized":1,"nested_phase_a_driver_only_via_launcher":True,"direct_phase_a_driver_authorized":False,"phase_a_worker_invocations_authorized":2,"rng_proxy_required_for_every_runtime_delegate":True,"runtime_delegate_calls_authorized":2000,"runtime_delegate_calls_consumed":0,"worker_environment_exact":EXPECTED_WORKER_ENV,"service_environment_overrides_authorized":False,"retry_authorized":False,"cache_reuse_authorized":False,"training_authorized":False,"reward_read_authorized":False,"dev_hidden_final_outcome_read_authorized":False,"submission_authorized":False}
    if authorization!=exact:raise RuntimeError("v520 authorization values")
    if authority.get("per_call_rng_evidence_contract")!=PER_CALL_RNG_CONTRACT:raise RuntimeError("v520 per-call RNG contract")

def freeze_worker_failure(error):
    try:print(json.dumps({"event":"worker_failure","error_type":type(error).__name__,"error":str(error),"progress":PROGRESS},sort_keys=True),file=sys.stderr,flush=True)
    except Exception:pass
    closed={}
    if FAILURE_CONTEXT is not None:
        for key in ("events_stream","progress_stream"):
            stream=FAILURE_CONTEXT.get(key)
            if stream is not None and not stream.closed:
                try:stream.flush();os.fsync(stream.fileno());stream.close()
                except Exception:pass
            closed[key]=stream is None or stream.closed
    return closed

def synthetic_failure_self_test():
    global FAILURE_CONTEXT
    with tempfile.TemporaryDirectory(prefix="v485-worker-failure-") as directory:
        partial=Path(directory)/"process_a.partial";partial.mkdir();identity={"role":"process_a"};atomic_json(partial/"identity.json",identity)
        progress_stream=(partial/"progress.ndjson").open("x",encoding="utf-8");events_stream=(partial/"call_events.ndjson").open("x",encoding="utf-8")
        append_event(progress_stream,{"event":"start"});append_event(events_stream,{"event":"synthetic_after_identity"})
        FAILURE_CONTEXT={"qualification_root":partial.parent,"role_name":"process_a","partial":partial,"identity":identity,"progress_stream":progress_stream,"events_stream":events_stream}
        closed=freeze_worker_failure(RuntimeError("synthetic identity-after failure"))
        passed=closed=={"events_stream":True,"progress_stream":True} and (partial/"identity.json").is_file() and (partial/"progress.ndjson").stat().st_size>0 and (partial/"call_events.ndjson").stat().st_size>0
        FAILURE_CONTEXT=None
        return passed

def bootstrap_scope(argv=None):
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument("--scope-source",type=Path);parser.add_argument("--scope-sha");parser.add_argument("--preregistration",type=Path);parser.add_argument("--preregistration-sha")
    known,_=parser.parse_known_args(argv)
    if known.scope_source is None or known.scope_sha is None or known.preregistration is None or known.preregistration_sha is None:raise RuntimeError("bootstrap arguments")
    scope_path=known.scope_source
    if scope_path.is_symlink() or not scope_path.is_file() or sha(scope_path)!=known.scope_sha:raise RuntimeError("bootstrap scope source")
    if known.preregistration.is_symlink() or not known.preregistration.is_file() or sha(known.preregistration)!=known.preregistration_sha:raise RuntimeError("bootstrap formal")
    formal=json.loads(known.preregistration.read_text());frozen=formal.get("execution_sources",{}).get("cache_scope_helper",{})
    if formal.get("format")!="strict-track2-v485-v482-v169-cache-determinism-qualification-preregistration-v1" or frozen!={"path":str(scope_path.resolve()),"sha256":known.scope_sha,"logical_bytes":scope_path.stat().st_size}:raise RuntimeError("bootstrap formal scope binding")
    root=Path(formal["qualification_output_root"]);intent_path=root/"attempt_intent.json"
    if intent_path.is_symlink() or not intent_path.is_file():raise RuntimeError("bootstrap intent")
    intent=json.loads(intent_path.read_text())
    if intent.get("preregistration_sha256")!=known.preregistration_sha or intent.get("scope_source_sha256")!=known.scope_sha or intent.get("execution_source_records")!=formal.get("execution_source_records") or intent.get("execution_sources_digest_sha256")!=formal.get("execution_sources_digest_sha256"):raise RuntimeError("bootstrap intent scope binding")
    return known

def synthetic_bootstrap_self_test():
    with tempfile.TemporaryDirectory(prefix="v485-worker-bootstrap-") as directory:
        root=Path(directory);scope_path=root/"scope.py";scope_path.write_text("VALUE=1\n");scope_sha=sha(scope_path);out=root/"qualification";out.mkdir()
        record={"role":"cache_scope_helper","path":str(scope_path.resolve()),"sha256":scope_sha,"logical_bytes":scope_path.stat().st_size};records=[record];formal={"format":"strict-track2-v485-v482-v169-cache-determinism-qualification-preregistration-v1","qualification_output_root":str(out.resolve()),"execution_sources":{"cache_scope_helper":{key:value for key,value in record.items() if key!="role"}},"execution_source_records":records,"execution_sources_digest_sha256":canonical_sha(records)}
        formal_path=root/"formal.json";atomic_json(formal_path,formal);formal_sha=sha(formal_path);atomic_json(out/"attempt_intent.json",{"preregistration_sha256":formal_sha,"scope_source_sha256":scope_sha,"execution_source_records":records,"execution_sources_digest_sha256":canonical_sha(records)})
        argv=["--scope-source",str(scope_path),"--scope-sha",scope_sha,"--preregistration",str(formal_path),"--preregistration-sha",formal_sha]
        clean=bootstrap_scope(argv).scope_sha==scope_sha;scope_path.write_text("VALUE=2\n");tamper_rejected=False
        try:bootstrap_scope(argv)
        except RuntimeError:tamper_rejected=True
        return clean and tamper_rejected


def request_rows(parent):
    selection_path=Path(parent["dataset"]["selection"]["path"])
    if sha(selection_path)!=parent["dataset"]["selection"]["sha256"]: raise RuntimeError("selection drift")
    selection=json.loads(selection_path.read_text())
    if len(selection.get("contexts",[]))!=200 or len(parent.get("temporal_contexts",[]))!=200: raise RuntimeError("exact200")
    requests=[];context_hashes=[]
    for index,(spec,frozen) in enumerate(zip(selection["contexts"],parent["temporal_contexts"])):
        row=Path(frozen["row_npz_path"]);receipt=Path(frozen["row_receipt_path"])
        if sha(row)!=frozen["row_npz_sha256"] or sha(receipt)!=frozen["row_receipt_sha256"]: raise RuntimeError("row drift")
        with np.load(row,allow_pickle=False) as data:
            history=np.asarray(data["history_actions"]);future=np.asarray(data["future_actions"])
            stored=np.asarray(data["pre_future_context_rgb"])
        if history.shape!=(4,14) or history.dtype!=np.float32 or not history.flags.c_contiguous or future.shape!=(6,8,14) or future.dtype!=np.float32 or not future.flags.c_contiguous or stored.shape!=(6,8,256,256,3) or stored.dtype!=np.uint8 or not stored.flags.c_contiguous: raise RuntimeError("row schema")
        context=np.repeat(np.ascontiguousarray(stored[0,0])[None],5,axis=0)
        if not np.array_equal(stored,np.broadcast_to(stored[0,0],stored.shape)) or raw_sha(context)!=frozen["constructed_repeat5_context_sha256"] or raw_sha(history)!=spec["history_action_sha256"]: raise RuntimeError("context/history drift")
        context_hashes.append(raw_sha(context))
        for branch,name in enumerate(BRANCHES):
            branch_future=np.ascontiguousarray(future[branch])
            if raw_sha(branch_future)!=spec["branch_action_sha256"][name]: raise RuntimeError("future drift")
            sample_id=index*5+branch;seed=stable_seed(spec["episode"],spec["start"]);instruction=str(spec["instruction"])
            digest=hashlib.sha256();digest.update(context.view(np.uint8));digest.update(history.view(np.uint8));digest.update(branch_future.view(np.uint8));digest.update(np.asarray([seed],dtype="<i8").view(np.uint8));digest.update(instruction.encode())
            requests.append({"sample_id":sample_id,"selection_order":index,"episode":int(spec["episode"]),"start":int(spec["start"]),"branch":branch,"branch_name":name,"seed":seed,"instruction":instruction,"context_sha256":raw_sha(context),"history_sha256":raw_sha(history),"future_sha256":raw_sha(branch_future),"seed_sha256":raw_sha(np.asarray([seed],dtype="<i8")),"instruction_sha256":hashlib.sha256(instruction.encode()).hexdigest(),"request_sha256":digest.hexdigest(),"context":context,"history":history,"future":branch_future})
    if [x["sample_id"] for x in requests]!=list(range(1000)): raise RuntimeError("canonical identities")
    return requests,context_hashes


def main():
    p=argparse.ArgumentParser()
    for name in ("preregistration","parent-preregistration","parent-failure","contract","authority-receipt","scope-source","rng-proxy-source","driver-source","auditor-source"): p.add_argument(f"--{name}",type=Path,required=True)
    for name in ("preregistration-sha","contract-sha","authority-receipt-sha","scope-sha","rng-proxy-sha","driver-sha","worker-sha","auditor-sha"): p.add_argument(f"--{name}",required=True)
    p.add_argument("--role",choices=("A","B"),required=True);args=p.parse_args()
    for value in (args.preregistration_sha,args.contract_sha,args.authority_receipt_sha,args.scope_sha,args.rng_proxy_sha,args.driver_sha,args.worker_sha,args.auditor_sha):
        if len(value)!=64 or value.startswith("PENDING"): raise RuntimeError("unfrozen closure")
    if sha(args.preregistration)!=args.preregistration_sha or sha(args.contract)!=args.contract_sha or sha(args.authority_receipt)!=args.authority_receipt_sha or sha(args.scope_source)!=args.scope_sha or sha(args.rng_proxy_source)!=args.rng_proxy_sha or sha(args.driver_source)!=args.driver_sha or sha(args.auditor_source)!=args.auditor_sha or sha(Path(__file__))!=args.worker_sha: raise RuntimeError("source/formal drift")
    formal=json.loads(args.preregistration.read_text());contract=json.loads(args.contract.read_text())
    authority=json.loads(args.authority_receipt.read_text())
    validate_v520_authorization(authority)
    if formal.get("format")!="strict-track2-v485-v482-v169-cache-determinism-qualification-preregistration-v1" or formal.get("authorization",{}).get("phase_a_cache_qualification_authorized") is not False or formal.get("authorization",{}).get("attempts_authorized")!=0 or formal.get("authorization",{}).get("training_authorized") is not False: raise RuntimeError("nonauthorizing formal")
    qualification_root=Path(formal["qualification_output_root"])
    if not qualification_root.is_absolute() or qualification_root.resolve()!=qualification_root:raise RuntimeError("qualification root must be canonical absolute")
    if authority.get("passed") is not True or authority.get("phase_a_cache_qualification_authorized") is not True or authority.get("training_authorized") is not False or authority.get("preregistration_sha256")!=args.preregistration_sha or authority.get("contract_sha256")!=args.contract_sha or Path(authority.get("qualification_output_root","")).resolve()!=qualification_root.resolve():raise RuntimeError("poststatic authority")
    expected_sources={"cache_scope_helper":(args.scope_source,args.scope_sha),"phase_a_process_worker":(Path(__file__),args.worker_sha),"phase_a_rng_isolation_proxy":(args.rng_proxy_source,args.rng_proxy_sha),"phase_a_driver":(args.driver_source,args.driver_sha),"phase_a_independent_auditor":(args.auditor_source,args.auditor_sha)}
    for role,(path,digest) in expected_sources.items():
        frozen=formal.get("execution_sources",{}).get(role,{})
        if frozen.get("path")!=str(path.resolve()) or frozen.get("sha256")!=digest or frozen.get("logical_bytes")!=path.stat().st_size:raise RuntimeError(f"formal source {role}")
    if args.contract_sha!=CONTRACT_SHA or contract.get("format")!="strict-track2-v485-v482-v169-cache-determinism-qualification-design-contract-v7": raise RuntimeError("contract")
    if sha(args.parent_preregistration)!=PARENT_PRE_SHA or sha(args.parent_failure)!=PARENT_FAILURE_SHA: raise RuntimeError("failed parent drift")
    if not qualification_root.is_dir() or qualification_root.is_symlink() or qualification_root.parent.is_symlink():raise RuntimeError("qualification root not initialized by driver")
    role_name="process_a" if args.role=="A" else "process_b";root=qualification_root/role_name;partial=root.with_name(root.name+".partial")
    failure_path=qualification_root/"failure_receipt.json"
    if root.exists() or failure_path.exists() or not partial.is_dir() or partial.is_symlink():raise FileExistsError(root)
    initial_entries=list(partial.iterdir())
    if len(initial_entries)!=1 or initial_entries[0].name!="process.log" or not initial_entries[0].is_file() or initial_entries[0].is_symlink():raise RuntimeError("driver-owned process log precreation")
    intent_path=qualification_root/"attempt_intent.json"
    if not intent_path.is_file() or intent_path.is_symlink():raise RuntimeError("committed intent absent")
    intent_sha=sha(intent_path);intent=json.loads(intent_path.read_text());expected_relative=f"{role_name}/identity.json";resolved=Path(sys.executable).resolve(strict=True)
    if intent.get("format")!="strict-track2-v522-v521-rng-isolated-cache-qualification-attempt-intent-v1" or intent.get("preregistration_sha256")!=args.preregistration_sha or intent.get("contract_sha256")!=CONTRACT_SHA or intent.get("authority_receipt_sha256")!=args.authority_receipt_sha or intent.get("qualification_output_root")!=str(qualification_root.resolve()) or intent.get("roles",{}).get(role_name)!=("canonical_0_to_999" if args.role=="A" else "reverse_999_to_0") or intent.get("expected_identity_relative_paths",{}).get(role_name)!=expected_relative or not isinstance(intent.get("attempt_nonce"),str) or len(intent["attempt_nonce"])!=64 or any(character not in "0123456789abcdef" for character in intent["attempt_nonce"]) or intent.get("sys_executable_resolved_path")!=str(resolved) or intent.get("sys_executable_sha256")!=sha(resolved) or intent.get("worker_source_sha256")!=args.worker_sha or intent.get("rng_proxy_source")!={"path":str(args.rng_proxy_source.resolve()),"sha256":args.rng_proxy_sha,"logical_bytes":args.rng_proxy_source.stat().st_size} or intent.get("identity_schema_version")!="strict-track2-v485-v169-cache-worker-identity-v2" or intent.get("execution_source_records")!=formal["execution_source_records"] or intent.get("execution_sources_digest_sha256")!=formal["execution_sources_digest_sha256"]:raise RuntimeError("attempt intent reverse binding")
    if any(key in intent for key in formal["fresh_process_identity_contract"]["intent_future_identity_fields_forbidden"]):raise RuntimeError("future identity leaked into intent")
    executable=Path(sys.executable)
    identity={"intent_path":str(intent_path.resolve()),"intent_sha256":intent_sha,"attempt_nonce":intent["attempt_nonce"],"role":role_name,"expected_identity_relative_path":expected_relative,"linux_boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),"pid":os.getpid(),"proc_self_stat_starttime_ticks":Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],"sys_executable_resolved_path":str(resolved),"sys_executable_sha256":sha(resolved),"worker_source_sha256":args.worker_sha,"identity_schema_version":"strict-track2-v485-v169-cache-worker-identity-v2"}
    if set(identity)!=set(formal["fresh_process_identity_contract"]["fields"]):raise RuntimeError("identity exact fields")
    identity_path=partial/"identity.json";atomic_json(identity_path,identity)
    print(json.dumps({"event":"worker_start","role":args.role,"identity":identity},sort_keys=True),flush=True)
    global FAILURE_CONTEXT
    FAILURE_CONTEXT={"qualification_root":qualification_root,"role_name":role_name,"partial":partial,"identity":identity,"log_path":partial/"process.log"}
    progress_path=partial/"progress.ndjson";progress_stream=progress_path.open("x",encoding="utf-8");append_event(progress_stream,{"event":"start","role":args.role,"identity":identity})
    FAILURE_CONTEXT["progress_stream"]=progress_stream
    parent=json.loads(args.parent_preregistration.read_text())
    runtime_path=Path(parent["source"]["runtime_path"])
    if not runtime_path.is_file() or runtime_path.is_symlink() or sha(runtime_path)!=parent["source"]["runtime_sha256"]:raise RuntimeError("parent runtime frozen SHA")
    runtime_module=load_module_exact(runtime_path,"v485_parent_runtime",parent["source"]["runtime_sha256"])
    rng_proxy_module=load_module_exact(args.rng_proxy_source,"v520_rng_isolation_proxy",args.rng_proxy_sha)
    closure_path=Path(parent["v169"]["closure_path"])
    if sha(closure_path)!=parent["v169"]["closure_sha256"] or runtime_module.verify_v169(json.loads(closure_path.read_text()))!=parent["v169"]["closure_digest"]: raise RuntimeError("v169 closure")
    requests,context_hashes=request_rows(parent)
    order=list(range(1000)) if args.role=="A" else list(reversed(range(1000)))
    cache_dir=partial/"cache"
    if args.role=="A": cache_dir.mkdir()
    raw=(cache_dir/"baseline_A_staging.raw") if args.role=="A" else (partial/"baseline_canonical.memmap")
    baseline=np.memmap(raw,dtype=np.uint8,mode="w+",shape=BASELINE_SHAPE)
    seed=np.empty(1000,dtype="<i8");request_hash=np.empty(1000,dtype="<U64");output_hash=np.empty(1000,dtype="<U64");events=[]
    events_path=partial/"call_events.ndjson";events_stream=events_path.open("x",encoding="utf-8")
    FAILURE_CONTEXT["events_stream"]=events_stream
    torch.use_deterministic_algorithms(True,warn_only=False)
    runtime=runtime_module.Track2V169ArmRoutedRuntime(parent["v169"]["release_path"],parent["v169"]["library_path"],"cuda:0")
    for execution_order,sample_id in enumerate(order):
        item=requests[sample_id]
        PROGRESS["current_call"]={"execution_order":execution_order,"sample_id":sample_id}
        proxy=rng_proxy_module.RngIsolationRuntimeProxy(runtime,item["seed"],torch,np,random)
        durable_stdout_event({"event":"rng_proxy_call_started","ordinal":execution_order,"sample_id":sample_id,"seed":item["seed"],"request_sha256":item["request_sha256"],"started":1,"completed":0})
        try:
            output,event=scope.predict_one(proxy,item["context"],item["history"],item["future"],item["seed"],item["instruction"])
        except BaseException:
            isolation=proxy.evidence
            if isolation is not None:
                rng_proxy_module.validate_evidence(isolation,require_completed=isolation.get("delegate_invocations_completed")==1)
                failed_row={"ordinal":execution_order,"sample_id":sample_id,"status":"failed_no_retry","sample_call_started":isolation["delegate_invocations_started"],"sample_call_completed":isolation["delegate_invocations_completed"],"rng_isolation":isolation,"rng_isolation_sha256":canonical_sha(isolation),"raw_warning":None}
                append_event(events_stream,failed_row);PROGRESS.update({"current_call":failed_row,"last_rng":isolation})
                durable_stdout_event({"event":"rng_proxy_call_failed","ordinal":execution_order,"sample_id":sample_id,"started":isolation["delegate_invocations_started"],"completed":isolation["delegate_invocations_completed"],"rng_isolation_sha256":canonical_sha(isolation)})
            raise
        isolation=proxy.evidence
        rng_proxy_module.validate_evidence(isolation,require_completed=True)
        baseline[sample_id]=output;seed[sample_id]=item["seed"];request_hash[sample_id]=item["request_sha256"];output_hash[sample_id]=event["output_sha256"]
        row={"format":"strict-track2-v520-per-call-rng-isolation-evidence-v1","ordinal":execution_order,"sample_id":sample_id,"episode":item["episode"],"start":item["start"],"branch":item["branch"],"seed":item["seed"],"request_sha256":item["request_sha256"],"output_sha256":event["output_sha256"],"output_schema":{"shape":[8,256,256,3],"dtype":"uint8","contiguous":True},"context_sha256":item["context_sha256"],"history_sha256":item["history_sha256"],"future_sha256":item["future_sha256"],"seed_sha256":item["seed_sha256"],"instruction_sha256":item["instruction_sha256"],"warning_category":event["warning"]["category"],"warning_full_message":event["warning"]["message"],"raw_warning":{"category":event["warning"]["category"],"message":event["warning"]["message"]},"raw_warnings_unchanged":True,"sample_call_started":isolation["delegate_invocations_started"],"sample_call_completed":isolation["delegate_invocations_completed"],"rng_isolation":isolation,"rng_isolation_sha256":canonical_sha(isolation),"deterministic_before":event["mode_before"]["enabled"],"warn_only_before":event["mode_before"]["warn_only"],"deterministic_inside":event["mode_during"]["enabled"],"warn_only_inside":event["mode_during"]["warn_only"],"deterministic_after":event["mode_after"]["enabled"],"warn_only_after":event["mode_after"]["warn_only"],"cpu_rng_before_sha256":event["rng_before"]["cpu_sha256"],"cpu_rng_after_sha256":event["rng_after"]["cpu_sha256"],"cuda_rng_before_sha256_by_device":event["rng_before"]["cuda"],"cuda_rng_after_sha256_by_device":event["rng_after"]["cuda"],"rng_unchanged":event["rng_unchanged"]};row["event_canonical_sha256"]=canonical_sha(row);events.append(row);append_event(events_stream,row)
        durable_stdout_event({"event":"rng_proxy_call_completed","ordinal":execution_order,"sample_id":sample_id,"started":1,"completed":1,"rng_isolation_sha256":row["rng_isolation_sha256"],"event_canonical_sha256":row["event_canonical_sha256"]})
        append_event(progress_stream,{"event":"call_committed","ordinal":execution_order,"sample_id":sample_id,"request_sha256":item["request_sha256"],"output_sha256":event["output_sha256"]})
        PROGRESS.update({"completed_calls":execution_order+1,"last_warning":event["warning"],"last_rng":{"scope_before":event["rng_before"],"scope_after":event["rng_after"],"isolation":isolation}})
    del runtime;baseline.flush();fd=os.open(str(raw),os.O_RDONLY);os.fsync(fd);os.close(fd);append_event(progress_stream,{"event":"process_complete","role":args.role,"calls":1000});events_stream.close();progress_stream.close()
    raw_digest=sha(raw)
    ancestry={"formal":{"path":str(args.preregistration.resolve()),"sha256":args.preregistration_sha},"contract":{"path":str(args.contract.resolve()),"sha256":args.contract_sha},"authority_receipt":{"path":str(args.authority_receipt.resolve()),"sha256":args.authority_receipt_sha},"scope_source":{"path":str(args.scope_source.resolve()),"sha256":args.scope_sha},"rng_proxy_source":{"path":str(args.rng_proxy_source.resolve()),"sha256":args.rng_proxy_sha,"logical_bytes":args.rng_proxy_source.stat().st_size},"driver_source":{"path":str(args.driver_source.resolve()),"sha256":args.driver_sha},"worker_source":{"path":str(Path(__file__).resolve()),"sha256":args.worker_sha},"auditor_source":{"path":str(args.auditor_source.resolve()),"sha256":args.auditor_sha},"parent_preregistration":{"path":str(args.parent_preregistration.resolve()),"sha256":PARENT_PRE_SHA},"parent_failure":{"path":str(args.parent_failure.resolve()),"sha256":PARENT_FAILURE_SHA},"v169_closure":{"path":str(closure_path.resolve()),"sha256":parent["v169"]["closure_sha256"],"digest":parent["v169"]["closure_digest"]},"runtime":{"path":str(runtime_path.resolve()),"sha256":parent["source"]["runtime_sha256"]},"interpreter":{"lexical_path":str(executable),"resolved_path":str(resolved),"resolved_sha256":sha(resolved),"resolved_bytes":resolved.stat().st_size,"python_version":platform.python_version(),"numpy_version":np.__version__,"torch_version":torch.__version__}}
    common={"ordered_scalar_requests":1000,"baseline_raw_array_sha256":raw_digest,"request_digest_sha256":hashlib.sha256("".join(request_hash).encode()).hexdigest(),"output_digest_sha256":hashlib.sha256("".join(output_hash).encode()).hexdigest(),"seed_digest_sha256":raw_sha(seed.reshape(200,5)),"request_sha256_array_digest_sha256":raw_sha(request_hash.reshape(200,5)),"output_sha256_array_digest_sha256":raw_sha(output_hash.reshape(200,5)),"sample_id_sha256":raw_sha(np.arange(1000,dtype="<i8").reshape(200,5)),"context_repeat5_digest_sha256":hashlib.sha256("".join(context_hashes).encode()).hexdigest(),"context_repeat5_sha256":context_hashes,"guard_event_manifest_sha256":canonical_sha([{"ordinal":x["ordinal"],"sample_id":x["sample_id"],"deterministic_before":x["deterministic_before"],"warn_only_before":x["warn_only_before"],"deterministic_inside":x["deterministic_inside"],"warn_only_inside":x["warn_only_inside"],"deterministic_after":x["deterministic_after"],"warn_only_after":x["warn_only_after"],"warning_category":x["warning_category"],"warning_full_message":x["warning_full_message"]} for x in events]),"rng_event_manifest_sha256":canonical_sha([{"ordinal":x["ordinal"],"sample_id":x["sample_id"],"cpu_rng_before_sha256":x["cpu_rng_before_sha256"],"cpu_rng_after_sha256":x["cpu_rng_after_sha256"],"cuda_rng_before_sha256_by_device":x["cuda_rng_before_sha256_by_device"],"cuda_rng_after_sha256_by_device":x["cuda_rng_after_sha256_by_device"],"rng_unchanged":x["rng_unchanged"]} for x in events]),"rng_isolation_event_manifest_sha256":canonical_sha([{"ordinal":x["ordinal"],"sample_id":x["sample_id"],"rng_isolation_sha256":x["rng_isolation_sha256"],"sample_call_started":x["sample_call_started"],"sample_call_completed":x["sample_call_completed"]} for x in events]),"source_and_ancestry_sha256":canonical_sha(ancestry)}
    if args.role=="A":
        cache=cache_dir/"scalar_v169_temporal_cache.npz";tmp=cache.with_name(cache.name+".tmp")
        with tmp.open("xb") as stream:
            np.savez_compressed(stream,baseline=np.asarray(baseline).reshape(200,5,8,256,256,3),seed=seed.reshape(200,5),request_sha256=request_hash.reshape(200,5),output_sha256=output_hash.reshape(200,5),sample_id=np.arange(1000,dtype="<i8").reshape(200,5));stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,cache);cache_sha=sha(cache);del baseline;raw.unlink()
        artifact={"cache_npz_path":str((root/"cache"/cache.name).resolve()),"cache_npz_sha256":cache_sha,"cache_npz_bytes":cache.stat().st_size,"baseline_raw_array_sha256":raw_digest}
    else:
        del baseline
        artifact={"memmap_path":str((root/raw.name).resolve()),"memmap_sha256":raw_digest,"memmap_bytes":raw.stat().st_size,"baseline_raw_array_sha256":raw_digest,"memmap_retained":True,"cleanup_authorized":False}
    receipt={"format":"strict-track2-v522-v521-rng-isolated-cache-worker-receipt-v1","status":"passed_unreviewed_process_output","passed":True,"role":args.role,"execution_order":"canonical_0_to_999" if args.role=="A" else "reverse_999_to_0","fresh_process_identity":identity,"process_wall_end_ns":time.time_ns(),"artifact":artifact,"identity_path":str((root/"identity.json").resolve()),"identity_sha256":sha(identity_path),"call_events_path":str((root/events_path.name).resolve()),"call_events_sha256":sha(events_path),"call_events_bytes":events_path.stat().st_size,"call_events_count":1000,"progress_path":str((root/"progress.ndjson").resolve()),"progress_sha256":sha(progress_path),"progress_bytes":progress_path.stat().st_size,"progress_count":1002,"driver_completion_receipt_future_owned":True,"process_log_future_owned_by_driver":True,"durable_call_markers_in_process_log":True,"durable_call_started_markers":1000,"durable_call_completed_markers":1000,"calls":1000,"warning_count":1000,"raw_warning_count":1000,"other_warning_count":0,"rng_unchanged_count":1000,"rng_isolation_event_count":1000,"rng_isolation_delegate_started_count":sum(x["sample_call_started"] for x in events),"rng_isolation_delegate_completed_count":sum(x["sample_call_completed"] for x in events),"rng_isolation_python_all_four_stages_equal_count":sum(x["rng_isolation"]["python_all_four_stages_equal"] is True for x in events),"rng_isolation_numpy_all_four_stages_equal_count":sum(x["rng_isolation"]["numpy_all_four_stages_equal"] is True for x in events),"rng_isolation_torch_cpu_exit_restored_count":sum(x["rng_isolation"]["torch_cpu_exit_restored"] is True for x in events),"rng_isolation_torch_cuda_exit_restored_count":sum(x["rng_isolation"]["torch_cuda_exit_restored"] is True for x in events),"rng_isolation_event_manifest_sha256":common["rng_isolation_event_manifest_sha256"],"rng_proxy_source":{"path":str(args.rng_proxy_source.resolve()),"sha256":args.rng_proxy_sha,"logical_bytes":args.rng_proxy_source.stat().st_size},"request_digest_sha256":common["request_digest_sha256"],"output_digest_sha256":common["output_digest_sha256"],"seed_digest_sha256":common["seed_digest_sha256"],"request_sha256_array_digest_sha256":common["request_sha256_array_digest_sha256"],"output_sha256_array_digest_sha256":common["output_sha256_array_digest_sha256"],"sample_id_sha256":common["sample_id_sha256"],"context_repeat5_digest_sha256":common["context_repeat5_digest_sha256"],"guard_event_manifest_sha256":common["guard_event_manifest_sha256"],"rng_event_manifest_sha256":common["rng_event_manifest_sha256"],"source_and_ancestry":ancestry,"source_and_ancestry_sha256":common["source_and_ancestry_sha256"],"phase_a_preregistration_path":str(args.preregistration.resolve()),"phase_a_preregistration_sha256":args.preregistration_sha,"postregistration_authority_path":str(args.authority_receipt.resolve()),"postregistration_authority_sha256":args.authority_receipt_sha,"training_launched":False,"folds":0,"policy_updates":0,"reward_loaded":False,"dev_outcome_success_hidden_final_used":False,"rl_authorized":False}
    if any(receipt[key]!=1000 for key in ("rng_isolation_event_count","rng_isolation_delegate_started_count","rng_isolation_delegate_completed_count","rng_isolation_python_all_four_stages_equal_count","rng_isolation_numpy_all_four_stages_equal_count","rng_isolation_torch_cpu_exit_restored_count","rng_isolation_torch_cuda_exit_restored_count")):raise RuntimeError("rng isolation exact1000 aggregate")
    receipt_path=partial/"receipt.json";atomic_json(receipt_path,receipt)
    if args.role=="A":
        payload=tree_evidence(partial,exclude=(partial/"process.log",partial/"completion_receipt.json"))
        if [row[0] for row in payload["inventory"]]!=["cache/scalar_v169_temporal_cache.npz","call_events.ndjson","identity.json","progress.ndjson","receipt.json"]:raise RuntimeError("process A exact child payload")
        schema=contract["cache_artifact_and_manifest_contract"]
        manifest={"format":"strict-track2-v485-qualified-v169-cache-candidate-v1","cache_npz_path":artifact["cache_npz_path"],"cache_npz_sha256":artifact["cache_npz_sha256"],"cache_npz_logical_bytes":artifact["cache_npz_bytes"],"baseline_raw_array_sha256":raw_digest,"npz_key_order":schema["npz_exact_keys"],"npz_array_schema":schema["npz_exact_array_contract"],"ordered_scalar_requests":1000,"request_digest_sha256":common["request_digest_sha256"],"output_digest_sha256":common["output_digest_sha256"],"seed_digest_sha256":common["seed_digest_sha256"],"request_sha256_array_digest_sha256":common["request_sha256_array_digest_sha256"],"output_sha256_array_digest_sha256":common["output_sha256_array_digest_sha256"],"sample_id_sha256":common["sample_id_sha256"],"context_repeat5_digest_sha256":common["context_repeat5_digest_sha256"],"context_repeat5_sha256":common["context_repeat5_sha256"],"guard_event_manifest_sha256":common["guard_event_manifest_sha256"],"rng_event_manifest_sha256":common["rng_event_manifest_sha256"],"source_and_ancestry_sha256":common["source_and_ancestry_sha256"],"process_a_worker_receipt_path":str((root/"receipt.json").resolve()),"process_a_worker_receipt_sha256":sha(receipt_path),"phase_a_preregistration_path":str(args.preregistration.resolve()),"phase_a_preregistration_sha256":args.preregistration_sha,"phase_a_formal_path":str(args.preregistration.resolve()),"phase_a_formal_sha256":args.preregistration_sha,"process_a_payload_tree_algorithm":"relative POSIX sorted; sha256 + two spaces + relative path + newline; canonical compact JSON triples; manifest excluded","process_a_payload_tree_file_count":payload["file_count"],"process_a_payload_tree_logical_file_bytes":payload["logical_file_bytes"],"process_a_payload_tree_sha256sum_lines_digest_sha256":payload["sha256sum_lines_digest_sha256"],"process_a_payload_tree_canonical_json_digest_sha256":payload["canonical_json_digest_sha256"]}
        if set(manifest)!=set(schema["manifest_exact_fields"]):raise RuntimeError("manifest exact fields")
        atomic_json(cache_dir/"manifest.json",manifest);fd=os.open(str(cache_dir),os.O_RDONLY);os.fsync(fd);os.close(fd)
    fd=os.open(str(partial),os.O_RDONLY);os.fsync(fd);os.close(fd);os.replace(partial,root);fd=os.open(str(qualification_root),os.O_RDONLY);os.fsync(fd);os.close(fd);os._exit(0)


if __name__=="__main__":
    if "--synthetic-failure-self-test" in sys.argv:
        passed=synthetic_failure_self_test();print(json.dumps({"passed":passed,"identity_after_failure_streams_closed":passed},sort_keys=True));raise SystemExit(0 if passed else 3)
    if "--synthetic-bootstrap-self-test" in sys.argv:
        passed=synthetic_bootstrap_self_test();print(json.dumps({"passed":passed,"scope_tamper_rejected_before_import":passed},sort_keys=True));raise SystemExit(0 if passed else 3)
    scope=None
    known=bootstrap_scope()
    scope=load_module_exact(known.scope_source,"v485_cache_scope",known.scope_sha)
    try: raise SystemExit(main())
    except Exception as error:
        freeze_worker_failure(error)
        os._exit(1)
