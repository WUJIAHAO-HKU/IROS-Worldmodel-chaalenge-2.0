#!/usr/bin/env python3
"""Independent, self-bound design-only static audit for v485 Phase-A sources."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

CONTRACT_SHA="8c4a583aa19de6927cc1ef787920b65317893e27ad01371d0b95b622f260cf64"
SMOKE_WORKER_PATH=Path("/dev/shm/smoke_v482_v169_detcompat_worker.py")
EXACT_WARNING=("median CUDA with indices output does not have a deterministic implementation, but you set "
"'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues "
"to help us prioritize adding deterministic support for this operation. (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:93.)")

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8<<20),b""):digest.update(block)
    return digest.hexdigest()
def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
def atomic_json(path,payload):
    path=Path(path);tmp=path.with_name(path.name+".tmp")
    if path.exists() or tmp.exists():raise FileExistsError(path)
    with tmp.open("x") as stream:
        json.dump(payload,stream,sort_keys=True,indent=2);stream.write("\n");stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path);fd=os.open(str(path.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
def constant(tree,name):
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id==name for target in node.targets):return ast.literal_eval(node.value)
    raise RuntimeError(f"constant absent: {name}")
def calls(tree,name):
    result=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            target=node.func
            if isinstance(target,ast.Name) and target.id==name:result.append(node)
            if isinstance(target,ast.Attribute) and target.attr==name:result.append(node)
    return result
def exact_tree(root):
    root=Path(root);rows=[]
    if not root.is_dir() or root.is_symlink():raise RuntimeError("parent root")
    for path in sorted(root.rglob("*"),key=lambda value:value.relative_to(root).as_posix()):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):raise RuntimeError("parent tree nonregular")
        if path.is_file():rows.append(("./"+path.relative_to(root).as_posix(),sha(path),path.stat().st_size))
    lines="".join(f"{digest}  {relative}\n" for relative,digest,_ in rows).encode()
    return len(rows),sum(size for _,_,size in rows),hashlib.sha256(lines).hexdigest()
def regular(path,want):return Path(path).is_file() and not Path(path).is_symlink() and sha(path)==want
def named_values(value,name):
    if isinstance(value,dict):
        for key,item in value.items():
            if key==name:yield item
            yield from named_values(item,name)
    elif isinstance(value,list):
        for item in value:yield from named_values(item,name)
def all_values(value):
    if isinstance(value,dict):
        for item in value.values():yield from all_values(item)
    elif isinstance(value,list):
        for item in value:yield from all_values(item)
    else:yield value

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--contract",type=Path,required=True);parser.add_argument("--preregistration",type=Path,required=True);parser.add_argument("--preregistration-sha",required=True);parser.add_argument("--static-source",type=Path,required=True);parser.add_argument("--static-sha",required=True);parser.add_argument("--output",type=Path,required=True)
    for name in ("scope","worker","driver","auditor","materializer","launcher"):
        parser.add_argument(f"--{name}-source",type=Path,required=True);parser.add_argument(f"--{name}-sha",required=True)
    args=parser.parse_args();self_path=Path(__file__).resolve()
    paths={name:getattr(args,name+"_source") for name in ("scope","worker","driver","auditor","materializer","launcher")};paths["static_auditor"]=args.static_source
    wants={name:getattr(args,name+"_sha") for name in ("scope","worker","driver","auditor","materializer","launcher")};wants["static_auditor"]=args.static_sha
    role_map={"materializer":"phase_a_materializer","driver":"phase_a_driver","worker":"phase_a_process_worker","scope":"phase_a_cache_scope_helper","auditor":"phase_a_independent_auditor","static_auditor":"phase_a_static_auditor","launcher":"phase_a_launcher"}
    checks={}
    checks["static_self_bound"]=args.static_source.resolve()==self_path and args.static_sha==sha(self_path)
    checks["contract_exact"]=regular(args.contract,CONTRACT_SHA)
    checks["formal_exact"]=regular(args.preregistration,args.preregistration_sha)
    contract=json.loads(args.contract.read_text());formal=json.loads(args.preregistration.read_text())
    checks["contract_design_only"]=contract.get("format")=="strict-track2-v485-v482-v169-cache-determinism-qualification-design-contract-v7" and contract.get("authorization",{}).get("current_contract_authorizes_static_review") is True and contract.get("authorization",{}).get("phase_a_cache_qualification_authorized") is False and contract.get("authorization",{}).get("training_authorized") is False and contract.get("authorization",{}).get("policy_updates")==0
    guards=formal.get("authorization",{})
    checks["formal_nonauthorizing"]=formal.get("format")=="strict-track2-v485-v482-v169-cache-determinism-qualification-preregistration-v1" and formal.get("status")=="preregistered_cache_qualification_pending_postregistration_static_authority" and guards=={"phase_a_cache_qualification_authorized":False,"attempts_authorized":0,"cache_reuse_authorized":False,"training_authorized":False,"folds_authorized":0,"policy_updates":0,"s1_authorized":False,"zero_update_authorized":False,"rl_authorized":False,"submission_authorized":False}
    checks["qualification_root_absent"]=not os.path.lexists(formal["qualification_output_root"])
    checks["all_sources_exact"]=all(regular(paths[name],wants[name]) for name in paths)
    checks["formal_sources_exact"]=set(formal.get("execution_sources",{}))==set(role_map.values()) and all(formal["execution_sources"][role_map[name]]["path"]==str(paths[name].resolve()) and formal["execution_sources"][role_map[name]]["sha256"]==wants[name] and formal["execution_sources"][role_map[name]]["logical_bytes"]==paths[name].stat().st_size for name in paths)
    checks["formal_source_records_digest"]=formal.get("execution_source_records")==[{"role":role,**formal["execution_sources"][role]} for role in contract["phase_a_output_and_receipt_schema"]["source_closure_contract"]["exact_roles"]] and formal.get("execution_sources_digest_sha256")==canonical_sha(formal["execution_source_records"])
    checks["formal_contract_sections_exact"]=all(formal.get(key)==contract.get(key) for key in ("immutable_failed_parent","immutable_failed_parent_tree","unchanged_parent_semantics","only_permitted_code_change","warning_cadence_qualification_evidence","fresh_process_cache_smoke","fresh_process_identity_contract","cache_artifact_and_manifest_contract","phase_a_output_and_receipt_schema","phase_b_qualified_cache_consumer_boundary"))
    interpreter=formal["execution_interpreter"];lexical=Path(interpreter["lexical_path"]);intermediate=Path(interpreter["symlink_target"]);resolved=lexical.resolve(strict=True)
    version_run=subprocess.run([str(lexical),"-c","import json,numpy,torch,platform;print(json.dumps({'python':platform.python_version(),'numpy':numpy.__version__,'torch':torch.__version__},sort_keys=True))"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
    actual_versions=json.loads(version_run.stdout) if version_run.returncode==0 else {}
    checks["execution_interpreter_actual"]=lexical.is_symlink() and interpreter["lexical_is_symlink"] is True and os.readlink(lexical)==interpreter["symlink_target"] and intermediate.is_symlink() and os.readlink(intermediate)=="python3.11" and str(resolved)==interpreter["resolved_path"] and interpreter["resolved_is_regular_file"] is True and regular(resolved,interpreter["resolved_sha256"]) and resolved.stat().st_size==interpreter["resolved_bytes"] and actual_versions=={"python":interpreter["python_version"],"numpy":interpreter["numpy_version"],"torch":interpreter["torch_version"]}
    texts={name:path.read_text() for name,path in paths.items()};trees={name:ast.parse(text) for name,text in texts.items() if name!="launcher"}
    checks["helper_exact_warning"]=constant(trees["scope"],"ALLOWED_WARNING_CATEGORY")=="UserWarning" and constant(trees["scope"],"ALLOWED_WARNING_MESSAGE")==EXACT_WARNING and "warning_evidence != [{\"category\": ALLOWED_WARNING_CATEGORY, \"message\": ALLOWED_WARNING_MESSAGE}]" in texts["scope"]
    scope_imports={alias.name for node in ast.walk(trees["scope"]) if isinstance(node,ast.Import) for alias in node.names}|{node.module or "" for node in ast.walk(trees["scope"]) if isinstance(node,ast.ImportFrom)}
    checks["scope_has_no_parent_runtime_import"]=all(not any(token in name.lower() for token in ("v169","wam_pipeline","runtime")) for name in scope_imports)
    checks["determinism_scope_exact"]="use_deterministic_algorithms(False" not in "\n".join(texts.values()) and "torch.use_deterministic_algorithms(True, warn_only=True)" in texts["scope"] and "finally:" in texts["scope"] and "torch.use_deterministic_algorithms(True, warn_only=False)" in texts["scope"] and "rng_after != rng_before" in texts["scope"]
    forbidden=("backward","step","optimizer","train_head")
    checks["no_training_reward_outcome"]=not any(calls(tree,name) for tree in trees.values() for name in forbidden) and not any(token in texts["worker"] for token in ('data["temporal_rgb"]','data["target','reward_model','success','hidden','final_split'))
    worker=texts["worker"];driver=texts["driver"];auditor=texts["auditor"];materializer=texts["materializer"]
    identity_index=worker.index("atomic_json(identity_path,identity)");runtime_index=worker.index('load_module_exact(runtime_path,"v485_parent_runtime"');ctor_index=worker.index("Track2V169ArmRoutedRuntime")
    intent_index=worker.index('intent_path=qualification_root/"attempt_intent.json"')
    checks["identity_before_v169"]=intent_index<identity_index<runtime_index<ctor_index and all(field in worker for field in contract["fresh_process_identity_contract"]["fields"])
    checks["intent_nonce_roles_reverse_binding"]="secrets.token_hex(32)" in driver and '"roles":{"process_a":"canonical_0_to_999","process_b":"reverse_999_to_0"}' in driver and '"expected_identity_relative_paths"' in driver and 'intent_sha256":intent_sha' in worker and 'attempt intent reverse binding' in worker and 'future identity leaked into intent' in worker
    checks["driver_exact_A_then_B_then_audits"]=driver.index('base+["--role","A"]')<driver.index('base+["--role","B"]')<driver.index('["--stage","precleanup"]')<driver.index('["--stage","final"]')
    run_function=next(node for node in trees["driver"].body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name=="run")
    checks["driver_run_signature_and_reachable_audits"]=len(run_function.args.args)==3 and run_function.args.args[2].arg=="log_path" and 'run(audit_base+["--stage","precleanup"],1800,' in driver and 'run(audit_base+["--stage","final"],1800,' in driver and "def synthetic_run_self_test" in driver
    checks["output_root_only_formal_authority"]='p.add_argument("--output-root"' not in driver and 'p.add_argument("--output-root"' not in worker and 'p.add_argument("--root"' not in auditor and 'formal["qualification_output_root"]' in driver and 'authority.get("qualification_output_root"' in driver and 'formal["qualification_output_root"]' in worker
    checks["exact_raw_dtypes_no_masking"]="astype(" not in worker and "astype(" not in auditor and 'dtype="<U64"' in worker and 'np.dtype("<U64")' in auditor and "flags.c_contiguous" in worker and "flags.c_contiguous" in auditor
    checks["fresh_tuple_and_identity_after_failure_fixture"]="process_a/identity.json" in driver and "process_b/identity.json" in driver and "proc_self_stat_starttime_ticks" in worker and 'def freeze_worker_failure(error)' in worker and 'freeze_worker_failure(error)' in worker and 'def synthetic_failure_self_test()' in worker and 'identity_after_failure_streams_closed' in worker and 'os._exit(1)' in worker and 'FAILURE_CONTEXT["failure_path"]' not in worker
    checks["A_nested_fsync_and_whole_promote"]='os.open(str(cache_dir),os.O_RDONLY)' in worker and 'os.open(str(partial),os.O_RDONLY)' in worker and 'os.replace(partial,root);fd=os.open(str(qualification_root),os.O_RDONLY);os.fsync(fd);os.close(fd);os._exit(0)' in worker
    checks["A_acyclic_receipt_payload_manifest"] = worker.index('atomic_json(receipt_path,receipt)') < worker.index('payload=tree_evidence(partial,exclude=(partial/"process.log",partial/"completion_receipt.json"))') < worker.index('atomic_json(cache_dir/"manifest.json",manifest)')
    checks["driver_owned_full_lifetime_logs_and_completion"]='stream=prepare_worker_log(partial)' in driver and 'subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT' in driver and 'returncode=process.wait(timeout=timeout)' in driver and 'stream.flush();os.fsync(stream.fileno())' in driver and 'atomic_json(base/"completion_receipt.json",completion)' in driver and 'fsync_tree(base)' in driver and 'os.replace(partial,final)' in driver and 'strict-track2-v485-v482-v169-cache-phase-a-driver-completion-v1' in driver and 'process_log_sha256' not in worker and 'process_log_bytes' not in worker and 'os.dup2' not in worker
    checks["driver_completion_real_dispatch_fixture"]='success=run_worker(' in driver and '"--fail"' in driver and '"--sleep"' in driver and 'failure["exit_type"]=="nonzero_exit"' in driver and 'timeout_receipt["exit_type"]=="timeout"' in driver and 'open_receipt["exit_type"]=="spawn_error"' in driver and 'not (open_root/"process_b.partial").exists()' in driver
    checks["worker_progress_events_fsynced"]='progress.ndjson' in worker and 'call_events.ndjson' in worker and 'append_event(progress_stream' in worker and 'append_event(events_stream' in worker and 'progress_count":1002' in worker and 'driver_completion_receipt_future_owned' in worker and 'process_log_future_owned_by_driver' in worker
    checks["driver_staged_terminal_trees"]='tree_fields("pre_report_tree",pre_report)' in driver and 'tree_fields("pre_terminal_tree",pre_terminal)' in driver and 'stage4->stage5 tree' in driver and 'stage5->stage6 tree' in driver and 'terminal_receipt_excluded_from_pre_terminal_tree' in driver and 'failure_receipt.json' in driver
    checks["driver_failure_intent_stage_completion"]='CURRENT_STAGE="pre_intent"' in driver and '"stage":CURRENT_STAGE' in driver and '"intent_path"' in driver and '"intent_sha256"' in driver and '"attempt_nonce"' in driver and '"completion_receipts":completion_receipts' in driver and 'CURRENT_STAGE="process_a"' in driver and 'CURRENT_STAGE="process_b"' in driver and 'CURRENT_STAGE="precleanup_audit"' in driver and 'CURRENT_STAGE="final_audit"' in driver and 'CURRENT_STAGE="finalize_report"' in driver and 'CURRENT_STAGE="finalize_terminal"' in driver
    checks["intent_rename_failure_boundary"]='os.replace(prep,root)\n    FAILURE_ROOT=root;CURRENT_STAGE="intent_parent_fsync"' in driver
    checks["terminal_unique_quiet_commit"]='atomic_json(root/"terminal_receipt.json",terminal);TERMINAL_COMMITTED=True;return 0' in driver and 'terminal_exists=terminal_doc.get("format")' in driver and 'not TERMINAL_COMMITTED and not terminal_exists' in driver and 'print(json.dumps(terminal' not in driver
    checks["auditor_independent_rebuild_full_closure"]="def rebuild_requests(parent)" in auditor and 'independent request rebuild' in auditor and all(name in auditor for name in ("context_sha256","history_sha256","future_sha256","seed_sha256","instruction_sha256")) and 'for key,value in ancestry.items()' in auditor and 'formal source binding' in auditor
    checks["parent_runtime_path_SHA_before_import_and_independent_audit"]='sha(runtime_path)!=parent["source"]["runtime_sha256"]' in worker and worker.index('sha(runtime_path)!=parent["source"]["runtime_sha256"]')<worker.index('load_module_exact(runtime_path,"v485_parent_runtime"') and 'independent parent runtime frozen SHA' in auditor
    checks["scope_bootstrap_verified_bytes_before_import"]='def bootstrap_scope(argv=None)' in worker and 'scope=load_module_exact(known.scope_source,"v485_cache_scope",known.scope_sha)' in worker and worker.index('known=bootstrap_scope()')<worker.index('scope=load_module_exact(known.scope_source') and 'scope_tamper_rejected_before_import' in worker
    checks["driver_actual_rehash_exact7_roles"]=all(role in driver for role in contract["phase_a_output_and_receipt_schema"]["source_closure_contract"]["exact_roles"]) and all(token in driver for token in ("args.materializer_source","args.static_auditor_source","args.launcher_source")) and 'def actual_source_closure(formal,expected_sources)' in driver and driver.count('actual_source_closure(formal,expected_sources)')>=3 and 'actual_execution_source_records' in driver and 'actual_execution_sources_digest_sha256' in driver
    checks["auditor_exact_schema_and_AB_bytes"]='data.files != ["baseline","seed","request_sha256","output_sha256","sample_id"]' in auditor and 'np.array_equal(flat[begin:begin+8],mmap[begin:begin+8])' in auditor and 'array_raw_sha(flat[i])' in auditor and 'memmap.unlink()' in auditor and 'retained_outputs_byteexact' in auditor and 'stage2->stage3 tree' in auditor and 'stage3->stage4 tree' in auditor and 'def progress(' in auditor
    checks["auditor_completion_and_cleanup_crosslinks"]='def validate_completion(root,base,role,intent,worker_receipt)' in auditor and 'strict-track2-v485-v482-v169-cache-phase-a-driver-completion-v1' in auditor and 'completion.get("wait_seconds"),float' in auditor and 'cleanup.get("former_memmap_path")!=pre_evidence["B_memmap_path"]' in auditor and 'cleanup.get("former_memmap_sha256")!=pre_evidence["B_memmap_sha256"]' in auditor and 'cleanup.get("former_memmap_bytes")!=pre_evidence["B_memmap_bytes"]' in auditor and 'retained completion receipts' in auditor
    checks["final_auditor_rehashes_full_closure"]='def final_source_closure(worker_receipt)' in auditor and 'closure=final_source_closure(ar)' in auditor and 'final_formal_contract_authority_exact7_parent_runtime_rehashed' in auditor and 'actual_execution_sources_digest_sha256' in auditor and 'final parent runtime' in auditor
    checks["materializer_whole_dir_and_sections"]=".registration-prep" in materializer and 'os.replace(prep,registration_root)' in materializer and 'fresh_process_identity_contract' in materializer and 'phase_b_qualified_cache_consumer_boundary' in materializer and 'exact_tree(tree_spec["root"])' in materializer and 'verify_smoke(contract)' in materializer and 'failed cache partial not exact empty' in materializer and 'parent_formal["framework"]' in materializer and 'execution interpreter lexical closure' in materializer and 'execution interpreter intermediate closure' in materializer and 'execution interpreter versions' in materializer and 'failed-parent derived completed_folds' in materializer and 'failed-parent derived cleanup emptiness' in materializer and 'partial_entries!=' in materializer
    tree=exact_tree(contract["immutable_failed_parent_tree"]["root"]);spec=contract["immutable_failed_parent_tree"]
    checks["static_parent_tree_rehash"]=tree==(spec["file_count"],spec["logical_file_bytes"],spec["sha256sum_lines_digest_sha256"])
    checks["static_parent_files_absences"]=all(regular(entry["path"],entry["sha256"]) for entry in contract["immutable_failed_parent"].values() if isinstance(entry,dict) and set(("path","sha256"))<=set(entry)) and all(not os.path.lexists(entry["path"]) for entry in contract["immutable_failed_parent"]["required_absence"])
    partial=Path(spec["failed_cache_partial_root"]);checks["static_failed_cache_state"]=not os.path.lexists(spec["failed_cache_root"]) and partial.is_dir() and not partial.is_symlink() and not any(partial.iterdir())
    smoke=contract["warning_cadence_qualification_evidence"];smoke_docs={};smoke_files=True
    for role in ("aggregate_receipt","forward_receipt","reverse_receipt"):
        entry=smoke[role];smoke_files=smoke_files and regular(entry["path_at_review"],entry["sha256"])
        if smoke_files:smoke_docs[role]=json.loads(Path(entry["path_at_review"]).read_text())
    smoke_facts=False
    if smoke_files:
        aggregate=smoke_docs["aggregate_receipt"];forward=smoke_docs["forward_receipt"];reverse=smoke_docs["reverse_receipt"]
        def output_map(doc):
            result={}
            for request_id,output_sha in zip(doc["request_ids"],doc["output_sha256"]):
                if request_id in result and result[request_id]!=output_sha:return None
                result[request_id]=output_sha
            return result
        fm=output_map(forward);rm=output_map(reverse)
        actual_facts={"fresh_processes":2 if aggregate.get("two_fresh_processes_same_seed_bitexact") is True else 0,"calls_per_process":forward.get("calls") if forward.get("calls")==reverse.get("calls") else -1,"total_calls":aggregate.get("calls_total"),"allowed_warning_per_call":1 if all(doc.get("allowed_deterministic_warning_count")==doc.get("calls") for doc in (forward,reverse)) else -1,"other_warning_count":forward.get("other_deterministic_warning_count",-1)+reverse.get("other_deterministic_warning_count",-1),"repeat8_byteexact":aggregate.get("repeat8_each_fresh_process_bitexact") is True and forward.get("within_process_repeat8_bitexact") is True and reverse.get("within_process_repeat8_bitexact") is True,"forward_reverse_identity_byteexact":aggregate.get("four_request_reorder_invariant") is True and fm is not None and fm==rm,"gpu_peak_mib":aggregate.get("gpu_peak_mib"),"training_launched":aggregate.get("training_launched") is True or forward.get("training_launched") is True or reverse.get("training_launched") is True,"policy_updates":max(aggregate.get("policy_updates",-1),forward.get("policy_updates",-1),reverse.get("policy_updates",-1)),"v218_dual_health_restored":aggregate.get("v218_health_restored") is True}
        smoke_facts=aggregate.get("forward_receipt_sha256")==smoke["forward_receipt"]["sha256"] and aggregate.get("reverse_receipt_sha256")==smoke["reverse_receipt"]["sha256"] and aggregate.get("forward_npz_sha256")==forward.get("npz_sha256") and aggregate.get("reverse_npz_sha256")==reverse.get("npz_sha256") and actual_facts==smoke["facts"]
    smoke_worker_ok=regular(SMOKE_WORKER_PATH,smoke["worker_source_sha256"])
    if smoke_files:
        smoke_worker_ok=smoke_worker_ok and smoke_docs["aggregate_receipt"].get("worker_source_sha256")==smoke["worker_source_sha256"] and smoke_docs["forward_receipt"].get("source_sha256")==smoke["worker_source_sha256"] and smoke_docs["reverse_receipt"].get("source_sha256")==smoke["worker_source_sha256"]
    checks["static_small_smoke_rehash"]=smoke_files and smoke_facts and smoke_worker_ok
    launcher=texts["launcher"];lock_index=launcher.index('exec 9>/var/lock/v485_v169_cache_qualification.lock')
    checks["launcher_argument_driven_self_bound_zero_state"]=all(token in launcher[:lock_index] for token in ('--preregistration','--prereg-sha','--authority','--authority-sha','--contract-sha','LAUNCHER=$(readlink -f -- "$0")','LAUNCHER_SHA=$(sha_file "$LAUNCHER")','execution_source_records','qualification root must be absent/absolute','bridge unhealthy','gpu service unhealthy')) and "PENDING_PREREGISTRATION" not in launcher and launcher.index('if [[ "${1:-}" == \'--synthetic-self-test\' ]]')<lock_index and launcher.index('for value in "$PRE"')<lock_index and launcher.index('"$RLPY" - "$PRE"')<lock_index
    checks["launcher_runtime_cleanup_resources"]=all(token in launcher for token in ('WHOLE_TIMEOUT=9000','GPU_LIMIT_MIB=24576','SHM_MIN_BYTES=6442450944','setsid env','taskset -c 0-11','query-compute-apps=pid','kill -TERM -- "-$pgid"','kill -KILL -- "-$pgid"','process_group_empty_before_restore','gpu_compute_pids_empty_before_restore','v218_health_restored','trap cleanup EXIT',"trap 'exit 130' INT","trap 'exit 143' TERM",'terminal_receipt.json','failure_receipt.json','success_receipt.json')) and launcher.index('trap early_cleanup EXIT')<launcher.index('mkdir "$ATTEMPTS"') and launcher.index("trap cleanup EXIT")<launcher.index("STAGE='stop_v218'") and launcher.index("STAGE='restore_v218'")<launcher.rindex('STAGE=\'complete\';write_receipt "$ATTEMPT/success_receipt.json"') and 'wait "$ACTIVE_PID";DRIVER_RC=$?;set -e;terminate_active' in launcher
    checks["launcher_authority_and_driver_exact"]=all(token in launcher for token in ("strict-track2-v485-v169-cache-qualification-postregistration-authority-v1","authorized_exact_one_phase_a_cache_qualification_attempt","auth['attempts_authorized']==1",'--authority-receipt "$AUTHORITY"','--scope-source "$SCOPE"','--worker-source "$WORKER"','--auditor-source "$AUDITOR"','--materializer-source "$MATERIALIZER"','--static-auditor-source "$STATIC_AUDITOR"','--launcher-source "$LAUNCHER"','--restart-source','--authority-contract','authority_design_contract','restart_v218_source','STATIC_CHECK_COUNT=47',"STATIC_CHECK_KEYSET_SHA='57676e2e75b6c88cfcf521a1b1c4780315d43547158d1314df8e8444ed86834b'",'check_key_set_sha256','checks_sha256','sources_digest_sha256',"set(s['checks'].values())=={True}")) and 'training_authorized\':False' in launcher and 'policy_updates\':0' in launcher and 'bash "$RESTART_SOURCE" start' in launcher and 'restart_v218_source\':ev' in launcher
    fixture_env=dict(os.environ);fixture_env["CUDA_VISIBLE_DEVICES"]=""
    helper_test=subprocess.run([sys.executable,str(paths["scope"]),"--self-test"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120,env=fixture_env)
    worker_test=subprocess.run([sys.executable,str(paths["worker"]),"--synthetic-failure-self-test"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=120,env=fixture_env)
    bootstrap_test=subprocess.run([sys.executable,str(paths["worker"]),"--synthetic-bootstrap-self-test"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=120,env=fixture_env)
    auditor_test=subprocess.run([sys.executable,str(paths["auditor"]),"--synthetic-self-test"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120,env=fixture_env)
    driver_test=subprocess.run([sys.executable,str(paths["driver"]),"--synthetic-self-test"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120,env=fixture_env)
    launcher_syntax=subprocess.run(["bash","-n",str(paths["launcher"])],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120,env=fixture_env)
    launcher_test=subprocess.run(["bash",str(paths["launcher"]),"--synthetic-self-test"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120,env=fixture_env)
    launcher_missing=subprocess.run(["bash",str(paths["launcher"])],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120,env=fixture_env)
    checks["CPU_synthetic_fixtures"]=helper_test.returncode==0 and json.loads(helper_test.stdout)["passed"] is True and worker_test.returncode==0 and json.loads(worker_test.stdout)=={"identity_after_failure_streams_closed":True,"passed":True} and "synthetic identity-after failure" in worker_test.stderr and bootstrap_test.returncode==0 and json.loads(bootstrap_test.stdout)=={"passed":True,"scope_tamper_rejected_before_import":True} and auditor_test.returncode==0 and json.loads(auditor_test.stdout)["passed"] is True and driver_test.returncode==0 and json.loads(driver_test.stdout)=={"passed":True,"stages":["precleanup","final"]} and launcher_syntax.returncode==0 and launcher_test.returncode==0 and json.loads(launcher_test.stdout)["passed"] is True and launcher_missing.returncode==78
    checks={key:bool(value) for key,value in checks.items()};passed=all(checks.values())
    sources={role_map[name]:{"path":str(path.resolve()),"sha256":sha(path),"logical_bytes":path.stat().st_size} for name,path in paths.items()}
    source_records=[{"role":role,**sources[role]} for role in contract["phase_a_output_and_receipt_schema"]["source_closure_contract"]["exact_roles"]]
    check_keys=sorted(checks)
    receipt={"format":"strict-track2-v485-v169-cache-qualification-static-audit-v1","status":"passed_no_execution_authority" if passed else "failed","passed":passed,"checks":checks,"check_keys":check_keys,"check_key_set_sha256":canonical_sha(check_keys),"checks_sha256":canonical_sha(checks),"contract":{"path":str(args.contract.resolve()),"sha256":CONTRACT_SHA},"preregistration":{"path":str(args.preregistration.resolve()),"sha256":args.preregistration_sha},"sources":sources,"sources_digest_sha256":canonical_sha(source_records),"static_auditor_self_sha256":args.static_sha,"runtime_observation":{"phase_a_executed":False,"training_launched":False,"folds":0,"policy_updates":0,"rl_authorized":False},"phase_a_cache_qualification_authorized":False,"training_authorized":False,"submission_authorized":False}
    atomic_json(args.output,receipt);print(json.dumps(receipt,sort_keys=True));return 0 if passed else 3
if __name__=="__main__":raise SystemExit(main())
