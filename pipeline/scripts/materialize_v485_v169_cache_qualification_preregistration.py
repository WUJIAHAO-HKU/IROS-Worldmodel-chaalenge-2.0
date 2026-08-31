#!/usr/bin/env python3
"""Atomically register a nonauthorizing v485 Phase-A qualification lineage."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

CONTRACT_SHA="8c4a583aa19de6927cc1ef787920b65317893e27ad01371d0b95b622f260cf64"
SMOKE_WORKER_PATH=Path("/dev/shm/smoke_v482_v169_detcompat_worker.py")

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
def regular_file(path,want):
    path=Path(path)
    if not path.is_file() or path.is_symlink() or sha(path)!=want:raise RuntimeError(f"immutable file drift: {path}")
    return {"path":str(path.resolve()),"sha256":want,"logical_bytes":path.stat().st_size}
def exact_tree(root):
    root=Path(root)
    if not root.is_dir() or root.is_symlink():raise RuntimeError("failed-parent root")
    rows=[]
    for path in sorted(root.rglob("*"),key=lambda value:value.relative_to(root).as_posix()):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):raise RuntimeError("failed-parent nonregular")
        if path.is_file():
            relative="./"+path.relative_to(root).as_posix();rows.append({"relative_path":relative,"sha256":sha(path),"bytes":path.stat().st_size})
    lines="".join(f"{row['sha256']}  {row['relative_path']}\n" for row in rows).encode()
    return {"file_count":len(rows),"logical_file_bytes":sum(row["bytes"] for row in rows),"sha256sum_lines_digest_sha256":hashlib.sha256(lines).hexdigest(),"files":rows}
def nested_values(value):
    if isinstance(value,dict):
        for item in value.values():yield from nested_values(item)
    elif isinstance(value,list):
        for item in value:yield from nested_values(item)
    else:yield value
def named_values(value,name):
    if isinstance(value,dict):
        for key,item in value.items():
            if key==name:yield item
            yield from named_values(item,name)
    elif isinstance(value,list):
        for item in value:yield from named_values(item,name)
def verify_smoke(contract):
    spec=contract["warning_cadence_qualification_evidence"];receipts={}
    for role in ("aggregate_receipt","forward_receipt","reverse_receipt"):
        entry=spec[role];path=Path(entry["path_at_review"]);regular_file(path,entry["sha256"]);receipts[role]=json.loads(path.read_text())
    aggregate=receipts["aggregate_receipt"];forward=receipts["forward_receipt"];reverse=receipts["reverse_receipt"]
    if aggregate.get("forward_receipt_sha256")!=spec["forward_receipt"]["sha256"] or aggregate.get("reverse_receipt_sha256")!=spec["reverse_receipt"]["sha256"] or aggregate.get("forward_npz_sha256")!=forward.get("npz_sha256") or aggregate.get("reverse_npz_sha256")!=reverse.get("npz_sha256"):raise RuntimeError("small-smoke receipt crosslinks")
    def output_map(doc):
        result={}
        for request_id,output_sha in zip(doc["request_ids"],doc["output_sha256"]):
            if request_id in result and result[request_id]!=output_sha:raise RuntimeError("small-smoke request output inconsistency")
            result[request_id]=output_sha
        return result
    forward_map=output_map(forward);reverse_map=output_map(reverse)
    actual_facts={"fresh_processes":2 if aggregate.get("two_fresh_processes_same_seed_bitexact") is True else 0,"calls_per_process":forward.get("calls") if forward.get("calls")==reverse.get("calls") else -1,"total_calls":aggregate.get("calls_total"),"allowed_warning_per_call":1 if all(doc.get("allowed_deterministic_warning_count")==doc.get("calls") for doc in (forward,reverse)) else -1,"other_warning_count":forward.get("other_deterministic_warning_count",-1)+reverse.get("other_deterministic_warning_count",-1),"repeat8_byteexact":aggregate.get("repeat8_each_fresh_process_bitexact") is True and forward.get("within_process_repeat8_bitexact") is True and reverse.get("within_process_repeat8_bitexact") is True,"forward_reverse_identity_byteexact":aggregate.get("four_request_reorder_invariant") is True and forward_map==reverse_map,"gpu_peak_mib":aggregate.get("gpu_peak_mib"),"training_launched":aggregate.get("training_launched") is True or forward.get("training_launched") is True or reverse.get("training_launched") is True,"policy_updates":max(aggregate.get("policy_updates",-1),forward.get("policy_updates",-1),reverse.get("policy_updates",-1)),"v218_dual_health_restored":aggregate.get("v218_health_restored") is True}
    if actual_facts!=spec["facts"]:raise RuntimeError(f"small-smoke derived facts mismatch: {actual_facts}")
    worker_record=regular_file(SMOKE_WORKER_PATH,spec["worker_source_sha256"])
    if forward.get("source_sha256")!=worker_record["sha256"] or reverse.get("source_sha256")!=worker_record["sha256"] or aggregate.get("worker_source_sha256")!=worker_record["sha256"]:raise RuntimeError("small-smoke worker source receipt closure")
    return {"receipts":{key:{"path":str(Path(spec[key]["path_at_review"]).resolve()),"sha256":spec[key]["sha256"]} for key in receipts},"worker_source":worker_record,"facts":actual_facts,"verified":True}
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--contract",type=Path,required=True);parser.add_argument("--registration-root",type=Path,required=True);parser.add_argument("--qualification-output-root",type=Path,required=True)
    for name in ("scope","worker","driver","auditor","static-auditor","launcher","materializer"):
        parser.add_argument(f"--{name}-source",type=Path,required=True);parser.add_argument(f"--{name}-sha",required=True)
    args=parser.parse_args()
    if sha(args.contract)!=CONTRACT_SHA:raise RuntimeError("contract drift")
    contract=json.loads(args.contract.read_text())
    if contract.get("format")!="strict-track2-v485-v482-v169-cache-determinism-qualification-design-contract-v7" or contract.get("status")!="phase_a_design_only_pending_independent_static_review_and_postregistration_smoke_authority":raise RuntimeError("contract schema")
    auth=contract["authorization"]
    if auth["phase_a_cache_qualification_authorized"] is not False or auth["training_authorized"] is not False or auth["cache_reuse_authorized"] is not False or auth["folds_authorized"]!=0 or auth["policy_updates"]!=0:raise RuntimeError("contract authority")
    role_map=(("phase_a_materializer","materializer"),("phase_a_driver","driver"),("phase_a_process_worker","worker"),("phase_a_cache_scope_helper","scope"),("phase_a_independent_auditor","auditor"),("phase_a_static_auditor","static_auditor"),("phase_a_launcher","launcher"));sources={}
    for frozen_role,cli_role in role_map:
        path=getattr(args,cli_role+"_source");want=getattr(args,cli_role+"_sha")
        if len(want)!=64 or want.startswith("PENDING"):raise RuntimeError(f"pending source {frozen_role}")
        sources[frozen_role]=regular_file(path,want)
    if tuple(sources)!=tuple(contract["phase_a_output_and_receipt_schema"]["source_closure_contract"]["exact_roles"]):raise RuntimeError("source role order")
    if sources["phase_a_materializer"]["path"]!=str(Path(__file__).resolve()) or sources["phase_a_materializer"]["sha256"]!=sha(Path(__file__)):raise RuntimeError("materializer self closure")
    failed_files={}
    for name,entry in contract["immutable_failed_parent"].items():
        if isinstance(entry,dict) and set(("path","sha256"))<=set(entry):failed_files[name]=regular_file(entry["path"],entry["sha256"])
    failure_docs=[json.loads(Path(contract["immutable_failed_parent"][name]["path"]).read_text()) for name in ("launcher_failure_receipt","trainer_hard_failure_receipt")]
    parent_formal=json.loads(Path(contract["immutable_failed_parent"]["v482_r3_formal"]["path"]).read_text());framework=parent_formal["framework"];execution_interpreter=framework["execution_interpreter"]
    lexical=Path(execution_interpreter["lexical_path"])
    if not lexical.is_symlink() or execution_interpreter["lexical_is_symlink"] is not True or os.readlink(lexical)!=execution_interpreter["symlink_target"]:raise RuntimeError("execution interpreter lexical closure")
    intermediate=Path(execution_interpreter["symlink_target"])
    if not intermediate.is_symlink() or os.readlink(intermediate)!="python3.11":raise RuntimeError("execution interpreter intermediate closure")
    resolved_interpreter=lexical.resolve(strict=True)
    if str(resolved_interpreter)!=execution_interpreter["resolved_path"] or not resolved_interpreter.is_file() or resolved_interpreter.is_symlink() or execution_interpreter["resolved_is_regular_file"] is not True or sha(resolved_interpreter)!=execution_interpreter["resolved_sha256"] or resolved_interpreter.stat().st_size!=execution_interpreter["resolved_bytes"]:raise RuntimeError("execution interpreter resolved closure")
    version_run=subprocess.run([str(lexical),"-c","import json,numpy,torch,platform;print(json.dumps({'python':platform.python_version(),'numpy':numpy.__version__,'torch':torch.__version__},sort_keys=True))"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
    actual_versions=json.loads(version_run.stdout) if version_run.returncode==0 else {}
    if actual_versions!={"python":execution_interpreter["python_version"],"numpy":execution_interpreter["numpy_version"],"torch":execution_interpreter["torch_version"]}:raise RuntimeError("execution interpreter versions")
    derived_failure_facts={}
    for key,value in contract["immutable_failed_parent"]["required_failure_facts"].items():
        if key=="error_contains":
            if not any(isinstance(item,str) and value in item for doc in failure_docs for item in nested_values(doc)):raise RuntimeError("failed-parent error fact")
        elif key=="completed_folds":
            partial_root=Path(contract["immutable_failed_parent"]["trainer_hard_failure_receipt"]["path"]).parent
            partial_entries=sorted(item.name for item in partial_root.iterdir())
            completed=sum(1 for item in partial_root.glob("fold*_receipt.json") if item.is_file() and not item.is_symlink())
            if partial_entries!=[Path(contract["immutable_failed_parent"]["only_partial_result_file"]).name] or completed!=value or completed!=0 or failure_docs[1].get("all200_training_performed") is not False:raise RuntimeError("failed-parent derived completed_folds")
            derived_failure_facts[key]=completed
        elif key=="gpu_and_process_group_empty_before_restore":
            combined=failure_docs[0].get("gpu_compute_pids_empty_before_restore") is True and failure_docs[0].get("process_group_empty_before_restore") is True
            if combined is not value:raise RuntimeError("failed-parent derived cleanup emptiness")
            derived_failure_facts[key]=combined
        elif not any(value in tuple(named_values(doc,key)) for doc in failure_docs):raise RuntimeError(f"failed-parent fact: {key}")
    tree_spec=contract["immutable_failed_parent_tree"];tree=exact_tree(tree_spec["root"])
    for key in ("file_count","logical_file_bytes","sha256sum_lines_digest_sha256"):
        if tree[key]!=tree_spec[key]:raise RuntimeError(f"failed-parent tree {key}")
    if os.path.lexists(tree_spec["failed_cache_root"]):raise RuntimeError("failed cache root exists")
    partial=Path(tree_spec["failed_cache_partial_root"])
    if not partial.is_dir() or partial.is_symlink() or any(partial.iterdir()):raise RuntimeError("failed cache partial not exact empty")
    for entry in contract["immutable_failed_parent"]["required_absence"]:
        if os.path.lexists(entry["path"]):raise RuntimeError(f"required old output exists: {entry['path']}")
    smoke=verify_smoke(contract)
    registration_root=args.registration_root.resolve();qualification_root=args.qualification_output_root.resolve()
    if not args.registration_root.is_absolute() or not args.qualification_output_root.is_absolute() or args.registration_root!=registration_root or args.qualification_output_root!=qualification_root or registration_root==qualification_root or registration_root.is_relative_to(qualification_root) or qualification_root.is_relative_to(registration_root):raise RuntimeError("root containment")
    if not registration_root.parent.is_dir() or registration_root.parent.is_symlink() or not qualification_root.parent.is_dir() or qualification_root.parent.is_symlink():raise RuntimeError("root parent")
    prep=registration_root.with_name(registration_root.name+".registration-prep")
    if os.path.lexists(registration_root) or os.path.lexists(prep) or os.path.lexists(qualification_root):raise FileExistsError("registration/qualification root")
    evidence={"immutable_failed_parent_files":failed_files,"immutable_failed_parent_tree":{"root":str(Path(tree_spec["root"]).resolve()),**{key:tree[key] for key in ("file_count","logical_file_bytes","sha256sum_lines_digest_sha256")}},"derived_failure_facts":derived_failure_facts,"execution_interpreter_actual":{"lexical_path":str(lexical),"symlink_target":os.readlink(lexical),"intermediate_path":str(intermediate),"intermediate_symlink_target":os.readlink(intermediate),"resolved_path":str(resolved_interpreter),"resolved_sha256":sha(resolved_interpreter),"resolved_bytes":resolved_interpreter.stat().st_size,"versions":actual_versions},"failed_cache_root_absent":True,"failed_cache_partial_exact_empty_directory":True,"required_old_outputs_absent":contract["immutable_failed_parent"]["required_absence"],"warning_cadence_small_smoke":smoke}
    source_records=[{"role":role,**record} for role,record in sources.items()]
    pre={"format":"strict-track2-v485-v482-v169-cache-determinism-qualification-preregistration-v1","status":"preregistered_cache_qualification_pending_postregistration_static_authority","seed":contract["seed"],"contract":{"path":str(args.contract.resolve()),"sha256":CONTRACT_SHA},"immutable_failed_parent":contract["immutable_failed_parent"],"immutable_failed_parent_tree":contract["immutable_failed_parent_tree"],"verified_registration_evidence":evidence,"verified_registration_evidence_sha256":canonical_sha(evidence),"unchanged_parent_semantics":contract["unchanged_parent_semantics"],"only_permitted_code_change":contract["only_permitted_code_change"],"warning_cadence_qualification_evidence":contract["warning_cadence_qualification_evidence"],"fresh_process_cache_smoke":contract["fresh_process_cache_smoke"],"fresh_process_identity_contract":contract["fresh_process_identity_contract"],"cache_artifact_and_manifest_contract":contract["cache_artifact_and_manifest_contract"],"phase_a_output_and_receipt_schema":contract["phase_a_output_and_receipt_schema"],"phase_b_qualified_cache_consumer_boundary":contract["phase_b_qualified_cache_consumer_boundary"],"execution_interpreter":execution_interpreter,"execution_sources":sources,"execution_source_records":source_records,"execution_sources_digest_sha256":canonical_sha(source_records),"qualification_output_root":str(qualification_root),"required_output_absence_at_registration":True,"authorization":{"phase_a_cache_qualification_authorized":False,"attempts_authorized":0,"cache_reuse_authorized":False,"training_authorized":False,"folds_authorized":0,"policy_updates":0,"s1_authorized":False,"zero_update_authorized":False,"rl_authorized":False,"submission_authorized":False},"postregistration_static_authority_required":True,"runtime_observation":{"phase_a_executed":False,"training_launched":False,"folds":0,"policy_updates":0}}
    prep.mkdir();atomic_json(prep/"preregistration.json",pre);fd=os.open(str(prep),os.O_RDONLY);os.fsync(fd);os.close(fd);os.replace(prep,registration_root);fd=os.open(str(registration_root.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
    output=registration_root/"preregistration.json";print(json.dumps({"path":str(output),"sha256":sha(output),"registration_root":str(registration_root)},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
