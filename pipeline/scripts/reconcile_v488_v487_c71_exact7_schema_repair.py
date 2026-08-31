#!/usr/bin/env python3
"""Immutable, read-only reconciliation of three v485 static false positives.

This writer never imports v169, never creates the Phase-A qualification root and
never mutates the frozen b73 lineage.  It verifies the old 44 PASS predicates by
locking every predicate input byte, then independently recomputes the three
source-only predicates with AST-aware semantics.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from pathlib import Path

FORMAL_SHA="b73fad8071d8df0b1bed3dc53212bf8a86b2350911620feac81d96a1505492e0"
CONTRACT_SHA="8c4a583aa19de6927cc1ef787920b65317893e27ad01371d0b95b622f260cf64"
OLD_STATIC_SHA="d4cadd94141378f0cf89892847894abefc91357304f44d470698221a6613fc5f"
OLD_RECEIPT_SHA="4a287cd3b92ff281a785f454f8844e9695a58fd4b1de277e054ab4c22df48afc"
OLD_LOG_SHA="7a3e5aa7b627ff92ec82cb7073cac70dea35e26774bda76ef302c547a3ab070b"
CHECK_KEYSET_SHA="57676e2e75b6c88cfcf521a1b1c4780315d43547158d1314df8e8444ed86834b"
FALSE_KEYS={"determinism_scope_exact","driver_owned_full_lifetime_logs_and_completion","no_training_reward_outcome"}
REG_LINES_SHA="f00ebe9d953b360b17fae68a2f32581a4add89bb90b86702be142c18b8e728df"
REG_CANON_SHA="199e880325f99e75b7d70e566d8da17c51093bafba3da1c3acd20d4a85c22c63"
OUTPUT_PATH=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v486_v485_phase_a_static_reconciliation_seed1628_20260824/transparent_static_audit.json")
RECON_FORMAT="strict-track2-v486-v485-phase-a-static-false-positive-reconciliation-preregistration-v1"
RECON_STATUS="preregistered_exact_one_readonly_static_reconciliation_authorized"
RECON_AUTH={"readonly_static_reconciliation_authorized":True,"attempts_authorized":1,"attempts_consumed":0,"retry_authorized":False,"phase_a_cache_qualification_authorized":False,"cache_reuse_authorized":False,"training_authorized":False,"folds_authorized":0,"policy_updates":0,"s1_authorized":False,"zero_update_authorized":False,"rl_authorized":False,"submission_authorized":False,"reward_read_authorized":False,"dev_hidden_final_outcome_read_authorized":False}
REPAIR_FORMAL_PATH=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v488_v487_c71_exact7_schema_repair_prereg_seed1630_20260825/preregistration.json")
REPAIR_FORMAT="strict-track2-v488-v487-c71-exact7-schema-repair-preregistration-v1"
REPAIR_STATUS="preregistered_readonly_source_repair_pending_postregistration_authority"
REPAIR_TOP_KEYS={"format","status","seed","design_contract","materializer_source","source_closure","source_closure_sha256","f813_registration_tree","v487_authority_tree","v487_helper_evidence_tree","failed_reconciliation_attempt_tree","failed_attempt_no_retry","transparent_static_receipt_path","exact7_schema_repair_proof","future_reconciler_r2_source","required_postregistration_authority","authorization","runtime_observation","input_pre_snapshot","input_post_snapshot","input_snapshots_exactly_equal"}
REPAIR_SOURCE_ROLES={"parent_v485_preregistration","f813_reconciliation_preregistration","v487_authority_receipt","v487_authority_helper","v487_helper_process_receipt","old_reconciler_c71","old_v487_wrapper","future_reconciler_r2"}
REPAIR_AUTH={"readonly_source_repair_authorized":False,"postregistration_authority_required":True,"attempts_authorized":0,"retry_old_v487_authorized":False,"phase_a_cache_qualification_authorized":False,"cache_reuse_authorized":False,"training_authorized":False,"folds_authorized":0,"policy_updates":0,"s1_authorized":False,"zero_update_authorized":False,"rl_authorized":False,"submission_authorized":False,"reward_read_authorized":False,"dev_hidden_final_outcome_read_authorized":False}
REPAIR_RUNTIME={"repair_formal_registered":True,"reconciler_r2_executed":False,"transparent_receipt_created":False,"phase_a_executed":False,"training_launched":False,"folds":0,"policy_updates":0}
OLD_C71_SHA="c71ba00b0c92efda03f9df149263d0c476de86ea7338ba3719a72f3305959984"
OLD_C71_PATH=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/reconcile_v486_v485_static_false_positive.py")
REPAIR_DESIGN_PATH=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/v488_v487_c71_exact7_schema_repair_contract.json")
REPAIR_DESIGN_FORMAT="strict-track2-v488-v487-c71-exact7-schema-repair-design-contract-v1"
REPAIR_DESIGN_STATUS="design_only_frozen_sources_pending_independent_review_no_authority"
REPAIR_MATERIALIZER_PATH=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/materialize_v488_v487_c71_exact7_schema_repair_preregistration.py")

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8<<20),b""):digest.update(block)
    return digest.hexdigest()

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",", ":")).encode()).hexdigest()

def regular(path,want):
    path=Path(path)
    if not path.is_file() or path.is_symlink() or sha(path)!=want:raise RuntimeError(f"file closure: {path}")
    return {"path":str(path.resolve()),"sha256":want,"logical_bytes":path.stat().st_size}

def atomic_json(path,payload):
    path=Path(path);tmp=path.with_name(path.name+".tmp")
    if os.path.lexists(path) or os.path.lexists(tmp):raise FileExistsError(path)
    with tmp.open("x",encoding="utf-8") as stream:
        json.dump(payload,stream,sort_keys=True,indent=2);stream.write("\n");stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path);fd=os.open(str(path.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)

def exact_tree(root):
    root=Path(root);rows=[]
    if not root.is_dir() or root.is_symlink():raise RuntimeError("old REG root")
    for path in sorted(root.rglob("*")):
        if path.is_symlink():raise RuntimeError("old REG symlink")
        if path.is_file():rows.append([path.relative_to(root).as_posix(),sha(path),path.stat().st_size])
        elif not path.is_dir():raise RuntimeError("old REG nonregular")
    lines="".join(f"{digest}  {rel}\n" for rel,digest,_ in rows).encode()
    return {"inventory":rows,"file_count":len(rows),"logical_file_bytes":sum(row[2] for row in rows),"sha256sum_lines_digest_sha256":hashlib.sha256(lines).hexdigest(),"canonical_json_triples_digest_sha256":hashlib.sha256(json.dumps(rows,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()}

def call_name(node):
    if isinstance(node,ast.Name):return node.id
    if isinstance(node,ast.Attribute):return node.attr
    return ""

def qualified_name(node):
    if isinstance(node,ast.Name):return node.id
    if isinstance(node,ast.Attribute):
        prefix=qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""

def constant_strings(node):
    return [item.value for item in ast.walk(node) if isinstance(item,ast.Constant) and isinstance(item.value,str)]

def named_function(tree,name):
    matches=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==name]
    if len(matches)!=1:raise RuntimeError(f"exact function {name}")
    return matches[0]

def exact_bool_keyword(call,name,value):
    matches=[item.value for item in call.keywords if item.arg==name]
    return len(matches)==1 and isinstance(matches[0],ast.Constant) and matches[0].value is value

def deterministic_call(call,mode,warn_only):
    if not isinstance(call,ast.Call) or qualified_name(call.func)!="torch.use_deterministic_algorithms":return False
    positional=bool(call.args) and isinstance(call.args[0],ast.Constant) and call.args[0].value is mode
    keyword=any(item.arg=="mode" and isinstance(item.value,ast.Constant) and item.value.value is mode for item in call.keywords)
    return (positional or keyword) and exact_bool_keyword(call,"warn_only",warn_only)

def assigned_name(node,name):
    return isinstance(node,(ast.Assign,ast.AnnAssign)) and any(isinstance(item,ast.Name) and item.id==name for item in (node.targets if isinstance(node,ast.Assign) else [node.target]))

def compare_names(node,left,operator,right):
    return isinstance(node,ast.Compare) and isinstance(node.left,ast.Name) and node.left.id==left and len(node.ops)==1 and isinstance(node.ops[0],operator) and len(node.comparators)==1 and isinstance(node.comparators[0],ast.Name) and node.comparators[0].id==right

def input_snapshot(named_paths,tree_roots,absence_paths):
    files=[]
    for label,path in sorted(named_paths.items()):
        path=Path(path)
        if not path.is_file() or path.is_symlink():raise RuntimeError(f"snapshot regular file: {label}")
        files.append({"label":label,"path":str(path.resolve()),"sha256":sha(path),"logical_bytes":path.stat().st_size})
    trees={label:exact_tree(path) for label,path in sorted(tree_roots.items())}
    absences=[]
    for label,path in sorted(absence_paths.items()):
        path=Path(path);absent=not os.path.lexists(path)
        absences.append({"label":label,"path":str(path.resolve()),"absent":absent})
        if not absent:raise RuntimeError(f"snapshot required absence: {label}")
    payload={"files":files,"trees":trees,"absences":absences}
    return {**payload,"snapshot_sha256":canonical_sha(payload)}

def forbidden_determinism_false(trees):
    hits=[]
    for role,tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and call_name(node.func)=="use_deterministic_algorithms":
                false_arg=bool(node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value is False)
                false_kw=any(item.arg=="mode" and isinstance(item.value,ast.Constant) and item.value.value is False for item in node.keywords)
                if false_arg or false_kw:hits.append({"role":role,"line":node.lineno})
    return hits

def determinism_scope_proof(trees):
    tree=trees["phase_a_cache_scope_helper"];fn=named_function(tree,"predict_one")
    false_hits=forbidden_determinism_false(trees)
    warn_calls=[node for node in ast.walk(fn) if isinstance(node,ast.Call) and deterministic_call(node,True,True)]
    strict_calls=[node for node in ast.walk(fn) if isinstance(node,ast.Call) and deterministic_call(node,True,False)]
    tries=[node for node in fn.body if isinstance(node,ast.Try)]
    rng_before=[node for node in fn.body if assigned_name(node,"rng_before")]
    rng_after=[node for node in fn.body if assigned_name(node,"rng_after")]
    structure=False;details={}
    if len(warn_calls)==len(strict_calls)==len(tries)==len(rng_before)==len(rng_after)==1:
        warn=warn_calls[0];strict=strict_calls[0];scope_try=tries[0]
        predict_calls=[node for statement in scope_try.body for node in ast.walk(statement) if isinstance(node,ast.Call) and call_name(node.func)=="predict"]
        warning_contexts=[node for statement in scope_try.body for node in ast.walk(statement) if isinstance(node,ast.With) and any(isinstance(item.context_expr,ast.Call) and qualified_name(item.context_expr.func)=="warnings.catch_warnings" and exact_bool_keyword(item.context_expr,"record",True) for item in node.items)]
        simplefilters=[node for statement in scope_try.body for node in ast.walk(statement) if isinstance(node,ast.Call) and qualified_name(node.func)=="warnings.simplefilter" and node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value=="always"]
        final_calls=[node for statement in scope_try.finalbody for node in ast.walk(statement) if isinstance(node,ast.Call)]
        restore_only=len([node for node in final_calls if deterministic_call(node,True,False)])==1 and not any(deterministic_call(node,True,True) for node in final_calls)
        rng_guard=any(isinstance(node,ast.If) and compare_names(node.test,"rng_after",ast.NotEq,"rng_before") and any(isinstance(item,ast.Raise) for item in ast.walk(node)) for node in fn.body)
        mode_checks=[node for node in ast.walk(fn) if isinstance(node,ast.Call) and call_name(node.func)=="strict_state"]
        structure=(rng_before[0].lineno<warn.lineno<scope_try.lineno<=strict.lineno<rng_after[0].lineno and strict in final_calls and restore_only and len(predict_calls)==1 and len(warning_contexts)==1 and len(simplefilters)==1 and rng_guard and len(mode_checks)>=3)
        details={"rng_before_line":rng_before[0].lineno,"warn_only_enable_line":warn.lineno,"try_line":scope_try.lineno,"predict_line":predict_calls[0].lineno if len(predict_calls)==1 else None,"strict_restore_line":strict.lineno,"rng_after_line":rng_after[0].lineno,"rng_inequality_hard_raise":rng_guard,"strict_state_call_count":len(mode_checks),"exact_one_predict_in_try":len(predict_calls)==1,"exact_warning_context_and_filter":len(warning_contexts)==1 and len(simplefilters)==1,"finally_contains_only_required_mode_transition":restore_only}
    result={"ast_false_mode_calls":false_hits,"function":"predict_one","structured_try_finally_and_order":structure,"details":details}
    result["passed"]=not false_hits and structure
    return result

def driver_log_proof(tree,text):
    fn=named_function(tree,"run_worker")
    prepare=[];popen=[];wait=[]
    for node in ast.walk(fn):
        if not isinstance(node,ast.Call):continue
        name=call_name(node.func)
        if name=="prepare_worker_log":prepare.append(node)
        if name=="Popen":popen.append(node)
        if name=="wait":wait.append(node)
    prepare_ok=any(len(node.args)==2 and isinstance(node.args[0],ast.Name) and node.args[0].id=="partial" and isinstance(node.args[1],ast.Name) and node.args[1].id=="log_opener" for node in prepare)
    popen_ok=False
    for node in popen:
        kwargs={item.arg:item.value for item in node.keywords}
        stdout=kwargs.get("stdout");stderr=kwargs.get("stderr");session=kwargs.get("start_new_session")
        popen_ok=popen_ok or (isinstance(stdout,ast.Name) and stdout.id=="stream" and isinstance(stderr,ast.Attribute) and stderr.attr=="STDOUT" and isinstance(session,ast.Constant) and session.value is True)
    wait_ok=bool(wait) and all(any(item.arg=="timeout" for item in node.keywords) for node in wait)
    flush=[node for node in ast.walk(fn) if isinstance(node,ast.Call) and qualified_name(node.func)=="stream.flush"]
    log_fsync=[node for node in ast.walk(fn) if isinstance(node,ast.Call) and qualified_name(node.func)=="os.fsync" and node.args and any(isinstance(item,ast.Call) and qualified_name(item.func)=="stream.fileno" for item in ast.walk(node.args[0]))]
    completion=[node for node in ast.walk(fn) if isinstance(node,ast.Call) and call_name(node.func)=="atomic_json" and "completion_receipt.json" in constant_strings(node)]
    failed_ifs=[node for node in fn.body if isinstance(node,ast.If) and isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Name) and node.test.left.id=="status" and len(node.test.ops)==1 and isinstance(node.test.ops[0],ast.NotEq) and len(node.test.comparators)==1 and isinstance(node.test.comparators[0],ast.Constant) and node.test.comparators[0].value=="passed"]
    failed_order=False;failed_lines={}
    if len(failed_ifs)==1:
        failed_if=failed_ifs[0]
        tree_sync=[node for node in ast.walk(failed_if) if isinstance(node,ast.Call) and call_name(node.func)=="fsync_tree" and node.args and isinstance(node.args[0],ast.Name) and node.args[0].id=="base"]
        promote=[node for node in ast.walk(failed_if) if isinstance(node,ast.Call) and qualified_name(node.func)=="os.replace" and len(node.args)==2 and all(isinstance(item,ast.Name) for item in node.args) and [item.id for item in node.args]==["partial","final"]]
        root_sync=[node for node in ast.walk(failed_if) if isinstance(node,ast.Call) and qualified_name(node.func)=="os.fsync" and not any(isinstance(item,ast.Call) and qualified_name(item.func)=="stream.fileno" for item in ast.walk(node))]
        if len(tree_sync)==len(promote)==1 and root_sync:
            failed_order=tree_sync[0].lineno<promote[0].lineno<=max(node.lineno for node in root_sync)
            failed_lines={"tree_fsync_line":tree_sync[0].lineno,"atomic_promote_line":promote[0].lineno,"root_fsync_line":max(node.lineno for node in root_sync)}
    lifecycle_order=bool(wait and flush and log_fsync and completion) and max(node.lineno for node in wait)<min(node.lineno for node in flush)<=min(node.lineno for node in log_fsync)<completion[0].lineno and failed_order
    completion_format="strict-track2-v485-v482-v169-cache-phase-a-driver-completion-v1" in constant_strings(fn)
    result={"function":"run_worker","prepare_worker_log_two_argument_call":prepare_ok,"popen_redirect_and_new_session":popen_ok,"all_waits_bounded":wait_ok,"wait_call_lines":[node.lineno for node in wait],"log_flush_lines":[node.lineno for node in flush],"log_fsync_lines":[node.lineno for node in log_fsync],"completion_write_lines":[node.lineno for node in completion],"failed_branch_lines":failed_lines,"structured_wait_log_completion_tree_promote_order":lifecycle_order,"completion_format_present_as_ast_constant":completion_format}
    result["passed"]=prepare_ok and popen_ok and wait_ok and lifecycle_order and completion_format
    return result

def training_reward_proof(trees,worker_text):
    runtime_roles={role:trees[role] for role in ("phase_a_cache_scope_helper","phase_a_process_worker","phase_a_driver") if role in trees}
    forbidden_call_exact={"backward","step","train","fit","train_head","optimizer","get_model","load_model","load_state_dict","from_pretrained","load_reward","load_reward_model","score_reward","evaluate_reward","read_reward","read_outcome"}
    forbidden_semantic_parts={"reward","rewards","model","models","outcome","success","hidden","dev","final","optimizer","backward"}
    forbidden_attr_exact={"reward","rewards","model","models","outcome","success","hidden","dev","final","target","targets","temporal_rgb","optimizer","training_model","policy_model"}
    forbidden_key_exact={"reward","rewards","model","models","outcome","success","hidden","dev","final","target","targets","temporal_rgb","training_target","reward_model"}
    forbidden_name_exact={"reward","rewards","model","models","outcome","success","hidden","dev","target","targets","optimizer"}
    forbidden_calls=[];forbidden_imports=[];forbidden_attributes=[];forbidden_subscripts=[];forbidden_names=[];benign_nonsemantic_names=[]
    for role,tree in runtime_roles.items():
        functions=[item for item in ast.walk(tree) if isinstance(item,(ast.FunctionDef,ast.AsyncFunctionDef))]
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                leaf=call_name(node.func).lower();parts=set(re.split(r"[._]",leaf))
                if leaf in forbidden_call_exact or parts&forbidden_semantic_parts:forbidden_calls.append({"role":role,"name":qualified_name(node.func),"line":node.lineno})
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[item.name for item in node.names] if isinstance(node,ast.Import) else [node.module or ""]
                if any(set(re.split(r"[._]",name.lower()))&forbidden_semantic_parts for name in names):forbidden_imports.append({"role":role,"names":names,"line":node.lineno})
            if isinstance(node,ast.Attribute) and node.attr.lower() in forbidden_attr_exact:forbidden_attributes.append({"role":role,"name":qualified_name(node),"line":node.lineno})
            if isinstance(node,ast.Subscript) and isinstance(node.slice,ast.Constant) and isinstance(node.slice.value,str) and node.slice.value.lower() in forbidden_key_exact:forbidden_subscripts.append({"role":role,"key":node.slice.value,"line":node.lineno})
            if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load) and node.id.lower() in forbidden_name_exact:
                owners=[item for item in functions if item.lineno<=node.lineno<=getattr(item,"end_lineno",item.lineno)]
                owner=min(owners,key=lambda item:getattr(item,"end_lineno",item.lineno)-item.lineno).name if owners else None
                record={"role":role,"function":owner,"name":node.id,"line":node.lineno}
                if node.id=="success" and owner=="synthetic_run_self_test":record["classification"]="synthetic completion receipt fixture, not task success/outcome";benign_nonsemantic_names.append(record)
                else:forbidden_names.append(record)
    worker_tree=runtime_roles["phase_a_process_worker"]
    required_false={"training_launched":False,"folds":0,"policy_updates":0,"reward_loaded":False,"dev_outcome_success_hidden_final_used":False,"rl_authorized":False}
    dicts=[]
    for node in ast.walk(worker_tree):
        if not isinstance(node,ast.Dict):continue
        values={}
        for key,value in zip(node.keys,node.values):
            if isinstance(key,ast.Constant) and isinstance(key.value,str) and isinstance(value,ast.Constant):values[key.value]=value.value
        if all(values.get(key,object())==value for key,value in required_false.items()):dicts.append({key:values[key] for key in required_false})
    false_provenance=len(dicts)>=1
    benign_path_identifiers=[]
    for role,tree in runtime_roles.items():
        for fn in [node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))]:
            for node in ast.walk(fn):
                if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load) and node.id=="final":
                    classification="process-role filesystem destination" if fn.name=="run_worker" else "independent Phase-A audit receipt, not dev/hidden/final outcome"
                    benign_path_identifiers.append({"role":role,"function":fn.name,"identifier":"final","line":node.lineno,"classification":classification})
    result={"scanned_runtime_roles":sorted(runtime_roles),"forbidden_semantic_parts":sorted(forbidden_semantic_parts),"forbidden_calls":forbidden_calls,"forbidden_imports":forbidden_imports,"forbidden_attributes":forbidden_attributes,"forbidden_subscript_or_data_accesses":forbidden_subscripts,"forbidden_name_accesses":forbidden_names,"benign_nonsemantic_names":benign_nonsemantic_names,"benign_final_identifiers":benign_path_identifiers,"required_false_provenance":required_false,"matching_false_provenance_dict_count":len(dicts),"explicit_false_provenance_fields":false_provenance}
    result["passed"]=not forbidden_calls and not forbidden_imports and not forbidden_attributes and not forbidden_subscripts and not forbidden_names and false_provenance
    return result

def synthetic_evidence():
    fixtures={
        "literal":'x="use_deterministic_algorithms(False"',
        "forbidden":'torch.use_deterministic_algorithms(False)',
        "scope":'''def predict_one(runtime):\n rng_before=rng_state_sha256()\n torch.use_deterministic_algorithms(True,warn_only=True)\n try:\n  if strict_state()!={"enabled":True,"warn_only":True}: raise RuntimeError()\n  with warnings.catch_warnings(record=True) as caught:\n   warnings.simplefilter("always")\n   output=runtime.predict()\n finally:\n  torch.use_deterministic_algorithms(True,warn_only=False)\n if strict_state()!={"enabled":True,"warn_only":False}: raise RuntimeError()\n rng_after=rng_state_sha256()\n if rng_after!=rng_before: raise RuntimeError()\n return strict_state(),output''',
        "driver":'''def run_worker(partial,log_opener,timeout):\n stream=prepare_worker_log(partial,log_opener)\n process=subprocess.Popen([],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)\n returncode=process.wait(timeout=timeout)\n stream.flush();os.fsync(stream.fileno())\n completion={"format":"strict-track2-v485-v482-v169-cache-phase-a-driver-completion-v1"}\n atomic_json(base/"completion_receipt.json",completion)\n if status!="passed":\n  fsync_tree(base)\n  if base==partial:\n   os.replace(partial,final);fd=os.open(str(root),os.O_RDONLY);os.fsync(fd);os.close(fd)\n return completion''',
        "worker":'receipt={"training_launched":False,"folds":0,"policy_updates":0,"reward_loaded":False,"dev_outcome_success_hidden_final_used":False,"rl_authorized":False}',
    }
    literal=ast.parse(fixtures["literal"]);forbidden=ast.parse(fixtures["forbidden"])
    scope_tree=ast.parse(fixtures["scope"]);driver_tree=ast.parse(fixtures["driver"]);worker_tree=ast.parse(fixtures["worker"])
    checks={"literal_not_call":forbidden_determinism_false({"x":literal})==[],"false_call_detected":len(forbidden_determinism_false({"x":forbidden}))==1,"two_argument_driver_accepted":driver_log_proof(driver_tree,fixtures["driver"])["passed"],"false_provenance_names_allowed":training_reward_proof({"phase_a_cache_scope_helper":scope_tree,"phase_a_process_worker":worker_tree,"phase_a_driver":driver_tree},fixtures["worker"])["passed"] and determinism_scope_proof({"phase_a_cache_scope_helper":scope_tree,"phase_a_process_worker":worker_tree,"phase_a_driver":driver_tree})["passed"]}
    result={"passed":all(checks.values()),"check_count":len(checks),"checks":checks,"fixture_sources_sha256":canonical_sha(fixtures)}
    result["evidence_sha256"]=canonical_sha(result)
    return result

def dual_formal_identity(f813_execution_sources,repair_sources,repair_future,self_record):
    expected_old={"path":str(OLD_C71_PATH),"sha256":OLD_C71_SHA,"logical_bytes":36229}
    if not isinstance(f813_execution_sources,dict) or f813_execution_sources.get("reconciler")!=expected_old:raise RuntimeError("f813 old reconciler identity")
    if not isinstance(repair_sources,dict) or repair_sources.get("old_reconciler_c71")!=expected_old:raise RuntimeError("repair old reconciler identity")
    if repair_sources.get("future_reconciler_r2")!=self_record or repair_future!=self_record:raise RuntimeError("repair r2 identity")
    return True

def repair_formal_boundary(repair):
    if not isinstance(repair,dict) or set(repair)!=REPAIR_TOP_KEYS:raise RuntimeError("repair formal keyset")
    if repair.get("format")!=REPAIR_FORMAT or repair.get("status")!=REPAIR_STATUS or repair.get("seed")!=1630:raise RuntimeError("repair formal identity")
    if repair.get("authorization")!=REPAIR_AUTH or repair.get("runtime_observation")!=REPAIR_RUNTIME:raise RuntimeError("repair formal state")
    if repair.get("required_postregistration_authority") is not True or repair.get("failed_attempt_no_retry") is not True:raise RuntimeError("repair formal authority boundary")
    return True

def dual_formal_synthetic():
    old={"path":str(OLD_C71_PATH),"sha256":OLD_C71_SHA,"logical_bytes":36229};self_record={"path":"/frozen/reconciler-r2.py","sha256":"1"*64,"logical_bytes":123}
    f813={"reconciler":old,"reconciliation_materializer":{"path":"/frozen/materializer.py","sha256":"2"*64,"logical_bytes":456}}
    repair={"old_reconciler_c71":old,"future_reconciler_r2":self_record}
    checks={"actual_dual_identity_pass":dual_formal_identity(f813,repair,self_record,self_record) is True}
    for name,mutated_f813,mutated_repair,mutated_future in [
        ("f813_self_substitution_rejected",{**f813,"reconciler":self_record},repair,self_record),
        ("repair_old_c71_tamper_rejected",f813,{**repair,"old_reconciler_c71":self_record},self_record),
        ("repair_self_tamper_rejected",f813,{**repair,"future_reconciler_r2":old},self_record),
        ("repair_formal_future_tamper_rejected",f813,repair,old),
    ]:
        try:dual_formal_identity(mutated_f813,mutated_repair,mutated_future,self_record);checks[name]=False
        except RuntimeError:checks[name]=True
    formal={key:None for key in REPAIR_TOP_KEYS};formal.update({"format":REPAIR_FORMAT,"status":REPAIR_STATUS,"seed":1630,"authorization":REPAIR_AUTH,"runtime_observation":REPAIR_RUNTIME,"required_postregistration_authority":True,"failed_attempt_no_retry":True})
    checks["repair_formal_exact21_pass"]=repair_formal_boundary(formal) is True
    formal_mutations={"missing_key":{key:value for key,value in formal.items() if key!="source_closure_sha256"},"extra_key":{**formal,"unexpected":False},"status":{**formal,"status":"tampered"},"authorization":{**formal,"authorization":{**REPAIR_AUTH,"attempts_authorized":1}}}
    for label,mutated in formal_mutations.items():
        try:repair_formal_boundary(mutated);checks[f"repair_formal_{label}_rejected"]=False
        except RuntimeError:checks[f"repair_formal_{label}_rejected"]=True
    return {"passed":all(checks.values()),"check_count":len(checks),"checks":checks,"evidence_sha256":canonical_sha(checks)}

def synthetic_self_test():
    result={"legacy_false_positive_fixtures":synthetic_evidence(),"dual_formal_identity_fixtures":dual_formal_synthetic()};result["passed"]=all(value["passed"] for value in result.values());print(json.dumps(result,sort_keys=True));return 0 if result["passed"] else 3

def main():
    if os.sys.argv[1:]==["--synthetic-self-test"]:return synthetic_self_test()
    parser=argparse.ArgumentParser()
    parser.add_argument("--preregistration",type=Path,required=True);parser.add_argument("--preregistration-sha",required=True)
    parser.add_argument("--contract",type=Path,required=True);parser.add_argument("--contract-sha",required=True)
    parser.add_argument("--old-static-source",type=Path,required=True);parser.add_argument("--old-static-sha",required=True)
    parser.add_argument("--old-static-receipt",type=Path,required=True);parser.add_argument("--old-static-receipt-sha",required=True)
    parser.add_argument("--old-static-log",type=Path,required=True);parser.add_argument("--old-static-log-sha",required=True)
    parser.add_argument("--reconciliation-preregistration",type=Path,required=True);parser.add_argument("--reconciliation-preregistration-sha",required=True)
    parser.add_argument("--repair-preregistration",type=Path,required=True);parser.add_argument("--repair-preregistration-sha",required=True)
    parser.add_argument("--reconciliation-contract",type=Path,required=True);parser.add_argument("--reconciliation-contract-sha",required=True)
    parser.add_argument("--reconciler-source",type=Path,required=True);parser.add_argument("--reconciler-sha",required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    if (args.preregistration_sha,args.contract_sha,args.old_static_sha,args.old_static_receipt_sha,args.old_static_log_sha)!=(FORMAL_SHA,CONTRACT_SHA,OLD_STATIC_SHA,OLD_RECEIPT_SHA,OLD_LOG_SHA):raise RuntimeError("frozen SHA arguments")
    formal_record=regular(args.preregistration,FORMAL_SHA);contract_record=regular(args.contract,CONTRACT_SHA);old_source_record=regular(args.old_static_source,OLD_STATIC_SHA);old_receipt_record=regular(args.old_static_receipt,OLD_RECEIPT_SHA);volatile_log_record=regular(args.old_static_log,OLD_LOG_SHA)
    recon_formal_record=regular(args.reconciliation_preregistration,args.reconciliation_preregistration_sha);recon_contract_record=regular(args.reconciliation_contract,args.reconciliation_contract_sha)
    if args.repair_preregistration.resolve()!=REPAIR_FORMAL_PATH or len(args.repair_preregistration_sha)!=64 or any(char not in "0123456789abcdef" for char in args.repair_preregistration_sha):raise RuntimeError("repair formal arguments")
    repair_formal_record=regular(args.repair_preregistration,args.repair_preregistration_sha)
    self_path=Path(__file__).resolve();self_record=regular(args.reconciler_source,args.reconciler_sha)
    if args.reconciler_source.resolve()!=self_path or args.reconciler_sha!=sha(self_path):raise RuntimeError("reconciler self closure")
    formal=json.loads(args.preregistration.read_text());contract=json.loads(args.contract.read_text());old=json.loads(args.old_static_receipt.read_text());recon=json.loads(args.reconciliation_preregistration.read_text());repair=json.loads(args.repair_preregistration.read_text())
    if recon.get("format")!=RECON_FORMAT or recon.get("status")!=RECON_STATUS or recon.get("authorization")!=RECON_AUTH:raise RuntimeError("reconciliation formal boundary")
    if recon.get("design_contract")!={"path":recon_contract_record["path"],"sha256":args.reconciliation_contract_sha}:raise RuntimeError("reconciliation contract binding")
    repair_formal_boundary(repair)
    repair_design_spec=repair.get("design_contract");repair_materializer_spec=repair.get("materializer_source")
    if not isinstance(repair_design_spec,dict) or set(repair_design_spec)!={"path","sha256","logical_bytes"} or Path(repair_design_spec["path"])!=REPAIR_DESIGN_PATH:raise RuntimeError("repair design record")
    if not isinstance(repair_materializer_spec,dict) or set(repair_materializer_spec)!={"path","sha256","logical_bytes"} or Path(repair_materializer_spec["path"])!=REPAIR_MATERIALIZER_PATH:raise RuntimeError("repair materializer record")
    repair_design_record=regular(repair_design_spec["path"],repair_design_spec["sha256"]);repair_materializer_record=regular(repair_materializer_spec["path"],repair_materializer_spec["sha256"])
    if repair_design_record!=repair_design_spec or repair_materializer_record!=repair_materializer_spec:raise RuntimeError("repair design/materializer current")
    repair_design=json.loads(Path(repair_design_record["path"]).read_text())
    if repair_design.get("format")!=REPAIR_DESIGN_FORMAT or repair_design.get("status")!=REPAIR_DESIGN_STATUS or repair_design.get("materializer_source")!=repair_materializer_record:raise RuntimeError("repair design boundary")
    repair_sources=repair.get("source_closure")
    if not isinstance(repair_sources,dict) or set(repair_sources)!=REPAIR_SOURCE_ROLES or repair.get("source_closure_sha256")!=canonical_sha(repair_sources):raise RuntimeError("repair source closure schema")
    repair_source_records={}
    for role,record in sorted(repair_sources.items()):
        if not isinstance(record,dict) or set(record)!={"path","sha256","logical_bytes"}:raise RuntimeError(f"repair source schema {role}")
        actual=regular(record["path"],record["sha256"])
        if actual!=record:raise RuntimeError(f"repair source current {role}")
        repair_source_records[role]=actual
    if repair_design.get("source_closure")!=repair_source_records:raise RuntimeError("repair design source closure")
    expected_old_c71={"path":str(OLD_C71_PATH),"sha256":OLD_C71_SHA,"logical_bytes":36229}
    if repair_source_records["parent_v485_preregistration"]!=formal_record or repair_source_records["f813_reconciliation_preregistration"]!=recon_formal_record or repair_source_records["old_reconciler_c71"]!=expected_old_c71:raise RuntimeError("repair parent/formal/old c71 binding")
    if repair_source_records["future_reconciler_r2"]!=self_record or repair.get("future_reconciler_r2_source")!=self_record or repair_design.get("source_closure",{}).get("future_reconciler_r2")!=self_record:raise RuntimeError("repair r2 self binding")
    if repair.get("transparent_static_receipt_path")!=str(OUTPUT_PATH):raise RuntimeError("repair transparent path")
    if exact_tree(args.reconciliation_preregistration.parent)!=repair.get("f813_registration_tree"):raise RuntimeError("repair f813 tree")
    if exact_tree(Path(repair_source_records["v487_authority_receipt"]["path"]).parent)!=repair.get("v487_authority_tree"):raise RuntimeError("repair v487 authority tree")
    if exact_tree(Path(repair_source_records["v487_helper_process_receipt"]["path"]).parent)!=repair.get("v487_helper_evidence_tree"):raise RuntimeError("repair helper evidence tree")
    if exact_tree(Path(repair_design["failed_reconciliation_attempt_tree"]["root"]))!=repair.get("failed_reconciliation_attempt_tree"):raise RuntimeError("repair failed attempt tree")
    if repair.get("input_snapshots_exactly_equal") is not True or repair.get("input_pre_snapshot")!=repair.get("input_post_snapshot"):raise RuntimeError("repair snapshot equality")
    repair_snapshot=repair["input_pre_snapshot"]
    if not isinstance(repair_snapshot,dict) or set(repair_snapshot)!={"files","trees","absences","snapshot_sha256"} or repair_snapshot["snapshot_sha256"]!=canonical_sha({key:repair_snapshot[key] for key in ("files","trees","absences")}):raise RuntimeError("repair snapshot digest")
    if not isinstance(repair_snapshot["files"],list) or not isinstance(repair_snapshot["trees"],dict) or not isinstance(repair_snapshot["absences"],list):raise RuntimeError("repair snapshot collections")
    repair_snapshot_names=[row.get("name") for row in repair_snapshot["files"]]
    failed_relatives=[row[0] for row in repair["failed_reconciliation_attempt_tree"]["inventory"]]
    expected_repair_snapshot_names={"contract","materializer",*[f"source::{role}" for role in REPAIR_SOURCE_ROLES],*[f"exact7_current::{role}" for role in repair_design["exact7_schema_repair_proof"]["roles_in_order"]],*[f"failed_attempt::{rel}" for rel in failed_relatives]}
    if len(repair_snapshot_names)!=len(set(repair_snapshot_names)) or set(repair_snapshot_names)!=expected_repair_snapshot_names:raise RuntimeError("repair snapshot exact file coverage")
    if set(repair_snapshot["trees"])!={"f813_REG","v487_authority_REG","v487_helper_evidence","failed_reconciliation_attempt"}:raise RuntimeError("repair snapshot exact tree coverage")
    repair_absence_names=[row.get("name") for row in repair_snapshot["absences"]]
    expected_repair_absences={*repair_design["required_absences_before_registration"],"registration_prep"}
    if len(repair_absence_names)!=len(set(repair_absence_names)) or set(repair_absence_names)!=expected_repair_absences or any(row.get("absent") is not True for row in repair_snapshot["absences"]):raise RuntimeError("repair snapshot exact absence coverage")
    if Path(recon.get("transparent_static_receipt_path","")).resolve()!=OUTPUT_PATH or args.output.resolve()!=OUTPUT_PATH or args.output.parent!=args.reconciliation_preregistration.parent:raise RuntimeError("transparent output registration")
    if os.path.lexists(args.output) or os.path.lexists(args.output.with_name(args.output.name+".tmp")):raise FileExistsError(args.output)
    execution_sources=recon.get("execution_sources",{});dual_formal_identity(execution_sources,repair_source_records,repair.get("future_reconciler_r2_source"),self_record)
    reconciliation_source_records={}
    for role,record in sorted(execution_sources.items()):
        if not isinstance(record,dict) or set(record)!={"path","sha256","logical_bytes"}:raise RuntimeError(f"reconciliation source schema {role}")
        actual=regular(record["path"],record["sha256"])
        if actual!=record:raise RuntimeError(f"reconciliation source record {role}")
        reconciliation_source_records[role]=actual
    persistent_log_record=regular(recon["persistent_old_static_log"]["path"],OLD_LOG_SHA)
    if persistent_log_record!=recon["persistent_old_static_log"] or persistent_log_record["logical_bytes"]!=6475 or recon.get("volatile_log_source_at_registration")!={"path":volatile_log_record["path"],"sha256":OLD_LOG_SHA,"logical_bytes":6475}:raise RuntimeError("persistent/volatile log provenance")
    expected_initial_spec={"exact_relative_paths":["immutable_evidence/v485_static_b73.log","preregistration.json"],"file_count":2,"persistent_log_sha256":OLD_LOG_SHA,"persistent_log_bytes":6475,"preregistration_self_excluded_from_digest":True,"no_other_entries":True}
    if recon.get("registration_initial_inventory")!=expected_initial_spec:raise RuntimeError("reconciliation REG acyclic inventory spec")
    initial_tree=exact_tree(args.reconciliation_preregistration.parent)
    if initial_tree["file_count"]!=2 or initial_tree["logical_file_bytes"]!=recon_formal_record["logical_bytes"]+persistent_log_record["logical_bytes"] or [row[0] for row in initial_tree["inventory"]]!=expected_initial_spec["exact_relative_paths"] or initial_tree["inventory"][0][1:]!=[OLD_LOG_SHA,6475] or initial_tree["inventory"][1][1:]!=[args.reconciliation_preregistration_sha,recon_formal_record["logical_bytes"]]:raise RuntimeError("reconciliation REG actual exact2")
    repair_initial_tree=exact_tree(args.repair_preregistration.parent)
    if repair_initial_tree["inventory"]!=[["preregistration.json",args.repair_preregistration_sha,repair_formal_record["logical_bytes"]]] or repair_initial_tree["file_count"]!=1 or repair_initial_tree["logical_file_bytes"]!=repair_formal_record["logical_bytes"]:raise RuntimeError("repair REG actual exact1")
    if formal["format"]!="strict-track2-v485-v482-v169-cache-determinism-qualification-preregistration-v1" or formal["authorization"]!={"phase_a_cache_qualification_authorized":False,"attempts_authorized":0,"cache_reuse_authorized":False,"training_authorized":False,"folds_authorized":0,"policy_updates":0,"s1_authorized":False,"zero_update_authorized":False,"rl_authorized":False,"submission_authorized":False}:raise RuntimeError("formal boundary")
    if old["format"]!="strict-track2-v485-v169-cache-qualification-static-audit-v1" or old["status"]!="failed" or old["passed"] is not False:raise RuntimeError("old static terminal")
    if old["preregistration"]!={"path":formal_record["path"],"sha256":FORMAL_SHA} or old["contract"]!={"path":contract_record["path"],"sha256":CONTRACT_SHA} or old["static_auditor_self_sha256"]!=OLD_STATIC_SHA:raise RuntimeError("old static ancestry")
    check_keys=sorted(old["checks"])
    if len(check_keys)!=47 or old["check_keys"]!=check_keys or old["check_key_set_sha256"]!=CHECK_KEYSET_SHA or canonical_sha(check_keys)!=CHECK_KEYSET_SHA or old["checks_sha256"]!=canonical_sha(old["checks"]):raise RuntimeError("old check schema")
    if {key for key,value in old["checks"].items() if value is False}!=FALSE_KEYS or not all(value is True for key,value in old["checks"].items() if key not in FALSE_KEYS):raise RuntimeError("old 44/3 partition")
    role_order=contract["phase_a_output_and_receipt_schema"]["source_closure_contract"]["exact_roles"]
    if set(formal["execution_sources"])!=set(role_order) or old["sources"]!=formal["execution_sources"]:raise RuntimeError("source map")
    source_records=[];texts={};trees={}
    for role in role_order:
        record=formal["execution_sources"][role];actual=regular(record["path"],record["sha256"])
        if actual!=record:raise RuntimeError(f"source record {role}")
        source_records.append({"role":role,**record});texts[role]=Path(record["path"]).read_text()
        if Path(record["path"]).suffix==".py":trees[role]=ast.parse(texts[role])
    source_digest=canonical_sha(source_records)
    if formal["execution_source_records"]!=source_records or formal["execution_sources_digest_sha256"]!=source_digest or old["sources_digest_sha256"]!=source_digest:raise RuntimeError("source digest")
    exact7=recon.get("exact7_source_closure")
    if not isinstance(exact7,dict) or set(exact7)!={"all_records_must_equal_parent_formal_and_old_receipt_and_current_files","canonical_records_digest_sha256","records","roles_in_order"}:raise RuntimeError("reconciliation exact7 closure schema")
    if exact7["all_records_must_equal_parent_formal_and_old_receipt_and_current_files"] is not True or exact7["roles_in_order"]!=role_order:raise RuntimeError("reconciliation exact7 closure roles")
    if exact7["records"]!=source_records or exact7["canonical_records_digest_sha256"]!=source_digest or source_digest!="9adc5bdbfaa0022b745c44abac2ac2bd02f801f751ebce35d0116c0a8e8e53e0":raise RuntimeError("reconciliation exact7 closure records")
    derived_sources={record["role"]:{key:record[key] for key in ("path","sha256","logical_bytes")} for record in exact7["records"]}
    if derived_sources!=formal["execution_sources"]:raise RuntimeError("reconciliation exact7 closure source map")
    repair_proof=repair.get("exact7_schema_repair_proof")
    if not isinstance(repair_proof,dict) or set(repair_proof)!={"actual_exact4_keyset","old_c71_expected_exact3_keyset","records_equal","observed_current_records","observed_current_records_digest_sha256","roles_equal","digest_equal","map_equal","canonical_records_digest_sha256","new_r2_required_rule"}:raise RuntimeError("repair exact7 proof schema")
    if repair_proof["actual_exact4_keyset"]!=sorted(exact7) or repair_proof["old_c71_expected_exact3_keyset"]!=["execution_source_records","execution_sources","execution_sources_digest_sha256"]:raise RuntimeError("repair exact7 proof keysets")
    if any(repair_proof[key] is not True for key in ("records_equal","roles_equal","digest_equal","map_equal")) or repair_proof["observed_current_records"]!=source_records or repair_proof["observed_current_records_digest_sha256"]!=source_digest or repair_proof["canonical_records_digest_sha256"]!=source_digest:raise RuntimeError("repair exact7 proof records")
    if repair_proof["new_r2_required_rule"]!=repair_design["exact7_schema_repair_proof"]["new_r2_required_rule"]:raise RuntimeError("repair exact7 proof rule")
    reg_tree=exact_tree(args.preregistration.parent)
    if reg_tree!={"inventory":[["preregistration.json",FORMAL_SHA,55761],["static_audit.json",OLD_RECEIPT_SHA,7141]],"file_count":2,"logical_file_bytes":62902,"sha256sum_lines_digest_sha256":REG_LINES_SHA,"canonical_json_triples_digest_sha256":REG_CANON_SHA}:raise RuntimeError("old REG exact2")
    qualification_root=Path(formal["qualification_output_root"])
    absences=[qualification_root,args.preregistration.parent/"phase_a_launcher_attempts",args.preregistration.parent/"authority_receipt.json"]
    if any(os.path.lexists(path) for path in absences):raise RuntimeError("forbidden output exists")
    named_inputs={"parent_preregistration":args.preregistration,"phase_a_design_contract":args.contract,"old_static_source":args.old_static_source,"old_static_receipt":args.old_static_receipt,"volatile_old_static_log":args.old_static_log,"f813_reconciliation_preregistration":args.reconciliation_preregistration,"reconciliation_design_contract":args.reconciliation_contract,"repair_preregistration":args.repair_preregistration,"repair_design_contract":repair_design_record["path"],"repair_materializer_source":repair_materializer_record["path"],"reconciler_r2_cli_source":args.reconciler_source,"old_reconciler_c71":repair_source_records["old_reconciler_c71"]["path"],"persistent_old_static_log":persistent_log_record["path"]}
    for record in source_records:named_inputs[f"phase_a_exact7::{record['role']}"]=record["path"]
    for role,record in reconciliation_source_records.items():named_inputs[f"reconciliation_execution_source::{role}"]=record["path"]
    for role,record in repair_source_records.items():named_inputs[f"repair_source::{role}"]=record["path"]
    snapshot_absences={"phase_a_qualification_root":qualification_root,"old_phase_a_launcher_attempts":args.preregistration.parent/"phase_a_launcher_attempts","old_phase_a_authority_receipt":args.preregistration.parent/"authority_receipt.json","transparent_static_receipt":args.output,"transparent_static_receipt_tmp":args.output.with_name(args.output.name+".tmp")}
    snapshot_trees={"old_b73_registration":args.preregistration.parent,"f813_initial_registration":args.reconciliation_preregistration.parent,"v488_repair_registration":args.repair_preregistration.parent,"v487_authority_registration":Path(repair_source_records["v487_authority_receipt"]["path"]).parent,"v487_helper_evidence":Path(repair_source_records["v487_helper_process_receipt"]["path"]).parent,"failed_v487_attempt":Path(repair_design["failed_reconciliation_attempt_tree"]["root"])}
    input_pre_snapshot=input_snapshot(named_inputs,snapshot_trees,snapshot_absences)
    synthetic=synthetic_evidence()
    if synthetic.get("passed") is not True or synthetic.get("check_count")!=4 or set(synthetic["checks"].values())!={True}:raise RuntimeError("synthetic reconciliation proof")
    determinism=determinism_scope_proof(trees)
    driver=driver_log_proof(trees["phase_a_driver"],texts["phase_a_driver"])
    training=training_reward_proof(trees,texts["phase_a_process_worker"])
    corrected={"determinism_scope_exact":determinism["passed"],"driver_owned_full_lifetime_logs_and_completion":driver["passed"],"no_training_reward_outcome":training["passed"]}
    if corrected!={key:True for key in FALSE_KEYS}:raise RuntimeError("corrected checks failed")
    checks=dict(old["checks"]);checks.update(corrected)
    if set(checks.values())!={True}:raise RuntimeError("reconciled checks")
    input_post_snapshot=input_snapshot(named_inputs,snapshot_trees,snapshot_absences)
    if input_post_snapshot!=input_pre_snapshot:raise RuntimeError("reconciliation inputs changed during proof")
    parent_runtime=old["runtime_observation"]
    reconciliation_runtime={"reconciliation_executed":True,"phase_a_executed":False,"v169_imported_or_run":False,"qualification_root_created":False,"training_launched":False,"folds":0,"policy_updates":0,"reward_read_or_loaded":False,"dev_hidden_final_outcome_read":False}
    receipt={"format":"strict-track2-v485-v169-cache-qualification-static-audit-v1","status":"passed_no_execution_authority","passed":True,"reconciliation_format":"strict-track2-v486-v485-phase-a-transparent-static-reconciliation-v1","reconciliation_status":"passed_readonly_reconciliation_no_phase_a_execution_authority","repair_format":"strict-track2-v488-v487-c71-exact7-schema-repair-transparent-reconciliation-v1","checks":checks,"check_keys":check_keys,"check_key_set_sha256":CHECK_KEYSET_SHA,"checks_sha256":canonical_sha(checks),"contract":old["contract"],"preregistration":old["preregistration"],"sources":old["sources"],"sources_digest_sha256":source_digest,"static_auditor_self_sha256":OLD_STATIC_SHA,"reconciliation_preregistration":{"path":recon_formal_record["path"],"sha256":recon_formal_record["sha256"]},"reconciliation_design_contract":{"path":recon_contract_record["path"],"sha256":recon_contract_record["sha256"]},"repair_preregistration":repair_formal_record,"repair_design_contract":repair_design_record,"repair_materializer_source":repair_materializer_record,"old_reconciler_c71_source":repair_source_records["old_reconciler_c71"],"reconciler_r2_source":self_record,"repair_source_closure":repair_source_records,"repair_registration_tree":repair_initial_tree,"f813_registration_tree":initial_tree,"receipt_writer":self_record,"persistent_old_static_log":persistent_log_record,"volatile_log_source_at_registration":volatile_log_record,"registration_initial_inventory":initial_tree,"reconciliation_ancestry":{"old_static_source":old_source_record,"old_failed_static_receipt":old_receipt_record,"old_registration_tree":reg_tree,"old_true_check_count":44,"old_false_check_names":sorted(FALSE_KEYS)},"repair_formal_ancestry":{"f813_reconciliation_preregistration":recon_formal_record,"repair_preregistration":repair_formal_record,"failed_v487_attempt_tree":repair["failed_reconciliation_attempt_tree"],"v487_authority_tree":repair["v487_authority_tree"],"v487_helper_evidence_tree":repair["v487_helper_evidence_tree"],"exact7_schema_repair_proof":repair_proof},"synthetic_recomputation_evidence":synthetic,"input_pre_snapshot":input_pre_snapshot,"input_post_snapshot":input_post_snapshot,"input_snapshots_exactly_equal":True,"false_positive_recomputation":{"determinism_scope_exact":determinism,"driver_owned_full_lifetime_logs_and_completion":driver,"no_training_reward_outcome":training},"immutable_absences":[str(path.resolve()) for path in absences],"runtime_observation":parent_runtime,"parent_runtime_observation":parent_runtime,"reconciliation_runtime_observation":reconciliation_runtime,"phase_a_cache_qualification_authorized":False,"cache_reuse_authorized":False,"training_authorized":False,"folds_authorized":0,"policy_updates":0,"s1_authorized":False,"zero_update_authorized":False,"rl_authorized":False,"submission_authorized":False,"reward_read_authorized":False,"dev_hidden_final_outcome_read_authorized":False,"reconciliation_only":True,"phase_a_executed":False,"v169_imported_or_run":False}
    atomic_json(args.output,receipt);print(json.dumps({"path":str(args.output.resolve()),"sha256":sha(args.output),"checks_sha256":receipt["checks_sha256"]},sort_keys=True));return 0

if __name__=="__main__":raise SystemExit(main())
