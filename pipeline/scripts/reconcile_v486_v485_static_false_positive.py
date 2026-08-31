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

def synthetic_self_test():
    result=synthetic_evidence();print(json.dumps(result,sort_keys=True));return 0 if result["passed"] else 3

def main():
    if os.sys.argv[1:]==["--synthetic-self-test"]:return synthetic_self_test()
    parser=argparse.ArgumentParser()
    parser.add_argument("--preregistration",type=Path,required=True);parser.add_argument("--preregistration-sha",required=True)
    parser.add_argument("--contract",type=Path,required=True);parser.add_argument("--contract-sha",required=True)
    parser.add_argument("--old-static-source",type=Path,required=True);parser.add_argument("--old-static-sha",required=True)
    parser.add_argument("--old-static-receipt",type=Path,required=True);parser.add_argument("--old-static-receipt-sha",required=True)
    parser.add_argument("--old-static-log",type=Path,required=True);parser.add_argument("--old-static-log-sha",required=True)
    parser.add_argument("--reconciliation-preregistration",type=Path,required=True);parser.add_argument("--reconciliation-preregistration-sha",required=True)
    parser.add_argument("--reconciliation-contract",type=Path,required=True);parser.add_argument("--reconciliation-contract-sha",required=True)
    parser.add_argument("--reconciler-source",type=Path,required=True);parser.add_argument("--reconciler-sha",required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    if (args.preregistration_sha,args.contract_sha,args.old_static_sha,args.old_static_receipt_sha,args.old_static_log_sha)!=(FORMAL_SHA,CONTRACT_SHA,OLD_STATIC_SHA,OLD_RECEIPT_SHA,OLD_LOG_SHA):raise RuntimeError("frozen SHA arguments")
    formal_record=regular(args.preregistration,FORMAL_SHA);contract_record=regular(args.contract,CONTRACT_SHA);old_source_record=regular(args.old_static_source,OLD_STATIC_SHA);old_receipt_record=regular(args.old_static_receipt,OLD_RECEIPT_SHA);volatile_log_record=regular(args.old_static_log,OLD_LOG_SHA)
    recon_formal_record=regular(args.reconciliation_preregistration,args.reconciliation_preregistration_sha);recon_contract_record=regular(args.reconciliation_contract,args.reconciliation_contract_sha)
    self_path=Path(__file__).resolve();self_record=regular(args.reconciler_source,args.reconciler_sha)
    if args.reconciler_source.resolve()!=self_path or args.reconciler_sha!=sha(self_path):raise RuntimeError("reconciler self closure")
    formal=json.loads(args.preregistration.read_text());contract=json.loads(args.contract.read_text());old=json.loads(args.old_static_receipt.read_text());recon=json.loads(args.reconciliation_preregistration.read_text())
    if recon.get("format")!=RECON_FORMAT or recon.get("status")!=RECON_STATUS or recon.get("authorization")!=RECON_AUTH:raise RuntimeError("reconciliation formal boundary")
    if recon.get("design_contract")!={"path":recon_contract_record["path"],"sha256":args.reconciliation_contract_sha}:raise RuntimeError("reconciliation contract binding")
    if Path(recon.get("transparent_static_receipt_path","")).resolve()!=OUTPUT_PATH or args.output.resolve()!=OUTPUT_PATH or args.output.parent!=args.reconciliation_preregistration.parent:raise RuntimeError("transparent output registration")
    if os.path.lexists(args.output) or os.path.lexists(args.output.with_name(args.output.name+".tmp")):raise FileExistsError(args.output)
    execution_sources=recon.get("execution_sources",{});reconciler_entry=execution_sources.get("reconciler")
    if not isinstance(reconciler_entry,dict) or reconciler_entry!={"path":self_record["path"],"sha256":self_record["sha256"],"logical_bytes":self_record["logical_bytes"]}:raise RuntimeError("reconciliation source binding")
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
    if recon.get("exact7_source_closure")!={"execution_sources":formal["execution_sources"],"execution_source_records":source_records,"execution_sources_digest_sha256":source_digest}:raise RuntimeError("reconciliation exact7 closure")
    reg_tree=exact_tree(args.preregistration.parent)
    if reg_tree!={"inventory":[["preregistration.json",FORMAL_SHA,55761],["static_audit.json",OLD_RECEIPT_SHA,7141]],"file_count":2,"logical_file_bytes":62902,"sha256sum_lines_digest_sha256":REG_LINES_SHA,"canonical_json_triples_digest_sha256":REG_CANON_SHA}:raise RuntimeError("old REG exact2")
    qualification_root=Path(formal["qualification_output_root"])
    absences=[qualification_root,args.preregistration.parent/"phase_a_launcher_attempts",args.preregistration.parent/"authority_receipt.json"]
    if any(os.path.lexists(path) for path in absences):raise RuntimeError("forbidden output exists")
    named_inputs={"parent_preregistration":args.preregistration,"phase_a_design_contract":args.contract,"old_static_source":args.old_static_source,"old_static_receipt":args.old_static_receipt,"volatile_old_static_log":args.old_static_log,"reconciliation_preregistration":args.reconciliation_preregistration,"reconciliation_design_contract":args.reconciliation_contract,"reconciler_cli_source":args.reconciler_source,"persistent_old_static_log":persistent_log_record["path"]}
    for record in source_records:named_inputs[f"phase_a_exact7::{record['role']}"]=record["path"]
    for role,record in reconciliation_source_records.items():named_inputs[f"reconciliation_execution_source::{role}"]=record["path"]
    snapshot_absences={"phase_a_qualification_root":qualification_root,"old_phase_a_launcher_attempts":args.preregistration.parent/"phase_a_launcher_attempts","old_phase_a_authority_receipt":args.preregistration.parent/"authority_receipt.json","transparent_static_receipt":args.output,"transparent_static_receipt_tmp":args.output.with_name(args.output.name+".tmp")}
    snapshot_trees={"old_b73_registration":args.preregistration.parent,"new_v486_initial_registration":args.reconciliation_preregistration.parent}
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
    receipt={"format":"strict-track2-v485-v169-cache-qualification-static-audit-v1","status":"passed_no_execution_authority","passed":True,"reconciliation_format":"strict-track2-v486-v485-phase-a-transparent-static-reconciliation-v1","reconciliation_status":"passed_readonly_reconciliation_no_phase_a_execution_authority","checks":checks,"check_keys":check_keys,"check_key_set_sha256":CHECK_KEYSET_SHA,"checks_sha256":canonical_sha(checks),"contract":old["contract"],"preregistration":old["preregistration"],"sources":old["sources"],"sources_digest_sha256":source_digest,"static_auditor_self_sha256":OLD_STATIC_SHA,"reconciliation_preregistration":{"path":recon_formal_record["path"],"sha256":recon_formal_record["sha256"]},"reconciliation_design_contract":{"path":recon_contract_record["path"],"sha256":recon_contract_record["sha256"]},"receipt_writer":self_record,"persistent_old_static_log":persistent_log_record,"volatile_log_source_at_registration":volatile_log_record,"registration_initial_inventory":initial_tree,"reconciliation_ancestry":{"old_static_source":old_source_record,"old_failed_static_receipt":old_receipt_record,"old_registration_tree":reg_tree,"old_true_check_count":44,"old_false_check_names":sorted(FALSE_KEYS)},"synthetic_recomputation_evidence":synthetic,"input_pre_snapshot":input_pre_snapshot,"input_post_snapshot":input_post_snapshot,"input_snapshots_exactly_equal":True,"false_positive_recomputation":{"determinism_scope_exact":determinism,"driver_owned_full_lifetime_logs_and_completion":driver,"no_training_reward_outcome":training},"immutable_absences":[str(path.resolve()) for path in absences],"runtime_observation":parent_runtime,"parent_runtime_observation":parent_runtime,"reconciliation_runtime_observation":reconciliation_runtime,"phase_a_cache_qualification_authorized":False,"cache_reuse_authorized":False,"training_authorized":False,"folds_authorized":0,"policy_updates":0,"s1_authorized":False,"zero_update_authorized":False,"rl_authorized":False,"submission_authorized":False,"reward_read_authorized":False,"dev_hidden_final_outcome_read_authorized":False,"reconciliation_only":True,"phase_a_executed":False,"v169_imported_or_run":False}
    atomic_json(args.output,receipt);print(json.dumps({"path":str(args.output.resolve()),"sha256":sha(args.output),"checks_sha256":receipt["checks_sha256"]},sort_keys=True));return 0

if __name__=="__main__":raise SystemExit(main())
