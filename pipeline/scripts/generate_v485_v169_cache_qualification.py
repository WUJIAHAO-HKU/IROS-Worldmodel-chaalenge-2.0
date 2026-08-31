#!/usr/bin/env python3
"""One-shot driver for two fresh v169 cache qualification processes; never trains."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import secrets
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


CONTRACT_SHA="8c4a583aa19de6927cc1ef787920b65317893e27ad01371d0b95b622f260cf64"
FAILURE_ROOT=None
CURRENT_STAGE="pre_intent"
INTENT_BINDING=None
TERMINAL_COMMITTED=False

class WorkerLogSetupError(RuntimeError):pass


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
def tree_evidence(root):
    root=Path(root);inventory=[]
    for path in sorted(root.rglob("*"),key=lambda value:value.relative_to(root).as_posix()):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):raise RuntimeError("terminal tree nonregular")
        if path.is_file():inventory.append([path.relative_to(root).as_posix(),sha(path),path.stat().st_size])
    lines="".join(f"{digest}  {relative}\n" for relative,digest,_ in inventory).encode()
    return {"file_count":len(inventory),"logical_file_bytes":sum(row[2] for row in inventory),"sha256sum_lines_digest_sha256":hashlib.sha256(lines).hexdigest(),"canonical_json_digest_sha256":canonical_sha(inventory),"inventory":inventory}
def tree_fields(prefix,evidence):return {f"{prefix}_file_count":evidence["file_count"],f"{prefix}_logical_file_bytes":evidence["logical_file_bytes"],f"{prefix}_sha256sum_lines_digest_sha256":evidence["sha256sum_lines_digest_sha256"],f"{prefix}_canonical_json_digest_sha256":evidence["canonical_json_digest_sha256"],f"{prefix}_inventory":evidence["inventory"]}
def validate_tree_fields(payload,prefix):
    inventory=payload[f"{prefix}_inventory"]
    if inventory!=sorted(inventory,key=lambda row:row[0]) or len({row[0] for row in inventory})!=len(inventory):raise RuntimeError("stored tree inventory")
    lines="".join(f"{digest}  {relative}\n" for relative,digest,_ in inventory).encode()
    expected={"file_count":len(inventory),"logical_file_bytes":sum(row[2] for row in inventory),"sha256sum_lines_digest_sha256":hashlib.sha256(lines).hexdigest(),"canonical_json_digest_sha256":canonical_sha(inventory)}
    if any(payload[f"{prefix}_{key}"]!=value for key,value in expected.items()):raise RuntimeError("stored tree digest")
def atomic_json(path,payload):
    path=Path(path);tmp=path.with_name(path.name+".tmp")
    if path.exists() or tmp.exists():raise FileExistsError(path)
    with tmp.open("x") as f:json.dump(payload,f,sort_keys=True,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);fd=os.open(str(path.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)

def actual_source_closure(formal,expected_sources):
    exact_roles=formal["phase_a_output_and_receipt_schema"]["source_closure_contract"]["exact_roles"]
    if set(expected_sources)!=set(exact_roles):raise RuntimeError("exact seven source roles")
    records=[]
    for role in exact_roles:
        lexical=Path(expected_sources[role])
        if lexical.is_symlink() or not lexical.is_file():raise RuntimeError(f"source nonregular {role}")
        path=lexical.resolve(strict=True)
        record={"role":role,"path":str(path),"sha256":sha(path),"logical_bytes":path.stat().st_size}
        if formal.get("execution_sources",{}).get(role)!={key:value for key,value in record.items() if key!="role"}:raise RuntimeError(f"formal source {role}")
        records.append(record)
    digest=canonical_sha(records)
    if records!=formal.get("execution_source_records") or digest!=formal.get("execution_sources_digest_sha256"):raise RuntimeError("source records/digest")
    return records,digest


def run(command,timeout,log_path=None):
    if log_path is None:
        result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=timeout,start_new_session=True)
    else:
        log_path=Path(log_path)
        if log_path.exists():raise FileExistsError(log_path)
        with log_path.open("xb") as stream:
            result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout,start_new_session=True)
            stream.flush();os.fsync(stream.fileno())
    if result.returncode:raise RuntimeError({"command":command,"returncode":result.returncode})

def fsync_tree(root):
    root=Path(root)
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):raise RuntimeError("completion tree nonregular")
        if path.is_file():
            fd=os.open(str(path),os.O_RDONLY);os.fsync(fd);os.close(fd)
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()),key=lambda value:len(value.parts),reverse=True):
        fd=os.open(str(directory),os.O_RDONLY);os.fsync(fd);os.close(fd)
    fd=os.open(str(root),os.O_RDONLY);os.fsync(fd);os.close(fd)

def prepare_worker_log(partial,opener=None):
    partial=Path(partial);partial.mkdir();log_path=partial/"process.log"
    try:return log_path.open("xb",buffering=0) if opener is None else opener(log_path)
    except Exception as error:
        try:
            mode="ab" if log_path.is_file() and not log_path.is_symlink() else "xb"
            with log_path.open(mode,buffering=0) as fallback:
                fallback.write((json.dumps({"event":"driver_log_setup_failure","error_type":type(error).__name__,"error":str(error)},sort_keys=True)+"\n").encode());fallback.flush();os.fsync(fallback.fileno())
        except Exception:
            if partial.is_dir() and not partial.is_symlink() and not any(partial.iterdir()):partial.rmdir()
            raise
        raise WorkerLogSetupError(str(error)) from error

def run_worker(command,timeout,root,role,intent,log_opener=None):
    role_name="process_a" if role=="A" else "process_b";partial=root/(role_name+".partial");final=root/role_name
    if partial.exists() or final.exists():raise FileExistsError(final)
    wait_started=time.monotonic_ns();timed_out=False;returncode=None;process=None;launch_error=None;stream=None
    try:stream=prepare_worker_log(partial,log_opener)
    except WorkerLogSetupError as error:launch_error=error
    log_path=partial/"process.log"
    if stream is not None:
      with stream:
        try:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:returncode=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out=True
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:returncode=process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    returncode=process.wait(timeout=30)
        except Exception as error:
            launch_error=error
            if process is not None and process.poll() is None:
                try:os.killpg(process.pid,signal.SIGKILL)
                except Exception:pass
                try:returncode=process.wait(timeout=30)
                except Exception:returncode=None
        stream.flush();os.fsync(stream.fileno())
    wait_finished=time.monotonic_ns();base=final if final.is_dir() and not final.is_symlink() else partial
    if not base.is_dir() or base.is_symlink():raise RuntimeError("worker evidence directory")
    actual_log=base/"process.log";final_log=final/"process.log"
    actual_identity=base/"identity.json";actual_worker_receipt=base/"receipt.json";final_identity=final/"identity.json";final_worker_receipt=final/"receipt.json"
    identity_ok=actual_identity.is_file() and not actual_identity.is_symlink();worker_receipt_ok=actual_worker_receipt.is_file() and not actual_worker_receipt.is_symlink()
    worker_passed=False
    if worker_receipt_ok:
        try:
            worker_doc=json.loads(actual_worker_receipt.read_text());worker_passed=worker_doc.get("passed") is True and worker_doc.get("role")==role
        except Exception:worker_passed=False
    if launch_error is not None and process is None:exit_type="spawn_error";exit_code=None;exit_signal=None;child_completed=False
    elif timed_out:exit_type="timeout";exit_code=None;exit_signal=None;child_completed=process is not None and process.poll() is not None
    elif isinstance(returncode,int) and returncode<0:exit_type="signal";exit_code=None;exit_signal=-returncode;child_completed=True
    elif isinstance(returncode,int) and returncode>0:exit_type="nonzero_exit";exit_code=returncode;exit_signal=None;child_completed=True
    elif returncode==0:exit_type="clean_exit";exit_code=0;exit_signal=None;child_completed=True
    else:exit_type="spawn_error";exit_code=None;exit_signal=None;child_completed=False
    worker_passed=bool(worker_passed and exit_type=="clean_exit")
    status="passed" if identity_ok and worker_receipt_ok and child_completed and worker_passed else "failed"
    authorities={"cache_reuse_authorized":False,"phase_b_preregistration_authorized":False,"training_authorized":False,"folds_authorized":0,"policy_updates":0,"s1_authorized":False,"zero_update_authorized":False,"rl_authorized":False,"submission_authorized":False,"retry_authorized":False}
    completion={"format":"strict-track2-v485-v482-v169-cache-phase-a-driver-completion-v1","status":status,"role":role_name,"intent_path":str((root/"attempt_intent.json").resolve()),"intent_sha256":sha(root/"attempt_intent.json"),"attempt_nonce":intent["attempt_nonce"],"identity_present":identity_ok,"identity_path":str(final_identity.resolve()) if identity_ok else None,"identity_sha256":sha(actual_identity) if identity_ok else None,"worker_receipt_present":worker_receipt_ok,"worker_receipt_path":str(final_worker_receipt.resolve()) if worker_receipt_ok else None,"worker_receipt_sha256":sha(actual_worker_receipt) if worker_receipt_ok else None,"exit_type":exit_type,"exit_code":exit_code,"signal":exit_signal,"wait_seconds":float((wait_finished-wait_started)/1e9),"process_log_path":str(final_log.resolve()),"process_log_sha256":sha(actual_log),"process_log_logical_bytes":actual_log.stat().st_size,"child_completed":child_completed,"worker_passed":worker_passed,"authorities":authorities}
    if status=="passed":
        fd=os.open(str(root),os.O_RDONLY);os.fsync(fd);os.close(fd)
    atomic_json(base/"completion_receipt.json",completion)
    if status!="passed":
        fsync_tree(base)
        if base==partial:
            os.replace(partial,final);fd=os.open(str(root),os.O_RDONLY);os.fsync(fd);os.close(fd)
        raise RuntimeError({"worker_role":role,"exit_type":exit_type,"exit_code":exit_code,"signal":exit_signal,"launch_error":None if launch_error is None else repr(launch_error),"completion_receipt":str((final/"completion_receipt.json").resolve())})
    return completion

def synthetic_run_self_test():
    with tempfile.TemporaryDirectory(prefix="v485-driver-synthetic-") as directory:
        root=Path(directory);fake=root/"fake_auditor.py"
        fake.write_text("import argparse\np=argparse.ArgumentParser();p.add_argument('--stage',required=True);a,_=p.parse_known_args();print(a.stage)\n",encoding="utf-8")
        observed=[]
        for stage in ("precleanup","final"):
            log=root/f"{stage}.log";run([sys.executable,str(fake),"--stage",stage],30,log);observed.append(log.read_text().strip())
        worker=root/"fake_worker.py"
        worker.write_text("import argparse,json,os,time\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--role');p.add_argument('--fail',action='store_true');p.add_argument('--sleep',action='store_true');a=p.parse_args();n='process_a' if a.role=='A' else 'process_b';q=a.root/(n+'.partial');print('worker-'+a.role,flush=True)\nif a.sleep:time.sleep(60)\nif a.fail:raise SystemExit(7)\n(q/'identity.json').write_text('{}');(q/'receipt.json').write_text(json.dumps({'passed':True,'role':a.role}));os.replace(q,a.root/n)\n",encoding="utf-8")
        success_root=root/"success";success_root.mkdir();atomic_json(success_root/"attempt_intent.json",{"attempt_nonce":"1"*64})
        success=run_worker([sys.executable,str(worker),"--root",str(success_root),"--role","A"],30,success_root,"A",{"attempt_nonce":"1"*64})
        failure_root=root/"failure";failure_root.mkdir();atomic_json(failure_root/"attempt_intent.json",{"attempt_nonce":"2"*64})
        failed=False
        try:run_worker([sys.executable,str(worker),"--root",str(failure_root),"--role","B","--fail"],30,failure_root,"B",{"attempt_nonce":"2"*64})
        except RuntimeError:failed=True
        failure=json.loads((failure_root/"process_b/completion_receipt.json").read_text())
        timeout_root=root/"timeout";timeout_root.mkdir();atomic_json(timeout_root/"attempt_intent.json",{"attempt_nonce":"3"*64});timed_out=False
        try:run_worker([sys.executable,str(worker),"--root",str(timeout_root),"--role","A","--sleep"],0.1,timeout_root,"A",{"attempt_nonce":"3"*64})
        except RuntimeError:timed_out=True
        timeout_receipt=json.loads((timeout_root/"process_a/completion_receipt.json").read_text())
        open_root=root/"open_failure";open_root.mkdir();atomic_json(open_root/"attempt_intent.json",{"attempt_nonce":"4"*64});open_failed=False
        try:run_worker([sys.executable,"unused"],30,open_root,"B",{"attempt_nonce":"4"*64},lambda _:(_ for _ in ()).throw(PermissionError("synthetic log open")))
        except RuntimeError:open_failed=True
        open_receipt=json.loads((open_root/"process_b/completion_receipt.json").read_text())
        return observed==["precleanup","final"] and success["status"]=="passed" and failed and failure["status"]=="failed" and failure["exit_type"]=="nonzero_exit" and failure["exit_code"]==7 and timed_out and timeout_receipt["status"]=="failed" and timeout_receipt["exit_type"]=="timeout" and timeout_receipt["exit_code"] is None and timeout_receipt["child_completed"] is True and open_failed and not (open_root/"process_b.partial").exists() and open_receipt["status"]=="failed" and open_receipt["exit_type"]=="spawn_error" and open_receipt["identity_present"] is False and open_receipt["worker_receipt_present"] is False


def main():
    p=argparse.ArgumentParser()
    for name in ("preregistration","contract","authority-receipt","scope-source","worker-source","auditor-source","materializer-source","static-auditor-source","launcher-source"):p.add_argument(f"--{name}",type=Path,required=True)
    for name in ("preregistration-sha","authority-receipt-sha","scope-sha","worker-sha","auditor-sha","materializer-sha","static-auditor-sha","launcher-sha"):p.add_argument(f"--{name}",required=True)
    args=p.parse_args()
    hashes=[args.preregistration_sha,args.authority_receipt_sha,args.scope_sha,args.worker_sha,args.auditor_sha,args.materializer_sha,args.static_auditor_sha,args.launcher_sha]
    if any(len(x)!=64 or x.startswith("PENDING") for x in hashes):raise RuntimeError("v485 pending closure: zero state")
    actual=((args.preregistration,args.preregistration_sha),(args.contract,CONTRACT_SHA),(args.authority_receipt,args.authority_receipt_sha),(args.scope_source,args.scope_sha),(args.worker_source,args.worker_sha),(args.auditor_source,args.auditor_sha),(args.materializer_source,args.materializer_sha),(args.static_auditor_source,args.static_auditor_sha),(args.launcher_source,args.launcher_sha))
    if any(not path.is_file() or path.is_symlink() or sha(path)!=want for path,want in actual):raise RuntimeError("v485 source/formal closure")
    formal=json.loads(args.preregistration.read_text());authority=json.loads(args.authority_receipt.read_text())
    if (formal.get("format")!="strict-track2-v485-v482-v169-cache-determinism-qualification-preregistration-v1" or formal.get("authorization",{}).get("phase_a_cache_qualification_authorized") is not False or formal.get("authorization",{}).get("attempts_authorized")!=0 or formal.get("authorization",{}).get("training_authorized") is not False or formal.get("authorization",{}).get("folds_authorized")!=0 or formal.get("authorization",{}).get("policy_updates")!=0):raise RuntimeError("nonauthorizing formal")
    formal_root=Path(formal["qualification_output_root"])
    if not formal_root.is_absolute() or formal_root.resolve()!=formal_root or formal_root.parent.is_symlink():raise RuntimeError("qualification output root")
    if authority.get("passed") is not True or authority.get("phase_a_cache_qualification_authorized") is not True or authority.get("training_authorized") is not False or authority.get("preregistration_sha256")!=args.preregistration_sha or authority.get("contract_sha256")!=CONTRACT_SHA or Path(authority.get("qualification_output_root","")).resolve()!=formal_root.resolve():raise RuntimeError("poststatic authority")
    expected_sources={"phase_a_materializer":args.materializer_source,"phase_a_driver":Path(__file__),"phase_a_process_worker":args.worker_source,"phase_a_cache_scope_helper":args.scope_source,"phase_a_independent_auditor":args.auditor_source,"phase_a_static_auditor":args.static_auditor_source,"phase_a_launcher":args.launcher_source}
    initial_source_records,initial_source_digest=actual_source_closure(formal,expected_sources)
    global FAILURE_ROOT,CURRENT_STAGE,INTENT_BINDING,TERMINAL_COMMITTED
    root=formal_root;prep=root.with_name(root.name+".attempt-prep")
    if root.exists() or prep.exists():raise FileExistsError(root)
    prep.mkdir(parents=True)
    executable=Path(sys.executable).resolve(strict=True);interpreter=formal["execution_interpreter"]
    if interpreter["resolved_path"]!=str(executable) or interpreter["resolved_sha256"]!=sha(executable) or interpreter["resolved_bytes"]!=executable.stat().st_size or interpreter["python_version"]!=platform.python_version():raise RuntimeError("execution interpreter")
    nonce=secrets.token_hex(32)
    intent={"format":"strict-track2-v485-v169-cache-qualification-attempt-intent-v1","preregistration_path":str(args.preregistration.resolve()),"preregistration_sha256":args.preregistration_sha,"contract_path":str(args.contract.resolve()),"contract_sha256":CONTRACT_SHA,"authority_receipt_path":str(args.authority_receipt.resolve()),"authority_receipt_sha256":args.authority_receipt_sha,"qualification_output_root":str(root.resolve()),"roles":{"process_a":"canonical_0_to_999","process_b":"reverse_999_to_0"},"expected_identity_relative_paths":{"process_a":"process_a/identity.json","process_b":"process_b/identity.json"},"attempt_nonce":nonce,"sys_executable_resolved_path":str(executable),"sys_executable_sha256":sha(executable),"execution_interpreter":interpreter,"worker_source_sha256":args.worker_sha,"identity_schema_version":"strict-track2-v485-v169-cache-worker-identity-v2","execution_source_records":initial_source_records,"execution_sources_digest_sha256":initial_source_digest,"driver_source_sha256":sha(Path(__file__)),"scope_source_sha256":args.scope_sha,"auditor_source_sha256":args.auditor_sha,"start_ns":time.time_ns(),"one_shot":True,"retry_authorized":False,"training_authorized":False,"folds_authorized":0,"policy_updates":0,"rl_authorized":False}
    atomic_json(prep/"attempt_intent.json",intent);fd=os.open(str(prep),os.O_RDONLY);os.fsync(fd);os.close(fd);os.replace(prep,root)
    FAILURE_ROOT=root;CURRENT_STAGE="intent_parent_fsync";INTENT_BINDING={"intent_path":str((root/"attempt_intent.json").resolve()),"intent_sha256":sha(root/"attempt_intent.json"),"attempt_nonce":nonce}
    fd=os.open(str(root.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
    parent=Path(formal["immutable_failed_parent"]["v482_r3_formal"]["path"]);failure=Path(formal["immutable_failed_parent"]["launcher_failure_receipt"]["path"])
    base=[sys.executable,str(args.worker_source),"--preregistration",str(args.preregistration),"--preregistration-sha",args.preregistration_sha,"--parent-preregistration",str(parent),"--parent-failure",str(failure),"--contract",str(args.contract),"--contract-sha",CONTRACT_SHA,"--authority-receipt",str(args.authority_receipt),"--authority-receipt-sha",args.authority_receipt_sha,"--scope-source",str(args.scope_source),"--scope-sha",args.scope_sha,"--driver-source",str(Path(__file__)),"--driver-sha",sha(Path(__file__)),"--worker-sha",args.worker_sha,"--auditor-source",str(args.auditor_source),"--auditor-sha",args.auditor_sha]
    CURRENT_STAGE="process_a";completion_a=run_worker(base+["--role","A"],3600,root,"A",intent)
    if not (root/"process_a/identity.json").is_file() or not (root/"process_a/completion_receipt.json").is_file() or completion_a["status"]!="passed" or (root/"failure_receipt.json").exists():raise RuntimeError("process A terminal state")
    CURRENT_STAGE="process_b";completion_b=run_worker(base+["--role","B"],3600,root,"B",intent)
    if not (root/"process_b/identity.json").is_file() or not (root/"process_b/completion_receipt.json").is_file() or completion_b["status"]!="passed" or (root/"failure_receipt.json").exists():raise RuntimeError("process B terminal state")
    audit_base=[sys.executable,str(args.auditor_source),"--preregistration",str(args.preregistration),"--preregistration-sha",args.preregistration_sha,"--authority-receipt",str(args.authority_receipt),"--authority-receipt-sha",args.authority_receipt_sha]
    CURRENT_STAGE="precleanup_audit";run(audit_base+["--stage","precleanup"],1800,root.parent/(root.name+".precleanup_audit.log"))
    CURRENT_STAGE="final_audit";run(audit_base+["--stage","final"],1800,root.parent/(root.name+".final_audit.log"))
    final_path=root/"independent_final_audit.json";final=json.loads(final_path.read_text())
    if final.get("passed") is not True:raise RuntimeError("final audit")
    validate_tree_fields(final,"terminal_base_tree")
    pre_report=tree_evidence(root)
    terminal_base_inventory=final["terminal_base_tree_inventory"]
    if pre_report["inventory"]!=sorted(terminal_base_inventory+[["independent_final_audit.json",sha(final_path),final_path.stat().st_size]],key=lambda row:row[0]):raise RuntimeError("stage4->stage5 tree")
    CURRENT_STAGE="finalize_report";report_source_records,report_source_digest=actual_source_closure(formal,expected_sources)
    report={"format":"strict-track2-v485-v169-cache-qualification-report-v1","passed":True,"preregistration_path":str(args.preregistration.resolve()),"preregistration_sha256":args.preregistration_sha,"contract_sha256":CONTRACT_SHA,"postregistration_authority_path":str(args.authority_receipt.resolve()),"postregistration_authority_sha256":args.authority_receipt_sha,"actual_execution_source_records":report_source_records,"actual_execution_sources_digest_sha256":report_source_digest,"attempt_intent_sha256":sha(root/"attempt_intent.json"),"process_a_receipt_sha256":sha(root/"process_a/receipt.json"),"process_b_receipt_sha256":sha(root/"process_b/receipt.json"),"process_a_log":{"path":str((root/"process_a/process.log").resolve()),"sha256":sha(root/"process_a/process.log"),"logical_bytes":(root/"process_a/process.log").stat().st_size},"process_b_log":{"path":str((root/"process_b/process.log").resolve()),"sha256":sha(root/"process_b/process.log"),"logical_bytes":(root/"process_b/process.log").stat().st_size},"process_a_call_events_sha256":sha(root/"process_a/call_events.ndjson"),"process_b_call_events_sha256":sha(root/"process_b/call_events.ndjson"),"precleanup_audit_sha256":sha(root/"independent_precleanup_audit.json"),"cleanup_receipt_sha256":sha(root/"process_b_cleanup_receipt.json"),"final_audit_sha256":sha(final_path),**tree_fields("pre_report_tree",pre_report),"qualified_cache_reconciliation_review_authorized":True,"training_authorized":False,"cache_reuse_authorized":False,"folds":0,"policy_updates":0,"s1_authorized":False,"zero_update_authorized":False,"rl_authorized":False}
    atomic_json(root/"qualification_report.json",report)
    pre_terminal=tree_evidence(root);report_path=root/"qualification_report.json"
    if pre_terminal["inventory"]!=sorted(pre_report["inventory"]+[["qualification_report.json",sha(report_path),report_path.stat().st_size]],key=lambda row:row[0]):raise RuntimeError("stage5->stage6 tree")
    expected=set(formal["phase_a_output_and_receipt_schema"]["required_relative_outputs"])-{"terminal_receipt.json"}
    if {row[0] for row in pre_terminal["inventory"]}!=expected or (root/"process_b/baseline_canonical.memmap").exists() or (root/"failure_receipt.json").exists():raise RuntimeError("preterminal exact membership")
    CURRENT_STAGE="finalize_terminal";terminal_source_records,terminal_source_digest=actual_source_closure(formal,expected_sources)
    terminal={"format":"strict-track2-v485-v169-cache-qualification-terminal-receipt-v1","passed":True,"preregistration_sha256":args.preregistration_sha,"contract_sha256":CONTRACT_SHA,"postregistration_authority_sha256":args.authority_receipt_sha,"qualification_report_sha256":sha(report_path),"final_audit_sha256":sha(final_path),"actual_execution_source_records":terminal_source_records,"actual_execution_sources_digest_sha256":terminal_source_digest,"only_atomic_write_after_inventory":"terminal_receipt.json","terminal_receipt_excluded_from_pre_terminal_tree":True,**tree_fields("pre_terminal_tree",pre_terminal),"retry_authorized":False,"qualified_cache_reconciliation_review_authorized":True,"training_authorized":False,"cache_reuse_authorized":False,"folds":0,"policy_updates":0,"rl_authorized":False}
    atomic_json(root/"terminal_receipt.json",terminal);TERMINAL_COMMITTED=True;return 0


if __name__=="__main__":
    if "--synthetic-self-test" in sys.argv:
        passed=synthetic_run_self_test();print(json.dumps({"passed":passed,"stages":["precleanup","final"]},sort_keys=True));raise SystemExit(0 if passed else 3)
    try: raise SystemExit(main())
    except Exception as error:
        terminal_path=FAILURE_ROOT/"terminal_receipt.json" if FAILURE_ROOT is not None else None
        try:
            terminal_doc=json.loads(terminal_path.read_text()) if terminal_path is not None and terminal_path.is_file() and not terminal_path.is_symlink() else {}
            terminal_exists=terminal_doc.get("format")=="strict-track2-v485-v169-cache-qualification-terminal-receipt-v1" and terminal_doc.get("passed") is True
        except Exception:terminal_exists=False
        if FAILURE_ROOT is not None and not TERMINAL_COMMITTED and not terminal_exists and not (FAILURE_ROOT/"failure_receipt.json").exists():
            evidence=[]
            for path in sorted(FAILURE_ROOT.rglob("*")):
                if path.is_symlink():evidence.append({"relative_path":path.relative_to(FAILURE_ROOT).as_posix(),"kind":"symlink_forbidden"})
                elif path.is_file():evidence.append({"relative_path":path.relative_to(FAILURE_ROOT).as_posix(),"sha256":sha(path),"logical_bytes":path.stat().st_size})
            completion_receipts=[]
            for role_name in ("process_a","process_b"):
                candidates=(FAILURE_ROOT/role_name/"completion_receipt.json",FAILURE_ROOT/(role_name+".partial")/"completion_receipt.json")
                present=[path for path in candidates if path.is_file() and not path.is_symlink()]
                if len(present)>1:raise RuntimeError("duplicate completion receipt")
                if present:completion_receipts.append({"role":role_name,"path":str(present[0].resolve()),"sha256":sha(present[0]),"logical_bytes":present[0].stat().st_size})
            failure_tree=tree_evidence(FAILURE_ROOT)
            atomic_json(FAILURE_ROOT/"failure_receipt.json",{"format":"strict-track2-v485-v169-cache-qualification-failure-receipt-v1","status":"failed_no_retry","passed":False,"stage":CURRENT_STAGE,**(INTENT_BINDING or {}),"completion_receipts":completion_receipts,"error_type":type(error).__name__,"error":str(error),"existing_files":evidence,**{f"failure_state_tree_{key}":value for key,value in failure_tree.items()},"retry_authorized":False,"phase_a_cache_qualification_authorized":False,"training_authorized":False,"cache_reuse_authorized":False,"folds":0,"policy_updates":0,"rl_authorized":False})
        raise
