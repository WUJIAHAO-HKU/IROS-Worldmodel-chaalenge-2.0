#!/usr/bin/env python3
"""Read-only pre/post/final audit for the frozen v460 10x20 collector."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
import h5py, numpy as np

EXPECTED_TREE = "cdc982fed2913385915b59c40c3d386f2e50a708844fcad655021bb1b7694040"
EXPECTED_RECEIPT = "790b8954c659c8038e778eecfba6cebdb84f8e5a477c2956d2755e58c7cf24d4"
EXPECTED_V457_PREREG = "a272626c5e9022a9808088ce5458e249e429802f9b3189ddb24f2e93283562eb"
EXPECTED_PREPARE = "961b977ca2720182904c3eb0c83c75e7b6b2f36f6d6d5b9c1419db44ef700395"
EXPECTED_GENERATOR = "340d811c6222854f4a3cb9add93185011ae2b80c079b3a85fcb35c9281c4aee9"
EXPECTED_CONTRACT = "d23af656e4069ba0c09f61e85ef8ce108e433f53bad28f2e2e07cb9f4f4c608b"
EXPECTED_V460_CONTRACT = "925aa0a6843b268725f048cae54f68b4883c781c1d655ed33e85f6ebcbbbf923"
EXPECTED_ENDPOINT_COLLECTOR = "6c64e4a42f273f43b9e64523bf12e4dc5f24618d5673848db6c1582b5547e846"
VARIANTS = ("factual","no_transport","scale_0p4","scale_1p25","reverse_direction_0p4","factual_duplicate")
TRANSPORT = VARIANTS[1:5]
FORBIDDEN_ENDPOINT_FIELDS={"source_window","source_window_sha256","target_frames","public_factual_endpoint_sha256","factual_public_endpoint_rgb_mae","factual_public_endpoint_rgb_mae_diagnostic","reward","success","done","termination","truncation","outcome","policy_action"}
EXPECTED_ENDPOINT_KEYS={"episode","dataset_seed","start","variants","instruction","branch_context_rgb","branch_context_state","branch_context_pose","branch_context_bottle_position","history_actions","future_actions","endpoint_rgb","endpoint_state","endpoint_bottle_position","context_rgb_sha256","context_state_sha256","executed_action_sha256"}

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()
def tree_sha(dataset: Path) -> str:
    files=sorted((*dataset.glob("episode*.npz"),dataset/"generation_report.json"))
    if len(files)!=5 or any(not x.is_file() for x in files):return "invalid"
    lines=sorted(f"{sha(x)}  dataset/{x.name}\n" for x in files)
    return hashlib.sha256("".join(lines).encode()).hexdigest()
def branches(history,future):
    # Factual and duplicate are byte-exact source actions.  Reconstructing them
    # as anchor + 1 * delta changes some float32 low bits and invalidates the
    # source-action digest even though the numerical values are nearly equal.
    anchor=history[-1,7:13];delta=future[:,7:13]-anchor
    factual=future.copy()
    out=[factual]
    for scale in (0.,.4,1.25,-.4):
        x=future.copy();x[:,7:13]=anchor+scale*delta;out.append(x)
    out.append(future.copy())
    return np.stack(out)
def validate_spec_source(spec,dataset):
    hdf5=Path(spec["source_hdf5"]);start=int(spec["start"]);expected=(dataset/"data"/f"episode{int(spec['episode'])}.hdf5").resolve()
    checks={"source_files":hdf5.is_file() and hdf5.resolve()==expected and "source_window" not in spec and "source_window_sha256" not in spec}
    if not checks["source_files"]:return False,checks
    checks["source_hashes"]=sha(hdf5)==spec["source_hdf5_sha256"]
    with h5py.File(hdf5,"r") as h:actions=np.asarray(h["joint_action/vector"],np.float32)
    history=actions[start:start+4];future=actions[start+4:start+12];expected=branches(history,future)
    checks["indexing"]=history.shape==(4,14) and future.shape==(8,14)
    checks["action_hashes"]=arrsha(history)==spec["history_action_sha256"] and [arrsha(x) for x in expected]==[spec["branch_action_sha256"][name] for name in VARIANTS]
    endpoint={name:float(np.linalg.norm(expected[i,-1,7:13]-future[-1,7:13])) for i,name in enumerate(VARIANTS) if name not in ("factual","factual_duplicate")}
    path={name:float(np.linalg.norm(expected[i,:,7:13]-future[:,7:13])) for i,name in enumerate(VARIANTS) if name not in ("factual","factual_duplicate")}
    checks["v461_action_difference"]=endpoint==spec.get("endpoint_diffs") and path==spec.get("path_diffs") and min(endpoint.values())>=.01 and min(path.values())>=.01
    return all(checks.values()),checks
def atomic_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+".tmp")
    with tmp.open("w") as f:json.dump(payload,f,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def process_is_live(pid):
    try:os.kill(int(pid),0);return True
    except (ProcessLookupError,ValueError):return False
    except PermissionError:return True
def generator_pids():
    found=[]
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
     try:parts=[x.decode(errors="replace") for x in cmdline.read_bytes().split(b"\0") if x]
     except OSError:continue
     pid=int(cmdline.parent.name)
     if pid not in (os.getpid(),os.getppid()) and len(parts)>1 and Path(parts[1]).name=="generate_v456_endpoint200_sharded.py":found.append(pid)
    return found
def validate_forensics(root):
    forensic=root/"_forensic_uncommitted";qpaths=sorted(forensic.glob("batch_*/*/quarantine_receipt.json")) if forensic.exists() else [];fpaths=sorted(forensic.glob("batch_*/failures/batch_failure_receipt-*.json")) if forensic.exists() else [];valid=True;details={"quarantine":{},"failures":{}}
    if forensic.exists():
     for batch in forensic.iterdir():
      valid &= batch.is_dir() and batch.name.startswith("batch_") and batch.name[6:].isdigit() and 0<=int(batch.name[6:])<10
      if batch.is_dir():
       for child in batch.iterdir():valid &= child.is_dir() and (child.name=="failures" or (child/"quarantine_receipt.json").is_file())
    for path in qpaths:
     rec=json.loads(path.read_text());actual={str(x.relative_to(path.parent)):sha(x) for x in sorted(path.parent.rglob("*")) if x.is_file() and x!=path};ok=rec.get("format")=="strict-track2-v461-uncommitted-work-quarantine-v1" and rec.get("deleted_or_reused") is False and Path(rec.get("destination","/missing")).resolve()==path.parent.resolve() and rec.get("files_sha256")==actual;valid&=ok;details["quarantine"][str(path.relative_to(root))]=ok
    for path in fpaths:
     rec=json.loads(path.read_text());cleanup=rec.get("cleanup",{});pids=list(map(int,cleanup.get("initial_pids",[])));ok=rec.get("format")=="strict-track2-v461-batch-failure-receipt-v1" and rec.get("deleted_or_overwritten") is False and rec.get("cleanup_passed") is True and cleanup.get("alive_after_cleanup")==[] and not any(process_is_live(pid) for pid in pids);valid&=ok;details["failures"][str(path.relative_to(root))]=ok
    qhash={str(x.relative_to(root)):sha(x) for x in qpaths};fhash={str(x.relative_to(root)):sha(x) for x in fpaths}
    return valid and not generator_pids(),details,qhash,fhash
def source_closure(args,contract,selection):
    receipt=json.loads(args.reconciliation.read_text());v457=json.loads(args.v457_preregistration.read_text())
    immutable=v457.get("immutable_files",{});npzs=v457.get("immutable_npz",[])
    pilot_files={x.name:sha(x) for x in args.pilot_dataset.glob("episode*.npz")}
    split=json.loads(args.split.read_text());seeds=list(map(int,args.seed_file.read_text().split()));train=sorted(map(int,split["train_episodes"]));arms={int(k):v for k,v in split["arm_by_episode"].items()};instructions={int(k):str(v) for k,v in split["episode_to_instruction"].items()};contexts=selection.get("contexts",[]);right=sorted(e for e in train if arms.get(e)=="right");selection_sha=args.selection_sha_file.read_text().strip() if args.selection_sha_file.is_file() else ""
    v460=json.loads(args.v460_contract.read_text())
    checks={
      "prepare_sha":sha(args.prepare)==EXPECTED_PREPARE,
      "generator_sha":sha(args.generator)==EXPECTED_GENERATOR,
      "contract_sha":sha(args.contract)==EXPECTED_CONTRACT,
      "v460_contract_sha":sha(args.v460_contract)==EXPECTED_V460_CONTRACT and v460.get("format")=="strict-track2-v461-endpoint-only-collection-contract-v1",
      "endpoint_collector_sha":sha(args.endpoint_collector)==EXPECTED_ENDPOINT_COLLECTOR==v460.get("collector",{}).get("sha256"),
      "v460_parent_binding":v460.get("selection_parent",{}).get("parent_contract_sha256")==EXPECTED_CONTRACT and v460.get("selection_parent",{}).get("contexts")==200 and v460.get("guards",{}).get("rl_authorized") is False,
      "selection_contract_binding":selection.get("frozen_contract_sha256")==sha(args.contract),
      "selection_sha_lock":selection_sha==sha(args.selection),
      "public_split_seed_sha":selection.get("source_sha256",{}).get("split")==sha(args.split)==contract.get("source_boundary",{}).get("required_public_split_sha256")=="8c1ecfb7127b2d30862420d311d9b3e0f7d1337bec6b4c214cb9db372fc78a87" and selection.get("source_sha256",{}).get("seed_file")==sha(args.seed_file)==contract.get("source_boundary",{}).get("required_public_seed_file_sha256")=="2a9302903b92b6828ad6cb99c0b04af50fcbbf176cdd827ad34abeced04ea419",
      "public_train_right_metadata":len(train)==40 and len(seeds)==50 and right==[12,15,20,25,28,30,32,33,37,40,42,44,46,47,49] and sorted({int(x["episode"]) for x in contexts})==right and all(int(x["episode"]) in train and arms[int(x["episode"])]=="right" and int(x["dataset_seed"])==seeds[int(x["episode"])] and str(x["instruction"])==instructions[int(x["episode"])] for x in contexts),
      "v457_receipt_sha":sha(args.reconciliation)==EXPECTED_RECEIPT==contract["prerequisites"]["required_v457_receipt_sha256"],
      "v457_prereg_sha":sha(args.v457_preregistration)==EXPECTED_V457_PREREG==contract["prerequisites"]["required_v457_preregistration_sha256"],
      "v455_tree_sha":tree_sha(args.pilot_dataset)==EXPECTED_TREE,
      "v457_receipt_pass":receipt.get("passed") is True and receipt.get("endpoint_parent_data_authorized") is True and all(receipt.get("checks",{}).values()) and all(receipt.get("hash_checks",{}).values()),
      "v457_prereg_npz_binding":len(npzs)==4 and {x["name"]:x["sha256"] for x in npzs}==pilot_files,
      "v457_report_binding":immutable.get("v455_generation_report",{}).get("sha256")==sha(args.pilot_dataset/"generation_report.json"),
      "v455_collector_binding":immutable.get("v455_generator",{}).get("sha256")==sha(args.v455_collector),
      "selection_v455_evidence":selection.get("v455_generation_report_sha256")==sha(args.pilot_dataset/"generation_report.json") and selection.get("v455_immutable_npz_sha256")==pilot_files,
      "contract_receipt_npz_binding":contract["prerequisites"]["required_v455_generation_report_sha256"]==sha(args.pilot_dataset/"generation_report.json") and contract["prerequisites"]["required_v455_endpoint_npz_sha256"]==pilot_files,
    }
    return checks
def validate_manifest(args,root):
    path=root/"generation_manifest.json"
    if not path.is_file():return False
    manifest=json.loads(path.read_text());provenance=manifest.get("endpoint_collector_runtime_provenance",{});modules=provenance.get("modules",{})
    return manifest.get("format")=="strict-track2-v461-endpoint200-generation-manifest-v1" and manifest.get("contract_sha256")==sha(args.contract) and manifest.get("v460_contract_sha256")==sha(args.v460_contract) and manifest.get("endpoint_collector_sha256")==sha(args.endpoint_collector) and Path(provenance.get("sys_executable","/missing")).resolve()==Path(sys.executable).resolve() and set(modules)=={"open3d","toppra","mplib","sapien"} and all(Path(record.get("file","/missing")).is_file() and str(record.get("version","")) for record in modules.values())
def validate_row(batch:Path,spec:dict):
    stem=f"episode{int(spec['episode'])}_start{int(spec['start']):05d}";rowdir=batch/"rows"/stem;npz=rowdir/"endpoint.npz";rp=rowdir/"receipt.json"
    if not npz.is_file() or not rp.is_file():return False,{"missing":stem}
    row=json.loads(rp.read_text());checks={"atomic_row_exact":rowdir.is_dir() and {x.name for x in rowdir.iterdir()}=={"endpoint.npz","receipt.json"},"identity":row.get("episode")==spec["episode"] and row.get("start")==spec["start"],"npz_sha":row.get("npz_sha256")==sha(npz),"receipt_action_sha":row.get("selection_action_sha256")==spec["branch_action_sha256"],"receipt_endpoint_only":not set(row).intersection(FORBIDDEN_ENDPOINT_FIELDS) and row.get("endpoint_only") is True and row.get("public_window_or_endpoint_diagnostic_consumed") is False}
    with np.load(npz,allow_pickle=False) as z:
      keys=set(z.files);history=z["history_actions"];future=z["future_actions"];contexts=z["branch_context_rgb"];states=z["branch_context_state"];poses=z["branch_context_pose"];bottles=z["branch_context_bottle_position"];endpoint=z["endpoint_rgb"];endstate=z["endpoint_state"];endbottle=z["endpoint_bottle_position"]
      expected=branches(history,future[0]);effects={};counts=0
      for i,name in enumerate(TRANSPORT,1):
       rgb=float(np.abs(endpoint[i].astype(np.float32)-endpoint[0].astype(np.float32)).mean());q=float(np.linalg.norm(endstate[i]-endstate[0]));o=float(np.linalg.norm(endbottle[i]-endbottle[0]));passed=rgb>=1 and (q>=.01 or o>=.005);effects[name]=passed;counts+=passed
      checks.update({
       "schema":keys==EXPECTED_ENDPOINT_KEYS and list(map(str,z["variants"]))==list(VARIANTS) and history.shape==(4,14) and future.shape==(6,8,14) and contexts.shape==(6,256,256,3) and endpoint.shape==(6,256,256,3) and contexts.dtype==endpoint.dtype==np.uint8 and states.shape[0]==poses.shape[0]==bottles.shape[0]==endstate.shape[0]==endbottle.shape[0]==6,
       "formula":np.allclose(future,expected,rtol=0,atol=1e-7),
       "selection_action":arrsha(history)==spec["history_action_sha256"] and [arrsha(x) for x in future]==[spec["branch_action_sha256"][name] for name in VARIANTS],
       "context_exact":all(np.array_equal(contexts[0],contexts[i]) and np.array_equal(states[0],states[i]) and np.array_equal(poses[0],poses[i]) and np.array_equal(bottles[0],bottles[i]) for i in range(1,6)),
       "context_hashes":list(map(str,z["context_rgb_sha256"]))==[arrsha(x) for x in contexts] and list(map(str,z["context_state_sha256"]))==[arrsha(x) for x in states],
       "executed_hashes":list(map(str,z["executed_action_sha256"]))==[arrsha(x) for x in future],
       "duplicate":float(np.abs(endpoint[0].astype(np.float32)-endpoint[5].astype(np.float32)).mean())<=.5,
       "effect_receipt_exact":counts==row.get("effect_count") and all(effects[name]==row["effects"][name]["passed"] for name in TRANSPORT),
       "forbidden_absent":not keys.intersection(FORBIDDEN_ENDPOINT_FIELDS|{"intermediate_frames"}),
      })
    return all(checks.values()),checks
def validate_batch(path:Path,batch_id:int,specs:list[dict]):
    report_path=path/"batch_report.json"
    if not report_path.is_file():return False,{"report_missing":True},{}
    report=json.loads(report_path.read_text());rows={};effect={name:0 for name in TRANSPORT};context_effect_actual=0
    for spec in specs:
      ok,detail=validate_row(path,spec);rows[f"{spec['episode']}:{spec['start']}"]={"passed":ok,"checks":detail}
      stem=f"episode{int(spec['episode'])}_start{int(spec['start']):05d}";row=json.loads((path/"rows"/stem/"receipt.json").read_text())
      with np.load(path/"rows"/stem/"endpoint.npz",allow_pickle=False) as z:
       endpoint=z["endpoint_rgb"];endstate=z["endpoint_state"];endbottle=z["endpoint_bottle_position"];npz_effects=[]
       for i,name in enumerate(TRANSPORT,1):
        rgb=float(np.abs(endpoint[i].astype(np.float32)-endpoint[0].astype(np.float32)).mean());q=float(np.linalg.norm(endstate[i]-endstate[0]));o=float(np.linalg.norm(endbottle[i]-endbottle[0]));passed=rgb>=1 and (q>=.01 or o>=.005);effect[name]+=int(passed);npz_effects.append(passed)
       context_effect_actual+=int(sum(npz_effects)>=3)
    rowsdir=path/"rows";npzs=list(rowsdir.glob("*/endpoint.npz"));rps=list(rowsdir.glob("*/receipt.json"));size=sum(x.stat().st_size for x in npzs);expected_names={f"episode{int(x['episode'])}_start{int(x['start']):05d}" for x in specs};actual_names={x.name for x in rowsdir.iterdir()} if rowsdir.is_dir() else set();extra={x.name for x in path.iterdir()}-{"rows","batch_report.json"}
    actual_receipts={(int(json.loads(x.read_text())["episode"]),int(json.loads(x.read_text())["start"])):json.loads(x.read_text())["npz_sha256"] for x in rps};report_receipts={(int(x["episode"]),int(x["start"])):x["npz_sha256"] for x in report.get("rows",[])}
    forensic_ok,_forensic_details,qhash,fhash=validate_forensics(path.parent);batch_prefix=f"_forensic_uncommitted/batch_{batch_id:03d}/";batch_q={k:v for k,v in qhash.items() if k.startswith(batch_prefix)};batch_f={k:v for k,v in fhash.items() if k.startswith(batch_prefix)}
    checks={"report_format":report.get("format")=="strict-track2-v461-endpoint200-batch-short-gate-v1","report_pass":report.get("passed") is True,"batch_id":report.get("batch_id")==batch_id,"exact20":len(specs)==len(npzs)==len(rps)==len(report.get("rows",[]))==20,"report_receipts_exact":report_receipts==actual_receipts and set(actual_receipts)=={(int(x["episode"]),int(x["start"])) for x in specs},"atomic_rows_exact":actual_names==expected_names,"no_extra_or_partial":not extra,"forensic_receipts":forensic_ok and report.get("quarantine_receipt_count")==len(batch_q) and report.get("quarantine_receipts_sha256")==batch_q and report.get("failure_receipt_count")==len(batch_f) and report.get("failure_receipts_sha256")==batch_f,"batch_margins":all(sum(int(x["fold"]==v) for x in specs)==4 for v in range(5)) and all(3<=sum(int(x["phase_bin"]==v) for x in specs)<=5 for v in range(5)) and all(4<=sum(int(x["motion_bin"]==v) for x in specs)<=6 for v in range(4)),"rows":all(x["passed"] for x in rows.values()),"contexts_effect_16_of_20":context_effect_actual>=16,"each_transport_effect_14_of_20":effect==report.get("transport_effect_counts") and all(x>=14 for x in effect.values()),"gpu":0<=report.get("gpu_peak_mib",-1)<=24576,"wall":report.get("wall_seconds",1e99)<=900,"bytes":size==report.get("output_bytes") and size<=41943040}
    return all(checks.values()),checks,rows
def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=("pre","post","final"),required=True);p.add_argument("--batch-id",type=int)
    for name in ("selection","selection-sha-file","contract","v460-contract","reconciliation","v457-preregistration","pilot-dataset","prepare","generator","v455-collector","endpoint-collector","split","seed-file","dataset","output-dir","audit-dir","lock-file","receipt-output"):p.add_argument(f"--{name}",type=Path,required=True)
    a=p.parse_args();selection=json.loads(a.selection.read_text());contract=json.loads(a.contract.read_text());root=a.output_dir
    closure=source_closure(a,contract,selection);lock=json.loads(a.lock_file.read_text()) if a.lock_file.is_file() else {};forensic_ok,forensic_details,global_qhash,global_fhash=validate_forensics(root);contexts=selection.get("contexts",[]);episode_counts={e:sum(int(x["episode"]==e) for x in contexts) for e in sorted({int(x["episode"]) for x in contexts})};starts={e:sorted(int(x["start"]) for x in contexts if int(x["episode"])==e) for e in episode_counts};coverage=selection.get("selected_coverage",{});base={"exclusive_launcher_lock_acquired":lock.get("acquired") is True and isinstance(lock.get("pid"),int) and bool(lock.get("run_id")),"forensic_receipts_and_no_live_workers":forensic_ok,"selection_format":selection.get("format")=="strict-track2-v461-endpoint200-preregistration-v1","contract_format":contract.get("format")=="strict-track2-v456-endpoint-residual-parent-frozen-contract-v1","exact10x20":len(contexts)==200 and len(selection.get("batches",[]))==10 and all(len(x["contexts"])==20 for x in selection["batches"]),"selection_unique_nonadjacent":len({(int(x["episode"]),int(x["start"])) for x in contexts})==200 and all(all(b-a>=2 for a,b in zip(value,value[1:])) for value in starts.values()),"selection_fold_margins":all(sum(int(x["fold"]==f) for x in contexts)==40 for f in range(5)) and all(sum(int(x["fold"]==f and x["phase_bin"]==v) for x in contexts)==8 for f in range(5) for v in range(5)) and all(sum(int(x["fold"]==f and x["motion_bin"]==v) for x in contexts)==10 for f in range(5) for v in range(4)),"selection_episode_bounds":len(episode_counts)==15 and min(episode_counts.values())>=6 and max(episode_counts.values())<=17 and max(episode_counts.values())-min(episode_counts.values())==11,"selection_coverage_receipt":coverage.get("episode_counts")=={str(k):v for k,v in episode_counts.items()} and coverage.get("episode_count_min")==min(episode_counts.values()) and coverage.get("episode_count_max")==max(episode_counts.values()) and coverage.get("episode_count_range")==11,"selection_action_difference_gate":selection.get("action_difference_gates")=={"right6_endpoint_l2_min":0.01,"right6_full_path_tensor_l2_min":0.01,"applied_before_binning":True},"no_rl":selection.get("guards",{}).get("rl_authorized") is False and contract.get("guards",{}).get("formal_rl_authorized") is False}
    details={};status="";manifest_exists=(root/"generation_manifest.json").is_file();manifest_ok=validate_manifest(a,root) if manifest_exists else False
    if a.mode in ("pre","post"):
      if a.batch_id is None or not 0<=a.batch_id<10:raise RuntimeError("batch-id 0..9 required")
      specs=sorted((x for x in selection["contexts"] if x["batch_id"]==a.batch_id),key=lambda x:(x["fold"],x["phase_bin"],x["motion_bin"],x["episode"],x["start"]));complete=root/f"batch_{a.batch_id:03d}";partial=root/f"batch_{a.batch_id:03d}.partial";source_rows={};source_ok=True
      for spec in specs:
       ok,detail=validate_spec_source(spec,a.dataset);source_rows[f"{spec['episode']}:{spec['start']}"]={"passed":ok,"checks":detail};source_ok&=ok
      base["batch_source_action_contract"]=source_ok
      base["generation_manifest_provenance"]=manifest_ok if (a.mode=="post" or manifest_exists) else True
      if a.mode=="post":base["batch_work_tree_cleared"]=not (root/"work"/f"batch_{a.batch_id:03d}").exists()
      if complete.exists() and partial.exists():base["batch_state_exclusive"]=False
      elif complete.exists():
       valid,checks,rows=validate_batch(complete,a.batch_id,specs);base["complete_batch_valid"]=valid;details={"source_rows":source_rows,"batch_checks":checks,"rows":rows};status="skip_authorized" if a.mode=="pre" else "post_pass"
      elif a.mode=="post":base["complete_batch_required"]=False;status="post_failed"
      else:
       valid=True;rows={};base["partial_has_no_failed_report_or_workdir"]=not (partial/"batch_report.json").exists() and not any(partial.glob(".*.work"))
       if partial.exists():
        expected_partial_names={f"episode{int(x['episode'])}_start{int(x['start']):05d}" for x in specs};actual_partial_names={x.name for x in (partial/"rows").iterdir()} if (partial/"rows").is_dir() else set();base["partial_known_rows_only"] = actual_partial_names <= expected_partial_names and ({x.name for x in partial.iterdir()} <= {"rows"})
        for spec in specs:
         stem=f"episode{int(spec['episode'])}_start{int(spec['start']):05d}"
         if (partial/"rows"/stem).exists():
          ok,detail=validate_row(partial,spec);rows[stem]={"passed":ok,"checks":detail};valid&=ok
       base["partial_receipt_npz_valid"]=valid;details={"source_rows":source_rows,"rows":rows};status="resume_authorized" if rows else "start_authorized"
    else:
      reports=[];valid=True;source_valid=True;wall_sum=0.
      for bid in range(10):
       specs=sorted((x for x in selection["contexts"] if x["batch_id"]==bid),key=lambda x:(x["fold"],x["phase_bin"],x["motion_bin"],x["episode"],x["start"]));ok,checks,_=validate_batch(root/f"batch_{bid:03d}",bid,specs);valid&=ok;wall_sum+=float(json.loads((root/f"batch_{bid:03d}"/"batch_report.json").read_text()).get("wall_seconds",1e99)) if (root/f"batch_{bid:03d}"/"batch_report.json").is_file() else 1e99
       for spec in specs:source_valid&=validate_spec_source(spec,a.dataset)[0]
       reports.append({"batch_id":bid,"passed":ok,"checks":checks})
      audit_pairs=True
      for bid in range(10):
       pre_receipts=sorted(a.audit_dir.glob(f"batch_{bid:03d}_pre_*.json"));post_receipts=sorted(a.audit_dir.glob(f"batch_{bid:03d}_post_*.json"));audit_pairs&=bool(pre_receipts and post_receipts and any(json.loads(x.read_text()).get("passed") is True for x in pre_receipts) and any(json.loads(x.read_text()).get("passed") is True for x in post_receipts))
      final_path=root/"generation_report.json";final=json.loads(final_path.read_text()) if final_path.is_file() else {};allowed_root={f"batch_{i:03d}" for i in range(10)}|{"generation_manifest.json","generation_report.json","_forensic_uncommitted"};root_extra={x.name for x in root.iterdir()}-allowed_root if root.exists() else {"missing"}
      base.update({"generation_manifest_provenance":manifest_ok,"ten_complete_batches":valid,"all_200_source_actions_rechecked":source_valid,"all_pre_post_audit_pairs_pass":audit_pairs,"no_extra_partial_or_work_batches":not root_extra and not (root/"work").exists(),"sum_batch_wall_le_3h":wall_sum<=10800,"final_forensic_binding":final.get("quarantine_receipt_count")==len(global_qhash) and final.get("quarantine_receipts_sha256")==global_qhash and final.get("failure_receipt_count")==len(global_fhash) and final.get("failure_receipts_sha256")==global_fhash,"final_report":final.get("format")=="strict-track2-v461-endpoint200-generation-report-v1" and final.get("passed") is True and final.get("exact_batches")==10 and final.get("exact_contexts")==200 and final.get("guards",{}).get("rl_authorized") is False});details={"forensics":forensic_details,"batches":reports,"sum_batch_wall_seconds":wall_sum};status="final_pass"
    checks={**closure,**base};passed=all(checks.values());receipt={"format":"strict-track2-v461-endpoint200-batch-audit-v1","created_at":datetime.now(timezone.utc).isoformat(),"mode":a.mode,"batch_id":a.batch_id,"status":status,"passed":passed,"checks":checks,"exclusive_lock":lock,"details":details,"guards":{"simulator_or_collection_code_in_auditor":False,"reward_loaded":False,"model_training":False,"policy_updates":0,"rl_authorized":False}}
    atomic_json(a.receipt_output,receipt);print(json.dumps(receipt,indent=2));return 0 if passed else 2
if __name__=="__main__":raise SystemExit(main())
