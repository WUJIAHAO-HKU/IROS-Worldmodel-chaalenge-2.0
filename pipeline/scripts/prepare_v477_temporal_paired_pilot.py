#!/usr/bin/env python3
"""Materialize the frozen four-context v477 temporal paired-simulator pilot."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
import h5py,numpy as np

FORMAT="strict-track2-v477-public-train-paired-temporal8-pilot-preregistration-v1"
CONTRACT_FORMAT="strict-track2-v477-public-train-paired-temporal8-pilot-contract-v1"
CONTRACT_SHA="9fbb494f55115786b2b22e77d91c172ec1a52df77cec224083ff6e82391e53a1"
SELECTION_SHA="edf2ea5e91c7b6a464c62a5d3c042c11bc69d4ca851cc03c4879cad3dcf55919"
MANIFEST_SHA="49b8f8b10732355d327ca8d83d2b22e16a2b8cd2ed92e7ab07dc3445000ed218"
REPORT_SHA="f128bd5e8b6c3b976033c7cc0da4afb0c602fb8e6383a5d6ea7dfb3cf8b3cde5"
VARIANTS=("factual","no_transport","scale_0p4","scale_1p25","reverse_direction_0p4","factual_duplicate")

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()
def support_tree(root):
 root=Path(root).resolve();paths=sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts and p.suffix.lower() in (".py",".yml",".yaml"));items=[[p.relative_to(root).as_posix(),sha(p)] for p in paths]
 return hashlib.sha256(json.dumps(items,sort_keys=True,separators=(",",":")).encode()).hexdigest(),len(items)
def branches(history,future):
 anchor=history[-1,7:13];delta=future[:,7:13]-anchor;out={}
 for name,scale in (("no_transport",0.),("scale_0p4",.4),("scale_1p25",1.25),("reverse_direction_0p4",-.4)):
  value=future.copy();value[:,7:13]=anchor+scale*delta;out[name]=value
 return {"factual":future.copy(),**out,"factual_duplicate":future.copy()}
def atomic(path,obj):
 path=Path(path);tmp=path.with_name(path.name+".tmp")
 if path.exists() or tmp.exists():raise FileExistsError(path)
 with tmp.open("x",encoding="utf-8") as f:json.dump(obj,f,sort_keys=True,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);fd=os.open(str(path.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
def main():
 ap=argparse.ArgumentParser()
 for name in ("contract","selection","generation-manifest","generation-report","v461-dataset","windows","collector","generator","auditor","resize-source","support-root","task-config","output"):ap.add_argument("--"+name,required=True,type=Path)
 a=ap.parse_args();contract=json.loads(a.contract.read_text());selection=json.loads(a.selection.read_text());manifest=json.loads(a.generation_manifest.read_text());generation=json.loads(a.generation_report.read_text())
 if sha(a.contract)!=CONTRACT_SHA or contract.get("format")!=CONTRACT_FORMAT or contract.get("status")!="frozen_design_contract_unexecuted":raise RuntimeError("v477 contract drift")
 if sha(a.selection)!=SELECTION_SHA or sha(a.generation_manifest)!=MANIFEST_SHA or sha(a.generation_report)!=REPORT_SHA or generation.get("passed") is not True:raise RuntimeError("v461 ancestry drift")
 selected={(int(x["episode"]),int(x["start"])):x for x in selection["contexts"]};specs=[]
 for fixed in contract["pilot_selection_rule"]["fixed_contexts"]:
  key=(int(fixed["episode"]),int(fixed["start"]));row=selected.get(key)
  if row is None:raise RuntimeError(f"missing v477 context {key}")
  for name in ("fold","phase_bin","motion_bin","cost_rank","batch_id","dataset_seed","source_hdf5_sha256","history_action_sha256"):
   if row[name]!=fixed[name]:raise RuntimeError(f"v477 fixed metadata drift {key} {name}")
  source=Path(row["source_hdf5"]).resolve();window=(a.windows/f"episode{key[0]}_{key[1]:05d}.npz").resolve();oracle=(a.v461_dataset/f"batch_{int(row['batch_id']):03d}"/"rows"/f"episode{key[0]}_start{key[1]:05d}"/"endpoint.npz").resolve()
  if window!=Path(fixed["source_window"]).resolve() or oracle!=Path(fixed["v461_endpoint_npz_path"]).resolve() or sha(source)!=fixed["source_hdf5_sha256"] or sha(window)!=fixed["source_window_sha256"] or sha(oracle)!=fixed["v461_endpoint_npz_sha256"]:raise RuntimeError(f"v477 source/window/oracle drift {key}")
  with h5py.File(source,"r") as h5:actions=np.asarray(h5["joint_action/vector"],np.float32)
  history=actions[key[1]:key[1]+4].copy();future=actions[key[1]+4:key[1]+12].copy();branch=branches(history,future)
  if arrsha(history)!=fixed["history_action_sha256"] or arrsha(future)!=fixed["factual_future8_action_sha256"] or any(arrsha(branch[n])!=row["branch_action_sha256"][n] for n in VARIANTS):raise RuntimeError(f"v477 action SHA drift {key}")
  with np.load(window,allow_pickle=False) as z:
   if not np.array_equal(np.asarray(z["history_actions"],np.float32),history) or not np.array_equal(np.asarray(z["future_actions"],np.float32),future) or np.asarray(z["target_frames"]).shape!=(8,256,256,3):raise RuntimeError(f"v477 public diagnostic alignment {key}")
  with np.load(oracle,allow_pickle=False) as z:
   if not np.array_equal(z["history_actions"],history) or not np.array_equal(z["future_actions"],np.stack([branch[n] for n in VARIANTS])):raise RuntimeError(f"v477 endpoint action oracle {key}")
   hashes={"context_rgb_sha256":arrsha(z["branch_context_rgb"][0]),"context_state_sha256":arrsha(z["branch_context_state"][0]),"context_pose_sha256":arrsha(z["branch_context_pose"][0]),"context_bottle_sha256":arrsha(z["branch_context_bottle_position"][0])}
  if any(hashes[n]!=fixed[n] for n in hashes):raise RuntimeError(f"v477 context oracle hash {key}")
  specs.append({**row,"source_hdf5":str(source),"source_hdf5_sha256":sha(source),"source_window":str(window),"source_window_sha256":sha(window),"v461_endpoint_npz":str(oracle),"v461_endpoint_npz_sha256":sha(oracle),"context_oracle_sha256":hashes})
 if [(x["episode"],x["start"]) for x in specs]!=[(25,70),(49,79),(40,118),(28,121)]:raise RuntimeError("v477 fixed order")
 support_digest,support_count=support_tree(a.support_root)
 closure={"prepare":{"path":str(Path(__file__).resolve()),"sha256":sha(Path(__file__).resolve())},"collector":{"path":str(a.collector.resolve()),"sha256":sha(a.collector)},"generator":{"path":str(a.generator.resolve()),"sha256":sha(a.generator)},"auditor":{"path":str(a.auditor.resolve()),"sha256":sha(a.auditor)},"resize_source":{"path":str(a.resize_source.resolve()),"sha256":sha(a.resize_source)},"support_root":{"path":str(a.support_root.resolve()),"source_config_tree_sha256":support_digest,"source_config_file_count":support_count,"tree_algorithm":"canonical JSON sorted [relative POSIX path,file SHA256] for *.py/*.yml/*.yaml excluding .git"},"task_config":{"path":str(a.task_config.resolve()),"sha256":sha(a.task_config)}}
 payload={"format":FORMAT,"status":"preregistered_public_train_paired_temporal8_pilot_authorized","seed":1621,"classification":contract["classification"],"contract":{"path":str(a.contract.resolve()),"sha256":sha(a.contract)},"ancestry":{"selection":{"path":str(a.selection.resolve()),"sha256":sha(a.selection)},"generation_manifest":{"path":str(a.generation_manifest.resolve()),"sha256":sha(a.generation_manifest)},"generation_report":{"path":str(a.generation_report.resolve()),"sha256":sha(a.generation_report)},"v461_dataset":str(a.v461_dataset.resolve())},"contexts":specs,"branches":contract["branches"],"collector_semantics":contract["collector_semantics"],"output_schema":contract["output_schema"],"gates":contract["frozen_pilot_gates"],"resources":contract["resources"],"per_dim_action_lower":selection["per_dim_action_lower"],"per_dim_action_upper":selection["per_dim_action_upper"],"execution_closure":closure,"runtime_provenance_expected":manifest["endpoint_collector_runtime_provenance"],"evidence_sha256":{k:v["sha256"] for k,v in closure.items() if "sha256" in v},"guards":{**contract["guards"],"technical_intervention_effect_whole_pilot_gate":True,"technical_effect_is_not_task_outcome_selection":True,"all_fixed_rows_branches_retained":True,"public_factual_diagnostic_only":True}}
 atomic(a.output,payload);print(json.dumps({"output":str(a.output),"contexts":len(specs)},sort_keys=True))
if __name__=="__main__":raise SystemExit(main())
