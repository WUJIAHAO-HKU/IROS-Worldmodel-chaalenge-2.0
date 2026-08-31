"""Preregister the single-shot public-train v469 closed-form residual KNN S0."""
from __future__ import annotations
import argparse,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path

FORMAT="strict-track2-v469-closed-form-residual-knn-preregistration-v1"
CONTRACT_SHA="e79b8009b07281d2d9cd2a9b4b91a127a4ba366db1a05e38ee1bce00abd8f2e8"
V468_PREREG=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v468_endpoint_residual_seed1616_20260823/preregistration.json")
V468_PREREG_SHA="e5975053c95b2e523702a4dae0988d33a9861caadc2c57294819cac4b409172b"

def sha(path:Path)->str:
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(8<<20),b""):h.update(block)
 return h.hexdigest()
def atomic_json(path:Path,value:dict)->None:
 if path.exists() or path.with_name(path.name+".tmp").exists():raise RuntimeError("v469 refuses prereg overwrite/stale tmp")
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+".tmp")
 with tmp.open("w",encoding="utf-8") as f:json.dump(value,f,indent=2,sort_keys=True);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def main()->int:
 p=argparse.ArgumentParser()
 for n in ("contract","v468-preregistration","selection","generation-report","dataset","probe","auditor","launcher","output"):p.add_argument(f"--{n}",type=Path,required=True)
 a=p.parse_args();contract=json.loads(a.contract.read_text());old=json.loads(a.v468_preregistration.read_text());selection=json.loads(a.selection.read_text());generation=json.loads(a.generation_report.read_text())
 if sha(a.contract)!=CONTRACT_SHA or contract.get("format")!="strict-track2-v469-closed-form-residual-knn-contract-v1" or contract.get("status")!="preregistered_public_train_s0_authorized":raise RuntimeError("v469 frozen contract/status drift")
 if a.v468_preregistration.resolve()!=V468_PREREG or sha(a.v468_preregistration)!=V468_PREREG_SHA or old.get("format")!="strict-track2-v468-canonical-notransport-paired-delta-preregistration-v1":raise RuntimeError("v469 requires immutable v468 closure")
 if selection.get("format")!="strict-track2-v461-endpoint200-preregistration-v1" or len(selection.get("contexts",[]))!=200 or generation.get("passed") is not True or generation.get("exact_contexts")!=200:raise RuntimeError("v469 immutable v461 input drift")
 if sha(a.selection)!=old["v461"]["selection"]["sha256"] or sha(a.generation_report)!=old["v461"]["generation_report"]["sha256"] or a.dataset.resolve()!=Path(old["v461"]["dataset"]).resolve():raise RuntimeError("v469 v461 receipt drift")
 final_path=Path(old["v461"]["final_receipt"]["path"]);final=json.loads(final_path.read_text())
 if sha(final_path)!=old["v461"]["final_receipt"]["sha256"] or final.get("passed") is not True or final.get("mode")!="final" or not all(final.get("checks",{}).values()):raise RuntimeError("v469 v461 final receipt drift")
 files=old["v461"]["files"]
 if len(files)!=410:raise RuntimeError("v469 exact dataset tree")
 for row in files:
  path=a.dataset/row["relative"]
  if not path.is_file() or sha(path)!=row["sha256"]:raise RuntimeError(f"v469 dataset drift: {path}")
 folds=sorted({int(x["fold"]) for x in selection["contexts"]})
 if folds!=list(range(5)) or any(len({int(x["episode"]) for x in selection["contexts"] if int(x["fold"])==f})!=3 for f in folds):raise RuntimeError("v469 outer fold drift")
 value={"format":FORMAT,"created_at":datetime.now(timezone.utc).isoformat(),"seed":1616,"contract":{"path":str(a.contract.resolve()),"sha256":sha(a.contract)},"source":{"prepare_sha256":sha(Path(__file__)),"probe_path":str(a.probe.resolve()),"probe_sha256":sha(a.probe),"auditor_path":str(a.auditor.resolve()),"auditor_sha256":sha(a.auditor),"launcher_path":str(a.launcher.resolve()),"launcher_sha256":sha(a.launcher)},"v468":{"preregistration_path":str(a.v468_preregistration.resolve()),"preregistration_sha256":sha(a.v468_preregistration),"v169":old["v169"],"phase_shuffle":old["phase_shuffle"]},"v461":{"selection_path":str(a.selection.resolve()),"selection_sha256":sha(a.selection),"generation_report_path":str(a.generation_report.resolve()),"generation_report_sha256":sha(a.generation_report),"dataset":str(a.dataset.resolve()),"files":files},"estimator":contract["frozen_estimator"],"features":contract["frozen_features"],"targets":contract["frozen_targets"],"gates":contract["unchanged_s0_gates"],"guards":{"reward_loaded":False,"policy_updates":0,"rl_authorized":False,"dev_final_hidden_used":False,"hyperparameter_sweep":False}}
 atomic_json(a.output,value);print(json.dumps({"passed":True,"format":FORMAT,"contexts":200,"contract_sha256":CONTRACT_SHA},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
