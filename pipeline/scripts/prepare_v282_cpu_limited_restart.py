#!/usr/bin/env python3
"""Amend v278 after an operator restart to cap CPU without changing policy."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
BASE=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge');OFF=BASE/'artifacts/strict_track2_official_20260810'
NAME='v278_v271_fresh_fullbudget_h200_r4_step10_lr2e5_beta001_seed1471_retry2_20260820';RUN=OFF/'runs'/NAME;REG=OFF/'run_registry'/NAME
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def main():
 out=REG/'cpu_limit_restart_amendment.json'
 if out.exists():raise FileExistsError(out)
 prereg_path=REG/'preregistration.json';prereg=json.loads(prereg_path.read_text())
 assert prereg['run_path']==str(RUN);assert not (RUN/'audit/V278_RETRY2_TRAINING_ACCEPTED').exists()
 if list(RUN.glob('**/checkpoints/**/*.pt')):raise RuntimeError('unexpected checkpoint exists')
 files={
  'pretraining_preregistration':prereg_path,
  'interrupted_launcher_log':REG/'interrupted_attempt_1/launcher.log',
  'cpu_limited_runner':BASE/'pipeline/scripts/run_strict_track2_conservative_kl.sh',
  'cpu_limited_launcher':BASE/'pipeline/scripts/launch_v278_retry2_v271_resumable_rl.sh',
  'cpu_limited_services':BASE/'pipeline/scripts/restart_v271_v274_services.sh',
  'cpu_health_watcher':BASE/'pipeline/scripts/watch_v278_retry2_training_health.sh',
  'candidate_freeze_tool':BASE/'pipeline/scripts/freeze_v277_v274_candidate.py'}
 payload={'format':'strict-track2-v282-v278-cpu-limited-operator-restart-amendment-v1','created_utc':datetime.now(timezone.utc).isoformat(),'reason':'operator restarted the container after observing 100% CPU utilization during the first actor update','interrupted_attempt':1,'interrupted_before_any_valid_checkpoint':True,'restart_from_step':0,'resource_limits':{'instance_quota_cores':12,'process_affinity':'0-5','maximum_affinity_cores':6,'omp_threads':2,'mkl_threads':2,'openblas_threads':2,'numexpr_threads':2,'rayon_threads':2,'tensorflow_intraop_threads':2,'tensorflow_interop_threads':2},'unchanged_training_contract':prereg['frozen_training'],'policy_or_data_changed':False,'policy_outcomes_read':False,'official_submission':False,'passed':True,'evidence':{k:{'path':str(v),'sha256':sha(v)} for k,v in files.items()}}
 out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps(payload,indent=2,sort_keys=True))
if __name__=='__main__':main()
