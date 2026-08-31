#!/usr/bin/env python3
"""Freeze the first public batch gate for the v274 full-budget candidate."""
from __future__ import annotations
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
out,checkpoint,training_report,manifest,batch=map(Path,sys.argv[1:6]);variant=sys.argv[6]
if out.exists():raise FileExistsError(out)
training=json.loads(training_report.read_text());assert training['passed'] is True;assert training['checkpoint']==str(checkpoint);assert checkpoint.stat().st_size>8_000_000_000
record={
 'format':'strict-track2-v275-v274-public-screen-preregistration-v2',
 'created_at':datetime.now(timezone.utc).isoformat(),
 'variant':variant,
 'purpose':'public-only gate after fresh official-policy full-budget v271 training',
 'selection_uses_policy_outcomes':False,
 'batch00_gate':{
  'batch':0,
  'episode_composition':{'left':4,'right':12,'total':16},
  'thresholds':{'right_success':6,'left_success':3,'total_success':10,'grasp_once':14,'right_grasp':10},
  'on_reject':'do not access batch01 or reserved final128; revise only from public training data',
 },
 'thresholds':{'right_success':6,'left_success':3,'total_success':10,'grasp_once':14,'right_grasp':10},
 'public112_gate':{
  'count':112,
  'batches':['00','01','02','03','04','05','06'],
  'additional_batches_after_batch00':['01','02','03','04','05','06'],
  'thresholds':{
   'total_success':75,
   'left_success_rate':0.60,
   'right_success_rate':0.60,
   'grasp_at_least_success':True,
  },
  'target_mapping':'75/112=66.96%, above 85/128=66.41%',
 },
 'freeze_rule':'freeze exactly this candidate only after both public gates pass, then allow one local final128 evaluation',
 'checkpoint':str(checkpoint),
 'checkpoint_sha256':sha(checkpoint),
 'training_report':str(training_report),
 'training_report_sha256':sha(training_report),
 'manifest':str(manifest),
 'manifest_sha256':sha(manifest),
 'batch00':str(batch),
 'batch00_sha256':sha(batch),
 'real_submission':False,
 'reserved_final128_used':False,
 'batch01_or_later_used':False,
}
out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
