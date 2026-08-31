#!/usr/bin/env python3
"""Materialize one authority for the v527 read-only OOF gate validator."""
import argparse, ctypes, errno, hashlib, importlib.util, json, os, signal, stat
from pathlib import Path

ROOT=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge'); S=ROOT/'pipeline/scripts'; J=ROOT/'artifacts/strict_track2_joint_augmentation_20260810'
SELF_PATH=S/'materialize_v527_v526_v525_oof_readonly_validation_execution_authority.py'
CONTRACT_PATH=S/'v527_v526_v525_oof_readonly_validation_execution_authority_contract.json'
VALIDATOR=S/'audit_v527_v526_v525_oof_readonly_validation.py'; VALIDATOR_SHA='3289297e8ca6ff026f3f5fed64860ac00742374bb10ee5c6012cde82ba2f4756'; VALIDATOR_BYTES=52025
PREREG=S/'v527_v526_v525_oof_readonly_gate_preregistration.json'; PREREG_SHA='9fc4e53117309a77aac773a8eccc8550ad82446b8224872ca47afb8d763167ac'; PREREG_BYTES=101898
AUTHORITY_ROOT=J/'v527_v526_v525_oof_readonly_validation_execution_authority_seed1662_20260827'; AUTHORITY_PREP=AUTHORITY_ROOT.with_name(AUTHORITY_ROOT.name+'.authority-prep')
ATTEMPT_ROOT=J/'v527_v526_v525_oof_readonly_validation_attempt_seed1662_20260827'; ATTEMPT_PREP=ATTEMPT_ROOT.with_name(ATTEMPT_ROOT.name+'.attempt-prep')
EXTERNAL_EVIDENCE_ROOT=J/'v527_v526_v525_oof_readonly_validation_external_evidence_seed1662_20260827'; EXTERNAL_EVIDENCE_PREP=EXTERNAL_EVIDENCE_ROOT.with_name(EXTERNAL_EVIDENCE_ROOT.name+'.evidence-prep')
STAGE_A_AUTH=J/'v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_execution_authority_seed1661_20260827'
STAGE_A_EVIDENCE=J/'v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_execution_authority_materialization_evidence_seed1661_20260827'
CANDIDATE_ROOT=J/'v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_seed1661_20260827'; EXTERNAL_ROOT=J/'v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_attempt_seed1661_20260827'
QUAL=Path('/root/v524_v523_phase_a_cache_qualification_seed1660_20260826')
V525_CONTRACT=S/'v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_execution_authority_contract.json'; V525_MATERIALIZER=S/'materialize_v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_execution_authority.py'
V525_RECONCILER=S/'reconcile_v525_v524_phase_a_worker_receipt_format_readonly.py'; V525_FORENSIC=S/'v525_v524_phase_a_worker_receipt_format_failure_forensic.json'
V526_FACT=S/'v526_v525_stage_b_predeploy_self_match_failure_forensic.json'; V526_DEPLOYER=Path('/root/deploy_v526_v525_stage_b_transport_pair_once.py')
STAGE_B_HELPER=S/'invoke_v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_once.py'; STAGE_B_SCRIPT=Path('/root/v525_phase_a_worker_receipt_format_readonly_reconciliation_once.sh')
V526_HELPER_TEMP=S/'.invoke_v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_once.py.v526-owned-deploy-temp'; V526_SCRIPT_TEMP=Path('/root/.v525_phase_a_worker_receipt_format_readonly_reconciliation_once.sh.v526-owned-deploy-temp')
CONTRACT_FORMAT='strict-track2-v527-v526-v525-oof-readonly-validation-execution-authority-design-contract-v1'; CONTRACT_STATUS='design_only_frozen_readonly_oof_gate_sources_pending_independent_review_no_authority'
OUTPUT_FORMAT='strict-track2-v527-v526-v525-oof-readonly-validation-execution-authority-v1'; OUTPUT_STATUS='authorized_exact_one_external_v527_oof_readonly_gate_validator_attempt'

AUTHORIZATION={'oof_readonly_gate_validator_invocations_authorized':1,'oof_readonly_gate_validator_invocations_consumed':0,'oof_model_execution_invocations_authorized':0,'phase_a_launcher_invocations_authorized':0,'phase_a_driver_invocations_authorized':0,'phase_a_worker_invocations_authorized':0,'rng_proxy_delegate_invocations_authorized':0,'phase_a_replay_invocations_authorized':0,'training_invocations_authorized':0,'cache_reuse_invocations_authorized':0,'reward_read_invocations_authorized':0,'dev_hidden_final_outcome_read_invocations_authorized':0,'submission_invocations_authorized':0,'retry_authorized':False,'oof_readonly_gate_validation_authorized':True,'oof_model_execution_authorized':False,'phase_a_replay_authorized':False,'training_authorized':False,'cache_reuse_authorized':False,'reward_read_authorized':False,'dev_hidden_final_outcome_read_authorized':False,'submission_authorized':False,'qualification_readonly_validation_authorized':True,'qualification_mutation_authorized':False}
RUNTIME={'execution_authority_materialized':True,'oof_readonly_gate_validator_executed':False,'oof_model_execution_invocations':0,'phase_a_launcher_executed':False,'phase_a_driver_executed':False,'phase_a_worker_invocations':0,'rng_proxy_delegate_invocations':0,'phase_a_replayed':False,'training_launched':False,'cache_reused':False,'reward_reads':0,'dev_hidden_final_outcome_reads':0,'submission_executed':False,'qualification_mutated':False}
EXECUTION_BOUNDARY={'authority_materialization_only':True,'oof_readonly_gate_validator_invocations':0,'oof_model_execution_invocations':0,'phase_a_launcher_invocations':0,'phase_a_driver_invocations':0,'phase_a_worker_invocations':0,'rng_proxy_delegate_invocations':0,'phase_a_replay_invocations':0,'training_invocations':0,'cache_reuse_invocations':0,'reward_reads':0,'dev_hidden_final_outcome_reads':0,'submission_invocations':0,'authority_publication_commit_semantics':'current_exact1_visibility_after_noreplace_directory_rename','crash_durability_claimed':False,'postcommit_diagnostics_best_effort_success_priority':True,'postcommit_failure_no_rollback_relink_retry_or_second_publish':True}

CONTRACT_SOURCE_ORDER=['authority_materializer','oof_readonly_gate_validator','oof_readonly_gate_preregistration','v482_s0_execution_preregistration_ancestry','v482_oof_design_contract','v482_static_auditor','v482_s0_validator','v482_trainer_schema_ancestry','v525_authority_design_contract','v525_authority_materializer','v525_readonly_reconciler','v525_failure_forensic','v525_stage_a_authority_receipt','v525_stage_a_process_receipt','v525_stage_b_candidate_receipt','v525_stage_b_external_process_receipt','v525_stage_b_external_stdout','v526_stage_b_deploy_failure_fact','v526_stage_b_deployer','v525_stage_b_transport_helper','v525_stage_b_transport_script','v524_qualification_terminal','v524_qualification_report','v524_qualification_independent_audit','v524_process_a_receipt','v524_process_b_receipt']
AUTHORITY_SOURCE_ORDER=['authority_design_contract',*CONTRACT_SOURCE_ORDER]
SOURCE_ALIASES={'authority_design_contract':'authority_design_contract',**{role:role+'_source' for role in CONTRACT_SOURCE_ORDER}}
CHECK_KEYS=sorted({'contract_schema','prereg_schema','source_order_exact','source_aliases_exact','source_closure_current','v482_ancestry_current_not_inherited','stage_a_authority_exact1','stage_a_evidence_exact6','stage_b_candidate_exact1_nonterminal','stage_b_external_evidence_exact6_passed','stage_b_dual_bind','normalized_view_exact','recursive_diff_exact2','qualification_exact20','qualification_receipts_current','per_call_events_2000','process_markers_4000','raw_warnings_exact','rng_exit_restored','v526_deployment_pair_current','oof_gate_candidate_contract_exact','oof_gate_external_terminal_evidence_contract_exact','authorization_boundary','runtime_boundary','execution_boundary','fresh_roots_absent','input_prepost_equal','services_current','gpu_empty','execution_pids_empty','qualification_readonly','publish_noreplace_exact1','historical_training_authority_not_inherited'})
CONTRACT_TOP_KEYS={'format','status','seed','lineage','source_closure','source_role_order','source_aliases','source_closure_sha256','authority_materializer_source','oof_readonly_gate_validator_source','oof_readonly_gate_preregistration_source','v482_oof_schema_ancestry','v525_stage_a_authority_registration_tree','v525_stage_a_authority_receipt','v525_stage_a_evidence_tree','v525_stage_a_process_receipt','v525_stage_b_candidate_tree','v525_stage_b_candidate_receipt','v525_stage_b_external_evidence_tree','v525_stage_b_external_process_receipt','v525_stage_b_external_stdout','v525_stage_b_execution_partition','v525_normalized_qualification_view','v524_qualification_tree','v524_qualification_terminal','v524_qualification_report','v524_qualification_independent_audit','v524_process_a_receipt','v524_process_b_receipt','v526_stage_b_transport_deployment','oof_gate_input_contract','oof_gate_output_contract','oof_gate_candidate_contract','oof_gate_external_terminal_evidence_contract','historical_absences','current_absences_after_authority','authority_receipt_contract','authorization','runtime_observation','execution_boundary'}
ANCHOR_KEYS={'v482_oof_schema_ancestry','v525_stage_a_authority_registration_tree','v525_stage_a_authority_receipt','v525_stage_a_evidence_tree','v525_stage_a_process_receipt','v525_stage_b_candidate_tree','v525_stage_b_candidate_receipt','v525_stage_b_external_evidence_tree','v525_stage_b_external_process_receipt','v525_stage_b_external_stdout','v525_stage_b_execution_partition','v525_normalized_qualification_view','v524_qualification_tree','v524_qualification_terminal','v524_qualification_report','v524_qualification_independent_audit','v524_process_a_receipt','v524_process_b_receipt','v526_stage_b_transport_deployment','oof_gate_input_contract','oof_gate_output_contract','oof_gate_candidate_contract','oof_gate_external_terminal_evidence_contract'}
AUTH_TOP_KEYS={'format','status','passed','source_closure','source_role_order','source_aliases','source_closure_sha256',*SOURCE_ALIASES.values(),*ANCHOR_KEYS,'oof_validation_attempt_root','historical_absences','required_absences','checks','check_keys','check_key_set_sha256','checks_sha256','input_pre_snapshot','input_post_snapshot','input_snapshots_exactly_equal','authorization','runtime_observation','execution_boundary'}

EXPECTED_RECORDS={
'oof_readonly_gate_validator':(VALIDATOR,'3289297e8ca6ff026f3f5fed64860ac00742374bb10ee5c6012cde82ba2f4756',52025),'oof_readonly_gate_preregistration':(PREREG,'9fc4e53117309a77aac773a8eccc8550ad82446b8224872ca47afb8d763167ac',101898),
'v482_s0_execution_preregistration_ancestry':(J/'v482_temporal8_residual_s0_r3_seed1624_20260824/preregistration.json','426d4f7a774520458f6af41d20660cf31202791f99603dacec114f771da6a881',294684),'v482_oof_design_contract':(S/'v482_temporal_film_residual_model_design_contract.json','19eea774f4b8871528054a352bf6c7ed6cefd39afcfc1956c9cf7a969335fddb',37030),'v482_static_auditor':(S/'audit_v482_temporal8_residual_static.py','088447774ad4ae0b67ec3470e4170ca5c8527e5fdfd0ee12fc000549bf785a2e',16435),'v482_s0_validator':(S/'audit_v482_temporal8_residual_s0.py','a24a20a8c0670e7bf53819bb48e78d204b482fbbdbe1060db61957488b3eed69',20439),'v482_trainer_schema_ancestry':(S/'train_v482_temporal8_residual_5fold.py','674b68afed3b6c39663d629db38be11aa2c752e54692458b43d4254eb08b8c1d',48428),
'v525_authority_design_contract':(V525_CONTRACT,'749373c708147ff7079c3efb2196a046a185edacccfca32a44a59d87e5d50124',70602),'v525_authority_materializer':(V525_MATERIALIZER,'3b8434571c7444c489689c2b7601bb56e79a46d79780843875f4cc5b6fcf9e82',27325),'v525_readonly_reconciler':(V525_RECONCILER,'81aa7da1eb525b5d03f8053da4dbce81e372b0660b6aa31e068dc3e2604b13f3',48538),'v525_failure_forensic':(V525_FORENSIC,'ea04aede510143cf9f1c6388c222b64397a8f5dfdf3c15a5c10c56a7d5b058d5',8137),
'v525_stage_a_authority_receipt':(STAGE_A_AUTH/'authority_receipt.json','1f468f2c5ed2f5928e9491e7113f1f34e1095dcd17c57c2a9c973107043a4057',95906),'v525_stage_a_process_receipt':(STAGE_A_EVIDENCE/'process_receipt.json','fb30407693464e98feae18eb81ec52905a62a5a4d160889505e86ecaaa8a31f8',31038),'v525_stage_b_candidate_receipt':(CANDIDATE_ROOT/'reconciliation_receipt.json','f577a285de806fdd9e27ed70708640da1b3686eebee66c5378cb9dcd0c5581f2',43485),'v525_stage_b_external_process_receipt':(EXTERNAL_ROOT/'process_receipt.json','e4674978630ebb85a824e31751ffba40b4cb06b9d737f7c5dc068635490df845',38888),'v525_stage_b_external_stdout':(EXTERNAL_ROOT/'reconciler_stdout.log','a56ed2bb027b5360a8c46ce1563f6449539ab6ae727995fa643084225a96d3e0',1832),
'v526_stage_b_deploy_failure_fact':(V526_FACT,'6534a349ec692c51d6a71a50f181ec5aefbb790a1283a56aca5ad2137881a309',4029),'v526_stage_b_deployer':(V526_DEPLOYER,'2032367324f8581933f56a600fb5f511e252c1fd8422d264705f6ced50e94423',21235),'v525_stage_b_transport_helper':(STAGE_B_HELPER,'ed6e30d8c155dbd9deccf8715773d3c42ae66770996e1912364c14fd3d779574',63291),'v525_stage_b_transport_script':(STAGE_B_SCRIPT,'748142b0ab07ea526ee33544ce390a65cd5da18090924a41a66e938245cb88e9',2567),
'v524_qualification_terminal':(QUAL/'terminal_receipt.json','93d38e010b37275a740bf3ccae2b935b08e40b6953657c12838780e5e2b542a1',83762),'v524_qualification_report':(QUAL/'qualification_report.json','6514b497f994bc502fc46dc6c5a90865dbb126402935cfa359a1af2fe0fbc173',87681),'v524_qualification_independent_audit':(QUAL/'independent_final_audit.json','3d702b36a3589d737c9f21e0a1182c59f636753faa1143a2adc85d90d3a674bc',9935),'v524_process_a_receipt':(QUAL/'process_a/receipt.json','89b7ce9e0ee6263fba2cb5b7e393664330776cfa7914ad9715e6ff2ed7346414',8871),'v524_process_b_receipt':(QUAL/'process_b/receipt.json','90ac9ff4eab3a06c02a2937ebe0f35a20e68145862e418ef50048c07c2d15214',8912)}

def cbytes(v): return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def csha(v): return hashlib.sha256(cbytes(v)).hexdigest()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''): h.update(block)
 return h.hexdigest()
def regular(path,digest=None,size=None):
 path=Path(path); st=os.lstat(path)
 if path.is_symlink() or not stat.S_ISREG(st.st_mode): raise RuntimeError('nonregular '+str(path))
 row={'path':str(path),'sha256':sha(path),'logical_bytes':st.st_size}
 if digest is not None and (row['sha256']!=digest or row['logical_bytes']!=size): raise RuntimeError('record '+str(path))
 return row
def load_validator():
 regular(VALIDATOR,VALIDATOR_SHA,VALIDATOR_BYTES); sp=importlib.util.spec_from_file_location('v527_validator',VALIDATOR); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def exact_tree(root): return load_validator().exact_tree(Path(root))
def json_exact(a,b): return cbytes(a)==cbytes(b)
def absences():
 historical={'authority_root':AUTHORITY_ROOT,'authority_prep':AUTHORITY_PREP,'oof_validation_attempt_root':ATTEMPT_ROOT,'oof_validation_attempt_prep':ATTEMPT_PREP,'external_evidence_root':EXTERNAL_EVIDENCE_ROOT,'external_evidence_prep':EXTERNAL_EVIDENCE_PREP}
 return historical,{k:v for k,v in historical.items() if k!='authority_root'}
def snapshot(files,trees,absence_map,validator):
 values={'files':{k:regular(v) for k,v in sorted(files.items())},'trees':{k:validator.exact_tree(v) for k,v in sorted(trees.items())},'absences':{k:{'path':str(v),'absent':not os.path.lexists(v)} for k,v in sorted(absence_map.items())},'services':validator.service_health(),'gpu_compute_pids':validator.gpu_compute_pids(),'relevant_execution_pids':validator.relevant_execution_pids()}
 values['canonical_sha256']=csha(values); return values
def rename_noreplace(source,target):
 libc=ctypes.CDLL(None,use_errno=True); fn=getattr(libc,'renameat2',None)
 if fn is None: raise RuntimeError('renameat2 unavailable')
 rc=fn(-100,os.fsencode(source),-100,os.fsencode(target),1)
 if rc:
  err=ctypes.get_errno(); raise FileExistsError(target) if err==errno.EEXIST else OSError(err,os.strerror(err),str(target))
def fsync_dir(path):
 fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
 try: os.fsync(fd)
 finally: os.close(fd)
def read_fd_exact(fd,expected_bytes,pread=os.pread):
 out=bytearray();offset=0
 while offset<expected_bytes:
  block=pread(fd,min(1<<20,expected_bytes-offset),offset)
  if not block: raise RuntimeError('held member short read')
  out.extend(block);offset+=len(block)
 if pread(fd,1,expected_bytes): raise RuntimeError('held member EOF drift')
 return bytes(out)
def assert_owned_member(path,fd,identity,expected_sha,expected_bytes,pread=os.pread):
 fst=os.fstat(fd);lst=os.lstat(path)
 if path.is_symlink() or not stat.S_ISREG(fst.st_mode) or not stat.S_ISREG(lst.st_mode) or (fst.st_dev,fst.st_ino)!=(lst.st_dev,lst.st_ino) or (fst.st_dev,fst.st_ino)!=identity: raise RuntimeError('held member identity')
 data=read_fd_exact(fd,expected_bytes,pread)
 if hashlib.sha256(data).hexdigest()!=expected_sha: raise RuntimeError('held member digest')
 return {'path':str(path),'sha256':expected_sha,'logical_bytes':expected_bytes}
def cleanup_owned_prep(prep,identity,member_fd=None,member_identity=None,expected_sha=None,expected_bytes=None):
 try: st=os.lstat(prep)
 except FileNotFoundError:return
 if identity!=(st.st_dev,st.st_ino):return
 items=list(prep.iterdir())
 if items:
  if len(items)!=1 or items[0].name!='authority_receipt.json' or member_fd is None or member_identity is None:return
  try:
   fst=os.fstat(member_fd);lst=os.lstat(items[0])
   if items[0].is_symlink() or not stat.S_ISREG(fst.st_mode) or not stat.S_ISREG(lst.st_mode) or (fst.st_dev,fst.st_ino)!=(lst.st_dev,lst.st_ino) or (fst.st_dev,fst.st_ino)!=member_identity:return
   # The held descriptor and current pathname still name our exclusively-created
   # inode.  Its current (possibly partial) record is therefore owned cleanup;
   # an inode replacement is foreign and is preserved above.
   current_bytes=fst.st_size;current=read_fd_exact(member_fd,current_bytes)
   if len(current)!=current_bytes:return
  except (OSError,RuntimeError):return
  items[0].unlink()
 prep.rmdir();fsync_dir(prep.parent)
def publish_exact1(receipt,files,trees,stable,stable_pre,validator,writer=os.write,pread=os.pread,phase_hook=None):
 if os.path.lexists(AUTHORITY_ROOT) or os.path.lexists(AUTHORITY_PREP): raise RuntimeError('authority prestate')
 hook=phase_hook or (lambda _name:None);oldmask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM}); identity=None;member_identity=None;fd=None;committed=False;expected_sha=None;data=b''
 try:
  os.mkdir(AUTHORITY_PREP,0o700);st=os.lstat(AUTHORITY_PREP);identity=(st.st_dev,st.st_ino)
  member=AUTHORITY_PREP/'authority_receipt.json'; flags=os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW;fd=os.open(member,flags,0o600);fst=os.fstat(fd);member_identity=(fst.st_dev,fst.st_ino)
  try:
   data=json.dumps(receipt,sort_keys=True,indent=2,ensure_ascii=False).encode()+b'\n';expected_sha=hashlib.sha256(data).hexdigest();view=memoryview(data)
   while view:
    written=writer(fd,view)
    if type(written) is not int or written<=0 or written>len(view): raise RuntimeError('held member write')
    view=view[written:]
   os.fsync(fd)
  except BaseException: raise
  assert_owned_member(member,fd,member_identity,expected_sha,len(data),pread);hook('after_member_write')
  fsync_dir(AUTHORITY_PREP)
  if snapshot(files,trees,stable,validator)!=stable_pre: raise RuntimeError('third snapshot drift')
  assert_owned_member(member,fd,member_identity,expected_sha,len(data),pread);hook('before_rename');assert_owned_member(member,fd,member_identity,expected_sha,len(data),pread)
  rename_noreplace(AUTHORITY_PREP,AUTHORITY_ROOT); committed=True
  # Visibility is the commit point.  Every later operation is diagnostic-only:
  # it cannot roll back, relink, retry, or turn this invocation into a second publish.
  diagnostics=[];tree=None;final_member=AUTHORITY_ROOT/'authority_receipt.json'
  for name,operation in (
   ('phase_hook',lambda:hook('after_rename_before_postcommit')),
   ('parent_fsync',lambda:fsync_dir(AUTHORITY_ROOT.parent)),
   ('held_readback',lambda:assert_owned_member(final_member,fd,member_identity,expected_sha,len(data),pread)),
   ('current_tree',lambda:validator.exact_tree(AUTHORITY_ROOT))):
   try:
    value=operation()
    if name=='current_tree':tree=value
   except BaseException as exc:diagnostics.append({'operation':name,'error_type':type(exc).__name__,'error':str(exc)})
  expected_inventory=[['authority_receipt.json',expected_sha,len(data)]]
  current_exact=tree is not None and tree.get('file_count')==1 and tree.get('inventory')==expected_inventory
  return {'file_count':1 if current_exact else None,'inventory':tree.get('inventory') if tree else None,'visibility_committed':True,'current_exact1_consumable':current_exact,'postcommit_diagnostics_passed':not diagnostics and current_exact,'postcommit_diagnostics':diagnostics,'retry_authorized':False,'rollback_or_relink_performed':False,'second_publish_performed':False}
 finally:
  if not committed: cleanup_owned_prep(AUTHORITY_PREP,identity,fd,member_identity,expected_sha,len(data) if data else None)
  if fd is not None:
   try:os.close(fd)
   except OSError:pass
  signal.pthread_sigmask(signal.SIG_SETMASK,oldmask)

def validate_context(args):
 if args.contract!=CONTRACT_PATH or args.materializer_source!=SELF_PATH or args.authority_root!=AUTHORITY_ROOT: raise RuntimeError('canonical CLI')
 contract_record=regular(CONTRACT_PATH,args.contract_sha,args.contract_bytes); materializer_record=regular(SELF_PATH,args.materializer_sha,args.materializer_bytes); contract=json.loads(CONTRACT_PATH.read_text()); validator=load_validator(); prereg=json.loads(PREREG.read_text());validator.validate_prereg(prereg)
 if set(contract)!=CONTRACT_TOP_KEYS or contract.get('format')!=CONTRACT_FORMAT or contract.get('status')!=CONTRACT_STATUS or type(contract.get('seed')) is not int or contract['seed']!=1662: raise RuntimeError('contract schema')
 if not json_exact(contract.get('authorization'),AUTHORIZATION) or not json_exact(contract.get('runtime_observation'),RUNTIME) or not json_exact(contract.get('execution_boundary'),EXECUTION_BOUNDARY): raise RuntimeError('boundary')
 sources=contract.get('source_closure',{}); expected={'authority_materializer':materializer_record,**{k:regular(*v) for k,v in EXPECTED_RECORDS.items()}}
 if list(contract.get('source_role_order',[]))!=CONTRACT_SOURCE_ORDER or set(sources)!=set(CONTRACT_SOURCE_ORDER) or not json_exact(sources,expected) or contract.get('source_closure_sha256')!=csha(sources) or contract.get('source_aliases')!=SOURCE_ALIASES: raise RuntimeError('source closure')
 if contract.get('authority_materializer_source')!=materializer_record or contract.get('oof_readonly_gate_validator_source')!=sources['oof_readonly_gate_validator'] or contract.get('oof_readonly_gate_preregistration_source')!=sources['oof_readonly_gate_preregistration']: raise RuntimeError('source aliases')
 current=validator.validate_current(prereg)
 if os.path.lexists(V526_HELPER_TEMP) or os.path.lexists(V526_SCRIPT_TEMP): raise RuntimeError('v526 deploy temp current')
 deployment={'failure_fact':sources['v526_stage_b_deploy_failure_fact'],'deployer':sources['v526_stage_b_deployer'],'transport_helper':sources['v525_stage_b_transport_helper'],'transport_script':sources['v525_stage_b_transport_script'],'pair_current':True,'owned_deploy_temps':{'helper':{'path':str(V526_HELPER_TEMP),'absent':True},'script':{'path':str(V526_SCRIPT_TEMP),'absent':True}},'historical_self_match_not_retried':True}
 expected_anchors={'v482_oof_schema_ancestry':prereg['ancestry_sources'],'v525_stage_a_authority_registration_tree':current['stage_a_authority_tree'],'v525_stage_a_authority_receipt':current['stage_a_authority_receipt'],'v525_stage_a_evidence_tree':current['stage_a_evidence_tree'],'v525_stage_a_process_receipt':current['stage_a_process_receipt'],'v525_stage_b_candidate_tree':current['stage_b_candidate_tree'],'v525_stage_b_candidate_receipt':current['stage_b_candidate_receipt'],'v525_stage_b_external_evidence_tree':current['stage_b_external_evidence_tree'],'v525_stage_b_external_process_receipt':current['stage_b_external_process_receipt'],'v525_stage_b_external_stdout':sources['v525_stage_b_external_stdout'],'v525_stage_b_execution_partition':{'transport_helper_invocations':1,'readonly_reconciler_invocations':1,'authority_materializer_invocations':0,'phase_a_launcher_invocations':0,'phase_a_driver_invocations':0,'phase_a_worker_invocations':0,'rng_proxy_delegate_invocations':0,'training_invocations':0,'oof_invocations':0},'v525_normalized_qualification_view':prereg['normalized_qualification_view'],'v524_qualification_tree':current['qualification_tree'],'v524_qualification_terminal':current['qualification_terminal'],'v524_qualification_report':current['qualification_report'],'v524_qualification_independent_audit':current['qualification_independent_audit'],'v524_process_a_receipt':sources['v524_process_a_receipt'],'v524_process_b_receipt':sources['v524_process_b_receipt'],'v526_stage_b_transport_deployment':deployment,'oof_gate_input_contract':prereg['input_contract'],'oof_gate_output_contract':prereg['output_contract'],'oof_gate_candidate_contract':validator.CANDIDATE_OUTPUT_CONTRACT,'oof_gate_external_terminal_evidence_contract':validator.EXTERNAL_TERMINAL_EVIDENCE_CONTRACT}
 if any(not json_exact(contract.get(k),v) for k,v in expected_anchors.items()): raise RuntimeError('contract anchor')
 lineage={'name':'v527_v526_v525_oof_readonly_validation','seed':1662,'fresh_authority_root':str(AUTHORITY_ROOT),'fresh_attempt_root':str(ATTEMPT_ROOT),'stage_b_candidate_and_external_terminal_dual_bound':True,'normalized_view_only_for_validation':True,'historical_v482_training_authority_not_inherited':True,'oof_model_execution_authorized':False}
 historical,current_abs=absences()
 if not json_exact(contract.get('lineage'),lineage) or contract.get('historical_absences')!={k:{'path':str(v)} for k,v in historical.items()} or contract.get('current_absences_after_authority')!={k:{'path':str(v)} for k,v in current_abs.items()}: raise RuntimeError('lineage/absences')
 if any(os.path.lexists(v) for v in historical.values()): raise RuntimeError('fresh roots')
 schema=contract.get('authority_receipt_contract',{}); expected_schema={'format':OUTPUT_FORMAT,'status':OUTPUT_STATUS,'top_keys':sorted(AUTH_TOP_KEYS),'check_keys':CHECK_KEYS,'check_key_set_sha256':csha(CHECK_KEYS),'checks_sha256':csha({k:True for k in CHECK_KEYS}),'authorization_exact':AUTHORIZATION,'runtime_observation_exact':RUNTIME,'execution_boundary_exact':EXECUTION_BOUNDARY,'oof_gate_input_contract_exact':prereg['input_contract'],'oof_gate_output_contract_exact':prereg['output_contract'],'oof_gate_candidate_contract_exact':validator.CANDIDATE_OUTPUT_CONTRACT,'oof_gate_external_terminal_evidence_contract_exact':validator.EXTERNAL_TERMINAL_EVIDENCE_CONTRACT}
 if not json_exact(schema,expected_schema): raise RuntimeError('receipt contract')
 files={'contract':str(CONTRACT_PATH),'materializer':str(SELF_PATH),**{'source_'+k:v['path'] for k,v in sources.items()}};trees={'stage_a_authority':STAGE_A_AUTH,'stage_a_evidence':STAGE_A_EVIDENCE,'stage_b_candidate':CANDIDATE_ROOT,'stage_b_external':EXTERNAL_ROOT,'qualification':QUAL};snapshot_absences={**historical,'v526_helper_deploy_temp':V526_HELPER_TEMP,'v526_script_deploy_temp':V526_SCRIPT_TEMP};pre=snapshot(files,trees,snapshot_absences,validator)
 return contract,contract_record,materializer_record,sources,expected_anchors,historical,current_abs,snapshot_absences,files,trees,pre,validator

def main():
 if os.sys.argv[1:]==['--synthetic-self-test']:
  checks={'contract_top_count':len(CONTRACT_TOP_KEYS)==40,'source_count':len(CONTRACT_SOURCE_ORDER)==26,'authority_source_count':len(AUTHORITY_SOURCE_ORDER)==27,'authorization_count':len(AUTHORIZATION)==24,'strict_count_types':all(type(AUTHORIZATION[k]) is int for k in AUTHORIZATION if k.endswith('_invocations_authorized') or k.endswith('_invocations_consumed')),'output_non_oof':AUTHORIZATION['oof_model_execution_authorized'] is False and AUTHORIZATION['oof_readonly_gate_validation_authorized'] is True,'fresh_paths':AUTHORITY_ROOT!=ATTEMPT_ROOT and AUTHORITY_PREP!=ATTEMPT_PREP};print(json.dumps({'passed':all(checks.values()),'checks':checks,'checks_sha256':csha(checks)},sort_keys=True));return 0 if all(checks.values()) else 1
 p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,required=True);p.add_argument('--contract-sha',required=True);p.add_argument('--contract-bytes',type=int,required=True);p.add_argument('--materializer-source',type=Path,required=True);p.add_argument('--materializer-sha',required=True);p.add_argument('--materializer-bytes',type=int,required=True);p.add_argument('--authority-root',type=Path,required=True);p.add_argument('--read-only-preflight',action='store_true');a=p.parse_args()
 contract,cr,mr,sources,anchors,historical,current_abs,snapshot_absences,files,trees,pre,validator=validate_context(a);post=snapshot(files,trees,snapshot_absences,validator)
 if pre!=post: raise RuntimeError('input prepost drift')
 checks={k:True for k in CHECK_KEYS};checks['gpu_empty']=not post['gpu_compute_pids'];checks['execution_pids_empty']=not post['relevant_execution_pids']
 authority_sources={'authority_design_contract':cr,**sources};receipt={'format':OUTPUT_FORMAT,'status':OUTPUT_STATUS,'passed':True,'source_closure':authority_sources,'source_role_order':AUTHORITY_SOURCE_ORDER,'source_aliases':SOURCE_ALIASES,'source_closure_sha256':csha(authority_sources),**{SOURCE_ALIASES[k]:v for k,v in authority_sources.items()},**anchors,'oof_validation_attempt_root':str(ATTEMPT_ROOT),'historical_absences':{k:{'path':str(v),'absent':True} for k,v in historical.items()},'required_absences':{k:{'path':str(v),'absent':True} for k,v in current_abs.items()},'checks':checks,'check_keys':CHECK_KEYS,'check_key_set_sha256':csha(CHECK_KEYS),'checks_sha256':csha(checks),'input_pre_snapshot':pre,'input_post_snapshot':post,'input_snapshots_exactly_equal':True,'authorization':AUTHORIZATION,'runtime_observation':RUNTIME,'execution_boundary':EXECUTION_BOUNDARY}
 if set(receipt)!=AUTH_TOP_KEYS or not all(type(v) is bool and v is True for v in checks.values()): raise RuntimeError('authority receipt schema')
 if a.read_only_preflight: print(json.dumps({'passed':True,'status':'passed_read_only_preflight_no_materialization','contract_top_count':len(contract),'authority_top_count':len(receipt),'check_count':len(CHECK_KEYS),'source_count':len(sources),'authority_source_count':len(authority_sources)},sort_keys=True));return 0
 stable={k:v for k,v in snapshot_absences.items() if k not in {'authority_root','authority_prep'}};stable_pre=snapshot(files,trees,stable,validator);publish_exact1(receipt,files,trees,stable,stable_pre,validator);return 0
if __name__=='__main__': raise SystemExit(main())
