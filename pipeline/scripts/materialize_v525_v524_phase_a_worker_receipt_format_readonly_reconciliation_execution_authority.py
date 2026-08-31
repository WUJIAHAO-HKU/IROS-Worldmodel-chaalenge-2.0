#!/usr/bin/env python3
"""Materialize the v525 authority for one read-only v524 receipt reconciliation."""
import argparse, copy, hashlib, importlib.util, json, os, stat, urllib.request
from pathlib import Path

ROOT=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge'); S=ROOT/'pipeline/scripts'; J=ROOT/'artifacts/strict_track2_joint_augmentation_20260810'
SELF_PATH=S/'materialize_v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_execution_authority.py'
CONTRACT_PATH=S/'v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_execution_authority_contract.json'
RECONCILER=S/'reconcile_v525_v524_phase_a_worker_receipt_format_readonly.py'; RECONCILER_SHA='81aa7da1eb525b5d03f8053da4dbce81e372b0660b6aa31e068dc3e2604b13f3'; RECONCILER_BYTES=48538
FORENSIC=S/'v525_v524_phase_a_worker_receipt_format_failure_forensic.json'; FORENSIC_SHA='ea04aede510143cf9f1c6388c222b64397a8f5dfdf3c15a5c10c56a7d5b058d5'; FORENSIC_BYTES=8137
OLD_CONTRACT=S/'v524_v523_phase_a_exact_source_roles_repair_execution_authority_contract.json'; OLD_CONTRACT_SHA='f3fdd5851231a984aa1ce04685a5433fd5f99e73065271969c30c62597c9b95d'; OLD_CONTRACT_BYTES=39332
OLD_MATERIALIZER=S/'materialize_v524_v523_phase_a_exact_source_roles_repair_execution_authority.py'; OLD_MATERIALIZER_SHA='1b33ba41297a998379cceebe3fcb4085a59f017d46dc481fdc2d3859622767dd'; OLD_MATERIALIZER_BYTES=35575
OLD_AUTH=J/'v524_v523_phase_a_exact_source_roles_repair_execution_authority_seed1660_20260826'; OLD_AUTH_RECEIPT=OLD_AUTH/'authority_receipt.json'
ATTEMPT=J/'v524_v523_phase_a_cache_qualification_attempt_seed1660_20260826'; QUAL=Path('/root/v524_v523_phase_a_cache_qualification_seed1660_20260826')
PROXY=S/'v520_v519_rng_isolated_v169_runtime_proxy.py'; PROXY_SHA='2b8ab0e2aec4a8de5d4bf8cffbb3525bbcd69b9b03270012a4a48f69f430b51b'; PROXY_BYTES=8926
AUTHORITY_ROOT=J/'v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_execution_authority_seed1661_20260827'
RECONCILIATION_ROOT=J/'v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_seed1661_20260827'
EXTERNAL_EVIDENCE_ROOT=J/'v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_attempt_seed1661_20260827'
EXTERNAL_EVIDENCE_PREP=EXTERNAL_EVIDENCE_ROOT.with_name(EXTERNAL_EVIDENCE_ROOT.name+'.attempt-prep')
OLD_FORMAT='strict-track2-v523-v522-rng-isolated-cache-worker-receipt-v1'; NORMALIZED_FORMAT='strict-track2-v524-v523-rng-isolated-cache-worker-receipt-v1'
ORIGINAL_CSHA='d00018deb7ea3ae660061e8879dc4a4f6d56bc65bd089c28c4a0580239c08101'; NORMALIZED_CSHA='c409c2b8f85641c23bc68fb06512cd664c84108d4b8b629480906193737f4b32'; DIFF_CSHA='fdc7999068d0e630272e6cb226b6b342692b5f8324cd506d593b5993b5f52779'
CONTRACT_FORMAT='strict-track2-v525-v524-phase-a-worker-receipt-format-readonly-reconciliation-execution-authority-design-contract-v1'; CONTRACT_STATUS='design_only_frozen_sources_pending_independent_review_no_authority'
OUTPUT_FORMAT='strict-track2-v525-v524-phase-a-worker-receipt-format-readonly-reconciliation-execution-authority-v1'; OUTPUT_STATUS='authorized_exact_one_external_v525_readonly_reconciler_attempt'
RECONCILER_CANDIDATE_CONTRACT={'format':'strict-track2-v525-v524-phase-a-worker-receipt-format-readonly-reconciliation-receipt-v1','status':'computation_verified_readonly_launcher_consumer_schema_mismatch','passed':None,'candidate_verified':True,'standalone_consumable':False,'external_terminal_required':True,'publication_success_claimed':False,'commit_semantics':'current_exact1_visibility_after_noreplace_directory_rename','crash_durability_claimed':False,'postcommit_parent_dir_fsync_best_effort':True}
EXTERNAL_TERMINAL_EVIDENCE_CONTRACT={'format':'strict-track2-v525-v524-phase-a-worker-receipt-format-readonly-reconciliation-external-terminal-evidence-v1','evidence_root':str(EXTERNAL_EVIDENCE_ROOT),'evidence_prep':str(EXTERNAL_EVIDENCE_PREP),'expected_file_count':6,'terminal_member':'process_receipt.json','terminal_status':'passed_external_terminal','authoritative_success_only_via_external_terminal':True,'candidate_root':str(RECONCILIATION_ROOT),'candidate_expected_file_count':1,'candidate_and_evidence_dual_bind_required':True,'current_exact1_candidate_required':True,'current_exact6_evidence_required':True,'fresh_transport_helper_source_required':True,'fresh_transport_script_source_required':True,'postcommit_diagnostics_adjudicated_by_external_terminal':True,'retry_authorized':False,'phase_a_replay_authorized':False}
CONTRACT_SOURCE_ORDER=['authority_materializer','readonly_reconciler','failure_forensic','v524_authority_design_contract','v524_authority_materializer','v524_authority_receipt','v524_attempt_intent','v524_attempt_stdout','v524_attempt_stderr','v524_attempt_terminal','v524_qualification_terminal','v524_qualification_report','v524_qualification_independent_audit','v524_process_a_receipt','v524_process_b_receipt','rng_proxy_current_only']
AUTHORITY_SOURCE_ORDER=['authority_design_contract',*CONTRACT_SOURCE_ORDER]
SOURCE_ALIASES={'authority_design_contract':'authority_design_contract',**{r:r+'_source' for r in CONTRACT_SOURCE_ORDER}}
AUTHORIZATION={'readonly_reconciler_invocations_authorized':1,'readonly_reconciler_invocations_consumed':0,'launcher_invocations_authorized':0,'driver_invocations_authorized':0,'worker_invocations_authorized':0,'rng_proxy_delegate_invocations_authorized':0,'phase_a_replay_invocations_authorized':0,'training_invocations_authorized':0,'oof_invocations_authorized':0,'retry_authorized':False,'phase_a_replay_authorized':False,'training_authorized':False,'cache_reuse_authorized':False,'reward_read_authorized':False,'dev_hidden_final_outcome_read_authorized':False,'oof_authorized':False,'qualification_readonly_validation_authorized':True,'qualification_mutation_authorized':False}
RUNTIME={'execution_authority_materialized':True,'readonly_reconciler_executed':False,'phase_a_launcher_executed':False,'phase_a_driver_executed':False,'phase_a_worker_invocations':0,'rng_proxy_delegate_invocations':0,'phase_a_replayed':False,'training_launched':False,'oof_executed':False,'qualification_mutated':False}
EXECUTION_BOUNDARY={'authority_materialization_only':True,'readonly_reconciler_invocations':0,'phase_a_launcher_invocations':0,'phase_a_driver_invocations':0,'phase_a_worker_invocations':0,'rng_proxy_delegate_invocations':0,'phase_a_replay_invocations':0,'training_invocations':0,'oof_invocations':0,'reward_reads':0,'dev_hidden_final_outcome_reads':0}
CHECK_KEYS=sorted({'contract_schema','source_order_exact','source_aliases_exact','source_closure_current','v524_authority_exact1','v524_authority_schema','v524_authority_sources_current','v524_attempt_exact4','v524_attempt_failed_no_retry','v524_qualification_exact20','v524_qualification_terminal_passed','v524_qualification_report_passed','v524_independent_audit_passed','v524_worker_receipts_current','v524_worker_counts_exact','format_mismatch_exact2','original_digest_exact','normalized_digest_exact','recursive_diff_digest_exact','forensic_exact','forensic_semantics','services_restored_current','qualification_readonly','fresh_roots_absent','authorization_boundary','runtime_boundary','execution_boundary','input_prepost_equal','gpu_empty','no_live_phase_a_process','publish_noreplace_exact1'})
CONTRACT_TOP_KEYS={'format','status','seed','lineage','source_closure','source_role_order','source_aliases','source_closure_sha256','authority_materializer_source','readonly_reconciler_source','failure_forensic_source','v524_authority_registration_tree','v524_authority_receipt','v524_authority_semantics','v524_attempt_tree','v524_attempt_intent','v524_attempt_stdout','v524_attempt_stderr','v524_attempt_terminal','v524_execution_partition','v524_qualification_tree','v524_qualification_terminal','v524_qualification_report','v524_qualification_independent_audit','v524_process_a_receipt','v524_process_b_receipt','normalization_contract','reconciler_candidate_contract','external_terminal_evidence_contract','historical_absences','current_absences_after_authority','authority_receipt_contract','authorization','runtime_observation','execution_boundary'}
AUTH_TOP_KEYS={'format','status','passed','source_closure','source_role_order','source_aliases','source_closure_sha256',*SOURCE_ALIASES.values(),'v524_authority_registration_tree','v524_authority_receipt','v524_authority_semantics','v524_attempt_tree','v524_attempt_intent','v524_attempt_stdout','v524_attempt_stderr','v524_attempt_terminal','v524_execution_partition','v524_qualification_tree','v524_qualification_terminal','v524_qualification_report','v524_qualification_independent_audit','v524_process_a_receipt','v524_process_b_receipt','normalization_contract','reconciler_candidate_contract','external_terminal_evidence_contract','reconciliation_output_root','historical_absences','required_absences','checks','check_keys','check_key_set_sha256','checks_sha256','input_pre_snapshot','input_post_snapshot','input_snapshots_exactly_equal','authorization','runtime_observation','execution_boundary'}
AUTH_TREE=(1,64970,'93f58f95c489f79f4d58cb36cd3d16815444b69674e6bd6270dbe09f1d956e7f','8253ef78bf8eadcd7e47e3825cb94c109c2030f66103c4513c5829006136dd72')
ATTEMPT_TREE=(4,44268,'83b63f8a8174ea8d3e1077c263ff3d0dd43403d37992b45c99c811d365eb2c7a','14c23172981c21dd6ec783fa187b6ee51aaa1fe9b01b8ca5620cc418ffe4ff91')
QUAL_TREE=(20,950124602,'6cdde75692d37b7eead3333d867b22dc6404f545bb5861b2f30b425fb4270dac','223308d02ed5a4a082b4de639292030444a627bbbac7f361fac1f9c6a7fcbd1f')

def csha(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def json_exact(a,b): return json.dumps(a,sort_keys=True,separators=(',',':'),ensure_ascii=False)==json.dumps(b,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def sha(p):
 d=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): d.update(b)
 return d.hexdigest()
def regular(path,digest=None,size=None):
 path=Path(path); st=os.lstat(path)
 if path.is_symlink() or not stat.S_ISREG(st.st_mode): raise RuntimeError('nonregular '+str(path))
 r={'path':str(path),'sha256':sha(path),'logical_bytes':st.st_size}
 if digest is not None and (r['sha256']!=digest or r['logical_bytes']!=size): raise RuntimeError('record '+str(path))
 return r
def exact_tree(root):
 root=Path(root); st=os.lstat(root)
 if root.is_symlink() or not stat.S_ISDIR(st.st_mode): raise RuntimeError('tree root')
 inv=[]
 for p in sorted(root.rglob('*'),key=lambda x:x.relative_to(root).as_posix()):
  s=os.lstat(p)
  if p.is_symlink(): raise RuntimeError('tree member')
  if stat.S_ISDIR(s.st_mode): continue
  if not stat.S_ISREG(s.st_mode): raise RuntimeError('tree member')
  inv.append([p.relative_to(root).as_posix(),sha(p),s.st_size])
 lines=''.join(f'{d}  {n}\n' for n,d,_ in inv).encode()
 return {'root':str(root),'inventory':inv,'file_count':len(inv),'logical_file_bytes':sum(x[2] for x in inv),'sha256sum_lines_digest_sha256':hashlib.sha256(lines).hexdigest(),'canonical_json_triples_digest_sha256':csha(inv)}
def tree_tuple(t): return (t['file_count'],t['logical_file_bytes'],t['sha256sum_lines_digest_sha256'],t['canonical_json_triples_digest_sha256'])
def load_base():
 regular(OLD_MATERIALIZER,OLD_MATERIALIZER_SHA,OLD_MATERIALIZER_BYTES); sp=importlib.util.spec_from_file_location('v524_frozen',OLD_MATERIALIZER); old=importlib.util.module_from_spec(sp); sp.loader.exec_module(old); _prior,base=old.load_base()
 base.CONTRACT_PATH=CONTRACT_PATH; base.MATERIALIZER_PATH=SELF_PATH; base.WRAPPER_PATH=RECONCILER; base.AUTHORITY_ROOT=AUTHORITY_ROOT; base.WRAPPER_ATTEMPT_ROOT=RECONCILIATION_ROOT
 base.CONTRACT_FORMAT=CONTRACT_FORMAT; base.CONTRACT_STATUS=CONTRACT_STATUS; base.OUTPUT_FORMAT=OUTPUT_FORMAT; base.OUTPUT_STATUS=OUTPUT_STATUS; base.AUTHORIZATION=AUTHORIZATION; base.RUNTIME=RUNTIME; base.EXECUTION_BOUNDARY=EXECUTION_BOUNDARY; base.CONTRACT_SOURCE_ORDER=CONTRACT_SOURCE_ORDER; base.AUTHORITY_SOURCE_ORDER=AUTHORITY_SOURCE_ORDER; base.SOURCE_ALIASES=SOURCE_ALIASES; base.CONTRACT_TOP_KEYS=CONTRACT_TOP_KEYS; base.AUTH_TOP_KEYS=AUTH_TOP_KEYS; base.CHECK_KEYS=CHECK_KEYS
 return base
def recursive_diff(a,b,path=()):
 if type(a) is not type(b): return [{'path':list(path),'original':a,'normalized':b}]
 if isinstance(a,dict):
  if set(a)!=set(b): return [{'path':list(path),'original_keys':sorted(a),'normalized_keys':sorted(b)}]
  return sum((recursive_diff(a[k],b[k],path+(k,)) for k in sorted(a)),[])
 if isinstance(a,list):
  if len(a)!=len(b): return [{'path':list(path),'original_length':len(a),'normalized_length':len(b)}]
  return sum((recursive_diff(x,y,path+(i,)) for i,(x,y) in enumerate(zip(a,b))),[])
 return [] if a==b else [{'path':list(path),'original':a,'normalized':b}]
def service_health():
 out={}
 for n,u in (('8005_v1_health','http://127.0.0.1:8005/v1/health'),('18084_health','http://127.0.0.1:18084/health')):
  with urllib.request.urlopen(u,timeout=10) as r:
   b=r.read(); out[n]={'http_code':r.status,'body_sha256':hashlib.sha256(b).hexdigest(),'body_bytes':len(b),'returncode':0}
 return out
def phase_pids():
 marks=('launch_v524_v523_phase_a_cache_qualification.py','generate_v524_v523_phase_a_cache_qualification.py','generate_v524_v523_phase_a_cache_qualification_worker.py','reconcile_v525_v524_phase_a_worker_receipt_format_readonly.py'); out=[]
 for e in Path('/proc').iterdir():
  if not e.name.isdigit() or int(e.name)==os.getpid(): continue
  try: cmd=(e/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  except (OSError,ProcessLookupError): continue
  if any(m in cmd for m in marks): out.append({'pid':int(e.name),'cmdline':cmd.strip()})
 return sorted(out,key=lambda x:x['pid'])
def absence_paths():
 hist={'authority_root':AUTHORITY_ROOT,'authority_prep':AUTHORITY_ROOT.with_name(AUTHORITY_ROOT.name+'.registration-prep'),'reconciliation_root':RECONCILIATION_ROOT,'reconciliation_prep':RECONCILIATION_ROOT.with_name(RECONCILIATION_ROOT.name+'.reconciliation-prep'),'external_evidence_root':EXTERNAL_EVIDENCE_ROOT,'external_evidence_prep':EXTERNAL_EVIDENCE_PREP}
 return hist,{k:v for k,v in hist.items() if k!='authority_root'}
def validate_state(contract):
 auth_tree=exact_tree(OLD_AUTH); attempt_tree=exact_tree(ATTEMPT); qual_tree=exact_tree(QUAL)
 if tree_tuple(auth_tree)!=AUTH_TREE or tree_tuple(attempt_tree)!=ATTEMPT_TREE or tree_tuple(qual_tree)!=QUAL_TREE: raise RuntimeError('tree anchor')
 auth_rec=regular(OLD_AUTH_RECEIPT,'15aa29b6d83a807bfeed2298c75ddbe158d5bffdd508a2145a3164aecf6bcedf',64970); auth=json.loads(OLD_AUTH_RECEIPT.read_text())
 if len(auth)!=71 or auth.get('passed') is not True or len(auth.get('checks',{}))!=33 or not all(type(v) is bool and v is True for v in auth['checks'].values()) or len(auth.get('source_closure',{}))!=27: raise RuntimeError('old authority')
 for r in auth['source_closure'].values():
  if regular(r['path'],r['sha256'],r['logical_bytes'])!=r: raise RuntimeError('old source')
 names={'v524_attempt_intent':ATTEMPT/'intent.json','v524_attempt_stdout':ATTEMPT/'phase_a_stdout.log','v524_attempt_stderr':ATTEMPT/'phase_a_stderr.log','v524_attempt_terminal':ATTEMPT/'terminal_receipt.json','v524_qualification_terminal':QUAL/'terminal_receipt.json','v524_qualification_report':QUAL/'qualification_report.json','v524_qualification_independent_audit':QUAL/'independent_final_audit.json','v524_process_a_receipt':QUAL/'process_a/receipt.json','v524_process_b_receipt':QUAL/'process_b/receipt.json'}
 records={k:regular(v) for k,v in names.items()}; terminal=json.loads((ATTEMPT/'terminal_receipt.json').read_text()); qt=json.loads((QUAL/'terminal_receipt.json').read_text()); qr=json.loads((QUAL/'qualification_report.json').read_text()); qa=json.loads((QUAL/'independent_final_audit.json').read_text())
 if terminal.get('status')!='failed_no_retry' or terminal.get('passed') is not False or terminal.get('error')!='qualification A RNG receipt' or terminal.get('retry_authorized') is not False or terminal.get('immutable_snapshots_exactly_equal') is not True or terminal.get('immutable_snapshot_before')!=terminal.get('immutable_snapshot_after'): raise RuntimeError('attempt terminal')
 if any(x.get('passed') is not True for x in (qt,qr,qa)): raise RuntimeError('qualification terminal')
 original={'process_a':json.loads((QUAL/'process_a/receipt.json').read_text()),'process_b':json.loads((QUAL/'process_b/receipt.json').read_text())}; count_keys=('calls','call_events_count','warning_count','raw_warning_count','rng_unchanged_count','rng_isolation_event_count','rng_isolation_delegate_started_count','rng_isolation_delegate_completed_count','rng_isolation_python_all_four_stages_equal_count','rng_isolation_numpy_all_four_stages_equal_count','rng_isolation_torch_cpu_exit_restored_count','rng_isolation_torch_cuda_exit_restored_count','durable_call_started_markers','durable_call_completed_markers')
 for n,role in (('process_a','A'),('process_b','B')):
  r=original[n]
  if r.get('format')!=OLD_FORMAT or r.get('passed') is not True or r.get('role')!=role or any(type(r.get(k)) is not int or r.get(k)!=1000 for k in count_keys) or r.get('other_warning_count')!=0: raise RuntimeError('worker receipt')
 if csha(original)!=ORIGINAL_CSHA: raise RuntimeError('original digest')
 normalized=copy.deepcopy(original); normalized['process_a']['format']=NORMALIZED_FORMAT; normalized['process_b']['format']=NORMALIZED_FORMAT; diff=recursive_diff(original,normalized)
 expected=[{'path':['process_a','format'],'original':OLD_FORMAT,'normalized':NORMALIZED_FORMAT},{'path':['process_b','format'],'original':OLD_FORMAT,'normalized':NORMALIZED_FORMAT}]
 if diff!=expected or csha(diff)!=DIFF_CSHA or csha(normalized)!=NORMALIZED_CSHA: raise RuntimeError('normalization')
 forensic=json.loads(FORENSIC.read_text()); fr=regular(FORENSIC,FORENSIC_SHA,FORENSIC_BYTES)
 if forensic.get('status')!='failed_no_retry_computation_complete_readonly_reconciliation_required' or forensic.get('readonly_normalization_contract',{}).get('recursive_diff_canonical_sha256')!=DIFF_CSHA or forensic.get('sole_mismatch',{}).get('false_predicate_count')!=2: raise RuntimeError('forensic')
 norm={'original_worker_receipts':original,'normalized_worker_receipts':normalized,'original_worker_receipts_canonical_sha256':ORIGINAL_CSHA,'normalized_worker_receipts_canonical_sha256':NORMALIZED_CSHA,'recursive_diff':diff,'recursive_diff_count':2,'recursive_diff_canonical_sha256':DIFF_CSHA,'recursive_diff_full_paths':[['process_a','format'],['process_b','format']],'standalone_phase_a_success_claimed':False,'new_computation_claimed':False}
 exact={'v524_authority_registration_tree':auth_tree,'v524_authority_receipt':auth_rec,'v524_authority_semantics':{'passed':True,'top_key_count':71,'check_count':33,'source_closure_count':27,'current':True,'consumed_by_v524_attempt':True},'v524_attempt_tree':attempt_tree,**records,'v524_execution_partition':{'launcher_invocations':1,'nested_phase_a_driver_invocations':1,'direct_phase_a_driver_invocations':0,'phase_a_worker_invocations':2,'rng_proxy_delegate_invocations':2000,'phase_a_retry_invocations':0,'training_invocations':0,'oof_invocations':0},'v524_qualification_tree':qual_tree,'normalization_contract':norm}
 for k,v in exact.items():
  if not json_exact(contract.get(k),v): raise RuntimeError('contract anchor '+k)
 if service_health()!=terminal['v218_health_after'] or terminal['v218_health_before']!=terminal['v218_health_after'] or phase_pids(): raise RuntimeError('runtime current')
 return exact,fr
def main():
 if os.sys.argv[1:]==['--synthetic-self-test']:
  checks={'seed':1661==1661,'source_count':len(CONTRACT_SOURCE_ORDER)==16,'auth_source_count':len(AUTHORITY_SOURCE_ORDER)==17,'diff2':len(recursive_diff({'process_a':{'format':OLD_FORMAT},'process_b':{'format':OLD_FORMAT}},{'process_a':{'format':NORMALIZED_FORMAT},'process_b':{'format':NORMALIZED_FORMAT}}))==2,'authorization':AUTHORIZATION['readonly_reconciler_invocations_authorized']==1 and AUTHORIZATION['phase_a_replay_authorized'] is False,'publish':load_base().publication_self_test()}; print(json.dumps({'passed':all(checks.values()),'checks':checks,'checks_sha256':csha(checks)},sort_keys=True)); return 0 if all(checks.values()) else 1
 p=argparse.ArgumentParser(); p.add_argument('--contract',type=Path,required=True);p.add_argument('--contract-sha',required=True);p.add_argument('--contract-bytes',type=int,required=True);p.add_argument('--materializer-source',type=Path,required=True);p.add_argument('--materializer-sha',required=True);p.add_argument('--materializer-bytes',type=int,required=True);p.add_argument('--authority-root',type=Path,required=True);p.add_argument('--read-only-preflight',action='store_true');a=p.parse_args()
 if a.contract!=CONTRACT_PATH or a.materializer_source!=SELF_PATH or a.authority_root!=AUTHORITY_ROOT: raise RuntimeError('canonical CLI')
 contract_record=regular(CONTRACT_PATH,a.contract_sha,a.contract_bytes); materializer_record=regular(SELF_PATH,a.materializer_sha,a.materializer_bytes); contract=json.loads(CONTRACT_PATH.read_text())
 if set(contract)!=CONTRACT_TOP_KEYS or contract.get('format')!=CONTRACT_FORMAT or contract.get('status')!=CONTRACT_STATUS or type(contract.get('seed')) is not int or contract.get('seed')!=1661: raise RuntimeError('contract schema')
 sources=contract['source_closure']; observed={k:regular(v['path'],v['sha256'],v['logical_bytes']) for k,v in sources.items()}
 if observed!=sources or list(contract['source_role_order'])!=CONTRACT_SOURCE_ORDER or set(sources)!=set(CONTRACT_SOURCE_ORDER) or contract['source_aliases']!={k:SOURCE_ALIASES[k] for k in CONTRACT_SOURCE_ORDER} or contract['source_closure_sha256']!=csha(sources) or sources['authority_materializer']!=materializer_record or sources['readonly_reconciler']!=regular(RECONCILER,RECONCILER_SHA,RECONCILER_BYTES) or sources['failure_forensic']!=regular(FORENSIC,FORENSIC_SHA,FORENSIC_BYTES): raise RuntimeError('source closure')
 expected_paths={'authority_materializer':SELF_PATH,'readonly_reconciler':RECONCILER,'failure_forensic':FORENSIC,'v524_authority_design_contract':OLD_CONTRACT,'v524_authority_materializer':OLD_MATERIALIZER,'v524_authority_receipt':OLD_AUTH_RECEIPT,'v524_attempt_intent':ATTEMPT/'intent.json','v524_attempt_stdout':ATTEMPT/'phase_a_stdout.log','v524_attempt_stderr':ATTEMPT/'phase_a_stderr.log','v524_attempt_terminal':ATTEMPT/'terminal_receipt.json','v524_qualification_terminal':QUAL/'terminal_receipt.json','v524_qualification_report':QUAL/'qualification_report.json','v524_qualification_independent_audit':QUAL/'independent_final_audit.json','v524_process_a_receipt':QUAL/'process_a/receipt.json','v524_process_b_receipt':QUAL/'process_b/receipt.json','rng_proxy_current_only':PROXY}
 if any(sources[k]['path']!=str(v) for k,v in expected_paths.items()): raise RuntimeError('source role path')
 if contract['authority_materializer_source']!=materializer_record or contract['readonly_reconciler_source']!=sources['readonly_reconciler'] or contract['failure_forensic_source']!=sources['failure_forensic']: raise RuntimeError('aliases')
 exact,fr=validate_state(contract); historical,current=absence_paths()
 expected_lineage={'name':'v525_v524_phase_a_worker_receipt_format_readonly_reconciliation','fresh_authority_root':str(AUTHORITY_ROOT),'fresh_reconciliation_output_root':str(RECONCILIATION_ROOT),'supersedes_v524_failed_launcher_consumer_no_retry':True,'qualification_computation_replay_required':False,'sole_normalization':'process_a_and_process_b_receipt_format'}
 if not json_exact(contract.get('lineage'),expected_lineage) or contract.get('historical_absences')!={k:{'path':str(v)} for k,v in historical.items()} or contract.get('current_absences_after_authority')!={k:{'path':str(v)} for k,v in current.items()}: raise RuntimeError('lineage/absence contract')
 if any(os.path.lexists(x) for x in historical.values()): raise RuntimeError('fresh roots')
 if not json_exact(contract.get('reconciler_candidate_contract'),RECONCILER_CANDIDATE_CONTRACT) or not json_exact(contract.get('external_terminal_evidence_contract'),EXTERNAL_TERMINAL_EVIDENCE_CONTRACT): raise RuntimeError('candidate/external terminal contract')
 if not json_exact(contract.get('authorization'),AUTHORIZATION) or not json_exact(contract.get('runtime_observation'),RUNTIME) or not json_exact(contract.get('execution_boundary'),EXECUTION_BOUNDARY): raise RuntimeError('boundary')
 files={'contract':str(CONTRACT_PATH),'materializer':str(SELF_PATH),**{'source_'+k:v['path'] for k,v in sources.items()}}; trees={'v524_authority':str(OLD_AUTH),'v524_attempt':str(ATTEMPT),'v524_qualification':str(QUAL)}; base=load_base(); pre=base.snapshot(files,trees,current); checks={k:True for k in CHECK_KEYS}; checks['gpu_empty']=base.gpu_empty();checks['no_live_phase_a_process']=not phase_pids()
 authority_sources={'authority_design_contract':contract_record,**sources}; receipt={'format':OUTPUT_FORMAT,'status':OUTPUT_STATUS,'passed':True,'source_closure':authority_sources,'source_role_order':AUTHORITY_SOURCE_ORDER,'source_aliases':SOURCE_ALIASES,'source_closure_sha256':csha(authority_sources),**{SOURCE_ALIASES[k]:v for k,v in authority_sources.items()},**exact,'reconciler_candidate_contract':RECONCILER_CANDIDATE_CONTRACT,'external_terminal_evidence_contract':EXTERNAL_TERMINAL_EVIDENCE_CONTRACT,'reconciliation_output_root':str(RECONCILIATION_ROOT),'historical_absences':{k:{'path':str(v),'absent':True} for k,v in historical.items()},'required_absences':{k:{'path':str(v),'absent':True} for k,v in current.items()},'checks':checks,'check_keys':CHECK_KEYS,'check_key_set_sha256':csha(CHECK_KEYS),'checks_sha256':csha(checks),'input_pre_snapshot':pre,'input_post_snapshot':base.snapshot(files,trees,current),'input_snapshots_exactly_equal':True,'authorization':AUTHORIZATION,'runtime_observation':RUNTIME,'execution_boundary':EXECUTION_BOUNDARY}
 schema=contract['authority_receipt_contract']
 if set(receipt)!=AUTH_TOP_KEYS or schema.get('format')!=OUTPUT_FORMAT or schema.get('status')!=OUTPUT_STATUS or schema.get('top_keys')!=sorted(AUTH_TOP_KEYS) or schema.get('check_keys')!=CHECK_KEYS or schema.get('check_key_set_sha256')!=csha(CHECK_KEYS) or schema.get('checks_sha256')!=csha({k:True for k in CHECK_KEYS}) or not json_exact(schema.get('reconciler_candidate_contract_exact'),RECONCILER_CANDIDATE_CONTRACT) or not json_exact(schema.get('external_terminal_evidence_contract_exact'),EXTERNAL_TERMINAL_EVIDENCE_CONTRACT) or not json_exact(schema.get('authorization_exact'),AUTHORIZATION) or not json_exact(schema.get('runtime_observation_exact'),RUNTIME) or not json_exact(schema.get('execution_boundary_exact'),EXECUTION_BOUNDARY) or receipt['input_pre_snapshot']!=receipt['input_post_snapshot'] or not all(type(v) is bool and v is True for v in checks.values()): raise RuntimeError('receipt gates')
 if a.read_only_preflight: print(json.dumps({'passed':True,'status':'passed_read_only_preflight_no_materialization','contract_top_count':len(contract),'authority_top_count':len(receipt),'check_count':len(CHECK_KEYS),'source_count':len(sources),'authority_source_count':len(authority_sources)},sort_keys=True)); return 0
 stable={k:v for k,v in current.items() if k!='authority_prep'}; stable_pre=base.snapshot(files,trees,stable); base.publish_exact1(AUTHORITY_ROOT,receipt,files,trees,stable,stable_pre); return 0
if __name__=='__main__': raise SystemExit(main())
