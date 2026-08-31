#!/usr/bin/env python3
"""Exact fixed-budget all200 training; no search, early stop, retry, or resume."""
from __future__ import annotations
import argparse, copy, ctypes, hashlib, importlib, importlib.util, io, json, os, random, signal, stat, sys, types
from pathlib import Path
import numpy as np

ROOT=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge'); J=ROOT/'artifacts/strict_track2_joint_augmentation_20260810'; S=ROOT/'pipeline/scripts'
SELF=S/'train_v542_v541_fixed_budget_all200_runtime_package_repair.py'; PREREG=S/'v542_v541_fixed_budget_all200_training_runtime_package_repair_preregistration.json'; MANIFEST=S/'v542_v541_fixed_budget_all200_training_runtime_package_repair_manifest.json'; CONTRACT=S/'v542_v541_fixed_budget_all200_training_runtime_package_repair_execution_authority_contract.json'
AUDITOR=S/'audit_v542_v541_fixed_budget_all200_runtime_package_repair.py'; LAUNCHER=S/'launch_v542_v541_fixed_budget_all200_runtime_package_repair.py'
OLD_PREREG=J/'v482_temporal8_residual_s0_r3_seed1624_20260824/preregistration.json'; OLD_TRAINER=S/'train_v482_temporal8_residual_5fold.py'; OLD_CONTRACT=S/'v482_temporal_film_residual_model_design_contract.json'
CACHE_MANIFEST=Path('/root/v524_v523_phase_a_cache_qualification_seed1660_20260826/process_a/cache/manifest.json')
PIPELINE_ROOT=ROOT/'pipeline'; WAM_PACKAGE=PIPELINE_ROOT/'wam_pipeline'; WAM_INIT=WAM_PACKAGE/'__init__.py'; V169_RUNTIME=WAM_PACKAGE/'v169_arm_routed_runtime.py'; V482_RUNTIME=WAM_PACKAGE/'v482_temporal8_residual_runtime.py'
PRODUCTION_PACKAGE_EXACT={
 str(WAM_INIT):('1d8c8f56ecd70fa05b21f3fe8b870a12424dbc5ea6a6ad9944d202023dc66f2a',118),
 str(V169_RUNTIME):('0044d49ae2a3083f0b638b701fb398b407d7c436c3bb0ee1798a3233e076c7c1',3277),
 str(V482_RUNTIME):('6939e3f6f52c1eb83bfd4324372d7e75ed4ee6d493777a39f38c26e5f4db7471',21745),
 str(OLD_TRAINER):('674b68afed3b6c39663d629db38be11aa2c752e54692458b43d4254eb08b8c1d',48428),
}
AUTH_ROOT=J/'v542_v541_fixed_budget_all200_training_runtime_package_repair_execution_authority_seed1673_20260828'; AUTH=AUTH_ROOT/'authority_receipt.json'; AUTH_PREP=AUTH_ROOT.with_name(AUTH_ROOT.name+'.authority-prep')
ATTEMPT_ROOT=J/'v542_v541_fixed_budget_all200_training_runtime_package_repair_attempt_seed1673_20260828'; ATTEMPT_PREP=ATTEMPT_ROOT.with_name(ATTEMPT_ROOT.name+'.attempt-prep')
OUTPUT=Path('/root/v542_v541_fixed_budget_all200_training_runtime_package_repair_seed1624_20260828'); OUTPUT_PREP=OUTPUT.with_name(OUTPUT.name+'.training-prep')
EXACT6=sorted(['attempt_intent.json','checkpoint.pt','independent_audit.json','source_manifest.json','training_events.ndjson','training_receipt.json'])
CHECKPOINT_FORMAT='strict-track2-v542-v541-fixed-budget-all200-checkpoint-v1'; RENAME_NOREPLACE=1; AT_FDCWD=-100
ACTIVE_ROLE_ORDER=['authority_design_contract','authority_materializer','fixed_budget_training_preregistration','fixed_budget_training_manifest','fixed_budget_trainer','fixed_budget_training_independent_auditor','fixed_budget_training_launcher']
ACTIVE_PATHS={'authority_design_contract':str(CONTRACT),'authority_materializer':str(S/'materialize_v542_v541_fixed_budget_all200_training_runtime_package_repair_execution_authority.py'),'fixed_budget_training_preregistration':str(PREREG),'fixed_budget_training_manifest':str(MANIFEST),'fixed_budget_trainer':str(SELF),'fixed_budget_training_independent_auditor':str(AUDITOR),'fixed_budget_training_launcher':str(LAUNCHER)}
SOURCE_ROLE_ORDER='authority_materializer fixed_budget_training_preregistration fixed_budget_training_manifest fixed_budget_trainer fixed_budget_training_independent_auditor fixed_budget_training_launcher v541_failure_forensic v541_unconsumed_authority_receipt v541_failed_transport_script v541_failed_transport_record v540_authority_receipt v540_attempt_intent v540_execution_receipt v540_independent_audit v540_public_s1_events v540_public_s1_metrics v540_source_manifest v540_zero_update_receipt v540_zero_update_trace v534_authority_receipt v534_actual_oof_execution_receipt v534_actual_oof_independent_audit v534_actual_oof_source_manifest v482_preregistration v482_runtime_source v482_model_design_contract v482_trainer_source v524_cache_manifest v524_qualification_terminal v524_qualification_report v524_qualification_audit v474_public_s1_selection'.split()
SOURCE_PATHS={'authority_materializer':ACTIVE_PATHS['authority_materializer'],'fixed_budget_training_preregistration':str(PREREG),'fixed_budget_training_manifest':str(MANIFEST),'fixed_budget_trainer':str(SELF),'fixed_budget_training_independent_auditor':str(AUDITOR),'fixed_budget_training_launcher':str(LAUNCHER),'v541_failure_forensic':str(S/'v542_v541_fixed_budget_training_runtime_package_import_failure_forensic.json'),'v541_unconsumed_authority_receipt':str(J/'v541_v540_fixed_budget_all200_training_execution_authority_seed1672_20260828/authority_receipt.json'),'v541_failed_transport_script':'/root/v541_v540_fixed_budget_all200_training_once.sh','v541_failed_transport_record':str(S/'v541_v540_fixed_budget_all200_training_transport_deployment_record.json'),'v540_authority_receipt':str(J/'v540_v539_public_s1_zero_update_poststage_owned_output_prep_repair_execution_authority_seed1671_20260828/authority_receipt.json'),'v540_attempt_intent':'/root/v540_v539_public_s1_zero_update_gate_seed1671_20260828/attempt_intent.json','v540_execution_receipt':'/root/v540_v539_public_s1_zero_update_gate_seed1671_20260828/execution_receipt.json','v540_independent_audit':'/root/v540_v539_public_s1_zero_update_gate_seed1671_20260828/independent_audit.json','v540_public_s1_events':'/root/v540_v539_public_s1_zero_update_gate_seed1671_20260828/public_s1_events.ndjson','v540_public_s1_metrics':'/root/v540_v539_public_s1_zero_update_gate_seed1671_20260828/public_s1_metrics.json','v540_source_manifest':'/root/v540_v539_public_s1_zero_update_gate_seed1671_20260828/source_manifest.json','v540_zero_update_receipt':'/root/v540_v539_public_s1_zero_update_gate_seed1671_20260828/zero_update_receipt.json','v540_zero_update_trace':'/root/v540_v539_public_s1_zero_update_gate_seed1671_20260828/zero_update_trace.json','v534_authority_receipt':str(J/'v534_v533_actual_oof_environment_repair_execution_authority_seed1667_20260827/authority_receipt.json'),'v534_actual_oof_execution_receipt':'/root/v534_v533_actual_oof_seed1667_20260827/execution_receipt.json','v534_actual_oof_independent_audit':'/root/v534_v533_actual_oof_seed1667_20260827/independent_audit.json','v534_actual_oof_source_manifest':'/root/v534_v533_actual_oof_seed1667_20260827/source_manifest.json','v482_preregistration':str(OLD_PREREG),'v482_runtime_source':str(ROOT/'pipeline/wam_pipeline/v482_temporal8_residual_runtime.py'),'v482_model_design_contract':str(OLD_CONTRACT),'v482_trainer_source':str(OLD_TRAINER),'v524_cache_manifest':str(CACHE_MANIFEST),'v524_qualification_terminal':'/root/v524_v523_phase_a_cache_qualification_seed1660_20260826/terminal_receipt.json','v524_qualification_report':'/root/v524_v523_phase_a_cache_qualification_seed1660_20260826/qualification_report.json','v524_qualification_audit':'/root/v524_v523_phase_a_cache_qualification_seed1660_20260826/independent_final_audit.json','v474_public_s1_selection':str(J/'v474_v473_parent_s1_seed1617_r6_20260824/action_only_selection.json')}
SOURCE_ALIASES={role:f'{role}_source' for role in SOURCE_ROLE_ORDER}; SOURCE_ALIASES.update({'v482_runtime_source':'v482_runtime_source_source'})
AUTHORITY_SOURCE_ALIASES={'authority_design_contract':'authority_design_contract',**SOURCE_ALIASES}
EXPECTED_LINEAGE={'canonical_v542_training_not_performed':True,'fresh_fixed_budget_training_authority_only':True,'formal_all200_exact500_budget_frozen':True,'v541_failed_training_output_not_published':True,'v541_authority_consumed_field_zero':True,'v541_failed_lineage_retry_authorized':False,'v541_runtime_package_import_failure_forensic_current':True,'production_package_qualified_loader_repair_frozen':True,'v540_public_s1_zero_update_passed_current':True,'v540_metrics_semantic_parse_or_config_selection_authorized':False,'v540_durable_boundary_one_current':True,'v534_actual_oof_folds_reused_readonly':True,'v524_qualification_exact20_readonly':True,'v482_model_and_training_components_frozen':True,'base_model_data_and_sources_readonly':True,'hidden_private_final_reward_inputs_denied':True,'retry_search_earlystop_resume_authorized':False}
CONTRACT_BASE_KEYS=set('format status seed lineage active_source_role_order active_source_paths active_source_records source_closure source_role_order source_aliases source_closure_sha256 budget_contract authorization base_training_contract output_contract v540_gate_transition historical_absences current_absences_after_authority authority_receipt_contract runtime_observation execution_boundary'.split())
AUTHORITY_BASE_KEYS=set('format status passed seed lineage active_source_role_order active_source_paths active_source_records source_closure source_role_order source_aliases source_closure_sha256 budget_contract authorization base_training_contract output_contract v540_gate_transition fresh_attempt_root fresh_output_root historical_absences required_absences checks check_keys check_key_set_sha256 checks_sha256 input_pre_snapshot input_post_snapshot input_snapshots_exactly_equal runtime_observation execution_boundary'.split())
CONTRACT_KEYS=CONTRACT_BASE_KEYS|set(SOURCE_ALIASES.values()); AUTHORITY_KEYS=AUTHORITY_BASE_KEYS|set(AUTHORITY_SOURCE_ALIASES.values())
RECEIPT_CONTRACT_KEYS=set('active_source_paths_exact active_source_role_order_exact authority_active_source_records_count authority_design_record_included authorization_exact base_training_contract_exact budget_contract_exact check_key_set_sha256 check_keys checks_sha256 contract_active_source_records_count contract_active_source_records_exact contract_design_record_excluded_to_avoid_self_hash_cycle execution_boundary_exact format output_contract_exact runtime_observation_exact status top_keys v540_gate_transition_exact source_closure_contract_count'.split())
CHECK_KEYS='contract_schema preregistration_schema manifest_schema authorization_exact_strict budget_arithmetic_exact schedule_permutation_exact runtime_boundary_exact execution_boundary_exact source_role_order_exact source_aliases_exact source_closure32_current active_sources6_current contract_active_records6 authority_active_records7 v541_failure_ancestry_current v540_authority_immutable_consumed0 v540_durable_boundary1 v540_metrics_not_parsed v534_oof_current_readonly v524_qualification_exact20_readonly v524_cache_current_readonly v482_package_qualified_training_components_current v474_selection_current base_data_sources_readonly fresh_roots6_absent input_prepost_equal services_current gpu_empty execution_pids_empty all_unsafe_disabled publish_noreplace_exact1 lineage_exact two_stage_boundary_only output_exact6_contract'.split()
SNAPSHOT_KEYS={'absences','canonical_sha256','files','gpu_compute_pids','relevant_execution_pids','services','trees'}
HISTORICAL_ABSENCES={'authority_root':{'path':str(AUTH_ROOT),'absent':True},'authority_prep':{'path':str(AUTH_PREP),'absent':True},'attempt_root':{'path':str(ATTEMPT_ROOT),'absent':True},'attempt_prep':{'path':str(ATTEMPT_PREP),'absent':True},'output_root':{'path':str(OUTPUT),'absent':True},'output_prep':{'path':str(OUTPUT_PREP),'absent':True}}
CURRENT_ABSENCES={key:value for key,value in HISTORICAL_ABSENCES.items() if key!='authority_root'}
EXPECTED_SERVICES={'18084':{'body_bytes':18,'body_sha256':'31bf75f4c0a97cc1f7b60df824fa390b3be9ba014f29b63c87c698ba63d9a9fd','http_code':200,'json_model':{'status':'ready'}},'8005':{'body_bytes':106,'body_sha256':'08dbc59225418d4d3064f9e122caeadbdf7c740d38fcbbaf708ad9baa44abab0','http_code':200,'json_model':{'api_version':'1.0','model_version':'track2-v218-public-knn-blend-alpha070-route-aware','status':'ready'}}}
EXPECTED_TREES={'qualification_exact20':{'canonical_json_triples_digest_sha256':'223308d02ed5a4a082b4de639292030444a627bbbac7f361fac1f9c6a7fcbd1f','file_count':20,'logical_file_bytes':950124602,'sha256sum_lines_digest_sha256':'6cdde75692d37b7eead3333d867b22dc6404f545bb5861b2f30b425fb4270dac'},'v534_actual_oof_exact11':{'canonical_json_triples_digest_sha256':'7bbc2464dc6fc600e4d2fd3ed8cc012ceb14839260a96c878fb667c3ad6b6eaa','file_count':11,'logical_file_bytes':1581516,'sha256sum_lines_digest_sha256':'c6ed14b53d92b7da929a625e624722723010af2c64c076a7bfb2a64a2bd5cc9a'},'v540_public_s1_zero_update_exact8':{'canonical_json_triples_digest_sha256':'71117c8570f8fd62865eb2573b2b2a402eeec78ca8a83881c9bdd442c434459f','file_count':8,'logical_file_bytes':6999111,'sha256sum_lines_digest_sha256':'24dee8d2fee5ee7e8a2e6aa3ca130ee9ad0935c8def7fed5baa1ceb35bd544fc'}}
EXPECTED_RUNTIME_OBSERVATION={'authority_materializer_invocations':1,'fixed_budget_training_boundary_invocations':0,'training_launcher_invocations':0,'training_trainer_invocations':0,'auditor_import_invocations':0,'forward_invocations':0,'loss_invocations':0,'backward_invocations':0,'optimizer_zero_grad_invocations':0,'gradient_clip_invocations':0,'optimizer_step_invocations':0,'scheduler_instances':0,'scheduler_step_invocations':0,'final_checkpoint_writes':0,'output_publication_invocations':0,'config_selection_invocations':0,'early_stop_invocations':0,'intermediate_checkpoint_writes':0,'resume_invocations':0,'retry_invocations':0,'search_invocations':0,'fold_training_invocations':0,'actual_oof_execution_invocations':0,'public_s1_zero_update_invocations':0,'hidden_private_final_reward_read_invocations':0}
EXPECTED_EXECUTION_BOUNDARY={'authority_materialization_only':True,'training_execution_authorized':False,'direct_training_authorized':False,'fixed_budget_training_boundary_invocations':0,'training_launcher_invocations':0,'training_trainer_invocations':0,'auditor_import_invocations':0,'optimizer_step_invocations':0,'output_publication_invocations':0,'hidden_private_final_reward_inputs_authorized':False,'metrics_semantic_parse_or_config_selection_authorized':False,'authority_publication_noreplace':True,'postcommit_diagnostics_best_effort_success_priority':True,'rollback_retry_or_second_publish_after_commit':False}
PREREG_KEYS=set('active_source_paths active_source_role_order authorization base_training_contract budget_contract format fresh_paths output_contract seed status v540_gate_transition'.split())
MANIFEST_KEYS=set('active_source_paths active_source_role_order base_input_records budget_contract execution_contract format output_contract schedule_contract seed status v540_gate_records'.split())
EXPECTED_AUTHORIZATION={'auditor_import_invocations_authorized':1,'backward_invocations_authorized':500,'config_selection_invocations_authorized':0,'early_stop_invocations_authorized':0,'final_checkpoint_writes_authorized':1,'fixed_budget_training_boundary_invocations_authorized':1,'fixed_budget_training_boundary_invocations_consumed':0,'forward_invocations_authorized':500,'gradient_clip_invocations_authorized':500,'hidden_private_or_final_input_authorized':False,'intermediate_checkpoint_writes_authorized':0,'loss_invocations_authorized':500,'optimizer_step_invocations_authorized':500,'optimizer_zero_grad_invocations_authorized':500,'output_publication_invocations_authorized':1,'resume_invocations_authorized':0,'retry_authorized':False,'scheduler_instances_authorized':0,'scheduler_step_invocations_authorized':0,'search_invocations_authorized':0,'training_authorized':True,'training_launcher_invocations_authorized':1,'training_trainer_invocations_authorized':1}
EXPECTED_BASE_TRAINING={'action_head_only':True,'base_and_inputs_readonly':True,'channels':16,'gradient_accumulation_steps':1,'gradient_clip_global_norm':1.0,'model':'TemporalResidualUNet128FiLM','optimizer':{'betas':[0.9,0.95],'eps':1e-08,'learning_rate':0.0003,'name':'AdamW','weight_decay':0.0001},'precision':{'autocast':'bf16','loss_accumulation':'fp32','master_parameters':'fp32'},'scheduler':'none','trainable_parameter_count':878579,'training_seed':1624}
EXPECTED_OUTPUT_CONTRACT={'build_order':['attempt_intent.json','training_events.ndjson','checkpoint.pt','training_receipt.json','independent_audit.json','source_manifest.json'],'exact_count':6,'exact_names':EXACT6,'held_o_rdwr_excl_nofollow_members':6,'noreplace_directory_publication':True,'postcommit_success_priority':True}
EXPECTED_FAILURE_ANCESTRY={
 'v541_failure_forensic':{'path':str(S/'v542_v541_fixed_budget_training_runtime_package_import_failure_forensic.json'),'sha256':'58e0067017de098441531227309ca39bc5150aab63a862bb397b12dd4e891a2e','logical_bytes':5383},
 'v541_unconsumed_authority_receipt':{'path':str(J/'v541_v540_fixed_budget_all200_training_execution_authority_seed1672_20260828/authority_receipt.json'),'sha256':'c3ad20c08c4f7d18f1d0fd462f1c6875c053b99ffb2e9964c1a9e38a3da3ce63','logical_bytes':57048},
 'v541_failed_transport_script':{'path':'/root/v541_v540_fixed_budget_all200_training_once.sh','sha256':'9b4aa7187555f5ab197021fedfe1e3fdb857f61804468e9e6f413426ec630281','logical_bytes':7037},
 'v541_failed_transport_record':{'path':str(S/'v541_v540_fixed_budget_all200_training_transport_deployment_record.json'),'sha256':'67d0cca5b591265af5684294a4d0501efeed1850c8efc2de4eb11a3553d111b8','logical_bytes':1249},
}

def cbytes(x): return (json.dumps(x,sort_keys=True,separators=(',',':'))+'\n').encode()
def sha_bytes(raw): return hashlib.sha256(raw).hexdigest()
def canonical_sha(x): return sha_bytes(json.dumps(x,sort_keys=True,separators=(',',':')).encode())
def json_exact(left,right): return json.dumps(left,sort_keys=True,separators=(',',':'))==json.dumps(right,sort_keys=True,separators=(',',':'))
def exact_file(path,digest=None,logical_bytes=None):
 path=Path(path); st=os.lstat(path)
 if not stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode): raise RuntimeError('regular '+str(path))
 raw=path.read_bytes(); observed=sha_bytes(raw)
 if digest is not None and observed!=digest: raise RuntimeError('sha '+str(path))
 if logical_bytes is not None and len(raw)!=logical_bytes: raise RuntimeError('bytes '+str(path))
 return {'path':str(path),'sha256':observed,'logical_bytes':len(raw)}
def exact_record(row):
 if set(row)!={'path','sha256','logical_bytes'} or type(row['logical_bytes']) is not int: raise RuntimeError('record')
 return exact_file(row['path'],row['sha256'],row['logical_bytes'])
def load_json(path): return json.loads(Path(path).read_bytes())
def import_source(name,path):
 spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
def load_production_training_components():
 for path,(digest,logical_bytes) in PRODUCTION_PACKAGE_EXACT.items(): exact_file(path,digest,logical_bytes)
 pipeline_literal=str(PIPELINE_ROOT)
 before=list(sys.path); inserted=False
 cached_before={name:sys.modules.get(name) for name in ('wam_pipeline','wam_pipeline.v169_arm_routed_runtime','wam_pipeline.v482_temporal8_residual_runtime')}
 for name,module in cached_before.items():
  if module is not None:
   expected={'wam_pipeline':WAM_INIT,'wam_pipeline.v169_arm_routed_runtime':V169_RUNTIME,'wam_pipeline.v482_temporal8_residual_runtime':V482_RUNTIME}[name]
   if Path(getattr(module,'__file__','')).resolve()!=expected.resolve(): raise RuntimeError('foreign package cache '+name)
 try:
  if pipeline_literal not in sys.path: sys.path.insert(0,pipeline_literal); inserted=True
  package=importlib.import_module('wam_pipeline')
  v169=importlib.import_module('wam_pipeline.v169_arm_routed_runtime')
  v482=importlib.import_module('wam_pipeline.v482_temporal8_residual_runtime')
  spec=importlib.util.spec_from_file_location('pipeline.scripts.train_v482_temporal8_residual_5fold',OLD_TRAINER)
  old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
  if (Path(package.__file__).resolve()!=WAM_INIT.resolve() or Path(v169.__file__).resolve()!=V169_RUNTIME.resolve()
      or Path(v482.__file__).resolve()!=V482_RUNTIME.resolve() or old.__package__!='pipeline.scripts'):
   raise RuntimeError('production package identity')
  return old
 finally:
  if inserted:
   if not sys.path or sys.path[0]!=pipeline_literal: raise RuntimeError('sys.path ownership drift')
   sys.path.pop(0)
  if sys.path!=before: raise RuntimeError('sys.path restore')
def schedule_rows(seed,steps):
 values=np.arange(steps*2,dtype=np.int64); np.random.default_rng(seed).shuffle(values); return values.reshape(steps,2)
def schedule_sha(rows): return hashlib.sha256(np.ascontiguousarray(rows,dtype='<i8').view(np.uint8)).hexdigest()

def validate_budget(budget,steps):
 expected={'sequence_instances':steps*2,'batch_sequences':2,'prefix_frames_per_sequence':8,'prefix_frame_examples_per_step':16,'optimizer_steps':steps,'forward_calls':steps,'loss_calls':steps,'backward_calls':steps,'optimizer_zero_grad_calls':steps,'gradient_clip_calls':steps,'scheduler_instances':0,'scheduler_steps':0,'checkpoint_writes':1,'publication_calls':1,'early_stop_calls':0,'fold_training_calls':0,'intermediate_checkpoint_writes':0,'resume_calls':0,'retry_calls':0,'search_calls':0,'total_prefix_frame_examples':steps*16}
 if set(budget)!={'all200_schedule_seed','all200_schedule_sha256','all200_schedule_shape','backward_calls','batch_sequences','branch_labels','branches_per_context','checkpoint_writes','contexts','early_stop_calls','fold_training_calls','forward_calls','gradient_clip_calls','intermediate_checkpoint_writes','loss_calls','optimizer_steps','optimizer_zero_grad_calls','prefix_frame_examples_per_step','prefix_frames_per_sequence','publication_calls','resume_calls','retry_calls','schedule_flatten_exact_permutation_start','schedule_flatten_exact_permutation_stop_exclusive','schedule_no_drop_no_pad_no_repeat','scheduler_instances','scheduler_steps','search_calls','sequence_instances','total_prefix_frame_examples'}: raise RuntimeError('budget keyset')
 for key,value in expected.items():
  if type(budget.get(key)) is not int or budget[key]!=value: raise RuntimeError('budget '+key)
 if (type(budget.get('contexts')) is not int or type(budget.get('branches_per_context')) is not int
     or budget['contexts']*budget['branches_per_context']!=steps*2 or type(budget.get('branch_labels')) is not list
     or len(budget['branch_labels'])!=budget['branches_per_context'] or budget.get('schedule_no_drop_no_pad_no_repeat') is not True
     or type(budget.get('all200_schedule_seed')) is not int or budget['all200_schedule_seed']!=2624
     or type(budget.get('schedule_flatten_exact_permutation_start')) is not int or budget['schedule_flatten_exact_permutation_start']!=0
     or type(budget.get('schedule_flatten_exact_permutation_stop_exclusive')) is not int or budget['schedule_flatten_exact_permutation_stop_exclusive']!=steps*2): raise RuntimeError('budget arithmetic')
 if steps==500 and (budget['contexts']!=200 or budget['branches_per_context']!=5 or budget['branch_labels']!=['factual','no_transport','scale_0p4','scale_1p25','reverse_direction_0p4']): raise RuntimeError('production budget literals')
 rows=schedule_rows(int(budget['all200_schedule_seed']),steps)
 if rows.shape!=(steps,2) or sorted(rows.reshape(-1).tolist())!=list(range(steps*2)): raise RuntimeError('schedule permutation')
 if steps==500 and (budget.get('all200_schedule_shape')!=[500,2] or schedule_sha(rows)!=budget.get('all200_schedule_sha256')): raise RuntimeError('schedule digest')
 return rows

def validate_documents(prereg,manifest,contract,authority,args,*,profile_steps=500,enforce_production_paths=True):
 if (set(prereg)!=PREREG_KEYS or set(manifest)!=MANIFEST_KEYS or set(contract)!=CONTRACT_KEYS or set(authority)!=AUTHORITY_KEYS
     or prereg.get('format')!='strict-track2-v542-v541-fixed-budget-all200-training-preregistration-v1'
     or manifest.get('format')!='strict-track2-v542-v541-fixed-budget-all200-training-manifest-v1'
     or contract.get('format')!='strict-track2-v542-v541-fixed-budget-all200-training-execution-authority-design-contract-v1'
     or prereg.get('status')!='preregistered_design_only_execution_not_performed'
     or manifest.get('status')!='frozen_design_only_execution_not_performed'
     or contract.get('status')!='design_only_frozen_fixed_budget_training_sources_pending_independent_authority_materialization'):
  raise RuntimeError('document exact top schemas')
 if type(prereg.get('seed')) is not int or prereg['seed']!=1673 or any(type(doc.get('seed')) is not int or doc['seed']!=1673 for doc in (manifest,contract,authority)): raise RuntimeError('seed strict int')
 if any(not json_exact(doc.get('active_source_role_order'),ACTIVE_ROLE_ORDER) or not json_exact(doc.get('active_source_paths'),ACTIVE_PATHS) for doc in (prereg,manifest,contract,authority)): raise RuntimeError('active role/path literal')
 for key in ('budget_contract','authorization','base_training_contract','output_contract','v540_gate_transition'):
  if ((key in manifest and not json_exact(prereg.get(key),manifest.get(key))) or not json_exact(prereg[key],contract.get(key)) or not json_exact(prereg[key],authority.get(key))): raise RuntimeError('embedded exact '+key)
 if not json_exact(prereg.get('base_training_contract'),EXPECTED_BASE_TRAINING) or not json_exact(prereg.get('output_contract'),EXPECTED_OUTPUT_CONTRACT): raise RuntimeError('base/output literal')
 if not json_exact(contract.get('lineage'),EXPECTED_LINEAGE) or not json_exact(authority.get('lineage'),EXPECTED_LINEAGE): raise RuntimeError('lineage literal')
 if not json_exact(contract.get('runtime_observation'),EXPECTED_RUNTIME_OBSERVATION) or not json_exact(authority.get('runtime_observation'),EXPECTED_RUNTIME_OBSERVATION): raise RuntimeError('runtime observation literal')
 if not json_exact(contract.get('execution_boundary'),EXPECTED_EXECUTION_BOUNDARY) or not json_exact(authority.get('execution_boundary'),EXPECTED_EXECUTION_BOUNDARY): raise RuntimeError('execution boundary literal')
 if (not json_exact(contract.get('historical_absences'),HISTORICAL_ABSENCES) or not json_exact(contract.get('current_absences_after_authority'),CURRENT_ABSENCES)
     or not json_exact(authority.get('historical_absences'),HISTORICAL_ABSENCES) or not json_exact(authority.get('required_absences'),CURRENT_ABSENCES)): raise RuntimeError('absence maps literal')
 records6,records7=contract.get('active_source_records'),authority.get('active_source_records')
 if not isinstance(records6,dict) or set(records6)!=set(ACTIVE_ROLE_ORDER[1:]) or not isinstance(records7,dict) or set(records7)!=set(ACTIVE_ROLE_ORDER): raise RuntimeError('active records6/7')
 for role in ACTIVE_ROLE_ORDER:
  row=records7[role]
  if set(row)!={'path','sha256','logical_bytes'} or row['path']!=ACTIVE_PATHS[role] or type(row['logical_bytes']) is not int: raise RuntimeError('active record '+role)
  if role!='authority_design_contract' and not json_exact(records6[role],row): raise RuntimeError('active record shared '+role)
  if enforce_production_paths: exact_record(row)
 design={'path':str(args.contract),'sha256':args.contract_sha,'logical_bytes':args.contract.stat().st_size}
 if not json_exact(records7['authority_design_contract'],design): raise RuntimeError('design record')
 closure6,closure7=contract.get('source_closure'),authority.get('source_closure')
 if (not json_exact(contract.get('source_role_order'),SOURCE_ROLE_ORDER) or not json_exact(authority.get('source_role_order'),['authority_design_contract',*SOURCE_ROLE_ORDER])
     or not json_exact(contract.get('source_aliases'),SOURCE_ALIASES) or not json_exact(authority.get('source_aliases'),AUTHORITY_SOURCE_ALIASES)
     or not isinstance(closure6,dict) or set(closure6)!=set(SOURCE_ROLE_ORDER) or not isinstance(closure7,dict) or set(closure7)!={'authority_design_contract',*SOURCE_ROLE_ORDER}
     or canonical_sha(closure6)!=contract.get('source_closure_sha256') or canonical_sha(closure7)!=authority.get('source_closure_sha256')): raise RuntimeError('source registry')
 if not json_exact(closure7['authority_design_contract'],design) or not json_exact(authority.get('authority_design_contract'),design): raise RuntimeError('design closure/alias')
 for role in SOURCE_ROLE_ORDER:
  row=closure6[role]
  if row.get('path')!=SOURCE_PATHS[role] or not json_exact(closure7.get(role),row) or not json_exact(contract.get(SOURCE_ALIASES[role]),row) or not json_exact(authority.get(AUTHORITY_SOURCE_ALIASES[role]),row): raise RuntimeError('source closure '+role)
  if enforce_production_paths: exact_record(row)
 for role,row in EXPECTED_FAILURE_ANCESTRY.items():
  if not json_exact(closure6.get(role),row): raise RuntimeError('v541 failure ancestry '+role)
 if enforce_production_paths:
  failed_authority=load_json(EXPECTED_FAILURE_ANCESTRY['v541_unconsumed_authority_receipt']['path'])
  if type(failed_authority.get('authorization',{}).get('fixed_budget_training_boundary_invocations_consumed')) is not int or failed_authority['authorization']['fixed_budget_training_boundary_invocations_consumed']!=0: raise RuntimeError('v541 authority consumed0')
 receipt=contract.get('authority_receipt_contract')
 if (not isinstance(receipt,dict) or set(receipt)!=RECEIPT_CONTRACT_KEYS or not json_exact(receipt.get('active_source_role_order_exact'),ACTIVE_ROLE_ORDER)
     or not json_exact(receipt.get('active_source_paths_exact'),ACTIVE_PATHS) or type(receipt.get('contract_active_source_records_count')) is not int or receipt['contract_active_source_records_count']!=6
     or receipt.get('contract_design_record_excluded_to_avoid_self_hash_cycle') is not True or type(receipt.get('authority_active_source_records_count')) is not int or receipt['authority_active_source_records_count']!=7
     or receipt.get('authority_design_record_included') is not True or type(receipt.get('source_closure_contract_count')) is not int or receipt['source_closure_contract_count']!=32
     or not json_exact(receipt.get('contract_active_source_records_exact'),records6) or not json_exact(receipt.get('authorization_exact'),authority.get('authorization'))
     or not json_exact(receipt.get('budget_contract_exact'),contract.get('budget_contract')) or not json_exact(receipt.get('base_training_contract_exact'),contract.get('base_training_contract'))
     or not json_exact(receipt.get('output_contract_exact'),contract.get('output_contract')) or not json_exact(receipt.get('v540_gate_transition_exact'),contract.get('v540_gate_transition'))
     or not json_exact(receipt.get('runtime_observation_exact'),EXPECTED_RUNTIME_OBSERVATION) or not json_exact(receipt.get('execution_boundary_exact'),EXPECTED_EXECUTION_BOUNDARY)): raise RuntimeError('receipt contract exact21')
 checks=authority.get('checks')
 if (authority.get('passed') is not True or not isinstance(checks,dict) or set(checks)!=set(CHECK_KEYS) or any(value is not True for value in checks.values())
     or receipt.get('top_keys')!=sorted(AUTHORITY_KEYS) or receipt.get('check_keys')!=CHECK_KEYS or authority.get('check_keys')!=CHECK_KEYS
     or canonical_sha(CHECK_KEYS)!=authority.get('check_key_set_sha256') or authority.get('check_key_set_sha256')!=receipt.get('check_key_set_sha256')
     or canonical_sha(checks)!=authority.get('checks_sha256') or authority.get('checks_sha256')!=receipt.get('checks_sha256')
     or authority.get('format')!=receipt.get('format') or authority.get('status')!=receipt.get('status')): raise RuntimeError('authority terminal/check schema')
 pre,post=authority.get('input_pre_snapshot'),authority.get('input_post_snapshot')
 for snap in (pre,post):
  if (not isinstance(snap,dict) or set(snap)!=SNAPSHOT_KEYS or not json_exact(snap.get('absences'),HISTORICAL_ABSENCES)
      or not json_exact(snap.get('files'),closure7) or snap.get('gpu_compute_pids')!=[] or snap.get('relevant_execution_pids')!=[]
      or not json_exact(snap.get('services'),EXPECTED_SERVICES) or not json_exact(snap.get('trees'),EXPECTED_TREES)
      or canonical_sha({key:value for key,value in snap.items() if key!='canonical_sha256'})!=snap.get('canonical_sha256')): raise RuntimeError('input snapshot literal')
 if authority.get('input_snapshots_exactly_equal') is not True or not json_exact(pre,post): raise RuntimeError('input prepost')
 budget=copy.deepcopy(prereg['budget_contract'])
 rows=validate_budget(budget,profile_steps)
 auth=authority['authorization']
 expected_authorization=copy.deepcopy(EXPECTED_AUTHORIZATION)
 for key in ('backward_invocations_authorized','forward_invocations_authorized','gradient_clip_invocations_authorized','loss_invocations_authorized','optimizer_step_invocations_authorized','optimizer_zero_grad_invocations_authorized'): expected_authorization[key]=profile_steps
 if not json_exact(auth,expected_authorization): raise RuntimeError('authorization exact keyset')
 expected_counts={'fixed_budget_training_boundary_invocations_authorized':1,'fixed_budget_training_boundary_invocations_consumed':0,'training_launcher_invocations_authorized':1,'training_trainer_invocations_authorized':1,'auditor_import_invocations_authorized':1,'optimizer_step_invocations_authorized':profile_steps,'optimizer_zero_grad_invocations_authorized':profile_steps,'forward_invocations_authorized':profile_steps,'loss_invocations_authorized':profile_steps,'backward_invocations_authorized':profile_steps,'gradient_clip_invocations_authorized':profile_steps,'final_checkpoint_writes_authorized':1,'output_publication_invocations_authorized':1,'scheduler_instances_authorized':0,'scheduler_step_invocations_authorized':0,'config_selection_invocations_authorized':0,'early_stop_invocations_authorized':0,'intermediate_checkpoint_writes_authorized':0,'resume_invocations_authorized':0,'search_invocations_authorized':0}
 for key,value in expected_counts.items():
  if type(auth.get(key)) is not int or auth[key]!=value: raise RuntimeError('authorization '+key)
 if auth.get('training_authorized') is not True or auth.get('retry_authorized') is not False or auth.get('hidden_private_or_final_input_authorized') is not False: raise RuntimeError('authorization bool')
 transition=authority['v540_gate_transition']
 if transition!={'config_or_budget_selected_from_v540_metrics':False,'exact_count':8,'metrics_semantic_parse_authorized':False,'passed_ancestry_only':True,'retry_authorized':False,'root':'/root/v540_v539_public_s1_zero_update_gate_seed1671_20260828','tree_canonical_sha256':'71117c8570f8fd62865eb2573b2b2a402eeec78ca8a83881c9bdd442c434459f','tree_lines_sha256':'24dee8d2fee5ee7e8a2e6aa3ca130ee9ad0935c8def7fed5baa1ceb35bd544fc','tree_total_logical_bytes':6999111}: raise RuntimeError('v540 transition')
 if enforce_production_paths:
  for row in manifest['v540_gate_records'].values(): exact_record(row)
  for row in manifest['base_input_records'].values(): exact_record(row)
  v540_authority=load_json(SOURCE_PATHS['v540_authority_receipt'])
  if type(v540_authority['authorization']['public_s1_zero_update_boundary_invocations_consumed']) is not int or v540_authority['authorization']['public_s1_zero_update_boundary_invocations_consumed']!=0: raise RuntimeError('v540 authority immutable consumed0')
  for path in (AUTH_PREP,ATTEMPT_ROOT,ATTEMPT_PREP,OUTPUT,OUTPUT_PREP):
   if os.path.lexists(path): raise RuntimeError('fresh path '+str(path))
 return budget,rows

def rng_snapshot(torch):
 return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state().clone(),'torch_cuda':[x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}
def rng_restore(torch,snap):
 random.setstate(snap['python']); np.random.set_state(snap['numpy']); torch.set_rng_state(snap['torch_cpu']);
 if torch.cuda.is_available(): torch.cuda.set_rng_state_all(snap['torch_cuda'])
def state_sha(torch,state):
 h=hashlib.sha256()
 for name,value in sorted(state.items()): h.update(name.encode()+b'\0'); h.update(np.ascontiguousarray(value.detach().cpu().numpy()).view(np.uint8))
 return h.hexdigest()
def named_tensor_sha(torch,rows): return state_sha(torch,{name:value for name,value in rows})
def object_sha(torch,value):
 stream=io.BytesIO(); torch.save(value,stream); return sha_bytes(stream.getvalue())
def dataset_snapshot(oldpre):
 root=Path(oldpre['dataset']['root']).resolve(); declared={row['relative']:row['sha256'] for row in oldpre['dataset']['files']}; actual={}
 for path in sorted(root.rglob('*')):
  if path.is_symlink(): raise RuntimeError('dataset symlink')
  if path.is_file(): actual[path.relative_to(root).as_posix()]=sha_bytes(path.read_bytes())
 if actual!=declared: raise RuntimeError('dataset tree current')
 return sha_bytes(json.dumps(actual,sort_keys=True,separators=(',',':')).encode())

def training_loop(*,torch,model,optimizer,rows,batch_builder,loss_builder,expected_steps,event_hook=None):
 events=[]; counts={key:0 for key in ('forward_calls','loss_calls','backward_calls','optimizer_zero_grad_calls','gradient_clip_calls','optimizer_steps')}
 for ordinal,batch in enumerate(rows):
  if ordinal>=expected_steps: raise RuntimeError('over budget')
  payload=batch_builder(batch)
  optimizer.zero_grad(set_to_none=True); counts['optimizer_zero_grad_calls']+=1
  prediction=model(*payload[:-1]); counts['forward_calls']+=1
  loss=loss_builder(prediction,payload[-1]); counts['loss_calls']+=1
  loss.backward(); counts['backward_calls']+=1
  gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); counts['gradient_clip_calls']+=1
  if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient)): raise RuntimeError('nonfinite loss or gradient')
  optimizer.step(); counts['optimizer_steps']+=1
  if not all(bool(torch.isfinite(p).all()) for p in model.parameters()): raise RuntimeError('nonfinite parameter')
  row={'ordinal':ordinal,'batch_sequence_ids':[int(x) for x in batch],'loss_finite':True,'gradient_norm_finite':True,'optimizer_step_committed':True}
  events.append(row)
  if event_hook is not None: event_hook(ordinal,row)
 if len(events)!=expected_steps or any(value!=expected_steps for value in counts.values()): raise RuntimeError('under budget')
 return events,counts

class Held:
 def __init__(self,path):
  self.path=Path(path); self.fd=os.open(self.path,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600); st=os.fstat(self.fd); self.dev,self.ino=st.st_dev,st.st_ino; self.record={'path':str(self.path),'sha256':sha_bytes(b''),'logical_bytes':0}
 def write(self,raw):
  os.ftruncate(self.fd,0); view=memoryview(raw)
  while view:
   n=os.write(self.fd,view)
   if n<=0: raise RuntimeError('short write')
   view=view[n:]
  os.fsync(self.fd); data=os.pread(self.fd,len(raw)+1,0)
  if data!=raw: raise RuntimeError('held payload')
  self.record={'path':str(self.path),'sha256':sha_bytes(raw),'logical_bytes':len(raw)}; return self.record
 def current_at(self,path):
  path=Path(path); st=os.fstat(self.fd); ls=os.lstat(path)
  if (st.st_dev,st.st_ino,ls.st_dev,ls.st_ino)!=(self.dev,self.ino,self.dev,self.ino): raise RuntimeError('held identity')
  raw=os.pread(self.fd,self.record['logical_bytes']+1,0)
  if len(raw)!=self.record['logical_bytes'] or sha_bytes(raw)!=self.record['sha256']: raise RuntimeError('held current')
  return {'path':str(path),'sha256':self.record['sha256'],'logical_bytes':self.record['logical_bytes']}
 def current(self): return self.current_at(self.path)
 def owned_current(self):
  try: self.current(); return True
  except BaseException: return False
 def close(self): os.close(self.fd)

def exact_output_tree(root,held):
 root=Path(root); st=os.lstat(root)
 if not stat.S_ISDIR(st.st_mode) or stat.S_ISLNK(st.st_mode): raise RuntimeError('output directory')
 names=sorted(entry.name for entry in os.scandir(root))
 if names!=EXACT6: raise RuntimeError('output exact6')
 records={name:held[name].current_at(root/name) for name in EXACT6}
 return {'names':names,'records':records,'canonical_sha256':sha_bytes(cbytes(records))}

def rename_noreplace(source,target):
 libc=ctypes.CDLL(None,use_errno=True); fn=libc.renameat2; fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]; fn.restype=ctypes.c_int
 if fn(AT_FDCWD,os.fsencode(source),AT_FDCWD,os.fsencode(target),RENAME_NOREPLACE)!=0: raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))

def production_inputs(torch,args,old=None):
 oldpre=load_json(OLD_PREREG); old=old or load_production_training_components()
 oldargs=types.SimpleNamespace(contract=OLD_CONTRACT,v169_release=Path(oldpre['v169']['release_path']),v169_library=Path(oldpre['v169']['library_path']))
 contexts,samples=old.load_data(oldargs,oldpre)
 manifest=load_json(CACHE_MANIFEST); cache=Path(manifest['cache_npz_path']); exact_file(cache,manifest['cache_npz_sha256'],manifest['cache_npz_logical_bytes'])
 with np.load(cache,allow_pickle=False) as data:
  if list(data.files)!=manifest['npz_key_order']: raise RuntimeError('cache keys')
  baselines=np.asarray(data['baseline'])
 if baselines.shape!=(200,5,8,256,256,3) or baselines.dtype!=np.uint8: raise RuntimeError('baseline schema')
 baselines=np.ascontiguousarray(baselines.reshape(1000,8,256,256,3))
 lower=np.asarray(oldpre['action_bounds']['lower'],np.float32); upper=np.asarray(oldpre['action_bounds']['upper'],np.float32)
 torch.manual_seed(2624); model=old.TemporalResidualUNet128FiLM(16).to(torch.device(args.device)); initial=copy.deepcopy(model.state_dict())
 if old.state_sha(initial)!=oldpre['all200_initial_state_sha256']: raise RuntimeError('initial state')
 optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(0.9,0.95),eps=1e-8,weight_decay=1e-4)
 def batch_builder(batch):
  _,base,context,features,target,base_target=old.tensors(contexts,samples,baselines,batch,batch,lower,upper,'action',torch.device(args.device))
  return (base,context,features,(base_target,target))
 def loss_builder(prediction,target_pair): return old.loss_value(prediction,target_pair[0],target_pair[1])
 return old,oldpre,model,optimizer,batch_builder,loss_builder,lower,upper

def fixture_inputs(torch,steps):
 torch.manual_seed(1624); model=torch.nn.Linear(2,1,bias=True); optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(0.9,0.95),eps=1e-8,weight_decay=1e-4)
 def batch_builder(batch):
  x=torch.tensor([[float(batch[0]),float(batch[1])]],dtype=torch.float32); y=torch.tensor([[float(sum(batch)%3)]],dtype=torch.float32); return (x,y)
 def loss_builder(prediction,target): return torch.nn.functional.mse_loss(prediction,target)
 return None,None,model,optimizer,batch_builder,loss_builder,np.zeros(14,np.float32),np.ones(14,np.float32)

def execute(args,*,validation_profile=None,enforce_production_paths=True,event_hook=None):
 import torch
 prereg,manifest,contract,authority=map(load_json,(args.preregistration,args.manifest,args.contract,args.authority_receipt))
 steps=500 if validation_profile is None else int(validation_profile['optimizer_steps'])
 budget,rows=validate_documents(prereg,manifest,contract,authority,args,profile_steps=steps,enforce_production_paths=enforce_production_paths)
 if validation_profile is not None:
  mode=validation_profile.get('schedule_mode','exact')
  if mode=='under': rows=rows[:-1]
  elif mode=='over': rows=np.concatenate((rows,rows[:1]),axis=0)
  elif mode!='exact': raise RuntimeError('validation profile schedule mode')
 use_production_inputs=validation_profile is None or validation_profile.get('production_inputs') is True
 production_components=load_production_training_components() if use_production_inputs else None
 root=OUTPUT if validation_profile is None else Path(validation_profile['output_root']); prep=OUTPUT_PREP if validation_profile is None else Path(validation_profile['output_prep'])
 if os.path.lexists(root) or os.path.lexists(prep): raise RuntimeError('output prestate')
 snap=rng_snapshot(torch); restored=False; held={}; promoted=False; prep_identity=None; publication_old_mask=None
 try:
  os.mkdir(prep,0o700); pst=os.lstat(prep); prep_identity=(pst.st_dev,pst.st_ino)
  for name in EXACT6: held[name]=Held(prep/name)
  input_builder=(lambda t,a:production_inputs(t,a,production_components)) if use_production_inputs else (lambda t,a:fixture_inputs(t,steps))
  components,oldpre,model,optimizer,batch_builder,loss_builder,lower,upper=input_builder(torch,args)
  input_before=dataset_snapshot(oldpre) if use_production_inputs else None
  initial_sha=state_sha(torch,model.state_dict()); initial_parameters_sha=named_tensor_sha(torch,model.named_parameters()); initial_buffers_sha=named_tensor_sha(torch,model.named_buffers()); initial_optimizer_sha=object_sha(torch,optimizer.state_dict()); initial_mode=model.training
  events,loop_counts=training_loop(torch=torch,model=model,optimizer=optimizer,rows=rows,batch_builder=batch_builder,loss_builder=loss_builder,expected_steps=steps,event_hook=event_hook)
  model.eval(); final_sha=state_sha(torch,model.state_dict()); final_parameters_sha=named_tensor_sha(torch,model.named_parameters()); final_buffers_sha=named_tensor_sha(torch,model.named_buffers()); final_optimizer_sha=object_sha(torch,optimizer.state_dict())
  input_after=dataset_snapshot(oldpre) if use_production_inputs else None
  if input_before!=input_after: raise RuntimeError('dataset prepost drift')
  if validation_profile is None:
   for row in manifest['base_input_records'].values(): exact_record(row)
   for row in manifest['v540_gate_records'].values(): exact_record(row)
  rng_restore(torch,snap); restored=True
  checkpoint={'format':CHECKPOINT_FORMAT,'training_scope':'all200-action-fixed-budget','channels':16,'precision':'bf16','feature_schema':getattr(components,'FEATURE_SCHEMA','fixture-v1'),'closure_digest':sha_bytes(cbytes(manifest.get('base_input_records',{}))),'schedule_sha256':budget['all200_schedule_sha256'],'optimizer_steps':steps,'action_lower':lower,'action_upper':upper,'model':model.state_dict()}
  buffer=io.BytesIO(); torch.save(checkpoint,buffer); checkpoint_raw=buffer.getvalue(); events_raw=b''.join(cbytes(row) for row in events)
  counts={**loop_counts,'checkpoint_writes':1,'publication_calls':1,'early_stop_calls':0,'fold_training_calls':0,'intermediate_checkpoint_writes':0,'resume_calls':0,'retry_calls':0,'search_calls':0,'scheduler_instances':0,'scheduler_steps':0,'total_prefix_frame_examples':steps*16}
  intent={'format':'strict-track2-v542-v541-fixed-budget-all200-attempt-intent-v1','authority_receipt':exact_file(args.authority_receipt),'optimizer_steps':steps,'schedule_sha256':budget['all200_schedule_sha256'],'retry_authorized':False}
  receipt={'format':'strict-track2-v542-v541-fixed-budget-all200-training-receipt-v1','status':'passed_fixed_budget_training','passed':True,'counts':counts,'initial_model_state_sha256':initial_sha,'final_model_state_sha256':final_sha,'initial_parameter_sha256':initial_parameters_sha,'final_parameter_sha256':final_parameters_sha,'initial_buffer_sha256':initial_buffers_sha,'final_buffer_sha256':final_buffers_sha,'initial_optimizer_state_sha256':initial_optimizer_sha,'final_optimizer_state_sha256':final_optimizer_sha,'initial_model_training_mode':initial_mode,'final_model_training_mode':model.training,'dataset_tree_pre_sha256':input_before,'dataset_tree_post_sha256':input_after,'external_rng_restored_before_publication':True,'base_and_inputs_readonly':True,'retry_authorized':False}
  source_manifest={'format':'strict-track2-v542-v541-fixed-budget-all200-source-manifest-v1','authority_receipt':exact_file(args.authority_receipt),'budget_sha256':sha_bytes(cbytes(budget)),'checkpoint':{'sha256':sha_bytes(checkpoint_raw),'logical_bytes':len(checkpoint_raw)},'events':{'sha256':sha_bytes(events_raw),'logical_bytes':len(events_raw)},'input_records':manifest.get('base_input_records',{}),'trainer':exact_file(Path(__file__))}
  auditor=import_source('v541_fixed_budget_independent_auditor',args.auditor_source); audit=auditor.audit(budget={**budget,'optimizer_steps':steps,'total_prefix_frame_examples':steps*16},events_raw=events_raw,checkpoint_raw=checkpoint_raw,receipt=receipt,source_manifest=source_manifest,expected_checkpoint_format=CHECKPOINT_FORMAT)
  payloads={'attempt_intent.json':cbytes(intent),'training_events.ndjson':events_raw,'checkpoint.pt':checkpoint_raw,'training_receipt.json':cbytes(receipt),'independent_audit.json':cbytes(audit),'source_manifest.json':cbytes(source_manifest)}
  for name in ['attempt_intent.json','training_events.ndjson','checkpoint.pt','training_receipt.json','independent_audit.json','source_manifest.json']: held[name].write(payloads[name])
  publication_signals={signal.SIGINT,signal.SIGTERM}; publication_old_mask=signal.pthread_sigmask(signal.SIG_BLOCK,publication_signals)
  for item in held.values(): item.current()
  now=os.lstat(prep)
  if (now.st_dev,now.st_ino)!=prep_identity or sorted(p.name for p in prep.iterdir())!=EXACT6: raise RuntimeError('prep identity/tree')
  prep_fd=os.open(prep,os.O_RDONLY|os.O_DIRECTORY)
  try: os.fsync(prep_fd)
  finally: os.close(prep_fd)
  expected_tree={'names':EXACT6,'records':{name:{'path':str(root/name),'sha256':held[name].record['sha256'],'logical_bytes':held[name].record['logical_bytes']} for name in EXACT6}}
  expected_tree['canonical_sha256']=sha_bytes(cbytes(expected_tree['records']))
  rename_noreplace(prep,root); promoted=True; postcommit_diagnostic=None; postcommit_current_exact6_observed=False
  try:
   parent_fd=os.open(root.parent,os.O_RDONLY|os.O_DIRECTORY)
   try: os.fsync(parent_fd)
   finally: os.close(parent_fd)
   observed_tree=exact_output_tree(root,held)
   if observed_tree!=expected_tree: raise RuntimeError('postcommit tree')
   postcommit_current_exact6_observed=True
  except BaseException as error: postcommit_diagnostic={'type':type(error).__name__,'message':str(error)}
  publication_signals_drained=[]
  for pending in sorted(signal.sigpending()&publication_signals,key=int): signal.sigwait({pending}); publication_signals_drained.append(int(pending))
  signal.pthread_sigmask(signal.SIG_SETMASK,publication_old_mask); publication_old_mask=None
  return {'passed':True,'status':'passed_fixed_budget_training','public_steps':steps,'output_root':str(root),'counts':counts,'checkpoint_sha256':sha_bytes(checkpoint_raw),'events_sha256':sha_bytes(events_raw),'publication_visibility_committed':True,'postcommit_current_exact6_observed':postcommit_current_exact6_observed,'postcommit_diagnostic':postcommit_diagnostic,'publication_signals_drained':publication_signals_drained}
 finally:
  if publication_old_mask is not None: signal.pthread_sigmask(signal.SIG_SETMASK,publication_old_mask)
  if not restored: rng_restore(torch,snap)
  if not promoted:
   for item in held.values():
    try:
      if os.path.lexists(item.path) and item.owned_current(): os.unlink(item.path)
    except BaseException: pass
   try:
    if os.path.lexists(prep):
     st=os.lstat(prep)
     if prep_identity and (st.st_dev,st.st_ino)==prep_identity and not any(prep.iterdir()): os.rmdir(prep)
   except BaseException: pass
  for item in held.values():
   try: item.close()
   except BaseException: pass

def parser():
 p=argparse.ArgumentParser()
 for name in ('preregistration','manifest','contract','authority-receipt','auditor-source','launcher-source','output-root'): p.add_argument('--'+name,type=Path,required=True)
 for name in ('preregistration-sha','manifest-sha','contract-sha','authority-receipt-sha','trainer-sha','auditor-sha','launcher-sha'): p.add_argument('--'+name,required=True)
 p.add_argument('--seed',type=int,required=True); p.add_argument('--device',choices=['cuda'],required=True); return p
def main(argv=None):
 args=parser().parse_args(argv)
 for path,digest in ((Path(__file__).resolve(),args.trainer_sha),(args.preregistration,args.preregistration_sha),(args.manifest,args.manifest_sha),(args.contract,args.contract_sha),(args.authority_receipt,args.authority_receipt_sha),(args.auditor_source,args.auditor_sha),(args.launcher_source,args.launcher_sha)): exact_file(path,digest)
 if args.seed!=1673 or args.output_root!=OUTPUT: raise RuntimeError('literal args')
 result=execute(args); print(json.dumps({'passed':result['passed'],'status':result['status'],'public_steps':result['public_steps']},sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
