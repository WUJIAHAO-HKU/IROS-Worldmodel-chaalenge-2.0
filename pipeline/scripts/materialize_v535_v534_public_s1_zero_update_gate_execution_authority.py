#!/usr/bin/env python3
"""Materialize one fresh v535 public-S1 zero-update execution authority."""
from __future__ import annotations
import argparse,ctypes,errno,hashlib,importlib.util,json,os,signal,stat,subprocess,urllib.request
from pathlib import Path
ROOT=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge'); S=ROOT/'pipeline/scripts'; J=ROOT/'artifacts/strict_track2_joint_augmentation_20260810'
SELF=S/'materialize_v535_v534_public_s1_zero_update_gate_execution_authority.py'
CONTRACT=S/'v535_v534_public_s1_zero_update_gate_execution_authority_contract.json'
PREREG=S/'v535_v534_public_s1_zero_update_gate_preregistration.json'; MANIFEST=S/'v535_v534_public_s1_zero_update_gate_manifest.json'
EVALUATOR=S/'evaluate_v535_v534_public_s1_zero_update_gate.py'; AUDITOR=S/'audit_v535_v534_public_s1_zero_update_gate.py'; LAUNCHER=S/'launch_v535_v534_public_s1_zero_update_gate.py'
PREREG_EXPECTED=('020b1e67b5d95286b6689cb00e564e66b5a0962908d8c3e1bad12d66abbc7106',75267)
MANIFEST_EXPECTED=('a9876660419e58c883810ce5e8806d1fd5d008d56e7834521f17b0767f778ca4',77608)
EVALUATOR_EXPECTED=('8f95e503464725ab95c4caab4fef4552f1ed0aaedc6b2414dc578120fcaabd3c',48432)
AUDITOR_EXPECTED=('1a908135dd47e0a963b4356ec243f185a903940b8bfc992169abe71404e6f372',5186)
LAUNCHER_EXPECTED=('45297eadb144fc670c774d2319c74e246987ae8d5343f08bb149ada2bdfce017',4270)
AUTH_ROOT=J/'v535_v534_public_s1_zero_update_gate_execution_authority_seed1668_20260827'; AUTH_PREP=AUTH_ROOT.with_name(AUTH_ROOT.name+'.authority-prep')
ATTEMPT_ROOT=J/'v535_v534_public_s1_zero_update_gate_attempt_seed1668_20260827'; ATTEMPT_PREP=ATTEMPT_ROOT.with_name(ATTEMPT_ROOT.name+'.attempt-prep')
OUTPUT_ROOT=Path('/root/v535_v534_public_s1_zero_update_gate_seed1668_20260827'); OUTPUT_PREP=Path(str(OUTPUT_ROOT)+'.output-prep')
V534_ROOT=Path('/root/v534_v533_actual_oof_seed1667_20260827'); QUAL_ROOT=Path('/root/v524_v523_phase_a_cache_qualification_seed1660_20260826')
CONTRACT_FORMAT='strict-track2-v535-v534-public-s1-zero-update-gate-execution-authority-design-contract-v1'
CONTRACT_STATUS='design_only_frozen_public_s1_zero_update_sources_pending_independent_authority_materialization'
OUTPUT_FORMAT='strict-track2-v535-v534-public-s1-zero-update-gate-execution-authority-v1'
OUTPUT_STATUS='authorized_exact_one_public_s1_zero_update_boundary_pending_external_execution'
AUTHORIZATION={
'actual_oof_execution_invocations_authorized':0,'backward_authorized':False,'backward_invocations_authorized':0,'buffer_update_invocations_authorized':0,'cache_write_invocations_authorized':0,'dev_hidden_final_outcome_read_invocations_authorized':0,'direct_auditor_invocations_authorized':0,'direct_evaluator_invocations_authorized':0,'hidden_private_input_read_invocations_authorized':0,'hidden_private_or_final_input_authorized':False,'model_or_cache_write_authorized':False,'model_write_invocations_authorized':0,'optimizer_authorized':False,'optimizer_step_invocations_authorized':0,'optimizer_zero_grad_invocations_authorized':0,'parameter_or_buffer_mutation_authorized':False,'parameter_update_invocations_authorized':0,'phase_a_replay_invocations_authorized':0,'public_s1_evaluator_invocations_authorized':1,'public_s1_zero_update_boundary_invocations_authorized':1,'public_s1_zero_update_boundary_invocations_consumed':0,'public_s1_zero_update_gate_authorized':True,'readonly_validator_invocations_authorized':0,'retry_authorized':False,'reward_read_authorized':False,'reward_read_invocations_authorized':0,'scheduler_authorized':False,'scheduler_step_invocations_authorized':0,'submission_authorized':False,'submission_invocations_authorized':0,'training_authorized':False,'training_invocations_authorized':0,'zero_update_gate_invocations_authorized':1}
RUNTIME={'authority_materializer_invocations':1,'public_s1_zero_update_boundary_invocations':0,'public_s1_evaluator_invocations':0,'zero_update_gate_invocations':0,'launcher_invocations':0,'independent_auditor_invocations':0,'actual_oof_execution_invocations':0,'backward_invocations':0,'optimizer_step_invocations':0,'optimizer_zero_grad_invocations':0,'scheduler_step_invocations':0,'parameter_update_invocations':0,'buffer_update_invocations':0,'model_write_invocations':0,'cache_write_invocations':0,'reward_read_invocations':0,'hidden_private_input_read_invocations':0,'training_invocations':0,'submission_invocations':0}
EXECUTION_BOUNDARY={'authority_materialization_only':True,'public_s1_zero_update_boundary_invocations':0,'public_s1_evaluator_invocations':0,'zero_update_gate_invocations':0,'direct_execution_authorized':False,'public_only':True,'hidden_private_final_reward_inputs_authorized':False,'training_or_update_authorized':False,'authority_publication_noreplace':True,'authority_publication_commit_semantics':'current_exact1_visibility_after_noreplace_directory_rename','crash_durability_claimed':False,'postcommit_diagnostics_best_effort_success_priority':True,'rollback_relink_retry_or_second_publish_after_commit':False}
ACTIVE_SOURCE_ROLE_ORDER=['authority_design_contract','authority_materializer','public_s1_zero_update_preregistration','public_s1_zero_update_manifest','public_s1_evaluator','public_s1_independent_auditor','public_s1_launcher']
ACTIVE_SOURCE_PATHS={'authority_design_contract':str(CONTRACT),'authority_materializer':str(SELF),'public_s1_zero_update_preregistration':str(PREREG),'public_s1_zero_update_manifest':str(MANIFEST),'public_s1_evaluator':str(EVALUATOR),'public_s1_independent_auditor':str(AUDITOR),'public_s1_launcher':str(LAUNCHER)}
SOURCE_ROLES=['authority_materializer','public_s1_zero_update_preregistration','public_s1_zero_update_manifest','public_s1_evaluator','public_s1_independent_auditor','public_s1_launcher','v534_authority_receipt','v534_actual_oof_attempt_intent','v534_actual_oof_execution_receipt','v534_actual_oof_independent_audit','v534_actual_oof_events','v534_actual_oof_metrics','v534_actual_oof_fold0','v534_actual_oof_fold1','v534_actual_oof_fold2','v534_actual_oof_fold3','v534_actual_oof_fold4','v534_actual_oof_source_manifest','v524_qualification_terminal','v524_qualification_report','v524_qualification_audit','v524_cache_manifest','v474_public_s1_selection','v474_public_s1_selection_receipt','v169_closure','v169_runtime_source','v169_release_manifest','v169_library_manifest','v482_preregistration','v482_model_design_contract']
AUTHORITY_SOURCE_ROLES=['authority_design_contract',*SOURCE_ROLES]
SOURCE_ALIASES={'authority_design_contract':'authority_design_contract',**{r:r+'_source' for r in SOURCE_ROLES}}; SOURCE_ALIASES['v169_runtime_source']='v169_runtime_source_source'
LINEAGE={'canonical_v535_execution_not_performed':True,'fresh_public_s1_zero_update_authority_only':True,'v534_actual_oof_passed_current':True,'v534_authority_immutable_consumed_zero':True,'v534_output_exact11_current':True,'public_episode_disjoint_action_only_selection_current':True,'historical_reward_authority_not_inherited':True,'v524_qualification_exact20_readonly':True,'zero_update_proof_mandatory':True,'training_or_update_authorized':False,'hidden_private_final_inputs_denied':True,'two_stage_authority_then_gate_only':True}
EMBEDDED_FIELDS=['v534_actual_oof_durable_transition','public_s1_input_contract','public_s1_output_contract','zero_update_contract','hidden_input_denial_contract']
CHECK_KEYS=sorted({'contract_schema','preregistration_schema','manifest_schema','authorization_exact_strict','runtime_boundary_exact','execution_boundary_exact','source_role_order_exact','source_aliases_exact','source_closure30_current','active_sources6_current','contract_active_records6','authority_active_records7','v534_durable_exact11_current','v534_authority_immutable_consumed0','v534_durable_boundary1','qualification_exact20_readonly','public_selection_current','public_windows_current','public_hidden_denial_exact','zero_update_contract_exact','fresh_roots6_absent','input_prepost_equal','services_current','gpu_empty','execution_pids_empty','historical_reward_authority_not_inherited','all_update_training_hidden_unsafe_disabled','publish_noreplace_exact1','lineage_exact','public_output_exact8_contract','two_stage_boundary_only'})
CONTRACT_TOP_KEYS={'format','status','seed','lineage','active_source_role_order','active_source_paths','active_source_records','source_closure','source_role_order','source_aliases','source_closure_sha256',*{SOURCE_ALIASES[r] for r in SOURCE_ROLES},*EMBEDDED_FIELDS,'historical_absences','current_absences_after_authority','authority_receipt_contract','authorization','runtime_observation','execution_boundary'}
AUTH_TOP_KEYS={'format','status','passed','seed','lineage','active_source_role_order','active_source_paths','active_source_records','source_closure','source_role_order','source_aliases','source_closure_sha256',*SOURCE_ALIASES.values(),*EMBEDDED_FIELDS,'historical_absences','required_absences','fresh_attempt_root','fresh_output_root','checks','check_keys','check_key_set_sha256','checks_sha256','input_pre_snapshot','input_post_snapshot','input_snapshots_exactly_equal','authorization','runtime_observation','execution_boundary'}
def cbytes(v): return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def csha(v): return hashlib.sha256(cbytes(v)).hexdigest()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def regular(path,expected=None):
 path=Path(path); st=os.lstat(path)
 if path.is_symlink() or not stat.S_ISREG(st.st_mode): raise RuntimeError('nonregular:'+str(path))
 row={'path':str(path),'sha256':sha(path),'logical_bytes':st.st_size}
 if expected is not None and (row['sha256'],row['logical_bytes'])!=expected: raise RuntimeError('record:'+str(path))
 return row
def exact_keys(v,keys,label):
 if type(v) is not dict or set(v)!=set(keys): raise RuntimeError(label+':keys')
def load_evaluator():
 regular(EVALUATOR,EVALUATOR_EXPECTED); spec=importlib.util.spec_from_file_location('v535_authority_evaluator_schema',EVALUATOR); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def source_specs():
 return {
'public_s1_zero_update_preregistration':(PREREG,PREREG_EXPECTED),'public_s1_zero_update_manifest':(MANIFEST,MANIFEST_EXPECTED),'public_s1_evaluator':(EVALUATOR,EVALUATOR_EXPECTED),'public_s1_independent_auditor':(AUDITOR,AUDITOR_EXPECTED),'public_s1_launcher':(LAUNCHER,LAUNCHER_EXPECTED),
'v534_authority_receipt':(J/'v534_v533_actual_oof_environment_repair_execution_authority_seed1667_20260827/authority_receipt.json',('54c7d75b911f4a70a28d9eb69b77cccd321d18a636f1d4ce82e086d68dda599e',324362)),
'v534_actual_oof_attempt_intent':(V534_ROOT/'attempt_intent.json',('5daa1953c2498ea8427c71d4884c1aac7b49697397a0196afe626527a92db836',6538)),'v534_actual_oof_execution_receipt':(V534_ROOT/'execution_receipt.json',('1a094542b3edd53cf47ef6d6379a2fa2de7cdf31483a65df0fc2d8cba195fb7b',10373)),'v534_actual_oof_independent_audit':(V534_ROOT/'independent_audit.json',('223f718b5bb2aa396452cb85ff522cfdb9465ea4cea14ce46cf74dabd8eab8ff',1705)),'v534_actual_oof_events':(V534_ROOT/'oof_call_events.ndjson',('16e10610f2b954226d381c5e0da4d9130c9f9dbdd7ad321f2acf39c1763ef856',990262)),'v534_actual_oof_metrics':(V534_ROOT/'metrics.npz',('815d71d8433ca32da7059deb67d9788bbc723d3fe25ae8601abd7efb192492ef',570338)),
'v534_actual_oof_fold0':(V534_ROOT/'fold_0_receipt.json',('a1cabcf85ec9a72b70f2b60c30df5f68b21a1825958afbaf73569821ec312c67',318)),'v534_actual_oof_fold1':(V534_ROOT/'fold_1_receipt.json',('fd527d9d0c8c18c920adc77e4e6500c19e99e50a1ff3a3bbc3621496294b1771',318)),'v534_actual_oof_fold2':(V534_ROOT/'fold_2_receipt.json',('1062d5b3c76f8abdd7df602c4392c9d5ba9cec53b591e146de05740055c276ea',318)),'v534_actual_oof_fold3':(V534_ROOT/'fold_3_receipt.json',('3b89ba184ffaec8e1037c502a43786837534ee0ece604f4e56520b5934309a31',318)),'v534_actual_oof_fold4':(V534_ROOT/'fold_4_receipt.json',('cee9e7c62a679668ec56bf1b51824103f4c0bef38d245d0e1c9e6012ff3dd512',318)),'v534_actual_oof_source_manifest':(V534_ROOT/'source_manifest.json',('5be99639ca21497d2ca6f76fe2f8704fdb2e4838d06934c113e2773e67ea62f6',710)),
'v524_qualification_terminal':(QUAL_ROOT/'terminal_receipt.json',('93d38e010b37275a740bf3ccae2b935b08e40b6953657c12838780e5e2b542a1',83762)),'v524_qualification_report':(QUAL_ROOT/'qualification_report.json',('6514b497f994bc502fc46dc6c5a90865dbb126402935cfa359a1af2fe0fbc173',87681)),'v524_qualification_audit':(QUAL_ROOT/'independent_final_audit.json',('3d702b36a3589d737c9f21e0a1182c59f636753faa1143a2adc85d90d3a674bc',9935)),'v524_cache_manifest':(QUAL_ROOT/'process_a/cache/manifest.json',('716d62575ea149242f0e0918ed508e22ae803ff12d3b66fb26bb4b8f5a3e6f94',18453)),
'v474_public_s1_selection':(J/'v474_v473_parent_s1_seed1617_r6_20260824/action_only_selection.json',('b8e8002897b9ec0e1b386eda6d72a6bc2a09c8e322764a1c6c27b96cf902bd19',297226)),'v474_public_s1_selection_receipt':(J/'v474_v473_parent_s1_seed1617_r6_20260824/action_only_selection_receipt.json',('31a54148eace7adc31fbb5ea21efb2629600d02d1c4bf4c78f69ada8ed69fcad',211877)),
'v169_closure':(J/'v474_v473_median4_parent_release_seed1618_20260824/v169_closure.json',('f8b4891fc12fdbc87a7abd93b8a7709cc77614b294e6a7dfa6cd109dac43fb29',3407000)),'v169_runtime_source':(ROOT/'pipeline/wam_pipeline/v482_temporal8_residual_runtime.py',('6939e3f6f52c1eb83bfd4324372d7e75ed4ee6d493777a39f38c26e5f4db7471',21745)),'v169_release_manifest':(J/'v169_instruction_arm_routed_release/v169_arm_routed_manifest.json',('0a91316e14a220c61acb6fceed3177ad72b2d8b70f12c212883ddb80a6817f51',1141)),'v169_library_manifest':(ROOT/'artifacts/splits/adjust_bottle_50episodes_full.json',('792673011ccf82aa52becdffef7f1854c297de33200df0b550b1178f41692a31',720)),'v482_preregistration':(J/'v482_temporal8_residual_s0_r3_seed1624_20260824/preregistration.json',('426d4f7a774520458f6af41d20660cf31202791f99603dacec114f771da6a881',294684)),'v482_model_design_contract':(S/'v482_temporal_film_residual_model_design_contract.json',('19eea774f4b8871528054a352bf6c7ed6cefd39afcfc1956c9cf7a969335fddb',37030))}
def source_records(materializer_record):
 out={'authority_materializer':materializer_record}
 for role,(p,e) in source_specs().items(): out[role]=regular(p,e)
 if list(out)!=SOURCE_ROLES: raise RuntimeError('source order')
 return out
def absences(): return {'authority_root':AUTH_ROOT,'authority_prep':AUTH_PREP,'attempt_root':ATTEMPT_ROOT,'attempt_prep':ATTEMPT_PREP,'output_root':OUTPUT_ROOT,'output_prep':OUTPUT_PREP}
def gpu_processes():
 p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
 if p.returncode: raise RuntimeError('nvidia-smi')
 return sorted({int(x) for x in p.stdout.splitlines() if x.strip()})
def services_snapshot():
 out={}
 for key,url in [('8005','http://127.0.0.1:8005/v1/health'),('18084','http://127.0.0.1:18084/health')]:
  with urllib.request.urlopen(url,timeout=10) as r: b=r.read(); out[key]={'http_code':r.status,'body_sha256':hashlib.sha256(b).hexdigest(),'body_bytes':len(b),'json_model':json.loads(b)}
 return out
def relevant_pids():
 names={EVALUATOR.name,AUDITOR.name,LAUNCHER.name,'train_v482_temporal8_residual_5fold.py'}; excluded=set(); pid=os.getpid()
 while pid>1 and pid not in excluded:
  excluded.add(pid)
  try: pid=int((Path('/proc')/str(pid)/'stat').read_text().split()[3])
  except: break
 rows=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name) in excluded: continue
  try: argv=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
  except: continue
  match=sorted(set(argv)&{str(S/n) for n in names})
  if match: rows.append({'pid':int(p.name),'argv':argv,'matches':match})
 return rows
def snapshot(module,records,absence_map):
 v={'files':{r:regular(Path(x['path'])) for r,x in records.items()},'trees':{'v534_actual_oof_exact11':module.tree(V534_ROOT),'qualification_exact20':module.tree(QUAL_ROOT)},'absences':{k:{'path':str(p),'absent':not os.path.lexists(p)} for k,p in sorted(absence_map.items())},'services':services_snapshot(),'gpu_compute_pids':gpu_processes(),'relevant_execution_pids':relevant_pids()}; v['canonical_sha256']=csha(v); return v
def rename_noreplace(source, target):
    function = getattr(ctypes.CDLL(None, use_errno=True), 'renameat2', None)
    if function is None:
        raise RuntimeError('renameat2 unavailable')
    if function(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(target)
        raise OSError(error, os.strerror(error), str(target))


def fsync_dir(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def held_read(fd, size):
    data = os.pread(fd, size + 1, 0)
    if len(data) != size or os.pread(fd, 1, size) != b'':
        raise RuntimeError('held EOF')
    return data


def gate_owned_staged(prep, prep_identity, member, member_identity, descriptor, payload):
    prep_metadata = os.lstat(prep)
    if prep.is_symlink() or not stat.S_ISDIR(prep_metadata.st_mode) or (prep_metadata.st_dev, prep_metadata.st_ino) != prep_identity:
        raise RuntimeError('prep identity')
    if sorted(entry.name for entry in os.scandir(prep)) != ['authority_receipt.json']:
        raise RuntimeError('prep exact1')
    member_metadata = os.lstat(member)
    fd_metadata = os.fstat(descriptor)
    if member.is_symlink() or not stat.S_ISREG(member_metadata.st_mode) or (member_metadata.st_dev, member_metadata.st_ino) != member_identity or (fd_metadata.st_dev, fd_metadata.st_ino) != member_identity:
        raise RuntimeError('held identity')
    if fd_metadata.st_size != len(payload) or held_read(descriptor, len(payload)) != payload:
        raise RuntimeError('held payload')


def publish_exact1(receipt, stable_snapshot, module, records):
    if os.path.lexists(AUTH_ROOT) or os.path.lexists(AUTH_PREP):
        raise RuntimeError('authority prestate')
    oldmask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    prep_identity = None
    member_identity = None
    descriptor = None
    committed = False
    payload = (json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False) + '\n').encode()
    digest = hashlib.sha256(payload).hexdigest()
    try:
        os.mkdir(AUTH_PREP, 0o700)
        metadata = os.lstat(AUTH_PREP)
        prep_identity = (metadata.st_dev, metadata.st_ino)
        member = AUTH_PREP / 'authority_receipt.json'
        descriptor = os.open(member, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        metadata = os.fstat(descriptor)
        member_identity = (metadata.st_dev, metadata.st_ino)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if type(written) is not int or written <= 0 or written > len(payload) - offset:
                raise RuntimeError('held write')
            offset += written
        os.fsync(descriptor)
        gate_owned_staged(AUTH_PREP, prep_identity, member, member_identity, descriptor, payload)
        fsync_dir(AUTH_PREP)
        stable_absences = {k: v for k, v in absences().items() if k not in {'authority_root', 'authority_prep'}}
        third = snapshot(module, records, stable_absences)
        if third != stable_snapshot:
            raise RuntimeError('third snapshot drift')
        gate_owned_staged(AUTH_PREP, prep_identity, member, member_identity, descriptor, payload)
        rename_noreplace(AUTH_PREP, AUTH_ROOT)
        committed = True
        diagnostics = []
        tree = None
        for name, operation in [
            ('parent_fsync', lambda: fsync_dir(AUTH_ROOT.parent)),
            ('held_readback', lambda: held_read(descriptor, len(payload))),
            ('current_tree', lambda: module.tree(AUTH_ROOT)),
        ]:
            try:
                result = operation()
                if name == 'current_tree':
                    tree = result
            except BaseException as error:
                diagnostics.append({'operation': name, 'error_type': type(error).__name__, 'error': str(error)})
        exact = tree is not None and tree.get('inventory') == [['authority_receipt.json', digest, len(payload)]]
        return {'visibility_committed': True, 'current_exact1_consumable': exact, 'postcommit_diagnostics_passed': not diagnostics and exact, 'postcommit_diagnostics': diagnostics, 'retry_authorized': False}
    finally:
        if not committed and prep_identity is not None and AUTH_PREP.is_dir() and not AUTH_PREP.is_symlink():
            try:
                metadata = os.lstat(AUTH_PREP)
                member = AUTH_PREP / 'authority_receipt.json'
                if (metadata.st_dev, metadata.st_ino) == prep_identity and descriptor is not None and os.path.lexists(member):
                    current = os.lstat(member)
                    fd_metadata = os.fstat(descriptor)
                    if not member.is_symlink() and (current.st_dev, current.st_ino) == member_identity and (fd_metadata.st_dev, fd_metadata.st_ino) == member_identity and fd_metadata.st_size == len(payload) and held_read(descriptor, len(payload)) == payload:
                        member.unlink()
                if not any(AUTH_PREP.iterdir()):
                    AUTH_PREP.rmdir(); fsync_dir(AUTH_PREP.parent)
            except BaseException:
                pass
        if descriptor is not None:
            try: os.close(descriptor)
            except OSError: pass
        signal.pthread_sigmask(signal.SIG_SETMASK, oldmask)


def expected_sources(materializer_record): return source_records(materializer_record)
def validate_context(args):
 materializer_record=regular(args.materializer_source,(args.materializer_sha,args.materializer_bytes)); design=regular(args.contract,(args.contract_sha,args.contract_bytes)); module=load_evaluator(); prereg=json.loads(PREREG.read_text()); manifest=json.loads(MANIFEST.read_text()); contract=json.loads(args.contract.read_text())
 sources=expected_sources(materializer_record); active6={r:sources[r] for r in ACTIVE_SOURCE_ROLE_ORDER[1:]}
 exact_keys(contract,CONTRACT_TOP_KEYS,'contract')
 if contract['format']!=CONTRACT_FORMAT or contract['status']!=CONTRACT_STATUS or contract['seed']!=1668 or type(contract['seed']) is not int or contract['lineage']!=LINEAGE or contract['active_source_role_order']!=ACTIVE_SOURCE_ROLE_ORDER or contract['active_source_paths']!=ACTIVE_SOURCE_PATHS or contract['active_source_records']!=active6: raise RuntimeError('contract identity/schema')
 if contract['source_role_order']!=SOURCE_ROLES or contract['source_closure']!=sources or contract['source_closure_sha256']!=csha(sources) or contract['source_aliases']!=SOURCE_ALIASES: raise RuntimeError('contract closure')
 for role,alias in SOURCE_ALIASES.items():
  if role!='authority_design_contract' and contract.get(alias)!=sources[role]: raise RuntimeError('contract alias:'+role)
 if prereg['authorization']!=AUTHORIZATION or manifest['authorization']!=AUTHORIZATION or contract['authorization']!=AUTHORIZATION: raise RuntimeError('authorization exact')
 for k,v in AUTHORIZATION.items():
  if type(contract['authorization'][k]) is not type(v): raise RuntimeError('authorization type')
 for f in EMBEDDED_FIELDS:
  if prereg[f]!=manifest[f] or contract[f]!=prereg[f]: raise RuntimeError('embedded:'+f)
 if prereg['active_source_role_order']!=ACTIVE_SOURCE_ROLE_ORDER or prereg['active_source_paths']!=ACTIVE_SOURCE_PATHS or manifest['active_source_role_order']!=ACTIVE_SOURCE_ROLE_ORDER or manifest['active_source_paths']!=ACTIVE_SOURCE_PATHS: raise RuntimeError('registry')
 return module,prereg,manifest,contract,design,sources
def parse():
 p=argparse.ArgumentParser(); p.add_argument('--contract',type=Path); p.add_argument('--contract-sha'); p.add_argument('--contract-bytes',type=int); p.add_argument('--materializer-source',type=Path); p.add_argument('--materializer-sha'); p.add_argument('--materializer-bytes',type=int); p.add_argument('--authority-root',type=Path); p.add_argument('--read-only-preflight',action='store_true'); p.add_argument('--synthetic-self-test',action='store_true'); return p.parse_args()
def main():
 args=parse()
 if args.synthetic_self_test:
  checks={'top':len(CONTRACT_TOP_KEYS)>50 and len(AUTH_TOP_KEYS)>50,'sources':len(SOURCE_ROLES)==30 and len(AUTHORITY_SOURCE_ROLES)==31,'auth':len(AUTHORIZATION)==33 and all(type(v) is int for k,v in AUTHORIZATION.items() if k.endswith('_invocations_authorized') or k.endswith('_invocations_consumed')),'ones':AUTHORIZATION['public_s1_zero_update_boundary_invocations_authorized']==AUTHORIZATION['public_s1_evaluator_invocations_authorized']==AUTHORIZATION['zero_update_gate_invocations_authorized']==1 and AUTHORIZATION['public_s1_zero_update_boundary_invocations_consumed']==0,'unsafe':all(AUTHORIZATION[k] is False for k in ['retry_authorized','training_authorized','backward_authorized','optimizer_authorized','scheduler_authorized','reward_read_authorized','submission_authorized','hidden_private_or_final_input_authorized','model_or_cache_write_authorized','parameter_or_buffer_mutation_authorized'])}; print(json.dumps({'passed':all(checks.values()),'checks':checks,'checks_sha256':csha(checks)},sort_keys=True)); return 0 if all(checks.values()) else 1
 if any(value is None for value in (args.contract,args.contract_sha,args.contract_bytes,args.materializer_source,args.materializer_sha,args.materializer_bytes,args.authority_root)): raise RuntimeError('required production arguments')
 if args.authority_root!=AUTH_ROOT: raise RuntimeError('authority root')
 module,prereg,manifest,contract,design,sources=validate_context(args); absence_map=absences()
 if any(os.path.lexists(p) for p in absence_map.values()): raise FileExistsError('fresh roots')
 pre=snapshot(module,sources,absence_map)
 if pre['gpu_compute_pids'] or pre['relevant_execution_pids'] or not all(x['absent'] for x in pre['absences'].values()): raise RuntimeError('prestate')
 if args.read_only_preflight:
  print(json.dumps({'passed':True,'status':'passed_read_only_preflight_no_materialization','contract_top_count':len(CONTRACT_TOP_KEYS),'authority_top_count':len(AUTH_TOP_KEYS),'check_count':len(CHECK_KEYS),'source_count':len(SOURCE_ROLES),'authority_source_count':len(AUTHORITY_SOURCE_ROLES),'input_snapshot_sha256':pre['canonical_sha256']},sort_keys=True)); return 0
 post=snapshot(module,sources,absence_map)
 if pre!=post: raise RuntimeError('snapshot drift')
 authority_sources={'authority_design_contract':design,**sources}; checks={k:True for k in CHECK_KEYS}; receipt={'format':OUTPUT_FORMAT,'status':OUTPUT_STATUS,'passed':True,'seed':1668,'lineage':LINEAGE,'active_source_role_order':ACTIVE_SOURCE_ROLE_ORDER,'active_source_paths':ACTIVE_SOURCE_PATHS,'active_source_records':{'authority_design_contract':design,**contract['active_source_records']},'source_closure':authority_sources,'source_role_order':AUTHORITY_SOURCE_ROLES,'source_aliases':SOURCE_ALIASES,'source_closure_sha256':csha(authority_sources),**{SOURCE_ALIASES[r]:x for r,x in authority_sources.items()},**{f:prereg[f] for f in EMBEDDED_FIELDS},'historical_absences':contract['historical_absences'],'required_absences':contract['current_absences_after_authority'],'fresh_attempt_root':str(ATTEMPT_ROOT),'fresh_output_root':str(OUTPUT_ROOT),'checks':checks,'check_keys':CHECK_KEYS,'check_key_set_sha256':csha(CHECK_KEYS),'checks_sha256':csha(checks),'input_pre_snapshot':pre,'input_post_snapshot':post,'input_snapshots_exactly_equal':True,'authorization':AUTHORIZATION,'runtime_observation':RUNTIME,'execution_boundary':EXECUTION_BOUNDARY}
 exact_keys(receipt,AUTH_TOP_KEYS,'authority receipt')
 module.validate_documents(prereg,manifest,contract,receipt,argparse.Namespace(contract=args.contract,contract_sha=args.contract_sha),require_fresh=True)
 stable_abs={k:v for k,v in absence_map.items() if k not in {'authority_root','authority_prep'}}; stable=snapshot(module,sources,stable_abs)
 result=publish_exact1(receipt,stable,module,sources)
 print(json.dumps({'passed':True,'authority_root':str(AUTH_ROOT),'publication':result,'receipt_sha256':sha(AUTH_ROOT/'authority_receipt.json'),'receipt_bytes':(AUTH_ROOT/'authority_receipt.json').stat().st_size},sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
