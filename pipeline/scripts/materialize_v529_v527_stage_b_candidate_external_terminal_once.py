#!/usr/bin/env python3
"""Validator-zero external terminal for the already-current v527 candidate."""
from __future__ import annotations
import argparse,ctypes,fcntl,hashlib,json,os,signal,stat,subprocess,sys,urllib.request
from pathlib import Path

ROOT=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge');J=ROOT/'artifacts/strict_track2_joint_augmentation_20260810';S=ROOT/'pipeline/scripts'
SELF=S/'materialize_v529_v527_stage_b_candidate_external_terminal_once.py'
SCRIPT=Path('/root/v529_v527_stage_b_candidate_external_terminal_once.sh')
DEPLOY=S/'v529_v527_stage_b_candidate_external_terminal_transport_deployment_record.json'
DESIGN=S/'v529_v527_stage_b_candidate_external_terminal_design_contract.json'
FORENSIC=S/'v529_v527_stage_b_mapped_fixture_default_argument_canonical_publication_forensic.json'
DESIGN_EXPECTED=('6bdf6ddc527a1eac7eccff9cf5763e615e557668cf624c1a595b7d927334fcfc',7604)
FORENSIC_EXPECTED=('4ab9cd21892d4efddd7c6b52ecd9f42b21f324f9b9013854f6e1f1ae8c379576',7575)
CANDIDATE=J/'v527_v526_v525_oof_readonly_validation_attempt_seed1662_20260827';CANDIDATE_PREP=CANDIDATE.with_name(CANDIDATE.name+'.attempt-prep');CANDIDATE_RECEIPT=CANDIDATE/'oof_readonly_gate_validation_receipt.json'
AUTH=J/'v527_v526_v525_oof_readonly_validation_execution_authority_seed1662_20260827';AUTH_RECEIPT=AUTH/'authority_receipt.json'
STAGE_A=J/'v527_v526_v525_oof_readonly_validation_execution_authority_materialization_evidence_seed1662_20260827'
QUAL=Path('/root/v524_v523_phase_a_cache_qualification_seed1660_20260826')
OUT=J/'v529_v527_stage_b_candidate_external_terminal_evidence_seed1663_20260827';PREP=OUT.with_name(OUT.name+'.evidence-prep')
EXACT5=['argv.json','helper_stderr.log','helper_stdout.log','intent.json','transport_helper.py'];EXACT6=sorted(EXACT5+['process_receipt.json'])
AT_FDCWD=-100;RENAME_NOREPLACE=1

def cbytes(v): return (json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode()
def csha(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def sha(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): d.update(b)
 return d.hexdigest()
def regular(p,expected=None):
 if p!=p.resolve() or p.is_symlink() or not p.is_file() or not stat.S_ISREG(os.lstat(p).st_mode): raise RuntimeError(f'regular:{p}')
 r={'path':str(p),'sha256':sha(p),'logical_bytes':p.stat().st_size}
 if expected is not None and (r['sha256'],r['logical_bytes'])!=expected: raise RuntimeError(f'record:{p}')
 return r
def tree(root):
 if root!=root.resolve() or root.is_symlink() or not root.is_dir(): raise RuntimeError(f'tree:{root}')
 inv=[]
 for p in sorted(root.rglob('*'),key=lambda x:x.relative_to(root).as_posix()):
  if p.is_symlink(): raise RuntimeError('symlink')
  if p.is_file(): inv.append([p.relative_to(root).as_posix(),sha(p),p.stat().st_size])
  elif not p.is_dir(): raise RuntimeError('special')
 lines=''.join(f'{d}  {n}\n' for n,d,_ in inv).encode()
 return {'root':str(root),'inventory':inv,'file_count':len(inv),'logical_file_bytes':sum(x[2] for x in inv),'sha256sum_lines_digest_sha256':hashlib.sha256(lines).hexdigest(),'canonical_json_triples_digest_sha256':csha(inv)}
def fsync_dir(p):
 fd=os.open(p,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
 try: os.fsync(fd)
 finally: os.close(fd)
def write_all(fd,data,writer=os.write):
 off=0
 while off<len(data):
  n=writer(fd,data[off:])
  if type(n) is not int or n<=0 or n>len(data)-off: raise RuntimeError('short write')
  off+=n
def write_exclusive(p,data):
 fd=os.open(p,os.O_RDWR|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
 try:
  write_all(fd,data);os.fsync(fd)
  if os.pread(fd,len(data)+1,0)!=data or os.pread(fd,1,len(data))!=b'': raise RuntimeError('write EOF')
 finally: os.close(fd)
def rename_noreplace(a,b):
 libc=ctypes.CDLL(None,use_errno=True);fn=getattr(libc,'renameat2',None)
 if fn is None: raise RuntimeError('renameat2')
 if fn(AT_FDCWD,os.fsencode(a),AT_FDCWD,os.fsencode(b),RENAME_NOREPLACE)!=0:
  e=ctypes.get_errno();raise OSError(e,os.strerror(e),str(b))
def expected_deploy(h,s):
 return {'format':'strict-track2-v529-stage-b-transport-pair-deployment-record-v1','status':'preregistered_atomic_noreplace_transport_pair_pending_external_deployment','classification':'design_only_non_authority_non_execution_transport_pair_manifest','deployment_executed':False,'pair_exact_count':2,'targets':[{'role':'external_terminal_helper',**h},{'role':'transport_script',**s}],'overwrite_authorized':False,'canonical_readback_required':True}
def deployment(h,s):
 r=regular(DEPLOY);raw=DEPLOY.read_bytes();v=json.loads(raw)
 if raw!=cbytes(v) or cbytes(v)!=cbytes(expected_deploy(h,s)): raise RuntimeError('deployment record')
 return r
def health():
 out={}
 for name,url in [('8005','http://127.0.0.1:8005/v1/health'),('18084','http://127.0.0.1:18084/health')]:
  with urllib.request.urlopen(url,timeout=10) as r: body=r.read();code=r.status
  try:model=json.loads(body)
  except Exception:model=None
  out[name]={'http_code':code,'body_sha256':hashlib.sha256(body).hexdigest(),'body_bytes':len(body),'json_model':model}
 if not all(x['http_code']==200 and x['body_bytes']>0 for x in out.values()): raise RuntimeError('health')
 return out
def gpu():
 p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=20)
 if p.returncode: raise RuntimeError('nvidia-smi')
 return sorted({int(x) for x in p.stdout.splitlines() if x.strip()})
def relevant_pids():
 names={'audit_v527_v526_v525_oof_readonly_validation.py','launch_v524_v523_phase_a_cache_qualification.py','generate_v524_v523_phase_a_cache_qualification.py','generate_v524_v523_phase_a_cache_qualification_worker.py'}
 excluded=set();pid=os.getpid()
 while pid>1 and pid not in excluded:
  excluded.add(pid)
  try:pid=int((Path('/proc')/str(pid)/'stat').read_text().split()[3])
  except Exception:break
 rows=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name) in excluded:continue
  try:argv=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
  except Exception:continue
  hit=sorted({Path(x).name for x in argv if Path(x).name in names})
  if hit:rows.append({'pid':int(p.name),'argv':argv,'matched_entrypoints':hit})
 return sorted(rows,key=lambda x:x['pid'])
def assert_tree(t,spec):
 if t['file_count']!=spec['file_count'] or t['sha256sum_lines_digest_sha256']!=spec['tree_lines'] or t['canonical_json_triples_digest_sha256']!=spec['tree_canon']: raise RuntimeError('tree anchor')
 if 'logical_file_bytes' in spec and t['logical_file_bytes']!=spec['logical_file_bytes']: raise RuntimeError('tree bytes')
def load_design():
 r=regular(DESIGN,DESIGN_EXPECTED);v=json.loads(DESIGN.read_bytes())
 if v.get('format')!='strict-track2-v529-v527-stage-b-candidate-external-terminal-design-contract-v1' or v.get('seed')!=1663:return (_ for _ in ()).throw(RuntimeError('design schema'))
 return v,r
def snapshot(design,allow_owned_prep=False,allow_output=False):
 forensic=regular(FORENSIC,FORENSIC_EXPECTED);fv=json.loads(FORENSIC.read_text())
 if fv.get('status')!='canonical_nonterminal_candidate_published_once_by_failed_mapped_fixture_no_retry' or fv.get('retry_authorized') is not False: raise RuntimeError('forensic')
 cr=regular(CANDIDATE_RECEIPT,(design['candidate']['receipt']['sha256'],design['candidate']['receipt']['logical_bytes']));ct=tree(CANDIDATE);assert_tree(ct,{'file_count':design['candidate']['tree_file_count'],'tree_lines':design['candidate']['tree_sha256sum_lines_digest_sha256'],'tree_canon':design['candidate']['tree_canonical_json_triples_digest_sha256']})
 if os.path.lexists(CANDIDATE_PREP):raise RuntimeError('candidate prep')
 raw=CANDIDATE_RECEIPT.read_bytes();candidate=json.loads(raw)
 if len(candidate)!=29:raise RuntimeError('candidate keys')
 required={'format':'strict-track2-v527-v526-v525-oof-readonly-gate-validation-candidate-v1','status':'readonly_oof_gate_inputs_verified_candidate_pending_external_terminal','passed':None,'candidate_verified':True,'input_validation_passed':True,'oof_gate_passed':None,'standalone_consumable':False,'external_terminal_required':True,'publication_success_claimed':False,'standalone_oof_execution_authority':False,'qualification_mutated':False,'retry_authorized':False}
 if any(type(candidate.get(k)) is not type(v) or candidate.get(k)!=v for k,v in required.items()):raise RuntimeError('candidate schema')
 if candidate['input_pre_snapshot']!=candidate['input_post_snapshot'] or candidate.get('input_snapshots_exactly_equal') is not True:raise RuntimeError('candidate snapshots')
 view=candidate['normalized_qualification_view'];ns=design['normalized_view']
 if (csha(view['original_worker_receipts'])!=ns['original_csha'] or csha(view['normalized_worker_receipts'])!=ns['normalized_csha'] or csha(view['recursive_diff'])!=ns['diff_csha'] or view['recursive_diff_count']!=2 or view['recursive_diff_full_paths']!=ns['diff_paths']):raise RuntimeError('normalized view')
 inp=candidate['input_pre_snapshot'];qt=tree(QUAL);assert_tree(qt,design['qualification'])
 if inp['qualification_tree']!=qt:raise RuntimeError('qualification current')
 v525=tree(Path(design['v525_external_current']['root']));assert_tree(v525,design['v525_external_current'])
 if inp['stage_b_external_evidence_tree']!=v525 or inp['stage_b_external_process_receipt']!=regular(Path(design['v525_external_current']['root'])/'process_receipt.json',(design['v525_external_current']['process_sha256'],design['v525_external_current']['process_bytes'])):raise RuntimeError('v525 external current')
 v528=tree(Path(design['v528_deployment_evidence']['root']));assert_tree(v528,design['v528_deployment_evidence'])
 v528_receipt=regular(Path(design['v528_deployment_evidence']['root'])/'deployment_receipt.json',(design['v528_deployment_evidence']['receipt_sha256'],design['v528_deployment_evidence']['receipt_bytes']))
 final7=[regular(Path(row['path']),(row['sha256'],row['logical_bytes'])) for row in design['v528_final7_current']]
 at=tree(AUTH);assert_tree(at,{'file_count':1,'tree_lines':design['stage_a_authority']['tree_lines'],'tree_canon':design['stage_a_authority']['tree_canon']});ar=regular(AUTH_RECEIPT,(design['stage_a_authority']['receipt_sha256'],design['stage_a_authority']['receipt_bytes']))
 st=tree(STAGE_A);assert_tree(st,design['stage_a_evidence']);sp=regular(STAGE_A/'process_receipt.json',(design['stage_a_evidence']['process_sha256'],design['stage_a_evidence']['process_bytes']))
 services=health();pids=relevant_pids();gp=gpu()
 if pids or gp or inp['services']!=services or inp['gpu_compute_pids']!=[] or inp['relevant_execution_pids']!=[]:raise RuntimeError('live/current')
 if (not allow_output and os.path.lexists(OUT)) or (not allow_owned_prep and os.path.lexists(PREP)):raise RuntimeError('output prestate')
 return {'design':regular(DESIGN,DESIGN_EXPECTED),'forensic':forensic,'candidate_tree':ct,'candidate_receipt':cr,'candidate_receipt_canonical_sha256':csha(candidate),'authority_tree':at,'authority_receipt':ar,'stage_a_evidence_tree':st,'stage_a_process_receipt':sp,'qualification_tree':qt,'v525_external_tree':v525,'v528_deployment_tree':v528,'v528_deployment_receipt':v528_receipt,'v528_final7_current':final7,'normalized_view_sha256':csha(view),'services':services,'gpu_compute_pids':gp,'relevant_pids':pids}
def atomic_terminal(path,value,state,hook=lambda _p:None,writer=os.write,linker=os.link,unlinker=lambda p:p.unlink(),parent_fsync=fsync_dir,temp_name=None):
 temp=path.with_name(temp_name or ('.'+path.name+'.noreplace-tmp'));payload=cbytes(value)
 if os.path.lexists(path) or os.path.lexists(temp):raise FileExistsError(path)
 fd=os.open(temp,os.O_RDWR|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600);linked=False
 try:
  m=os.fstat(fd);identity=(m.st_dev,m.st_ino);write_all(fd,payload,writer);os.fsync(fd)
  if os.pread(fd,len(payload)+1,0)!=payload or os.pread(fd,1,len(payload))!=b'':raise RuntimeError('terminal EOF')
  hook(temp);lm=os.lstat(temp)
  if (lm.st_dev,lm.st_ino)!=identity or regular(temp)!={'path':str(temp),'sha256':hashlib.sha256(payload).hexdigest(),'logical_bytes':len(payload)}:raise RuntimeError('terminal ownership')
  linker(temp,path,follow_symlinks=False);linked=True;state['visible']=True
  if (os.lstat(path).st_dev,os.lstat(path).st_ino)!=identity:raise RuntimeError('terminal target identity')
  unlinker(temp);state['committed']=True
  try:parent_fsync(path.parent)
  except BaseException:pass
 except BaseException:
  if linked and not state.get('committed'):
   tm=os.lstat(temp);pm=os.lstat(path)
   if ((tm.st_dev,tm.st_ino)!=(pm.st_dev,pm.st_ino) or (tm.st_dev,tm.st_ino)!=identity):raise RuntimeError('terminal exact2 foreign')
   path.unlink();temp.unlink();fsync_dir(path.parent);linked=False
  if not linked and os.path.lexists(temp):
   lm=os.lstat(temp)
   if not temp.is_symlink() and stat.S_ISREG(lm.st_mode) and (lm.st_dev,lm.st_ino)==identity and (os.fstat(fd).st_dev,os.fstat(fd).st_ino)==identity:temp.unlink();fsync_dir(path.parent)
   else:raise RuntimeError('foreign terminal preserved')
  raise
 finally:os.close(fd)
def held_member(path,payload):
 fd=os.open(path,os.O_RDWR|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600);m=os.fstat(fd);identity=(m.st_dev,m.st_ino)
 try:
  write_all(fd,payload);os.fsync(fd)
  if os.pread(fd,len(payload)+1,0)!=payload or os.pread(fd,1,len(payload))!=b'':raise RuntimeError('member EOF')
  return {'fd':fd,'identity':identity,'sha256':hashlib.sha256(payload).hexdigest(),'logical_bytes':len(payload),'name':path.name}
 except BaseException:
  os.close(fd)
  if os.path.lexists(path):
   lm=os.lstat(path)
   if not path.is_symlink() and (lm.st_dev,lm.st_ino)==identity:path.unlink();fsync_dir(path.parent)
  raise
def validate_members(root,members):
 for name,row in members.items():
  p=root/name;m=os.lstat(p);fm=os.fstat(row['fd']);payload=os.pread(row['fd'],row['logical_bytes']+1,0)
  if (p.is_symlink() or not stat.S_ISREG(m.st_mode) or (m.st_dev,m.st_ino)!=row['identity'] or (fm.st_dev,fm.st_ino)!=row['identity'] or payload!=os.pread(row['fd'],row['logical_bytes'],0) or len(payload)!=row['logical_bytes'] or os.pread(row['fd'],1,row['logical_bytes'])!=b'' or hashlib.sha256(payload).hexdigest()!=row['sha256']):raise RuntimeError(f'member drift:{name}')
def build_prep(helper_record,script_record,deploy_record,design_record,member_hook=lambda _name:None):
 PREP.mkdir(mode=0o700);fsync_dir(PREP.parent);identity=(os.lstat(PREP).st_dev,os.lstat(PREP).st_ino);members={}
 try:
  member_hook('transport_helper.py');members['transport_helper.py']=held_member(PREP/'transport_helper.py',SELF.read_bytes())
  member_hook('argv.json');members['argv.json']=held_member(PREP/'argv.json',cbytes({'format':'strict-track2-v529-stage-b-external-terminal-argv-v1','transport_helper':helper_record,'transport_script':script_record,'transport_deployment_record':deploy_record,'design_contract':design_record}))
  member_hook('helper_stdout.log');members['helper_stdout.log']=held_member(PREP/'helper_stdout.log',b'')
  member_hook('helper_stderr.log');members['helper_stderr.log']=held_member(PREP/'helper_stderr.log',b'')
  member_hook('intent.json');members['intent.json']=held_member(PREP/'intent.json',cbytes({'format':'strict-track2-v529-stage-b-external-terminal-intent-v1','status':'committed_before_validator_zero_external_terminal','retry_authorized':False}))
  fsync_dir(PREP);validate_members(PREP,members)
  if [x[0] for x in tree(PREP)['inventory']]!=EXACT5:raise RuntimeError('prep exact5')
  return identity,members
 except BaseException:
  close_members(members);cleanup_owned(PREP,identity,members);raise
def close_members(members):
 for row in members.values():
  if row['fd'] is not None:os.fsync(row['fd']);os.close(row['fd']);row['fd']=None
def cleanup_owned(root,identity,members):
 if not root.is_dir() or root.is_symlink():return
 m=os.lstat(root)
 if (m.st_dev,m.st_ino)!=identity:raise RuntimeError('foreign root')
 foreign=[]
 for p in list(root.iterdir()):
  row=members.get(p.name);lm=os.lstat(p)
  current=regular(p) if row is not None and not p.is_symlink() and stat.S_ISREG(lm.st_mode) else None
  if (row is not None and current is not None and (lm.st_dev,lm.st_ino)==row['identity']
      and current['sha256']==row['sha256'] and current['logical_bytes']==row['logical_bytes']):
   p.unlink()
  else:foreign.append(p.name)
 if foreign:raise RuntimeError('foreign members preserved:'+','.join(sorted(foreign)))
 root.rmdir();fsync_dir(root.parent)
def run(helper_record,script_record,deploy_record,design,design_record,hook=lambda _p:None,terminal_options=None):
 terminal_options=terminal_options or {}
 baseline=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM});committed=False;identity=None;members={};promoted=False
 try:
  first=snapshot(design);second=snapshot(design)
  if first!=second:raise RuntimeError('double snapshot')
  identity,members=build_prep(helper_record,script_record,deploy_record,design_record)
  rename_noreplace(PREP,OUT);promoted=True;fsync_dir(OUT.parent);validate_members(OUT,members)
  validation_error=None
  try:
   third=snapshot(design,allow_output=True)
   if third!=second:raise RuntimeError('preterminal third snapshot')
  except BaseException as e:validation_error=e;third=None
  passed=validation_error is None
  line1={'line_origin':'external_terminal_helper','candidate_receipt':first['candidate_receipt'],'candidate_tree':first['candidate_tree'],'validator_invocations_current':0}
  line2={'line_origin':'external_terminal_helper','passed':passed,'external_terminal_helper_invocations':1,'historical_validator_invocations':1,'current_validator_invocations':0,'oof_model_execution_invocations':0,'phase_a_invocations':0,'training_invocations':0,'error':None if passed else str(validation_error)}
  outfd=members['helper_stdout.log']['fd'];payload=cbytes(line1)+cbytes(line2);write_all(outfd,payload);os.ftruncate(outfd,len(payload));os.fsync(outfd);members['helper_stdout.log'].update({'sha256':hashlib.sha256(payload).hexdigest(),'logical_bytes':len(payload)})
  validate_members(OUT,members)
  receipt={'format':'strict-track2-v529-v527-stage-b-candidate-external-terminal-process-receipt-v1','status':'passed_external_terminal' if passed else 'failed_no_retry','passed':passed,'candidate_consumable':passed,'standalone_candidate_consumable':False,'external_terminal_helper_invocations':1,'historical_oof_readonly_gate_validator_invocations':1,'current_oof_readonly_gate_validator_invocations':0,'oof_model_execution_invocations':0,'phase_a_invocations':0,'training_invocations':0,'cache_reuse_invocations':0,'reward_read_invocations':0,'dev_hidden_final_outcome_read_invocations':0,'submission_invocations':0,'retry_authorized':False,'transport_helper':helper_record,'transport_script':script_record,'transport_deployment_record':deploy_record,'design_contract':design_record,'candidate_receipt':first['candidate_receipt'],'candidate_tree':first['candidate_tree'],'initial_snapshot':first,'preterminal_snapshot':third,'initial_and_preterminal_snapshots_equal':passed,'preterminal_error':None if passed else {'error_type':type(validation_error).__name__,'error':str(validation_error)},'stdout':regular(OUT/'helper_stdout.log'),'stderr':regular(OUT/'helper_stderr.log'),'stdout_contract':{'line_count':2,'line_origin':'external_terminal_helper','self_owned_no_child_capture':True},'terminal_publication_contract':{'noreplace':True,'commit_semantics':'current_exact6_visibility_after_owned_temp_unlink','crash_durability_claimed':False,'postcommit_parent_dir_fsync_best_effort':True,'no_rollback_relink_retry_or_second_publish_after_commit':True,'safe_precommit_rollback_then_failed_terminal_authorized':True}}
  state={'visible':False,'committed':False}
  foreign_terminal_preserved=False
  try:atomic_terminal(OUT/'process_receipt.json',receipt,state,hook,**terminal_options)
  except BaseException as terminal_error:
   foreign_terminal_preserved=os.path.lexists(OUT/'.process_receipt.json.noreplace-tmp')
   failure={**receipt,'status':'failed_no_retry','passed':False,'candidate_consumable':False,'preterminal_error':{'error_type':type(terminal_error).__name__,'error':str(terminal_error)},'foreign_terminal_temp_preserved':foreign_terminal_preserved,'terminal_tree_contract':'exact7_nonconsumable' if foreign_terminal_preserved else 'exact6_nonconsumable'}
   state={'visible':False,'committed':False};atomic_terminal(OUT/'process_receipt.json',failure,state,temp_name='.process_receipt.json.failed-noreplace-tmp');receipt=failure;passed=False
  committed=state['committed'];t=tree(OUT)
  expected_names=sorted(EXACT6+(['.process_receipt.json.noreplace-tmp'] if foreign_terminal_preserved else []))
  if not committed or t['file_count']!=len(expected_names) or [x[0] for x in t['inventory']]!=expected_names:raise RuntimeError('terminal exact6/exact7')
  close_members(members)
  postcommit=None;postcommit_error=None
  try:
   postcommit=snapshot(design,allow_output=True)
   if postcommit!=first:raise RuntimeError('postcommit current closure drift')
  except BaseException as e:postcommit_error={'error_type':type(e).__name__,'error':str(e),'success_priority_after_commit':True}
  for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,signal.SIG_IGN)
  return {'passed':receipt['passed'],'receipt':receipt,'tree':t,'postcommit_snapshot_sha256':None if postcommit is None else csha(postcommit),'postcommit_diagnostic':postcommit_error,'foreign_terminal_temp_preserved':foreign_terminal_preserved}
 except BaseException:
  for row in members.values():
   if row.get('fd') is not None:
    try:os.close(row['fd'])
    except OSError:pass
    row['fd']=None
  if identity is not None and PREP.exists():cleanup_owned(PREP,identity,members)
  if identity is not None and promoted and OUT.exists() and tree(OUT)['file_count']==5:
   failure={'format':'strict-track2-v529-v527-stage-b-candidate-external-terminal-process-receipt-v1','status':'failed_no_retry','passed':False,'candidate_consumable':False,'external_terminal_helper_invocations':1,'current_oof_readonly_gate_validator_invocations':0,'retry_authorized':False};state={'visible':False,'committed':False};atomic_terminal(OUT/'process_receipt.json',failure,state)
  raise
 finally:
  if not committed:signal.pthread_sigmask(signal.SIG_SETMASK,baseline)
def parse():
 p=argparse.ArgumentParser();p.add_argument('--self-sha');p.add_argument('--self-bytes',type=int);p.add_argument('--script-sha');p.add_argument('--script-bytes',type=int);p.add_argument('--read-only-self-test',action='store_true');return p.parse_args()
def emit_result(result):
 print(json.dumps({'passed':result['passed'],'evidence_tree':result['tree'],'status':result['receipt']['status'],'candidate_consumable':result['receipt']['candidate_consumable']},sort_keys=True))
 return 0 if result['passed'] else 79
def main():
 a=parse();h=regular(SELF,(a.self_sha,a.self_bytes));s=regular(SCRIPT,(a.script_sha,a.script_bytes));d=deployment(h,s);design,dr=load_design()
 if a.read_only_self_test:
  first=snapshot(design);second=snapshot(design)
  if first!=second:raise RuntimeError('selftest snapshots')
  print(json.dumps({'passed':True,'read_only_self_test':True,'validator_invocations':0,'oof_model_execution_invocations':0,'snapshot_sha256':csha(first)},sort_keys=True));return 0
 result=run(h,s,d,design,dr);return emit_result(result)
if __name__=='__main__':raise SystemExit(main())
