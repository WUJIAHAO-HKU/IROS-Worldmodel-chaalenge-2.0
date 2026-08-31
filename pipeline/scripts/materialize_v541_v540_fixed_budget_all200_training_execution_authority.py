#!/usr/bin/env python3
"""Materialize one fresh v541 fixed-budget all200 training authority."""
from __future__ import annotations
import argparse,ctypes,errno,hashlib,importlib.util,json,os,signal,stat,subprocess,urllib.request
from pathlib import Path

ROOT=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge'); S=ROOT/'pipeline/scripts'; J=ROOT/'artifacts/strict_track2_joint_augmentation_20260810'
SELF=S/'materialize_v541_v540_fixed_budget_all200_training_execution_authority.py'
CONTRACT=S/'v541_v540_fixed_budget_all200_training_execution_authority_contract.json'
PREREG=S/'v541_v540_fixed_budget_all200_training_preregistration.json'; MANIFEST=S/'v541_v540_fixed_budget_all200_training_manifest.json'
TRAINER=S/'train_v541_v540_fixed_budget_all200.py'; AUDITOR=S/'audit_v541_v540_fixed_budget_all200.py'; LAUNCHER=S/'launch_v541_v540_fixed_budget_all200.py'
PREREG_EXPECTED=('d429b0aa8e8386e28fe6e90db2981da5c7cc57f64d0dcb250a69d66784a0bd34',5698)
MANIFEST_EXPECTED=('c973dcb5a6d2d1418e957c04d3d3cf2655977e3dd35a72bc42a0e911d360ab64',6497)
TRAINER_EXPECTED=('67896c46f6df51aa3a29076376ff5df0ac7980f8fe97a0f0212907d02f42e316',46906)
AUDITOR_EXPECTED=('57e628ba412532cb12d97d9927b8ad574899e14b50c2934e8e88f8adba175a98',4784)
LAUNCHER_EXPECTED=('8b9a569f19f08a23b448f87e6e7e1d7a95d20e4b048874aece9d9e213ec3f16b',2697)
AUTH_ROOT=J/'v541_v540_fixed_budget_all200_training_execution_authority_seed1672_20260828'; AUTH_PREP=AUTH_ROOT.with_name(AUTH_ROOT.name+'.authority-prep')
ATTEMPT_ROOT=J/'v541_v540_fixed_budget_all200_training_attempt_seed1672_20260828'; ATTEMPT_PREP=ATTEMPT_ROOT.with_name(ATTEMPT_ROOT.name+'.attempt-prep')
OUTPUT_ROOT=Path('/root/v541_v540_fixed_budget_all200_training_seed1624_20260828'); OUTPUT_PREP=OUTPUT_ROOT.with_name(OUTPUT_ROOT.name+'.training-prep')
V534_ROOT=Path('/root/v534_v533_actual_oof_seed1667_20260827'); V540_ROOT=Path('/root/v540_v539_public_s1_zero_update_gate_seed1671_20260828'); QUAL_ROOT=Path('/root/v524_v523_phase_a_cache_qualification_seed1660_20260826')
OUTPUT_FORMAT='strict-track2-v541-v540-fixed-budget-all200-training-execution-authority-v1'
OUTPUT_STATUS='authorized_exact_one_fixed_budget_training_boundary_pending_external_execution'

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
def load_trainer():
 regular(TRAINER,TRAINER_EXPECTED); spec=importlib.util.spec_from_file_location('v541_authority_trainer_schema',TRAINER); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def tree(root):
 rows=[]; lines=bytearray(); total=0
 for path in sorted(root.rglob('*')):
  if path.is_symlink(): raise RuntimeError('tree symlink:'+str(path))
  if path.is_file():
   rel=path.relative_to(root).as_posix(); digest=sha(path); size=path.stat().st_size; rows.append([rel,digest,size]); lines.extend(f'{digest}  {rel}\n'.encode()); total+=size
 return {'file_count':len(rows),'logical_file_bytes':total,'sha256sum_lines_digest_sha256':hashlib.sha256(lines).hexdigest(),'canonical_json_triples_digest_sha256':csha(rows)}
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
 names={str(TRAINER),str(AUDITOR),str(LAUNCHER),str(S/'train_v482_temporal8_residual_5fold.py')}; excluded=set(); pid=os.getpid()
 while pid>1:
  excluded.add(pid)
  try: pid=int((Path('/proc')/str(pid)/'stat').read_text().split()[3])
  except BaseException: break
 rows=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name) in excluded: continue
  try: argv=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
  except BaseException: continue
  match=sorted(set(argv)&names)
  if match: rows.append({'pid':int(p.name),'argv':argv,'matches':match})
 return rows
def snapshot(records,absence_map):
 v={'files':{role:regular(row['path']) for role,row in records.items()},'trees':{'qualification_exact20':tree(QUAL_ROOT),'v534_actual_oof_exact11':tree(V534_ROOT),'v540_public_s1_zero_update_exact8':tree(V540_ROOT)},'absences':{key:{'path':str(path),'absent':not os.path.lexists(path)} for key,path in sorted(absence_map.items())},'services':services_snapshot(),'gpu_compute_pids':gpu_processes(),'relevant_execution_pids':relevant_pids()}; v['canonical_sha256']=csha(v); return v
def rename_noreplace(source,target):
 fn=getattr(ctypes.CDLL(None,use_errno=True),'renameat2',None)
 if fn is None: raise RuntimeError('renameat2 unavailable')
 if fn(-100,os.fsencode(source),-100,os.fsencode(target),1)!=0:
  error=ctypes.get_errno()
  if error==errno.EEXIST: raise FileExistsError(target)
  raise OSError(error,os.strerror(error),str(target))
def fsync_dir(path):
 fd=os.open(path,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
 try: os.fsync(fd)
 finally: os.close(fd)
def held_read(fd,size):
 data=os.pread(fd,size+1,0)
 if len(data)!=size or os.pread(fd,1,size)!=b'': raise RuntimeError('held EOF')
 return data
def gate_owned_staged(prep,prep_identity,member,member_identity,fd,payload):
 pst=os.lstat(prep)
 if prep.is_symlink() or not stat.S_ISDIR(pst.st_mode) or (pst.st_dev,pst.st_ino)!=prep_identity or sorted(x.name for x in os.scandir(prep))!=['authority_receipt.json']: raise RuntimeError('prep identity/exact1')
 mst=os.lstat(member); fst=os.fstat(fd)
 if member.is_symlink() or not stat.S_ISREG(mst.st_mode) or (mst.st_dev,mst.st_ino)!=member_identity or (fst.st_dev,fst.st_ino)!=member_identity or fst.st_size!=len(payload) or held_read(fd,len(payload))!=payload: raise RuntimeError('held identity/payload')
def current_exact1_tree(root,digest,size):
 if root.is_symlink() or not root.is_dir() or sorted(x.name for x in os.scandir(root))!=['authority_receipt.json']: raise RuntimeError('current exact1')
 record=regular(root/'authority_receipt.json',(digest,size)); inv=[['authority_receipt.json',digest,size]]; lines=f'{digest}  authority_receipt.json\n'.encode()
 return {'root':str(root),'file_count':1,'logical_file_bytes':size,'inventory':inv,'sha256sum_lines_digest_sha256':hashlib.sha256(lines).hexdigest(),'canonical_json_triples_digest_sha256':csha(inv)}
def publish_exact1(receipt,stable_snapshot,records):
 if os.path.lexists(AUTH_ROOT) or os.path.lexists(AUTH_PREP): raise RuntimeError('authority prestate')
 oldmask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM}); prep_identity=member_identity=None; fd=None; committed=False
 payload=(json.dumps(receipt,sort_keys=True,indent=2,ensure_ascii=False)+'\n').encode(); digest=hashlib.sha256(payload).hexdigest()
 try:
  os.mkdir(AUTH_PREP,0o700); pst=os.lstat(AUTH_PREP); prep_identity=(pst.st_dev,pst.st_ino); member=AUTH_PREP/'authority_receipt.json'
  fd=os.open(member,os.O_RDWR|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600); fst=os.fstat(fd); member_identity=(fst.st_dev,fst.st_ino)
  offset=0
  while offset<len(payload):
   written=os.write(fd,payload[offset:])
   if type(written) is not int or written<=0 or written>len(payload)-offset: raise RuntimeError('held write')
   offset+=written
  os.fsync(fd); gate_owned_staged(AUTH_PREP,prep_identity,member,member_identity,fd,payload); fsync_dir(AUTH_PREP)
  stable_abs={k:v for k,v in absences().items() if k not in {'authority_root','authority_prep'}}
  if snapshot(records,stable_abs)!=stable_snapshot: raise RuntimeError('third snapshot drift')
  gate_owned_staged(AUTH_PREP,prep_identity,member,member_identity,fd,payload); rename_noreplace(AUTH_PREP,AUTH_ROOT); committed=True
  diagnostics=[]; observed=None
  for name,op in [('parent_fsync',lambda:fsync_dir(AUTH_ROOT.parent)),('held_readback',lambda:held_read(fd,len(payload))),('current_tree',lambda:current_exact1_tree(AUTH_ROOT,digest,len(payload)))]:
   try:
    result=op()
    if name=='current_tree': observed=result
   except BaseException as error: diagnostics.append({'operation':name,'error_type':type(error).__name__,'error':str(error)})
  exact=observed is not None and observed['inventory']==[['authority_receipt.json',digest,len(payload)]]
  return {'visibility_committed':True,'current_exact1_consumable':exact,'postcommit_diagnostics_passed':not diagnostics and exact,'postcommit_diagnostics':diagnostics,'retry_authorized':False}
 finally:
  if not committed and prep_identity is not None and AUTH_PREP.is_dir() and not AUTH_PREP.is_symlink():
   try:
    pst=os.lstat(AUTH_PREP); member=AUTH_PREP/'authority_receipt.json'
    if (pst.st_dev,pst.st_ino)==prep_identity and fd is not None and os.path.lexists(member):
     mst=os.lstat(member); fst=os.fstat(fd)
     if not member.is_symlink() and (mst.st_dev,mst.st_ino)==member_identity and (fst.st_dev,fst.st_ino)==member_identity and fst.st_size==len(payload) and held_read(fd,len(payload))==payload: member.unlink()
    if not any(AUTH_PREP.iterdir()): AUTH_PREP.rmdir(); fsync_dir(AUTH_PREP.parent)
   except BaseException: pass
  if fd is not None:
   try: os.close(fd)
   except OSError: pass
  signal.pthread_sigmask(signal.SIG_SETMASK,oldmask)
def validate_context(args):
 materializer=regular(args.materializer_source,(args.materializer_sha,args.materializer_bytes)); design=regular(args.contract,(args.contract_sha,args.contract_bytes)); trainer=load_trainer()
 regular(PREREG,PREREG_EXPECTED); regular(MANIFEST,MANIFEST_EXPECTED); regular(AUDITOR,AUDITOR_EXPECTED); regular(LAUNCHER,LAUNCHER_EXPECTED)
 prereg=json.loads(PREREG.read_text()); manifest=json.loads(MANIFEST.read_text()); contract=json.loads(args.contract.read_text())
 sources=contract.get('source_closure'); active6=contract.get('active_source_records')
 if type(sources) is not dict or set(sources)!=set(trainer.SOURCE_ROLE_ORDER) or type(active6) is not dict or set(active6)!=set(trainer.ACTIVE_ROLE_ORDER[1:]): raise RuntimeError('contract source schema')
 if sources.get('authority_materializer')!=materializer or active6.get('authority_materializer')!=materializer: raise RuntimeError('materializer bind')
 if trainer.ACTIVE_ROLE_ORDER!=list(prereg['active_source_role_order']) or trainer.ACTIVE_PATHS!=prereg['active_source_paths']: raise RuntimeError('active literals')
 return trainer,prereg,manifest,contract,design,sources
def parse():
 p=argparse.ArgumentParser(); p.add_argument('--contract',type=Path); p.add_argument('--contract-sha'); p.add_argument('--contract-bytes',type=int); p.add_argument('--materializer-source',type=Path); p.add_argument('--materializer-sha'); p.add_argument('--materializer-bytes',type=int); p.add_argument('--authority-root',type=Path); p.add_argument('--read-only-preflight',action='store_true'); p.add_argument('--synthetic-self-test',action='store_true'); return p.parse_args()
def main():
 args=parse()
 if args.synthetic_self_test:
  trainer=load_trainer(); checks={'contract_top':len(trainer.CONTRACT_KEYS)==49,'authority_top':len(trainer.AUTHORITY_KEYS)==59,'checks':len(trainer.CHECK_KEYS)==34,'sources':len(trainer.SOURCE_ROLE_ORDER)==28 and len(trainer.AUTHORITY_SOURCE_ALIASES)==29,'active':len(trainer.ACTIVE_ROLE_ORDER)==7,'authorization':len(json.loads(PREREG.read_text())['authorization'])==23}; print(json.dumps({'passed':all(checks.values()),'checks':checks,'checks_sha256':csha(checks)},sort_keys=True)); return 0 if all(checks.values()) else 1
 if any(v is None for v in (args.contract,args.contract_sha,args.contract_bytes,args.materializer_source,args.materializer_sha,args.materializer_bytes,args.authority_root)): raise RuntimeError('required production arguments')
 if args.authority_root!=AUTH_ROOT: raise RuntimeError('authority root')
 trainer,prereg,manifest,contract,design,sources=validate_context(args); absence_map=absences()
 if any(os.path.lexists(path) for path in absence_map.values()): raise FileExistsError('fresh roots')
 authority_sources={'authority_design_contract':design,**sources}; pre=snapshot(authority_sources,absence_map)
 if pre['gpu_compute_pids'] or pre['relevant_execution_pids'] or not all(x['absent'] for x in pre['absences'].values()): raise RuntimeError('prestate')
 checks={key:True for key in trainer.CHECK_KEYS}; receipt_contract=contract['authority_receipt_contract']
 def make_receipt(post):
  return {'format':OUTPUT_FORMAT,'status':OUTPUT_STATUS,'passed':True,'seed':1672,'lineage':trainer.EXPECTED_LINEAGE,'active_source_role_order':trainer.ACTIVE_ROLE_ORDER,'active_source_paths':trainer.ACTIVE_PATHS,'active_source_records':{'authority_design_contract':design,**contract['active_source_records']},'source_closure':authority_sources,'source_role_order':['authority_design_contract',*trainer.SOURCE_ROLE_ORDER],'source_aliases':trainer.AUTHORITY_SOURCE_ALIASES,'source_closure_sha256':csha(authority_sources),**{trainer.AUTHORITY_SOURCE_ALIASES[role]:row for role,row in authority_sources.items()},'budget_contract':prereg['budget_contract'],'authorization':prereg['authorization'],'base_training_contract':prereg['base_training_contract'],'output_contract':prereg['output_contract'],'v540_gate_transition':prereg['v540_gate_transition'],'fresh_attempt_root':str(ATTEMPT_ROOT),'fresh_output_root':str(OUTPUT_ROOT),'historical_absences':trainer.HISTORICAL_ABSENCES,'required_absences':trainer.CURRENT_ABSENCES,'checks':checks,'check_keys':trainer.CHECK_KEYS,'check_key_set_sha256':csha(trainer.CHECK_KEYS),'checks_sha256':csha(checks),'input_pre_snapshot':pre,'input_post_snapshot':post,'input_snapshots_exactly_equal':True,'runtime_observation':trainer.EXPECTED_RUNTIME_OBSERVATION,'execution_boundary':trainer.EXPECTED_EXECUTION_BOUNDARY}
 provisional=make_receipt(pre); exact_keys(provisional,trainer.AUTHORITY_KEYS,'authority receipt'); trainer.validate_documents(prereg,manifest,contract,provisional,argparse.Namespace(contract=args.contract,contract_sha=args.contract_sha),profile_steps=500,enforce_production_paths=True)
 if args.read_only_preflight:
  print(json.dumps({'passed':True,'status':'passed_read_only_preflight_no_materialization','contract_top_count':len(trainer.CONTRACT_KEYS),'authority_top_count':len(trainer.AUTHORITY_KEYS),'check_count':len(trainer.CHECK_KEYS),'source_count':len(trainer.SOURCE_ROLE_ORDER),'authority_source_count':1+len(trainer.SOURCE_ROLE_ORDER),'input_snapshot_sha256':pre['canonical_sha256']},sort_keys=True)); return 0
 post=snapshot(authority_sources,absence_map)
 if pre!=post: raise RuntimeError('snapshot drift')
 receipt=make_receipt(post); trainer.validate_documents(prereg,manifest,contract,receipt,argparse.Namespace(contract=args.contract,contract_sha=args.contract_sha),profile_steps=500,enforce_production_paths=True)
 stable_abs={k:v for k,v in absence_map.items() if k not in {'authority_root','authority_prep'}}; stable=snapshot(authority_sources,stable_abs)
 result=publish_exact1(receipt,stable,authority_sources)
 print(json.dumps({'passed':True,'authority_root':str(AUTH_ROOT),'publication':result,'receipt_sha256':sha(AUTH_ROOT/'authority_receipt.json'),'receipt_bytes':(AUTH_ROOT/'authority_receipt.json').stat().st_size},sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
