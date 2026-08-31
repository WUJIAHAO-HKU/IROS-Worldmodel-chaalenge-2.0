"""V474 frozen-v169 + median4 appearance + group-context action residual runtime."""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
import numpy as np
from .v169_arm_routed_runtime import Track2V169ArmRoutedRuntime
FORMAT="track2-v474-median4-c-group-e-parent-release-v1"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def explicit_right(text):
 value=str(text or "").lower();return "right arm" in value and "left arm" not in value
def postclose_request(history,future):
 h=np.asarray(history);f=np.asarray(future);return bool(h.shape==(4,14) and f.shape==(8,14) and np.isfinite(h).all() and np.isfinite(f).all() and float(h[-1,13])<.5 and np.all(f[:,13]<.5))
def gate_decision(history,future,instruction):
 h=np.asarray(history);f=np.asarray(future);ok=postclose_request(h,f) and explicit_right(instruction)
 return {"gate":bool(ok),"phase":"postclose" if ok else "g0"}
def context_feature(rgb):
 x=np.asarray(rgb,np.float32)
 if x.shape!=(256,256,3):raise ValueError("v474 context image")
 return x.reshape(8,32,8,32,3).mean((1,3)).reshape(192).astype(np.float32)
def action_feature(history,future):
 h=np.asarray(history,np.float32);f=np.asarray(future,np.float32)
 if h.shape!=(4,14) or f.shape!=(8,14) or not np.isfinite(h).all() or not np.isfinite(f).all():raise ValueError("v474 actions")
 anchor=h[-1,7:13];return np.concatenate((f[-1,7:13]-anchor,(f[:,7:13]-anchor).reshape(48))).astype(np.float32)
def no_transport(history,future):return np.array_equal(np.asarray(future,np.float32)[:,7:13],np.broadcast_to(np.asarray(history,np.float32)[-1,7:13],(8,6)))
def inside(path,root):
 path=Path(path).resolve();root=Path(root).resolve();return path==root or root in path.parents
def directory_target_sha(path):
 path=Path(path).resolve();items=[]
 for parent,dirs,files in os.walk(path,followlinks=False):
  for name in sorted(dirs+files):
   item=Path(parent)/name;relative=str(item.relative_to(path))
   if item.is_symlink():items.append(["link",relative,os.readlink(item)])
   elif item.is_file():items.append(["file",relative,sha(item)])
   else:items.append(["dir",relative,None])
 return hashlib.sha256(json.dumps(items,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def release_inventory(release,joint_root):
 release=Path(release).absolute();joint_root=Path(joint_root).resolve();items=[];seen_dirs=set();seen_files=set();active=set()
 def visit(logical,relative):
  resolved=logical.resolve()
  if not inside(resolved,joint_root) or not resolved.is_dir():raise RuntimeError("v474 inventory directory escape")
  dir_key=(resolved.stat().st_dev,resolved.stat().st_ino)
  if dir_key in active:raise RuntimeError("v474 inventory cycle")
  if dir_key in seen_dirs:raise RuntimeError("v474 inventory duplicate directory target")
  active.add(dir_key);seen_dirs.add(dir_key)
  for child in sorted(logical.iterdir(),key=lambda x:x.name):
   rel=relative/child.name;target=child.resolve()
   if not inside(target,joint_root):raise RuntimeError("v474 inventory target escape")
   if child.is_symlink():items.append(("directory_symlink" if target.is_dir() else "file_symlink","v169_release_link",str(rel),str(child.absolute()),os.readlink(child),str(target)))
   if target.is_dir():visit(child,rel)
   elif target.is_file():
    target_key=(target.stat().st_dev,target.stat().st_ino)
    if target_key in seen_files:raise RuntimeError("v474 inventory duplicate file target")
    seen_files.add(target_key)
    if not child.is_symlink():items.append(("file","v169_release_target" if not inside(target,release.resolve()) else "v169_release",str(rel),str(child.absolute()),None,str(target)))
   else:raise RuntimeError("v474 inventory unsupported target")
  active.remove(dir_key)
 visit(release,Path("."));return sorted(items)
def verify_v169(v):
 if not isinstance(v,dict):raise RuntimeError("v474 v169 closure")
 release=Path(v["release"]).absolute();library=Path(v["library"]).resolve();joint_root=release.resolve().parent;roots={"v169_release":release,"v169_release_target":release,"v169_release_link":release,"v169_base_release":(release/"v168_release/base_release").resolve(),"v169_library":library};seen=set()
 for records in (v["release_files"],v["library_files"]):
  for row in records:
   if set(row)!={"record_type","base_kind","relative","lexical_path","link_target","resolved_path","target_sha"} or row["base_kind"] not in roots:raise RuntimeError("v474 closure schema")
   base=roots[row["base_kind"]];lexical=(base/row["relative"]).absolute();resolved=lexical.resolve();key=(row["record_type"],row["base_kind"],row["relative"],row["resolved_path"])
   if key in seen:raise RuntimeError("v474 duplicate closure record")
   seen.add(key);allowed_root=library if row["base_kind"]=="v169_library" else joint_root
   if os.path.commonpath((str(base),str(lexical)))!=str(base) or not inside(resolved,allowed_root) or str(lexical)!=row["lexical_path"] or str(resolved)!=row["resolved_path"]:raise RuntimeError("v474 closure path drift")
   is_link=lexical.is_symlink()
   if row["link_target"]!=(os.readlink(lexical) if is_link else None):raise RuntimeError("v474 symlink text drift")
   if row["record_type"]=="directory_symlink":
    if not is_link or not resolved.is_dir() or directory_target_sha(resolved)!=row["target_sha"]:raise RuntimeError("v474 directory symlink drift")
   elif not resolved.is_file() or sha(resolved)!=row["target_sha"]:raise RuntimeError("v474 file closure drift")
 declared=sorted((row["record_type"],row["base_kind"],row["relative"],row["lexical_path"],row["link_target"],row["resolved_path"]) for row in v["release_files"])
 if declared!=release_inventory(release,joint_root) or sum(x["record_type"]=="directory_symlink" for x in v["release_files"])!=4 or sum(x["record_type"]=="file_symlink" for x in v["release_files"])!=2 or len({x["resolved_path"] for x in v["release_files"]})!=len(v["release_files"]):raise RuntimeError("v474 exact release inventory")
 for key,path in (("release_manifest_sha256",Path(v["release"])/"v169_arm_routed_manifest.json"),("library_manifest_sha256",Path(v["library_manifest"]))):
  if not path.is_file() or sha(path)!=v[key]:raise RuntimeError(f"v474 {key}")
 if not Path(v["runtime_source"]["path"]).is_file() or sha(v["runtime_source"]["path"])!=v["runtime_source"]["sha256"]:raise RuntimeError("v474 v169 runtime source")
 return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()
class Track2V474V473Median4Parent:
 def __init__(self,release_dir,device="cuda"):
  root=Path(release_dir).resolve();mp=root/"v474_median4_c_group_e_manifest.json";manifest=json.loads(mp.read_text());lib=(root/manifest["library"]).resolve();vp=(root/manifest["v169_closure"]).resolve()
  guards={"format":FORMAT,"median_definition":"sort-float32-middle-two-times-0.5","endpoint_only":True,"official_reward_runtime_used":False,"policy_modified":False,"formal":False,"endpoint_parent_data_authorized":False,"s1_authorized":False,"rl_authorized":False}
  if any(manifest.get(k)!=v for k,v in guards.items()) or not (sha(Path(__file__))==manifest["runtime_source_sha256"]==manifest["sha256"]["runtime_source"]) or root not in lib.parents or root not in vp.parents or not (sha(lib)==manifest["all200_library_sha256"]==manifest["sha256"]["library"]) or sha(vp)!=manifest["sha256"]["v169_closure"]:raise RuntimeError("bad v474 release")
  v=json.loads(vp.read_text());digest=verify_v169(v)
  if digest!=manifest["v169_closure_digest"] or v["release_manifest_sha256"]!=manifest["v169_release_manifest_sha256"] or v["library_manifest_sha256"]!=manifest["v169_library_manifest_sha256"]:raise RuntimeError("v474 closure digest")
  with np.load(lib,allow_pickle=False) as z:
   if sorted(z.files)!=["action","action_sha","appearance","context","context_sha","transport"]:raise RuntimeError("v474 library keys")
   self.context=np.asarray(z["context"],np.float32);self.action=np.asarray(z["action"],np.float32);self.C=np.asarray(z["appearance"],np.int16);self.E=np.asarray(z["transport"],np.int16);self.action_sha=np.asarray(z["action_sha"]).astype(str);self.context_sha=np.asarray(z["context_sha"]).astype(str)
  if self.context.shape!=(200,192) or self.action.shape!=(200,5,54) or self.C.shape!=(200,256,256,3) or self.E.shape!=(200,5,256,256,3) or self.action_sha.shape!=(200,5) or self.context_sha.shape!=(200,) or not np.array_equal(self.E[:,1],np.zeros_like(self.E[:,1])):raise RuntimeError("v474 library schema")
  self.cm=self.context.mean(0);self.cs=np.maximum(self.context.std(0),1e-4);flat=self.action.reshape(-1,54);self.am=flat.mean(0);self.asd=np.maximum(flat.std(0),1e-4);self.v169=Track2V169ArmRoutedRuntime(Path(v["release"]),Path(v["library"]),device)
 def _v169(self,context,history,future,seeds,instructions):
  out=[]
  for i in range(0,len(context),4):out.append(self.v169.predict_batch(context[i:i+4],history[i:i+4],future[i:i+4],seeds[i:i+4],instructions[i:i+4]))
  return np.concatenate(out) if out else np.empty((0,8,256,256,3),np.uint8)
 def _residual(self,rgb,history,future,override_action_feature=None):
  q=context_feature(rgb);dc=np.square((self.context-q)/self.cs).mean(1);chosen=np.lexsort((self.context_sha,self.action_sha[:,1],dc.astype(np.float64)))[:4];s=np.sort(self.C[chosen].astype(np.float32),axis=0);c=np.float32(.5)*(s[1]+s[2])
  if no_transport(history,future):return c,[str(self.context_sha[i]) for i in chosen],[]
  aq=action_feature(history,future) if override_action_feature is None else np.asarray(override_action_feature,np.float32)
  if aq.shape!=(54,) or not np.isfinite(aq).all():raise ValueError("v474 override action feature")
  maps=[];actions=[]
  for i in chosen:
   da=np.square((self.action[i]-aq)/self.asd).mean(1);k=int(np.lexsort((np.repeat(self.context_sha[i],5),self.action_sha[i],da.astype(np.float64)))[0]);maps.append(self.E[i,k]);actions.append(str(self.action_sha[i,k]))
  return c+np.mean(np.asarray(maps,dtype=np.float32),axis=0),[str(self.context_sha[i]) for i in chosen],actions
 def _predict(self,context,history,future,seeds,instructions,override=None):
  context=np.asarray(context);history=np.asarray(history,np.float32);future=np.asarray(future,np.float32);seeds=np.asarray(seeds);n=len(context)
  if context.shape!=(n,5,256,256,3) or context.dtype!=np.uint8 or history.shape!=(n,4,14) or future.shape!=(n,8,14) or seeds.shape!=(n,) or len(instructions)!=n or not np.isfinite(history).all() or not np.isfinite(future).all():raise ValueError("v474 batch contract")
  if override is not None:
   override=np.asarray(override,np.float32)
   if override.shape!=(n,54) or not np.isfinite(override).all():raise ValueError("v474 override batch")
  baseline=self._v169(context,history,future,seeds,list(instructions))
  if baseline.shape!=(n,8,256,256,3) or baseline.dtype!=np.uint8:raise RuntimeError("v474 v169 output")
  output=baseline.copy();decisions=[]
  for h,f,t in zip(history,future,instructions):
   d=gate_decision(h,f,t);d.update({"explicit_right":explicit_right(t),"postclose":postclose_request(h,f),"no_transport":bool(no_transport(h,f)),"context_neighbor_sha256":[],"action_prototype_sha256":[],"frame7_changed":False});decisions.append(d)
  for i,d in enumerate(decisions):
   if d["gate"]:
    residual,contexts,actions=self._residual(context[i,-1],history[i],future[i],None if override is None else override[i]);output[i,7]=np.clip(np.rint(baseline[i,7].astype(np.float32)+residual),0,255).astype(np.uint8);d["context_neighbor_sha256"]=contexts;d["action_prototype_sha256"]=actions;d["frame7_changed"]=bool(not np.array_equal(output[i,7],baseline[i,7]))
  return baseline,output,decisions
 def predict_batch_with_baseline(self,context,history,future,seeds,instructions):return self._predict(context,history,future,seeds,instructions)
 def predict_batch_s1_action_override(self,context,history,true_future,seeds,instructions,override_action_features):return self._predict(context,history,true_future,seeds,instructions,override_action_features)
 def predict_batch(self,*args):return self.predict_batch_with_baseline(*args)[1]
 def gate_decision(self,history,future,instruction):return gate_decision(history,future,instruction)
 def predict_with_baseline(self,context,history,future,seed,instruction):
  b,o,d=self.predict_batch_with_baseline(np.asarray(context)[None],np.asarray(history)[None],np.asarray(future)[None],np.asarray([seed]),[instruction]);return b[0],o[0],d[0]
 def predict(self,context,history,future,seed,instruction):return self.predict_with_baseline(context,history,future,seed,instruction)[1]
