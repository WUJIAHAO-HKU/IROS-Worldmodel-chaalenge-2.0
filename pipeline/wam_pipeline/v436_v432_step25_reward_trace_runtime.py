"""Passed v432-step25 output plus v354-parent counterfactual telemetry."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
import numpy as np
from .arm_routed_autoregressive_runtime import Track2ArmRoutedAutoregressiveUNet
from .autoregressive_unet_runtime import Track2AutoregressiveUNet

class Track2V436V432Step25RewardTrace:
    def __init__(self, release_dir: str | Path, device="cuda") -> None:
        root=Path(release_dir); m=json.loads((root/"v436_diagnostic_manifest.json").read_text())
        if m.get("format")!="track2-v436-v432-step25-diagnostic-release-v1": raise RuntimeError("bad v436 release")
        if m.get("classification")!="parent world-model diagnostic only" or m.get("formal_candidate_authorized") is not False:
            raise RuntimeError("v436 release classification mismatch")
        for key, relative in (("left","left_expert"),("parent_right","parent_right"),("candidate_right","candidate_right")):
            path=root/m[relative]/"model.pt"
            if hashlib.sha256(path.read_bytes()).hexdigest()!=m["model_sha256"][key]:
                raise RuntimeError(f"v436 model hash mismatch: {key}")
        self.left=Track2AutoregressiveUNet(root/m["left_expert"],device)
        self.parent=Track2AutoregressiveUNet(root/m["parent_right"],device)
        self.candidate=Track2AutoregressiveUNet(root/m["candidate_right"],device)
        self.trace_dir=Path(os.environ["WAM_V436_TRACE_DIR"]); self.trace_dir.mkdir(parents=True,exist_ok=True)
        if any(self.trace_dir.iterdir()): raise RuntimeError("nonempty v436 trace")
        self.route_path=self.trace_dir/"route_trace.jsonl"; self.index=0

    @staticmethod
    def arm(history,future,instruction):
        return Track2ArmRoutedAutoregressiveUNet.active_arm(history,future,instruction)

    def _flush(self,baseline,candidate,prompts,rows):
        stem=f"batch_{self.index:05d}"
        np.save(self.trace_dir/f"{stem}_parent.npy",np.asarray(baseline[:,-1],dtype=np.uint8),allow_pickle=False)
        np.save(self.trace_dir/f"{stem}_candidate.npy",np.asarray(candidate[:,-1],dtype=np.uint8),allow_pickle=False)
        (self.trace_dir/f"{stem}_prompts.json").write_text(json.dumps(list(prompts),ensure_ascii=False)+"\n")
        with self.route_path.open("a",encoding="utf-8") as f:f.write(json.dumps({"batch":rows},separators=(",",":"))+"\n")
        self.index+=1

    def predict_batch(self,context,history,future,seeds,instructions):
        arms=np.asarray([self.arm(h,a,t) for h,a,t in zip(history,future,instructions)])
        out=np.empty((len(arms),8,256,256,3),dtype=np.uint8); base=np.empty_like(out)
        for arm,expert,parent in (("left",self.left,self.left),("right",self.candidate,self.parent)):
            mask=arms==arm
            if not mask.any():continue
            ix=np.flatnonzero(mask); texts=[instructions[i] for i in ix]
            out[mask]=expert.predict_batch(context[mask],history[mask],future[mask],np.asarray(seeds)[mask],texts)
            base[mask]=parent.predict_batch(context[mask],history[mask],future[mask],np.asarray(seeds)[mask],texts)
        self._flush(base,out,instructions,[{"arm":str(x),"learned_request":x=="right"} for x in arms])
        return out

    def predict(self,context,history,future,seed,instruction):
        arm=self.arm(history,future,instruction); expert=self.left if arm=="left" else self.candidate; parent=self.left if arm=="left" else self.parent
        out=expert.predict(context,history,future,seed,instruction); base=parent.predict(context,history,future,seed,instruction)
        self._flush(base[None],out[None],[instruction],[{"arm":arm,"learned_request":arm=="right"}]); return out
