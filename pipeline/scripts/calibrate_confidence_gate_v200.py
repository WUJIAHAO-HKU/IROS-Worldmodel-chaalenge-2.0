#!/usr/bin/env python3
"""Sweep a deployable confidence/agreement hard gate for high-frequency reprojection."""

import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from wam_pipeline.autoregressive_texture_memory_v190 import OneStepActionTextureMemoryUNet
from train_autoregressive_texture_memory_v190 import highpass
from train_autoregressive_unet import WindowDataset,frames_for_model


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument("--windows",required=True);p.add_argument("--split-manifest",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True);args=p.parse_args();device=torch.device("cuda")
    split=json.loads(Path(args.split_manifest).read_text());data=WindowDataset(Path(args.windows),split["validation_episodes"]);loader=DataLoader(data,batch_size=1,num_workers=2);checkpoint=Path(args.checkpoint);state=torch.load(checkpoint/"model.pt",map_location="cpu",weights_only=True)
    model=OneStepActionTextureMemoryUNet(use_source_expert=True).to(device);model.load_state_dict(state["state_dict"],strict=True);norm=np.load(checkpoint/"action_normalization.npz");mean=torch.from_numpy(norm["mean"]).to(device);std=torch.from_numpy(norm["std"]).to(device)
    configs=[(c,d,a) for c in (.4,.6,.8) for d in (.02,.05) for a in (.25,.5,1.)];names=[f"c{c}_d{d}_a{a}" for c,d,a in configs]
    parent_rgb=torch.zeros(8);parent_texture=torch.zeros(8);values={name:{"rgb":torch.zeros(8),"texture":torch.zeros(8),"mask":0.,"arms":{0:[0.,0.,0.,0.,0],1:[0.,0.,0.,0.,0]}} for name in names};count=0
    for context,history,future,target in loader:
        raw_history=history.clone();raw_future=future.clone();context=frames_for_model(context).to(device);memory=context.clone();target=frames_for_model(target).to(device);history=((history.to(device)-mean)/std).float();future=((future.to(device)-mean)/std).float();parent_frames=[];candidate_frames={name:[] for name in names}
        for index,action in enumerate(future.unbind(1)):
            _,detail=model(context,torch.cat((history,action[:,None]),1),memory,True);base=detail["base"].clamp(0,1);selected=detail["selected"].clamp(0,1);parent_frames.append(base);confidence=detail["source_logits"].softmax(1).max(1,keepdim=True).values;disagreement=(selected-base).abs().mean(1,keepdim=True);delta=highpass(selected)-highpass(base)
            for name,(threshold,distance,alpha) in zip(names,configs):
                mask=((confidence>=threshold)&(disagreement<=distance)&(index>=3)).to(base.dtype);candidate_frames[name].append((base+alpha*mask*delta).clamp(0,1));values[name]["mask"]+=float(mask.mean())
            context=torch.cat((context[:,1:],base[:,None]),1);history=torch.cat((history[:,1:],action[:,None]),1)
        parent=torch.stack(parent_frames,1);parent_error=(parent-target).abs().mean((2,3,4)).cpu();parent_hp=(highpass(parent.flatten(0,1))-highpass(target.flatten(0,1))).abs().mean((1,2,3)).reshape(1,8).cpu();parent_rgb+=parent_error[0];parent_texture+=parent_hp[0];count+=1
        sequence=torch.cat((raw_history[:,-1:],raw_future),1);delta_action=sequence.diff(dim=1).abs().mean(1);arm=int((delta_action[:,7:].mean(1)>delta_action[:,:7].mean(1)).item())
        for name in names:
            candidate=torch.stack(candidate_frames[name],1);rgb=(candidate-target).abs().mean((2,3,4)).cpu();texture=(highpass(candidate.flatten(0,1))-highpass(target.flatten(0,1))).abs().mean((1,2,3)).reshape(1,8).cpu();values[name]["rgb"]+=rgb[0];values[name]["texture"]+=texture[0];record=values[name]["arms"][arm];record[0]+=float(parent_error.mean());record[1]+=float(rgb.mean());record[2]+=float(parent_hp.mean());record[3]+=float(texture.mean());record[4]+=1
    parent_rgb=255*parent_rgb/count;parent_texture=255*parent_texture/count;results=[]
    for name,(confidence,distance,alpha) in zip(names,configs):
        value=values[name];rgb=255*value["rgb"]/count;texture=255*value["texture"]/count;result={"name":name,"confidence":confidence,"distance":distance,"alpha":alpha,"mask_rate":value["mask"]/(count*8),"rgb_mae":float(rgb.mean()),"texture_mae":float(texture.mean()),"rgb_improvement_percent":100*float((parent_rgb.mean()-rgb.mean())/parent_rgb.mean()),"texture_improvement_percent":100*float((parent_texture.mean()-texture.mean())/parent_texture.mean()),"frame_rgb_mae":rgb.tolist(),"arms":{}}
        scores=[result["rgb_improvement_percent"],result["texture_improvement_percent"]]
        for arm,(pr,vr,pt,vt,n) in value["arms"].items():
            pr=255*pr/n;vr=255*vr/n;pt=255*pt/n;vt=255*vt/n;item={"rgb_improvement_percent":100*(pr-vr)/pr,"texture_improvement_percent":100*(pt-vt)/pt};result["arms"][f"arm{arm}"]=item;scores.extend(item.values())
        result["gate_score"]=min(scores);results.append(result)
    results.sort(key=lambda item:item["gate_score"],reverse=True);report={"format":"v20.0-confidence-agreement-hard-gate","episodes":split["validation_episodes"],"windows":len(data),"parent_rgb_mae":float(parent_rgb.mean()),"parent_texture_mae":float(parent_texture.mean()),"results":results};output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(results[:5]))


if __name__=="__main__":main()
