#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from wam_pipeline.arm_routed_autoregressive_runtime import Track2ArmRoutedAutoregressiveUNet


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or a.std() == 0 or b.std() == 0: return None
    return float(np.corrcoef(a, b)[0, 1])


def concordance(values, target):
    agrees, ties, total = 0, 0, 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            dv, dt = values[i] - values[j], target[i] - target[j]
            if dv == 0 or dt == 0: ties += 1
            else: agrees += int(dv * dt > 0)
            total += 1
    return (agrees / max(total - ties, 1), agrees, ties, total)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--details', type=Path, required=True)
    p.add_argument('--audit-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise SystemExit('refusing to overwrite')
    with np.load(a.details, allow_pickle=False) as d:
        scales=d['scales']; terminal=d['rewards'][..., -1]; distances=d['distances']; motions=d['motions']; alignments=d['alignments']; alphas=d['alphas']; rows=d['selected_rows']
    mapping=[]
    files=sorted(a.audit_dir.glob('rollout_*.npz'))
    for fi,path in enumerate(files):
        with np.load(path,allow_pickle=False) as x:
            hs=x['history_actions'].astype(np.float32); fs=x['future_actions'].astype(np.float32)
            instructions=[str(v) for v in json.loads(str(x['instructions_json']))]
        for ai,(h,f,text) in enumerate(zip(hs,fs,instructions)):
            route=Track2ArmRoutedAutoregressiveUNet.active_arm(h,f,text)
            if route=='right' and h[-1,13]<0.5 and (f[:,13]<0.5).mean()>=0.75: mapping.append((fi,ai))
    assert len(mapping)==len(terminal)
    expert=-distances
    groups={}
    for qi,(fi,ai) in enumerate(mapping): groups.setdefault((fi,ai//4),[]).append(qi)
    reports=[]
    for si,scale in enumerate(scales):
        reward=terminal[:,si]
        metric={}
        for name,target in [('expert',expert),('motion',motions),('alignment',alignments),('alpha',alphas)]:
            pooled=[]
            group_corr=[]
            top_wins=[]
            for indices in groups.values():
                if len(indices)<2: continue
                idx=np.asarray(indices)
                c=concordance(reward[idx],target[idx]); pooled.append(c[1:])
                gc=corr(reward[idx],target[idx])
                if gc is not None: group_corr.append(gc)
                best=idx[int(np.argmax(reward[idx]))]
                top_wins.append(float(target[best]>=np.max(target[idx])-1e-12))
            agrees=sum(x[0] for x in pooled); ties=sum(x[1] for x in pooled); total=sum(x[2] for x in pooled)
            metric[name]={
                'groups_with_2plus':len(pooled),
                'pair_concordance':float(agrees/max(total-ties,1)),
                'pairs':int(total),'ties':int(ties),
                'mean_group_correlation':float(np.mean(group_corr)) if group_corr else None,
                'top_reward_is_best_target_fraction':float(np.mean(top_wins)) if top_wins else None}
        robust=reward<0.1
        top=np.argsort(reward)[-8:][::-1]
        reports.append({
          'scale':float(scale),'reward_gt_0p1':int((reward>=0.1).sum()),
          'robust_reward_motion_correlation':corr(reward[robust],motions[robust]),
          'robust_reward_expert_correlation':corr(reward[robust],expert[robust]),
          'group_ranking':metric,
          'top_queries':[{'query':int(i),'file':int(mapping[i][0]),'action':int(mapping[i][1]),
                          'reward':float(reward[i]),'motion':float(motions[i]),
                          'expert_similarity':float(expert[i]),'alignment':float(alignments[i]),
                          'alpha':float(alphas[i]),'selected_row':int(rows[i])} for i in top]})
    report={'format':'strict-track2-v242-group-ranking-analysis-v1','post_queries':len(mapping),
            'groups_total':len(groups),'groups_with_2plus':sum(len(x)>=2 for x in groups.values()),
            'alpha_nonzero':int((alphas>0).sum()),'unique_selected_rows':int(len(set(rows.tolist()))),
            'scales':reports,
            'guards':{'public_data_only':True,'policy_modified':False,'hidden_or_final_data':False,'real_submission':False}}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
