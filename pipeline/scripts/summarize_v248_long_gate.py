#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
def summary(path,mask,threshold):
 x=json.loads(path.read_text());scores=np.asarray(x['raw_scores']['candidate'],float)[mask];peak=scores.max(1);return {'count':len(peak),'mean':float(scores.mean()),'peak_mean':float(peak.mean()),'peak_max':float(peak.max()),'hits':int((peak>=threshold).sum()),'hit_rate':float((peak>=threshold).mean())}
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--registry',type=Path,required=True);a=p.parse_args();pre=json.loads((a.registry/'preregistration.json').read_text());f=pre['fixed_audit']
 with np.load(a.run/'audit/public_success_baseline.npz',allow_pickle=False) as x:sm=x['arm_right'].astype(bool)
 with np.load(a.run/'audit/public_failure_baseline.npz',allow_pickle=False) as x:fm=x['arm_right'].astype(bool)
 s=summary(a.run/'audit/public_success_reward.json',sm,f['threshold']);q=summary(a.run/'audit/public_failure_reward.json',fm,f['threshold']);capture=json.loads(Path(pre['input_capture_gate']).read_text());checks={'capture_gate':capture['passed'],'success_recall':s['hit_rate']>=f['success_hit_rate_min'],'failure_consistency':q['hit_rate']<=f['failure_hit_rate_max'],'margin':s['hit_rate']-q['hit_rate']>=f['margin_min'],'service_contract':(a.run/'audit/service_acceptance.json').is_file()};report={'format':'strict-track2-v248-long128-gate-v1','success':s,'failure':q,'margin':s['hit_rate']-q['hit_rate'],'checks':checks,'passed':all(checks.values()),'guards':pre['guards']};(a.run/'audit/long_gate_report.json').write_text(json.dumps(report,indent=2)+'\n');(a.run/('V248_LONG_GATE_PASSED' if report['passed'] else 'V248_LONG_GATE_REJECTED')).touch();print(json.dumps(report,indent=2))
if __name__=='__main__':main()
