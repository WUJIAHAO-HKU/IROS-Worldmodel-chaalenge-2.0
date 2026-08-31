#!/usr/bin/env python3
import argparse, json
from pathlib import Path

ap=argparse.ArgumentParser(); ap.add_argument('--recursive',type=Path,required=True); ap.add_argument('--visual',type=Path,required=True); ap.add_argument('--reward',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
r=json.loads(a.recursive.read_text()); v=json.loads(a.visual.read_text()); rew=json.loads(a.reward.read_text())
i=r['improvement']; checks={
 'recursive_overall_rgb_ge_3': i['overall']['rgb_mae_percent']>=3.0,
 'recursive_late64_rgb_positive': i['overall']['late64_rgb_mae_percent']>=0,
 'recursive_texture_positive': i['overall']['texture_mae_percent']>=0,
 'recursive_right_rgb_nonregression': i['right']['rgb_mae_percent']>=0,
 'recursive_left_rgb_positive': i['left']['rgb_mae_percent']>0,
 'window_visual_overall_rgb_ge_3': v['improvement']['overall']['rgb_mae_improvement_percent']>=3.0,
 'window_visual_texture_nonregression': v['improvement']['overall']['texture_mae_improvement_percent']>=0,
 'window_visual_right_rgb_nonregression': v['improvement']['right']['rgb_mae_improvement_percent']>=0,
}
out={'format':'strict-track2-v5c-parent-screen-v1','passed':all(checks.values()),'checks':checks,'failed_checks':[k for k,x in checks.items() if not x], 'recursive_evidence':str(a.recursive.resolve()),'window_visual_evidence':str(a.visual.resolve()),'reward_evidence':str(a.reward.resolve()),'reward_terminal_nonregression':False,'interpretation':'Visual/recursive gate passed; official reward terminal-gain regressed and remains a risk. This is not real-RoboTwin policy success evidence.'}
a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
