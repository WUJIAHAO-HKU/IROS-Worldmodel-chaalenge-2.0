#!/usr/bin/env python3
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
B=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge');O=B/'artifacts/strict_track2_official_20260810';N='v249b_terminal_alignment_floor_proxy_seed1453_20260819'
def main():
 r=O/'run_registry'/N
 if r.exists():raise SystemExit('refusing overwrite')
 r.mkdir(parents=True);p=B/'pipeline/scripts/analyze_v246_terminal_successor_proxy.py';x={'format':'strict-track2-v249-alignment-floor-proxy-preregistration-v1','registered_at':datetime.now(timezone.utc).isoformat(),'grid':{'temperature':[.25,.5,1,3,5],'alignment_gate':['positive','positive_squared','smooth','floor025','floor050','floor075'],'alpha_scale':[2,4,8,16,32,64]},'selection_rule':'one-step success-like>=8 and positive expert/alignment ranking; combine with v248 public long false-positive alignment analysis','source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'guards':{'runtime_uses_reward':False,'public_data_only':True,'policy_modified':False,'hidden_or_final_data':False,'real_submission':False}};(r/'preregistration.json').write_text(json.dumps(x,indent=2)+'\n')
if __name__=='__main__':main()
