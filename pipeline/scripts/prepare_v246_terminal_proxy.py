#!/usr/bin/env python3
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
B=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge');O=B/'artifacts/strict_track2_official_20260810';N='v246_public_clean_terminal_proxy_seed1449_20260819'
def main():
 r=O/'run_registry'/N
 if r.exists():raise SystemExit('refusing overwrite')
 r.mkdir(parents=True);p=B/'pipeline/scripts/analyze_v246_terminal_successor_proxy.py';x={'format':'strict-track2-v246-terminal-proxy-preregistration-v1','registered_at':datetime.now(timezone.utc).isoformat(),'grid':{'temperature':[1,3,5],'alignment_gate':['positive','positive_squared','smooth'],'alpha_scale':[2,4,8]},'target_rule':'visually nearest official public clean episode, final available row at start>=112','selection_rule':'success-like >=8, positive expert/alignment global correlation, expert/alignment group concordance >=.55; maximize conservative score','source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'guards':{'runtime_uses_reward':False,'public_data_only':True,'policy_modified':False,'hidden_or_final_data':False,'real_submission':False}};(r/'preregistration.json').write_text(json.dumps(x,indent=2)+'\n')
if __name__=='__main__':main()
