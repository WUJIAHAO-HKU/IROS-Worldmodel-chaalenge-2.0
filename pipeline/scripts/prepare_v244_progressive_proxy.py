#!/usr/bin/env python3
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
BASE=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge');OFF=BASE/'artifacts/strict_track2_official_20260810';NAME='v244_public_clean_progressive_proxy_seed1446_20260819'
def sha(p):
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def main():
 reg=OFF/'run_registry'/NAME
 if reg.exists():raise SystemExit('refusing overwrite')
 reg.mkdir(parents=True);src=BASE/'pipeline/scripts/analyze_v244_progressive_successor_proxy.py'
 x={'format':'strict-track2-v244-progressive-proxy-preregistration-v1','registered_at':datetime.now(timezone.utc).isoformat(),'candidate_grid':{'action_weight':[.5,1,2.5,4],'offset':[8,12,16,24],'temperature':[1,3,5],'alpha_scale':[1,2,4]},'selection_rule':'positive global expert/alignment correlations, >=.55 group concordance for both, >=.5 high-reward public successor coverage; maximize conservative score','source_sha256':sha(src),'guards':{'public_data_only':True,'policy_modified':False,'hidden_or_final_data':False,'real_submission':False}}
 (reg/'preregistration.json').write_text(json.dumps(x,indent=2)+'\n')
if __name__=='__main__':main()
