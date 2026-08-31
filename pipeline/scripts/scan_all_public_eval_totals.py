from pathlib import Path
import re

root = Path('/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_official_20260810/real_robotwin_eval')
pat = re.compile(r"\[INFO .*?RLinf\] \{(.*?'eval/num_trajectories':\s*(\d+).*?)\}")
metric = re.compile(r"'eval/(success_once|right_success|left_success|arm_right|arm_left)':\s*array\(([-+0-9.eE]+)")
groups={}
for p in root.rglob('launcher.log'):
    try: text=p.read_text(errors='ignore')
    except: continue
    matches=list(pat.finditer(text))
    if not matches: continue
    m=matches[-1]
    count=int(m.group(2)); vals={k:float(v) for k,v in metric.findall(m.group(1))}
    if 'success_once' not in vals: continue
    key=p.parent.parent
    row=groups.setdefault(key,{'count':0,'success':0,'right_count':0,'right_success':0,'left_count':0,'left_success':0,'batches':0})
    row['count']+=count; row['success']+=round(vals['success_once']*count)
    row['right_count']+=round(vals.get('arm_right',0)*count); row['right_success']+=round(vals.get('right_success',0)*count)
    row['left_count']+=round(vals.get('arm_left',0)*count); row['left_success']+=round(vals.get('left_success',0)*count)
    row['batches']+=1
for key,row in sorted(groups.items(), key=lambda kv:(kv[1]['success'],kv[1]['count']), reverse=True):
    if row['count'] >= 32:
        print(f"{row['success']}/{row['count']} R={row['right_success']}/{row['right_count']} L={row['left_success']}/{row['left_count']} batches={row['batches']} {key}")
