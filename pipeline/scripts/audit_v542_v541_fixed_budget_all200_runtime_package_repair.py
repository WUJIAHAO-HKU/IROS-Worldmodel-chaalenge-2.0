#!/usr/bin/env python3
"""Independent fixed-budget output auditor; never trains or mutates model/data."""
from __future__ import annotations
import hashlib, io, json

def cbytes(value): return (json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode()
def sha(raw): return hashlib.sha256(raw).hexdigest()

COUNT_KEYS=['backward_calls','checkpoint_writes','early_stop_calls','fold_training_calls','forward_calls','gradient_clip_calls','intermediate_checkpoint_writes','loss_calls','optimizer_steps','optimizer_zero_grad_calls','publication_calls','resume_calls','retry_calls','scheduler_instances','scheduler_steps','search_calls','total_prefix_frame_examples']

def audit(*, budget:dict, events_raw:bytes, checkpoint_raw:bytes, receipt:dict, source_manifest:dict, expected_checkpoint_format:str) -> dict:
    if type(budget) is not dict or type(receipt) is not dict or type(source_manifest) is not dict: raise RuntimeError('audit input schema')
    expected={
        'backward_calls':budget['optimizer_steps'],'checkpoint_writes':1,'early_stop_calls':0,'fold_training_calls':0,
        'forward_calls':budget['optimizer_steps'],'gradient_clip_calls':budget['optimizer_steps'],
        'intermediate_checkpoint_writes':0,'loss_calls':budget['optimizer_steps'],
        'optimizer_steps':budget['optimizer_steps'],'optimizer_zero_grad_calls':budget['optimizer_steps'],
        'publication_calls':1,'resume_calls':0,'retry_calls':0,'scheduler_instances':0,'scheduler_steps':0,
        'search_calls':0,'total_prefix_frame_examples':budget['total_prefix_frame_examples']}
    if set(receipt.get('counts',{}))!=set(COUNT_KEYS): raise RuntimeError('count keyset')
    for key,value in expected.items():
        if type(receipt['counts'].get(key)) is not int or receipt['counts'][key]!=value: raise RuntimeError('count '+key)
    lines=events_raw.splitlines()
    if len(lines)!=budget['optimizer_steps'] or events_raw!=(b'\n'.join(lines)+b'\n'): raise RuntimeError('event count')
    prior=None
    for ordinal,line in enumerate(lines):
        row=json.loads(line)
        if set(row)!={'batch_sequence_ids','gradient_norm_finite','loss_finite','ordinal','optimizer_step_committed'}: raise RuntimeError('event schema')
        if type(row['ordinal']) is not int or row['ordinal']!=ordinal: raise RuntimeError('event ordinal')
        if type(row['batch_sequence_ids']) is not list or len(row['batch_sequence_ids'])!=2 or any(type(x) is not int for x in row['batch_sequence_ids']): raise RuntimeError('event batch')
        if row['loss_finite'] is not True or row['gradient_norm_finite'] is not True or row['optimizer_step_committed'] is not True: raise RuntimeError('event status')
        prior=row
    try:
        import torch
        checkpoint=torch.load(io.BytesIO(checkpoint_raw),map_location='cpu',weights_only=False)
    except Exception as error: raise RuntimeError('checkpoint decode') from error
    if (set(checkpoint)!={'action_lower','action_upper','channels','closure_digest','feature_schema','format','model','optimizer_steps','precision','schedule_sha256','training_scope'}
            or checkpoint['format']!=expected_checkpoint_format or checkpoint['training_scope']!='all200-action-fixed-budget'
            or type(checkpoint['optimizer_steps']) is not int or checkpoint['optimizer_steps']!=budget['optimizer_steps']
            or checkpoint['schedule_sha256']!=budget['all200_schedule_sha256'] or checkpoint['channels']!=16 or checkpoint['precision']!='bf16'):
        raise RuntimeError('checkpoint contract')
    if not checkpoint['model'] or any(not bool(torch.isfinite(value).all()) for value in checkpoint['model'].values()): raise RuntimeError('checkpoint finite')
    if set(source_manifest)!={'authority_receipt','budget_sha256','checkpoint','events','format','input_records','trainer'}: raise RuntimeError('source manifest')
    if source_manifest['checkpoint']!={'sha256':sha(checkpoint_raw),'logical_bytes':len(checkpoint_raw)} or source_manifest['events']!={'sha256':sha(events_raw),'logical_bytes':len(events_raw)}: raise RuntimeError('source manifest payload')
    checks={'budget_counts_exact':True,'events_exact_budget':True,'events_all_committed':True,'checkpoint_exact_final_step':True,'checkpoint_finite':True,'source_manifest_payloads':True,'no_early_stop_search_retry_resume_intermediate':True}
    return {'format':'strict-track2-v542-v541-fixed-budget-all200-independent-audit-v1','passed':True,'checks':checks,'checks_sha256':hashlib.sha256(json.dumps(checks,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'optimizer_steps':budget['optimizer_steps'],'checkpoint_sha256':sha(checkpoint_raw),'events_sha256':sha(events_raw),'retry_authorized':False}

if __name__=='__main__': raise SystemExit('import-only auditor')
