#!/usr/bin/env python3
"""Fresh v507 exact-once authority for Phase-A cache qualification; no Phase-A execution."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path

ROOT = Path('/root/autodl-tmp/IROS_WAM_2.0 challenge')
J = ROOT / 'artifacts/strict_track2_joint_augmentation_20260810'
SCRIPTS = ROOT / 'pipeline/scripts'
SELF_PATH = SCRIPTS / 'materialize_v507_v506_phase_a_cache_qualification_execution_authority.py'
CONTRACT_PATH = SCRIPTS / 'v507_v506_phase_a_cache_qualification_execution_authority_contract.json'
LAUNCHER_PATH = SCRIPTS / 'launch_v507_v506_phase_a_cache_qualification.py'
BASE_PATH = SCRIPTS / 'materialize_v503_v502_compact_reconciliation_execution_authority.py'
BASE_SHA = 'bfdf33ab525d3b9e98ea116604f309354160cf98021478c94c177caa5060d879'
BASE_BYTES = 42234
AUTHORITY_ROOT = J / 'v507_v506_phase_a_cache_qualification_execution_authority_seed1648_20260825'
ATTEMPT_ROOT = J / 'v507_v506_phase_a_cache_qualification_attempt_seed1648_20260825'
QUALIFICATION_ROOT = Path('/root/v485_v169_cache_qualification_seed1627_20260824')

V506_AUTH_ROOT = J / 'v506_v505_v503_compact_reconciliation_execution_authority_seed1647_20260825'
V506_AUTH_RECEIPT = V506_AUTH_ROOT / 'authority_receipt.json'
V506_AUTH_SHA = '33b72c430efbc0c96d21445b3f14ae31eef29f9a35566ced6cb8c9d582b33543'
V506_AUTH_BYTES = 37311
V506_EVIDENCE = J / 'v506_v505_v503_compact_reconciliation_execution_authority_materialization_evidence_seed1647_20260825'
V506_PROCESS = V506_EVIDENCE / 'process_receipt.json'
V506_PROCESS_SHA = '2d68df738dfdc2d7b0a1a324aadeaf484f49314f02d06e83f93d454f8ad46129'
V506_PROCESS_BYTES = 18469
V506_ATTEMPT = J / 'v506_v505_v503_compact_single_reconciler_attempt_seed1647_20260825'
V506_TERMINAL = V506_ATTEMPT / 'terminal_receipt.json'
V506_TERMINAL_SHA = '3c213ef203b8ee86ebbd83b9c1f5f89f886c4444a46fd4e2b6652973a8a7e931'
V506_TERMINAL_BYTES = 19783
F813_ROOT = J / 'v486_v485_phase_a_static_reconciliation_seed1628_20260824'
TRANSPARENT = F813_ROOT / 'transparent_static_audit.json'
TRANSPARENT_SHA = '7740cabe53661da4832f2624642af01dac922e05aa6a62c6ef7882e8cb56a477'
TRANSPARENT_BYTES = 61920

PHASE_CONTRACT = SCRIPTS / 'v485_v482_v169_cache_determinism_scope_repair_contract.json'
PHASE_CONTRACT_SHA = '8c4a583aa19de6927cc1ef787920b65317893e27ad01371d0b95b622f260cf64'
PHASE_CONTRACT_BYTES = 43960
PHASE_PREREG = J / 'v485_v169_cache_qualification_prereg_seed1627_20260824/preregistration.json'
PHASE_PREREG_SHA = 'b73fad8071d8df0b1bed3dc53212bf8a86b2350911620feac81d96a1505492e0'
PHASE_PREREG_BYTES = 55761
PHASE_STATIC = TRANSPARENT
PHASE_STATIC_SHA = TRANSPARENT_SHA
PHASE_STATIC_BYTES = TRANSPARENT_BYTES

CONTRACT_FORMAT = 'strict-track2-v507-v506-phase-a-cache-qualification-execution-authority-design-contract-v1'
CONTRACT_STATUS = 'design_only_frozen_sources_pending_independent_review_no_authority'
OUTPUT_FORMAT = 'strict-track2-v507-v506-phase-a-cache-qualification-execution-authority-v1'
OUTPUT_STATUS = 'authorized_exact_one_external_v507_phase_a_cache_qualification_attempt'

CONTRACT_SOURCE_ORDER = [
    'authority_materializer', 'fresh_phase_a_launcher', 'frozen_v485_launcher',
    'phase_a_design_contract', 'phase_a_preregistration', 'phase_a_static_audit',
    'cache_scope_helper', 'phase_a_worker', 'phase_a_driver',
    'phase_a_independent_auditor', 'phase_a_output_materializer',
    'phase_a_static_auditor', 'restart_v218_source',
]
AUTHORITY_SOURCE_ORDER = ['authority_design_contract', *CONTRACT_SOURCE_ORDER]
SOURCE_ALIASES = {'authority_design_contract': 'authority_design_contract', **{
    role: role + '_source' for role in CONTRACT_SOURCE_ORDER
}}

AUTHORIZATION = {
    'phase_a_cache_qualification_launcher_authorized': True,
    'launcher_invocations_authorized': 1, 'launcher_invocations_consumed': 0,
    'nested_phase_a_driver_invocations_authorized': 1,
    'nested_phase_a_driver_only_via_launcher': True,
    'direct_phase_a_driver_authorized': False, 'retry_authorized': False,
    'cache_reuse_authorized': False, 'phase_b_preregistration_authorized': False,
    'training_authorized': False, 'folds_authorized': 0, 'policy_updates': 0,
    's1_authorized': False, 'zero_update_authorized': False, 'rl_authorized': False,
    'submission_authorized': False, 'reward_read_authorized': False,
    'dev_hidden_final_outcome_read_authorized': False,
}
RUNTIME = {
    'execution_authority_materialized': True, 'phase_a_launcher_executed': False,
    'phase_a_driver_executed': False, 'qualification_output_created': False,
    'cache_reused': False, 'training_launched': False, 'folds': 0,
    'policy_updates': 0, 'reward_read': False, 'dev_hidden_final_outcome_read': False,
}
EXECUTION_BOUNDARY = {
    'authority_materialization_only': True, 'phase_a_launcher_invocations': 0,
    'phase_a_driver_invocations': 0, 'training_invocations': 0,
    'cache_reuse_invocations': 0, 'reward_reads': 0,
    'dev_hidden_final_outcome_reads': 0,
}

CONTRACT_TOP_KEYS = {
    'format', 'status', 'seed', 'lineage', 'source_closure', 'source_role_order',
    'source_aliases', 'source_closure_sha256', 'authority_materializer_source',
    'phase_a_launcher_source', 'phase_a_design_contract_record',
    'phase_a_preregistration_record', 'phase_a_static_audit_record',
    'v506_authority_receipt', 'v506_authority_registration_tree',
    'v506_materialization_evidence_tree', 'v506_materialization_process_receipt',
    'v506_reconciliation_attempt_tree', 'v506_reconciliation_terminal_receipt',
    'transparent_static_receipt', 'f813_registration_tree', 'qualification_output_root',
    'historical_absences', 'current_absences_after_authority',
    'authority_receipt_contract', 'authorization', 'runtime_observation',
    'execution_boundary',
}
CHECK_KEYS = sorted({
    'authority_contract_current', 'authority_materializer_current',
    'source_closure_current', 'source_order_exact', 'source_aliases_exact',
    'phase_a_preregistration_exact', 'phase_a_preregistration_nonauthorizing',
    'phase_a_static_exact', 'phase_a_static_pass_nonauthorizing',
    'old_exact7_sources_exact', 'v506_authority_exact1', 'v506_authority_schema',
    'v506_materialization_evidence_exact6', 'v506_process_partition',
    'v506_attempt_exact4', 'v506_terminal_pass', 'transparent_pass_check47',
    'f813_exact3', 'qualification_root_absent', 'historical_absences',
    'current_absences', 'input_snapshots_equal', 'no_live_phase_a_process',
    'gpu_empty', 'authorization_boundary', 'execution_boundary',
    'no_pending_values',
})
AUTH_TOP_KEYS = {
    'format', 'status', 'passed', 'source_closure', 'source_role_order',
    'source_aliases', 'source_closure_sha256', *SOURCE_ALIASES.values(),
    'v506_authority_receipt', 'v506_authority_registration_tree',
    'v506_materialization_evidence_tree', 'v506_materialization_process_receipt',
    'v506_reconciliation_attempt_tree', 'v506_reconciliation_terminal_receipt',
    'transparent_static_receipt', 'f813_registration_tree',
    'qualification_output_root', 'phase_a_attempt_root', 'historical_absences',
    'required_absences', 'checks', 'check_keys', 'check_key_set_sha256',
    'checks_sha256', 'input_pre_snapshot', 'input_post_snapshot',
    'input_snapshots_exactly_equal', 'authorization', 'runtime_observation',
    'execution_boundary', 'phase_a_cache_qualification_authorized',
    'training_authorized', 'preregistration_sha256', 'contract_sha256',
}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def csha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def regular(path: Path, digest: str | None = None, logical_bytes: int | None = None) -> dict:
    metadata = os.lstat(path)
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError('nonregular: ' + str(path))
    row = {'path': str(path), 'sha256': sha(path), 'logical_bytes': metadata.st_size}
    if digest is not None and row['sha256'] != digest:
        raise RuntimeError('sha: ' + str(path))
    if logical_bytes is not None and row['logical_bytes'] != logical_bytes:
        raise RuntimeError('bytes: ' + str(path))
    return row

def load_base():
    regular(BASE_PATH, BASE_SHA, BASE_BYTES)
    spec = importlib.util.spec_from_file_location('v503_frozen_base_for_v507', BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError('base spec')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def configure_base(base) -> None:
    base.CONTRACT_PATH = CONTRACT_PATH; base.MATERIALIZER_PATH = SELF_PATH
    base.WRAPPER_PATH = LAUNCHER_PATH; base.AUTHORITY_ROOT = AUTHORITY_ROOT
    base.WRAPPER_ATTEMPT_ROOT = ATTEMPT_ROOT; base.CONTRACT_FORMAT = CONTRACT_FORMAT
    base.CONTRACT_STATUS = CONTRACT_STATUS; base.OUTPUT_FORMAT = OUTPUT_FORMAT
    base.OUTPUT_STATUS = OUTPUT_STATUS; base.AUTHORIZATION = AUTHORIZATION
    base.RUNTIME = RUNTIME; base.EXECUTION_BOUNDARY = EXECUTION_BOUNDARY
    base.CONTRACT_SOURCE_ORDER = CONTRACT_SOURCE_ORDER
    base.AUTHORITY_SOURCE_ORDER = AUTHORITY_SOURCE_ORDER; base.SOURCE_ALIASES = SOURCE_ALIASES
    base.CONTRACT_TOP_KEYS = CONTRACT_TOP_KEYS; base.AUTH_TOP_KEYS = AUTH_TOP_KEYS
    base.CHECK_KEYS = CHECK_KEYS

def expected_fixed_sources() -> dict:
    return {
        'frozen_v485_launcher': regular(SCRIPTS/'launch_v485_v169_cache_qualification.sh', 'f596a0411c9b0db03812440ab1ece036b27489e8f6bd978a443d1a2a9ae8587e', 23827),
        'phase_a_design_contract': regular(PHASE_CONTRACT, PHASE_CONTRACT_SHA, PHASE_CONTRACT_BYTES),
        'phase_a_preregistration': regular(PHASE_PREREG, PHASE_PREREG_SHA, PHASE_PREREG_BYTES),
        'phase_a_static_audit': regular(PHASE_STATIC, PHASE_STATIC_SHA, PHASE_STATIC_BYTES),
        'cache_scope_helper': regular(SCRIPTS/'v485_v169_cache_scope.py', 'ccf6d0a5bd5273a12f946e994a9aa615ae613116d21a9ce196032b10629d11f5', 4519),
        'phase_a_worker': regular(SCRIPTS/'generate_v485_v169_cache_qualification_worker.py', 'f3c440c09e4cede17b5825c2117da7b39ee62c353c3538dd6e5c518f4da74cba', 31353),
        'phase_a_driver': regular(SCRIPTS/'generate_v485_v169_cache_qualification.py', '9199d8489fe6aea732365552a2cdbe60f24408dc383bdbd31d4148f6a0931d0a', 28333),
        'phase_a_independent_auditor': regular(SCRIPTS/'audit_v485_v169_cache_qualification.py', 'e890c84b4a9086890a18a9b82f3dc9bab546ad7f43ea27e392efcf6862782737', 42699),
        'phase_a_output_materializer': regular(SCRIPTS/'materialize_v485_v169_cache_qualification_preregistration.py', '4c31832a5b1436c8efda258cc41dde6b37f169debcf7d1c3ee2a98489c1e9f4b', 16459),
        'phase_a_static_auditor': regular(SCRIPTS/'audit_v485_v169_cache_qualification_static.py', 'd4cadd94141378f0cf89892847894abefc91357304f44d470698221a6613fc5f', 29437),
        'restart_v218_source': regular(SCRIPTS/'restart_v218_services.sh', 'e54980c82266af22668a46cc9ee80b39ece66af53cc6d64d44621808807d51af', 2794),
    }

def no_live_phase_a_process() -> bool:
    entrypoints = {str(LAUNCHER_PATH), str(SCRIPTS/'generate_v485_v169_cache_qualification.py'), str(SCRIPTS/'generate_v485_v169_cache_qualification_worker.py')}
    for process in Path('/proc').iterdir():
        if not process.name.isdigit(): continue
        try: argv = [x.decode('utf-8', 'replace') for x in (process/'cmdline').read_bytes().split(b'\0') if x]
        except (FileNotFoundError, PermissionError, ProcessLookupError): continue
        if any(item in entrypoints for item in argv): return False
    return True

def decode_absences(base, contract: dict):
    historical_keys = {'authority_root','authority_prep','phase_a_attempt_root','phase_a_attempt_prep','qualification_output_root'}
    current_keys = historical_keys - {'authority_root'}
    historical = base.decode_absences(contract['historical_absences'], historical_keys)
    current = base.decode_absences(contract['current_absences_after_authority'], current_keys)
    expected = {
        'authority_root': AUTHORITY_ROOT,
        'authority_prep': AUTHORITY_ROOT.with_name(AUTHORITY_ROOT.name+'.registration-prep'),
        'phase_a_attempt_root': ATTEMPT_ROOT,
        'phase_a_attempt_prep': ATTEMPT_ROOT.with_name(ATTEMPT_ROOT.name+'.attempt-prep'),
        'qualification_output_root': QUALIFICATION_ROOT,
    }
    if historical != expected or current != {k:v for k,v in expected.items() if k!='authority_root'}:
        raise RuntimeError('absence declarations')
    if any(os.path.lexists(path) for path in historical.values()):
        raise RuntimeError('fresh prestate')
    return historical, current

def validate_context(base, args):
    if args.contract != CONTRACT_PATH or args.materializer_source != SELF_PATH or args.authority_root != AUTHORITY_ROOT:
        raise RuntimeError('canonical CLI')
    contract_record = regular(args.contract, args.contract_sha, args.contract_bytes)
    materializer_record = regular(args.materializer_source, args.materializer_sha, args.materializer_bytes)
    contract = json.loads(args.contract.read_text())
    if set(contract) != CONTRACT_TOP_KEYS or contract['format'] != CONTRACT_FORMAT or contract['status'] != CONTRACT_STATUS or contract['seed'] != 1648:
        raise RuntimeError('contract schema')
    sources = contract['source_closure']
    if (contract['source_role_order'] != CONTRACT_SOURCE_ORDER or set(sources) != set(CONTRACT_SOURCE_ORDER)
            or contract['source_aliases'] != {r:SOURCE_ALIASES[r] for r in CONTRACT_SOURCE_ORDER}
            or contract['source_closure_sha256'] != csha(sources)):
        raise RuntimeError('source schema')
    observed = {role: regular(Path(row['path']), row['sha256'], row['logical_bytes']) for role,row in sources.items()}
    if observed != sources or observed['authority_materializer'] != materializer_record or observed['fresh_phase_a_launcher']['path'] != str(LAUNCHER_PATH):
        raise RuntimeError('source drift')
    fixed = expected_fixed_sources()
    if any(observed[role] != row for role,row in fixed.items()): raise RuntimeError('fixed source')
    expected_lineage = {
        'name':'v507_v506_phase_a_cache_qualification',
        'fresh_authority_root':str(AUTHORITY_ROOT),
        'fresh_phase_a_attempt_root':str(ATTEMPT_ROOT),
        'binds_v506_reconciliation_final':True,
    }
    schema = contract.get('authority_receipt_contract',{})
    if (contract.get('lineage') != expected_lineage
            or contract.get('authority_materializer_source') != materializer_record
            or contract.get('phase_a_launcher_source') != observed['fresh_phase_a_launcher']
            or contract.get('phase_a_design_contract_record') != fixed['phase_a_design_contract']
            or contract.get('phase_a_preregistration_record') != fixed['phase_a_preregistration']
            or contract.get('phase_a_static_audit_record') != fixed['phase_a_static_audit']
            or contract.get('authorization') != AUTHORIZATION or contract.get('runtime_observation') != RUNTIME
            or contract.get('execution_boundary') != EXECUTION_BOUNDARY
            or set(schema) != {'format','status','top_keys','check_keys','check_key_set_sha256','checks_sha256','authorization_exact','runtime_observation_exact'}
            or schema.get('format') != OUTPUT_FORMAT or schema.get('status') != OUTPUT_STATUS
            or schema.get('top_keys') != sorted(AUTH_TOP_KEYS) or schema.get('check_keys') != CHECK_KEYS
            or schema.get('check_key_set_sha256') != csha(CHECK_KEYS)
            or schema.get('checks_sha256') != csha({k:True for k in CHECK_KEYS})
            or schema.get('authorization_exact') != AUTHORIZATION or schema.get('runtime_observation_exact') != RUNTIME):
        raise RuntimeError('contract aliases/boundary')

    prereg = json.loads(PHASE_PREREG.read_text()); static_audit = json.loads(PHASE_STATIC.read_text())
    exact7 = prereg['execution_sources']
    expected7 = {
        'phase_a_launcher': fixed['frozen_v485_launcher'], 'phase_a_cache_scope_helper': fixed['cache_scope_helper'],
        'phase_a_process_worker': fixed['phase_a_worker'], 'phase_a_driver': fixed['phase_a_driver'],
        'phase_a_independent_auditor': fixed['phase_a_independent_auditor'],
        'phase_a_materializer': fixed['phase_a_output_materializer'], 'phase_a_static_auditor': fixed['phase_a_static_auditor'],
    }
    if exact7 != expected7 or prereg.get('execution_sources_digest_sha256') != '9adc5bdbfaa0022b745c44abac2ac2bd02f801f751ebce35d0116c0a8e8e53e0': raise RuntimeError('old exact7')
    if (prereg.get('authorization',{}).get('phase_a_cache_qualification_authorized') is not False
            or prereg.get('runtime_observation') != {'phase_a_executed':False,'training_launched':False,'folds':0,'policy_updates':0}
            or prereg.get('qualification_output_root') != str(QUALIFICATION_ROOT)):
        raise RuntimeError('prereg boundary')
    if (static_audit.get('passed') is not True or static_audit.get('status') != 'passed_no_execution_authority'
            or len(static_audit.get('checks',{})) != 47 or not all(v is True for v in static_audit['checks'].values())
            or static_audit.get('phase_a_cache_qualification_authorized') is not False):
        raise RuntimeError('static boundary')

    auth_record = regular(V506_AUTH_RECEIPT, V506_AUTH_SHA, V506_AUTH_BYTES); auth_tree = base.tree(V506_AUTH_ROOT)
    evidence_tree = base.tree(V506_EVIDENCE); process_record = regular(V506_PROCESS,V506_PROCESS_SHA,V506_PROCESS_BYTES)
    attempt_tree = base.tree(V506_ATTEMPT); terminal_record = regular(V506_TERMINAL,V506_TERMINAL_SHA,V506_TERMINAL_BYTES)
    transparent_record = regular(TRANSPARENT,TRANSPARENT_SHA,TRANSPARENT_BYTES); f813_tree = base.tree(F813_ROOT)
    if auth_tree['inventory'] != [['authority_receipt.json',V506_AUTH_SHA,V506_AUTH_BYTES]]: raise RuntimeError('v506 auth tree')
    if evidence_tree['file_count'] != 6 or process_record['sha256'] != V506_PROCESS_SHA: raise RuntimeError('v506 evidence')
    if attempt_tree['file_count'] != 4 or terminal_record['sha256'] != V506_TERMINAL_SHA: raise RuntimeError('v506 attempt')
    terminal = json.loads(V506_TERMINAL.read_text()); transparent = json.loads(TRANSPARENT.read_text())
    if (terminal.get('passed') is not True or terminal.get('nested_r2_invocations') != 1
            or terminal.get('reconciler_exit_code') != 0 or terminal.get('retry_authorized') is not False
            or terminal.get('immutable_inputs_exactly_equal') is not True or terminal.get('f813_exact2_to_sole_exact3') is not True
            or any(terminal.get(k) not in (False,0) for k in ('phase_a_executed','training_launched','reward_read','dev_hidden_final_outcome_read','policy_updates','folds'))):
        raise RuntimeError('v506 terminal')
    if transparent.get('passed') is not True or len(transparent.get('checks',{})) != 47 or not all(v is True for v in transparent['checks'].values()): raise RuntimeError('transparent')
    expected_f813 = sorted([
        ['immutable_evidence/v485_static_b73.log','7a3e5aa7b627ff92ec82cb7073cac70dea35e26774bda76ef302c547a3ab070b',6475],
        ['preregistration.json','f8137d02a692a9c6243f13ce8a7674e4ac90fbd31c85b3e8c32af82f8afa42a8',21296],
        ['transparent_static_audit.json',TRANSPARENT_SHA,TRANSPARENT_BYTES],
    ])
    if f813_tree['inventory'] != expected_f813: raise RuntimeError('F813 exact3')
    aliases = {
        'v506_authority_receipt': auth_record, 'v506_authority_registration_tree': auth_tree,
        'v506_materialization_evidence_tree': evidence_tree, 'v506_materialization_process_receipt': process_record,
        'v506_reconciliation_attempt_tree': attempt_tree, 'v506_reconciliation_terminal_receipt': terminal_record,
        'transparent_static_receipt': transparent_record, 'f813_registration_tree': f813_tree,
    }
    if any(contract.get(k) != v for k,v in aliases.items()): raise RuntimeError('lineage alias')
    if contract['qualification_output_root'] != str(QUALIFICATION_ROOT): raise RuntimeError('qualification path')
    historical,current = decode_absences(base,contract)
    files = {'contract':str(CONTRACT_PATH),'materializer':str(SELF_PATH),**{('source_'+k):v['path'] for k,v in sources.items()},
             'v506_authority':str(V506_AUTH_RECEIPT),'v506_process':str(V506_PROCESS),'v506_terminal':str(V506_TERMINAL),'transparent':str(TRANSPARENT)}
    trees = {'v506_authority':str(V506_AUTH_ROOT),'v506_evidence':str(V506_EVIDENCE),'v506_attempt':str(V506_ATTEMPT),'f813':str(F813_ROOT)}
    pre = base.snapshot(files,trees,current)
    return contract,contract_record,materializer_record,sources,aliases,historical,current,files,trees,pre

def main() -> int:
    base=load_base(); configure_base(base)
    if os.sys.argv[1:]==['--synthetic-self-test']:
        checks={'contract_top_count':len(CONTRACT_TOP_KEYS)==28,'authority_top_nonempty':len(AUTH_TOP_KEYS)>35,
                'check_count':len(CHECK_KEYS)==27,'contract_source_count':len(CONTRACT_SOURCE_ORDER)==13,
                'authority_source_count':len(AUTHORITY_SOURCE_ORDER)==14,'source_alias_bijection':len(set(SOURCE_ALIASES.values()))==14,
                'authorization_boundary':AUTHORIZATION['phase_a_cache_qualification_launcher_authorized'] is True and AUTHORIZATION['direct_phase_a_driver_authorized'] is False and AUTHORIZATION['training_authorized'] is False,
                'fresh_roots':'v507_' in AUTHORITY_ROOT.name and 'v507_' in ATTEMPT_ROOT.name,'publication_noreplace_owned_cleanup':base.publication_self_test()}
        print(json.dumps({'passed':all(checks.values()),'checks':checks,'checks_sha256':csha(checks)},sort_keys=True)); return 0 if all(checks.values()) else 1
    parser=argparse.ArgumentParser(); parser.add_argument('--contract',type=Path,required=True); parser.add_argument('--contract-sha',required=True); parser.add_argument('--contract-bytes',type=int,required=True); parser.add_argument('--materializer-source',type=Path,required=True); parser.add_argument('--materializer-sha',required=True); parser.add_argument('--materializer-bytes',type=int,required=True); parser.add_argument('--authority-root',type=Path,required=True)
    args=parser.parse_args(); (contract,contract_record,materializer_record,sources,aliases,historical,current,files,trees,pre)=validate_context(base,args)
    checks={key:True for key in CHECK_KEYS}; checks.update({'no_live_phase_a_process':no_live_phase_a_process(),'gpu_empty':base.gpu_empty(),'no_pending_values':not base.pending(contract)})
    if not all(checks.values()): raise RuntimeError('checks')
    authority_sources={'authority_design_contract':contract_record,**sources}
    receipt={'format':OUTPUT_FORMAT,'status':OUTPUT_STATUS,'passed':True,'source_closure':authority_sources,'source_role_order':AUTHORITY_SOURCE_ORDER,'source_aliases':SOURCE_ALIASES,'source_closure_sha256':csha(authority_sources),
             **{SOURCE_ALIASES[k]:v for k,v in authority_sources.items()},**aliases,
             'qualification_output_root':str(QUALIFICATION_ROOT),'phase_a_attempt_root':str(ATTEMPT_ROOT),
             'phase_a_cache_qualification_authorized':True,'training_authorized':False,
             'preregistration_sha256':PHASE_PREREG_SHA,'contract_sha256':PHASE_CONTRACT_SHA,
             'historical_absences':{k:{'path':str(v),'absent':True} for k,v in historical.items()},'required_absences':{k:{'path':str(v),'absent':True} for k,v in current.items()},
             'checks':checks,'check_keys':CHECK_KEYS,'check_key_set_sha256':csha(CHECK_KEYS),'checks_sha256':csha(checks),
             'input_pre_snapshot':pre,'input_post_snapshot':base.snapshot(files,trees,current),'input_snapshots_exactly_equal':True,
             'authorization':AUTHORIZATION,'runtime_observation':RUNTIME,'execution_boundary':EXECUTION_BOUNDARY}
    if receipt['input_pre_snapshot']!=receipt['input_post_snapshot'] or set(receipt)!=AUTH_TOP_KEYS: raise RuntimeError('receipt schema/snapshot')
    base.publish_exact1(AUTHORITY_ROOT,receipt,files,trees,current,pre); return 0

if __name__=='__main__': raise SystemExit(main())
