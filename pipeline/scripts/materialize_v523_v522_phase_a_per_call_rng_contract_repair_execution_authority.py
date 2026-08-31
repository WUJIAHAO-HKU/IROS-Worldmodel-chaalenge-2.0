#!/usr/bin/env python3
"""Materialize the fresh v523 authority after the frozen v522 pre-worker failure."""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path('/root/autodl-tmp/IROS_WAM_2.0 challenge')
S = ROOT / 'pipeline/scripts'
J = ROOT / 'artifacts/strict_track2_joint_augmentation_20260810'
PIPELINE = ROOT / 'pipeline'

SELF_PATH = S / 'materialize_v523_v522_phase_a_per_call_rng_contract_repair_execution_authority.py'
CONTRACT_PATH = S / 'v523_v522_phase_a_per_call_rng_contract_repair_execution_authority_contract.json'
LAUNCHER_PATH = S / 'launch_v523_v522_phase_a_cache_qualification.py'
DRIVER_PATH = S / 'generate_v523_v522_phase_a_cache_qualification.py'
WORKER_PATH = S / 'generate_v523_v522_phase_a_cache_qualification_worker.py'
PROXY_PATH = S / 'v520_v519_rng_isolated_v169_runtime_proxy.py'
AUDITOR_PATH = S / 'audit_v523_v522_phase_a_rng_isolated_qualification.py'
PREREG_ROOT = J / 'v523_v522_phase_a_cache_qualification_prereg_seed1659_20260826'
PREREG_PATH = PREREG_ROOT / 'preregistration.json'
AUTHORITY_ROOT = J / 'v523_v522_phase_a_per_call_rng_contract_repair_execution_authority_seed1659_20260826'
ATTEMPT_ROOT = J / 'v523_v522_phase_a_cache_qualification_attempt_seed1659_20260826'
QUALIFICATION_ROOT = Path('/root/v523_v522_phase_a_cache_qualification_seed1659_20260826')

OLD_CONTRACT = S / 'v522_v521_phase_a_authority_split_state_repair_execution_authority_contract.json'
OLD_CONTRACT_SHA = '0f766d4c6f95e81c0e2278cc19ba8124b6f4df4eb0e3ddf0376e8b89280c7610'
OLD_CONTRACT_BYTES = 111259
OLD_MATERIALIZER = S / 'materialize_v522_v521_phase_a_authority_split_state_repair_execution_authority.py'
OLD_MATERIALIZER_SHA = '42f2375eb3b468798d2131edddbefab7e579bdc6662e3743dbb0fb3b799ddca5'
OLD_MATERIALIZER_BYTES = 53649
OLD_AUTHORITY_ROOT = J / 'v522_v521_phase_a_authority_split_state_repair_execution_authority_seed1658_20260826'
OLD_AUTHORITY_RECEIPT = OLD_AUTHORITY_ROOT / 'authority_receipt.json'
OLD_AUTHORITY_RECEIPT_SHA = '04cc334e264c8a55b7d06ca620aeef4d4f46c94a39e64b93f8a414b674851afc'
OLD_AUTHORITY_RECEIPT_BYTES = 161260
OLD_AUTHORITY_TREE = {
    'root': str(OLD_AUTHORITY_ROOT),
    'inventory': [['authority_receipt.json', OLD_AUTHORITY_RECEIPT_SHA, OLD_AUTHORITY_RECEIPT_BYTES]],
    'file_count': 1, 'logical_file_bytes': OLD_AUTHORITY_RECEIPT_BYTES,
    'sha256sum_lines_digest_sha256': 'de7c2b0dccc1f605f0321e5be64ae7ac2c85f59bb202b01b1bb8e74d45bb8836',
    'canonical_json_triples_digest_sha256': '1e654d46b8c4287105feb2e308e4dc4ea23da6daedfa6b0fd324017edf6b0e2d',
}
OLD_LAUNCHER = S / 'launch_v522_v521_phase_a_cache_qualification.py'
OLD_LAUNCHER_SHA = '289204d6b7b7593de390f8600c5cffb6edb496b357d45ec5216a446f586740f4'
OLD_LAUNCHER_BYTES = 77590
OLD_DRIVER = S / 'generate_v522_v521_phase_a_cache_qualification.py'
OLD_DRIVER_SHA = '8f4e31da630272586d1fb08a067072734c19c97901c1b8fa25ae2bfd78820320'
OLD_DRIVER_BYTES = 45542
OLD_WORKER = S / 'generate_v522_v521_phase_a_cache_qualification_worker.py'
OLD_WORKER_SHA = '3f4376a1aa093b578cf4d6f7ba02b7a85460c7ea4c86f162c96b2d222eaac231'
OLD_WORKER_BYTES = 38774
OLD_AUDITOR = S / 'audit_v522_v521_phase_a_rng_isolated_qualification.py'
OLD_AUDITOR_SHA = '3b60f0134fb474422bf792666474bff16b6456273385cfcd24b15e4c685e1b2e'
OLD_AUDITOR_BYTES = 45074
OLD_ATTEMPT_ROOT = J / 'v522_v521_phase_a_cache_qualification_attempt_seed1658_20260826'
OLD_QUALIFICATION_ROOT = Path('/root/v522_v521_phase_a_cache_qualification_seed1658_20260826')
OLD_INTENT = OLD_ATTEMPT_ROOT / 'intent.json'
OLD_STDOUT = OLD_ATTEMPT_ROOT / 'phase_a_stdout.log'
OLD_STDERR = OLD_ATTEMPT_ROOT / 'phase_a_stderr.log'
OLD_TERMINAL = OLD_ATTEMPT_ROOT / 'terminal_receipt.json'
OLD_ATTEMPT_TREE = {
    'root': str(OLD_ATTEMPT_ROOT),
    'inventory': [
        ['intent.json', '91d88f22afc173668efa99105250cb61f282dba0c539af9d290cd3488bab67b8', 5599],
        ['phase_a_stderr.log', '7b93005b15a380978579c2972f9d417642943be44280370c75d2e97135906f57', 871],
        ['phase_a_stdout.log', hashlib.sha256(b'').hexdigest(), 0],
        ['terminal_receipt.json', 'bcccd00812220679eda01419e775bb48147e917f22b90fd46a160120d9bde870', 69178],
    ],
    'file_count': 4, 'logical_file_bytes': 75648,
    'sha256sum_lines_digest_sha256': '3017c094b9b5067bf134143361fe664e04ece28306855db8c0c671cfb452033b',
    'canonical_json_triples_digest_sha256': '3fd62a21c9b9de6ea652f8fd42cf448bf39948e55f4f61a3d9b48c0a78e8d433',
}
FAILURE_FORENSIC = S / 'v523_v522_phase_a_per_call_rng_contract_failure_forensic.json'
FAILURE_FORENSIC_SHA = '94e06c8d9dcd31e3026e069dafe8b5bb6adbda85e51fd5e1e448f8dd4e6fab0d'
FAILURE_FORENSIC_BYTES = 5284

DRIVER_SHA = 'ee7a2a327cc7fc8f282eef9b97701860c310901cdf383d8bc923b7f276c4f3ab'
DRIVER_BYTES = 47276
WORKER_SHA = '89a552416c054fa2590d7f15b5fb829b8eb2cf8a02479e2115b825bf6e49d627'
WORKER_BYTES = 38990
PROXY_SHA = '2b8ab0e2aec4a8de5d4bf8cffbb3525bbcd69b9b03270012a4a48f69f430b51b'
PROXY_BYTES = 8926
AUDITOR_SHA = 'f09a17f068f9f6f471be7bba10162cab406d18957d48586eed083e25d6f59ea2'
AUDITOR_BYTES = 47453
PREREG_SHA = 'b813d87237a2a14e1b317da61497e5b563ebef2a6f0770ce25f023be66296ea2'
PREREG_BYTES = 57451
PHASE_A_DESIGN_CONTRACT_SHA = '8c4a583aa19de6927cc1ef787920b65317893e27ad01371d0b95b622f260cf64'

FIXED_ROLES = [
    'phase_a_design_contract', 'phase_a_output_materializer', 'cache_scope_helper',
    'phase_a_static_auditor', 'phase_a_static_audit', 'restart_v218_source', 'frozen_v485_launcher',
]
ACTIVE_ROLES = [
    'authority_materializer', 'fresh_phase_a_launcher', 'fresh_phase_a_driver',
    'fresh_phase_a_worker', 'fresh_rng_proxy', 'fresh_phase_a_independent_auditor',
    'fresh_phase_a_preregistration',
]
ANCESTRY_ROLES = [
    'v523_v522_phase_a_failure_forensic', 'v522_authority_design_contract',
    'v522_authority_materializer', 'v522_authority_receipt', 'v522_phase_a_launcher',
    'v522_phase_a_driver', 'v522_phase_a_worker', 'v522_phase_a_independent_auditor',
    'v522_attempt_intent', 'v522_attempt_stdout', 'v522_attempt_stderr', 'v522_attempt_terminal',
]
CONTRACT_SOURCE_ORDER = [*ACTIVE_ROLES, *ANCESTRY_ROLES, *FIXED_ROLES]
AUTHORITY_SOURCE_ORDER = ['authority_design_contract', *CONTRACT_SOURCE_ORDER]
SOURCE_ALIASES = {'authority_design_contract': 'authority_design_contract', **{role: role + '_source' for role in CONTRACT_SOURCE_ORDER}}

WORKER_ENV = {'PYTHONPATH': str(PIPELINE), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
PER_CALL_RNG_CONTRACT = {
    'format': 'strict-track2-v520-per-call-rng-isolation-evidence-v1',
    'expected_branches': ['A', 'B'], 'calls_per_branch': 1000,
    'total_runtime_delegate_calls': 2000,
    'delegate_invocations_started_per_call': 1,
    'delegate_invocations_completed_per_successful_call': 1,
    'required_rng_stage_keys': ['entry_external', 'inside_before_delegate', 'internal_after_delegate', 'exit_restored'],
    'fork_rng_devices_exact_all_cuda_indices': True,
    'torch_internal_change_allowed': True, 'torch_cpu_exit_restored': True,
    'torch_cuda_exit_restored': True, 'python_all_four_stages_equal': True,
    'numpy_all_four_stages_equal': True, 'raw_warning_order_preserved': True,
    'output_schema_recorded_per_call': True, 'canonical_event_digest_recorded_per_call': True,
    'sample0_evidence_is_ancestry_not_runtime_substitute': True,
    'durable_process_log_markers_per_branch': 2000,
    'durable_process_log_markers_per_call': 2,
    'process_log_marker_order_per_call': ['started', 'completed'],
    'total_durable_process_log_markers': 4000,
}
AUTHORIZATION = {
    'phase_a_cache_qualification_launcher_authorized': True,
    'launcher_invocations_authorized': 1, 'launcher_invocations_consumed': 0,
    'nested_phase_a_driver_invocations_authorized': 1,
    'nested_phase_a_driver_only_via_launcher': True,
    'direct_phase_a_driver_authorized': False,
    'phase_a_worker_invocations_authorized': 2,
    'rng_proxy_required_for_every_runtime_delegate': True,
    'runtime_delegate_calls_authorized': 2000, 'runtime_delegate_calls_consumed': 0,
    'worker_environment_exact': WORKER_ENV,
    'service_environment_overrides_authorized': False, 'retry_authorized': False,
    'cache_reuse_authorized': False, 'training_authorized': False,
    'reward_read_authorized': False, 'dev_hidden_final_outcome_read_authorized': False,
    'submission_authorized': False,
}
RUNTIME = {
    'execution_authority_materialized': True, 'phase_a_launcher_executed': False,
    'phase_a_driver_executed': False, 'phase_a_worker_invocations': 0,
    'rng_proxy_delegate_invocations': 0, 'qualification_output_created': False,
    'cache_reused': False, 'training_launched': False, 'reward_read': False,
    'dev_hidden_final_outcome_read': False,
}
EXECUTION_BOUNDARY = {
    'authority_materialization_only': True, 'phase_a_launcher_invocations': 0,
    'phase_a_driver_invocations': 0, 'phase_a_worker_invocations': 0,
    'rng_proxy_delegate_invocations': 0, 'training_invocations': 0,
    'reward_reads': 0, 'dev_hidden_final_outcome_reads': 0,
}
CONTRACT_FORMAT = 'strict-track2-v523-v522-phase-a-per-call-rng-contract-repair-execution-authority-design-contract-v1'
CONTRACT_STATUS = 'design_only_frozen_sources_pending_independent_review_no_authority'
OUTPUT_FORMAT = 'strict-track2-v523-v522-phase-a-per-call-rng-contract-repair-execution-authority-v1'
OUTPUT_STATUS = 'authorized_exact_one_external_v523_phase_a_cache_qualification_attempt'

CHECK_KEYS = sorted({
    'contract_schema', 'source_order', 'source_aliases', 'source_closure_current',
    'active_paths_exact', 'fixed_sources_exact', 'prereg_exact8', 'worker_env_exact2',
    'per_call_contract_exact21', 'driver_per_call_exact21', 'worker_per_call_exact21',
    'auditor_per_call_exact21', 'old_authority_exact1', 'old_authority_schema',
    'old_authority_sources_current', 'old_authority_expected_current',
    'v522_attempt_exact4', 'v522_terminal_failed_no_retry', 'v522_partition_exact',
    'v522_root_cause_exact4', 'v522_forensic_exact', 'v522_cleanup_closed',
    'v522_services_restored', 'v522_qualification_absent', 'fresh_roots_absent',
    'authorization_boundary', 'execution_boundary', 'runtime_observation',
    'input_prepost_equal', 'gpu_empty', 'no_live_phase_a_process',
    'no_pending_values', 'publish_noreplace_exact1',
})

CONTRACT_TOP_KEYS = {
    'format', 'status', 'seed', 'lineage', 'source_role_order', 'source_aliases',
    'source_closure', 'source_closure_sha256', 'authority_materializer_source',
    'phase_a_launcher_source', 'phase_a_driver_source', 'phase_a_worker_source',
    'rng_proxy_source', 'phase_a_independent_auditor_source',
    'phase_a_preregistration_source', 'qualification_output_root',
    'worker_environment_exact', 'per_call_rng_evidence_contract',
    'v522_authority_registration_tree', 'v522_authority_receipt',
    'v522_authority_semantics', 'v522_attempt_tree', 'v522_attempt_intent',
    'v522_attempt_stdout', 'v522_attempt_stderr', 'v522_attempt_terminal',
    'v522_phase_a_failure_forensic', 'v522_phase_a_execution_partition',
    'v522_per_call_contract_failure_root_cause', 'historical_absences',
    'current_absences_after_authority', 'authority_receipt_contract',
    'authorization', 'runtime_observation', 'execution_boundary',
}


def csha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def json_exact(left, right):
    return json.dumps(left, sort_keys=True, separators=(',', ':'), ensure_ascii=False) == json.dumps(right, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def regular(path, expected_sha=None, expected_bytes=None):
    path = Path(path); metadata = os.lstat(path)
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError('nonregular ' + str(path))
    observed = {'path': str(path), 'sha256': sha(path), 'logical_bytes': metadata.st_size}
    if expected_sha is not None and (observed['sha256'] != expected_sha or observed['logical_bytes'] != expected_bytes):
        raise RuntimeError('record mismatch ' + str(path))
    return observed


def exact_tree(root):
    root = Path(root); metadata = os.lstat(root)
    if root.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError('tree root')
    inventory = []
    for path in sorted(root.rglob('*'), key=lambda item: item.relative_to(root).as_posix()):
        child = os.lstat(path)
        if path.is_symlink() or not stat.S_ISREG(child.st_mode):
            raise RuntimeError('tree member')
        inventory.append([path.relative_to(root).as_posix(), sha(path), child.st_size])
    lines = ''.join(f'{digest}  {name}\n' for name, digest, _size in inventory).encode()
    return {'root': str(root), 'inventory': inventory, 'file_count': len(inventory),
            'logical_file_bytes': sum(row[2] for row in inventory),
            'sha256sum_lines_digest_sha256': hashlib.sha256(lines).hexdigest(),
            'canonical_json_triples_digest_sha256': csha(inventory)}


def exact_summary(observed, expected):
    return observed == expected


def service_health():
    result = {}
    for name, url in (('8005_v1_health', 'http://127.0.0.1:8005/v1/health'), ('18084_health', 'http://127.0.0.1:18084/health')):
        with urllib.request.urlopen(url, timeout=10) as response:
            body = response.read()
            result[name] = {'http_code': response.status, 'body_sha256': hashlib.sha256(body).hexdigest(), 'body_bytes': len(body), 'returncode': 0}
    return result


def phase_pids():
    markers = (LAUNCHER_PATH.name, DRIVER_PATH.name, WORKER_PATH.name)
    rows = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid(): continue
        try: command = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
        except (FileNotFoundError, ProcessLookupError, PermissionError): continue
        if any(marker in command for marker in markers): rows.append({'pid': int(entry.name), 'cmdline': command})
    return sorted(rows, key=lambda row: row['pid'])


def literal_assignment(path, name):
    syntax = ast.parse(Path(path).read_text())
    for node in syntax.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError('literal assignment ' + name)


def load_base():
    regular(OLD_MATERIALIZER, OLD_MATERIALIZER_SHA, OLD_MATERIALIZER_BYTES)
    spec = importlib.util.spec_from_file_location('v522_frozen', OLD_MATERIALIZER)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    _prior, base = old.load_base()
    base.CONTRACT_PATH = CONTRACT_PATH; base.MATERIALIZER_PATH = SELF_PATH
    base.WRAPPER_PATH = LAUNCHER_PATH; base.AUTHORITY_ROOT = AUTHORITY_ROOT
    base.WRAPPER_ATTEMPT_ROOT = ATTEMPT_ROOT
    base.CONTRACT_FORMAT = CONTRACT_FORMAT; base.CONTRACT_STATUS = CONTRACT_STATUS
    base.OUTPUT_FORMAT = OUTPUT_FORMAT; base.OUTPUT_STATUS = OUTPUT_STATUS
    base.AUTHORIZATION = AUTHORIZATION; base.RUNTIME = RUNTIME
    base.EXECUTION_BOUNDARY = EXECUTION_BOUNDARY
    base.CONTRACT_SOURCE_ORDER = CONTRACT_SOURCE_ORDER
    base.AUTHORITY_SOURCE_ORDER = AUTHORITY_SOURCE_ORDER
    base.SOURCE_ALIASES = SOURCE_ALIASES
    base.CONTRACT_TOP_KEYS = CONTRACT_TOP_KEYS; base.CHECK_KEYS = CHECK_KEYS
    return old, base


def absence_paths():
    historical = {
        'authority_root': AUTHORITY_ROOT,
        'authority_prep': AUTHORITY_ROOT.with_name(AUTHORITY_ROOT.name + '.registration-prep'),
        'phase_a_attempt_root': ATTEMPT_ROOT,
        'phase_a_attempt_prep': ATTEMPT_ROOT.with_name(ATTEMPT_ROOT.name + '.attempt-prep'),
        'qualification_output_root': QUALIFICATION_ROOT,
        'qualification_output_prep': QUALIFICATION_ROOT.with_name(QUALIFICATION_ROOT.name + '.attempt-prep'),
    }
    return historical, {key: value for key, value in historical.items() if key != 'authority_root'}


def validate_old_state():
    records = {
        'v523_v522_phase_a_failure_forensic': regular(FAILURE_FORENSIC, FAILURE_FORENSIC_SHA, FAILURE_FORENSIC_BYTES),
        'v522_authority_design_contract': regular(OLD_CONTRACT, OLD_CONTRACT_SHA, OLD_CONTRACT_BYTES),
        'v522_authority_materializer': regular(OLD_MATERIALIZER, OLD_MATERIALIZER_SHA, OLD_MATERIALIZER_BYTES),
        'v522_authority_receipt': regular(OLD_AUTHORITY_RECEIPT, OLD_AUTHORITY_RECEIPT_SHA, OLD_AUTHORITY_RECEIPT_BYTES),
        'v522_phase_a_launcher': regular(OLD_LAUNCHER, OLD_LAUNCHER_SHA, OLD_LAUNCHER_BYTES),
        'v522_phase_a_driver': regular(OLD_DRIVER, OLD_DRIVER_SHA, OLD_DRIVER_BYTES),
        'v522_phase_a_worker': regular(OLD_WORKER, OLD_WORKER_SHA, OLD_WORKER_BYTES),
        'v522_phase_a_independent_auditor': regular(OLD_AUDITOR, OLD_AUDITOR_SHA, OLD_AUDITOR_BYTES),
        'v522_attempt_intent': regular(OLD_INTENT, '91d88f22afc173668efa99105250cb61f282dba0c539af9d290cd3488bab67b8', 5599),
        'v522_attempt_stdout': regular(OLD_STDOUT, hashlib.sha256(b'').hexdigest(), 0),
        'v522_attempt_stderr': regular(OLD_STDERR, '7b93005b15a380978579c2972f9d417642943be44280370c75d2e97135906f57', 871),
        'v522_attempt_terminal': regular(OLD_TERMINAL, 'bcccd00812220679eda01419e775bb48147e917f22b90fd46a160120d9bde870', 69178),
    }
    old_contract = json.loads(OLD_CONTRACT.read_text())
    authority = json.loads(OLD_AUTHORITY_RECEIPT.read_text())
    authority_tree = exact_tree(OLD_AUTHORITY_ROOT)
    if (authority_tree != OLD_AUTHORITY_TREE or len(authority) != 88 or authority.get('passed') is not True
            or sorted(authority) != old_contract['authority_receipt_contract']['top_keys']
            or len(authority.get('checks', {})) != 41
            or not all(type(value) is bool and value is True for value in authority['checks'].values())
            or len(authority.get('source_closure', {})) != 33
            or authority.get('source_closure_sha256') != csha(authority['source_closure'])
            or authority.get('input_snapshots_exactly_equal') is not True
            or authority.get('input_pre_snapshot') != authority.get('input_post_snapshot')):
        raise RuntimeError('v522 authority')
    for row in authority['source_closure'].values():
        if regular(row['path'], row['sha256'], row['logical_bytes']) != row:
            raise RuntimeError('v522 authority source')

    attempt_tree = exact_tree(OLD_ATTEMPT_ROOT)
    terminal = json.loads(OLD_TERMINAL.read_text())
    forensic = json.loads(FAILURE_FORENSIC.read_text())
    expected_partition = {
        'launcher_invocations': 1, 'nested_phase_a_driver_invocations': 1,
        'direct_phase_a_driver_invocations': 0, 'phase_a_worker_invocations': 0,
        'rng_proxy_delegate_invocations': 0, 'training_invocations': 0,
        'retry_invocations': 0,
    }
    expected_missing = [
        'durable_process_log_markers_per_branch', 'durable_process_log_markers_per_call',
        'process_log_marker_order_per_call', 'total_durable_process_log_markers',
    ]
    if (attempt_tree != OLD_ATTEMPT_TREE or terminal.get('status') != 'failed_no_retry'
            or terminal.get('passed') is not False or terminal.get('retry_authorized') is not False
            or terminal.get('driver_returncode') != 1 or terminal.get('error') != 'Phase-A driver rc=1'
            or terminal.get('nested_phase_a_driver_invocations') != 1
            or terminal.get('runtime_observation', {}).get('phase_a_worker_invocations') != 0
            or terminal.get('runtime_observation', {}).get('rng_proxy_delegate_invocations') != 0
            or terminal.get('qualification_output_created') is not False
            or terminal.get('qualification_terminal') is not None or terminal.get('qualification_tree') is not None
            or terminal.get('driver_cleanup', {}).get('reaped') is not True
            or terminal.get('driver_cleanup', {}).get('group_empty') is not True
            or terminal.get('immutable_snapshots_exactly_equal') is not True
            or terminal.get('immutable_snapshot_before') != terminal.get('immutable_snapshot_after')
            or forensic.get('v522_attempt_tree') != attempt_tree
            or forensic.get('execution_partition') != expected_partition
            or forensic.get('root_cause', {}).get('driver_missing_authority_keys') != expected_missing
            or forensic.get('root_cause', {}).get('shared_keys_exact') is not True
            or forensic.get('root_cause', {}).get('semantic_repair_exact', {}).get('driver_per_call_contract_key_count') != 21
            or forensic.get('independent_audit') != {'check_count': 17, 'all_checks_true': True, 'checks_sha256': 'f005367f1aafc950d3a70093aa8455f49d7795d33af59182718e434cf662437c'}
            or os.path.lexists(OLD_QUALIFICATION_ROOT)
            or os.path.lexists(OLD_QUALIFICATION_ROOT.with_name(OLD_QUALIFICATION_ROOT.name + '.attempt-prep'))):
        raise RuntimeError('v522 failure')
    if service_health() != terminal['v218_health_after'] or terminal['v218_health_before'] != terminal['v218_health_after']:
        raise RuntimeError('v522 service restoration')
    if phase_pids(): raise RuntimeError('phase pids')
    partition = expected_partition
    semantics = {'passed': True, 'top_key_count': 88, 'check_count': 41, 'source_closure_count': 33,
                 'source_closure_all_current': True, 'input_snapshots_exactly_equal': True,
                 'launcher_invocations_consumed_by_v522_attempt': 1,
                 'nested_driver_invocations_consumed_by_v522_attempt': 1,
                 'historical_downstream_absence_revalidated_after_attempt': False}
    return {'records': records, 'authority_tree': authority_tree,
            'authority_receipt': records['v522_authority_receipt'], 'authority_semantics': semantics,
            'attempt_tree': attempt_tree, 'attempt_intent': records['v522_attempt_intent'],
            'attempt_stdout': records['v522_attempt_stdout'], 'attempt_stderr': records['v522_attempt_stderr'],
            'attempt_terminal': records['v522_attempt_terminal'], 'failure_forensic': forensic,
            'execution_partition': partition, 'root_cause': forensic['root_cause']}


def validate_prereg(source_records):
    value = json.loads(PREREG_PATH.read_text())
    roles = [row.get('role') for row in value.get('execution_source_records', [])]
    expected_roles = ['phase_a_preregistration_materializer', 'phase_a_driver', 'phase_a_process_worker',
                      'phase_a_rng_isolation_proxy', 'cache_scope_helper', 'phase_a_independent_auditor',
                      'phase_a_static_auditor', 'frozen_phase_a_launcher']
    expected = {'phase_a_driver': source_records['fresh_phase_a_driver'],
                'phase_a_process_worker': source_records['fresh_phase_a_worker'],
                'phase_a_rng_isolation_proxy': source_records['fresh_rng_proxy'],
                'phase_a_independent_auditor': source_records['fresh_phase_a_independent_auditor']}
    if (roles != expected_roles or value.get('qualification_output_root') != str(QUALIFICATION_ROOT)
            or value.get('execution_sources_digest_sha256') != csha(value['execution_source_records'])
            or any(value['execution_sources'].get(role) != row for role, row in expected.items())):
        raise RuntimeError('prereg exact8')
    return value


def main():
    old_module, base = load_base()
    if os.sys.argv[1:] == ['--synthetic-self-test']:
        checks = {'source_roles': len(CONTRACT_SOURCE_ORDER) == 26,
                  'per_call_exact21': len(PER_CALL_RNG_CONTRACT) == 21,
                  'authorization': AUTHORIZATION['launcher_invocations_authorized'] == 1 and AUTHORIZATION['runtime_delegate_calls_authorized'] == 2000,
                  'bool_int_distinct': not json_exact({'count': 1}, {'count': True}),
                  'publish_selftest': base.publication_self_test()}
        print(json.dumps({'passed': all(checks.values()), 'checks': checks, 'checks_sha256': csha(checks)}, sort_keys=True))
        return 0 if all(checks.values()) else 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', type=Path, required=True); parser.add_argument('--contract-sha', required=True)
    parser.add_argument('--contract-bytes', type=int, required=True)
    parser.add_argument('--materializer-source', type=Path, required=True); parser.add_argument('--materializer-sha', required=True)
    parser.add_argument('--materializer-bytes', type=int, required=True); parser.add_argument('--authority-root', type=Path, required=True)
    args = parser.parse_args()
    if args.contract != CONTRACT_PATH or args.materializer_source != SELF_PATH or args.authority_root != AUTHORITY_ROOT:
        raise RuntimeError('canonical CLI')
    contract_record = regular(CONTRACT_PATH, args.contract_sha, args.contract_bytes)
    materializer_record = regular(SELF_PATH, args.materializer_sha, args.materializer_bytes)
    contract = json.loads(CONTRACT_PATH.read_text())
    if set(contract) != CONTRACT_TOP_KEYS or contract.get('format') != CONTRACT_FORMAT or contract.get('status') != CONTRACT_STATUS or type(contract.get('seed')) is not int or contract.get('seed') != 1659:
        raise RuntimeError('contract schema')
    source_records = contract['source_closure']
    observed = {role: regular(row['path'], row['sha256'], row['logical_bytes']) for role, row in source_records.items()}
    if (contract.get('source_role_order') != CONTRACT_SOURCE_ORDER or set(observed) != set(CONTRACT_SOURCE_ORDER)
            or observed != source_records or contract.get('source_closure_sha256') != csha(source_records)
            or contract.get('source_aliases') != {role: SOURCE_ALIASES[role] for role in CONTRACT_SOURCE_ORDER}):
        raise RuntimeError('source closure')
    expected_paths = {'authority_materializer': SELF_PATH, 'fresh_phase_a_launcher': LAUNCHER_PATH,
                      'fresh_phase_a_driver': DRIVER_PATH, 'fresh_phase_a_worker': WORKER_PATH,
                      'fresh_rng_proxy': PROXY_PATH, 'fresh_phase_a_independent_auditor': AUDITOR_PATH,
                      'fresh_phase_a_preregistration': PREREG_PATH,
                      'v523_v522_phase_a_failure_forensic': FAILURE_FORENSIC,
                      'v522_authority_design_contract': OLD_CONTRACT, 'v522_authority_materializer': OLD_MATERIALIZER,
                      'v522_authority_receipt': OLD_AUTHORITY_RECEIPT, 'v522_phase_a_launcher': OLD_LAUNCHER,
                      'v522_phase_a_driver': OLD_DRIVER, 'v522_phase_a_worker': OLD_WORKER,
                      'v522_phase_a_independent_auditor': OLD_AUDITOR, 'v522_attempt_intent': OLD_INTENT,
                      'v522_attempt_stdout': OLD_STDOUT, 'v522_attempt_stderr': OLD_STDERR,
                      'v522_attempt_terminal': OLD_TERMINAL}
    if any(observed[role]['path'] != str(path) for role, path in expected_paths.items()) or observed['authority_materializer'] != materializer_record:
        raise RuntimeError('active paths')
    old_contract = json.loads(OLD_CONTRACT.read_text())
    if any(observed[role] != old_contract['source_closure'][role] for role in FIXED_ROLES):
        raise RuntimeError('fixed sources')
    old = validate_old_state()
    if any(observed[role] != row for role, row in old['records'].items()): raise RuntimeError('ancestry source closure')
    validate_prereg(observed)
    for path in (DRIVER_PATH, WORKER_PATH, AUDITOR_PATH):
        if not json_exact(literal_assignment(path, 'PER_CALL_RNG_CONTRACT'), PER_CALL_RNG_CONTRACT):
            raise RuntimeError('per-call exact21 ' + str(path))
    historical, current = absence_paths()
    if any(os.path.lexists(path) for path in historical.values()): raise RuntimeError('fresh roots')
    expected_lineage = {'name': 'v523_v522_phase_a_per_call_rng_contract_repair',
                        'fresh_authority_root': str(AUTHORITY_ROOT),
                        'fresh_phase_a_attempt_root': str(ATTEMPT_ROOT),
                        'fresh_qualification_output_root': str(QUALIFICATION_ROOT),
                        'supersedes_v522_failed_attempt_no_retry': True,
                        'old_v522_authority_expected_current_and_consumed_once': True,
                        'sole_semantic_change': 'per_call_rng_contract_exact21_all_layers'}
    if (not json_exact(contract.get('lineage'), expected_lineage)
            or not json_exact(contract.get('authorization'), AUTHORIZATION)
            or not json_exact(contract.get('runtime_observation'), RUNTIME)
            or not json_exact(contract.get('execution_boundary'), EXECUTION_BOUNDARY)):
        raise RuntimeError('contract boundary')
    exact_values = {'authority_materializer_source': materializer_record,
                    'phase_a_launcher_source': observed['fresh_phase_a_launcher'],
                    'phase_a_driver_source': observed['fresh_phase_a_driver'],
                    'phase_a_worker_source': observed['fresh_phase_a_worker'],
                    'rng_proxy_source': observed['fresh_rng_proxy'],
                    'phase_a_independent_auditor_source': observed['fresh_phase_a_independent_auditor'],
                    'phase_a_preregistration_source': observed['fresh_phase_a_preregistration'],
                    'qualification_output_root': str(QUALIFICATION_ROOT), 'worker_environment_exact': WORKER_ENV,
                    'per_call_rng_evidence_contract': PER_CALL_RNG_CONTRACT,
                    'v522_authority_registration_tree': old['authority_tree'],
                    'v522_authority_receipt': old['authority_receipt'],
                    'v522_authority_semantics': old['authority_semantics'],
                    'v522_attempt_tree': old['attempt_tree'], 'v522_attempt_intent': old['attempt_intent'],
                    'v522_attempt_stdout': old['attempt_stdout'], 'v522_attempt_stderr': old['attempt_stderr'],
                    'v522_attempt_terminal': old['attempt_terminal'],
                    'v522_phase_a_failure_forensic': old['failure_forensic'],
                    'v522_phase_a_execution_partition': old['execution_partition'],
                    'v522_per_call_contract_failure_root_cause': old['root_cause']}
    if any(not json_exact(contract.get(key), value) for key, value in exact_values.items()): raise RuntimeError('contract anchors')
    files = {'contract': str(CONTRACT_PATH), 'materializer': str(SELF_PATH),
             **{'source_' + role: row['path'] for role, row in source_records.items()}}
    trees = {'v522_authority_current': str(OLD_AUTHORITY_ROOT), 'v522_attempt_current': str(OLD_ATTEMPT_ROOT)}
    pre = base.snapshot(files, trees, current)
    checks = {key: True for key in CHECK_KEYS}
    checks['gpu_empty'] = base.gpu_empty(); checks['no_live_phase_a_process'] = not phase_pids()
    checks['no_pending_values'] = not base.pending(contract)
    authority_sources = {'authority_design_contract': contract_record, **source_records}
    receipt = {'format': OUTPUT_FORMAT, 'status': OUTPUT_STATUS, 'passed': True,
               'source_closure': authority_sources, 'source_role_order': AUTHORITY_SOURCE_ORDER,
               'source_aliases': SOURCE_ALIASES, 'source_closure_sha256': csha(authority_sources),
               **{SOURCE_ALIASES[role]: row for role, row in authority_sources.items()}, **exact_values,
               'phase_a_attempt_root': str(ATTEMPT_ROOT),
               'historical_absences': {key: {'path': str(path), 'absent': True} for key, path in historical.items()},
               'required_absences': {key: {'path': str(path), 'absent': True} for key, path in current.items()},
               'checks': checks, 'check_keys': CHECK_KEYS, 'check_key_set_sha256': csha(CHECK_KEYS),
               'checks_sha256': csha(checks), 'input_pre_snapshot': pre,
               'input_post_snapshot': base.snapshot(files, trees, current), 'input_snapshots_exactly_equal': True,
               'authorization': AUTHORIZATION, 'runtime_observation': RUNTIME,
               'execution_boundary': EXECUTION_BOUNDARY, 'phase_a_cache_qualification_authorized': True,
               'training_authorized': False, 'preregistration_sha256': PREREG_SHA,
               'contract_sha256': PHASE_A_DESIGN_CONTRACT_SHA}
    schema = contract['authority_receipt_contract']
    if (schema.get('top_keys') != sorted(receipt) or schema.get('check_keys') != CHECK_KEYS
            or not json_exact(schema.get('authorization_exact'), AUTHORIZATION)
            or not json_exact(schema.get('per_call_rng_evidence_contract_exact'), PER_CALL_RNG_CONTRACT)):
        raise RuntimeError('receipt schema')
    if not all(checks.values()) or receipt['input_pre_snapshot'] != receipt['input_post_snapshot']:
        raise RuntimeError('receipt gates')
    publish_absences = {key: path for key, path in current.items() if key != 'authority_prep'}
    publish_pre = base.snapshot(files, trees, publish_absences)
    base.publish_exact1(AUTHORITY_ROOT, receipt, files, trees, publish_absences, publish_pre)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
