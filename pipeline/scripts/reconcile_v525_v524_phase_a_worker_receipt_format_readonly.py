#!/usr/bin/env python3
"""Read-only reconciliation of completed v524 Phase-A worker receipt formats.

This program never launches Phase-A code.  It validates the immutable completed
qualification, constructs two in-memory normalized worker-receipt views, proves
that the recursive difference is exactly the two format leaves, and publishes
one reconciliation receipt into a fresh root.
"""
from __future__ import annotations

import argparse
import ast
import copy
import ctypes
import hashlib
import importlib.util
import json
import os
import signal
import stat
import urllib.request
from pathlib import Path

ROOT = Path('/root/autodl-tmp/IROS_WAM_2.0 challenge')
J = ROOT / 'artifacts/strict_track2_joint_augmentation_20260810'
S = ROOT / 'pipeline/scripts'
SELF = S / 'reconcile_v525_v524_phase_a_worker_receipt_format_readonly.py'
FORENSIC = S / 'v525_v524_phase_a_worker_receipt_format_failure_forensic.json'
AUTHORITY_CONTRACT = S / 'v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_execution_authority_contract.json'
AUTHORITY_MATERIALIZER = S / 'materialize_v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_execution_authority.py'
FRESH_AUTHORITY_ROOT = J / 'v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_execution_authority_seed1661_20260827'
FRESH_AUTHORITY_RECEIPT = FRESH_AUTHORITY_ROOT / 'authority_receipt.json'
FORENSIC_EXPECTED = ('ea04aede510143cf9f1c6388c222b64397a8f5dfdf3c15a5c10c56a7d5b058d5', 8137)
V524_AUTH = J / 'v524_v523_phase_a_exact_source_roles_repair_execution_authority_seed1660_20260826'
V524_ATTEMPT = J / 'v524_v523_phase_a_cache_qualification_attempt_seed1660_20260826'
V524_QUALIFICATION = Path('/root/v524_v523_phase_a_cache_qualification_seed1660_20260826')
OUTPUT_ROOT = J / 'v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_seed1661_20260827'
OUTPUT_PREP = OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + '.reconciliation-prep')
OUTPUT_RECEIPT = OUTPUT_ROOT / 'reconciliation_receipt.json'
OLD_FORMAT = 'strict-track2-v523-v522-rng-isolated-cache-worker-receipt-v1'
NORMALIZED_FORMAT = 'strict-track2-v524-v523-rng-isolated-cache-worker-receipt-v1'
OUTPUT_FORMAT = 'strict-track2-v525-v524-phase-a-worker-receipt-format-readonly-reconciliation-receipt-v1'
OUTPUT_STATUS = 'computation_verified_readonly_launcher_consumer_schema_mismatch'
PUBLICATION_SIGNAL_SET = {signal.SIGINT, signal.SIGTERM}
PUBLICATION_TEST_HOOKS: dict[str, object] = {}
EXTERNAL_EVIDENCE_ROOT = J / 'v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_attempt_seed1661_20260827'
EXTERNAL_EVIDENCE_PREP = EXTERNAL_EVIDENCE_ROOT.with_name(EXTERNAL_EVIDENCE_ROOT.name + '.attempt-prep')
RECONCILER_CANDIDATE_CONTRACT = {
    'format': OUTPUT_FORMAT,
    'status': OUTPUT_STATUS,
    'passed': None,
    'candidate_verified': True,
    'standalone_consumable': False,
    'external_terminal_required': True,
    'publication_success_claimed': False,
    'commit_semantics': 'current_exact1_visibility_after_noreplace_directory_rename',
    'crash_durability_claimed': False,
    'postcommit_parent_dir_fsync_best_effort': True,
}
EXTERNAL_TERMINAL_EVIDENCE_CONTRACT = {
    'format': 'strict-track2-v525-v524-phase-a-worker-receipt-format-readonly-reconciliation-external-terminal-evidence-v1',
    'evidence_root': str(EXTERNAL_EVIDENCE_ROOT),
    'evidence_prep': str(EXTERNAL_EVIDENCE_PREP),
    'expected_file_count': 6,
    'terminal_member': 'process_receipt.json',
    'terminal_status': 'passed_external_terminal',
    'authoritative_success_only_via_external_terminal': True,
    'candidate_root': str(OUTPUT_ROOT),
    'candidate_expected_file_count': 1,
    'candidate_and_evidence_dual_bind_required': True,
    'current_exact1_candidate_required': True,
    'current_exact6_evidence_required': True,
    'fresh_transport_helper_source_required': True,
    'fresh_transport_script_source_required': True,
    'postcommit_diagnostics_adjudicated_by_external_terminal': True,
    'retry_authorized': False,
    'phase_a_replay_authorized': False,
}
ORIGINAL_CSHA = 'd00018deb7ea3ae660061e8879dc4a4f6d56bc65bd089c28c4a0580239c08101'
NORMALIZED_CSHA = 'c409c2b8f85641c23bc68fb06512cd664c84108d4b8b629480906193737f4b32'
DIFF_CSHA = 'fdc7999068d0e630272e6cb226b6b342692b5f8324cd506d593b5993b5f52779'
V524_AUTH_TREE = (1, 64970, '93f58f95c489f79f4d58cb36cd3d16815444b69674e6bd6270dbe09f1d956e7f', '8253ef78bf8eadcd7e47e3825cb94c109c2030f66103c4513c5829006136dd72')
V524_ATTEMPT_TREE = (4, 44268, '83b63f8a8174ea8d3e1077c263ff3d0dd43403d37992b45c99c811d365eb2c7a', '14c23172981c21dd6ec783fa187b6ee51aaa1fe9b01b8ca5620cc418ffe4ff91')
V524_QUALIFICATION_TREE = (20, 950124602, '6cdde75692d37b7eead3333d867b22dc6404f545bb5861b2f30b425fb4270dac', '223308d02ed5a4a082b4de639292030444a627bbbac7f361fac1f9c6a7fcbd1f')
AUTHORITY_RECORD = ('15aa29b6d83a807bfeed2298c75ddbe158d5bffdd508a2145a3164aecf6bcedf', 64970)
ATTEMPT_TERMINAL_RECORD = ('07c00c0a3f8bc52b2f0a642276be46f34d131473a876aecf9f354eca2e16498a', 38680)
QUALIFICATION_RECORDS = {
    'terminal_receipt.json': ('93d38e010b37275a740bf3ccae2b935b08e40b6953657c12838780e5e2b542a1', 83762),
    'qualification_report.json': ('6514b497f994bc502fc46dc6c5a90865dbb126402935cfa359a1af2fe0fbc173', 87681),
    'independent_final_audit.json': ('3d702b36a3589d737c9f21e0a1182c59f636753faa1143a2adc85d90d3a674bc', 9935),
    'independent_precleanup_audit.json': ('9ceeaf9d47e901b88628a095d18b8bdc785105ebfc2b6bd0f434ca573418ef32', 9629),
    'process_b_cleanup_receipt.json': ('9786761a7f8159d3115fe5ab45f604eba20927feb0be2b26ce75931e73a0fe44', 3470),
    'process_a/receipt.json': ('89b7ce9e0ee6263fba2cb5b7e393664330776cfa7914ad9715e6ff2ed7346414', 8871),
    'process_b/receipt.json': ('90ac9ff4eab3a06c02a2937ebe0f35a20e68145862e418ef50048c07c2d15214', 8912),
    'process_a/call_events.ndjson': ('9b83ed569d597e887cba4d483f4fae145cdaf9330a0866f51f49eea6799bed89', 5431045),
    'process_b/call_events.ndjson': ('66ac9f81e846768ea95629050a544804ea4e1cb245f7b41f388b208fcc82cd24', 5431045),
    'process_a/process.log': ('ed26dc2ab5f02459e12accd420889cb4e7ae89c06eee6d80573c18b341781f56', 467837),
    'process_b/process.log': ('a78896e81c0e9a9679b3efb3263872220cd6f1bef6285add3120fcce1bfc8136', 467837),
}
COUNT_KEYS = (
    'calls', 'call_events_count', 'warning_count', 'raw_warning_count', 'rng_unchanged_count',
    'rng_isolation_event_count', 'rng_isolation_delegate_started_count',
    'rng_isolation_delegate_completed_count', 'rng_isolation_python_all_four_stages_equal_count',
    'rng_isolation_numpy_all_four_stages_equal_count', 'rng_isolation_torch_cpu_exit_restored_count',
    'rng_isolation_torch_cuda_exit_restored_count', 'durable_call_started_markers',
    'durable_call_completed_markers',
)
EXPECTED_DIFF = [
    {'path': ['process_a', 'format'], 'original': OLD_FORMAT, 'normalized': NORMALIZED_FORMAT},
    {'path': ['process_b', 'format'], 'original': OLD_FORMAT, 'normalized': NORMALIZED_FORMAT},
]
RAW_MEDIAN_WARNING = "median CUDA with indices output does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:93.)"
PER_CALL_FORMAT = 'strict-track2-v520-per-call-rng-isolation-evidence-v1'
RNG_FORMAT = 'strict-track2-v520-v519-rng-isolation-four-stage-v1'
RNG_KEYS = {
    'format', 'exact_seed', 'cuda_device_count', 'cuda_device_indices',
    'fork_rng_devices_exact_all_cuda_indices', 'delegate_invocations_started',
    'delegate_invocations_completed', 'entry_external', 'inside_before_delegate',
    'internal_after_delegate', 'exit_restored', 'python_all_four_stages_equal',
    'numpy_all_four_stages_equal', 'torch_internal_change_allowed',
    'python_exit_restored', 'numpy_exit_restored', 'torch_cpu_exit_restored',
    'torch_cuda_exit_restored', 'exception_finally_restoration_complete',
    'four_stage_canonical_sha256',
}


def cbytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n').encode()


def csha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def hash_fd(fd: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    offset = 0
    while True:
        block = os.pread(fd, 1 << 20, offset)
        if not block:
            break
        digest.update(block)
        offset += len(block)
    if os.pread(fd, 1, offset) != b'':
        raise RuntimeError('EOF')
    return digest.hexdigest(), offset


def regular(path: Path, expected: tuple[str, int] | None = None) -> dict:
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        meta = os.fstat(fd)
        if not stat.S_ISREG(meta.st_mode):
            raise RuntimeError(f'not regular: {path}')
        observed = hash_fd(fd)
        current = os.lstat(path)
        if path.is_symlink() or (current.st_dev, current.st_ino) != (meta.st_dev, meta.st_ino):
            raise RuntimeError(f'identity drift: {path}')
        if expected is not None and observed != expected:
            raise RuntimeError(f'record mismatch: {path}: {observed}')
        return {'path': str(path), 'sha256': observed[0], 'logical_bytes': observed[1]}
    finally:
        os.close(fd)


def exact_tree(root: Path) -> dict:
    if root != root.resolve() or root.is_symlink() or not root.is_dir():
        raise RuntimeError(f'tree root: {root}')
    inventory = []
    for path in sorted(root.rglob('*'), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise RuntimeError(f'tree member: {path}')
        if path.is_file():
            row = regular(path)
            inventory.append([path.relative_to(root).as_posix(), row['sha256'], row['logical_bytes']])
        elif not path.is_dir():
            raise RuntimeError(f'tree member: {path}')
    lines = ''.join(f'{digest}  {name}\n' for name, digest, _ in inventory).encode()
    return {'root': str(root), 'inventory': inventory, 'file_count': len(inventory),
            'logical_file_bytes': sum(row[2] for row in inventory),
            'sha256sum_lines_digest_sha256': hashlib.sha256(lines).hexdigest(),
            'canonical_json_triples_digest_sha256': csha(inventory)}


def assert_tree(tree: dict, expected: tuple[int, int, str, str]) -> None:
    observed = (tree['file_count'], tree['logical_file_bytes'], tree['sha256sum_lines_digest_sha256'], tree['canonical_json_triples_digest_sha256'])
    if observed != expected:
        raise RuntimeError(f'tree mismatch: {observed}')


def json_load(path: Path) -> dict:
    return json.loads(path.read_text())


def recursive_diff(left: object, right: object, path: tuple[object, ...] = ()) -> list[dict]:
    if type(left) is not type(right):
        return [{'path': list(path), 'original': left, 'normalized': right}]
    if isinstance(left, dict):
        if set(left) != set(right):
            return [{'path': list(path), 'original_keys': sorted(left), 'normalized_keys': sorted(right)}]
        rows = []
        for key in sorted(left):
            rows.extend(recursive_diff(left[key], right[key], path + (key,)))
        return rows
    if isinstance(left, list):
        if len(left) != len(right):
            return [{'path': list(path), 'original_length': len(left), 'normalized_length': len(right)}]
        rows = []
        for index, (a, b) in enumerate(zip(left, right)):
            rows.extend(recursive_diff(a, b, path + (index,)))
        return rows
    return [] if left == right else [{'path': list(path), 'original': left, 'normalized': right}]


def records_current(value: object) -> bool:
    if isinstance(value, dict):
        if set(value) == {'path', 'sha256', 'logical_bytes'}:
            try:
                return regular(Path(value['path']), (value['sha256'], value['logical_bytes'])) == value
            except (OSError, RuntimeError, TypeError):
                return False
        return all(records_current(item) for item in value.values())
    if isinstance(value, list):
        return all(records_current(item) for item in value)
    return True


def load_materializer(path: Path, expected: tuple[str, int]):
    regular(path, expected)
    spec = importlib.util.spec_from_file_location('v525_authority_materializer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = ('CONTRACT_TOP_KEYS', 'AUTH_TOP_KEYS', 'CHECK_KEYS', 'CONTRACT_SOURCE_ORDER',
                'AUTHORITY_SOURCE_ORDER', 'SOURCE_ALIASES', 'AUTHORIZATION', 'RUNTIME',
                'OUTPUT_FORMAT', 'OUTPUT_STATUS', 'RECONCILER_CANDIDATE_CONTRACT',
                'EXTERNAL_TERMINAL_EVIDENCE_CONTRACT')
    if any(not hasattr(module, key) for key in required):
        raise RuntimeError('materializer interface')
    return module


def validate_authority(args, forensic_record: dict, self_record: dict) -> tuple[object, dict, dict]:
    module = load_materializer(args.authority_materializer_source, (args.authority_materializer_sha, args.authority_materializer_bytes))
    contract_record = regular(args.authority_contract, (args.authority_contract_sha, args.authority_contract_bytes))
    authority_record = regular(args.authority_receipt, (args.authority_receipt_sha, args.authority_receipt_bytes))
    contract = json_load(args.authority_contract)
    authority = json_load(args.authority_receipt)
    if set(contract) != set(module.CONTRACT_TOP_KEYS) or set(authority) != set(module.AUTH_TOP_KEYS):
        raise RuntimeError('authority schema')
    if authority.get('format') != module.OUTPUT_FORMAT or authority.get('status') != module.OUTPUT_STATUS or authority.get('passed') is not True:
        raise RuntimeError('authority semantics')
    if set(authority.get('checks', {})) != set(module.CHECK_KEYS) or not all(value is True for value in authority['checks'].values()):
        raise RuntimeError('authority checks')
    if authority.get('source_role_order') != module.AUTHORITY_SOURCE_ORDER or len(authority.get('source_closure', {})) != len(module.AUTHORITY_SOURCE_ORDER):
        raise RuntimeError('authority sources')
    if authority.get('authorization') != module.AUTHORIZATION or authority.get('runtime_observation') != module.RUNTIME:
        raise RuntimeError('authority boundary')
    if (cbytes(module.RECONCILER_CANDIDATE_CONTRACT) != cbytes(RECONCILER_CANDIDATE_CONTRACT)
            or cbytes(module.EXTERNAL_TERMINAL_EVIDENCE_CONTRACT) != cbytes(EXTERNAL_TERMINAL_EVIDENCE_CONTRACT)):
        raise RuntimeError('materializer external terminal interface')
    for name, expected in (('reconciler_candidate_contract', RECONCILER_CANDIDATE_CONTRACT),
                           ('external_terminal_evidence_contract', EXTERNAL_TERMINAL_EVIDENCE_CONTRACT)):
        if cbytes(contract.get(name)) != cbytes(expected) or cbytes(authority.get(name)) != cbytes(expected):
            raise RuntimeError(f'authority external terminal contract: {name}')
    authorization = authority['authorization']
    required_ints = {'readonly_reconciler_invocations_authorized': 1, 'readonly_reconciler_invocations_consumed': 0,
                     'launcher_invocations_authorized': 0, 'driver_invocations_authorized': 0,
                     'worker_invocations_authorized': 0, 'rng_proxy_delegate_invocations_authorized': 0,
                     'phase_a_replay_invocations_authorized': 0, 'training_invocations_authorized': 0,
                     'oof_invocations_authorized': 0}
    if any(type(authorization.get(key)) is not int or authorization.get(key) != value for key, value in required_ints.items()):
        raise RuntimeError('authority counts')
    for key in ('retry_authorized', 'phase_a_replay_authorized', 'training_authorized', 'cache_reuse_authorized',
                'reward_read_authorized', 'dev_hidden_final_outcome_read_authorized', 'oof_authorized'):
        if authorization.get(key) is not False:
            raise RuntimeError(f'unsafe authority: {key}')
    if not records_current(contract) or not records_current(authority):
        raise RuntimeError('authority current')
    records = list(authority.get('source_closure', {}).values())
    if forensic_record not in records or self_record not in records:
        raise RuntimeError('authority reconciler/forensic sources')
    return module, contract_record, authority_record, contract, authority


def validate_worker_receipts(original: dict) -> None:
    for name, role in (('process_a', 'A'), ('process_b', 'B')):
        receipt = original[name]
        if receipt.get('format') != OLD_FORMAT or receipt.get('passed') is not True or receipt.get('role') != role:
            raise RuntimeError(f'original worker receipt: {name}')
        if any(type(receipt.get(key)) is not int or receipt.get(key) != 1000 for key in COUNT_KEYS):
            raise RuntimeError(f'worker counts: {name}')
        if receipt.get('durable_call_markers_in_process_log') is not True or receipt.get('other_warning_count') != 0:
            raise RuntimeError(f'worker aggregate: {name}')
        if receipt.get('training_launched') is not False or receipt.get('reward_loaded') is not False:
            raise RuntimeError(f'worker unsafe: {name}')


def validate_rng_isolation(value: dict, seed: int) -> None:
    if set(value) != RNG_KEYS or value.get('format') != RNG_FORMAT:
        raise RuntimeError('RNG evidence schema')
    if type(seed) is not int or type(value.get('exact_seed')) is not int or value['exact_seed'] != seed:
        raise RuntimeError('RNG exact seed')
    if type(value.get('cuda_device_count')) is not int or value['cuda_device_count'] != 1 or value.get('cuda_device_indices') != [0]:
        raise RuntimeError('RNG CUDA devices')
    if type(value.get('delegate_invocations_started')) is not int or value['delegate_invocations_started'] != 1:
        raise RuntimeError('RNG delegate started')
    if type(value.get('delegate_invocations_completed')) is not int or value['delegate_invocations_completed'] != 1:
        raise RuntimeError('RNG delegate completed')
    for key in ('fork_rng_devices_exact_all_cuda_indices', 'python_all_four_stages_equal',
                'numpy_all_four_stages_equal', 'torch_internal_change_allowed', 'python_exit_restored',
                'numpy_exit_restored', 'torch_cpu_exit_restored', 'torch_cuda_exit_restored',
                'exception_finally_restoration_complete'):
        if value.get(key) is not True:
            raise RuntimeError(f'RNG flag: {key}')
    stages = ('entry_external', 'inside_before_delegate', 'internal_after_delegate', 'exit_restored')
    if any(not isinstance(value.get(stage), dict) for stage in stages):
        raise RuntimeError('RNG stages')
    entry, inside, internal, exit_state = (value[stage] for stage in stages)
    if not (entry['python'] == inside['python'] == internal['python'] == exit_state['python']):
        raise RuntimeError('Python RNG four stages')
    if not (entry['numpy'] == inside['numpy'] == internal['numpy'] == exit_state['numpy']):
        raise RuntimeError('NumPy RNG four stages')
    if entry['torch'] != exit_state['torch'] or inside['torch'] != internal['torch']:
        raise RuntimeError('Torch RNG restoration/internal stability')
    if value['four_stage_canonical_sha256'] != csha({stage: value[stage] for stage in stages}):
        raise RuntimeError('RNG four-stage digest')


def validate_per_call_and_markers(name: str, role: str) -> dict:
    events_path = V524_QUALIFICATION / name / 'call_events.ndjson'
    log_path = V524_QUALIFICATION / name / 'process.log'
    event_lines = events_path.read_text().splitlines()
    log_lines = log_path.read_text().splitlines()
    if len(event_lines) != 1000 or len(log_lines) != 2001:
        raise RuntimeError(f'per-call line counts: {name}')
    worker_start = json.loads(log_lines[0])
    if worker_start.get('event') != 'worker_start' or worker_start.get('role') != role:
        raise RuntimeError(f'worker start: {name}')
    for ordinal, line in enumerate(event_lines):
        row = json.loads(line)
        unsigned = dict(row)
        event_digest = unsigned.pop('event_canonical_sha256', None)
        sample_id = ordinal if role == 'A' else 999 - ordinal
        if (row.get('format') != PER_CALL_FORMAT or type(row.get('ordinal')) is not int
                or row['ordinal'] != ordinal or type(row.get('sample_id')) is not int
                or row['sample_id'] != sample_id or event_digest != csha(unsigned)):
            raise RuntimeError(f'event identity/digest: {name}:{ordinal}')
        if (type(row.get('sample_call_started')) is not int or row['sample_call_started'] != 1
                or type(row.get('sample_call_completed')) is not int or row['sample_call_completed'] != 1):
            raise RuntimeError(f'event call accounting: {name}:{ordinal}')
        if row.get('output_schema') != {'shape': [8, 256, 256, 3], 'dtype': 'uint8', 'contiguous': True}:
            raise RuntimeError(f'event output schema: {name}:{ordinal}')
        warning = {'category': 'UserWarning', 'message': RAW_MEDIAN_WARNING}
        if (row.get('raw_warning') != warning or row.get('warning_category') != 'UserWarning'
                or row.get('warning_full_message') != RAW_MEDIAN_WARNING or row.get('raw_warnings_unchanged') is not True):
            raise RuntimeError(f'event raw warning: {name}:{ordinal}')
        if (row.get('deterministic_before') is not True or row.get('deterministic_inside') is not True
                or row.get('deterministic_after') is not True or row.get('warn_only_before') is not False
                or row.get('warn_only_inside') is not True or row.get('warn_only_after') is not False):
            raise RuntimeError(f'event warning scope: {name}:{ordinal}')
        validate_rng_isolation(row.get('rng_isolation'), row.get('seed'))
        if row.get('rng_isolation_sha256') != csha(row['rng_isolation']) or row.get('rng_unchanged') is not True:
            raise RuntimeError(f'event RNG digest: {name}:{ordinal}')
        if (row.get('cpu_rng_before_sha256') != row.get('cpu_rng_after_sha256')
                or row.get('cuda_rng_before_sha256_by_device') != row.get('cuda_rng_after_sha256_by_device')):
            raise RuntimeError(f'event external RNG: {name}:{ordinal}')
        started = json.loads(log_lines[1 + 2 * ordinal])
        completed = json.loads(log_lines[2 + 2 * ordinal])
        if (set(started) != {'event', 'ordinal', 'sample_id', 'seed', 'request_sha256', 'started', 'completed'}
                or started != {'event': 'rng_proxy_call_started', 'ordinal': ordinal, 'sample_id': sample_id,
                               'seed': row['seed'], 'request_sha256': row['request_sha256'],
                               'started': 1, 'completed': 0}):
            raise RuntimeError(f'start marker: {name}:{ordinal}')
        expected_completed = {'event': 'rng_proxy_call_completed', 'ordinal': ordinal, 'sample_id': sample_id,
                              'event_canonical_sha256': row['event_canonical_sha256'],
                              'rng_isolation_sha256': row['rng_isolation_sha256'],
                              'started': 1, 'completed': 1}
        if completed != expected_completed:
            raise RuntimeError(f'complete marker: {name}:{ordinal}')
    return {'role': role, 'call_events_count': 1000, 'started_markers': 1000,
            'completed_markers': 1000, 'total_markers': 2000,
            'raw_median_user_warnings': 1000, 'other_warnings': 0,
            'python_numpy_four_stage_equal': 1000, 'torch_cpu_cuda_exit_restored': 1000,
            'delegate_started_completed': 1000, 'cuda_device_indices': [0],
            'process_log_marker_order': 'strict_alternating_started_completed'}


def validate_current_inputs() -> dict:
    auth_tree = exact_tree(V524_AUTH)
    attempt_tree = exact_tree(V524_ATTEMPT)
    qualification_tree = exact_tree(V524_QUALIFICATION)
    assert_tree(auth_tree, V524_AUTH_TREE)
    assert_tree(attempt_tree, V524_ATTEMPT_TREE)
    assert_tree(qualification_tree, V524_QUALIFICATION_TREE)
    regular(V524_AUTH / 'authority_receipt.json', AUTHORITY_RECORD)
    regular(V524_ATTEMPT / 'terminal_receipt.json', ATTEMPT_TERMINAL_RECORD)
    for name, record in QUALIFICATION_RECORDS.items():
        regular(V524_QUALIFICATION / name, record)
    attempt_terminal = json_load(V524_ATTEMPT / 'terminal_receipt.json')
    if attempt_terminal.get('status') != 'failed_no_retry' or attempt_terminal.get('error') != 'qualification A RNG receipt':
        raise RuntimeError('attempt terminal')
    terminal = json_load(V524_QUALIFICATION / 'terminal_receipt.json')
    report = json_load(V524_QUALIFICATION / 'qualification_report.json')
    audit = json_load(V524_QUALIFICATION / 'independent_final_audit.json')
    if terminal.get('passed') is not True or report.get('passed') is not True or audit.get('passed') is not True:
        raise RuntimeError('qualification computation')
    original = {'process_a': json_load(V524_QUALIFICATION / 'process_a/receipt.json'),
                'process_b': json_load(V524_QUALIFICATION / 'process_b/receipt.json')}
    validate_worker_receipts(original)
    per_call_validation = {
        'process_a': validate_per_call_and_markers('process_a', 'A'),
        'process_b': validate_per_call_and_markers('process_b', 'B'),
    }
    if csha(original) != ORIGINAL_CSHA:
        raise RuntimeError('original receipts digest')
    normalized = copy.deepcopy(original)
    for name in ('process_a', 'process_b'):
        normalized[name]['format'] = NORMALIZED_FORMAT
    rows = recursive_diff(original, normalized)
    if rows != EXPECTED_DIFF or csha(rows) != DIFF_CSHA or csha(normalized) != NORMALIZED_CSHA:
        raise RuntimeError('normalization exact2')
    marker = report.get('durable_marker_summary', {})
    if (marker.get('total_calls') != 2000 or marker.get('total_markers') != 4000
            or marker.get('all_pairs_alternate_started_completed') is not True):
        raise RuntimeError('markers')
    return {'authority_tree': auth_tree, 'attempt_tree': attempt_tree, 'qualification_tree': qualification_tree,
            'original_worker_receipts': original, 'normalized_worker_receipts': normalized,
            'per_call_validation': per_call_validation,
            'recursive_diff': rows, 'qualification_terminal': regular(V524_QUALIFICATION / 'terminal_receipt.json', QUALIFICATION_RECORDS['terminal_receipt.json']),
            'qualification_report': regular(V524_QUALIFICATION / 'qualification_report.json', QUALIFICATION_RECORDS['qualification_report.json']),
            'independent_final_audit': regular(V524_QUALIFICATION / 'independent_final_audit.json', QUALIFICATION_RECORDS['independent_final_audit.json'])}


def validate_embedded_authority_inputs(contract: dict, authority: dict, current: dict) -> None:
    normalization = {
        'original_worker_receipts': current['original_worker_receipts'],
        'normalized_worker_receipts': current['normalized_worker_receipts'],
        'original_worker_receipts_canonical_sha256': ORIGINAL_CSHA,
        'normalized_worker_receipts_canonical_sha256': NORMALIZED_CSHA,
        'recursive_diff': current['recursive_diff'],
        'recursive_diff_count': 2,
        'recursive_diff_canonical_sha256': DIFF_CSHA,
        'recursive_diff_full_paths': [['process_a', 'format'], ['process_b', 'format']],
        'standalone_phase_a_success_claimed': False,
        'new_computation_claimed': False,
    }
    exact = {
        'v524_authority_registration_tree': current['authority_tree'],
        'v524_attempt_tree': current['attempt_tree'],
        'v524_qualification_tree': current['qualification_tree'],
        'normalization_contract': normalization,
    }
    for key, value in exact.items():
        if contract.get(key) != value or authority.get(key) != value:
            raise RuntimeError(f'embedded authority input: {key}')
    source_records = {
        'v524_process_a_receipt': QUALIFICATION_RECORDS['process_a/receipt.json'],
        'v524_process_b_receipt': QUALIFICATION_RECORDS['process_b/receipt.json'],
        'v524_qualification_terminal': QUALIFICATION_RECORDS['terminal_receipt.json'],
        'v524_qualification_report': QUALIFICATION_RECORDS['qualification_report.json'],
        'v524_qualification_independent_audit': QUALIFICATION_RECORDS['independent_final_audit.json'],
    }
    for key, expected in source_records.items():
        for container in (contract, authority):
            row = container.get(key)
            if not isinstance(row, dict) or (row.get('sha256'), row.get('logical_bytes')) != expected:
                raise RuntimeError(f'embedded source record: {key}')


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def rename_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    fn = getattr(libc, 'renameat2', None)
    if fn is None or fn(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def publication_hook(name: str) -> None:
    hook = PUBLICATION_TEST_HOOKS.get(name)
    if hook is not None:
        hook()


def identity(path: Path) -> tuple[int, int]:
    meta = os.lstat(path)
    return meta.st_dev, meta.st_ino


def write_exclusive(path: Path, payload: bytes) -> dict:
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    owned_identity = None
    try:
        meta = os.fstat(fd)
        if not stat.S_ISREG(meta.st_mode):
            raise RuntimeError('receipt not regular')
        owned_identity = (meta.st_dev, meta.st_ino)
        publication_hook('after_receipt_open')
        view = memoryview(payload)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise RuntimeError('short write')
            view = view[count:]
        os.fsync(fd)
        if os.pread(fd, len(payload) + 1, 0) != payload or os.pread(fd, 1, len(payload)) != b'':
            raise RuntimeError('write readback')
        current = os.lstat(path)
        if path.is_symlink() or (current.st_dev, current.st_ino) != (meta.st_dev, meta.st_ino):
            raise RuntimeError('receipt identity drift')
        return {'dev': meta.st_dev, 'ino': meta.st_ino,
                'sha256': hashlib.sha256(payload).hexdigest(), 'logical_bytes': len(payload)}
    except BaseException:
        try:
            if owned_identity is not None and identity(path) == owned_identity and not path.is_symlink():
                os.unlink(path)
                fsync_dir(path.parent)
        except OSError:
            pass
        raise
    finally:
        os.close(fd)


def owned_prep_current(prep_identity: tuple[int, int], receipt_identity: dict) -> bool:
    try:
        if identity(OUTPUT_PREP) != prep_identity or OUTPUT_PREP.is_symlink() or not OUTPUT_PREP.is_dir():
            return False
        members = list(OUTPUT_PREP.iterdir())
        if members != [OUTPUT_PREP / 'reconciliation_receipt.json']:
            return False
        member = members[0]
        if identity(member) != (receipt_identity['dev'], receipt_identity['ino']):
            return False
        fd = os.open(member, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        try:
            return hash_fd(fd) == (receipt_identity['sha256'], receipt_identity['logical_bytes'])
        finally:
            os.close(fd)
    except (OSError, RuntimeError):
        return False


def cleanup_owned_prep(prep_identity: tuple[int, int] | None, receipt_identity: dict | None) -> bool:
    if prep_identity is None:
        return False
    try:
        if identity(OUTPUT_PREP) != prep_identity or OUTPUT_PREP.is_symlink() or not OUTPUT_PREP.is_dir():
            return False
    except OSError:
        return False
    members = list(OUTPUT_PREP.iterdir())
    if receipt_identity is None:
        if members:
            return False
    else:
        if not owned_prep_current(prep_identity, receipt_identity):
            return False
        os.unlink(OUTPUT_PREP / 'reconciliation_receipt.json')
    os.rmdir(OUTPUT_PREP)
    fsync_dir(J)
    return True


def expected_output_tree(receipt_identity: dict) -> dict:
    inventory = [['reconciliation_receipt.json', receipt_identity['sha256'], receipt_identity['logical_bytes']]]
    lines = f"{receipt_identity['sha256']}  reconciliation_receipt.json\n".encode()
    return {'root': str(OUTPUT_ROOT), 'inventory': inventory, 'file_count': 1,
            'logical_file_bytes': receipt_identity['logical_bytes'],
            'sha256sum_lines_digest_sha256': hashlib.sha256(lines).hexdigest(),
            'canonical_json_triples_digest_sha256': csha(inventory)}


def validate_visible_output(prep_identity: tuple[int, int], receipt_identity: dict) -> dict:
    if identity(OUTPUT_ROOT) != prep_identity or OUTPUT_ROOT.is_symlink() or not OUTPUT_ROOT.is_dir():
        raise RuntimeError('output root identity')
    member = OUTPUT_RECEIPT
    if identity(member) != (receipt_identity['dev'], receipt_identity['ino']):
        raise RuntimeError('output receipt identity')
    record = regular(member, (receipt_identity['sha256'], receipt_identity['logical_bytes']))
    tree = exact_tree(OUTPUT_ROOT)
    if tree != expected_output_tree(receipt_identity) or record['path'] != str(member):
        raise RuntimeError('output exact1')
    return tree


def publish(receipt: dict) -> tuple[dict, dict, dict]:
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, PUBLICATION_SIGNAL_SET)
    prep_identity = None
    receipt_identity = None
    committed = False
    postcommit_diagnostic_passed = False
    try:
        publication_hook('before_prestate')
        if os.path.lexists(OUTPUT_ROOT) or os.path.lexists(OUTPUT_PREP):
            raise FileExistsError(OUTPUT_ROOT)
        OUTPUT_PREP.mkdir(mode=0o700)
        prep_identity = identity(OUTPUT_PREP)
        fsync_dir(J)
        publication_hook('after_prep_visible')
        if identity(OUTPUT_PREP) != prep_identity or OUTPUT_PREP.is_symlink() or not OUTPUT_PREP.is_dir():
            raise RuntimeError('prep root identity drift')
        receipt_identity = write_exclusive(OUTPUT_PREP / 'reconciliation_receipt.json', cbytes(receipt))
        publication_hook('after_receipt_write')
        if not owned_prep_current(prep_identity, receipt_identity):
            raise RuntimeError('owned prep drift')
        fsync_dir(OUTPUT_PREP)
        publication_hook('before_rename')
        if signal.sigpending() & PUBLICATION_SIGNAL_SET:
            raise InterruptedError('pending publication signal')
        rename_noreplace(OUTPUT_PREP, OUTPUT_ROOT)
        committed = True  # Choice-A: successful NOREPLACE visibility is the commit point.
        tree = expected_output_tree(receipt_identity)
        try:
            publication_hook('after_rename_before_parent_fsync')
            if signal.sigpending() & PUBLICATION_SIGNAL_SET:
                raise InterruptedError('pending postcommit publication signal')
            fsync_dir(J)
            publication_hook('after_parent_fsync')
            tree = validate_visible_output(prep_identity, receipt_identity)
            publication_hook('after_current_gate')
            postcommit_diagnostic_passed = True
        except BaseException:
            # The candidate is already current.  It is intentionally non-consumable
            # without the future external terminal evidence; never rollback or publish twice.
            postcommit_diagnostic_passed = False
        return tree, {
            'commit_semantics': 'current_exact1_visibility_after_noreplace_directory_rename',
            'visibility_committed': True,
            'crash_durability_claimed': False,
            'postcommit_parent_dir_fsync_best_effort': True,
            'postcommit_diagnostic_passed': postcommit_diagnostic_passed,
            'second_publication_attempted': False,
        }, {'path': str(OUTPUT_RECEIPT), 'sha256': receipt_identity['sha256'],
            'logical_bytes': receipt_identity['logical_bytes']}
    except BaseException:
        if not committed:
            cleanup_owned_prep(prep_identity, receipt_identity)
        raise
    finally:
        if not committed:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)


def synthetic_self_test() -> dict:
    source = Path(__file__).read_text()
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    forbidden_calls = {'Popen', 'run', 'call', 'check_call', 'check_output', 'system',
                       'execv', 'execve', 'spawnv', 'spawnve'}
    observed_forbidden_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else None
            if name in forbidden_calls:
                observed_forbidden_calls.append(name)
    minimal = {'process_a': {'format': OLD_FORMAT, 'passed': True}, 'process_b': {'format': OLD_FORMAT, 'passed': True}}
    fixed = copy.deepcopy(minimal)
    fixed['process_a']['format'] = NORMALIZED_FORMAT
    fixed['process_b']['format'] = NORMALIZED_FORMAT
    rows = recursive_diff(minimal, fixed)
    checks = {
        'no_popen_name': 'Popen' not in names and 'Popen' not in attrs,
        'no_subprocess_import': all(not (isinstance(node, (ast.Import, ast.ImportFrom)) and any(alias.name == 'subprocess' for alias in node.names)) for node in ast.walk(tree)),
        'recursive_diff_exact2_paths': [row['path'] for row in rows] == [['process_a', 'format'], ['process_b', 'format']],
        'normalization_does_not_mutate_original': minimal['process_a']['format'] == OLD_FORMAT and minimal['process_b']['format'] == OLD_FORMAT,
        'constants_exact': csha(EXPECTED_DIFF) == DIFF_CSHA and FORENSIC_EXPECTED[1] > 0,
        'all_execution_entry_calls_absent': not observed_forbidden_calls,
        'candidate_contract_nonterminal': (
            RECONCILER_CANDIDATE_CONTRACT['passed'] is None
            and RECONCILER_CANDIDATE_CONTRACT['candidate_verified'] is True
            and RECONCILER_CANDIDATE_CONTRACT['standalone_consumable'] is False
            and RECONCILER_CANDIDATE_CONTRACT['external_terminal_required'] is True
            and RECONCILER_CANDIDATE_CONTRACT['publication_success_claimed'] is False),
        'external_terminal_contract_exact6': (
            EXTERNAL_TERMINAL_EVIDENCE_CONTRACT['expected_file_count'] == 6
            and EXTERNAL_TERMINAL_EVIDENCE_CONTRACT['current_exact1_candidate_required'] is True
            and EXTERNAL_TERMINAL_EVIDENCE_CONTRACT['current_exact6_evidence_required'] is True),
        'publication_signal_block_precedes_prestate': (
            source.index('signal.pthread_sigmask(signal.SIG_BLOCK') < source.index("publication_hook('before_prestate')")),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    return {'passed': True, 'checks': checks, 'check_count': len(checks), 'phase_a_invocations': 0,
            'readonly_reconciler_invocations': 0, 'output_root_created': False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic-self-test', action='store_true')
    parser.add_argument('--authority-contract', type=Path)
    parser.add_argument('--authority-contract-sha')
    parser.add_argument('--authority-contract-bytes', type=int)
    parser.add_argument('--authority-receipt', type=Path)
    parser.add_argument('--authority-receipt-sha')
    parser.add_argument('--authority-receipt-bytes', type=int)
    parser.add_argument('--authority-materializer-source', type=Path)
    parser.add_argument('--authority-materializer-sha')
    parser.add_argument('--authority-materializer-bytes', type=int)
    parser.add_argument('--reconciler-source', type=Path)
    parser.add_argument('--reconciler-sha')
    parser.add_argument('--reconciler-bytes', type=int)
    parser.add_argument('--forensic', type=Path)
    parser.add_argument('--forensic-sha')
    parser.add_argument('--forensic-bytes', type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.synthetic_self_test:
        print(json.dumps(synthetic_self_test(), sort_keys=True))
        return 0
    required = ('authority_contract', 'authority_contract_sha', 'authority_contract_bytes', 'authority_receipt',
                'authority_receipt_sha', 'authority_receipt_bytes', 'authority_materializer_source',
                'authority_materializer_sha', 'authority_materializer_bytes', 'reconciler_source',
                'reconciler_sha', 'reconciler_bytes', 'forensic', 'forensic_sha', 'forensic_bytes')
    if any(getattr(args, key) is None for key in required):
        raise RuntimeError('missing CLI')
    if (args.reconciler_source != SELF or args.forensic != FORENSIC
            or args.authority_contract != AUTHORITY_CONTRACT
            or args.authority_materializer_source != AUTHORITY_MATERIALIZER
            or args.authority_receipt != FRESH_AUTHORITY_RECEIPT):
        raise RuntimeError('canonical source paths')
    self_record = regular(args.reconciler_source, (args.reconciler_sha, args.reconciler_bytes))
    forensic_record = regular(args.forensic, (args.forensic_sha, args.forensic_bytes))
    if (forensic_record['sha256'], forensic_record['logical_bytes']) != FORENSIC_EXPECTED:
        raise RuntimeError('forensic exact')
    forensic = json_load(FORENSIC)
    if forensic.get('status') != 'failed_no_retry_computation_complete_readonly_reconciliation_required':
        raise RuntimeError('forensic semantics')
    _, contract_record, authority_record, contract, authority = validate_authority(args, forensic_record, self_record)
    before = validate_current_inputs()
    validate_embedded_authority_inputs(contract, authority, before)
    services_before = service_health()
    if gpu_compute_pids() or phase_a_pids():
        raise RuntimeError('live execution')
    after = validate_current_inputs()
    services_after = service_health()
    if csha(before) != csha(after) or services_before != services_after or gpu_compute_pids() or phase_a_pids():
        raise RuntimeError('readonly drift')
    receipt = {
        **RECONCILER_CANDIDATE_CONTRACT,
        'external_terminal_evidence_contract': EXTERNAL_TERMINAL_EVIDENCE_CONTRACT,
        'standalone_phase_a_success_claimed': False, 'new_computation_claimed': False,
        'readonly_reconciliation_completed': True, 'readonly_reconciler_invocations': 1,
        'phase_a_launcher_invocations': 0, 'phase_a_driver_invocations': 0,
        'phase_a_worker_invocations': 0, 'rng_proxy_delegate_invocations': 0,
        'training_invocations': 0, 'oof_invocations': 0, 'retry_authorized': False,
        'contract': contract_record, 'authority_receipt': authority_record,
        'authority_materializer_source': regular(args.authority_materializer_source, (args.authority_materializer_sha, args.authority_materializer_bytes)),
        'reconciler_source': self_record, 'failure_forensic': forensic_record,
        'v524_authority_tree': before['authority_tree'], 'v524_attempt_tree': before['attempt_tree'],
        'v524_qualification_tree': before['qualification_tree'],
        'qualification_terminal': before['qualification_terminal'], 'qualification_report': before['qualification_report'],
        'independent_final_audit': before['independent_final_audit'],
        'original_worker_receipts': before['original_worker_receipts'],
        'normalized_worker_receipts': before['normalized_worker_receipts'],
        'per_call_validation': before['per_call_validation'],
        'original_worker_receipts_canonical_sha256': ORIGINAL_CSHA,
        'normalized_worker_receipts_canonical_sha256': NORMALIZED_CSHA,
        'recursive_diff': before['recursive_diff'], 'recursive_diff_count': 2,
        'recursive_diff_canonical_sha256': DIFF_CSHA,
        'input_pre_snapshot_sha256': csha(before), 'input_post_snapshot_sha256': csha(after),
        'input_snapshots_exactly_equal': True,
        'service_health_pre': services_before, 'service_health_post': services_after,
        'service_health_exactly_equal': True, 'gpu_compute_pids': [], 'phase_a_pids': [],
        'training_authorized': False, 'cache_reuse_authorized': False, 'reward_read_authorized': False,
        'dev_hidden_final_outcome_read_authorized': False, 'oof_authorized': False,
    }
    tree, publication, receipt_record = publish(receipt)
    print(json.dumps({'candidate_verified': True, 'standalone_consumable': False,
                      'external_terminal_required': True, 'publication_success_claimed': False,
                      'status': OUTPUT_STATUS, 'reconciliation_receipt': receipt_record,
                      'reconciliation_tree': tree, 'publication': publication, 'recursive_diff_count': 2,
                      'phase_a_launcher_invocations': 0, 'phase_a_driver_invocations': 0,
                      'phase_a_worker_invocations': 0, 'rng_proxy_delegate_invocations': 0}, sort_keys=True))
    return 0


def service_health() -> dict:
    result = {}
    for name, url in (('8005_v1_health', 'http://127.0.0.1:8005/v1/health'),
                      ('18084_health', 'http://127.0.0.1:18084/health')):
        with urllib.request.urlopen(url, timeout=10) as response:
            body = response.read()
            result[name] = {'http_code': response.status, 'body_sha256': hashlib.sha256(body).hexdigest(),
                            'body_bytes': len(body), 'json_model': json.loads(body)}
    if not all(row['http_code'] == 200 and row['body_bytes'] > 0 for row in result.values()):
        raise RuntimeError('service health')
    return result


def gpu_compute_pids() -> list[int]:
    class ProcessInfo(ctypes.Structure):
        _fields_ = [('pid', ctypes.c_uint), ('usedGpuMemory', ctypes.c_ulonglong),
                    ('gpuInstanceId', ctypes.c_uint), ('computeInstanceId', ctypes.c_uint)]
    nvml = ctypes.CDLL('libnvidia-ml.so.1')
    if nvml.nvmlInit_v2() != 0:
        raise RuntimeError('nvml init')
    try:
        count = ctypes.c_uint()
        if nvml.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
            raise RuntimeError('nvml count')
        found = set()
        for index in range(count.value):
            handle = ctypes.c_void_p()
            if nvml.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) != 0:
                raise RuntimeError('nvml handle')
            size = ctypes.c_uint(0)
            status = nvml.nvmlDeviceGetComputeRunningProcesses_v3(handle, ctypes.byref(size), None)
            if status not in (0, 7):
                raise RuntimeError('nvml processes size')
            if size.value:
                rows = (ProcessInfo * size.value)()
                if nvml.nvmlDeviceGetComputeRunningProcesses_v3(handle, ctypes.byref(size), rows) not in (0, 7):
                    raise RuntimeError('nvml processes')
                found.update(int(rows[i].pid) for i in range(size.value))
        return sorted(found)
    finally:
        nvml.nvmlShutdown()


def phase_a_pids() -> list[dict]:
    markers = ('launch_v524_v523_phase_a_cache_qualification.py',
               'generate_v524_v523_phase_a_cache_qualification.py',
               'generate_v524_v523_phase_a_cache_qualification_worker.py')
    rows = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if any(marker in command for marker in markers):
            rows.append({'pid': int(entry.name), 'cmdline': command})
    return sorted(rows, key=lambda row: row['pid'])


if __name__ == '__main__':
    raise SystemExit(main())
