#!/usr/bin/env python3
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

B = Path('/root/autodl-tmp/IROS_WAM_2.0 challenge')
O = B / 'artifacts/strict_track2_official_20260810'
J = B / 'artifacts/strict_track2_joint_augmentation_20260810'
N = 'v255_v254_delta_regime_service_seed1457_20260819'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    registry = O / 'run_registry' / N
    run = J / N
    if registry.exists() or run.exists():
        raise SystemExit('refusing overwrite')
    analysis = O / 'run_registry/v252_v250_specific_terminal_long128_seed1455_20260819/delta_action_metric_analysis_v3.json'
    rejected = J / 'v252_v250_specific_terminal_long128_seed1455_20260819/audit/long_gate_report.json'
    if json.loads(rejected.read_text()).get('passed') is not False:
        raise ValueError('v252 rejection evidence missing')
    registry.mkdir(parents=True)
    (run / 'audit').mkdir(parents=True)
    (run / 'local_dev_token.txt').write_text('local-dev-token\n')
    sources = [
        B / 'pipeline/wam_pipeline/v254_delta_regime_terminal_runtime.py',
        B / 'pipeline/wam_pipeline/v250_specific_terminal_successor_runtime.py',
        B / 'pipeline/wam_pipeline/backends.py',
        B / 'pipeline/scripts/replay_v245_post_grasp_reward.py',
        B / 'pipeline/scripts/restart_v254_services.sh',
        B / 'pipeline/scripts/launch_v255_service_audit.sh',
    ]
    payload = {
        'format': 'strict-track2-v255-service-preregistration-v1',
        'registered_at': datetime.now(timezone.utc).isoformat(),
        'fixed_model': {
            'model_version': 'track2-v254-delta-regime-terminal',
            'delta_clean_max': .01,
            'clean_alignment_floor': .95,
            'delta_ood_min': .8,
            'ood_alignment_floor': .5,
            'alpha_scale': 8.,
            'target_rule': 'visual public-clean terminal successor',
        },
        'fixed_gate': {
            'success_like_min': 8, 'group_std_min': .01,
            'alignment_global_min': .2, 'alignment_group_min': .45,
            'regime_alpha_global_min': .7, 'service_contract': True,
        },
        'inputs': {
            'rejected_v252_long_gate': str(rejected),
            'delta_action_analysis': str(analysis),
            'delta_action_analysis_sha256': sha(analysis),
        },
        'implementation': {str(path): sha(path) for path in sources},
        'guards': {
            'runtime_uses_reward': False, 'public_data_only': True,
            'policy_modified': False, 'hidden_or_final_data': False,
            'real_submission': False,
        },
    }
    text = json.dumps(payload, indent=2) + '\n'
    (registry / 'preregistration.json').write_text(text)
    (run / 'release_registration.json').write_text(text)


if __name__ == '__main__':
    main()
