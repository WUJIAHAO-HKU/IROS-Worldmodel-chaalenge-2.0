#!/usr/bin/env python3
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

B = Path('/root/autodl-tmp/IROS_WAM_2.0 challenge')
O = B / 'artifacts/strict_track2_official_20260810'
J = B / 'artifacts/strict_track2_joint_augmentation_20260810'
N = 'v252_v250_specific_terminal_long128_seed1455_20260819'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    registry = O / 'run_registry' / N
    run = J / N
    if registry.exists() or run.exists():
        raise SystemExit('refusing overwrite')
    registry.mkdir(parents=True)
    (run / 'audit').mkdir(parents=True)
    (run / 'local_dev_token.txt').write_text('local-dev-token\n')
    sources = [
        B / 'pipeline/wam_pipeline/v250_specific_terminal_successor_runtime.py',
        B / 'pipeline/wam_pipeline/backends.py',
        B / 'pipeline/scripts/export_v217_service_recursive_cache.py',
        B / 'pipeline/scripts/summarize_v252_long_gate.py',
        B / 'pipeline/scripts/restart_v252_services.sh',
        B / 'pipeline/scripts/launch_v252_long_gate.sh',
    ]
    prereg = {
        'format': 'strict-track2-v252-long128-preregistration-v1',
        'registered_at': datetime.now(timezone.utc).isoformat(),
        'frozen_model_version': 'track2-v250-specific-terminal-successor',
        'input_capture_gate': str(J / 'v251_v250_specific_terminal_service_seed1454_20260819/audit/post_grasp_reward_report.json'),
        'fixed_audit': {
            'chunks': 16,
            'threshold': 0.9,
            'success_hit_rate_min': 0.75,
            'failure_hit_rate_max': 0.2,
            'margin_min': 0.55,
            'official_http_acceptance': True,
        },
        'implementation': {str(path): sha256(path) for path in sources},
        'guards': {
            'public_data_only': True,
            'policy_training': False,
            'hidden_or_final_data': False,
            'real_submission': False,
        },
    }
    payload = json.dumps(prereg, indent=2) + '\n'
    (registry / 'preregistration.json').write_text(payload)
    (run / 'preregistration.json').write_text(payload)


if __name__ == '__main__':
    main()
