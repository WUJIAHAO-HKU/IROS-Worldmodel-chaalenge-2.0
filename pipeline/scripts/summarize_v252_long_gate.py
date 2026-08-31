#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(path: Path, mask: np.ndarray, threshold: float) -> dict:
    payload = json.loads(path.read_text())
    scores = np.asarray(payload['raw_scores']['candidate'], dtype=float)[mask]
    peak = scores.max(axis=1)
    return {
        'count': len(peak),
        'mean': float(scores.mean()),
        'peak_mean': float(peak.mean()),
        'peak_max': float(peak.max()),
        'hits': int((peak >= threshold).sum()),
        'hit_rate': float((peak >= threshold).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads((args.registry / 'preregistration.json').read_text())
    fixed = prereg['fixed_audit']
    with np.load(args.run / 'audit/public_success_baseline.npz', allow_pickle=False) as data:
        success_mask = data['arm_right'].astype(bool)
    with np.load(args.run / 'audit/public_failure_baseline.npz', allow_pickle=False) as data:
        failure_mask = data['arm_right'].astype(bool)
    success = summarize(args.run / 'audit/public_success_reward.json', success_mask, fixed['threshold'])
    failure = summarize(args.run / 'audit/public_failure_reward.json', failure_mask, fixed['threshold'])
    capture = json.loads(Path(prereg['input_capture_gate']).read_text())
    checks = {
        'capture_gate': capture['passed'],
        'success_recall': success['hit_rate'] >= fixed['success_hit_rate_min'],
        'failure_consistency': failure['hit_rate'] <= fixed['failure_hit_rate_max'],
        'margin': success['hit_rate'] - failure['hit_rate'] >= fixed['margin_min'],
        'service_contract': (args.run / 'audit/service_acceptance.json').is_file(),
    }
    report = {
        'format': 'strict-track2-v252-long128-gate-v1',
        'success': success,
        'failure': failure,
        'margin': success['hit_rate'] - failure['hit_rate'],
        'checks': checks,
        'passed': all(checks.values()),
        'guards': prereg['guards'],
    }
    (args.run / 'audit/long_gate_report.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.run / ('V252_LONG_GATE_PASSED' if report['passed'] else 'V252_LONG_GATE_REJECTED')).touch()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
