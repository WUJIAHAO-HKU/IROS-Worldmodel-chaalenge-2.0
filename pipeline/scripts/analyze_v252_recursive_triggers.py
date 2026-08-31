#!/usr/bin/env python3
"""Diagnose v250 recursive terminal triggers on public long-horizon caches."""
import argparse
import json
from pathlib import Path

import numpy as np

from wam_pipeline.arm_routed_autoregressive_runtime import Track2ArmRoutedAutoregressiveUNet
from wam_pipeline.v216_public_knn_blend_runtime import _visual_descriptor
from wam_pipeline.v245_clean_progressive_successor_runtime import ACTION_WEIGHT, DISTANCE_SCALE, MOTION_SCALE, _path_length


def parse_name(name: str) -> tuple[int, int]:
    episode, start = Path(name).stem.split('_')
    return int(episode[7:]), int(start)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--success-windows', type=Path, required=True)
    parser.add_argument('--failure-windows', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('refusing overwrite')
    with np.load(args.library, allow_pickle=False) as data:
        episode = data['episode_id'].astype(int)
        paths = data['path'].astype(str)
        visual = data['visual'].astype(np.float32)
        action = data['action'].astype(np.float32)
        mean = data['normalization_mean'].astype(np.float32)
        std = data['normalization_std'].astype(np.float32)
    clean = np.flatnonzero(episode >= 20000)
    row_start = np.asarray([parse_name(path)[1] for path in paths], dtype=int)
    rows = []
    for split, window_root in [('success', args.success_windows), ('failure', args.failure_windows)]:
        baseline_path = args.run / 'audit' / f'public_{split}_baseline.npz'
        candidate_path = args.run / 'audit' / f'public_{split}_candidate.npz'
        reward_path = args.run / 'audit' / f'public_{split}_reward.json'
        with np.load(baseline_path, allow_pickle=False) as data:
            sequences = data['path'].astype(str)
            arms = data['arm_right'].astype(bool)
            instructions = data['instruction'].astype(str)
        with np.load(candidate_path, allow_pickle=False) as data:
            frames = data['candidate']
        rewards = np.asarray(json.loads(reward_path.read_text())['raw_scores']['candidate'], dtype=float)
        index = {parse_name(path.name): path for path in window_root.glob('episode*_*.npz')}
        for sequence in np.flatnonzero(arms):
            ep, start = parse_name(sequences[sequence])
            with np.load(index[(ep, start)], allow_pickle=False) as data:
                context = data['context_frames'].copy()
                history = data['history_actions'].astype(np.float32).copy()
            chunks = []
            for chunk in range(16):
                with np.load(index[(ep, start + 8 * chunk)], allow_pickle=False) as data:
                    future = data['future_actions'].astype(np.float32).copy()
                route = Track2ArmRoutedAutoregressiveUNet.active_arm(history, future, instructions[sequence])
                post = route == 'right' and history[-1, 13] < .5 and (future[:, 13] < .5).mean() >= .75
                query_visual = _visual_descriptor(context[-1])
                query_action = ((np.concatenate((history, future), axis=0) - mean) / std).reshape(-1)
                visual_distance = ((visual[clean] - query_visual) ** 2).mean(1)
                action_distance = ((action[clean] - query_action) ** 2).mean(1)
                score = visual_distance / max(float(np.median(visual_distance)), 1e-9)
                score += ACTION_WEIGHT * action_distance / max(float(np.median(action_distance)), 1e-9)
                base = int(clean[int(np.argmin(score))])
                raw = action[base].reshape(-1, 14) * std + mean
                query_delta = future[-1, 7:13] - history[-1, 7:13]
                library_delta = raw[-1, 7:13] - raw[-9, 7:13]
                denominator = float(np.linalg.norm(query_delta) * np.linalg.norm(library_delta))
                alignment = float(np.dot(query_delta, library_delta) / denominator) if denominator > 1e-8 else 0.0
                distance = float(action_distance[int(np.argmin(score))])
                confidence = float(np.exp(-distance / (.5 * DISTANCE_SCALE)))
                gate = float(np.clip((alignment - .5) / .5, 0, 1))
                motion = _path_length(history, future)
                alpha = float(np.clip(16 * confidence * gate * np.clip(motion / MOTION_SCALE, 0, 1), 0, 1)) if post else 0.0
                chunks.append({
                    'chunk': chunk, 'post': bool(post), 'base': base,
                    'base_episode': int(episode[base]), 'base_start': int(row_start[base]),
                    'visual_distance': float(visual_distance[int(np.argmin(score))]),
                    'action_distance': distance, 'alignment': alignment,
                    'motion': float(motion), 'alpha': alpha,
                    'reward_peak': float(rewards[sequence, chunk * 8:(chunk + 1) * 8].max()),
                })
                prediction = frames[sequence, chunk * 8:(chunk + 1) * 8]
                context = np.concatenate((context, prediction), axis=0)[-5:]
                history = np.concatenate((history, future), axis=0)[-4:]
            peak_chunk = int(np.argmax([chunk['reward_peak'] for chunk in chunks]))
            trigger_chunks = [chunk for chunk in chunks if chunk['alpha'] >= .5]
            rows.append({
                'split': split, 'sequence': int(sequence), 'path': sequences[sequence],
                'hit': bool(max(chunk['reward_peak'] for chunk in chunks) >= .9),
                'peak_chunk': peak_chunk, 'peak': chunks[peak_chunk],
                'first_trigger': trigger_chunks[0] if trigger_chunks else None,
                'chunks': chunks,
            })
    compact = {}
    for split in ('success', 'failure'):
        selected = [row for row in rows if row['split'] == split]
        hits = [row for row in selected if row['hit']]
        compact[split] = {
            'count': len(selected), 'hits': len(hits),
            'hit_first_triggers': [row['first_trigger'] for row in hits],
            'hit_paths': [row['path'] for row in hits],
        }
    report = {
        'format': 'strict-track2-v252-recursive-trigger-analysis-v1',
        'summary': compact,
        'rows': rows,
        'guards': {'public_data_only': True, 'policy_modified': False, 'hidden_or_final_data': False, 'real_submission': False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(compact, indent=2))


if __name__ == '__main__':
    main()
