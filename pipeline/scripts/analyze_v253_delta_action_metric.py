#!/usr/bin/env python3
"""Assess an offset-invariant right-arm delta-action retrieval metric."""
import argparse
import json
from pathlib import Path

import numpy as np

from wam_pipeline.arm_routed_autoregressive_runtime import Track2ArmRoutedAutoregressiveUNet
from wam_pipeline.v216_public_knn_blend_runtime import _visual_descriptor


def parse_name(name: str) -> tuple[int, int]:
    episode, start = Path(name).stem.split('_')
    return int(episode[7:]), int(start)


def stats(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {key: float(value) for key, value in zip(
        ('min', 'q25', 'median', 'q75', 'max'), np.quantile(values, (0, .25, .5, .75, 1))
    )}


def corr(left, right):
    left, right = np.asarray(left, float), np.asarray(right, float)
    return float(np.corrcoef(left, right)[0, 1]) if left.std() and right.std() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--capture-details', type=Path, required=True)
    parser.add_argument('--long-run', type=Path, required=True)
    parser.add_argument('--recursive-analysis', type=Path, required=True)
    parser.add_argument('--success-windows', type=Path, required=True)
    parser.add_argument('--failure-windows', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('refusing overwrite')
    with np.load(args.library, allow_pickle=False) as data:
        episodes = data['episode_id'].astype(int)
        visual = data['visual'].astype(np.float32)
        normalized = data['action'].astype(np.float32)
        mean = data['normalization_mean'].astype(np.float32)
        std = data['normalization_std'].astype(np.float32)
    clean = np.flatnonzero(episodes >= 20000)
    raw = normalized[clean].reshape(-1, 12, 14) * std + mean
    library_delta = np.diff(raw[:, :, 7:13], axis=1)
    delta_scale = np.maximum(library_delta.reshape(-1, 6).std(axis=0), 1e-6)
    library_feature = (library_delta / delta_scale).reshape(len(clean), -1)

    def feature(context, history, future):
        query = np.concatenate((history, future), axis=0)
        query_feature = (np.diff(query[:, 7:13], axis=0) / delta_scale).reshape(-1)
        delta_distance = ((library_feature - query_feature) ** 2).mean(axis=1)
        visual_distance = ((visual[clean] - _visual_descriptor(context[-1])) ** 2).mean(axis=1)
        score = visual_distance / max(float(np.median(visual_distance)), 1e-9)
        score += 4 * delta_distance / max(float(np.median(delta_distance)), 1e-9)
        local = int(np.argmin(score))
        query_delta = future[-1, 7:13] - history[-1, 7:13]
        library_delta_terminal = raw[local, -1, 7:13] - raw[local, -9, 7:13]
        denominator = float(np.linalg.norm(query_delta) * np.linalg.norm(library_delta_terminal))
        alignment = float(np.dot(query_delta, library_delta_terminal) / denominator) if denominator > 1e-8 else 0.0
        return {
            'delta_distance': float(delta_distance[local]),
            'delta_ratio': float(delta_distance[local] / max(float(np.median(delta_distance)), 1e-9)),
            'visual_distance': float(visual_distance[local]),
            'visual_ratio': float(visual_distance[local] / max(float(np.median(visual_distance)), 1e-9)),
            'alignment': alignment,
            'base': int(clean[local]),
        }

    capture_rows = []
    for file_index, path in enumerate(sorted(args.capture.glob('rollout_*.npz'))):
        with np.load(path, allow_pickle=False) as data:
            contexts = data['context_frames'].copy()
            histories = data['history_actions'].astype(np.float32)
            futures = data['future_actions'].astype(np.float32)
            texts = list(map(str, json.loads(str(data['instructions_json']))))
        for action_index, (context, history, future, text) in enumerate(zip(contexts, histories, futures, texts)):
            route = Track2ArmRoutedAutoregressiveUNet.active_arm(history, future, text)
            if route == 'right' and history[-1, 13] < .5 and (future[:, 13] < .5).mean() >= .75:
                capture_rows.append({'file': file_index, 'action': action_index, **feature(context, history, future)})
    with np.load(args.capture_details, allow_pickle=False) as data:
        terminal_rewards = data['rewards'][:, -1].astype(float)
        original_alignment = data['alignment'].astype(float)
        original_motion = data['motion'].astype(float)
    if len(capture_rows) != len(terminal_rewards):
        raise RuntimeError('capture order/count mismatch')
    for row, reward, alignment, motion in zip(capture_rows, terminal_rewards, original_alignment, original_motion):
        row['terminal_reward'] = float(reward)
        row['original_alignment'] = float(alignment)
        row['original_motion'] = float(motion)

    recursive = json.loads(args.recursive_analysis.read_text())
    recursive_lookup = {(row['split'], row['path']): row for row in recursive['rows']}
    long_rows = []
    long_all_rows = []
    for split, window_root in [('success', args.success_windows), ('failure', args.failure_windows)]:
        baseline_path = args.long_run / 'audit' / f'public_{split}_baseline.npz'
        candidate_path = args.long_run / 'audit' / f'public_{split}_candidate.npz'
        with np.load(baseline_path, allow_pickle=False) as data:
            paths = data['path'].astype(str)
            arms = data['arm_right'].astype(bool)
        with np.load(candidate_path, allow_pickle=False) as data:
            frames = data['candidate']
        index = {parse_name(path.name): path for path in window_root.glob('episode*_*.npz')}
        for sequence in np.flatnonzero(arms):
            name = paths[sequence]
            recursive_row = recursive_lookup[(split, name)]
            wanted = recursive_row['first_trigger']['chunk'] if recursive_row['first_trigger'] else None
            episode, start = parse_name(name)
            with np.load(index[(episode, start)], allow_pickle=False) as data:
                context = data['context_frames'].copy()
                history = data['history_actions'].astype(np.float32).copy()
            for chunk in range(16):
                with np.load(index[(episode, start + 8 * chunk)], allow_pickle=False) as data:
                    future = data['future_actions'].astype(np.float32).copy()
                row = {
                    'split': split, 'path': name, 'chunk': chunk,
                    'original_post': recursive_row['chunks'][chunk]['post'],
                    'original_alignment': recursive_row['chunks'][chunk]['alignment'],
                    'original_motion': recursive_row['chunks'][chunk]['motion'],
                    **feature(context, history, future),
                }
                long_all_rows.append(row)
                if chunk == wanted:
                    long_rows.append(row)
                prediction = frames[sequence, chunk * 8:(chunk + 1) * 8]
                context = np.concatenate((context, prediction), axis=0)[-5:]
                history = np.concatenate((history, future), axis=0)[-4:]

    report = {
        'format': 'strict-track2-v253-delta-action-metric-analysis-v1',
        'delta_scale': delta_scale.tolist(),
        'capture': {
            'count': len(capture_rows),
            'delta_distance': stats([row['delta_distance'] for row in capture_rows]),
            'delta_ratio': stats([row['delta_ratio'] for row in capture_rows]),
            'reward_vs_negative_distance_correlation': corr(terminal_rewards, [-row['delta_distance'] for row in capture_rows]),
            'success_like_distance': stats([row['delta_distance'] for row in capture_rows if row['terminal_reward'] >= .1]),
        },
        'long_first_trigger': {
            split: {
                'count': sum(row['split'] == split for row in long_rows),
                'delta_distance': stats([row['delta_distance'] for row in long_rows if row['split'] == split]),
                'delta_ratio': stats([row['delta_ratio'] for row in long_rows if row['split'] == split]),
            } for split in ('success', 'failure')
        },
        'capture_rows': capture_rows,
        'long_rows': long_rows,
        'long_all_rows': long_all_rows,
        'guards': {'public_data_only': True, 'policy_modified': False, 'hidden_or_final_data': False, 'real_submission': False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('capture', 'long_first_trigger')}, indent=2))


if __name__ == '__main__':
    main()
