"""Materialize a reviewed window without mixing future labels into inputs.

This prepares arrays, not model weights. Default dynamics inputs contain only
historical RGB and time. --oracle-history is an explicitly privileged baseline.
"""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from worldbridge.data import iter_pairs


def prepare(recipe_path, out, oracle_history=False):
    recipe_path = Path(recipe_path).resolve()
    recipe = json.loads(recipe_path.read_text(encoding='utf-8'))
    if recipe.get('schema') != 'worldbridge.training_window.v1':
        raise ValueError('Unsupported recipe schema')
    ep = Path(recipe['episode'])
    if not ep.is_absolute():
        ep = (recipe_path.parent / ep).resolve()
    meta = json.loads((ep / 'episode.json').read_text(encoding='utf-8'))
    start = int(recipe['context']['start'])
    cutoff = int(recipe['context']['end_inclusive'])
    end = int(recipe['target']['end_inclusive'])
    if not 0 <= start <= cutoff < end < meta['frame_count']:
        raise ValueError('Invalid causal frame boundary')
    if recipe['target']['start'] != cutoff + 1:
        raise ValueError('Future must start immediately after cutoff')
    task = recipe['task']
    if task not in ('dynamics', 'renderer'):
        raise ValueError('Unsupported task')
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be new or empty; existing samples are never overwritten')
    historical_rgb, future_rgb, future_proxy, past_time, future_time = [], [], [], [], []
    for pair in iter_pairs(ep, condition=recipe['condition']):
        i = pair['frame']
        if i > end:
            break
        if i < start:
            continue
        if i <= cutoff:
            historical_rgb.append(pair['rgb'])
            past_time.append(pair['time_us'])
        else:
            future_rgb.append(pair['rgb'])
            future_time.append(pair['time_us'])
            if task == 'renderer':
                future_proxy.append(pair['proxy'])
    if len(historical_rgb) != cutoff-start+1 or len(future_rgb) != end-cutoff:
        raise ValueError('Decoded frame count does not match recipe')
    inputs = dict(past_rgb=np.stack(historical_rgb), past_time_us=np.asarray(past_time, dtype='int64'))
    labels = dict(future_rgb=np.stack(future_rgb), future_time_us=np.asarray(future_time, dtype='int64'))
    hs, fs = slice(start, cutoff+1), slice(cutoff+1, end+1)
    with np.load(ep / 'joints.npz', allow_pickle=False) as z:
        for name in ('world_xyz', 'valid', 'root_world', 'root_valid', 'velocity_world', 'velocity_valid'):
            labels['future_' + name] = z[name][fs].copy()
            if oracle_history:
                inputs['past_oracle_' + name] = z[name][hs].copy()
        labels['joint_valid_finite'] = labels['future_valid'] & np.isfinite(labels['future_world_xyz']).all(axis=-1)
        labels['instance_ids'] = z['instance_ids'].copy()
        labels['joint_names'] = z['joint_names'].copy()
        labels['parent_joint'] = z['parent_joint'].copy()
    with np.load(ep / 'points.npz', allow_pickle=False) as z:
        labels['future_point_world_xyz'] = z['world_xyz'][fs].copy()
        labels['future_point_valid'] = z['valid'][fs] & np.isfinite(labels['future_point_world_xyz']).all(axis=-1)
        labels['point_instance_ids'] = z['instance_ids'].copy()
        labels['point_ids'] = z['point_ids'].copy()
        labels['tracking_kind'] = z['tracking_kind'].copy()
    # Future camera is an explicit condition for renderer training, a target-only
    # descriptor for dynamics. No future camera is silently fed to a predictor.
    future_views, future_projections = [], []
    with gzip.open(ep / 'frames.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            frame = json.loads(line)
            i = frame['index']
            if cutoff < i <= end:
                camera = frame['cameras']['main']
                future_views.append(camera['view'])
                future_projections.append(camera['projection'])
            if i >= end:
                break
    cameras = dict(future_camera_view=np.asarray(future_views, dtype='float32'),
                   future_camera_projection=np.asarray(future_projections, dtype='float32'))
    if task == 'renderer':
        inputs.update(future_proxy=np.stack(future_proxy), **cameras)
    else:
        labels.update(cameras)
        assert not any(k.startswith('future_') for k in inputs)
    assert len(inputs['past_rgb']) == cutoff-start+1
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'student_inputs.npz', **inputs)
    np.savez_compressed(out / 'supervision.npz', **labels)
    provenance = dict(recipe=recipe, resolved_episode=str(ep), input_mode='privileged_historical_engine_state' if oracle_history else 'past_rgb_and_time_only',
        array_resolution='384x216; no chosen video model preprocessing applied',
        input_shapes={k: list(v.shape) for k, v in inputs.items()},
        label_shapes={k: list(v.shape) for k, v in labels.items()},
        model_trained=False, source_modified=False,
        ready_for_training=False,
        remaining=['visual_quality_review', 'scene_identity_session_split', 'model_specific_preprocessing_and_trainer'])
    (out / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--oracle-history', action='store_true', help='Explicit privileged historical state baseline')
    args = parser.parse_args()
    result = prepare(args.recipe, args.out, args.oracle_history)
    print(json.dumps({'out': str(Path(args.out).resolve()), 'input_mode': result['input_mode'],
                      'model_trained': False}, ensure_ascii=True))
