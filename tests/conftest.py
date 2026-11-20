import gzip
import importlib.util
import json
from pathlib import Path
import sys
import uuid

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def load_module():
    def load(relative_path):
        name = 'test_target_' + uuid.uuid4().hex
        spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    return load


@pytest.fixture
def tiny_episode(tmp_path):
    """Synthetic labels only; no game data, video codecs or GPU required."""
    ep = tmp_path / 'dataset/gta5/synthetic_case'
    proxy = ep / 'proxy/main'
    proxy.mkdir(parents=True)
    times = np.arange(4, dtype='int64') * 33333
    xyz = np.full((4, 1, 20, 3), np.nan, dtype='float32')
    xyz[:, 0, 0] = np.c_[np.arange(4), np.zeros(4), np.ones(4)]
    valid = np.isfinite(xyz).all(axis=-1)
    roots = xyz[:, :, 0]
    meta = dict(game='gta5', case_id=ep.name, frame_count=4, duration_s=.099999,
        views={'main': {'source_video': str(ep/'source.mp4'), 'camera_source': 'synthetic'}},
        source_training_eligibility=False, site_id='synthetic',
        point_track_semantics={'entities': 1, 'points_per_entity': 2})
    (ep/'episode.json').write_text(json.dumps(meta), encoding='utf-8')
    (ep/'qa.json').write_text(json.dumps(dict(static_triangles=0, track_only_entities=0,
        visual_review_status='required')), encoding='utf-8')
    (ep/'program.json').write_text('{}', encoding='utf-8')
    (proxy/'render_report.json').write_text(json.dumps(dict(complete_episode=True,
        frames=4, resolution=[3, 2], preview_fps=30, chunks=[])), encoding='utf-8')
    np.savez_compressed(ep/'timeline.npz', time_us=times, source_video_frame=np.arange(4))
    np.savez_compressed(ep/'joints.npz', world_xyz=xyz, valid=valid, root_world=roots,
        root_valid=np.ones((4, 1), dtype=bool), velocity_world=np.ones((4, 1, 3), dtype='float32'),
        velocity_valid=np.ones((4, 1), dtype=bool), instance_ids=np.array([1]),
        joint_names=np.array(['joint_'+str(i) for i in range(20)]),
        parent_joint=np.array([-1]+[0]*19), time_us=times)
    np.savez_compressed(ep/'points.npz', world_xyz=xyz[:, :, :2], valid=valid[:, :, :2],
        instance_ids=np.array([1]), point_ids=np.arange(2), tracking_kind=np.array('synthetic_points'))
    with gzip.open(ep/'frames.jsonl.gz', 'wt', encoding='utf-8') as f:
        for i in range(4):
            f.write(json.dumps(dict(index=i, cameras={'main': {'view': np.eye(4).tolist(),
                'projection': np.eye(4).tolist()}}))+'\n')
    return ep

