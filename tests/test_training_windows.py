import json
import numpy as np
import pytest


def make_recipe(path, episode, task='dynamics', cutoff=1, end=3):
    # Exercise resolution relative to the recipe, independently of shell cwd.
    value = dict(schema='worldbridge.training_window.v1', episode='dataset/gta5/synthetic_case',
        task=task, condition='control', context={'start': 0, 'end_inclusive': cutoff},
        target={'start': cutoff+1, 'end_inclusive': end})
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


def fake_pairs(ep, condition):
    for i in range(4):
        yield dict(frame=i, rgb=np.full((2,3,3), i, dtype='uint8'),
                   proxy=np.full((2,3,3), 200+i, dtype='uint8'), time_us=i*33333)


def test_dynamics_keeps_future_out_of_inputs(tiny_episode, tmp_path, load_module, monkeypatch):
    module = load_module('scripts/prepare_window.py')
    monkeypatch.setattr(module, 'iter_pairs', fake_pairs)
    recipe = make_recipe(tmp_path/'window.json', tiny_episode)
    out = tmp_path/'sample'
    result = module.prepare(recipe, out)
    with np.load(out/'student_inputs.npz') as x, np.load(out/'supervision.npz') as y:
        assert set(x.files) == {'past_rgb','past_time_us'}
        assert x['past_rgb'][:,0,0,0].tolist() == [0,1]
        assert y['future_rgb'][:,0,0,0].tolist() == [2,3]
        assert x['past_time_us'].max() < y['future_time_us'].min()
        assert not y['joint_valid_finite'][:,:,1:].any()
    assert result['input_mode']=='past_rgb_and_time_only'
    assert result['ready_for_training'] is False
    assert result['resolved_episode']==str(tiny_episode.resolve())


def test_renderer_allows_known_future_proxy(tiny_episode, tmp_path, load_module, monkeypatch):
    module = load_module('scripts/prepare_window.py')
    monkeypatch.setattr(module, 'iter_pairs', fake_pairs)
    recipe = make_recipe(tmp_path/'renderer.json', tiny_episode, task='renderer')
    module.prepare(recipe, tmp_path/'sample')
    with np.load(tmp_path/'sample/student_inputs.npz') as x:
        assert x['future_proxy'][:,0,0,0].tolist()==[202,203]
        assert x['future_camera_view'].shape==(2,4,4)
        assert not any('oracle' in key for key in x.files)


def test_oracle_is_explicit_and_past_only(tiny_episode, tmp_path, load_module, monkeypatch):
    module = load_module('scripts/prepare_window.py')
    monkeypatch.setattr(module, 'iter_pairs', fake_pairs)
    recipe=make_recipe(tmp_path/'oracle.json', tiny_episode)
    result=module.prepare(recipe,tmp_path/'oracle',oracle_history=True)
    with np.load(tmp_path/'oracle/student_inputs.npz') as x:
        assert x['past_oracle_root_world'][:,0,0].tolist()==[0,1]
        assert not any(key.startswith('future_') for key in x.files)
    assert result['input_mode']=='privileged_historical_engine_state'


def test_invalid_boundary_and_overwrite_are_rejected(tiny_episode, tmp_path, load_module, monkeypatch):
    module=load_module('scripts/prepare_window.py')
    monkeypatch.setattr(module,'iter_pairs',fake_pairs)
    recipe=make_recipe(tmp_path/'invalid.json',tiny_episode,cutoff=3,end=2)
    with pytest.raises(ValueError,match='boundary'):
        module.prepare(recipe,tmp_path/'sample')
    recipe=make_recipe(tmp_path/'valid.json',tiny_episode)
    module.prepare(recipe,tmp_path/'sample')
    with pytest.raises(ValueError,match='never overwritten'):
        module.prepare(recipe,tmp_path/'sample')

