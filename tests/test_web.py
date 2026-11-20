import json
from fastapi.testclient import TestClient


def test_empty_data_root_is_usable(tmp_path, load_module, monkeypatch):
    monkeypatch.setenv('CODEASMEM_DATA_ROOT',str(tmp_path/'missing'))
    monkeypatch.delenv('CODEASMEM_AUDIT',raising=False)
    module=load_module('web/app.py')
    client=TestClient(module.app)
    assert client.get('/').status_code==200
    summary=client.get('/api/summary').json()
    assert summary['episodes']==summary['frames']==summary['point_records']==0
    assert summary['structural_passed']==0
    assert client.get('/api/cases').json()==[]
    assert client.get('/api/code/loader').json()['symbol']=='iter_pairs'


def test_dynamic_catalog_quality_and_local_review(tiny_episode, tmp_path, load_module, monkeypatch):
    root=tiny_episode.parent.parent
    monkeypatch.setenv('CODEASMEM_DATA_ROOT',str(root))
    monkeypatch.delenv('CODEASMEM_AUDIT',raising=False)
    module=load_module('web/app.py')
    module.STATE=tmp_path/'web_state'
    client=TestClient(module.app)
    summary=client.get('/api/summary').json()
    assert summary['episodes']==1 and summary['frames']==4 and summary['point_records']==8
    assert summary['structural_unvalidated']==1 and summary['structural_passed']==0
    base='/api/cases/gta5/synthetic_case'
    assert client.get(base).json()['source_training_eligibility'] is False
    assert client.post(base+'/recipe',json=dict(task='dynamics',start=0,cutoff=3,end=2)).status_code==422
    recipe=client.post(base+'/recipe',json=dict(task='dynamics',start=0,cutoff=1,end=3)).json()
    assert recipe['training_ready'] is False
    assert 'future_ground_truth_proxy' in recipe['excluded_student_inputs']
    response=client.post(base+'/review',json=dict(status='candidate',note='Synthetic test only'))
    assert response.status_code==200
    assert client.get(base).json()['source_training_eligibility'] is False
    assert (module.STATE/'reviews.local.json').is_file()
    assert client.post(base+'/review',json=dict(status='pending'),headers={'Origin':'https://example.invalid'}).status_code==403


def test_present_validation_is_used(tiny_episode, load_module, monkeypatch):
    root=tiny_episode.parent.parent
    (root/'batch_validation.json').write_text(json.dumps(dict(scope='synthetic',cases=[
        dict(game='gta5',case_id=tiny_episode.name,issues=[],visual_quality_approved=False)
    ])),encoding='utf-8')
    monkeypatch.setenv('CODEASMEM_DATA_ROOT',str(root))
    monkeypatch.delenv('CODEASMEM_AUDIT',raising=False)
    client=TestClient(load_module('web/app.py').app)
    summary=client.get('/api/summary').json()
    assert summary['structural_passed']==1 and summary['visual_approved']==0

