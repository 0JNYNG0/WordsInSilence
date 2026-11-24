"""CLI -> selected contract -> real references -> HTTP replay, without camera or NN reruns.

For a staged code check, WIS_SERVICE_BASE_ROOT points to the unchanged data/dependency root.
The NN output is explicitly stubbed: this tests feedback transport, not model performance.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib, importlib, importlib.util, json, os, sys, tempfile

CODE=Path(__file__).resolve().parents[1]
BASE=Path(os.environ.get('WIS_SERVICE_BASE_ROOT',str(CODE)))
sys.path.insert(0,str(CODE/'src'))
import wis_lab
if BASE!=CODE: wis_lab.__path__.append(str(BASE/'src/wis_lab'))
from wis_lab import cli
cli.ROOT=str(BASE);cli.CONFIG_DIR=str(BASE/'config')
package=importlib.import_module('wis_lab.service')
spec=importlib.util.spec_from_file_location('wis_lab.service.app',CODE/'src/wis_lab/service/app.py')
svc=importlib.util.module_from_spec(spec);sys.modules[spec.name]=svc;spec.loader.exec_module(svc)
package.app=svc
from fastapi.testclient import TestClient
import uvicorn

path=BASE/'config/observed_feedback_candidate_v1.json'
default=BASE/'config/contract.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
default_before=sha(default)
camera_calls=[]
def no_camera(*args,**kwargs):
    camera_calls.append(True)
    raise AssertionError('Camera must not open during this test')

with patch.object(svc.CameraWorker,'start',no_camera), patch.object(uvicorn,'run') as run:
    assert cli.main(['serve','--contract',str(path)])==0
    run.assert_called_once()
    assert run.call_args.args[0] is svc.app
    assert svc.store.contract['feedback']['observed_movement_structure'] is True
    assert svc.store.lineage['contract_id']==svc.store.contract['contract_id']
    assert svc.store.lineage['selector']=='uniform_time:'+str(svc.store.contract['sampling_A']['budget'])
    with TestClient(svc.app) as client:
        health=client.get('/api/v1/health')
        assert health.status_code==200
        info=health.json()
        assert info['camera'] is None and info['camera_session_id'] is None
        assert info['contract_sha256']==sha(path)
        assert info['feedback_policy']['observed_movement_structure'] is True
        refs=client.get('/api/v1/references').json()
        assert refs['prepared_count']==18
        # One public stored input tests HTTP serialization and selected-reference binding.
        with patch.object(svc.neural,'recognise',return_value={
            'model_status':'NOT_RUN_TRANSPORT_SMOKE','ranking':None,'is_a_verdict':False}) as nn_stub:
            response=client.post('/api/v1/replays',json={'observation_id':'ALDA','compare_to_word_id':'ALDA'})
            assert response.status_code==200,response.text[:1000]
            data=response.json()
            nn_stub.assert_called_once()
        assert data['counts_as_human_attempt'] is False
        assert data['contract_id']==info['contract_id']
        assert data['accepted'] is None
        follow=data['demonstration_following']
        assert follow['status']=='AVAILABLE'
        movement=follow['axis_deviation']['movement_observation']
        assert movement['status']=='AVAILABLE' and movement['accepted'] is None
        assert 'movement' not in follow['axis_deviation']['axes']
        assert movement['reference']['hands']['R']['vertical_turns']==3
        assert svc.store.camera is None and not svc.store.sessions and not svc.store.trials
    try:
        svc.Store(str(BASE/'config/does_not_exist_service_test.json'))
    except FileNotFoundError: pass
    else: raise AssertionError('Invalid explicit contract silently fell back to default')

    # Reference generation must use the selected sampling budget, not silently load default.
    cfg=json.loads(path.read_text(encoding='utf-8'));cfg['sampling_A']['budget']=16
    refs,lineage=cli.load_reference_set(contract=cfg)
    assert lineage['selector']=='uniform_time:16'
    assert all(len(r['features'])<=16 for r in refs) and any(len(r['features'])==16 for r in refs)
    default_store=svc.Store()
    assert default_store.contract['contract_id']==json.loads(default.read_text(encoding='utf-8'))['contract_id']
    assert not default_store.contract.get('feedback',{}).get('observed_movement_structure')

assert not camera_calls and sha(default)==default_before
print(json.dumps({'status':'PASS_CANDIDATE_CLI_AND_HTTP_FEEDBACK', 'public_replays':1,
                  'camera_calls':0,'neural_inference_calls':0,'neural_stub_calls':1,
                  'human_attempts_created':0,'default_config_unchanged':True,
                  'selected_contract_and_reference_budget_match':True},ensure_ascii=False))
