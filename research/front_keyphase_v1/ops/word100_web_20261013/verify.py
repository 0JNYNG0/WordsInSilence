"""Camera-free API lifecycle and independent input/logit checks, using TRAINING examples only."""
import copy
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np
import torch

U = Path(__file__).resolve().parent
BASE = U.parents[1]
sys.path[:0] = [str(BASE/'src'), str(BASE.parent/'sign_model_pilot_v1/src')]
from fastapi.testclient import TestClient
from wis_lab.service import app as service, word100, neural
from wis_lab.core.observation import Observation, empty_frames, sha256_file
from pilot.model_masked_v1 import MaskedSignTCN
torch.set_num_threads(4)

def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def save(name,x): (U/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf-8')
checks=[]
def check(name, cond):
    assert cond,name
    checks.append(name)
    save('VERIFY_PROGRESS.json',dict(last_check=name,passed=len(checks)))
    print(name,flush=True)

class FakeCamera:
    clock_id='offline_api_test'
    shadow_assignment=False
    begins=0
    def begin_recording(self, **kw): self.begins+=1
    def request_stop(self, reason): return time.perf_counter_ns()
    def take(self): return copy.deepcopy(obs),dict(end_candidate_ns=time.perf_counter_ns(),end_reason='manual')
    def stop(self): pass
    def end_candidate(self): return None
    def status(self): return {'recording':False,'error':None}

cfg=word100.config(); items=cfg['items']
check('12 practice items,100 recognizer classes',len(items)==12 and len(cfg['labels'])==100)
ck=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True)
model=MaskedSignTCN(n_classes=100,in_channels=280,**ck['config'])
model.load_state_dict(ck['state_dict']);model.eval()
inference=[]
for item in items[:2]:
    obs=Observation.load(item['observation'])
    r=word100.recognise(obs)
    record=read(Path('W:/WordsInSilence/data/AIHub_103/word100_inputs_20261013/records')/(item['clip_id']+'.json'))
    with np.load(record['channel_file'],allow_pickle=False) as z: x=z['x_whole']
    with torch.inference_mode(): direct=model(torch.from_numpy(x[None]),torch.ones((1,len(x))))[0].numpy()
    np.testing.assert_allclose(r['logits'],direct,rtol=1e-6,atol=1e-6)
    check('all100 inference matches frozen prepared input '+item['word_id'],r['candidate_count']==100 and r['accepted'] is None)
    inference.append(dict(word_id=item['word_id'],max_abs_diff=float(np.max(np.abs(direct-r['logits'])))))
obs=Observation.load(items[0]['observation'])
empty=Observation(**empty_frames(5));empty.timestamp_us[:]=np.arange(5)*33333
check('empty hands refused, not classified',word100.recognise(empty)['model_status']=='INSUFFICIENT_OBSERVATION')
tbase=dict(research_model=word100.identity(),reference_binding=word100.binding(items[0]['word_id'],items[0]['clip_id']))
r1=word100.result(obs,dict(tbase,requested_word_id=items[0]['word_id']))
r2=word100.result(obs,dict(tbase,requested_word_id=items[1]['word_id']))
check('requested word cannot change logits',r1['practice100']['logits']==r2['practice100']['logits'])

service.store=service.Store()
client=TestClient(service.app)
with patch.object(service.CameraWorker,'start',side_effect=AssertionError('REAL CAMERA FORBIDDEN')):
    check('catalog does not open camera',client.get('/api/v1/practice100/catalog').status_code==200)
    check('unknown video blocked',client.get('/api/v1/practice100/references/UNKNOWN/clip.mp4').status_code==404)
    rr=client.get('/api/v1/practice100/references/'+items[0]['word_id']+'/clip.mp4',headers={'Range':'bytes=0-31'})
    # This installed Starlette returns200/full files (also valid for an optional Range).
    # Verify bytes in either case; native browser replay/seek is checked separately.
    expected=Path(items[0]['video_path']).read_bytes()
    check('reference video exact served bytes',
          (rr.status_code==206 and rr.content==expected[:32]) or (rr.status_code==200 and rr.content==expected))
    s=client.post('/api/v1/sessions',json={'open_camera':False,'participant_id':'SYNTHETIC_API_QA','client_id':'qa'}).json()
    sid=s['session_id']; session=service.store.sessions[sid]
    # Keep mock data out of human runs and reports.
    session['run_dir']=str(U/'synthetic_api_session');Path(session['run_dir'],'trials').mkdir(parents=True,exist_ok=True)
    cam=FakeCamera();service.store.camera=cam;service.store.camera_session=sid
    payload=dict(client_id='qa',request_id='qa1',requested_word_id=items[0]['word_id'],
                 demonstration_clip_id=items[0]['clip_id'],research_model_key=word100.KEY)
    bad=dict(payload,demonstration_clip_id=items[1]['clip_id'])
    check('mismatched reference cannot start recording',client.post(f'/api/v1/sessions/{sid}/trials',json=bad).status_code==400 and cam.begins==0)
    tt=client.post(f'/api/v1/sessions/{sid}/trials',json=payload);check('valid explicit trial',tt.status_code==200)
    tid=tt.json()['trial_id']
    check('duplicate request starts once',client.post(f'/api/v1/sessions/{sid}/trials',json=payload).json()['deduplicated'] and cam.begins==1)
    with patch.object(service,'predict',side_effect=AssertionError('OLD18_DTW_FORBIDDEN')),patch.object(neural,'recognise',side_effect=AssertionError('OLD_MODEL_FORBIDDEN')):
        rr=client.post(f'/api/v1/trials/{tid}/stop',json={});check('new model finish bypasses old18 DTW',rr.status_code==200)
    result=rr.json()['result'];save('MOCK_RESULT.json',dict(result,verification_origin='TRAINING_REFERENCE_MOCK_NOT_HUMAN_ACCURACY'))
    check('finished result preserved without correctness verdict',result['accepted'] is None and result['execution_assessment']=='UNCONFIGURED' and result['practice100']['candidate_count']==100)
    rp=Path(session['run_dir'])/'trials'/f'{tid}.result.json'
    check('result file persisted before FEEDBACK',rp.exists() and rr.json()['state']=='FEEDBACK')
    for value,actual in [('yes',items[0]['word_id']),('no',None)]:
        rv=client.post(f'/api/v1/trials/{tid}/review',json=dict(session_id=sid,actual_word_id=actual,followed_guidance=value,followed_demonstration_form=None))
        check('review stored '+value,rv.status_code==200 and rv.json()['stored'])
    check('foreign session review blocked',client.post(f'/api/v1/trials/{tid}/review',json=dict(session_id='other',actual_word_id=None,followed_guidance='no')).status_code==409)
    tt=client.post(f'/api/v1/sessions/{sid}/trials',json=dict(payload,request_id='qa_cancel')).json()
    check('cancel never gives success',client.post(f"/api/v1/trials/{tt['trial_id']}/cancel",json={}).json()['result'] is None)
    check('close releases camera',client.post(f'/api/v1/sessions/{sid}/close',json={}).status_code==200 and service.store.camera is None)
    check('legacy references still18',client.get('/api/v1/references').status_code==200)
guard=read(BASE/'ops/word100_training_20261013/PRODUCTION_GUARD.json')
check('production checkpoint hashes unchanged',all(sha256_file(p)==h for p,h in guard.items()))
check('neural adapter source unchanged',sha256_file(BASE/'src/wis_lab/service/neural.py')==read(BASE/'ops/word100_training_20261013/FROZEN_EXPERIMENT.json')['source_sha256'][str(BASE/'src/wis_lab/service/neural.py')])
save('CHECK.json',dict(status='PASS',checks=checks,input_parity=inference,real_camera_opened=False,training_run=False,
     limitations=['Mock lifecycle + frozen training inputs only, not live-user accuracy or handshape validation']))
print(json.dumps({'status':'PASS','checks':len(checks)}))
