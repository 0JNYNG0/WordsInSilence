"""Independent numeric checks + read-only integration; never open a real camera."""
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

U=Path(__file__).resolve().parent;B=U.parents[1]
sys.path.insert(0,str(B/'src'))
from wis_lab.service import word100 as W,word100_analysis as A,app as service
from wis_lab.core.observation import Observation,sha256_file
from wis_lab.core import features as F
from wis_lab.core.dtw import dtw
from fastapi.testclient import TestClient

def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
checks=[]
def check(name,value):
    assert value,name
    checks.append(name);print(name,flush=True)

refs=A.references();check('100 training-only demo references, original12 practice unchanged',len(refs)==100 and len(W.config()['items'])==12)
ref=Observation.load(refs['WORD0003']['observation'])
base=A.compare(ref,ref)
check('self comparison zero in every measured axis',base['status']=='OK' and all(m['value']==0 for m in base['metrics']))
scaled=copy.deepcopy(ref);scaled.xy_px=scaled.xy_px*1.7+np.array([81,144],np.float32)
res=A.compare(scaled,ref)
check('translation/body scale invariance',res['status']=='OK' and max(m['value'] for m in res['metrics'])<0.001)
q=A._sample(scaled);r=A._sample(ref);oracle=dtw(q,r,band_ratio=A.BAND)
check('DTW matches existing independent accumulated-path implementation',abs(res['dtw_distance']-oracle['distance'])<1e-7)

# Repeat one fully observed measured frame: closed-form spatial/finger changes have no timing ambiguity.
frame=next(i for i in range(ref.n_frames) if ref.usable[i,[11,12]].all() and ref.usable[i,33:75].all() and
           np.isfinite(F.bend_angles(ref.xy_px[i,54:75])).all() and np.isfinite(F.bend_angles(ref.xy_px[i,33:54])).all())
static=copy.deepcopy(ref)
for field in ('xy_px','observed','usable','conf','hand_score'):
    setattr(static,field,np.repeat(getattr(ref,field)[frame:frame+1],12,axis=0))
static.timestamp_us=np.arange(12,dtype=np.int64)*33333;static.frame_id=np.arange(12,dtype=np.int64)
shift=copy.deepcopy(static);shift.xy_px[:,54:75,0]+=20
res=A.compare(shift,static);metrics={x['key']:x for x in res['metrics']}
width=np.linalg.norm(static.xy_px[0,11]-static.xy_px[0,12]);expected=20/float(width)*100
check('position reports shoulder-width units independently',abs(metrics['location']['sides'][1]['value']-expected)<0.001)
check('constant hand displacement does not become trajectory movement',metrics['movement']['sides'][1]['value']<0.001)
bent=copy.deepcopy(static);bent.xy_px[:,62,0]+=20  # right index tip54+8
res=A.compare(bent,static)
def angles(points):
    vals=[]
    for ch in [(0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)]:
        for k in range(1,4):
            u=points[ch[k-1]]-points[ch[k]];v=points[ch[k+1]]-points[ch[k]]
            vals.append(np.degrees(np.arccos(np.clip(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)),-1,1))))
    return np.array(vals)
expected=np.abs(angles(bent.xy_px[0,54:75])-angles(static.xy_px[0,54:75])).mean()
measured=res['metrics'][0]['sides'][1]['value']
check('15 hand flexion angles match separate trigonometry',expected>0 and abs(measured-expected)<0.01)
missing=copy.deepcopy(static);missing.xy_px[:,33:54]=np.nan;missing.usable[:,33:54]=False;missing.observed[:,33:54]=False
res=A.compare(missing,static)
check('missing hand not represented as zero deviation',res['status']=='OK' and all(x['sides'][0]['value'] is None for x in res['metrics']))
missing.xy_px[:,54:75]=np.nan;missing.usable[:,54:75]=False;missing.observed[:,54:75]=False
check('no hands gives insufficient rather than a match',A.compare(missing,static)['status']=='INSUFFICIENT')

session='ses_79977f57893d';reports=[]
with (patch.object(service.CameraWorker,'start',side_effect=AssertionError('NO_REAL_CAMERA')),
      patch.object(W,'recognise',side_effect=AssertionError('DO_NOT_RERUN_USER_INFERENCE'))):
    client=TestClient(service.app)
    history=client.get('/api/v1/practice100/history').json()['items']
    rows=[x for x in history if x['session_id']==session]
    check('all3 original attempts remain visible,2 confirmations',len(rows)==3 and sum(x['reviewed'] for x in rows)==2)
    for row in rows:
        r0,path=A.saved_result(session,row['trial_id'])
        url=f"/api/v1/practice100/saved/{session}/{row['trial_id']}"
        response=client.get(url);check('saved result endpoint '+row['trial_id'],response.status_code==200)
        r=response.json();rank=r['practice100']['ranking'];raw=np.asarray(r0['practice100']['logits'])
        indices=sorted(range(100),key=lambda i:(-raw[i],i))[:5]
        check('top5 exactly from original logits '+row['trial_id'],[x['word_id'] for x in rank]==[W.config()['word_codes'][i] for i in indices])
        check('requested demo, not wrong top1, used for DTW '+row['trial_id'],r['analysis']['dtw']['reference_word_id']==r0['requested_word_id'])
        check('recognition unchanged '+row['trial_id'],r0['practice100']['logits']==r['practice100']['logits'] and r['accepted'] is None)
        capture=client.get(url+'/capture');check('stored replay available '+row['trial_id'],capture.status_code==200 and len(capture.json()['frames'])>0)
        first=client.get(url+'/frames/0');check('saved JPEG frame '+row['trial_id'],first.status_code==200 and first.content[:2]==b'\xff\xd8')
        reports.append(r)
    check('foreign-session result IDs refused',client.get('/api/v1/practice100/saved/ses_000000000000/'+rows[0]['trial_id']).status_code==404)
    check('invalid frame refused',client.get(url+'/frames/-1').status_code==404)
    check('unknown reference refused',client.get('/api/v1/practice100/references/WORD9999/clip.mp4').status_code==404)
    # Top3 includes labels outside the12 practice dropdown; verify those exact video bytes too.
    words={x['word_id'] for r in reports for x in r['practice100']['ranking'][:3]}
    for code in sorted(words):
        response=client.get(f'/api/v1/practice100/references/{code}/clip.mp4')
        import hashlib
        check('top3 reference bytes '+code,response.status_code==200 and hashlib.sha256(response.content).hexdigest()==refs[code]['source_sha256'])

guard=read(U/'ORIGINAL_GUARD.json')
check('original recordings/results/reviews and model bytes unchanged',all(sha256_file(p)==h for p,h in guard.items()))
write(U/'SAVED_ANALYSES.json',reports)
write(U/'CHECK.json',dict(status='PASS',checks=checks,real_camera_opened=False,retraining=False,
    user_inference_rerun=False,scope='Geometry correctness and integration, not validated sign correctness or causal misrecognition diagnosis'))
print(json.dumps(dict(status='PASS',checks=len(checks))))
