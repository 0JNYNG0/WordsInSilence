"""Replay frozen current-view caches; no detector, classifier or training."""
import json,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
from visibility import HandVisibility,PARAMS
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
scope=read(U/'SCOPE.json');visual=read(U/'VISUAL_LABELS.json')
assert sha(U/'visibility.py')==scope['code_sha256'] and PARAMS==scope['parameters']
assert visual['predictions_seen'] is False
assert read(U/'SYNTHETIC_TESTS.json')['status']=='PASS'
for p,h in scope['pins'].items():assert sha(p)==h,p
out=[];counts={g:{'whole':Counter(),'core':Counter()} for g in ['known36','additional12']}
for r in scope['records']:
    raw=np.load(r['paths']['raw']);obs=np.load(r['paths']['candidate']);ass=np.load(r['paths']['assignment'])['sides']
    assert np.array_equal(raw['timestamp_us'],obs['timestamp_us'])
    assert np.array_equal(raw['frame_id'],obs['frame_id'])
    machine=HandVisibility();frames=[]
    for f,t in enumerate(raw['timestamp_us']):
        n=int(raw['raw_count'][f]);sides=ass[f,:n]
        pred=machine.step(obs['xy_px'][f],obs['usable'][f],raw['pose_conf'][f],int(t),int(raw['width'][f]),int(raw['height'][f]),
                          assigned=(bool(0 in sides),bool(1 in sides)),unassigned_raw=bool(any(s<0 for s in sides)))
        core=any(a<=t/1e6<=b for a,b in r['intervals'])
        for z in pred:
            counts[r['group']]['whole'][z['state']]+=1
            if core:counts[r['group']]['core'][z['state']]+=1
        frames.append({'frame':f,'timestamp_us':int(t),'core':core,'hands':pred})
    out.append({'clip_id':r['clip_id'],'label':r['label'],'word_code':r['word_code'],'group':r['group'],'frames':frames})
lookup={r['clip_id']:r for r in out};comparisons=[]
for z in visual['labels']:
    pred=lookup[z['clip_id']]['frames'][z['frame']]['hands'][z['side']]
    comparisons.append({**z,'prediction':pred})
result={'scope_sha256':sha(U/'SCOPE.json'),'labels_sha256':sha(U/'VISUAL_LABELS.json'),
        'records':out,'counts':counts,'visual_comparisons':comparisons,'new_training':0,'new_detection':0,'new_classifier_inference':0,'user_recording_reads':0}
with (U/'PREDICTIONS.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
for p,h in scope['pins'].items():assert sha(p)==h,p
tables={}
for origin in ['prior_known_gap_review','additional12_blind_review']:
    tab={}
    for x in comparisons:
        if x['origin']!=origin:continue
        tab.setdefault(x['visual_category'],Counter())[x['prediction']['state']]+=1
    tables[origin]=tab
print(json.dumps({'counts':counts,'comparison':tables,
 'additional':[(x['review_id'],x['visual_category'],x['prediction']['state']) for x in comparisons if x['origin']=='additional12_blind_review']},ensure_ascii=False,indent=2))
