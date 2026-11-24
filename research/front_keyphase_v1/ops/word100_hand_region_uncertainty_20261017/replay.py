import json,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
from region import HandRegion,PARAMS,HandVisibility
U=Path(__file__).resolve().parent;B=U.parent/'word100_hand_visibility_20261017'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=read(U/'SCOPE.json');labels=read(U/'VISUAL_LABELS.json')
assert labels['predictions_seen'] is False and PARAMS==s['parameters']
assert sha(U/'region.py')==s['code_sha256']['region.py']
assert sha(B/'visibility.py')==s['code_sha256']['baseline']
assert read(U/'SYNTHETIC_TESTS.json')['status']=='PASS'
for p,h in s['pins'].items():assert sha(p)==h,p
previous={r['clip_id']:r for r in read(B/'PREDICTIONS.json')['records']}
records=[];counts={g:{k:Counter() for k in ['baseline_whole','candidate_whole','baseline_core','candidate_core']} for g in ['previous48','fresh12']}
for r in s['records']:
    raw=np.load(r['paths']['raw']);obs=np.load(r['paths']['candidate']);ass=np.load(r['paths']['assignment'])['sides']
    assert np.array_equal(raw['timestamp_us'],obs['timestamp_us']) and np.array_equal(raw['frame_id'],obs['frame_id'])
    m=HandRegion();b=HandVisibility();rows=[]
    for f,t in enumerate(raw['timestamp_us']):
        sides=ass[f,:int(raw['raw_count'][f])]
        args=(obs['xy_px'][f],obs['usable'][f],raw['pose_conf'][f],int(t),int(raw['width'][f]),int(raw['height'][f]))
        kwargs={'assigned':(bool(0 in sides),bool(1 in sides)), 'unassigned_raw':bool(any(x<0 for x in sides))}
        candidate=m.step(*args,**kwargs)
        baseline=previous[r['clip_id']]['frames'][f]['hands'] if r['clip_id'] in previous else b.step(*args,**kwargs)
        core=any(a<=t/1e6<=z for a,z in r['intervals'])
        for side in (0,1):
            assert candidate[side]['baseline_state']==baseline[side]['state']
            for k,hand in [('baseline',baseline[side]),('candidate',candidate[side])]:
                counts[r['group']][k+'_whole'][hand['state']]+=1
                if core:counts[r['group']][k+'_core'][hand['state']]+=1
        rows.append({'frame':f,'timestamp_us':int(t),'core':core,'baseline':baseline,'hands':candidate})
    records.append({'clip_id':r['clip_id'],'label':r['label'],'group':r['group'],'frames':rows})
lookup={r['clip_id']:r for r in records};comp=[]
for x in labels['labels']:
    f=lookup[x['clip_id']]['frames'][x['frame']]
    comp.append({**x,'baseline':f['baseline'][x['side']],'candidate':f['hands'][x['side']]})
out={'scope_sha256':sha(U/'SCOPE.json'),'labels_sha256':sha(U/'VISUAL_LABELS.json'), 'records':records,'counts':counts,'visual_comparisons':comp,
     'new_training':0,'new_detection':0,'classifier_inference':0,'user_recording_reads':0}
with (U/'PREDICTIONS.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
for p,h in s['pins'].items():assert sha(p)==h
tables={}
for x in comp:
    d=tables.setdefault(x['origin'],{})
    d.setdefault(x['visual_category'],Counter())[(x['baseline']['state']+' -> '+x['candidate']['state'])]+=1
print(json.dumps({'counts':counts,'visual_tables':tables},ensure_ascii=False,indent=2))
