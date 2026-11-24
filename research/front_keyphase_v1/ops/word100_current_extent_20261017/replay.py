import json,hashlib,sys
from pathlib import Path
from collections import Counter
import numpy as np
from spatial import extend_evidence,PARAMS
U=Path(__file__).resolve().parent;O=U.parent;B=O/'word100_hand_evidence_split_20261017'
sys.path.insert(0,str(B));from separate import SeparatedHandDiagnostics
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=read(U/'SCOPE.json');assert s['parameters']==PARAMS
for p,h in {**s['source_pins'],**s['protected_pins'],**s['dependencies']}.items():assert sha(p)==h,p
assert read(U/'CONTRACT_TESTS.json')['status']=='PASS'
assert read(U/'VISUAL_LABELS.json')['predictions_seen'] is False
old={r['clip_id']:r for r in read(B/'OUTPUT.json')['records']}
counts={'extent':Counter(),'overlap':Counter()};rows=[];preserved=0
for r in s['records']:
 raw=np.load(r['paths']['raw']);obs=np.load(r['paths']['candidate']);ass=np.load(r['paths']['assignment'])['sides']
 assert np.array_equal(raw['frame_id'],obs['frame_id']) and np.array_equal(raw['timestamp_us'],obs['timestamp_us'])
 adapter=SeparatedHandDiagnostics();frames=[]
 for i,t in enumerate(raw['timestamp_us']):
  w,h=int(raw['width'][i]),int(raw['height'][i]);xy,mask,conf=obs['xy_px'][i],obs['usable'][i],raw['pose_conf'][i]
  if r['clip_id'] in old:previous=old[r['clip_id']]['frames'][i]['hands']
  else:
   sides=ass[i,:int(raw['raw_count'][i])]
   previous=adapter.step(xy,mask,conf,int(t),w,h,assigned=(bool(0 in sides),bool(1 in sides)),unassigned_raw=bool(any(x<0 for x in sides)))
  copies=[a.copy() for a in (xy,mask,conf)]
  out=extend_evidence(previous,xy,mask,conf,w,h)
  assert all(np.array_equal(a,z,equal_nan=True) for a,z in zip((xy,mask,conf),copies))
  for side,q in enumerate(out):
   assert all(q[k]==v for k,v in previous[side].items());preserved+=1
   counts['extent'][q['current_extent']['status']]+=1;counts['overlap'][q['current_overlap']['status']]+=1
  frames.append({'frame':i,'timestamp_us':int(t),'core':any(a<=t/1e6<=b for a,b in r['intervals']),'hands':out})
 rows.append({'clip_id':r['clip_id'],'label':r['label'],'group':r['group'],'frames':frames})
labels=read(O/'word100_hand_region_uncertainty_20261017/VISUAL_LABELS.json')['labels']+read(U/'VISUAL_LABELS.json')['labels']
lookup={r['clip_id']:r for r in rows}
comparisons=[{**x,'candidate':lookup[x['clip_id']]['frames'][x['frame']]['hands'][x['side']]} for x in labels]
output={'scope_sha256':sha(U/'SCOPE.json'),'new_labels_sha256':sha(U/'VISUAL_LABELS.json'),'records':rows,'comparisons':comparisons,'counts':counts,
 'preserved_hand_states':preserved,'new_training':0,'classifier_inference':0,'new_detection':0,'user_recording_reads':0}
with (U/'OUTPUT.json').open('x',encoding='utf-8') as f:json.dump(output,f,ensure_ascii=False,indent=2)
tables={}
for x in comparisons:
 t=tables.setdefault(x['origin'],{});c=t.setdefault(x['visual_category'],Counter());h=x['candidate']
 c[h['current_extent']['status']+' / '+h['current_overlap']['status']]+=1
for p,h in {**s['source_pins'],**s['protected_pins'],**s['dependencies']}.items():assert sha(p)==h,p
print(json.dumps({'preserved':preserved,'counts':counts,'tables':tables},ensure_ascii=False,indent=2))
