from common import *
import math, collections
from PIL import Image
s=read(U/'SCOPE.json');c=read(U/'CATALOG.json');check(s['pins'])
pts_check=read(U/'VIDEO_PTS_CHECK.json');assert pts_check['status']=='PASS'
pts_by_word={r['word_code']:{f['frame_id']:f['pts_seconds']for f in r['frames']}for r in pts_check['items']}
assert c['scope_sha256']==sha(U/'SCOPE.json')
assert [i['clip_id'] for i in c['items']]==[i['clip_id'] for i in s['items']]
assert len(c['items'])==100
checks=[];images=0;frames=0;unknown_frames=0;mark_kind=collections.Counter();nospans=[];maxerr=0
for r in c['items']:
 assert r['signer_id']==1 and r['role']=='train_candidate' and not r['expert_reviewed']
 assert sha(r['video_path'])==r['source_sha256'] and sha(r['observation'])==r['observation_sha256']
 data=np.load(r['observation']);xy=data['xy_px'];ok=data['usable'];t=data['timestamp_us']/1e6;fid=data['frame_id'];lo,hi=r['word_interval']
 assert len(r['spans'])==len(r['marks'])+1 and len(r['marks'])<=5
 assert r['spans'][0]['start']==lo and r['spans'][-1]['end']==hi
 whole=np.flatnonzero((t>=lo)&(t<=hi)).tolist();union=[]
 for j,p in enumerate(r['spans']):
  assert p['start']<p['end'] and not p['expert_reviewed']
  assert all(p[k] is None for k in ['required_optional','required_hands','precedence','repeat_policy','repeat_range'])
  if j:assert p['start']==r['spans'][j-1]['end']
  idx=p['source_frame_indices'];union+=idx
  assert all(p['start']<=t[i] and (t[i]<p['end'] or (j==len(r['spans'])-1 and t[i]<=p['end'])) for i in idx)
  for hand,off in [('L',33),('R',54)]:
   counts=[sum(bool(v) for v in ok[i,off:off+21]) for i in idx]
   v=p['observation'][hand]
   assert v==dict(full21_frames=counts.count(21),partial_frames=sum(0<n<21 for n in counts),zero_frames=counts.count(0),total_frames=len(idx))
   unknown_frames+=sum(n<21 for n in counts)
  for im in p['images']:
   i=im['observation_index'];assert im['frame_id']==int(fid[i]) and im['seconds']==float(t[i])
   assert abs(im['seconds']-pts_by_word[r['word_code']][im['frame_id']])<1e-6
   assert sha(U/im['image'])==im['image_sha256']
   with Image.open(U/im['image']) as img:assert img.size==(640,360)
   images+=1
 assert union==whole and len(union)==len(set(union));frames+=len(whole)
 assert sha(U/r['sheet'])==r['sheet_sha256']
 if not r['marks']:nospans.append(dict(label=r['label'],word_code=r['word_code'],duration=hi-lo))
 for m in r['marks']:
  mark_kind[m['kind']]+=1;i=m['index'];assert m['time']==t[i]
  before=[j for j in range(len(t)) if t[i]-.133333<=t[j]<t[i]];after=[j for j in range(len(t)) if t[i]<t[j]<=t[i]+.133333]
  off=33 if m['hand']=='L' else 54
  assert len(before)>=2 and len(after)>=2
  assert ok[before+[i]+after,off:off+21].all() and ok[before+[i]+after,11:13].all()
  vals=[]
  for ids in [before,after]:
   seq=[]
   for j in ids:
    points=xy[j].astype(float)
    if m['kind']=='position_change':
     scale=math.dist(points[11],points[12]);palm=[sum(points[off+n,k] for n in [0,5,9,13,17])/5 for k in [0,1]]
     seq.append([(palm[k]-(points[11,k]+points[12,k])/2)/scale for k in [0,1]])
    else:
     angles=[]
     for fingers in [(0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)]:
      for f in range(1,4):
       a=points[off+fingers[f-1]]-points[off+fingers[f]];b=points[off+fingers[f+1]]-points[off+fingers[f]]
       angles.append(sum(a[k]*b[k] for k in [0,1])/(math.hypot(*a)*math.hypot(*b)))
     seq.append(angles)
   vals.append(np.median(seq,axis=0))
  actual=math.dist(vals[0],vals[1]) if m['kind']=='position_change' else sum(abs(a-b) for a,b in zip(*vals))/15
  err=abs(actual-m['value']);maxerr=max(maxerr,err);assert err<1e-5,(r['word_code'],m,actual)
  checks.append(dict(word_code=r['word_code'],time=m['time'],kind=m['kind'],scalar=actual,saved=m['value'],error=err))
 for a,b in zip(r['marks'],r['marks'][1:]):assert b['time']-a['time']>=.30-1e-10
 # Every stored complete-hand run is a literal mask summary, not a physical absence label.
 for hand,off in [('L',33),('R',54)]:
  n=0
  for rr in r['hand_observation_runs'][hand]:
   assert rr['start']==t[n];stop=n+rr['samples'];assert all(bool(ok[j,off:off+21].all())==rr['value'] for j in range(n,stop));n=stop
  assert n==len(t)
write(U/'INDEPENDENT.json',dict(status='PASS',at=now(),catalog_sha256=sha(U/'CATALOG.json'),verifier_sha256=sha(__file__),
 public_words=100,spans=sum(len(r['spans']) for r in c['items']),source_frames_in_annotated_intervals=frames,hand_frames_with_less_than21points=unknown_frames,
 images_verified=images,scalar_bookmarks_checked=len(checks),max_numeric_error=maxerr,mark_kinds=dict(mark_kind),no_interior_bookmarks=nospans,protected_files=len(s['pins']),
 checks=checks,semantic_contracts_confirmed=0,verification='Same Codex separate scalar calculation and saved artifact verification, not independent agent or KSL expert.',
 warnings=['Automatic spans may split one gesture or merge multiple meaningful phases.','Numerical observations do not certify semantic correctness or hand identity.','All references one signer; no normal-variant or repeat-policy conclusion.']))
print('PASS',len(checks),'scalar bookmark checks;',images,'images;',maxerr,'max error')
