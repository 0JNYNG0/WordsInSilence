from pathlib import Path
import json,hashlib
import numpy as np
from sequence_core import features
U=Path(__file__).resolve().parent;O=U.parent;D=U/'state_path';D.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(D/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
prior=read(U/'metric_recovery/EVALUATION.json');assert not prior['summary']['preliminary_gate']
write('PLAN.json',dict(
 diagnosis='A direct class head predicts movement on exactly repeated observations and gives incompatible predictions under reversal. This is an architecture violation independent of probe labels.',
 remedy='Learn visible configuration from local native context; movement uses difference of learned configuration states. Enforce identical input = no observed shape change and reversal = opposite change algebraically.',
 context='Three native neighbors each side; partial-prefix inputs exclude all future beyond the cut. No invented coordinates, no minimum duration.',
 whole_interval='Keep the interior state path for repetition/uncertainty; equal endpoints do not prove no interior events.',
 original_endpoints_immutable=True,do_not_assume_missing_state=True,
 data='Existing FIT source annotations plus viewed107 new FIT points; no new probe annotation',
 caveat='Observed projected finger configuration only, not educational completion or exact 3D shape.'))
labels=read(U/'LABELS.json');points=read(U/'metric_recovery/LABELS.json')['rows'];manifest={r['clip_id']:r for r in read(O/'word100_shared_motion_learning_20261109/READY.json')['rows']};cache={};pins={}
def load(cid):
 if cid not in cache:
  m=manifest[cid];p=Path(m['channel']);assert sha(p)==m['channel_sha256'];z=np.load(p);cache[cid]=(z['feedback'if'feedback'in z.files else'full'],z['frame_id']);pins[str(p)]=sha(p)
 return cache[cid]
def context(f,k):
 out=np.zeros((7,f.shape[-1]),np.float32)
 for j,idx in enumerate(range(k-3,k+4)):
  if 0<=idx<len(f):out[j]=f[idx]
 return out
contexts=[];paths=[];meta=[]
for r in labels['rows']:
 x,fid=load(r['clip_id']);a,b=r['input_indices'];end=b+1 if r['origin']=='RGB_open_to_bent_partial'else len(x)
 xx=x[:end].copy()
 if r['freeze']:xx[:]=xx[a]
 f,ok=features(xx,r['hand']);pa=context(f,a);pb=context(f,b);path=np.stack([context(f,i)for i in range(a,b+1)])
 if r['reverse']:pa,pb=pb[::-1].copy(),pa[::-1].copy();path=path[::-1,::-1].copy()
 contexts.append(np.stack([pa,pb]));paths.append(path);meta.append(dict(**r,context_input_end=end-1,future_after_cut_used=False))
px=[]
for r in points:
 x,fid=load(r['clip_id']);f,ok=features(x,r['hand']);i=int(np.flatnonzero(fid==r['frame'])[0]);px.append(context(f,i))
np.savez_compressed(D/'arrays.npz',pairs=np.stack(contexts),points=np.stack(px),**{f'path{i}':p for i,p in enumerate(paths)})
write('LABELS.json',dict(rows=meta,point_rows=points,pins=pins,source_label_sha256=sha(U/'metric_recovery/LABELS.json')))
print(json.dumps(dict(pairs=len(meta),point_contexts=len(px),context_native_frames=7)))
