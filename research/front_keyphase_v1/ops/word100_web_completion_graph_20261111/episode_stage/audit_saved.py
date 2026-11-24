"""Second arithmetic implementation over saved outputs; never calls a model."""
from pathlib import Path
import sys, hashlib, json
import numpy as np
W=Path(__file__).resolve().parent; U=W.parent
sys.path.insert(0,str(U))
from graph import read,write,stamp
F=[2,3,4,5,7]; A={2:0,3:1,4:2,5:4,7:3}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 out=W/'SAVED_VERIFICATION.json';assert not out.exists()
 conf=[]
 for r in read(W/'TRAINING_COMPLETE.json')['runs']:
  assert sha(r['path'])==r['sha256'];d=np.load(Path(r['path']).parent/'PROBE.npz');cm=np.zeros((5,3,3),int)
  for i in range(len(d['y'])):
   for j in range(5):
    y=int(d['y'][i,j])
    if y>=0:cm[j,y,int(np.argmax(d['p'][i,j]))]+=1
  assert cm.tolist()==r['probe']['confusion'];conf.append(dict(seed=r['seed'],correct=int(sum(np.trace(m) for m in cm)),total=int(cm.sum())))
 ev=read(W/'EVALUATION_COMPLETE.json')['rows']; lookup={(r['seed'],r['clip_id'],r['kind']):r for r in ev};checked=0;missing=[]
 for b in read(W/'SCORES.json')['behaviors']:
  r=lookup[b['seed'],b['clip_id'],b['kind']];d=np.load(r['source']);p=np.load(r['path'])['p'];q=d['z'][:,292:392].reshape(-1,10,10);j=F.index(b['family']);axis=A[b['family']]*2
  m=set(np.flatnonzero((p[:,j,1]>=.5)&(q[:,axis,8]>.5)).tolist());h=set(np.flatnonzero((p[:,j,2]>=.5)&(q[:,axis,8]>.5)).tolist())
  assert len(m)==b['new_motion'] and len(h)==b['new_hold'];assert m=={i for a,z in b['motion_intervals'] for i in range(a,z+1)};checked+=1
  if b['kind']=='all_hand_missing':
   valid=d['x'][:,186:237];assert np.all(valid[:,9:]==0)
   # Body relation motion may remain observable. It cannot prove handshape/contact.
   qualified={i for i in m if q[i,axis,9]>.5}
   assert not qualified
   missing.append(dict(seed=b['seed'],clip=b['clip_id'],family=b['family'],body_motion_proposals=len(m),hand_qualified_evidence=len(qualified),semantic_completion=None))
 for r in ev:assert sha(r['path'])==r['sha256'] and sha(r['source'])==r['source_sha256']
 write(out,dict(at=stamp(),status='PASS_SAVED_ARITHMETIC_NOT_G2',reviewer='same root Codex second implementation, not independent human',probe=conf,checked_cards=checked,prediction_files=len(ev),all_missing=missing,inference_calls=0,web_changed=False,decision='REWORK: diagnostic movement coverage improves but wrong-action and held-pose contamination remain. Body-only proposals must never stand in for hand-specific support.'))
 print('Verified',len(ev),'saved prediction files and',checked,'cards; all missing has zero hand-qualified evidence; G2 REWORK.')
if __name__=='__main__':main()
