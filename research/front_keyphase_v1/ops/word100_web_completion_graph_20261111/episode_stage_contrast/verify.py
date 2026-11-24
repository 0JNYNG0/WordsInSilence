"""Independent arithmetic implementation on saved artifacts; no model forwards."""
from pathlib import Path
import sys,hashlib,collections
import numpy as np
W=Path(__file__).resolve().parent;U=W.parent;P=U/'episode_stage'
sys.path.insert(0,str(U))
from graph import read,write,stamp
F=[2,3,4,5,7];A={2:0,3:1,4:2,5:4,7:3}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not (W/'VERIFICATION.json').exists();probe=[]
 for r in read(W/'TRAINING_COMPLETE.json')['runs']:
  assert r['steps']==1200 and sha(r['path'])==r['sha256'];d=np.load(Path(r['path']).parent/'PROBE.npz');y=d['y'];old=d['old_mask']
  for name,mask in [('old',old),('added',(y>=0)&~old),('all',y>=0)]:
   for which,key in [('baseline','baseline_p'),('candidate','p')]:
    pred=np.argmax(d[key],-1);correct=int(((pred==y)&mask).sum());total=int(mask.sum());cm=np.zeros((5,3,3),int)
    for j in range(5):
     for a,b in zip(y[:,j][mask[:,j]],pred[:,j][mask[:,j]]):cm[j,a,b]+=1
    expected=r['probe'][name][which];assert (correct,total)==(expected['correct'],expected['total']) and cm.tolist()==expected['confusion'];probe.append(dict(seed=r['seed'],subset=name,model=which,correct=correct,total=total))
  oldsaved=np.load(P/'runs'/str(r['seed'])/'PROBE.npz');assert np.array_equal(d['baseline_p'][old.any(1)],oldsaved['p'])
 ev=read(W/'EVALUATION_COMPLETE.json')['rows'];em={(r['seed'],r['clip_id'],r['kind']):r for r in ev};raw={};missing_qualified=0
 for r in ev:
  assert sha(r['path'])==r['sha256'] and sha(r['source'])==r['source_sha256'];raw[r['seed'],r['clip_id'],r['kind']]=(np.load(r['path'])['p'],np.load(r['source']))
 for filename in ['BINDING_SCORES.json','BINDING_V2_SCORES.json']:
  v2='V2' in filename;result=read(W/filename)
  for r in result['rows']:
   p,d=raw[r['seed'],r['clip_id'],r['kind']];q=d['z'][:,292:392].reshape(-1,10,10);axis=A[r['family']]*2;j=F.index(r['family']);dt=np.concatenate([np.diff(d['ts']),[np.median(np.diff(d['ts']))]])/1e6
   for c in r['candidates']:
    a,b=c['span'];ix=[i for i in range(a,b+1) if q[i,axis,8]>.5];assert ix==c['body_observed']
    vals=p[ix,j,1]+p[ix,j,2] if v2 else p[ix,j,1];vals=np.clip(vals,1e-6,1-1e-6);score=sum(float(np.log(v/(1-v)))*dt[i] for i,v in zip(ix,vals));assert abs(score-c['score'])<1e-4
    assert c['has_motion']==bool(np.any(p[ix,j,1]>=.5))
   eligible=[c for c in r['candidates'] if c['has_motion'] and (v2 or c['score']>0)];expected=max(eligible,key=lambda c:c['score']) if eligible else None;assert r['chosen']==expected
   if r['kind']=='all_hand_missing' and expected:assert not expected['hand_qualified']
  for s in result['summary']:
   rr=[r for r in result['rows'] if r['seed']==s['seed'] and r['kind']==s['kind'] and 'truth' in r];correct=wrong=0
   for r in rr:
    ix=set(r['chosen']['body_observed']) if r['chosen'] else set();correct+=set(r['truth']['witness'])<=ix;wrong+=len(set(r['truth']['wrong_witness'])&ix)
   assert correct==s['new_all_witness'] and wrong==s['new_wrong_points']
 protected=read(U/'PROTECTIONS.json');assert all(sha(p)==h for p,h in protected.items())
 write(W/'VERIFICATION.json',dict(at=stamp(),status='PASS_SAVED_ARITHMETIC_G2_REWORK',probe=probe,predictions=len(ev),protected_hashes=len(protected),all_missing_hand_qualified=0,old_probe_reused_exactly=True,reviewer='root Codex separate arithmetic, not independent human/KSL review',decision='Keep learned component improvement as isolated candidate; no runtime adoption. Corrected action-span binding preserves limited normal witnesses but includes other action witnesses on2/3 reordered clips. Need joint association/ambiguity diagnosis, not more blind epochs.',inference_calls=0))
 print('Verified probe matrices, old-output reuse,246 predictions, both binding calculations,',len(protected),'protected hashes. G2 REWORK.')
if __name__=='__main__':main()
