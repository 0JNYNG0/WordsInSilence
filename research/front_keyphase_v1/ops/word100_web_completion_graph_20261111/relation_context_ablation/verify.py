"""Second arithmetic path: outputs reused, no source/model inference replay."""
from pathlib import Path
import sys,hashlib,importlib.util
import numpy as np
W=Path(__file__).resolve().parent;U=W.parent;B=U/'episode_stage_contrast'
sys.path.insert(0,str(U))
from graph import read,write,stamp
F=[2,3,4,5,7];A={2:0,3:1,4:2,5:4,7:3}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 assert not (W/'VERIFICATION.json').exists();runs=read(W/'TRAINING_COMPLETE.json')['runs'];probe=[]
 for r in runs:
  assert sha(r['path'])==r['sha256'] and r['steps']==1200
  d=np.load(Path(r['path']).parent/'PROBE.npz');old=np.load(r['baseline_path']);assert sha(r['baseline_path'])==r['baseline_sha256'];assert np.array_equal(d['baseline_p'],old['p']) and np.array_equal(d['y'],old['y'])
  for subset,mask in [('old',d['old_mask']),('added',(d['y']>=0)&~d['old_mask']),('all',d['y']>=0)]:
   for model,key in [('baseline','baseline_p'),('candidate','p')]:
    pred=d[key].argmax(-1);cm=np.zeros((5,3,3),int)
    for j in range(5):
     for a,b in zip(d['y'][:,j][mask[:,j]],pred[:,j][mask[:,j]]):cm[j,a,b]+=1
    wanted=r['probe'][subset][model];assert cm.tolist()==wanted['confusion'];assert sum(np.trace(c) for c in cm)==wanted['correct'] and cm.sum()==wanted['total'];probe.append(dict(seed=r['seed'],subset=subset,model=model,correct=wanted['correct'],total=wanted['total']))
 raw={}
 for r in read(W/'EVALUATION_COMPLETE.json')['rows']:
  assert sha(r['path'])==r['sha256'] and sha(r['source'])==r['source_sha256'];raw[r['seed'],r['clip_id'],r['kind']]=(np.load(r['path'])['p'],np.load(r['source']))
 saved=read(W/'BINDING_V2_SCORES.json');old=read(B/'BINDING_V2_SCORES.json');oldrows={(r['seed'],r['clip_id'],r['kind'],r['family']):r for r in old['rows']};comparison=[]
 for r in saved['rows']:
  p,d=raw[r['seed'],r['clip_id'],r['kind']];q=d['z'][:,292:392].reshape(-1,10,10);axis=2*A[r['family']];j=F.index(r['family']);dt=np.r_[np.diff(d['ts']),np.median(np.diff(d['ts']))]/1e6
  for c in r['candidates']:
   ix=[t for t in range(c['span'][0],c['span'][1]+1) if q[t,axis,8]>.5];assert ix==c['body_observed'];s=0.
   for t in ix:
    v=float(np.clip(p[t,j,1]+p[t,j,2],1e-6,1-1e-6));s+=np.log(v/(1-v))*dt[t]
   assert abs(s-c['score'])<1e-4 and c['has_motion']==bool(np.any(p[ix,j,1]>=.5))
  eligible=[c for c in r['candidates'] if c['has_motion']];chosen=max(eligible,key=lambda c:c['score']) if eligible else None;assert chosen==r['chosen']
  if r['kind']=='all_hand_missing' and chosen:assert not chosen['hand_qualified']
 for seed in [42,43,44]:
  for kind in ['full_normal','dynamic_complete_blocks']:
   rr=[r for r in saved['rows'] if r['seed']==seed and r['kind']==kind and 'truth' in r];base=[oldrows[r['seed'],r['clip_id'],r['kind'],r['family']] for r in rr]
   def count(rows):
    return dict(witness_groups=sum(set(r['truth']['witness'])<=set(r['chosen']['body_observed'] if r['chosen'] else []) for r in rows),wrong_points=sum(len(set(r['truth']['wrong_witness'])&set(r['chosen']['body_observed'] if r['chosen'] else [])) for r in rows))
   comparison.append(dict(seed=seed,kind=kind,groups=len(rr),baseline=count(base),candidate=count(rr)))
  rr=[r for r in saved['rows'] if r['seed']==seed and r['kind'] in ['full_normal','speed075','speed125']];comparison.append(dict(seed=seed,kind='normal_speed',cards=len(rr),baseline_missing=sum(oldrows[r['seed'],r['clip_id'],r['kind'],r['family']]['chosen'] is None for r in rr),candidate_missing=sum(r['chosen'] is None for r in rr)))
 protected=read(U/'PROTECTIONS.json');assert all(sha(p)==h for p,h in protected.items())
 write(W/'VERIFICATION.json',dict(at=stamp(),status='PASS_ARITHMETIC_CANDIDATE_REJECTED',probe=probe,comparison=comparison,predictions=len(raw),protected_hashes=len(protected),old_outputs_reused_exactly=True,source_inference_replayed=False,web_changed=False,decision='Local-relative-only ablation reduces probe correctness and loses normal/speed candidates without reducing reordered contamination. Do not adopt. Retain context-rich stage model; next improve supervised context/association, not discard context wholesale.',reviewer='root Codex separate arithmetic; no independent human/KSL evaluation'))
 print('Verified',len(raw),'saved predictions;',len(protected),'protected hashes; local-only candidate rejected.')
 print(comparison)

if __name__=='__main__':main()
