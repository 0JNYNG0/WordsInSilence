"""Check decoder assumptions on FIT/probe labels only; no new DEV decisions."""
from source_boundary_core import *
from collections import defaultdict
meta=read(B/'INPUTS_COMPLETE.json');out=[]
for run in read(B/'TRAINING_COMPLETE.json')['rows']:
 if run['condition']!='direct_boundary':continue
 p=np.load(run['predictions'])
 for i,r in enumerate(meta['sequences']):
  z=np.load(r['path']);l=p[f'logits_{i}'];prob=1/(1+np.exp(-np.clip(l[:,:,2],-40,40)))
  for j,f in enumerate(F):
   aa=np.flatnonzero(z['bands'][:,j,0]);bb=np.flatnonzero(z['bands'][:,j,1])
   if not len(aa) or not len(bb):continue
   pairs=[(int(a),int(b),float(prob[a:b+1,j].mean())) for a in aa for b in bb if a<b]
   known=z['mask'][:,j]>0;pos=known&(z['y'][:,j]>0);neg=known&(z['y'][:,j]==0)
   out.append(dict(seed=run['seed'],clip_id=r['clip_id'],split=r['split'],view=r['view'],reverse=r['reverse'],family=f,boundary_pair_count=len(pairs),any_true_band_pair_passes_mean_presence=any(v[2]>=.5 for v in pairs),min_mean_presence=min(v[2] for v in pairs),max_mean_presence=max(v[2] for v in pairs),sparse_positive_count=int(pos.sum()),sparse_positive_correct=int((prob[pos,j]>=.5).sum()),unlabelled_inside=int((~known[aa.min():bb.max()+1]).sum()),extent_frames=int(bb.max()-aa.min()+1)))
summary=[]
for seed in [42,43,44]:
 for split in ['train','probe']:
  for view in ['full','upper','actual_crop']:
   rr=[r for r in out if r['seed']==seed and r['split']==split and r['view']==view];summary.append(dict(seed=seed,split=split,view=view,groups=len(rr),gate_rejects_all_true_band_pairs=sum(not r['any_true_band_pair_passes_mean_presence'] for r in rr),known_positive_accuracy=sum(r['sparse_positive_correct'] for r in rr)/max(1,sum(r['sparse_positive_count'] for r in rr))))
write(B/'DECODE_CONTRACT_AUDIT.json',dict(at=stamp(),rows=out,summary=summary,claim='Sparse moving-frame presence is not a trained dense action-membership probability. Requiring its entire span mean >=.5 is an additional unsupported inference assumption.',not_claimed='Changing the gate solves boundary generalization. Independent probe boundary peaks already miss source bands.',no_new_inference=True,no_new_development_decisions=True))
for r in summary:
 if r['seed']==42:print(r)
