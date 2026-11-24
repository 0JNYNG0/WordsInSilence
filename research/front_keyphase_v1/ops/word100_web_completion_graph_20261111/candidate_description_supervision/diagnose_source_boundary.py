"""Describe failed boundary supervision separately from decoding; no tuning."""
from source_boundary_core import *
from collections import defaultdict
meta=read(B/'INPUTS_COMPLETE.json');out=[]
for run in read(B/'TRAINING_COMPLETE.json')['rows']:
 assert sha(run['model'])==run['model_sha256'] and sha(run['predictions'])==run['predictions_sha256'];pred=np.load(run['predictions'])
 if run['condition']!='direct_boundary':continue
 for i,r in enumerate(meta['sequences']):
  z=np.load(r['path']);p=pred[f'logits_{i}'];assert p.shape==(len(z['x']),5,3)
  for j,f in enumerate(F):
   use=z['mask'][:,j]>0
   if use.any():
    y=z['y'][use,j];pr=p[use,j,2]>=0
    out.append(dict(seed=run['seed'],clip_id=r['clip_id'],split=r['split'],view=r['view'],reverse=r['reverse'],family=f,target='presence',n=int(use.sum()),correct=int((pr==y).sum()),positive=int(y.sum()),tp=int((pr&(y==1)).sum()),negative=int((y==0).sum()),tn=int((~pr&(y==0)).sum())))
   for k,name in enumerate(['start','end']):
    valid=z['valid'][:,j,k];band=z['bands'][:,j,k]
    if band.any():
     at=int(np.where(valid,p[:,j,k],-np.inf).argmax());out.append(dict(seed=run['seed'],clip_id=r['clip_id'],split=r['split'],view=r['view'],reverse=r['reverse'],family=f,target=name,n=1,correct=int(band[at]),chosen=at,band=np.flatnonzero(band).tolist()))
groups=defaultdict(lambda:[0,0])
for r in out:
 key=(r['seed'],r['split'],r['view'],r['reverse'],r['target']);groups[key][0]+=r['correct'];groups[key][1]+=r['n']
summary=[dict(seed=k[0],split=k[1],view=k[2],reverse=k[3],target=k[4],correct=v[0],n=v[1]) for k,v in groups.items()]
write(B/'TARGET_DIAGNOSIS.json',dict(at=stamp(),rows=out,summary=summary,saved_logits_only=True,no_tuning=True,scope='Sparse presence and onset/offset bands; not educational accuracy or unseen upstream-person evaluation'))
for r in summary:
 if r['seed']==42 and r['view']=='full' and not r['reverse']:print(r)
comp=read(B/'evaluation/COMPARISON.json')
for r in comp['summary']:
 if r['kind']in['full_normal','dynamic_complete_blocks'] and r['condition']=='body_window':print(r)
