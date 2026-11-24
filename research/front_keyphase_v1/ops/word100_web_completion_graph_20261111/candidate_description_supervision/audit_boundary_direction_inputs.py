from boundary_direction_core import *
from collections import defaultdict,Counter
meta=read(M/'INPUTS_COMPLETE.json');assert sha(meta['path'])==meta['sha256'];z=np.load(meta['path']);rows=meta['rows'];pred=z['baseline_logits'].argmax(1);groups=defaultdict(list);conflicts=[]
for i,r in enumerate(rows):
 if r['origin']=='reviewed_boundary_variant':
  groups[r['clip_id'],r['view'],r['family'],r['reverse'],r['split']].append(i)
  delta=r['geometry']['signed_change'];want=r['target']-1
  if want*delta<0:conflicts.append(dict(index=i,**r))
summary=[]
for k,ix in groups.items():
 p=np.array(ix);tt=z['y'][p];pp=pred[p];summary.append(dict(clip_id=k[0],view=k[1],family=k[2],reverse=k[3],split=k[4],variants=len(p),correct=int((tt==pp).sum()),prediction_states=dict(Counter(map(int,pp))),min_signed=float(min(rows[i]['geometry']['signed_change']for i in ix)),max_signed=float(max(rows[i]['geometry']['signed_change']for i in ix))))
assert all(r['person']in([8,9]if r['split']=='probe'else[1,2,3,4,5,6,7])for r in rows)
assert not({r['source_sha256']for r in rows if r['split']=='train'}&{r['source_sha256']for r in rows if r['split']=='probe'})
write(M/'INPUT_AUDIT.json',dict(at=stamp(),rows=summary,coordinate_direction_conflicts=conflicts,training_ready=not conflicts,limits='Coordinate direction conflict is a reason to inspect observation/label agreement, not permission to relabel RGB truth from model output.',counts=dict(Counter((r['split']+'_'+r['origin'])for r in rows)),copied_head_matches=True,split_checked=True))
print('rows',len(rows),'direction conflicts',len(conflicts),'groups affected',len(set((r['clip_id'],r['view'],r['family'])for r in conflicts)))
for r in summary:
 if r['split']=='train'and not r['reverse']and r['correct']<r['variants']:print(r)
for r in conflicts[:10]:print('CONFLICT',r)
