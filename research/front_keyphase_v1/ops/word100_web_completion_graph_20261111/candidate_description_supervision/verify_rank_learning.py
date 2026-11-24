from ranking_core import *
from collections import Counter,defaultdict
import math
E=K/'evaluation';meta=read(K/'INPUTS_COMPLETE.json');cfg=read(K/'FROZEN.json');comp=read(E/'COMPARISON.json');shadow=read(E/'shadow_components/RESULTS.json')
for p,h in {**cfg['pins'],**meta['pins'],**read(E/'FROZEN.json')['pins']}.items():assert sha(p)==h,p
assert len(comp['rows'])==246;assert {r['person']for r in meta['sequences']if r['split']=='train'}=={2,3,4,5,6,7};assert {r['person']for r in meta['sequences']if r['split']=='probe'}=={8,9}
assert not({r['source_sha256']for r in meta['sequences']if r['split']=='train'}&{r['source_sha256']for r in meta['sequences']if r['split']=='probe'})
for r in meta['sequences']:assert sha(r['path'])==r['sha256']
bags=[r for r in meta['bags']if not r['skipped']];target_rows=[]
for run in read(K/'TRAINING_COMPLETE.json')['rows']:
 for k in ['model','predictions']:assert sha(run[k])==run[k+'_sha256']
 z=np.load(run['predictions'])
 if run['condition'].startswith('rank_'):
  for i,r in enumerate(bags):
   assert sha(r['path'])==r['sha256'];d=np.load(r['path']);p=z[f'logits_{i}'][:,F.index(r['family'])];assert np.isfinite(p).all();j=int(p.argmax());target_rows.append(dict(condition=run['condition'],seed=run['seed'],clip_id=r['clip_id'],view=r['view'],reverse=r['reverse'],split=r['split'],family=r['family'],chosen=d['spans'][j].tolist(),compatible=bool(d['y'][j]),target='reviewed_source_extent'))
 else:
  for i,r in enumerate(meta['sequences']):
   d=np.load(r['path']);p=z[f'logits_{i}'];assert np.isfinite(p).all()
   for j,f in enumerate(F):
    for h,t in enumerate(['start','end']):
     band=d['bands'][:,j,h];valid=d['valid'][:,j,h]
     if band.any():
      at=int(np.where(valid,p[:,j,h],-np.inf).argmax());target_rows.append(dict(condition=run['condition'],seed=run['seed'],clip_id=r['clip_id'],view=r['view'],reverse=r['reverse'],split=r['split'],family=f,chosen=at,compatible=bool(band[at]),target=t))
ls=lambda v:-max(0.,-float(v))-math.log1p(math.exp(-abs(float(v))))
pm={(r['seed'],r['clip_id'],r['kind']):r for r in comp['predictions']};count=0
for r in comp['rows']:
 p=pm[r['seed'],r['clip_id'],r['kind']];assert sha(p['path'])==p['sha256'] and sha(p['source'])==p['source_sha256'];z=np.load(p['path'])
 for mode in ['rank_original','rank_extended','boundary_extended']:
  for i,c in enumerate(r['conditions'][mode]['cards']):
   sp=z[f'spans_{i}'];j=F.index(c['family'])
   if mode.startswith('rank_'):scores=[float(v)if eligible>=0 else -math.inf for v,eligible in zip(z[f'{mode}_logits_{i}'],z[f'presence_{i}'])]
   else:
    v=z['boundary_logits'][:,j];lp=[ls(q)for q in v[:,2]];pp=[math.exp(q)for q in lp];scores=[float(v[a,0])+float(v[b,1])+math.fsum(lp[a:b+1])/(b-a+1)if math.fsum(pp[a:b+1])/(b-a+1)>=.5 else -math.inf for a,b in sp]
   if scores and max(scores)>-math.inf:
    at=max(range(len(scores)),key=scores.__getitem__);assert c['chosen']['span']==sp[at].tolist();assert math.isclose(c['chosen']['score'],scores[at],abs_tol=1e-9)
   else:assert c['chosen']is None
   assert c['semantic_completion']is None;count+=1
for s in comp['summary']:
 rr=[r for r in comp['rows']if r['seed']==s['seed']and r['kind']==s['kind']];cc=[c for r in rr for c in r['conditions'][s['condition']]['cards']];tt=[c for c in cc if 'truth'in c];obs=lambda c:set(c['chosen']['body_observed'])if c['chosen']else set()
 assert sum(set(c['truth']['witness'])<=obs(c)for c in tt)==s['all_witness'];assert sum(len(set(c['truth']['wrong_witness'])&obs(c))for c in tt)==s['wrong_points'];assert sum(c['chosen']is None for c in cc)==s['missing'];assert dict(Counter(o['state']for r in rr for o in r['conditions'][s['condition']]['order']))==s['order']
diff=[]
for s in shadow['summary']:
 cc=[c for r in shadow['rows']if r['seed']==s['seed']and r['kind']==s['kind']for c in r['conditions'][s['condition']]];assert dict(Counter(c['status']for c in cc))==s['statuses']
 for task in ['shape','direction']:assert dict(Counter(v['state']for c in cc for v in c['evidence']if v['task']==task))==s[task]
for r in shadow['rows']:
 if r['kind']not in['full_normal','speed_075','speed_125']:continue
 for a,b in zip(r['conditions']['baseline'],r['conditions']['rank_extended']):
  aa=[(v['task'],v['state'])for v in a['evidence']];bb=[(v['task'],v['state'])for v in b['evidence']]
  if aa!=bb:diff.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=a['card'],before=aa,after=bb))
assert all(v['state']=='observation_unknown'for r in shadow['rows']if r['kind']=='all_hand_missing'for cs in r['conditions'].values()for c in cs for v in c['evidence']if v['task']=='shape')
protected=read(U/'PROTECTIONS.json');assert all(sha(p)==h for p,h in protected.items());groups=defaultdict(lambda:[0,0])
for r in target_rows:
 key=(r['condition'],r['seed'],r['split'],r['view'],r['reverse'],r['target']);groups[key][0]+=int(r['compatible']);groups[key][1]+=1
sm=[dict(condition=k[0],seed=k[1],split=k[2],view=k[3],reverse=k[4],target=k[5],correct=v[0],n=v[1])for k,v in groups.items()]
write(K/'TARGET_DIAGNOSIS.json',dict(at=stamp(),rows=target_rows,summary=sm,normal_component_changes=diff,no_inference=True))
write(K/'VERIFICATION.json',dict(at=stamp(),saved_arithmetic='PASS',argmax_checked=count,protected_files=len(protected),protected_unchanged=True,source_clips=32,fit_original=16,fit_extended=24,probe_clips=8,skipped_training_bags=[r for r in meta['bags']if r['skipped']],no_new_inference=True,reviewer='same Codex separate process, not independent human/KSL truth',semantic_completion=None,web_changed=False,final8_access=False,private_access=False))
print('Verified',count,'selections; protected',len(protected))
for r in sm:
 if r['seed']==42 and r['view']=='full'and not r['reverse']:print(r)
print('Normal component changes',diff)
