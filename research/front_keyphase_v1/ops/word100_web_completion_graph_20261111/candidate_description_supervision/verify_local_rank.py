"""Fresh-process audit of saved predictions; no model inference or label changes."""
from local_rank_core import *
from collections import Counter,defaultdict
import math

E=L/'evaluation'
cfg=read(L/'FROZEN.json');meta=read(L/'INPUTS_COMPLETE.json')
comp=read(E/'COMPARISON.json');shadow=read(E/'shadow_components/RESULTS.json')
assert not (L/'VERIFICATION.json').exists()
for p,h in {**cfg['pins'],**meta['pins'],**read(E/'FROZEN.json')['pins']}.items():assert sha(p)==h,p
bags=meta['bags'];assert len(bags)==238
assert {r['person']for r in bags if r['split']=='train'}=={2,3,4,5,6,7}
assert {r['person']for r in bags if r['split']=='probe'}=={8,9}
assert not({r['source_sha256']for r in bags if r['split']=='train'}&{r['source_sha256']for r in bags if r['split']=='probe'})
data=[];targets=[]
for r in bags:
 assert sha(r['path'])==r['sha256']and sha(r['original_path'])==r['original_sha256']
 d=np.load(r['path']);old=np.load(r['original_path'])
 assert np.array_equal(d['y'],old['y'])and np.array_equal(d['spans'],old['spans'])
 data.append(d)
for run in read(L/'TRAINING_COMPLETE.json')['rows']:
 for k in ['model','predictions']:assert sha(run[k])==run[k+'_sha256']
 z=np.load(run['predictions'])
 for i,r in enumerate(bags):
  p=z[f'logits_{i}'][:,F.index(r['family'])];assert np.isfinite(p).all();j=int(p.argmax())
  targets.append(dict(condition=run['condition'],seed=run['seed'],clip_id=r['clip_id'],person=r['person'],view=r['view'],reverse=r['reverse'],split=r['split'],seen_by_this_head=r['split']=='train',family=r['family'],chosen=data[i]['spans'][j].tolist(),compatible=bool(data[i]['y'][j]),target='reviewed_source_extent',semantic_completion=None))
pm={(r['seed'],r['clip_id'],r['kind']):r for r in comp['predictions']};count=0
assert len(comp['rows'])==246
for r in comp['rows']:
 p=pm[r['seed'],r['clip_id'],r['kind']];assert sha(p['path'])==p['sha256']and sha(p['source'])==p['source_sha256'];z=np.load(p['path'])
 for mode in ['local_full','local_no_hands']:
  for i,c in enumerate(r['conditions'][mode]['cards']):
   sp=z[f'spans_{i}'];scores=[float(v)if a>=0 else -math.inf for v,a in zip(z[f'{mode}_logits_{i}'],z[f'presence_{i}'])]
   if scores and max(scores)>-math.inf:
    j=max(range(len(scores)),key=scores.__getitem__);assert c['chosen']['span']==sp[j].tolist();assert math.isclose(c['chosen']['score'],scores[j],abs_tol=1e-9)
   else:assert c['chosen']is None
   assert c['semantic_completion']is None;count+=1
for s in comp['summary']:
 rr=[r for r in comp['rows']if r['seed']==s['seed']and r['kind']==s['kind']];cc=[c for r in rr for c in r['conditions'][s['condition']]['cards']];tt=[c for c in cc if 'truth'in c];obs=lambda c:set(c['chosen']['body_observed'])if c['chosen']else set()
 assert sum(set(c['truth']['witness'])<=obs(c)for c in tt)==s['all_witness']
 assert sum(len(set(c['truth']['wrong_witness'])&obs(c))for c in tt)==s['wrong_points']
 assert sum(c['chosen']is None for c in cc)==s['missing']
 assert dict(Counter(o['state']for r in rr for o in r['conditions'][s['condition']]['order']))==s['order']
for s in shadow['summary']:
 cc=[c for r in shadow['rows']if r['seed']==s['seed']and r['kind']==s['kind']for c in r['conditions'][s['condition']]]
 assert dict(Counter(c['status']for c in cc))==s['statuses']
 for task in ['shape','direction']:assert dict(Counter(v['state']for c in cc for v in c['evidence']if v['task']==task))==s[task]
assert all(v['state']=='observation_unknown'for r in shadow['rows']if r['kind']=='all_hand_missing'for cs in r['conditions'].values()for c in cs for v in c['evidence']if v['task']=='shape')
changes=[]
for r in shadow['rows']:
 if r['kind']not in ['full_normal','speed_075','speed_125']:continue
 for mode in ['local_full','local_no_hands']:
  for before,after in zip(r['conditions']['baseline'],r['conditions'][mode]):
   a=[(v['task'],v['state'])for v in before['evidence']];b=[(v['task'],v['state'])for v in after['evidence']]
   if a!=b:changes.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],condition=mode,card=before['card'],before=a,after=b))
groups=defaultdict(lambda:[0,0])
for r in targets:
 k=(r['condition'],r['seed'],r['split'],r['view'],r['reverse']);groups[k][0]+=int(r['compatible']);groups[k][1]+=1
sm=[dict(condition=k[0],seed=k[1],split=k[2],view=k[3],reverse=k[4],correct=v[0],n=v[1])for k,v in groups.items()]
protected=read(U/'PROTECTIONS.json');assert all(sha(p)==h for p,h in protected.items())
write(L/'TARGET_DIAGNOSIS.json',dict(at=stamp(),rows=targets,summary=sm,normal_component_changes=changes,no_inference=True,warning='Boundary-compatible source witnesses are not educational completion; upstream common model saw probe clips.'))
write(L/'VERIFICATION.json',dict(at=stamp(),saved_arithmetic='PASS',argmax_checked=count,protected_files=len(protected),protected_unchanged=True,source_clips=32,fit_clips=24,probe_clips=8,no_new_inference=True,reviewer='same Codex separate process, not independent human/KSL truth',semantic_completion=None,web_changed=False,final8_access=False,private_access=False))
print('Verified local selections',count,'protected',len(protected))
for r in sm:
 if r['view']=='full'and not r['reverse']:print(r)
print('Normal changed components',len(changes))
