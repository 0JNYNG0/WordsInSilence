"""Audit saved probabilities and invariants, without any new model inference."""
from projected_direction_core import *
M=V
from collections import Counter, defaultdict

E=M/'evaluation'
assert not (M/'VERIFICATION.json').exists()
meta=read(M/'INPUTS_COMPLETE.json');cfg=read(M/'FROZEN.json');result=read(E/'RESULTS.json')
for p,h in {**cfg['pins'],**read(E/'FROZEN.json')['pins']}.items():assert sha(p)==h,p
assert sha(meta['path'])==meta['sha256'] and sha(meta['source'])==meta['source_sha256']
rows=meta['rows'];z=np.load(meta['path']);assert len(rows)==len(z['y'])==2406
assert all(r['axis']in[1,2,3,4]for r in rows)
tr=[r for r in rows if r['split']=='train'];pr=[r for r in rows if r['split']=='probe']
assert {r['person']for r in tr}<={1,2,3,4,5,6,7}
assert {r['person']for r in pr}=={8,9}
assert not({r['source_sha256']for r in tr}&{r['source_sha256']for r in pr})
assert dict(Counter(r['split']+'_'+r['origin']for r in rows))==meta['counts']
def verdict(p,d,ok):
 if not ok:return 'observation_unknown'
 if d==1:return 'supported'if p[1]>=.65 and max(p[0],p[2])<.65 else'uncertain'
 if p[d]>=.65 and p[2-d]<.65 and p[1]<.65:return 'supported'
 if p[2-d]>=.65 and p[d]<.35 and p[1]<.65:return 'opposite_direction'
 if p[1]>=.65:return 'movement_not_established'
 return 'uncertain'
for p in result['direction_predictions']:
 a=np.asarray(p['logits']);soft=np.exp(a-a.max());soft/=soft.sum()
 assert np.allclose(soft,p['probabilities'],atol=1e-6)
 assert verdict(soft,p['desired'],p['eligible'])==p['state']
 assert sha(p['source'])==p['source_sha256']
original=read(L/'evaluation/shadow_components/RESULTS.json')
om={(r['seed'],r['clip_id'],r['kind']):r for r in original['rows']}
changes=[];invariant=0
assert len(result['rows'])==246
for r in result['rows']:
 old=om[r['seed'],r['clip_id'],r['kind']]
 assert r['conditions']['baseline']==old['conditions']['baseline']
 assert r['conditions']['local_old']==old['conditions']['local_full']
 for a,b in zip(r['conditions']['local_old'],r['conditions']['local_tuned']):
  assert a['card']==b['card'] and a['span']==b['span']
  assert len(a['evidence'])==len(b['evidence'])
  for x,y in zip(a['evidence'],b['evidence']):
   if x['task']!='direction'or x.get('axis',0)!=2:assert x==y;invariant+=1
   elif x['state']!=y['state']:changes.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=a['card'],span=a['span'],before=x['state'],after=y['state']))
  states=[v['state']for v in b['evidence']]
  expected='association_unknown'if b['span']is None else 'difference_evidence'if ('difference'in states or'opposite_direction'in states)else'supported_features_only'if states and all(v=='supported'for v in states)else'insufficient_evidence'
  assert b['status']==expected
for s in result['summary']:
 cc=[c for r in result['rows']if r['seed']==s['seed']and r['kind']==s['kind']for c in r['conditions'][s['condition']]]
 assert len(cc)==s['cards'] and dict(Counter(c['status']for c in cc))==s['statuses']
 for task in ['shape','direction']:assert dict(Counter(v['state']for c in cc for v in c['evidence']if v['task']==task))==s[task]
groups=[]
for run in read(M/'TRAINING_COMPLETE.json')['rows']:
 for k in ['model','predictions']:assert sha(run[k])==run[k+'_sha256']
 logits=np.load(run['predictions'])['logits'];assert logits.shape==(2406,3)and np.isfinite(logits).all()
 for split in ['train','probe']:
  for origin in sorted({r['origin']for r in rows}):
   ii=[i for i,r in enumerate(rows)if r['split']==split and r['origin']==origin]
   if ii:groups.append(dict(seed=run['seed'],split=split,origin=origin,correct=int((logits[ii].argmax(-1)==z['y'][ii]).sum()),n=len(ii)))
protected=read(U/'PROTECTIONS.json');assert all(sha(p)==h for p,h in protected.items())
write(M/'VERIFICATION.json',dict(at=stamp(),saved_arithmetic='PASS',rows=246,direction_predictions=len(result['direction_predictions']),preserved_shape_nonaxis2=invariant,protected_files=len(protected),protected_unchanged=True,train_rows=len(tr),probe_rows=len(pr),target_summary=groups,changes=changes,no_inference=True,reviewer='Same Codex fresh process; not independent human/KSL truth',adopted=False,reason='Normal direction recovered and combined fist+reverse improved, but direction-only reversal regressed for two seeds. No complete gate pass.'))
print('Verified',len(result['direction_predictions']),'directions;',invariant,'unchanged shape/non-axis2;',len(protected),'protected files')
for r in groups:print(r)
print('Changed directions',len(changes))
