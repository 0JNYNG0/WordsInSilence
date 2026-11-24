from episode_core import *
import copy
OLD_MAP={0:0,1:1,2:2,3:3,4:4,5:4,6:5,7:5,8:5,9:5,10:5,11:6,12:7,13:7}
def role_spans(rp,q,ts):
 out=[]
 for fam in range(1,8):
  ks=[0] if fam<=3 else [4,5] if fam==4 else [6,7];valid=np.logical_and.reduce([q[:,k,-2:].max(-1)>.5 for k in ks])
  for a,b in bc.groups((rp[:,fam]>=.65)&valid):
   if b-a>=1 and ts[b]-ts[a]>=60000:out.append(dict(a=a,b=b,family=fam,confidence=float(rp[a:b+1,fam].mean())))
 return out
def direction(base,x,ts,fh,a,b,axis,moving='R',anchor='L'):
 geo,ok,info=prior.geometry(x,ts,a,b,axis,moving,anchor)
 with torch.inference_mode():p=base.movement(torch.tensor(fh[None]),torch.tensor(geo[None]),torch.tensor([[0,a,b]])).softmax(-1)[0].numpy()
 k=int(p.argmax());agree=k==1 or np.sign(info.get('signed_change',0))==k-1
 return dict(span=[a,b],axis=axis,moving=moving,prediction=k,confidence=float(p.max()),eligible=bool(ok and agree),geometry_eligible=bool(ok),geometry=info)
def conflict(spans,oldprob):
 checks=[]
 for s in spans:
  votes=0
  for p in oldprob[s['a']:s['b']+1]:
   k=int(np.argmax(p));votes+=p[k]>=.65 and OLD_MAP[k]!=s['family']
  checks.append(dict(span=[s['a'],s['b']],family=s['family'],votes=votes,frames=s['b']-s['a']+1,conflict=votes>(s['b']-s['a']+1)/2))
 return checks
def add_warning(c,kind,message):
 c.update(status='difference_candidate',display_status='difference',educational_wrong=False,message=message,candidate_message=message);c.setdefault('difference_components',[])
 if kind not in c['difference_components']:c['difference_components'].append(kind)
def supplement(before,rp,bp,q,x,ts,fh,base,contracts):
 out=copy.deepcopy(before);new,partials=decode(rp,bp,q);ss=role_spans(rp,q,ts);bindings=[];cache={}
 # Preserve every old candidate. New proposals supplement, never erase difficult movements.
 all_events=before['phase_events']+before['phase_partials']+new+partials
 for c,spec in zip(out['cards'],contracts):
  matches=[s for s in ss if s['family'] in spec['families']];cc=conflict(matches,before['phase_probability']);bound=[];seen=set()
  if spec['axis'] is not None:
   for e in all_events:
    if e['axis']!=spec['axis'] or e['moving']!='R' or e['b']-e['a']<3:continue
    a,b=e['a'],e['b'];key=(a,b,spec['axis'])
    if key in seen:continue
    overlap=len({i for s in matches for i in range(max(a,s['a']),min(b,s['b'])+1)})
    if overlap/(b-a+1)<.25:continue
    seen.add(key)
    if key not in cache:cache[key]=direction(base,x,ts,fh,a,b,spec['axis'])
    bound.append(dict(cache[key],complete=e.get('complete',False)))
  signs={e['prediction']-1 for e in bound if e['eligible'] and e['confidence']>=.65 and e['prediction']!=1};expected=spec['sign'];state='not_specified' if expected is None else 'unknown'
  if expected is not None and len(signs)==1:state='supported' if signs=={expected} else 'contradicted'
  elif expected is not None and len(signs)>1:state='mixed'
  blocked=any(z['conflict'] for z in cc)
  bind=dict(card=c['id'],families=spec['families'],spans=matches,movements=bound,state=state,expected_sign=expected,conflict_checks=cc,conflict=blocked,required=spec['required'],parent=spec.get('parent'));bindings.append(bind);c['episode_binding']=bind
  if state=='contradicted' and not blocked and not c.get('observation_quarantine_overlap'):add_warning(c,'episode_direction','표시된 움직임이 참고와 반대 방향으로 보여요. 이동 방향을 확인해 보세요.')
 ordering=[];required=[i for i,c in enumerate(contracts) if c['required'] and not c.get('parent')]
 for ia,ib in zip(required,required[1:]):
  a,b=bindings[ia],bindings[ib];state=lc.order_state(a['spans'],b['spans']);blocked=a['conflict'] or b['conflict'];ordering.append(dict(pair=[a['card'],b['card']],state=state,conflict=blocked,first=a['spans'],second=b['spans']))
  if state=='contradicted' and not blocked and not any(out['cards'][i].get('observation_quarantine_overlap') for i in [ia,ib]):add_warning(out['cards'][ib],'episode_order','표시된 행동의 순서가 참고와 반대로 보여요. 먼저 할 행동부터 이어 보세요.')
 out.update(episode_events=new,episode_partials=partials,episode_role_probabilities=rp.tolist(),episode_boundary_probabilities=bp.tolist(),episode_bindings=bindings,episode_order_checks=ordering,episode_absence_is_not_omission=True)
 return out
