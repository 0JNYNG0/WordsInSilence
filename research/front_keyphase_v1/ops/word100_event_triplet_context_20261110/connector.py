from triplet_core import *
import copy
order_state=lc.order_state
def warning(card,kind,message,sp):
 card.update(status='difference_candidate',display_status='difference',educational_wrong=False,message=message,candidate_message=message)
 card.setdefault('difference_components',[])
 if kind not in card['difference_components']:card['difference_components'].append(kind)
 card.setdefault('guide_link_difference_spans',[]).extend(sp)

def apply_links(baseline,prob,x,ts,fh,base,contracts,context_spans):
 out=copy.deepcopy(baseline);ss=context_spans;bindings=[];directions=[];ordering=[]
 ev=baseline['phase_events']+baseline['phase_partials'];cache={}
 for c,spec in zip(out['cards'],contracts):
  assert c['id']==spec['id']
  # A failed expected handshape is not a reason to remove the attempted movement.
  matches=[s for s in ss if s['context'] in spec['contexts']];bound=[]
  if spec['axis'] is not None:
   for e in ev:
    if e['axis']!=spec['axis'] or e['moving']!='R' or e['b']-e['a']<3:continue
    overlap=sum(max(0,min(e['b'],s['b'])-max(e['a'],s['a'])+1) for s in matches)
    if overlap/(e['b']-e['a']+1)<.25:continue
    a,b=e['a'],e['b'];key=(a,b,spec['axis'])
    if key not in cache:
     geo,ok,info=prior.geometry(x,ts,a,b,spec['axis'],'R','L')
     with torch.inference_mode():mp=base.movement(torch.tensor(fh[None]),torch.tensor(geo[None]),torch.tensor([[0,a,b]])).softmax(-1)[0].numpy()
     pred=int(mp.argmax());delta=info.get('signed_change',0);agree=pred==1 or np.sign(delta)==pred-1
     cache[key]=dict(span=[a,b],axis=spec['axis'],moving='R',prediction=pred,confidence=float(mp.max()),eligible=bool(ok and agree),geometry=info,complete=e.get('complete',False))
    bound.append(cache[key])
  bid=dict(card=c['id'],contexts=spec['contexts'],context_spans=matches,movement_links=bound,required=spec['required'],parent=spec.get('parent'),original_window_used=False)
  bindings.append(bid)
  signs={e['prediction']-1 for e in bound if e['eligible'] and e['confidence']>=.65 and e['prediction']!=1}
  expected=spec['sign'];state='not_specified' if expected is None else 'unknown'
  if expected is not None and len(signs)==1:state='supported' if signs=={expected} else 'contradicted'
  if expected is not None and len(signs)>1:state='mixed'
  check=dict(card=c['id'],state=state,expected=expected,observed_signs=sorted(signs),evidence=bound)
  directions.append(check)
  if state=='contradicted' and not c.get('observation_quarantine_overlap'):
   warning(c,'guide_link_direction','표시된 동작에서 이동 방향이 참고와 반대로 보여요. 화살표 방향을 확인해 보세요.',[e['span'] for e in bound if e['eligible'] and e['confidence']>=.65])
  c['guide_link_observation']=bid;c['guide_link_direction']=check
  if not matches:
   c['guide_link_absence']=dict(state='unconfirmed',reason='No reliable independent body-context span; absence alone is not proof of omission.',completion_allowed=False)
 # Optional arrival states and same-card alternating subcontexts are not independent mandatory steps.
 required=[(i,c) for i,c in enumerate(contracts) if c['required'] and not c.get('parent')]
 for (ia,ca),(ib,cb) in zip(required,required[1:]):
  aa=bindings[ia]['context_spans'];bb=bindings[ib]['context_spans'];state=order_state(aa,bb)
  check=dict(pair=[ca['id'],cb['id']],state=state,first=aa,second=bb,optional_children_excluded=True);ordering.append(check)
  if state=='contradicted' and not any(out['cards'][i].get('observation_quarantine_overlap') for i in [ia,ib]):
   warning(out['cards'][ib],'guide_link_order','표시된 두 행동의 순서가 참고와 반대로 보여요. 먼저 해야 할 행동부터 이어 보세요.',[[s['a'],s['b']] for s in aa+bb])
 out.update(guide_link_version='independent_body_context_v1',guide_link_bindings=bindings,guide_link_order_checks=ordering,guide_link_direction_checks=directions,guide_link_probabilities=prob.tolist(),guide_link_absence_is_not_omission=True)
 return out
