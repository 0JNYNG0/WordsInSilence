from role_core import *

def role_spans(prob,q,ts):
 out=[]
 for fam in range(1,10):
  ks=[0] if fam<=3 else [4,5] if fam==4 else [6,7] if fam<8 else []
  valid=np.logical_and.reduce([q[:,k,-2:].max(-1)>.5 for k in ks]) if ks else np.ones(len(q),bool)
  for a,b in bc.groups((prob[:,fam]>=.65)&valid):
   if b>a and ts[b]-ts[a]>=60000:out.append(dict(a=a,b=b,family=fam,confidence=float(prob[a:b+1,fam].mean())))
 return out

def unique(events):
 return list({(e['a'],e['b'],e['axis'],e['moving']):e for e in events if e['b']-e['a']>=3}.values())

def bind(before,prob,ob,events,net,frozen,x,ts,fh,base,motion,cache,mode):
 """No test kind, word number branch, target direction or gold spans in selection."""
 out=copy.deepcopy(before);contracts=jc.ep.family_contracts()[before['word']];q=ob['q'][0].numpy();spans=role_spans(prob,q,ts)
 # Family->axis information is a shared declarative guide schema, not a video
 # frame boundary or the expected direction. Unknown candidates stay recorded.
 axes={}
 for cc in jc.ep.family_contracts().values():
  for spec in cc:
   if spec['axis'] is not None:
    for fam in spec['families']:axes.setdefault(fam,set()).add(spec['axis'])
 candidates=copy.deepcopy(events)
 for s in spans:
  for axis in axes.get(s['family'],[]):
   candidates.append(dict(a=s['a'],b=s['b'],axis=axis,moving='R',source='role_proposal',complete=False,boundary_supported=False))
 candidates=unique(candidates);ss=[(0,e['a'],e['b'],e['axis']*2+(e['moving']=='L')) for e in candidates];sf=segment_features(frozen,ob,ss)
 quality=net.quality(torch.tensor(sf)).softmax(-1).detach().numpy() if len(sf) else np.zeros((0,4));ledger=[]
 for i,(e,p) in enumerate(zip(candidates,quality)):
  a,b=e['a'],e['b'];famprob=prob[a:b+1].mean(0);fam=int(famprob.argmax());coverage=float((prob[a:b+1,fam]>=.65).mean());k=e['axis']*2+(e['moving']=='L');valid=float((q[a:b+1,k,-2:].max(-1)>.5).mean())
  accepted=fam in range(1,8) and coverage>=.65 and valid>=.8 and int(p.argmax())==2 and p[2]>=.65
  item=dict(e);item.update(event=i,family=fam,family_probability=famprob.tolist(),role_coverage=coverage,quality=p.tolist(),valid=valid,accepted=bool(accepted),expected_direction_used=False,semantic_complete=False);ledger.append(item)
 pairs=[]
 for i,a in enumerate(ledger):
  for j,b in enumerate(ledger):
   if i==j or a['axis']!=b['axis'] or a['moving']!=b['moving'] or a['b']>b['a']:continue
   if b['a']-a['b']>max(a['b']-a['a']+1,b['b']-b['a']+1):continue
   pf=torch.tensor(np.r_[sf[i],sf[j],(b['a']-a['b'])/100.],dtype=torch.float32)
   with torch.inference_mode():pp=net.pair(pf).softmax(-1).numpy()
   pairs.append(dict(first=i,second=j,label=int(pp.argmax()),probability=pp.tolist()))
 # Retain rejected candidates in the ledger. One observed piece cannot be used
 # as two independent actions; optional guide child can share its parent.
 selected=[]
 for e in sorted([e for e in ledger if e['accepted']],key=lambda e:(-e['quality'][2],e['b']-e['a'])):
  if any(e['axis']==f['axis'] and e['moving']==f['moving'] and max(e['a'],f['a'])<min(e['b'],f['b']) for f in selected):continue
  if any(p['label']!=0 and max(p['probability'])>=.65 and ledger[p['first']]['a']>=e['a'] and ledger[p['second']]['b']<=e['b'] for p in pairs):continue
  selected.append(e)
 selected.sort(key=lambda e:e['a']);bindings=[]
 for card,spec in zip(out['cards'],contracts):
  matches=[s for s in spans if s['family'] in spec['families']];ds=[]
  for e in selected:
   if e['family'] not in spec['families'] or e['axis']!=spec['axis'] or e['moving']!='R':continue
   key=(e['a'],e['b'],e['axis'],e['moving'],'L')
   if key not in cache:
    od=jc.integration.direction(base,x,ts,fh,*key[:3],key[3],key[4]);nd=mc.direction(motion,x,ts,*key[:3],key[3],key[4]);cache[key]=mc.fused(od,nd)
   ds.append(dict(cache[key],event=e['event']))
  signs={d['prediction']-1 for d in ds if d['eligible'] and d['confidence']>=.65 and d['prediction']!=1};expected=spec['sign'];state='not_specified' if expected is None else 'unknown'
  if expected is not None and len(signs)==1:state='supported' if signs=={expected} else 'contradicted'
  elif len(signs)>1:state='mixed'
  b=dict(card=card['id'],spans=matches,directions=ds,state=state,semantic_complete=False);bindings.append(b);card['learned_role_binding']=b
  if state=='contradicted' and not card.get('observation_quarantine_overlap'):
   jc.integration.add_warning(card,'role_direction','표시된 구간의 이동 방향을 바꿔 보세요. '+spec['instruction']);card['highlight_spans']=[d['span'] for d in ds if d['eligible'] and d['confidence']>=.65]
 required=[i for i,s in enumerate(contracts) if s['required'] and not s.get('parent')];order=[]
 for ia,ib in zip(required,required[1:]):
  aa,bb=bindings[ia]['spans'],bindings[ib]['spans'];state=lc.order_state(aa,bb);order.append(dict(first=contracts[ia]['id'],second=contracts[ib]['id'],state=state,first_spans=aa,second_spans=bb))
  if state=='contradicted' and not any(out['cards'][i].get('observation_quarantine_overlap') for i in [ia,ib]):
   c=out['cards'][ib];jc.integration.add_warning(c,'role_order','이 두 행동의 순서를 확인해 보세요. 먼저 '+contracts[ia]['instruction']);c['highlight_spans']=[[s['a'],s['b']] for s in aa+bb]
 completion=[];returns=[s for s in spans if s['family']==9]
 for j,i in enumerate(required):
  c=out['cards'][i];spec=contracts[i];observed=bindings[i]['spans'];st='action_observed_completion_unconfirmed' if observed else 'unobserved_or_unassigned';evidence={}
  # No assertion that any detected endpoint completes the sign. A cautious
  # omission candidate needs a preceding action, observed return, continuous
  # relevant joints, and no positive evidence of the requested action.
  if j>0 and not observed:
   previous=bindings[required[j-1]]['spans'];ends=[s for s in returns if previous and s['a']>max(a['b'] for a in previous)]
   if previous and ends:
    a=min(s['a'] for s in previous);b=min(s['b'] for s in ends);ks=sorted({k for fam in spec['families'] for k in ([0] if fam<=3 else [4,5] if fam==4 else [6,7])});valid=np.logical_and.reduce([q[a:b+1,k,-2:].max(-1)>.5 for k in ks]);gaps=bc.groups(~valid);maxgap=max((ts[a+v]-ts[a+u] for u,v in gaps),default=0);maxprob=float(prob[a:b+1,spec['families']].sum(-1).max());coverage=float(valid.mean());evidence=dict(span=[a,b],coverage=coverage,max_missing_us=int(maxgap),max_role_probability=maxprob,return_spans=ends)
    if coverage>=.9 and maxgap<100000 and maxprob<.35 and not c.get('observation_quarantine_overlap'):
     st='omission_candidate';jc.integration.add_warning(c,'role_omission','앞 동작 뒤에 이 행동이 확인되지 않았어요. '+spec['instruction']);c['highlight_spans']=[[a,b]]
    else:st='unconfirmed_observation_or_assignment'
  completion.append(dict(card=c['id'],state=st,evidence=evidence,semantic_complete=False))
 out.update(role_binding_mode=mode,role_candidate_ledger=ledger,role_selected=selected,role_pair_checks=pairs,role_order_checks=order,role_completion_checks=completion,role_probabilities=prob.tolist())
 return out
