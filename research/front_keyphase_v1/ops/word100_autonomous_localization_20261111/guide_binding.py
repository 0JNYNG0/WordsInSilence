from localization_core import *

def profiles():
 """Build observed endpoint ranges from FIT labels, never evaluation examples."""
 cfg=jc.ep.family_contracts();data={};rows={r['clip_id']:r for r in fitrows()}
 for ar in read(U/'FIT_LABELS.json')['rows']:
  r=rows[ar['clip_id']];word=r['word_id']
  if word not in cfg:continue
  for spec in cfg[word]:
   if spec['axis']!=ar['axis'] or ar['moving']!='R':continue
   key=spec['id'];v,valid=prior.wrist_series(prior.physical(r['x_full']),prior.motion.KINDS[ar['axis']],'R','L')
   for e in ar['events']:
    # Roles use checked guide annotations. Expected direction is not used to
    # select/erase an event at runtime; here these are original normal FIT clips.
    a,b=e['onset'],e['offset']
    if not valid[a[0]:a[1]+1].all() or not valid[b[0]:b[1]+1].all():continue
    data.setdefault(key,[]).append(dict(clip_id=r['clip_id'],start=v[a[0]:a[1]+1].mean(0).tolist(),end=v[b[0]:b[1]+1].mean(0).tolist(),direction=e['direction']))
 out={}
 for key,rs in data.items():
  starts=np.asarray([r['start'] for r in rs]);ends=np.asarray([r['end'] for r in rs])
  out[key]=dict(rows=rs,start_min=starts.min(0).tolist(),start_max=starts.max(0).tolist(),end_min=ends.min(0).tolist(),end_max=ends.max(0).tolist())
 return out

def attach(before,events,x,ts,fh,base,motion,profiles,combined,old_cache):
 out=copy.deepcopy(before);contracts=jc.ep.family_contracts()[before['word']];rp=jc.legacy_role(before);q=old.relation_features(x);spans=jc.integration.role_spans(rp,q,ts);bindings=[];ledger=[];groups={s['id']:[] for s in contracts}
 for idx,e in enumerate(events):
  opts=[]
  for spec in contracts:
   if spec['axis']!=e['axis'] or e['moving']!='R':continue
   a,b=e['a'],e['b'];overlap=sum(any(s['family'] in spec['families'] and s['a']<=i<=s['b'] for s in spans) for i in range(a,b+1))/(b-a+1)
   conf=float(rp[a:b+1,spec['families']].sum(-1).mean())
   if overlap>=.25:opts.append((conf,spec['id']))
  opts.sort(reverse=True);assigned=opts[0][1] if opts and (len(opts)==1 or opts[0][0]-opts[1][0]>.1) else None
  item=dict(event=idx,span=[e['a'],e['b']],axis=e['axis'],moving=e['moving'],assigned=assigned,alternatives=opts,reason='body_context_overlap' if assigned else 'unassigned_or_ambiguous',expected_direction_used=False)
  ledger.append(item)
  if assigned:groups[assigned].append(e)
 for card,spec in zip(out['cards'],contracts):
  matches=[s for s in spans if s['family'] in spec['families']];conflicts=jc.integration.conflict(matches,before['phase_probability']);blocked=any(s['conflict'] for s in conflicts);mov=[]
  for e in groups[spec['id']]:
   key=(e['a'],e['b'],e['axis'],e['moving'],'L')
   if key not in old_cache:
    # New localizer-created interval, not re-scoring a completed old interval.
    old_cache[key]=jc.integration.direction(base,x,ts,fh,*key[:3],key[3],key[4])
   od=old_cache[key];nd=mc.direction(motion,x,ts,*key[:3],key[3],key[4]);d=mc.fused(od,nd) if combined else copy.deepcopy(od)
   endpoint=dict(state='unconfirmed',semantic_complete=False)
   pr=profiles.get(spec['id'])
   if pr and len(pr['rows'])>=3 and e['boundary_supported'] and d['eligible']:
    v,valid=prior.wrist_series(x,prior.motion.KINDS[e['axis']],e['moving'],'L')
    # Only the requested relation axis (first scalar) is a completion clue;
    # per-person absolute screen position is never a completion requirement.
    start,end=float(v[e['a'],0]),float(v[e['b'],0]);sok=pr['start_min'][0]<=start<=pr['start_max'][0];eok=pr['end_min'][0]<=end<=pr['end_max'][0]
    endpoint.update(start_relation_in_fit_range=bool(sok),end_relation_in_fit_range=bool(eok),state='observed_relation_endpoints' if sok and eok else 'endpoint_not_confirmed',semantic_complete=False,reason='Endpoint evidence alone does not prove required shape, full sequence or educational correctness')
   d.update(localizer_event=e,endpoint=endpoint);mov.append(d)
  signs={d['prediction']-1 for d in mov if d['eligible'] and d['confidence']>=.65 and d['prediction']!=1};expected=spec['sign'];state='not_specified' if expected is None else 'unknown'
  if expected is not None and len(signs)==1:state='supported' if signs=={expected} else 'contradicted'
  elif expected is not None and len(signs)>1:state='mixed'
  bind=dict(card=card['id'],instruction=spec['instruction'],state=state,observed=mov,body_context_spans=matches,conflict=blocked,unmatched_is_not_omission=True);bindings.append(bind)
  if state=='contradicted' and not blocked and not card.get('observation_quarantine_overlap'):
   msg='표시된 구간의 이동 방향이 반대로 보여요. '+spec['instruction'];jc.integration.add_warning(card,'temporal_localization_direction',msg);card['highlight_spans']=[d['span'] for d in mov if d['eligible'] and d['confidence']>=.65]
  card['automatic_action_binding']=bind
 # Sequence preserves observation timestamps; no expected-order sorting or DTW
 # rewriting is used to manufacture a completed sequence.
 out.update(localization_bindings=bindings,localization_ledger=ledger,observed_event_sequence=sorted(events,key=lambda e:e['a']))
 return out

if __name__=='__main__':
 assert not (U/'GUIDE_ENDPOINT_PROFILES.json').exists();write(U/'GUIDE_ENDPOINT_PROFILES.json',dict(profiles=profiles(),scope='Observed relative endpoint evidence; no semantic completion claim or calibrated acceptance boundary'))
