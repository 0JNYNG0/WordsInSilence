"""Shared event association: observed core, chronological provenance, and retained alternatives."""
from core import *
import copy
sys.path.insert(0,str(D));consumer=load('_completion_consumer',D/'consumer.py')
saved=sys.modules.get('core');sys.modules['core']=old
prior=load('_completion_old_compare',N/'compare.py')
if saved is not None:sys.modules['core']=saved
GROUPS=[[0],[1],[2],[3],[4],[5,7],[6,8,9],[10],[11,12]]
MAP={s:g for g,ss in enumerate(GROUPS) for s in ss}
runs=consumer.assess.__globals__['runs']
def project(p):
 out=np.zeros((len(p),14),np.float32);out[:,0]=p[:,0]
 for g,ss in enumerate(GROUPS):out[:,ss[0]+1]=p[:,1+2*g]+p[:,2+2*g]
 return out
def decide(word,p,oldphase,oldgeo,d0,d1,attrs,raw,ts,fid):
 phase=project(p);base=consumer.assess(word,phase,d0,d1,raw,ts,fid);oldbase=consumer.assess(word,oldphase,d0,d1,raw,ts,fid);geobase=consumer.assess(word,oldgeo,d0,d1,raw,ts,fid)
 hand=np.stack([raw[:,237:258].sum(1)>=5,raw[:,258:279].sum(1)>=5],1);previous=-1;some=False;assigned=np.zeros(len(ts),bool)
 for card,oc,gc in zip(base['cards'],oldbase['cards'],geobase['cards']):
  available=hand[:,[0 if h=='L' else 1 for h in card['hands']]].all(1);evidence=[];allcore=[];archived=[];unresolved=[];before_previous=False
  for group in card['groups']:
   gs=sorted({MAP[s] for s in group});ep=sum(p[:,1+2*g]+p[:,2+2*g] for g in gs);cp=sum(p[:,2+2*g] for g in gs)
   rr=runs((ep>=.65)&available,ts,60);cr=runs((cp>=.65)&available,ts,60);candidates=[]
   for a,b in rr:
    cores=[(s,e) for s,e in cr if s>=a and e<=b];after=[(s,e) for s,e in cores if s>previous]
    if cores and not after:before_previous=True
    if after:candidates.append(dict(span=[a,b],cores=[list(x) for x in after],peak=float(cp[a:b+1].max())))
   # Each repeated occurrence remains explicit; the first supported core advances order.
   chosen=min(candidates,key=lambda c:c['cores'][0][0]) if candidates else None
   span=chosen['span'] if chosen else None
   if chosen:allcore.append(chosen['cores'][0]);assigned[span[0]:span[1]+1]=True
   accepted=[list(x) for x in rr];other=np.max(np.stack([p[:,2+2*g] for g in range(9) if g not in gs]),axis=0)
   for route,source in [('deployed',oc),('geometry',gc)]:
    for oe in source['phase_evidence']:
     if oe['group']!=group:continue
     for a,b in oe['available_runs']:
      # Exclude an old identity only if a different observed core explains most of it.
      other_support=available[a:b+1]&(other[a:b+1]>=.65)
      reidentified=bool(other_support.mean()>=.65 and runs(other_support,ts[a:b+1],60))
      archived.append(dict(route=route,span=[a,b],group=group,reidentified_as_other_core=reidentified))
      if not reidentified:
       accepted.append([a,b]);unresolved.append([a,b])
   evidence.append(dict(group=group,span=span,peak=float(ep.max()),core_runs=[list(x) for x in cr],event_runs=[list(x) for x in rr],candidates=candidates,available_runs=[list(x) for x in sorted(set(tuple(v) for v in accepted))]))
  complete=len(allcore)==len(card['groups'])
  if len(allcore)>1:complete=complete and len({sum(c) for c in allcore})>1
  span=[min(e['span'][0] for e in evidence),max(e['span'][1] for e in evidence)] if complete else None
  if complete:
   status='supported';previous=max(c[1] for c in allcore);some=True
  elif before_previous:status='order_candidate'
  elif not card.get('required',True):status='optional'
  elif any(e['event_runs'] for e in evidence):status='incomplete_candidate'
  elif some and base['return_to_rest_evidence'] and card['unknown_fraction']<=.2:status='missing_candidate'
  else:status='unknown'
  # Preserve observation gaps as uncertainty, never convert them into omitted movement.
  missing=runs(~available,ts,300)
  active=base['active_indices'];relevant=[(a,b) for a,b in missing if not(b<active[0] or a>active[1])]
  if status in ['missing_candidate','incomplete_candidate'] and relevant:status='unknown'
  if span and any(not(b<span[0] or a>span[1]) for a,b in relevant):status='unknown'
  card.update(status=status,query_indices=span,query_frames=[int(fid[k]) for k in span] if span else None,query_seconds=[float(ts[k]/1e6) for k in span] if span else None,phase_evidence=evidence,core_observed=complete,core_intervals=allcore,candidate_archive=archived,unresolved_old_intervals=unresolved,reference_only=span is None)
 out=consumer.apply_coverage(consumer.refine(base,attrs,raw,ts,fid,phase));out=prior.follow.apply_guard(out,attrs,raw,ts)
 for c in out['cards']:
  if c['display_status']=='similar' and not c['core_observed']:raise AssertionError('similar without core')
  if c['status']=='incomplete_candidate':c['message']='움직임은 보이지만, 필요한 모습까지 이어졌는지 확인하기 어려워요.'
  c['candidate_message']=c['message'];c['educational_wrong']=False
 out.update(progression_version='observed_core_v1',all_old_candidates_archived=True,unassigned_observable_intervals=runs(hand.any(1)&~assigned,ts,60),exact_repeat_count_verified=False,no_dtw_completion=True)
 return out
