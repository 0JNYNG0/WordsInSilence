"""Required components cannot borrow support from other measured components.

Data describes which component and time scope is being checked. This evaluator
has no word/card-name branches, coordinate thresholds, or target-shape inference.
An unresolved source variant prevents a comprehensive claim, not an error claim.
"""
import copy

def apply_coverage(result):
 out=copy.deepcopy(result)
 for card in out['cards']:
  requirements=card.get('coverage_requirements',[])
  if not requirements:continue
  observations=card.get('component_observations',{})
  coverage=[]
  for req in requirements:
   key=req['component'];obs=observations.get(key,{})
   source_ok=req.get('source_applicability')=='verified'
   # A numerical score or a different hand's observation cannot satisfy a claim.
   bound=obs.get('scope')==req.get('scope') and obs.get('component')==key
   valid=source_ok and bound and obs.get('observation_sufficient') is True and obs.get('validated') is True
   state=obs.get('state','unknown') if valid else 'unknown'
   if state not in ('supported','contradicted','unknown'):state='unknown'
   coverage.append(dict(component=key,label=req['label'],scope=req.get('scope'),state=state,source_resolved=source_ok))
  pending=[r for r in coverage if r['state']=='unknown'];bad=[r for r in coverage if r['state']=='contradicted']
  card['component_coverage']=coverage;card['coverage_complete']=not pending
  card['unverified_components']=[r['label'] for r in pending]
  if bad:
   card['status']='difference_candidate';card['display_status']='difference'
   card['difference_components']=list(dict.fromkeys(card.get('difference_components',[])+[r['component'] for r in bad]))
   card['message']=' · '.join(r['label'] for r in bad)+' 차이를 확인해 보세요.'
  elif pending and card.get('display_status')=='similar':
   card['status']='partial_evidence';card['display_status']='uncertain'
   card['message']=card.get('coverage_message','확인한 특징은 있지만, '+ ' · '.join(r['label'] for r in pending)+'는 아직 확인하지 못했어요.')
  elif pending and card.get('display_status')!='difference':
   card['message']=card.get('coverage_message','일부 필요한 특징을 아직 확인하지 못했어요. 틀렸다는 뜻은 아니에요.')
  if pending and card.get('display_status')=='difference':
   card['message']+=' 아직 확인하지 못한 특징도 있어요.'
  card['candidate_message']=card['message'];card['educational_wrong']=False
 out['component_coverage_version']='scoped_components_v1'
 return out
