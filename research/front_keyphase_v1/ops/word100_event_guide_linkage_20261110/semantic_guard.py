"""Conflict veto only. Keeps all evidence and cannot add a positive judgment."""
import copy
OLD_CONTEXT={0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:6,9:7,10:7,11:8,12:9,13:9}

def context_conflict(span,old_prob):
 votes=[]
 for j in range(span['a'],span['b']+1):
  row=old_prob[j];idx=max(range(len(row)),key=lambda k:row[k]);votes.append(row[idx]>=.65 and OLD_CONTEXT[idx]!=span['context'])
 return dict(span=[span['a'],span['b']],new_context=span['context'],old_conflicting_frames=sum(votes),frames=len(votes),conflict=sum(votes)>len(votes)/2)

def guard(candidate,baseline):
 out=copy.deepcopy(candidate);byid={c['id']:c for c in baseline['cards']};audit={}
 for b in candidate['guide_link_bindings']:
  cc=[context_conflict(s,baseline['phase_probability']) for s in b['context_spans']]
  audit[b['card']]=dict(checks=cc,conflict=any(r['conflict'] for r in cc),same_source_models_not_independent_truth=True)
 veto=[]
 for card in out['cards']:
  base=byid[card['id']];new=set(card.get('difference_components',[]))-set(base.get('difference_components',[]));blocked=[]
  if 'guide_link_direction' in new and audit[card['id']]['conflict']:blocked.append('guide_link_direction')
  if 'guide_link_order' in new:
   relevant=[r for r in candidate['guide_link_order_checks'] if r['pair'][1]==card['id'] and r['state']=='contradicted']
   if any(any(audit[k]['conflict'] for k in r['pair']) for r in relevant):blocked.append('guide_link_order')
  if blocked:
   veto.append(dict(card=card['id'],components=blocked,reason='Two observed-context estimates disagree; new correction is not established.'))
   card['difference_components']=[k for k in card.get('difference_components',[]) if k not in blocked]
   if not (set(card['difference_components'])-set(base.get('difference_components',[]))):
    for k in ['status','display_status','educational_wrong','message','candidate_message']:
     if k in base:card[k]=copy.deepcopy(base[k])
     else:card.pop(k,None)
   card['guide_link_correction_veto']=blocked
 out['guide_link_context_conflicts']=audit;out['guide_link_vetoes']=veto;out['guide_link_conflict_guard']='v1_no_new_positive_or_deleted_evidence'
 return out
