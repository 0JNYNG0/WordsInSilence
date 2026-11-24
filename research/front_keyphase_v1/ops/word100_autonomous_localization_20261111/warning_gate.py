"""Safety-only post-score repair using stored decisions, no inference/tuning."""
from pathlib import Path
import json,copy,hashlib
U=Path(__file__).resolve().parent;R=U/'boundary_integrity_repair';M=U.parent/'word100_short_motion_completion_20261111';G=U/'guide_warning_gate';G.mkdir(exist_ok=True)
rd=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wr(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),'utf-8')
assert not (G/'COMPLETE.json').exists()
wr(G/'FROZEN.json',dict(reason='A partial entry/exit motion is retained as an observed attempt, but is not sufficient alone to identify a guide action and issue a new correction.',rule='Only events with both previously fixed learned boundary tests supported can produce a new guide-direction warning. All partial and wrong-direction/shape candidates remain in the ledger. Existing last-completed combined feedback is preserved.',not_semantic_completion='Boundary support does not prove the required sign was completed. Endpoint/shape/order checks remain separate.',no_threshold_change=True,no_model_change=True,post_score_repair=True,not_fresh_validation=True))
prior={(r['seed'],r['clip_id'],r['kind']):r for r in rd(M/'FULLPATH_COMPLETE.json')['rows'] if r['mode']=='combined'};out=[]
def compact(cards):
 req=[c for c in cards if c.get('required',True)];warning=any(c['display_status']=='difference' for c in req);return dict(display=[c['display_status'] for c in cards],warning=warning,similar_no_warning=all(c['display_status']=='similar' for c in req) and not warning)
for r in rd(R/'FULLPATH_COMPLETE.json')['rows']:
 if r['mode']!='combined':continue
 pr=prior[r['seed'],r['clip_id'],r['kind']];before=rd(pr['result']);candidate=rd(r['result']);cards=copy.deepcopy(before['cards']);decisions=[]
 for card,b,proposed in zip(cards,candidate['bindings'],candidate['cards']):
  supported=[d for d in b['observed'] if d['localizer_event']['boundary_supported'] and d['eligible'] and d['confidence']>=.65 and d['prediction']!=1];signs={d['prediction']-1 for d in supported}
  # Reuse the saved guide contradiction state; never infer the desired sign from
  # a model result or choose whichever interpretation makes a test pass.
  allow=b['state']=='contradicted' and len(signs)==1 and not b['conflict'] and not card.get('observation_quarantine_overlap')
  added=allow and 'temporal_localization_direction' in proposed.get('difference_components',[])
  if added and card['display_status']!='difference':
   for k in ['status','display_status','educational_wrong','message','candidate_message','highlight_spans']: 
    if k in proposed:card[k]=copy.deepcopy(proposed[k])
   card.setdefault('difference_components',[]).append('temporal_localization_direction')
  card['automatic_action_binding']=copy.deepcopy(b);card['automatic_action_binding']['correction_allowed']=bool(allow);card['automatic_action_binding']['semantic_complete']=False
  decisions.append(dict(card=card['id'],raw_state=b['state'],supported_events=len(supported),partial_events=sum(not d['localizer_event']['boundary_supported'] for d in b['observed']),new_warning_allowed=bool(allow),all_observed_attempts_retained=True))
 p=G/'feedback'/str(r['seed'])/f"{r['clip_id']}_{r['kind']}.json";wr(p,dict(cards=cards,ledger=candidate['ledger'],decisions=decisions,baseline=pr['result'],baseline_sha256=pr['sha256'],candidate=r['result'],candidate_sha256=r['sha256'],protected=candidate['protected']));after=compact(cards)
 out.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],before=pr['after'],after=after,result=str(p),sha256=sha(p)))
wr(G/'FULLPATH_COMPLETE.json',dict(rows=out));summary={}
for seed in [42,43,44]:
 for kind in sorted({r['kind'] for r in out}):
  rows=[r for r in out if r['seed']==seed and r['kind']==kind];summary[f'{seed}/{kind}']=dict(total=len(rows),before_warnings=sum(r['before']['warning'] for r in rows),after_warnings=sum(r['after']['warning'] for r in rows),before_similar=sum(r['before']['similar_no_warning'] for r in rows),after_similar=sum(r['after']['similar_no_warning'] for r in rows),changed=[r['clip_id'] for r in rows if r['before']!=r['after']])
wr(G/'RESULT.json',summary);wr(G/'COMPLETE.json',dict(rows=len(out),neural_inferences=0,post_score_conservative_safety_gate=True));print(json.dumps({k:v for k,v in summary.items() if v['changed']},ensure_ascii=False))
