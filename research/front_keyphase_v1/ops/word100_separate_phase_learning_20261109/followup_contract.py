"""Post-hoc, generic evidence-consistency guard. No weights or thresholds changed.

The frozen six-model comparison is retained. A discarded plausible run cannot
silently contradict the selected run while the card claims similarity. This
guard withholds judgement; it does not declare which run was the true event.
"""
from lib import *
import copy

def apply_guard(decision, attrs, raw, ts):
 out=copy.deepcopy(decision)
 for card in out['cards']:
  chosen=card.get('query_indices');reviews=[];seen=set()
  for phase in card.get('phase_evidence',[]):
   for span in phase.get('available_runs',[]):
    if tuple(span) in seen or (chosen and chosen[0]<=span[0] and span[1]<=chosen[1]):continue
    seen.add(tuple(span))
    features=positive.shape_evidence(attrs,raw,ts,span,card['features'],card.get('shape_scope')=='late')
    if any(f['state']=='contradicted' for f in features):reviews.append(dict(span=span,features=features))
  card['alternative_run_conflicts']=reviews
  if reviews and card.get('display_status')=='similar':
   card.update(previous_guard_display='similar',display_status='uncertain',status='unknown',
       uncertainty_reason='candidate_intervals_disagree',
       candidate_message='앞뒤 구간의 손모양 근거가 달라, 이 부분은 아직 판단하기 어려워요.',
       message='앞뒤 구간의 손모양 근거가 달라, 이 부분은 아직 판단하기 어려워요.',
       educational_wrong=False)
  # No successful phase is inferred from the mere presence of visible hands.
  if card.get('display_status')=='uncertain' and not card.get('uncertainty_reason') and chosen is None:
   indices=[0 if hand=='L' else 1 for hand in card['hands']]
   hand=np.stack([raw[:,237:258].sum(1)>=5,raw[:,258:279].sum(1)>=5],1)
   observable=evidence.runs(hand[:,indices].all(1),ts,read(D/'FROZEN.json')['phase']['min_evidence_ms'])
   card['uncertainty_reason']='phase_unresolved_despite_some_visible_hands' if observable else 'insufficient_observation'
 return out

def main():
 assert not (P/'FOLLOWUP_RESULT.json').exists()
 contract=dict(scope='Post-hoc diagnostic followup after frozen comparison; no new training or neural inference',
   mechanism='Highest-peak candidate may omit another run with observed opposite handshape.',
   rule='Check all discarded candidate runs separately with unchanged positive.shape_evidence. If any contradicts the card and selected card was similar, withhold as uncertainty, never auto-correct.',
   thresholds='Reuse existing thresholds; no word/person branches; preserve gaps between runs.',
   inputs_sha256=sha(P/'EVAL_INPUTS.json'),baseline_scores_sha256=sha(P/'EVALUATION_COMPLETE.json'),
   implementation_sha256=sha(P/'followup_contract.py'),scope_not_claimed=['No recovery of a phase that was not proposed','No independently validated event boundary','No deployment','No expert educational correctness'])
 write(P/'FOLLOWUP_FROZEN.json',contract)
 inputs={(r['clip_id'],r['kind']):r for r in read(P/'EVAL_INPUTS.json')['rows']};output={};all_changes=[]
 for name in read(P/'EVALUATION_COMPLETE.json')['models']:
  rows=[]
  for case in read(P/'scores'/f'{name}_cases.json'):
   src=inputs[(case['clip_id'],case['kind'])];z=np.load(src['path']);old=read(case['decision']);new=apply_guard(old,z['p'],z['raw_x'],z['timestamp_us']);changed=[]
   for a,b in zip(old['cards'],new['cards']):
    if a['display_status']!=b['display_status']:changed.append(dict(card=a['id'],before=a['display_status'],after=b['display_status'],selected=b['query_indices'],conflicts=b['alternative_run_conflicts']))
   row=dict(clip_id=case['clip_id'],kind=case['kind'],before=case['display'],after=[c['display_status'] for c in new['cards']],required=case['required'],changes=changed,uncertainty_reasons=[c.get('uncertainty_reason') for c in new['cards']])
   rows.append(row)
   if changed:all_changes.append(dict(model=name,**row))
  path=P/'followup'/f'{name}.json';write(path,rows)
  output[name]=dict(rows=str(path),sha256=sha(path),changed_clips=sum(bool(r['changes']) for r in rows),normal_full_changed=sum(bool(r['changes']) for r in rows if r['kind']=='full_normal'),normal_crop_changed=sum(bool(r['changes']) for r in rows if r['kind']=='crop_normal'))
 write(P/'FOLLOWUP_RESULT.json',dict(models=output,changed=all_changes,applied_to_web=False,no_neural_reinference=True,prior_results_unchanged=True))
 print(json.dumps(dict(models=output,changes=[dict(model=r['model'],clip=r['clip_id'],kind=r['kind'],before=r['before'],after=r['after']) for r in all_changes]),ensure_ascii=False))
if __name__=='__main__':main()
