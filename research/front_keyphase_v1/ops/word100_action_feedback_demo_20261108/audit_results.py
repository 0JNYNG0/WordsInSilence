"""Separate saved-output arithmetic; never retrains or selects seeds."""
from task import *
import numpy as np
def main():
 first=read(U/'EVALUATION.json');local=read(U/'LOCAL_PHASE_COMPLETE.json');prep=read(U/'PREPARED.json');cards=read(U/'CARDS.json')['words'];anchors=read(G/'BLIND_STAGE_ANCHORS.json')['rows'];out=[];checks=[]
 for family,e in [('context',first),('local',local)]:
  for seed in [42,43,44]:
   rs=[r for r in e['rows'] if r['seed']==seed];assert len(rs)==84
   correct=0;total=0;lost_last=[]
   for r in rs:
    data=read(r['output']);assert sha(r['output'])==r['sha256'];checks.append(r['sha256'])
    if r['kind']=='normal':
     source=next(q for q in prep['rows'] if q['clip_id']==r['clip_id']);z=np.load(source['path']);phase=np.asarray(data['phase_probability'])
     for a in [a for a in anchors if a['clip_id']==r['clip_id']]:
      idx=int(np.flatnonzero(z['frame_id']==a['frame_id'])[0]);pred=int(phase[idx].argmax())-1;label=prep['guide_ids'][pred] if pred>=0 else 'rest';correct+=label in a['allowed_same_physical_configuration'];total+=1
    if r['kind']=='omit_last_main_then_rest' and r['word'] in ['WORD0739','WORD2523']:
     last=[c for c in cards[r['word']] if c.get('required',True)][-1];gs=[prep['guide_ids'][j] for g in last['groups'] for j in g];aa=[a for a in anchors if a['clip_id']==r['clip_id'] and a['stage_id'] in gs];source=next(q for q in prep['rows'] if q['clip_id']==r['clip_id']);z=np.load(source['path']);removed_fids=set(z['frame_id'][r['removed']].tolist());lost_last.append(dict(clip_id=r['clip_id'],deleted_reviewed_anchors=sum(a['frame_id'] in removed_fids for a in aa),reviewed_anchors=len(aa),card_status=r['statuses'][-1]))
   out.append(dict(family=family,seed=seed,stage_correct=correct,stage_total=total,loss_of_reviewed_anchors=lost_last))
 # The original 12-cut gate was stricter than its semantic ground truth justified:
 # a half-cut can retain a valid alternation or sufficient stroke. Keep its first FAIL.
 write(U/'CONTROL_SCOPE_AUDIT.json',dict(first_gate_retained=True,first_gate_not_valid_as_all_word_semantic_error_accuracy=True,reason='Half-main deletion does not guarantee omission of a required whole action, especially a repeated action or a stroke. Counts are engineering detections, not learner-error sensitivity.',strict_independent_whole_event_error_gold_available=False,stage_rows=out,source_anchors_sha256=sha(G/'BLIND_STAGE_ANCHORS.json'),all_stage_totals_92=all(x['stage_total']==92 for x in out),checked_outputs=len(checks),same_codex_separate_arithmetic_not_independent_human=True))
 write(U/'RELEASE_DECISION.json',dict(demonstration_seed=42,seed_choice='fixed before evaluation',new_phase_branch_adopted=False,reason='Local branch did not consistently reduce normal warnings or prove reliable omission detection.',automatic_educational_correction_enabled=False,automatic_pass_enabled=False,normal_hand_comparison_display_enabled=True,neutral_uncertainty_on_candidates=True,first_evaluation_retained=True,training_complete=True))
 print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
