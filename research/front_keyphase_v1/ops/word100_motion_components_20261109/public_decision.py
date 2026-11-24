from common import *
from component_coverage import apply_coverage
import copy,collections,ast

def main():
 assert not (U/'PUBLIC_DECISION.json').exists()
 source=O/'word100_guide_video_feedback_20261028/GUIDE_CARDS.json'
 contract=dict(word='WORD2523',card='spread_hands',coverage_message='두 손의 이동과 손끝 모으기는 따로 확인해야 해요. 마지막 손끝 모양은 아직 확인하지 못했어요.',coverage_requirements=[dict(component='fingertip_convergence:'+h,label=('오른손' if h=='R' else '왼손')+' 마지막 손끝 모양',scope='late_core',source_applicability='unresolved_source_variant') for h in ['R','L']],provenance=dict(path=str(source),sha256=sha(source),candidate_origin_ids=['23905','5990'],note='Cached23905 contains each-hand gathering suffix;5990 does not. Do not impose one variant on the other. No current expert or framewise all-tip annotation. Current runtime open_four output cannot establish convergence.'))
 write(U/'COVERAGE_CONTRACTS.json',dict(contracts=[contract]))
 review=read(U/'FINGERTIP_REVIEW_INPUT.json');write(U/'FINGERTIP_REVIEW.json',dict(inputs_sha256=sha(U/'FINGERTIP_REVIEW_INPUT.json'),reviewer='same Codex visual review, not independent human/expert',public_clips=len(review['rows']),frames=sum(len(r['frames']) for r in review['rows']),hand_crops=sum(2*len(r['frames']) for r in review['rows']),finding='Visible long fingers form a flat grouped posture. Thumb and all-tip contact are not consistently visible. Returning hands in last samples are not automatically the required final core shape.',labels_created=0,reason='Do not equate open_four=0, a fist, two-hand distance, or a sign name with within-hand fingertip convergence.'))
 old=read(O/'word100_positive_direction_20261109/COMPARISON.json')['rows'];out=[]
 for r in old:
  before=read(r['decision'])['after'];request=copy.deepcopy(before)
  for c in request['cards']:
   if r['word']==contract['word'] and c['id']==contract['card']:c.update({k:contract[k] for k in ['coverage_requirements','coverage_message']})
  after=apply_coverage(request)
  # Known contradictory evidence is preserved, even with an unresolved component.
  for a,b in zip(before['cards'],after['cards']):
   if a.get('display_status')=='difference':assert b['display_status']=='difference'
   if r['word']!=contract['word']:assert a==b
  out.append(dict(clip_id=r['clip_id'],variant=r['kind'],word=r['word'],before=[c.get('display_status') for c in before['cards']],after=[c.get('display_status') for c in after['cards']],result=after))
 write(U/'COVERAGE_REPLAY.json',dict(rows=out,scope='Cached public outputs, deterministic contract replay; not new detector/model accuracy',private=False))
 rr=read(U/'SCORES.json');saved=read(U/'EVALUATION_COMPLETE.json');checks=0
 for model,variants in saved['results'].items():
  for variant,aggregate in variants.items():
   selected=[r for r in rr if r['model']==model and r['variant']==variant]
   assert len(selected)==32==aggregate['n'];checks+=1
   assert sum(r['eligible'] and r['pred']==r['target'] for r in selected)==aggregate['raw_correct'];checks+=1
   assert sum(not(r['eligible'] and r['confident']) for r in selected)==aggregate['withheld'];checks+=1
   assert len({(r['clip_id'],r['pair']) for r in selected})==32;checks+=1
 # Every saved fitted model is fixed before public inference. No DEV fit entries.
 fit=read(U/'FIT_MANIFEST.json');assert {m['person'] for m in fit}==set(range(1,10));checks+=1
 assert all('REAL1' not in m['clip_id'] for m in fit);checks+=1
 for run in read(U/'TRAINING_COMPLETE.json')['runs']:
  path=U/'runs'/f"{run['condition']}_{run['seed']}"/'model.pt';assert sha(path)==run['checkpoint_sha256'];checks+=1
 for p,s in read(U/'FROZEN.json')['pins'].items():assert sha(p)==s;checks+=1
 for name in ['motion.py','component_coverage.py']:
  strings=[n.value for n in ast.walk(ast.parse((U/name).read_text('utf-8'))) if isinstance(n,ast.Constant) and isinstance(n.value,str)];assert not any('WORD' in s or '이집트' in s or '잘하다' in s for s in strings);checks+=1
 write(U/'PUBLIC_VERIFICATION.json',dict(passed=True,checks=checks,review_type='Separate saved-output arithmetic/code checks by same Codex; not independent expert labels',scores_sha256=sha(U/'SCORES.json'),training_manifest_sha256=sha(U/'FIT_MANIFEST.json'),source_labels='Same Codex sparse public RGB review',private_learning=False))
 write(U/'PUBLIC_DECISION.json',dict(status='PARTIAL_DIRECTION_IMPROVEMENT_AND_COVERAGE_FIX',primary='motion_only_42',direction=dict(previous_geometry='26/32',previous_latent_finetune='28/32',new='31/32',actual_movement='20/21 -> 21/21',hold='6/11 -> 10/11',normal_arm_stroke='3/3 baseline, 1/3 previous fine-tune, 3/3 new',automatic_combined_fault='New head misses1/3 wrong-shape+reverse; deployed geometry catches3/3',deploy_model=False,reason='Supplied-window improvement is real within historical DEV; combined auto-window regression prevents replacement. No further DEV threshold or weight tuning.'),coverage=dict(deploy_research=True,claim='Stops unsupported component borrowing; does not detect final fingertip error yet',source_variant_unresolved=True,verified_classifier=False),preserve=['C','T3-F','current RTMW TCN','current Conv+GRU weights','current signed direction measurement'],limits=['4 words/9 FIT IDs/3 historical DEV IDs, no fresh test','Each relation type currently represented in few expressions; generic code is not all-word accuracy','Synthetic reverse/crop/coordinate transforms are not real learner-error accuracy','No educational correctness or hidden fingertip truth'],public_frozen_at=time.time(),private_used=False,pins={f:sha(U/f) for f in ['motion.py','component_coverage.py','COVERAGE_CONTRACTS.json','SCORES.json','AUTO_WINDOW_CHECK.json','PUBLIC_VERIFICATION.json']}))
 print(json.dumps(dict(checks=checks,coverage_public_replays=len(out),changed=sum(r['before']!=r['after'] for r in out)),ensure_ascii=False))
if __name__=='__main__':main()
