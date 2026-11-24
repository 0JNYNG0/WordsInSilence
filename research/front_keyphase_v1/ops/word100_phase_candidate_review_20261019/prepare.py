from common import *
import shutil
refs=read(O.parent/'config/word100_analysis_references.json')['items']
assert len(refs)==100 and len({r['word_id'] for r in refs})==100
prior=read(E/'SCOPE.json'); pins=dict(prior['pins'])
rows={r['clip_id']:r for r in read(B/'PREP_CONTEXT.json')['ordered_rows']}
reviews={r['clip_id']:r for r in read(AUD/'VISUAL_REVIEW.json')['reference_reviews']}
items=[]
for r in refs:
 assert r['signer_id']==1 and r['source_role']=='train_candidate'
 cid=r['clip_id']; row=rows[cid]; recpath=B/'data/records'/f'{cid}.json'; rec=read(recpath)
 assert len(row['intervals'])==1 and row['role']=='train_candidate'
 assert sha(r['video_path'])==r['source_sha256']
 obs=rec['views']['sensor']['observation']
 for p in [recpath,Path(obs),Path(r['video_path'])]: pins[str(p)]=sha(p)
 items.append(dict(word_code=r['word_id'],label=r['label'],clip_id=cid,signer_id=1,role='train_candidate',video_path=r['video_path'],source_sha256=r['source_sha256'],observation=obs,observation_sha256=sha(obs),record=str(recpath),word_interval=row['intervals'][0],existing_visual_review=reviews[cid],roi=rec['calibration']['box_source_pixels']))
for p in [E/'PHASE_REVIEW_CATALOG.json',E/'NEXT_PLAN.json',E/'FINAL_AUDIT.json',AUD/'VISUAL_REVIEW.json',O.parent/'config/word100_analysis_references.json']: pins[str(p)]=sha(p)
check(pins)
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
 (U/'before').mkdir(exist_ok=True);shutil.copyfile(p,U/'before'/p.name)
write(U/'SCOPE.json',dict(at=now(),authorization='공개100단어의 동작 단계 후보 표시와 필수/선택/반복 검토표. 추가학습은 결과 함께 확인 후.',status='FROZEN_REVIEW_PREPARATION_NO_TRAINING',items=items,pins=pins,
 proposal_rule=dict(window_seconds=.133333,min_between_marks_seconds=.30,max_interior_marks=5,local_peak_radius_seconds=.10,
  rule='Compare before/after window medians only when all21same-hand points and both shoulders are usable throughout. Palm-centre displacement normalized by shoulder width and mean change in15bend cosines are ranked separately. Local maxima above within-clip positive median become bookmark candidates; alternate two ranked lists; merge close marks. No label/model scores. Limits are display budget, not sign correctness thresholds.',
  unknown='Missing or partial21point hand is an observation flag, never an omitted phase or a boundary by itself.',
  intervals='Cover whole-word annotated interval continuously, without changing raw/full video; these are review spans, not semantic phases.',
  invariance='No absolute duration/number of phases/repeats is a correctness rule. One signer per word is insufficient to establish allowed variants.'),
 semantic_fields=['required_optional','required_hands','precedence','repeat_policy','repeat_range','boundary_revision'],
 semantic_labels_confirmed=0,training_runs=0,new_model_inference=0,user_recordings_read=0,heldout_raw_read=0,production_changes=0))
print('Frozen100 public TRAIN references; no training or model inference.')
