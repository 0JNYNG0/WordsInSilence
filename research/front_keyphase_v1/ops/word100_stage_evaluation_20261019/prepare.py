from common import *
import shutil
assert not (U/'SCOPE.json').exists()
rows=[r for r in read(B/'training_CONTEXT.json')['ordered_rows'] if r['role']=='train_candidate']
rows=sorted(rows,key=lambda r:r['clip_id'])
assert len(rows)==899 and {r['signer_id'] for r in rows}==set(range(1,10))
refs=read(ROOT/'research/front_keyphase_v1/config/word100_analysis_references.json')['items']
assert len(refs)==100 and all(r['signer_id']==1 for r in refs)
pins={}
def pin(p):pins[str(p)]=sha(p)
for p in [B/'training_CONTEXT.json',B/'PREP_CONTEXT.json',AUD/'VISUAL_REVIEW.json',AUD/'VISUAL_MANIFEST.json',AUD/'SUMMARY.json',M3/'CONCLUSION.json',M3/'COMPARISON_INDEPENDENT.json',ROOT/'research/front_keyphase_v1/config/word100_analysis_references.json',U/'criteria.py',U/'prepare.py']:pin(p)
for p,h in read(M3/'TRAIN_CONTEXT.json')['pins'].items():
    # Preserve all already pinned inputs, models, source code and web files.
    pins[p]=h
for r in rows:
    p=B/'data/records'/f"{r['clip_id']}.json";pin(p);rec=read(p)
    for view in ['sensor','masked']:pin(Path(rec['views'][view]['channel_file']))
for r in refs:
    assert sha(r['video_path'])==r['source_sha256'];pin(Path(r['video_path']))
check(pins)
scope={'at':now(),'authorization':'공개자료에서 동작 단계 누락/순서 평가 기준 정리·검증. 정상 속도/반복을 오류로 단정하지 않기. 추가학습은 결과 함께 검토 후.',
 'status':'FROZEN_DIAGNOSTIC_NO_TRAINING','rows':rows,'references':refs,'views':['sensor','masked'],
 'conditions':['original','clock_slow','clock_fast','clock_local_speed','dwell_samples','omit_middle','swap_first_two','repeat_middle','mask_middle'],
 'public_pairs':'100 fixed ID01 references x9variants x2views; all799 other TRAIN same-word signers x2views, no resampling. Total3398 pair/view rows, each global+local axis.',
 'metrics':'Observed global-hand/body-scale XY and local-hand XY separately. Mean common-joint Euclidean cost. Symmetric directed monotone frame coverage vs unrestricted nearest-neighbour coverage, no learned threshold or mixed pass score.',
 'limitations':['Temporal thirds are mechanical probes, never semantic stage labels.','Clock-only controls test evaluator independence of absolute time, not detector robustness at new frame rates.','Dwell duplicates identical sampled states, not real repeated motion cycles.','Whole cycle repetition count can affect meaning: no universal permit/forbid.','Missing geometric evidence remains unknown; no omission verdict.','Source same-word label does not prove all performances or their steps are correct.','No order/omission threshold chosen from these outputs.'],
 'criteria':{'identity':'Keep TCN word identity separate from completion correctness.','phase_contract':'Requires reviewed observable phases, necessary/optional relations, allowed cycle structure and hand roles. No Korean text splitting into phases.','time':'Absolute duration is not an error criterion; use monotone progress for all preserved events.','repeats':'Dwell/sample duplicates distinct from movement cycles. Reviewer may declare permitted cycles; unknown policy -> review.','observation':'Screen-out/occlusion/identity uncertainty -> insufficient evidence, never absence-as-wrong.','negative_label':'Only reviewed missing required event or reversed required precedence becomes a labelled error. Mechanical edits remain UNREVIEWED_PROBE.','report':'Show normal variation retention, reviewed error sensitivity, abstention and semantic-label coverage separately; no single combined accuracy.'},
 'semantic_ground_truth_available':False,'training_runs':0,'new_model_inference':0,'new_DEV_inference':0,'user_recordings_read':0,'heldout_raw_read':0,
 'pins':pins,'sources':[{'url':'https://jeffe.cs.illinois.edu/teaching/compgeom/refs/Sakoe-Chiba-DTW.pdf','use':'Monotone time alignment background; speech method, not KSL correctness evidence.'},{'url':'https://aclanthology.org/L12-1057/','use':'ASL repetition can serve linguistic purposes; no transfer of exact KSL repetition rules.'},{'url':'https://aclanthology.org/2023.findings-emnlp.846/','use':'Sign-language segmentation distinct from arbitrary temporal thirds.'}]}
write(U/'SCOPE.json',scope)
backup=U/'before';backup.mkdir()
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copyfile(p,backup/p.name)
# A review worksheet for every fixed public reference, not invented labels.
reviews={r['clip_id']:r for r in read(AUD/'VISUAL_REVIEW.json')['reference_reviews']}
write(U/'PHASE_REVIEW_CATALOG.json',{'status':'DRAFT_REQUIRES_SEMANTIC_REVIEW','items':[{'word_code':r['word_id'],'label':r['label'],'clip_id':r['clip_id'],'existing_visual_note':reviews[r['clip_id']]['note'],'existing_sheet':str(AUD/reviews[r['clip_id']]['sheet']),'status':'UNREVIEWED_SEMANTIC_PHASES','required_events':None,'precedence_edges':None,'allowed_repeat_cycles':None,'minimum_visible_evidence':None,'expert_reviewed':False} for r in refs]})
print('FROZEN:899 public TRAIN/100refs;3398 geometry pairs; no learning, model inference or automatic semantic labels.')
