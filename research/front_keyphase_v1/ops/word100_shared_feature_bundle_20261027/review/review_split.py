import json, hashlib, collections
from pathlib import Path
U=Path(__file__).resolve().parents[1]
f=json.loads((U/'FROZEN.json').read_text(encoding='utf-8'))
r=json.loads((U/'RUBRIC.json').read_text(encoding='utf-8'))
rows=f['rows']; roles=collections.Counter(x['role'] for x in rows)
checks={
 '36_clips':len(rows)==36,
 'unique_clip_ids':len({x['clip_id'] for x in rows})==36,
 'unique_source_content':len({x['source']['sha256'] for x in rows})==36,
 'split_counts':dict(roles)=={'fit_candidate':16,'held_person':8,'held_expression_seen_person':8,'held_expression_new_person':4},
 'new_clips_24':sum(x['is_new'] for x in rows)==24,
 'fixed_static_ordinals':r['static_sample_ordinals']==[5,8,11],
 'rubric_pre_image_freeze_recorded':r['frozen_before_new_RGB_review'] is True,
}
fit={x['word_id'] for x in rows if x['role']=='fit_candidate'}
held={x['word_id'] for x in rows if x['role'].startswith('held_expression')}
checks['expression_disjoint']=not (fit & held)
checks['fit_8_and_held4']=len(fit)==8 and len(held)==4
checks['person_roles']=all(x['recorded_person_id'] in ([1,2] if x['role'] in ('fit_candidate','held_expression_seen_person') else [3]) for x in rows)
out={
 'status':'PASS_WITH_SCOPE_LIMITS' if all(checks.values()) else 'ISSUE',
 'reviewer':'Separate Codex review; not independent human ground truth',
 'checks':checks,'role_counts':dict(roles),
 'pins':{p:hashlib.sha256((U/p).read_bytes()).hexdigest() for p in ('FROZEN.json','RUBRIC.json')},
 'judgment':'Valid finite preparation design for coarse visible-feature feasibility. This is not an independent C/T3-F test and does not establish educational correctness.',
 'required_reporting':[
   'Keep fixed splits and fixed ordinals even when feature coverage fails. Report positive, negative and unknown counts per feature and split, plus distinct clips, expressions and recorded IDs.',
   'Unknown is not a negative. Invisible fingers or uncertain hand identity must not become absent-feature labels.',
   'Static samples, sequences, clips and people are different units. Three or sixteen frames from one clip are not independent participants.',
   'The two held-expression groups must remain separate: familiar recorded IDs versus ID03 also excluded from fit.',
   'All 36 videos were previously TCN training material. Frozen C/T3-F embeddings cannot make this an unseen-expression test of that pretrained representation.',
   'Two Codex image reviews supply consistency evidence only. No independent human, native signer, exact contact, hidden finger or linguistic-error ground truth is created.',
   'The minimum two fit expressions per polarity is a composition gate, not proof of sufficient sample size or generalization.',
   'Sequence zero means no open/grouped change observed among the sampled frames with adequate visibility. It cannot label the complete continuous video as having no change.',
   'No training is run in this preparation unit; report coverage readiness separately from correctness-learning readiness.'
 ],
 'image_review_protocol':{
   'new_images_not_yet_seen_when_this_review_written':True,
   'alias_mapping_previously_visible_in_frozen_manifest':True,
   'blindness_limit':'Image-first independent of root labels/source profiles, but not fully blind because manifest word IDs were seen.',
   'static_samples':[5,8,11],
   'negative_rule':'0 requires visible contradictory evidence, not missing evidence.',
   'sequence_scope':'16 sampled images; per-hand change evidence must cite sample ordinals.',
   'no_source_or_root_label_lookup_before_label_freeze':True
 }
}
(U/'review'/'SPLIT_METHOD_REVIEW.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':out['status'],'checks':checks,'counts':dict(roles)}))
