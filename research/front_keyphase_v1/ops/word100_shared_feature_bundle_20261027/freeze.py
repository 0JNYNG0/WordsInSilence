import json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
U=Path(__file__).resolve().parent; ROOT=U.parents[3]; OPS=U.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 assert not (U/'FROZEN.json').exists()
 old=['WORD0949','WORD1139','WORD1290','WORD2551']
 new=['WORD0003','WORD0071','WORD0500','WORD0678','WORD0739','WORD1133','WORD1567','WORD2523']
 held=['WORD0003','WORD0071','WORD0500','WORD2523']
 fit=old+[x for x in new if x not in held]
 prep=OPS/'word100_linguistic_feedback_pipeline_20261027/FEATURE_LABEL_PREPARATION.json'
 profiles=OPS/'word100_linguistic_feedback_pipeline_20261027/FEATURE_PROFILES.json'
 selected=[]
 for w in sorted(old+new):
  p=next(r for r in read(prep)['rows'] if r['word_id']==w)
  for person in [1,2,3]:
   row=next(r for r in p['public_records'] if r['recorded_person_id']==person)
   record=read(Path(row['record'])); assert record['role']=='train_candidate'
   role='held_expression_new_person' if w in held and person==3 else 'held_expression_seen_person' if w in held else 'held_person' if person==3 else 'fit_candidate'
   selected.append(dict(word_id=w,label=p['label'],recorded_person_id=person,clip_id=row['clip_id'],role=role,record=row['record'],record_sha256=sha(Path(row['record'])),source=record['source'],intervals=row['word_interval_seconds'],is_new=w in new))
 old_index=OPS/'word100_source_token_alignment_20261027/visual_inputs/INDEX.json'
 old_obs=OPS/'word100_source_token_alignment_20261027/TOKEN_OBSERVATIONS.json'
 for i,r in enumerate([r for r in selected if r['is_new']],1):r['alias']=f'X{i:02d}'
 for r in selected:
  if not r['is_new']:r['alias']=next(x['alias'] for x in read(old_index)['rows'] if x['clip_id']==r['clip_id'])
 pins={str(p):sha(p) for p in [prep,profiles,old_index,old_obs]}
 save(U/'FROZEN.json',dict(at=datetime.now(timezone.utc).isoformat(),scope='User continuation: shared-feature public preparation and frozen held-person/expression split',
   selection_basis='Eight added expressions selected from existing dictionary descriptions before new RGB outcome review, targeting overlapping index/open/grouped/body/hand relations and contrasts. Source text used for retrieval only; image labels not generated from text.',
   rows=selected,fit_expressions=fit,held_expressions=held,recorded_people=[1,2,3],
   counts=dict(old_clips=12,new_clips=24,total_clips=36,fit=16,held_person=8,held_expression_seen_person=8,held_expression_new_person=4),
   expected_scope='All belong to previous TCN TRAIN. This is prospective split for new feature work, not an unseen test of C/T3-F.',
   evaluation_policy='Freeze now, no result-driven swaps. Report lack of positive/negative or expression coverage. Rows are data roles, not authorization to train on draft labels.',
   feature_coverage_gate='For a binary coarse feature require positive and negative examples across >=2 fit expressions each, plus positives and negatives in held-expression data; report per-person/expression counts. This is a minimal composition check, not statistical sufficiency or truth validation.',
   labels='present/absent/unknown from actual visible RGB; lack of visibility is unknown. Side and timestamp retained. Motion is sequence evidence, not instantaneous frame truth. Same-AI crossreview not independent expert truth.',
   preservation=['C','T3-F','operating web/DTW','old results','source videos'],excluded=['user/mentor footage','new detector','facial expressions','downloads','Git','training on unverified source descriptions'],
   source_pins=pins,prior_observations=str(old_obs)))
 (U/'before').mkdir(exist_ok=True)
 for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',OPS/'STATE.json']:shutil.copy2(p,U/'before'/p.name)
 save(U/'SOURCE_PROFILES.json',dict(profiles=[p for p in read(profiles)['profiles'] if p['word_id'] in old+new],warning='Retrieval candidates only; not observed labels'))
 print('Frozen 36 clips:16fit/8held person/8held expression seen person/4held both')
if __name__=='__main__':main()
