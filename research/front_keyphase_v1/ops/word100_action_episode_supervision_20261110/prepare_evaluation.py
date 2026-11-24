from episode_core import *
assert not (U/'EVALUATION_FROZEN.json').exists()
ex=read(U/'EXTRACTION_COMPLETE.json')['rows'];xm={r['word_id']:r for r in ex}
defs={
 'WORD0739':[(0,2,[63,66],[72,75],0),(1,3,[75,78],[81,84],2)],
 'WORD1133':[(2,4,[39,45],[57,63],2)],
 'WORD1567':[(4,5,[48,51],[57,60],0),(4,5,[60,63],[66,69],2),(4,5,[69,72],[75,78],0)],
 'WORD2523':[(3,7,[48,51],[54,57],2)]}
gold=[]
for word,ds in defs.items():
 r=xm[word];assert sha(r['sheet'])==r['sheet_sha256'];assert sha(r['raw'])==r['raw_sha256']
 for axis,fam,on,off,direction in ds:gold.append(dict(clip_id=r['clip_id'],role='additional_evaluation',person=14,word_id=word,axis=axis,family=fam,moving='R',anchor='L',onset=on,offset=off,direction=direction,sheet=r['sheet'],sheet_sha256=r['sheet_sha256'],source_sha256=r['source_sha256']))
dev=[]
for r in read(B/'OBSERVED_BOUNDARIES.json')['rows']:
 if r['role']!='development':continue
 word=r['clip_id'].split('_')[2];spec=[c for c in family_contracts()[word] if c['axis']==r['axis'] and not c.get('parent')];assert len(spec)==1
 for e in r['events']:dev.append(dict(clip_id=r['clip_id'],role='historical_development',person=r['person'],word_id=word,axis=r['axis'],family=spec[0]['families'][0],moving=r['moving'],anchor=r['anchor'],onset=e['onset'],offset=e['offset'],direction=e['direction'],sheet=r['sheet'],sheet_sha256=r['sheet_sha256'],source_sha256=r['source_sha256']))
write(U/'EVALUATION_LABELS.json',dict(historical=dev,additional=gold,review='Same Codex source montage review before predictions; ranges reflect 3-frame sampling. No hidden contact/expert correctness assertion.',extra_scope='RecordID14 not used for feedback training. Previously used by older TCN evaluation, not globally untouched.',intermediate='Between source reviewed onset/offset, not guessed fraction.'))
write(U/'LABEL_REVIEW.json',dict(revisited_fit_sheets=[str(B/'review/NIA_SL_WORD1133_REAL01_F.jpg'),str(B/'review/NIA_SL_WORD2523_REAL01_F.jpg')],findings=['Arm travel to near-wrist arrival is continuous; child state is part of parent action.','Hand separation has actual movement after gathered appearance, followed by hold; hold and lowering are not further separating motion.'],additional_reviewed=4,additional_events=len(gold),historical_events=len(dev),independent_expert=False))
write(U/'EVALUATION_FROZEN.json',dict(primary_seed=42,historical_conditions=78,additional_conditions=['normal','whole_reverse'],additional_originals=4,additional_components_only=True,matching='Axis/moving-hand one-to-one maximum IoU>=0.5. Unmatched events remain denominator. Manual-window diagnosis separate from automatic.',family_mapping=MAP,score_based_tuning=False,pins={str(p):sha(p) for p in [U/'EVALUATION_LABELS.json',U/'evaluate.py',U/'integration.py',U/'episode_core.py',U/'EXTRACTION_COMPLETE.json']}))
print('Labels, manual/automatic comparison and evaluation frozen')
