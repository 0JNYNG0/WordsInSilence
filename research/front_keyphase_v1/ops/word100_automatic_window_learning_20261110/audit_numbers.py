from window_core import *
from collections import Counter
rr=prior.rows();fit=[r for r in rr if r['role']=='fit'];dev=[r for r in rr if r['role']=='development'];main=[r for r in dev if r['word']>=0];labels=read(P/'ADDITIONAL_LABELS.json');ev=read(P/'EVALUATION_COMPLETE.json')
shape_by_word={r['word_id']:0 for r in main};shape_by_feature=Counter();motions=Counter()
for r in main:
 shape_by_word[r['word_id']]+=int(r['direct_ym'].sum());motions[r['word_id']]+=len(r['pairs'])
 for j,v in enumerate(r['direct_ym'].sum(0)):shape_by_feature[j]+=int(v)
allwords={r['word_id'] for r in fit};missing={str(person):sorted(allwords-{r['word_id'] for r in fit if r['person']==person}) for person in sorted({r['person'] for r in fit})}
added_motion=sum(map(len,labels['motion'].values()));added_shape=sum(len(q['features']) for rows in labels['shape'].values() for q in rows);added_orders=sum(map(len,labels['orders'].values()))
assert (len(fit),len(allwords),added_motion,added_shape,added_orders,sum(shape_by_word.values()),sum(motions.values()))==(899,100,31,76,63,228,32)
d=dict(fit_clips=899,fit_words=100,fit_by_record_id=dict(Counter(r['person'] for r in fit)),missing_word_ids=missing,development_clips=len(dev),fine_label_scope=read(P/'READY.json')['fine_label_words'],new_motion_intervals=added_motion,new_shape_feature_cells=added_shape,new_order_pairs=added_orders,reviewed_shape_by_word=shape_by_word,reviewed_shape_by_feature=dict(shape_by_feature),movement_intervals_by_word=dict(motions),previous_primary={k:ev['models'][k]['by_variant']['normal'] for k in ['coupled_42','factorized_42']},source_files=[str(P/'READY.json'),str(P/'ADDITIONAL_LABELS.json'),str(P/'EVALUATION_COMPLETE.json')],no_model_inference=True,independent_people_claim=False,expert_truth_claim=False)
assert d['previous_primary']['coupled_42']['reviewed_shape']['correct']==227 and d['previous_primary']['coupled_42']['movement']['correct']==32 and d['previous_primary']['factorized_42']['movement']['correct']==29
write(U/'NUMBERS_CHECK.json',d);print(json.dumps({k:d[k] for k in ['fit_by_record_id','reviewed_shape_by_word','movement_intervals_by_word']},ensure_ascii=False))
