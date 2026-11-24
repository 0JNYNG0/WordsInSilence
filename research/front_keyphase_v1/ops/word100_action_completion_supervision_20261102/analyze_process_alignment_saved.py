"""Read saved scalar predictions/targets only; no model or alternative decision rule."""
from pathlib import Path
from collections import Counter
import hashlib
import json

U = Path(__file__).resolve().parent
read = lambda n: json.loads((U / n).read_text(encoding='utf-8-sig'))
sha = lambda n: hashlib.sha256((U / n).read_bytes()).hexdigest()
raw = read('PROCESS_EVALUATION.json')
auto = read('PROCESS_AUTO_EVALUATION.json')
raw_by = {r['view_id']: r for r in raw['rows']}
rows = []
for v in auto['view_rows']:
    if v['truth'] != 1:
        continue
    r = raw_by[v['view_id']]
    positive = [a for a in r['interval_scores'] if a['truth'] == 1]
    candidates = v['qualification']['interval_candidates']
    if not positive or not candidates:
        continue
    pairs = []
    for c in candidates:
        cs, ce = c['start'], c['end']
        for a in positive:
            aa = a['output_indices']; s, e = aa[0], aa[-1]
            if cs == s and ce == e: relation = 'same_interval'
            elif cs <= s and ce >= e: relation = 'selected_contains_annotation'
            elif s <= cs and e >= ce: relation = 'selected_inside_annotation'
            elif ce < s: relation = 'selected_before_annotation'
            elif cs > e: relation = 'selected_after_annotation'
            elif cs < s: relation = 'overlap_but_selected_ends_earlier'
            else: relation = 'overlap_but_selected_starts_later'
            pairs.append(dict(relation=relation, selected_source_extent=[r['source_frames'][cs], r['source_frames'][ce]],
                annotated_source_extent=[r['source_frames'][s], r['source_frames'][e]],
                selected_probability=c['motion_probability'], annotated_probability=a['probability'],
                selected_length=ce-cs+1, annotated_length=e-s+1))
    rows.append(dict(view_id=v['view_id'], role=v['role'], query=v['physical_query'], kind=v['kind'],
        candidates_all_low=all(c['motion_probability'] < .8 for c in candidates),
        annotated_positive_any_high=any(a['probability'] >= .8 for a in positive), pairs=pairs))

summary = {}
for role in ('fit', 'development'):
    rr = [r for r in rows if r['role'] == role]
    lows = [r for r in rr if r['candidates_all_low'] and r['annotated_positive_any_high']]
    summary[role] = dict(
        scope_comparable_positive_views=len(rr),
        high_annotated_low_selected_views=len(lows),
        low_views_with_any_containing_pair=sum(any(p['relation'] in ('same_interval', 'selected_contains_annotation') for p in r['pairs']) for r in lows),
        all_candidate_annotation_pair_index_relations=dict(Counter(p['relation'] for r in rr for p in r['pairs'])),
        high_annotated_low_selected_pair_index_relations=dict(Counter(p['relation'] for r in lows for p in r['pairs'])),
        per_query={q: dict(views=sum(r['query'] == q for r in rr),
            high_annotated_low_selected=sum(r['query'] == q for r in lows),
            pair_index_relations=dict(Counter(p['relation'] for r in rr if r['query'] == q for p in r['pairs'])))
            for q in sorted({r['query'] for r in rr})})

fit = read('PROCESS_VIEW_TARGETS.json')['rows']
witnesses = []
for r in fit:
    if r['observed_process_evidence'] != 1:
        continue
    sparse = {p['output_index']: p for p in r['sparse_relation_targets']}
    for side in ('setup', 'terminal'):
        for w in r['bound_' + side + '_witnesses']:
            i = w['output_index']; target = sparse.get(i)
            value = target[side + '_relation'] if target else None
            witnesses.append(dict(view_id=r['view_id'], query_id=r['query_id'], side=side,
                source_frame=w['source_frame'], output_index=i, sparse_target=value,
                coverage='known_positive' if value == 1 else 'known_negative_conflict' if value == 0 else 'not_consumed_as_known_point',
                saved_probability=raw_by[r['view_id']]['point_probabilities'][i][0 if side == 'setup' else 1]))

report = dict(status='SAVED_ALIGNMENT_DIAGNOSTIC_ONLY', scope_summary=summary, scope_rows=rows,
    fit_positive_direct_point_witness_consumption=dict(records=len(witnesses),
        dependent_source_queries=len({w['query_id'] for w in witnesses}),
        counts=dict(Counter(w['coverage'] for w in witnesses)),
        unconsumed_or_conflicting=[w for w in witnesses if w['coverage'] != 'known_positive']),
    limitations=[
        'Index overlap is not local semantic truth. No candidate was relabelled or rescored.',
        'Every candidate-to-annotation pair is counted; these are correlated diagnostic pairs, not independent events.',
        'Bound witnesses are read as documented physical evidence. Unconsumed points are not silently promoted from unknown to a training target.',
        'No counterfactual last-setup, threshold or candidate selection rule was evaluated.'
    ], new_inference=0, new_training=0, new_score=0,
    source_pins={n: sha(n) for n in ('PROCESS_EVALUATION.json', 'PROCESS_AUTO_EVALUATION.json', 'PROCESS_VIEW_TARGETS.json')})
p = U / 'PROCESS_ALIGNMENT_COUNTS.json'
assert not p.exists()
p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(summary=summary, point_consumption={k:v for k,v in report['fit_positive_direct_point_witness_consumption'].items() if k != 'unconsumed_or_conflicting'}), ensure_ascii=False))
