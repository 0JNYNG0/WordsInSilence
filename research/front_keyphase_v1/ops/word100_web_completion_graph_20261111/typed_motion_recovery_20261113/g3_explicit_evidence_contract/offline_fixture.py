"""Wrap saved JSON/PTS into an explicit request. No model modules imported."""
from pathlib import Path
import copy
import numpy as np


def request_from_saved(row, ranked, condition, contract, pts_cache):
    source = next(s for s in contract['sources'].values() if s['clip_id'] == row['clip_id'] and s['kind'] == row['kind'])
    path = source['source']
    if path not in pts_cache:
        with np.load(path, allow_pickle=False) as z:
            pts_cache[path] = list(map(int, z['ts']))
    word = source['word']
    specs = {c['id']: c for c in contract['cards'][word]}
    rank_cards = {c['card']: c for c in ranked['conditions'][condition]['cards']}
    cards = []
    for c in row['conditions'][condition]:
        spec = specs[c['card']]
        rc = rank_cards[c['card']]
        actual = None if rc['chosen'] is None else rc['chosen']['span']
        if actual != c['span']:
            raise ValueError('saved selector/inspector span mismatch')
        inspections = []
        for e in c['evidence']:
            shape = e['task'] == 'shape'
            sequence = e['state'] == 'sequence_required'
            key = f"shape:{e['feature']}" if shape else next(k for k in spec['descriptors'] if k.startswith('sequence:')) if sequence else f"direction:{e['axis']}"
            item = dict(descriptor=key, source_sha256=source['sha256'], observer=copy.deepcopy(contract['observer']),
                inspection_span=e.get('span'), state=e['state'])
            if shape:
                item.update(feature=e['feature'], expected=e['expected'], probability=e['probability'],
                    association_qualified=e['qualified'], observed_count=e['observed_count'],
                    observation_eligible=e['observed_count'] / (e['span'][1] - e['span'][0] + 1) >= .8)
            elif not sequence:
                item.update(axis=e['axis'], desired=e['desired'], probabilities=e['probabilities'],
                    observation_eligible=e['state'] != 'observation_unknown',
                    eligibility_source='Saved original .8 geometry availability decision; not recomputed by adapter.')
            else:
                item['probabilities'] = None
            inspections.append(item)
        cards.append(dict(id=c['card'], selected_span=copy.deepcopy(c['span']), saved_status=c['status'],
            selector_profile=f"{condition}:{row['seed']}:{spec['family']}", inspections=inspections,
            completion=c['completion']))
    return dict(schema='g3_explicit_evidence_v1', word=word, condition=condition, seed=row['seed'],
        source=dict(id=row['clip_id'], scenario=row['kind'], path=path, sha256=source['sha256'],
            artifact_kind='observation_npz', frame_indexing='decoded_observation_0_based', pts_us=pts_cache[path]),
        semantic_result=copy.deepcopy(contract['semantic_result']), selection_result=copy.deepcopy(contract['selection_result']),
        cards=cards, research_only=True, completion=None, whole_sign_correctness=None)
