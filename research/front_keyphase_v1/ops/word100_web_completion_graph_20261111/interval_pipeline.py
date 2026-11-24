"""Common, descriptor-driven correspondence component; no word-name branches."""
import numpy as np
from occurrence_binding import select_occurrence,union_indices


def select_cards(descriptions,specifications,phase,role,hand,ts,config):
    cards=[]
    for spec in specifications:
        card=next(k for k in descriptions if k['id']==spec['id'])
        family=spec['families'][0]
        valid=hand[:,[0 if h=='L' else 1 for h in card['hands']]].all(1)
        groups=[]
        for group in card['groups']:
            selected=select_occurrence(phase[:,np.array(group)+1].sum(1),role[:,family],valid,ts,
                config['threshold'],config['min_evidence_ms'],bool(spec.get('parent')))
            groups.append(dict(group=group,**selected))
        spans=[g['span'] for g in groups if g['span']]
        span=[min(a for a,b in spans),max(b for a,b in spans)] if len(spans)==len(groups) else None
        cards.append(dict(id=card['id'],family=family,span=span,groups=groups,
            evidence_intervals=spans,observed_indices=union_indices(spans),required=spec['required'],
            parent=spec.get('parent'),semantic_completion=None))
    return cards


def order_observations(cards):
    main=[c for c in cards if c['required'] and not c['parent']]
    return [(a,b) for a,b in zip(main,main[1:]) if a['family']!=b['family']]
