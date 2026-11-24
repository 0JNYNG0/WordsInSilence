"""Attribute evidence consumer. No word IDs, trained parameters or coordinate edits.

Contradictions are annotations, never replacements of the saved predictions.
Equal negatives say nothing about untested alternative hand configurations.
"""
from itertools import combinations

GROUPS = {
    'R_shape': ('R_index_only', 'R_open_four', 'R_closed_four'),
    'L_shape': ('L_open_four', 'L_closed_four'),
    'height': ('R_higher', 'L_higher'),
}


def conflicts(values):
    return {g: [list(p) for p in combinations(names, 2)
                if values.get(p[0]) == 1 and values.get(p[1]) == 1]
            for g, names in GROUPS.items()
            if sum(values.get(t) == 1 for t in names) > 1}


def compare_attribute(q, r, target):
    """Keep the local difference even if another output contradicts it."""
    a, b = q.get(target), r.get(target)
    qconf, rconf = conflicts(q), conflicts(r)
    relevant = [g for g, names in GROUPS.items() if target in names]
    clash = any(g in qconf or g in rconf for g in relevant)
    if a is None or b is None:
        status = 'INSUFFICIENT_OBSERVATION'
    elif a != b:
        status = 'ATTRIBUTE_DIFFERENCE_CANDIDATE'
    elif a == 1:
        status = 'SAME_POSITIVE_ATTRIBUTE'
    else:
        status = 'SAME_NEGATIVE_ATTRIBUTE_ONLY'
    return dict(target=target, query=a, reference=b, status=status,
                raw_attribute_difference=None if a is None or b is None else a != b,
                query_conflicts=qconf, reference_conflicts=rconf,
                inconsistent_relevant_group=clash,
                supports_single_attribute_claim=a is not None and b is not None and not clash,
                whole_handshape_equal=None, same_action_stage=None,
                educational_correction=None)


def sequence_evidence(values, sample_ids):
    """Do not smooth, fill gaps, count a return as a repeat, or force guide order."""
    assert len(values) == len(sample_ids)
    return dict(samples=[dict(sample_id=i, value=v) for i, v in zip(sample_ids, values)],
                adjacent_changes=[dict(from_id=sample_ids[i-1], to_id=sample_ids[i],
                                       before=values[i-1], after=values[i])
                                  for i in range(1, len(values))
                                  if values[i-1] is not None and values[i] is not None
                                  and values[i-1] != values[i]],
                unknown_sample_ids=[i for i, v in zip(sample_ids, values) if v is None],
                obligatory_repeat_count=None, inferred_missing_states=[])
