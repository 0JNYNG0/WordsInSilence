"""Preserve relation support; neither exact event localization nor a classifier.

No word/person identity, source seam, truth or score optimization is accepted.
An arrival relation can persist while an action is still unfolding. Preserve
that whole run as endpoint support, without calling its first/last frame the
true completion time. A future reviewed selector must still judge the process.
"""
from bisect import bisect_right
import numbers
import numpy as np

HIGH = .8
DEFAULT_MAX_PAIRS = 256  # Resource budget, never a semantic/accuracy selector.


def _runs(indices):
    result = []
    for i in indices:
        if not result or i != result[-1][1] + 1:
            result.append([int(i), int(i)])
        else:
            result[-1][1] = int(i)
    return result


def inclusive_indices(bounds):
    """Lazy reconstruction of a compact inclusive range, including singleton."""
    if len(bounds) != 2 or bounds[0] > bounds[1]:
        raise ValueError('Expected inclusive [start, end]')
    return range(bounds[0], bounds[1] + 1)


def relation_support(point_probabilities, image_available, max_pairs=DEFAULT_MAX_PAIRS):
    p = np.asarray(point_probabilities, dtype=float)
    raw_available = np.asarray(image_available)
    if raw_available.ndim != 1 or p.shape != (len(raw_available), 2):
        raise ValueError('Expected one availability value and two relation probabilities per real row')
    if not np.isin(raw_available, [False, True]).all():
        raise ValueError('Availability must contain only boolean or numeric 0/1 values')
    available = raw_available.astype(bool)
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Probabilities must be finite and between zero and one')
    if isinstance(max_pairs, bool) or not isinstance(max_pairs, numbers.Integral) or max_pairs < 1:
        raise ValueError('max_pairs must be an explicit positive integer resource budget')
    groups = _runs(np.flatnonzero(available))
    contexts = []
    pair_count = 0
    for group_id, group in enumerate(groups):
        starts = _runs(i for i in inclusive_indices(group) if p[i, 0] >= HIGH)
        arrivals = _runs(i for i in inclusive_indices(group) if p[i, 1] >= HIGH)
        contexts.append(dict(group_id=group_id, real_range=group,
                             setup_relation_run_ranges=starts, arrival_relation_run_ranges=arrivals))
        ends = [a[1] for a in arrivals]
        pair_count += sum(len(ends) - bisect_right(ends, s[0]) for s in starts)
    exceeded = pair_count > max_pairs
    result = dict(status=('UNRESOLVED_BUDGET_EXCEEDED' if exceeded else 'RELATION_SUPPORT_ONLY_NOT_PROCESS_DECISION'),
        representation='compact_inclusive_ranges_v2', contexts=contexts, candidates=[],
        unavailable_ranges=_runs(np.flatnonzero(~available)),
        candidate_pair_count=pair_count, max_pairs=int(max_pairs), budget_exceeded=exceeded,
        candidate_generation_complete=not exceeded, candidates_silently_truncated=False,
        budget_is_semantic_selection=False, short_evidence_dropped=0,
        source_frame_or_edit_metadata_used=False,
        threshold=HIGH, scored_candidates=0, exact_event_boundaries=False,
        runtime_adoption=False,
        limitation='An envelope can include holds or multiple actions; it is context support, not a verified event or a reason to accept an incomplete action.',
        budget_policy='If all pairs do not fit the explicit resource budget, return no candidates and preserve all relation runs. Do not score a silently selected subset.')
    if exceeded:
        return result
    candidates = []
    for context in contexts:
        group_id = context['group_id']
        starts = context['setup_relation_run_ranges']
        arrivals = context['arrival_relation_run_ranges']
        ends = [a[1] for a in arrivals]
        for s_id, start in enumerate(starts):
            for t_id in range(bisect_right(ends, start[0]), len(arrivals)):
                arrival = arrivals[t_id]
                # Do not require dwell. One observed frame can be valid support.
                # Both relations may coexist. The process itself still needs
                # at least two different observation times, not a pause.
                possible_end = [max(arrival[0], start[0] + 1), arrival[1]]
                possible_start = [start[0], min(start[1], arrival[1] - 1)]
                candidates.append(dict(
                    group_id=group_id, setup_run_id=s_id, arrival_run_id=t_id,
                    start_support_range=possible_start,
                    end_support_range=possible_end,
                    context_envelope_range=[possible_start[0], possible_end[1]],
                    former_first_only_range=[possible_start[0], possible_end[0]],
                    selected_start=None, selected_end=None, process_probability=None,
                    directly_observed_process=None, inferred_continuity=None,
                    whole_word_correctness=None, learner_error=None))
    candidates.sort(key=lambda c:(c['end_support_range'][0],c['start_support_range'][0]))
    assert len(candidates) == pair_count
    result['candidates'] = candidates
    return result
