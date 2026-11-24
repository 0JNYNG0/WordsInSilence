"""Preserve relation support; neither exact event localization nor a classifier.

No word/person identity, source seam, truth or score optimization is accepted.
An arrival relation can persist while an action is still unfolding. Preserve
that whole run as endpoint support, without calling its first/last frame the
true completion time. A future reviewed selector must still judge the process.
"""
import numpy as np

HIGH = .8


def _runs(indices):
    result = []
    for i in indices:
        if not result or i != result[-1][-1] + 1:
            result.append([])
        result[-1].append(int(i))
    return result


def relation_support(point_probabilities, image_available):
    p = np.asarray(point_probabilities, dtype=float)
    available = np.asarray(image_available, dtype=bool)
    if available.ndim != 1 or p.shape != (len(available), 2):
        raise ValueError('Expected one availability value and two relation probabilities per real row')
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Probabilities must be finite and between zero and one')
    groups = _runs(np.flatnonzero(available))
    contexts = []
    candidates = []
    for group_id, group in enumerate(groups):
        starts = _runs([i for i in group if p[i, 0] >= HIGH])
        arrivals = _runs([i for i in group if p[i, 1] >= HIGH])
        contexts.append(dict(group_id=group_id, real_indices=group,
                             setup_relation_runs=starts, arrival_relation_runs=arrivals))
        for s_id, start in enumerate(starts):
            for t_id, arrival in enumerate(arrivals):
                # Do not require dwell. One observed frame can be valid support.
                # Both relations may coexist. The process itself still needs
                # at least two different observation times, not a pause.
                possible_end = [i for i in arrival if i > start[0]]
                if not possible_end:
                    continue
                possible_start = [i for i in start if i < possible_end[-1]]
                candidates.append(dict(
                    group_id=group_id, setup_run_id=s_id, arrival_run_id=t_id,
                    start_support_indices=possible_start,
                    end_support_indices=possible_end,
                    context_envelope_indices=list(range(possible_start[0],possible_end[-1]+1)),
                    former_first_only_indices=list(range(possible_start[0],possible_end[0]+1)),
                    selected_start=None, selected_end=None, process_probability=None,
                    directly_observed_process=None, inferred_continuity=None,
                    whole_word_correctness=None, learner_error=None))
    candidates.sort(key=lambda c:(c['end_support_indices'][0],c['start_support_indices'][0]))
    return dict(status='RELATION_SUPPORT_ONLY_NOT_PROCESS_DECISION',
        contexts=contexts, candidates=candidates,
        unavailable_indices=np.flatnonzero(~available).tolist(),
        short_evidence_dropped=0, candidates_silently_truncated=False,
        source_frame_or_edit_metadata_used=False,
        threshold=HIGH, scored_candidates=0, exact_event_boundaries=False,
        runtime_adoption=False,
        limitation='An envelope can include holds or multiple actions; it is context support, not a verified event or a reason to accept an incomplete action.')
