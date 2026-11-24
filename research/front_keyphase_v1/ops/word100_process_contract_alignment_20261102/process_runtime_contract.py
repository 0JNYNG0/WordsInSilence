"""One connected-process target for global and proposed local observations.

The proposal is unchanged. Local process probability is NOT net motion and is
NOT an exact boundary certificate. RGB context support is reported separately.
No labels, source edits, person IDs or target outcomes enter qualification.
"""
from pathlib import Path
import sys
import numpy as np

AB = Path(__file__).resolve().parent.parent / 'word100_action_completion_supervision_20261102'
sys.path.insert(0, str(AB))
from process_qualification import propose_motion_intervals, HIGH, LOW


def qualify(raw_probability, point_probabilities, image_available, local_probabilities):
    p = float(raw_probability)
    assert np.isfinite(p) and 0 <= p <= 1
    spec = propose_motion_intervals(point_probabilities, image_available)
    scores = np.asarray(local_probabilities, dtype=float)
    assert scores.shape == (len(spec['candidates']),)
    assert np.isfinite(scores).all() and ((0 <= scores) & (scores <= 1)).all()
    candidates = [dict(c, connected_process_probability=float(s))
                  for c, s in zip(spec['candidates'], scores)]
    chains = [c for c in candidates if c['connected_process_probability'] >= HIGH]
    available = np.asarray(image_available, dtype=bool)
    display = 1 if p >= HIGH and chains else None
    if display == 1:
        status = 'OBSERVED_PROCESS_SUPPORT'
    elif not available.any():
        status = 'INSUFFICIENT_OBSERVATION'
    elif p <= LOW and chains:
        status = 'PROCESS_AND_LOCAL_EVIDENCE_DISAGREE'
    elif p <= LOW:
        status = 'PROCESS_NOT_CONFIRMED'
    else:
        status = 'PROCESS_EVIDENCE_UNCONFIRMED'
    return dict(status=status, qualified_display=display, raw_probability=p,
                raw_display=1 if p >= HIGH else 0 if p <= LOW else None,
                witnessed_chain=chains[0] if display else None,
                interval_candidates=candidates,
                candidate_count=len(candidates),
                uncapped_candidates=spec['uncapped_candidates'],
                candidate_cap_reached=spec['capped'],
                unavailable_indices=np.flatnonzero(~available).tolist(),
                source_seam_oracle_diagnostic=False,
                observation_only=True, observed_process_absence_certified=False,
                whole_word_correctness=None, learner_error=None,
                required_repetitions=None, complete_action_verified=False,
                exact_event_boundaries=False, selected_span_truth=None,
                rgb_context_may_extend_beyond_nominal_span=True,
                raw_low_is_not_word_error=True,
                local_target='Directly labelled connected process, not net motion')


def known_seam_diagnostic(runtime_result, seams):
    """Filter same scores for a privileged diagnostic; never runtime evidence."""
    import copy
    out = copy.deepcopy(runtime_result)
    for c in out['interval_candidates']:
        c['crosses_known_source_seam'] = any(c['start'] < s <= c['end'] for s in seams)
    valid = [c for c in out['interval_candidates']
             if c['connected_process_probability'] >= HIGH and not c['crosses_known_source_seam']]
    out['qualified_display'] = 1 if out['raw_probability'] >= HIGH and valid else None
    out['witnessed_chain'] = valid[0] if out['qualified_display'] else None
    if out['qualified_display'] == 1:
        out['status'] = 'OBSERVED_PROCESS_SUPPORT'
    elif out['status'] == 'INSUFFICIENT_OBSERVATION':
        pass
    elif out['raw_probability'] <= LOW and valid:
        out['status'] = 'PROCESS_AND_LOCAL_EVIDENCE_DISAGREE'
    elif out['raw_probability'] <= LOW:
        out['status'] = 'PROCESS_NOT_CONFIRMED'
    else:
        out['status'] = 'PROCESS_EVIDENCE_UNCONFIRMED'
    out['source_seam_oracle_diagnostic'] = True
    out['same_runtime_candidates_and_scores_reused'] = True
    out['source_seam_indices'] = sorted(set(map(int, seams)))
    return out
