"""Counterexamples for the proposed readout; no model accuracy claim."""
from pathlib import Path
import hashlib, json
import numpy as np
from progress_order import compare, support

U = Path(__file__).resolve().parent
ts = np.arange(12, dtype=np.int64) * 33333
visible = np.ones(12, bool)
def scores(indices):
    p = np.full(12, -1., np.float32)
    p[indices] = 1
    return p

checks = []
def check(name, condition):
    assert condition, name
    checks.append(name)

r = compare(ts, [1, 7], [6, 10], scores([2, 3, 4]), scores([7, 8, 9]), visible, visible)
check('progression_can_be_ordered_without_erasing_extent_overlap',
      r['original_candidate_extent_order']['state'] == 'ambiguous_overlap'
      and r['observed_body_progression_order']['state'] == 'forward')
check('ordered_progression_is_never_completion', r['completion'] is None and r['first']['completion'] is None)
r = compare(ts, [6, 10], [1, 7], scores([7, 8, 9]), scores([2, 3, 4]), visible, visible)
check('reversal_is_read_from_observation_not_corrected_to_expected_order', r['observed_body_progression_order']['state'] == 'reverse')
r = compare(ts, [1, 10], [1, 10], scores([2, 3, 4]), scores([2, 3, 4]), visible, visible)
check('same_observation_does_not_become_two_sequential_actions', r['observed_body_progression_order']['state'] == 'ambiguous_overlap')
r = compare(ts, [1, 10], [1, 10], scores([2, 3, 8, 9]), scores([5, 6]), visible, visible)
check('late_returning_progression_not_dropped_to_force_forward_order', r['observed_body_progression_order']['state'] == 'ambiguous_overlap')
check('single_point_is_insufficient_not_wrong_or_body_missing', support(ts, [1, 10], scores([3]), visible)['state'] == 'insufficient_progression_support')
check('high_scores_on_unobserved_body_never_count', support(ts, [1, 10], np.ones(12), np.zeros(12, bool))['support_indices'] == [])
check('unobserved_body_separate_from_weak_model_support', support(ts, [1, 10], np.ones(12), np.zeros(12, bool))['state'] == 'body_unknown')
check('no_candidate_is_unknown_not_absent_or_complete', support(ts, None, np.ones(12), visible)['completion'] is None)
a = compare(ts, [1, 7], [6, 10], scores([2, 3, 4]), scores([7, 8, 9]), visible, visible)
b = compare(ts * 3 + 17, [1, 7], [6, 10], scores([2, 3, 4]), scores([7, 8, 9]), visible, visible)
check('monotone_time_change_preserves_temporal_relation', a['observed_body_progression_order']['state'] == b['observed_body_progression_order']['state'])
for name, times, span, p, mask in [
    ('duplicate_pts_rejected', np.zeros(12, np.int64), [1, 10], scores([3, 4]), visible),
    ('noninteger_pts_rejected', ts.astype(float), [1, 10], scores([3, 4]), visible),
    ('out_of_bounds_span_rejected', ts, [1, 12], scores([3, 4]), visible),
    ('nonfinite_scores_rejected', ts, [1, 10], np.full(12, np.nan), visible),
    ('implicit_float_mask_rejected', ts, [1, 10], scores([3, 4]), np.ones(12)),
]:
    try:
        support(times, span, p, mask)
    except ValueError:
        check(name, True)
    else:
        raise AssertionError(name)
out = U / 'PROGRESS_ORDER_CONTRACT_CHECK_V2.json'
assert not out.exists()
pins = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), U/'progress_order.py']}
out.write_text(json.dumps(dict(status='PURE_READOUT_CONTRACT_VERIFIED_NOT_MODEL_VALIDATION', checks=checks,
    count=len(checks), no_inference=True, no_training=True, web_changed=False, pins=pins), indent=2)+'\n', encoding='utf-8')
print('Readout counterexamples passed:', len(checks))
