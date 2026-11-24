"""Independent scalar rule audit and recount of saved outputs; no candidate import.

Same operator, separate implementation. This verifies calculation/contracts, not
the physical correctness of a heuristic or independent human ground truth.
"""
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np

U = Path(__file__).resolve().parent
def read(p): return json.loads(Path(p).read_text('utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name, obj):
    with (U / name).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

scope = read(U / 'SCOPE.json')
pred = read(U / 'PREDICTIONS.json')
params = scope['parameters']
assert sha(U / 'SCOPE.json') == pred['scope_sha256']
assert sha(U / 'visibility.py') == scope['code_sha256']
assert sha(U / 'VISUAL_LABELS.json') == pred['labels_sha256']
assert read(U / 'VISUAL_LABELS.json')['predictions_seen'] is False
assert read(U / 'SYNTHETIC_TESTS.json')['status'] == 'PASS'
protected = read(U.parent / 'word100_overlap_sequence_20261017/RECOUNT.json')['protected_pins']
for path, digest in {**scope['pins'], **protected}.items():
    assert sha(path) == digest, path

# Manual post-run safety inspection recorded after viewing all twelve sheets.
selection = read(U / 'ASSERTION_SELECTION.json')
assert len(selection['runs']) == 35 and len(selection['items']) == 94
overlap_ids = {'A018', 'A019', 'A020', 'A030', 'A077'}
assert {x['id'] for x in selection['items'] if x['state'] == 'overlap_likely'} == overlap_ids
review = {
    'at': datetime.now().astimezone().isoformat(timespec='seconds'),
    'operator': 'Codex root visual inspection; no second human/agent/expert',
    'design': 'Post-output safety inspection; not blinded validation. First/middle/last of each confident core run.',
    'selection_sha256': sha(U / 'ASSERTION_SELECTION.json'),
    'reviewed_ids': [x['id'] for x in selection['items']],
    'runs': dict(Counter(x['state'] for x in selection['runs'])),
    'sampled_frames': dict(Counter(x['state'] for x in selection['items'])),
    'page_sha256': {p: sha(U / p) for p in selection['pages']},
    'observations': [
        'All 89 boundary samples show lowered/outside or partly cut hands at the sensor bottom boundary.',
        'No clearly entirely interior active hand labelled boundary/outside was found among sampled frames.',
        'Five overlap samples from three runs show contact/overlap of hands consistent with a suspicion label.',
        'No claim that all frames in each run, all data, or expert anatomical ground truth were reviewed.'
    ],
    'disagreements_observed_in_this_sample': []
}
save('ASSERTION_REVIEW.json', review)

counts = {g: {'whole': Counter(), 'core': Counter()} for g in ('known36', 'additional12')}
checked = 0
scalar_mismatches = []
lookup = {r['clip_id']: r for r in pred['records']}
assert set(lookup) == {r['clip_id'] for r in scope['records']}

for record in scope['records']:
    observed = np.load(record['paths']['candidate'])
    raw = np.load(record['paths']['raw'])
    assignment = np.load(record['paths']['assignment'])['sides']
    rows = lookup[record['clip_id']]['frames']
    assert len(rows) == len(raw['timestamp_us'])
    assert np.array_equal(observed['timestamp_us'], raw['timestamp_us'])
    assert np.array_equal(observed['frame_id'], raw['frame_id'])
    tracks = [None, None]
    old_t, old_size = None, None
    for i, row in enumerate(rows):
        t = int(raw['timestamp_us'][i]) / 1e6
        w, h = int(raw['width'][i]), int(raw['height'][i])
        assert row['frame'] == i and row['timestamp_us'] == int(raw['timestamp_us'][i])
        core = any(a <= t <= b for a, b in record['intervals'])
        assert row['core'] == core
        if (old_t is not None and t <= old_t) or old_size not in (None, (w, h)):
            tracks = [None, None]
        old_t, old_size = t, (w, h)
        tracks = [z if z is not None and t-z[0] <= params['history_seconds'] else None for z in tracks]
        points = observed['xy_px'][i].astype(float).tolist()
        mask = observed['usable'][i].tolist()
        confidence = raw['pose_conf'][i].astype(float).tolist()
        def finite(p): return all(math.isfinite(z) for z in p)
        def interior(p, margin=0):
            return finite(p) and margin <= p[0] <= w-margin and margin <= p[1] <= h-margin
        def norm(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
        def pose_ok(j, actual=True):
            return finite(points[j]) and math.isfinite(confidence[j]) and confidence[j] >= params['pose_confidence'] and (not actual or (mask[j] and interior(points[j])))
        palms = []
        for side in (0, 1):
            ix = [33+21*side+k for k in (0, 5, 9, 13, 17)]
            palms.append([sum(points[j][a] for j in ix)/5 for a in (0, 1)] if all(mask[j] and finite(points[j]) for j in ix) else None)
        shoulder = pose_ok(11) and pose_ok(12)
        scale = norm(points[11], points[12]) if shoulder else 0
        assigned = assignment[i, :int(raw['raw_count'][i])].tolist()
        for side in (0, 1):
            s, e, q = 11+side, 13+side, 15+side
            state = 'unknown'
            if palms[side] is not None:
                state = 'observed'
            elif side in assigned:
                state = 'partial_observation'
            elif any(x < 0 for x in assigned):
                state = 'assignment_uncertain'
            elif shoulder and scale > 1e-6 and pose_ok(s) and pose_ok(e) and pose_ok(q, False):
                lengths = [norm(points[s], points[e])/scale, norm(points[e], points[q])/scale]
                if all(params['arm_segment_min_shoulders'] <= z <= params['arm_segment_max_shoulders'] for z in lengths):
                    x, y = points[q]
                    distances = [x/scale, (w-x)/scale, y/scale, (h-y)/scale]
                    edge = min(range(4), key=distances.__getitem__)
                    outward = [points[e][0]-x, x-points[e][0], points[e][1]-y, y-points[e][1]][edge]/scale
                    if distances[edge] <= params['boundary_shoulders'] and outward > params['boundary_shoulders']:
                        state = 'boundary_outside_likely'
                    elif interior(points[q], params['boundary_shoulders']*scale):
                        state = 'inside_missing'
                        if palms[1-side] is not None and tracks[side] is not None:
                            if norm(points[q], palms[1-side])/scale <= params['overlap_shoulders'] and norm(points[q], tracks[side][1])/scale <= params['history_displacement_shoulders']:
                                state = 'overlap_likely'
            saved = row['hands'][side]
            assert saved['side'] == side and saved['coordinate_repair'] is False
            if saved['state'] != state:
                scalar_mismatches.append([record['clip_id'], i, side, saved['state'], state])
            checked += 1
            counts[record['group']]['whole'][state] += 1
            if core: counts[record['group']]['core'][state] += 1
        for side in (0, 1):
            if palms[side] is not None: tracks[side] = (t, palms[side])

assert not scalar_mismatches, scalar_mismatches[:5]
assert counts == pred['counts']
tables = defaultdict(lambda: defaultdict(Counter))
groups = defaultdict(Counter)
manual = read(U / 'VISUAL_LABELS.json')['labels']
assert len(manual) == len(pred['visual_comparisons']) == 87
for expected, stored in zip(manual, pred['visual_comparisons']):
    assert all(stored[k] == v for k, v in expected.items())
    state = lookup[expected['clip_id']]['frames'][expected['frame']]['hands'][expected['side']]
    assert state == stored['prediction']
    origin, category = expected['origin'], expected['visual_category']
    tables[origin][category][state['state']] += 1
    if category == 'observed_control':
        groups[origin]['observed_controls'] += 1
        assert state['state'] == 'observed'
        continue
    groups[origin]['missing_reviewed'] += 1
    location = ('outside' if state['state'] == 'boundary_outside_likely' else 'inside' if state['state'] in ('inside_missing', 'overlap_likely') else 'undetermined')
    target = 'outside' if category == 'boundary_or_outside' else 'inside'
    # Check exact vocabulary rather than silently accepting a mislabeled category.
    assert category in ('boundary_or_outside', 'inside_overlap', 'inside_other'), category
    groups[origin]['undetermined' if location == 'undetermined' else 'broad_agree' if location == target else 'broad_disagree'] += 1

# Verify each post-output assertion sample really belongs to the selected run.
runmap = {r['run_id']: r for r in selection['runs']}
for item in selection['items']:
    run = runmap[item['run_id']]
    assert item['frame'] in run['frames'] and item['clip_id'] == run['clip_id'] and item['side'] == run['side']
    row = lookup[item['clip_id']]['frames'][item['frame']]
    assert row['core'] and row['hands'][item['side']]['state'] == item['state']
for path, digest in {**scope['pins'], **protected}.items(): assert sha(path) == digest
result = {
    'status': 'PASS', 'operator': 'Codex root, separate scalar reference code; not a second reviewer',
    'at': datetime.now().astimezone().isoformat(timespec='seconds'),
    'source_pins_checked': len(scope['pins']), 'protected_pins_checked': len(protected),
    'protected_pins': protected, 'scalar_state_checks': checked, 'mismatches': scalar_mismatches,
    'counts': counts, 'visual_confusion': tables, 'broad_location_comparison': groups,
    'observed_controls_unchanged': 16, 'post_output_safety_samples': 94,
    'limits': 'Arithmetic/contracts verified. Physical validity limited by selected visual samples. Not recognition accuracy.',
    'new_classifier_inference': 0, 'new_detection': 0, 'new_training': 0, 'user_recording_reads': 0,
    'evidence_sha256': {n: sha(U/n) for n in ['SCOPE.json', 'visibility.py', 'VISUAL_LABELS.json', 'PREDICTIONS.json', 'SYNTHETIC_TESTS.json', 'ASSERTION_REVIEW.json']}
}
save('INDEPENDENT_CHECK.json', result)
print(json.dumps({k: result[k] for k in ['status', 'scalar_state_checks', 'source_pins_checked', 'protected_pins_checked', 'broad_location_comparison']}, ensure_ascii=False, indent=2))
