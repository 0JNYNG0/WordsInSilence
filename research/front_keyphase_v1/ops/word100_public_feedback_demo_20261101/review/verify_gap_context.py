"""Append-only independent verification of three saved native-point gap contexts.

No producer import, inference, video decoding, label change, or score recomputation.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import traceback

import numpy as np

REVIEW = Path(__file__).resolve().parent
UNIT = REVIEW.parent
OPS = UNIT.parent
PINS, CHECKS, FAILURES, FACTS = {}, Counter(), [], {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(group, value, message):
    CHECKS[group] += 1
    if not bool(value):
        FAILURES.append({'group': group, 'message': message})


def eq(group, actual, expected, message):
    check(group, actual == expected, message)


def pin(path):
    path = str(Path(path).resolve())
    value = digest(path)
    if path in PINS:
        eq('preservation', value, PINS[path], path)
    else:
        PINS[path] = value
    return value


def read(path):
    pin(path)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def array(path, keys):
    pin(path)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in keys}


def main():
    # Existing verification and outputs are protected, not rewritten or rerun.
    eq('prior_preserved', pin(REVIEW / 'VERIFY_DEMO.json'),
       '5bec27ffd29a62db14cfc0f6f89cd315b50d3cef92a95bf7b35c15907f874353', 'original completed verification')
    for name in ['verify_demo.py', 'PLAN_REVIEW.json']:
        pin(REVIEW / name)
    for name in ['DENSE_MANIFEST.json', 'DEMO_DATA.json', 'EXECUTION_LEDGER.json',
                 'TRAINING_DECISION.json', 'FROZEN.json', 'GAP_CONTEXT_FIRST_FAILURE.txt', 'gap_context.py']:
        pin(UNIT / name)
    scope = read(UNIT / 'GAP_CONTEXT_SCOPE.json')
    result = read(UNIT / 'DEVELOPMENT_GAP_CONTEXT.json')
    audit = read(UNIT / 'AUDIT.json')
    dataset = read(OPS / 'word100_semantic_feature_cycle_20261101/DATASET.json')
    selection = {}
    for seed in [42, 43, 44]:
        rows = [r for r in audit['records'] if r['seed'] == seed and r['role'] == 'development'
                and r['stratum'] == 'geometry_gate_failed']
        selection[seed] = rows
        eq('selection', len(rows), 11, '11 known failed-gate cells per seed')
    fixed = sorted({(r['clip_id'], r['frame_id']) for r in selection[42]})
    eq('selection', len(fixed), 3, 'three predetermined center scenes')
    for seed in [43, 44]:
        eq('selection', {(r['clip_id'], r['frame_id'], r['target']) for r in selection[seed]},
           {(r['clip_id'], r['frame_id'], r['target']) for r in selection[42]}, 'same scenes/targets across seeds')
    for name, rows in [('scope', scope['rows']), ('result', result['rows'])]:
        eq('selection', [(r['clip_id'], r['frame_id']) for r in rows], fixed, name + ' exact coverage/order')
    facts = []
    for source, saved in zip(scope['rows'], result['rows']):
        cid, frame = source['clip_id'], source['frame_id']
        eq('lineage', {key: saved[key] for key in source}, source, 'source pointers copied exactly')
        expected_record = OPS / f'word100_body_boundary_20261017/data/records/{cid}.json'
        eq('lineage', Path(source['record']).resolve(), expected_record.resolve(), 'official public record path')
        record = read(source['record'])
        eq('lineage', pin(source['record']), source['record_sha256'], 'record digest')
        eq('lineage', record['clip_id'], cid, 'record identity')
        eq('lineage', record['views']['native']['observation'], source['observation'], 'record native view pointer')
        eq('lineage', pin(source['observation']), source['observation_sha256'], 'observation digest')
        a = array(source['observation'], ['xy_px', 'observed', 'frame_id', 'timestamp_us'])
        n = len(a['frame_id'])
        eq('source_schema', a['xy_px'].shape, (n, 75, 2), '75 native points')
        eq('source_schema', a['observed'].shape, (n, 75), 'native observation mask')
        eq('source_schema', str(a['observed'].dtype), 'bool', 'boolean observation mask')
        check('source_time', np.array_equal(np.diff(a['frame_id']), np.ones(n - 1)), 'actual native IDs consecutive; no gap joining')
        check('source_time', (np.diff(a['timestamp_us']) > 0).all(), 'PTS increasing')
        ix = np.flatnonzero(a['frame_id'] == frame)
        eq('source_time', len(ix), 1, 'unique selected frame')
        i = int(ix[0])
        eq('source_time', saved['seconds'], int(a['timestamp_us'][i]) / 1e6, 'exact selected PTS')
        direct_rows = [r for r in dataset['rows'] if r['clip_id'] == cid and r['center_frame'] == frame]
        eq('selection', len(direct_rows), 1, 'exact direct label center')
        direct = direct_rows[0]
        eq('selection', direct['role'], 'development', 'original role')
        eq('selection', direct['person'], 4, 'recorded person ID')
        eq('source_time', direct['center_timestamp_us'], int(a['timestamp_us'][i]), 'label/native PTS match')
        b = array(direct['input'], ['frame_id', 'timestamp_us', 'mask'])
        eq('lineage', pin(direct['input']), direct['input_sha256'], 'original saved input pin')
        jx = np.flatnonzero(b['frame_id'] == frame)
        eq('source_time', len(jx), 1, 'unique input center')
        j = int(jx[0]); m = b['mask'][j]
        eq('source_time', int(b['timestamp_us'][j]), int(a['timestamp_us'][i]), 'input/native PTS match')
        left = bool(m[0] and m[1] and np.count_nonzero(m[11:25]) >= 10)
        right = bool(m[28] and m[29] and np.count_nonzero(m[39:53]) >= 10)
        both = bool(m[8] and m[9] and m[36] and m[37])
        expected_gates = [right, right, left, right, left, both, both]
        expected_targets = [t for k, t in enumerate(dataset['selected_targets'])
                            if direct['target_mask'][k] and not expected_gates[k]]
        audited_targets = sorted(r['target'] for r in selection[42] if r['clip_id'] == cid and r['frame_id'] == frame)
        eq('selection', audited_targets, sorted(expected_targets), 'all and only known failed-gate features')
        sides = {}
        for side, indexes in [('L', np.arange(33, 54)), ('R', np.arange(54, 75))]:
            finite = np.isfinite(a['xy_px'][:, indexes]).all(axis=2)
            counts = np.count_nonzero(a['observed'][:, indexes] & finite, axis=1)
            eq('point_count', saved['sides'][side]['observed_points'], int(counts[i]), cid + '/' + side)
            missing_indices = np.flatnonzero(counts == 0)
            runs = np.split(missing_indices, np.flatnonzero(np.diff(missing_indices) > 1) + 1)
            run = next((run for run in runs if i in run), None)
            if run is None:
                expected_length, expected_ids = 0, None
                before, after = None, None
            else:
                lo, hi = int(run[0]), int(run[-1])
                expected_length = len(run)
                expected_ids = [int(a['frame_id'][lo]), int(a['frame_id'][hi])]
                check('absence_run', all(int(counts[k]) == 0 for k in run), 'every point-count zero inside native run')
                check('absence_run', lo == 0 or counts[lo - 1] > 0, 'maximal run before boundary')
                check('absence_run', hi == n - 1 or counts[hi + 1] > 0, 'maximal run after boundary')
                before = {'frame_id': int(a['frame_id'][lo - 1]), 'points': int(counts[lo - 1])} if lo else None
                after = {'frame_id': int(a['frame_id'][hi + 1]), 'points': int(counts[hi + 1])} if hi + 1 < n else None
            eq('absence_run', saved['sides'][side]['zero_point_run_frames'], expected_length, 'contiguous native run count')
            eq('absence_run', saved['sides'][side]['zero_point_run_ids'], expected_ids, 'contiguous native run IDs')
            sides[side] = {'points_at_center': int(counts[i]), 'zero_point_run_frames': expected_length,
                           'zero_point_run_ids': expected_ids, 'previous_observed_boundary': before,
                           'next_observed_boundary': after}
        eq('point_count', sorted(x['points_at_center'] for x in sides.values()), [0, 21], 'one zero-point side and one observed side')
        facts.append({'clip_id': cid, 'frame_id': frame, 'timestamp_us': int(a['timestamp_us'][i]),
                      'failed_gate_known_targets': audited_targets, 'sides': sides})
    FACTS.update(unique_scenes=3, repeated_seeds=3, known_failed_gate_cells_per_seed=11,
                 repeated_seed_cells=33, newly_scored_cells=0, newly_labeled_cells=0, rows=facts)


if __name__ == '__main__':
    destination = REVIEW / 'VERIFY_GAP_CONTEXT.json'
    if destination.exists():
        raise SystemExit('Prior verification exists; preserve it.')
    try:
        main()
    except BaseException:
        FAILURES.append({'group': 'verifier_exception', 'message': traceback.format_exc()})
    for path, expected in list(PINS.items()):
        eq('preservation_after', digest(path), expected, path)
    report = dict(status='PASS' if not FAILURES else 'FAIL', at=datetime.now(timezone.utc).isoformat(),
                  verifier='Separate-context Codex saved-data verification, not human/KSL truth',
                  assertions=sum(CHECKS.values()), assertion_groups=dict(CHECKS), failures=FAILURES,
                  facts=FACTS, source_pins=PINS,
                  verifier_source={'path': str(Path(__file__).resolve()), 'sha256': digest(__file__)},
                  method='Direct native observed-and-finite point counts; zero-index contiguous run splitting independent of producer traversal; source paths/time and predetermined AUDIT selection checked.',
                  limitations=[
                      'Native assigned hand point absence does not prove that the hand was visually absent or that the raw detector had no unassigned candidate.',
                      'These are three preselected development centers and 11 known feature cells per seed; repeated seed values do not supply independent scenes.',
                      'Observed gaps of one or two native frames do not support accuracy claims for sustained or severe missing input.',
                      'No model scores, labels or target masks were changed or recomputed; no new visibility or educational correctness truth was created.',
                      'The first failed 80-clip lookup is preserved as GAP_CONTEXT_FIRST_FAILURE.txt; authoritative 96-dataset source records supply the corrected lineage.'
                  ],
                  actions={'inference': 0, 'training': 0, 'producer_calls': 0, 'video_decodes': 0,
                           'score_recomputation': 0, 'previous_review_changes': 0, 'old_artifact_writes': 0})
    with destination.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status': report['status'], 'assertions': report['assertions'], 'failures': len(FAILURES),
                      'path': str(destination)}, ensure_ascii=False))
    if FAILURES:
        print(json.dumps(FAILURES, ensure_ascii=False, indent=2))
        raise SystemExit(1)
