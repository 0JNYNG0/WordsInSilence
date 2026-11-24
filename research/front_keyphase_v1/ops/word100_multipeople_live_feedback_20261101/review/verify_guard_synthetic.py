"""Independent bounded guard counterexamples. Inject saved-style paths: no DP."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import traceback
import numpy as np

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


A = module('_independent_guard_masks', Path(__file__).with_name('verify_guard_saved.py'))
G = module('_independent_guard_subject', U / 'evidence_guard.py')
results = []


def check(ok, name):
    results.append(dict(name=name, passed=bool(ok)))
    if not ok:
        raise AssertionError(name)


def equal(actual, expected):
    if expected is None:
        return actual is None
    if isinstance(expected, float):
        return actual is not None and abs(actual - expected) <= 1e-12
    return actual == expected


def support_case(name, qm, rm, links, shape=False):
    frames = list(range(len(rm)))
    own = A.independent_support(qm, rm, frames, links, shape)
    got = G._support(qm, rm, links, [0, len(rm) - 1], shape)
    mapping = {'expected': 'reference_expected', 'observed': 'reference_observed'}
    keys = ['expected', 'observed', 'query_expected', 'query_observed', 'reference_coverage',
            'query_coverage', 'coverage', 'dimension_reference_expected', 'dimension_reference_observed',
            'dimension_query_expected', 'dimension_query_observed', 'dimension_coverage', 'query_indices']
    check(all(equal(got[mapping.get(k, k)], own[k]) for k in keys), name)
    return got


def make_arrays(names):
    x = np.zeros((16, 69), dtype=np.float64)
    mask = np.ones_like(x, bool)
    probability = np.tile([.05, .9, .05, .05, .9, .5, .5], (16, 1))
    return dict(x=x, mask=mask, probabilities=probability, frame_id=np.arange(16),
                timestamp_us=np.arange(16) * 33333, feature_names=names)


def base_result(labels, primary, rest=False):
    nonrequired = ['R_shape', 'R_angles', 'hand_relation'] if rest else []
    features = [dict(name=labels[f], status='NOT_REQUIRED_REFERENCE_REST' if f in nonrequired else G.SIMILAR,
                coverage=[1., 1.], distances=[0., 0.], peak_distances=[0., 0.],
                normal_range_limit=.1, original_expected=[6, 6], scoped_expected=[0, 0] if f in nonrequired else [6, 6])
                for f in A.FAMILIES]
    stage = dict(id='s', canonical_stage_id='synthetic:one', title='synthetic only',
                 status=G.SIMILAR, message='original retained', features=features, query_range_seconds=[0., 1.])
    references = [dict(id=rid, person=i, distance=0., path=primary.tolist(),
                      stages=[dict(id='s', canonical_stage_id='synthetic:one', families={'sentinel': {'value': 7}})])
                  for i, rid in enumerate(['ref1', 'ref2'], 1)]
    return dict(stages=[stage], references=references, selected_whole_references=['ref1', 'ref2'])


class FixedCache:
    def __init__(self, path):
        self.path = path.tolist()
        self.calls = 0

    def get(self, a, r, **kwargs):
        self.calls += 1
        return dict(key='synthetic-fixed-no-DP', path=self.path)


def main():
    names = A.read(U / 'PUBLIC_INPUTS.json')['feature_names']
    pins = {str(p): A.sha(p) for p in [U / 'evidence_guard.py', U / 'engine.py', U / 'GUARD_FROZEN.json']}
    support_case('many-to-one query cells retained', np.ones((3, 1), bool), np.ones((2, 1), bool), [[0, 0], [1, 0], [2, 1]])
    support_case('duplicate links do not inflate cells', np.ones((3, 1), bool), np.ones((2, 1), bool), [[0, 0], [1, 0], [1, 0], [2, 1]])
    q = np.zeros((20, 1), bool)
    q[:13] = True
    boundary = support_case('exact .65 support retained', q, np.ones((1, 1), bool), [[i, 0] for i in range(20)])
    check(G._eligible(boundary), 'existing .65 boundary inclusive')
    q[12] = False
    below = support_case('12 of20 below .65', q, np.ones((1, 1), bool), [[i, 0] for i in range(20)])
    check(not G._eligible(below), 'below boundary insufficient')
    empty = support_case('no reference evidence remains null', np.ones((2, 1), bool), np.zeros((1, 1), bool), [[0, 0], [1, 0]], True)
    check(empty['coverage'] is None and not G._eligible(empty), 'empty is not successful zero difference')
    lowdim = support_case('shape single missing attribute not hidden by mean',
        np.tile([True, True, False], (2, 1)), np.ones((2, 3), bool), [[0, 0], [1, 1]], True)
    check(lowdim['coverage'] == 0., 'per-shape-dimension minimum')
    support_case('query unknown retained while reference observed once', np.array([[1], [0], [0]], bool),
                 np.ones((1, 1), bool), [[0, 0], [1, 0], [2, 0]])

    base = G._base()
    a = make_arrays(names)
    r = make_arrays(names)
    row = dict(clip_id='ref1', person=1, stages=[dict(id='s', canonical_stage_id='synthetic:one', frames=[5, 10],
        confirmed_rest_sides=[], boundary_provenance='synthetic_test_not_truth')])
    # Direct mask reconstruction: raw finite support, short context and conflicts.
    ai = [i for i, n in enumerate(names) if n.startswith('R_') and '_angle' in n and 'elbow' not in n]
    a['mask'][5:11, ai] = False
    mine = A.independent_masks(a, names)
    check(all(np.array_equal(mine[f], base.channels(a)[f][1]) for f in A.FAMILIES), 'independent raw69 posterior mask reconstruction')
    a['probabilities'][3, [0, 1, 3]] = [.9, .9, .1]
    mine = A.independent_masks(a, names)
    check(not mine['R_shape'][3].any(), 'conflicting head positives remain unsupported')
    check(np.array_equal(mine['R_shape'], base.channels(a)['R_shape'][1]), 'conflict support equals frozen engine')
    a['probabilities'][3, [0, 1, 3]] = [.05, .9, .05]
    identity = np.stack([np.arange(16), np.arange(16)], 1)
    escaped = np.asarray([[i, i] for i in range(5)] + [[4, j] for j in range(5, 11)] +
                         [[i, i + 6] for i in range(5, 10)] + [[i, 15] for i in range(10, 16)])
    check(A.complete_path(escaped, 16, 16), 'synthetic escaped primary is a full unit monotone path')
    original = base_result(base.LABELS, escaped)
    before = copy.deepcopy(original)
    source_before = {k: v.copy() for k, v in a.items() if isinstance(v, np.ndarray)}
    rows = {rid: dict(row, clip_id=rid, person=i) for i, rid in enumerate(['ref1', 'ref2'], 1)}
    arrays = {rid: r for rid in rows}
    fixed = FixedCache(identity)
    guarded = G.guard_result(a, rows, arrays, original, cache=fixed)
    check(guarded['stages'][0]['status'] == G.INSUFFICIENT, 'escaped missing support downgrades original similar')
    check(fixed.calls == 2, 'injected auxiliary paths; no public or synthetic DP')
    check(original == before, 'base result deep copy unchanged')
    check(all(np.array_equal(a[k], v, equal_nan=True) for k, v in source_before.items()), 'query arrays unchanged')
    check(guarded['selected_whole_references'] == original['selected_whole_references'], 'selected two whole references unchanged')
    check(guarded['references'] == original['references'], 'original paths/raw family sentinels unchanged')
    check(all(f['distances'] == b['distances'] and f['coverage'] == b['coverage'] and
              f['peak_distances'] == b['peak_distances'] for f, b in zip(guarded['stages'][0]['features'], original['stages'][0]['features'])),
          'all old metric values preserved')
    for status in ['DIFFERENCE_CANDIDATE', G.INSUFFICIENT, 'NOT_REQUIRED_REFERENCE_REST']:
        test = copy.deepcopy(before)
        test['stages'][0]['status'] = status
        for feature in test['stages'][0]['features']:
            feature['status'] = status
        got = G.guard_result(a, rows, arrays, test, cache=FixedCache(identity))
        check(got['stages'][0]['status'] == status and all(f['status'] == status for f in got['stages'][0]['features']), 'non-similar status preserved:' + status)
    clean = make_arrays(names)
    good = G.guard_result(clean, rows, arrays, base_result(base.LABELS, identity), cache=FixedCache(identity))
    check(good['stages'][0]['status'] == G.SIMILAR, 'fully supported agreement is not automatically rejected')
    armcols = [names.index(s + '_' + site + '_' + d) for s in ['L', 'R'] for site in ['pose_wrist', 'elbow'] for d in ['x', 'y']]
    clean['mask'][:, armcols] = False
    unsupported_pose = G.guard_result(clean, rows, arrays, base_result(base.LABELS, identity), cache=FixedCache(identity))
    check(unsupported_pose['stages'][0]['status'] == G.INSUFFICIENT, 'unsupported auxiliary pose cannot authorize similar')
    check(any('AUXILIARY_POSE_CORRESPONDENCE_SUPPORT_BELOW_0_65' in reason
              for f in unsupported_pose['stages'][0]['features'] for reason in f['guard_support']['reasons']), 'pose reason explicit')
    restrows = {rid: dict(rr, stages=[dict(rr['stages'][0], confirmed_rest_sides=['R'])]) for rid, rr in rows.items()}
    scope = A.independent_scope(restrows['ref1'], A.independent_masks(r, names))
    actual_scope = base.reference_scope(restrows['ref1'], base.channels(r))[0]
    check(all(np.array_equal(scope[f], actual_scope[f]) for f in A.FAMILIES), 'reference-only rest scope independent reconstruction')
    resting = G.guard_result(a, restrows, arrays, base_result(base.LABELS, identity, True), cache=FixedCache(identity))
    check(resting['stages'][0]['status'] == G.SIMILAR, 'reference-rest excluded R shape does not create fake missing error')
    check([f['status'] for f in resting['stages'][0]['features'] if f['name'] in
           [base.LABELS['R_shape'], base.LABELS['R_angles'], base.LABELS['hand_relation']]] == ['NOT_REQUIRED_REFERENCE_REST'] * 3,
          'scope exclusion remains separate from similar evidence')
    key = G.arm_key(a, r)
    changed_hand = copy.deepcopy(a)
    changed_hand['x'][:, ai] += .123
    changed_hand['probabilities'][:] = .99
    check(G.arm_key(changed_hand, r) == key, 'hand/posterior cannot choose auxiliary cache path')
    changed_hand['mask'][0, armcols[0]] = False
    check(G.arm_key(changed_hand, r) != key, 'raw pose mask changes cache identity')
    check(all(A.sha(p) == h for p, h in pins.items()), 'source pins unchanged')
    return dict(status='PASS_INDEPENDENT_GUARD_COUNTEREXAMPLES', checks=len(results), failures=0,
        tests=results, source_pins=pins, verifier_sha256=A.sha(__file__),
        independent_arithmetic_verifier_sha256=A.sha(Path(__file__).with_name('verify_guard_saved.py')),
        execution=dict(public_data_alignment=0, synthetic_alignment=0, observer_forward=0, training=0,
                       synthetic_fixed_path_guard_calls=8),
        limits=['Synthetic counterexamples validate mechanics only; no educational accuracy or auxiliary-path truth claim.',
                'Public saved-result verification remains separate and pending.'])


if __name__ == '__main__':
    destination = U / 'review/GUARD_SYNTHETIC_VERIFICATION.json'
    if destination.exists():
        raise FileExistsError('Preserve first verification')
    try:
        report = main()
    except BaseException:
        report = dict(status='FAIL_PRESERVED', tests=results, traceback=traceback.format_exc(),
                      verifier_sha256=A.sha(__file__))
    with destination.open('x', encoding='utf8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: report.get(k) for k in ['status', 'checks', 'failures']}, ensure_ascii=False))
    raise SystemExit(0 if report['status'].startswith('PASS') else 1)
