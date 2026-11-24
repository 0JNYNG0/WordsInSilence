"""Independent synthetic comparator tests; never invokes bank/adapter/model/video.

Each run writes a new named JSON. The initial snapshot demonstrates counterexamples;
later runs must preserve that first evidence and pin the newly reviewed engine.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import traceback

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
UNIT = HERE.parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(module_path):
    spec = importlib.util.spec_from_file_location('_synthetic_engine_review', module_path)
    engine = importlib.util.module_from_spec(spec); spec.loader.exec_module(engine)
    source = json.loads((UNIT / 'PUBLIC_INPUTS.json').read_text(encoding='utf-8-sig'))
    names = source['feature_names']
    checks, counterexamples = [], []

    def verify(name, test):
        try:
            detail = test()
            passed = bool(detail.pop('pass')) if isinstance(detail, dict) else bool(detail)
            checks.append(dict(name=name, passed=passed, detail=detail if isinstance(detail, dict) else None))
            if not passed:
                counterexamples.append(checks[-1])
        except BaseException:
            checks.append(dict(name=name, passed=False, exception=traceback.format_exc()))
            counterexamples.append(checks[-1])

    def sample(n=20):
        x = np.zeros((n, 69), float)
        x[:, 11:25] = .2; x[:, 39:53] = .2
        x[:, [25, 53]] = 1
        p = np.tile([.9, .1, .1, .1, .9, .9, .1], (n, 1))
        return dict(x=x, mask=np.ones((n, 69), bool), probabilities=p,
                    frame_id=np.arange(n), timestamp_us=np.arange(n) * 33333,
                    feature_names=list(names))

    def row(n=20, cid='reference1', person=1):
        return dict(clip_id=cid, person=person, intervals=[[0, 10]], stages=[
            dict(id='main', canonical_stage_id='main', title='합성 주요 구간', frames=[0, n - 1],
                 description='합성 구간', source_url=None, boundary_source='automatic_transfer',
                 boundary_inferred=True)])

    def comparisons(q, r):
        one = engine.compare_one(q, r, row(len(r['x'])))
        two = copy.deepcopy(one); two['id'] = 'reference2'; two['person'] = 2
        return [one, two]

    def aggregate(cs):
        return engine.aggregate(cs, {key: .05 for key in engine.FAMILIES})

    def shape_status(cs, key='R_shape'):
        stages, selected = aggregate(cs)
        found = [f for f in stages[0]['features'] if f['name'] == engine.LABELS[key]]
        return found[0]['status'] if len(found) == 1 else 'OMITTED'

    def identity():
        a = sample(); original = copy.deepcopy(a)
        cs = comparisons(a, a)
        preserved = all(np.array_equal(a[k], original[k], equal_nan=True) for k in ['x', 'mask', 'probabilities', 'frame_id', 'timestamp_us'])
        path = np.asarray(cs[0]['path'])
        valid = path[0].tolist() == [0, 0] and path[-1].tolist() == [19, 19]
        valid &= set(path[:, 0]) == set(range(20)) and set(path[:, 1]) == set(range(20))
        distances = [f['distance'] for f in cs[0]['stages'][0]['families'].values()]
        return dict(pass_=preserved, **{'pass': preserved and valid and all(v == 0 for v in distances)}, distances=distances)

    verify('identity_all_frames_and_input_unchanged', identity)

    def missing_posteriors():
        q, r = sample(), sample(); q['probabilities'][:, :5] = np.nan
        cs = comparisons(q, r); value = cs[0]['stages'][0]['families']['R_shape']
        status = shape_status(cs)
        return {'pass': status == 'INSUFFICIENT_EVIDENCE', 'R_shape': value, 'status': status,
                'meaning': 'Angles cannot supply missing categorical handshape observations.'}

    verify('unknown_handshape_not_promoted_by_14_observed_angles', missing_posteriors)

    def unknown_reference():
        q, r = sample(), sample(); r['probabilities'][:, :5] = np.nan
        cs = comparisons(q, r); status = shape_status(cs)
        return {'pass': status == 'INSUFFICIENT_EVIDENCE', 'status': status,
                'meaning': 'Unknown reference handshape remains a visible unsupported opportunity.'}

    verify('unknown_reference_handshape_not_silently_omitted', unknown_reference)

    for side, columns in [('R', (0, 1)), ('R', (0, 3)), ('R', (1, 3)), ('L', (2, 4))]:
        def conflict(side=side, columns=columns):
            q, r = sample(), sample(); q['probabilities'][:, :5] = .1
            q['probabilities'][:, columns] = .5
            status = shape_status(comparisons(q, r), side + '_shape')
            return {'pass': status == 'INSUFFICIENT_EVIDENCE', 'status': status,
                    'probability_columns': list(columns), 'threshold': .5}
        verify('exclusive_attribute_conflict_at_threshold_' + side + str(columns), conflict)

    def split_masks():
        q, r = sample(), sample(); q['mask'][:, :28] = False
        cs = comparisons(q, r)
        return {'pass': shape_status(cs, 'R_shape') == 'SIMILAR_REFERENCE_RANGE'
                and shape_status(cs, 'L_shape') == 'INSUFFICIENT_EVIDENCE',
                'R': shape_status(cs, 'R_shape'), 'L': shape_status(cs, 'L_shape')}

    verify('left_missing_does_not_change_right_evidence_identity', split_masks)

    def denominator():
        q, r = sample(), sample(); q['mask'][:] = False
        before = comparisons(r, r)[0]['stages'][0]['families']
        after = comparisons(q, r)[0]['stages'][0]['families']
        return {'pass': all(after[k]['expected'] == before[k]['expected'] and after[k]['observed'] == 0
                            and after[k]['distance'] is None for k in engine.FAMILIES),
                'expected_before': {k: before[k]['expected'] for k in engine.FAMILIES},
                'expected_after': {k: after[k]['expected'] for k in engine.FAMILIES}}

    verify('query_missing_never_shrinks_reference_expected_denominator', denominator)

    def known_and_unknown_link():
        q, r = sample(20), sample(19); q['mask'][1] = False; q['probabilities'][1] = np.nan
        original = engine.path_dtw
        forced = np.array([[0, 0]] + [[i, i - 1] for i in range(1, 20)], dtype=np.int32)
        try:
            engine.path_dtw = lambda _: forced.copy()
            families = engine.compare_one(q, r, row(19))['stages'][0]['families']
        finally:
            engine.path_dtw = original
        v = families['arm_path']
        qcoverage = v.get('query_coverage')
        return {'pass': qcoverage is not None and qcoverage < 1 and v['coverage'] == 1,
                'family': v, 'meaning': 'One known and one unknown query linked to the same reference must retain query missingness.'}

    verify('many_to_one_known_link_does_not_hide_unknown_query_opportunity', known_and_unknown_link)

    def only_arm():
        q, r = sample(), sample()
        for a in (q, r):
            a['mask'][:] = False; a['mask'][:, [2, 3, 4, 5, 30, 31, 32, 33]] = True
            a['probabilities'][:] = np.nan
        stages, ids = aggregate(comparisons(q, r))
        return {'pass': stages[0]['status'] != 'SIMILAR_REFERENCE_RANGE', 'stage': stages[0],
                'meaning': 'Arm-only agreement cannot be presented as handshape and both-hand relation agreement.'}

    verify('only_arm_known_does_not_become_whole_stage_similarity', only_arm)

    def short_peak():
        q, r = sample(), sample(); q['x'][9, 41] += .7
        cs = comparisons(q, r)
        families = cs[0]['stages'][0]['families']; key = 'R_finger_angles' if 'R_finger_angles' in families else 'R_shape'
        family = families[key]
        peak = family.get('max_abs', family.get('peak_difference'))
        return {'pass': peak is not None and peak >= .699999,
                'family': family, 'meaning': 'A quantile may remain small, but a brief raw change and its source time must survive.'}

    verify('single_frame_finger_peak_evidence_preserved', short_peak)

    def invalid_mask_values():
        q, r = sample(), sample(); q['mask'][7] = False; q['x'][7] = np.nan
        costs = engine.matrix(engine.channels(q), engine.channels(r), np.ones(20, bool))
        return {'pass': bool(np.isfinite(costs).all()), 'nonfinite_cells': int((~np.isfinite(costs)).sum())}

    verify('masked_nonfinite_values_do_not_poison_DTW', invalid_mask_values)

    def same_whole_refs():
        cs = comparisons(sample(), sample())
        cs[0]['distance'] = .1; cs[1]['distance'] = .2
        third = copy.deepcopy(cs[1]); third['id'] = 'reference3'; third['person'] = 3; third['distance'] = .3
        # Third is locally best in one family, but cannot replace a selected
        # whole-reference identity just in that one family.
        for family in engine.FAMILIES:
            third['stages'][0]['families'][family]['distance'] = 0
            cs[0]['stages'][0]['families'][family]['distance'] = .02
            cs[1]['stages'][0]['families'][family]['distance'] = .03
        stages, selected = aggregate(cs + [third])
        return {'pass': selected == ['reference1', 'reference2'] and all(f['reference_ids'] == selected
                           for stage in stages for f in stage['features']), 'selected': selected}

    verify('one_fixed_pair_of_whole_references_not_per_family_patchwork', same_whole_refs)

    def exact_path_optimum():
        cost = np.array([[.2, .4, .9], [.6, .1, .4], [.8, .3, .2]])
        values = []
        def enumerate_paths(i, j, total):
            if i == 2 and j == 2:
                values.append(total); return
            for di, dj in [(1, 1), (1, 0), (0, 1)]:
                ii, jj = i + di, j + dj
                if ii < 3 and jj < 3:
                    enumerate_paths(ii, jj, total + cost[ii, jj] + (.03 if di != dj else 0))
        enumerate_paths(0, 0, cost[0, 0])
        path = engine.path_dtw(cost)
        observed = sum(cost[i, j] for i, j in path)
        observed += sum(.03 for d in np.diff(path, axis=0) if d[0] != d[1])
        return {'pass': abs(observed - min(values)) < 1e-12, 'expected_minimum': min(values), 'observed': observed}

    verify('DTW_matches_independent_small_path_enumeration', exact_path_optimum)

    def boundary_provenance():
        output = comparisons(sample(), sample())[0]['stages'][0]
        return {'pass': output.get('boundary_inferred') is True or output.get('boundary_source') == 'automatic_transfer',
                'output_keys': list(output), 'meaning': 'Reference boundary transferred to another person is inferred, not direct truth.'}

    verify('automatically_transferred_stage_boundary_provenance_retained', boundary_provenance)
    return checks, counterexamples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', default=str(UNIT / 'engine.py'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    module_path = Path(args.engine).resolve(); output = HERE / args.output
    if not module_path.is_relative_to(UNIT.resolve()) or output.resolve().parent != HERE or output.exists():
        raise SystemExit('Use current-unit code and a fresh review JSON output.')
    before = digest(module_path)
    checks, failures = main(module_path)
    report = {'status': 'PASS' if not failures else 'ISSUES_FOUND', 'checks': len(checks), 'failures': failures,
              'results': checks, 'engine': {'path': str(module_path), 'sha256': before},
              'engine_unchanged': digest(module_path) == before,
              'verifier_sha256': digest(__file__),
              'scope': 'Synthetic mathematical/source-preservation tests only. Independent expectations; comparator pure functions execute, no model/adapter/bank/public data comparison.',
              'static_adapter_note': 'Draft adapter had no actual-frame/time cap before detection; root/adapter owner notified. Needs final code review after repair.',
              'model_inference': 0, 'training': 0, 'public_path_runs': 0, 'camera_access': 0, 'user_data_access': 0}
    with output.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status': report['status'], 'checks': len(checks), 'failed_checks': len(failures),
                      'output': str(output)}, ensure_ascii=False))
