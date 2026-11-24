"""Independent AJ stored-value audit. No evaluator/model/encoder imports.

Run only after the parent confirms the single control inference has finished.
This checks arithmetic/provenance, not linguistic correctness or causal neural
feature use. Original/derived videos are SHA-read only, never decoded.
"""
from pathlib import Path
from collections import Counter
from bisect import bisect_left
import hashlib
import json
import math
import time
import traceback
import numpy as np

U = Path(__file__).resolve().parent
NORMAL = {'reencoded_normal', 'speed_0.8', 'speed_1.25', 'trim_initial_rest'}
KINDS = NORMAL | {'action_cut', 'main_blackout'}
WORDS = {'WORD0739', 'WORD1133', 'WORD1567', 'WORD2523'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def load_npz(path):
    with np.load(path, allow_pickle=False) as a:
        return {key: a[key].copy() for key in a.files}


def rebuild_masks(a, r, geometry_valid):
    """Reconstruct availability from exact cache values, without aj_data."""
    x, m = a['x'], a['mask']
    arm = x[:, 2:4] - x[:, 4:6]
    hand = x[:, 36:38] - x[:, 4:6]
    den = np.sum(arm * arm, axis=1)
    arm_valid = m[:, [2, 3, 4, 5, 36, 37]].all(1) & np.isfinite(den) & (den > .0025)
    along = np.sum(hand * arm, axis=1) / np.maximum(den, 1e-6)
    side = (hand[:, 0] * arm[:, 1] - hand[:, 1] * arm[:, 0]) / np.maximum(den, 1e-6)
    z = np.column_stack((x[:, :61], along, side, a['retained_probability'], r['probability']))
    valid = np.column_stack((m[:, :61] & np.isfinite(x[:, :61]), arm_valid, arm_valid,
                             np.isfinite(a['retained_probability']), np.isfinite(r['probability'])))
    image = a['image_available']
    valid &= image[:, None]
    valid &= np.isfinite(np.where(valid, z, 0).astype(np.float32))
    keep = np.asarray(geometry_valid)[73:] > .5
    rgb = image & np.isfinite(a['rgb']).all(1)
    observed = rgb | valid[:, keep].any(1)
    spans = []
    begin = None
    for i, value in enumerate(observed.tolist() + [False]):
        if value and begin is None:
            begin = i
        elif not value and begin is not None:
            spans.append([begin, i - 1])
            begin = None
    anchors = np.zeros(len(observed), dtype=bool)
    for lo, hi in spans:
        if hi > lo:
            anchors[lo:hi + 1] = True
    return image, rgb, observed, anchors, spans


def rgb_row_support(ts, rgb):
    """Exact positive-coefficient center interpolation, implemented separately."""
    n = len(ts)
    assert n >= 16
    starts = list(range(0, n - 15, 2))
    if starts[-1] != n - 16:
        starts.append(n - 16)
    centers = [int(ts[s]) + int(ts[s + 15]) for s in starts]
    supports = []
    for i, timestamp in enumerate(ts):
        if not rgb[i]:
            supports.append([])
            continue
        doubled = 2 * int(timestamp)
        if doubled <= centers[0]:
            active = [0]
        elif doubled >= centers[-1]:
            active = [len(centers) - 1]
        else:
            right = bisect_left(centers, doubled)
            active = [right] if centers[right] == doubled else [right - 1, right]
        supports.append(sorted({j for w in active for j in range(starts[w], starts[w] + 16)}))
    return supports, starts, centers


def expected_bag(cid, t, scope, diag, p, anchors, ids, ts, support):
    n = len(ids)
    lo, hi = [0, n - 1] if diag else scope['indices']
    ii = list(range(lo, hi + 1))
    eligible = [i for i in ii if anchors[i]]
    complete = all(bool(anchors[i]) for i in ii)
    out = dict(bag_id=cid + ':' + str(t) + (':whole_context' if diag else ''),
               kind='full_native_diagnostic_only' if diag else 'fixed_existing_prototype_scope',
               label=None, frames=[int(ids[lo]), int(ids[hi])], indices=[lo, hi],
               timestamp_us=[int(ts[lo]), int(ts[hi])], labelled_evidence_complete=complete,
               prediction_coverage=len(eligible) / len(ii),
               unavailable_native_frames=[int(ids[i]) for i in ii if not anchors[i]],
               supported=True, probability=None, anchor=None, status='INSUFFICIENT_OBSERVATION')
    if eligible:
        # Python max keeps the first maximum, like the evaluator's np.argmax.
        at = max(eligible, key=lambda i: float(p[i, t]))
        value = float(p[at, t])
        high = value >= .8
        status = ('MOVEMENT_EVIDENCE' if complete else 'MOVEMENT_EVIDENCE_WITH_GAPS') if high else (
            'INSUFFICIENT_OBSERVATION' if not complete else 'LOW_MOVEMENT_SCORE' if value <= .2 else 'UNCONFIRMED')
        left = right = at
        while left > 0 and anchors[left - 1]:
            left -= 1
        while right + 1 < n and anchors[right + 1]:
            right += 1
        temporal = list(range(max(left, at - 7), min(right, at + 7) + 1))
        rgb_rows = sorted({j for i in temporal for j in support[i]})
        out.update(probability=value, anchor=at, anchor_frame=int(ids[at]), anchor_timestamp_us=int(ts[at]),
                   raw_positive=value >= .5, high_confirmation=high, status=status,
                   absence_assessable=complete, whole_action_complete=None,
                   whole_action_error=None, educational_error=None,
                   context_support=dict(temporal_native_rows=temporal, rgb_native_rows=rgb_rows,
                                        rgb_native_timestamps_us=[int(ts[j]) for j in rgb_rows],
                                        structural_support_only=True, exact_onset_or_observed_event_boundary=False))
    out.update(target_index=t, guide_id=scope['guide_id'], occurrence_truth=None,
               whole_sign_correctness=None, complete_action=None, educational_error=None,
               correction_enabled=False)
    if diag:
        out['not_used_to_select_or_expand_primary_scope'] = True
    else:
        out['fixed_scope'] = scope
    return out


def main():
    if not (U / 'RESULT.json').is_file() or not (U / 'RUN_FINISHED.json').is_file():
        raise RuntimeError('No complete saved result; do not run inference or fabricate a result')
    if (U / 'SAVED_VERIFIED.json').exists():
        raise RuntimeError('Saved verification is already complete; do not repeat')
    checks = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    frozen = read(U / 'INPUT_FROZEN.json')
    plan = read(U / 'PLAN_FROZEN.json')
    execution = read(U / 'EXECUTION_FROZEN.json')
    review = read(U / 'CODE_REVIEW.json')
    result = read(U / 'RESULT.json')
    finished = read(U / 'RUN_FINISHED.json')
    check('finished result identity', finished['status'] == 'COMPLETE' and finished['result_sha256'] == sha(U / 'RESULT.json'))
    check('approved one inference', review['approved_for_one_inference'] is True)
    all_pins = {}
    for label, pins in [('input', frozen['pins']), ('plan', plan['pins']), ('execution', execution['pins']), ('review', review['code_pins'])]:
        for path, digest in pins.items():
            if path in all_pins:
                check(label + ' consistent pin ' + path, all_pins[path] == digest)
            all_pins[path] = digest
    for path, digest in all_pins.items():
        check('unchanged bytes ' + path, sha(path) == digest)
    check('frozen head exact', result['model_sha256'] == plan['model']['sha256'] == frozen['model']['sha256'])
    check('frozen thresholds', plan['thresholds'] == execution['thresholds'] == dict(raw=.5, high=.8, low=.2))
    check('execution scope', execution['inputs'] == execution['forwards'] == result['model_forwards'] == 24)
    check('no training encoder DTW', all(result[k] == 0 for k in ['training', 'new_extraction', 'new_DTW']))
    check('no copied old scores or modified results', result['native_AH_prediction_scores_copied'] is False and result['original_results_changed'] is False)
    check('goal not overclaimed', result['whole_goal_complete'] is False)
    check('supported five physical queries', frozen['supported_targets'] == list(range(5)))
    check('matching geometry value and mask flags', frozen['geometry_valid'][:73] == frozen['geometry_valid'][73:])
    rows = {r['case_id']: r for r in result['rows']}
    inputs = {r['case_id']: r for r in frozen['rows']}
    check('24 unique actual input cases', len(rows) == len(result['rows']) == len(inputs) == 24 and set(rows) == set(inputs))
    check('four source words six controls', set(inputs) == {w + '_' + k for w in WORDS for k in KINDS})
    check('30 scope denominator', sum(len(r['scopes']) for r in inputs.values()) == 30)
    check('twenty expectations ten unknown', Counter(s['expected_occurrence'] for r in inputs.values() for s in r['scopes']) == {1: 20, None: 10})
    verified_rows = []
    expected_by_case = {}
    prediction_paths = set()
    input_paths = set()
    for cid, source in inputs.items():
        saved = rows[cid]
        key = cid + ' '
        word, kind = source['word_id'], source['kind']
        check(key + 'same public DEV identity', source['person'] == 12 and source['role'] == 'development_controls_not_training' and saved['word_id'] == word and saved['kind'] == kind)
        a = load_npz(source['analysis']['path'])
        r = load_npz(source['relation']['path'])
        ids, ts = a['frame_id'], a['timestamp_us']
        n = len(ids)
        check(key + 'new derived input cache path', 'word100_local_event_learning_20261103' not in source['analysis']['path'] and 'actual_controls' in source['analysis']['path'])
        input_paths.add(source['analysis']['path'])
        check(key + 'retained native axes', ids.tolist() == source['native_frame_ids'] and ts.tolist() == source['timestamp_us'] and np.array_equal(ids, np.arange(n)) and np.all(np.diff(ts) > 0))
        check(key + 'relation native axes', np.array_equal(r['frame_id'], ids) and np.array_equal(r['timestamp_us'], ts))
        check(key + 'cache geometry dimensions', a['x'].shape == a['mask'].shape == (n, 69) and a['mask'].dtype.kind == 'b')
        check(key + 'shape relation rgb dimensions', a['retained_probability'].shape == (n, 7) and r['probability'].shape == (n, 3) and a['rgb'].shape == (n, 768))
        image, rgb, observed, anchors, spans = rebuild_masks(a, r, frozen['geometry_valid'])
        for field, values in [('image_available', image), ('rgb_available', rgb), ('evidence_available', observed), ('motion_anchor', anchors)]:
            check(key + 'independent ' + field, values.dtype.kind == 'b' and values.tolist() == source[field])
        check(key + 'fragments rebuilt', spans == source['observed_fragments'])
        check(key + 'blank cannot anchor', not np.any(anchors[~image]) and not np.any(observed[~image]))
        support, starts, centers = rgb_row_support(ts, rgb)
        check(key + 'exact RGB structural recipe', starts == source['rgb_lineage']['window_starts'] and centers == source['rgb_lineage']['center_twice_us'])
        check(key + 'no unsaved window value claim', source['rgb_lineage']['saved_window_vectors'] is False and source['rgb_lineage']['reinterpolation_values_independently_reconstructed'] is False)
        destination = Path(saved['prediction']['path']).resolve()
        check(key + 'new own prediction file', destination.parent == (U / 'predictions').resolve() and destination.name == cid + '.npz')
        prediction_paths.add(str(destination))
        check(key + 'prediction bytes pinned', sha(destination) == saved['prediction']['sha256'])
        pred = load_npz(destination)
        logits, probability = pred['logits'], pred['probabilities']
        check(key + 'finite five columns', logits.shape == probability.shape == (n, 5) and np.isfinite(logits).all() and np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all())
        safe = logits.astype(np.float64)
        sigmoid = np.exp(-np.logaddexp(0.0, -safe))
        check(key + 'stored sigmoid arithmetic', np.allclose(probability, sigmoid, rtol=2e-7, atol=8e-8))
        for name, value in [('frame_id', ids), ('timestamp_us', ts), ('evidence_available', observed), ('rgb_available', rgb), ('motion_anchor', anchors)]:
            check(key + 'saved ' + name, np.array_equal(pred[name], value) and pred[name].dtype == value.dtype)
        check(key + 'masked logits never real evidence', np.all(logits[~anchors] == 0) and np.all(probability[~anchors] == .5))
        check(key + 'native output denominator preserved', saved['native_frames'] == n and saved['original_axis_preserved'] is True)
        check(key + 'old sidecars preserved', saved['source_ledger'] == source['source_ledger'] and saved['cache_binding'] == source['analysis'])
        check(key + 'missing native rows reported', saved['image_unavailable_frames'] == ids[~image].tolist() and saved['unobserved_frames'] == ids[~observed].tolist())
        check(key + 'no whole action label', saved['whole_action_correctness'] is None)
        check(key + 'cached head timing only', type(saved['head_seconds']) in [int, float] and math.isfinite(saved['head_seconds']) and saved['head_seconds'] >= 0)
        path = np.asarray(source['prototype_path'], dtype=np.int64)
        old = read(source['old_motion_feedback']['path'])
        check(key + 'single preexisting path retained', source['prototype_path'] == old['path'] and path.ndim == 2 and path.shape[1] == 2 and path[:, 0].min() >= 0 and path[:, 0].max() < n)
        old_by_target = {v['target']: v for v in old['motion_observations']}
        check(key + 'target count unchanged', len(saved['observations']) == len(saved['full_native_diagnostics']) == len(source['scopes']) == len(old_by_target))
        primary_expected, full_expected = [], []
        for scope, actual, full in zip(source['scopes'], saved['observations'], saved['full_native_diagnostics']):
            t = scope['target_index']; q = key + 'target' + str(t) + ' '
            lo, hi = scope['indices']; clo, chi = scope['canonical_frames']
            native = path[(path[:, 1] >= clo) & (path[:, 1] <= chi), 0]
            check(q + 'scope was not score selected', len(native) > 0 and [int(native.min()), int(native.max())] == [lo, hi])
            check(q + 'old motion identity', scope['canonical_frames'] == old_by_target[t]['canonical_frames'] and scope['guide_id'] == old_by_target[t]['guide_id'] and scope['frame_ids'] == old_by_target[t]['frame_ids'])
            check(q + 'all fixed native slots retained', scope['frame_ids'] == ids[lo:hi + 1].tolist() and scope['timestamp_us'] == ts[lo:hi + 1].tolist())
            check(q + 'truth remains limited', scope['expected_occurrence'] == (1 if kind in NORMAL else None) and scope['expectation_is_new_independent_gold'] is False and all(scope[k] is None for k in ['whole_clause_truth', 'educational_error', 'complete_action']))
            expected = expected_bag(cid, t, scope, False, probability, anchors, ids, ts, support)
            whole = expected_bag(cid, t, scope, True, probability, anchors, ids, ts, support)
            for name, value in expected.items():
                check(q + 'primary ' + name, actual.get(name, '__ABSENT__') == value)
            for name, value in whole.items():
                check(q + 'diagnostic ' + name, full.get(name, '__ABSENT__') == value)
            check(q + 'exact primary schema', set(actual) == set(expected))
            check(q + 'exact diagnostic schema', set(full) == set(whole))
            if actual['prediction_coverage'] < 1:
                check(q + 'gap cannot prove absence', not actual.get('absence_assessable', False))
                if not actual.get('high_confirmation', False):
                    check(q + 'low with gap is insufficient', actual['status'] == 'INSUFFICIENT_OBSERVATION')
            if actual['probability'] is not None:
                check(q + 'same column broad monotonic', full['probability'] is not None and full['probability'] >= actual['probability'])
            primary_expected.append(expected); full_expected.append(whole)
            at = actual['anchor']
            verified_rows.append(dict(case_id=cid, target_index=t, kind=kind, fixed_scope=scope['indices'],
                fixed_status=actual['status'], fixed_probability=actual['probability'],
                full_native_status=full['status'], full_native_probability=full['probability'],
                expected_occurrence=scope['expected_occurrence'], expectation_is_independent_gold=False,
                anchor_source_frame=None if at is None else source['source_ledger']['source_frame_indices'][at],
                anchor_original_timestamp_us=None if at is None else source['source_ledger']['original_source_timestamp_us'][at],
                scope_accuracy=None, educational_error=None, complete_action=None))
        expected_by_case[cid] = dict(primary=primary_expected, full=full_expected)
    check('exact24 unique predictions and inputs', len(prediction_paths) == len(input_paths) == 24)
    expected_summary = []
    for kind in sorted(KINDS):
        cases = [cid for cid, v in inputs.items() if v['kind'] == kind]
        bags = [b for cid in cases for b in expected_by_case[cid]['primary']]
        whole = [b for cid in cases for b in expected_by_case[cid]['full']]
        summary = dict(kind=kind, clips=len(cases), scopes=len(bags),
                       high_occurrence=sum(b.get('high_confirmation', False) for b in bags),
                       full_native_high_diagnostic=sum(b.get('high_confirmation', False) for b in whole),
                       insufficient=sum(b['status'] == 'INSUFFICIENT_OBSERVATION' for b in bags),
                       scopes_with_gaps=sum(b['prediction_coverage'] < 1 for b in bags),
                       normal_preservation_expectation=kind in NORMAL,
                       exact_scope_accuracy=None, whole_sign_error_truth=None)
        expected_summary.append(summary)
    check('independent summary reconstruction', result['summary'] == expected_summary)
    for s in result['summary']:
        check(s['kind'] + ' summary types', all(type(s[k]) is int for k in ['clips', 'scopes', 'high_occurrence', 'full_native_high_diagnostic', 'insufficient', 'scopes_with_gaps']) and type(s['normal_preservation_expectation']) is bool)
    preservation = [r for r in verified_rows if r['kind'] in NORMAL]
    unknown = [r for r in verified_rows if r['kind'] not in NORMAL]
    check('all20 and10 retained in reporting', len(preservation) == 20 and len(unknown) == 10 and all(r['expected_occurrence'] is None for r in unknown))
    report = dict(status='PASS_INDEPENDENT_SAVED_CONTROL_ARITHMETIC', checks_count=len(checks), failed_checks=[],
                  checks=checks, inputs=24, primary_scopes=30, diagnostic_scopes=30,
                  summary=expected_summary, rows=verified_rows,
                  normal_preservation=dict(denominator=20,
                      primary_high=sum(r['fixed_probability'] is not None and r['fixed_probability'] >= .8 for r in preservation),
                      full_native_high=sum(r['full_native_probability'] is not None and r['full_native_probability'] >= .8 for r in preservation)),
                  unknown_edit_scopes=10, original_models_and_inputs_unchanged=True,
                  new_model_imports=0, new_model_loads=0, new_forwards=0, training=0, new_DTW=0, source_video_decodes=0,
                  scope='Independent saved arithmetic by another Codex agent; not independent human KSL truth.',
                  limitations=['24 controls depend on four public DEV12 sources; not24 independent people.',
                      'Normal20 is a preservation expectation; cut5/blackout5 lack new occurrence truth.',
                      'Stored logits and lineage verified, model forward not independently rerun.',
                      'RGB support is structural; unsaved encoder window values were not reconstructed.',
                      'High movement evidence does not certify clause, order, repetitions or complete action.',
                      'Low observed score is not automatic absence or educational error.'],
                  whole_goal_complete=False,
                  pins={str(U / p): sha(U / p) for p in ['INPUT_FROZEN.json', 'PLAN_FROZEN.json', 'EXECUTION_FROZEN.json', 'CODE_REVIEW.json', 'RESULT.json', 'RUN_FINISHED.json', 'verify_saved.py']})
    write(U / 'SAVED_VERIFIED.json', report)
    print(json.dumps({k: report[k] for k in ['status', 'checks_count', 'normal_preservation', 'unknown_edit_scopes']}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        write(U / ('VERIFY_FAILURE_' + str(time.time_ns()) + '.json'),
              dict(status='FAILED_SAVED_AUDIT', traceback=traceback.format_exc(), new_model_forwards=0))
        raise
