"""Independently reconstruct saved AC metrics/qualification without any model.

No evaluator, model, training, runtime-qualification or RGB-support helper is
imported. Run only after the one frozen EVALUATION.json exists.
"""
from pathlib import Path
from collections import Counter
from fractions import Fraction
import datetime
import hashlib
import json
import math
import traceback
import numpy as np

U = Path(__file__).resolve().parent
A = U.parent / 'word100_action_completion_supervision_20261102'
CHECKS = Counter()
PIN_CACHE = {}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check(value, reason, group='contract'):
    CHECKS[group] += 1
    if not value:
        raise AssertionError(reason)


def equal(actual, expected, path='value'):
    if isinstance(expected, dict):
        check(isinstance(actual, dict) and set(actual) == set(expected), f'{path}: dictionary keys', 'stored_arithmetic')
        for k in expected:
            equal(actual[k], expected[k], path + '.' + str(k))
    elif isinstance(expected, list):
        check(isinstance(actual, list) and len(actual) == len(expected), f'{path}: list shape', 'stored_arithmetic')
        for i, (a, b) in enumerate(zip(actual, expected)):
            equal(a, b, path + '/' + str(i))
    elif isinstance(expected, float):
        check(isinstance(actual, (int, float)) and math.isfinite(actual) and abs(actual-expected) <= 1e-10, f'{path}: number differs', 'stored_arithmetic')
    else:
        check(actual == expected, f'{path}: value differs', 'stored_arithmetic')


def pin(path, expected):
    path = str(Path(path).resolve())
    if path not in PIN_CACHE:
        PIN_CACHE[path] = digest(path)
    check(PIN_CACHE[path] == expected, 'Hash changed: ' + path, 'hash')


def probability(x):
    check(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1, 'Invalid probability', 'probability')


def raw_label(p):
    return 1 if p >= .8 else 0 if p <= .2 else None


def expected_proposals(points, available):
    # Scan each observed stretch and record the start of each confident run.
    # Unlike the production function this uses a single event scan per stretch.
    stretches, current = [], []
    for j, ok in enumerate(available):
        if ok:
            current.append(j)
        elif current:
            stretches.append(current)
            current = []
    if current:
        stretches.append(current)
    candidates = []
    for ix in stretches:
        starts, terminal_groups, previous = [], [], None
        for j in ix:
            if points[j][0] >= .8 and (previous is None or points[previous][0] < .8):
                starts.append(j)
            if points[j][1] >= .8:
                if previous is None or points[previous][1] < .8:
                    terminal_groups.append([])
                terminal_groups[-1].append(j)
            previous = j
        for group in terminal_groups:
            for s in starts:
                t = next((v for v in group if v > s), None)
                if t is not None:
                    candidates.append(dict(start=s, end=t, output_indices=list(range(s, t+1)),
                        setup_index=s, terminal_index=t, setup_probability=points[s][0], terminal_probability=points[t][1]))
    candidates.sort(key=lambda c: (c['end'], c['start']))
    return candidates[:8], len(candidates)


def expected_status(p, available, qualifying):
    if p >= .8 and qualifying:
        return 'OBSERVED_PROCESS_SUPPORT'
    if not any(available):
        return 'INSUFFICIENT_OBSERVATION'
    if p <= .2 and qualifying:
        return 'PROCESS_AND_LOCAL_EVIDENCE_DISAGREE'
    return 'PROCESS_NOT_CONFIRMED' if p <= .2 else 'PROCESS_EVIDENCE_UNCONFIRMED'


def context_for(support, nominal):
    # Reconstruct structural support from encoder slots and rational window
    # coefficients. Do not trust cached row_encoded_support unions or helper.
    check(nominal == list(range(nominal[0], nominal[-1]+1)), 'Nominal cache selection is not contiguous', 'support')
    encoded, consumed, active = set(), set(), set()
    av, finite = support['image_available'], support['rgb_finite']
    for j in nominal:
        terms = support['row_window_terms'][j]
        check(sum((Fraction(num, den) for _, num, den in terms), Fraction(0)) == 1, 'Interpolation weights do not sum to1', 'support')
        for w, num, den in terms:
            check(Fraction(num, den) > 0, 'Zero/negative support coefficient included', 'support')
            slots = support['windows'][w]['slot_rows']
            encoded.update(slots)
            if av[j] and finite[j]:
                consumed.update(slots)
                active.add(w)
    raw, use, windows = sorted(encoded), sorted(consumed), sorted(active)
    observed = [j for j in use if av[j]]
    synthetic = [j for j in use if not av[j]]
    sf, st = support['source_frame_ids'], support['source_timestamp_us']
    ns = set(nominal)
    return dict(nominal_cache_rows=nominal, nominal_source_frames=[sf[j] for j in nominal],
        nominal_source_timestamps_us=[st[j] for j in nominal], nominal_unavailable_rows=[j for j in nominal if not av[j]],
        encoded_context_cache_rows=raw, consumed_context_cache_rows=use,
        observed_context_source_frames=[sf[j] for j in observed], observed_context_source_timestamps_us=[st[j] for j in observed],
        synthetic_context_cache_rows=synthetic, synthetic_context_source_coordinates=[sf[j] for j in synthetic],
        outside_nominal_consumed_rows=[j for j in use if j not in ns],
        outside_nominal_observed_source_frames=[sf[j] for j in observed if j not in ns],
        contributing_window_indices=windows,
        internal_padding_slots=sum(not m for w in windows for m in support['windows'][w]['real_slot_mask']),
        structural_context_is_direct_nominal_evidence=False, local_process_truth=None, exact_completion_boundary=None)


def raw_metrics(rows):
    known = [r for r in rows if r['truth'] is not None]
    classes = {}
    for label in (0, 1):
        subset = [r for r in known if r['truth'] == label]
        classes[str(label)] = dict(total=len(subset), raw_correct=sum((r['probability'] >= .5) == bool(label) for r in subset),
            confident_correct=sum(r['raw_display'] == label for r in subset),
            confident_wrong=sum(r['raw_display'] == 1-label for r in subset), abstained=sum(r['raw_display'] is None for r in subset))
    correct = sum((r['probability'] >= .5) == bool(r['truth']) for r in known)
    balanced = sum(v['raw_correct']/v['total'] for v in classes.values())/2 if all(v['total'] for v in classes.values()) else None
    return dict(counts=dict(Counter(str(r['truth']) for r in rows)), known=len(known), raw_correct=correct,
        raw_accuracy=correct/len(known) if known else None, class_counts=classes, raw_balanced_accuracy=balanced,
        unknown_raw_confident=sum(r['truth'] is None and r['raw_display'] is not None for r in rows),
        disclaimer='Raw confidence is not traceable observed evidence. Unknown truth remains unknown even if raw probability is high.')


def qualified_metrics(rows):
    pos = [r for r in rows if r['truth'] == 1]
    zero = [r for r in rows if r['truth'] == 0]
    yes = lambda r: r['qualification']['qualified_display'] == 1
    return dict(total=len(rows), known_positive=len(pos), known_zero=len(zero), unknown=sum(r['truth'] is None for r in rows),
        confirmed_positive_correct=sum(yes(r) for r in pos), confirmed_positive_wrong=sum(yes(r) for r in zero),
        unknown_confirmed_positive=sum(r['truth'] is None and yes(r) for r in rows),
        known_positive_recall=sum(yes(r) for r in pos)/len(pos) if pos else None,
        raw_low_on_known_positive=sum(r['qualification']['raw_display'] == 0 for r in pos),
        raw_low_on_known_zero=sum(r['qualification']['raw_display'] == 0 for r in zero),
        all_absence_claims=False, qualified_negative_is_not_defined=True)


def expected_sources():
    cases = {}
    for role, tf, im in [('fit', 'FIT_VIEW_TARGETS_V2.json', 'PROCESS_INPUT_MANIFEST.json'),
                         ('development', 'DEV_VIEW_TARGETS_V2.json', 'DEV_PROCESS_INPUT_MANIFEST.json')]:
        source = {r['pixel_view_key']: r for r in read(A/im)['rows']}
        for r in read(U/tf)['rows']:
            s = source[r['pixel_view_key']]
            check(r['role'] == s['role'] == role, 'Role binding')
            axis = [dict(output_index=f['output_index'], source_frame=f['source_native_frame'],
                         source_timestamp_us=f['source_native_timestamp_us'], input_timestamp_us=f['view_timestamp_us'],
                         image_available=f['image_available']) for f in r['ordered_frame_map']]
            cases[r['view_id']] = dict(original=r, source=s, axis=axis, origin='reviewed_view',
                person=r['recorded_person_id'], input=s['input'], input_sha256=s['input_sha256'],
                sensor_selection=None, expected_sensor_frame_ids=list(range(len(axis))))
    for r in read(A/'PROCESS_RUNTIME_SUPERVISION_BINDING.json')['rows']:
        axis = [dict(output_index=f['output_index'], source_frame=f['source_frame'],
                     source_timestamp_us=f['source_timestamp_us'], input_timestamp_us=f['source_timestamp_us'],
                     image_available=f['image_available']) for f in r['proposal_axis']]
        cases[r['query_id']+'__runtime_proposal'] = dict(original=r, source=r, axis=axis, origin='runtime_proposal',
            person=r['person'], input=r['input'], input_sha256=r['input_sha256'],
            sensor_selection=r['input_selection_indices'], expected_sensor_frame_ids=[x['source_frame'] for x in axis])
    check(len(cases) == 357, 'Expected357 fixed cases')
    return cases


def main():
    report_path = U/'EVALUATION_VERIFIED.json'
    check(not report_path.exists(), 'No silent rerun of completed saved verification')
    check((U/'EVALUATION.json').exists(), 'Wait for one fixed EVALUATION; never execute inference here')
    c, train, ev = read(U/'ALIGNMENT_FROZEN.json'), read(U/'TRAIN_RESULT.json'), read(U/'EVALUATION.json')
    check(ev['status'] == 'ONE_FIXED_ALIGNED_CANDIDATE_EVALUATED_NOT_ADOPTION', 'Evaluation status')
    check(ev['contract_sha256'] == train['contract_sha256'] == digest(U/'ALIGNMENT_FROZEN.json'), 'Frozen contract linkage')
    check(ev['checkpoint_sha256'] == train['checkpoint_sha256'] == digest(train['checkpoint']), 'Checkpoint linkage')
    check(train['normalization_sha256'] == digest(train['normalization']), 'Normalization linkage')
    check(train['epochs'] == 80 and train['seed'] == 42 and train['parameters'] == 47980, 'Fixed training budget')
    for p, h in c['pins'].items():
        pin(p, h)
    support_doc, support_verified = read(U/'RGB_SUPPORT_MANIFEST.json'), read(U/'RGB_SUPPORT_VERIFIED.json')
    check(ev['rgb_support_manifest_sha256'] == support_verified['manifest_sha256'] == digest(U/'RGB_SUPPORT_MANIFEST.json'), 'Support manifest verification linkage')
    check(ev['rgb_support_verification_sha256'] == digest(U/'RGB_SUPPORT_VERIFIED.json'), 'Support verifier linkage')
    check(support_verified['helper_sha256'] == digest(U/'process_rgb_support.py'), 'Support helper verification linkage')
    nominal_path = Path(c['nominal_span_manifest']['path'])
    check(ev['nominal_span_manifest_sha256'] == c['nominal_span_manifest']['sha256'] == digest(nominal_path), 'Nominal manifest linkage')
    nominal_rows = [r for r in read(nominal_path)['rows'] if r['role'] == 'fit']
    nominal_values = [r['observed_process_evidence'] for r in nominal_rows]
    equal(ev['nominal_binding'], dict(status='BOUND_DIRECT_NOMINAL_PROCESS_LABELS', records=len(nominal_values),
        known=sum(v is not None for v in nominal_values), positive=sum(v == 1 for v in nominal_values),
        negative=sum(v == 0 for v in nominal_values), unknown=sum(v is None for v in nominal_values),
        manifest=str(nominal_path.resolve()), manifest_sha256=digest(nominal_path)), 'nominal_binding')
    nominal_by_case = {}
    for t in nominal_rows:
        nominal_by_case.setdefault(t['case_id'], []).append(t)
    expected = expected_sources()
    check(len(ev['rows']) == len(expected), 'Evaluation denominator')
    check({r['case_id'] for r in ev['rows']} == set(expected), 'Missing or duplicate fixed case')
    sensor_cache = {}
    process_forwards, motion_forwards = set(), set()
    nominal_checked = 0
    for row in ev['rows']:
        cid = row['case_id']
        item = expected[cid]
        original, axis = item['original'], item['axis']
        n = len(axis)
        equal(row['axis'], axis, cid+'.axis')
        check(row['view_id'] == cid and row['query_id'] == original['query_id'], 'Saved identity')
        for key in ['clip_id', 'role', 'physical_query', 'source_sha256']:
            check(row[key] == original[key], 'Source/role/query changed: '+key)
        check(row['person'] == item['person'] and row['origin'] == item['origin'], 'Person/origin mismatch')
        check(row['kind'] == original.get('construction_kind_audit_only', 'automatic_runtime_proposal'), 'Control category changed')
        check(row['input_sha256'] == item['input_sha256'], 'Input hash mismatch')
        check(c['pins'].get(str(Path(item['input']).resolve())) == item['input_sha256'], 'Actual sensor not frozen')
        check(row['truth'] == original['observed_process_evidence'], 'Truth altered from direct supervision')
        check(row['frames'] == n and row['image_available'] == [x['image_available'] for x in axis], 'Saved availability changed')
        ih = item['input_sha256']
        if ih not in sensor_cache:
            pin(item['input'], ih)
            with np.load(item['input'], allow_pickle=False) as z:
                # Inspect sensor axes and finite RGB only, never any predictions.
                sensor_cache[ih] = {k: z[k].copy() for k in ['frame_id', 'timestamp_us', 'image_available']}
                sensor_cache[ih]['rgb_finite'] = np.isfinite(z['rgb']).all(1)
        sensor = sensor_cache[ih]
        ii = item['sensor_selection'] or list(range(n))
        check(sensor['frame_id'][ii].tolist() == item['expected_sensor_frame_ids'], 'Actual NPZ frame axis', 'axis')
        check(sensor['timestamp_us'][ii].tolist() == [x['input_timestamp_us'] for x in axis], 'Actual NPZ PTS', 'axis')
        check(sensor['image_available'][ii].tolist() == row['image_available'], 'Actual NPZ availability', 'axis')
        points = row['point_probabilities']
        check(len(points) == n and all(len(p) == 2 for p in points), 'Point probability shape')
        for p in points:
            for value in p:
                probability(value)
        p = row['probability']; probability(p)
        check(row['raw_display'] == raw_label(p), 'Raw threshold .8/.2 mismatch')
        process_forwards.add((cid, tuple(range(n))))
        spec, uncapped = expected_proposals(points, row['image_available'])
        q = row['qualification']
        check(q['candidate_count'] == len(spec) and len(q['interval_candidates']) == len(spec), 'Candidate count')
        check(q['uncapped_candidates'] == uncapped and q['candidate_cap_reached'] == (uncapped>8), 'Candidate cap')
        check(q['raw_probability'] == p and q['raw_display'] == raw_label(p), 'Qualification raw linkage')
        check(q['unavailable_indices'] == [j for j, yes in enumerate(row['image_available']) if not yes], 'Missing-image list')
        support_entry = support_doc['inputs'][ih]
        check(support_entry['source_sha256'] == row['source_sha256'], 'Support source mismatch')
        support = support_entry['support']
        check(support['rgb_finite'] == sensor['rgb_finite'].tolist(), 'Support effective finite mask mismatch')
        source_lookup = {f: j for j, f in enumerate(support['source_frame_ids'])}
        cache_axis = [source_lookup[x['source_frame']] for x in axis]
        check([support['source_timestamp_us'][j] for j in cache_axis] == [x['source_timestamp_us'] for x in axis], 'Support PTS mismatch')
        equal(q['global_rgb_context_support'], context_for(support, cache_axis), cid+'.global_support')
        for got, want in zip(q['interval_candidates'], spec):
            for key, value in want.items():
                equal(got[key], value, cid+'.proposal.'+key)
            score = got['connected_process_probability']; probability(score)
            span = got['output_indices']
            process_forwards.add((cid, tuple(span)))
            check(got['source_frame_ids'] == [axis[j]['source_frame'] for j in span], 'Candidate source frames')
            check(got['source_timestamps_us'] == [axis[j]['source_timestamp_us'] for j in span], 'Candidate native PTS')
            equal(got['rgb_context_support'], context_for(support, [cache_axis[j] for j in span]), cid+'.candidate_support')
            if span == list(range(n)):
                check(score == p, 'Identical whole/local selection did not reuse prediction')
        chains = [t for t in q['interval_candidates'] if t['connected_process_probability'] >= .8]
        display = 1 if p >= .8 and chains else None
        check(q['qualified_display'] == display and q['witnessed_chain'] == (chains[0] if display else None), 'Qualified evidence threshold')
        check(q['status'] == expected_status(p, row['image_available'], chains), 'Qualification status')
        for key in ['whole_word_correctness', 'learner_error', 'required_repetitions', 'selected_span_truth']:
            check(q[key] is None, 'Unsupported semantic truth')
        for key in ['source_seam_oracle_diagnostic', 'observed_process_absence_certified', 'complete_action_verified', 'exact_event_boundaries']:
            check(q[key] is False, 'Unsupported runtime completion/oracle claim')
        check(q['local_target'] == 'Directly labelled connected process, not net motion', 'Local target conflated')
        seams = original.get('seam_output_indices', [])
        oracle = row['oracle_known_seam_diagnostic']
        valid = []
        for got, base in zip(oracle['interval_candidates'], q['interval_candidates']):
            crosses = any(base['start'] < s <= base['end'] for s in seams)
            equal(got, dict(base, crosses_known_source_seam=crosses), cid+'.oracle_candidate')
            if not crosses and base['connected_process_probability'] >= .8:
                valid.append(got)
        oracle_display = 1 if p >= .8 and valid else None
        check(oracle['qualified_display'] == oracle_display, 'Oracle display')
        check(oracle['witnessed_chain'] == (valid[0] if oracle_display else None), 'Oracle chain')
        check(oracle['status'] == expected_status(p, row['image_available'], valid), 'Oracle status stale')
        check(oracle['source_seam_oracle_diagnostic'] is True and oracle['same_runtime_candidates_and_scores_reused'] is True, 'Oracle scope')
        check(oracle['source_seam_indices'] == sorted(set(seams)), 'Oracle seam provenance')

        point_expected = []
        for t in original['sparse_relation_targets']:
            j = t['output_index']
            point_expected.append(dict(output_index=j, source_frame=t['source_frame'], setup_truth=t['setup_relation'],
                terminal_truth=t['terminal_relation'], setup_probability=points[j][0], terminal_probability=points[j][1]))
        equal(row['sparse_point_scores'], point_expected, cid+'.point_scores')
        check(len(row['interval_scores']) == len(original['fully_retained_motion_interval_targets']), 'Netmotion target count')
        seen_motion = {}
        for got, target in zip(row['interval_scores'], original['fully_retained_motion_interval_targets']):
            check(got['truth'] == target['label'] and got['output_indices'] == target['output_indices'] and got['source_frames'] == target['source_frames'], 'Netmotion target scope changed')
            probability(got['probability'])
            key = tuple(target['output_indices'])
            if key in seen_motion:
                check(got['probability'] == seen_motion[key], 'Duplicate motion selection differs')
            seen_motion[key] = got['probability']
            motion_forwards.add((cid, key))
        target_spans = nominal_by_case.get(cid, [])
        check(len(row['nominal_process_scores']) == len(target_spans), 'Nominal diagnostics missing')
        seen_process = {tuple(range(n)): p, **{tuple(v['output_indices']): v['connected_process_probability'] for v in q['interval_candidates']}}
        for got, t in zip(row['nominal_process_scores'], target_spans):
            check(row['role'] == 'fit', 'DEV used as nominal training diagnostic')
            for key in ['span_id', 'output_indices', 'source_frames', 'source_timestamps_us', 'image_available', 'binding_kind', 'direct_review']:
                equal(got[key], t[key], cid+'.nominal.'+key)
            check(got['truth'] == t['observed_process_evidence'], 'Nominal truth copied/changed')
            probability(got['probability'])
            check(got['raw_display'] == raw_label(got['probability']), 'Nominal raw display')
            key = tuple(t['output_indices'])
            if key in seen_process:
                check(got['probability'] == seen_process[key], 'Identical process span not memoized')
            seen_process[key] = got['probability']
            process_forwards.add((cid, key))
            equal(got['rgb_context_support'], context_for(support, [cache_axis[j] for j in key]), cid+'.nominal_support')
            check(got['whole_word_correctness'] is None and got['learner_error'] is None, 'Nominal semantic escalation')
            nominal_checked += 1
        check(row['recovered'] == (row['truth'] == 1 and display == 1), 'Recovered denominator/truth')
        check(row['learned_selected_chain_exact_time_truth'] is None and row['whole_word_correctness'] is None and row['learner_error'] is None, 'Unsupported selected-chain truth')
        check(row['additional_DTW'] == 0 and row['end_to_end_latency_measured'] is False, 'Timing scope escalation')
        check(0 <= row['runtime_head_seconds'] <= row['diagnostic_inclusive_seconds']+1e-6, 'Timing scope inconsistent')
    equal(ev['forward_counts'], {'process': len(process_forwards), 'netmotion': len(motion_forwards)}, 'forward_budget')

    summaries = {}
    physical_queries = sorted({r['physical_query'] for r in ev['rows']})
    for role in ['fit', 'development']:
        vv = [r for r in ev['rows'] if r['role'] == role and r['origin'] == 'reviewed_view']
        aa = [r for r in ev['rows'] if r['role'] == role and r['origin'] == 'runtime_proposal']
        per = {q: raw_metrics([r for r in vv if r['physical_query'] == q]) for q in physical_queries}
        src = {q: raw_metrics([r for r in vv if r['query_id'] == q]) for q in sorted({r['query_id'] for r in vv})}
        ba = [v['raw_balanced_accuracy'] for v in per.values() if v['raw_balanced_accuracy'] is not None]
        sb = [v['raw_balanced_accuracy'] for v in src.values() if v['raw_balanced_accuracy'] is not None]
        summaries[role] = dict(raw=raw_metrics(vv), raw_per_query=per,
            raw_macro_query_balanced_accuracy=sum(ba)/len(ba), raw_macro_source_query_balanced_accuracy=sum(sb)/len(sb),
            qualification=qualified_metrics(vv), automatic=qualified_metrics(aa),
            automatic_per_query={q: qualified_metrics([r for r in aa if r['physical_query'] == q]) for q in physical_queries},
            native_per_query={q: qualified_metrics([r for r in vv if r['kind'] == 'native' and r['physical_query'] == q]) for q in physical_queries},
            by_kind={k: qualified_metrics([r for r in vv if r['kind'] == k]) for k in sorted({r['kind'] for r in vv})})
    equal(ev['summaries'], summaries, 'summaries')

    # Existing baseline predictions are read as already-completed evidence only.
    # Do not rescore old probability vectors against new point/interval labels.
    old_raw = read(A/'PROCESS_EVALUATION.json')
    old_auto = read(A/'PROCESS_AUTO_EVALUATION.json')
    by_id = {r['case_id']: r for r in ev['rows']}
    common_points = common_intervals = added_points = added_intervals = 0
    for old in old_raw['rows']:
        r = by_id[old['view_id']]
        check(r['truth'] == old['truth'] and r['role'] == old['role'] and r['physical_query'] == old['physical_query'], 'Whole baseline common labels differ')
        check([a['source_frame'] for a in r['axis']] == old['source_frames'], 'Whole baseline frame coverage differs')
        check([a['source_timestamp_us'] for a in r['axis']] == old['source_timestamps_us'], 'Whole baseline PTS differs')
        check(r['image_available'] == old['image_available'], 'Whole baseline missing evidence differs')
        points_new = {v['output_index']: v for v in r['sparse_point_scores']}
        for v in old['sparse_point_scores']:
            w = points_new[v['output_index']]
            check((v['source_frame'], v['setup_truth'], v['terminal_truth']) == (w['source_frame'], w['setup_truth'], w['terminal_truth']), 'Common point truth changed')
        motion_new = {tuple(v['output_indices']): v for v in r['interval_scores']}
        for v in old['interval_scores']:
            w = motion_new[tuple(v['output_indices'])]
            check(v['source_frames'] == w['source_frames'] and v['truth'] == w['truth'], 'Common interval truth changed')
        common_points += len(old['sparse_point_scores'])
        common_intervals += len(old['interval_scores'])
        added_points += len(r['sparse_point_scores']) - len(old['sparse_point_scores'])
        added_intervals += len(r['interval_scores']) - len(old['interval_scores'])
    for old in old_auto['view_rows']:
        check(by_id[old['view_id']]['truth'] == old['truth'], 'Qualified whole-view baseline label differs')
    for old in old_auto['automatic_rows']:
        r = by_id[old['query_id']+'__runtime_proposal']
        check(r['truth'] == old['truth'], 'Automatic baseline denominator/label differs')
        check([x['source_frame'] for x in r['axis']] == old['source_frame_ids'], 'Automatic proposal frame scope differs')
        check([x['source_timestamp_us'] for x in r['axis']] == old['source_timestamps_us'], 'Automatic proposal PTS differs')
        check(r['image_available'] == old['image_available'], 'Automatic baseline availability differs')
    check(ev['old_models_executed'] == 0 and ev['extraction'] == 0 and ev['new_DTW'] == 0, 'Forbidden repeat execution')
    check(ev['nominal_diagnostic_is_fit_only'] is True and ev['repeated_development'] is True, 'Evaluation limitation missing')
    check(ev['whole_goal_complete'] is False and ev['independent_human_expert'] is False, 'Completion/expert claim')
    check(ev['end_to_end_speed_improvement_claim'] is False and ev['selected_span_local_truth_not_inferred'] is True, 'Unsupported latency/local-truth claim')
    result = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), status='PASS_SAVED_EVALUATION_RECONSTRUCTED_NO_MODEL_EXECUTION',
        evaluation_sha256=digest(U/'EVALUATION.json'), checkpoint_sha256=ev['checkpoint_sha256'],
        contract_sha256=ev['contract_sha256'], verifier_sha256=digest(__file__),
        checks=dict(CHECKS), total_checks=sum(CHECKS.values()), rows=357, nominal_fit_diagnostics=nominal_checked,
        forward_counts_reconstructed={'process': len(process_forwards), 'netmotion': len(motion_forwards)},
        summaries=summaries, checked_input_pins=len(PIN_CACHE),
        baseline_scope=dict(whole_views=297, automatic_queries=60, common_point_rows=common_points,
            additional_point_rows=added_points, common_netmotion_intervals=common_intervals,
            additional_netmotion_intervals=added_intervals,
            official_comparison='Same whole-process labels/qualified and automatic denominators only; added v2 point/motion supervision is a separate diagnostic scope.',
            old_scores_recomputed=False),
        model_imports=0, new_model_inference=0, new_training=0, new_sensor_extraction=0,
        repeated_development_not_final_independent_test=True, independent_human_expert=False,
        end_to_end_speed_improvement_verified=False, selected_span_exact_time_truth=None,
        whole_goal_complete=False)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'checks': result['total_checks'], 'rows':357,
                      'sha256': digest(report_path), 'development': summaries['development']}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        p = U/('FIRST_SAVED_VERIFY_FAILURE_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')+'.txt')
        p.write_text(traceback.format_exc(), encoding='utf-8')
        raise
