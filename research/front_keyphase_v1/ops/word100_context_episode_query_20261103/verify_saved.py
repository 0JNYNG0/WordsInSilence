"""Saved six-scope audit; NumPy only, no model imports/loads/forwards or decode."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import traceback
import numpy as np

U = Path(__file__).resolve().parent
O = U.parent
COUNTS = Counter()
PINS = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    path = Path(path)
    PINS[str(path)] = sha(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def check(condition, category, detail):
    COUNTS[category] += 1
    if not condition:
        raise AssertionError(f'{category}: {detail}')


def pin(path, value):
    check(sha(path) == value, 'hash', path)
    PINS[str(path)] = value


def main():
    out = U / 'SAVED_VERIFIED.json'
    assert not out.exists()
    result = read(U / 'RESULT.json')
    summary = read(U / 'SAVED_SUMMARY.json')
    frozen = read(U / 'FROZEN.json')
    execution = read(U / 'EXECUTION_FROZEN.json')
    labels = read(U / 'LATE_VISUAL_REVIEW.json')
    started = read(U / 'RUN_STARTED.json')
    finished = read(U / 'RUN_FINISHED.json')
    review = read(U / 'REVIEW_CODE.json')
    tests = read(U / 'REVIEW_STRUCTURAL_TESTS.json')
    preflight = read(U / 'PREFLIGHT.json')
    first = read(U / 'FIRST_PREFLIGHT_FAILURE.json')
    interpretation = read(U / 'INTERPRETATION.json')
    old_label_code = O / 'word100_learned_motion_order_20261102/label_motion.py'
    PINS[str(old_label_code)] = sha(old_label_code)
    check('net physical direction over this observed episode; no exact onset, per-frame class, required repetition count, or linguistic error label' in old_label_code.read_text(encoding='utf-8-sig'), 'target_contract', 'Original interval direction scope')
    check(interpretation['no_new_accuracy_claim'] is True and interpretation['whole_goal_complete'] is False and 'scope sensitivity' in interpretation['causal_limit'] and 'not causally isolated' in interpretation['causal_limit'], 'target_contract', 'Occurrence/net-direction distinction and causal limitation')
    for doc, field in [(frozen, 'pins'), (execution, 'pins'), (summary, 'pins'), (labels, 'pins'), (review, 'code_pins'), (tests, 'code_pins')]:
        for path, value in doc[field].items():
            pin(path, value)
    check(finished['status'] == 'COMPLETE' and started['pid'] == finished['pid'] and started['at'] < finished['at'], 'execution', 'Terminal one-run identity')
    check(started['scopes'] == 6 and finished['result_sha256'] == sha(U / 'RESULT.json'), 'execution', 'Completed saved six-result hash')
    check(execution['no_window_search'] is True and execution['no_training'] is True and execution['old_completed_exact_scopes_reused_or_rerun'] == 0, 'execution', 'No old inference rerun or scope search')
    check(execution['thresholds'] == 'inherited .8/.2, geometry .5, min3frames', 'execution', 'Frozen thresholds')
    check(first['model_execution_before_correction'] == 0 and first['old_evaluations_rerun'] is False, 'first_failure', 'Corrected before model execution')
    PINS[str(U / first['initial_code'])] = sha(U / first['initial_code'])
    check(tests['status'] == 'PASS' and len(tests['checks']) == tests['check_count'] == review['structural_tests']['checks'] == 61, 'prior_structural_checks', '61 saved tests')
    check(len({c['name'] for c in tests['checks']}) == 61 and all(c['passed'] is True for c in tests['checks']) and tests['failures'] == 0, 'prior_structural_checks', 'Unique successful saved cases')
    check(tests['model_imports'] == tests['model_loads'] == tests['model_forwards'] == tests['media_views'] == 0, 'prior_structural_checks', 'Model-free structural method')
    check(review['approved_for_six_scope_diagnostic'] is True and review['structural_tests']['sha256'] == sha(U / 'REVIEW_STRUCTURAL_TESTS.json'), 'prior_structural_checks', 'Approval report bound to actual tests')

    requests = {(r['clip_id'], r['scope']): r for r in execution['scopes']}
    rows = {(r['clip_id'], r['scope']): r for r in result['rows']}
    cases = {r['clip_id']: r for r in frozen['rows']}
    lrows = {r['clip_id']: r for r in labels['rows']}
    sums = {r['clip_id']: r for r in summary['rows']}
    check(len(requests) == len(rows) == len(result['rows']) == 6 and set(rows) == set(requests), 'scope', 'Six fixed unique scopes')
    check(set(cases) == set(lrows) == set(sums) and {r['person'] for r in cases.values()} == {1, 4, 7}, 'scope', 'Exactly three selected FIT records')
    old_file = O / 'word100_component_rgb_context_20261102/SCOPE_RESULT.json'
    old_all = read(old_file)
    old = {r['record']['clip_id']: r for r in old_all['rows'] if r['record']['clip_id'] in cases}
    check(len(old) == 3, 'old_saved_result', 'Only selected FIT evidence summarized')
    compact = []
    all_status = Counter()
    for cid, case in cases.items():
        check(case['role'] == 'fit' and case['word_id'] == 'WORD1133', 'scope', cid)
        pin(case['source'], case['source_sha256'])
        pin(case['input'], case['input_sha256'])
        with np.load(case['input'], allow_pickle=False) as a:
            values = {k: a[k].copy() for k in ['frame_id', 'timestamp_us', 'x', 'mask', 'rgb']}
            n = len(values['frame_id'])
            av = a['image_available'].copy() if 'image_available' in a else np.ones(n, dtype=bool)
            windows = {k: a[k].copy() for k in ['window_starts', 'window_center_us', 'window_features'] if k in a}
        ids = values['frame_id']; ts = values['timestamp_us']; x = values['x']; native_mask = values['mask']
        check(np.array_equal(ids, np.arange(n)) and ts.dtype.kind in 'iu' and np.all(np.diff(ts) > 0), 'native_axis', cid)
        check(av.shape == (n,) and av.dtype.kind == 'b', 'native_axis', 'Boolean availability')
        shape_path = O / 'word100_multipeople_live_feedback_20261101/bank_predictions' / f'{cid}.npz'
        relation_path = O / 'word100_learned_body_relation_20261102/runs/rgb_geometry_42/predictions' / f'{cid}.npz'
        with np.load(shape_path, allow_pickle=False) as a:
            shape = a['probabilities'].copy()
            check(np.array_equal(a['frame_id'], ids) and np.array_equal(a['timestamp_us'], ts), 'native_axis', 'Shape cache exact')
        with np.load(relation_path, allow_pickle=False) as a:
            relation = a['probability'].copy()
            check(np.array_equal(a['frame_id'], ids) and np.array_equal(a['timestamp_us'], ts), 'native_axis', 'Relation cache exact')
        # Independently reconstruct only validity, not features or predictions.
        base_valid = np.isfinite(x[:, :61]) & native_mask[:, :61].astype(bool)
        forearm = x[:, 2:4] - x[:, 4:6]
        hand = x[:, 36:38] - x[:, 4:6]
        squared_length = np.sum(forearm * forearm, axis=1)
        valid_projection = native_mask[:, [2, 3, 4, 5, 36, 37]].all(axis=1) & np.isfinite(squared_length) & (squared_length > .0025)
        along = np.sum(hand * forearm, axis=1) / np.maximum(squared_length, 1e-6)
        side = (hand[:, 0] * forearm[:, 1] - hand[:, 1] * forearm[:, 0]) / np.maximum(squared_length, 1e-6)
        valid = np.column_stack([base_valid, valid_projection & np.isfinite(along), valid_projection & np.isfinite(side), np.isfinite(shape), np.isfinite(relation)]) & av[:, None]
        check(valid.shape == (n, 73), 'feature_mask', 'Frozen 73 inputs')
        finite_rgb = np.isfinite(values['rgb']).all(axis=1)
        available = av & finite_rgb
        # Independent positive-weight interpolation support via searchsorted.
        starts = np.unique(np.r_[np.arange(0, n - 15, 2), n - 16])
        centers2 = ts[starts] + ts[starts + 15]
        if windows:
            check(set(windows) == {'window_starts', 'window_center_us', 'window_features'}, 'rgb_lineage', 'All cached metadata present')
            check(np.array_equal(windows['window_starts'], starts) and np.array_equal(windows['window_center_us'], centers2 / 2), 'rgb_lineage', 'Saved encoder windows')
            reconstructed = np.column_stack([np.interp(ts, centers2 / 2, windows['window_features'][:, j]) for j in range(768)]).astype(np.float32)
            check(np.array_equal(reconstructed[available], values['rgb'][available]), 'rgb_lineage', 'Stored RGB values reconstructed without encoding')
            lineage = 'SAVED_WINDOWS_AND_VALUES_EXACTLY_RECONSTRUCTED'
        else:
            public = read(O / 'word100_multipeople_live_feedback_20261101/PUBLIC_INPUTS.json')
            records = [r for r in public['rows'] if r['clip_id'] == cid]
            check(len(records) == 1 and records[0]['role'] == 'fit' and records[0]['frames'] == n and records[0]['reused_input'] is False, 'rgb_lineage', 'Recipe-only original route')
            check(records[0]['source_sha256'] == case['source_sha256'] and records[0]['input_sha256'] == case['input_sha256'], 'rgb_lineage', 'Recipe-only cache identity')
            lineage = 'VERIFIED_GENERATOR_RECIPE_WITHOUT_SAVED_WINDOW_VALUES'
        for kind in ('broad', 'late'):
            row = rows[cid, kind]; ob = row['observation']; request = requests[cid, kind]
            check({k: row[k] for k in request} == request, 'scope', 'Exact frozen request preserved')
            lo, hi = case[kind + '_frames']; ii = np.arange(lo, hi + 1)
            check(row['frames'] == [lo, hi] and ob['frame_ids'] == ids[ii].tolist() and ob['timestamp_us'] == ts[ii].tolist(), 'native_axis', 'Every requested frame/time preserved')
            check(ob['start_seconds'] == ts[lo] / 1e6 and ob['end_seconds'] == ts[hi] / 1e6, 'native_axis', 'Native playback times')
            check(ob['unavailable_frame_ids'] == ids[ii[~available[ii]]].tolist() and ob['partial_geometry_frame_ids'] == ids[ii[~valid[ii].all(axis=1)]].tolist(), 'feature_mask', 'No missing rows removed')
            coverage = int(valid[ii].sum()) / valid[ii].size
            image_coverage = int(available[ii].sum()) / len(ii)
            check(abs(ob['feature_coverage'] - coverage) < 1e-15 and ob['image_coverage'] == image_coverage, 'feature_mask', 'Coverage denominators')
            p = ob['probabilities']
            check(len(p) == 6 and all(type(v) in (float, int) and math.isfinite(v) and 0 <= v <= 1 for v in p), 'probability', 'Six saved finite independent sigmoid outputs')
            check(ob['target_index'] == 0 and ob['probability'] == p[0] and ob['continuity_probability'] == p[5], 'probability', 'Target0 and continuity indices')
            expected_status = ('MOTION_OBSERVATION_INSUFFICIENT' if image_coverage < .8 or len(ii) < 3 or coverage < .5 else
                               'MOTION_CONTINUITY_UNCONFIRMED' if p[5] < .8 else
                               'DIRECTED_MOTION_OBSERVED' if p[0] >= .8 else 'EXPECTED_MOTION_UNCONFIRMED')
            check(ob['raw_observer_status'] == expected_status, 'status', 'Independent fixed target0 status arithmetic')
            all_status[expected_status] += 1
            encoded_windows = set(); active_windows = set()
            for i in ii:
                t2 = int(ts[i]) * 2
                upper = int(np.searchsorted(centers2, t2, side='left'))
                if upper == len(centers2):
                    ws = [upper - 1]
                elif upper == 0 or centers2[upper] == t2:
                    ws = [upper]
                else:
                    ws = [upper - 1, upper]
                encoded_windows.update(ws)
                if available[i]: active_windows.update(ws)
            encoded = sorted({j for wi in encoded_windows for j in range(int(starts[wi]), int(starts[wi]) + 16)})
            consumed = sorted({j for wi in active_windows for j in range(int(starts[wi]), int(starts[wi]) + 16)})
            observed = [j for j in consumed if av[j]]; synthetic = [j for j in consumed if not av[j]]
            expected_support = dict(nominal_cache_rows=ii.tolist(), nominal_source_frames=ids[ii].tolist(), nominal_source_timestamps_us=ts[ii].tolist(),
                nominal_unavailable_rows=ii[~av[ii]].tolist(), encoded_context_cache_rows=encoded, consumed_context_cache_rows=consumed,
                observed_context_source_frames=ids[observed].tolist(), observed_context_source_timestamps_us=ts[observed].tolist(),
                synthetic_context_cache_rows=synthetic, synthetic_context_source_coordinates=ids[synthetic].tolist(),
                outside_nominal_consumed_rows=[j for j in consumed if j < lo or j > hi],
                outside_nominal_observed_source_frames=[int(ids[j]) for j in observed if j < lo or j > hi],
                contributing_window_indices=sorted(active_windows), internal_padding_slots=0,
                structural_context_is_direct_nominal_evidence=False, local_process_truth=None, exact_completion_boundary=None)
            check(ob['rgb_context_support'] == expected_support, 'rgb_support', 'Independent contributing windows, native rows and outside scope')
            check(ob['rgb_lineage'] == lineage and ob['scope_attribution'] == 'NEURAL_CONTEXT_EVIDENCE_NOT_EXACT_FRAME_TRUTH', 'rgb_support', 'Input route limitations retained')
            check(all(ob[k] is None for k in ('complete_action', 'whole_clause_correctness', 'educational_error')) and ob['correction_enabled'] is False and ob['reference_video_comparisons'] == 0, 'no_semantic_promotion', 'Partial observation only')
            check('잘못했다는 판정은 아닙니다' in ob['message_ko'], 'no_semantic_promotion', 'No movement low-score correction')
            pf = next(r for r in preflight['rows'] if r['person'] == case['person'] and r['scope'] == kind)
            check(pf['frames'] == len(ii) and pf['lineage'] == lineage and pf['rgb_context'] == [observed[0], observed[-1]], 'preflight', 'Same preflight scope/support')
            compact.append(dict(clip_id=cid, person=case['person'], scope=kind, frames=[lo, hi], probability=p[0], continuity=p[5], status=expected_status,
                                rgb_context=[observed[0], observed[-1]], out_of_scope_rgb_rows=len(expected_support['outside_nominal_consumed_rows'])))
        b = rows[cid, 'broad']; l = rows[cid, 'late']; s = sums[cid]; label = lrows[cid]; prior = old[cid]
        check(label['local_motion_presence'] == label['y'] == 0 and label['mask'] is True and label['actually_viewed_native_frames'] == l['observation']['frame_ids'] and label['native_timestamps_us'] == l['observation']['timestamp_us'], 'direct_label', 'Local hold observation only')
        check(label['old_S_negative_label_read_or_copied'] is False and all(label[k] is None for k in ('full_action_error', 'whole_sign_correctness', 'educational_error', 'source_clause_correctness', 'completion_truth')), 'direct_label', 'Not an erroneous sign')
        check(prior['record']['y'] == 1 and prior['record']['mask'] is True and prior['record']['target'] == 'right_hand_traverses_left_arm_toward_wrist', 'old_saved_result', 'Existing direct positive physical target')
        witness = prior['record']['native_frames']
        check(set(witness).issubset(b['observation']['frame_ids']) and prior['record']['native_timestamps_us'] == ts[witness].tolist(), 'old_saved_result', 'Known sparse movement witnesses within broad scope')
        check(prior['extent'] != b['frames'] and prior['extent'] != l['frames'], 'old_saved_result', 'No completed exact-scope re-evaluation')
        expected_summary = dict(clip_id=cid, person=case['person'], broad_native_frames=b['frames'], existing_positive_witness=witness,
            old_narrow_scope=prior['extent'], old_narrow_probability=prior['probabilities'][0], broad_probability=b['observation']['probability'],
            broad_continuity=b['observation']['continuity_probability'], broad_status=b['observation']['raw_observer_status'], late_native_frames=l['frames'],
            late_direct_local_presence=0, late_probability=l['observation']['probability'], late_continuity=l['observation']['continuity_probability'],
            late_status=l['observation']['raw_observer_status'], late_RGB_context_frames=l['observation']['rgb_context_support']['observed_context_source_frames'],
            late_short_window_limit=label['short_window_limit'])
        check(s == expected_summary, 'summary', 'Exact saved previous/current fields without rescoring')

    counts = dict(selected_fit_records=3, new_scopes=6,
        broad_with_known_motion_reported=sum(r['scope'] == 'broad' and r['status'] == 'DIRECTED_MOTION_OBSERVED' for r in compact),
        locally_still_late_scopes_reported_as_motion=sum(r['scope'] == 'late' and r['status'] == 'DIRECTED_MOTION_OBSERVED' for r in compact))
    check(counts == summary['counts'] and counts['broad_with_known_motion_reported'] == counts['locally_still_late_scopes_reported_as_motion'] == 0, 'summary', '0/3 reported and 0/3 false motion displays')
    check(all_status == {'EXPECTED_MOTION_UNCONFIRMED': 5, 'MOTION_CONTINUITY_UNCONFIRMED': 1}, 'summary', 'Status totals')
    check(all(result[k] is False for k in ('source_labels_edited', 'old_scores_recomputed', 'live_web_changed', 'whole_goal_complete')) and result['training'] == 0, 'execution', 'No claim of training or adoption')
    check(result['previous_S_zeros_not_used_as_hold_truth'] is True, 'direct_label', 'Old interval zeros not adopted as new truth')
    pin(result['checkpoint'], result['checkpoint_sha256'])
    check(execution['pins'][result['checkpoint']] == result['checkpoint_sha256'], 'hash', 'Fixed old checkpoint')
    seconds = result['motion_head_batch_seconds']; load = result['model_load_seconds']
    check(math.isfinite(seconds) and 0 < seconds < finished['at'] - started['at'] and math.isfinite(load) and load > 0, 'timing', 'Saved positive durations within one execution')
    check('excludes MediaPipe/SignRep/RGB extraction and whole video latency' in result['latency_scope'] and round(seconds * 1000, 4) == 7.9395, 'timing', '7.9395ms is six cached-input head calls as one batch, not full latency')
    text_path = U / 'RESULT_KO.txt'; text = text_path.read_text(encoding='utf-8'); PINS[str(text_path)] = sha(text_path)
    check('정상 동작의 일부이고 틀린 수어가 아니다' in text and '새 사용자 정확도가 아니다' in text and '0.007939초' in text and '전체 영상 분석시간 또는 실제 사용자의 대기시간이 아니다' in text, 'report_scope', 'No semantic or latency exaggeration')
    for path, value in list(PINS.items()):
        check(sha(path) == value, 'unchanged_after_audit', path)
    report = dict(status='PASS_SIX_SAVED_SCOPE_RESULTS_NO_ADOPTION_OR_ACCURACY_CLAIM', at=datetime.now(timezone.utc).isoformat(),
        result_sha256=sha(U / 'RESULT.json'), checks=sum(COUNTS.values()), checks_by_category=dict(COUNTS), rows=compact, counts=counts,
        status_counts=dict(all_status), prior_structural_checks=dict(count=61, saved_artifact_pins_verified=True, tests_rerun_by_this_verifier=0),
        timing=dict(motion_head_six_scope_batch_seconds=seconds, motion_head_six_scope_batch_milliseconds=seconds*1000, model_load_seconds=load,
                    whole_video_latency_measured=False, verifier_remeasured_latency=False),
        interpretation=['Existing narrow saved probabilities are reused exactly; the new broad windows contain the same positive sparse witnesses but all return low target0 probability.',
                        'The attempted broad contains-event use is unsupported by this diagnostic. The old target is whole-interval net physical direction, whereas the new desired target is movement occurrence/location inside a broader flow. This is not an original-classifier accuracy score or proof of deterioration.',
                        'The result does not isolate which feature, endpoint, duration or recurrent-state effect caused the low probabilities. Structural outside-RGB context is not a demonstrated cause.',
                        'The three short late windows have direct local absence labels; low scores do not independently validate full-action omission sensitivity.',
                        'ID07 late scope is only three frames and additionally fails the continuity threshold. It is not counted as a confident semantic negative.',
                        'All six RGB supports extend outside the nominal interval. ID07 support is recipe-based; unsaved intermediate window values cannot be reconstructed.',
                        'No automatic new-user localization, new-person accuracy, full-action completion or source-clause correctness is established.'],
        verification_limits=['Saved arithmetic, masks, native axes, source/cache/checkpoint identities and RGB support were independently reconstructed; neural probabilities were not recomputed.',
                             'The earlier61 pure structural tests were checked through their saved results and code hashes, not rerun.',
                             'Same Codex verification, not independent human/KSL expert ground truth.'],
        model_modules_imported=0, model_checkpoints_loaded=0, new_inference=0, new_training=0, new_rgb_extraction=0,
        whole_goal_complete=False, pins={**PINS, str(Path(__file__).resolve()):sha(__file__)})
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], checks=report['checks'], counts=counts, status_counts=dict(all_status)), ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        p = U / 'SAVED_VERIFIER_FIRST_FAILURE.json'
        if not p.exists():
            p.write_text(json.dumps(dict(error=traceback.format_exc(), checks=dict(COUNTS), verifier_sha256=sha(__file__), result_modified=False), ensure_ascii=False, indent=2), encoding='utf-8')
        raise
