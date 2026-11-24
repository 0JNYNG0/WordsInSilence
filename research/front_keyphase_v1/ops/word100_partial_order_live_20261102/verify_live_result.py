"""Independent saved-artifact arithmetic; no model imports, inference or POST."""
from pathlib import Path
import json
import hashlib
import datetime
import traceback
import numpy as np

AA = Path(__file__).resolve().parent
JOB = 'dff00320fc0b497db638b0408244db20'
D = AA / 'public_runtime_runs' / JOB / 'analysis'
Z = AA.parent / 'word100_partial_order_feedback_20261102'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    destination = AA / 'LIVE_RESULT_VERIFIED.json'
    assert not destination.exists()
    checks = []

    def check(name, value):
        assert bool(value), name
        checks.append(name)

    frozen = read(AA / 'EXECUTION_FROZEN_INITIAL.json')
    current_frozen = read(AA / 'EXECUTION_FROZEN.json')
    check('original181pin_count', len(frozen['pins']) == 181)
    amendment = read(AA / 'UI_DISPLAY_AMENDMENT.json')
    check('amendment_did_not_replace_execution_freeze',
          not amendment['execution_frozen']['modified'] and
          sha(AA / 'EXECUTION_FROZEN_INITIAL.json') == amendment['execution_frozen']['sha256'])
    changed = []
    for path, expected in frozen['pins'].items():
        current = sha(path)
        if current != expected:
            changed.append(dict(path=path, frozen_sha256=expected, current_sha256=current))
        else:
            checks.append('frozen_file_exact:' + path)
    js = str((AA / 'demo.js').resolve())
    check('only_declared_UI_file_changed', len(changed) == 1 and changed[0]['path'] == js)
    check('original_UI_bytes_preserved',
          sha(amendment['previous_demo_js']['path']) == amendment['previous_demo_js']['sha256'] == frozen['pins'][js])
    check('current_UI_bytes_match_amendment',
          sha(amendment['current_demo_js']['path']) == amendment['current_demo_js']['sha256'] == changed[0]['current_sha256'])
    lineage = current_frozen['display_only_amendment']
    check('amended_freeze_lineage',
          lineage['initial_sha256'] == sha(AA / 'EXECUTION_FROZEN_INITIAL.json') and
          lineage['amendment_sha256'] == sha(AA / 'UI_DISPLAY_AMENDMENT.json') and
          lineage['fresh_job_executed_under_initial_freeze'] == JOB and
          lineage['new_inference'] == 0 and lineage['live_backend_process_unchanged'] is True)
    expected_current = json.loads(json.dumps(frozen))
    expected_current['pins'][js] = changed[0]['current_sha256']
    expected_current['display_only_amendment'] = lineage
    check('current_freeze_only_one_UI_pin_plus_lineage', current_frozen == expected_current)
    check('UI_amendment_only_optional_timing_labels',
          amendment['scope'] == 'DISPLAY_ONLY_OPTIONAL_TIMING_FIELDS' and
          amendment['product_files_changed'] == ['demo.js'] and
          amendment['new_model_inference'] == 0 and amendment['training'] == 0)

    base = read(D / 'SOURCE_FEEDBACK.json')
    order = read(D / 'ORDER_FEEDBACK.json')
    live = read(D / 'LIVE_FEEDBACK.json')
    contract = read(D / 'ORDER_INPUT_CONTRACT.json')
    intent = read(D.parent / 'INTENT.json')
    pairs = read(D / 'ORDER_PAIRS.json')
    context = read(Z / 'RUNTIME_CONTEXT.json')
    browser = read(AA / 'LIVE_ORDER_BROWSER_VERIFIED.json')
    a = dict(np.load(D / 'analysis.npz', allow_pickle=False))
    q = dict(np.load(D / 'order_arrays.npz', allow_pickle=False))
    for key, value in base.items():
        check('base_field_exact:' + key, order[key] == value and live[key] == value)
    for key, value in order.items():
        check('order_field_exact_in_live:' + key, live[key] == value)
    check('only_additive_order_fields', set(order) - set(base) == {'partial_order_feedback', 'partial_order_runtime'})
    check('only_additive_live_timing_fields', set(live) - set(order) == {'combined_feedback_latency_seconds', 'combined_latency_scope'})
    check('public_word_job_binding',
          contract['job_id'] == JOB and contract['word_id'] == base['word_id'] == 'WORD0739' and
          intent['word'] == 'WORD0739' and intent['public_only'] is True)
    check('input_path_hash_binding', Path(contract['input']).resolve() == (D / 'analysis.npz').resolve() and sha(contract['input']) == contract['input_sha256'])
    check('base_path_hash_binding', Path(contract['base']).resolve() == (D / 'SOURCE_FEEDBACK.json').resolve() and sha(contract['base']) == contract['base_sha256'])
    check('actual_public_video_hash_binding', sha(intent['source']) == base['source_sha256'] == contract['source_sha256'])
    check('no_training_tuning_selection', not contract['training'] and not contract['tuning'] and not contract['candidate_selection'] and not live['training'] and not live['tuning'] and live['new_training'] == 0)
    ids = a['frame_id']; ts = a['timestamp_us']; times = ts / 1e6; available = np.asarray(a['image_available'], bool)
    n = len(ids)
    check('native123_frames', n == 123 == base['frames'] and ids.dtype.kind in 'iu' and ts.dtype.kind in 'iu')
    check('strict_native_axis', np.all(np.diff(ids) > 0) and np.all(np.diff(ts) > 0))
    check('native_axis_hash', hashlib.sha256(ids.tobytes() + ts.tobytes()).hexdigest() == contract['native_axis_sha256'])
    for key in ['frame_id', 'timestamp_us', 'image_available']:
        check('order_axis_exact:' + key, np.array_equal(q[key], a[key]))
    check('overlay_times_exact', np.array_equal(base['overlay']['times'], times))
    check('overlay_frame_count', len(base['overlay']['points']) == n)
    for name in ['raw.npz', 'observations.npz', 'baseline.npz']:
        with np.load(D / name, allow_pickle=False) as sensor:
            for key in ['frame_id', 'timestamp_us']:
                check(name + ':' + key, np.array_equal(sensor[key], a[key]))
    for key, width in [('physical', 9), ('role', 2), ('occurrence', 8), ('current', 10), ('flow', 6)]:
        check('saved_shape_finite:' + key, q[key].shape == (n, width) and np.isfinite(q[key]).all())
    for key in ['physical', 'role', 'current']:
        check('probability_simplex:' + key, np.all((q[key] >= 0) & (q[key] <= 1)) and np.allclose(q[key].sum(1), 1, atol=1e-6, rtol=1e-6))
    check('occurrence_probability_range', np.all((q['occurrence'] >= 0) & (q['occurrence'] <= 1)))
    missing = np.flatnonzero(~available).tolist()
    check('fresh_image_gap_count_zero', missing == [] and base['evidence_gaps'] == [])
    check('joint_gap_inventory_consistent', np.flatnonzero(~np.asarray(a['mask'], bool)[:, :61].any(1)).tolist() == [] and base['joint_evidence_gaps'] == [])

    path = np.asarray(base['path'], int); m = int(path[-1, 1]) + 1
    check('path_native_axes', base['path_axes'] == ['query_native_index', 'learned_canonical_index'])
    check('path_endpoints', path[0].tolist() == [0, 0] and path[-1].tolist() == [n - 1, m - 1])
    step = np.diff(path, axis=0)
    check('path_monotone_coverage', np.all((step >= 0) & (step <= 1)) and np.all(step.sum(1) > 0) and np.array_equal(np.unique(path[:, 0]), np.arange(n)) and np.array_equal(np.unique(path[:, 1]), np.arange(m)))
    phase = np.asarray([path[path[:, 0] == i, 1].mean() / max(m - 1, 1) for i in range(n)], np.float32)
    check('saved_flow_reuses_base_path', np.array_equal(phase, q['flow'][:, 0]) and np.array_equal(q['flow'][:, 5] > 0, available))
    check('one_learned_prototype_reference0', base['learned_prototype_comparisons'] == 1 and base['reference_video_reads'] == 0 and base['individual_reference_comparisons'] == 0 and base['runtime_individual_reference_video_comparisons'] == 0 and live['partial_order_runtime']['additional_DTW'] == 0 and live['partial_order_runtime']['individual_reference_video_comparisons'] == 0)

    graph = context['graphs']['WORD0739']
    check('exact_reference_queries', {(p['first'], p['second']) for p in pairs} == {tuple(x) for x in graph['precedences']} and len(pairs) == 3)
    payload = live['partial_order_feedback']
    items = {(x['first']['id'], x['second']['id']): x for x in payload['items']}
    arithmetic = []
    for p in pairs:
        first, second = p['first'], p['second']
        # Independent float64 arithmetic over saved float32 outputs. This is
        # a reproducibility check, not model evaluation against a new label.
        weights = q['physical'][:, :8].astype(np.float64) * q['occurrence'].astype(np.float64)
        wa = weights[:, first] * available; wb = weights[:, second] * available
        fw = float((wb * np.r_[0., np.cumsum(wa)[:-1]]).sum())
        rv = float((wa * np.r_[0., np.cumsum(wb)[:-1]]).sum())
        probability = (fw + 1e-8) / (fw + rv + 2e-8)
        check(f'pair{first}_{second}_soft_probability_arithmetic', abs(probability - p['forward_probability']) < 2e-6)
        check(f'pair{first}_{second}_mass_arithmetic', np.isclose(fw + rv, p['pair_mass'], atol=1e-4, rtol=2e-6))
        af = np.flatnonzero(available & (weights[:, first] >= .8)).tolist()
        bf = np.flatnonzero(available & (weights[:, second] >= .8)).tolist()
        pa = np.flatnonzero(available & (q['physical'][:, first] >= .8)).tolist()
        pb = np.flatnonzero(available & (q['physical'][:, second] >= .8)).tolist()
        overlap = sorted(set(af) & set(bf)); direction = None; gaps = []
        status = 'PAIR_OCCURRENCE_UNCONFIRMED'
        if af and bf:
            gaps = [i for i in range(min(af[0], bf[0]), max(af[-1], bf[-1]) + 1) if not available[i]]
            if gaps: status = 'PAIR_OBSERVATION_GAP'
            elif overlap: status = 'PAIR_OVERLAPPING_EVIDENCE'
            elif af[-1] < bf[0]:
                direction = 1; status = 'OBSERVED_A_BEFORE_B' if probability >= .8 else 'PAIR_READOUT_DISAGREEMENT'
            elif bf[-1] < af[0]:
                direction = 0; status = 'OBSERVED_B_BEFORE_A' if probability <= .2 else 'PAIR_READOUT_DISAGREEMENT'
            else: status = 'PAIR_INTERLEAVED_EVIDENCE'
        raw_direction = None
        if pa and pb:
            if pa[-1] < pb[0]: raw_direction = 1
            elif pb[-1] < pa[0]: raw_direction = 0
            elif not set(pa) & set(pb): raw_direction = 'interleaved'
        learned_status = status
        if status in ['OBSERVED_A_BEFORE_B', 'OBSERVED_B_BEFORE_A'] and raw_direction in [0, 1] and raw_direction != direction:
            status = 'RETAINED_PHYSICAL_ORDER_DISAGREEMENT'
        check(f'pair{first}_{second}_status_contract', p['status'] == status and p['learned_pair_status'] == learned_status and p['physical_direction'] == raw_direction and p['observed_direction'] == direction)
        check(f'pair{first}_{second}_gaps_overlap_axis', p['gaps_inside_evidence'] == gaps and p['overlap_frames'] == overlap and p['unobserved_frames'] == missing and p['comparison_frames'] == n)
        for prefix, frames in [('first_evidence', af), ('second_evidence', bf), ('raw_physical_first', pa), ('raw_physical_second', pb)]:
            check(f'pair{first}_{second}_{prefix}_indices_times', p[prefix + '_frames'] == frames and np.array_equal(p[prefix + '_seconds'], times[frames]))
        check(f'pair{first}_{second}_native_frame_ids', p['first_native_frame_ids'] == ids[af].tolist() and p['second_native_frame_ids'] == ids[bf].tolist() and p['unobserved_native_frame_ids'] == ids[missing].tolist())
        item = items[first, second]
        check(f'pair{first}_{second}_display_preserves_evidence', item['observation_status'] == p['status'] and item['first_evidence']['input_frames'] == af and item['second_evidence']['input_frames'] == bf and item['first_evidence']['native_seconds'] == p['first_evidence_seconds'] and item['second_evidence']['native_seconds'] == p['second_evidence_seconds'])
        check(f'pair{first}_{second}_raw_interleaving_preserved', item['retained_physical_context']['first']['input_frames'] == pa and item['retained_physical_context']['second']['input_frames'] == pb and item['retained_physical_context']['role_qualified'] is False)
        for obj in [p, item]:
            check(f'pair{first}_{second}_not_completion_' + ('raw' if obj is p else 'display'), obj['whole_sign_correctness'] is None and obj['complete_action_verified'] is False and obj['missing_action_verified'] is False)
        arithmetic.append(dict(first=first, second=second, probability_saved=p['forward_probability'], probability_recomputed=probability, absolute_probability_error=abs(probability - p['forward_probability']), mass_absolute_error=abs(fw + rv - p['pair_mass']), status=status, raw_physical_direction=raw_direction, first_frames=af, second_frames=bf))
    check('whole_word_unknown', base['whole_sign_correctness'] is None and base['all_semantic_requirements_checked'] is False and payload['whole_sign_correctness'] is None and payload['education_error'] is None)
    check('completion_omission_repeat_unverified', payload['completion_unknown'] is True and payload['complete_action_verified'] is False and payload['missing_action_verified'] is False and payload['repeat_count'] is None and payload['repeat_count_verified'] is False and payload['repeated_sequence_verified'] is False)
    check('no_semantic_clause_certification', payload['source_clause_verified'] is False and payload['full_span_semantic_verified'] is False and payload['evaluation_truth_used'] is False)
    check('display_times_not_boundaries', all(not x[side]['span_is_exact_action_boundary'] for x in payload['items'] for side in ['first_evidence', 'second_evidence']))

    runtime = live['partial_order_runtime']; latency = live['latency']; combined = live['combined_feedback_latency_seconds']
    fresh = browser['fresh']; wall = fresh['wall_seconds_from_button']
    check('runtime_job_and_hashes', runtime['job_id'] == JOB and runtime['input_sha256'] == contract['input_sha256'] and runtime['base_sha256'] == contract['base_sha256'])
    checkpoint = Path('W:/WordsInSilence_derived/word100_direct_occurrence_order_20261102/runs/tied_readout/model.pt')
    check('tied_checkpoint_identity', runtime['model_sha256'] == frozen['pins'][str(checkpoint.resolve())])
    check('positive_finite_recorded_latency', all(np.isfinite(v) and v >= 0 for v in latency.values()) and np.isfinite(combined) and runtime['head_seconds'] > 0)
    check('latency_nesting', combined >= latency['total_analysis_seconds'] + runtime['head_seconds'] and latency['total_analysis_seconds'] >= latency['total_seconds'] >= latency['mediapipe_seconds'] and latency['total_seconds'] >= latency['rgb_seconds'])
    check('combined_time_scope', live['combined_latency_scope'] == 'server analysis after request begins, not browser upload or model boot')
    check('browser_saved_fresh_matches_server', fresh['job_id'] == JOB and fresh['frames'] == n and fresh['source_sha256'] == contract['source_sha256'] and fresh['combined_feedback_latency_seconds'] == combined and fresh['partial_order_runtime'] == runtime and fresh['base_latency'] == latency and wall >= combined)
    check('one_reported_public_POST_no_camera_upload', browser['actual_fresh_public_posts_across_attempts'] == 1 and browser['actual_upload_posts'] == 0 and browser['camera_calls'] == 0 and browser['private_footage_access'] is False)
    check('only_one_AA_public_job_directory', sorted(p.name for p in (AA / 'public_runtime_runs').iterdir() if p.is_dir()) == [JOB])
    observed_files = [D / name for name in ['SOURCE_FEEDBACK.json', 'ORDER_FEEDBACK.json', 'LIVE_FEEDBACK.json', 'ORDER_PAIRS.json', 'ORDER_INPUT_CONTRACT.json', 'analysis.npz', 'order_arrays.npz']] + [AA / 'EXECUTION_FROZEN_INITIAL.json', AA / 'EXECUTION_FROZEN.json', AA / 'UI_DISPLAY_AMENDMENT.json', AA / 'LIVE_ORDER_BROWSER_VERIFIED.json', Path(__file__)]
    report = dict(
        status='PASS_SAVED_RUNTIME_ARITHMETIC_WITH_DECLARED_UI_ONLY_AMENDMENT',
        at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), reviewer='Codex /root/retention_direction_review',
        verification='Separate saved-data/code arithmetic; not independent human KSL truth, new accuracy evaluation or model inference.',
        public_job=JOB, word='WORD0739', frames=n, native_axis_exact=True,
        source_base_fields_preserved=len(base), order_fields_preserved_in_live=len(order),
        saved_pair_arithmetic=arithmetic, observed_pair_count=len(pairs), image_unavailable_frames=missing,
        original_freeze=dict(total_pins=181, unchanged_pins=180, declared_exceptions=changed,
            exception_record=str(AA / 'UI_DISPLAY_AMENDMENT.json'), original_freeze_unchanged=True,
            note='Only optional timing labels changed in demo.js. Original bytes are preserved; model/input/output/worker/server pins remain exact.'),
        current_freeze=dict(pins_verified=181, only_change_from_initial='One demo.js hash and explicit amendment lineage',
            fresh_job_executed_under='EXECUTION_FROZEN_INITIAL.json'),
        learned_prototype_comparisons=1, additional_DTW=0, individual_reference_video_comparisons=0,
        execution_count_scope='Stored runtime declarations, preserved code path and one-job/browser records; not a new instrumented execution trace.',
        latency=dict(combined_server_seconds=combined, source_pipeline_seconds=latency['total_analysis_seconds'],
            order_component_seconds=runtime['head_seconds'], added_server_elapsed_seconds=combined-latency['total_analysis_seconds'],
            remaining_roundtrip_serialization_seconds=combined-latency['total_analysis_seconds']-runtime['head_seconds'],
            browser_wall_seconds=wall, browser_minus_server_seconds=wall-combined,
            capture_seconds=latency['mediapipe_seconds'], rgb_seconds=latency['rgb_seconds'],
            extraction_parallel=True, rule='MediaPipe/capture and RGB run concurrently (V/demo_adapter.py); do not sum their overlapping durations. One laptop run is not a latency distribution.',
            model_load_seconds=None, model_load_not_measured_for_this_job=True),
        correctness_scope=dict(whole_sign_correctness=None, complete_action_verified=False,
            missing_action_verified=False, repeat_count_verified=False, independent_human_annotation=False),
        checks=checks, check_count=len(checks), failed_checks=[],
        first_attempt=dict(path=str(AA / 'LIVE_RESULT_VERIFICATION_FIRST_FAILURE.txt'),
            reason='Parent accepted the declared UI-only amendment while verification was starting. The first attempt expected the original hash at EXECUTION_FROZEN.json. Verification now checks the exact original backup and permits only the declared single UI pin plus amendment lineage.'),
        new_model_inference=0, new_training=0, new_POST=0, old_benchmark_rescoring=False,
        next_priority=dict(primary='Important-action sufficiency/omission and repetition evidence, before speed optimization.',
            rationale='The saved cut WORD1133 can retain an observed pair after annotated A witnesses are removed; WORD1567 has no distinct order query and repetition remains unknown. In this fresh run the order component costs only0.032 seconds of8.774 server seconds, so accelerating it would not close the main feedback gap.',
            bounded_next='Prepare one generic FIT supervision/validation slice distinguishing directly observed main-action evidence from truncated residual evidence, while marking genuine configuration/movement coexistence and unknown completion explicitly. Reuse existing FIT source views and transformations; add only directly needed FIT judgments. Do not infer completion from span/frame coverage or label all physical cuts linguistic errors. Required repetition needs separate actual supervision before a count verdict.',
            speed_secondary='Keep the current measured wait visible. If latency work follows, profile capture/RGB concurrency rather than tune this32ms head or rerun completed comparisons.'),
        source_hashes={str(p.resolve()):sha(p) for p in observed_files}, whole_goal_complete=False)
    with destination.open('x', encoding='utf8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], checks=len(checks), frames=n, pairs=len(pairs), frozen_exact=180,
                         declared_UI_exception=1, combined_seconds=combined, head_seconds=runtime['head_seconds']), ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        failure = AA / 'LIVE_RESULT_VERIFICATION_FIRST_FAILURE.txt'
        if not failure.exists():
            failure.write_text(traceback.format_exc(), encoding='utf8')
        raise
