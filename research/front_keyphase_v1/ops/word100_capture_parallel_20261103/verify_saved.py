"""Separate saved-array comparison; no video decoder, sensor or model imports.

Root supplies --result after candidate execution completes. Result interface:
rows[{case_id, output_directory, capture_wall_seconds}]. Existing baseline arrays are
read, never rerun. This verifier is not an invocation of the capture candidate.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import traceback
import numpy as np

U = Path(__file__).resolve().parent
CHECKS = []
COMPARISONS = []


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def check(name, value):
    CHECKS.append({'check': name, 'pass': bool(value)})
    return bool(value)


def require(name, value):
    if not check(name, value):
        raise ValueError(name)


def load(p):
    with np.load(p, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def same(a, b):
    if a.dtype.kind in 'fc' and b.dtype.kind in 'fc':
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


def compare_array(name, a, b):
    dtype_ok = check(name + ':dtype', a.dtype == b.dtype)
    shape_ok = check(name + ':shape', a.shape == b.shape)
    exact = dtype_ok and shape_ok and same(a, b)
    check(name + ':all_values_exact_nan_equal', exact)
    report = {'array': name, 'exact': exact, 'baseline_dtype': str(a.dtype),
              'candidate_dtype': str(b.dtype), 'baseline_shape': list(a.shape), 'candidate_shape': list(b.shape)}
    if shape_ok and not exact:
        if a.dtype.kind in 'fc' and b.dtype.kind in 'fc':
            equal = (a == b) | (np.isnan(a) & np.isnan(b))
            finite = np.isfinite(a) & np.isfinite(b)
            report['finite_max_absolute_difference'] = float(np.max(np.abs(a[finite].astype(np.float64)-b[finite].astype(np.float64)))) if finite.any() else None
            report['nan_position_changes'] = int(np.count_nonzero(np.isnan(a) != np.isnan(b)))
            report['infinite_position_or_sign_changes'] = int(np.count_nonzero((np.isinf(a) | np.isinf(b)) & (a != b)))
        else:
            equal = a == b
        report['unequal_cells'] = int(np.count_nonzero(~equal))
        report['first_unequal_flat_indices'] = np.flatnonzero(~equal)[:20].tolist()
    COMPARISONS.append(report)


def verify_profile(case_id, profile, raw, base, capture_seconds):
    """Validate recorded scheduling, not detector execution or inferred speedup."""
    names = ['pose', 'hand', 'face']; n = base['native_frames']
    frames = profile['frames']; totals = profile['totals']; calls = profile['calls']
    require(case_id + ':profile complete native denominator', len(frames) == n
            and profile['status'] == 'CAPTURE_COMPLETE_NOT_YET_BASELINE_VERIFIED')
    check(case_id + ':schedulers joined/clean', profile['schedulers_closed'] is True and profile['cleanup_error'] is None)
    check(case_id + ':same frame concurrency only', profile['frames_parallel'] is False and profile['detectors_same_frame_parallel'] is True)
    check(case_id + ':original point layouts', profile['original_point_layout'] == 75 and profile['face_points'] == 478)
    check(case_id + ':unmodified original algorithms', profile['original_capture_and_extract_bytecode_reused'] is True
          and profile['original_assignment_unchanged'] is True and profile['original_modules_patched'] is False)
    check(case_id + ':profile no performance claim', profile['performance_improvement_claimed'] is False
          and profile['exact_baseline_equality_verified'] is False)
    check(case_id + ':profile source binding', profile['source_sha256'] == base['source_sha256'])
    for p, h in profile['pins'].items():
        require(case_id + ':detector/implementation pin:' + p, sha(p) == h)
    require(case_id + ':profile wall finite/positive', isinstance(profile['wall_seconds'], (int, float))
            and math.isfinite(profile['wall_seconds']) and 0 < profile['wall_seconds'] <= capture_seconds)
    for metric, total in totals.items():
        check(case_id + ':finite timing:' + metric, isinstance(total, (int, float)) and not isinstance(total, bool)
              and math.isfinite(total) and total >= 0)
        check(case_id + ':positive timing call count:' + metric, type(calls.get(metric)) is int and calls[metric] > 0)
    check(case_id + ':timing/calls key sets', set(totals) == set(calls))
    ids = {name: set() for name in names}; duration = {name: [] for name in names}
    previous_end = None; frame_envelopes = []; task_unions = []; frame_overlap = []
    for row, frame in enumerate(frames):
        tag = case_id + ':frame' + str(row)
        check(tag + ':ordinal and original detector time', frame['row'] == row
              and frame['detector_timestamp_ms'] == int(raw['detector_timestamp_ms'][row]))
        check(tag + ':one call per detector', frame['calls'] == dict(pose=1, hand=1, face=1))
        check(tag + ':original image shape', frame['rgb_shape'] == [int(raw['height'][row]), int(raw['width'][row]), 3])
        check(tag + ':original consumption order/join', frame['consumed'] == names and frame['frame_barrier_complete'] is True)
        require(tag + ':three task records', set(frame['tasks']) == set(names))
        intervals = []
        for name in names:
            task = frame['tasks'][name]; start, end, elapsed = task['start_seconds'], task['end_seconds'], task['duration_seconds']
            require(tag + ':' + name + ':finite timing', all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) for v in [start, end, elapsed]))
            check(tag + ':' + name + ':exact duration', end >= start and elapsed == end - start)
            check(tag + ':' + name + ':no task error', task['error'] is None)
            check(tag + ':' + name + ':valid worker identity', type(task['thread_id']) is int and task['thread_id'] > 0)
            ids[name].add(task['thread_id']); duration[name].append(elapsed); intervals.append((start, end))
        first = min(v[0] for v in intervals); last = max(v[1] for v in intervals)
        if previous_end is not None:
            check(tag + ':next frame begins after complete previous barrier', first >= previous_end)
            check(tag + ':strict increasing detector milliseconds', frame['detector_timestamp_ms'] > frames[row - 1]['detector_timestamp_ms'])
        previous_end = last; frame_envelopes.append(last - first)
        # Union is per-frame task occupancy, with overlapping detector time only
        # counted once. It still excludes decoding/assembly and is not full wall.
        ordered = sorted(intervals); left, right = ordered[0]; union = 0.
        for start, end in ordered[1:]:
            if start <= right:
                right = max(right, end)
            else:
                union += right - left; left, right = start, end
        union += right - left
        task_unions.append(union); frame_overlap.append(sum(end-start for start, end in intervals)-union)
    check(case_id + ':one fixed worker per detector', all(len(s) == 1 for s in ids.values()))
    check(case_id + ':three distinct detector workers', len(set.union(*ids.values())) == 3)
    for name, metric in [('pose', 'pose_detect_seconds'), ('hand', 'hand_detect_seconds'), ('face', 'face_process_seconds')]:
        check(case_id + ':exact calls:' + metric, calls.get(metric) == n)
        check(case_id + ':duration sum:' + metric, math.isclose(sum(duration[name]), totals[metric], rel_tol=0, abs_tol=1e-9))
    for metric in ['face_detect_seconds', 'complete_frame_process_seconds', 'raw_process_with_pose_hand_wait_seconds', 'RGB_ndarray_conversion_seconds']:
        check(case_id + ':native call denominator:' + metric, calls.get(metric) == n)
    check(case_id + ':face detector nested inside face process', totals['face_detect_seconds'] <= totals['face_process_seconds'] + 1e-9)
    check(case_id + ':frame envelope within complete-frame processing', sum(frame_envelopes) <= totals['complete_frame_process_seconds'] + 1e-9)
    check(case_id + ':serial frame envelopes within capture wall', sum(frame_envelopes) <= profile['wall_seconds'] + 1e-9)
    check(case_id + ':overlap nonnegative', all(v >= -1e-9 for v in frame_overlap))
    return {'frames_checked': n, 'exact_calls_each_detector': n, 'worker_ids': {k: sorted(v) for k, v in ids.items()},
            'task_duration_sum_seconds_not_wall': sum(sum(v) for v in duration.values()),
            'task_interval_union_seconds_not_wall': sum(task_unions), 'detector_overlap_seconds': sum(frame_overlap),
            'sum_frame_task_envelopes_seconds': sum(frame_envelopes), 'profile_wall_seconds': profile['wall_seconds'],
            'scope_ko': '각 검출기 호출은 겹친다. 얼굴 detect는 얼굴 process 안의 중복 계층이다. 이 값들을 더해 전체 wall latency로 표시하지 않는다.'}


def verify(result_path):
    plan_path = U/'INPUT_PLAN.json'; plan = read(plan_path); result = read(result_path)
    expected = {r['case_id']: r for r in plan['rows']}
    records = result['rows']
    started = read(U/'RUN_STARTED.json'); finished = read(U/'RUN_FINISHED.json')
    require('candidate execution budget marker', started['planned_candidate_captures'] == 2
            and started['baseline_captures'] == started['signrep_forwards'] == started['feedback_head_forwards'] == started['new_training'] == 0)
    require('finished result identity', finished['status'] == 'complete' and finished['result_sha256'] == sha(result_path))
    require('no old execution or semantic work', all(result[k] == 0 for k in ['baseline_reruns', 'semantic_or_SignRep_forwards', 'training', 'new_DTW']))
    require('no causal speed/general completion claim', result['speed_improvement_proven'] is False
            and result['live_changed'] is False and result['whole_goal_complete'] is False)
    require('two prespecified inputs', len(expected) == 2 and plan['maximum_new_candidate_runs'] == 2)
    require('exact candidate case set without duplicates', len(records) == len(expected) and {r['case_id'] for r in records} == set(expected))
    for p, h in plan['pins'].items():
        require('input/baseline pin:' + p, sha(p) == h)
    frozen_path = U/'EXECUTION_FROZEN.json'
    if frozen_path.exists():
        frozen = read(frozen_path)
        for p, h in frozen['pins'].items():
            require('execution pin:' + p, sha(p) == h)
    cases = []
    for r in records:
        case_id = r['case_id']; base = expected[case_id]
        dest = Path(r['output_directory']).resolve(); baseline = Path(base['baseline_directory']).resolve()
        require(case_id + ':new output within AL', dest.is_relative_to(U.resolve()) and dest != baseline and dest.is_dir())
        require(case_id + ':source hash unchanged', sha(base['source']) == base['source_sha256'])
        require(case_id + ':result source binding', r['source_sha256'] == base['source_sha256'])
        request = read(dest/'RUNNER_CAPTURE_REQUEST.json')
        cap_start = read(dest/'CAPTURE_STARTED.json'); cap_end = read(dest/'CAPTURE_FINISHED.json')
        require(case_id + ':one requested source', request['case_id'] == case_id and request['source_sha256'] == base['source_sha256'])
        require(case_id + ':actual capture source', cap_start['source_sha256'] == base['source_sha256']
                and Path(cap_start['source']).resolve() == Path(base['source']).resolve())
        require(case_id + ':actual capture code pin', cap_start['implementation_sha256'] == sha(U/'parallel_capture.py'))
        require(case_id + ':no classifier/training', cap_start['new_training'] == cap_start['new_classifier_inferences'] == 0)
        require(case_id + ':capture finished', cap_end['status'] == 'complete' and r['status'] == 'COMPLETE')
        file_reports = []; independent_equal = []
        for name, oldhash in base['baseline_files'].items():
            old = baseline/name; new = dest/name
            require(case_id + ':baseline immutable:' + name, sha(old) == oldhash)
            require(case_id + ':candidate exists:' + name, new.is_file())
            if name.endswith('.npz'):
                a, b = load(old), load(new)
                check(case_id + ':' + name + ':key set', set(a) == set(b))
                independent_equal.append(set(a) == set(b))
                for k in sorted(set(a) | set(b)):
                    independent_equal.append(k in a and k in b and a[k].dtype == b[k].dtype
                                             and a[k].shape == b[k].shape and same(a[k], b[k]))
                for k in sorted(set(a) & set(b)):
                    compare_array(case_id + ':' + name + ':' + k, a[k], b[k])
            else:
                check(case_id + ':' + name + ':exact JSON', read(old) == read(new))
                independent_equal.append(read(old) == read(new))
            file_reports.append({'file': name, 'baseline_sha256': oldhash, 'candidate_sha256': sha(new),
                                 'container_bytes_identical': oldhash == sha(new)})
        # The candidate must retain native axes, including detector clock, in
        # every stored family. Equality of file sizes alone is not accepted.
        raw = load(dest/'raw.npz'); face = load(dest/'face.npz')
        for name in ['raw.npz', 'baseline.npz', 'observations.npz', 'face.npz']:
            a = load(dest/name)
            check(case_id + ':' + name + ':frozen native frames', np.array_equal(a['frame_id'], base['frame_id']))
            check(case_id + ':' + name + ':frozen PTS', np.array_equal(a['timestamp_us'], base['timestamp_us']))
        check(case_id + ':pose/face detector clock', np.array_equal(raw['detector_timestamp_ms'], face['detector_timestamp_ms']))
        for name, array_file in [('baseline.sidecar.json', 'baseline.npz'), ('observations.sidecar.json', 'observations.npz')]:
            old, new = read(baseline/name), read(dest/name)
            check(case_id + ':' + name + ':baseline artifact hash', old['observation_sha256'] == sha(baseline/array_file))
            check(case_id + ':' + name + ':candidate artifact hash', new['observation_sha256'] == sha(dest/array_file))
            old_semantic = {k: v for k, v in old.items() if k != 'observation_sha256'}
            new_semantic = {k: v for k, v in new.items() if k != 'observation_sha256'}
            check(case_id + ':' + name + ':all semantic fields exact', old_semantic == new_semantic)
            independent_equal.append(old_semantic == new_semantic)
            check(case_id + ':' + name + ':same public source', new['source_sha256'] == base['source_sha256']
                  and Path(new['source_path']).resolve() == Path(base['source']).resolve())
            # observation_sha256 is a produced-artifact identity, not a physical
            # value. Report it separately; every other existing field must stay.
            for k, value in old.items():
                if k == 'observation_sha256':
                    continue
                if k == 'source_path':
                    check(case_id + ':' + name + ':' + k, Path(new[k]).resolve() == Path(value).resolve())
                else:
                    check(case_id + ':' + name + ':' + k, k in new and new[k] == value)
            file_reports.append({'file': name, 'baseline_sha256': sha(baseline/name), 'candidate_sha256': sha(dest/name),
                'artifact_identity_field': 'observation_sha256', 'old_identity': old.get('observation_sha256'),
                'new_identity': new.get('observation_sha256'), 'new_only_fields': sorted(set(new)-set(old))})
        seconds = r.get('capture_wall_seconds')
        require(case_id + ':finite positive capture seconds', isinstance(seconds, (float, int)) and not isinstance(seconds, bool)
                and math.isfinite(seconds) and seconds > 0)
        comparison = read(dest/'ARRAY_COMPARISON.json')
        check(case_id + ':reported checks denominator', len(comparison['rows']) == len(independent_equal))
        check(case_id + ':all reported comparison flags independently reproduced',
              [v['equal'] for v in comparison['rows']] == independent_equal)
        unequal = sum(not v for v in independent_equal)
        check(case_id + ':reported unequal count', comparison['unequal_checks'] == r['unequal_checks'] == unequal)
        check(case_id + ':reported preservation', comparison['all_equal'] == r['exact_observation_preserved']
              == cap_end['exact_observation_preserved'] == all(independent_equal))
        profile = read(dest/'CAPTURE_TIMING.json')
        check(case_id + ':saved profile unmodified', profile == r['profile'])
        profile_check = verify_profile(case_id, profile, raw, base, seconds)
        check(case_id + ':native frame denominator', r['native_frames'] == base['native_frames'])
        check(case_id + ':uncontrolled timing disclosed', r['timing_comparison_is_controlled'] is False)
        cases.append({'case_id': case_id, 'registered_id': base['registered_id'], 'word_id': base['word_id'],
            'source_sha256': base['source_sha256'], 'native_frames': base['native_frames'], 'output_directory': str(dest),
            'files': file_reports, 'candidate_capture_seconds': seconds,
            'all_arrays_and_semantics_exact': all(independent_equal), 'unequal_comparisons': unequal,
            'profile': profile,
            'independent_profile_arithmetic': profile_check,
            'saved_baseline_capture_seconds': base['baseline_capture_seconds'],
            'timing_comparability': 'Historical _capture had concurrent RGB/image_information work; do not attribute timing difference solely to sensor parallelization.',
            'physical_or_educational_accuracy': None})
    check('overall all_exact arithmetic', result['all_exact'] == all(c['all_arrays_and_semantics_exact'] for c in cases))
    # Inputs must still be unchanged after all saved-array reads.
    for p, h in plan['pins'].items():
        require('after pin:' + p, sha(p) == h)
    return {'input_plan_sha256': sha(plan_path), 'result_path': str(result_path), 'result_sha256': sha(result_path),
            'cases': cases, 'input_pins_checked_before_and_after': len(plan['pins']),
            'criterion': 'All keys/dtypes/shapes/values and same NaN positions exact; no tolerance relaxation.',
            'limits_ko': ['공개 입력 두 개의 구현 동등성 검산이다. 수어 정확도·모든 사람/가림/교차의 일반화 검증이 아니다.',
                         '기존 baseline은 저장값만 읽었으며 다시 실행하지 않았다.',
                         '후보 실행 횟수는 root 실행 로그·동결 marker로 별도 확인한다. 이 도구는 센서나 모델을 실행할 수 없다.',
                         '다른 동시부하가 있는 과거 baseline과의 시간차는 병렬화만의 인과효과가 아니다.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', default=str(U/'RESULT.json'), help='Root completed candidate JSON with rows/case_id/output_directory/capture_wall_seconds')
    args = parser.parse_args(); result_path = Path(args.result).resolve()
    if not result_path.is_relative_to(U.resolve()):
        raise ValueError('Result must be the fixed AL public candidate result.')
    output = U/'SAVED_CAPTURE_VERIFIED.json'
    if output.exists():
        raise FileExistsError('Preserve first verification; no overwrite.')
    report = {'status': 'RUNNING', 'verifier_sha256': sha(__file__), 'new_sensor_runs': 0,
              'new_model_loads': 0, 'new_model_forwards': 0, 'new_training': 0, 'new_video_decodes': 0,
              'baseline_reruns': 0}
    try:
        report.update(verify(result_path))
        report['status'] = 'PASS_EXACT_SAVED_CAPTURE_ARRAYS' if all(c['pass'] for c in CHECKS) else 'FAIL_CAPTURE_OUTPUT_DIFFERENCE'
    except BaseException:
        report['status'] = 'FAIL_SAVED_VERIFICATION'; report['exception'] = traceback.format_exc()
        raise
    finally:
        report.update(check_count=len(CHECKS), failed_count=sum(not c['pass'] for c in CHECKS), checks=CHECKS,
                      array_comparisons=COMPARISONS)
        with output.open('x', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps({k: report[k] for k in ['status', 'check_count', 'failed_count']}))


if __name__ == '__main__':
    main()
