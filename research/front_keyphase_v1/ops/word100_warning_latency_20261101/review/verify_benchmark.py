"""Read-only independent arithmetic/hash/array audit; never run adapters or GPUs."""
from pathlib import Path
import hashlib
import itertools
import json
import math
import traceback
import numpy as np

U = Path(__file__).resolve().parents[1]
checks = 0
failures = []
pins = {}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def check(ok, name):
    global checks
    checks += 1
    if not bool(ok):
        failures.append(name)


def pin(p, expected=None):
    p = str(Path(p).resolve())
    h = sha(p)
    check(expected is None or h == expected, 'pin:' + p)
    check(p not in pins or pins[p] == h, 'stable_pin:' + p)
    pins[p] = h
    return h


def read(p):
    pin(p)
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def comparison(p, q):
    pin(p)
    pin(q)
    records = {}
    with np.load(p, allow_pickle=False) as aa, np.load(q, allow_pickle=False) as bb:
        check(aa.files == bb.files, 'array_schema:' + str(p) + ':' + str(q))
        for k in aa.files:
            a, b = aa[k], bb[k]
            shape = a.shape == b.shape
            dtype = a.dtype == b.dtype
            numeric = a.dtype.kind in 'fciu'
            finite = bool(shape and np.array_equal(np.isfinite(a), np.isfinite(b))) if numeric else None
            equal = bool(shape and dtype and (np.array_equal(a, b, equal_nan=True) if numeric else np.array_equal(a, b)))
            delta = None
            changed = None
            if shape:
                values_equal = a == b
                if a.dtype.kind in 'fc':
                    values_equal |= np.isnan(a) & np.isnan(b)
                changed = int((~values_equal).sum())
                if a.dtype.kind in 'fiu':
                    known = np.isfinite(a) & np.isfinite(b)
                    if known.any():
                        delta = float(np.max(np.abs(a[known].astype(float) - b[known].astype(float))))
            records[k] = dict(shape=list(a.shape), dtype=str(a.dtype), dtype_equal=dtype,
                equal=equal, max_abs=delta, finite_mask_equal=finite, changed_cells=changed)
    return records


def run():
    frozen = read(U / 'FROZEN.json')
    bench = read(U / 'BENCHMARK.json')
    gate = read(U / 'ENCODER_GATE.json')
    started = read(U / 'BENCHMARK_STARTED.json')
    recovery = read(U / 'BENCHMARK_RECOVERY.json')
    pin(U / 'candidate_adapter.py', started['candidate_sha256'])
    pin(U / 'FROZEN.json', started['frozen_sha256'])
    pin(U / 'benchmark_first_version.py', recovery['old_script_sha256'])
    pin(U / 'benchmark.py', recovery['script_sha256'])
    pin(U / 'encoder_gate.py')
    pin(U / 'BENCHMARK_FIRST_FAILURE.txt')
    pin(U / 'benchmark.log')
    pin(U / 'benchmark_recovery.log')
    for p, h in frozen['source_pins'].items():
        pin(p, h)
    check(bench['source_pins'] == frozen['source_pins'], 'protected_source_map')
    check(frozen['order'] == ['serial', 'candidate', 'candidate', 'serial'], 'frozen_ABBA')
    check(len(bench['runs']) == 8, 'all8_timed_runs_retained')
    initial_log = (U / 'benchmark.log').read_text(encoding='utf-8-sig')
    recovery_log = (U / 'benchmark_recovery.log').read_text(encoding='utf-8-sig')
    check(initial_log.count('START clip1_') == 4 and recovery_log.count('RESUME_SAVED clip1_') == 4 and
          'START clip1_' not in recovery_log and recovery_log.count('START clip2_') == 4, 'recovery_does_not_repeat_completed4_inferences')
    check('TypeError' in (U / 'BENCHMARK_FIRST_FAILURE.txt').read_text(encoding='utf8'), 'first_dtype_failure_preserved')
    warm = read(U / 'WARMUP.json')
    recovery_warm = read(U / 'RECOVERY_WARMUP.json')
    source_results = []
    all_comparisons = []
    all_exact = True
    for ci, source in enumerate(frozen['public_clips'], 1):
        pin(source['source'], source['source_sha256'])
        runs = [r for r in bench['runs'] if r['clip'] == source['clip_id']]
        check([r['condition'] for r in runs] == frozen['order'], f'clip{ci}:run_order')
        check([r['id'] for r in runs] == [f'clip{ci}_{i}_{kind}' for i, kind in enumerate(frozen['order'], 1)], f'clip{ci}:unique_run_ids')
        folders = []
        complete = []
        for r in runs:
            path = Path(r['output'])
            folders.append(path)
            c = read(path / 'COMPLETE.json')
            complete.append(c)
            check(c == r['metadata'], r['id'] + ':metadata_exact')
            check(read(U / 'bench' / (r['id'] + '.json')) == r, r['id'] + ':run_ledger_exact')
            check(c['source_sha256'] == source['source_sha256'], r['id'] + ':frozen_source_sha')
            check(c['status'] == 'COMPLETE_RESEARCH_ESTIMATES' and c['frames'] == source['frames'] and
                  not c['cached_predictions_used'] and not c['training'], r['id'] + ':full_fresh_public_output')
            check(c['device']['precision'] == 'f16' and c['device']['actual'] == ['GPU.0'] and
                  c['runtime_reuse']['signrep_initialized_before_video'] and c['runtime_reuse']['heads_initialized_before_video'],
                  r['id'] + ':resident_same_device_precision')
            check(r['wall_seconds'] > 0 and c['latency']['total_adapter_seconds'] <= r['wall_seconds'], r['id'] + ':valid_whole_wall_time')
            for name, h in c['artifacts'].items():
                pin(path / name, h)
            check(read(path / 'CALIBRATION.json') == c['calibration'], r['id'] + ':calibration_metadata_exact')
            with np.load(path / 'analysis.npz', allow_pickle=False) as a, np.load(path / 'rgb_windows.npz', allow_pickle=False) as z:
                ids, ts = a['frame_id'], a['timestamp_us']
                expected_starts = list(range(0, len(ids) - 15, 2))
                if expected_starts[-1] != len(ids) - 16:
                    expected_starts.append(len(ids) - 16)
                check(z['starts'].tolist() == expected_starts and z['features'].shape == (len(expected_starts), 768), r['id'] + ':all16_stride2_tail_windows')
                check(np.array_equal(z['center_timestamp_us'], (ts[z['starts']] + ts[z['starts'] + 15]) / 2), r['id'] + ':window_center_PTS')
                centers = [i for i in range(6, len(ids)-6) if np.all(np.diff(ids[i-6:i+7]) == 1)]
                check(a['new_four_center_indices'].tolist() == centers and np.array_equal(a['posterior_available'], np.isfinite(a['probabilities'])),
                      r['id'] + ':head_boundary_no_padding')
            sidecar = read(path / 'observations.sidecar.json')
            check(sidecar['width'] == 1920 and sidecar['height'] == 1080 and sidecar['source_sha256'] == source['source_sha256'], r['id'] + ':source_resolution_unchanged')
        for i, j in itertools.combinations(range(4), 2):
            records = {}
            for name in ['analysis.npz', 'rgb_windows.npz', 'observations.npz', 'face.npz', 'raw.npz', 'assignment.npz', 'baseline.npz']:
                rec = comparison(folders[i] / name, folders[j] / name)
                records[name] = rec
                exact = all(v['equal'] for v in rec.values())
                all_exact &= exact
                # An actual mismatch is recorded as a failed equivalence gate,
                # not concealed or confused with an arithmetic verifier error.
                if i == 0 and name in ['analysis.npz', 'rgb_windows.npz', 'observations.npz', 'face.npz', 'raw.npz']:
                    producer = next(r['arrays'] for r in bench['equality'] if r['clip'] == source['clip_id'] and
                                    r['comparison'] == f'1_vs_{j+1}' and r['file'] == name)
                    check(all(producer[k] == {f: v[f] for f in ['shape', 'equal', 'max_abs', 'finite_mask_equal']}
                              for k, v in rec.items()), f'clip{ci}:{i+1}:{j+1}:{name}:saved_equality_arithmetic')
            cal_equal = read(folders[i] / 'CALIBRATION.json') == read(folders[j] / 'CALIBRATION.json')
            trace_equal = read(folders[i] / 'assignment_trace.json') == read(folders[j] / 'assignment_trace.json')
            sides = [read(folders[k] / 'observations.sidecar.json') for k in [i, j]]
            quality_equal = sides[0]['quality'] == sides[1]['quality']
            all_exact &= cal_equal and trace_equal and quality_equal
            all_comparisons.append(dict(clip=source['clip_id'], comparison=f'{i+1}_vs_{j+1}',
                arrays=records, full_calibration_equal=cal_equal, assignment_trace_equal=trace_equal,
                observation_quality_equal=quality_equal))
        serial = [r['wall_seconds'] for r in runs if r['condition'] == 'serial']
        candidate = [r['wall_seconds'] for r in runs if r['condition'] == 'candidate']
        sm, cm = sum(serial) / 2, sum(candidate) / 2
        source_results.append(dict(clip=source['clip_id'], serial_wall_seconds=serial, candidate_wall_seconds=candidate,
            serial_mean_seconds=sm, candidate_mean_seconds=cm, mean_seconds_saved=sm-cm,
            reduction_fraction=(sm-cm)/sm, candidate_max_below_serial_min=max(candidate) < min(serial),
            independent_public_people=1, warmup_session='initial' if ci == 1 else 'recovery',
            comparison_scope='Within this clip ABBA only; two repetitions per condition.'))
    encoder = []
    check(len(gate['results']) == 2, 'two_fixed_geometry_encoder_tests')
    for ci, source in enumerate(frozen['public_clips'], 1):
        row = gate['results'][ci-1]
        check(row['clip'] == source['clip_id'] and row['native_frames'] == source['frames'], f'encoder{ci}:identity')
        result = comparison(U / f'gate_{ci}_serial.npz', U / f'gate_{ci}_candidate.npz')
        equal = all(v['equal'] for v in result.values())
        traces = row['records']
        inputs_equal = traces[0]['input_sha256'] == traces[1]['input_sha256']
        check([r['condition'] for r in traces] == ['serial', 'candidate'] and len(traces[0]['input_sha256']) == row['windows'], f'encoder{ci}:all_input_tensor_hashes')
        check(row['passed'] == (equal and inputs_equal), f'encoder{ci}:gate_arithmetic')
        check(all(row['equality'][k] == {f: v[f] for f in ['shape', 'equal', 'max_abs', 'finite_mask_equal']} for k, v in result.items()), f'encoder{ci}:stored_output_equality')
        encoder.append(dict(clip=source['clip_id'], windows=row['windows'], all_input_tensor_hashes_equal=inputs_equal,
            all_output_arrays_bit_exact=equal, arrays=result, input_tensor_hashes=traces[0]['input_sha256']))
    check(gate['status'] == ('PASS' if all(v['all_input_tensor_hashes_equal'] and v['all_output_arrays_bit_exact'] for v in encoder) else 'FAIL'), 'encoder_final_status')
    for p, h in list(pins.items()):
        check(sha(p) == h, 'end_pin:' + p)
    return dict(source_pins=pins, all_fresh_run_output_arrays_and_calibration_exact=all_exact,
        timing_by_clip=source_results, full_run_comparisons=all_comparisons, fixed_geometry_encoder=encoder,
        warmup_seconds=[warm['seconds'], recovery_warm['seconds']],
        recovery=dict(first_failure='numpy string equal_nan verification error after first four completed runs',
            old_code_preserved=True, completed_clip1_inference_repeated=False, fresh_clip2_runs=4,
            first_clip_source_identity_checked_post_execution=True, original4_results_unchanged=True),
        execution=dict(model_load=0, GPU=0, public_decode=0, forward=0, DP=0, training=0),
        limitations=['Two public clips and two repetitions per condition, not a broad latency distribution or deployment benchmark.',
            'First serial run can include module import/cold costs; all eight measured runs remain in the report.',
            'Recovery creates a new process/warmup before clip2; comparisons are within each clip only.',
            'Whole-adapter wall times exclude upload, downstream comparison/guard and UI; prior15.72/31.52s were different completed end-to-end observations.',
            'Exact stored arrays on these runs establish no information loss here; they do not prove all future detector executions deterministic or full sign correctness.'])


if __name__ == '__main__':
    out = U / 'review/BENCHMARK_VERIFICATION.json'
    if out.exists():
        raise FileExistsError('Preserve first audit')
    value = {}
    try:
        value = run()
    except BaseException:
        p = U / 'review/BENCHMARK_VERIFICATION_FIRST_FAILURE.txt'
        if not p.exists():
            p.write_text(traceback.format_exc(), encoding='utf8')
        failures.append(traceback.format_exc())
    value.update(status='PASS_INDEPENDENT_STORED_BENCHMARK_AUDIT' if not failures else 'FAIL_PRESERVED',
        checks=checks, failure_count=len(failures), failures=failures, verifier_sha256=sha(__file__))
    with out.open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: value.get(k) for k in ['status', 'checks', 'failure_count', 'all_fresh_run_output_arrays_and_calibration_exact', 'timing_by_clip']}))
    raise SystemExit(0 if not failures else 1)
