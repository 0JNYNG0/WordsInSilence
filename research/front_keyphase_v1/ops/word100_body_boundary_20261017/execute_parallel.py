"""Two-process executor for unchanged, frozen per-clip preparation.

This file is versioned execution plumbing only: it does not change input rules,
detector settings, the frozen row order, or the per-clip preparation function.
--limit counts newly dispatched rows, excluding hash-verified existing records.
--seconds closes dispatch; already running rows finish and keep their evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
sys.path.insert(0, str(U))
import prepare

legacy = prepare.legacy
EXECUTION = U / 'PARALLEL_EXECUTION_CONTEXT.json'
WORKER_LOG = None


def utc():
    return datetime.now(timezone.utc).isoformat()


def archives_unchanged(context):
    for path, expected in context['archive_stat'].items():
        actual = Path(path).stat()
        assert [actual.st_size, actual.st_mtime_ns] == expected, path


def freeze():
    legacy.authorize()
    context = legacy.read(prepare.CONTEXT)
    legacy.check_pins(context['pins'])
    archives_unchanged(context)
    pins = {str(p.resolve()): legacy.sha256_file(p) for p in
            [Path(__file__), prepare.CONTEXT, U / 'prepare.py', U / 'AUTHORIZATION.json']}
    legacy.write_new(EXECUTION, {
        'schema': 'word100_body_parallel_execution_v1', 'status': 'FROZEN',
        'created_utc': utc(), 'workers': 2, 'start_method': 'spawn',
        'pins': pins, 'preparation_context_sha256': legacy.sha256_file(prepare.CONTEXT),
        'per_row_function': 'prepare.legacy.prepare_row, after prepare.configure()',
        'output_root': str(prepare.DATA.resolve()),
        'rule_changes': 0, 'training': 0,
        'failure_policy': 'stop new dispatch on first observed failure; await in-flight evidence; no automatic retry/delete',
        'limit_semantics': 'newly dispatched rows; reused records do not count',
        'seconds_semantics': 'dispatch deadline; in-flight rows may finish after it',
        'source_policy': 'frozen source pins plus archive stat and selected ZIP-entry CRC/source SHA only; never whole ZIP hashing',
    })
    print(legacy.read(EXECUTION)['status'], flush=True)


def memory_rss():
    try:
        import psutil
        p = psutil.Process()
        return {'rss_bytes': p.memory_info().rss, 'threads': p.num_threads()}
    except ImportError:
        return {'rss_bytes': None, 'threads': None, 'reason': 'psutil unavailable'}


def init_worker(context_hash, execution_hash, run_id):
    global WORKER_LOG
    prepare.configure()
    assert legacy.sha256_file(prepare.CONTEXT) == context_hash
    assert legacy.sha256_file(EXECUTION) == execution_hash
    legacy.check_pins(legacy.read(EXECUTION)['pins'])
    directory = U / 'parallel_logs' / run_id
    directory.mkdir(parents=True, exist_ok=True)
    WORKER_LOG = directory / f'worker_{os.getpid()}.log'
    # Each worker owns one log, including native MediaPipe stderr.
    fd = os.open(str(WORKER_LOG), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    finally:
        os.close(fd)
    sys.stdout.reconfigure(encoding='utf-8', write_through=True)
    sys.stderr.reconfigure(encoding='utf-8', write_through=True)


def do_row(row, context_hash):
    start = time.perf_counter()
    cpu = os.times()
    cid = row['clip_id']
    result = {'clip_id': cid, 'pid': os.getpid(), 'started_utc': utc(),
              'worker_log': str(WORKER_LOG), 'memory_start': memory_rss()}
    try:
        assert legacy.sha256_file(prepare.CONTEXT) == context_hash
        assert legacy.shutil.disk_usage(U).free >= 10_000_000_000
        contract = legacy.read(legacy.LAB / 'config/contract.json')
        rec, reused = legacy.prepare_row(row, context_hash, contract)
        result.update(status='ROW_COMPLETE', reused=reused, frames=rec['frames'],
                      method=rec['calibration']['method'], visual_flags=len(rec['visual_flags']),
                      new_detector_conditions=sum(not rec[k].get('reused', False)
                          for k in ('native_capture', 'sensor_capture')))
    except Exception as exc:
        result.update(status='ROW_FAILED', error_type=type(exc).__name__,
                      message=str(exc), traceback=traceback.format_exc())
        # Unique per-row first evidence; shared PREP_FIRST_FAILURE is owned by parent.
        path = U / 'parallel_failures' / (cid + '.json')
        if not path.exists():
            legacy.write_new(path, result)
    result.update(ended_utc=utc(), seconds=time.perf_counter() - start,
                  cpu_seconds=(os.times().user + os.times().system) - (cpu.user + cpu.system),
                  memory_end=memory_rss())
    return result


def bounded_dispatch(pool, rows, worker, worker_args, limit, seconds, on_result, clock=time.perf_counter):
    """At most two submitted rows; never enqueue the whole corpus."""
    start = clock()
    cursor = 0
    in_flight = {}
    results = []
    failed = False
    while True:
        while not failed and len(in_flight) < 2 and cursor < len(rows) and cursor < limit and clock() - start < seconds:
            row = rows[cursor]
            cursor += 1
            future = pool.submit(worker, row, *worker_args)
            in_flight[future] = row
        if not in_flight:
            break
        completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
        # All completed tasks are inspected before any new dispatch.
        for future in completed:
            row = in_flight.pop(future)
            try:
                result = future.result()
            except Exception as exc:
                result = {'status': 'ROW_FAILED', 'clip_id': row['clip_id'],
                          'error_type': type(exc).__name__, 'message': str(exc),
                          'traceback': traceback.format_exc(), 'worker_crash': True}
            results.append(result)
            if result['status'] != 'ROW_COMPLETE':
                failed = True
            on_result(result)
    return results, cursor, failed


def run(stage, limit, seconds):
    assert limit > 0 and seconds > 0
    prepare.configure()
    legacy.authorize()
    execution = legacy.read(EXECUTION)
    assert execution['workers'] == 2 and execution['start_method'] == 'spawn'
    legacy.check_pins(execution['pins'])
    context = legacy.read(prepare.CONTEXT)
    context_hash = legacy.sha256_file(prepare.CONTEXT)
    assert context_hash == execution['preparation_context_sha256']
    assert not (U / 'PREP_FIRST_FAILURE.json').exists(), 'Review existing first failure before any retry'
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + f'_{os.getpid()}'
    lock = U / 'PREP_RUNNING.json'
    legacy.write_new(lock, {'pid': os.getpid(), 'mode': 'two_process', 'run_id': run_id,
                            'context_sha256': context_hash, 'stage': stage})
    start = time.perf_counter()
    results = []
    reused = 0
    active_cid = None
    selected = []
    result = None

    def record_result(item):
        legacy.event(stage='parallel_row_complete' if item['status']=='ROW_COMPLETE' else 'parallel_row_failed',
                     run_id=run_id, row=item)
        legacy.write_new(U / 'parallel_results' / run_id / (item['clip_id'] + '.json'), item)
        if item['status'] != 'ROW_COMPLETE' and not (U / 'PREP_FIRST_FAILURE.json').exists():
            legacy.write_new(U / 'PREP_FIRST_FAILURE.json', {
                'status': 'FAILED_NO_TRAINING', 'mode': 'two_process', 'run_id': run_id,
                'context_sha256': context_hash, 'first_observed_failure': item,
                'clip_id': item['clip_id'], 'training': 0})
        legacy.write_status({'status': 'FAILED_AWAITING_INFLIGHT_EVIDENCE' if (U / 'PREP_FIRST_FAILURE.json').exists()
                            else 'PREPARING_TWO_PROCESSES', 'run_id': run_id, 'last_row': item,
                            'training': 0, 'context_sha256': context_hash})

    try:
        legacy.check_pins(context['pins'])
        archives_unchanged(context)
        selected = [r for r in context['ordered_rows'] if stage == 'all' or r['clip_id'] in context['early_ids']]
        pending = []
        contract = legacy.read(legacy.LAB / 'config/contract.json')
        # Validate all prior records before starting any new row. Do not skip partial evidence.
        for row in selected:
            active_cid = row['clip_id']
            if (prepare.DATA / 'records' / (active_cid + '.json')).exists():
                _, was_reused = legacy.prepare_row(row, context_hash, contract)
                assert was_reused
                reused += 1
            else:
                assert not (prepare.DATA / 'clips' / active_cid).exists(), 'Partial evidence requires review: ' + active_cid
                pending.append(row)
        assert legacy.shutil.disk_usage(U).free >= 10_000_000_000
        legacy.event(stage='parallel_start', run_id=run_id, workers=2, selected=len(selected),
                     verified_reused=reused, pending=len(pending), dispatch_limit=limit, dispatch_seconds=seconds)
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=init_worker,
                                 initargs=(context_hash, legacy.sha256_file(EXECUTION), run_id)) as pool:
            results, dispatched, failed = bounded_dispatch(pool, pending, do_row, (context_hash,),
                                                           limit, seconds, record_result)
        # The pool is now closed and all native log writes have completed.
        legacy.check_pins(context['pins'])
        legacy.check_pins(execution['pins'])
        archives_unchanged(context)
        for item in results:
            if item['status'] == 'ROW_COMPLETE':
                r = next(r for r in selected if r['clip_id'] == item['clip_id'])
                _, was_reused = legacy.prepare_row(r, context_hash, contract)
                assert was_reused
        completed = sum((prepare.DATA / 'records' / (r['clip_id'] + '.json')).exists() for r in selected)
        logs = U / 'parallel_logs' / run_id
        result = {'status': 'FAILED_NO_TRAINING' if failed else
                  'PREPARED_PENDING_INDEPENDENT_AND_VISUAL_REVIEW' if completed == len(selected) else 'CHUNK_COMPLETE',
                  'run_id': run_id, 'stage': stage, 'workers': 2, 'selected': len(selected),
                  'verified_reused': reused, 'dispatched_new': dispatched,
                  'new_records': sum(r['status'] == 'ROW_COMPLETE' for r in results),
                  'completed': completed, 'seconds': time.perf_counter() - start,
                  'dispatch_limit': limit, 'dispatch_seconds': seconds, 'rows': results,
                  'source_pins_verified_start_and_end': True,
                  'execution_context_sha256': legacy.sha256_file(EXECUTION),
                  'context_sha256': context_hash, 'training': 0,
                  'worker_log_pins': {str(p): legacy.sha256_file(p) for p in logs.glob('*.log')}}
        legacy.write_new(U / 'parallel_results' / run_id / 'RESULT.json', result)
        legacy.write_status(result)
        print(legacy.json.dumps({k: v for k, v in result.items() if k not in ('rows', 'worker_log_pins')}), flush=True)
    except Exception as exc:
        failure = {'status': 'FAILED_NO_TRAINING', 'mode': 'two_process', 'run_id': run_id,
                   'clip_id': active_cid, 'error_type': type(exc).__name__, 'message': str(exc),
                   'traceback': traceback.format_exc(), 'context_sha256': context_hash, 'training': 0}
        if not (U / 'PREP_FIRST_FAILURE.json').exists():
            legacy.write_new(U / 'PREP_FIRST_FAILURE.json', failure)
        legacy.write_new(U / 'parallel_results' / run_id / 'CONTROLLER_FAILURE.json', failure)
        legacy.write_status(failure)
        raise
    finally:
        if lock.exists() and legacy.read(lock)['pid'] == os.getpid():
            lock.unlink()
    if result['status'] == 'FAILED_NO_TRAINING':
        raise SystemExit(1)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'run'])
    parser.add_argument('--stage', choices=['early', 'all'], default='all')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--seconds', type=float, default=1800)
    args = parser.parse_args()
    prepare.configure()
    if args.action == 'freeze':
        freeze()
    else:
        run(args.stage, args.limit, args.seconds)
