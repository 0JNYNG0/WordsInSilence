"""Narrow recovery of cached20 source-path resolution; never runs a detector.

Frozen prep.py, PREP_CONTEXT and first failure remain untouched. The experiment
context stays the same because the source bytes, calibration and input contract
are unchanged; the separately pinned recovery context records the path patch.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
import prep

U = Path(__file__).resolve().parent
RC = U / 'PREP_RECOVERY_CONTEXT.json'
RESULT = U / 'PREP_RECOVERY_RESULT.json'
FIRST = U / 'PREP_FIRST_FAILURE.json'


def deny_capture(*args, **kwargs):
    raise RuntimeError('Recovery forbids detector execution; existing paired cache only')


def freeze():
    prep.authorize()
    assert not RC.exists(), 'Reuse frozen recovery context'
    context = prep.read(prep.CONTEXT)
    prep.check_pins(context['pins'])
    failure = prep.read(FIRST)
    ch = prep.sha256_file(prep.CONTEXT)
    assert failure['context_sha256'] == ch
    assert failure['type'] == 'AssertionError'
    assert failure['message'] == 'Cache-only mode cannot extract source or run detector'
    prior_path = prep.STABLE / 'PLAN.json'
    prior = prep.read(prior_path)
    items = {r['clip_id']: r for r in prior['items']}
    prior_pins = {str(Path(p).resolve()): h for p, h in prior['pins'].items()}
    patches = {}
    selected = []
    pins = {str(p.resolve()): prep.sha256_file(p) for p in
            [Path(__file__), prep.CONTEXT, FIRST, prior_path, U / 'AUTHORIZATION.json']}
    for row in context['ordered_rows']:
        if not (row.get('reuse_native') and row.get('reuse_sensor')):
            continue
        cid = row['clip_id']; selected.append(cid)
        assert cid in items and row['role'] == 'train_candidate'
        if row.get('retained_video'):
            continue
        video = Path(items[cid]['video_path']).resolve()
        old = prep.read(row['legacy_record_path'])
        expected = old['source_sha256']
        assert expected == items[cid]['source_sha256']
        assert prior_pins[str(video)] == expected, 'Prior immutable plan must pin source video'
        assert prep.sha256_file(video) == expected
        assert video.stat().st_size == row['video_entries'][0]['bytes']
        pins[str(video)] = expected
        patches[cid] = {'field': 'retained_video', 'old': None, 'new': str(video),
                        'source_sha256': expected, 'source_bytes': video.stat().st_size}
    assert len(selected) == 20 and len(patches) == 8
    failed_dir = (prep.DATA / 'clips' / failure['clip_id']).resolve()
    assert failed_dir.parent == (prep.DATA / 'clips').resolve()
    assert failed_dir.exists() and not list(failed_dir.iterdir()), 'Only known empty partial directory may be moved'
    assert not (prep.DATA / 'records' / (failure['clip_id'] + '.json')).exists()
    completed = {}
    for cid in selected:
        p = prep.DATA / 'records' / (cid + '.json')
        if p.exists():
            rec = prep.read(p)
            assert rec['context_sha256'] == ch
            prep.check_pins(rec['outputs'])
            completed[str(p.resolve())] = prep.sha256_file(p)
    assert len(completed) == 12
    pins.update(completed)
    prep.write_new(RC, {
        'schema': 'word100_cache_path_recovery_v1', 'status': 'FROZEN_BEFORE_RECOVERY',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'cause': 'Cache-only source lookup considered only web reference directory; eight immutable prior-study videos are elsewhere.',
        'experiment_context_sha256': ch, 'first_failure_sha256': prep.sha256_file(FIRST),
        'selected_clip_ids': selected, 'patches': patches, 'pins': pins,
        'preserved_existing_records': completed, 'empty_partial_dir': str(failed_dir),
        'evidence_destination': str((U / 'recovery' / 'evidence' / (failure['clip_id'] + '_empty_partial')).resolve()),
        'source_change': False, 'condition_change': False, 'calibration_change': False,
        'new_detector_conditions': 0, 'training': 0,
        'provenance': 'New records retain identical experiment context; this recovery context and result map their hashes to the source-path correction.'})
    print(json.dumps({'status': 'RECOVERY_FROZEN', 'path_patches': 8, 'existing_records': 12}))


def run():
    prep.authorize()
    rc = prep.read(RC)
    prep.check_pins(rc['pins'])
    context = prep.read(prep.CONTEXT)
    prep.check_pins(context['pins'])
    if RESULT.exists():
        result = prep.read(RESULT)
        prep.check_pins(result['record_pins'])
        print('Verified completed recovery reused; no extraction.', flush=True)
        return
    lock = U / 'PREP_RUNNING.json'
    prep.write_new(lock, {'pid': os.getpid(), 'recovery_context_sha256': prep.sha256_file(RC)})
    try:
        old = Path(rc['empty_partial_dir']).resolve()
        target = Path(rc['evidence_destination']).resolve()
        assert old.parent == (prep.DATA / 'clips').resolve()
        assert target.parent == (U / 'recovery' / 'evidence').resolve()
        if old.exists():
            assert not list(old.iterdir()), 'Partial directory changed; preserve and stop'
            assert not target.exists(), 'Existing evidence must never be overwritten'
            target.parent.mkdir(parents=True, exist_ok=True)
            old.rename(target)
        else:
            assert target.is_dir() and not list(target.iterdir()), 'Expected preserved empty evidence'
        prep.capture = deny_capture
        contract = prep.read(prep.LAB / 'config/contract.json')
        rows = {r['clip_id']: r for r in context['ordered_rows']}
        results = []; record_pins = {}
        for cid in rc['selected_clip_ids']:
            row = copy.deepcopy(rows[cid])
            if cid in rc['patches']:
                patch = rc['patches'][cid]
                assert row['retained_video'] is None
                row['retained_video'] = patch['new']
                assert prep.sha256_file(row['retained_video']) == patch['source_sha256']
            record, reused = prep.prepare_row(row, rc['experiment_context_sha256'], contract, cache_only=True)
            p = prep.DATA / 'records' / (cid + '.json')
            record_pins[str(p.resolve())] = prep.sha256_file(p)
            results.append({'clip_id': cid, 'existing_record_reused': reused,
                            'source_path_patched': cid in rc['patches'],
                            'record_path': str(p.resolve()), 'record_sha256': prep.sha256_file(p)})
            prep.event(stage='cache_recovery_row', clip_id=cid, reused_record=reused)
        prep.check_pins(rc['pins'])
        result = {'status': 'CACHED20_PREPARED_PENDING_INDEPENDENT_AND_VISUAL_CHECK',
                  'experiment_context_sha256': rc['experiment_context_sha256'],
                  'recovery_context_sha256': prep.sha256_file(RC), 'first_failure_preserved': True,
                  'rows': results, 'record_pins': record_pins, 'records': len(results),
                  'new_detector_conditions': 0, 'new_source_extraction': 0, 'training': 0}
        prep.write_new(RESULT, result)
        print(json.dumps({'status': result['status'], 'records': len(results), 'detector': 0}, ensure_ascii=False))
    except Exception as exc:
        fp = U / 'PREP_RECOVERY_FIRST_FAILURE.json'
        if not fp.exists():
            prep.write_new(fp, {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        if lock.exists() and prep.read(lock)['pid'] == os.getpid(): lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('freeze', 'run'))
    args = parser.parse_args()
    globals()[args.action]()
