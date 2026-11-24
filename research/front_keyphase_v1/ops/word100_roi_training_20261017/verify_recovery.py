"""Read-only independent recovery and failed-calibration evidence verification."""
from pathlib import Path
import json
import sys
import zipfile

import numpy as np

sys.dont_write_bytecode = True
from verify_inputs import U, read, sha, load, write_new


def calibration_windows(o):
    """Enumerate candidate windows directly; do not call stable_rest."""
    ts = o['timestamp_us']; xy = o['xy_px'].astype(np.float64)
    stop = int(np.searchsorted(ts, ts[0] + 1000000, side='right'))
    required = [11, 12, 13, 14, 33, 54]
    eligible = (o['observed'][:, required] & o['usable'][:, required]
                & np.isfinite(xy[:, required]).all(2)).all(1)
    totals = {'duration_eligible': 0, 'complete_observation': 0,
              'arms_down_geometry': 0, 'stable': 0}
    successes = []
    for end in range(7, stop):
        for start in range(end - 6):
            if ts[end] - ts[start] < 300000:
                continue
            totals['duration_eligible'] += 1
            if not eligible[start:end + 1].all() or (np.diff(ts[start:end + 1]) > 100000).any():
                continue
            totals['complete_observation'] += 1
            w = xy[start:end + 1]
            scale = float(np.median(np.linalg.norm(w[:, 11] - w[:, 12], axis=1)))
            if not np.isfinite(scale) or scale <= 0:
                continue
            geometry = []
            for shoulder, elbow, wrist in [(11, 13, 33), (12, 14, 54)]:
                upper_drop = w[:, elbow, 1] - w[:, shoulder, 1]
                forearm_drop = w[:, wrist, 1] - w[:, elbow, 1]
                horizontal = np.abs(w[:, wrist, 0] - w[:, elbow, 0])
                geometry.append(((upper_drop >= .2 * scale) & (forearm_drop >= .2 * scale)
                                 & (horizontal <= .5 * forearm_drop)).all())
            if not all(geometry):
                continue
            totals['arms_down_geometry'] += 1
            pts = w[:, [11, 12, 33, 54]]
            if np.linalg.norm(pts - np.median(pts, axis=0), axis=2).max() > .05 * scale:
                continue
            totals['stable'] += 1
            successes.append([start, end])
    return {'prefix_frames': stop, 'window_counts': totals, 'qualifying_windows': successes}


def main():
    context = read(U / 'PREP_CONTEXT.json'); rc = read(U / 'PREP_RECOVERY_CONTEXT.json')
    result = read(U / 'PREP_RECOVERY_RESULT.json'); first = read(U / 'PREP_FIRST_FAILURE.json')
    assert sha(U / 'PREP_CONTEXT.json') == rc['experiment_context_sha256'] == result['experiment_context_sha256']
    assert sha(U / 'PREP_FIRST_FAILURE.json') == rc['first_failure_sha256']
    assert sha(U / 'PREP_RECOVERY_CONTEXT.json') == result['recovery_context_sha256']
    assert first['type'] == 'AssertionError' and first['message'] == 'Cache-only mode cannot extract source or run detector'
    pins = {}
    for mapping in [rc['pins'], rc['preserved_existing_records'], result['record_pins']]:
        for p, h in mapping.items():
            assert sha(p) == h, p
            pins[p] = h
    assert len(rc['preserved_existing_records']) == 12 and len(rc['patches']) == 8
    assert len(result['rows']) == result['records'] == 20
    assert sum(r['existing_record_reused'] for r in result['rows']) == 12
    assert sum(r['source_path_patched'] for r in result['rows']) == 8
    assert result['new_detector_conditions'] == result['new_source_extraction'] == result['training'] == 0
    byid = {r['clip_id']: r for r in context['ordered_rows']}
    prior = read(U.with_name('word100_stable_rest_roi_20261017') / 'PLAN.json')
    prior_rows = {r['clip_id']: r for r in prior['items']}
    assert set(rc['selected_clip_ids']) == set(prior_rows)
    sources = []
    archives = {}
    try:
        for cid, patch in rc['patches'].items():
            row = byid[cid]; old = read(row['legacy_record_path']); p = Path(patch['new'])
            assert patch['field'] == 'retained_video' and patch['old'] is None and row['retained_video'] is None
            assert p.resolve() == Path(prior_rows[cid]['video_path']).resolve()
            assert sha(p) == patch['source_sha256'] == old['source_sha256'] == prior_rows[cid]['source_sha256']
            source = row['video_entries'][0]
            assert p.stat().st_size == patch['source_bytes'] == source['bytes']
            if source['archive'] not in archives:
                archives[source['archive']] = zipfile.ZipFile(source['archive'])
            member = archives[source['archive']].getinfo(source['member'])
            assert member.CRC == source['crc32'] and member.file_size == source['bytes']
            record = read(U / 'data/records' / (cid + '.json'))
            assert record['source']['sha256'] == patch['source_sha256']
            assert record['native_capture']['reused'] and record['sensor_capture']['reused']
            sources.append({'clip_id': cid, 'sha256': patch['source_sha256'], 'bytes': source['bytes'], 'archive_crc_verified': True})
    finally:
        for archive in archives.values():
            archive.close()
    evidence = Path(rc['evidence_destination'])
    assert evidence.is_dir() and not list(evidence.iterdir())
    probe = read(U / 'CALIBRATION_PROBE_RESULT.json'); scope = read(U / 'CALIBRATION_PROBE_SCOPE.json')
    assert scope['context_sha256'] == sha(U / 'PREP_CONTEXT.json')
    assert sha(U / 'probe_calibration.py') == scope['code_sha256']
    assert {r['clip_id'] for r in probe['rows']} == set(scope['clip_ids'])
    probe_rows = []
    for r in probe['rows']:
        assert r['role'] == byid[r['clip_id']]['role'] == 'train_candidate'
        for p, h in r['capture']['pins'].items():
            assert sha(p) == h, p
        assert sha(r['source_path']) == r['source_sha256'] == read(byid[r['clip_id']]['legacy_record_path'])['source_sha256']
        ob = load(r['capture']['candidate']); actual = calibration_windows(ob)
        assert not actual['qualifying_windows'] and r['calibration']['status'] == 'NOT_CALIBRATED'
        probe_rows.append({'clip_id': r['clip_id'], 'label': r['label'], **actual,
                           'independent_calibration_status': 'NOT_CALIBRATED'})
    report = {'status': 'PASS_NARROW_RECOVERY_AND_TWO_CALIBRATION_FAILURES_CONFIRMED',
              'recovery_pins_verified': len(pins), 'existing_records_preserved': 12,
              'source_path_only_patches': 8, 'new_records': 8, 'source_bytes_unchanged': True,
              'frozen_context_unchanged': True, 'first_failure_preserved': True,
              'empty_partial_directory_preserved': str(evidence), 'sources': sources,
              'calibration_probe_independent': probe_rows,
              'new_detector_runs_by_verifier': 0, 'classifier_runs_by_verifier': 0,
              'training_runs_by_verifier': 0, 'training_readiness': 'NOT_PASSED_CALIBRATION_BLOCKER',
              'note': 'The two failures disprove universal readiness; they do not estimate all-dataset failure prevalence.'}
    write_new(U / 'RECOVERY_AND_PROBE_INDEPENDENT.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'sources'}, ensure_ascii=False))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
