"""Versioned 100-word input preparation. No model loading or training.

freeze: pin the approved proposal, implementation and existing source evidence.
run: bounded sequential rows, keeping first failure and never overwriting it.
The caller must inspect readiness/visual flags before launching any training.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import zipfile
import numpy as np

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
LAB = U.parents[1]
ROOT = LAB.parents[1]
DATA = U / 'data'
OLD = U.with_name('word100_training_20261013')
STABLE = U.with_name('word100_stable_rest_roi_20261017')
FULL = U.with_name('word100_fullframe_roi_pair_20261016')
CAP = U.with_name('word100_hand_assignment_20261016')
JOINT = U.with_name('word100_hand_assignment_joint_20261016')
PROPOSAL = U.with_name('word100_roi_training_plan_20261017') / 'PROPOSED_EXPERIMENT.json'
LEGACY = Path('W:/WordsInSilence/data/AIHub_103/word100_inputs_20261013')
REFERENCES = Path('W:/WordsInSilence/data/AIHub_103/word100_web_references_20261013')
CONTEXT = U / 'PREP_CONTEXT.json'
sys.path[:0] = [str(LAB / 'src'), str(ROOT / 'research/sign_model_pilot_v1/src')]
from wis_lab.core.observation import Observation, sha256_file
from wis_lab.service.neural import _channels
from pilot.web_input import to_native_like, active_span
from pilot.data import build_channels, PilotDataError


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_module('roi_training_frozen_base', FULL / 'run_pair.py')
roi = load_module('roi_training_frozen_roi', FULL / 'roi_adapter.py')
cal = load_module('roi_training_frozen_cal', STABLE / 'stable_rest.py')


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write('\n')


def write_status(value):
    path = U / 'PREP_STATUS.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf-8')
    tmp.replace(path)


def event(**value):
    value.update(pid=os.getpid(), time_utc=datetime.now(timezone.utc).isoformat())
    with (U / 'PREP_PROGRESS.jsonl').open('a', encoding='utf-8') as out:
        out.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(value, ensure_ascii=False), flush=True)


def check_pins(pins):
    for p, expected in pins.items():
        if sha256_file(p) != expected:
            raise RuntimeError('Pinned input changed: ' + p)


class ReadinessError(RuntimeError):
    pass


def authorize():
    auth = read(U / 'AUTHORIZATION.json')
    assert auth.get('approved') is True and auth.get('status') == 'APPROVED', 'Approved preparation authorization required'
    return auth


def ordered_manifest(rows, prior_ids):
    train = [r for r in rows if r['role'] == 'train_candidate']
    refs = {min((r for r in train if r['word_code'] == code), key=lambda r: (r['signer_id'], r['clip_id']))['clip_id']
            for code in {r['word_code'] for r in train}}
    assert len(refs) == 100
    ordered = sorted(rows, key=lambda r: (0 if r['clip_id'] in refs else 1 if r['clip_id'] in prior_ids else
                                          2 if r['role'] == 'train_candidate' else 3,
                                          r['word_code'], r['signer_id'], r['clip_id']))
    return ordered, sorted(refs)


def freeze():
    authorize()
    assert not CONTEXT.exists(), 'Reuse existing frozen PREP_CONTEXT; no overwrite'
    rows = read(OLD / 'MANIFEST.json')
    assert Counter(r['role'] for r in rows) == {'train_candidate': 899, 'development_only': 300}
    assert len({r['clip_id'] for r in rows}) == 1199
    prior = read(STABLE / 'PLAN.json')
    prior_items = {r['clip_id']: r for r in prior['items']}
    sensor = {r['clip_id']: r for r in read(STABLE / 'CAPTURE_RESULT.json')['rows']}
    ordered, refs = ordered_manifest(rows, set(prior_items))
    fixed = [Path(__file__), U / 'prep_test.py', U / 'AUTHORIZATION.json', PROPOSAL,
             OLD / 'MANIFEST.json', OLD / 'INPUT_CHECK.json', OLD / 'FROZEN_EXPERIMENT.json',
             STABLE / 'PLAN.json', STABLE / 'CAPTURE_RESULT.json', STABLE / 'stable_rest.py',
             FULL / 'run_pair.py', FULL / 'roi_adapter.py', CAP / 'capture_raw.py', JOINT / 'assignment.py',
             LAB / 'config/contract.json', LAB / 'src/wis_lab/core/observation.py',
             LAB / 'src/wis_lab/capture/video_source.py', LAB / 'src/wis_lab/capture/landmarker.py',
             LAB / 'src/wis_lab/service/neural.py', ROOT / 'research/sign_model_pilot_v1/src/pilot/data.py',
             ROOT / 'research/sign_model_pilot_v1/src/pilot/web_input.py',
             ROOT / 'Tools/models/hand_landmarker.task', ROOT / 'Tools/models/pose_landmarker_lite.task']
    pins = {str(p.resolve()): sha256_file(p) for p in fixed}
    # Verify the previous candidate evidence before copying its reuse identities.
    check_pins(prior['pins'])
    checked = read(OLD / 'INPUT_CHECK.json')
    archives = {}
    for row in ordered:
        cid = row['clip_id']
        assert row['signer_id'] in (range(1, 10) if row['role'] == 'train_candidate' else (11, 12, 13))
        rp = LEGACY / 'records' / (cid + '.json')
        assert sha256_file(rp) == checked['record_hashes'][cid]
        rec = read(rp)
        assert rec['role'] == row['role'] and rec['signer_id'] == row['signer_id']
        pins[str(rp.resolve())] = sha256_file(rp)
        for key, hk in [('observation', 'observation_sha256'), ('channel_file', 'channel_sha256')]:
            assert rec[key] and sha256_file(rec[key]) == rec[hk]
            pins[str(Path(rec[key]).resolve())] = rec[hk]
        side = Path(rec['observation']).with_suffix('.sidecar.json')
        pins[str(side.resolve())] = sha256_file(side)
        row['legacy_record_path'] = str(rp.resolve())
        e = row['video_entries'][0]
        ap = Path(e['archive']); st = ap.stat()
        archives[str(ap)] = [st.st_size, st.st_mtime_ns]
        vp = REFERENCES / (cid + '.mp4')
        row['retained_video'] = str(vp) if vp.exists() else None
        if vp.exists():
            assert sha256_file(vp) == rec['source_sha256']
            pins[str(vp.resolve())] = rec['source_sha256']
        if cid in prior_items:
            it, sr = prior_items[cid], sensor[cid]
            check_pins(sr['outputs'])
            native = it['cached']['native']
            row['reuse_native'] = {'baseline': str(Path(native['raw_dir']) / 'baseline.npz'),
                                   'raw': str(Path(native['raw_dir']) / 'raw.npz'),
                                   'candidate': native['observation'], 'reused': True}
            row['reuse_sensor'] = {'baseline': str(Path(sr['raw']).with_name('baseline.npz')),
                                   'raw': sr['raw'], 'candidate': sr['observation'],
                                   'box': sr['sensor_box'], 'reused': True}
            for cache in [row['reuse_native'], row['reuse_sensor']]:
                cache['pins'] = {}
                for key in ('baseline', 'raw', 'candidate'):
                    p = Path(cache[key]); cache['pins'][str(p.resolve())] = sha256_file(p)
                    if key != 'raw':
                        s = p.with_suffix('.sidecar.json'); cache['pins'][str(s.resolve())] = sha256_file(s)
                pins.update(cache['pins'])
    write_new(CONTEXT, {'schema': 'word100_roi_preparation_v1', 'status': 'FROZEN_BEFORE_EXTRACTION',
                       'proposal_path': str(PROPOSAL.resolve()), 'manifest_path': str((OLD / 'MANIFEST.json').resolve()),
                       'ordered_rows': ordered, 'fixed_reference_ids': refs, 'prior20_ids': sorted(prior_items),
                       'pins': pins, 'archive_stat': archives, 'output_root': str(DATA.resolve()),
                       'limit_meaning': 'newly completed records this invocation, including reuse of detector caches; already committed records verified and skipped without consuming limit',
                       'training': 0, 'rest_wrist_filter': 'OFF', 'user_data': 0})
    event(stage='frozen', rows=1199, fixed_references=100, cache_native=20, cache_sensor=20)


def validate_channels(x):
    assert x.ndim == 2 and x.shape[1] == 280 and np.isfinite(x).all()
    assert x[0, -1] == 0 and np.isin(x[:, 186:279], (0., 1.)).all()
    bm = x[:, 186:237].astype(bool); hm = x[:, 237:279].astype(bool)
    assert not x[:, :102].reshape(-1, 51, 2)[~bm].any()
    assert not x[:, 102:186].reshape(-1, 42, 2)[~hm].any()


def features(obs, row):
    """Exactly the original whole/context span policy, after any requested ROI."""
    obs.validate()
    assert obs.n_frames and np.array_equal(obs.frame_id, np.arange(obs.n_frames))
    try:
        whole, reason, support = _channels(obs)
    except PilotDataError as exc:
        return None, {'reason': 'FEATURE_NORMALIZATION_FAILED', 'detail': str(exc), 'whole_span': None}
    if whole is None:
        return None, {'reason': reason, 'support': support, 'whole_span': None}
    nat, present = to_native_like(obs.xy_px, obs.usable)
    lo, hi = active_span(present)
    rebuilt = build_channels(nat[lo:hi+1], present[lo:hi+1], obs.timestamp_us[lo:hi+1])
    np.testing.assert_array_equal(rebuilt['x'], whole['x'])
    arrays = {'x_whole': whole['x'], 'whole_frame_id': obs.frame_id[lo:hi+1],
              'whole_ts': obs.timestamp_us[lo:hi+1]}
    info = {'reason': None, 'support': support, 'whole_span': [int(lo), int(hi)]}
    if row['role'] == 'train_candidate':
        a, b = row['intervals'][0]; ts = obs.timestamp_us
        ids = np.flatnonzero((ts >= round((a-.3)*1e6)) & (ts <= round((b+.3)*1e6)))
        if not len(ids):
            return None, {**info, 'reason': 'EMPTY_CONTEXT_INTERVAL'}
        cl, ch = max(lo, int(ids[0])), min(hi, int(ids[-1]))
        if ch < cl or int(present[cl:ch+1, 25:67].any(1).sum()) < 4:
            return None, {**info, 'reason': 'TOO_FEW_CONTEXT_HAND_FRAMES'}
        try:
            context = build_channels(nat[cl:ch+1], present[cl:ch+1], ts[cl:ch+1])
        except PilotDataError as exc:
            return None, {**info, 'reason': 'CONTEXT_NORMALIZATION_FAILED', 'detail': str(exc)}
        arrays.update(x_context=context['x'], context_frame_id=obs.frame_id[cl:ch+1], context_ts=ts[cl:ch+1])
        info.update(context_span=[int(cl), int(ch)])
    for key in ('x_whole', 'x_context'):
        if key in arrays: validate_channels(arrays[key])
    return arrays, info


def missing_run_seconds(mask, timestamps):
    idx = np.flatnonzero(mask)
    if not len(idx): return 0.
    groups = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    step = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0.
    return max(float(timestamps[g[-1]] - timestamps[g[0]]) + step for g in groups) / 1e6


def visibility_flags(row, native, masked, sensor):
    if row['role'] != 'train_candidate': return []
    a, b = row['intervals'][0]; ts = native.timestamp_us
    core = (ts >= round(a*1e6)) & (ts <= round(b*1e6))
    flags = []
    for name, old, new in [('native_to_mask', native, masked), ('mask_to_sensor', masked, sensor)]:
        for side, start in [('left', 33), ('right', 54)]:
            ids = np.arange(start, start+21)
            oldany = old.usable[:, ids].any(1); newany = new.usable[:, ids].any(1)
            newly_absent = core & oldany & ~newany
            duration = missing_run_seconds(newly_absent, ts)
            palm = start + np.array([0, 5, 9, 13, 17])
            drop = (float(old.usable[core][:, palm].all(1).mean()) -
                    float(new.usable[core][:, palm].all(1).mean())) if core.any() else 0.
            if duration >= .3 or drop >= .1:
                flags.append({'comparison': name, 'side': side, 'newly_absent_run_seconds': duration,
                              'palm5_drop_pp': 100*drop, 'newly_absent_frame_ids': native.frame_id[newly_absent].tolist(),
                              'status': 'VISUAL_REVIEW_PENDING_NOT_SEMANTIC_FAILURE'})
    return flags


def source_file(row, legacy, tempdir):
    e = row['video_entries'][0]
    with zipfile.ZipFile(e['archive']) as archive:
        zi = archive.getinfo(e['member'])
        assert zi.CRC == e['crc32'] and zi.file_size == e['bytes']
        if row['retained_video']:
            path = Path(row['retained_video']); temporary = False
        else:
            assert e['bytes'] < 200_000_000
            path = tempdir / (row['clip_id'] + '.mp4'); temporary = True
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(zi) as src, path.open('xb') as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
    assert path.stat().st_size == e['bytes']
    assert sha256_file(path) == legacy['source_sha256'], 'Source differs from existing verified extraction'
    return path, temporary


def capture(video, cid, directory, contract, box=None):
    from wis_lab.capture.video_source import extract
    capmod = load_module('roi_training_capture_runtime', CAP / 'capture_raw.py')
    assmod = load_module('roi_training_assign_runtime', JOINT / 'assignment.py')
    directory.mkdir(parents=True)
    lm = capmod.RawCaptureLandmarker(contract); token = lm.instance_token
    try:
        ob = extract(str(video), lm if box is None else capmod.CropProxy(lm, box), cid)
    finally:
        lm.close()
    assert lm.raw_rows, 'No decoded frames'
    raw = {key: np.stack([r[key] for r in lm.raw_rows]) for key in lm.raw_rows[0]}
    capmod.verify_capture(raw, ob, contract)
    assert ob.sidecar['dropped_non_monotonic_pts'] == 0
    ob.sidecar['detector_instance'] = token
    if box is not None:
        ob.sidecar.update(source_bbox_px=box, width=box[2]-box[0], height=box[3]-box[1])
    ob.save(str(directory / 'baseline.npz'))
    with (directory / 'raw.npz').open('xb') as out: np.savez_compressed(out, **raw)
    candidate, sides, trace = base.associate(raw, ob, assmod.HandAssigner, contract)
    candidate.save(str(directory / 'candidate.npz'))
    with (directory / 'assignment.npz').open('xb') as out: np.savez_compressed(out, sides=sides)
    write_new(directory / 'trace.json', trace)
    result = {key: str(directory / (key + '.npz')) for key in ('baseline', 'raw', 'candidate')}
    result.update(reused=False, detector_instance=token, box=box,
                  pins={str(p.resolve()): sha256_file(p) for p in directory.iterdir() if p.is_file()})
    return result


def save_view(obs, directory, row):
    directory.mkdir(parents=True)
    op = directory / 'observation.npz'; obs.save(str(op))
    arrays, info = features(obs, row)
    cp = None
    if arrays is not None:
        cp = directory / 'channels.npz'
        with cp.open('xb') as out: np.savez_compressed(out, **arrays)
    return {'observation': str(op), 'channel_file': str(cp) if cp else None, 'feature_info': info}


def prepare_row(row, context_hash, contract, cache_only=False):
    cid = row['clip_id']; record_path = DATA / 'records' / (cid + '.json')
    if record_path.exists():
        rec = read(record_path)
        assert rec['context_sha256'] == context_hash
        check_pins(rec['outputs'])
        check_pins(rec['native_capture']['pins']); check_pins(rec['sensor_capture']['pins'])
        return rec, True
    d = DATA / 'clips' / cid
    assert not d.exists(), 'Partial evidence exists; review before recovery: ' + str(d)
    d.mkdir(parents=True)
    old = read(row['legacy_record_path'])
    if cache_only:
        assert row.get('reuse_native') and row.get('reuse_sensor') and row.get('retained_video'), 'Cache-only mode cannot extract source or run detector'
    video, temporary = source_file(row, old, DATA / 'temporary')
    native_rec = row.get('reuse_native') or capture(video, cid, d / 'native_capture', contract)
    check_pins(native_rec['pins'])
    native = Observation.load(native_rec['candidate'])
    legacy_ob = Observation.load(old['observation'])
    assert np.array_equal(native.frame_id, legacy_ob.frame_id)
    assert np.array_equal(native.timestamp_us, legacy_ob.timestamp_us)
    width, height = int(legacy_ob.sidecar['width']), int(legacy_ob.sidecar['height'])
    calibration = cal.calibrate_rest(native, width, height)
    write_new(d / 'calibration.json', calibration)
    if calibration['status'] != 'CALIBRATED':
        raise ReadinessError(cid + ': NOT_CALIBRATED; source/native/calibration evidence preserved')
    box = calibration['box_source_pixels']
    sensor_rec = row.get('reuse_sensor')
    if sensor_rec is not None and sensor_rec['box'] != box:
        sensor_rec = None
    if cache_only:
        assert sensor_rec is not None, 'Frozen boundary mismatch; cache-only mode will not run a detector'
    sensor_rec = sensor_rec or capture(video, cid, d / 'sensor_capture', contract, box)
    check_pins(sensor_rec['pins'])
    sensor_ob = Observation.load(sensor_rec['candidate'])
    assert np.array_equal(sensor_ob.frame_id, native.frame_id)
    assert np.array_equal(sensor_ob.timestamp_us, native.timestamp_us)
    masked = roi.apply_roi(native, box)
    sensor_masked = roi.apply_roi(sensor_ob, box)
    views = {}
    for mode, obs in [('native', native), ('masked', masked), ('sensor', sensor_masked),
                       ('m0_sensor', Observation.load(sensor_rec['baseline']))]:
        views[mode] = save_view(obs, d / mode, row)
        if row['role'] == 'train_candidate' and mode != 'm0_sensor' and views[mode]['channel_file'] is None:
            write_new(d / 'INPUT_GATE_FAILURE.json', {'mode': mode, **views[mode]['feature_info']})
            raise ReadinessError(cid + ': required TRAIN view unavailable: ' + mode)
    views['m0_native'] = {'observation': old['observation'], 'channel_file': old['channel_file'],
                          'feature_info': old['feature_check'], 'reused_legacy': True}
    flags = visibility_flags(row, native, masked, sensor_masked)
    write_new(d / 'VISUAL_FLAGS.json', flags)
    e = row['video_entries'][0]
    rec = {'clip_id': cid, 'role': row['role'], 'signer_id': row['signer_id'], 'label_index': row['label_index'],
           'word_code': row['word_code'], 'canonical_label': row['canonical_label'], 'frames': native.n_frames,
           'context_sha256': context_hash, 'source': {k: e[k] for k in ('archive', 'member', 'crc32', 'bytes')},
           'calibration': calibration, 'calibration_path': str(d / 'calibration.json'),
           'native_capture': native_rec, 'sensor_capture': sensor_rec, 'views': views, 'visual_flags': flags,
           'training': 0, 'rest_wrist_filter': 'OFF'}
    rec['source']['sha256'] = old['source_sha256']
    rec['outputs'] = {str(p.resolve()): sha256_file(p) for p in d.rglob('*') if p.is_file()}
    write_new(record_path, rec)
    if temporary:
        # One known file only. Verify absolute location before removing our temporary video.
        resolved = video.resolve(); expected = (DATA / 'temporary').resolve()
        assert resolved.parent == expected and resolved.name == cid + '.mp4'
        resolved.unlink()
    return rec, False


def run(limit, max_seconds, min_free_gb, cache_only=False):
    authorize()
    assert not (U / 'PREP_FIRST_FAILURE.json').exists(), 'First failure preserved; explicit review required'
    context = read(CONTEXT); check_pins(context['pins'])
    for p, old in context['archive_stat'].items():
        st = Path(p).stat(); assert [st.st_size, st.st_mtime_ns] == old, 'Archive changed: ' + p
    assert Path(context['output_root']).resolve() == DATA.resolve()
    lock = U / 'PREP_RUNNING.json'
    write_new(lock, {'pid': os.getpid(), 'context_sha256': sha256_file(CONTEXT)})
    start = time.perf_counter(); completed = skipped = cache_reused_rows = 0; cid = None
    try:
        contract = read(LAB / 'config/contract.json')
        for row in context['ordered_rows']:
            if cache_only and not (row.get('reuse_native') and row.get('reuse_sensor')):
                continue
            if time.perf_counter() - start >= max_seconds:
                break
            cid = row['clip_id']
            if (DATA / 'records' / (cid + '.json')).exists():
                _, was_reused = prepare_row(row, sha256_file(CONTEXT), contract, cache_only)
                assert was_reused
                skipped += 1
                continue
            if completed >= limit:
                break
            assert shutil.disk_usage(U).free >= min_free_gb * 1e9, 'Insufficient local output space'
            event(stage='row_start', clip_id=cid, role=row['role'])
            rec, was_reused = prepare_row(row, sha256_file(CONTEXT), contract, cache_only)
            assert not was_reused
            completed += 1
            cache_reused_rows += int(rec['native_capture']['reused'] or rec['sensor_capture']['reused'])
            event(stage='row_complete', clip_id=cid, role=row['role'], reused_record=was_reused,
                  frames=rec['frames'], visual_flags=len(rec['visual_flags']), invocation_rows=completed,
                  seconds=time.perf_counter()-start)
        total = sum((DATA / 'records' / (r['clip_id'] + '.json')).exists() for r in context['ordered_rows'])
        result = {'status': 'PREPARED_PENDING_INDEPENDENT_AND_VISUAL_CHECK' if total == 1199 else 'CHUNK_COMPLETE',
                  'total_completed': total, 'total_requested': 1199, 'invocation_completed_including_reuse': completed,
                  'invocation_preexisting_records_verified': skipped, 'invocation_detector_cache_reused_rows': cache_reused_rows,
                  'seconds': time.perf_counter()-start,
                  'limit': limit, 'max_seconds': max_seconds, 'training': 0, 'cache_only': cache_only,
                  'cache_only_scope': 'All reusable paired caches within the same frozen1199 ordering; no manual word selection' if cache_only else None}
        write_status(result); event(**result)
    except Exception as exc:
        failure = {'status': 'FAILED_STOPPED_BEFORE_TRAINING', 'clip_id': cid,
                   'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc(),
                   'context_sha256': sha256_file(CONTEXT), 'training': 0}
        if not (U / 'PREP_FIRST_FAILURE.json').exists(): write_new(U / 'PREP_FIRST_FAILURE.json', failure)
        write_status(failure); event(stage='failed', clip_id=cid, reason=str(exc))
        raise
    finally:
        # Do not guess whether a stale lock belongs to another process.
        if lock.exists() and read(lock)['pid'] == os.getpid(): lock.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'run', 'status'))
    parser.add_argument('--limit', type=int, default=1200)
    parser.add_argument('--max-seconds', type=float, default=1800)
    parser.add_argument('--min-free-gb', type=float, default=10.)
    parser.add_argument('--cache-only', action='store_true')
    args = parser.parse_args()
    assert args.limit > 0 and args.max_seconds > 0 and args.min_free_gb > 0
    if args.action == 'freeze': freeze()
    elif args.action == 'run': run(args.limit, args.max_seconds, args.min_free_gb, args.cache_only)
    else: print(json.dumps(read(U / 'PREP_STATUS.json') if (U / 'PREP_STATUS.json').exists() else {'status': 'NOT_RUN'}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
