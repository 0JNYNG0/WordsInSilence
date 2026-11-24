"""New development actual edited-video features, never slices of clean temporal contexts.

No feedback heads/DTW/training, no private data. A completed input is only
hash-verified and reused. An interrupted input requires explicit inspection.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib, importlib.util, json, os, shutil, sys, time, traceback
import numpy as np
from process_rgb_features import extract_rgb

U = Path(__file__).resolve().parent
O = U.parent
V = O / 'word100_learned_body_relation_20261102'
OUT = Path('W:/WordsInSilence_derived/word100_action_completion_supervision_20261102/development_process_inputs')
VIDEO_ROOT = OUT.parent / 'development_process_videos'
LAB = O.parent
PROJECT = LAB.parents[1]

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()
def write(p, v): Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
def now(): return datetime.now(timezone.utc).isoformat()

def validate_arrays(a, q, meta):
    n=q['frames']
    assert set(a)=={'x','mask','rgb','prelinear','frame_id','timestamp_us','image_available'}
    assert a['x'].shape==a['mask'].shape==(n,69)
    assert a['rgb'].shape==a['prelinear'].shape==(n,768)
    assert a['mask'].dtype==bool and a['image_available'].dtype==bool
    assert np.array_equal(a['frame_id'],np.arange(n))
    assert np.array_equal(a['timestamp_us'],q['view_timestamps_us'])
    assert np.array_equal(a['image_available'],q['image_available'])
    assert all(np.isfinite(a[k]).all() for k in ['x','rgb','prelinear'])
    assert meta['real_frames']==meta['returned_rows']==n and meta['exact_decode_pts_verified']
    assert meta['real_frame_id']==list(range(n)) and meta['real_timestamp_us']==q['view_timestamps_us']
    assert not meta['padding_is_observation'] and not meta['padding_is_motion_label']
    assert not meta['timestamps_extended'] and not meta['frame_ids_extended']
    assert meta['padding_count']==max(0,16-n)
    if n<16:
        assert meta['model_window_indices']==[list(range(n))+[n-1]*(16-n)]
        assert meta['source_mask']==[[True]*n+[False]*(16-n)]
        assert np.array_equal(a['rgb'],np.repeat(a['rgb'][:1],n,axis=0))
        assert np.array_equal(a['prelinear'],np.repeat(a['prelinear'][:1],n,axis=0))
    else:
        assert all(len(w)==16 and all(mask) for w,mask in zip(meta['model_window_indices'],meta['source_mask']))

def main():
    assert not (U / 'DEV_PROCESS_INPUT_MANIFEST.json').exists(), 'Completed extraction must not rerun'
    media = read(U / 'DEV_PROCESS_VIDEO_MANIFEST.json')
    rows = media['rows']
    assert len(rows) == 80 and len({r['pixel_view_key'] for r in rows}) == 80
    assert {r['person'] for r in rows} == set(range(11, 14))
    assert all(r['role'] == 'development' and r['source_vs_decoded_rgb_exact'] for r in rows)
    assert len({r['clip_id'] for r in rows}) == 12
    assert media['freeze_sha256'] == sha(U / 'DEV_PROCESS_VIDEO_FROZEN.json')
    assert media['training_allowed'] is False
    source_views={v['pixel_view_key']:v for v in read(U/'DEV_PARTIAL_VIEWS_MANIFEST.json')['views']}
    assert set(source_views)=={r['pixel_view_key'] for r in rows}
    for r in rows:
        assert r['clip_id'] in {f'NIA_SL_{word}_REAL{person:02d}_F'
            for word in ('WORD0739', 'WORD1133', 'WORD1567', 'WORD2523') for person in range(11, 14)}
        assert r['clip_id'].endswith(f"_REAL{r['person']:02d}_F")
        assert Path(r['video']).resolve() == (VIDEO_ROOT / r['pixel_view_key'] / 'view.mp4').resolve()
        assert r['freeze_sha256'] == media['freeze_sha256']
        assert 2 <= r['frames'] <= 1200
        assert all(len(r[k]) == r['frames'] for k in
                   ('source_frame_indices', 'source_timestamps_us', 'view_timestamps_us', 'image_available', 'pixel_operations'))
        assert all((op == 'copy') == available and op in ('copy', 'blackout')
                   for op, available in zip(r['pixel_operations'], r['image_available']))
        v=source_views[r['pixel_view_key']]
        assert r['source']==v['source_path'] and r['source_sha256']==v['source_sha256']
        assert r['source_frame_indices']==[e['source_native_frame'] for e in v['ordered_frame_map']]
        assert r['source_timestamps_us']==[e['source_native_timestamp_us'] for e in v['ordered_frame_map']]
        assert r['view_timestamps_us']==[e['view_timestamp_us'] for e in v['ordered_frame_map']]
    frozen_path = U / 'DEV_PROCESS_EXTRACTION_FROZEN.json'
    if not frozen_path.exists():
        parent = read(U / 'PROCESS_EXTRACTION_FROZEN.json')
        for p, h in parent['pins'].items(): assert sha(p) == h, p
        paths = [Path(__file__).resolve(), U / 'process_rgb_features.py',
                 U / 'DEV_PROCESS_VIDEO_MANIFEST.json', U / 'DEV_PROCESS_VIDEO_FROZEN.json',
                 U / 'DEV_PARTIAL_VIEWS_MANIFEST.json', U / 'PROCESS_SHORT_INPUT_REVIEW.json',
                 U / 'PROCESS_EXTRACTION_FROZEN.json', U / 'PROCESS_INPUT_MANIFEST.json', U / 'DEV_PROCESS_CODE_INPUT_AUDIT.json', U / 'dev_materialize_process_videos.py', V / 'demo_adapter.py']
        # Preserve the validated FIT sensor dependency pins. No FIT input,
        # score, or head is recomputed; only DEV pixels enter new extraction.
        paths += [O / 'word100_signrep_feasibility_20261029/common.py',
                  O / 'word100_hand_assignment_20261016/capture_raw.py',
                  O / 'word100_hand_assignment_joint_20261016/assignment.py',
                  LAB / 'config/contract.json',
                  LAB / 'src/wis_lab/capture/video_source.py',
                  LAB / 'src/wis_lab/capture/landmarker.py',
                  LAB / 'src/wis_lab/capture/face_landmarks.py',
                  LAB / 'src/wis_lab/core/observation.py']
        paths += [PROJECT / 'Tools/models' / name for name in
                  ('hand_landmarker.task', 'pose_landmarker_lite.task', 'face_landmarker.task')]
        pins = dict(parent['pins']); pins.update({str(p): sha(p) for p in paths})
        write(frozen_path, dict(at=now(), pins=pins, unique_inputs=80, query_aliases=80,
            purpose='Fresh features from exactly verified edited pixels, including native bounded-context controls.',
            sensors='Existing full-frame MediaPipe joint_v3/face and frozen SignRep; no feedback head calls.',
            short_input='Last actual image padded internally to16 only; N real times retained; not motion truth.',
            role='development', recorded_ids=list(range(11, 14)), no_private=True,
            training_allowed=False,repeated_development=True,independent_final_test=False,
            required_actual_rgb_device='GPU.0',device_failure='Stop before feature inference and investigate; no silent CPU run.',
            labels_as_features=False, source_frame_index_as_feature=False, edit_kind_as_feature=False,
            new_feedback_scoring=0, new_DTW=0, new_training=0))
    frozen = read(frozen_path)
    for p, h in frozen['pins'].items(): assert sha(p) == h, p
    sys.path.insert(0, str(V))
    spec = importlib.util.spec_from_file_location('_process_sensor_adapter', V / 'demo_adapter.py')
    adapter = importlib.util.module_from_spec(spec); sys.modules[spec.name] = adapter; spec.loader.exec_module(adapter)
    shared = adapter.shared
    compiled, device = None, None
    observer = shared.base._module('_process_geometry', O / 'word100_guide_video_feedback_20261028/motion_observer.py')
    OUT.mkdir(parents=True, exist_ok=True)
    result = []
    write(U / 'DEV_PROCESS_INPUT_PROGRESS.json', dict(at=now(), status='RUNNING', pid=os.getpid(), done=0, total=80, device=device))
    for q in rows:
        key = q['pixel_view_key']; source = Path(q['video'])
        d = (OUT / key).resolve(); assert d.is_relative_to(OUT.resolve())
        assert sha(source) == q['video_sha256']
        if (d / 'COMPLETE.json').exists():
            r = read(d / 'COMPLETE.json')
            assert r['extraction_freeze_sha256'] == sha(frozen_path)
            assert r['video_sha256'] == q['video_sha256'] and r['pixel_view_key'] == key
            for p, h in r['output_pins'].items(): assert sha(p) == h, p
            assert r['role']=='development' and r['person']==q['person']
            with np.load(r['input'],allow_pickle=False) as saved:
                validate_arrays({k:saved[k] for k in saved.files},q,read(r['rgb_context']))
            result.append(r)
            continue
        assert not d.exists(), f'Partial input requires inspection; no silent restart: {d}'
        assert shutil.disk_usage(OUT).free > 2 * 1024**3, 'Less than2GB free'
        if compiled is None:
            compiled,device=shared.compile_rgb()
            assert device['actual']==['GPU.0'], f'Expected existing GPU.0 route; investigate before inference: {device}'
        d.mkdir()
        write(d / 'STARTED.json', dict(at=now(), pid=os.getpid(), video=str(source), sha256=q['video_sha256']))
        try:
            start = time.perf_counter()
            ids, ts, w, h = shared.source_axis(source)
            assert np.array_equal(ids, np.arange(q['frames']))
            assert np.array_equal(ts, q['view_timestamps_us'])
            assert [0, 0, w, h] == q['actual_sensor_extent']
            with ThreadPoolExecutor(max_workers=2) as pool:
                rgb_job = pool.submit(extract_rgb, shared, source, ids, ts, w, h, compiled, None)
                availability_job = pool.submit(adapter.image_information, source, ids)
                tick = time.perf_counter(); ob, face = shared.base._capture(source, d, None)
                mp_seconds = time.perf_counter() - tick
                rgb, pre, rgb_seconds, rgb_meta = rgb_job.result()
                available = availability_job.result()
            assert rgb_meta['supported']
            assert np.array_equal(ob.frame_id, ids) and np.array_equal(ob.timestamp_us, ts)
            assert np.array_equal(available, q['image_available']), 'Image availability disagrees with actual pixels'
            observed = {k: getattr(ob, k).copy() for k in ['xy_px', 'usable', 'observed', 'frame_id', 'timestamp_us']}
            f = observer.extract_features(observed)
            a = dict(x=np.nan_to_num(f['x']).astype(np.float32), mask=f['mask'], rgb=rgb, prelinear=pre,
                     frame_id=ids, timestamp_us=ts, image_available=available)
            assert a['x'].shape == (len(ids), 69) and a['mask'].shape == a['x'].shape
            assert a['mask'].dtype == bool and available.dtype == bool
            assert np.array_equal(f['mask'], np.isfinite(f['x']))
            assert rgb.shape == pre.shape == (len(ids), 768)
            assert all(np.isfinite(a[k]).all() for k in ['x', 'rgb', 'prelinear'])
            validate_arrays(a,q,rgb_meta)
            # Preserve detector output separately from image availability. Candidate
            # losses/inference must not turn tracked coordinates on blank pixels
            # into direct observation. No original arrays are overwritten.
            np.savez_compressed(d / 'input.npz', **a)
            write(d / 'RGB_CONTEXT.json', rgb_meta)
            with np.load(d/'input.npz',allow_pickle=False) as saved:
                validate_arrays({k:saved[k] for k in saved.files},q,rgb_meta)
            outputs = [p for p in d.iterdir() if p.is_file() and p.name not in ['STARTED.json', 'COMPLETE.json']]
            r = dict(at=now(), pixel_view_key=key, aliases=q['aliases'], clip_id=q['clip_id'], person=q['person'], role='development',
                source=q['source'], source_sha256=q['source_sha256'], video=str(source), video_sha256=sha(source),
                input=str(d / 'input.npz'), input_sha256=sha(d / 'input.npz'), frames=len(ids),
                view_timestamps_us=ts.tolist(), source_frame_indices=q['source_frame_indices'],
                source_timestamps_us=q['source_timestamps_us'], actual_sensor_extent=[0, 0, w, h],
                source_gap_output_indices_audit_only=(np.flatnonzero(np.diff(q['source_frame_indices']) != 1) + 1).tolist(),
                motion_context_note='Geometry uses supplied edited-video PTS/indices. A jump across a source gap is an edit-seam response, not observed travel through removed frames. Audit-only source-gap metadata must not become a label or feature.',
                image_available=available.tolist(), rgb_context=str(d / 'RGB_CONTEXT.json'),
                internal_padding_count=rgb_meta['padding_count'], device=device, parallel_observers=True,
                seconds=time.perf_counter()-start, mediapipe_seconds=mp_seconds, rgb_seconds=rgb_seconds,
                extraction_freeze_sha256=sha(frozen_path), output_pins={str(p):sha(p) for p in outputs},
                new_feedback_scoring=0, new_DTW=0, new_training=0,
                training_allowed=False,repeated_development=True,independent_final_test=False,
                unknown_policy='image_available is separate from raw detector masks; no padding/blank labels assigned')
            assert r['video_sha256'] == q['video_sha256']
            write(d / 'COMPLETE.json', r); result.append(r)
            write(U / 'DEV_PROCESS_INPUT_PROGRESS.json', dict(at=now(), status='RUNNING', pid=os.getpid(), done=len(result), total=80,
                current=key, device=device, last_seconds=r['seconds']))
            print('process inputs',len(result),'/80',q['clip_id'],q['frames'],round(r['seconds'],3),flush=True)
        except BaseException:
            (d / 'FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8'); raise
    for p,h in frozen['pins'].items(): assert sha(p)==h,p
    write(U/'DEV_PROCESS_INPUT_MANIFEST.json',dict(at=now(),status='NEW_EDITED_PUBLIC_FEATURES_PREPARED',rows=result,
        extraction_freeze=str(frozen_path),extraction_freeze_sha256=sha(frozen_path),unique_inputs=80,query_aliases=80,
        registered_ids=list(range(11,14)),source_clips=12,new_training=0,new_feedback_scoring=0,new_DTW=0,
        no_private=True,training_allowed=False,repeated_development=True,independent_final_test=False,whole_goal_complete=False))
    write(U/'DEV_PROCESS_INPUT_PROGRESS.json',dict(at=now(),status='COMPLETE',active=None,done=80,total=80))
    print('Prepared80 actual view inputs; no feedback scoring or training',flush=True)

if __name__=='__main__':
    try: main()
    except BaseException:
        p=U/'DEV_PROCESS_INPUT_FIRST_FAILURE.txt'
        if not p.exists(): p.write_text(traceback.format_exc(),encoding='utf8')
        raise
