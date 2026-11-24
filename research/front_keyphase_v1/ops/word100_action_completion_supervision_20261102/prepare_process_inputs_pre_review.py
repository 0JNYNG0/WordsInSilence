"""New actual edited-video features, never slices of clean temporal contexts.

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
AA = O / 'word100_partial_order_live_20261102'
V = O / 'word100_learned_body_relation_20261102'
OUT = Path('W:/WordsInSilence_derived/word100_action_completion_supervision_20261102/process_inputs')

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()
def write(p, v): Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
def now(): return datetime.now(timezone.utc).isoformat()

def main():
    assert not (U / 'PROCESS_INPUT_MANIFEST.json').exists(), 'Completed extraction must not rerun'
    media = read(U / 'PROCESS_VIDEO_MANIFEST.json')
    rows = media['rows']
    assert len(rows) == 216 and len({r['pixel_view_key'] for r in rows}) == 216
    assert {r['person'] for r in rows} == set(range(1, 10))
    assert all(r['role'] == 'fit' and r['source_vs_decoded_rgb_exact'] for r in rows)
    frozen_path = U / 'PROCESS_EXTRACTION_FROZEN.json'
    if not frozen_path.exists():
        parent = read(AA / 'EXECUTION_FROZEN.json')
        for p, h in parent['pins'].items(): assert sha(p) == h, p
        paths = [Path(__file__).resolve(), U / 'process_rgb_features.py',
                 U / 'PROCESS_VIDEO_MANIFEST.json', U / 'PROCESS_VIDEO_FROZEN.json',
                 U / 'PARTIAL_VIEWS_MANIFEST.json', U / 'PROCESS_SHORT_INPUT_REVIEW.json',
                 AA / 'EXECUTION_FROZEN.json', V / 'demo_adapter.py']
        pins = dict(parent['pins']); pins.update({str(p): sha(p) for p in paths})
        write(frozen_path, dict(at=now(), pins=pins, unique_inputs=216, query_aliases=217,
            purpose='Fresh features from exactly verified edited pixels, including native bounded-context controls.',
            sensors='Existing full-frame MediaPipe joint_v3/face and frozen SignRep; no feedback head calls.',
            short_input='Last actual image padded internally to16 only; N real times retained; not motion truth.',
            role='fit', recorded_ids=list(range(1, 10)), no_private=True,
            labels_as_features=False, source_frame_index_as_feature=False, edit_kind_as_feature=False,
            new_feedback_scoring=0, new_DTW=0, new_training=0))
    frozen = read(frozen_path)
    for p, h in frozen['pins'].items(): assert sha(p) == h, p
    sys.path.insert(0, str(V))
    spec = importlib.util.spec_from_file_location('_process_sensor_adapter', V / 'demo_adapter.py')
    adapter = importlib.util.module_from_spec(spec); sys.modules[spec.name] = adapter; spec.loader.exec_module(adapter)
    shared = adapter.shared
    compiled, device = shared.compile_rgb()
    observer = shared.base._module('_process_geometry', O / 'word100_guide_video_feedback_20261028/motion_observer.py')
    OUT.mkdir(parents=True, exist_ok=True)
    result = []
    write(U / 'PROCESS_INPUT_PROGRESS.json', dict(at=now(), status='RUNNING', pid=os.getpid(), done=0, total=216, device=device))
    for q in rows:
        key = q['pixel_view_key']; source = Path(q['video'])
        d = (OUT / key).resolve(); assert d.is_relative_to(OUT.resolve())
        assert sha(source) == q['video_sha256']
        if (d / 'COMPLETE.json').exists():
            r = read(d / 'COMPLETE.json')
            assert r['extraction_freeze_sha256'] == sha(frozen_path)
            assert r['video_sha256'] == q['video_sha256'] and r['pixel_view_key'] == key
            for p, h in r['output_pins'].items(): assert sha(p) == h, p
            result.append(r)
            continue
        assert not d.exists(), f'Partial input requires inspection; no silent restart: {d}'
        assert shutil.disk_usage(OUT).free > 2 * 1024**3, 'Less than2GB free'
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
            assert rgb.shape == pre.shape == (len(ids), 768)
            assert all(np.isfinite(a[k]).all() for k in ['x', 'rgb', 'prelinear'])
            # Preserve detector output separately from image availability. Candidate
            # losses/inference must not turn tracked coordinates on blank pixels
            # into direct observation. No original arrays are overwritten.
            np.savez_compressed(d / 'input.npz', **a)
            write(d / 'RGB_CONTEXT.json', rgb_meta)
            outputs = [p for p in d.iterdir() if p.is_file() and p.name not in ['STARTED.json', 'COMPLETE.json']]
            r = dict(at=now(), pixel_view_key=key, aliases=q['aliases'], clip_id=q['clip_id'], person=q['person'], role='fit',
                source=q['source'], source_sha256=q['source_sha256'], video=str(source), video_sha256=sha(source),
                input=str(d / 'input.npz'), input_sha256=sha(d / 'input.npz'), frames=len(ids),
                view_timestamps_us=ts.tolist(), source_frame_indices=q['source_frame_indices'],
                source_timestamps_us=q['source_timestamps_us'], actual_sensor_extent=[0, 0, w, h],
                image_available=available.tolist(), rgb_context=str(d / 'RGB_CONTEXT.json'),
                internal_padding_count=rgb_meta['padding_count'], device=device, parallel_observers=True,
                seconds=time.perf_counter()-start, mediapipe_seconds=mp_seconds, rgb_seconds=rgb_seconds,
                extraction_freeze_sha256=sha(frozen_path), output_pins={str(p):sha(p) for p in outputs},
                new_feedback_scoring=0, new_DTW=0, new_training=0,
                unknown_policy='image_available is separate from raw detector masks; no padding/blank labels assigned')
            assert r['video_sha256'] == q['video_sha256']
            write(d / 'COMPLETE.json', r); result.append(r)
            write(U / 'PROCESS_INPUT_PROGRESS.json', dict(at=now(), status='RUNNING', pid=os.getpid(), done=len(result), total=216,
                current=key, device=device, last_seconds=r['seconds']))
            print('process inputs',len(result),'/216',q['clip_id'],q['frames'],round(r['seconds'],3),flush=True)
        except BaseException:
            (d / 'FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8'); raise
    for p,h in frozen['pins'].items(): assert sha(p)==h,p
    write(U/'PROCESS_INPUT_MANIFEST.json',dict(at=now(),status='NEW_EDITED_PUBLIC_FEATURES_PREPARED',rows=result,
        extraction_freeze=str(frozen_path),extraction_freeze_sha256=sha(frozen_path),unique_inputs=216,query_aliases=217,
        registered_ids=list(range(1,10)),source_clips=36,new_training=0,new_feedback_scoring=0,new_DTW=0,
        no_private=True,whole_goal_complete=False))
    write(U/'PROCESS_INPUT_PROGRESS.json',dict(at=now(),status='COMPLETE',active=None,done=216,total=216))
    print('Prepared216 actual view inputs; no feedback scoring or training',flush=True)

if __name__=='__main__':
    try: main()
    except BaseException:
        p=U/'PROCESS_INPUT_FIRST_FAILURE.txt'
        if not p.exists(): p.write_text(traceback.format_exc(),encoding='utf8')
        raise
