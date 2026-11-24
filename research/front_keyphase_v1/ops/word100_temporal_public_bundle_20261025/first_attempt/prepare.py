"""Package existing public observations and original RGB; no ML inference/training.

All input files are read-only. Source labels live in the manifest, not model arrays.
The bundle is development/regression data, not a newly unseen test set.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
import json, hashlib, traceback, time
import cv2
import numpy as np

U = Path(__file__).resolve().parent
O = U.parent
FULL = O / 'word100_fullframe_hand_merge_20261020'
SMALL = O / 'word100_small_multipeople_angle_20261021'
TRACK = O / 'word100_image_tracking_pair_20261020'
PALM = np.array([0, 5, 9, 13, 17])

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()

def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def runs(mask):
    edge = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edge == 1).tolist(), np.flatnonzero(edge == -1).tolist()))

def main():
    assert not (U / 'MANIFEST.json').exists(), 'Completed bundle must not be overwritten'
    (U / 'inputs').mkdir(exist_ok=True)
    (U / 'preview').mkdir(exist_ok=True)
    f = read(FULL / 'FROZEN.json')
    t = read(TRACK / 'FROZEN.json')
    small = read(SMALL / 'PREPARED_INDEX.json')
    small_by = {r['clip_id']: r for r in small}
    track_by = {r['clip_id']: r for r in t['rows']}
    full_idx = {r['clip_id']: r for r in read(FULL / 'INDEX.json')}
    training = read(O / 'word100_training_20261013/MANIFEST.json')
    labels = {r['clip_id']: r for r in training}
    rows = {r['clip_id']: dict(r, cohort=['diagnostic36']) for r in f['rows']}
    for r in small:
        if r['clip_id'] in rows:
            rows[r['clip_id']]['cohort'].append('multipeople27')
        else:
            rows[r['clip_id']] = dict(r, cohort=['multipeople27'])
    assert len(rows) == 61
    historical = {}
    for obj in (f, t, read(SMALL / 'FROZEN.json')):
        historical.update(obj['pins'])
    for r in full_idx.values():
        historical[r['path']] = r['sha256']
        historical.update(r['outputs'])
    for r in small:
        historical.update(r.get('output_hashes', {}))
    historical = {str(Path(p).resolve()): h for p, h in historical.items()}
    checked = {}
    def pin(path):
        path = str(Path(path).resolve())
        if path not in checked:
            value = sha(path)
            if path in historical:
                assert value == historical[path], ('Historical hash mismatch', path)
            checked[path] = value
        return checked[path]

    protected = {p: sha(p) for p in f['protected']}
    # Current before/after pins do not assert old reports equal current runtime.
    write(U / 'PROTECTED_BEFORE.json', protected)
    write(U / 'SCOPE.json', {
        'authorized': 'Public input/evaluation preparation; report before any training.',
        'selection': 'Union of fixed diagnostic36 and multipeople27; overlap2; no new score selection.',
        'input_pipeline': 'Original full-frame RGB/native detection; merge then fixed ROI. Sensor results excluded from primary model arrays.',
        'no_actions': ['training', 'model inference', 'new detector extraction', 'download', 'user or mentor recording access', 'production edits'],
        'source_manifests': {str(FULL / 'FROZEN.json'): pin(FULL / 'FROZEN.json'), str(SMALL / 'PREPARED_INDEX.json'): pin(SMALL / 'PREPARED_INDEX.json')},
        'timing': 'Original decoded frame index and timestamp; no speed conversion or discarded frames.'})
    manifest, events, samples = [], [], []
    total = Counter()
    for cid in sorted(rows):
        r = rows[cid]; label = labels[cid]
        assert r['signer_id'] in range(1, 10) and label['role'] == 'train_candidate'
        assert r['word_code'] == label['word_code'] and r['label'] == label['canonical_label']
        video = Path(r['video'])
        source = track_by[cid]['source'] if cid in track_by else r['source']
        assert pin(video) == source['sha256'], (cid, 'RGB not original bytes')
        paths = {k: r[k] for k in ('raw', 'assignment', 'native', 'old_masked')}
        if cid in full_idx:
            paths['merged_roi'] = str(FULL / 'outputs' / cid / 'masked.npz')
            paths['detections'] = full_idx[cid]['path']
        else:
            paths['merged_roi'] = r['observation']
            paths['detections'] = r['detections']
        if cid in small_by:
            paths['angle_evidence'] = small_by[cid]['prepared']
            paths['quality_evidence'] = small_by[cid]['quality']
        pins = {str(Path(p).resolve()): pin(p) for p in paths.values()}
        native, old, merged, raw, assign = (load(paths[k]) for k in ('native', 'old_masked', 'merged_roi', 'raw', 'assignment'))
        n = len(native['frame_id']); times = native['timestamp_us']
        assert native['xy_px'].shape == (n, 75, 2)
        assert np.array_equal(native['frame_id'], np.arange(n))
        assert (np.diff(times) > 0).all() and times[0] == 0
        for z in (old, merged, raw):
            for k in ('frame_id', 'timestamp_us'):
                assert np.array_equal(z[k], native[k]), (cid, k)
        for z in (native, old, merged):
            assert z['xy_px'].shape == (n, 75, 2)
            assert np.isfinite(z['xy_px'][z['usable']]).all()
        assert assign['sides'].shape == (n, 2)
        detections = read(paths['detections'])
        assert len(detections['frames']) == n
        for i, q in enumerate(detections['frames']):
            assert q['frame_id'] == i and q['timestamp_us'] == int(times[i])
            assert 'full' in q, (cid, 'not full-frame detections')
        # Saved merge must not alter existing ROI observations or body geometry.
        keep = old['usable']
        assert np.array_equal(merged['xy_px'][keep], old['xy_px'][keep], equal_nan=True)
        assert merged['usable'][keep].all()
        assert np.array_equal(merged['xy_px'][:, :33], old['xy_px'][:, :33], equal_nan=True)
        x0, y0, x1, y1 = r['box']
        def inside(xy):
            return np.isfinite(xy).all(-1) & (xy[..., 0] >= x0) & (xy[..., 0] < x1) & (xy[..., 1] >= y0) & (xy[..., 1] < y1)
        assert not (old['usable'] & ~inside(old['xy_px'])).any()
        assert not (merged['usable'] & ~inside(merged['xy_px'])).any()
        data = {'frame_id': native['frame_id'], 'timestamp_us': times, 'roi_box': np.array(r['box'], np.int32),
                'width': raw['width'], 'height': raw['height'], 'assignment_sides': assign['sides']}
        for prefix, z in (('native', native), ('baseline_roi', old), ('redetected_roi', merged)):
            for key in ('xy_px', 'observed', 'usable', 'conf', 'hand_score'):
                data[prefix + '_' + key] = z[key]
        for key in ('raw_count', 'hand_xy_px', 'raw_label', 'raw_score', 'hand_finite', 'pose_xy_px', 'pose_usable', 'pose_conf'):
            data['raw_' + key] = raw[key]
        # Provenance means source of supplied coordinates, never anatomical truth.
        data['point_origin_roi'] = np.where(old['usable'], 1, np.where(merged['usable'], 2, 0)).astype(np.uint8)
        output = U / 'inputs' / (cid + '.npz')
        assert not output.exists()
        np.savez_compressed(output, **data)
        core = np.zeros(n, bool)
        for a, b in r['intervals']:
            core |= (times >= a * 1e6) & (times <= b * 1e6)
        # Availability episodes only, not claims of occlusion, inactivity or true shape.
        palm = old['usable'][:, 33:].reshape(n, 2, 21)[:, :, PALM].all(-1)
        native_palm = native['usable'][:, 33:].reshape(n, 2, 21)[:, :, PALM].all(-1)
        new_palm = merged['usable'][:, 33:].reshape(n, 2, 21)[:, :, PALM].all(-1)
        for side in (0, 1):
            for start, end in runs(~palm[:, side]):
                e = {'clip_id': cid, 'side': side, 'start_index': start, 'end_index_exclusive': end,
                     'start_us': int(times[start]), 'last_us': int(times[end-1]), 'frames': end-start,
                     'core_frames': int(core[start:end].sum()),
                     'full_native_palm_frames': int(native_palm[start:end, side].sum()),
                     'redetected_roi_palm_frames': int(new_palm[start:end, side].sum()),
                     'reappears': bool(end < n and palm[end, side]),
                     'reason': 'UNREVIEWED_AVAILABILITY_GAP_NOT_AN_OCCLUSION_LABEL'}
                events.append(e)
        # Full original frame decode: verifies frames, dimensions and clocks, no inference.
        cap = cv2.VideoCapture(str(video)); assert cap.isOpened(), cid
        fps = cap.get(cv2.CAP_PROP_FPS); assert fps > 0
        selected = sorted(set([0, n-1] + [int(i) for i in np.quantile(np.flatnonzero(core), [.25, .5, .75])]))
        thumbnails = []; dec = 0
        while True:
            ok, image = cap.read()
            if not ok: break
            assert dec < n and image.shape[:2] == (int(raw['height'][dec]), int(raw['width'][dec]))
            if dec in selected:
                thumb = cv2.resize(image, (384, 216))
                cv2.line(thumb, (0, round(y1 * 216 / image.shape[0])), (383, round(y1 * 216 / image.shape[0])), (0,255,255), 2)
                cv2.putText(thumb, f'{cid.split("_")[2]} ID{r["signer_id"]:02d} f{dec}', (8,20), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1)
                thumbnails.append(thumb)
                samples.append({'clip_id': cid, 'frame_id': dec, 'timestamp_us': int(times[dec]), 'visibility': 'UNANNOTATED', 'keypoints_2d': None, 'purpose': 'source/ROI sanity and future blinded visible-coordinate review'})
            dec += 1
        cap.release(); assert dec == n, (cid, dec, n)
        clock_error = float(np.max(np.abs(times - np.rint(np.arange(n) * 1e6 / fps))))
        assert clock_error <= 1.0, (cid, 'clock/frame misalignment', clock_error)
        preview = U / 'preview' / (cid + '.jpg')
        assert cv2.imwrite(str(preview), np.concatenate(thumbnails, axis=1))
        row = {'clip_id': cid, 'source_group': cid, 'gloss_key': 'AIHub_103:' + r['word_code'],
               'word_code': r['word_code'], 'display_label': r['label'], 'signer_id': r['signer_id'],
               'cohorts': r['cohort'], 'bundle_role': 'known_public_development_regression',
               'classifier_prior_role': 'train_candidate', 'refiner_training_use': 'NOT_ASSIGNED_NOT_TRAINED',
               'frames': n, 'fps': fps, 'video_dimensions': [int(raw['width'][0]), int(raw['height'][0])],
               'rgb_path': str(video.resolve()), 'rgb_sha256': source['sha256'], 'rgb_bytes': video.stat().st_size,
               'source_archive': source['archive'], 'source_member': source['member'],
               'roi_box_source_pixels': r['box'], 'evaluation_annotation_intervals_only': r['intervals'],
               'arrays': str(output), 'arrays_sha256': sha(output), 'source_assets': paths, 'source_pins': pins,
               'historical_pins_matched': sum(p in historical for p in pins),
               'frame_decode_count': dec, 'max_clock_error_us': clock_error,
               'preview': str(preview), 'selection_history': track_by.get(cid, {}).get('selection_reasons', ['fixed_three_word_nine_signer_cohort']),
               'pointwise_anatomical_ground_truth': 'NOT_AVAILABLE', 'hidden_3d_ground_truth': 'NOT_AVAILABLE'}
        manifest.append(row)
        total.update(clips=1, frames=n, rgb_bytes=video.stat().st_size, array_bytes=output.stat().st_size,
                     retained_usable_roi_points=int(keep.sum()), added_available_points=int((merged['usable'] & ~keep).sum()))
        print(f'{len(manifest)}/61 {cid} frames={n}', flush=True)
    words = defaultdict(list)
    for r in manifest: words[r['word_code']].append(r)
    links = []
    for q in manifest:
        refs = [r['clip_id'] for r in words[q['word_code']] if r['signer_id'] != q['signer_id']]
        links.append({'query_clip': q['clip_id'], 'gloss_key': q['gloss_key'], 'reference_clips': refs,
                      'status': 'PUBLIC_REFERENCE_LINKS_READY' if refs else 'NO_OTHER_SIGNER_IN_THIS_BUNDLE',
                      'scope': 'same source word ID; no per-word anatomical correctness implied; no self/person reference'})
    glossary = []
    for code in sorted({r['word_code'] for r in training}):
        rr = [r for r in training if r['word_code'] == code]
        mapping = {(r['canonical_label'], r['label_index']) for r in rr}; assert len(mapping) == 1
        name, idx = next(iter(mapping))
        glossary.append({'gloss_key': 'AIHub_103:' + code, 'source_word_id': code, 'display_label': name,
                         'classifier_index': idx, 'source': 'AIHub_103_existing_word100_manifest',
                         'in_bundle_clips': [r['clip_id'] for r in words.get(code, [])],
                         'cross_corpus_equivalence': 'UNVERIFIED_NO_AUTOMATIC_MERGE',
                         'phonological_attributes': None})
    write(U / 'MANIFEST.json', manifest)
    write(U / 'GLOSS_REFERENCE_INDEX.json', {'glossary': glossary, 'reference_links': links})
    write(U / 'GAP_CANDIDATES.json', events)
    write(U / 'REVIEW_QUEUE.json', {'status': 'UNANNOTATED_NOT_GROUND_TRUTH', 'samples': samples})
    write(U / 'SOURCE_PINS.json', checked)
    for p, expected in protected.items(): assert sha(p) == expected, ('Protected changed', p)
    summary = {'status': 'INPUTS_READY_FOR_REVIEW_NOT_TRAINING_AUTHORIZATION', 'at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
               'counts': dict(total), 'words': len(words), 'signers': sorted({r['signer_id'] for r in manifest}),
               'person_clip_counts': dict(Counter(r['signer_id'] for r in manifest)),
               'reference_links': sum(len(x['reference_clips']) for x in links),
               'queries_without_other_person_reference': sum(not x['reference_clips'] for x in links),
               'gap_candidate_intervals': len(events), 'review_queue_frames': len(samples),
               'source_files_pinned': len(checked), 'protected_files_unchanged': len(protected),
               'limits': ['Known development data, not unseen generalization', 'Availability is not anatomical correctness', 'No hidden finger/3D truth', 'Sensor tracking is not same-condition full-frame baseline'],
               'new_training': 0, 'new_ml_inference': 0, 'user_recording_access': 0}
    write(U / 'PREPARE_RESULT.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception:
        if not (U / 'FIRST_ERROR.txt').exists():
            (U / 'FIRST_ERROR.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
