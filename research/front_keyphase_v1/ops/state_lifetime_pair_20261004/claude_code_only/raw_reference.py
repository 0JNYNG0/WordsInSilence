"""Independent saved-artifact audit only: no model, decoder, MediaPipe or camera imports."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

FIELDS = ('xy_px', 'timestamp_us', 'frame_id', 'observed', 'usable', 'conf', 'hand_score')
TAGS = ('B_video', 'C_hand_image')


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def exact(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def stats(values):
    a = np.asarray(values, np.float64)
    return {'n': len(a), 'p50': float(np.percentile(a, 50)), 'p95': float(np.percentile(a, 95)), 'max': float(a.max())}


def expected_sides(raw, xy, usable, width, height):
    refs = {s: xy[j] for s, j in ((0, 15), (1, 16)) if usable[j]}
    wrists = np.asarray([[h['normalized_xyz'][0][0] * width, h['normalized_xyz'][0][1] * height] for h in raw])
    scores = [h['handedness'][0]['score'] for h in raw]
    labels = [0 if h['handedness'][0]['category_name'] == 'Left' else 1 for h in raw]
    sides, used_hands, used_sides = {}, set(), set()
    if refs:
        choices = [(float(np.linalg.norm(wrists[i] - p)), i, s) for i in range(len(raw)) for s, p in refs.items()]
        choices.sort(key=lambda item: item[0])
        for _, i, s in choices:
            if i not in used_hands and s not in used_sides:
                sides[i] = s
                used_hands.add(i)
                used_sides.add(s)
        sides.update({i: None for i in range(len(raw)) if i not in sides})
    else:
        for i, s in enumerate(labels):
            sides[i] = None if s in used_sides else s
            used_sides.add(s)
    return sides, refs, scores, labels, wrists


def verify_frame(f, z, row, mode, rule):
    fid, ts = int(z['obs_frame_id'][row]), int(z['obs_timestamp_us'][row])
    require((f['row'], f['frame_id'], f['timestamp_us']) == (row, fid, ts), 'diagnostic frame identity')
    di = int(np.flatnonzero(z['decoded_frame_id'] == fid)[0])
    height, width, channels = z['decoded_rgb_shape'][di].tolist()
    require(channels == 3, 'RGB channels')
    ms = ts // 1000
    require(f['pose_ms'] == ms, 'pose timestamp')
    require(f['rgb_sha256_before'] == f['rgb_sha256_after'], 'RGB changed')
    # Diagnostic raw-byte RGB hash intentionally excludes dtype/shape; decoder digest includes them.
    require(len(f['rgb_sha256_before']) == 64, 'RGB diagnostic digest')
    for part, expected_mode, api in (('pose', 'VIDEO', 'detect_for_video'), ('hand', mode, 'detect' if mode == 'IMAGE' else 'detect_for_video')):
        d = f[part]
        require(d['mode'] == expected_mode and d['api'] == api and d['landmarker_ms'] == ms, 'task mode/time')
        require(d['timestamp_ms_passed'] == (None if expected_mode == 'IMAGE' else ms), 'wrong API timestamp')
        require(d['t1_ns'] >= d['t0_ns'] and d['detect_ns'] == d['t1_ns'] - d['t0_ns'], 'timer accounting')
    require(f['process_ns'] >= f['pose']['detect_ns'] + f['hand']['detect_ns'], 'process timing smaller than calls')
    raw = f['hand_raw']
    require(len(raw) == f['n_hands'] and len(raw) <= 2, 'raw detection count')
    for i, hand in enumerate(raw):
        require(hand['index'] == i, 'raw detection order')
        for key in ('normalized_xyz', 'world_xyz'):
            points = np.asarray(hand[key], np.float64)
            require(points.shape == (21, 3) and np.isfinite(points).all(), 'raw hand coordinates')
        require(len(hand['handedness']) >= 1, 'missing handedness')
    sides, refs, scores, labels, wrists = expected_sides(raw, z['obs_xy_px'][row], z['obs_usable'][row], width, height)
    if raw:
        d = f['assignment']
        actual = {a['index']: a['side'] for a in d['index_to_side']}
        require(actual == sides, 'independent assignment disagrees')
        require(all(a['score'] == scores[a['index']] for a in d['index_to_side']), 'assignment score')
        require(d['fallback_raw_handedness'] == (not refs), 'fallback flag')
        require(d['pose_wrist_ref_px'] == {str(s): p.tolist() for s, p in refs.items()}, 'pose wrist reference')
        expected_distances = [{'index': i, 'side': s, 'px': math.dist(wrists[i], refs[s])} for i in range(len(raw)) for s in refs]
        require(d['distances_px'] == expected_distances, 'raw-to-pose distances')
        comparisons = sum(s is not None for s in sides.values()) if refs else 0
        disagreements = sum(s is not None and s != labels[i] for i, s in sides.items()) if refs else 0
        require(d['handedness_comparisons_delta'] == comparisons and d['handedness_disagreements_delta'] == disagreements, 'assignment counters')
    else:
        require(f['assignment'] is None, 'assignment when no hands')
    require(f['unassigned_delta'] == sum(s is None for s in sides.values()), 'unassigned count')
    xy = np.full((42, 2), np.nan, np.float32)
    observed, usable = np.zeros(42, bool), np.zeros(42, bool)
    hs = np.full(2, np.nan, np.float32)
    for i, side in sides.items():
        if side is None:
            continue
        sl = slice(21 * side, 21 * (side + 1))
        pts = np.asarray(raw[i]['normalized_xyz'], np.float64)[:, :2] * np.asarray([width, height], np.float64)
        xy[sl] = pts
        observed[sl] = True
        hs[side] = scores[i]
        margin = rule['bounds_margin_ratio']
        usable[sl] = ((xy[sl, 0] >= -margin * width) & (xy[sl, 0] <= (1 + margin) * width)
                      & (xy[sl, 1] >= -margin * height) & (xy[sl, 1] <= (1 + margin) * height)
                      & (scores[i] >= rule['hand_score_threshold']))
    require(exact(xy, z['obs_xy_px'][row, 33:]), 'raw hand to output xy differs')
    require(exact(observed, z['obs_observed'][row, 33:]), 'raw hand observation mask')
    require(exact(usable, z['obs_usable'][row, 33:]), 'raw hand usable mask')
    require(exact(hs, z['obs_hand_score'][row]), 'raw handedness score')
    require(np.isnan(z['obs_conf'][row, 33:]).all(), 'hand confidence contract')


def audit(config_path, result_dir):
    cfg, out = read(config_path), Path(result_dir).resolve()
    require(out == Path(cfg['output_dir']).resolve(), 'output path')
    for p, h in cfg['pins'].items():
        require(sha(p) == h, 'input pin: ' + p)
    result = read(out / 'RESULT.json')
    require(result['status'] == 'COMPLETED_SELF_REPORTED_PENDING_INDEPENDENT_REVIEW', 'result incomplete')
    manifest = read(out / 'MANIFEST.json')
    require(set(manifest) == {p.name for p in out.iterdir() if p.is_file()} - {'MANIFEST.json'}, 'manifest completeness')
    for name, h in manifest.items():
        require(Path(name).name == name and sha(out / name) == h, 'result artifact hash')
    src = result['source_hashes']
    require(src['config_sha256'] == sha(config_path), 'config hash')
    for field, name in (('runner_sha256', 'hand_image_pair.py'), ('test_sha256', 'test_hand_image.py'), ('test_receipt_sha256', 'TEST_RECEIPT.json')):
        require(src[field] == sha(Path(config_path).parent / name), 'execution source hash')
    require(all(not errors for errors in result['gates'].values()), 'recorded gate failure')
    rule = read(cfg['extraction']['contract_path'])['usable_rule']
    arrays, diagnostics, clips, counts = {}, {}, [], Counter()
    for clip in cfg['clips']:
        cid = clip['clip_id']
        prior = np.load(clip['baseline_b_npz'], allow_pickle=False)
        zs = {}
        for tag, mode in zip(TAGS, ('VIDEO', 'IMAGE')):
            files = result['clips'][cid]['artifacts'][tag]
            with np.load(out / files['npz'], allow_pickle=False) as npz:
                z = {k: npz[k].copy() for k in npz.files}
            zs[tag] = z
            meta = read(out / files['meta'])
            require(meta['npz_sha256'] == sha(out / files['npz']), 'NPZ metadata hash')
            require(meta['clip_id'] == cid and meta['tag'] == tag and meta['arm'] == 'B_even_rgb_only', 'NPZ metadata identity')
            ids = np.arange(0, clip['expected_frames_from_prior'], 2, dtype=np.int64)
            require(exact(z['obs_frame_id'], ids) and exact(z['selected_frame_id'], ids), 'even original IDs')
            require(exact(z['obs_timestamp_us'], z['decoded_timestamp_us'][ids]), 'selected real timestamps')
            require(len(ids) == clip['expected_selected_even_frames'], 'selected row count')
            diag = read(out / (Path(files['npz']).stem + '.diag.json'))
            require(diag['clip_id'] == cid and diag['tag'] == tag and diag['errors'] == [] and not diag['rgb_content_stored'], 'diagnostic identity/errors')
            require(len(diag['frames']) == len(ids), 'diagnostic row count')
            prov = diag['provenance']
            require(prov['pose_running_mode'] == 'VIDEO' and prov['hand_running_mode'] == mode and prov['wrapper_sha256'] == src['runner_sha256'], 'provenance mode/hash')
            require(prov['hand_options'] == dict(num_hands=2, min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5), 'hand options')
            require(diag['construction']['base_factory_ns'] > 0, 'base construction timer')
            require((diag['construction']['image_create_ns'] > 0) if mode == 'IMAGE' else diag['construction']['image_create_ns'] is None, 'image construction timer')
            expected_tasks = ['pose_VIDEO', 'hand_VIDEO'] + (['hand_IMAGE'] if mode == 'IMAGE' else [])
            require([t['task'] for t in diag['tasks']] == expected_tasks, 'task creation order')
            for task in diag['tasks']:
                expected_calls = 0 if mode == 'IMAGE' and task['task'] == 'hand_VIDEO' else len(ids)
                require(task['calls'] == expected_calls, 'task calls')
                require(task['close'] == {'attempted': True, 'ok': True, 'error': None}, 'task close')
                prefix = task['task'].lower()
                counts[prefix + '_tasks'] += 1
                counts[prefix + '_calls'] += task['calls']
                counts[prefix + '_closed_ok'] += 1
                counts[prefix + '_zero_call_tasks'] += task['calls'] == 0
            for row, frame in enumerate(diag['frames']):
                verify_frame(frame, z, row, mode, rule)
            arrays[cid, tag], diagnostics[cid, tag] = z, diag
            if tag == 'B_video':
                for key in prior.files:
                    require(key in z and exact(z[key], prior[key]), 'B differs from old raw: ' + key)
        prior.close()
        b, c = (zs[t] for t in TAGS)
        for key in b:
            if key.startswith('decoded_') or key in ('obs_frame_id', 'obs_timestamp_us', 'selected_frame_id'):
                require(exact(b[key], c[key]), 'B/C frame metadata mismatch')
        for field in ('xy_px', 'observed', 'usable', 'conf'):
            require(exact(b['obs_' + field][:, :33], c['obs_' + field][:, :33]), 'B/C pose differs')
        clips.append({'clip_id': cid, 'rows_per_arm': len(b['obs_frame_id']), 'B_all_prior_arrays_byte_exact': True,
                      'BC_pose_and_RGB_ID_PTS_exact': True, 'raw_hand_to_assignment_to_observation_exact': True})
    require(dict(counts) == result['task_counts'], 'task totals')
    expected_counts = {'decoded_frames': 1706, 'landmarker_process_calls': 856, 'landmarker_constructed': 12,
                       'landmarker_closed': 12, 'decoder_opened': 12, 'model_loads': 1, 'tcn_forwards': 8,
                       'camera_calls': 0, 'training_runs': 0, 'downloads': 0}
    require(result['counters'] == expected_counts, 'run counts')
    for tag in TAGS:
        ds = [diagnostics[clip['clip_id'], tag] for clip in cfg['clips']]
        steady = [f for d in ds for f in d['frames'][1:]]
        t = result['timing'][tag]
        require(t['construction_per_landmarker'] == [d['construction'] for d in ds], 'construction timing summary')
        for part in ('pose', 'hand'):
            require(t['steady'][part] == stats([f[part]['detect_ns'] for f in steady]), 'detect percentile summary')
        require(t['steady']['process'] == stats([f['process_ns'] for f in steady]), 'process percentile summary')
        require(t['first_call_per_landmarker'] == [{'pose_ns': d['frames'][0]['pose']['detect_ns'], 'hand_ns': d['frames'][0]['hand']['detect_ns'], 'process_ns': d['frames'][0]['process_ns']} for d in ds], 'first call timing summary')
    require(not any(k in sys.modules for k in ('torch', 'mediapipe', 'av', 'cv2')), 'forbidden verification import')
    for path, h in cfg['pins'].items():
        require(sha(path) == h, 'input changed during audit')
    for name, h in manifest.items():
        require(sha(out / name) == h, 'result changed during audit')
    return {'status': 'PASS_ARTIFACT_AUDIT_NO_MODEL_REINFERENCE', 'config_sha256': sha(config_path),
            'result_sha256': sha(out / 'RESULT.json'), 'pins_checked': len(cfg['pins']), 'result_files_checked': len(manifest),
            'clips': clips, 'frames_with_raw_hand_assignment_audit': 856, 'task_instances_closed': 30,
            'reported_counts_verified': expected_counts, 'task_counts_verified': dict(counts),
            'timing_percentiles_verified': True, 'verification_execution': {'model_forwards': 0, 'model_loads': 0,
            'video_decodes': 0, 'mediapipe_calls': 0, 'camera_calls': 0, 'training': 0},
            'limits': ['Raw assignment consistency is not anatomical ground truth.', 'Model logits independently verified by a separate agent.',
                       'Detection/side-set counts do not verify temporal identity.', 'Offline timer agreement is not deployment FPS validation.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--result', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    report = audit(a.config, a.result)
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k not in ('clips', 'task_counts_verified')}, ensure_ascii=True))
