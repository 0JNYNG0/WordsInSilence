"""Read-only saved-artifact audit. Never imports a model, decoder, MediaPipe, or runner."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'hand_image_mode_pair_20261004' / 'independent_artifact_verify.py'
spec = importlib.util.spec_from_file_location('verified_hand_raw_audit', OLD)
raw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raw)
require, read, sha, exact, stats = raw.require, raw.read, raw.sha, raw.exact, raw.stats
FIELDS, TAGS = raw.FIELDS, raw.TAGS


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def audit(config_path, result_dir):
    cfg, out = read(config_path), Path(result_dir).resolve()
    require(out == Path(cfg['output_dir']).resolve(), 'output path')
    before = {p: sha(p) for p in cfg['pins']}
    require(before == cfg['pins'], 'input pins before audit')
    res = read(out / 'RESULT.json')
    require(res['status'] == 'COMPLETED_PENDING_INDEPENDENT_VERIFICATION', 'result incomplete')
    manifest = read(out / 'MANIFEST.json')['files']
    require(set(manifest) == {p.name for p in out.iterdir() if p.is_file()} - {'MANIFEST.json'}, 'manifest completeness')
    for name, digest in manifest.items():
        require(Path(name).name == name and sha(out / name) == digest, 'artifact hash ' + name)
    prov = res['provenance']
    require(prov['config_sha256'] == sha(config_path), 'config hash')
    for field, name in (('runner_sha256', 'nikl18_pair.py'), ('test_sha256', 'test_nikl18.py'),
                        ('receipt_sha256', 'TEST_RECEIPT.json')):
        require(prov[field] == sha(HERE / name), 'executed source/receipt ' + field)
    for field, key in (('cadence_module_sha256', 'cadence_module_path'), ('hand_image_module_sha256', 'hand_image_module_path')):
        require(prov[field] == sha(cfg['reuse'][key]), 'helper hash')
    require(res['hashes_before'] == res['hashes_after'], 'run input hashes changed')
    state = res['model_state']
    require(state['before'] == state['after'] == state['payload_state_sha256']
            == res['model']['payload_state_sha256'] and state['unchanged'], 'reported model state preservation')
    require(res['evaluation_selection'] == 'all_stored_obs', 'wrong scoring selection')
    require([r['clip_id'] for r in res['rows']] == [c['clip_id'] for c in cfg['clips']], 'result clip order/set')
    require(len(cfg['clips']) == 18 and sum(c['expected_frames'] for c in cfg['clips']) == 1337, 'fixed population')
    rule = read(cfg['extraction']['contract_path'])['usable_rule']
    all_diag, counts, clips = {}, Counter(), []
    for clip, row_result in zip(cfg['clips'], res['rows']):
        cid, n = clip['clip_id'], clip['expected_frames']
        require(row_result['evaluation_selection'] == 'all_stored_obs' and row_result['n_frames'] == n, 'full row metadata')
        prior = load_npz(clip['baseline_npz_path'])
        require(str(prior['schema']) == 'wis_observation_2d_v1' and set(prior) == {'schema', *FIELDS}, 'cache schema/fields')
        arms = {}
        for tag, mode in zip(TAGS, ('VIDEO', 'IMAGE')):
            files = row_result['persisted'][tag]
            z = load_npz(out / files['npz'])
            meta = read(out / files['meta'])
            ids = np.arange(n, dtype=np.int64)
            require(meta['clip_id'] == cid and meta['tag'] == tag and meta['arm'] == 'A_full_then_even', 'raw metadata identity')
            require(meta['npz_sha256'] == sha(out / files['npz']), 'NPZ metadata hash')
            require(exact(z['obs_frame_id'], ids) and exact(z['decoded_frame_id'], ids), 'full original IDs')
            require(exact(z['selected_frame_id'], ids[::2]), 'internal even bookkeeping')
            require(exact(z['obs_timestamp_us'], z['decoded_timestamp_us']), 'full real timestamps')
            require(exact(z['decoded_processed'], np.ones(n, bool)), 'all frames processed')
            require(meta['counts'] == {'decoded_frames': n, 'process_calls': n, 'stored_rows': n,
                                       'selected_rows': len(ids[::2])}, 'full/even helper counters')
            tb = Fraction(meta['time_base'])
            require(str(tb) == clip['time_base'], 'time base')
            pts = z['decoded_pts']
            want_ts = np.asarray([int(round(float(int(p - pts[0]) * tb) * 1e6)) for p in pts], np.int64)
            require(exact(want_ts, z['obs_timestamp_us']) and np.all(np.diff(pts) > 0), 'PTS-derived clock')
            require(exact(z['decoded_rgb_shape'], np.tile(np.asarray([clip['expected_height'], clip['expected_width'], 3], np.int64), (n, 1))), 'RGB dimensions')
            require(all(len(str(s)) == 64 for s in z['decoded_rgb_sha256']), 'decoded RGB digest shape')
            diag = read(out / (Path(files['npz']).stem + '.instr.json'))
            require(diag['tag'] == tag and diag['mode'] == mode and diag['errors'] == [], 'diagnostic tag/mode/errors')
            require(len(diag['frames']) == n, 'diagnostic full row count')
            dp = diag['provenance']
            require(dp['pose_running_mode'] == 'VIDEO' and dp['hand_running_mode'] == mode, 'mode provenance')
            require(dp['wrapper_sha256'] == prov['hand_image_module_sha256'], 'reused wrapper provenance')
            require(dp['hand_options'] == dict(num_hands=2, min_hand_detection_confidence=0.5,
                                               min_hand_presence_confidence=0.5, min_tracking_confidence=0.5), 'options')
            require(diag['construction']['base_factory_ns'] > 0, 'base construction timer')
            require((diag['construction']['image_create_ns'] > 0) if mode == 'IMAGE'
                    else diag['construction']['image_create_ns'] is None, 'IMAGE construction timer')
            tasks = ['pose_VIDEO', 'hand_VIDEO'] + (['hand_IMAGE'] if mode == 'IMAGE' else [])
            require([t['task'] for t in diag['tasks']] == tasks, 'task creation order')
            for task in diag['tasks']:
                calls = 0 if mode == 'IMAGE' and task['task'] == 'hand_VIDEO' else n
                require(task['calls'] == calls, 'task actual calls')
                require(task['close'] == {'attempted': True, 'ok': True, 'error': None}, 'task close')
                prefix = task['task'].lower()
                for suffix, value in (('_tasks', 1), ('_calls', calls), ('_closed_ok', 1), ('_zero_call_tasks', int(calls == 0))):
                    counts[prefix + suffix] += value
            for ri, frame in enumerate(diag['frames']):
                raw.verify_frame(frame, z, ri, mode, rule)
            if tag == 'B_video':
                for field in FIELDS:
                    require(exact(z['obs_' + field], prior[field]), cid + ' B cache mismatch ' + field)
            all_diag[cid, tag], arms[tag] = diag, z
        b, c = (arms[t] for t in TAGS)
        for field in b:
            if field.startswith('decoded_') or field in ('obs_frame_id', 'obs_timestamp_us', 'selected_frame_id'):
                require(exact(b[field], c[field]), cid + ' B/C metadata ' + field)
        for field in ('xy_px', 'observed', 'usable', 'conf'):
            require(exact(b['obs_' + field][:, :33], c['obs_' + field][:, :33]), cid + ' B/C pose ' + field)
        # The independent model reviewer checks actual preprocessing and logits; verify span identity here.
        for tag, score_key in zip(TAGS, ('B', 'C')):
            z, scored = arms[tag], row_result[score_key]
            valid = z['obs_usable'][:, 33:].any(1)
            hits = np.flatnonzero(valid)
            if not len(hits):
                require(score_key == 'C' and scored['status'] == 'UNAVAILABLE_NO_ACTIVE_HANDS', 'unavailable omitted')
                require('logits' not in scored and 'score' not in scored, 'unavailable scored')
            else:
                lo, hi = int(hits[0]), int(hits[-1])
                inp = scored['input']
                require(inp['span'] == [lo, hi] and inp['frame_id'] == list(range(lo, hi + 1)), 'active span IDs')
                require(inp['timestamp_us'] == z['obs_timestamp_us'][lo:hi + 1].tolist(), 'active span timestamps')
                require(inp['x_shape'] == [1, hi - lo + 1, 280], 'input full-span shape')
                lg = np.asarray(scored['logits'], np.float64)
                require(lg.shape == (93,) and np.isfinite(lg).all(), 'reported logits shape/finite')
        clips.append({'clip_id': cid, 'rows_per_arm': n, 'B_cache_seven_fields_byte_exact': True,
                      'BC_pose_RGB_ID_PTS_exact': True, 'raw_assignment_to_observation_exact': True})
    require(dict(counts) == res['task_counts'], 'task totals')
    unavailable = sum(r['C']['status'] == 'UNAVAILABLE_NO_ACTIVE_HANDS' for r in res['rows'])
    expected_counts = {'decoded_frames': 2674, 'landmarker_process_calls': 2674, 'landmarker_constructed': 36,
                       'landmarker_closed': 36, 'decoder_opened': 36, 'model_loads': 1, 'tcn_forwards': 36 - unavailable,
                       'camera_calls': 0, 'training_runs': 0, 'downloads': 0}
    require(res['counters'] == expected_counts, 'actual execution counters')
    expected_actual = {k: v for k, v in cfg['budgets'].items()
                       if k not in ('selected_frames_per_arm', 'forward_calls_max', 'model_loads', 'independent_forward_calls')}
    expected_actual.update(forward_calls=36 - unavailable)
    expected_actual.update({f'selected_frames_per_arm:{tag}': 1337 for tag in TAGS})
    require(res['actual_counts'] == expected_actual, 'helper actual budget mapping')
    for tag in TAGS:
        ds = [all_diag[c['clip_id'], tag] for c in cfg['clips']]
        steady = [f for d in ds for f in d['frames'][1:]]
        t = res['timing'][tag]
        require(t['construction_per_landmarker'] == [d['construction'] for d in ds], 'construction summary')
        for part in ('pose', 'hand'):
            require(t['steady'][part] == stats([f[part]['detect_ns'] for f in steady]), 'task timer summary')
        require(t['steady']['process'] == stats([f['process_ns'] for f in steady]), 'process timer summary')
        require(t['first_call_per_landmarker'] == [{'pose_ns': d['frames'][0]['pose']['detect_ns'],
                'hand_ns': d['frames'][0]['hand']['detect_ns'], 'process_ns': d['frames'][0]['process_ns']}
                for d in ds], 'first-frame timer summary')
    require(not any(k in sys.modules for k in ('torch', 'mediapipe', 'av', 'cv2', 'wis_lab', 'pilot')), 'forbidden audit imports')
    require({p: sha(p) for p in cfg['pins']} == before, 'input changed during audit')
    require(all(sha(out / n) == h for n, h in manifest.items()), 'result changed during audit')
    return {'status': 'PASS_ARTIFACT_AUDIT_NO_MODEL_REINFERENCE', 'config_sha256': sha(config_path),
            'result_sha256': sha(out / 'RESULT.json'), 'verifier_sha256': sha(__file__),
            'reused_verifier_sha256': sha(OLD), 'pins_checked': len(before), 'artifact_files_checked': len(manifest),
            'clips': clips, 'raw_frames_independently_audited': 2674, 'tasks_closed': 90,
            'counters_verified': expected_counts, 'task_counts_verified': dict(counts),
            'timing_summaries_verified': True, 'model_forward': 0, 'model_load': 0,
            'video_decode': 0, 'mediapipe_calls': 0, 'camera_calls': 0, 'training': 0,
            'limits': ['Assignment consistency does not establish anatomical coordinate correctness.',
                       'NIKL18 are training/reference demonstrations, not independent generalization data.',
                       'Model logits are checked separately; this audit does not execute a model.',
                       'Offline timings do not establish actual webcam FPS.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--result', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    report = audit(args.config, args.result)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k not in ('clips', 'task_counts_verified')}, ensure_ascii=False))
