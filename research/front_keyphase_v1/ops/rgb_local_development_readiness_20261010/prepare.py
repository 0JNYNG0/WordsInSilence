"""Offline input preparation and provenance audit. No sign scoring or training."""
from pathlib import Path
import collections
import hashlib
import json
import sys
import time
import traceback
import zipfile

import av
import numpy as np
import torch
from PIL import Image, ImageDraw

U = Path(__file__).resolve().parent
B = U.parents[1]
OUT = B / 'runs/ksl77_rgb_paired_v1'
sys.path[:0] = [str(B / 'ops/rgb_pose_readiness_20261010'),
                str(B / 'ops/rgb_tensor_adapter_20261010'),
                str(B / 'ops/rgb_pose_bridge_20261010'),
                str(B / 'ops/rgb_features_and_browser_20261010')]
from prepare_rgb_pairs import make_crop, sha, load_npz, HANDS
from rgb_input import crop_original, validate, pack, scatter_features
from rgb_bridge import FrozenRGBEncoder
from feature_cache import validate as validate_features


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, data):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


def audit113():
    config = read(B / 'ops/rgb_pose_readiness_20261010/CONFIG_ALL.json')
    native = {r['sample_id']: r for r in read(B.parent / 'sign_model_pilot_v1/expand/manifest.json')}
    meta = read(B / 'ops/context_training_pair_20260921/feature_metadata.json')
    parent = {r['id']: r for r in meta['base']}
    protected = {r['id'].split('_a')[0] for r in meta['annotation_dev'] + meta['whole_dev']}
    archive = Path('C:/Users/local-user/Downloads/수어 영상/1.Training/[라벨]01_real_sen_morpheme.zip')
    results = []
    with zipfile.ZipFile(archive) as z:
        for row in config['rows']:
            sid = row['sample_id']; n = native[sid]
            member = f"morpheme/{row['signer_id']:02d}/{row['clip_id']}_morpheme.json"
            raw = z.read(member)
            ann = json.loads(raw)['data'][n['annotation_index']]
            original = load_npz(row['native_source'])
            a = load_npz(row['input_path'])
            ts = a['timestamp_us'] / 1e6
            checks = {
                'official_label': ann['attributes'][0]['name'] == row['label'] == n['word_label'],
                'official_start': abs(ann['start'] - row['start_s']) < 1e-6,
                'official_end': abs(ann['end'] - row['end_s']) < 1e-6,
                'official_window_contains_all_pts': bool(((ts >= ann['start']) & (ts < ann['end'])).all()),
                'native_hash': sha(row['native_source']) == n['npz_sha256'],
                'saved_pose_hash': sha(row['input_path']) == row['input_sha256'],
                'same_native_frame_ids': np.array_equal(a['frame_id'], original['frame_index']),
                'frame_count_and_extents': len(a['x']) == n['frame_count'] and int(a['frame_id'][0]) == n['first_frame'] and int(a['frame_id'][-1]) == n['last_frame'],
                'source_training_role': n['split_role'] == 'train',
                'no_protected_development_origin': row['clip_id'] not in protected,
            }
            assert all(checks.values()), (sid, checks)
            exposed = sid in parent
            if exposed:
                assert parent[sid]['sha256'] == n['npz_sha256']
            results.append({'sample_id': sid, 'label': row['label'], 'signer_id': row['signer_id'],
                            'clip_id': row['clip_id'], 'annotation_member': member,
                            'annotation_member_sha256': hashlib.sha256(raw).hexdigest(),
                            'annotation_index': n['annotation_index'], 'checks': checks,
                            'same_native_sample_in_parent_training': exposed,
                            'parent_training_path': parent[sid]['path'] if exposed else None,
                            'semantic_expert_review': False})
    label_counts = collections.Counter(r['label'] for r in results)
    result = {'status': 'PASS_SOURCE_AND_TIMING_AUDIT', 'rows': results,
              'counts': {'segments': len(results), 'origins': len({r['clip_id'] for r in results}),
                         'classes': len(label_counts), 'people': dict(collections.Counter(r['signer_id'] for r in results)),
                         'same_native_samples_in_parent_training': sum(r['same_native_sample_in_parent_training'] for r in results),
                         'classes_with_one_segment': sum(v == 1 for v in label_counts.values()),
                         'classes_with_two_segments': sum(v == 2 for v in label_counts.values()),
                         'study_eat_segments': sum(r['label'] in ['공부', '먹다'] for r in results)},
              'interpretation': 'Prepared MP arrays were not used by832; their underlying native-coordinate samples WERE used when parent membership is true. They may train an auxiliary branch, but cannot become independent whole-system evaluation by splitting the prepared cache.',
              'semantic_limit': 'Official label and annotated timing verified; sign form and correctness not independently certified.'}
    write(U / 'ANNOTATION_AND_EXPOSURE_AUDIT.json', result)
    print('113 audit:', result['counts'], flush=True)


def allocate(x):
    n = len(x['x'])
    return {'pose_x': x['x'].copy(), 'frame_id': x['frame_id'].copy(), 'timestamp_us': x['timestamp_us'].copy(),
            'rgb': np.zeros((n, 2, 224, 224, 3), np.uint8),
            'pixel_valid': np.zeros((n, 2, 224, 224), bool),
            'available': np.zeros((n, 2), bool), 'observed_rgb': np.zeros((n, 2), bool),
            'region_estimated': np.zeros((n, 2), bool),
            'landmark_observed_count': np.zeros((n, 2), np.int32),
            'bbox_xyxy': np.full((n, 2, 4), -1, np.int32)}


def main():
    plan = read(U / 'SCOPE.json')
    assert all(sha(p) == h for p, h in plan['source_pins'].items())
    resume = '--resume-clock-repair' in sys.argv
    if resume:
        recovery = read(U/'CLOCK_CONTRACT_RECOVERY.json')
        assert not (OUT/'MANIFEST.json').exists()
        assert all(sha(p) == h for p, h in recovery['completed_outputs_before_repair'].items())
    else:
        OUT.mkdir(exist_ok=False); (OUT / 'caches').mkdir(); (OUT / 'features').mkdir(); (OUT / 'visual').mkdir()
    pins = dict(plan['source_pins']); pins[str(U / 'SCOPE.json')] = sha(U / 'SCOPE.json'); pins[str(Path(__file__))] = sha(__file__)
    if not resume: audit113()
    else:
        assert read(U/'ANNOTATION_AND_EXPOSURE_AUDIT.json')['status'] == 'PASS_SOURCE_AND_TIMING_AUDIT'
        pins[str(U/'CLOCK_CONTRACT_RECOVERY.json')] = sha(U/'CLOCK_CONTRACT_RECOVERY.json')
    original_plan = read(B / 'ops/ksl77_development_probe_20260926/FROZEN_PLAN.json')
    extraction = {r['id']: r for r in read(B / 'ops/ksl77_development_probe_20260926/EXTRACTION.json')['rows']}
    old_results = {r['id']: r for r in read(B / 'ops/ksl77_development_probe_20260926/RESULT.json')['rows']}
    weights = read(B / 'ops/rgb_pose_bridge_20261010/WEIGHTS.json')
    torch.set_num_threads(2)
    encoder = FrozenRGBEncoder(weights['path'], weights['sha256'])
    pins[weights['path']] = weights['sha256']
    manifest = []; totals = collections.Counter(); start = time.perf_counter()
    for row in original_plan['rows']:
        sid = row['id']; e = extraction[sid]; old = old_results[sid]
        ip = B / 'runs/ksl77_development_probe_v1' / sid / 'model_input.npz'
        for p, h in [(row['path'], row['source_sha256']), (e['observation'], e['observation_sha256']), (str(ip), old['input_sha256'])]:
            assert sha(p) == h, p
            pins[p] = h
        sidecar_path = Path(e['observation']).with_suffix('.sidecar.json')
        assert sha(sidecar_path) == e['sidecar_sha256']
        assert read(sidecar_path)['clock'] == 'video PTS * time_base, microseconds from first decoded frame'
        pins[str(sidecar_path)] = e['sidecar_sha256']
        cp = OUT/'caches'/(sid+'.npz'); fp = OUT/'features'/(sid+'.npz'); lp = OUT/(sid+'_frames.json')
        if resume and cp.exists():
            assert fp.exists() and lp.exists() and (OUT/'visual'/(sid+'.jpg')).exists()
            a = load_npz(cp); feat = load_npz(fp); validate(a); validate_features(feat)
            original = load_npz(ip)
            assert np.array_equal(a['pose_x'], original['x'])
            for k in ('frame_id', 'timestamp_us'): assert np.array_equal(a[k], original[k])
            n = len(a['pose_x'])
            m = {'sample_id': sid, 'cache_path': str(cp), 'cache_sha256': sha(cp), 'feature_path': str(fp), 'feature_sha256': sha(fp),
                 'pose_path': str(ip), 'pose_sha256': old['input_sha256'], 'observation_path': e['observation'],
                 'observation_sha256': e['observation_sha256'], 'raw_path': row['path'], 'raw_sha256': row['source_sha256'],
                 'frame_ledger_path': str(lp), 'frame_ledger_sha256': sha(lp), 'label_metadata_only': row['candidate_label'],
                 'signer_id': None, 'split': 'previous2' if row['previous_diagnostic_exposure'] else 'development34',
                 'training_use': False, 'frames': n, 'available_images': int(a['available'].sum()),
                 'missing_slots': int((~a['available']).sum()), 'estimated_images': 0,
                 'encoder_sha256': weights['sha256'], 'reused_before_clock_repair': True}
            manifest.append(m); totals.update(frames=n, available_images=m['available_images'], missing_slots=m['missing_slots'], clips=1)
            print('reuse prepared', sid, flush=True)
            continue
        obs = load_npz(e['observation']); x = load_npz(ip); a = allocate(x)
        ids = a['frame_id']; ts = a['timestamp_us']; n = len(ids)
        lookup = {int(f): k for k, f in enumerate(obs['frame_id'])}
        oi = np.array([lookup[int(f)] for f in ids])
        assert np.array_equal(ts, obs['timestamp_us'][oi])
        fid_to_row = {int(f): j for j, f in enumerate(ids)}
        reasons = collections.Counter(); ledger = []; seen = set(); panels = []
        preview_rows = set(np.linspace(0, n - 1, 3).round().astype(int).tolist())
        dimensions = set()
        with av.open(row['path']) as container:
            stream = container.streams.video[0]
            assert int(stream.metadata.get('rotate', 0)) % 360 == 0
            assert stream.sample_aspect_ratio is None or stream.sample_aspect_ratio == 1
            origin = None
            for fid, frame in enumerate(container.decode(stream)):
                assert frame.pts is not None
                if origin is None: origin = frame.pts*frame.time_base
                if fid not in fid_to_row:
                    continue
                j = fid_to_row[fid]; assert frame.pts is not None
                pts = round((frame.pts * frame.time_base-origin) * 1000000)
                assert abs(pts - int(ts[j])) <= 1, (sid, fid, pts, int(ts[j]))
                image = frame.to_ndarray(format='rgb24'); dimensions.add(image.shape[:2])
                seen.add(j); i = oi[j]
                for hand, sl in enumerate(HANDS):
                    a['landmark_observed_count'][j, hand] = int(obs['observed'][i, sl].sum())
                    crop = make_crop(image, obs['xy_px'][i, sl], obs['usable'][i, sl], obs['xy_px'][i, [11, 12]])
                    reasons[crop['crop_reason']] += 1
                    if crop['crop_valid']:
                        assert a['landmark_observed_count'][j, hand] == 21
                        a['bbox_xyxy'][j, hand] = crop['bbox_xyxy']
                        a['rgb'][j, hand], a['pixel_valid'][j, hand] = crop_original(image, crop['bbox_xyxy'])
                        a['available'][j, hand] = a['observed_rgb'][j, hand] = True
                ledger.append({'frame_id': fid, 'pts_us': pts, 'absolute_pts_us': round(frame.pts*frame.time_base*1000000),
                               'origin_pts_us': round(origin*1000000), 'raw_rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest()})
                if j in preview_rows:
                    thumb = Image.fromarray(image); thumb.thumbnail((480, 330))
                    d = ImageDraw.Draw(thumb); scale = thumb.width / image.shape[1]
                    for hand in range(2):
                        if a['available'][j, hand]:
                            d.rectangle(tuple(a['bbox_xyxy'][j, hand] * scale), outline=('red', 'green')[hand], width=3)
                    panels.append((fid, thumb, a['rgb'][j].copy()))
        assert seen == set(range(n)) and len(dimensions) == 1
        validate(a)
        cp = OUT / 'caches' / (sid + '.npz')
        with cp.open('xb') as f: np.savez_compressed(f, **a)
        packed = pack(a, 'B')
        features = scatter_features(encoder(packed['images']), packed['slots'], n).numpy()
        feat = {k: a[k] for k in ['pose_x', 'frame_id', 'timestamp_us', 'available', 'observed_rgb', 'region_estimated', 'landmark_observed_count', 'bbox_xyxy']}
        feat['features'] = features; validate_features(feat)
        fp = OUT / 'features' / (sid + '.npz')
        with fp.open('xb') as f: np.savez_compressed(f, **feat)
        canvas = Image.new('RGB', (480 * len(panels), 600), 'white'); draw = ImageDraw.Draw(canvas)
        for col, (fid, thumb, hands) in enumerate(panels):
            xx = col * 480; draw.text((xx + 10, 8), f'{sid} frame {fid} | L red / R green', fill='black')
            canvas.paste(thumb, (xx, 35))
            for hand in range(2): canvas.paste(Image.fromarray(hands[hand]), (xx + 8 + 238 * hand, 365))
        canvas.save(OUT / 'visual' / (sid + '.jpg'), quality=90)
        lp = OUT / (sid + '_frames.json'); write(lp, ledger)
        m = {'sample_id': sid, 'cache_path': str(cp), 'cache_sha256': sha(cp), 'feature_path': str(fp), 'feature_sha256': sha(fp),
             'pose_path': str(ip), 'pose_sha256': old['input_sha256'], 'observation_path': e['observation'],
             'observation_sha256': e['observation_sha256'], 'raw_path': row['path'], 'raw_sha256': row['source_sha256'],
             'frame_ledger_path': str(lp), 'frame_ledger_sha256': sha(lp),
             'label_metadata_only': row['candidate_label'], 'signer_id': None,
             'split': 'previous2' if row['previous_diagnostic_exposure'] else 'development34',
             'training_use': False, 'frames': n, 'crop_reasons': dict(reasons),
             'available_images': int(a['available'].sum()), 'missing_slots': int((~a['available']).sum()),
             'estimated_images': 0, 'encoder_sha256': weights['sha256']}
        manifest.append(m); totals.update(frames=n, available_images=m['available_images'], missing_slots=m['missing_slots'], clips=1)
        write(OUT / 'progress.json.tmp', {'completed': len(manifest), 'expected': 36})
        (OUT / 'progress.json.tmp').replace(OUT / 'progress.json')
        print(sid, n, 'frames,', m['available_images'], 'images', flush=True)
    assert len(manifest) == 36 and sum(m['split'] == 'development34' for m in manifest) == 34
    assert all(sha(p) == h for p, h in pins.items())
    write(OUT / 'SOURCE_PINS.json', pins); write(OUT / 'MANIFEST.json', manifest)
    write(U / 'BUILD_RESULT.json', {'status': 'PREPARED_PENDING_INDEPENDENT_VERIFICATION', 'counts': dict(totals),
           'encoder_forward_images': encoder.forward_images, 'seconds': time.perf_counter() - start,
           'reused_prepared_clips': sum(m.get('reused_before_clock_repair', False) for m in manifest),
           'all_sources_unchanged': True, 'sign_predictions': 0, 'training_runs': 0,
           'bytes': sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())})


if __name__ == '__main__':
    try: main()
    except Exception:
        p = U / 'FIRST_FAILURE.json'
        if not p.exists(): write(p, {'traceback': traceback.format_exc()})
        raise
