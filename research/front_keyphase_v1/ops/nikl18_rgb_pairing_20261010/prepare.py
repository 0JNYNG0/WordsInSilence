"""Attach original RGB to the exact existing NIKL training timeline. No optimizer."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys
import traceback
import time
import av
import numpy as np
import torch
from PIL import Image, ImageDraw

U = Path(__file__).resolve().parent
B = U.parents[1]
OUT = B/'runs/nikl18_rgb_paired_v1'
sys.path[:0] = [str(B/'ops/rgb_pose_readiness_20261010'), str(B/'ops/rgb_tensor_adapter_20261010'),
                str(B/'ops/rgb_pose_bridge_20261010'), str(B/'ops/rgb_features_and_browser_20261010'),
                str(B.parent/'sign_model_pilot_v1/src')]
from prepare_rgb_pairs import make_crop, sha, HANDS
from rgb_input import crop_original, validate, pack, scatter_features
from rgb_bridge import FrozenRGBEncoder
from feature_cache import validate as validate_features
from pilot.web_input import to_native_like, active_span
from pilot.data import build_channels


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def load(p):
    with np.load(p, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def write(p, obj):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    start = time.perf_counter(); plan = read(U/'SCOPE.json'); pins = dict(plan['source_pins'])
    assert all(sha(p) == h for p, h in pins.items())
    pins[str(U/'SCOPE.json')] = sha(U/'SCOPE.json'); pins[str(Path(__file__))] = sha(__file__)
    OUT.mkdir(exist_ok=False)
    for sub in ('caches', 'features', 'visual'): (OUT/sub).mkdir()
    source = read(B/'ops/nikl18_hand_image_regression_20261004/PREPARED_CONFIG.json')
    meta = read(B/'ops/context_training_pair_20260921/feature_metadata.json')
    idx = {r['id']: i for i, r in enumerate(meta['base'])}
    training = load(B/'ops/motion_input_contract_20260922/training_motion_inputs.npz')
    weights = read(B/'ops/rgb_pose_bridge_20261010/WEIGHTS.json')
    pins[weights['path']] = weights['sha256']; torch.set_num_threads(2)
    encoder = FrozenRGBEncoder(weights['path'], weights['sha256'])
    manifest = []; totals = Counter()
    for row in source['clips']:
        sid = row['clip_id']; i = idx[sid]
        for key in ('raw', 'baseline_npz', 'baseline_sidecar'):
            p, h = row[key+'_path'], row[key+'_sha256']; assert sha(p) == h; pins[p] = h
        sidecar = read(row['baseline_sidecar_path'])
        assert sidecar['clock'] == 'video PTS * time_base, microseconds from first decoded frame'
        assert not sidecar['mirrored'] and sidecar['rotation_deg'] == 0
        obs = load(row['baseline_npz_path'])
        nat, present = to_native_like(obs['xy_px'], obs['usable']); lo, hi = active_span(present)
        ts = obs['timestamp_us'][lo:hi+1]; ids = obs['frame_id'][lo:hi+1]
        left, right = training['base_offsets'][i:i+2]
        pose = training['base_x'][left:right].copy()
        rebuilt = build_channels(nat[lo:hi+1], present[lo:hi+1], ts)['x']
        assert np.array_equal(rebuilt, pose), sid
        assert training['classes'][training['base_y'][i]] == row['label'] == meta['base'][i]['label']
        n = len(pose)
        a = {'pose_x': pose, 'timestamp_us': ts.copy(), 'frame_id': ids.copy(),
             'rgb': np.zeros((n, 2, 224, 224, 3), np.uint8), 'pixel_valid': np.zeros((n, 2, 224, 224), bool),
             'available': np.zeros((n, 2), bool), 'observed_rgb': np.zeros((n, 2), bool),
             'region_estimated': np.zeros((n, 2), bool), 'landmark_observed_count': np.zeros((n, 2), np.int32),
             'bbox_xyxy': np.full((n, 2, 4), -1, np.int32)}
        wanted = {int(fid): j for j, fid in enumerate(ids)}
        selected = set(np.linspace(0, n-1, 3).round().astype(int)); panels = []; seen = set(); ledger = []; reasons = Counter()
        with av.open(row['raw_path']) as c:
            stream = c.streams.video[0]
            assert int(stream.metadata.get('rotate', 0)) % 360 == 0
            assert stream.sample_aspect_ratio is None or stream.sample_aspect_ratio == 1
            origin = None
            for fid, frame in enumerate(c.decode(stream)):
                assert frame.pts is not None
                if origin is None: origin = frame.pts*frame.time_base
                if fid not in wanted: continue
                j = wanted[fid]; oi = lo+j; seen.add(j)
                assert int(obs['frame_id'][oi]) == fid
                pts = round((frame.pts*frame.time_base-origin)*1000000)
                assert abs(pts-int(ts[j])) <= 1, (sid, fid, pts, int(ts[j]))
                rgb = frame.to_ndarray(format='rgb24')
                assert rgb.shape[:2] == (row['expected_height'], row['expected_width'])
                for hand, sl in enumerate(HANDS):
                    a['landmark_observed_count'][j, hand] = obs['observed'][oi, sl].sum()
                    crop = make_crop(rgb, obs['xy_px'][oi, sl], obs['usable'][oi, sl], obs['xy_px'][oi, [11, 12]])
                    reasons[crop['crop_reason']] += 1
                    if crop['crop_valid']:
                        assert a['landmark_observed_count'][j, hand] == 21
                        a['bbox_xyxy'][j, hand] = crop['bbox_xyxy']
                        a['rgb'][j, hand], a['pixel_valid'][j, hand] = crop_original(rgb, crop['bbox_xyxy'])
                        a['available'][j, hand] = a['observed_rgb'][j, hand] = True
                ledger.append({'frame_id': fid, 'relative_pts_us': pts, 'raw_rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()})
                if j in selected:
                    thumb = Image.fromarray(rgb); thumb.thumbnail((480, 330)); d = ImageDraw.Draw(thumb)
                    for hand in range(2):
                        if a['available'][j, hand]: d.rectangle(tuple(a['bbox_xyxy'][j, hand]*thumb.width/rgb.shape[1]), outline=('red', 'green')[hand], width=3)
                    panels.append((fid, thumb, a['rgb'][j].copy()))
        assert len(seen) == n; validate(a)
        cp = OUT/'caches'/(sid+'.npz')
        with cp.open('xb') as f: np.savez_compressed(f, **a)
        p = pack(a, 'B'); features = scatter_features(encoder(p['images']), p['slots'], n).numpy()
        feat = {k: a[k] for k in ('pose_x', 'frame_id', 'timestamp_us', 'available', 'observed_rgb', 'region_estimated', 'landmark_observed_count', 'bbox_xyxy')}
        feat['features'] = features; validate_features(feat)
        fp = OUT/'features'/(sid+'.npz')
        with fp.open('xb') as f: np.savez_compressed(f, **feat)
        lp = OUT/(sid+'_frames.json'); write(lp, ledger)
        canvas = Image.new('RGB', (480*len(panels), 600), 'white'); d = ImageDraw.Draw(canvas)
        for col, (fid, thumb, hands) in enumerate(panels):
            xx = col*480; d.text((xx+8, 8), f'{sid} frame {fid} | L red / R green', fill='black'); canvas.paste(thumb, (xx, 35))
            for hand in range(2): canvas.paste(Image.fromarray(hands[hand]), (xx+8+238*hand, 365))
        canvas.save(OUT/'visual'/(sid+'.jpg'), quality=90)
        manifest.append({'sample_id': sid, 'label_metadata_only': row['label'], 'class_index_metadata_only': int(training['base_y'][i]),
                         'signer_id': None, 'source': 'NIKL_REFERENCE_ALREADY_IN_PARENT_TRAINING', 'split': 'train_only',
                         'base_training_index': i, 'observation_row_span': [lo, hi], 'frames': n,
                         'cache_path': str(cp), 'cache_sha256': sha(cp), 'feature_path': str(fp), 'feature_sha256': sha(fp),
                         'observation_path': row['baseline_npz_path'], 'observation_sha256': row['baseline_npz_sha256'],
                         'raw_path': row['raw_path'], 'raw_sha256': row['raw_sha256'], 'frame_ledger_path': str(lp), 'frame_ledger_sha256': sha(lp),
                         'available_images': int(a['available'].sum()), 'missing_slots': int((~a['available']).sum()),
                         'crop_reasons': dict(reasons), 'pose_exactly_matches_existing_training': True, 'encoder_sha256': weights['sha256']})
        totals.update(clips=1, frames=n, available_images=int(a['available'].sum()), missing_slots=int((~a['available']).sum()))
        print(sid, n, 'frames', int(a['available'].sum()), 'images', flush=True)
    assert len(manifest) == 18 and all(sha(p) == h for p, h in pins.items())
    write(OUT/'MANIFEST.json', manifest); write(OUT/'SOURCE_PINS.json', pins)
    write(U/'BUILD_RESULT.json', {'status': 'PREPARED_PENDING_INDEPENDENT_VERIFICATION', 'counts': dict(totals),
          'encoder_forward_images': encoder.forward_images, 'seconds': time.perf_counter()-start,
          'bytes': sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()), 'all_sources_unchanged': True,
          'new_landmark_extraction': False, 'training_runs': 0, 'sign_predictions': 0})


if __name__ == '__main__':
    try: main()
    except Exception:
        if not (U/'FIRST_FAILURE.json').exists(): write(U/'FIRST_FAILURE.json', {'traceback': traceback.format_exc()})
        raise
