"""Independent decode, geometry, pixel and feature recomputation; no producer imports."""
from pathlib import Path
import collections
import hashlib
import json
import math
import time

import av
import numpy as np
from PIL import Image
import torch
from torchvision.models import mobilenet_v3_small

U = Path(__file__).resolve().parent
B = U.parents[1]
OUT = B / 'runs/ksl77_rgb_paired_v1'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def load(p):
    with np.load(p, allow_pickle=False) as f: return {k: f[k] for k in f.files}


def expected_box(points, usable, width, height):
    if not usable.all() or not np.isfinite(points).all(): return None
    if not ((points >= 0).all() and (points[:, 0] < width).all() and (points[:, 1] < height).all()): return None
    p = points.astype(np.float64)
    bounds = [p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]
    extent = max(bounds[2]-bounds[0], bounds[3]-bounds[1])
    if extent <= 0: return None
    size = max(2, math.ceil(1.5*extent))
    left = math.floor((bounds[0]+bounds[2]-size)/2)
    top = math.floor((bounds[1]+bounds[3]-size)/2)
    return (left, top, left+size, top+size)


def main():
    started = time.perf_counter(); torch.set_num_threads(2)
    rows = read(OUT/'MANIFEST.json'); pins = read(OUT/'SOURCE_PINS.json')
    assert all(sha(p) == h for p, h in pins.items())
    weights = read(B/'ops/rgb_pose_bridge_20261010/WEIGHTS.json')
    ref = mobilenet_v3_small(weights=None).eval()
    ref.load_state_dict(torch.load(weights['path'], map_location='cpu', weights_only=True), strict=True)
    ref.requires_grad_(False)
    counts = collections.Counter(); per_split = collections.defaultdict(collections.Counter)
    max_error = 0.; max_pts_error = 0
    expected_fields = {'pose_x', 'frame_id', 'timestamp_us', 'features', 'available', 'observed_rgb', 'region_estimated', 'landmark_observed_count', 'bbox_xyxy'}
    for r in rows:
        for key in ('cache', 'feature', 'pose', 'observation', 'raw', 'frame_ledger'):
            assert sha(r[key+'_path']) == r[key+'_sha256']
        a, f, p, o = (load(r[k+'_path']) for k in ('cache', 'feature', 'pose', 'observation'))
        assert set(f) == expected_fields
        assert np.array_equal(a['pose_x'], p['x'])
        for k in ('frame_id', 'timestamp_us'): assert np.array_equal(a[k], p[k])
        for k in expected_fields-{'features'}: assert np.array_equal(a[k], f[k])
        assert not a['region_estimated'].any()
        assert np.array_equal(a['available'], a['observed_rgb'])
        assert not a['rgb'][~a['available']].any() and not f['features'][~a['available']].any()
        ids = {int(fid): i for i, fid in enumerate(a['frame_id'])}
        obsids = {int(fid): i for i, fid in enumerate(o['frame_id'])}
        ledger = {v['frame_id']: v for v in read(r['frame_ledger_path'])}
        nframes = 0; nimages = 0
        with av.open(r['raw_path']) as container:
            origin_us = None
            for fid, frame in enumerate(container.decode(video=0)):
                raw_us = frame.pts*frame.time_base*1000000
                if origin_us is None: origin_us = raw_us
                if fid not in ids: continue
                i = ids[fid]; oi = obsids[fid]; nframes += 1
                pts = round(raw_us-origin_us)
                err = abs(pts-int(a['timestamp_us'][i])); assert err <= 1
                max_pts_error = max(max_pts_error, err)
                assert o['timestamp_us'][oi] == a['timestamp_us'][i]
                rgb = frame.to_ndarray(format='rgb24')
                assert hashlib.sha256(rgb.tobytes()).hexdigest() == ledger[fid]['raw_rgb_sha256']
                im = Image.fromarray(rgb); mask = Image.new('L', im.size, 255)
                for hand in range(2):
                    sl = slice(33+21*hand, 54+21*hand)
                    box = expected_box(o['xy_px'][oi, sl], o['usable'][oi, sl], im.width, im.height)
                    assert a['landmark_observed_count'][i, hand] == o['observed'][oi, sl].sum()
                    assert bool(a['available'][i, hand]) == (box is not None)
                    if box is None:
                        assert (a['bbox_xyxy'][i, hand] == -1).all()
                        continue
                    assert tuple(a['bbox_xyxy'][i, hand]) == box
                    # PIL.crop independently pads out-of-image pixels black.
                    expected = np.array(im.crop(box).resize((224, 224), Image.Resampling.BILINEAR))
                    valid = np.array(mask.crop(box).resize((224, 224), Image.Resampling.NEAREST)) > 0
                    assert np.array_equal(expected, a['rgb'][i, hand])
                    assert np.array_equal(valid, a['pixel_valid'][i, hand])
                    nimages += 1
        assert nframes == len(a['pose_x']) == r['frames']
        assert nimages == r['available_images']
        slots = np.argwhere(a['available'])
        for start in range(0, len(slots), 13):
            batch = slots[start:start+13]
            pixels = np.stack([a['rgb'][i, h] for i, h in batch])
            images = torch.tensor(pixels.transpose(0, 3, 1, 2), dtype=torch.float32)/255
            for ch, (mean, std) in enumerate(zip((.485, .456, .406), (.229, .224, .225))):
                images[:, ch] = (images[:, ch]-mean)/std
            with torch.inference_mode(): answer = ref.avgpool(ref.features(images)).flatten(1).numpy()
            saved = np.stack([f['features'][i, h] for i, h in batch])
            error = float(np.max(np.abs(answer-saved))); assert error <= 1e-5
            max_error = max(max_error, error)
        numbers = dict(clips=1, frames=nframes, available_images=nimages, missing_slots=2*nframes-nimages)
        counts.update(numbers); per_split[r['split']].update(numbers)
        print('verified', r['sample_id'], flush=True)
    build = read(U/'BUILD_RESULT.json')
    assert dict(counts) == build['counts']
    assert counts['clips'] == 36 and per_split['development34']['clips'] == 34 and per_split['previous2']['clips'] == 2
    assert all(sha(p) == h for p, h in pins.items())
    # Independent parent-membership recount, not the producer's flags.
    rows113 = read(B/'ops/rgb_pose_readiness_20261010/CONFIG_ALL.json')['rows']
    meta = read(B/'ops/context_training_pair_20260921/feature_metadata.json')
    parentids = {r['id'] for r in meta['base']}
    parent_matches = sum(r['sample_id'] in parentids for r in rows113)
    assert parent_matches == read(U/'ANNOTATION_AND_EXPOSURE_AUDIT.json')['counts']['same_native_samples_in_parent_training']
    result = {'status': 'PASS', 'counts': dict(counts), 'per_split': {k: dict(v) for k, v in per_split.items()},
              'same_native_training_samples_recount': parent_matches,
              'max_pts_error_us': max_pts_error, 'max_feature_abs_error': max_error, 'frozen_tolerance': 1e-5,
              'all_raw_pixels_boxes_sides_timestamps_pose_checked': True,
              'all_source_hashes_unchanged': True, 'manifest_sha256': sha(OUT/'MANIFEST.json'),
              'seconds': time.perf_counter()-started,
              'limits': 'Input correspondence only. No new sign accuracy, anatomical identity guarantee or independent-person result.'}
    with (U/'INDEPENDENT_CHECK.json').open('x', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
