"""Independent full pixel/feature check plus direct equality to archived training inputs."""
from pathlib import Path
import collections
import hashlib
import importlib.util
import json
import time
import av
import numpy as np
from PIL import Image
import torch
from torchvision.models import mobilenet_v3_small

U = Path(__file__).resolve().parent; B = U.parents[1]; OUT = B/'runs/nikl18_rgb_paired_v1'
spec = importlib.util.spec_from_file_location('independent_ksl_geometry', B/'ops/rgb_local_development_readiness_20261010/verify.py')
ind = importlib.util.module_from_spec(spec); spec.loader.exec_module(ind)


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def load(p):
    with np.load(p, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def main():
    started = time.perf_counter(); torch.set_num_threads(2)
    rows = read(OUT/'MANIFEST.json'); pins = read(OUT/'SOURCE_PINS.json')
    assert len(rows) == 18 and all(sha(p) == h for p, h in pins.items())
    training = load(B/'ops/motion_input_contract_20260922/training_motion_inputs.npz')
    meta = read(B/'ops/context_training_pair_20260921/feature_metadata.json')
    weights = read(B/'ops/rgb_pose_bridge_20261010/WEIGHTS.json')
    model = mobilenet_v3_small(weights=None).eval()
    model.load_state_dict(torch.load(weights['path'], map_location='cpu', weights_only=True), strict=True)
    model.requires_grad_(False); counts = collections.Counter(); max_error = 0.; max_time_error = 0
    for r in rows:
        for key in ('cache', 'feature', 'observation', 'raw', 'frame_ledger'):
            assert sha(r[key+'_path']) == r[key+'_sha256']
        a, f, o = [load(r[k+'_path']) for k in ('cache', 'feature', 'observation')]
        bi = r['base_training_index']; lo, hi = training['base_offsets'][bi:bi+2]
        assert meta['base'][bi]['id'] == r['sample_id']
        assert np.array_equal(a['pose_x'], training['base_x'][lo:hi])
        # Derive frame membership directly from the original hand mask, without producer's trim call.
        active = np.flatnonzero(o['usable'][:, 33:75].any(1)); begin, end = int(active[0]), int(active[-1])
        assert [begin, end] == r['observation_row_span']
        for key in ('frame_id', 'timestamp_us'): assert np.array_equal(a[key], o[key][begin:end+1])
        expected_fields = {'pose_x', 'frame_id', 'timestamp_us', 'features', 'available', 'observed_rgb', 'region_estimated', 'landmark_observed_count', 'bbox_xyxy'}
        assert set(f) == expected_fields
        for key in expected_fields-{'features'}: assert np.array_equal(a[key], f[key])
        assert not a['region_estimated'].any() and np.array_equal(a['available'], a['observed_rgb'])
        assert not a['rgb'][~a['available']].any() and not a['pixel_valid'][~a['available']].any() and not f['features'][~a['available']].any()
        wanted = {int(fid): i for i, fid in enumerate(a['frame_id'])}
        obindex = {int(fid): i for i, fid in enumerate(o['frame_id'])}
        ledger = {l['frame_id']: l for l in read(r['frame_ledger_path'])}; matched = 0; photos = 0
        with av.open(r['raw_path']) as c:
            origin = None
            for fid, frame in enumerate(c.decode(video=0)):
                raw_us = frame.pts*frame.time_base*1000000
                if origin is None: origin = raw_us
                if fid not in wanted: continue
                i = wanted[fid]; oi = obindex[fid]; matched += 1
                pts = round(raw_us-origin); error = abs(pts-int(a['timestamp_us'][i])); assert error <= 1
                max_time_error = max(max_time_error, error)
                rgb = frame.to_ndarray(format='rgb24'); assert hashlib.sha256(rgb.tobytes()).hexdigest() == ledger[fid]['raw_rgb_sha256']
                image = Image.fromarray(rgb); mask = Image.new('L', image.size, 255)
                for hand in range(2):
                    sl = slice(33+21*hand, 54+21*hand)
                    box = ind.expected_box(o['xy_px'][oi, sl], o['usable'][oi, sl], image.width, image.height)
                    assert bool(a['available'][i, hand]) == (box is not None)
                    assert a['landmark_observed_count'][i, hand] == o['observed'][oi, sl].sum()
                    if box is None:
                        assert (a['bbox_xyxy'][i, hand] == -1).all(); continue
                    assert tuple(a['bbox_xyxy'][i, hand]) == box
                    pixels = np.array(image.crop(box).resize((224, 224), Image.Resampling.BILINEAR))
                    valid = np.array(mask.crop(box).resize((224, 224), Image.Resampling.NEAREST)) > 0
                    assert np.array_equal(pixels, a['rgb'][i, hand]) and np.array_equal(valid, a['pixel_valid'][i, hand]); photos += 1
        assert matched == len(a['pose_x']) == r['frames'] and photos == r['available_images']
        slots = np.argwhere(a['available'])
        for start in range(0, len(slots), 13):
            batch = slots[start:start+13]; rgb = np.stack([a['rgb'][i, hand] for i, hand in batch])
            inputs = torch.tensor(rgb.transpose(0, 3, 1, 2), dtype=torch.float32)/255
            for ch, (mean, std) in enumerate(zip((.485, .456, .406), (.229, .224, .225))): inputs[:, ch] = (inputs[:, ch]-mean)/std
            with torch.inference_mode(): expected = model.avgpool(model.features(inputs)).flatten(1).numpy()
            saved = np.stack([f['features'][i, hand] for i, hand in batch])
            error = float(np.max(np.abs(expected-saved))); assert error <= 1e-5; max_error = max(max_error, error)
        counts.update(clips=1, frames=matched, available_images=photos, missing_slots=2*matched-photos)
        print('verified', r['sample_id'], flush=True)
    assert dict(counts) == read(U/'BUILD_RESULT.json')['counts']
    assert all(sha(p) == h for p, h in pins.items())
    result = {'status': 'PASS', 'counts': dict(counts), 'all18_exactly_match_existing_training_x': True,
              'all_frames_pixels_side_slots_provenance_checked': True, 'max_relative_time_error_us': max_time_error,
              'max_feature_abs_error': max_error, 'frozen_feature_tolerance': 1e-5, 'sources_unchanged': True,
              'manifest_sha256': sha(OUT/'MANIFEST.json'), 'seconds': time.perf_counter()-started,
              'independent_geometry_code_sha256': sha(B/'ops/rgb_local_development_readiness_20261010/verify.py'),
              'no_claim': ['sign accuracy', 'correct hand anatomy', 'new person generalization']}
    with (U/'INDEPENDENT_CHECK.json').open('x', encoding='utf-8') as f: json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
