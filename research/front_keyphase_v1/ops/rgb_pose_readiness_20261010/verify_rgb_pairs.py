"""Independent output audit: no import of the preparation implementation."""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
from PIL import Image


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def geometry(xy, mask, width, height):
    count = sum(bool(v) for v in mask)
    if count == 0: return 'missing', None
    if count < 21: return 'partial', None
    pairs = [(float(x), float(y)) for x, y in xy]
    if not all(math.isfinite(x) and math.isfinite(y) for x, y in pairs): return 'nonfinite', None
    if not all(0 <= x < width and 0 <= y < height for x, y in pairs): return 'outside', None
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    dx, dy = max(xs)-min(xs), max(ys)-min(ys)
    if max(dx, dy) <= 0: return 'degenerate', None
    size = max(2, math.ceil(3 * max(dx, dy) / 2))
    bx = math.floor((min(xs)+max(xs)-size)/2)
    by = math.floor((min(ys)+max(ys)-size)/2)
    return 'valid', (bx, by, bx+size, by+size)


def main(folder, report_path, pixel_mode):
    folder = Path(folder)
    summary = json.loads((folder/'SUMMARY.json').read_text(encoding='utf-8'))
    config = json.loads((folder/'CONFIG.json').read_text(encoding='utf-8'))
    pins = json.loads((folder/'SOURCE_PINS.json').read_text(encoding='utf-8'))
    assert all(digest(k) == v for k, v in pins.items()), 'source hash drift'
    recs = {r['sample_id']: r for r in summary['rows_detail']}
    assert set(recs) == {r['sample_id'] for r in config['rows']}
    all_ids = sorted(recs)
    sample_ids = set(all_ids if pixel_mode == 'all' else
                     [all_ids[int(i)] for i in np.linspace(0, len(all_ids)-1, min(12, len(all_ids)))])
    pixel_requests = defaultdict(list)
    checked = 0
    metadata_slots = 0
    valid_points_outside_box = 0
    min_raw_side, max_raw_side = None, None
    for r in config['rows']:
        sid = r['sample_id']
        z = arrays(folder/f'{sid}.npz')
        obs, inp = arrays(r['observation_path']), arrays(r['input_path'])
        assert digest(folder/f'{sid}.npz') == recs[sid]['output_sha256']
        fid = inp['frame_id']
        n = len(fid)
        assert z['rgb'].shape == (n, 2, 128, 128, 3) and z['rgb'].dtype == np.uint8
        assert z['pixel_valid'].dtype == bool and z['crop_valid'].dtype == bool
        np.testing.assert_array_equal(z['frame_id'], fid)
        np.testing.assert_array_equal(z['timestamp_us'], inp['timestamp_us'])
        np.testing.assert_array_equal(z['timestamp_us'], obs['timestamp_us'][fid])
        for j in range(n):
            w, h = map(int, z['source_width_height'][j])
            for side, start in enumerate((33, 54)):
                xy, mask = obs['xy_px'][fid[j], start:start+21], obs['usable'][fid[j], start:start+21]
                np.testing.assert_array_equal(z['hand_xy_px'][j, side], xy)
                np.testing.assert_array_equal(z['hand_usable'][j, side], mask)
                reason, box = geometry(xy, mask, w, h)
                assert z['crop_reason'][j, side] == reason
                assert bool(z['crop_valid'][j, side]) == (box is not None)
                if box is None:
                    assert (z['bbox_xyxy'][j, side] == -1).all()
                    assert not z['rgb'][j, side].any() and not z['pixel_valid'][j, side].any()
                    assert np.isnan(z['padding_fraction'][j, side])
                    assert np.isnan(z['shoulder_relative_size'][j, side])
                    assert np.isnan(z['frame_relative_size'][j, side])
                else:
                    np.testing.assert_array_equal(z['bbox_xyxy'][j, side], box)
                    x0, y0, x1, y1 = box
                    size = x1-x0
                    area = (min(w, x1)-max(0, x0))*(min(h, y1)-max(0, y0))
                    assert abs(z['padding_fraction'][j, side] - (1-area/size**2)) < 1e-12
                    assert abs(z['frame_relative_size'][j, side] - size/w) < 1e-12
                    if obs['usable'][fid[j], [11, 12]].all():
                        shoulder = obs['xy_px'][fid[j], [11, 12]].astype(float)
                        span = math.hypot(*(shoulder[1]-shoulder[0]))
                    else:
                        span = float('nan')
                    expected_scale = size/span if math.isfinite(span) and span > 0 else float('nan')
                    np.testing.assert_allclose(z['shoulder_relative_size'][j, side], expected_scale,
                                               rtol=1e-12, atol=1e-12, equal_nan=True)
                    if not all(x0 <= float(x) < x1 and y0 <= float(y) < y1 for x, y in xy):
                        valid_points_outside_box += 1
                    min_raw_side = size if min_raw_side is None else min(min_raw_side, size)
                    max_raw_side = size if max_raw_side is None else max(max_raw_side, size)
                metadata_slots += 1
            if sid in sample_ids and (pixel_mode == 'all' or j in {0, n//2, n-1}):
                pixel_requests[r['raw_path']].append((int(fid[j]), sid, j))
        checked += n
    pixel_frames = pixel_slots = png_checks = 0
    for video, requests in pixel_requests.items():
        by_frame = defaultdict(list)
        for fid, sid, j in requests: by_frame[fid].append((sid, j))
        cache = {sid: arrays(folder/f'{sid}.npz') for _, sid, _ in requests}
        with av.open(video) as con:
            stream = con.streams.video[0]
            for fid, frame in enumerate(con.decode(stream)):
                if fid in by_frame:
                    rgb = frame.to_ndarray(format='rgb24')
                    rawhash = hashlib.sha256(rgb.tobytes()).hexdigest()
                    us = round(frame.pts*stream.time_base*1_000_000)
                    for sid, j in by_frame[fid]:
                        z = cache[sid]
                        fact = recs[sid]['frames_rgb'][j]
                        assert fact['frame_id'] == fid and fact['rgb_sha256'] == rawhash
                        assert abs(int(z['timestamp_us'][j])-us) <= 1
                        np.testing.assert_array_equal(z['source_width_height'][j], [frame.width, frame.height])
                        raw_png = folder/f'{sid}_f{fid:06d}_raw.png'
                        if raw_png.exists():
                            np.testing.assert_array_equal(np.asarray(Image.open(raw_png).convert('RGB')), rgb)
                            png_checks += 1
                        for side in (0, 1):
                            if not z['crop_valid'][j, side]: continue
                            box = tuple(map(int, z['bbox_xyxy'][j, side]))
                            # PIL.crop constructs padding independently of the producer's ndarray canvas.
                            expect = Image.fromarray(rgb).crop(box).resize((128, 128), Image.Resampling.BILINEAR)
                            np.testing.assert_array_equal(z['rgb'][j, side], np.asarray(expect))
                            mask = Image.new('L', (frame.width, frame.height), 255).crop(box)
                            expect_mask = np.asarray(mask.resize((128, 128), Image.Resampling.NEAREST)) > 0
                            np.testing.assert_array_equal(z['pixel_valid'][j, side], expect_mask)
                            pixel_slots += 1
                        pixel_frames += 1
                if fid >= max(by_frame): break
    assert checked == summary['frames']
    assert valid_points_outside_box == 0, 'valid hand point clipped by requested box'
    result = {'status': 'PASS', 'independent_of_preparer_module': True,
              'rows_checked': len(recs), 'frame_rows_checked': checked,
              'hand_slots_geometry_mask_metadata_checked': metadata_slots,
              'decoded_pixel_frame_rows': pixel_frames, 'valid_crop_pixel_checks': pixel_slots,
              'raw_png_checks': png_checks, 'pixel_mode': pixel_mode,
              'pixel_sample_ids': sorted(sample_ids), 'all_source_hashes_unchanged': True,
              'valid_points_outside_requested_box': valid_points_outside_box,
              'raw_requested_side_px': {'min': min_raw_side, 'max': max_raw_side},
              'limitations': 'Independent arithmetic and pixel checks, not semantic/hand identity ground truth or recognition accuracy.'}
    with Path(report_path).open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--pixel-mode', choices=('all', 'sample'), required=True)
    a = parser.parse_args()
    main(a.folder, a.report, a.pixel_mode)
