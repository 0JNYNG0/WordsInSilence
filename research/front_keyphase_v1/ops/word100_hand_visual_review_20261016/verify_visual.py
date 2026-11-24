"""Independent saved-data and image conservation checks; no inference or assignment.

Only this file and VISUAL_DATA_VERIFICATION.json are verifier-owned. All previous
units, detector arrays, model files and source videos are read-only.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
import av
import numpy as np
from PIL import Image

U = Path(__file__).resolve().parent
P = U.with_name('word100_hand_assignment_20261016')
J = U.with_name('word100_hand_assignment_joint_20261016')
PUBLIC = U/'public'
EXPECTED = {
    'sobok': ('NIA_SL_WORD0651_REAL03_F', 'crop', 57),
    'believe-native': ('NIA_SL_WORD1139_REAL01_F', 'native', 71),
    'believe-crop': ('NIA_SL_WORD1139_REAL01_F', 'crop', 56),
    'believe-entry': ('NIA_SL_WORD1139_REAL01_F', 'crop', 46),
    'coldcoffee': ('NIA_SL_WORD0106_REAL01_F', 'crop', 52),
    'reception': ('NIA_SL_WORD0766_REAL03_F', 'crop', 46),
    'eighty-nine': ('NIA_SL_WORD2721_REAL02_F', 'crop', 43),
    'seongnam': ('NIA_SL_WORD1675_REAL03_F', 'crop', 42),
    'joy': ('NIA_SL_WORD1140_REAL01_F', 'crop', 94),
    'seasons': ('NIA_SL_WORD2572_REAL01_F', 'crop', 34),
}


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def json_safe(x):
    if isinstance(x, np.ndarray):
        return json_safe(x.tolist())
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, (float, np.floating)) and not np.isfinite(x):
        return None
    if isinstance(x, np.generic):
        return x.item()
    return x


def protected_pins():
    pins = {}
    def add(mapping):
        for name, value in mapping.items():
            name = str(Path(name).resolve())
            assert name not in pins or pins[name] == value, ('inconsistent source pin', name)
            pins[name] = value
    add(read(J/'COMPLETION.json')['artifacts_sha256'])
    add({str(J/'COMPLETION.json'): '1f919536a15638c9d616861f60a526d2b6162d48c2344f88745c6a9661dbd166'})
    add(read(J/'PROTECTED_FILES.json'))
    for c in [P/'RAW_CAPTURE_CONTEXT.json', J/'BROADENED_CAPTURE_CONTEXT.json', J/'REPLAY_CONTEXT.json']:
        add(read(c)['pins'])
    for group in read(U.parent/'word100_crop_readiness_20261016/SOURCE_GUARD.json').values():
        add(group)
    for c in [P/'RAW_CAPTURE_RESULT.json', J/'BROADENED_CAPTURE_RESULT.json']:
        for row in read(c)['rows']:
            add(row['output_sha256'])
    for row in read(J/'REPLAY_RESULT.json')['rows']:
        add(row['outputs'])
    add(read(U/'EVIDENCE_EXPORT.json')['sourcePins'])
    return pins


def check_pins(pins):
    mismatches = [str(p) for p, h in pins.items() if not Path(p).is_file() or sha(p) != h]
    assert not mismatches, ('protected source changed', mismatches)
    return {'files': len(pins), 'mismatches': 0}


def parse_data():
    text = (PUBLIC/'data.js').read_text('utf-8-sig').strip()
    prefix = 'window.HAND_REVIEW'
    assert text.startswith(prefix), 'Expected a plain JSON assignment, not executable exporter code'
    payload = text[len(prefix):].lstrip()
    assert payload.startswith('=')
    payload = payload[1:].strip().removesuffix(';').strip()
    def reject(token):
        raise AssertionError('Non-JSON numeric token: '+token)
    return json.loads(payload, parse_constant=reject)


def local_image(name):
    p = (PUBLIC/name).resolve()
    assert p.is_relative_to(PUBLIC.resolve()) and p.is_file(), ('image escapes output or missing', name)
    return p


def audit(data):
    evidence = read(U/'EVIDENCE_EXPORT.json')
    assert sha(PUBLIC/'data.js') == evidence['dataJsSha256']
    for image in evidence['images']:
        assert sha(U/image['path']) == image['sha256']
    assert len(evidence['images']) == 250
    cases = data['cases']
    assert len(cases) == 10 and {c['id'] for c in cases} == set(EXPECTED)
    items = {i['clip_id']: i for c in [read(P/'RAW_CAPTURE_CONTEXT.json'), read(J/'BROADENED_CAPTURE_CONTEXT.json')]
             for i in c['items']}
    replay_rows = {(r['clip_id'], r['mode']): r for r in read(J/'REPLAY_RESULT.json')['rows']}
    counts = Counter()
    checks = []
    for case in cases:
        cid, mode, target = EXPECTED[case['id']]
        assert (case['clipId'], case['mode'], case['targetFrame']) == (cid, mode, target)
        item, row = items[cid], replay_rows[cid, mode]
        assert case['label'] == item['label']
        assert item['source_role'] == 'train_candidate'
        raw = arrays(Path(row['raw_dir'])/'raw.npz')
        current = arrays(J/'replay'/cid/mode/'assignment.npz')['sides']
        previous = arrays(J/'replay'/cid/mode/'previous_assignment.npz')['sides']
        trace = read(J/'replay'/cid/mode/'trace.json')
        fs = case['frames']
        indexes = [f['index'] for f in fs]
        assert indexes == sorted(set(indexes)) and target in indexes
        assert indexes == list(range(min(indexes), max(indexes)+1)), ('omitted display frame', case['id'])
        assert indexes == list(range(max(0, target-12), min(len(raw['raw_count'])-1, target+12)+1))
        w, h = case['width'], case['height']
        assert np.all(raw['width'][indexes] == w) and np.all(raw['height'][indexes] == h)
        for f in fs:
            k = f['index']
            assert 0 <= k < len(raw['raw_count'])
            assert k == int(raw['frame_id'][k]), ('array row differs from frame ID', cid, k)
            assert f['row'] == k and f['timestampUs'] == int(raw['timestamp_us'][k])
            assert f['timeSeconds'] == int(raw['timestamp_us'][k])/1e6
            count = int(raw['raw_count'][k])
            assert len(f['raw']) == count
            assert [d['index'] for d in f['raw']] == list(range(count))
            for i, detection in enumerate(f['raw']):
                assert detection['labelCode'] == int(raw['raw_label'][k, i])
                assert detection['label'] == str(raw['raw_label_name'][k, i])
                assert detection['score'] == float(raw['raw_score'][k, i])
                assert detection['points'] == json_safe(raw['hand_xy_px'][k, i]), (case['id'], k, i, 'points changed')
                assert len(detection['points']) == 21
                assert detection['baselineSide'] == int(raw['baseline_side'][k, i])
                assert detection['previousSide'] == int(previous[k, i])
                assert detection['currentSide'] == int(current[k, i])
                assert detection['reason'] == trace[k]['reasons'][i]
                counts['raw_detections'] += 1
                counts['raw_point_coordinates'] += 42
            wrists = f['poseWrists']
            assert len(wrists) == 2
            for side, wrist in enumerate(wrists):
                assert wrist['side'] == side
                assert wrist['point'] == json_safe(raw['pose_xy_px'][k, 15+side])
                assert wrist['usable'] == bool(raw['pose_usable'][k, 15+side])
                assert wrist['observed'] == bool(raw['pose_observed'][k, 15+side])
                assert wrist['confidence'] == json_safe(raw['pose_conf'][k, 15+side])
            assert f['trace'] == json_safe(trace[k]), (case['id'], k, 'trace mismatch')
            counts['frames'] += 1
            counts['zero_hand_frames'] += count == 0
        # Decode once per displayed case; never run MediaPipe or assignment.
        wanted = {f['index']: f for f in fs}
        image_checks = 0
        with av.open(item['video_path']) as video:
            origin_pts = None
            for k, frame in enumerate(video.decode(video=0)):
                if origin_pts is None and frame.pts is not None:
                    origin_pts = frame.pts
                if k in wanted:
                    f = wanted[k]
                    decoded = frame.to_image().convert('RGB')
                    if mode == 'crop':
                        decoded = decoded.crop(tuple(item['box']))
                    assert decoded.size == (w, h)
                    image_path = local_image(f['image'])
                    assert image_path.suffix.lower() == '.jpg'
                    image_size = (min(960, w), round(h*min(960, w)/w))
                    assert (f['imageWidth'], f['imageHeight']) == image_size
                    if image_size != decoded.size:
                        decoded = decoded.resize(image_size, Image.Resampling.LANCZOS)
                    encoded = BytesIO()
                    decoded.save(encoded, format='JPEG', quality=90, subsampling=0, optimize=True)
                    assert encoded.getvalue() == image_path.read_bytes(), (case['id'], k, 'source/crop/resize/JPEG differs')
                    with Image.open(image_path) as actual:
                        assert actual.size == image_size
                    if frame.pts is not None:
                        original_pts_us = round(float((frame.pts-origin_pts) * frame.time_base)*1e6)
                        assert abs(original_pts_us-int(raw['timestamp_us'][k])) <= 1, (cid, k, 'source PTS mismatch')
                    image_checks += 1
                if k >= max(indexes):
                    break
        assert image_checks == len(fs)
        counts['source_images_exact_after_documented_resize_and_jpeg'] += image_checks
        checks.append({'id': case['id'], 'clip_id': cid, 'mode': mode, 'target_frame': target,
                       'frames': len(fs), 'raw_detections': sum(len(f['raw']) for f in fs),
                       'status': 'PASS'})
    return {'status': 'PASS', 'cases': len(cases), 'counts': dict(counts), 'case_checks': checks,
            'checks': ['raw 21-point XY exact', 'handedness labels/scores exact',
                       'baseline/previous/current side assignments exact', 'pose wrists and usability exact',
                       'trace and absent frames preserved', 'frame IDs/timestamps/PTS preserved',
                       'source/crop images reproduce exact JPEG bytes after declared resize and encoding'],
            'image_transform': 'Source RGB decode, fixed crop when applicable, proportional width<=960 LANCZOS resize, JPEG quality90/subsampling0/optimizeTrue. Coordinates remain original mode-input pixels.',
            'limitation': 'Verifies display fidelity only. Does not certify MediaPipe anatomical accuracy, classification or generalization.'}


def main():
    report = {'verifier': 'assignment_verify', 'time_utc': datetime.now(timezone.utc).isoformat(),
              'training': 0, 'detector_inference': 0, 'classifier_inference': 0,
              'algorithm_or_production_changes': 0}
    pins = protected_pins()
    report['protected_before'] = check_pins(pins)
    if '--guard-only' in sys.argv:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    try:
        report['visual_data'] = audit(parse_data())
        report['protected_after'] = check_pins(pins)
        report['input_sha256'] = {str(PUBLIC/'data.js'): sha(PUBLIC/'data.js'),
                                  str(Path(__file__)): sha(__file__)}
        report['status'] = 'PASS'
    except Exception:
        report['status'] = 'FAIL'
        report['error'] = traceback.format_exc()
        report['protected_after'] = check_pins(pins)
    target = U/'VISUAL_DATA_VERIFICATION.json'
    if target.exists():
        old = read(target)
        assert old['status'] != 'PASS', 'Successful final audit is immutable; create a separate follow-up on explicit authorization'
        report['prior_failed_audits'] = old.get('prior_failed_audits', []) + [old]
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
