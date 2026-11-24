"""Link frozen input artifacts only. No image generation, model or network use."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

UNIT = Path(__file__).resolve().parent
BASE = UNIT.parents[1]
MANIFEST = BASE / 'ops/rgb_pose_readiness_20261010/PAIRED_INPUT_MANIFEST.json'
PIXELS = BASE / 'runs/rgb_temporal_roi_v1/PIXEL_RECORDS.json'
EVENTS = BASE / 'runs/rgb_temporal_roi_v1/NATURAL_EVENTS.json'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    assert not (UNIT / 'INPUT_INDEX.json').exists(), 'Do not overwrite completed output'
    pins = {}

    def pin(p, expected=None):
        p = str(Path(p).resolve())
        if p not in pins:
            pins[p] = sha(p)
        if expected is not None:
            assert pins[p] == expected, ('source hash mismatch', p)
        return pins[p]

    for p in [MANIFEST, PIXELS, EVENTS, UNIT/'SCOPE.json', Path(__file__)]:
        pin(p)
    manifest, pixels, events = read(MANIFEST), read(PIXELS), read(EVENTS)
    by_sample = {m['sample_id']: m for m in manifest}
    assert len(by_sample) == len(manifest) == 113
    event_keys = {(e['sample_id'], e['row'], e['side']): e for e in events}
    assert len(event_keys) == len(events) == 88
    pixel_keys = {(p['sample_id'], p['row'], p['side']): p for p in pixels}
    assert len(pixel_keys) == len(pixels) == 78
    assert set(pixel_keys) == {k for k,e in event_keys.items() if e['gate']['accepted']}
    samples, counts, uncertainty = [], Counter(), Counter()
    used = set()
    for m in manifest:
        sid = m['sample_id']
        pose_path, rgb_path = m['source']['input_path'], m['output_path']
        pin(pose_path, m['source']['input_sha256'])
        pin(rgb_path, m['output_sha256'])
        with np.load(pose_path, allow_pickle=False) as pose, np.load(rgb_path, allow_pickle=False) as rgb:
            n = m['frames']
            assert pose['x'].shape == (n, 280)
            assert rgb['crop_valid'].shape == (n, 2)
            assert np.array_equal(pose['frame_id'], rgb['frame_id'])
            assert np.array_equal(pose['timestamp_us'], rgb['timestamp_us'])
            assert np.all(np.diff(pose['timestamp_us']) > 0)
            rows = []
            for row in range(n):
                frame, ts = int(pose['frame_id'][row]), int(pose['timestamp_us'][row])
                hands = []
                for side_index, side in enumerate(('left', 'right')):
                    valid = bool(rgb['crop_valid'][row, side_index])
                    key = (sid, row, side)
                    observed = {'path': rgb_path, 'row': row, 'side_index': side_index} if valid else None
                    proposal = None
                    if key in pixel_keys:
                        p, e = pixel_keys[key], event_keys[key]
                        assert not valid
                        assert p['clip_id'] == m['clip_id'] and p['source_path'] == m['source']['raw_path']
                        assert (p['frame_id'], p['timestamp_us'], p['decoded_pts_us']) == (frame, ts, ts)
                        assert e['proposed_box'] == p['proposed_box']
                        pin(p['path'], p['sha256'])
                        with np.load(p['path'], allow_pickle=False) as crop:
                            assert int(crop['frame_id']) == frame and int(crop['timestamp_us']) == ts
                            assert bool(crop['region_estimated']) and not bool(crop['hand_landmarks_observed'])
                            assert np.array_equal(crop['bbox_xyxy'], p['proposed_box'])
                            assert crop['rgb'].shape == (128, 128, 3)
                            assert crop['pixel_valid'].shape == (128, 128)
                        assert not rgb['hand_usable'][row, side_index].any()
                        proposal = {'path': p['path'], 'sha256': p['sha256'], 'region_estimated': True,
                                    'hand_landmarks_observed': False,
                                    'diagnostic_only_prior_classification': p['prior_classification']}
                        uncertainty[p['prior_classification']] += 1
                        used.add(key)
                    hands.append({'side': side, 'landmark_observed_count': int(rgb['hand_usable'][row, side_index].sum()),
                                  'observed_rgb': observed, 'estimated_region_rgb': proposal})
                    counts['hand_slots'] += 1
                    counts['B_observed_rgb'] += int(valid)
                    counts['C_available_rgb'] += int(valid or proposal is not None)
                    counts['C_added_estimated_rgb'] += int(proposal is not None)
                rows.append({'row': row, 'frame_id': frame, 'timestamp_us': ts, 'hands': hands})
            samples.append({'sample_id': sid, 'clip_id': m['clip_id'], 'label_metadata_only': m['label'],
                            'signer_metadata_only': m['signer_id'], 'split': 'development_input_fixture_only',
                            'semantic_alignment_verified': False,
                            'shared_pose_path': pose_path, 'shared_pose_sha256': pin(pose_path),
                            'observed_rgb_path': rgb_path, 'observed_rgb_sha256': pin(rgb_path), 'rows': rows})
            counts['rows'] += n
    assert used == set(pixel_keys)
    expected = {'rows': 2789, 'hand_slots': 5578, 'B_observed_rgb': 5315,
                'C_available_rgb': 5393, 'C_added_estimated_rgb': 78}
    assert dict(counts) == expected, counts
    assert uncertainty == {'clear_same': 74, 'uncertain': 4}, uncertainty
    assert all(sha(p) == h for p,h in pins.items())
    result = {'schema': 'wis_rgb_comparison_input_index_v1',
              'status': 'VERIFIED_INPUT_FIXTURE_NOT_TRAINING_READY',
              'arms': {'A': 'shared pose only', 'B': 'shared pose + observed_rgb',
                       'C': 'shared pose + observed_rgb + estimated_region_rgb'},
              'never_model_inputs': ['label_metadata_only', 'signer_metadata_only', 'sample_id', 'clip_id',
                                     'diagnostic_only_prior_classification'],
              'rules': ['Preserve pose and timeline; never replace observed RGB with a proposal.',
                        'Missing RGB is masked; do not encode it as an observed black image.',
                        'Keep pixel padding masks, bbox and body-relative size in source artifacts.',
                        'No word accuracy or unseen-person evaluation is represented by this index.'],
              'source_sha256': pins, 'counts': expected, 'samples': samples}
    write(UNIT/'INPUT_INDEX.json', result)
    write(UNIT/'BUILD_CHECK.json', {'status': 'PASS', 'counts': expected, 'segments': len(samples),
                                  'proposal_prior_classification': dict(uncertainty),
                                  'input_files_pinned': len(pins), 'source_files_unchanged': True,
                                  'training': 0, 'model_inference': 0, 'new_crops': 0,
                                  'index_sha256': sha(UNIT/'INPUT_INDEX.json')})
    print(json.dumps({'status': 'PASS', 'counts': expected, 'pinned': len(pins)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
