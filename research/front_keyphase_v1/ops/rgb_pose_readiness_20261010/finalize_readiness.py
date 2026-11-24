"""Aggregate completed preparation/audits, without decoding or model execution."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    run = here.parent.parent / 'runs' / 'rgb_pose_readiness_v1'
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    config = read(here / 'CONFIG_ALL.json')
    tests = read(here / 'SYNTHETIC_CHECK.json')
    assert tests['status'] == 'PASS' and tests['tests'] == 10
    for name, field in [('prepare_rgb_pairs.py', 'implementation_sha256'),
                        ('test_rgb_pairs.py', 'test_sha256'), ('SCOPE.json', 'scope_sha256')]:
        assert digest(here / name) == tests[field], f'Frozen code/contract changed: {name}'
    assert digest(Path(config['manifest_path'])) == config['manifest_sha256']
    audits = [read(here / f'{name}_INDEPENDENT_CHECK.json') for name in ('PILOT', 'REMAINING')]
    assert all(a['status'] == 'PASS' and a['all_source_hashes_unchanged'] for a in audits)
    assert audits[0]['decoded_pixel_frame_rows'] == 141
    assert audits[1]['decoded_pixel_frame_rows'] == 36
    for name in ('PILOT', 'REMAINING'):
        assert read(here / f'{name}_VISUAL_REVIEW.json')['result'].startswith('PASS')
    summaries = [read(run / part / 'SUMMARY.json') for part in ('pilot', 'remaining')]
    rows = [r for s in summaries for r in s['rows_detail']]
    assert len(rows) == len({r['sample_id'] for r in rows}) == 113
    assert {r['sample_id'] for r in rows} == {r['sample_id'] for r in config['rows']}
    assert all(s['source_hashes_unchanged'] for s in summaries)
    assert all(s['new_inference'] == s['new_training'] == s['new_pose_extraction'] == 0 for s in summaries)
    counts = Counter()
    frames_state = Counter()
    by_word = defaultdict(list)
    padded = shoulder_unknown = npz_bytes = 0
    output_manifest = []
    for r in rows:
        out = Path(r['output_path'])
        assert digest(out) == r['output_sha256']
        npz_bytes += out.stat().st_size
        with np.load(out, allow_pickle=False) as z:
            counts.update(z['crop_reason'].reshape(-1).tolist())
            frames_state.update(int(v) for v in z['crop_valid'].sum(axis=1))
            padded += int(((z['padding_fraction'] > 0) & z['crop_valid']).sum())
            shoulder_unknown += int((~np.isfinite(z['shoulder_relative_size']) & z['crop_valid']).sum())
        by_word[r['label']].append(r)
        output_manifest.append({k: r[k] for k in ('sample_id', 'clip_id', 'label', 'signer_id',
                                                  'frames', 'output_path', 'output_sha256', 'source')})
    classes = [{'label': k, 'segments': len(v), 'unique_clips': len({r['clip_id'] for r in v}),
                'signers': sorted({r['signer_id'] for r in v})} for k, v in sorted(by_word.items())]
    total_frames = sum(r['frames'] for r in rows)
    assert total_frames == sum(a['frame_rows_checked'] for a in audits) == 2789
    assert sum(counts.values()) == 2 * total_frames
    result = {
        'status': 'RGB_POSE_INPUT_PREPARATION_VERIFIED_NO_MODEL_TRAINING', 'date_kst': '2026-10-10',
        'roles': 'Codex local implementation/execution and separate-code verification under explicit narrow user exception; not a second human/external review.',
        'segments': len(rows), 'unique_clips': len({r['clip_id'] for r in rows}),
        'classes': len(classes), 'signers': sorted({r['signer_id'] for r in rows}),
        'frame_rows': total_frames, 'hand_slots': 2 * total_frames,
        'hand_slot_reasons': dict(counts), 'frames_by_valid_hand_count': dict(frames_state),
        'valid_crops_with_source_edge_padding': padded,
        'valid_crops_without_shoulder_scale': shoulder_unknown,
        'npz_bytes': npz_bytes, 'timing_max_error_us': max(s['timing_max_error_us'] for s in summaries),
        'class_segment_count_histogram': dict(Counter(c['segments'] for c in classes)),
        'class_signer_count_histogram': dict(Counter(len(c['signers']) for c in classes)),
        'synthetic_tests_passed': tests['tests'],
        'independent_checks': {
            'frame_rows_metadata': total_frames, 'hand_slots_geometry_and_missing_metadata': 2 * total_frames,
            'source_decoded_pixel_frame_rows': sum(a['decoded_pixel_frame_rows'] for a in audits),
            'valid_crop_pixels_and_pixel_masks': sum(a['valid_crop_pixel_checks'] for a in audits),
            'visual_rows': 12, 'visual_frame_rows': 36, 'visual_hand_slots': 72,
            'all_source_hashes_unchanged': True,
            'scope': 'All metadata/boxes/missing slots; pixel equality and valid pixel-mask equality on all pilot frames plus deterministic36 remaining frame rows. Visual inspection sampled separately.'},
        'constraints': {'model_training': 0, 'model_inference': 0, 'new_pose_extraction': 0,
                        'downloads': 0, 'camera_access': 0, 'user_recordings': 0,
                        'production_changes': 0, 'git_mutation': 0},
        'limitations': [
            'Both source people are known baseline-training signers, so this is not unseen-person evaluation.',
            'Time/pixel alignment is verified; semantic sign boundaries and landmark anatomical correctness remain uncertified.',
            'RGB hand crops require valid stored hand observations; visible hands with missing landmarks have no image fallback.',
            'Rectangles include face/background and original blur. Resizing and scale metadata do not measure true depth.',
            'No new accuracy, hand-form correctness, or generalization result.'],
        'evidence': ['SCOPE.json', 'ROLE_EXCEPTION.json', 'SYNTHETIC_CHECK.json',
                     'PILOT_INDEPENDENT_CHECK.json', 'REMAINING_INDEPENDENT_CHECK.json',
                     'PILOT_VISUAL_REVIEW.json', 'REMAINING_VISUAL_REVIEW.json'],
        'producer_sha256': digest(here / 'prepare_rgb_pairs.py'),
        'independent_verifier_sha256': digest(here / 'verify_rgb_pairs.py'),
        'per_class': classes}
    for name, value in [('RESULT.json', result), ('PAIRED_INPUT_MANIFEST.json', output_manifest)]:
        with (here / name).open('x', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_class'}, ensure_ascii=True))


if __name__ == '__main__':
    main()
