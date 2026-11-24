"""Independent source-to-index audit; does not import or run the builder."""
import hashlib
import json
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
base = here.parents[1]
index = json.loads((here/'INPUT_INDEX.json').read_text(encoding='utf-8'))
manifest = json.loads((base/'ops/rgb_pose_readiness_20261010/PAIRED_INPUT_MANIFEST.json').read_text(encoding='utf-8'))
pixels = json.loads((base/'runs/rgb_temporal_roi_v1/PIXEL_RECORDS.json').read_text(encoding='utf-8'))


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


sources = {s['sample_id']: s for s in manifest}
proposals = {(p['sample_id'], p['frame_id'], p['side']): p for p in pixels}
assert len(sources) == len(manifest) == len(index['samples']) == 113
assert len(proposals) == len(pixels) == 78
assert {s['sample_id'] for s in index['samples']} == set(sources)
seen = set()
total_rows = total_slots = expected_b = expected_c = added = uncertain = 0
for sample in index['samples']:
    original = sources[sample['sample_id']]
    assert sample['shared_pose_path'] == original['source']['input_path']
    assert sample['shared_pose_sha256'] == original['source']['input_sha256']
    assert sample['observed_rgb_path'] == original['output_path']
    assert sample['observed_rgb_sha256'] == original['output_sha256']
    assert sample['clip_id'] == original['clip_id']
    assert sample['label_metadata_only'] == original['label']
    assert sample['signer_metadata_only'] == original['signer_id']
    assert sample['split'] == 'development_input_fixture_only'
    assert sample['semantic_alignment_verified'] is False
    with np.load(original['output_path'], allow_pickle=False) as images, np.load(original['source']['input_path'], allow_pickle=False) as pose:
        assert len(sample['rows']) == original['frames'] == len(pose['x'])
        np.testing.assert_array_equal([r['frame_id'] for r in sample['rows']], images['frame_id'])
        np.testing.assert_array_equal([r['timestamp_us'] for r in sample['rows']], pose['timestamp_us'])
        for i, row in enumerate(sample['rows']):
            assert row['row'] == i and len(row['hands']) == 2
            for j, hand in enumerate(row['hands']):
                side = ['left', 'right'][j]
                assert hand['side'] == side
                assert hand['landmark_observed_count'] == np.count_nonzero(images['hand_usable'][i,j])
                key = (sample['sample_id'], int(images['frame_id'][i]), side)
                observed = bool(images['crop_valid'][i,j])
                expected_observed = {'path': original['output_path'], 'row': i, 'side_index': j} if observed else None
                assert hand['observed_rgb'] == expected_observed
                estimate = hand['estimated_region_rgb']
                if key in proposals:
                    p = proposals[key]
                    assert estimate is not None and not observed
                    assert p['row'] == i and p['timestamp_us'] == row['timestamp_us']
                    assert estimate['path'] == p['path'] and estimate['sha256'] == p['sha256']
                    assert estimate['region_estimated'] is True
                    assert estimate['hand_landmarks_observed'] is False
                    assert estimate['diagnostic_only_prior_classification'] == p['prior_classification']
                    with np.load(p['path'], allow_pickle=False) as crop:
                        assert int(crop['frame_id']) == key[1]
                        assert int(crop['timestamp_us']) == int(pose['timestamp_us'][i])
                        assert crop['region_estimated'].item() is True
                        assert crop['hand_landmarks_observed'].item() is False
                    seen.add(key)
                    added += 1
                    uncertain += p['prior_classification'] == 'uncertain'
                else:
                    assert estimate is None
                expected_b += observed
                expected_c += observed or key in proposals
                total_slots += 1
            total_rows += 1
assert seen == set(proposals)
assert (total_rows, total_slots, expected_b, expected_c, added, uncertain) == (2789,5578,5315,5393,78,4)
counts = dict(rows=total_rows, hand_slots=total_slots, B_observed_rgb=expected_b,
              C_available_rgb=expected_c, C_added_estimated_rgb=added)
assert counts == index['counts']
for path, checksum in index['source_sha256'].items():
    assert digest(path) == checksum, path
build = json.loads((here/'BUILD_CHECK.json').read_text(encoding='utf-8'))
assert digest(here/'INPUT_INDEX.json') == build['index_sha256']
result = {'status':'PASS', 'verification':'Independent reconstruction; builder not imported',
          'counts':counts, 'segments':113, 'uncertain_proposals_preserved':uncertain,
          'source_files_unchanged':len(index['source_sha256']),
          'index_sha256':digest(here/'INPUT_INDEX.json'),
          'verifier_sha256':digest(__file__), 'model_or_video_execution':False}
with (here/'INDEPENDENT_CHECK.json').open('x', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(json.dumps(result, ensure_ascii=False))
