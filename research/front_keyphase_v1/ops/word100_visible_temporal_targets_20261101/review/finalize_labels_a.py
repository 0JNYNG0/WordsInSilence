"""Validate A's manually reviewed RGB draft; no label inference or model use."""
import hashlib
import json
from collections import Counter
from pathlib import Path

unit = Path(__file__).resolve().parent.parent
review = unit / 'review'
data = json.loads((review / 'LABELS_A_PROGRESS.json').read_text(encoding='utf-8'))
rubric = json.loads((unit / 'RUBRIC.json').read_text(encoding='utf-8'))
expected = [f'S{i:03d}_W{w}' for i in range(1, 29) for w in range(3)]
assert [r['sample_id'] for r in data['rows']] == expected
assert data['completed_windows'] == data['expected_windows'] == 84
assert data['pending_sample_ids'] == []
for row in data['rows']:
    assert set(row) == set(rubric['label_format'])
    assert row['center_relation'] in rubric['relation_values']
    for side in ('R', 'L'):
        shape = row['center_shape'][side]
        assert len(shape) == 3
        assert all(v is None or type(v) is int and v in (0, 1) for v in shape)
        assert sum(v == 1 for v in shape) <= 1
        assert row['motion'][side] in rubric['motion_values']
        for k in ('roi_center', 'roi_window'):
            assert row[k][side] is None or type(row[k][side]) is bool
    assert isinstance(row['notes'], str) and row['notes'].strip()
images = data['viewed_images']
expected_images = {(unit / 'visuals' / f'S{i:03d}' / fn).resolve() for i in range(1, 29)
                   for fn in ('states.jpg', 'window_0.jpg', 'window_1.jpg', 'window_2.jpg')}
assert len(images) == 112
assert {Path(im['path']).resolve() for im in images} == expected_images
for im in images:
    assert hashlib.sha256(Path(im['path']).read_bytes()).hexdigest() == im['sha256']
assert hashlib.sha256((unit / 'RUBRIC.json').read_bytes()).hexdigest() == data['rubric_sha256']
data['shape_order'] = rubric['shape_order']
data['row_evidence'] = {
    sample_id: [str((unit / 'visuals' / sample_id.split('_')[0] / fn).resolve())
                for fn in ('states.jpg', f"window_{sample_id[-1]}.jpg")]
    for sample_id in expected
}
data['validation'] = {
    'status': 'PASS_SCHEMA_COMPLETENESS_AND_EVIDENCE_HASHES',
    'rows': 84, 'unique_viewed_images': 112,
    'scope': 'Mechanical format and evidence integrity only; not visual-label correctness certification.',
    'shape_cells': dict(Counter('null' if v is None else str(v) for r in data['rows']
                               for s in ('R', 'L') for v in r['center_shape'][s])),
    'motion_windows_by_hand': dict(Counter('null' if r['motion'][s] is None else r['motion'][s]
                                          for r in data['rows'] for s in ('R', 'L'))),
}
payload = json.dumps(data, ensure_ascii=False, indent=2)
(review / 'LABELS_A.json').write_text(payload, encoding='utf-8')
(review / 'LABELS_A_PROGRESS.json').write_text(payload, encoding='utf-8')
print(json.dumps(data['validation'], ensure_ascii=False))
