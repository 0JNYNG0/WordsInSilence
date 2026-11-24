"""Read-only directory audit for the next candidate; no joint arrays or heldout ZIPs."""
from pathlib import Path
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone, timedelta

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
WEB = OUT.parents[1]
PILOT = WEB.parent / 'sign_model_pilot_v1'
sys.path.insert(0, str(PILOT))
import expand_build as reader

dest = OUT / 'word1544_local_availability.json'
if dest.exists():
    raise FileExistsError('Completed availability evidence already exists; do not repeat.')
source = Path(reader.AIHUB)
inventory = [{'name': p.name, 'bytes': p.stat().st_size}
             for p in sorted(source.glob('*.zip'))]
coverage_path = WEB / 'evidence/word_archive_mapping_20260918/coverage_256.csv'
with coverage_path.open(encoding='utf-8-sig', newline='') as f:
    metadata = [r for r in csv.DictReader(f) if r['word_number'] == '1544'
                and int(r['signer_id']) in list(range(1, 10)) + [11, 12, 13]]
assert len(metadata) == 12
rows = []
for r in metadata:
    signer = int(r['signer_id'])
    assert r['split_role'] == ('train_candidate' if signer <= 9 else 'development_only')
    archive = source / ('[라벨]%02d_real_sen_keypoint.zip' % signer)
    before = archive.stat()
    target = Path(r['video_basename']).stem
    control = 'NIA_SL_SEN0356_REAL%02d_F' % signer
    members, stats = reader.selected_directory(archive, {target, control})
    after = archive.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    target_members = [m for m in members if m['clip_id'] == target]
    control_members = [m for m in members if m['clip_id'] == control]
    assert control_members, 'Positive control missing; search not validated'
    rows.append({'clip_id': target, 'split_role': r['split_role'],
                 'raw_label': r['raw_label'],
                 'annotation_seconds': [float(r['annotation_start_s']), float(r['annotation_end_s'])],
                 'archive': str(archive), 'archive_bytes': before.st_size,
                 'target_keypoint_members': len(target_members),
                 'positive_control_keypoint_members': len(control_members),
                 'directory_scan': stats})
result = {'at_kst': datetime.now(timezone(timedelta(hours=9))).isoformat(),
          'status': 'NOT_READY_MISSING_LOCAL_MULTISIGNER_WORD_KEYPOINTS'
                    if not any(r['target_keypoint_members'] for r in rows) else 'REVIEW_TARGET_MEMBERS',
          'scope': 'Configured AI Hub datastore; only train01-09/dev11-13 keypoint ZIP central directories. Not proof of absence from all storage or AI Hub.',
          'inventory': inventory, 'rows': rows,
          'word_keypoint_archives_in_configured_store': [r for r in inventory if 'word_keypoint' in r['name']],
          'existing_coverage_sha256': hashlib.sha256(coverage_path.read_bytes()).hexdigest(),
          'semantic_gate': 'Existing visual review reports a form difference from NIKL; lexical equivalence remains unconfirmed. No automatic rejection or merge.',
          'joint_arrays_read': 0, 'heldout_archives_opened': 0,
          'downloads': 0, 'training_runs': 0}
dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': result['status'], 'checked_signers': len(rows),
                  'target_members': sum(r['target_keypoint_members'] for r in rows),
                  'positive_control_members': sum(r['positive_control_keypoint_members'] for r in rows),
                  'joint_arrays_read': 0, 'heldout_archives_opened': 0}))
