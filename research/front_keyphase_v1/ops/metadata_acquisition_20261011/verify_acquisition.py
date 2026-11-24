"""Independent saved-file/catalog calculations; no network, training or extraction."""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import re
import shutil
import zipfile

U = Path(__file__).resolve().parent
B = U.parent / 'aihub_modu_source_review_20261011'
ROOT = Path('W:/WordsInSilence/data')

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def digest(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

r = read(U / 'ACQUISITION_RESULT.json')
s = read(U / 'SUPPLEMENTARY_DOCUMENTS.json')
fresh = read(r['catalog']['destination'])['list']
prior = read(B / 'AIHUB103_PUBLIC_CATALOG.json')['files']
fields = ['streFileNm', 'fileSize', 'fileStreCours']
normalize = lambda fs: {int(f['fileSn']): tuple(f[k] for k in fields) for f in fs}
assert normalize(fresh) == normalize(prior)
# Validation video path is REAL/01_real_word_video.zip, unlike Training REAL/WORD/.
word = [f for f in fresh if re.fullmatch(r'\[(원천|라벨)\]\d+_real_word_(video|morpheme)\.zip', f['streFileNm'])]
key = [f for f in fresh if re.fullmatch(r'\[라벨\]\d+_real_word_keypoint\.zip', f['streFileNm'])]
assert len(word) == 35 and sum(int(f['fileSize']) for f in word) == 1181348380709
assert {int(f['fileSn']) for f in word} == {int(f['fileSn']) for f in read(B / 'ACQUISITION_SCOPE.json')['file_list']}
assert len(key) == 17 and sum(int(f['fileSize']) for f in key) == 211236960084
inventory = []
for item in r['labels']:
    assert digest(item['source']) == item['sha256'] == digest(item['destination'])
    with zipfile.ZipFile(item['destination']) as z:
        names = [f.filename for f in z.infolist() if not f.is_dir()]
        matches = [re.search(r'(WORD\d+)_REAL(\d+)_(?:(.+)_)?([DFLRU])_morpheme\.json$', n) for n in names]
        assert all(matches)
        mismatch = [n for n, m in zip(names, matches) if n.split('/')[1] != m.group(2)]
        inventory.append({
            'split': item['official_catalog_entry']['fileStreCours'].split('/')[1],
            'json_files': len(names),
            'word_code_count': len({m.group(1) for m in matches}),
            'folder_id_counts': dict(collections.Counter(n.split('/')[1] for n in names)),
            'filename_performer_ids': sorted({m.group(2) for m in matches}),
            'view_counts': dict(collections.Counter(m.group(4) for m in matches)),
            'folder_filename_id_mismatch_count': len(mismatch),
            'mismatch_examples': mismatch[:3],
            'extra_name_token_count': sum(m.group(3) is not None for m in matches),
            'interpretation': 'Candidate metadata only: folder/name IDs are not independently verified people, WORD codes are not necessarily unique glosses, five views are not five independent performances.',
        })
hashchecks = read(U / 'INDEPENDENT_FILE_HASHES.json')
assert all(x['match'] for x in hashchecks)
for d in s['documents']:
    assert digest(d['destination']) == d['sha256']
files = [p for p in ROOT.rglob('*') if p.is_file()]
assert not any(p.name.endswith('.partial') for p in files)
free = shutil.disk_usage('W:/').free
labels = sum(x['bytes'] for x in r['labels'])
videos = sum(int(f['fileSize']) for f in word if 'morpheme' not in f['streFileNm'])
all_bytes = sum(int(f['fileSize']) for f in fresh)
out = {
    'at_kst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
    'reviewer': 'Codex separate calculation and PowerShell hashing; no independent agent claimed',
    'catalog_refetched_exact_match': True, 'catalog_files': len(fresh),
    'copied_label_bytes': labels,
    'archive_crc_entries_passed': sum(x['entries'] for x in r['labels']),
    'powershell_hashes_all_match': True, 'supplementary_document_hashes_all_match': True,
    'source_label_hashes_unchanged': True, 'no_partial_files': True,
    'label_inventory': inventory,
    'capacity': {
        'unit': 'decimal bytes; GB=10^9 TB=10^12, Windows displayed TiB=2^40',
        'free_bytes_now': free, 'free_TiB': free / 1024**4,
        'WORD_video_plus_labels_bytes': sum(int(f['fileSize']) for f in word),
        'WORD_video_still_to_acquire_bytes': videos,
        'free_after_WORD_video_archives_only_bytes': free - videos,
        'optional_WORD_OpenPose_bytes': sum(int(f['fileSize']) for f in key),
        'all_AIHub103_bytes': all_bytes,
        'free_after_remaining_full_dataset_archives_bytes': free - (all_bytes - labels),
        'video_extraction_bytes': 'unknown: inspect each video ZIP directory before extraction',
        'planning_reserve_bytes': 300_000_000_000,
        'planning_reserve_is_assumption': True,
        'recommended_workflow': 'Select words, distinct people and front views; acquire one video ZIP at a time; budget extraction/cache; do not download all124.',
    },
    'stored_file_count_before_closeout': len(files),
    'stored_bytes_before_closeout': sum(p.stat().st_size for p in files),
    'network_new_saved_files': [r['catalog']['destination'], r['documents'][2]['destination'], s['documents'][0]['destination']],
    'new_corpus_video_downloads': 0, 'new_corpus_label_downloads': 0, 'model_training': 0,
    'remaining_active_transfer': False,
    'verification_first_failure_resolved': 'Original path-only filter omitted Validation file39483. Corrected exact filename categories and cross-checked all35 expected file IDs; no data changes.',
}
(U / 'VERIFIED_RESULT.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
auth = {
    'at_kst': out['at_kst'],
    'user_request': 'Identify needed datasets/capacity; download small annotation/needed materials first to newly added disk; may need pause after20:00.',
    'authorized_destination': str(ROOT),
    'completed_scope': 'Reuse existing WORD labels; small public catalog and official documentation only',
    'large_video_keypoint_downloads': 'Not started; this turn limited to small material',
    'automatic_resume_after20_00': False,
}
(U / 'AUTHORIZATION.json').write_text(json.dumps(auth, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'pass': True, 'capacity': out['capacity'], 'bytes_stored': out['stored_bytes_before_closeout'], 'metadata': inventory}, ensure_ascii=False))
