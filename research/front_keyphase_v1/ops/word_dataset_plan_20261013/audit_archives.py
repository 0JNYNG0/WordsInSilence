"""Read ZIP directories only; reuse verified annotation manifest. No media extraction."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import re
import zipfile

UNIT = Path(__file__).resolve().parent
OPS = UNIT.parent
ROOT = Path('W:/WordsInSilence/data/AIHub_103')
RX = re.compile(r'NIA_SL_(WORD\d+)_REAL(\d+)_(?:(.+)_)?([DFLRU])\.mp4$')

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def save(name, obj):
    with (UNIT/name).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main():
    assert not (UNIT/'ARCHIVE_AUDIT.json').exists(), 'Reuse completed audit'
    plan = read(OPS/'word_video_download_20261011/ACQUISITION_PLAN.json')
    actual = defaultdict(list)
    all_files = list((ROOT/'videos').rglob('*'))
    for p in all_files:
        if p.is_file() and p.suffix.lower() == '.zip':
            split = next((s for s in p.parts if s in ('1.Training','2.Validation')), 'unknown')
            actual[(split, p.name)].append(p)
    by_view, view_bytes, by_split, by_folder, by_filename_id = (Counter() for _ in range(5))
    groups = defaultdict(set)
    front = defaultdict(list)
    archives, odd, mismatches = [], [], []
    total_payload = 0
    names_hasher = hashlib.sha256()
    for entry in plan['archives']:
        split = entry['fileStreCours'].split('/')[1]
        paths = actual[(split, entry['streFileNm'])]
        assert len(paths) == 1, (entry['streFileNm'], paths)
        p = paths[0]
        before = p.stat()
        assert before.st_size == entry['fileSize'], (p, before.st_size, entry['fileSize'])
        a = {'path': str(p), 'split': split, 'fileSn': entry['fileSn'],
             'bytes': before.st_size, 'size_matches_catalog': True, 'views': Counter(),
             'uncompressed_bytes': 0, 'front_bytes': 0, 'other_files': []}
        with zipfile.ZipFile(p) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                a['uncompressed_bytes'] += info.file_size
                total_payload += info.file_size
                names_hasher.update(f'{entry["fileSn"]}\t{info.filename}\t{info.file_size}\t{info.CRC}\n'.encode())
                m = RX.search(info.filename)
                if not m:
                    a['other_files'].append(info.filename)
                    odd.append({'archive': str(p), 'member': info.filename})
                    continue
                code, sid, extra, view = m.groups()
                parts = info.filename.split('/')
                folder_match = re.fullmatch(r'(\d+)(?:-1)?', parts[-2])
                assert folder_match, (p, info.filename)
                folder = int(folder_match.group(1))
                sid = int(sid)
                by_view[view] += 1
                view_bytes[view] += info.file_size
                by_split[split] += 1
                by_folder[str(folder)] += 1
                by_filename_id[str(sid)] += 1
                a['views'][view] += 1
                groups[(split, folder, code, sid, extra)].add(view)
                if folder != sid or extra:
                    mismatches.append({'archive': str(p), 'member': info.filename, 'folder_id': folder,
                                       'filename_id': sid, 'extra': extra})
                if view == 'F':
                    a['front_bytes'] += info.file_size
                    clip_id = Path(info.filename).stem
                    front[(split, clip_id)].append({'archive': str(p), 'member': info.filename,
                        'word_code': code, 'folder_id': folder, 'folder_raw': parts[-2], 'filename_signer_id': sid,
                        'extra': extra, 'bytes': info.file_size, 'compressed_bytes': info.compress_size,
                        'crc32': info.CRC})
        after = p.stat()
        assert (before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns), p
        archives.append(a)
    src = OPS/'word_acquisition_inventory_20261011/FRONT_MANIFEST.jsonl'
    rows = [json.loads(l) for l in src.read_text(encoding='utf-8').splitlines() if l.strip()]
    matched, clean, missing, duplicates, joined_issues = 0,0,[],[],[]
    role_counts, role_bytes, durations = Counter(), Counter(), Counter()
    eligible_codes = defaultdict(set)
    labels_by_code = defaultdict(set)
    with (UNIT/'FRONT_VIDEO_JOIN.jsonl').open('x',encoding='utf-8') as out:
        for row in rows:
            videos = front.get((row['split'],row['clip_id']), [])
            eligible = False
            if not videos:
                missing.append(row['clip_id'])
            elif len(videos) != 1:
                duplicates.append(row['clip_id'])
            else:
                matched += 1
                v = videos[0]
                identity_match = (v['folder_id'] == row['folder_id'] and
                                  v['filename_signer_id'] == row['filename_signer_id'])
                eligible = bool(row['metadata_clean'] and identity_match and not v['extra'])
                if eligible:
                    clean += 1
                    role_counts[row['role']] += 1
                    role_bytes[row['role']] += v['bytes']
                    durations[row['role']] += row['duration']
                    eligible_codes[row['word_code']].add(row['filename_signer_id'])
                    labels_by_code[row['word_code']].update(row['labels'])
                elif row['metadata_clean']:
                    joined_issues.append({'clip_id':row['clip_id'],'reason':'video_annotation_identity_mismatch'})
            out.write(json.dumps({'split':row['split'],'clip_id':row['clip_id'],
                'word_code':row['word_code'],'labels':row['labels'],'role':row['role'],
                'duration_metadata':row['duration'],'intervals':row['intervals'],
                'prior_metadata_clean':row['metadata_clean'],'problems':row['problems'],
                'metadata_video_join_eligible':eligible,'video_entries':videos},ensure_ascii=False)+'\n')
    annotation_keys = {(r['split'],r['clip_id']) for r in rows}
    unmatched_videos = [str(k) for k in front if k not in annotation_keys]
    result = {
      'at_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
      'scope':'Archive central directory and existing annotation manifest only; no video payload/CRC/decode/pose/model reads.',
      'author_and_verifier':'Codex local audit; independent enumeration recorded separately',
      'archives':archives,'archive_count':len(archives),'archive_bytes':sum(a['bytes'] for a in archives),
      'all_sizes_match_catalog':True,'all_archives_directory_readable':True,
      'full_payload_integrity_verified':False,'uncompressed_all_bytes':total_payload,
      'video_count':sum(by_view.values()),'by_view':by_view,'uncompressed_bytes_by_view':view_bytes,
      'by_split':by_split,'by_folder_id':by_folder,'by_filename_id':by_filename_id,
      'multiview_group_count':len(groups),'complete_five_view_groups':sum(len(v)==5 for v in groups.values()),
      'other_file_count':len(odd),'other_files':odd,'identity_or_extra_token_issues':mismatches,
      'front_annotation_count':len(rows),'front_video_count':sum(len(v) for v in front.values()),
      'matched_annotation_count':matched,'metadata_video_join_eligible':clean,
      'missing_front_videos':missing,'duplicate_front_keys':duplicates,'extra_front_videos':unmatched_videos,
      'new_join_issues':joined_issues,'eligible_role_counts':role_counts,'eligible_role_bytes':role_bytes,
      'eligible_role_duration_seconds_metadata':durations,
      'all18_single_label_codes':sum(len(s)==18 and len(labels_by_code[c])==1 for c,s in eligible_codes.items()),
      'complete9_training_single_label_codes':sum(set(range(1,10)).issubset(s) and len(labels_by_code[c])==1 for c,s in eligible_codes.items()),
      'source_manifest_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
      'archive_directory_metadata_sha256':names_hasher.hexdigest(),
      'planned_zip_count':plan['all_archive_count'],
      'unexpected_video_directory_files':[str(p) for p in all_files if p.is_file() and p.suffix.lower()!='.zip'],
      'limits':['Directory readability and matching byte lengths are not full corruption verification.',
                'Folder/filename IDs are metadata, not biometric identity verification.',
                'Five camera views of one performance are not five independent performances.',
                'Metadata-clean rows still need visual/semantic/MediaPipe quality audit.']}
    save('ARCHIVE_AUDIT.json',result)
    brief = {k:v for k,v in result.items() if k not in ('archives','identity_or_extra_token_issues','other_files','new_join_issues','missing_front_videos','extra_front_videos')}
    brief.update(missing_front_count=len(missing),extra_front_count=len(unmatched_videos),new_join_issue_count=len(joined_issues))
    print(json.dumps(brief,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    main()
