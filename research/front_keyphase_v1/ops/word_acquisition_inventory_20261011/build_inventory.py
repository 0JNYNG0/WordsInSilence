"""Local morphology inventory. Never loads video payload, model or held-out arrays."""
from pathlib import Path
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
import re
import zipfile

UNIT = Path(__file__).resolve().parent
FRONT = UNIT.parents[1]
OLD = Path('C:/Users/local-user/.codex/.chatgpt-projects/g-p-6a12a39d590c8191960741b826fa75ff/TCN_일반화감사_20260918')
DATA = Path('W:/WordsInSilence/data/AIHub_103')
VIDEO = Path('C:/Users/local-user/Downloads/수어 영상/1.Training/[원천]01_real_word_video.zip')
RX = re.compile(r'NIA_SL_(WORD\d+)_REAL(\d+)_(?:(.+)_)?([DFLRU])(?:_morpheme\.json|\.mp4)$')

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write(name, value):
    with (UNIT / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def role(signer):
    if 1 <= signer <= 9: return 'train_candidate'
    if 11 <= signer <= 13: return 'development_only'
    if 14 <= signer <= 16: return 'reserved_metadata_only'
    if 17 <= signer <= 18: return 'official_validation_metadata_only'
    if signer == 10: return 'unassigned'
    return 'identity_unresolved'

def main():
    assert not (UNIT / 'SUMMARY.json').exists(), 'Completed inventory exists; reuse it'
    old = read(OLD / 'training_word_catalog.json')
    acquisition = read(FRONT / 'ops/metadata_acquisition_20261011/ACQUISITION_RESULT.json')
    policy = read(FRONT / 'ops/partial_encoder_pair_20260922/comparison_policy.json')
    metadata_path = Path(policy['paths']['metadata'])
    ready = read(FRONT / 'ops/partial_encoder_pair_20260922/TRAIN_PREFIX_READY.json')
    assert sha(metadata_path) == ready['metadata_sha256']
    metadata = read(metadata_path)
    train = metadata['base'] + metadata['pair_context']
    exposure_ids = {re.split(r'_a\d+', r['id'])[0] for r in train}
    exposure_signers = {r['signer_id'] for r in train if r.get('signer_id') is not None}
    exposure_labels = {r['label'] for r in train}
    scope = {
        'created_at_kst': dt.datetime.now().astimezone().isoformat(),
        'authorization': 'User: 뭐야 그럼 지금 너가 할 수 있는건 뭐야? 진행해봐.',
        'scope': 'Extend prior target18 training inventory to all WORD families and official Validation metadata; resolve available-video mapping; no downloads/training/inference/video decoding.',
        'old_target_inventory': str(OLD / 'training_word_catalog.json'),
        'existing_training_split': '01-09 candidate;10 unassigned;11-13 development;14-16 reserved metadata only;17-18 official Validation metadata only; conflicting IDs quarantined',
        'candidate_order': 'Descending number of distinct internally consistent training IDs per exact raw label. Display ties by label and WORD code; no automatic final top100 selection.',
        'quality_rules': 'Preserve raw label/code; reject empty/multiple segments, nonfinite/out-of-range times, name mismatch, folder/ID conflict, extra filename token from clean metadata counts. No manual relabel or clipping.',
        'meaning_limits': 'A source code is not a verified sign-form variant. Names/URLs are metadata evidence, not biometric signer verification. MediaPipe/RGB quality unmeasured.',
        'model_training_metadata_sha256': sha(metadata_path),
    }
    if not (UNIT/'SCOPE.json').exists(): write('SCOPE.json', scope)
    known_video = {}
    with zipfile.ZipFile(VIDEO) as z:
        for i in z.infolist():
            if i.filename.endswith('.mp4'):
                key = Path(i.filename).name[:-4]
                assert key not in known_video
                known_video[key] = {'archive_fileSn': 39546, 'member': i.filename, 'bytes': i.file_size,
                                    'compressed_bytes': i.compress_size, 'crc32': i.CRC}
    expected_video = read(OLD / 'word_video_archive_inventory.json')
    assert VIDEO.stat().st_size == expected_video['bytes'] and len(known_video) == expected_video['video_count']
    rows = []
    issue_counts = Counter()
    canonical_counts = Counter()
    canonical_train_labels = set()
    old_target_rows = {a['member']: a for values in old['matching_annotations'].values() for a in values}
    reused_target_checks = 0
    identity_details = []
    for archive in acquisition['labels']:
        path = Path(archive['destination'])
        assert sha(path) == archive['sha256']
        split = archive['official_catalog_entry']['fileStreCours'].split('/')[1]
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.is_dir() or not info.filename.endswith('_F_morpheme.json'): continue
                match = RX.search(info.filename)
                assert match is not None
                code, sid, extra, view = match.groups()
                sid = int(sid); folder = int(info.filename.split('/')[1])
                clip = Path(info.filename).name.removesuffix('_morpheme.json')
                doc = json.loads(z.read(info).decode('utf-8-sig'))
                meta = doc['metaData']; events = doc['data']
                name = meta.get('name', ''); url_basename = meta.get('url', '').rsplit('/', 1)[-1]
                duration = float(meta['duration'])
                labels = [a.get('name', '') for e in events for a in e.get('attributes', [])]
                intervals = [[e.get('start'), e.get('end')] for e in events]
                problems = []
                if folder != sid: problems.append('folder_filename_identity_conflict')
                if extra: problems.append('extra_filename_token')
                if name != clip + '.mp4': problems.append('metadata_name_mismatch')
                if url_basename != name: problems.append('url_basename_mismatch')
                if len(events) != 1 or len(labels) != 1: problems.append('not_one_label_segment')
                if not labels or any(not x.strip() for x in labels): problems.append('empty_label')
                import math
                valid_times = bool(events) and math.isfinite(duration) and duration > 0
                for a, b in intervals:
                    valid_times &= isinstance(a, (int,float)) and isinstance(b, (int,float)) and math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= duration
                if not valid_times: problems.append('invalid_time_interval')
                clean = not problems
                r = {'split': split, 'member': info.filename, 'clip_id': clip, 'word_code': code,
                     'folder_id': folder, 'filename_signer_id': sid, 'extra_token': extra,
                     'role': role(sid) if folder == sid and not extra else 'identity_unresolved',
                     'labels': labels, 'intervals': intervals, 'duration': duration,
                     'metadata_clean': clean, 'problems': problems,
                     'source_payload_sha256': hashlib.sha256(z.read(info)).hexdigest(),
                     'video': known_video.get(clip),
                     'exact_clip_in_current_train_metadata': clip in exposure_ids,
                     'signer_seen_in_current_train': sid in exposure_signers if folder == sid else None,
                     'label_seen_in_current_train': any(x in exposure_labels for x in labels)}
                rows.append(r); issue_counts.update(problems)
                if not extra:
                    canonical_counts[split] += 1
                    if split == '1.Training': canonical_train_labels.update(x.strip() for x in labels)
                if info.filename in old_target_rows and split == '1.Training':
                    prev = old_target_rows[info.filename]
                    assert prev['label'] in [x.strip() for x in labels]
                    assert [prev['start_s'], prev['end_s']] in intervals
                    assert prev['signer_id'] == sid
                    reused_target_checks += 1
                if folder != sid or extra:
                    identity_details.append({**{k:r[k] for k in ['split','member','word_code','folder_id','filename_signer_id','extra_token','problems']},
                                             'metadata_name': name, 'url_basename': url_basename,
                                             'resolution': 'Keep original and quarantine; metadata is insufficient to assign/correct the actual person.'})
    assert len(rows) == 54000
    assert canonical_counts['1.Training'] == old['canonical_front_clips']
    assert len(canonical_train_labels) == old['unique_label_strings']
    # Duplicate clip IDs, if any, cannot silently count as independent performances.
    counts = Counter(r['clip_id'] for r in rows)
    for r in rows:
        if counts[r['clip_id']] > 1:
            r['metadata_clean'] = False; r['problems'].append('duplicate_clip_id')
    with (UNIT / 'FRONT_MANIFEST.jsonl').open('x',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,separators=(',',':')) + '\n')
    family_groups = defaultdict(list)
    label_groups = defaultdict(list)
    for r in rows:
        family_groups[r['word_code']].append(r)
        if r['metadata_clean']: label_groups[r['labels'][0]].append(r)
    families = []
    for code, group in sorted(family_groups.items()):
        train_group = [r for r in group if r['role']=='train_candidate' and r['metadata_clean']]
        raw_train = Counter(r['labels'][0] for r in train_group)
        families.append({'word_code': code, 'raw_labels_all_metadata': dict(Counter(x for r in group for x in r['labels'])),
                         'clean_training_labels': dict(raw_train),
                         'clean_training_ids': sorted({r['filename_signer_id'] for r in train_group}),
                         'clean_role_counts': dict(Counter(r['role'] for r in group if r['metadata_clean'])),
                         'front_rows': len(group), 'quarantined': sum(not r['metadata_clean'] for r in group),
                         'local_front_video_count': sum(r['video'] is not None for r in group),
                         'local_clean_train_front_count': sum(r['video'] is not None for r in train_group),
                         'train_label_agreement': len(raw_train)==1})
    write('WORD_FAMILIES.json',families)
    labels = []
    for label, group in label_groups.items():
        tr = [r for r in group if r['role']=='train_candidate']
        ids = sorted({r['filename_signer_id'] for r in tr})
        labels.append({'label':label,'clean_train_distinct_ids':len(ids),'train_ids':ids,
                       'clean_train_clips':len(tr),'word_codes':sorted({r['word_code'] for r in group}),
                       'clean_role_counts':dict(Counter(r['role'] for r in group)),
                       'local_clean_training_videos':sum(r['video'] is not None for r in tr),
                       'already_model_label':label in exposure_labels})
    labels.sort(key=lambda r:(-r['clean_train_distinct_ids'],-r['clean_train_clips'],r['label']))
    write('LABEL_COUNTS.json',labels)
    write('IDENTITY_QUARANTINE.json',identity_details)
    target_rows=[]
    for target in old['target_words']:
        label = target['label']
        exact=[r for r in rows if label in r['labels']]
        target_rows.append({'target':label,'word_id':target['word_id'],'exact_word_codes':sorted({r['word_code'] for r in exact}),
                            'exact_front_rows':len(exact),'clean_train_ids':sorted({r['filename_signer_id'] for r in exact if r['metadata_clean'] and r['role']=='train_candidate'}),
                            'local_videos':sum(r['video'] is not None for r in exact),
                            'prior_training_catalog_reused':True,
                            'alias_candidates_not_merged':target.get('alias_candidates_not_merged',[])})
    write('TARGET18_COVERAGE.json',target_rows)
    local_clean=[r for r in rows if r['video'] is not None and r['metadata_clean'] and r['role']=='train_candidate']
    local_clean.sort(key=lambda r:(r['labels'][0],r['word_code']))
    write('LOCAL_READY_METADATA.json',local_clean)
    catalog=read(DATA/'manifests/AIHUB103_PUBLIC_CATALOG.json')['files']
    videos=[x for x in catalog if re.fullmatch(r'\[원천\]\d+_real_word_video.zip',x['streFileNm'])]
    mapping=[{**x,'mapping_status':'LOCAL_CENTRAL_DIRECTORY_VERIFIED' if x['fileSn']==39546 else 'MEMBER_MAPPING_UNKNOWN_NO_ARCHIVE_NUMBER_TO_SIGNER_ASSUMPTION',
              'download_now':False} for x in videos]
    write('VIDEO_ACQUISITION_QUEUE.json',mapping)
    write('LOCAL_ARCHIVE_MEMBERS.json',known_video)
    unknown = sum(r['video'] is None for r in rows)
    summary={'status':'COMPLETE_METADATA_INVENTORY_MEDIA_MAPPING_PARTIAL',
             'created_at_kst':dt.datetime.now().astimezone().isoformat(),
             'front_rows':len(rows),'word_codes':len(families),'clean_exact_label_strings':len(labels),
             'all_raw_label_strings':len({x for r in rows for x in r['labels']}),
             'canonical_training_labels_matches_old':len(canonical_train_labels),
             'reused_prior_target_rows_checked':reused_target_checks,
             'issue_counts':dict(issue_counts),'quarantined_front_rows':sum(not r['metadata_clean'] for r in rows),
             'identity_quarantine_rows':len(identity_details),'duplicate_clip_ids':sum(n>1 for n in counts.values()),
             'clean_training_id_distribution':dict(Counter(x['clean_train_distinct_ids'] for x in labels)),
             'local_front_videos':sum(r['video'] is not None for r in rows),
             'local_clean_train_front_videos':len(local_clean),
             'local_clean_front_video_bytes':sum(r['video']['bytes'] for r in local_clean),
             'local_all_front_video_bytes':sum(r['video']['bytes'] for r in rows if r['video']),
             'local_videos_people':sorted({r['filename_signer_id'] for r in rows if r['video']}),
             'unresolved_front_archive_mapping':unknown,
             'existing_archive_bytes_reused_without_copy_or_redownload':VIDEO.stat().st_size,
             'current_training_metadata_rows':len(train),'current_training_word_ids':sum('NIA_SL_WORD' in r['id'] for r in train),
             'current_training_signers':sorted(exposure_signers),
             'same_signer_different_clip_not_new_person_test':True,
             'ranking_is_final_vocabulary':False,'no_new_downloads':True,'training_runs':0,'inference_runs':0,
             'current_model_unchanged':True,'next':'Review first local valid WORD pilot metadata and index unresolved archives through official route; preserve person splits, no guessed large download.'}
    write('SUMMARY.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
