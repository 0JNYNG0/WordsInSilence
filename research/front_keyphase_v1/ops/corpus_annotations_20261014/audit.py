"""Read-only tar audit. Never extract or execute supplied archive files."""
import collections as C
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sqlite3
import tarfile
import time

OUT = Path(__file__).resolve().parent
SOURCE = Path('W:/ksl_json_sets.tar.gz')
ROOT = OUT.parents[1]

def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def counter_dict(c):
    return dict(c.most_common())

def main():
    started = time.time()
    stat0 = SOURCE.stat()
    dbpath = OUT / 'ANNOTATION_INDEX.sqlite'
    if dbpath.exists():
        raise RuntimeError('Existing index is preserved; do not silently rerun.')
    db = sqlite3.connect(dbpath)
    db.executescript('''
    CREATE TABLE documents (doc INTEGER PRIMARY KEY, year TEXT, path TEXT UNIQUE, bytes INTEGER,
      json_id TEXT, video TEXT, signer TEXT, text_sha256 TEXT, fps TEXT, resolution TEXT);
    CREATE TABLE events (doc INTEGER, gloss TEXT, express TEXT, start REAL, end REAL, tiers INTEGER);
    ''')
    years = {}
    words = {}
    index_rows = {}
    actual_sizes = {}
    errors = []
    problems = C.Counter()
    examples = {}
    processed = 0
    def issue(kind, path, detail=None):
        problems[kind] += 1
        if len(examples.setdefault(kind, [])) < 6:
            examples[kind].append({'path': path, 'detail': detail})
    def init_year():
        return dict(documents=0, bytes=0, top_keys=C.Counter(), metadata_keys=C.Counter(),
                    signers=C.Counter(), signer_profiles=C.defaultdict(C.Counter),
                    fps=C.Counter(), resolution=C.Counter(), topics=C.Counter(), categories=C.Counter(),
                    raw_tiers=C.Counter(), expressions=C.Counter(), valid_unique_events=0,
                    valid_primary_s=0, empty_sign_script=0, nms_documents=0, nms_tags=C.Counter(),
                    primary_duration_bins=C.Counter(), video_length_present=0,
                    declared_file_size_present=0, declared_file_size_sum=0., coordinate_keys=C.Counter())
    with tarfile.open(SOURCE, 'r|gz') as tf:
        for member in tf:
            if not member.isfile():
                continue
            path = member.name
            parts = path.split('/')
            if path.endswith('/_index.csv'):
                rows = list(csv.DictReader(io.StringIO(tf.extractfile(member).read().decode('utf-8-sig'))))
                index_rows[parts[1]] = rows
                continue
            if not path.endswith('.json'):
                continue
            year = parts[1]
            ys = years.setdefault(year, init_year())
            raw = tf.extractfile(member).read()
            actual_sizes[path] = member.size
            try:
                obj = json.loads(raw)
                if not isinstance(obj, dict):
                    raise ValueError('top level is not object')
            except Exception as exc:
                errors.append({'path': path, 'error': str(exc)})
                continue
            processed += 1
            ys['documents'] += 1
            ys['bytes'] += len(raw)
            ys['top_keys'].update(obj.keys())
            for key in ('landmarks', 'keypoints', 'pose', 'hand_landmarks'):
                if key in obj:
                    ys['coordinate_keys'][key] += 1
            meta = obj.get('potogrf') or {}
            ys['metadata_keys'].update(meta.keys())
            signer = str(meta.get('sl_speaker_id') or '').strip()
            ys['signers'][signer] += 1
            profile = tuple(str(meta.get(k, '')) for k in ('sl_speaker_sex', 'sl_speaker_age', 'sl_speaker_legion', 'sl_speaker_hand'))
            ys['signer_profiles'][signer][profile] += 1
            fps = str(meta.get('fps', ''))
            resolution = f"{meta.get('width', '')}x{meta.get('height', '')}"
            ys['fps'][fps] += 1
            ys['resolution'][resolution] += 1
            ys['video_length_present'] += bool(meta.get('length'))
            fs = meta.get('fileSize')
            if isinstance(fs, (float, int)) and math.isfinite(fs):
                ys['declared_file_size_present'] += 1
                ys['declared_file_size_sum'] += fs
            korean = obj.get('krlgg_sntenc') or {}
            ys['topics'][str(korean.get('thema', ''))] += 1
            ys['categories'][str(korean.get('category', ''))] += 1
            video = str(obj.get('vido_file_nm') or '').strip()
            jid = str(obj.get('id') or '')
            if not video:
                issue('missing_video', path)
            if not signer:
                issue('missing_signer', path)
            if Path(video).stem != jid:
                issue('json_id_video_name_mismatch', path, [jid, video])
            txt = str(korean.get('koreanText') or '')
            tsha = hashlib.sha256(txt.encode('utf-8')).hexdigest() if txt else ''
            db.execute('INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?)',
                       (processed, year, path, len(raw), jid, video, signer, tsha, fps, resolution))
            script = obj.get('sign_script') or {}
            events = {}
            raw_valid = 0
            for tier, mask in [('sign_gestures_strong', 1), ('sign_gestures_weak', 2)]:
                items = script.get(tier) or []
                if not isinstance(items, list):
                    issue('invalid_tier_type', path, tier)
                    continue
                ys['raw_tiers'][tier] += len(items)
                starts = []
                for token in items:
                    if not isinstance(token, dict):
                        issue('invalid_event_object', path, tier)
                        continue
                    gloss = str(token.get('gloss_id') or '').strip()
                    express = str(token.get('express') or '').strip()
                    ys['expressions'][express] += 1
                    a, b = token.get('start'), token.get('end')
                    try:
                        a, b = float(a), float(b)
                        good = math.isfinite(a) and math.isfinite(b) and a >= 0 and b > a
                    except (ValueError, TypeError):
                        good = False
                    if not good or not gloss:
                        issue('invalid_time_or_empty_gloss', path, token)
                        continue
                    starts.append(a)
                    key = (gloss, express, a, b)
                    if events.get(key, 0) & mask:
                        issue('exact_duplicate_within_tier', path, list(key))
                    events[key] = events.get(key, 0) | mask
                    raw_valid += 1
                if starts != sorted(starts):
                    issue('tier_not_time_sorted', path, tier)
            if not events:
                ys['empty_sign_script'] += 1
            ys['valid_unique_events'] += len(events)
            problems['raw_valid_events_minus_exact_union_events'] += raw_valid - len(events)
            db.executemany('INSERT INTO events VALUES (?,?,?,?,?,?)',
                [(processed, *key, mask) for key, mask in events.items()])
            seen_words = set()
            primary = []
            for (gloss, express, a, b), mask in events.items():
                if mask == 3:
                    problems['exact_shared_hand_events'] += 1
                if not (mask & 1) or express != 's':
                    continue
                ys['valid_primary_s'] += 1
                primary.append((a, b, gloss))
                key = (year, gloss)
                ws = words.setdefault(key, dict(events=0, documents=0, signers=C.Counter(), duration_sum=0., example=None))
                ws['events'] += 1
                ws['signers'][signer] += 1
                ws['duration_sum'] += b - a
                if key not in seen_words:
                    ws['documents'] += 1
                    seen_words.add(key)
                if ws['example'] is None:
                    ws['example'] = dict(path=path, video=video, signer=signer, start=a, end=b)
                duration = b - a
                bucket = '<0.1s' if duration < .1 else '0.1-0.25s' if duration < .25 else '0.25-1s' if duration < 1 else '1-3s' if duration < 3 else '>=3s'
                ys['primary_duration_bins'][bucket] += 1
            primary.sort()
            if any(a[1] > b[0] for a, b in zip(primary, primary[1:])):
                issue('primary_s_document_with_overlap', path)
            nms = obj.get('nms_script') or {}
            if any(isinstance(v, list) and len(v) for v in nms.values()):
                ys['nms_documents'] += 1
            for key, val in nms.items():
                if isinstance(val, list):
                    ys['nms_tags'][key] += len(val)
            if processed % 20000 == 0:
                db.commit()
                print(json.dumps(dict(processed=processed, elapsed_sec=round(time.time()-started, 1))), flush=True)
    db.commit()
    print('Building query indexes and checking metadata totals', flush=True)
    db.executescript('CREATE INDEX event_doc ON events(doc); CREATE INDEX event_gloss ON events(gloss,express); CREATE INDEX doc_year ON documents(year);')
    integrity = db.execute('PRAGMA integrity_check').fetchone()[0]
    duplicates = {}
    for key in ('json_id', 'video', 'text_sha256'):
        duplicates[key] = [dict(year=y, total=n, unique=distinct, surplus=n-distinct) for y,n,distinct in
                          db.execute(f'SELECT year,COUNT(*),COUNT(DISTINCT {key}) FROM documents GROUP BY year')]
    index_checks = {}
    for year, rows in index_rows.items():
        rowpaths = [f'ksl_json_sets/{year}/' + r['rel_path'].replace('\\', '/') for r in rows]
        expected = {p for p in actual_sizes if p.split('/')[1] == year}
        index_checks[year] = dict(rows=len(rows), unique_paths=len(set(rowpaths)),
            missing_members=len(set(rowpaths)-expected), unindexed_members=len(expected-set(rowpaths)),
            byte_mismatches=sum(actual_sizes.get(p) != int(r['dst_bytes']) for p,r in zip(rowpaths, rows)),
            ok=counter_dict(C.Counter(r['ok'] for r in rows)),
            removed_fields=counter_dict(C.Counter(r['removed_fields'] for r in rows)),
            original_bytes_claimed=sum(int(r['src_bytes']) for r in rows),
            copy_bytes=sum(int(r['dst_bytes']) for r in rows))
    ranked = []
    for (year, gloss), ws in words.items():
        ranked.append(dict(year=year, gloss=gloss, events=ws['events'], documents=ws['documents'],
            signer_count=len([s for s in ws['signers'] if s]), signer_events=counter_dict(ws['signers']),
            mean_duration_s=ws['duration_sum']/ws['events'], example=ws['example']))
    ranked.sort(key=lambda x:(x['year'], -x['signer_count'], -x['events'], x['gloss']))
    write('WORD_DISTRIBUTION.json', ranked)
    frozen = json.loads((ROOT/'ops/word100_training_20261013/FROZEN_EXPERIMENT.json').read_text('utf-8-sig'))
    labels = frozen['labels']
    overlap = {}
    for label in labels + ['먹다','공부','너','감사합니다','운동']:
        # Discovery ONLY: never use this spelling rule as a semantic label mapping.
        pattern = re.compile(re.escape(label)+r'(?:\d+)?(?:#.*)?$')
        matches = [dict(year=r['year'], gloss=r['gloss'], events=r['events'], signer_count=r['signer_count'])
                   for r in ranked if pattern.fullmatch(r['gloss'])]
        overlap[label] = matches
    write('CURRENT100_LABEL_DISCOVERY.json', dict(warning='Spelling candidates only; suffixes/semantics must be reviewed; no automatic merge.', current100=labels, matches=overlap))
    for year, ys in years.items():
        ys['signer_profiles'] = {s:[dict(profile=list(p),documents=n) for p,n in pc.most_common()] for s,pc in ys['signer_profiles'].items()}
        for key, val in list(ys.items()):
            if isinstance(val, C.Counter):
                ys[key] = counter_dict(val)
        ww = [w for w in ranked if w['year']==year]
        ys['primary_s_distinct_raw_glosses'] = len(ww)
        ys['gloss_signer_count_distribution'] = counter_dict(C.Counter(w['signer_count'] for w in ww))
        ys['glosses_ge10_events_ge5_signers'] = sum(w['events']>=10 and w['signer_count']>=5 for w in ww)
        ys['top20_by_frequency'] = sorted(ww, key=lambda w:(-w['events'], w['gloss']))[:20]
    stat1 = SOURCE.stat()
    result = dict(status='COMPLETE', source=str(SOURCE), source_stat_unchanged=(stat0.st_size,stat0.st_mtime_ns)==(stat1.st_size,stat1.st_mtime_ns),
        documents_parsed=processed, parse_errors=errors, years=years, index_checks=index_checks,
        duplicate_fields=duplicates, issues=counter_dict(problems), issue_examples=examples,
        sqlite_integrity=integrity, events_sql_count=db.execute('SELECT COUNT(*) FROM events').fetchone()[0],
        all_primary_s_label_strings=len(set(w['gloss'] for w in ranked)),
        current100_spelling_candidates=sum(bool(overlap[l]) for l in labels),
        elapsed_seconds=time.time()-started,
        definitions={'events':'Exact (gloss,express,start,end) union across two tiers within a document; not guaranteed independent performances.',
          'word_ranking':'Valid strong-tier express=s events only; exact duplicate intervals removed. Strong not assumed anatomical right hand.',
          'signer_count':'Nonempty raw sl_speaker_id within each year, not cross-year confirmed identities.',
          'runtime':'Codex local audit; no training, no video download, no original extraction or execution.'})
    write('FULL_AUDIT.json', result)
    db.close()
    print(json.dumps(dict(status='COMPLETE', documents=processed, errors=len(errors), seconds=round(time.time()-started,1)),ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
