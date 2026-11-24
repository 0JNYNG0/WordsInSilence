"""Separate SQL recomputation and linkage checks; does not rerun the producer."""
import hashlib
import importlib.util
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

U = Path(__file__).resolve().parent
summary = json.loads((U/'SUMMARY.json').read_text(encoding='utf-8'))
vocab = json.loads((U/'VOCABULARY.json').read_text(encoding='utf-8'))['codes']
source = U.parent/'word_dataset_plan_20261013/FRONT_VIDEO_JOIN.jsonl'
rows = [json.loads(x) for x in source.read_text(encoding='utf-8').splitlines()]
original = {r['clip_id']: r for r in rows}
assert len(original) == 54000
con = sqlite3.connect(':memory:')
con.execute('CREATE TABLE t (code TEXT, clip TEXT, person INTEGER, name TEXT)')
records = []
for r in rows:
    if r['role'] == 'train_candidate' and r['metadata_video_join_eligible']:
        person = int(re.search(r'_REAL(\d+)_F', r['clip_id']).group(1))
        assert person in range(1,10) and len(r['labels']) == 1
        records.append((r['word_code'], r['clip_id'], person, r['labels'][0]))
con.executemany('INSERT INTO t VALUES (?, ?, ?, ?)', records)
sql_candidates = {r[0] for r in con.execute('''
    WITH ambiguous AS (SELECT name FROM t GROUP BY name HAVING count(DISTINCT code)>1)
    SELECT code FROM t GROUP BY code
    HAVING count(*)=9 AND count(DISTINCT person)=9 AND min(person)=1 AND max(person)=9
    AND count(DISTINCT name)=1 AND min(name) NOT IN (SELECT name FROM ambiguous)
''')}
assert sql_candidates == {v['code'] for v in vocab if v['metadata_candidate']}
assert len(sql_candidates) == summary['metadata_candidate_codes']
sql_hist = dict(con.execute('SELECT n,count(*) FROM (SELECT code,count(DISTINCT person) n FROM t GROUP BY code) GROUP BY n'))
assert {str(k): v for k,v in sql_hist.items()} == summary['training_people_histogram']
sql_stats = {c: (n, people, labels) for c,n,people,labels in con.execute('SELECT code,count(*),count(DISTINCT person),count(DISTINCT name) FROM t GROUP BY code')}
for v in vocab:
    n,p,l = sql_stats[v['code']]
    assert n == v['training_clips'] and p == v['training_people'] and l == len(v['training_labels'])
expected = {r['clip_id'] for r in rows if r['word_code'] in sql_candidates}
seen = set()
role_counts = Counter()
bad = conflict = 0
candidate_bytes = Counter()
for line in (U/'CANDIDATE_ROLES.jsonl').read_text(encoding='utf-8').splitlines():
    row = json.loads(line)
    assert row['clip_id'] not in seen
    seen.add(row['clip_id'])
    raw = original[row['clip_id']]
    for key, value in raw.items():
        assert row[key] == value, (row['clip_id'], key)
    target = con.execute('SELECT DISTINCT name FROM t WHERE code=?', (row['word_code'],)).fetchone()[0]
    assert target == row['training_label']
    assert row['raw_label_matches_training_label'] == (raw['labels'] == [target])
    role_counts[row['role']+':all_rows'] += 1
    role_counts[row['role']+':eligible_rows'] += int(raw['metadata_video_join_eligible'])
    if not raw['metadata_video_join_eligible']:
        bad += 1
    else:
        assert len(raw['video_entries']) == 1
        candidate_bytes[row['role']] += raw['video_entries'][0]['bytes']
        conflict += int(raw['labels'] != [target])
        if raw['role'] == 'train_candidate':
            assert raw['labels'] == [target]
assert seen == expected
assert len(seen) == summary['candidate_role_manifest_rows'] == summary['candidate_unique_clip_ids']
assert dict(role_counts) == summary['candidate_role_counts']
assert dict(candidate_bytes) == summary['candidate_eligible_video_bytes_by_role']
assert bad == summary['candidate_retained_bad_metadata_rows']
assert conflict == summary['candidate_eligible_nontraining_label_conflicts']

# This is an explicit dependency test of the implementation, separate from the SQL count oracle.
spec = importlib.util.spec_from_file_location('pilot_manifest_builder', U/'build_manifest.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
baseline = builder.classify_training(rows)
training_only = [r for r in rows if r['role'] == 'train_candidate']
assert builder.classify_training(training_only) == baseline
mutated = []
for row in rows:
    if row['role'] == 'train_candidate':
        mutated.append(row)
    else:
        mutated.append(dict(row, labels=['__DELIBERATE_HELDOUT_MUTATION__'], metadata_video_join_eligible=False))
assert builder.classify_training(mutated) == baseline
for path, expected_sha in summary['source_hashes'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected_sha
page=(U/'단어_학습구성표.html').read_text(encoding='utf-8')
assert page.count('<tr data-kind=') == 3000
assert page.count('data-kind="candidate"') == len(sql_candidates)
result = dict(status='PASS', method='Codex separate SQLite recomputation plus raw row comparisons and explicit dependency tests; not another agent/expert',
              codes_checked=3000, candidates=len(sql_candidates), training_rows=len(records),
              candidate_rows_preserved=len(seen), candidate_training_rows=role_counts['train_candidate:eligible_rows'],
              nontraining_removal_invariant=True, nontraining_labels_and_quality_mutation_invariant=True,
              unresolved_candidate_rows_retained=bad, differing_nontraining_labels_retained=conflict,
              source_hashes_preserved=len(summary['source_hashes']), sign_form_verified=False,
              model_training=0, video_decode=0)
(U/'INDEPENDENT_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
