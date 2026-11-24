"""Cross-check exported inventory against unchanged source ZIP bodies and saved prior audit."""
import collections
import hashlib
import json
from pathlib import Path
import zipfile

U=Path(__file__).resolve().parent
F=U.parents[1]
load=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
rows=[json.loads(line) for line in (U/'FRONT_MANIFEST.jsonl').read_text(encoding='utf-8').splitlines()]
summary=load(U/'SUMMARY.json')
assert len(rows)==54000 and len({(r['split'],r['member']) for r in rows})==54000
sources=load(F/'ops/metadata_acquisition_20261011/ACQUISITION_RESULT.json')['labels']
verified=0
for src in sources:
    split=src['official_catalog_entry']['fileStreCours'].split('/')[1]
    with Path(src['destination']).open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==src['sha256']
    with zipfile.ZipFile(src['destination']) as z:
        expected={x.filename for x in z.infolist() if not x.is_dir() and x.filename.endswith('_F_morpheme.json')}
        subset=[r for r in rows if r['split']==split]
        assert expected=={r['member'] for r in subset}
        for r in subset:
            raw=z.read(r['member']); d=json.loads(raw.decode('utf-8-sig'))
            assert hashlib.sha256(raw).hexdigest()==r['source_payload_sha256']
            assert r['labels']==[a.get('name','') for e in d['data'] for a in e.get('attributes',[])]
            assert r['intervals']==[[e.get('start'),e.get('end')] for e in d['data']]
            assert r['metadata_clean']==(not r['problems'])
            if r['role']=='train_candidate':
                assert r['split']=='1.Training' and 1<=r['filename_signer_id']<=9
                assert r['folder_id']==r['filename_signer_id'] and r['extra_token'] is None
            if r['folder_id']!=r['filename_signer_id'] or r['extra_token']:
                assert not r['metadata_clean'] and r['role']=='identity_unresolved'
            verified+=1
assert verified==54000
for f in load(U/'WORD_FAMILIES.json'):
    rr=[r for r in rows if r['word_code']==f['word_code']]
    assert len(rr)==f['front_rows']==18
    assert sum(not r['metadata_clean'] for r in rr)==f['quarantined']
    assert collections.Counter(r['role'] for r in rr if r['metadata_clean'])==f['clean_role_counts']
by_label=collections.defaultdict(list)
for r in rows:
    if r['metadata_clean']: by_label[r['labels'][0]].append(r)
for l in load(U/'LABEL_COUNTS.json'):
    training=[r for r in by_label[l['label']] if r['role']=='train_candidate']
    assert len(training)==l['clean_train_clips']
    assert len({r['filename_signer_id'] for r in training})==l['clean_train_distinct_ids']
assert sum(not r['metadata_clean'] for r in rows)==summary['quarantined_front_rows']==83
assert len(load(U/'LOCAL_READY_METADATA.json'))==1499
catalog=load(U/'VIDEO_ACQUISITION_QUEUE.json')
assert len(catalog)==33 and sum(x['mapping_status']=='LOCAL_CENTRAL_DIRECTORY_VERIFIED' for x in catalog)==1
assert not any(x['download_now'] for x in catalog)
assert sum(x['fileSize'] for x in catalog)==1181218593950
out={'status':'PASS','method':'Codex separate raw-ZIP-to-output and aggregate checks; no independent agent claim',
     'source_rows_matched':verified,'source_label_hashes_unchanged':True,
     'word_codes':3000,'front_rows_per_code':18,'quarantined_front_rows':83,
     'no_heldout_role_reassigned_to_train':True,'raw_labels_and_intervals_preserved':True,
     'archive_catalog_count':33,'verified_local_archive_mapping':1,'unknown_other_archives':32,
     'local_ready_manifest_count':1499,'new_downloads':0,'training':0,'video_payload_read':0,
     'scope_limit':'Metadata identities and quality only; actual person, form correctness and RGB/MediaPipe quality unverified.'}
(U/'VERIFIED_RESULT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))
