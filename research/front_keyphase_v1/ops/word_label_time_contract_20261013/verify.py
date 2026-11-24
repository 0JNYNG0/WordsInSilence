"""Separate standard-library SQL/count verification; no producer imports or models."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3

import numpy as np

U=Path(__file__).resolve().parent
def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (U/'INDEPENDENT_CHECK.json').exists()
    a=read(U/'ANALYSIS.json')
    assert all(digest(p)==v for p,v in a['source_hashes'].items())
    rows=[json.loads(x) for x in (U.parent/'word_dataset_plan_20261013/FRONT_VIDEO_JOIN.jsonl').read_text('utf-8').splitlines()]
    db=sqlite3.connect(':memory:')
    db.execute('create table records(code,clip,person,role,ok,raw,name,duration,start,end,bytes)')
    for r in rows:
        person=int(re.search(r'_REAL(\d+)_',r['clip_id']).group(1))
        raw=r['labels'][0] if len(r['labels'])==1 else None
        interval=r['intervals'][0] if len(r['intervals'])==1 else (None,None)
        db.execute('insert into records values(?,?,?,?,?,?,?,?,?,?,?)',(r['word_code'],r['clip_id'],person,r['role'],int(r['metadata_video_join_eligible']),raw,raw.strip() if raw else None,r['duration_metadata'],*interval,r['video_entries'][0]['bytes'] if len(r['video_entries'])==1 else 0))
    db.execute("create view tr as select * from records where role='train_candidate' and ok=1")
    sql="""select code from tr group by code having count(distinct name)=1
    and count(distinct person)=count(*) and count(*) between 8 and 9
    and min(name) not in(select name from tr group by name having count(distinct code)>1)"""
    good={r[0] for r in db.execute(sql)}
    expected={r['clip_id']:r for r in rows if r['word_code'] in good and r['role']=='train_candidate' and r['metadata_video_join_eligible']}
    manifest=[json.loads(x) for x in (U/'TRAINING_CANDIDATES_V2.jsonl').read_text('utf-8').splitlines()]
    assert len(good)==a['revised_candidates']==2494
    assert len(manifest)==len(expected)==a['revised_training_clips']==22388
    assert {r['clip_id'] for r in manifest}==set(expected)
    for r in manifest:
        orig=expected[r['clip_id']]
        assert r['raw_label']==orig['labels'][0] and r['derived_label']==r['raw_label'].strip()
        assert r['intervals']==orig['intervals'] and r['video_entries']==orig['video_entries']
    db.execute("update records set name='MUTATED',ok=0 where role<>'train_candidate'")
    assert good=={r[0] for r in db.execute(sql)}
    full=sum(r['duration_metadata'] for r in expected.values())
    core=sum(r['intervals'][0][1]-r['intervals'][0][0] for r in expected.values())
    assert abs(full-a['duration_seconds']['sum'])<1e-5 and abs(core-a['core_seconds']['sum'])<1e-5
    bounds=read(U/'TRAIN_TIME_BOUNDS.json')
    assert len(bounds)==len(expected) and {r['clip_id'] for r in bounds}==set(expected)
    for r in bounds:
        s,e=expected[r['clip_id']]['intervals'][0]
        assert r['start']==s and r['end']==e
        assert r['context_start']==max(0,s-.3) and r['context_end']==min(r['duration'],e+.3)
    pilot=read('W:/WordsInSilence/data/AIHub_103/pilot12_inputs_20261013/RESULT.json')
    audit={r['clip_id']:r for r in read(U/'PILOT_TIME_AUDIT.json')}
    totals=Counter()
    for c in pilot['clips']:
        assert digest(c['observation'])==c['observation_sha256']
        with np.load(c['observation']) as z:
            t=z['timestamp_us']/1e6; s,e=c['intervals'][0]; rr=audit[c['clip_id']]
            inside=[i for i,v in enumerate(t) if max(0,s-.3)<=v<=min(float(t[-1]),e+.3)]
            assert rr['context_frame_ids']==[int(z['frame_id'][i]) for i in inside]
            core_i=[i for i,v in enumerate(t) if s<=v<=e]
            assert set(core_i)<=set(inside) and inside==list(range(inside[0],inside[-1]+1))
            totals.update(full=len(t),core=len(core_i),context=len(inside))
    media=read(U/'MEDIA_REVIEW_INPUTS.json')
    assert len(media)==8 and len({r['clip_id'] for r in media})==8
    assert all(r['clip_id'] in {x['clip_id'] for x in rows if x['role']=='train_candidate'} for r in media)
    assert len(list(U.glob('case_*.jpg')))==4
    quarantined=[r for r in rows if int(re.search(r'_REAL(\d+)_',r['clip_id']).group(1))<=9 and not r['metadata_video_join_eligible']]
    originals=[json.loads(x) for x in (U.parent/'word_acquisition_inventory_20261011/FRONT_MANIFEST.jsonl').read_text('utf-8').splitlines()]
    by_clip={r['clip_id']:r for r in rows}
    folder_quarantined=[r for r in originals if int(r['folder_id'])<=9 and not by_clip[r['clip_id']]['metadata_video_join_eligible']]
    assert len(folder_quarantined)==70
    out=dict(status='PASS',verifier='Codex separate SQL and source reconstruction; not independent human/agent',
        checks=['source hashes unchanged','SQL candidate/code/row reconstruction','nontraining mutation invariant','raw label/paths/intervals preserved','time sum and contiguous context/core inclusion','pilot original observation hashes unchanged','only eight selected training videos reviewed'],
        candidate_codes=len(good),training_rows=len(expected),pilot_frame_counts=dict(totals),
        quarantine_with_filename_ids_01_09=len(quarantined),quarantine_by_filename_id=dict(Counter(int(re.search(r'_REAL(\d+)_',r['clip_id']).group(1)) for r in quarantined)),
        quarantined_training_folder_rows=len(folder_quarantined),quarantine_by_annotation_folder=dict(Counter(r['folder_id'] for r in folder_quarantined)),
        quarantine_note='Filename-based15 is not total training-folder70. Fifty-five additional rows have conflicting filename IDs (e.g. folder02/filename25). Neither folder nor filename alone fixes identity. No performer judged inaccurate.',
        semantic_gloss_expertise=False,classification_accuracy_measured=False)
    (U/'INDEPENDENT_CHECK.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__': main()
