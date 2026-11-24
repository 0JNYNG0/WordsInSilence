from pathlib import Path
import json,hashlib,sys,time,os
from collections import Counter
from flow_guard import guard
U=Path(__file__).resolve().parent;O=U.parent
SPLIT=O/'word100_core_context_split_20261101';REL=O/'word100_relational_sequence_compare_20261031';AUD=O/'word100_crossperson_stage_audit_20261101'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def source_files():
    files=[U/n for n in ['AUTHORIZATION.json','PLAN_KO.txt','POLICY.json','flow_guard.py','run.py','review/PLAN_REVIEW_DRAFT.json','review/BASELINE_CAUSE_AUDIT.json','review/CODE_REVIEW.json']]
    files += [SPLIT/'RESULT_MANIFEST.json',AUD/'ASSESSMENT.json',AUD/'COMPLETION.json']
    for row in read(SPLIT/'RESULT_MANIFEST.json')['results']:
        files.append(Path(row['file']))
        files.append(REL/('pairs' if row['kind']=='person_pair' else 'controls')/(row['name']+'.json'))
    return list(dict.fromkeys(files))
def freeze():
    assert read(U/'review/CODE_REVIEW.json')['status'].startswith('PASS')
    assert read(U/'review/PLAN_REVIEW_DRAFT.json')['status'].startswith('PASS')
    write(U/'FROZEN.json',dict(status='FROZEN_BEFORE_POLICY_EXECUTION',new_candidate_count=1,
      mechanism='unchanged full-sequence path plus generic local-window agreement policy',
      controls_baseline='Saved relational-feature full path; different origin from person-pair baseline. No pooled baseline accuracy claim.',
      people_baseline='Saved original global DTW path',no_new_search=True,no_training=True,
      expected=dict(person_pairs=36,controls=126,person_anchor_opportunities=117),
      pins={str(p):sha(p) for p in source_files()}))
def run():
    f=read(U/'FROZEN.json');assert all(sha(p)==h for p,h in f['pins'].items())
    assert not (U/'RESULT_MANIFEST.json').exists();write(U/'RUNNING.json',dict(pid=os.getpid(),started=time.time()))
    (U/'results').mkdir(exist_ok=True);manifest=[];rows=[]
    audit={(e['case'],e['segment_id']):e for e in read(AUD/'ASSESSMENT.json')['events']}
    for row in read(SPLIT/'RESULT_MANIFEST.json')['results']:
        d=read(row['file']);assert sha(row['file'])==row['sha256']
        p=REL/('pairs' if row['kind']=='person_pair' else 'controls')/(row['name']+'.json');base=read(p)
        n,m=base['query_frames'],base['reference_frames'];start=time.perf_counter()
        result=guard(base['path'],n,m,d['anchors']);elapsed=time.perf_counter()-start
        result.update(name=row['name'],kind=row['kind'],query=d['query'],reference=d['reference'],
          role=d['role'],control=d.get('control'),source_full_path=str(p),source_full_sha256=sha(p),
          source_local_result=row['file'],source_local_sha256=row['sha256'],incremental_policy_seconds=elapsed,
          frozen_sha256=sha(U/'FROZEN.json'))
        assert result['path']==base['path']
        for a in result['anchors']:
            # Retain original observation metrics byte-for-value; this policy does not re-score them.
            stage=next(x for x in base['stages'] if x['segment_id']==a['segment_id'])
            a['unchanged_global_groups']=stage['groups']
            if row['kind']=='person_pair':
                prior=audit[(row['name'],a['segment_id'])]
                a['saved_physical_audit_global']=prior['methods']['global_dtw']
                a['saved_physical_audit_local']=prior['methods']['core_first']
        dest=U/'results'/(row['name']+'.json');write(dest,result)
        manifest.append(dict(file=str(dest),sha256=sha(dest),kind=row['kind'],name=row['name']))
        rows.append(result)
    assert len(rows)==162 and sum(x['kind']=='person_pair' for x in rows)==36
    assert sum(len(x['anchors']) for x in rows if x['kind']=='person_pair')==117
    summary={}
    for group,subset in [('fit',[x for x in rows if x['kind']=='person_pair' and x['role']=='fit']),('dev',[x for x in rows if x['kind']=='person_pair' and x['role']!='fit'])]+[(k,[x for x in rows if x.get('control')==k]) for k in sorted({x['control'] for x in rows if x.get('control')})]:
        anchors=[a for x in subset for a in x['anchors']]
        summary[group]=dict(cases=len(subset),anchors=len(anchors),range_relation=dict(Counter(a['range_relation'] for a in anchors)),status=dict(Counter(a['status'] for a in anchors)),flags=dict(Counter(flag for a in anchors for flag in a['flags'])),path_unchanged=True)
    write(U/'SUMMARY.json',dict(status='POLICY_COMPLETE_PENDING_VERIFICATION',summary=summary,full_path_replacement=False,alignment_accuracy_improved=False,old_physical_summary=read(AUD/'ASSESSMENT.json')['summary']['dev'],no_local_agreement_is_correctness=True))
    write(U/'RESULT_MANIFEST.json',dict(results=manifest));(U/'RUNNING.json').unlink()
    print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
