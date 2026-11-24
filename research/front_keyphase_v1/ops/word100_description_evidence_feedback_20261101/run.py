from pathlib import Path
import json,hashlib,sys,time,os
from collections import Counter
import numpy as np
from evidence import trace,feedback,source_cards,runs
U=Path(__file__).resolve().parent;O=U.parent
REL=O/'word100_relational_sequence_compare_20261031'
OLD=O/'word100_ordered_motion_alignment_20261101'
PHASE=O/'word100_core_stage_evidence_20261031'
sys.path.insert(0,str(REL))
from relational import features,GROUPS,NAMES

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,allow_nan=False,separators=(',',':'))

def freeze():
    assert read(U/'review/PLAN_REVIEW.json')['status'].startswith('PASS')
    assert read(U/'review/CODE_REVIEW.json')['status'].startswith('PASS')
    a=read(U/'AUTHORIZATION.json')
    files=[U/n for n in ['AUTHORIZATION.json','PLAN_KO.txt','evidence.py','run.py','review/PLAN_REVIEW.json','review/CODE_REVIEW.json']]
    files += [REL/n for n in ['relational.py','SOURCE_FEATURE_CONTRACT.json','REFERENCE_RANGES.json','DEVELOPMENT.json']]
    files += [PHASE/'PHASE_DATA.json',OLD/'RESULT_MANIFEST.json',OLD/'COMPLETION.json']
    for x in read(OLD/'RESULT_MANIFEST.json')['results']:
        if x['kind']=='person_pair':
            assert sha(x['file'])==x['sha256'];files += [Path(x['file']),REL/'pairs'/(x['name']+'.json')]
    files += [Path(r['input']) for r in a['rows']]
    protected=read(O/'word100_semantic_episode_pilot_20261031/FROZEN.json')['protected']
    assert all(sha(p)==h for p,h in protected.items())
    write(U/'FROZEN.json',dict(pins={str(p):sha(p) for p in files},protected=protected,
        policy='saved-global-path/source-anchor-context; no semantic verdict; all missing retained',
        data_scope='consumed four expressions, fit persons1-3 and development person4',
        new_training=False,new_alignment=False,threshold_tuning=False))

def run():
    f=read(U/'FROZEN.json');assert all(sha(p)==h for p,h in f['pins'].items())
    assert not (U/'RESULT_MANIFEST.json').exists()
    write(U/'RUNNING.json',dict(pid=os.getpid(),started=time.time()))
    rows=read(U/'AUTHORIZATION.json')['rows'];by={r['alias']:r for r in rows}
    arrays={r['alias']:dict(np.load(r['input'])) for r in rows}
    feat={a:features(z) for a,z in arrays.items()}
    clips={c['alias']:c for c in read(PHASE/'PHASE_DATA.json')['clips']}
    contract=read(REL/'SOURCE_FEATURE_CONTRACT.json')
    clauses={c['id']:dict(c,word_id=e['word_id']) for e in contract['expressions'] for c in e['clauses']}
    decisions={(d['query'],d['reference'],x['segment_id'],x['group']):x
        for d in read(REL/'DEVELOPMENT.json')['results'] if d['condition']=='B_same_path' for x in d['decisions']}
    ranges={(d['reference'],d['segment_id'],d['group']):d for d in read(REL/'REFERENCE_RANGES.json')['ranges']['B_same_path']}
    results=[];manifest=[];start=time.perf_counter()
    for m in read(OLD/'RESULT_MANIFEST.json')['results']:
        if m['kind']!='person_pair':continue
        g=read(m['file']);q,r=g['query'],g['reference'];b=read(REL/'pairs'/(m['name']+'.json'))
        assert g['path']==b['path'];assert by[r]['role']=='fit' and by[q]['word_id']==by[r]['word_id']
        segs={s['segment_id']:s for s in clips[r]['segments']}
        stages=[]
        for s in b['stages']:
            sid=s['segment_id'];seg=segs[sid];guard=next(x for x in g['anchors'] if x['segment_id']==sid)
            pp=[p for p in b['path'] if s['reference_core'][0]<=p[1]<=s['reference_core'][1]]
            groups={}
            for group,metric in s['groups'].items():
                t=trace(feat[q][group],feat[r][group],pp,s['reference_core'])
                # Numerical check against STORED scores, not a new score or candidate selection.
                cnt=sum(x['observed_components'] for x in t)
                total=sum(sum(v or 0 for v in x['difference_by_component']) for x in t)
                assert cnt==metric['compared']
                if cnt:assert abs(total/cnt-metric['mean_difference'])<1e-10
                assert sum(x['expected_components'] for x in t)==metric['expected']
                assert sum(x['path_missing_cells'] for x in t)==metric['path_missing_cells']
                d=decisions.get((q,r,sid,group));rg=ranges.get((r,sid,group))
                groups[group]=dict(name=NAMES[group],saved_metric=metric,saved_decision=d,saved_range=rg,
                    feedback=feedback(metric,d),trace=t,
                    reference_missing_ranges=runs(x['reference_frame'] for x in t if x['expected_components']<x['possible_components']),
                    query_missing_reference_ranges=runs(x['reference_frame'] for x in t if x['path_missing_cells']))
            cards=source_cards(seg,r,by[r]['word_id'],clauses)
            for c in cards:
                c['query_anchor_links']=[dict(reference_frame=j,query_frames=[i for i,k in pp if k==j]) for j in c['reference_anchor_frames']]
            stages.append(dict(id=sid,reference_core=s['reference_core'],query_frames=s['query_frames'],
                path=pp,reference_rgb_description=seg['description_ko'],source_cards=cards,
                source_uncovered_anchors=seg.get('source_uncovered_anchors',[]),
                source_scope='listed source anchors only; numeric metrics cover full physical core',
                alignment=dict(status=guard['status'],flags=guard['flags'],range_relation=guard['range_relation'],
                    not_proof_of_correct_stage=True),groups=groups))
        result=dict(id=m['name'],query=q,reference=r,role=by[q]['role'],label=by[q]['label'],
            query_person=by[q]['person'],reference_person=by[r]['person'],
            query_pts=(arrays[q]['timestamp_us']/1e6).tolist(),reference_pts=(arrays[r]['timestamp_us']/1e6).tolist(),
            path=b['path'],stages=stages,source_pair=str(REL/'pairs'/(m['name']+'.json')),
            source_pair_sha256=sha(REL/'pairs'/(m['name']+'.json')),
            learned_description_understanding=False,educational_correctness=None)
        dest=U/'results'/(m['name']+'.json');write(dest,result)
        manifest.append(dict(file=str(dest),sha256=sha(dest)));results.append(result)
    summary={}
    for role in ['fit','development']:
        rr=[r for r in results if r['role']==role];st=[s for r in rr for s in r['stages']]
        gs=[g for s in st for g in s['groups'].values()]
        linked=[g for s in st for key,g in s['groups'].items() if any(key in c['applicable_numeric_groups'] for c in s['source_cards'])]
        summary[role]=dict(pairs=len(rr),stages=len(st),source_context_stages=sum(bool(s['source_cards']) for s in st),
            source_cards=sum(len(s['source_cards']) for s in st),group_opportunities=len(gs),
            all_status=dict(Counter(g['feedback']['status'] for g in gs)),
            linked_group_opportunities=len(linked),linked_status=dict(Counter(g['feedback']['status'] for g in linked)),
            alignment_uncertain_stages=sum(bool(s['alignment']['flags']) for s in st))
    summary.update(total_seconds=time.perf_counter()-start,new_inference=False,new_training=False,
        source_clauses=len(clauses),inactive_clauses=[c['id'] for c in clauses.values() if c['runtime_scope']=='inactive_different_form'],
        automatic_correct_or_wrong_verdicts=0,accuracy_improvement_measured=False)
    assert len(results)==36 and sum(len(r['stages']) for r in results)==117
    assert all(sha(p)==h for p,h in f['protected'].items())
    write(U/'RESULT_MANIFEST.json',dict(results=manifest));write(U/'SUMMARY.json',summary)
    write(U/'REPORT_DATA.json',dict(summary=summary,cases=results,group_names=NAMES,source_clause_catalog=list(clauses.values())))
    (U/'RUNNING.json').unlink();print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
