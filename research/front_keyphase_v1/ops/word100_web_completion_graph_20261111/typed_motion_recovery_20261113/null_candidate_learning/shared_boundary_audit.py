"""Separate timestamp-extent order audit. Never alter the frozen comparison."""
from pathlib import Path
from collections import Counter
import json,hashlib,datetime,itertools
import numpy as np
N=Path(__file__).resolve().parent
def rd(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(r):return r['seed'],r['clip_id'],r['kind']

def old_relation(a,b):
    if a is None or b is None:return 'unobserved'
    if max(a[0],b[0])<=min(a[1],b[1]):return 'ambiguous_overlap'
    return 'forward' if a[1]<b[0] else 'reverse'

def temporal_relation(a,b):
    if a is None or b is None:return 'unobserved','unobserved',None
    assert a[0]<a[1] and b[0]<b[1]
    overlap=max(0,min(a[1],b[1])-max(a[0],b[0]))
    if a[1]<=b[0]:return 'forward','forward_shared_boundary' if a[1]==b[0] else 'forward_separated',overlap
    if b[1]<=a[0]:return 'reverse','reverse_shared_boundary' if b[1]==a[0] else 'reverse_separated',overlap
    assert overlap>0
    return 'ambiguous_overlap','positive_duration_overlap',overlap

def main():
    output=N/'SHARED_BOUNDARY_AUDIT.json';assert not output.exists()
    source=N/'evaluation/COMPARISON.json';source_hash=sha(source);data=rd(source)
    pp={key(r):r for r in data['predictions']};timestamps={};pins={str(source):source_hash};records=[];changes=[]
    assert len(pp)==len(data['predictions'])==len(data['rows'])==246
    conditions=['baseline','local_full','local_null']
    for row in data['rows']:
        k=key(row);pr=pp[k];path=pr['source']
        if path not in timestamps:
            assert sha(path)==pr['source_sha256'];pins[path]=pr['source_sha256']
            ts=np.load(path)['ts'];assert ts.ndim==1 and np.isfinite(ts).all() and (np.diff(ts)>0).all()
            assert np.array_equal(ts,ts.astype(np.int64)),'Expected integer microsecond timestamps'
            timestamps[path]=ts.astype(np.int64)
        ts=timestamps[path]
        assert set(row['conditions'])==set(conditions)
        for condition in conditions:
            result=row['conditions'][condition];cards=result['cards'];orders=[]
            for i,j in itertools.combinations(range(len(cards)),2):
                if sorted([cards[i]['family'],cards[j]['family']])!=[2,3]:continue
                ci,cj=cards[i],cards[j];si=ci['chosen']['span'] if ci['chosen'] else None;sj=cj['chosen']['span'] if cj['chosen'] else None
                for span in [si,sj]:
                    if span is not None:assert 0<=span[0]<span[1]<len(ts)
                a=[int(ts[q]) for q in si] if si is not None else None
                b=[int(ts[q]) for q in sj] if sj is not None else None
                old=old_relation(si,sj);new,detail,overlap=temporal_relation(a,b)
                orders.append(dict(cards=[ci['card'],cj['card']],state=old))
                r=dict(seed=k[0],clip_id=k[1],kind=k[2],condition=condition,cards=[ci['card'],cj['card']],
                       spans=[si,sj],time_extents_us=[a,b],original_state=old,timestamp_extent_state=new,relation_detail=detail,overlap_duration_us=overlap,
                       source_path=path,source_sha256=pr['source_sha256'])
                records.append(r)
                if old!=new:
                    assert old=='ambiguous_overlap' and new in ['forward','reverse'] and overlap==0
                    assert detail.endswith('shared_boundary')
                    changes.append(r)
            assert orders==result['order'],'Original frozen interval arithmetic must be reproduced exactly'
    summaries=[]
    for s in data['summary']:
        rr=[r for r in records if r['seed']==s['seed'] and r['condition']==s['condition'] and r['kind']==s['kind']]
        before=dict(Counter(r['original_state'] for r in rr));after=dict(Counter(r['timestamp_extent_state'] for r in rr))
        assert before==s['order']
        summaries.append(dict(seed=s['seed'],condition=s['condition'],kind=s['kind'],order_pairs=len(rr),original=before,timestamp_extent=after,changed=sum(r['original_state']!=r['timestamp_extent_state'] for r in rr)))
    for p,h in pins.items():assert sha(p)==h
    report=dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),status='PASS_ARITHMETIC_ONLY',
        definition='Each span is an extent from its first sampled timestamp to its last sampled timestamp; not per-frame exposure bins [PTS,nextPTS). Overlap requires positive duration, max(start)<min(end). Shared boundary has zero overlap duration.',
        scope='Same rule applied to all baseline/local_full/local_null conditions, all seeds and all stored scenarios. No selection/model/threshold/annotation modification.',
        limits='An order arithmetic correction is not evidence that the selected segment is the correct action or semantically complete. Annotation boundary uncertainty stays unchanged. No new-user accuracy claim.',
        input=str(source),input_sha256=source_hash,input_preserved=True,rows=len(data['rows']),conditions=conditions,source_scenarios=len(timestamps),
        order_relations=len(records),changed_relations=len(changes),changed_by_condition=dict(Counter(r['condition'] for r in changes)),
        changed_by_kind=dict(Counter(r['kind'] for r in changes)),changes=changes,summary=summaries,relations=records,
        no_inference=True,no_training=True,web_changed=False,private_access=False,final8=False,
        reviewer='Delegated Codex motion_contract_review_1113 saved timestamp arithmetic, not independent human labels',script_sha256=sha(__file__),pins=pins)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
    print(json.dumps({k:report[k] for k in ['status','rows','order_relations','changed_relations','changed_by_condition','changed_by_kind','input_preserved']},ensure_ascii=False),flush=True)
    for r in changes:print(json.dumps({k:r[k] for k in ['seed','clip_id','kind','condition','spans','original_state','timestamp_extent_state','relation_detail']},ensure_ascii=False))

if __name__=='__main__':main()
