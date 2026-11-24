"""Feasibility only: prefix input vs full-clip observations; no tuning/training.

Previously inspected PUBLIC recordings; not a fresh held-out evaluation.
Every prefix is normalized again from its own raw points (no future-scale leak).
Agreement with a full clip is NOT accuracy or whole-sign correctness.
"""
from pathlib import Path
import sys, json, time, hashlib
import numpy as np
import torch

V=Path(__file__).resolve().parent
O=V.parent
U=O/'basic_demo_feedback_20261115'
D=U/'acquisition_motion_followup'
sys.path.insert(0,str(D))
from check_motion_bridge import Features, RelationObserver, adapter, bridge, Observer
from shape_consensus import ShapeConsensus

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def channels(raw,end):
    return adapter.convert(raw['xy_px'][:end],raw['score'][:end],
        raw['timestamp_us'][:end],raw['frame_id'][:end],
        int(raw['source_width']),int(raw['source_height']))

def main():
    assert not (V/'PREFIX_RESULTS.json').exists()
    torch.set_num_threads(2)
    features=Features();relation=RelationObserver();motion=Observer('mlp32');shape=ShapeConsensus()
    source=read(D/'confirmation/EXTRACTED.json')['rows']
    # Chosen from previous SOURCE_TRUTH anchors, before examining prefix outputs.
    anchors={'NIA_SL_WORD1290_REAL18_F':[57,60,66],
             'NIA_SL_WORD0099_REAL14_F':[33,41,57],
             'NIA_SL_WORD0099_REAL15_F':[48,60,72]}
    contract=dict(purpose=__doc__,anchors=anchors,lookahead_frames=[0,3,9],
        prediction_scope='Existing relation, movement, and right-hand four-long-finger observers only.',
        timing_scope='Warm CPU model/adapter calls on cached RTMW; excludes extraction/camera/UI; not a streaming load test.',
        training=False,web_changed=False,private_data=False,
        script_sha256=sha(Path(__file__)),sources=[dict(clip=r['clip_id'],raw_sha256=sha(Path(r['raw_path'])))for r in source])
    (V/'CONTRACT.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[];bases=[]
    for r in source:
        with np.load(r['raw_path'])as z:raw={k:z[k].copy()for k in z.files}
        n=len(raw['frame_id']);ch=channels(raw,n)
        f,q,_=features(ch);full,_=relation(f,q,ch['timestamp_us'])
        fingers=shape.sequence(ch['tcn280'],'R')['fingers']
        full_events=bridge(full,motion.sequence(ch)) if r['word']=='WORD1290' else None
        bases.append(dict(clip=r['clip_id'],events=full_events))
        for i in anchors[r['clip_id']]:
            for lag in [0,3,9]:
                end=min(n,i+lag+1);t=time.perf_counter();z=channels(raw,end)
                pf,pq,_=features(z);pstate,_=relation(pf,pq,z['timestamp_us'])
                pshape=shape.sequence(z['tcn280'],'R')['fingers']
                events=bridge(pstate,motion.sequence(z)) if r['word']=='WORD1290' else None
                rows.append(dict(clip=r['clip_id'],frame=i,lookahead=lag,available_frames=end,
                    full_relation=int(full[i]),prefix_relation=int(pstate[i]),
                    full_fingers=fingers[i].tolist(),prefix_fingers=pshape[i].tolist(),
                    relation_revisions=int(np.sum(pstate!=full[:end])),
                    compared_relation_frames=end,events=events,
                    elapsed_seconds=time.perf_counter()-t))
    summary=[]
    for lag in [0,3,9]:
        rs=[r for r in rows if r['lookahead']==lag]
        summary.append(dict(lookahead_frames=lag,anchor_comparisons=len(rs),
            relation_anchor_changed=sum(r['full_relation']!=r['prefix_relation']for r in rs),
            finger_anchor_changes=sum(sum(a!=b for a,b in zip(r['full_fingers'],r['prefix_fingers']))for r in rs),
            finger_anchor_comparisons=4*len(rs)))
    result=dict(contract=contract,summary=summary,full_results=bases,rows=rows,
        accuracy=None,live_ready=False,live_or_error_input_tested=False)
    (V/'PREFIX_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(summary=summary,gratitude=[dict(frame=r['frame'],lookahead=r['lookahead'],events=r['events'])for r in rows if '1290'in r['clip']]),ensure_ascii=False))

if __name__=='__main__':main()
