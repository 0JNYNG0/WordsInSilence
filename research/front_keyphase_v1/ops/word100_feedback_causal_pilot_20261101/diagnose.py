"""One new diagnostic from frozen saved values. No inference/training/path search."""
from pathlib import Path
from collections import Counter
import hashlib, json, time
import numpy as np
import action_evidence as ae

U=Path(__file__).resolve().parent; O=U.parent
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):
    with (U/n).open('x',encoding='utf-8') as f: json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    assert not (U/'POINT_DIAGNOSIS.json').exists(), 'Completed diagnostic is not rerun.'
    start=time.perf_counter(); frozen=read(U/'FROZEN_POINT_DIAGNOSIS.json')
    for p,h in frozen['source_pins'].items(): assert sha(p)==h,p
    case=read(O/'word100_action_feedback_casebook_20261101/INPUTS.json')
    data=read(O/'word100_semantic_feature_cycle_20261101/DATASET.json')
    old=read(O/'word100_stage_feedback_cycle_20261031/LEARNING_INPUTS.json')
    oldids={r['clip_id']:i for i,r in enumerate(old['rows'])}
    centers={(r['clip_id'],r['center_frame']):i for i,r in enumerate(data['rows'])}
    names=data['targets']; cols={n:i for i,n in enumerate(names)}
    arrays={}; input_pins={}; center_checks=0; records=[]; cached_nodes={}
    aliases=sorted({x[k] for x in frozen['opportunities'] for k in ['query','reference']})
    for a in aliases:
        row=case['rows'][a]; assert sha(row['input'])==row['input_sha256']
        with np.load(row['input'],allow_pickle=False) as z:
            arrays[a]={k:z[k].copy() for k in ['x','mask','frame_id','timestamp_us']}
        input_pins[row['input']]=row['input_sha256']
    for seed in frozen['seeds']:
        base=O/f'word100_stage_feedback_cycle_20261031/runs/pose_plus_signrep_{seed}'
        new=O/f'word100_semantic_feature_cycle_20261101/runs/protected_shape_split_features_{seed}'
        assert sha(base/'probabilities.npy')==read(base/'RUN.json')['prediction_sha256']
        assert sha(new/'probabilities.npy')==read(new/'COMPLETE.json')['probabilities_sha256']
        native=np.load(base/'probabilities.npy',allow_pickle=False)
        split=np.load(new/'probabilities.npy',allow_pickle=False)
        def node(a,f):
            nonlocal center_checks
            key=f'{seed}:{a}:{f}'
            if key in cached_nodes:return cached_nodes[key]
            row=case['rows'][a]; z=arrays[a]; ii=np.flatnonzero(z['frame_id']==f)
            assert len(ii)==1; i=int(ii[0]); assert i==f
            oi=oldids[row['clip_id']]; assert old['rows'][oi]['input_sha256']==row['input_sha256']
            ci=centers.get((row['clip_id'],f)); p={}; valid={}; sources={}
            for t in names:
                j=cols[t]; side=t[0]; off=28 if side=='R' else 0
                if j<3:
                    v=native[oi,i,old['targets'].index(t)]
                    if ci is not None:
                        assert v==split[ci,j]; center_checks+=1
                    sources[t]='protected_original_native'
                elif ci is not None:
                    v=split[ci,j]; sources[t]='protected_split_exact_center'
                else:
                    v=None; sources[t]='no_center_prediction'
                p[t]=None if v is None else float(v)
                if t.endswith('higher'):
                    support=bool(z['mask'][i,[8,9,36,37]].all())
                else:
                    support=bool(z['mask'][i,off:off+2].all() and
                                 z['mask'][i,off+11:off+25].sum()>=10)
                valid[t]=support and v is not None and bool(np.isfinite(v))
            if ci is not None:
                cr=data['rows'][ci]
                assert int(z['timestamp_us'][i])==cr['center_timestamp_us']
            values={t:int(v>=.5) if valid[t] else None for t,v in p.items()}
            palm_valid=bool(z['mask'][i,[8,9,36,37]].all())
            delta=float(z['x'][i,9]-z['x'][i,37]) if palm_valid else None
            n=dict(alias=a,clip_id=row['clip_id'],frame_id=f,timestamp_us=int(z['timestamp_us'][i]),
                   seed=seed,probabilities=p,support=valid,values=values,prediction_source=sources,
                   contradictions=ae.conflicts(values),center_dataset_row=ci,
                   palm_left_y_minus_right_y=delta,
                   palm_delta_definition='positive=right palm image-above left; not clear separation or equal-height judgement')
            cached_nodes[key]=n; return n
        for k,x in enumerate(frozen['opportunities']):
            r=node(x['reference'],x['reference_frame']); t=x['target']
            qs=[node(x['query'],f) for f in x['current_query_indices']]
            comparisons=[ae.compare_attribute(q['values'],r['values'],t) for q in qs]
            usable=all(c['raw_attribute_difference'] is not None for c in comparisons)
            mismatch=any(c['raw_attribute_difference'] is True for c in comparisons)
            truth_diff=any(v!=x['reference_label'] for v in x['current_query_labels'])
            raw_mismatch=any((q['probabilities'][t]>=.5)!=(r['probabilities'][t]>=.5)
                             for q in qs if q['probabilities'][t] is not None and r['probabilities'][t] is not None)
            anchors=[node(x['query'],v['query_frame']) for v in x['same_family_known_query_anchors']]
            records.append(dict(opportunity=k,seed=seed,source=x,reference_node=f'{seed}:{r["alias"]}:{r["frame_id"]}',
                query_nodes=[f'{seed}:{q["alias"]}:{q["frame_id"]}' for q in qs],comparisons=comparisons,
                truth_attribute_difference=truth_diff,raw_ungated_prediction_difference=raw_mismatch,
                usable=usable,observed_attribute_difference=mismatch if usable else None,
                false_attribute_difference=bool(usable and mismatch and not truth_diff),
                known_feature_diagnostic_only=dict(values=x['current_query_labels'],reference=x['reference_label'],
                   feature_difference=truth_diff,automatic_performance=False),
                candidate_anchors=[dict(frame_id=q['frame_id'],comparison=ae.compare_attribute(q['values'],r['values'],t)) for q in anchors],
                stage_oracle_available=False,whole_handshape_equal=None,educational_correctness=None))
    summary=[]
    for seed in frozen['seeds']:
        rr=[x for x in records if x['seed']==seed]
        cc=Counter(c['status'] for x in rr for c in x['comparisons'])
        summary.append(dict(seed=seed,opportunities=len(rr),usable=sum(x['usable'] for x in rr),
                unknown=sum(not x['usable'] for x in rr),false_attribute_difference=sum(x['false_attribute_difference'] for x in rr),
                statuses=dict(cc),conflicting_opportunities=sum(any(c['inconsistent_relevant_group'] for c in x['comparisons']) for x in rr),
                same_negative_is_not_full_handshape=sum(c['status']=='SAME_NEGATIVE_ATTRIBUTE_ONLY' for x in rr for c in x['comparisons'])))
    write('POINT_DIAGNOSIS.json',dict(status='COMPLETE_SAVED_POINT_DIAGNOSIS_ONLY',summary=summary,records=records,nodes=cached_nodes,
        protected_center_probability_exact_checks=center_checks,source_pins={**frozen['source_pins'],**input_pins},
        code_pins={str(p):sha(p) for p in [U/'diagnose.py',U/'action_evidence.py']},elapsed_seconds=time.perf_counter()-start,
        new_model_inferences=0,new_training=0,new_dtw_paths=0,not_runtime_accuracy='labels identify attribute differences only; all common truth values agree; repeated DEV'))
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
