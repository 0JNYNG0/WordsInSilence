"""Root independent saved-output arithmetic; no adapter/model imports."""
import hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    checks=[]
    def check(name,condition):
        assert condition,name
        checks.append(name)
    frozen=read(U/'FROZEN.json');result=read(U/'REPLAY_RESULT.json')
    check('terminal result sha',read(U/'REPLAY_FINISHED.json')['result_sha256']==sha(U/'REPLAY_RESULT.json'))
    for p,h in frozen['pins'].items():check('pin '+p,sha(p)==h)
    outputs={r['clip_id']:r for r in result['outputs']};seen=[]
    for row in frozen['rows']:
        e=row['identity'];cid=e['clip_id'];saved=outputs[cid]
        check(cid+' output sha',sha(saved['path'])==saved['sha256'])
        base=read(e['existing_feedback']);out=read(saved['path']);addition=out.pop('local_event_feedback')
        check(cid+' exact original preservation',out==base)
        check(cid+' immutable proposal count',len(addition['observations'])==len(row['proposals']))
        check(cid+' no new inference',all(addition[k]==0 for k in ['new_model_loads','new_model_forwards','new_DTW','new_training']))
        check(cid+' never semantic correctness',all(addition[k] is None for k in ['whole_clause_truth','whole_sign_correctness','complete_action','educational_error']))
        with np.load(e['AH_prediction'],allow_pickle=False) as a:
            ids=a['frame_id'];ts=a['timestamp_us'];p=a['probabilities'];anchors=a['motion_anchor'];observed=a['evidence_available']
        check(cid+' exact native axis',ids.tolist()==e['native_frame_ids'] and ts.tolist()==e['timestamp_us'])
        for proposal,ob in zip(row['proposals'],addition['observations']):
            t=proposal['target_index'];lo,hi=proposal['indices'];ix=np.arange(lo,hi+1);valid=ix[anchors[ix]]
            key=cid+':'+str(t)
            check(key+' unchanged proposal',ob['proposal']==proposal)
            check(key+' native denominator',ob['native_scope_denominator']==len(ix) and ob['anchor_count']==len(valid))
            complete=bool(anchors[ix].all() and observed[ix].all() and len(ix)>=2)
            check(key+' completeness',ob['scope_evidence_complete']==complete)
            check(key+' gap preserved',ob['unavailable_native_frames']==ids[ix[~observed[ix]]].tolist())
            high=valid[p[valid,t]>=.8]
            check(key+' high native frames',ob['high_anchor_native_frames']==ids[high].tolist() and ob['high_anchor_timestamp_us']==ts[high].tolist())
            value=float(p[valid,t].max()) if len(valid) else None
            check(key+' exact cached maximum',ob['probability']==value)
            at=int(valid[np.argmax(p[valid,t])]) if len(valid) else None
            check(key+' candidate native anchor',ob['maximum_anchor_native_frame']==(int(ids[at]) if at is not None else None))
            expected='LOCAL_MOVEMENT_EVIDENCE_OBSERVED' if len(high) else 'INSUFFICIENT_OBSERVATION' if not complete else 'LOCAL_MOVEMENT_NOT_CONFIRMED' if value is not None and value<=.2 else 'LOCAL_MOVEMENT_UNCERTAIN'
            check(key+' state',ob['status']==expected)
            check(key+' semantic nulls',all(ob[k] is None for k in ['whole_clause_truth','whole_sign_correctness','complete_action','educational_error','required_repetitions','source_variant_applicability_verified','source_local_applicability_verified']))
            stage=next(s for s in base['stages'] if s['guide_id']==proposal['guide_id'])
            guard=stage['neural_stage_agreement']>=.5 and stage['image_time_coverage']>=.8 and stage['unique_query_frames']>=2
            check(key+' inherited guard',ob['guide_time_binding']['status']==('SUPPORTED' if guard else 'UNCONFIRMED'))
            check(key+' guard scope unchanged',ob['guide_time_binding']['guard_scope_native_frames']==stage['frame_ids'])
            check(key+' source semantic limit',ob['source_constraint']==proposal['source_constraint'] and (t==0 or ob['source_binding']=='REFERENCE_OBSERVATION_ONLY'))
            for point,i in zip(ob['native_point_estimates'],ix):
                check(key+f' point {i}',point['frame_id']==int(ids[i]) and point['timestamp_us']==int(ts[i]) and point['motion_anchor']==bool(anchors[i]))
                check(key+f' probability mask {i}',point['probability']==(float(p[i,t]) if anchors[i] else None))
            seen.append(expected)
    check('counts',dict(Counter(seen))==result['counts'] and len(seen)==15 and len(outputs)==12)
    check('no execution expansion',all(result[k]==0 for k in ['source_videos_read','models_loaded','new_model_forwards','new_DTW','new_training']))
    report=dict(status='PASS_INDEPENDENT_SAVED_ADAPTER_VERIFICATION',checks_count=len(checks),failed_checks=[],
        primary_scopes=15,clips=12,counts=dict(Counter(seen)),all_original_outputs_unchanged=True,
        implementation_by='Codex egypt agent',verified_by='Codex root; separate arithmetic, not independent human KSL truth',
        new_model_forwards=0,new_training=0,new_DTW=0,source_video_reads=0,whole_goal_complete=False,
        pins={str(U/n):sha(U/n) for n in ['FROZEN.json','CODE_REVIEW.json','REPLAY_RESULT.json','verify_saved.py']})
    with (U/'SAVED_VERIFIED.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(report['status'],len(checks),dict(Counter(seen)))
if __name__=='__main__':main()
