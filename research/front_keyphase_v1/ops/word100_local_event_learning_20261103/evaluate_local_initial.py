"""One native sequence evaluation. Save scores before independent arithmetic."""
import time,os,traceback
import numpy as np
import torch
from train_local import U,O,read,write,sha,checkpins
from local_events import LocalEventNet

def evidence_support(p,observed,anchor):
    left=right=anchor
    while left>0 and observed[left-1]:left-=1
    while right+1<len(observed) and observed[right+1]:right+=1
    rows=list(range(max(left,anchor-7),min(right,anchor+7)+1))
    rgb=sorted({j for i in rows for j in p['support']['row_consumed_support'][i]})
    return dict(temporal_native_rows=rows,rgb_native_rows=rgb,
        rgb_native_timestamps_us=[int(p['ts'][i]) for i in rgb],
        structural_support_only=True,exact_onset_or_observed_event_boundary=False)

def score_bag(prob,anchors,p,b,target,supported):
    lo,hi=b['indices'];ix=np.arange(lo,hi+1);eligible=ix[anchors[ix]]
    complete=bool(anchors[ix].all());result=dict(bag_id=b['bag_id'],kind=b['kind'],label=b['label'],frames=b['frames'],
        indices=b['indices'],timestamp_us=[int(p['ts'][lo]),int(p['ts'][hi])],
        labelled_evidence_complete=b['evidence_complete'],prediction_coverage=float(anchors[ix].mean()),
        unavailable_native_frames=p['ids'][ix[~anchors[ix]]].tolist(),supported=supported,
        probability=None,anchor=None,status='UNSUPPORTED_QUERY' if not supported else 'INSUFFICIENT_OBSERVATION')
    if not supported or not len(eligible):return result
    at=int(eligible[np.argmax(prob[eligible,target])]);value=float(prob[at,target])
    result.update(probability=value,anchor=at,anchor_frame=int(p['ids'][at]),anchor_timestamp_us=int(p['ts'][at]),
        raw_positive=value>=.5,high_confirmation=value>=.8 and complete,
        status=('INSUFFICIENT_OBSERVATION' if not complete else 'MOVEMENT_EVIDENCE' if value>=.8 else 'LOW_MOVEMENT_SCORE' if value<=.2 else 'UNCONFIRMED'),
        whole_action_error=None,educational_error=None)
    result['context_support']=evidence_support(p,anchors,at)
    return result

def main():
    from data import load_inputs
    assert not (U/'EVAL_STARTED.json').exists(),'Never rerun completed/scored clips'
    d=read(U/'DATASET_FROZEN.json');ex=read(U/'EXECUTION_FROZEN.json');tr=read(U/'TRAIN_RESULT.json')
    assert read(U/'TRAIN_FINISHED.json')['status']=='COMPLETE'
    checkpins(ex['pins']);assert sha(tr['model'])==tr['model_sha256']
    inputs=load_inputs(d);torch.set_num_threads(4)
    model=LocalEventNet();load_start=time.perf_counter()
    model.load_state_dict(torch.load(tr['model'],map_location='cpu',weights_only=True));model.eval();model.requires_grad_(False)
    load_seconds=time.perf_counter()-load_start
    write(U/'EVAL_STARTED.json',dict(pid=os.getpid(),at=time.time(),clips=len(inputs),model_sha256=tr['model_sha256']))
    dest=U/'predictions';dest.mkdir(exist_ok=True);rows=[];timing=[]
    with torch.inference_mode():
        for r in d['rows']:
            cid=r['clip_id'];p=inputs[cid];x=torch.from_numpy(p['x'])[None]
            start=time.perf_counter();out=model(x,torch.tensor([len(p['ids'])]));elapsed=time.perf_counter()-start
            logits=out['logits'][0].numpy();prob=out['logits'][0].sigmoid().numpy();anchors=out['motion_anchor'][0].numpy()
            file=dest/(cid+'.npz');assert not file.exists()
            np.savez_compressed(file,logits=logits,probabilities=prob,motion_anchor=anchors,
                evidence_available=out['evidence_available'][0].numpy(),rgb_available=out['rgb_available'][0].numpy(),
                frame_id=p['ids'],timestamp_us=p['ts'])
            timing.append(dict(clip_id=cid,role=r['role'],seconds=elapsed,frames=len(p['ids'])))
            for g in (g for g in d['groups'] if g['clip_id']==cid):
                t=g['target_index'];supported=t in d['supported_targets']
                bags=[score_bag(prob,anchors,p,b,t,supported) for b in g['bags']]
                broad=dict(bag_id=g['group_id']+':broad',kind='broad_consistency',label=1,
                    frames=[int(p['ids'][0]),int(p['ids'][-1])],indices=[0,len(p['ids'])-1],evidence_complete=bool(anchors.all()))
                full=score_bag(prob,anchors,p,broad,t,supported)
                positive=[b for b in g['bags'] if b['label']==1 and b['evidence_complete']]
                full['anchor_inside_reviewed_positive_bracket']=None if full['anchor'] is None else any(b['indices'][0]<=full['anchor']<=b['indices'][1] for b in positive)
                full['positive_bracket_union']=[b['indices'] for b in positive]
                vals=[b['probability'] for b in bags if b['probability'] is not None]
                full['same_score_broad_monotonic']=full['probability'] is not None and all(full['probability']>=v for v in vals)
                rows.append(dict(group_id=g['group_id'],clip_id=cid,role=g['role'],person=g['person'],target_index=t,
                    source_sha256=r['source_sha256'],prediction=dict(path=str(file),sha256=sha(file)),
                    bags=bags,broad=full,whole_clause_correctness=None,educational_error=None))
            (U/'EVAL_PROGRESS.json').write_text(__import__('json').dumps(dict(done=len(timing),total=len(d['rows']),last=cid)),encoding='utf-8')
    checkpins(ex['pins']);assert sha(tr['model'])==tr['model_sha256']
    # Retain the frozen prior scores as their original net-direction task only.
    # No retroactive regrading against these new occurrence labels.
    baseline=O/'word100_context_episode_query_20261103/RESULT.json'
    write(U/'EVAL_RESULT.json',dict(status='NATIVE_OCCURRENCE_EVALUATION_COMPLETE',rows=rows,timing=timing,
        model_load_seconds=load_seconds,latency_scope='Cached-input local head only; excludes video decode, MediaPipe, SignRep and web latency',
        raw=.5,confirmed=.8,low=.2,whole_sequence_forwards=len(timing),reference_video_forwards=0,
        baseline=dict(path=str(baseline),sha256=sha(baseline),task='Frozen original net-direction scope diagnostic; not relabelled or rerun'),
        missing_rows_preserved=True,exact_onset_labels_available=False,backbone_recomputed=False,
        limitations=['Repeatedly exposed DEV IDs11-13, not fresh holdout','Physical interval labels from Codex review, not independent KSL expert truth','No certification of sign correctness, repetition count or omitted full action','Speed/pixel corruption and new-user latency remain unmeasured for this head'],
        whole_goal_complete=False))
    write(U/'EVAL_FINISHED.json',dict(pid=os.getpid(),at=time.time(),status='COMPLETE',result_sha256=sha(U/'EVAL_RESULT.json')))
    print('Saved native evaluation:',len(timing),'clips,',len(rows),'source-query groups',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        (U/('EVAL_FAILURE_'+str(time.time_ns())+'.txt')).write_text(traceback.format_exc(),encoding='utf-8');raise
