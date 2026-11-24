"""One fixed new-head forward for each already-extracted public pixel edit."""
from pathlib import Path
import sys,json,time,hashlib,os,traceback
import numpy as np
import torch
U=Path(__file__).resolve().parent;O=U.parent;AH=O/'word100_local_event_learning_20261103'
sys.path.insert(0,str(AH))
from local_events import LocalEventNet
from evaluate_local import score_bag

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def pins(items):
    for p,h in items.items():assert sha(p)==h,p

def main():
    import aj_data
    assert not (U/'RUN_STARTED.json').exists(),'Do not repeat a scored control'
    manifest=read(U/'INPUT_FROZEN.json');review=read(U/'CODE_REVIEW.json')
    assert review['approved_for_one_inference'] is True
    pins(review['code_pins'])
    prepared=aj_data.load_inputs()
    assert len(prepared)==24
    assert set(prepared)=={r['case_id'] for r in manifest['rows']}
    assert sum(len(p['scopes']) for p in prepared.values())==30
    for cid,p in prepared.items():
        assert cid==p['word_id']+'_'+p['kind']
        json.dumps(dict(scopes=p['scopes'],source_ledger=p['source_ledger'],cache_binding=p['cache_binding']),allow_nan=False)
    training=read(AH/'TRAIN_RESULT.json');modelpath=training['model']
    assert sha(modelpath)==training['model_sha256']
    execution_pins={str(p):sha(p) for p in [U/'INPUT_FROZEN.json',U/'CODE_REVIEW.json',U/'aj_data.py',U/'evaluate_controls.py',AH/'local_events.py',AH/'evaluate_local.py',Path(modelpath),AH/'TRAIN_RESULT.json']}
    write(U/'EXECUTION_FROZEN.json',dict(pins=execution_pins,inputs=24,forwards=24,thresholds=dict(high=.8,low=.2,raw=.5),
        training=0,normal_preservation_kinds=['reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest'],
        interpretation='Normal variants are preservation expectations, not independently reviewed exact scope truth. Cut/blackout never automatically occurrence0. No full-sign judgement.',
        no_score_based_selection=True,original_native_predictions_not_reused=True))
    torch.set_num_threads(4);model=LocalEventNet()
    model.load_state_dict(torch.load(modelpath,map_location='cpu',weights_only=True));model.eval();model.requires_grad_(False)
    assert torch.equal(model.geometry_valid,torch.tensor(manifest['geometry_valid'],dtype=torch.float32))
    write(U/'RUN_STARTED.json',dict(pid=os.getpid(),at=time.time(),new_model='AH local42',new_input_cases=24))
    folder=U/'predictions';folder.mkdir(exist_ok=False);rows=[]
    with torch.inference_mode():
        for cid,p in prepared.items():
            x=np.asarray(p['x'],dtype='float32');n=len(p['ids']);assert x.shape==(n,915)
            start=time.perf_counter();out=model(torch.from_numpy(x)[None],torch.tensor([n]));seconds=time.perf_counter()-start
            probability=out['logits'][0].sigmoid().numpy();anchors=out['motion_anchor'][0].numpy()
            assert np.array_equal(anchors,p['motion_anchor'])
            dest=folder/(cid+'.npz')
            np.savez_compressed(dest,logits=out['logits'][0].numpy(),probabilities=probability,motion_anchor=anchors,
                evidence_available=out['evidence_available'][0].numpy(),rgb_available=out['rgb_available'][0].numpy(),
                frame_id=p['ids'],timestamp_us=p['ts'])
            observations=[];full_diagnostics=[]
            for scope in p['scopes']:
                lo,hi=scope['indices'];t=scope['target_index']
                b=dict(bag_id=cid+':'+str(t),kind='fixed_existing_prototype_scope',label=None,
                    frames=[int(p['ids'][lo]),int(p['ids'][hi])],indices=[lo,hi],evidence_complete=bool(anchors[lo:hi+1].all()))
                result=score_bag(probability,anchors,p,b,t,True)
                result.update(target_index=t,guide_id=scope['guide_id'],fixed_scope=scope,
                    occurrence_truth=None,whole_sign_correctness=None,complete_action=None,educational_error=None,
                    correction_enabled=False)
                observations.append(result)
                whole=dict(bag_id=cid+':'+str(t)+':whole_context',kind='full_native_diagnostic_only',label=None,
                    frames=[int(p['ids'][0]),int(p['ids'][-1])],indices=[0,n-1],evidence_complete=bool(anchors.all()))
                diagnostic=score_bag(probability,anchors,p,whole,t,True)
                diagnostic.update(target_index=t,guide_id=scope['guide_id'],not_used_to_select_or_expand_primary_scope=True,
                    occurrence_truth=None,whole_sign_correctness=None,complete_action=None,educational_error=None,
                    correction_enabled=False)
                full_diagnostics.append(diagnostic)
            rows.append(dict(case_id=cid,word_id=cid.split('_',1)[0],kind=cid.split('_',1)[1],
                observations=observations,full_native_diagnostics=full_diagnostics,head_seconds=seconds,native_frames=n,prediction=dict(path=str(dest),sha256=sha(dest)),
                source_ledger=p['source_ledger'],cache_binding=p['cache_binding'],
                image_unavailable_frames=p['ids'][~np.asarray(p['image_available'])].tolist(),
                unobserved_frames=p['ids'][~out['evidence_available'][0].numpy()].tolist(),
                original_axis_preserved=True,whole_action_correctness=None))
            (U/'PROGRESS.json').write_text(json.dumps(dict(done=len(rows),total=24,last=cid)),encoding='utf-8')
    pins(execution_pins);pins(manifest['pins'])
    normal={'reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest'}
    summaries=[]
    for kind in sorted({r['kind'] for r in rows}):
        selected=[r for r in rows if r['kind']==kind];obs=[b for r in selected for b in r['observations']]
        summaries.append(dict(kind=kind,clips=len(selected),scopes=len(obs),
            high_occurrence=sum(b.get('high_confirmation',False) for b in obs),
            full_native_high_diagnostic=sum(b.get('high_confirmation',False) for r in selected for b in r['full_native_diagnostics']),
            insufficient=sum(b['status']=='INSUFFICIENT_OBSERVATION' for b in obs),
            scopes_with_gaps=sum(b['prediction_coverage']<1 for b in obs),
            normal_preservation_expectation=kind in normal,
            exact_scope_accuracy=None,whole_sign_error_truth=None))
    write(U/'RESULT.json',dict(status='ONE_FIXED_24_PIXEL_CONTROL_INFERENCE_COMPLETE',rows=rows,summary=summaries,
        model_sha256=training['model_sha256'],model_forwards=24,training=0,new_extraction=0,new_DTW=0,
        native_AH_prediction_scores_copied=False,original_results_changed=False,
        latency_scope='New tiny head on existing cached RGB/joint input only, not real-user or full pipeline latency',
        interpretation='Partial action deletion may retain a directional event. A high event is not proof of complete action, and missing observation is never normality.',
        whole_goal_complete=False))
    write(U/'RUN_FINISHED.json',dict(status='COMPLETE',pid=os.getpid(),at=time.time(),result_sha256=sha(U/'RESULT.json')))
    print(json.dumps(summaries,ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        with (U/('FIRST_OR_LATER_FAILURE_'+str(time.time_ns())+'.txt')).open('x',encoding='utf-8') as f:f.write(traceback.format_exc())
        raise
