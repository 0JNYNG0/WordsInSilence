"""Root-authorized once-only AH readout on four cached actual reverse videos."""
import ast
import importlib.util
import json
import os
import time
import traceback
from pathlib import Path
import numpy as np
from reverse_data import AN,AH,AJ,WORDS,MODEL_SHA,read,sha,require,write_new,verify_pins,load_inputs

def load_score_bag():
    # Reuse the frozen scorer without importing its training orchestration.
    path=AH/'evaluate_local.py';tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'score_bag','evidence_support'}]
    require(len(nodes)==2,'Missing frozen AH score functions')
    namespace={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['score_bag']

def main():
    require(not (AN/'REVERSE_RUN_STARTED.json').exists(),'Do not repeat a started reverse evaluation; inspect original artifacts')
    require(not (AN/'REVERSE_RESULT.json').exists(),'Completed results are immutable')
    review=read(AN/'REVERSE_CODE_REVIEW.json')
    require(review['approved_for_four_reverse_forwards'] is True,'Independent review/root execution approval required')
    verify_pins(review['code_pins'])
    manifest=read(AN/'reverse_inputs.json');prepared=load_inputs()
    require(len(prepared)==4 and set(prepared)=={w+'_reverse_main' for w in WORDS},'Four new reverse caches only')
    require(manifest['thresholds']==dict(high=.8,low=.2,raw=.5),'Fixed thresholds')
    require(manifest['scope_family']=='legacy_R_canonical_motion_DTW','Scope family changed')
    require(sum(len(p['scopes']) for p in prepared.values())==5,'Five preselected physical routes')
    for p in prepared.values():
        json.dumps(dict(scopes=p['scopes'],source_ledger=p['source_ledger'],saved_aa_result=p['saved_aa_result']),allow_nan=False)
    training=read(AH/'TRAIN_RESULT.json');modelpath=Path(training['model'])
    require(training['model_sha256']==MODEL_SHA and sha(modelpath)==MODEL_SHA,'Exact frozen AH checkpoint')
    execution_paths=[AN/'reverse_inputs.json',AN/'REVERSE_PREPARATION.json',AN/'REVERSE_CODE_REVIEW.json',
                     AN/'reverse_data.py',AN/'prepare_reverse.py',AN/'evaluate_reverse.py',
                     AH/'local_events.py',AH/'evaluate_local.py',AH/'TRAIN_RESULT.json',modelpath]
    execution_pins={str(p.resolve()):sha(p) for p in execution_paths}
    # Verify already-completed AJ remains unchanged; never call its loader/evaluator.
    aj_result=AJ/'RESULT.json';aj_hash=sha(aj_result)
    execution_pins[str(aj_result.resolve())]=aj_hash
    write_new(AN/'REVERSE_EXECUTION_FROZEN.json',dict(pins=execution_pins,input_pins=manifest['pins'],
        full_native_forwards=4,fixed_scope_readouts=5,full_native_diagnostics=5,
        thresholds=manifest['thresholds'],scope_family=manifest['scope_family'],current_AM_fidelity_claim=False,
        occurrence_gold_count=0,training=0,RGB_extraction=0,sensor_extraction=0,DTW=0,AJ24_reruns=0,
        no_score_based_selection=True,whole_goal_complete=False))
    # Exclusive marker precedes any model construction/forward. A failed attempt is not retried automatically.
    write_new(AN/'REVERSE_RUN_STARTED.json',dict(pid=os.getpid(),at=time.time(),allowed_forwards=4,checkpoint_sha256=MODEL_SHA))
    import torch
    spec=importlib.util.spec_from_file_location('_an_frozen_local_events',AH/'local_events.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.set_num_threads(4);start=time.perf_counter();model=module.LocalEventNet()
    model.load_state_dict(torch.load(modelpath,map_location='cpu',weights_only=True));model.eval();model.requires_grad_(False)
    load_seconds=time.perf_counter()-start
    require(torch.equal(model.geometry_valid,torch.tensor(manifest['geometry_valid'],dtype=torch.float32)),'Input/model masks differ')
    scorer=load_score_bag();folder=AN/'reverse_predictions';folder.mkdir(exist_ok=False);rows=[]
    with torch.inference_mode():
        for cid,p in prepared.items():
            x=np.asarray(p['x'],dtype=np.float32);n=len(p['ids']);require(x.shape==(n,915),'Native shape changed')
            start=time.perf_counter();out=model(torch.from_numpy(x)[None],torch.tensor([n]));seconds=time.perf_counter()-start
            logits=out['logits'][0].numpy();probability=out['logits'][0].sigmoid().numpy();anchors=out['motion_anchor'][0].numpy()
            require(np.array_equal(anchors,p['motion_anchor']),'Motion anchors changed')
            require(np.array_equal(out['evidence_available'][0].numpy(),p['evidence_available']),'Evidence mask changed')
            require(np.array_equal(out['rgb_available'][0].numpy(),p['available']),'RGB mask changed')
            dest=folder/(cid+'.npz')
            np.savez_compressed(dest,logits=logits,probabilities=probability,motion_anchor=anchors,
                evidence_available=out['evidence_available'][0].numpy(),rgb_available=out['rgb_available'][0].numpy(),
                frame_id=p['ids'],timestamp_us=p['ts'])
            observations=[];full=[]
            for scope in p['scopes']:
                lo,hi=scope['indices'];target=scope['target_index'];supported=target in manifest['supported_targets']
                b=dict(bag_id=cid+':'+str(target),kind='fixed_legacy_R_prototype_scope',label=None,
                       frames=[int(p['ids'][lo]),int(p['ids'][hi])],indices=[lo,hi],evidence_complete=bool(anchors[lo:hi+1].all()))
                observation=scorer(probability,anchors,p,b,target,supported)
                observation.update(target_index=target,guide_id=scope['guide_id'],fixed_scope=scope,
                    occurrence_truth=None,whole_sign_correctness=None,educational_error=None,complete_action=None,
                    correction_enabled=False,scope_family=manifest['scope_family'])
                observations.append(observation)
                broad=dict(bag_id=cid+':'+str(target)+':full_diagnostic',kind='full_native_diagnostic_only',label=None,
                           frames=[int(p['ids'][0]),int(p['ids'][-1])],indices=[0,n-1],evidence_complete=bool(anchors.all()))
                diagnostic=scorer(probability,anchors,p,broad,target,supported)
                diagnostic.update(target_index=target,guide_id=scope['guide_id'],scope_selection_use=False,
                    occurrence_truth=None,whole_sign_correctness=None,educational_error=None,complete_action=None,correction_enabled=False)
                full.append(diagnostic)
            rows.append(dict(case_id=cid,word_id=p['word_id'],person=12,role='previously_exposed_development_controls',
                kind='reverse_main',observations=observations,full_native_diagnostics=full,
                prediction=dict(path=str(dest.resolve()),sha256=sha(dest)),head_seconds=seconds,native_frames=n,
                source_ledger=p['source_ledger'],cache_binding=p['cache_binding'],saved_aa_result=p['saved_aa_result'],
                image_unavailable_frames=p['ids'][~p['image_available']].tolist(),unobserved_frames=p['ids'][~p['evidence_available']].tolist(),
                original_axis_preserved=True,old_order_and_shape_outputs_unchanged=True,whole_action_correctness=None))
            (AN/'REVERSE_PROGRESS.json').write_text(json.dumps(dict(done=len(rows),total=4,last=cid)),encoding='utf-8')
    verify_pins(execution_pins);verify_pins(manifest['pins']);require(sha(aj_result)==aj_hash,'AJ completed output changed')
    obs=[b for row in rows for b in row['observations']]
    summary=dict(clips=4,fixed_scopes=5,registered_people=1,registered_id=12,
                 high_occurrence=sum(b.get('high_confirmation',False) for b in obs),
                 insufficient=sum(b['status']=='INSUFFICIENT_OBSERVATION' for b in obs),
                 full_native_high_diagnostic=sum(b.get('high_confirmation',False) for row in rows for b in row['full_native_diagnostics']),
                 known_occurrence_gold=0,occurrence_accuracy=None,whole_sign_error_accuracy=None)
    write_new(AN/'REVERSE_RESULT.json',dict(status='ONE_FIXED_FOUR_REVERSE_INFERENCE_COMPLETE',rows=rows,summary=summary,
        model_sha256=MODEL_SHA,model_forwards=4,model_load_seconds=load_seconds,training=0,new_extraction=0,new_DTW=0,
        AJ24_reruns=0,original_results_changed=False,scope_family=manifest['scope_family'],current_AM_fidelity_claim=False,
        latency_scope='Only small AH head on existing actual reverse RGB/joint caches; not camera/video/full feedback time',
        interpretation='Direction occurrence and existing AA partial order are separate observations. Reversal alone is not educational error gold; high local evidence does not establish action completion.',
        whole_goal_complete=False))
    write_new(AN/'REVERSE_RUN_FINISHED.json',dict(status='COMPLETE',pid=os.getpid(),at=time.time(),result_sha256=sha(AN/'REVERSE_RESULT.json')))
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    try:main()
    except BaseException:
        path=AN/'REVERSE_FIRST_FAILURE.txt'
        if not path.exists():path.write_text(traceback.format_exc(),encoding='utf-8')
        raise
