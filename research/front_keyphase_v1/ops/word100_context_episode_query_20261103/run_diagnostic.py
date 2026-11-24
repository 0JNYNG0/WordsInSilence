"""One six-scope FIT inference; immutable outputs and explicit cached lineage."""
import os,sys,time
from pathlib import Path
import numpy as np
import torch
from prepare import U,O,F,R,read,sha,exclusive
from episode_query import EpisodeQuery,prepare_input,select_episode
from motion_model import public_arrays

def verified_recipe(row,a):
    if 'window_starts' in a:
        return None,{}
    bank=O/'word100_multipeople_live_feedback_20261101'
    manifest=bank/'PUBLIC_INPUTS.json'
    old=next(r for r in read(manifest)['rows'] if r['clip_id']==row['clip_id'])
    assert old['input_sha256']==row['input_sha256']==sha(old['input'])
    assert old['source_sha256']==row['source_sha256'] and old['role']=='fit'
    assert old['frames']==len(a['frame_id']) and not old['reused_input']
    files=[bank/'prepare_public.py',O/'word100_signrep_temporal_head_20261029/prepare_inputs.py',
           O/'word100_signrep_feasibility_20261029/common.py']
    pins={str(p):sha(p) for p in files}
    return dict(route='public_inputs_16_stride2_center_interp',frames=old['frames'],
                manifest_sha256=sha(manifest),code_pins=pins),dict(pins,**{str(manifest):sha(manifest)})

def main():
    assert not (U/'RUN_STARTED.json').exists(), 'Inspect terminal state, never blindly repeat inference'
    contract=read(U/'FROZEN.json')
    review=read(U/'LATE_VISUAL_REVIEW.json')
    code_review=read(U/'REVIEW_CODE.json')
    assert code_review['approved_for_six_scope_diagnostic'] is True
    for path,h in code_review['code_pins'].items(): assert sha(path)==h
    assert len(review['rows'])==3 and len(contract['rows'])==3
    assert {r['person'] for r in contract['rows']}=={1,4,7}
    assert {r['clip_id'] for r in review['rows']}=={r['clip_id'] for r in contract['rows']}
    for path,digest in contract['pins'].items():
        assert sha(path)==digest
    stats=read(O/'word100_learned_motion_order_20261102/NORMALIZATION.json')
    rgb_stats=read(R/'RGB_NORMALIZATION.json')
    old=read(O/'word100_component_rgb_context_20261102/SCOPE_RESULT.json')['rows']
    old_scope={(r['record']['clip_id'],tuple(r['extent'])) for r in old}
    trusted_path=O/'word100_component_rgb_context_20261102/SCOPE_FROZEN.json'
    trusted=read(trusted_path)
    pins=dict(contract['pins'])
    pins[str(trusted_path)]=sha(trusted_path)
    items=[];descriptors=[]
    for r in contract['rows']:
        for k in ['source','input']:
            assert sha(r[k])==r[k+'_sha256'];pins[r[k]]=r[k+'_sha256']
        linked=[v for v in trusted['rows'] if v['record']['clip_id']==r['clip_id']]
        assert len(linked)==1 and linked[0]['record']['person']==r['person'] and linked[0]['record']['role']=='fit'
        assert trusted['pins'][r['input']]==r['input_sha256']
        native=np.load(r['input'])
        v=next(v for v in review['rows'] if v['clip_id']==r['clip_id'])
        assert v['source']==r['source'] and v['source_sha256']==r['source_sha256']
        assert v['fixed_native_frame_interval']==r['late_frames']
        ii=list(range(r['late_frames'][0],r['late_frames'][1]+1))
        assert v['actually_viewed_native_frames']==ii
        assert v['native_timestamps_us']==native['timestamp_us'][ii].tolist()
        assert sha(v['reviewed_sheet']['path'])==v['reviewed_sheet']['sha256']
        for p in [O/'word100_multipeople_live_feedback_20261101/bank_predictions'/f"{r['clip_id']}.npz",
                  O/'word100_learned_body_relation_20261102/runs/rgb_geometry_42/predictions'/f"{r['clip_id']}.npz"]:
            assert sha(p)==trusted['pins'][str(p)]
            observed=np.load(p)
            assert np.array_equal(observed['frame_id'],native['frame_id'])
            assert np.array_equal(observed['timestamp_us'],native['timestamp_us'])
            pins[str(p)]=trusted['pins'][str(p)]
        a,z,mask=public_arrays(r)
        recipe,recipe_pins=verified_recipe(r,a);pins.update(recipe_pins)
        prepared=prepare_input(a,z,mask,stats,rgb_stats,recipe)
        for kind in ['broad','late']:
            bounds=r[kind+'_frames'];assert (r['clip_id'],tuple(bounds)) not in old_scope
            episode=select_episode(prepared,bounds,0)
            items.append((prepared,episode))
            descriptors.append(dict(clip_id=r['clip_id'],person=r['person'],role='fit',
                                    scope=kind,frames=bounds,input=r['input'],input_sha256=r['input_sha256']))
    assert len(items)==len(descriptors)==6
    for p in [U/'FROZEN.json',U/'LATE_VISUAL_REVIEW.json',U/'REVIEW_CODE.json',U/'episode_query.py',U/'run_diagnostic.py',
              R/'rgb_model.py',O/'word100_learned_motion_order_20261102/motion_model.py',
              O/'word100_learned_motion_path_20261102/path_model.py',
              O/'word100_process_contract_alignment_20261102/process_rgb_support.py']:
        pins[str(p)]=sha(p)
    exclusive(U/'EXECUTION_FROZEN.json',dict(pins=pins,scopes=descriptors,thresholds='inherited .8/.2, geometry .5, min3frames',
        no_window_search=True,no_training=True,old_completed_exact_scopes_reused_or_rerun=0))
    torch.set_num_threads(4)
    exclusive(U/'RUN_STARTED.json',dict(pid=os.getpid(),at=time.time(),scopes=6))
    start=time.perf_counter();model=EpisodeQuery();load_seconds=time.perf_counter()-start
    values,seconds=model.query(items)
    assert len(values)==len(items)==6
    rows=[dict(**d,observation=v) for d,v in zip(descriptors,values)]
    for p,h in pins.items(): assert sha(p)==h
    exclusive(U/'RESULT.json',dict(status='SIX_NEW_FIT_SCOPES_EXECUTED_ONCE',rows=rows,
        model_load_seconds=load_seconds,motion_head_batch_seconds=seconds,
        latency_scope='Cached inputs, motion head only; excludes MediaPipe/SignRep/RGB extraction and whole video latency',
        checkpoint=str(R/'runs/rgb_residual_42/model.pt'),checkpoint_sha256=sha(R/'runs/rgb_residual_42/model.pt'),
        training=0,source_labels_edited=False,old_scores_recomputed=False,live_web_changed=False,
        previous_S_zeros_not_used_as_hold_truth=True,whole_goal_complete=False))
    exclusive(U/'RUN_FINISHED.json',dict(pid=os.getpid(),at=time.time(),status='COMPLETE',result_sha256=sha(U/'RESULT.json')))
    print([(r['clip_id'],r['scope'],round(r['observation']['probability'],6),r['observation']['raw_observer_status']) for r in rows])

if __name__=='__main__': main()
