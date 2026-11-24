"""Separate saved-input/checkpoint verification. No neural forward or training."""
from pathlib import Path
from collections import Counter
import json, hashlib, datetime
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;U=HERE.parent;O=U.parent
W=U/'context_role_training';OLD=U/'occurrence_supervision'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    target=HERE/'context_training_verification.json'
    assert not target.exists(), 'Do not replace a completed audit'
    frozen=read(W/'FROZEN.json');admission=read(W/'ADMISSION.json');blockfile=OLD/'FIT_BLOCKS.json'
    assert admission['approved'] and sha(blockfile)==admission['fit_blocks_sha256']
    pin_checks=[]
    for path,h in frozen['pins'].items():
        assert sha(path)==h,path
        pin_checks.append(dict(path=path,sha256=h))
    assert frozen['conditions']==['source_context_control','permuted_context']
    assert frozen['seeds']==[42,43,44] and frozen['primary_seed']==42
    assert (frozen['epochs'],frozen['normal_views'],frozen['steps_per_run'])==(25,72,1800)
    assert (frozen['lr'],frozen['weight_decay'],frozen['teacher_kl_weight'])==(.0003,.01,.2)
    assert (frozen['paired_source_frames'],frozen['paired_views'],frozen['seam_guard_us'])==(34,10,133333)
    blocks_all=read(blockfile)['rows'];blocks={r['clip_id']:r for r in blocks_all if r['status']=='READY_LOCAL_ROLE_CONTEXT_AUGMENTATION'}
    assert len(blocks_all)==6 and len(blocks)==5
    source_admitted={(r['clip_id'],r['sequence_index']):r for r in read(OLD/'DIRECT_ROLE_AUDIT.json')['rows'] if r['category']!='uncertain'}
    previous={(s['clip_id'],s['view']):s for s in read(OLD/'INPUTS_COMPLETE.json')['rows'] if s['condition']=='reviewed_roles'}
    new=read(W/'INPUTS_COMPLETE.json')['rows'];assert len(new)==len(previous)==72
    assert len({(s['clip_id'],s['view']) for s in new})==72 and len({s['clip_id'] for s in new})==36
    assert set(blocks)<={s['clip_id'] for s in new}
    ready_path=O/'word100_shared_motion_learning_20261109/READY.json'
    ready=read(ready_path)['rows'];fit={r['clip_id']:r for r in ready if r['role']=='fit'}
    dev=[r for r in ready if r['role']!='fit'];dev_sha={r['source_sha256'] for r in dev}
    assert len(fit)==899 and len(dev)==300
    assert not {r['source_sha256'] for r in fit.values()}&dev_sha
    normal_details=[];paired_details=[];frame_set=set();family_counts=Counter();normal_direct=0
    seen_sources=set();raw_source_arrays_pinned=[]
    for s in new:
        cid=s['clip_id'];v=s['view'];old=previous[cid,v]
        assert all(s[k]==old[k] for k in ['person','source_sha256','path','sha256'])
        assert cid in fit and fit[cid]['source_sha256']==s['source_sha256'] and 1<=s['person']<=9
        assert s['source_sha256'] not in dev_sha
        assert sha(s['path'])==s['sha256']
        with np.load(s['path'],allow_pickle=False) as z:
            assert set(z.files)=={'z','y','w'} and np.isfinite(z['z']).all()
            assert z['z'].shape==(len(z['y']),684) and z['w'].shape==z['y'].shape
            assert set(np.unique(z['w']))<={0,1} and (z['y'][z['w']>0]>=0).all()
            assert (z['w']>0).any()
            normal_direct+=int(z['w'].sum())
            normal_details.append(dict(clip_id=cid,view=v,path=s['path'],sha256=s['sha256'],frames=len(z['y']),direct_frames=int(z['w'].sum()),same_prior_file_and_mask=True))
            if s['paired'] is None:
                assert cid not in blocks
                continue
            b=blocks[cid];ii=np.asarray(b['query_to_source_sequence_indices']);ts=np.asarray(b['query_timestamp_us'])
            assert np.array_equal(np.sort(ii),np.arange(len(z['y']))) and np.all(np.diff(ts)>0)
            seam_excluded=set()
            for seam in b['splices']:
                mask=np.abs(ts-seam['seam_midpoint_us'])<=133333 if seam['synthetic_discontinuity'] else np.zeros(len(ts),bool)
                indices=np.flatnonzero(mask).tolist();assert indices==seam['excluded_query_indices'];seam_excluded.update(indices)
            boundary=set(i for r in b['boundary_uncertainty'] for i in range(r['range'][0],r['range'][1]+1))
            assert sorted(boundary)==b['boundary_excluded_source_frame_ids']
            p=s['paired'];assert sha(p['path'])==p['sha256']
            with np.load(p['path'],allow_pickle=False) as q:
                assert set(q.files)=={'normal','permuted','y','source_index','query_index'}
                si=q['source_index'];qi=q['query_index'];yy=q['y'];assert len(yy)>0
                assert len(set(si.tolist()))==len(si)
                assert si.tolist()==[a['source_sequence_index'] for a in b['eligible_sampled_labels']]
                assert qi.tolist()==[a['query_index'] for a in b['eligible_sampled_labels']]
                assert yy.tolist()==[a['family'] for a in b['eligible_sampled_labels']]
                assert np.array_equal(ii[qi],si) and np.array_equal(z['y'][si],yy) and (z['w'][si]==1).all()
                assert np.array_equal(q['normal'],z['z'][si]), 'Paired control not exact source features'
                assert q['permuted'].shape==q['normal'].shape==(len(si),684)
                assert np.isfinite(q['permuted']).all() and np.isfinite(q['normal']).all()
                assert (yy>=1).all() and (yy<=7).all()
                for source_i,query_i,family in zip(si.tolist(),qi.tolist(),yy.tolist()):
                    a=source_admitted[cid,source_i]
                    assert not a['boundary_bands'] and a['source_images']
                    assert a['proposed_family']==family and a['source_sha256']==s['source_sha256']
                    assert source_i not in boundary and query_i not in seam_excluded
                    frame_set.add((cid,source_i));family_counts[family]+=1
                assert int(p['frames'])==len(si) and p['families']=={str(k):n for k,n in Counter(yy.tolist()).items()}
                row_deltas=np.max(np.abs(q['normal']-q['permuted']),axis=1)
                paired_details.append(dict(clip_id=cid,view=v,path=p['path'],sha256=p['sha256'],source_indices=si.tolist(),query_indices=qi.tolist(),labels=yy.tolist(),source_control_features_bit_equal=True,finite_recomputed_context=True,frames_with_different_context=int((row_deltas>0).sum()),max_abs_feature_difference=float(row_deltas.max()),min_max_abs_per_frame_difference=float(row_deltas.min()),valid_mask_and_source_provenance_checked=True))
        if cid in blocks and cid not in seen_sources:
            r=fit[cid];assert sha(r['path'])==r['sha256'];seen_sources.add(cid)
            raw_source_arrays_pinned.append(dict(path=r['path'],sha256=r['sha256'],content_loaded_by_this_verifier=False))
    assert len(paired_details)==10 and len(frame_set)==34 and sum(len(r['labels']) for r in paired_details)==68
    assert dict(family_counts)=={1:22,2:14,3:26,6:6} and normal_direct==1640
    assert len(seen_sources)==5

    train=read(W/'TRAINING_COMPLETE.json')['runs'];assert len(train)==6
    assert {(r['condition'],r['seed']) for r in train}=={(c,s) for c in frozen['conditions'] for s in frozen['seeds']}
    rolekeys={'role.0.weight','role.0.bias','role.2.weight','role.2.bias'}
    model_checks=[]
    for row in train:
        assert row['steps']==1800 and len(row['loss'])==25 and np.isfinite(row['loss']).all()
        assert sha(row['path'])==row['sha256']
        p=OLD/'runs/reviewed_roles'/str(row['seed'])/'model.pt'
        assert sha(p)==frozen['pins'][str(p)]
        original=torch.load(p,weights_only=True,map_location='cpu');final=torch.load(row['path'],weights_only=True,map_location='cpu')
        assert set(original)==set(final)
        changed=[k for k in original if not torch.equal(original[k],final[k])]
        assert set(changed)==rolekeys
        unchanged=[k for k in original if k not in rolekeys]
        assert all(k.startswith(('quality.','pair.')) for k in unchanged)
        assert all(torch.isfinite(v).all() for v in final.values())
        model_checks.append(dict(condition=row['condition'],seed=row['seed'],path=row['path'],sha256=row['sha256'],initial_checkpoint=str(p),initial_sha256=sha(p),epochs_logged=len(row['loss']),optimizer_steps_recorded=row['steps'],expected_optimizer_steps=25*72,changed_tensors=changed,non_role_bit_equal_tensors=unchanged,role_max_abs_delta={k:float((original[k]-final[k]).abs().max()) for k in changed}))
    protections=read(OLD/'INDEPENDENT_TRAIN_VERIFICATION.json')['protected_files'];assert len(protections)==83
    for p in protections:assert sha(p['path'])==p['sha256'],p['path']
    # Source inspection only; no model forward is imported or called by this verifier.
    code=(U/'context_role_training.py').read_text(encoding='utf-8')
    snippets=["if r['role']!='fit' or r['clip_id'] not in blocks:continue",'fh,_=role.lc.be.ev.frozen_forward(base,x,',"ob=role.features(temporal,x,ts,fh)","normal=z['z'][sidx],permuted=ob['z'][qidx]",'torch.manual_seed(seed);net.train()',"torch.optim.AdamW(net.role.parameters()",'loss=.5*loss+.5*torch.nn.functional.cross_entropy',"p[~mask].log_softmax(-1),teachers[i][~mask]",'opt.step();steps+=1']
    assert all(s in code for s in snippets)
    assert 'rc.public_rows()' not in code
    extra_pins=[W/'FROZEN.json',W/'ADMISSION.json',W/'INPUTS_COMPLETE.json',W/'TRAINING_COMPLETE.json',OLD/'DIRECT_ROLE_AUDIT.json',HERE/'context_augmentation_design.json',Path(__file__),ready_path]
    out=dict(status='PASS_MATCHED_CONTEXT_TRAINING_ARTIFACT_AUDIT_NOT_PERFORMANCE_PASS',at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),reviewer='Separate Codex artifact arithmetic/checkpoint/code audit; not independent human/KSL validation',counts=dict(normal_clips=36,normal_views=72,normal_direct_frame_views=1640,paired_clips=5,deferred_clips=1,paired_views=10,paired_source_frames=34,paired_frame_views=68,paired_unique_families={str(k):v//2 for k,v in family_counts.items()},models=6,epochs_each=25,steps_each=1800,protected_original_files=83),matched_conditions=dict(same_initial_matching_seed=True,same_normal_source_file_and_masks=True,same_paired_source_indices_and_targets=True,same_loss_weights=True,same_step_schedule=True,altered_context_only_in_paired_slot=True,teacher_kl_only_normal_unlabelled=True,teacher_not_truth=True),normal_inputs=normal_details,paired_inputs=paired_details,checkpoint_tensor_checks=model_checks,source_split=dict(normal_training_sources_all_existing_fit=True,paired_fit_record_ids=[1,4,7],dev_record_ids=sorted({r['person'] for r in dev}),normal_train_vs_dev_source_sha_overlap=0,all899_fit_vs300_dev_source_sha_overlap=0,dev_arrays_labels_or_scores_read=False,ready_metadata_only=True),protected_files=protections,frozen_pins_rechecked=pin_checks,paired_source_array_pins=raw_source_arrays_pinned,extra_pins=[dict(path=str(p),sha256=sha(p)) for p in extra_pins],limitations=['Saved recomputed context arrays are finite and differ from source-control rows; without another model forward this audit verifies source mappings and code provenance, not independently reproducing neural feature arithmetic.','Step count follows completed training logs and pinned25x72 loop; no replay of optimizer execution was performed.','The34 source anchors are5correlated recordings of2expressions. Egypt has only3family6 anchors and no family7 anchors; no claim its spread/closure was newly learned.','133333us exclusion prevents direct supervision at splice neighborhoods, but recurrent context may carry synthetic discontinuity influence farther.','Old w1 normal targets include reviewed intervals. New paired targets are approved sampled direct anchors only.','Frozen weights and fixed masks do not by themselves establish output accuracy or safe web promotion. No evaluation result is scored in this audit.'],training_or_inference_by_verifier=False,old_files_modified=False,private_access=False,web_changed=False)
    for p,h in frozen['pins'].items():assert sha(p)==h,'Artifact changed during audit'
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=out['status'],counts=out['counts'],source_split=out['source_split'],output=str(target)),ensure_ascii=False))

if __name__=='__main__':main()
