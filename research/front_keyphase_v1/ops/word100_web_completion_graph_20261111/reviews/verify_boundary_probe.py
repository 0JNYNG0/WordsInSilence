"""Read-only saved-array arithmetic. No neural module forward/encoder/training."""
from pathlib import Path
from collections import Counter
import json, hashlib, datetime, ast
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;U=HERE.parent;O=U.parent;W=U/'boundary_candidate_probe';R=O/'word100_candidate_role_learning_20261111';J=O/'word100_episode_joint_linkage_20261111'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def body(x):
    p=np.asarray(x[:,:102],float).reshape(-1,51,2);valid=x[:,186:237]>.5
    ok=valid[:,1]&valid[:,2];d=np.linalg.norm(p[:,1]-p[:,2],axis=1);scale=float(np.median(d[ok])) if ok.any() else 0.
    if scale<1e-5:return np.zeros_like(p),np.zeros_like(valid)
    return (p-(p[:,1]+p[:,2])[:,None]/2)/scale,valid&ok[:,None]

def relation(x,axis,side):
    p,v=body(x);mi,me=(6,4) if side=='R' else (5,3);oi,oe=(5,3) if side=='R' else (6,4)
    p[:,:,0]*=(np.sign(p[:,2,0]-p[:,1,0])*(1 if side=='R' else -1))[:,None]
    q,other=p[:,mi],p[:,oi]
    if axis==0:out=np.column_stack([-q[:,1],q[:,0],-q[:,1],q[:,0]]);pts=[mi]
    elif axis==1:
        arm=q-p[:,me];length=np.linalg.norm(arm,axis=1);out=np.column_stack([-arm[:,1]/np.maximum(length,1e-5),arm[:,0]/np.maximum(length,1e-5),length,-q[:,1]]);pts=[mi,me]
    elif axis==2:
        arm=other-p[:,oe];length=np.linalg.norm(arm,axis=1);rel=q-p[:,oe];out=np.column_stack([(rel*arm).sum(1)/np.maximum(length**2,1e-5),rel[:,0],rel[:,1],np.linalg.norm(q-other,axis=1)]);pts=[mi,oi,oe]
    else:
        diff=q-other;dist=np.linalg.norm(diff,axis=1);height=-diff[:,1];out=np.column_stack([dist if axis==3 else height,np.abs(diff[:,0]),height if axis==3 else dist,-q[:,1]]);pts=[mi,oi]
    good=v[:,pts].all(1)&np.isfinite(out).all(1)
    if axis in (1,2):good&=length>.05
    return out,good

def relations(x):
    replacement=x.copy();p=replacement[:,:102].reshape(-1,51,2);p[:,5]=p[:,9];p[:,6]=p[:,30];replacement[:,191]=replacement[:,195];replacement[:,192]=replacement[:,216]
    rows=[]
    for axis in range(5):
        for side in ['R','L']:
            a,av=relation(x,axis,side);b,bv=relation(replacement,axis,side)
            rows.append(np.c_[np.where(av[:,None],a,0),np.where(bv[:,None],b,0),av,bv].astype(np.float32))
    return np.stack(rows,1)

def groups(mask):
    found=[];a=None
    for i,v in enumerate(mask):
        if v and a is None:a=i
        if a is not None and (not v or i==len(mask)-1):found.append((a,i if v else i-1));a=None
    return found

def candidates(bp,q):
    rows=[];seen=set()
    def add(a,b,k,source,flag=False):
        key=(int(a),int(b),int(k))
        if b-a<3 or key in seen:return
        seen.add(key);valid=q[a:b+1,k,8:10].max(-1)>.5
        rows.append(dict(a=int(a),b=int(b),axis=k//2,moving='L' if k%2 else 'R',source=source,boundary_pair_proposal=flag,coverage=float(valid.mean())))
    for k in range(10):
        valid=q[:,k,8:10].max(-1)>.5;active=(bp[:,k,0]>=.5)&valid
        for a,aa in groups((bp[:,k,1]>=.5)&valid):
            for b,bb in groups((bp[:,k,2]>=.5)&valid):
                if b<aa:continue
                lo=a+int(np.argmax(bp[a:aa+1,k,1]));hi=b+int(np.argmax(bp[b:bb+1,k,2]))
                if hi-lo<3 or active[lo:hi+1].mean()<.6 or valid[lo:hi+1].mean()<.8:continue
                add(lo,hi,k,'boundary_pair',True)
        parts=groups(active)
        for a,b in parts:add(a,b,k,'active_fragment')
        for (a,b),(c,d) in zip(parts,parts[1:]):
            if valid[a:d+1].mean()>=.8:add(a,d,k,'fragment_union')
    return rows

def equivalent(a,b,path=''):
    if isinstance(a,dict):
        assert set(a)==set(b),(path,set(a)^set(b))
        for k in a:equivalent(a[k],b[k],path+'.'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),(path,len(a),len(b))
        for i,(x,y) in enumerate(zip(a,b)):equivalent(x,y,path+f'[{i}]')
    elif isinstance(a,(float,int)) and not isinstance(a,bool):assert np.isclose(a,b,rtol=1e-10,atol=1e-10),(path,a,b)
    else:assert a==b,(path,a,b)

def motion(values,mask,times,a,b):
    x=values[a:b+1].astype(float);m=mask[a:b+1]&np.isfinite(x);ts=times[a:b+1];dt=np.diff(ts).astype(float);dims=[]
    for k in range(x.shape[1]):
        good=m[:,k];starts=np.flatnonzero(good[:-1]&good[1:]);delta=x[starts+1,k]-x[starts,k];ad=np.abs(delta);variation=float(ad.sum());square=float(np.square(delta).sum());count=int(good.sum());nchange=int((delta!=0).sum());idx=int(np.argmax(ad)) if len(starts) else None
        obsstate='none' if count==0 else 'one_sample' if count==1 else 'gapped_samples' if not good.all() else 'all_samples_observed'
        pattern='no_observed_neighbor_pair' if len(starts)==0 else 'no_change_in_observed_neighbor_pairs' if nchange==0 else 'one_observed_change' if nchange==1 else 'multiple_observed_changes'
        dims.append(dict(dimension=k,observation_state=obsstate,sampled_change_pattern=pattern,observed_samples=count,observed_indices=(np.flatnonzero(good)+a).tolist(),unobserved_indices=(np.flatnonzero(~good)+a).tolist(),claimed_observed_nonfinite=int((mask[a:b+1,k]&~np.isfinite(x[:,k])).sum()),observed_sample_runs=[[i+a,j+a] for i,j in groups(good)],observed_neighbor_pairs=[[int(i+a),int(i+a+1)] for i in starts],observed_neighbor_dt_us=dt[starts].tolist(),observed_neighbor_delta=delta.tolist(),observed_neighbor_velocity_per_second=(delta/(dt[starts]/1e6)).tolist(),neighbor_pair_count=len(starts),exact_nonzero_neighbor_changes=nchange,endpoint_difference=float(x[-1,k]-x[0,k]) if good[0] and good[-1] else None,endpoint_difference_is_path_evidence=False,connected_signed_change=float(delta.sum()),connected_absolute_variation=variation,positive_variation=float(np.maximum(delta,0).sum()),negative_variation=float(-np.minimum(delta,0).sum()),largest_neighbor_step=float(delta[idx]) if idx is not None else None,largest_neighbor_pair=[int(starts[idx]+a),int(starts[idx]+a+1)] if idx is not None else None,largest_step_fraction_of_variation=float(ad[idx]/variation) if variation else None,effective_change_steps=variation**2/square if square else None,sampled_range=float(np.ptp(x[good,k])) if count else None))
    return dict(version=1,interval=[a,b],samples=b-a+1,duration_us=float(ts[-1]-ts[0]),sample_dt_us=dt.tolist(),dimensions=dims,interpolation_used=False,inputs_modified=False,intersample_path_observed=False,observation_discontinuity=None,physical_motion_complete=None,guide_requirement_satisfied=None,interpretation='Measurements only. A single sampled change may be real fast motion or a jump; multiple changes do not prove semantic completion.')

def main():
    dst=HERE/'boundary_probe_verification.json';assert not dst.exists()
    frozen=read(W/'FROZEN.json');summary=read(W/'COMPLETE.json');rows=read(W/'RESULTS.json')
    for p,h in frozen['pins'].items():assert sha(p)==h
    cks=torch.load(J/'runs/42/model.pt',weights_only=True,map_location='cpu');quality=torch.load(R/'runs/42/model.pt',weights_only=True,map_location='cpu')
    qm=cks['qm'].numpy();qs=cks['qs'].numpy();assert qm.shape==qs.shape==(10,10)
    assert tuple(cks['boundary.0.weight'].shape)==(64,145) and tuple(cks['boundary.2.weight'].shape)==(3,64)
    assert tuple(quality['quality.0.weight'].shape)==(64,304) and tuple(quality['quality.2.weight'].shape)==(4,64)
    oldproof=read(U/'occurrence_supervision/INDEPENDENT_TRAIN_VERIFICATION.json');protection={r['path']:r['sha256'] for r in oldproof['protected_files']}
    for r in oldproof['checkpoint_checks']:
        p,h=r['initial_path'],r['initial_sha256']
        if p in protection:assert protection[p]==h
        protection[p]=h
    for p,h in read(U/'context_role_training/FROZEN.json')['pins'].items():
        if p in protection:assert protection[p]==h
        protection[p]=h
    for p in [J/'runs/42/model.pt',R/'runs/42/model.pt']:assert sha(p)==protection[str(p)]
    idx=read(R/'EVALUATION_INPUTS.json')['rows']+[dict(s,kind='dynamic_complete_blocks') for s in read(U/'dynamic_blocks/INPUTS_COMPLETE.json')['rows']]
    source={(s['clip_id'],s['kind']):s for s in idx};assert len(source)==len(rows)==82
    phase={(r['clip_id'],r['kind']):r for r in read(U/'context_role_training/RESULTS.json') if r['condition']=='permuted_context' and r['seed']==42}
    families=read(O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json')['words']
    details=[];children=[];all_motion=[];types=Counter();sides=Counter();valid_patterns=Counter();maxnorm=0.;maxcachedq=0.;quality_count=0;phase_count=0
    for row in rows:
        key=row['clip_id'],row['kind'];s=source[key];assert sha(s['path'])==s['sha256'];assert sha(row['boundary_probability'])==row['boundary_sha256']
        with np.load(s['path'],allow_pickle=False) as data:
            x=data['x'];ts=data['ts'];z=data['z'];fh=data['fh'];T=len(ts);q=relations(x)
            h=z[:,228:292];cached_qnorm=z[:,292:392].reshape(T,10,10);assert z.shape==(T,684)
            assert np.array_equal(fh,z[:,392:456])
            if 'h' in data:assert np.array_equal(h,data['h'][0])
            if 'q' in data:
                qerror=float(np.max(np.abs(q-data['q'][0])));maxcachedq=max(maxcachedq,qerror);assert qerror==0
            normalized=(q-qm)/qs;qn=np.concatenate([normalized[:,:,:4]*q[:,:,8,None],normalized[:,:,4:8]*q[:,:,9,None],q[:,:,8:]],axis=-1)
            qerror=float(np.max(np.abs(qn-cached_qnorm)));maxnorm=max(maxnorm,qerror);assert qerror<1e-6
            ids=np.arange(10);query=np.c_[np.eye(5,dtype=np.float32)[ids//2],np.eye(2,dtype=np.float32)[ids%2]]
            boundary_in=np.concatenate([np.broadcast_to(h[:,None],(T,10,64)),cached_qnorm,np.broadcast_to(fh[:,None],(T,10,64)),np.broadcast_to(query,(T,10,7))],axis=-1)
            assert boundary_in.shape==(T,10,145) and np.isfinite(boundary_in).all()
            with np.load(row['boundary_probability']) as bpz:
                bp=bpz['boundary'];assert bp.shape==(T,10,3) and np.isfinite(bp).all() and ((bp>=0)&(bp<=1)).all()
            derived=candidates(bp,q);assert len(derived)==len(row['candidates'])
            all_quality=[]
            for expected,actual in zip(derived,row['candidates']):
                equivalent(expected,{k:actual[k] for k in expected})
                assert actual['physical_complete'] is None and actual['semantic_complete'] is None
                probs=np.asarray(actual['quality']);assert probs.shape==(4,) and np.isclose(probs.sum(),1,atol=2e-7) and (probs>=0).all()
                assert int(probs.argmax())==actual['quality_class'] and float(probs.max())==actual['quality_confidence']
                a,b=actual['a'],actual['b'];k=2*actual['axis']+(actual['moving']=='L');hh=h[a:b+1];zz=qn[a:b+1,k];ff=fh[a:b+1]
                feature=np.r_[hh.mean(0),hh[0],hh[-1],zz.mean(0),zz[0],zz[-1],zz[-1]-zz[0],ff.mean(0),np.eye(5)[k//2],np.eye(2)[k%2],(b-a+1)/100.].astype(np.float32)
                assert feature.shape==(304,) and np.isfinite(feature).all();all_quality.append(feature);quality_count+=1;types[actual['source']]+=1;sides[actual['moving']]+=1
            expected_cards=phase[key]['cards'];assert row['preserved_phase_cards']==expected_cards
            for card in expected_cards:
                if card['parent']:
                    children.append(dict(clip_id=key[0],kind=key[1],card=card['id'],span=card['span'],phase_candidates=card['groups'],preserved_exactly=True))
            expected_measurements=[]
            word=key[0].split('_')[2]
            for card in expected_cards:
                spec=next(c for c in families[word] if c['id']==card['id'])
                if spec['axis'] is None:continue
                k=spec['axis']*2
                values=q[:,k,:8];mask=np.repeat(q[:,k,8:10]>.5,4,axis=1)
                assert np.array_equal(mask[:,:4],np.broadcast_to((q[:,k,8]>.5)[:,None],(T,4)))
                assert np.array_equal(mask[:,4:],np.broadcast_to((q[:,k,9]>.5)[:,None],(T,4)))
                for v in q[:,k,8:10]>.5:valid_patterns[f'{int(v[0])}{int(v[1])}']+=1
                for group in card['groups']:
                    for candidate in group['candidates']:
                        a,b=candidate['span'];expected_measurements.append(dict(card=card['id'],axis=spec['axis'],moving='R',span=[a,b],was_selected=group['span']==[a,b],motion=motion(values,mask,ts,a,b)))
            equivalent(expected_measurements,row['phase_motion_measurements']);phase_count+=len(expected_measurements)
            all_motion.extend(dict(clip_id=key[0],kind=key[1],card=r['card'],span=r['span'],was_selected=r['was_selected'],dimension_count=8,dimension_observed_samples=[d['observed_samples'] for d in r['motion']['dimensions']],dimension_neighbor_pairs=[d['neighbor_pair_count'] for d in r['motion']['dimensions']]) for r in expected_measurements)
            details.append(dict(clip_id=key[0],kind=key[1],frames=T,boundary_input_shape=list(boundary_in.shape),boundary_input_sha256=hashlib.sha256(boundary_in.tobytes()).hexdigest(),relation_norm_max_abs_error=qerror,hidden_and_fh_slices_confirmed=True,boundary_candidates=len(derived),quality_input_shape=[len(all_quality),304],quality_feature_formula_hash=hashlib.sha256(np.asarray(all_quality,dtype=np.float32).reshape(-1,304).tobytes()).hexdigest(),phase_card_count=len(expected_cards),phase_cards_bitwise_json_equal=True,phase_motion_candidates=len(expected_measurements),probability_file_sha256=row['boundary_sha256']))
    assert quality_count==summary['boundary_candidates']==2802 and phase_count==summary['phase_motion_candidates']==124
    code=(U/'probe_boundary_candidates.py').read_text(encoding='utf-8');tree=ast.parse(code)
    subkeys=[n.slice.value for n in ast.walk(tree) if isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant) and isinstance(n.slice.value,str)]
    assert not {'target','source_indices','query_to_source_frame_ids','anchors','gold','label'}&set(subkeys)
    assert 'role.jc.candidate_spans(bp,q,[])' in code
    normal=[r for r in details if r['kind']=='full_normal'];assert len(normal)==12 and sum(r['phase_card_count'] for r in normal)==24
    normal_speed_child=[r for r in children if r['kind'] in ['full_normal','speed075','speed125']]
    proofs=[W/'FROZEN.json',W/'COMPLETE.json',W/'RESULTS.json',U/'candidate_motion_evidence.py',U/'context_role_training/RESULTS.json',R/'role_core.py',J/'joint_core.py',O/'word100_automatic_window_learning_20261110/window_core.py',O/'word100_broad_factorized_learning_20261110/core.py',O/'word100_motion_components_20261109/motion.py',O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json',Path(__file__)]
    result=dict(status='PASS_SAVED_BOUNDARY_CANDIDATE_AND_MOTION_ARITHMETIC_NOT_QUALITY_ACCURACY',reviewer='Separate Codex source/tensor/array audit. No neural forward, encoder inference or human ground truth.',at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),counts=dict(inputs=82,boundary_candidates=quality_count,phase_motion_candidates=phase_count,motion_dimensions=phase_count*8,normal_phase_cards_preserved=24,all_phase_cards_preserved=sum(r['phase_card_count'] for r in details),all_child_cards_preserved=len(children),normal_speed_child_cards_preserved=len(normal_speed_child),boundary_candidate_types=dict(types),boundary_candidate_sides=dict(sides)),architecture=dict(boundary='64h +10normalized_relation +64fh +5axis +2side =145',quality='3x64h +4x10relation +64fh +5axis +2side +1duration =304',frozen_boundary_checkpoint=str(J/'runs/42/model.pt'),frozen_quality_checkpoint=str(R/'runs/42/model.pt'),protected_checkpoint_hashes_match=True,raw_cached_q_max_abs_error=maxcachedq,normalized_q_slice_max_abs_error=maxnorm,quality_numeric_logits_reexecuted=False),input_and_candidate_checks=details,motion_measurement_checks=all_motion,mask_contract=dict(first_four_dimensions='body-observed wrist relation values; valid is q[...,8]',next_four_dimensions='hand-observed anatomical wrist relation values; valid is q[...,9]',repeat_four_is_correct=True,valid_pair_patterns_counted_per_diagnostic_card=dict(valid_patterns),nonfinite_and_masked_samples_not_bridged=True,endpoint_difference_not_path_evidence=True,observed_neighbor_timestamps_preserved=True),child_checks=children,target_leakage=dict(source_indices_not_accessed_by_probe=True,evaluation_target_not_accessed_by_probe=True,word_id_not_passed_to_boundary_or_quality=True,word_id_used_only_to_choose_existing_diagnostic_card_axis=True,all_ten_axis_side_candidates_generated=True,no_legacy_or_expected_span_passed_to_generator=True,existing_phase_result_preserved_not_used_as_boundary_inference_input=True),limitations=['No saved145D or304D inputs existed for direct bit comparison to runtime head inputs. Their source slices/normalization/dimensions/formulas are verified; quality logits and boundary neural outputs were not independently rerun.','Boundary proposals require at least4frames under the unchanged existing generator. Existing short child and all phase proposals are retained separately; this proposal generator must not silently replace the only path for short movements.','Phase-motion diagnostic currently uses moving=R / axis*2 only. Boundary generation covers both hands, but124phase measurements do not validate separate left-hand movement or finger shape.','First8values are two four-dimensional wrist-relative estimates, not8finger joints. Their validity flags indicate the existing observation contract, not correctness proof.','Existing quality feature duration uses(b-a+1)/100, a frame-count convention inherited from training, not seconds. It is preserved for checkpoint compatibility and is not physical completion truth.','A synthetic source-index discontinuity is intentionally not supplied to inference. Motion statistics across a visible unmasked jump remain measurements; no biological plausibility or completed movement is asserted.','Complete was renamed boundary_pair_proposal; all physical/semantic completion outputs remain null. Quality-class confidence has not been shown to mean educational correctness.','Normal24andchild preservation proves saved proposal/display inputs unchanged; no browser or end-to-end feedback test was executed.'],pins=[dict(path=str(p),sha256=sha(p)) for p in proofs],original_frozen_pins=frozen['pins'],training=False,neural_inference=False,private_access=False,web_changed=False)
    for p,h in frozen['pins'].items():assert sha(p)==h
    result['checkpoint_pre_probe_provenance']=[dict(path=str(U/'context_role_training/FROZEN.json'),sha256=sha(U/'context_role_training/FROZEN.json'),use='Joint42 checkpoint hash fixed before this boundary probe'),dict(path=str(U/'occurrence_supervision/INDEPENDENT_TRAIN_VERIFICATION.json'),sha256=sha(U/'occurrence_supervision/INDEPENDENT_TRAIN_VERIFICATION.json'),use='Role42 initial checkpoint hash recorded before this probe')]
    result['verifier_errors_preserved']=[dict(path=str(HERE/name),sha256=sha(HERE/name)) for name in ['FIRST_BOUNDARY_VERIFIER_ERROR.json','SECOND_BOUNDARY_VERIFIER_ERROR.json']]
    dst.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],counts=result['counts'],architecture=result['architecture'],output=str(dst)),ensure_ascii=False))

if __name__=='__main__':main()
