"""Prepare training-only raw temporal inputs and independently verify derived features."""
from pathlib import Path
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,sys,time,copy,importlib.util
import numpy as np
import torch
H=Path(__file__).resolve().parent;L=H.parents[1];PRIOR=L/'ops/partial_encoder_pair_20260922'
sys.path[:0]=[str(H),str(PRIOR)]
import local_io as old
spec=importlib.util.spec_from_file_location('motion_contract_v1',H/'generated_v1.py')
motion_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(motion_module)
motion_features=motion_module.motion_features
fit_velocity_scale=motion_module.fit_velocity_scale
MotionAugmentedTCN=motion_module.MotionAugmentedTCN
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):
    with (H/name).open('x',encoding='utf-8') as f:json.dump(old.plain(value),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
    start=time.time();audit=old.read(H/'IMPLEMENTATION_AUDIT.json');assert audit['status']=='PASS' and audit['module_sha256']==sha(H/'generated_v1.py')
    p=old.policy();parent,model,classes=old.parent_model(p);data,meta=old.train_inputs(p)
    ys={g:np.array([classes.index(r['label']) for r in meta[g]],dtype=np.int64) for g in data}
    fit=fit_velocity_scale(data['base'],ys['base']);assert fit['class_count']==93 and fit['clip_count']==2122
    numerator=np.zeros(186,dtype=np.float64);denominator=np.zeros(186,dtype=np.float64);counts=Counter(ys['base'].tolist())
    records={};new_motion_pairs=0;scale=fit['scale'];max_feature_error=0.
    for group,xs in data.items():
        max_rate=0.;scaled_max=0.;valid_count=0;frame_count=0
        for index,x in enumerate(xs):
            frame_count+=len(x)
            result=motion_features(torch.from_numpy(x[None]),torch.ones((1,len(x)),dtype=torch.float32))[0].numpy()
            valid=(x[1:,186:279]==1)&(x[:-1,186:279]==1)&(x[1:,279,None]>0)
            validxy=np.repeat(valid,2,axis=1);expected=np.zeros((len(x),279),dtype=np.float32)
            dt=np.broadcast_to(x[1:,279,None],validxy.shape)
            delta=x[1:,:186]-x[:-1,:186]
            expected[1:,:186][validxy]=delta[validxy]/dt[validxy]
            expected[1:,186:]=valid.astype(np.float32)
            diff=float(np.max(np.abs(expected-result)));max_feature_error=max(max_feature_error,diff)
            assert np.array_equal(result,expected),(group,index,diff)
            assert not result[0].any() and np.isfinite(result).all()
            valid_count+=int(valid.sum());new_motion_pairs+=int(valid.sum())
            max_rate=max(max_rate,float(np.max(np.abs(result[:,:186]))));scaled_max=max(scaled_max,float(np.max(np.abs(result[:,:186]/scale))))
            if group=='base':
                ns=validxy.sum(axis=0);observed=ns>0;weight=1/(93*counts[int(ys[group][index])])
                energies=np.sum(expected[1:,:186].astype(np.float64)**2,axis=0)
                numerator[observed]+=weight*energies[observed]/ns[observed];denominator[observed]+=weight
        records[group]={'clips':len(xs),'frames':frame_count,'valid_coordinate_pairs':valid_count,'max_abs_relative_rate':max_rate,'max_abs_scaled_rate':scaled_max}
    ratio=np.divide(numerator,denominator,out=np.zeros(186),where=denominator>0)
    oracle=np.ones(186);positive=ratio>0;oracle[positive]=np.sqrt(ratio[positive])
    assert np.array_equal(oracle.astype(np.float32),scale)
    assert np.allclose(denominator,fit['valid_clip_weight'],atol=1e-15,rtol=0)
    wrapper=MotionAugmentedTCN(model,scale).eval()
    # New wrapper integration only: exact zero-residual preservation on actual training shapes/domains.
    picks=sorted(set([0,int(np.argmin([len(x) for x in data['base']])),int(np.argmax([len(x) for x in data['base']]))]+[i for i,r in enumerate(meta['base']) if r['id'].startswith('NIKL_')]))
    max_parent_error=0.
    with torch.inference_mode():
        for i in picks:
            x=torch.from_numpy(data['base'][i][None]);mask=torch.ones((1,x.shape[1]))
            old_scores=model(x,mask);new_scores=wrapper(x,mask)
            max_parent_error=max(max_parent_error,float((old_scores-new_scores).abs().max()))
            assert torch.equal(old_scores,new_scores)
    arrays={'classes':np.array(classes),'velocity_scale':scale,'scale_valid_clip_weight':fit['valid_clip_weight']}
    for group,xs in data.items():
        packed,offsets=old.pack(xs);arrays[group+'_x']=packed;arrays[group+'_offsets']=offsets;arrays[group+'_y']=ys[group]
    with np.load(p['paths']['teacher_scores'],allow_pickle=False) as z:
        assert z['ids'].tolist()==[r['id'] for r in meta['base']]
        arrays['teacher_base_logits']=z['parent'].copy()
    with np.load(p['paths']['schedule'],allow_pickle=False) as z:
        arrays['base_indices']=z['base_indices'].copy();arrays['context_indices']=z['pair_indices'].copy()
    with (H/'training_motion_inputs.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    path=H/'training_motion_inputs.npz'
    # Serialization identity, including every boundary's zero dt; no performance evaluation.
    with np.load(path,allow_pickle=False) as z:
        for group in data:
            assert np.array_equal(z[group+'_x'],arrays[group+'_x'])
            assert np.all(z[group+'_x'][z[group+'_offsets'][:-1],279]==0)
    record={'status':'READY_FOR_COMPARISON_DESIGN_NOT_TRAINED','at_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
      'records':records,'new_motion_pairs_checked':new_motion_pairs,'max_numpy_feature_difference':max_feature_error,'rms_scale_exactly_matches_independent_formula':True,
      'zero_or_unobserved_scale_channels':fit['zero_or_unobserved_channels'],'scale_range':[float(scale.min()),float(scale.max())],
      'actual_zero_residual_cases':len(picks),'max_parent_output_difference':max_parent_error,
      'cache':str(path),'cache_sha256':sha(path),'cache_bytes':path.stat().st_size,'parent':p['parent'],
      'module_sha256':sha(H/'generated_v1.py'),'contract_sha256':sha(H/'CONTRACT.md'),'prior_metadata_sha256':sha(p['paths']['metadata']),
      'scale_fit_source':'original base2122 training only,93 class-balanced clip mean squared relative coordinate rates; context224 not used for fitting',
      'new_training':0,'new_data':0,'new_video_extraction':0,'development_arrays_read':0,'user_arrays_read':0,'reserved_arrays_read':0,
      'seconds':time.time()-start,'next':'Freeze a same-data motion-input candidate training comparison; preserve existing head-only control if exact objective/path equivalence is established.'}
    write('TRAIN_INPUT_READY.json',record);print(json.dumps(record,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
