"""Independently recompute saved proxy errors and training selection; no inference."""
from pathlib import Path
import json,hashlib,importlib.util
import numpy as np
U=Path(__file__).resolve().parent
B=U.parent/'word100_temporal_public_bundle_20261025'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def compute_metrics(cases):
    e=np.concatenate([r['e'].ravel() for r in cases]);a=np.concatenate([r['a'].ravel() for r in cases])
    w=np.concatenate([r['w'].ravel() for r in cases]);wr=np.concatenate([r['wr'].ravel() for r in cases])
    rel=np.concatenate([r['rel'].ravel() for r in cases])
    h=np.where(a<.05,a*a/.1,a-.025)
    result=dict(mean_euclidean=float(e.mean()),mean_l1=float(a.mean()),median_euclidean=float(np.median(e)),
        p95_euclidean=float(np.quantile(e,.95)),quality_weighted_huber_proxy=float((h*w).sum()/w.sum()),
        cases=len(cases),clips=len({r['clip_id'] for r in cases}),joint_frames=len(e),coordinate_values=len(a))
    for prefix,v in (('wrist',wr),('wrist_relative_finger',rel)):
        result.update({prefix+'_mean_euclidean':float(v.mean()),prefix+'_median_euclidean':float(np.median(v)),
            prefix+'_p95_euclidean':float(np.quantile(v,.95)),prefix+'_point_frames':len(v)})
    return result

def compare_metrics(a,b,context):
    diff=0.
    for k,v in a.items():
        assert np.isfinite(v) and np.isfinite(b[k]),(context,k,'nonfinite')
        if isinstance(v,int):assert v==b[k],(context,k,v,b[k])
        else:assert np.isclose(v,b[k],rtol=1e-5,atol=1e-7),(context,k,v,b[k])
        diff=max(diff,abs(v-b[k]))
    return diff

def main():
    rows={r['clip_id']:r for r in read(B/'MANIFEST.json')}
    scenarios={r['case_id']:r for r in read(B/'SCENARIOS.json')['cases']}
    splits={'train':{1,2,3,4,5,6},'dev':{7},'test':{8,9}}
    expected_cases={s:[cid for cid,c in scenarios.items() if c['model_input']['coordinate_transform']['kind']!='none'
        and int(rows[c['evaluator_only']['clip_id']]['signer_id']) in ids] for s,ids in splits.items()}
    assert sum(map(len,expected_cases.values()))==333
    assert all(len(v)==len(set(v)) for v in expected_cases.values())
    expected_runs={f'{c}_seed{s}' for c in ('B2','B3') for s in (17,29,43)}
    assert {p.name for p in (U/'runs').iterdir()}==expected_runs
    protocol=read(U/'FROZEN_PROTOCOL.json')
    for p,h in protocol['source_pins'].items():assert sha(p)==h,(p,'frozen source changed')
    teacher=module(U/'teacher_reliability.py','verify_teacher');policy=read(U/'TEACHER_POLICY.json')
    assert policy['implementation_sha256']==sha(U/'teacher_reliability.py')
    adapter=module(B/'load_inputs.py','verify_safe_adapter');source_weights={}
    for cid,row in rows.items():
        assert sha(row['arrays'])==row['arrays_sha256'],(cid,'public source changed')
        with np.load(row['arrays'],allow_pickle=False) as z:
            source_weights[cid]=teacher.weights_for(adapter.from_arrays(z,'B0'),policy)
    runs=[]; protected=read(B/'PROTECTED_BEFORE.json')
    for p,h in protected.items():assert sha(p)==h,(p,'protected changed')
    total_preserved=0;results={};total_estimated=0
    for folder in sorted((U/'runs').iterdir()):
        result=read(folder/'RESULT.json');history=read(folder/'history.json')
        expected=min(history,key=lambda r:r['dev_weighted_huber'])
        condition,seed=folder.name.split('_seed')
        assert result['condition']==condition and result['seed']==int(seed)
        assert result['selected_epoch']==expected['epoch']
        assert len(history)==80 and [r['epoch'] for r in history]==list(range(1,81))
        assert all(np.isfinite(r['dev_weighted_huber']) and np.isfinite(r['train_weighted_huber']) for r in history)
        steps=(len(expected_cases['train'])+15)//16
        assert result['optimizer_steps']==steps*80 and result['training_examples_per_epoch']==len(expected_cases['train'])
        assert all(r['train_examples']==len(expected_cases['train']) and r['optimizer_steps']==steps for r in history)
        assert result['checkpoint_sha256']==sha(folder/'best.pt')
        import torch
        checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        assert checkpoint['condition']==condition and checkpoint['seed']==int(seed)
        assert checkpoint['selected_epoch']==expected['epoch']
        assert np.isclose(checkpoint['dev_weighted_huber'],expected['dev_weighted_huber'])
        maxdiff=0.;point_count=0
        with np.load(folder/'synthetic_predictions.npz') as predictions:
            assert set(predictions.files)=={s+t for s in splits for t in ('_case_id','_prediction_normalized')}
            for split in ('train','dev','test'):
                errs=[];l1=[];wrists=[];relative=[]
                ids_stored=predictions[split+'_case_id'].tolist();preds=predictions[split+'_prediction_normalized']
                assert ids_stored==expected_cases[split],(folder.name,split,'case list mismatch')
                assert preds.shape==(len(ids_stored),11,42) and np.isfinite(preds).all()
                case_metrics=[]
                for cid,pred in zip(predictions[split+'_case_id'],predictions[split+'_prediction_normalized']):
                    case=scenarios[str(cid)];row=rows[case['evaluator_only']['clip_id']]
                    tr=case['model_input']['coordinate_transform'];side=tr['side_index']
                    lo,hi=tr['frame_range_half_open'];center_index=tr['shared_center_index']
                    ids=np.arange(center_index-5,center_index+6);selected=(ids>=lo)&(ids<hi)
                    with np.load(row['arrays']) as z:
                        xy=z['baseline_roi_xy_px'];good=z['baseline_roi_usable'][:,11]&z['baseline_roi_usable'][:,12]
                        left,right=xy[good,11],xy[good,12];width=np.linalg.norm(left-right,axis=1)
                        ok=np.isfinite(left).all(1)&np.isfinite(right).all(1)&(width>1)
                        centre=np.median((left[ok]+right[ok])/2,axis=0).astype(np.float32);scale=float(np.median(width[ok]))
                        sl=slice(33+21*side,54+21*side)
                        assert z['baseline_roi_usable'][lo:hi,sl].all()
                        truth=((xy[ids,sl]-centre)/scale).astype(np.float32)
                    got=pred.reshape(11,21,2);d=got-truth
                    weight=np.repeat(source_weights[row['clip_id']][ids,sl,None],2,axis=-1)
                    case_metrics.append(dict(case_id=str(cid),clip_id=row['clip_id'],hole=hi-lo,
                        e=np.linalg.norm(d,axis=-1)[selected],a=np.abs(d[selected]),w=weight[selected],
                        wr=np.linalg.norm(d[:,0],axis=-1)[selected],
                        rel=np.linalg.norm((got[:,1:]-got[:,:1])-(truth[:,1:]-truth[:,:1]),axis=-1)[selected]))
                    errs.extend(np.linalg.norm(d,axis=-1)[selected].ravel().tolist())
                    l1.extend(np.abs(d[selected]).ravel().tolist())
                    wrists.extend(np.linalg.norm(d[:,0],axis=-1)[selected].tolist())
                    relative.extend(np.linalg.norm((got[:,1:]-got[:,:1])-(truth[:,1:]-truth[:,:1]),axis=-1)[selected].ravel().tolist())
                actual=result['scores'][split]
                recomputed={'mean_euclidean':np.mean(errs),'mean_l1':np.mean(l1),'wrist_mean_euclidean':np.mean(wrists),'wrist_relative_finger_mean_euclidean':np.mean(relative)}
                for key,value in recomputed.items():
                    diff=abs(value-actual[key]);maxdiff=max(maxdiff,diff)
                    assert np.isclose(value,actual[key],rtol=1e-5,atol=1e-7),(folder.name,split,key,value,actual[key])
                assert len(errs)==actual['joint_frames'];point_count+=len(errs)
                calc=compute_metrics(case_metrics)
                maxdiff=max(maxdiff,compare_metrics(calc,actual,(folder.name,split,'full')))
                assert [r['case_id'] for r in actual['per_case']]==ids_stored
                for cc,stored in zip(case_metrics,actual['per_case']):
                    c=compute_metrics([cc])
                    perkeys=('joint_frames','mean_euclidean','median_euclidean','p95_euclidean',
                        'wrist_mean_euclidean','wrist_relative_finger_mean_euclidean')
                    maxdiff=max(maxdiff,compare_metrics({k:c[k] for k in perkeys},stored,(folder.name,split,cc['case_id'])))
                for hole in (1,3,7):
                    calc_hole=compute_metrics([c for c in case_metrics if c['hole']==hole])
                    maxdiff=max(maxdiff,compare_metrics(calc_hole,actual['by_hole_frames'][str(hole)],(folder.name,split,hole)))
                if split=='dev':
                    assert np.isclose(calc['quality_weighted_huber_proxy'],expected['dev_weighted_huber'],rtol=1e-5,atol=1e-7)
        assert {p.name for p in (folder/'natural_estimates').iterdir()}=={cid+'.npz' for cid in rows}
        estimated_count=0
        for row in rows.values():
            with np.load(row['arrays']) as base,np.load(folder/'natural_estimates'/(row['clip_id']+'.npz')) as out:
                valid=base['redetected_roi_usable']
                assert np.array_equal(base['redetected_roi_xy_px'][valid],out['xy_px'][valid])
                assert not (out['estimated']&valid).any()
                assert np.array_equal(out['observed'],base['redetected_roi_observed']&valid)
                assert np.array_equal(out['usable_observed'],valid)
                assert out['xy_px'].shape==base['redetected_roi_xy_px'].shape
                assert out['estimated'].shape==valid.shape and out['estimated'].dtype==np.bool_
                assert np.isfinite(out['xy_px']).all()
                missing_unchanged=~(valid|out['estimated'])
                # Safe adapter zeroes unusable values; source audit arrays may retain them.
                assert np.all(out['xy_px'][missing_unchanged]==0)
                x0,y0,x1,y1=np.asarray(base['roi_box']).reshape(-1)[:4];added=out['xy_px'][out['estimated']]
                assert ((added[:,0]>=x0)&(added[:,0]<x1)&(added[:,1]>=y0)&(added[:,1]<y1)).all()
                estimated_count+=int(out['estimated'].sum())
                assert np.array_equal(out['frame_id'],base['frame_id']) and np.array_equal(out['timestamp_us'],base['timestamp_us'])
                total_preserved+=int(valid.sum())
        assert estimated_count==result['natural']['counts']['estimated_joint_frames']
        total_estimated+=estimated_count;results[folder.name]=result
        runs.append({'run':folder.name,'status':'PASS','raw_metric_max_abs_difference':maxdiff,'joint_frame_errors_recomputed':point_count,'selected_epoch':expected['epoch']})
    assert len(runs)==6
    aggregate=read(U/'COMPARISON.json')['aggregate']
    for condition,splitdoc in aggregate.items():
        for split,tables in splitdoc.items():
            for key,table in tables.items():
                items=[((key,),table)] if key!='by_hole_frames' else [(('by_hole_frames',h,k),v) for h,metrics in table.items() for k,v in metrics.items()]
                for keys,t in items:
                    values=[]
                    for seed in (17,29,43):
                        value=results[f'{condition}_seed{seed}']['scores'][split]
                        for k in keys:value=value[k]
                        values.append(value)
                    assert values==t['per_seed']
                    compare_metrics({'mean':float(np.mean(values)),'min':min(values),'max':max(values)},t,(condition,split,keys))
    result={'status':'PASS','runs':runs,'preserved_observation_slots_checked':total_preserved,
        'estimated_slots_checked':total_estimated,'protected_unchanged':len(protected),'public_source_hashes_checked':len(rows),
        'expected_scenario_counts':{k:len(v) for k,v in expected_cases.items()},
        'weighted_huber_recomputed':True,'hole_1_3_7_metrics_recomputed':True,'checkpoint_metadata_checked':True,
        'aggregate_mean_min_max_checked':True,'teacher_policy_reused':True,'implementation_sha256':sha(__file__),
        'inference_rerun':False,'ground_truth':'Saved MP proxy only; not anatomical correctness'}
    with (U/'SAVED_SCORE_VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
