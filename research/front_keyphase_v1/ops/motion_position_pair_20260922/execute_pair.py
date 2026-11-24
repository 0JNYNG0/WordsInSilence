"""Local, one-shot execution of the frozen Claude-authored comparison."""
from pair_io import *
from datetime import datetime,timezone,timedelta
import time,traceback

def main():
    p=policy();impl=implementation();audit=read(H/'IMPLEMENTATION_AUDIT.json')
    assert audit['status']=='PASS' and sha(H/'generated_v1.py')==audit['module_sha256']
    assert not R.exists(), 'Never overwrite or rerun a completed/failed pair'
    base,context,a=inputs(p);ck,model,classes=parent(p)
    assert len(base)==2122 and len(context)==224 and a['classes'].tolist()==classes
    assert len(classes)==93 and a['teacher_base_logits'].shape==(2122,93)
    assert a['base_indices'].shape==a['context_indices'].shape==(660,16)
    fit=impl.fit_position_scale(base,a['base_y'])
    # Independent NumPy calculation: clip means first, then class-balanced weights.
    means=[];seen=[]
    for x in base:
        mask=np.repeat(x[:,186:279].astype(bool),2,axis=1)
        count=mask.sum(0);seen.append(count>0)
        means.append(np.divide(np.where(mask,x[:,:186].astype(float)**2,0).sum(0),count,
                               out=np.zeros(186),where=count>0))
    labels,counts=np.unique(a['base_y'],return_counts=True)
    weights=np.array([1/(len(labels)*counts[np.flatnonzero(labels==y)[0]]) for y in a['base_y']])
    num=(np.array(means)*weights[:,None]).sum(0);den=(np.array(seen)*weights[:,None]).sum(0)
    oracle=np.ones(186);good=(num>0)&(den>0);oracle[good]=np.sqrt(num[good]/den[good])
    assert np.array_equal(oracle.astype(np.float32),fit['scale'])
    assert np.allclose(den,fit['valid_clip_weight'],atol=1e-14,rtol=0)
    assert fit['clip_count']==2122
    R.mkdir()
    write(H/'POSITION_SCALE.json',dict(status='PASS_INDEPENDENT_TRAIN_ONLY_RMS',**fit,
         scope='original base2122 only; context/dev/user/reserved excluded',
         scale_min=float(fit['scale'].min()),scale_max=float(fit['scale'].max())))
    frozen=dict(at_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
       policy_sha256=sha(H/'comparison_policy.json'),
       files={str(q):sha(q) for q in [H/'generated_v1.py',H/'IMPLEMENTATION_AUDIT.json',H/'pair_io.py',H/'execute_pair.py',H/'POSITION_SCALE.json']},
       data=dict(base=2122,context=224,classes=93,steps=660,batch=32),
       evaluation_started=False,training_attempt=1)
    write(R/'preflight.json',frozen)
    def progress(arm,step,total):print(json.dumps(dict(arm=arm,updates=step,total=total)),flush=True)
    t=time.perf_counter()
    out=impl.run_pair(model,base,a['base_y'],context,a['pair_context_y'],a['teacher_base_logits'],
         a['base_indices'],a['context_indices'],fit['scale'],a['velocity_scale'],
         learning_rate=.001,weight_decay=.001,anchor_coefficient=.01,output_coefficient=1.,progress=progress)
    elapsed=time.perf_counter()-t;models={};arrays={**out['schedule'],'position_scale':fit['scale'],'velocity_scale':a['velocity_scale']}
    initial=model.state_dict()
    for arm in ['control','treatment']:
        v=out[arm];state=v['state_dict'];assert v['updates']==660
        assert set(v['trainable_names'])=={'velocity_weight','base.head.weight','base.head.bias'}
        count=sum(state[n].numel() for n in v['trainable_names']);assert count==23901
        changed=[k for k in initial if not torch.equal(initial[k],state['base.'+k])]
        assert set(changed)=={'head.weight','head.bias'}
        assert bool(torch.count_nonzero(state['velocity_weight']))
        assert v['buffers_unchanged'] and v['frozen_parameters_unchanged']
        path=R/('checkpoint_'+arm+'_final.pt')
        payload=dict(kind='research_paired_input_candidate_v1',classes=classes,state_dict=state,
           feature_mode=v['feature_mode'],parent=p['parent'],policy_sha256=frozen['policy_sha256'],
           implementation_sha256=audit['module_sha256'],motion_implementation_sha256=sha(MOTION/'generated_v1.py'),updates=660)
        with path.open('xb') as f:torch.save(payload,f)
        models[arm]=dict(path=str(path),sha256=sha(path),feature_mode=v['feature_mode'],
              changed_parent_tensors=changed,trainable_names=v['trainable_names'],trainable_count=count,
              added_projection_nonzero=True,frozen_encoder_and_buffers_unchanged=True)
        arrays[arm+'_losses']=v['losses']
    with (R/'training_arrays.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    settings=out['settings'];settings['coordinate_pair_count']=settings.pop('joint_count')
    write(R/'training.json',dict(status='TRAINING_COMPLETE_BOTH_CHECKPOINTS_FROZEN_BEFORE_DEVELOPMENT_EVALUATION',
       at_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),models=models,settings=settings,
       policy_sha256=frozen['policy_sha256'],preflight_sha256=sha(R/'preflight.json'),
       arrays_sha256=sha(R/'training_arrays.npz'),elapsed_training_seconds=elapsed,
       real_training_arms=2,attempts=1,user_arrays_read=0,reserved_arrays_read=0,development_arrays_read=0,
       default_promoted=False))
    print(json.dumps(dict(status='BOTH_FROZEN',elapsed_training_seconds=elapsed,models=models)),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        failure=H/'EXECUTION_FIRST_FAILURE.txt'
        if not failure.exists():failure.write_text(traceback.format_exc(),encoding='utf-8')
        raise
