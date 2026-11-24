"""Verify a new wrapper preserves the already-frozen control's saved predictions."""
from pathlib import Path
import hashlib,json,sys,importlib.util
import numpy as np
import torch
H=Path(__file__).resolve().parent;L=H.parents[1];P=L/'ops/partial_encoder_pair_20260922';R=L/'runs/partial_encoder_pair_v1'
sys.path[:0]=[str(P)]
import local_io as old
spec=importlib.util.spec_from_file_location('motion_contract_reuse',H/'generated_v1.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=old.policy();ready=old.read(H/'TRAIN_INPUT_READY.json');audit=old.read(H/'IMPLEMENTATION_AUDIT.json')
    assert audit['status']=='PASS' and audit['module_sha256']==ready['module_sha256']==sha(H/'generated_v1.py')
    assert sha(ready['cache'])==ready['cache_sha256']
    tr=old.read(R/'training.json');ev=old.read(R/'evaluation.json');s=tr['models']['control'];assert sha(s['path'])==s['sha256']
    payload=torch.load(s['path'],map_location='cpu',weights_only=True);model,classes=old.u.rebuild_from_payload(payload)
    with np.load(ready['cache'],allow_pickle=False) as z:scale=z['velocity_scale'].copy()
    wrapper=mod.MotionAugmentedTCN(model,scale,enabled=False).eval();torch.set_num_threads(2)
    groups,meta=old.evaluation_inputs(p);errors={};conditions=0;maxerr=0.
    with np.load(R/'evaluation_arrays.npz',allow_pickle=False) as saved,torch.inference_mode():
        for group,xs in groups.items():
            reference=saved[group+'__control__scores'];scores=[]
            for x in xs:
                q=torch.from_numpy(x[None]);m=torch.ones((1,len(x)),dtype=torch.float32)
                scores.append(wrapper(q,m)[0].numpy())
            current=np.stack(scores);err=float(np.max(np.abs(current-reference)));maxerr=max(maxerr,err)
            assert np.allclose(current,reference,atol=1e-4,rtol=1e-5),(group,err)
            assert np.array_equal(current.argmax(1),reference.argmax(1)),group
            rs=[r for r in ev['rows'] if r['group']==group]
            assert [r['id'] for r in rs]==[r['id'] for r in meta[group]]
            for row,values in zip(rs,current):
                assert row['models']['control']['full93']['pred']==classes[int(values.argmax())]
            errors[group]={'conditions':len(xs),'max_score_difference':err,'all_full93_top1_identical':True};conditions+=len(xs)
    report={'status':'PASS_CONTROL_REUSE_COMPATIBILITY','control':s,'records':errors,'conditions':conditions,'max_score_difference':maxerr,
      'unchanged_objective_required':'Same parent b8,base2122/context224,660x32 saved schedule,LR/WD,head parameter anchor and sampled base centered-logit teacher penalty as the prior comparison. New adapter disabled and zero initialized; all base encoder/buffers fixed.',
      'meaning':'New wrapper execution compatibility only; reuse prior control results, not a new independent accuracy test. Actual enabled-motion benefit not measured.',
      'control_checkpoint_sha256':s['sha256'],'module_sha256':sha(H/'generated_v1.py'),'training_input_ready_sha256':sha(H/'TRAIN_INPUT_READY.json'),
      'prior_evaluation_sha256':sha(R/'evaluation.json'),'new_training':0,'user_arrays_or_scores_read':0,'reserved_arrays_read':0,'development_inputs_read_for_compatibility_only':True}
    with (H/'CONTROL_REUSE_AUDIT.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':report['status'],'conditions':conditions,'max_score_difference':maxerr,'new_training':0}),flush=True)
if __name__=='__main__':main()
