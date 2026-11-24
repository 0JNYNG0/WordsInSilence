"""One predeclared temporal augmentation candidate. Offline; original inputs immutable."""
from pathlib import Path
import sys
import hashlib
import time
import os
import math
import argparse
import numpy as np
U=Path(__file__).resolve().parent
OLD=U.parent/'word100_training_20261013'
sys.path.insert(0,str(OLD))
from common import BASE,ROOT,read,write,sha,pin_check
import train as original
import torch
RUN=BASE/'runs/word100_temporal_robustness_v1'

def speed(x,factor):
    """Nearest observed time sample; never interpolate hand coordinates/masks across gaps.

    Uniform speed changes duration, keeping approximately the acquisition sampling rate.
    Returned source indices expose repeated/omitted frames. Normalisation stays clip-fixed.
    This is synthetic augmentation, not a repaired real observation.
    """
    assert len(x)>1 and .8<=factor<=1.25
    if factor==1.:return x.copy(),np.arange(len(x))
    elapsed=np.cumsum(x[:,-1],dtype=np.float64);elapsed-=elapsed[0]
    assert np.all(np.diff(elapsed)>0)
    n=max(2,round((len(x)-1)/factor)+1)
    target=np.linspace(0,elapsed[-1],n)
    right=np.searchsorted(elapsed,target).clip(0,len(x)-1);left=np.maximum(right-1,0)
    idx=np.where(target-elapsed[left]<=elapsed[right]-target,left,right)
    y=x[idx].copy(); y[:,-1]=elapsed[-1]/factor/(n-1);y[0,-1]=0
    return y,idx

def augmented_batch(clips,epoch,enable=True):
    changed=[]
    for c in clips:
        context=(epoch+c['parity'])%2==1
        x=c['context'] if context else c['whole']
        if enable:
            key=f"20261013:{epoch}:{c['id']}:speed_v1".encode()
            rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8],'little'))
            if rng.random()<.5:x,_=speed(x,float(rng.uniform(.8,1.25)))
        changed.append(dict(c,whole=x))
    return original.batch(changed)

def freeze():
    assert not (U/'FROZEN_EXPERIMENT.json').exists()
    f=pin_check();audit=read(U/'AUDIT.json'); baseline=read(BASE/'runs/word100_tcn_v1/whole_context/RESULT.json')
    pins=[OLD/'FROZEN_EXPERIMENT.json',OLD/'MANIFEST.json',OLD/'INPUT_CHECK.json',OLD/'train.py',OLD/'common.py',U/'audit.py',U/'AUDIT.json',U/'AUDIT_ROWS.json',U/'experiment.py',BASE/'runs/word100_tcn_v1/whole_context/RESULT.json',BASE/'runs/word100_tcn_v1/whole_context/PREDICTIONS.json',BASE/'runs/word100_tcn_v1/whole_context/best.pt']
    out=dict(status='FROZEN_BEFORE_CANDIDATE_TRAINING',author='Codex',created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        question='Does modest uniform speed augmentation improve the frozen100-word whole-context TCN on the same development set?',
        rationale='899 training examples show within-word core-duration ratios0.47..1.80; missing-hand/time-gap audit did not isolate a systematic error cause. Temporal augmentation is a bounded robustness hypothesis, not a proven error cause.',
        literature=dict(url='https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/papers/Jiang_Skeleton_Aware_Multi-Modal_Sign_Language_Recognition_CVPRW_2021_paper.pdf',use='Primary work supports investigating temporal sampling augmentation; our speed range, mask preservation and TCN effect are local choices, not their measured result.'),
        treatment=dict(training_probability=.5,speed_uniform=[.8,1.25],sampling='nearest original timestamp, both endpoints kept',coordinates='exact source rows; no interpolation',masks='binary original source rows',dt='synthetic duration divided by speed, uniform positive dt; original dt when identity',unchanged='model,100classes,899/300split,initialseed,order,optimizer,whole/context parity,checkpoint rule'),
        train=f['train'],model=f['model'],baseline=baseline,run=str(RUN),
        success=dict(primary='correct >290/300 AND no signer below baseline98/92/100',secondary='report top5,CE,per-class,paired gains/losses and two synthetic speed stresses; never rescue a failed primary gate using secondary scores',repeats='one candidate seed; no post-result coefficient/epoch/threshold search'),
        evaluation=dict(real='original untouched300 development cached whole inputs; no requested label/core at inference',stress='speed0.8 and1.25 on same300:600 correlated synthetic conditions, not600 independent people/videos',blind_test=False),
        constraints=['no user/mentor train/tune/selection','no heldout10/14-18','no original rewriting','no production replacement','no download/camera/Git'],source_sha256={str(p):sha(p) for p in pins})
    write(U/'FROZEN_EXPERIMENT.json',out)

def preflight():
    assert not (U/'PREFLIGHT.json').exists()
    frozen=read(U/'FROZEN_EXPERIMENT.json');assert all(sha(p)==h for p,h in frozen['source_sha256'].items())
    f=pin_check();tr,dv,invalid=original.datasets(f);assert not invalid
    torch.set_num_threads(4);original.seed(f['train']['seed']);m=original.model(f)
    initial=hashlib.sha256(b''.join(v.numpy().tobytes() for v in m.state_dict().values())).hexdigest()
    assert initial==frozen['baseline']['initial_state_sha256']
    n=0
    for c in tr:
        for x in [c['whole'],c['context']]:
            before=x.copy()
            for fac in [.8,1.,1.25]:
                y,idx=speed(x,fac)
                assert idx[0]==0 and idx[-1]==len(x)-1 and np.all(np.diff(idx)>=0)
                np.testing.assert_array_equal(y[:,:279],x[idx,:279]);assert np.isfinite(y).all()
                assert y[0,-1]==0 and (y[1:,-1]>0).all()
                assert abs(float(y[:,-1].sum())-float(x[:,-1].sum())/fac)<2e-5
                assert np.isin(y[:,186:279],[0.,1.]).all()
                assert (y[:,:102].reshape(-1,51,2)[y[:,186:237]==0]==0).all()
                assert (y[:,102:186].reshape(-1,42,2)[y[:,237:279]==0]==0).all()
                n+=1
            np.testing.assert_array_equal(before,x)
    for ep in [1,2]:
        for at in range(0,len(tr),32):
            cc=tr[at:at+32]
            for a,b in zip(augmented_batch(cc,ep,False),original.batch(cc,ep,'whole_context')):assert torch.equal(a,b)
    for a,b in zip(augmented_batch(tr[:32],3),augmented_batch(tr[:32],3)):assert torch.equal(a,b)
    write(U/'PREFLIGHT.json',dict(status='PASS',initial_state_sha256=initial,train=899,dev=300,transformed_sequences_checked=n,no_augmentation_batches_exact=True,input_immutable=True,identity_exact=True,source=sha(U/'experiment.py')))

def run():
    assert not RUN.exists(), 'Existing run requires reuse/recovery; never silently restart'
    frozen=read(U/'FROZEN_EXPERIMENT.json');assert all(sha(p)==h for p,h in frozen['source_sha256'].items())
    assert read(U/'PREFLIGHT.json')['status']=='PASS' and read(U/'PREFLIGHT.json')['source']==sha(__file__)
    f=pin_check();tr,dv,invalid=original.datasets(f);t=f['train'];RUN.mkdir()
    torch.set_num_threads(t['cpu_threads']);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    original.seed(t['seed']);m=original.model(f)
    initial=hashlib.sha256(b''.join(v.numpy().tobytes() for v in m.state_dict().values())).hexdigest()
    assert initial==frozen['baseline']['initial_state_sha256']
    opt=torch.optim.AdamW(m.parameters(),lr=t['lr'],weight_decay=t['weight_decay'])
    start=time.perf_counter();best=math.inf;best_ep=0;step=0;history=[];order_hashes=[]
    for ep in range(1,t['epochs_max']+1):
        m.train();order=np.random.default_rng(t['seed']+ep).permutation(len(tr));total=0.
        order_hashes.append(hashlib.sha256('|'.join(tr[i]['id'] for i in order).encode()).hexdigest())
        for at in range(0,len(order),t['batch_size']):
            cc=[tr[i] for i in order[at:at+t['batch_size']]];x,p,y=augmented_batch(cc,ep)
            opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(m(x,p),y)
            assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),t['gradient_clip_norm']);assert torch.isfinite(norm)
            opt.step();step+=1;total+=loss.item()*len(cc)
        ce,pred=original.predict(m,dv,t['batch_size'])
        if ce<best-1e-6:
            best=ce;best_ep=ep;ck=original.checkpoint(m,f,ep,step,ce,'whole_context_speed')
            ck['frozen_experiment_sha256']=sha(U/'FROZEN_EXPERIMENT.json');torch.save(ck,RUN/'best.pt')
        h=dict(epoch=ep,step=step,train_ce=total/len(tr),development_ce=ce,development_correct=sum(r['prediction']==r['label_index'] for r in pred),best_epoch=best_ep,seconds=time.perf_counter()-start)
        history.append(h);write(RUN/'history.json',history);write(RUN/'progress.json',dict(status='TRAINING',pid=os.getpid(),**h))
        torch.save(dict(model_state=m.state_dict(),optimizer_state=opt.state_dict(),torch_rng=torch.get_rng_state(),epoch=ep,step=step,history=history,context_sha256=sha(U/'FROZEN_EXPERIMENT.json')),RUN/'last.pt')
        print(h,flush=True)
        if ep-best_ep>=t['early_stopping_patience']:break
        assert time.perf_counter()-start<t['wall_safety_seconds_per_arm'],'Safety budget reached; saved checkpoint, no completion claim'
    ck=torch.load(RUN/'best.pt',map_location='cpu',weights_only=True);m.load_state_dict(ck['state_dict'])
    ce,pred=original.predict(m,dv,t['batch_size']);s,rr=original.summarize(pred,invalid,f)
    assert abs(ce-best)<1e-6
    write(RUN/'PREDICTIONS.json',rr)
    result=dict(status='COMPLETE_DEVELOPMENT_ONLY',selected_epoch=best_ep,epochs_executed=ep,steps=step,selected_development_ce=ce,initial_state_sha256=initial,checkpoint_sha256=sha(RUN/'best.pt'),metrics=s,seconds=time.perf_counter()-start,order_sha256=order_hashes,frozen_experiment_sha256=sha(U/'FROZEN_EXPERIMENT.json'))
    write(RUN/'RESULT.json',result)
    # All endpoint stresses were frozen before training, and have no selection role.
    base=original.model(f);base.load_state_dict(torch.load(BASE/'runs/word100_tcn_v1/whole_context/best.pt',map_location='cpu',weights_only=True)['state_dict'])
    stress=[]
    for fac in [.8,1.25]:
        cc=[dict(c,whole=speed(c['whole'],fac)[0]) for c in dv]
        for name,model in [('baseline',base),('candidate',m)]:
            loss,p=original.predict(model,cc,t['batch_size']);ss,pr=original.summarize(p,invalid,f)
            write(RUN/f'STRESS_{name}_{fac}.json',pr);stress.append(dict(model=name,speed=fac,correct=ss['correct'],total=300,ce=loss))
    write(RUN/'STRESS_SUMMARY.json',stress)
    write(RUN/'progress.json',dict(status='COMPLETE_PENDING_INDEPENDENT_VERIFICATION',seconds=time.perf_counter()-start))
    print(dict(correct=s['correct'],stress=stress),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','preflight','run']);args=p.parse_args();globals()[args.action]()
