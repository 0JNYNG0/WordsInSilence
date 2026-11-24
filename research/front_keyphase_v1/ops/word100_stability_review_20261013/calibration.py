"""Fixed, train-only population-statistics diagnostic; no optimizer or saved model."""
import argparse
import time
import numpy as np
import torch
from probe import U, RUNS, T, read, write, sha, pin_check, digest, bn
from pilot.model_masked_v1 import MaskedBatchNorm1d

class Moments:
    def __init__(self):
        self.n=0; self.mean=None; self.m2=None
    def add(self,x,mask):
        a=x.detach().numpy().transpose(0,2,1)[mask.detach().numpy()[:,0,:].astype(bool)].astype(np.float64)
        n=len(a)
        if not n:return
        mean=a.mean(0);m2=((a-mean)**2).sum(0)
        if not self.n:self.n=n;self.mean=mean;self.m2=m2;return
        delta=mean-self.mean;total=self.n+n
        self.m2+=m2+delta**2*self.n*n/total
        self.mean+=delta*n/total;self.n=total

def unit_check():
    rng=np.random.default_rng(75);x=rng.normal(size=(5,3,13)).astype(np.float32);mask=(rng.random((5,1,13))>.25).astype(np.float32)
    expected=x.transpose(0,2,1)[mask[:,0,:].astype(bool)].astype(np.float64)
    a=Moments()
    for i,j in [(0,2),(2,4),(4,5)]:a.add(torch.from_numpy(x[i:j]),torch.from_numpy(mask[i:j]))
    assert a.n==len(expected)
    assert np.allclose(a.mean,expected.mean(0),atol=1e-13,rtol=0)
    assert np.allclose(a.m2/(a.n-1),expected.var(0,ddof=1),atol=1e-13,rtol=0)

def freeze():
    assert not (U/'POPULATION_PLAN.json').exists()
    unit_check()
    write(U/'POPULATION_PLAN.json',dict(status='FROZEN_BEFORE_POPULATION_SCORES',
        reason='Cross-epoch swaps establish sensitivity, not a valid matched-state repair; test matched statistics without gradient training.',
        arms=['current_best','current_last','speed_best','speed_last'],
        algorithm='Sequential layerwise population moments: fixed model eval mode, dropout off, 899 original whole-span training clips, all valid frames weighted equally, float64 Chan moments, unbiased variance, earlier layers already recalculated.',
        calibration_data='899 training IDs01-09 only; no labels needed, no development/heldout/user data for statistics.',
        input_choice='Whole span matches deployment and development preprocessing; do not try alternative calibration pools after results.',
        evidence_rule='Report all four. Improvement of fixed selected best over its own score is diagnostic utility; recovery of last is stability evidence, never grounds to select last post hoc. Regressions refute this particular quick repair. No automatic deployment or success guarantee.',
        reference='Wu and Johnson 2021, Rethinking Batch in BatchNorm, Appendix A.3, https://arxiv.org/abs/2105.07576',
        implementation_sha256=sha(__file__),factorial_sha256=sha(U/'FACTORIAL.json'),gradient_steps=0,new_checkpoints=0,
        warning='Re-estimation fits normalization buffers on training inputs; not untouched-model inference, and not independent new-user validation.'))

def run():
    assert not (U/'POPULATION_RESULT.json').exists()
    p=read(U/'POPULATION_PLAN.json');assert sha(__file__)==p['implementation_sha256']
    assert sha(U/'FACTORIAL.json')==p['factorial_sha256']
    guards=read(U/'PLAN.json')['source_sha256'];assert all(sha(k)==v for k,v in guards.items())
    unit_check();f=pin_check();train,dev,invalid=T.datasets(f);assert not invalid
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    results={};start=time.perf_counter()
    for name,path in RUNS.items():
        for which in ['best','last']:
            ck=torch.load(path/f'{which}.pt',weights_only=True,map_location='cpu')
            state=ck['state_dict' if which=='best' else 'model_state']
            original=digest(state);m=T.model(f);m.load_state_dict(state);m.eval()
            moments={}
            for layername,layer in m.named_modules():
                if not isinstance(layer,MaskedBatchNorm1d):continue
                a=Moments()
                handle=layer.register_forward_pre_hook(lambda mod,args:a.add(args[0],args[1]))
                with torch.inference_mode():
                    for at in range(0,len(train),32):
                        x,mask,_=T.batch(train[at:at+32]);m(x,mask)
                handle.remove();assert a.n>1
                var=a.m2/(a.n-1);assert np.isfinite(var).all() and (var>=0).all()
                with torch.no_grad():
                    layer.running_mean.copy_(torch.from_numpy(a.mean).float())
                    layer.running_var.copy_(torch.from_numpy(var).float())
                moments[layername]=dict(valid_frame_count=a.n,mean=a.mean.tolist(),unbiased_variance=var.tolist())
            assert len(moments)==8
            assert len(set(v['valid_frame_count'] for v in moments.values()))==1
            assert all(torch.equal(v,state[k]) for k,v in m.state_dict().items() if not bn(k))
            assert digest(state)==original
            before=digest(m.state_dict());ce,pred=T.predict(m,dev,32);metric,rows=T.summarize(pred,[],f)
            assert digest(m.state_dict())==before
            key=f'{name}_{which}';write(U/f'population_{key}.json',rows)
            results[key]=dict(correct=metric['correct'],ce=ce,by_signer=metric['by_signer'],statistics=moments,
                learned_parameters_unchanged=True,checkpoint_sha256=sha(path/f'{which}.pt'))
            print(key,metric['correct'],flush=True)
    assert all(sha(k)==v for k,v in guards.items())
    write(U/'POPULATION_RESULT.json',dict(status='COMPLETE_PENDING_RECOUNT',results=results,seconds=time.perf_counter()-start,
        moment_unit_check='PASS_FLOAT64_DIRECT_MASKED_MOMENTS',source_guards='PASS',gradient_steps=0,new_checkpoint_files=0))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','run']);args=a.parse_args();globals()[args.action]()
