"""C8 bounded continuation: valid-only context + nonsingular forearm relation.
All ten body relations remain; unchanged150->96->5 architecture.
"""
from pathlib import Path
import json,hashlib,sys,time,traceback
import numpy as np
import torch
from torch import nn
sys.dont_write_bytecode=True
from observation_conditioning import relative_valid_context
BASE=Path(__file__).resolve().parent;P=BASE.parent;U=BASE/'conditioned_context'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,d):
    p=U/n;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p
class Head(nn.Module):
    def __init__(self,mean,std):
        super().__init__();self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32));self.register_buffer('std',torch.tensor(std,dtype=torch.float32));self.head=nn.Sequential(nn.Linear(150,96),nn.GELU(),nn.Linear(96,5))
    def forward(self,x):return self.head((x-self.mean)/self.std)
def freeze():
    U.mkdir(exist_ok=True);old=read(P/'FROZEN.json');rows=read(P/'INDEX.json')['rows'];z=np.load(P/'INPUTS.npz');x=z['absolute_context'].reshape(-1,3,10,5);out=np.empty((len(x),150),np.float32)
    base=[r for r in read(P/'TRAINING_COMPLETE.json')['rows']if r['condition']=='relative_context'];pins=dict(read(BASE/'FROZEN.json')['pins'])
    for pp in [Path(__file__),BASE/'observation_conditioning.py',BASE/'RGB_REVIEW.json']:pins[str(pp)]=sha(pp)
    for r in base:pins[r['prediction']]=r['prediction_sha256'];pins[r['model']]=r['model_sha256']
    checks=[]
    for r in rows:
        sl=slice(r['start'],r['stop']);q=x[sl,1,:,:4].copy();m=x[sl,1,:,4]>.5
        # Projection q0 = dot/length². Retain dot as the scalar and existing
        # opposite forearm feature's length as its conditioning evidence.
        for moving,anchor in [(4,3),(5,2)]:
            assert np.all(~m[:,moving]|m[:,anchor]),'projection anchor missing while relation valid'
            q[:,moving,0]*=q[:,anchor,2]**2
        new,cov=relative_valid_context(q,m,z['timestamp_us'][sl]);out[sl]=np.concatenate([new,cov[...,None]],axis=-1).reshape(-1,150)
        assert np.isfinite(out[sl]).all();assert np.allclose(cov[:,0],x[sl,1,:,4])
        checks.append(r['input_index'])
    np.savez_compressed(U/'INPUTS.npz',features=out);pins[str(U/'INPUTS.npz')]=sha(U/'INPUTS.npz')
    folds=[]
    for fold in old['folds']:
        p=fold['person'];xx=out[z['person']!=p];norm=dict(mean=xx.mean(0,dtype=np.float64).tolist(),std=np.maximum(xx.std(0,dtype=np.float64),1e-4).tolist())
        write(f'folds/{p}/NORMALIZATION.json',norm);normp=U/f'folds/{p}/NORMALIZATION.json';pins[str(normp)]=sha(normp);pins[fold['weights']]=sha(fold['weights'])
        folds.append(dict(**{k:v for k,v in fold.items()if k!='normalization'},normalization=str(normp)))
    # Short pulse is not erased; gaps alone do not invent a pulse.
    ts=np.arange(9)*33333;m=np.ones((9,1),bool);q=np.zeros((9,1,4));q[4]=1
    pulse,pm=relative_valid_context(q,m,ts);assert np.max(abs(pulse))>0
    constant=np.ones_like(q)*3;m[5:7]=False;static,sm=relative_valid_context(constant,m,ts);assert np.max(abs(static[:,1:]))==0 and np.any(sm<1)
    # All-missing context is explicitly unavailable, not a held zero-position.
    no,no_m=relative_valid_context(constant,np.zeros_like(m),ts);assert not no_m.any()
    write('TRAINING_FROZEN.json',dict(condition='availability_conditioned_context',baseline='C6 relative_context saved only',folds=folds,seeds=old['seeds'],steps=old['steps'],lr=old['lr'],weight_decay=old['weight_decay'],gradient_clip=old['gradient_clip'],baseline_rows=base,pins=pins,protected_pins=old['protected_pins'],
        anatomy='All10 body relations retained including both hands and opposite forearm. Projection numerator replaces ratio only for along_other_forearm R/L; denominator evidence remains in opposite forearm length. No clipping or word-specific threshold.',
        context='Before/after means over available observations only; coverage fractions retained; absent pairs yield change0 with unavailable mask, never asserted observed hold.',
        architecture='Identical150->96 GELU->5 fully connected shapes, parameter count, same scratch initial values. New representation, no old weight finetuning.',
        gate='Eachseed pooledmotion/ownhold and eachmovementfamily nondecreasing, nofold losesboth, at leastonestrictgain. Matched primary full/upper classification inconsistency must not increase vsC6; no claim removing all nuisance.',
        no_changed_labels=True,no_DEV_final8_private=True,no_old_forward=True,no_web_change=True,
        limits='Only body motion-vs-hold. Short pulse primitive test is synthetic, not KSL correctness. Crossing family is bilateral height witness, not finger/identity validation.'))
    check(pins);check(old['protected_pins']);write('PRETRAIN_VERIFICATION.json',dict(status='PASS',sequences_checked=len(checks),pulse_preserved=True,constant_with_gap_no_false_motion=True,all_missing_mask_explicit=True,frozen_sha256=sha(U/'TRAINING_FROZEN.json')))
    print('C8 frozen18;144sequences; no new sources',flush=True)
def run():
    c=read(U/'TRAINING_FROZEN.json');check(c['pins']);check(c['protected_pins']);write('TRAINING_STARTED.json',dict(time=time.time()))
    z=np.load(P/'INPUTS.npz');d={k:z[k]for k in z.files};features=np.load(U/'INPUTS.npz')['features'];result=[]
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    for fold in c['folds']:
        p=fold['person'];norm=read(fold['normalization']);w=np.load(fold['weights'])['weights'];active=(d['person']!=p)&(w.sum(1)>0)
        x=torch.tensor(features[active]);y=torch.tensor(d['y'][active],dtype=torch.float32);ww=torch.tensor(w[active])
        for seed in c['seeds']:
            assert not(BASE/'STOP').exists();torch.manual_seed(seed);net=Head(norm['mean'],norm['std']);initial=hashlib.sha256(b''.join(v.detach().numpy().tobytes()for v in net.head.parameters())).hexdigest()
            b=next(r for r in c['baseline_rows']if r['person']==p and r['seed']==seed);assert initial==b['initial_trainable_hash']
            opt=torch.optim.AdamW(net.parameters(),lr=c['lr'],weight_decay=c['weight_decay']);tick=time.perf_counter();losses=[]
            for step in range(c['steps']):
                loss=(nn.functional.binary_cross_entropy_with_logits(net(x),y,reduction='none')*ww).sum();assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),c['gradient_clip']);opt.step()
                if step%300==0 or step==c['steps']-1:losses.append(dict(step=step+1,loss=float(loss.detach())))
            elapsed=time.perf_counter()-tick;out=U/'runs'/f'{p}_{seed}';out.mkdir(parents=True,exist_ok=False);torch.save(net.state_dict(),out/'model.pt')
            with torch.inference_mode():pred=net(torch.tensor(features)).numpy()
            np.savez_compressed(out/'PREDICTIONS.npz',logits=pred);r=dict(person=p,seed=seed,seconds=elapsed,initial_trainable_hash=initial,losses=losses,prediction=str(out/'PREDICTIONS.npz'),prediction_sha256=sha(out/'PREDICTIONS.npz'),model=str(out/'model.pt'),model_sha256=sha(out/'model.pt'))
            write(f'runs/{p}_{seed}/COMPLETE.json',r);result.append(r);print('C8',len(result),'/18',p,seed,round(elapsed,2),flush=True)
    check(c['pins']);check(c['protected_pins']);write('TRAINING_COMPLETE.json',dict(rows=result,status='COMPLETE_DIAGNOSTIC_ONLY',no_old_inference=True))
if __name__=='__main__':
    try:freeze()if sys.argv[1]=='freeze'else run()
    except BaseException:
        U.mkdir(exist_ok=True);(U/f'FIRST_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
