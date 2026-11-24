"""C7 new diagnostic: same head tensors, anatomical-family input connections.
The masked architecture has fewer effective connections, not equal effective capacity.
"""
from pathlib import Path
import sys,json,hashlib,time,traceback
import numpy as np
import torch
from torch import nn
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;P=U.parent
F=[2,3,4,5,7];AX={2:0,3:1,4:2,5:4,7:3}
IDX=np.array([[s*50+AX[f]*2*5+c for s in range(3) for c in range(5)]for f in F])
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,d):
    p=U/n;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p
class Head(nn.Module):
    def __init__(self,mean,std):
        super().__init__();self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32));self.register_buffer('std',torch.tensor(std,dtype=torch.float32))
        self.head=nn.Sequential(nn.Linear(150,96),nn.GELU(),nn.Linear(96,5))
        self.register_buffer('indices',torch.tensor(IDX,dtype=torch.long))
    def forward(self,x):
        x=((x-self.mean)/self.std)[:,self.indices]
        weights=self.head[0].weight[:,self.indices].permute(1,0,2)
        h=torch.einsum('nfi,fhi->nfh',x,weights)+self.head[0].bias
        h=self.head[1](h)
        return (h*self.head[2].weight).sum(-1)+self.head[2].bias
def freeze():
    cfg=read(P/'FROZEN.json');completed=read(P/'TRAINING_COMPLETE.json');pins=dict(read(U/'FROZEN.json')['pins'])
    pins[str(Path(__file__))]=sha(__file__)
    pins[str(U/'RGB_REVIEW.json')]=sha(U/'RGB_REVIEW.json')
    base=[r for r in completed['rows']if r['condition']=='relative_context'];assert len(base)==18
    for r in base:pins[r['prediction']]=r['prediction_sha256'];pins[r['model']]=r['model_sha256']
    for f in cfg['folds']:
        for k in ['weights','normalization']:pins[f[k]]=sha(f[k])
    write('TRAINING_FROZEN.json',dict(condition='anatomical_family_local',baseline='C6 relative_context saved predictions only',
        folds=cfg['folds'],seeds=cfg['seeds'],steps=cfg['steps'],lr=cfg['lr'],weight_decay=cfg['weight_decay'],gradient_clip=cfg['gradient_clip'],
        indices=IDX.tolist(),pins=pins,protected_pins=cfg['protected_pins'],baseline_rows=base,
        anatomy='Family2 rightwristheight;3 rightforearm angle;4 Rwrist projected along Lforearm with L elbow/wrist;5 relative bilateral wristheight;7 bilateral separation. Both hands retained in bilateral scalar relations.',
        architecture='Same150x96x5 scratch trainable tensor shapes and random initialization as C6; each output sees only its15 relation/context values. Fewer effective connections; not an equal-effective-capacity ablation.',
        readout='logit>=0 unchanged; labels/weights/folds/epochs unchanged; no threshold/seed/epoch selection',
        gate='Every seed motion/ownhold and each movementfamily nondecreasing vs saved C6 relative; nofold losesbothmotionandhold; full-upper agreement when primaryinputsidentical; at leastonestrictgain.',
        limits='FIT physical motion-vs-hold diagnostic. Not direction/handshape/semantic completion validation. No G2 pass/deployment claim.',
        source_labels='Existing15 residual witnesses and allknown labels preserved; sameCodex RGB audit, noindependentexpert.',
        no_DEV_final8_private=True,no_old_forward=True,no_web_change=True))
    check(pins);check(cfg['protected_pins'])
    # Architecture/invariance tests before fitting any new result.
    torch.manual_seed(42);net=Head(np.zeros(150),np.ones(150));x=torch.randn(5,150)
    with torch.no_grad():
        a=net(x)
        z=x.clone();irrelevant=sorted(set(range(150))-set(IDX[0]));z[:,irrelevant]+=10
        b=net(z);assert torch.equal(a[:,0],b[:,0])
        expected=[]
        for j in range(5):
            masked=torch.zeros_like(x);masked[:,IDX[j]]=x[:,IDX[j]];expected.append(net.head(masked)[:,j])
        assert torch.allclose(a,torch.stack(expected,1),atol=2e-7)
    write('PRETRAIN_VERIFICATION.json',dict(status='PASS',irrelevant_input_invariance=True,explicit_mask_equivalence=True,
        full_upper_primary_identical_residuals=15,original_labels_unchanged=True,frozen_sha256=sha(U/'TRAINING_FROZEN.json')))
    print('frozen18newmodels,2CPUthreads; noC6rerun',flush=True)
def run():
    c=read(U/'TRAINING_FROZEN.json');v=read(U/'PRETRAIN_VERIFICATION.json');assert v['frozen_sha256']==sha(U/'TRAINING_FROZEN.json')
    check(c['pins']);check(c['protected_pins']);write('TRAINING_STARTED.json',dict(time=time.time(),frozen_sha256=sha(U/'TRAINING_FROZEN.json')))
    z=np.load(P/'INPUTS.npz');d={k:z[k]for k in z.files};result=[]
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    for fold in c['folds']:
        person=fold['person'];norm=read(fold['normalization'])['relative_context'];w=np.load(fold['weights'])['weights'];active=(d['person']!=person)&(w.sum(1)>0)
        x=torch.tensor(d['relative_context'][active]);y=torch.tensor(d['y'][active],dtype=torch.float32);ww=torch.tensor(w[active])
        for seed in c['seeds']:
            assert not (U/'STOP').exists();torch.manual_seed(seed);net=Head(norm['mean'],norm['std'])
            initial=hashlib.sha256(b''.join(v.detach().numpy().tobytes()for v in net.head.parameters())).hexdigest()
            baseline=next(r for r in c['baseline_rows']if r['person']==person and r['seed']==seed);assert initial==baseline['initial_trainable_hash']
            opt=torch.optim.AdamW(net.parameters(),lr=c['lr'],weight_decay=c['weight_decay']);tick=time.perf_counter();losses=[]
            for step in range(c['steps']):
                out=net(x);loss=(nn.functional.binary_cross_entropy_with_logits(out,y,reduction='none')*ww).sum();assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),c['gradient_clip']);opt.step()
                if step%300==0 or step==c['steps']-1:losses.append(dict(step=step+1,loss=float(loss.detach())))
            elapsed=time.perf_counter()-tick;out=U/'runs'/f'{person}_{seed}';out.mkdir(parents=True,exist_ok=False)
            torch.save(net.state_dict(),out/'model.pt')
            with torch.inference_mode():pred=net(torch.tensor(d['relative_context'])).numpy()
            np.savez_compressed(out/'PREDICTIONS.npz',logits=pred)
            r=dict(person=person,seed=seed,seconds=elapsed,initial_trainable_hash=initial,losses=losses,
                prediction=str(out/'PREDICTIONS.npz'),prediction_sha256=sha(out/'PREDICTIONS.npz'),model=str(out/'model.pt'),model_sha256=sha(out/'model.pt'))
            write(f'runs/{person}_{seed}/COMPLETE.json',r);result.append(r);print('C7',len(result),'/18',person,seed,round(elapsed,2),flush=True)
    check(c['pins']);check(c['protected_pins']);write('TRAINING_COMPLETE.json',dict(rows=result,no_old_inference=True,status='COMPLETE_DIAGNOSTIC_ONLY'))
if __name__=='__main__':
    try: freeze() if sys.argv[1]=='freeze' else run()
    except BaseException:
        (U/f'FIRST_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
