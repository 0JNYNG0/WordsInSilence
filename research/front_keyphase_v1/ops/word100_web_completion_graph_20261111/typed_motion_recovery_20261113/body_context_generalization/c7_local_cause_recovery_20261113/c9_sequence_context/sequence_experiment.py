"""C9 fixed matched-architecture mean-vs-native-order diagnostic. FIT only."""
from pathlib import Path
import json,hashlib,sys,time,traceback
import numpy as np
import torch
from torch import nn
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;C7=U.parent;C6=C7.parent
CONDITIONS=['mean_control','ordered'];PEOPLE=list(range(2,8));SEEDS=[42,43,44]
COORD=np.array([j*5+c for j in range(10)for c in range(4)]);MASK=np.array([j*5+4 for j in range(10)])
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):
    p=U/n;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8')as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p
def stop():
    for p in [U,C7,C6]:assert not(p/'STOP').exists(),str(p/'STOP')
def sequences(current,timestamps):
    n=len(current);ordered=np.zeros((n,9,51),np.float32);mean=ordered.copy()
    for i in range(n):
        indices=np.arange(i-4,i+5);valid=(indices>=0)&(indices<n);take=indices[valid]
        ordered[i,valid,:50]=current[take].reshape(-1,50)
        ordered[i,valid,50]=(timestamps[take]-timestamps[i])/1e6
        mean[i]=ordered[i]
        # Preserve the exact centre, masks, timestamps; remove coordinate order
        # only within before/after slots. Absent rows stay absent.
        for positions in [np.arange(0,5),np.arange(4,9)]:
            q=ordered[i,positions,:50].reshape(-1,10,5);m=q[...,4]>0
            avg=np.divide((q[...,:4]*m[...,None]).sum(0),m.sum(0)[:,None],out=np.zeros((10,4)),where=m.sum(0)[:,None]>0)
            for p in positions:
                if p==4:continue
                slot=mean[i,p,:50].reshape(10,5);slot[:,:4]=avg*(slot[:,4:5]>0)
    return mean,ordered
class Net(nn.Module):
    def __init__(self,mean,std):
        super().__init__();self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32));self.register_buffer('std',torch.tensor(std,dtype=torch.float32))
        self.projection=nn.Sequential(nn.Linear(51,24),nn.GELU())
        self.temporal=nn.Sequential(nn.Conv1d(24,24,3,padding=1),nn.GELU())
        self.context=nn.GRU(24,16,batch_first=True,bidirectional=True)
        self.head=nn.Linear(32,5)
    def forward(self,x):
        valid=(x[:,:,MASK].sum(-1)>0).to(x.dtype)
        z=(x-self.mean)/self.std
        # Zero normalized absent coordinates; mask fractions are never normalized.
        coords=z[:,:,:50].reshape(*z.shape[:2],10,5).clone();coords[...,:4]*=coords[...,4:5]
        z=torch.cat([coords.reshape(*z.shape[:2],50),z[:,:,50:]],-1)
        z=self.projection(z)*valid[:,:,None];z=self.temporal(z.transpose(1,2)).transpose(1,2)*valid[:,:,None]
        h,_=self.context(z);return self.head(h[:,4])
def init_hash(net):return hashlib.sha256(b''.join(p.detach().numpy().tobytes()for p in net.parameters())).hexdigest()
def prepare():
    stop();old=read(C6/'FROZEN.json');rows=read(C6/'INDEX.json')['rows'];z=np.load(C6/'INPUTS.npz');new=np.load(C7/'conditioned_context/INPUTS.npz')['features'].reshape(-1,3,10,5)
    current=new[:,0].copy();N=len(current);a=np.zeros((N,9,51),np.float32);b=a.copy()
    for r in rows:
        sl=slice(r['start'],r['stop']);a[sl],b[sl]=sequences(current[sl],z['timestamp_us'][sl])
    np.savez_compressed(U/'SEQUENCES.npz',mean_control=a,ordered=b)
    pins=dict(read(U/'FROZEN.json')['pins']);pins[str(Path(__file__))]=sha(__file__);pins[str(U/'SEQUENCES.npz')]=sha(U/'SEQUENCES.npz')
    pins[str(U/'pretrain_verify.py')]=sha(U/'pretrain_verify.py');pins[str(U/'verify_results.py')]=sha(U/'verify_results.py')
    folds=[];mask=current[...,4]>0;flat=current.reshape(N,50)
    for fold in old['folds']:
        p=fold['person'];tr=z['person']!=p;mu=np.zeros(51);sd=np.ones(51)
        for j in range(10):
            take=tr&mask[:,j];assert take.any()
            for c in range(4):
                k=j*5+c;mu[k]=flat[take,k].mean(dtype=np.float64);sd[k]=max(float(flat[take,k].std(dtype=np.float64)),1e-4)
        norm=dict(mean=mu.tolist(),std=sd.tolist(),training_people=sorted(set(z['person'][tr].tolist())))
        write(f'folds/{p}/NORMALIZATION.json',norm);path=U/f'folds/{p}/NORMALIZATION.json';pins[str(path)]=sha(path);pins[fold['weights']]=sha(fold['weights'])
        folds.append(dict(person=p,normalization=str(path),weights=fold['weights']))
    base=[r for r in read(C6/'TRAINING_COMPLETE.json')['rows']if r['condition']=='relative_context'];assert len(base)==18
    for r in base:pins[r['prediction']]=r['prediction_sha256']
    net=Net(np.zeros(51),np.ones(51));parameters=sum(p.numel()for p in net.parameters())
    write('TRAINING_FROZEN.json',dict(status='FROZEN_BEFORE_C9_SCORES',conditions=CONDITIONS,seeds=SEEDS,folds=folds,steps=old['steps'],lr=old['lr'],weight_decay=old['weight_decay'],gradient_clip=old['gradient_clip'],pins=pins,protected_pins=old['protected_pins'],baseline_rows=base,
        architecture='51->24 GELU; temporal Conv1d24->24 kernel3; bidirectionalGRU24->16; centre32->5',trainable_parameters=parameters,
        paired='Same architecture/statistics/initialtrainablevalues/labels/weights/steps; native9rows and masks/timestamps identical. Only within-window coordinate order information differs.',
        normalization='Observed native current rows from five training recordedIDs only; shared between conditions; stdfloor1e-4. Masks unchanged; timestamp in seconds unchanged.',
        geometry='C8 conditioned body relations, all10 including oppositearm/bilateral relations. Nine native rows centred; boundaries are unavailable zero padding with timestamp0, never repeated valid poses.',
        readout='Unchanged logit>=0; no seed/threshold/epoch selection.',
        gate='Each seed ordered must be nondecreasing vs pairedmean in pooledmovement, ownhold and everymovementfamily; nofold losesboth; strictgain. For adoptiondiagnostic also no pooledmovement/ownhold/family decline vs storedC6. Boundarysubset and bilateralheight reported explicitly. NoG2/web promotion.',
        labels='Existing sparse physical body movement/hold. Boundarybands unchanged; no dense truth or semanticcompletion derived. Reversal is physical stress test, not correctKSL.',
        scope='24 historicallyexposed publicFITsourceclips;6recordedIDs02..07;3views×2timeorientations; noDEV/final8/private data.',
        limits='Temporal averaging as currenterrorcause is not proven. This is the fixed information-preservation test, not claim of full semanticorder/completion or finger accuracy.',
        no_C6_C7_C8_rerun=True,no_web_change=True,no_new_teacher_truth=True))
    check(pins);check(old['protected_pins']);print('C9 prepared',N,'rows',parameters,'parameters;36fixedfits',flush=True)
def run():
    stop();c=read(U/'TRAINING_FROZEN.json');v=read(U/'PRETRAIN_VERIFICATION.json');assert v['status']=='PASS'and v['frozen_sha256']==sha(U/'TRAINING_FROZEN.json');check(c['pins']);check(c['protected_pins']);write('TRAINING_STARTED.json',dict(time=time.time(),pid=__import__('os').getpid()))
    z=np.load(C6/'INPUTS.npz');d={k:z[k]for k in z.files};inputs=np.load(U/'SEQUENCES.npz');results=[]
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    for fold in c['folds']:
        p=fold['person'];norm=read(fold['normalization']);w=np.load(fold['weights'])['weights'];active=(d['person']!=p)&(w.sum(1)>0);y=torch.tensor(d['y'][active],dtype=torch.float32);ww=torch.tensor(w[active])
        for seed in SEEDS:
            initial=None
            for condition in CONDITIONS:
                stop();torch.manual_seed(seed);net=Net(norm['mean'],norm['std']);hh=init_hash(net)
                if initial is None:initial=hh
                assert initial==hh;xx=inputs[condition];x=torch.tensor(xx[active]);opt=torch.optim.AdamW(net.parameters(),lr=c['lr'],weight_decay=c['weight_decay']);tick=time.perf_counter();losses=[]
                for step in range(c['steps']):
                    if step%100==0:stop()
                    loss=(nn.functional.binary_cross_entropy_with_logits(net(x),y,reduction='none')*ww).sum();assert torch.isfinite(loss)
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),c['gradient_clip']);opt.step()
                    if step%300==0 or step==c['steps']-1:losses.append(dict(step=step+1,loss=float(loss.detach())))
                elapsed=time.perf_counter()-tick;path=U/'runs'/f'{p}_{seed}_{condition}';path.mkdir(parents=True,exist_ok=False);torch.save(net.state_dict(),path/'model.pt');net.eval()
                with torch.inference_mode():pred=np.concatenate([net(torch.tensor(xx[k:k+512])).numpy()for k in range(0,len(xx),512)])
                np.savez_compressed(path/'PREDICTIONS.npz',logits=pred);r=dict(person=p,seed=seed,condition=condition,initial_trainable_hash=hh,seconds=elapsed,losses=losses,prediction=str(path/'PREDICTIONS.npz'),prediction_sha256=sha(path/'PREDICTIONS.npz'),model=str(path/'model.pt'),model_sha256=sha(path/'model.pt'))
                write(f'runs/{p}_{seed}_{condition}/COMPLETE.json',r);results.append(r);print('C9',len(results),'/36',p,seed,condition,round(elapsed,2),flush=True)
    check(c['pins']);check(c['protected_pins']);write('TRAINING_COMPLETE.json',dict(status='COMPLETE_FIXED_C9_ONLY',rows=results,no_old_inference=True))
if __name__=='__main__':
    try:prepare()if sys.argv[1]=='prepare'else run()
    except BaseException:
        (U/f'FIRST_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
